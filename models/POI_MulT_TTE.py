import torch

from models.base.ContextEncoder import ContextEncoder
from models.base.LayerNormGRU import LayerNormGRU
from models.base.RegionEncoder import PoiEncoder

import torch.nn.functional as F
import torch.nn as nn
import math
import copy
from info_nce import InfoNCE, info_nce
batch_first = False

class POI_MulT_TTE(torch.nn.Module):
    def __init__(self,
                 seq_hidden_dim, seq_layer,
                 decoder_layer,
                 bert_attention_heads,bert_hidden_size,pad_token_id,bert_hidden_layers,vocab_size=27300):
        super().__init__()
        # context encoder -> poi encoder -> temporal encoder -> decoder -> MLP
        self.context_encoder = ContextEncoder(seq_hidden_dim, bert_attention_heads,bert_hidden_size,pad_token_id,bert_hidden_layers,vocab_size)        
        self.temporal_block = LayerNormGRU(input_dim=seq_hidden_dim, hidden_dim=seq_hidden_dim, num_layers=seq_layer)
        self.cl_proj = nn.Sequential(
            nn.Linear(seq_hidden_dim, seq_hidden_dim//2),
            nn.LeakyReLU(),
            nn.Linear(seq_hidden_dim//2, seq_hidden_dim//2),
            nn.LayerNorm(seq_hidden_dim//2)
        )
        decoder_head = 1
        
        self.decoder = Decoder(d_model=seq_hidden_dim, N=decoder_layer, heads=decoder_head)
        
        self.mlp = nn.Sequential(
            nn.Linear(seq_hidden_dim + 33, seq_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(seq_hidden_dim, 1)
        )
        self.info_nce_loss = InfoNCE()
    def point_masking(self, x,padded_mask,mask_rate):
        # x: [B, L, D] 
        # padded_mask: [B, L] 
        B, L, D = x.shape
        device = x.device
        
        rand = torch.rand(B,L,device=device)
        pt_mask = (rand < mask_rate) & (padded_mask.bool()) # [B, L]
        
        x_masked = x.clone()
        x_masked[pt_mask] = 0.0 
        
        return x_masked, pt_mask
    def merge_algo(self, x1, x2):
        # x1, x2: [B, D]
        return (x1 + x2) / 2
    def merge_segments(self,x, padded_mask, merge_rate):
        # x: [B, L, D]
        # padded_mask: [B, L]
        B, L, D = x.shape
        device = x.device

        # ──────────────────────────────────────────────
        # 1. Decide where merges happen (independent per sample)
        # ──────────────────────────────────────────────
        can_merge = padded_mask[:, 1:] & padded_mask[:, :-1]           # [B, L-1]
        merge_decision = (torch.rand(B, L-1, device=device) < merge_rate) & can_merge  # [B, L-1]

        # ──────────────────────────────────────────────
        # 2. Compute cumulative merges before each position
        # ──────────────────────────────────────────────
        cum_merges = torch.cat([
            torch.zeros(B, 1, dtype=torch.long, device=device),
            merge_decision.cumsum(dim=1)   # cumsum along sequence
        ], dim=1)  # [B, L]

        # ──────────────────────────────────────────────
        # 3. Which original positions are kept? (not the second of a merge pair)
        # ──────────────────────────────────────────────
        is_second_of_merge = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=device),
            merge_decision
        ], dim=1)  # [B, L]  True if this pos is skipped (merged into previous)

        is_kept = ~is_second_of_merge & padded_mask   # [B, L]

        # ──────────────────────────────────────────────
        # 4. New position for each kept original timestep
        # ──────────────────────────────────────────────
        new_pos = torch.arange(L, device=device).expand(B, L) - cum_merges  # [B, L]

        # ──────────────────────────────────────────────
        # 5. Gather the features that survive
        # ──────────────────────────────────────────────
        # For kept normal positions → take original
        # For merge-start positions → take average of self and next
        is_merge_start = merge_decision   # [B, L-1] but we need [B, L]
        is_merge_start = torch.cat([
            merge_decision,
            torch.zeros(B, 1, dtype=torch.bool, device=device)
        ], dim=1)

        # Features to place
        features_to_scatter = x.clone()   # start with originals

        # Where we merge → overwrite the start position with average
        merge_start_mask = is_merge_start & padded_mask   # only if valid start
        if merge_start_mask.any():
            # Gather next position features
            next_features = torch.roll(x, shifts=-1, dims=1)   # [B, L, D]
            avg_features = (x + next_features) / 2

            # Put averaged value at the start position
            features_to_scatter = features_to_scatter.masked_scatter(
                merge_start_mask.unsqueeze(-1).expand(-1, -1, D),
                avg_features.masked_select(merge_start_mask.unsqueeze(-1).expand(-1, -1, D))
            )

        # Now only keep the features where is_kept
        kept_features = features_to_scatter[is_kept]   # [num_kept_total, D]
        kept_new_pos = new_pos[is_kept]                # [num_kept_total]
        kept_batch_idx = torch.nonzero(is_kept)[:, 0]  # [num_kept_total]

        # ──────────────────────────────────────────────
        # 6. Scatter into output tensor
        # ──────────────────────────────────────────────
        merged_x = torch.zeros(B, L, D, device=device)
        merged_x[kept_batch_idx, kept_new_pos] = kept_features

        # ──────────────────────────────────────────────
        # 7. Build updated mask & new lengths
        # ──────────────────────────────────────────────
        updated_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        updated_mask[kept_batch_idx, kept_new_pos] = True

        new_lengths = is_kept.sum(dim=1)   # [B]

        return merged_x, updated_mask, new_lengths
    
    def forward(self, input_, args):
        segment_mask = input_['valid_mask']
        seq_lens = input_['lens'].long()
        # context output
        h_ori, _, (weekrep,daterep,timerep) = self.context_encoder(input_, args) # (B,T,seq_hidden_dim)
        # point masking
        if self.training:
            h_merged, updated_mask, merged_lens = self.merge_segments(h_ori, segment_mask, merge_rate=args.merge_rate) # (B,T,seq_hidden_dim)
            new_lens = torch.cat([seq_lens, merged_lens], dim=0)
            # concat for GPU
            h = torch.cat([h_ori, h_merged], dim=0) # (2B,T,seq_hidden_dim)
        else:
            h = h_ori
            new_lens = seq_lens
        h = h.transpose(0,1).contiguous() if not batch_first else h.contiguous() # (T,2B,seq_hidden_dim)
        # temporal block
        hiddens, _ = self.temporal_block(h, seq_lens = new_lens)
        if self.training:
            mask_2view = torch.cat([segment_mask, updated_mask], dim=0) 
            mask_2view = mask_2view.transpose(0,1).contiguous() if not batch_first else mask_2view.contiguous() # (T,2B)
            mask_2view = mask_2view.unsqueeze(-1) # (T,2B,1)
            pooled = (hiddens * mask_2view.float()).sum(dim=0) # (2B,seq_hidden_dim)
            pooled_proj = self.cl_proj(pooled) # (2B,seq_hidden_dim//4)
            proj = F.normalize(pooled_proj, dim=-1) # (2B,seq_hidden_dim//4)
            proj_ori, proj_merged = proj.chunk(2, dim=0) # each (B,seq_hidden_dim//4)
            # TODO: infoNCE
            loss_cl = self.info_nce_loss(proj_ori, proj_merged) 
            hiddens_ori, _ = hiddens.chunk(2, dim=1)  # (T, B, D)
        else:
            loss_cl = None
            hiddens_ori = hiddens
            
        device_type = "cuda" if hiddens_ori.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            decoder = self.decoder(hiddens_ori.float(), seq_lens)
        decoder = decoder if batch_first else decoder.transpose(0,1).contiguous()
        # mean pooling
        decoder = decoder * segment_mask.unsqueeze(-1).float() # (B,T,seq_hidden_dim)
        pooled_decoder = decoder.sum(dim=1) # (B,seq_hidden_dim)
        pooled_decoder = torch.cat([pooled_decoder, weekrep, daterep, timerep], dim=-1) # (B,seq_hidden_dim + 33)
        output = self.mlp(pooled_decoder)

        return output, loss_cl


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.h = heads
        self.attn_1 = nn.MultiheadAttention(embed_dim=d_model,kdim=d_model,vdim=d_model, dropout=dropout, num_heads=self.h)

    def forward(self, q, k, v, len):
        # perform linear operation and split into N heads
        device = len.device
        max_len = torch.max(len).item()
        mask = torch.arange(max_len, device=device).unsqueeze(0) < len.unsqueeze(1)
        attn_output = self.attn_1(q, k, v, key_padding_mask=~mask, need_weights=False)[0]
        return attn_output

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        d_ff = d_model * 2
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x):
        return self.ffn(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads=1, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)


    def forward(self, x, len):
        x1 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x1, x1, x1, len))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Decoder(nn.Module):
    def __init__(self, d_model, N=3, heads=1, dropout=0.1):
        super().__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, lens):
        for i in range(self.N):
            x = self.layers[i](x, lens)
        return self.norm(x)