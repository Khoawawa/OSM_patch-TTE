import torch

from models.base.ContextEncoder import ContextEncoder
from models.base.LayerNormGRU import LayerNormGRU
from models.base.RegionEncoder import PoiEncoder

import torch.nn.functional as F
import torch.nn as nn
import math
import copy

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
        
        decoder_head = 1
        
        self.decoder = Decoder(d_model=seq_hidden_dim, N=decoder_layer, heads=decoder_head)
        
        self.mlp = nn.Sequential(
            nn.Linear(seq_hidden_dim + 33, seq_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(seq_hidden_dim, 1)
        )

    def forward(self, input_, args):
        segment_mask = input_['valid_mask']        
        # context output
        ctx_output, loss_1, (weekrep,daterep,timerep) = self.context_encoder(input_, args) # (B,T,seq_hidden_dim)
        # temporal modeling
        ctx_output = ctx_output if batch_first else ctx_output.transpose(0,1).contiguous() # (T,B,Res + Ctx)
        hiddens, _ = self.temporal_block(ctx_output, seq_lens = input_['lens'].long())
        # decoder
        device_type = "cuda" if hiddens.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            decoder = self.decoder(hiddens.float(), input_['lens'].long())
        decoder = decoder if batch_first else decoder.transpose(0,1).contiguous()
        # mean pooling
        decoder = decoder * segment_mask.unsqueeze(-1).float() # (B,T,seq_hidden_dim)
        pooled_decoder = decoder.sum(dim=1) # (B,seq_hidden_dim)
        # pooled_decoder = pooled_decoder / input_['lens'].unsqueeze(-1).float() # (B,seq_hidden_dim)
        pooled_decoder = torch.cat([pooled_decoder, weekrep, daterep, timerep], dim=-1) # (B,seq_hidden_dim + 33)
        output = self.mlp(pooled_decoder)

        return output, loss_1


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