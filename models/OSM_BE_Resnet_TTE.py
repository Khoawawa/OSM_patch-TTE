import torch

from models.base.ContextEncoder import ContextEncoder
from models.base.LayerNormGRU import LayerNormGRU
from models.base.AdaNorm import AdaRMSNorm

import torch.nn.functional as F
import torch.nn as nn
import math
import copy
batch_first = False
def assert_finite(t, name):
    if not torch.isfinite(t).all():
        print(f"\n NaN/Inf in {name}")
        print("shape:", t.shape)
        print("min:", torch.nanmin(t))
        print("max:", torch.nanmax(t))
        print("example values:", t.flatten()[:10])
        raise RuntimeError(name)

class MulT_TTE(torch.nn.Module):
    def __init__(self,
                 seq_hidden_dim, seq_layer,
                 decoder_layer,
                 bert_attention_heads,bert_hidden_size,pad_token_id,bert_hidden_layers,vocab_size=27300):
        super().__init__()
        self.context_encoder = ContextEncoder(bert_attention_heads,bert_hidden_size,pad_token_id,bert_hidden_layers,vocab_size) # trip specific encoder
        self.represent = nn.Sequential(
            nn.Linear(self.context_encoder.hidden_size, seq_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(seq_hidden_dim, seq_hidden_dim)
        )
        self.temporal_block = LayerNormGRU(input_dim=seq_hidden_dim, hidden_dim=seq_hidden_dim, num_layers=seq_layer)
        
        self.decoder = Decoder(d_model=seq_hidden_dim, N=decoder_layer)
        self.adanorm = AdaRMSNorm(d_model=seq_hidden_dim, d_context=self.context_encoder.datetimerep_size + 1)
        self.mlp = nn.Sequential(
            nn.Linear(seq_hidden_dim*2, seq_hidden_dim*2),
            nn.GELU(),
            nn.Linear(seq_hidden_dim*2, 1)
        )

    def sum_pooling(self, decoder, valid_mask):
        mask = valid_mask.float().unsqueeze(-1)
        masked_outputs = decoder * mask
        pooled = masked_outputs.sum(dim=1)
        return pooled
    def max_sum_pooling(self, h : torch.Tensor, valid_mask: torch.Tensor, seg_lens):
        mask = valid_mask.float().unsqueeze(-1)
        # bottleneck identifier: max pooling
        masked_h = h.masked_fill(mask == 0, -1e4)
        max_pooled = masked_h.max(dim=1).values
        # weighted sum pooling
        masked_h = h * mask
        weights = seg_lens.float().unsqueeze(-1) # (B,T)
        masked_weights = weights * mask
        weighted_masked_h = masked_h * masked_weights
        sum_pooled = weighted_masked_h.sum(dim=1)
        total_weight = masked_weights.sum(dim=1).clamp(min=1e-6)
        sum_pooled = sum_pooled / total_weight
        return torch.cat([max_pooled, sum_pooled], dim=-1)
    
    def forward(self, input_, args):
        # visual input
        valid_mask = input_['valid_mask']  # (B,T)
        seg_lens = input_['links'][:,:,1] # (B,T,1)
        # representation encoding
        representation, loss_1, datetimerep = self.context_encoder(input_, args)
        representation = self.represent(representation) # (B,T,seq_hidden_dim)
        representation = representation if batch_first else representation.transpose(0,1).contiguous() # (T,B,Res + Ctx)
        # temporal modeling
        hiddens, _ = self.temporal_block(representation, seq_lens = input_['lens'].long())
        decoder = self.decoder(hiddens, input_['lens'])
        decoder = decoder if batch_first else decoder.transpose(0,1).contiguous() # (B,T,seq_hidden_dim)
        # inject temporal context
        cum_lens = input_['links'][:,:,2]  # (B,T)
        progress = (cum_lens / cum_lens[:,-1:].clamp(min=1e-6)).unsqueeze(-1)  # (B,T,1)
        
        cond = torch.cat([datetimerep, progress], dim=-1)  # (B,T,89)
        decoder = self.adanorm(decoder, cond)
        assert_finite(decoder, "decoder after adanorm")
        # pooling
        pooled_decoder = self.max_sum_pooling(decoder, valid_mask, seg_lens) # (B, seq_hidden_dim*2)
        assert_finite(pooled_decoder, "pooled_decoder")
        # final MLP
        output = self.mlp(pooled_decoder) # (B,1)
        
        return output, loss_1

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            dropout=dropout,
            batch_first=False   # because your x is (T,B,D)
        )

    def forward(self, x, lens):
        device = lens.device
        max_len = x.size(0)
        mask = torch.arange(max_len, device=device)[None, :] >= lens[:, None]
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads=1, dropout=0.1):
        super().__init__()
        # self.norm_1 = nn.LayerNorm(d_model) #
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        # self.dropout_1 = nn.Dropout(dropout)   #
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        #self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)


    def forward(self, x, len):
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, len))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

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


if __name__ == "__main__":
    pass