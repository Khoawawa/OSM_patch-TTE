from models.ContrasiveModel import TrajContrasiveEncoder
import torch

from models.base.SegmentEncoder import SegmentEncoder
from models.base.LayerNormGRU import LayerNormGRU

import torch.nn.functional as F
import torch.nn as nn
import math
import copy
from pytorch_metric_learning import losses
batch_first = False

class POI_MulT_TTE(torch.nn.Module):
    def __init__(self,
                seq_hidden_dim, cl_queue_size,cl_hidden_dim,cl_head,cl_layer,
                seq_layer,
                decoder_layer):
        super().__init__()
        self.segment_encoder = SegmentEncoder(seq_hidden_dim,cl_queue_size,cl_hidden_dim,cl_head,cl_layer)
        
        self.temporal_block = LayerNormGRU(input_dim=seq_hidden_dim, hidden_dim=seq_hidden_dim, num_layers=seq_layer)
        
        decoder_head = 1
        
        self.decoder = Decoder(d_model=seq_hidden_dim, N=decoder_layer, heads=decoder_head)
        
        self.mlp = nn.Sequential(
            nn.Linear(seq_hidden_dim + 33, seq_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(seq_hidden_dim, 1)
        )
        # self.alpha_h = nn.Parameter(torch.tensor(0.2))
    def get_SE_alpha(self):
        return self.segment_encoder.get_alpha_h()
    def forward(self, input_, args):
        segment_mask = input_['valid_mask']  
        is_train = args.phase == 'train'      
        # context output
        seg_feats, cl_loss,datetimerep = self.segment_encoder(input_) # (B,T,seq_hidden_dim)
        
        seg_feats = seg_feats if batch_first else seg_feats.transpose(0,1).contiguous() # (T,B,Res + Ctx)
        h, _ = self.temporal_block(seg_feats, seq_lens = input_['lens'].long())
        # decoder
        with torch.amp.autocast(device_type="cuda" if h.is_cuda else "cpu", enabled=False):
            d = self.decoder(h.float(), input_['lens'].long())
        d = d if batch_first else d.transpose(0,1).contiguous()
        # sum pooling + MLP
        d = d * segment_mask.unsqueeze(-1).float() # (B,T,seq_hidden_dim)
        pooled_d = d.sum(dim=1) # (B,seq_hidden_dim)
        pooled_d = torch.cat([pooled_d, datetimerep], dim=-1) # (B,seq_hidden_dim + 33)
        z = self.mlp(pooled_d)

        return z, cl_loss


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