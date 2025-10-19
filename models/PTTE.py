import torch
import torch.nn as nn
from models.image_models.vit import ViTEncoder
from models.TemporalEncoder import TemporalEncoder
from utils.util import require
import copy
import torch.nn.functional as F

class PTTE(nn.Module):
    def __init__(self, image_out_dim=128, model_name="transformer", d_model=512, num_linear_stacks=2,**kwargs):
        super().__init__()
        # always FeatEncoderBlk -> TemporalEncoderBlk -> DecoderBlk (MLP?)
        self.feature_encoder_blk = FeatureEncoderBlock(image_out_dim=image_out_dim,blk_out_dim=d_model,num_linear_stacks=num_linear_stacks,**kwargs)
        self.temporal_encoder_blk = TemporalEncoderBlock(model_name, d_model = d_model,**kwargs)
        self.refine_blk = RefinementBlock(d_model,**kwargs)
        self.regressor = nn.Linear(d_model,1)
    def pooling(self, seq, mask, pool_method="sum"):
        if pool_method == "sum":
            valid_mask = (~mask).float().unsqueeze(-1)  # [B,T,1]
            return (seq * valid_mask).sum(dim=1) 
        else:
            raise NotImplementedError(f"Pooling method {pool_method} not implemented")
    def generate_padding_mask(self,lens):
        max_len = lens.max().item()
        B = len(lens) 
        mask = torch.arange(max_len).expand(B, max_len) >= lens.unsqueeze(1)
        assert mask.shape == torch.Size([B,max_len]) 
        return mask
    def forward(self, inputs, args):
        lens = inputs['lens']
        feats = self.feature_encoder_blk(inputs)
        padding_mask = self.generate_padding_mask(lens)
        sequence = self.temporal_encoder_blk(feats, padding_mask)
        refined = self.refine_blk(sequence,padding_mask)
        pooled = self.pooling(refined,padding_mask,pool_method=args.pool_method)
        output = self.regressor(pooled)
        return output
class TemporalEncoderBlock(nn.Module):
    def __init__(self, d_model=512, model_name="transformer",**kwargs):
        super().__init__()
        self.sequence = TemporalEncoder(model_name, d_model = d_model,**kwargs)
    def forward(self, feats: torch.Tensor, mask) -> torch.Tensor:
        sequence = self.sequence(feats,mask)
        return sequence
class FeatureEncoderBlock(nn.Module):
    def __init__(self, image_out_dim=128, blk_out_dim=512, num_linear_stacks=2,**kwargs):    
        super().__init__()
        # embedding modules --> attributes
        self.highwayemb = nn.Embedding(15,5, padding_idx=0)
        self.wkemb = nn.Embedding(8,3)
        self.dateemb = nn.Embedding(367, 10)
        self.timeemb = nn.Embedding(1441, 20)
        # spatial module
        self.gps_rep = nn.Linear(4, 16)
        # image module
        self.patch_encoder = ViTEncoder(out_dim=image_out_dim)
        # mapping to sequence module
        if num_linear_stacks == 1:
            self.representation = nn.Linear(image_out_dim+3+10+20, blk_out_dim)
        else:
            repr_hidden_dim = require('repr_hidden_dim', **kwargs)
            self.representation = nn.Sequential(
                nn.Linear(image_out_dim+3+10+20, repr_hidden_dim)
            )
            for _ in range(num_linear_stacks-1):
                self.representation.append(
                    nn.ReLU(),
                    nn.Linear(repr_hidden_dim, repr_hidden_dim)
                )
            self.representation.append(
                nn.ReLU(),
                nn.Linear(repr_hidden_dim, blk_out_dim)
            )
    
    def forward(self, inputs):
        feature = inputs['links']
        
        highwayrep = self.highwayemb(feature[:, :, 0].long()) # [B,T,5]
        wkrep = self.wkemb(feature[:, :, 3].long()) # [B,T,3]
        daterep = self.dateemb(feature[:, :, 4].long()) # [B,T,10]
        timerep = self.timeemb(feature[:, :, 5].long()) # [B,T,20]
        
        gpsrep = self.gps_rep(feature[:, :, 6:10]) # [B,T,16]
        
        patches = inputs['patch'] # [B,T,C,H,W]
        B, T, C,H,W = patches.shape  
        patchrep = self.patch_encoder(patches.reshape(B*T,C,H,W)) # [B*T,image_out_dim]
        patchrep = patchrep.reshape(B,T,-1) # [B,T,image_out_dim]
        
        datettimerep = torch.cat([wkrep, daterep, timerep], dim=-1) # [B,T,3+10+20=33]
        feats = torch.cat([feature[...,1:3],highwayrep, gpsrep, patchrep, datettimerep], dim=-1) # [B,T,2 + 5 + 16  + image_out_dim + 33]
        
        representation = self.representation(feats) # [B,T,blk_out_dim]
        
        return representation
        
class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
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
class RefinementLayer(nn.Module):
    def __init__(self, d_model, heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        return self.norm2(x)
class RefinementBlock(nn.Module):
    def __init__(self, d_model,**kwargs):
        super().__init__()
        N = require('N', **kwargs)
        heads = require('heads', **kwargs)
        d_ff = require('d_ff', **kwargs)
        dropout = require('dropout', **kwargs)
        self.layers = nn.ModuleList([
            RefinementLayer(d_model, dropout=dropout, heads=heads,d_ff=d_ff)
            for _ in range(N)
        ])
        self.norm = Norm(d_model)

    def forward(self, x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
