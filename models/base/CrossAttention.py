import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class LayerNormCA(torch.nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=8,batch_first=True):
        super().__init__()
        self.ca = CrossAttention(dim_q, dim_kv, num_heads=num_heads,batch_first=batch_first)
        self.ln_q = RMSNorm(dim_q)
        self.ln_kv = RMSNorm(dim_kv)
    def forward(self, query, key_value, attn_mask=None,key_padding_mask=None):
        # implement cross attention mechanism
        q_norm = self.ln_q(query)
        kv_norm = self.ln_kv(key_value)
        fused_feat = self.ca(q_norm, kv_norm,attn_mask=attn_mask,key_padding_mask=key_padding_mask)
        if torch.isnan(fused_feat).any():
            print("NaN detected: Output of self.ca is NaN!")
        
        return fused_feat + query # residual
    

class CrossAttention(torch.nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=8,batch_first=True):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=dim_q, num_heads=num_heads, batch_first=batch_first,kdim=dim_kv,vdim=dim_kv)
        
    def forward(self, query, key_value, attn_mask=None,key_padding_mask=None):
        # implement cross attention mechanism
        fused_feat, _ = self.attn(query, key_value, key_value, attn_mask=attn_mask,key_padding_mask=key_padding_mask)
        return fused_feat
    
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)

        return self.weight * x_normed.to(x.dtype)