import torch
import torch.nn.functional as F
import torch.nn as nn

class LayerNormCA(torch.nn.Module):
    def __init__(self, d_model,d_context, num_heads=8,batch_first=True):
        super().__init__()
        self.ln_q = RMSNorm(d_model)
        self.ln_kv = RMSNorm(d_context)
        self.ca = nn.MultiheadAttention(
            embed_dim=d_model,
            kdim=d_context,
            vdim=d_context,
            num_heads=num_heads,
            batch_first=batch_first
        )
    def forward(self, query, key_value, attn_mask=None,key_padding_mask=None):
        # implement cross attention mechanism
        q_norm = self.ln_q(query)
        kv_norm = self.ln_kv(key_value)
        fused_feat, _ = self.ca(q_norm, kv_norm, kv_norm, attn_mask=attn_mask,key_padding_mask=key_padding_mask)
        
        return fused_feat + query # residual
    
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)

        return self.weight * x_normed.to(x.dtype)