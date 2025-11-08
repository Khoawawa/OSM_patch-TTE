import torch
import torch.nn.functional as F
import torch.nn as nn
# from flash_attn_interface import flash_attn_func

class LayerNormCA(torch.nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=8,batch_first=True):
        super().__init__()
        self.ca = CrossAttention(dim_q, dim_kv, num_heads=num_heads,batch_first=batch_first)
        self.ln = torch.nn.LayerNorm(dim_q)
    def forward(self, query, key_value, attn_mask=None,key_padding_mask=None):
        # implement cross attention mechanism
        fused_feat = self.ca(query, key_value,attn_mask=attn_mask,key_padding_mask=key_padding_mask)
        if torch.isnan(fused_feat).any():
            print("NaN detected: Output of self.ca is NaN!")
        return self.ln(fused_feat)
    

class CrossAttention(torch.nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=8,batch_first=True):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=dim_q, num_heads=num_heads, batch_first=batch_first,kdim=dim_kv,vdim=dim_kv)
        
    def forward(self, query, key_value, attn_mask=None,key_padding_mask=None):
        # implement cross attention mechanism
        with torch.autocast(device_type='cuda', enabled=False):
            fused_feat, _ = self.attn(query.float(), key_value.float(), key_value.float(), attn_mask=attn_mask,key_padding_mask=key_padding_mask)
        return fused_feat

# class FlashMHA(torch.nn.Module):
#     def __init__(self, dim_q, dim_kv, num_heads=8,batch_first=True):
#         super().__init__()

#         self.q_proj = nn.Linear(dim_q, dim_q)
#         self.kv_proj = nn.Linear(dim_kv, dim_q)

#         self.out_proj = nn.Linear(dim_q, dim_q)