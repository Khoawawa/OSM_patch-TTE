import torch

class CrossAttention(torch.nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=8,batch_first=True):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=dim_q, num_heads=num_heads, batch_first=batch_first)
        self.kv_projection = torch.nn.Linear(dim_kv, dim_q)
        
    def forward(self, visual_feat, context_feat, attn_mask=None):
        # implement cross attention mechanism
        context_proj = self.kv_projection(context_feat)
        fused_feat, _ = self.attn(visual_feat, context_proj, context_proj, attn_mask=attn_mask)
        return fused_feat
