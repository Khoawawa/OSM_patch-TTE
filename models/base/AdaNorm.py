import torch
import torch.nn as nn
class AdaRMSNorm(nn.Module):
    def __init__(self, d_model, d_context):
        super().__init__()
        self.rms_norm = RMSNorm(d_model)
        self.context_norm = nn.LayerNorm(d_context)
        self.to_scale_shift = nn.Linear(d_context, d_model * 2)
        nn.init.zeros_(self.to_scale_shift.bias)
        nn.init.zeros_(self.to_scale_shift.weight)

    def forward(self, x, context):
        """
        x: (B, T, d_model)
        context: (B, T, d_context)
        """
        context = self.context_norm(context)
        scale_shift = self.to_scale_shift(context) # (B, T, 2*d_model)
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        return self.rms_norm(x) * (1 + scale) + shift
    
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)

        return self.weight * x_normed.to(x.dtype)
    