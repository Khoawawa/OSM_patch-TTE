import torch
import torch.nn as nn
class AdaRMSNorm(nn.Module):
    def __init__(self, d_model, d_context):
        super().__init__()
        self.rms_norm = RMSNorm(d_model) # Using the RMSNorm you wrote earlier
        # This projects the temporal context into scale and shift parameters
        self.to_scale_shift = nn.Linear(d_context, d_model * 2)

    def forward(self, x, context):
        """
        x: (B, T, d_model) - The Road/Route features
        context: (B, d_context) - The output from your TimeEncoding
        """
        # Generate scale (gamma) and shift (beta) from the temporal context
        # We initialize the shift to 0 and scale to 1 (via 1 + scale) for stability
        scale_shift = self.to_scale_shift(context).unsqueeze(1) # (B, 1, 2*d_model)
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