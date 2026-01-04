import torch
import torch.nn.functional as F
import torch.nn as nn

class AdaRMSNorm(nn.Module):
    def __init__(self, d_model, time_dim, eps=1e-8):
        super().__init__()
        self.norm = RMSNorm(d_model, eps)
        self.linear = nn.Linear(time_dim, 2 * d_model)

    def forward(self, h, time_emb):
        h_norm = self.norm(h)
        gamma, beta = self.linear(time_emb).chunk(2, dim=-1)
        gamma = gamma + 1.0
        return h_norm * gamma + beta
    
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)

        return self.weight * x_normed.to(x.dtype)