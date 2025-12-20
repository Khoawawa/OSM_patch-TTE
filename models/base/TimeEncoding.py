import torch
import torch.nn as nn
import math

class TimeEncoding(nn.Module):
    def __init__(self, out_dim, cycle):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.register_buffer('cycle', torch.tensor(cycle, dtype=torch.float32))

    def forward(self, minute):
        # minute: (B,)
        theta = 2 * math.pi * minute / self.cycle 
        enc = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
        return self.proj(enc)
