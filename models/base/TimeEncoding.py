import torch
import torch.nn as nn
import math

class TimeEncoding(nn.Module):
    def __init__(self, out_dim, cycle=1440, num_freqs=None):
        super().__init__()
        if num_freqs is None:
            num_freqs = out_dim // 2
        
        self.num_freqs = num_freqs
        self.base_cycle = float(cycle)

        freqs = torch.arange(1, num_freqs + 1)
        self.register_buffer('freqs', freqs.float())

        self.linear = (
            nn.Linear(num_freqs * 2, out_dim)
            if out_dim != num_freqs * 2
            else nn.Identity()
        )

    def forward(self, minute):
        minute = minute.float()

        # Accept (B,) or (B,T)
        if minute.dim() == 1:
            minute = minute.unsqueeze(1)  # (B,1)

        # minute: (B,T)
        # freqs: (F,)
        angles = (
            minute.unsqueeze(-1)              # (B,T,1)
            * self.freqs.view(1, 1, -1)        # (1,1,F)
            * (2 * math.pi)
            / self.base_cycle
        )                                      # (B,T,F)

        emb = torch.cat(
            [torch.sin(angles), torch.cos(angles)],
            dim=-1
        )                                      # (B,T,2F)

        return self.linear(emb) 
