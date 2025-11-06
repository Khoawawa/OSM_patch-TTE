import torch
import torch.nn as nn

class FilMAdapter(nn.Module):
    def __init__(self, patch_dim, gps_dim, offset_dim, adapter_hidden_dim=512, activation = "RELU"):
        super().__init__()
        self.fc1 = nn.Linear(patch_dim, adapter_hidden_dim)
        self.norm1 = nn.LayerNorm(adapter_hidden_dim)


        self.gps_proj = nn.Linear(gps_dim, 8)
        self.gps_gamma = nn.Linear(8, adapter_hidden_dim)
        self.gps_beta = nn.Linear(8, adapter_hidden_dim)

        self.fc2 = nn.Linear(adapter_hidden_dim, adapter_hidden_dim)
        self.norm2 = nn.LayerNorm(adapter_hidden_dim)

        self.offset_proj = nn.Linear(offset_dim, 8)
        self.offset_gamma = nn.Linear(8, adapter_hidden_dim)
        self.offset_beta = nn.Linear(8, adapter_hidden_dim)

        self.fc3 = nn.Linear(adapter_hidden_dim, patch_dim)
        self.norm3 = nn.LayerNorm(patch_dim)
        self.dropout = nn.Dropout(0.1)
        if activation == "RELU":
            self.activation = nn.LeakyReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError
        
        nn.init.zeros_(self.gps_gamma.weight)
        nn.init.zeros_(self.offset_gamma.weight)
        nn.init.zeros_(self.gps_gamma.bias)
        nn.init.zeros_(self.offset_gamma.bias)

    def forward(self, patch_embs, gps, offset, mask):
        # patch_embs: (B, N, patch_dim)
        # gps: (B, N, gps_dim)
        # offset: (B, N, offset_dim)
        # mask: (B, N)
        mask = mask.unsqueeze(-1)
        x = self.activation(self.norm1(self.fc1(patch_embs))) # adapt patch_embs to our dataset domain
        # dropout to prevent overfit
        x = self.dropout(x)
        # film conditioning
        # first stage set on global
        gps_embs = torch.tanh(self.gps_proj(gps))
        gamma_g, beta_g = self.gps_gamma(gps_embs), self.gps_beta(gps_embs)
        x = (1 + gamma_g) * x + beta_g
        x = x * mask # zero out masked patches
        x = x + self.activation(self.norm2(self.fc2(x)))
        # second stage
        offset_embs = torch.tanh(self.offset_proj(offset))
        gamma_o, beta_o = self.offset_gamma(offset_embs), self.offset_beta(offset_embs)
        x = (1 + gamma_o) * x + beta_o
        x = x * mask # zero out masked patches

        x = self.norm3(self.fc3(x))
        x = x * mask

        return x