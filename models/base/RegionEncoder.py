import torch
import torch.nn as nn
import torch.nn.functional as F

class RegionEncoder(nn.Module):
    def __init__(self, input_dim,hidden_dim,output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.feature_input_dim = input_dim
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, region_features, offset,valid_mask):
        # region_features: [N_total, 1, F,7,7]
        # offset: [N_total, 1, 2]
        # valid_mask: [B, L] bool tensor
        region_features = region_features.squeeze(1)  # [N_total, F,7,7]
        offset = offset.squeeze(1)  # [N_total, 2]
        region_pooled = self.pool(region_features) # [N_total, F,1,1]
        region_pooled = region_pooled.squeeze(-1).squeeze(-1) # [N_total, F]
        region_cat = torch.cat([region_pooled, offset], dim=-1) # [N_total, F+2]
        region_emb = self.mlp(region_cat) # [N_total, O]
        ctx_return = torch.zeros(valid_mask.shape[0], valid_mask.shape[1],self.output_dim, device=region_emb.device, dtype=region_emb.dtype)
        ctx_return[valid_mask] = region_emb
        return ctx_return # [B,L, O]
