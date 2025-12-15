import torch
import torch.nn as nn
import torch.nn.functional as F

class RegionEncoder(nn.Module):
    def __init__(self, input_dim,hidden_dim,output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.feature_input_dim = input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def compute_idw_weight(self, query_gps, region_centres):
        # query_gps: [B* L, 2]
        # region_centres: [B* L, N, 2]
        diff = query_gps[:,None,:] - region_centres
        dist_sq = torch.sum(diff ** 2, dim=-1)
        inv_dist = 1.0 / (dist_sq + 1e-12)
        weights = inv_dist / (torch.sum(inv_dist, dim=-1, keepdim=True) + 1e-12)
        return weights 
    def forward(self, region_features, valid_mask):
        # query_gps: [N_total, 2]
        # region_features: [N_total, 1, F]
        # region_centres: [N_total, 1, 2]
        # valid_mask: [B, L] bool tensor
        assert not torch.isnan(region_features).any(), "region_embs contains NaNs!"
        region_embs = self.mlp(region_features) # [N_total, O]
        assert not torch.isnan(region_embs).any(), "region_embs contains NaNs!"
        # softmax_wgts = self.compute_idw_weight(query_gps, region_centres)
        # ctx_embs = torch.sum(region_embs * softmax_wgts.unsqueeze(-1), dim=-2) 
        # assert not torch.isnan(region_embs).any(), "ctx_embs contains NaNs!"
        ctx_return = torch.zeros(valid_mask.shape[0], valid_mask.shape[1],self.output_dim, device=region_embs.device, dtype=region_embs.dtype)
        ctx_return[valid_mask] = region_embs
        return ctx_return # [B,L, O]
