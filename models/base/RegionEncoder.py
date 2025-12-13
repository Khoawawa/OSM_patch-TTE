import torch
import torch.nn as nn
import torch.nn.functional as F

class RegionEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.output_dim = input_dim
        self.feature_input_dim = input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
    def compute_idw_weight(self, query_gps, region_centres):
        # query_gps: [B, L, 2]
        # region_centres: [B, L, N, 2]
        dist_sq = torch.sum((query_gps.unsqueeze(2) - region_centres)**2, dim=-1)
        idw_weights = 1.0 / (dist_sq + 1e-6)
        softmax_weights = F.softmax(idw_weights, dim=-1)
        return softmax_weights 
    def forward(self, query_gps,region_centres, region_features, valid_mask):
        # query_gps: [B, 2]
        # region_bbox: [B, N, 4]
        # region_features: [B, N, F]
        region_embs = self.mlp(region_features) + region_features
        
        softmax_wgts = self.compute_idw_weight(query_gps, region_centres)
        ctx_embs = torch.sum(region_embs * softmax_wgts.unsqueeze(-1), dim=-2) 

        ctx_return = torch.zeros(valid_mask.shape[0], valid_mask.shape[1],self.output_dim, device=ctx_embs.device, dtype=ctx_embs.dtype)
        ctx_return[valid_mask] = ctx_embs
        return ctx_return # [B,L, O]

if __name__ == "__main__":
    query_gps = torch.randn(2, 3, 2)
    region_bbox = torch.randn(2, 3, 4,4)
    region_features = torch.randn(2, 3, 4, 6)
    region_encoder = RegionEncoder(6,6)
    print(region_encoder(query_gps, region_bbox, region_features).shape)