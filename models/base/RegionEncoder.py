import torch
import torch.nn as nn
import torch.nn.functional as F

class PoiEncoder(nn.Module):
    def __init__(self,d_segment_feat,d_bottleneck,m,num_poi_types):
        super().__init__()
        self.m = m

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=num_poi_types + 2, out_channels=d_segment_feat // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=d_segment_feat // 2, out_channels=d_segment_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )   

        self.seg_proj = nn.Linear(d_segment_feat, d_segment_feat)
        self.attn = nn.Conv2d(d_segment_feat, 1, kernel_size=1)
        
        self.interaction_mlp = nn.Sequential(
            nn.LayerNorm(d_segment_feat),
            nn.Linear(d_segment_feat, d_bottleneck),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_bottleneck, d_segment_feat)
        )
        
        xs = torch.linspace(-1, 1, m)
        ys = torch.linspace(-1, 1, m)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coord = torch.stack([xx, yy], dim=0)  # (2, m, m)
        self.register_buffer("coord_grid", coord, persistent=False)  # (2, m, m)
        
    def forward(self,segment_feat, poi_matrix, segment_mask,is_log = False):
        # segment_feat: (B,L,d_segment_feat)
        # poi_matrix: (B,L,m*m,T)
        log = dict() if is_log else None

        B,L,_,T = poi_matrix.shape

        poi_flatten = poi_matrix.view(B*L,self.m,self.m,T).permute(0,3,1,2)  # (B*L,T,m,m)
        # TODO: add 2 channel for pe
        coord = self.coord_grid.unsqueeze(0).expand(B*L,-1,-1,-1)  # (B*L,2,m,m)
        poi_flatten = torch.cat([poi_flatten, coord], dim=1)  # (B*L,T+2,m,m)
        # local feature extraction
        poi_feature = self.cnn(poi_flatten)  # (B*L, d_segment_feat, m, m)
        # attention pooling
        seg = self.seg_proj(segment_feat).view(B*L,-1,1,1)  # (B*L,d_segment_feat,1,1)
        poi_cond = poi_feature + seg  # (B*L,d_segment_feat,m,m)
        score = self.attn(poi_cond) # (B*L,1,m,m)
        weight = torch.softmax(score.flatten(1), dim=-1).view(B*L,1,self.m,self.m)  # (B*L,1,m,m)
        pooled_poi = (poi_feature * weight).sum(dim=(2,3))
        # mlp interaction
        pooled_poi = pooled_poi.view(B,L,-1)  # (B,L,d_segment_feat)
        extracted_feature = self.interaction_mlp(pooled_poi)  # (B,L,d_segment_feat)
        # residual connection
        segment_feat = segment_feat + extracted_feature  # (B,L,d_segment_feat)
        # Apply mask
        segment_feat = segment_feat * segment_mask.unsqueeze(-1).float()  # (B,L,d_segment_feat)
        
        if is_log:
            # How concentrated is the attention? 
            # High value = focusing on one POI cell. Low = looking at all POIs.
            log["attn_max"] = weight.max().item()
            log["attn_entropy"] = -(weight * torch.log(weight + 1e-9)).sum(dim=(2,3)).mean().item()
        
        return segment_feat  if not is_log else (segment_feat, log)
        