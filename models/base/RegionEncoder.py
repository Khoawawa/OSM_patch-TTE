import torch
import torch.nn as nn
import torch.nn.functional as F

class PoiEncoder(nn.Module):
    def __init__(self,d_segment_feat,m,num_poi_types):
        super().__init__()
        # cell emb, region emb, sinusodial positional encoding, cross attention
        self.m = m

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=num_poi_types, out_channels=d_segment_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=d_segment_feat, out_channels=d_segment_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )   

        # self.seg2gate = nn.Linear(d_segment_feat, m*m)
        self.seg_proj = nn.Linear(d_segment_feat, d_segment_feat)
        self.attn = nn.Conv2d(d_segment_feat, 1, kernel_size=1)
        # self.alpha = nn.Linear(2*d_segment_feat,d_segment_feat)
        self.interaction_norm = nn.LayerNorm(d_segment_feat*2)
        self.interaction_mlp = nn.Sequential(
            nn.Linear(d_segment_feat*2, d_segment_feat),
            nn.LeakyReLU(),
            nn.Linear(d_segment_feat, d_segment_feat)
        )

    def forward(self,segment_feat, poi_matrix, segment_mask,is_log = False):
        # segment_feat: (B,L,d_segment_feat)
        # poi_matrix: (B,L,m*m,T)
        log = dict() if is_log else None

        B,L,_,T = poi_matrix.shape

        poi_flatten = poi_matrix.view(B*L,self.m,self.m,T).permute(0,3,1,2)  # (B*L,T,m,m)
        poi_feature = self.cnn(poi_flatten)  # (B*L, d_segment_feat, m, m)
        # attention pooling
        seg = self.seg_proj(segment_feat).view(B*L,-1,1,1)  # (B*L,d_segment_feat,1,1)
        poi_cond = poi_feature + seg  # (B*L,d_segment_feat,m,m)
        score = self.attn(poi_cond) # (B*L,1,m,m)
        weight = torch.softmax(score.flatten(1), dim=-1).view(B*L,1,self.m,self.m)  # (B*L,1,m,m)
        pooled_poi = (poi_feature * weight).sum(dim=(2,3))
        pooled_poi = pooled_poi.view(B,L,-1)  # (B*L,d_segment_feat)
        
        concat_feature = torch.cat([segment_feat, pooled_poi], dim=-1)  # (B,L,d_segment_feat*2)
        concat_feature = self.interaction_norm(concat_feature)  # (B,L,d_segment_feat*2)
        interaction_feat = self.interaction_mlp(concat_feature)  # (B,L,d_segment_feat)
        
        segment_feat = segment_feat + interaction_feat  # (B,L,d_segment_feat)

        segment_feat = segment_feat * segment_mask.unsqueeze(-1).float()  # (B,L,d_segment_feat)
        
        if is_log:
            log["gate_mean"] = gated.mean().item()
            log["gate_std"]  = gated.std().item()
            log["gate_sat_low"]  = (gated < 0.05).float().mean().item()
            log["gate_sat_high"] = (gated > 0.95).float().mean().item()
        return segment_feat  if not is_log else (segment_feat, log)
        