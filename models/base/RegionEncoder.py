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
            nn.ReLU(),
            nn.Conv2d(in_channels=d_segment_feat, out_channels=d_segment_feat, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.seg2gate = nn.Linear(d_segment_feat, m*m)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self,segment_feat, poi_matrix, segment_mask,is_log = False):
        # segment_feat: (B,L,d_segment_feat)
        # poi_matrix: (B,L,m*m,T)
        log = dict() if is_log else None

        B,L,_,T = poi_matrix.shape

        poi_flatten = poi_matrix.view(B*L,self.m,self.m,T).permute(0,3,1,2)  # (B*L,T,m,m)
        poi_feature = self.cnn(poi_flatten)  # (B*L, d_segment_feat, m, m)

        segment_flatten = segment_feat.view(B*L,segment_feat.size(-1))  # (B*L,d_segment_feat)
        gate = self.seg2gate(segment_flatten).view(B*L,1,self.m,self.m)  # (B*L,1,m,m)
        gate = torch.sigmoid(gate)  # (B*L,1,m,m)

        gated = poi_feature * gate  # (B*L, d_segment_feat, m, m)
        poi_vec = self.pool(gated).flatten(1)  # (B*L, d_segment_feat)
        poi_vec = poi_vec.view(B,L,-1)  # (B,L,d_segment_feat)
        segment_feat = segment_feat + self.scale * poi_vec  # (B,L,d_segment_feat)

        segment_feat = segment_feat * segment_mask.unsqueeze(-1).float()  # (B,L,d_segment_feat)
        
        if is_log:
            log["gate_mean"] = gated.mean().item()
            log["gate_std"]  = gated.std().item()
            log["gate_sat_low"]  = (gated < 0.05).float().mean().item()
            log["gate_sat_high"] = (gated > 0.95).float().mean().item()
        return segment_feat  if not is_log else (segment_feat, log)
        