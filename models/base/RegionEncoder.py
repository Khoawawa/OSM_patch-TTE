import torch
import torch.nn as nn

class PoiEncoder(nn.Module):
    def __init__(self,d_poi,d_segment_feat,num_heads,num_poi_types):
        super().__init__()
        # cell emb, region emb, sinusodial positional encoding, cross attention
        self.poi_type = nn.Embedding(num_poi_types,d_poi)
        d_kv = d_poi
        
        self.q_norm = nn.LayerNorm(d_segment_feat)
        self.k_norm = nn.LayerNorm(d_kv)
        self.v_norm = nn.LayerNorm(d_kv)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_segment_feat,num_heads=num_heads,kdim=d_kv,vdim=d_kv,batch_first=True)
        
        self.ffn_norm = nn.LayerNorm(d_segment_feat)
        self.ffn = nn.Sequential(
            nn.Linear(d_segment_feat, 2 * d_segment_feat),
            nn.GELU(),
            nn.Linear(2 * d_segment_feat, d_segment_feat)
        )
        self.register_buffer("rel_pe", None)
        
    def encode_cell_embedding(self,poi_matrix):
        # poi_matrix: (B,L,m*m,T)
        
        total_pois = torch.sum(poi_matrix, dim=-1, keepdim=True) # (B,L,m*m,1)
        weights = poi_matrix / total_pois.clamp_min(1.0)  # (B,L,m*m,T)
        type_embeddings = self.poi_type.weight  # (T,d_poi)
        e_cell = torch.matmul(weights, type_embeddings)  # (B,L,m*m,d_poi)
        
        return e_cell  # (B,L,m*m,d_poi)
    def get_2d_relative_pe(self,m,d_poi,device):
        coords = torch.arange(m,device=device) - (m // 2)
        
        delta_rows, delta_cols = torch.meshgrid(coords, coords, indexing='ij')
        
        delta_rows = delta_rows.flatten()  # (m*m,)
        delta_cols = delta_cols.flatten()  # (m*m,)

        pe = torch.zeros((m*m, d_poi), device=device)
        
        div_term = torch.exp(torch.arange(0, d_poi // 2, 2, device=device) * -(torch.log(torch.tensor(10000.0)) / (d_poi // 2)))
        pe[:, 0: d_poi // 2:2] = torch.sin(delta_rows.unsqueeze(-1) * div_term)
        pe[:, 1: d_poi // 2:2] = torch.cos(delta_cols.unsqueeze(-1) * div_term)
        pe[:, d_poi // 2::2] = torch.sin(delta_cols.unsqueeze(-1) * div_term)
        pe[:, d_poi // 2 + 1::2] = torch.cos(delta_rows.unsqueeze(-1) * div_term)
        
        return pe # (m*m,d_poi)
    def forward(self,segment_feat, poi_matrix, segment_mask,m):
        # segment_feat: (B,L,d_segment_feat)
        # poi_matrix: (B,L,m*m,T)
        B,L = segment_feat.size(0), segment_feat.size(1)
        c_e = self.encode_cell_embedding(poi_matrix)  # (B,L,m*m,d_poi)
        if self.rel_pe is None or self.rel_pe.size(0) != m*m:
            self.rel_pe = self.get_2d_relative_pe(m, c_e.size(-1), device=c_e.device)  # (m*m,d_poi)
        c_e = c_e + self.rel_pe  # (B,L,m*m,d_poi)
        # cross attention
        c_e_flatten = c_e.view(-1, m*m, c_e.size(-1))  # (B*L,m*m,d_poi)
        c_e_k_norm = self.k_norm(c_e_flatten)
        c_e_v_norm = self.v_norm(c_e_flatten)
        
        segment_feat_flatten = segment_feat.view(-1, segment_feat.size(-1)).unsqueeze(1)  # (B*L,1,d_segment_feat)
        segment_feat_q_norm = self.q_norm(segment_feat_flatten)

        ca_output = self.cross_attention(query=segment_feat_q_norm, key=c_e_k_norm, value=c_e_v_norm, need_weights=False)[0]  # (B*L,1,d_segment_feat)
        ca_output = ca_output.squeeze(1)  # (B*L,d_segment_feat)
        
        ca_output = ca_output.view(B, L, -1)  # (B,L,d_segment_feat)
        ca_output = self.ffn_norm(ca_output)
        ca_output = self.ffn(ca_output)  # (B,L,d_segment_feat)
        ca_output = ca_output * segment_mask.unsqueeze(-1)  # (B,L,d_segment_feat)
        
        segment_feat = segment_feat + ca_output  # (B,L,d_segment_feat)
        
        return segment_feat  # (B,L,d_segment_feat)
        