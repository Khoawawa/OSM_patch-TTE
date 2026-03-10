import torch
import torch.nn as nn
from models.base.PositionalEncoding import PositionalEncodingIndex
from models.base.moco import MoCo
class MoCoTrajContrasiveEncoder(nn.Module):
    def __init__(self, in_dim, d_model, n_heads, dropout=0.1, num_layers=4, queue_size=1024, temperature=0.05):
        super().__init__()
        
        q_encoder = MSM(in_dim, d_model, n_heads, dropout, num_layers)
        k_encoder = MSM(in_dim, d_model, n_heads, dropout, num_layers)

        self.moco = MoCo(q_encoder, k_encoder, d_model, d_model, queue_size, temperature=temperature)
    
    def forward(self, traj1_emb, traj2_emb, traj1_len, traj2_len, traj2_merge_pad_mask):
        max_traj1_len = torch.max(traj1_len).item()
        max_traj2_len = torch.max(traj2_len).item()
        
        src_padding_mask1 = torch.arange(max_traj1_len, device=traj1_emb.device).unsqueeze(0) >= traj1_len.unsqueeze(1)  # (B, T1)
        src_padding_mask2 = torch.arange(max_traj2_len, device=traj2_emb.device).unsqueeze(0) >= traj2_len.unsqueeze(1)  # (B, T2)
        src_padding_mask2 = src_padding_mask2 | traj2_merge_pad_mask  # Combine with merge pad mask
        
        logits, labels, h = self.moco({'x': traj1_emb, 'src_key_padding_mask': src_padding_mask1},
                                    {'x': traj2_emb, 'src_key_padding_mask': src_padding_mask2})
        return logits, labels, h

    def loss(self, logits, labels):
        return self.moco.loss(logits, labels)
class MSM(nn.Module):
    def __init__(self, in_dim, d_model, n_heads, dropout=0.1, num_layers=4):
        super().__init__()
        self.pos_enc = PositionalEncodingIndex(d_model)
        self.in_proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dropout=dropout, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.after_norm = nn.LayerNorm(d_model)
    def forward(self, x, src_key_padding_mask=None):
        # x: (B, T, D) features after segment encoder
        x = self.in_proj(x)
        x = self.pos_enc(x, src_key_padding_mask)  # (B, T, D)
        
        h = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, D)
        h = self.after_norm(h)  # (B, T, D)
        
        return h
class TrajContrasiveEncoder(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, num_layers=4):
        super().__init__()
        self.pos_enc = PositionalEncodingIndex(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dropout=dropout, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.after_norm = nn.LayerNorm(d_model)
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    def forward(self, x, src_key_padding_mask=None, is_train=True):
        # x: (B, T, D) features after segment encoder
        x = self.pos_enc(x, src_key_padding_mask)  # (B, T, D)
        
        h = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, D)
        h = self.after_norm(h)  # (B, T, D)
        if is_train:
            if src_key_padding_mask is not None:
                mask = (~src_key_padding_mask).unsqueeze(-1)  # (B, T, 1)
                h_pooled = (h * mask).sum(dim=1) / mask.sum(dim=1)  # (B, D)
            else:
                h_pooled = h.mean(dim=1)  # (B, D)
            
            z = self.projector(h_pooled)  # (B, D)
        
        if is_train:
            return z, h
        
        return None, h
        
        