import torch
import torch.nn as nn
from models.base.PositionalEncoding import PositionalEncodingIndex

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
        
        