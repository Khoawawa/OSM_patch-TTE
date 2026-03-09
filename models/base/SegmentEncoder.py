import math

import torch
import torch.nn as nn
from models.base.PositionalEncoding import CyclicalTimeEncoding, PositionalEncoding1D
from models.ContrasiveModel import MoCoTrajContrasiveEncoder
class SegmentEncoder(nn.Module):
    def __init__(self, seq_hidden_dim, cl_queue_size,cl_hidden_dim=64, cl_head=8,cl_layer=4):
        super().__init__()

        self.highwayembed = nn.Embedding(15, 5, padding_idx=0)
        self.gpsembed = nn.Linear(4,16)
        
        self.weekembed = nn.Embedding(8, 3)
        self.dateembed = PositionalEncoding1D(10)
        self.timeembed = PositionalEncoding1D(d_model=20)
        
        cl_in_dim = 2 + 5 + 16
        self.cl = MoCoTrajContrasiveEncoder(cl_in_dim,cl_hidden_dim,cl_head,queue_size=cl_queue_size,num_layers=cl_layer)
        
        timene_dim = 3 + 10 + 20 + cl_hidden_dim
        self.timeneprojection = nn.Sequential(
            nn.Linear(timene_dim, timene_dim),
            nn.LeakyReLU(),
            nn.Linear(timene_dim, timene_dim)
        )
        self.time_norm = nn.LayerNorm(timene_dim)
        
        hidden_size = cl_in_dim + timene_dim
        self.represent = nn.Sequential(
            nn.Linear(hidden_size, seq_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(seq_hidden_dim, seq_hidden_dim)
        )
        
        self.mask_token = nn.Parameter(torch.zeros(1,1,cl_in_dim))
    def ada_merged_augment(self, x, mask_ratio=0.15):
        B, T, D = x.shape
        device = x.device
        
        merge_mask = torch.rand(B, T, device=device) < mask_ratio  # (B, T)
        
        merged = x[:, :-1] + x[:, 1:]
        
        x_aug = x.clone()
        x_aug[:, :-1] = torch.where(
            merge_mask.unsqueeze(-1),
            merged,
            x[:, :-1]
        )
        x_aug[:, 1:] = torch.where(
            merge_mask.unsqueeze(-1),
            merged,
            x[:, 1:]
        )
        return x_aug
    def point_masking(self, x, mask_ratio=0.15, mask_value=0.0):
        """
        x: (B, T, D)
        returns:
            masked_x: (B, T, D)
            mask:     (B, T)  True = masked
        """
        B, T, D = x.shape
        device = x.device

        # Bernoulli mask per time step
        mask = torch.rand(B, T, device=device) < mask_ratio  # (B, T)

        masked_x = x.clone()

        # Broadcast mask over feature dimension
        masked_x[mask] = mask_value
        # equivalent to:
        # masked_x = masked_x.masked_fill(mask.unsqueeze(-1), mask_value)

        return masked_x, mask
    def forward(self, inputs, args):
        # date
        dateinfo = inputs['dateinfo']
        
        weekrep   = self.weekembed(dateinfo[:, 0].long())
        daterep   = self.dateembed(dateinfo[:, 1])
        timerep   = self.timeembed(dateinfo[:, 2])
        datetimerep = torch.cat([weekrep, daterep, timerep], dim=-1)
        datetimerep = datetimerep.unsqueeze(1).expand(-1, inputs['links'].shape[1], -1) # (B,T,seq_hidden_dim)
        # spatial features
        feature = inputs['links']
        feature_lens = inputs['lens']
        highwayrep = self.highwayembed(feature[:, :, 0].long()) # 5
        
        gpsrep = torch.tanh(self.gpsembed(feature[:, :, 3:7].float())) # 16
        features = torch.cat([feature[..., 1:3], gpsrep,highwayrep], dim=-1) # 2 + 5 + 16 + 33
        # semantic features
        merged_features = self.ada_merged_augment(features)
        logits, labels, h = self.cl(features, merged_features, feature_lens, feature_lens)
        
        cl_loss = self.cl.loss(logits, labels)
        
        time_h = torch.cat([h, datetimerep], dim=-1)
        time_proj = self.timeneprojection(time_h) + time_h # (B,T,timene_dim)
        time_norm = self.time_norm(time_proj)
        
        features = torch.cat([features, time_norm], dim=-1)
        
        features = self.represent(features) # (B,T,seq_hidden_dim)
        
        return features, cl_loss, datetimerep
    