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
        
        self.pad_token = nn.Parameter(torch.zeros(1,1,cl_in_dim))
    def apply_merge(x, start_mask, pad_mask, pad_token):
        B, T, _ = x.shape

        span_mask = start_mask | pad_mask
        span_mask_f = span_mask.float().unsqueeze(-1)

        summed = (x * span_mask_f).sum(dim=1)
        counts = span_mask_f.sum(dim=1).clamp(min=1e-6)

        merged = summed / counts

        merged_expand = merged.unsqueeze(1).expand(-1, T, -1)

        x_aug = torch.where(start_mask.unsqueeze(-1), merged_expand, x)
        x_aug = torch.where(pad_mask.unsqueeze(-1), pad_token, x_aug)

        return x_aug
    def forward(self, inputs):
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
        merge_start_mask, merge_pad_mask = inputs['merge_mask']
        merged_features = self.apply_merge(features, merge_start_mask, merge_pad_mask, self.pad_token)
        logits, labels, h = self.cl(features, merged_features, feature_lens, feature_lens, merge_pad_mask)
        
        cl_loss = self.cl.loss(logits, labels)
        
        time_h = torch.cat([h, datetimerep], dim=-1)
        time_proj = self.timeneprojection(time_h) + time_h # (B,T,timene_dim)
        time_norm = self.time_norm(time_proj)
        
        features = torch.cat([features, time_norm], dim=-1)
        
        features = self.represent(features) # (B,T,seq_hidden_dim)
        
        return features, cl_loss, datetimerep
    