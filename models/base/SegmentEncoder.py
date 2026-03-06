import math

import torch
import torch.nn as nn
from models.base.PositionalEncoding import CyclicalTimeEncoding, PositionalEncoding1D

class SegmentEncoder(nn.Module):
    def __init__(self, seq_hidden_dim):
        super().__init__()

        self.highwayembed = nn.Embedding(15, 5, padding_idx=0)
        self.gpsembed = nn.Linear(4,16)
        
        self.weekembed = nn.Embedding(8, 3)
        self.dateembed = PositionalEncoding1D(10,period=365.0)
        self.timeembed = PositionalEncoding1D(d_model=20)
        
        self.timene_dim = 3 + 10 + 20
        self.timeneprojection = nn.Sequential(
            nn.Linear(self.timene_dim, self.timene_dim),
            nn.LeakyReLU(),
            nn.Linear(self.timene_dim, seq_hidden_dim)
        )
        
        self.hidden_size = 2 + 5 + 16
        
        self.represent = nn.Sequential(
            nn.Linear(self.hidden_size, seq_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(seq_hidden_dim, seq_hidden_dim)
        )
    def forward(self, inputs):
        #
        dateinfo = inputs['dateinfo']
        
        weekrep   = self.weekembed(dateinfo[:, 0].long())
        daterep   = self.dateembed(dateinfo[:, 1])
        timerep   = self.timeembed(dateinfo[:, 2])
        datetimerep = torch.cat([weekrep, daterep, timerep], dim=-1)
        
        time_proj = self.timeneprojection(datetimerep) # (B, T, seq_hidden_dim)
        time_proj = time_proj.unsqueeze(1)
        
        feature = inputs['links']
        highwayrep = self.highwayembed(feature[:, :, 0].long()) # 5
        
        gpsrep = torch.tanh(self.gpsembed(feature[:, :, 3:7].float())) # 16
        
        features = torch.cat([feature[..., 1:3], gpsrep,highwayrep], dim=-1) # 2 + 5 + 16 + 33
        features = self.represent(features) # (B,T,seq_hidden_dim)
        
        features = features + time_proj # (B,T,seq_hidden_dim)
        
        return features, datetimerep
    