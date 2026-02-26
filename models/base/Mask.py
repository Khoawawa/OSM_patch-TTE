import torch
import torch.nn as nn

class PointMasking(nn.Module):
    def __init__(self,mask_rate):
        super().__init__()
        self.mask_rate = mask_rate
    def forward(self, x,padded_mask):
        # x: [B, L, D] 
        # padded_mask: [B, L] 
        B, L, D = x.shape
        device = x.device
        
        rand = torch.rand(B,L,device=device)
        pt_mask = (rand < self.mask_rate) & (padded_mask.bool()) # [B, L]
        
        x_masked = x.clone()
        x_masked[pt_mask] = 0.0 
        
        return x_masked, pt_mask