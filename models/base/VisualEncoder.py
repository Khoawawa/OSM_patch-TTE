import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPVisionModel
from torchvision.models import resnet50
from models.base.CrossAttention import LayerNormCA
import torchvision
from models.base.FiLM import FilMAdapter
from models.base.PositionalEncoding import PositionalEncoding2D
batch_first=True

class CA_ResnetEncoder(nn.Module):
    def __init__(self, adapter_hidden_dim=512, use_precomputed=False):
        super().__init__()
        self.precomputed = use_precomputed

        if not use_precomputed:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            self.resnet = resnet50(weights=weights)
            self.resnet_out = self.resnet.fc.in_features
            self.resnet = nn.Sequential(*list(self.resnet.children())[:-2]) # remove classifier and pool for feature
            # freezing
            for param in self.resnet.parameters():
                param.requires_grad = False
        else:
            self.resnet_out = 2048  
        
        self.offset_pe = PositionalEncoding2D(128)
        self.gps_pe = PositionalEncoding2D(256)
        self.output_dim = 128 + 256
        self.adapter = nn.Sequential(
            nn.Linear(self.resnet_out, adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_hidden_dim, self.resnet_out)
        )

        self.ca = LayerNormCA(dim_q=128+256,dim_kv=self.resnet_out, num_heads=8,batch_first=batch_first)

    def forward(self, patches, patch_ids, valid_mask, patch_center_gps, offsets):
        # patches: (U, C, H, W)
        # patch_ids: (total_link,)
        # valid_mask: (B, T)
        # patch_center_gps: (total_link, 2)
        # offsets: (total_link, 2)

        if not self.precomputed:
            out = self.resnet(patches) # (U, resnet_out,7,7)
        else:
            out = patches # (U, resnet_out,7,7)
            
        B, T = valid_mask.shape
        C,H,W = out.shape[1:]

        out = out.view(out.shape[0], C, H * W).permute(0, 2, 1) # (U, 49, resnet_out)
        
        gathered_patch_embs = out[patch_ids] # (total_link, 49, resnet_out)
        # adapter
        adapter_out = self.adapter(gathered_patch_embs) # (L, 49, resnet_out)
        kv_embs = adapter_out # (L, 49, 2*resnet_out)
        gps_embs = self.gps_pe(patch_center_gps) # (L, 256)
        diff_embs = self.offset_pe(offsets) # # (L, 128)

        query_embs = torch.cat([gps_embs, diff_embs], dim=-1) # (L, 384)
        query_embs = query_embs.unsqueeze(1) # (L, 1, 384)
        # cross attention
        if (query_embs.abs() > 1e4).any() or (kv_embs.abs() > 1e4).any():
            print("Extreme values detected (>1e4)")
            
        attn_out = self.ca(query_embs,kv_embs) # (L, 1, 384)
        attn_out = attn_out.squeeze(1) # (L, 384)
        assert torch.isnan(attn_out).any() == False, "nan in attn_out"

        # map back to grid
        output_grid = torch.zeros(B,T,self.output_dim, device=out.device, dtype=out.dtype)
        output_grid[valid_mask] = attn_out.to(output_grid.dtype)
        
        return output_grid
    

class FiLm_ResnetEncoder(nn.Module):
    def __init__(self, adapter_hidden_dim=512,activation="RELU", use_precomputed=False):
        super().__init__()
        self.precomputed = use_precomputed

        if not use_precomputed:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            self.resnet = resnet50(weights=weights)
            # get output dim of resnet
            self.output_dim = self.resnet.fc.in_features
            self.resnet = nn.Sequential(*list(self.resnet.children())[:-2]) # remove classifier
            # freezing
            for param in self.resnet.parameters():
                param.requires_grad = False
        else:
            self.output_dim = 2048
        # adapter
        self.adapter = FilMAdapter(self.output_dim,2,2,adapter_hidden_dim,activation=activation)
    def forward(self, patches, patch_ids, valid_mask, patch_center_gps, offsets):
        # patches: (U, C, H, W)
        # patch_ids: (total_link,)
        # valid_mask: (B, T)
        if not self.precomputed:
            out = self.resnet(patches).flatten(1) # (U, resnet_out)
        else:
            out = patches.flatten(1) # (U, resnet_out)
            
        B, T = valid_mask.shape
        D = out.shape[-1]
        
        gathered_patch_embs = out[patch_ids] # (total_link, resnet_out)
        output_grid = torch.zeros(B, T, D, device=out.device, dtype=out.dtype)
        output_grid[valid_mask] = gathered_patch_embs # (total_link, resnet_out) -> (B, T, resnet_out)
        
        output = self.adapter(output_grid, patch_center_gps, offsets, valid_mask)
        # adapter
        return output
    