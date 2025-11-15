import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPVisionModel
from torchvision.models import resnet50
from models.base.CrossAttention import LayerNormCA
import torchvision
from models.base.FiLM import FilMAdapter
from models.base.PositionalEncoding import PositionalEncoding2D
batch_first=True

class CA_ResnetEncoder(nn.Module):
    def __init__(self, adapter_hidden_dim=512, topk=64, use_precomputed=False):
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

        self.output_dim = self.resnet_out
        self.adapter = nn.Sequential(
            nn.Linear(self.resnet_out, adapter_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_hidden_dim, self.resnet_out)
        )
        self.adapter_norm = nn.LayerNorm(self.resnet_out)
        # self.pos_encoder = PositionalEncoding2D(16)
        self.ca = LayerNormCA(d_model= self.resnet_out, d_context = self.resnet_out, num_heads=4,batch_first=batch_first)
        # self.ca_dropout = nn.Dropout(0.1)
        self.topk = topk
    def calc_offsets(self, patches):
        _, U, _ = patches.shape
        H = W = int(U ** 0.5)
        center_i = H // 2
        center_j = W // 2
        rel_offsets = torch.stack(
            torch.meshgrid(
                torch.arange(W,device=patches.device) - center_j,
                torch.arange(H,device=patches.device)-center_i, 
                indexing='xy'
            ), 
            dim=-1
        ).reshape(-1,2) # (49, 2)
        
        return rel_offsets
    def get_offset_patch_embs(self, patches, offsets):
        # patches: (L, 784, resnet_out + PE)
        # offsets: (L, 2)
        L, U, D = patches.shape
        H = W = int(U ** 0.5)
        center_i = H // 2
        center_j = W // 2
        # convert normalized offsets to pixel offsets
        dx = offsets[:,0] * (W//2)
        dy = offsets[:,1] * (H//2)
        
        i_t = (center_i + dy).clamp(0,H-1) # -3 -> 3
        j_t = (center_j + dx).clamp(0,W-1) # -3 -> 3
        
        i_t = torch.floor(i_t) # i_t.floor()
        j_t = torch.floor(j_t) # j_t.floor()

        idx_flat = (i_t * W + j_t).long() # (L,)
        
        patch_vectors = patches[torch.arange(L), idx_flat] # (L, resnet_out)

        # offset_for_pe = torch.stack([dx, dy], dim=-1)        
        # pe = self.pos_encoder(offset_for_pe)

        # patch_vectors = torch.cat([patch_vectors, pe], dim=-1)

        return patch_vectors.unsqueeze(1) # (L, 1, resnet_out + PE)    
    def forward(self, patches, patch_ids, valid_mask, offsets):
        # patches: (U, C, H, W)
        # patch_ids: (total_link,)
        # valid_mask: (B, T)
        # offsets: (total_link, 2)

        if not self.precomputed:
            out = self.resnet(patches) # (U, resnet_out,7,7)
        else:
            out = patches.flatten(2).transpose(1,2) # (U, 49, resnet_out)
            
        B, T = valid_mask.shape
        
        gathered_patch_embs = out[patch_ids] # (L, 49, resnet_out)
        # adapter_in = self.adapter_norm(gathered_patch_embs)a
        # adapter
        kv_patches = self.adapter(gathered_patch_embs) + gathered_patch_embs # (L, 49, O)
        # get offset and perform positional encoding
        # kv_pe = self.calc_offsets(adapter_out) # (49, 2)
        # kv_pe = self.pos_encoder(kv_pe) # (49, PE)
        # kv_pe = kv_pe.unsqueeze(0) # (1, 49, PE)
        # kv_pe = kv_pe.expand(adapter_out.shape[0],-1,-1) # (L, 49, PE)
        # kv_patches = torch.cat([adapter_out, kv_pe], dim=-1) # (L, 49, resnet_out + PE)
        # get query patch
        query_patch = self.get_offset_patch_embs(kv_patches, offsets) # (L, 1, resnet_out + PE)
        # cross-attention
        attn_out = self.ca(query_patch,kv_patches) # (L, 1, O + PE)
        # slice to remove PE
        attn_out = attn_out.squeeze(1) # (L, O + PE)
        # attn_out = attn_out[:, :self.output_dim] # (L, O)
        # attn_out = self.ca_dropout(attn_out) # (L, O)
        assert torch.isnan(attn_out).any() == False, "nan in attn_out"

        # map back to grid
        output_grid = torch.zeros(B,T,self.output_dim, device=out.device, dtype=out.dtype)
        output_grid[valid_mask] = attn_out.to(output_grid.dtype)
        
        return output_grid # (B, T, O)
    

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
    