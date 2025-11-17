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
    def __init__(self, adapter_hidden_dim=32, output_dim=512, use_precomputed=False):
        super().__init__()
        self.precomputed = use_precomputed
        self.output_dim = output_dim
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
             
        self.compress_linear = nn.Linear(self.resnet_out, self.output_dim)
        
        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, adapter_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_hidden_dim, self.output_dim)
        )
        self.adapter_norm = nn.LayerNorm(self.output_dim)
        self.pos_encoder = PositionalEncoding2D(d_model=self.output_dim)
        self.ca = LayerNormCA(d_model= self.output_dim, d_context = self.output_dim, num_heads=4,batch_first=batch_first)
        # precompute 
        # H = W = 7
        # center_i, center_j = H // 2, W // 2
        # grid_coords = torch.stack(
        #     torch.meshgrid(
        #         torch.arange(W) - center_j,
        #         torch.arange(H) - center_i, 
        #         indexing='xy'
        #     ), 
        #     dim=-1
        # ).reshape(-1, 2).float() # (49, 2)
        
        # grid_pe = self.pos_encoder(grid_coords)
        
        # self.register_buffer('grid_pe', grid_pe)
        
        # init weights for up projection adapter
        nn.init.constant_(self.adapter[3].weight, 0.0)
        nn.init.constant_(self.adapter[3].bias, 0.0)
        # self.ca_dropout = nn.Dropout(0.1)
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
        # pe = self.pos_encoder(torch.stack([dx, dy], dim=-1)) # (L, PE)
        # patch_vectors = patch_vectors + pe
        
        return patch_vectors.unsqueeze(1) # (L, 1, resnet_out + PE)    
    def forward(self, patches, patch_ids, valid_mask, offsets):
        # patches: (U, C, H, W)
        # patch_ids: (total_link,)
        # valid_mask: (B, T)
        # offsets: (total_link, 2)

        if not self.precomputed:
            out = self.resnet(patches) # (U, resnet_out,7,7)
            out = out.flatten(2).transpose(1,2) # (U, 49, resnet_out)
            
        else:
            out = patches.flatten(2).transpose(1,2) # (U, 49, resnet_out)
            
        B, T = valid_mask.shape
        
        gathered_patch_embs = out[patch_ids] # (L, 49, resnet_out)
        patches_compressed = self.compress_linear(gathered_patch_embs) # (L, 49, O)
        # get kv
        kv_with_pe = patches_compressed # baked-in positional encoding for kv
        # adapter
        adapter_in = self.adapter_norm(kv_with_pe)
        kv_patches = self.adapter(adapter_in) + kv_with_pe # (L, 49, O)
        # get offset and perform positional encoding
        # kv_pe = self.calc_offsets(adapter_out) # (49, 2)
        # kv_pe = self.pos_encoder(kv_pe) # (49, PE)
        # kv_pe = kv_pe.unsqueeze(0) # (1, 49, PE)
        # kv_pe = kv_pe.expand(adapter_out.shape[0],-1,-1) # (L, 49, PE)
        # kv_patches = torch.cat([adapter_out, kv_pe], dim=-1) # (L, 49, resnet_out + PE)
        # get query patch
        query_patch = self.get_offset_patch_embs(kv_patches, offsets) # (L, 1,O)
        # cross-attention
        attn_out = self.ca(query_patch,kv_patches) # (L, 1, O)
        # slice to remove PE
        attn_out = attn_out.squeeze(1) # (L, O)
        # attn_out = attn_out[:, :self.output_dim] # (L, O)
        # attn_out = self.ca_dropout(attn_out) # (L, O)
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
    