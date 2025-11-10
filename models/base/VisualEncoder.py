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
            self.resnet_out = 384 

        self.output_dim = 256
        self.adapter = nn.Sequential(
            nn.Linear(self.resnet_out, adapter_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_hidden_dim, self.output_dim)
        )

        self.ca = LayerNormCA(dim_q=self.output_dim,dim_kv=self.output_dim, num_heads=4,batch_first=batch_first)
        
        self.topk = topk
    def get_offset_patch_embs(self, patches, offsets):
        # patches: (L, 784, resnet_out)
        # offsets: (L, 2)
        _, U, D = patches.shape
        H = W = int(U ** 0.5)
        center_i = H // 2
        center_j = W // 2
        
        dx = offsets[:,0]
        dy = offsets[:,1]
        
        i_t = (center_i + dy).clamp(0,H-1)
        j_t = (center_j + dx).clamp(0,W-1)
        
        idx_flat = (i_t * W + j_t).long() # (L,)
        
        patch_vectors = torch.gather(
            patches, 1, idx_flat.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,D) # (L,) -> (L,1,1) 
        ) # (L,1,D)
        
        return patch_vectors
    def calc_cosine_sim(self, x1, x2):
        x1_norm = torch.nn.functional.normalize(x1, dim=-1)
        x2_norm = torch.nn.functional.normalize(x2, dim=-1)
        sim = torch.bmm(x1_norm, x2_norm.transpose(1,2)).squeeze(1)
        return sim # (L,784)
    
    def forward(self, patches, patch_ids, valid_mask, patch_center_gps, offsets):
        # patches: (U, C, H, W)
        # patch_ids: (total_link,)
        # valid_mask: (B, T)
        # patch_center_gps: (total_link, 2)
        # offsets: (total_link, 2)

        if not self.precomputed:
            out = self.resnet(patches) # (U, resnet_out,7,7)
        else:
            out = patches # (U, 784, resnet_out)
            
        B, T = valid_mask.shape
        
        gathered_patch_embs = out[patch_ids] # (L, 784, resnet_out)
        # adapter
        adapter_out = self.adapter(gathered_patch_embs) # (L, 784, O)
        # get query patch
        query_patch = self.get_offset_patch_embs(adapter_out, offsets) # (L, 1, O)
        # topk cosine similarity
        sim = self.calc_cosine_sim(query_patch, adapter_out) # (L, 784)
        _, indices = torch.topk(sim, self.topk, dim=1) # (L, topk)
        kv_patches = torch.gather(
            adapter_out, 1, indices.unsqueeze(-1).expand(-1,-1,adapter_out.shape[-1])
        ) # (L, topk, resnet_out)
        
        attn_out = self.ca(query_patch,kv_patches) # (L, 1, O)
        attn_out = attn_out.squeeze(1) # (L, O)
        
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
    