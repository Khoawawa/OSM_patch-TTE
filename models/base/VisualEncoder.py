import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPVisionModel
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

from models.base.CrossAttention import CrossAttention
class BE_Resnet_CA_Module(nn.Module):
    def __init__(self,adapter_hidden_dim=512,num_heads=8):
        super().__init__()
        self.resnet = BE_ResnetEncoder(adapter_hidden_dim=adapter_hidden_dim)
        
        self.diff_embs = nn.Linear(2, 8)
        self.gps_embs = nn.Linear(4, 16)
        
        kv_dim = self.diff_embs.out_features + self.gps_embs.out_features
        
        self.ca = CrossAttention(dim_q=adapter_hidden_dim,dim_kv=kv_dim, num_heads=num_heads)
    def forward(self, patches, patch_ids, valid_mask, gps, diff):
        patch_embs = self.resnet(patches, patch_ids, valid_mask) # (B, T, resnet_out)
        diff_embs = self.diff_embs(diff) # (B, T, 8)
        gps_embs = self.gps_embs(gps) # (B, T, 16)
        # cross attention
        kv_embs = torch.cat([diff_embs, gps_embs], dim=-1) # (B, T, 24)
        out = self.ca(patch_embs, kv_embs) # (B, T, resnet_out)
        return out, gps_embs # (B, T, resnet_out), (B, T, 16) for later
class BE_ResnetEncoder(nn.Module):
    def __init__(self,adapter_hidden_dim=512):
        super().__init__()
        self.resnet = BigEarthNetv2_0_ImageClassifier.from_pretrained("BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0")
        # get output dim of resnet
        self.output_dim = self.resnet.classifier.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1]) # remove classifier
        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(adapter_hidden_dim, self.output_dim)
        )
    def forward(self, patches, patch_ids, valid_mask):
        # patches: (U, C, H, W)
        # patch_ids: (total_link,)
        # valid_mask: (B, T)
        out = self.resnet(patches).flatten(1) # (U, resnet_out)
        out = self.adapter(out) # (U, resnet_out)
        B, T = valid_mask.shape
        D = out.shape[-1]
        
        gathered_patch_embs = out[patch_ids] # (total_link, resnet_out)
        output_grid = torch.zeros(B, T, D, device=out.device, dtype=out.dtype)
        output_grid[valid_mask] = gathered_patch_embs # (total_link, resnet_out) -> (B, T, resnet_out)
        # adapter
        output_grid = self.adapter(output_grid) # (B, T, resnet_out)
        return output_grid
class CLIPEncoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    def forward(self, patches, patch_ids, valid_mask):
        # patches: (U, C, H, W)
        # patch_ids: (total_link,)
        # valid_mask: (B, T)
        inputs = self.processor(patches, return_tensors="pt").to(patches.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            out = outputs.pooler_output # (U, vit_out)
        B, T = valid_mask.shape
        D = out.shape[-1]
        
        assert D == 512, "vit_out should be 512"
        
        gathered_patch_embs = out[patch_ids] # (total_link, vit_out)
        output_grid = torch.zeros(B, T, D, device=out.device, dtype=out.dtype)
        output_grid[valid_mask] = gathered_patch_embs # (total_link, vit_out) -> (B, T, vit_out)
        return output_grid
    
if __name__ == "__main__":
    device = 'cpu'
    U = 3
    C, H, W = 3, 224,224
    patches = torch.randn(U, C, H, W).to(device)
    
    total_link = 4
    patch_id = torch.tensor([0,1,2,0]).to(device)
    
    B, T = 2, 3
    valid_mask = torch.tensor([[1, 1, 0], [0, 1, 1]]).bool().to(device)
    
    clip_encoder = CLIPEncoder().to(device)
    out = clip_encoder(patches, patch_id, valid_mask)
    print(out.shape)
    print(out[0,0,:5])