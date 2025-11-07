import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPVisionModel
from torchvision.models import resnet50
from models.base.CrossAttention import LayerNormCA
import torchvision
from models.base.FiLM import FilMAdapter
from models.base.PositionalEncoding import PositionalEncoding
batch_first=True

class BE_Resnet_CA_Module(nn.Module):
    def __init__(self,adapter_hidden_dim=512,num_heads=8):
        super().__init__()
        self.resnet = BE_ResnetEncoder(adapter_hidden_dim=adapter_hidden_dim)
        
        self.offset_pe = PositionalEncoding(128)
        self.gps_pe = PositionalEncoding(256)
        
        kv_dim = self.diff_embs.out_features + self.gps_embs.out_features
        
        self.ca = LayerNormCA(dim_q=self.resnet.output_dim,dim_kv=kv_dim, num_heads=num_heads,batch_first=batch_first)
        self.ca_heads = num_heads
    def forward(self, patches, patch_ids, valid_mask, gps, diff):
        patch_embs = self.resnet(patches, patch_ids, valid_mask) # (B, T, resnet_out)
        diff_embs = self.diff_embs(diff) # (B, T, 8)
        gps_embs = self.gps_embs(gps) # (B, T, 16)
        # cross attention
        # prepare query
        query_seq = patch_embs if batch_first else patch_embs.transpose(0,1).contiguous() # (T, B, resnet_out)
        query_seq = query_seq * valid_mask.unsqueeze(-1)
        # prepare kv
        kv_embs = torch.cat([diff_embs, gps_embs], dim=-1) # (B, T, 24)
        kv_seq = kv_embs if batch_first else kv_embs.transpose(0,1).contiguous() # (T, B, 24)
        kv_seq = kv_seq * valid_mask.unsqueeze(-1)
        # prepare mask
        B, T = valid_mask.shape
        
        attn_mask = torch.full((T,T), True, device=query_seq.device,dtype=torch.bool)
        attn_mask = attn_mask.fill_diagonal_(False) # (T, T)
        out = self.ca(query_seq, kv_seq, attn_mask=attn_mask) # (T, B, resnet_out)
        out = out if batch_first else out.transpose(0,1).contiguous() # (B, T, resnet_out)
        # out = torch.nan_to_num(out, nan=0.0) # (B, T, resnet_out)
        out = out * valid_mask.unsqueeze(-1)
        if torch.isnan(out).any(): 
            idx = torch.isnan(out).any(dim=-1)
            print("NaN detected in output at positions: ", idx.nonzero(as_tuple=True))
            
        if torch.isinf(out).any():
            print("Inf detected in output")
        return out, gps_embs # (B, T, resnet_out), (B, T, 16)
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
        
        self.offset_pe = PositionalEncoding(128)
        self.gps_pe = PositionalEncoding(256)
        self.output_dim = 128 + 256
        self.adapter = nn.Sequential(
            nn.Linear(self.resnet_out, adapter_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(adapter_hidden_dim, adapter_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(adapter_hidden_dim, self.resnet_out)
        )

        self.ca = LayerNormCA(dim_q=128+256,dim_kv=self.resnet_out, num_heads=8,batch_first=batch_first)

    def forward(self, patches, patch_ids, valid_mask, patch_center_gps, offsets):
        # patches: (U, C, H, W)
        # patch_ids: (total_link,)
        # valid_mask: (B, T)
        if not self.precomputed:
            out = self.resnet(patches) # (U, resnet_out,7,7)
        else:
            out = patches # (U, resnet_out,7,7)
            
        B, T = valid_mask.shape
        C,H,W = out.shape[1:]

        out = out.view(out.shape[0], C, H * W).permute(0, 2, 1) # (U, 49, resnet_out)
        
        gathered_patch_embs = out[patch_ids] # (total_link, 49, resnet_out)
        # adapter
        kv_embs = self.adapter(gathered_patch_embs) # (L, 49, resnet_out)
        # pe center and offset
        valid_gps = patch_center_gps[valid_mask] # (L,2)
        valid_offsets = offsets[valid_mask] # (L,2)

        gps_embs = self.gps_pe(valid_gps) # (L, 256)
        diff_embs = self.offset_pe(valid_offsets) # # (L, 128)
        
        query_embs = torch.cat([gps_embs, diff_embs], dim=-1) # (L, 384)
        query_embs = query_embs.unsqueeze(1) # (L, 1, 384)
        # cross attention
        attn_out = self.ca(query_embs,kv_embs) # (L, 1, 384)
        attn_out = attn_out.squeeze(1) # (L, 384)
        # map back to grid
        output_grid = torch.zeros(B,T,self.output_dim, device=out.device, dtype=out.dtype)
        output_grid[valid_mask] = attn_out
        
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
    
class BE_ResnetEncoder(nn.Module):
    def __init__(self,adapter_hidden_dim=512, adapter_layers=2):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        self.resnet = resnet50(weights=weights)
        # get output dim of resnet
        self.output_dim = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1]) # remove classifier
        # freezing
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # adapter
        in2hid = nn.Sequential(
            nn.Linear(self.output_dim, adapter_hidden_dim),
            nn.LeakyReLU()
        )

        hid2out = nn.Linear(adapter_hidden_dim, self.output_dim)
        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, adapter_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(adapter_hidden_dim, self.output_dim)
        )
    def forward(self, patches, patch_ids, valid_mask):
        # patches: (U, C, H, W)
        # patch_ids: (total_link,)
        # valid_mask: (B, T)
        patches = patches.float() / 255.0
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