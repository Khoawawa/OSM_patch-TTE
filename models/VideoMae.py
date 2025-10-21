import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEImageProcessor
from transformers import AutoModel, AutoImageProcessor
import torchvision.models as models
import numpy as np
import timm
class VideoMAEBackbone(nn.Module):
    def __init__(self, model_name='MCG-NJU/videomae-base-finetuned-kinetics'):
        super().__init__()
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEModel.from_pretrained(model_name)

    def forward(self, x):
        # x: [B, T, C, H, W]
        videos = [v.permute(0, 2, 3, 1).cpu().numpy() for v in x]  # list of (T, H, W, C)
        inputs = self.processor(videos, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(x.device)
        outputs = self.model(pixel_values)
        return outputs.last_hidden_state

class ViTEncoder(nn.Module):
    def __init__(self, model="google/vit-base-patch16-224", freeze=True):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model,use_fast=True)
        self.model = AutoModel.from_pretrained(model)
        self.hidden_size = self.model.config.hidden_size
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    def forward(self, x, T, mask,device):
        # x: [[C, H, W] x B*T]
        
        b = int(len(x) / T)
        inputs = self.processor(images=x, return_tensors="pt",).to(device)
        outputs = self.model(**inputs)
        
        vit_embeds = outputs.pooler_output.view(b,T,-1)  # [B,T,D]
        return vit_embeds * mask.unsqueeze(-1)
        
class ResnetEncoder(nn.Module):
    def __init__(self, output_dim, pretrained=True, unfreeze_layer=0):
        super().__init__()
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(resnet.fc.in_features, output_dim)
        self.hidden_size = output_dim
        self.__freeze_layer(unfreeze_layer)
    def __freeze_layer(self, unfreeze_layer):
        for param in self.model.parameters():
            param.requires_grad = False
        if unfreeze_layer == 0:
            return

        
    def forward(self, x, lens): # [B*T(vary), C, H, W]
        feat = self.model(x) # [B*T, resnet_out,1, 1]
        feat = feat.squeeze(-1).squeeze(-1) # [B*T, resnet_out]
        feat = self.proj(feat)
        
        feats = []
        start = 0
        for len_ in lens:
            feats.append(feat[start: start + len_])
            start += len_

        padded_feats = nn.utils.rnn.pad_sequence(feats, batch_first=True)
        return  padded_feats
if __name__ == "__main__":
    pass
