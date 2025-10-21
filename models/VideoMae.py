import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEImageProcessor
from transformers import AutoModel, AutoImageProcessor
import numpy as np
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

if __name__ == "__main__":
    model = VideoMAEBackbone()
    video = list(torch.randn(2,16,3,224,224))
    out = model(video)
    print(out.shape)  # Expected: [2, num_patches, D]

class ViTEncoder(nn.Module):
    def __init__(self, model="google/vit-base-patch16-224-in21k", freeze=True):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model,use_fast=True)
        self.model = AutoModel.from_pretrained(model)
        self.hidden_size = self.model.config.hidden_size
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    def forward(self, x):
        # x: [B, T,C, H, W]
        b, t, c, h, w = x.shape
        images = x.view(b*t,c,h,w)
        
        inputs = self.processor(images=list(images), return_tensors="pt",).to(x.device)
        outputs = self.model(**inputs)
        
        return outputs.pooler_output.view(b,t,-1)  # [B,T,D]
