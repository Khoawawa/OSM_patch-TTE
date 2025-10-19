import torch
import timm
from torch import nn 

class ViTEncoder(nn.Module):
    def __init__(self,vit_backbone="vit_tiny_patch16_224", out_dim = 128, unfreeze_block=2):
        super().__init__()
        
        self.vit = timm.create_model(vit_backbone,pretrained=True)
        
        # freeze all params
        for param in self.vit.parameters():
            param.requires_grad = False
        
        for block in self.vit.blocks[-unfreeze_block:]:
            for param in block.parameters():
                param.requires_grad = True
        
        #projection head
        in_feats = self.vit.head.in_features
        self.vit.head = nn.Linear(in_feats, out_dim)

    def forward(self, x):
        return self.vit(x)
    
    
if __name__ =='__main__':
    model = ViTEncoder()
    x = torch.randn(1,3,3,224,224)
    print(x.shape)
    x = x.reshape(1*3, 3,224, 224)
    print(x.shape)
    y = model(x)
    print(y.shape)
    y_reshaped = y.reshape(1,3,128)
    print(y_reshaped.shape)
        