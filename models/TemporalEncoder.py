import torch.nn as nn
import torch
from utils.util import require

class TemporalEncoder(nn.Module):
    def __init__(self, model_name:str,d_model=512,**kwargs):
        '''
        
        '''
        super().__init__() 
        if model_name.lower() == "transformer":
            d_model = d_model
            nhead = require('nhead', **kwargs)
            dim_feedforward = require('dim_feedforward', **kwargs)
            dropout = require('dropout', **kwargs)
            activation = require('activation', **kwargs)
            num_layers = require('num_layers', **kwargs)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True
            )
            self.norm = nn.LayerNorm(d_model)
            self.model = nn.TransformerEncoder(encoder_layer,num_layers=num_layers,norm=self.norm)
        elif "itransformer" in model_name.lower():
            pass
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")
    
    def forward(self, x, mask):
        return self.model(x,src_key_padding_mask=mask)

if __name__ == "__main__":
    model = TemporalEncoder(model_name="transformer",d_model=512,nhead=8,dim_feedforward=2048,dropout=0.1,activation="relu",num_layers=6)
    x = torch.randn(1,10,512)
    y = model(x)
    print(y.shape)