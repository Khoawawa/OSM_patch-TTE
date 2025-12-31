import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM
from torch.nn.functional import tanh
from models.base.TimeEncoding import TimeEncoding

class ContextEncoder(nn.Module):
    def __init__(self,
                 bert_attention_heads, bert_hiden_size, pad_token_id, bert_hidden_layers, vocab_size=27300):
        super().__init__()
        self.bert_config = BertConfig(num_attention_heads = bert_attention_heads, hidden_size = bert_hiden_size, pad_token_id=pad_token_id,
                                      vocab_size=vocab_size, num_hidden_layers = bert_hidden_layers)
        self.seg_embedding_learning = BertForMaskedLM(self.bert_config)

        self.highwayembed = nn.Embedding(15, 5, padding_idx=0)
        
        self.gpsembed = nn.Linear(4,16)

        self.weekembed = TimeEncoding(4, cycle=7)
        self.dateembed = nn.Embedding(367, 8)
        self.timeembed = TimeEncoding(16, cycle=1440)
        self.datetimerep_size = 8 + 16 + 64  # week + date + time
        self.hidden_size = 2 + 5 + 16 + bert_hiden_size  # link length + highway type + gps + bert hidden size

    def seg_embedding(self, x):
        bert_output = self.seg_embedding_learning(input_ids=x[0], encoder_attention_mask=x[1],  labels=x[2], output_hidden_states=True)

        return bert_output["loss"], bert_output["hidden_states"][4], bert_output["logits"]
    
    def forward(self, inputs, args):
        feature = inputs['links']
        B, T = feature.shape[:2]
        # print("Lens: ", max(lens))
        highwayrep = self.highwayembed(feature[:, :, 0].long()) # 5
        #gps encoding
        gpsrep = tanh(self.gpsembed(feature[:, :, 6:10].float())) # 16
        # global time encoding
        weekrep = self.weekembed(feature[:, :, 3].long()) # 8
        daterep = self.dateembed(feature[:, :, 4].long())  # 16
        timerep = self.timeembed(feature[:, :, 5].long()) # 64
        datetimerep = torch.cat([weekrep, daterep, timerep], dim=-1) # 8 + 16 + 64 = 88

        loss_1, bert_hidden_states,_ = self.seg_embedding([inputs['linkindex'], inputs['encoder_attention_mask'], inputs['mask_label']]) 

        features = torch.cat([feature[..., 1:3], gpsrep,highwayrep, bert_hidden_states], dim=-1) 
        
        return features, loss_1, datetimerep
        
if __name__ == "__main__":
    model = ContextEncoder(8, 512, 0, 4)
    