import math

import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM
from base.PositionalEncoding import PositionalEncoding1D

class ContextEncoder(nn.Module):
    def __init__(self, seq_hidden_dim,
                 bert_attention_heads, bert_hiden_size, pad_token_id, bert_hidden_layers, vocab_size=27300):
        super().__init__()
        self.bert_config = BertConfig(num_attention_heads = bert_attention_heads, hidden_size = bert_hiden_size, pad_token_id=pad_token_id,
                                      vocab_size=vocab_size, num_hidden_layers = bert_hidden_layers)
        self.seg_embedding_learning = BertForMaskedLM(self.bert_config)

        self.highwayembed = nn.Embedding(15, 5, padding_idx=0)
        self.gpsembed = nn.Linear(4,16)
        self.weekembed = nn.Embedding(8, 3)
        self.dateembed = PositionalEncoding1D(10)
        self.timeembed = PositionalEncoding1D(20)
        
        self.timene_dim = 3 + 10 + 20 + bert_hiden_size
        self.timene = nn.Sequential(
            nn.Linear(self.timene_dim, self.timene_dim),
            nn.LeakyReLU(),
            nn.Linear(self.timene_dim, self.timene_dim)
        )
        self.hidden_size = 2 + 5 + 16 + self.timene_dim
        self.represent = nn.Sequential(
            nn.Linear(self.hidden_size, seq_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(seq_hidden_dim, seq_hidden_dim)
        )

    def seg_embedding(self, x):
        bert_output = self.seg_embedding_learning(input_ids=x[0], encoder_attention_mask=x[1],  labels=x[2], output_hidden_states=True)

        return bert_output["loss"], bert_output["hidden_states"][4], bert_output["logits"]
    
    def forward(self, inputs, args):
        feature = inputs['links']

        # print("Lens: ", max(lens))
        highwayrep = self.highwayembed(feature[:, :, 0].long()) # 5
        
        doy = feature[:, :, 4].float() / 365.0 * 2 * math.pi
        minute = feature[:, :, 5].float() / 1440.0 * 2 * math.pi
        
        weekrep = self.weekembed(feature[:, :, 3].long()) # 3
        daterep = self.dateembed(doy)  # 10
        timerep = self.timeembed(minute) # 20
        
        gpsrep = self.gpsembed(feature[:, :, 6:10].float()) # 16
        
        datetimerep = torch.cat([weekrep, daterep, timerep], dim=-1) # 3 + 10 + 20 = 33

        loss_1, hidden_states, prediction_scores = self.seg_embedding([inputs['linkindex'], inputs['encoder_attention_mask'], inputs['mask_label']])
        # 
        timene_input = torch.cat([self.seg_embedding_learning.bert.embeddings.word_embeddings(inputs['rawlinks']), datetimerep], dim=-1)
        timene = self.timene(timene_input)+timene_input
        features = torch.cat([feature[..., 1:3], gpsrep,highwayrep, timene], dim=-1) # 2 + 5 + 16 + 33 + bert_hiden_size
        features = self.represent(features) # (B,T,seq_hidden_dim)
        return features, loss_1, (weekrep[:,0], daterep[:,0], timerep[:,0])
        
if __name__ == "__main__":
    model = ContextEncoder(8, 512, 0, 4)
    