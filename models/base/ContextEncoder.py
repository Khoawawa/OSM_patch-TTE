import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.highwayembed = nn.Embedding(15, 5, padding_idx=0)
        self.weekembed = nn.Embedding(8, 3)
        self.dateembed = nn.Embedding(367, 10)
        self.timeembed = nn.Embedding(1441, 20)
        self.timene_dim = 3 + 10 + 20
        self.timene = nn.Sequential(
            nn.Linear(self.timene_dim, self.timene_dim),
            nn.LeakyReLU(),
            nn.Linear(self.timene_dim, self.timene_dim)
        )
        self.hidden_size = 2 + 5 + self.timene_dim
    def forward(self, inputs, args):
        feature = inputs['links']

        # print("Lens: ", max(lens))
        highwayrep = self.highwayembed(feature[:, :, 0].long()) # 5
        weekrep = self.weekembed(feature[:, :, 3].long()) # 3
        daterep = self.dateembed(feature[:, :, 4].long())  # 10
        timerep = self.timeembed(feature[:, :, 5].long()) # 20
        datetimerep = torch.cat([weekrep, daterep, timerep], dim=-1) # 3 + 10 + 20 = 33
        
        timene = self.timene(datetimerep)+datetimerep
        features = torch.cat([feature[..., 1:3], highwayrep, timene], dim=-1)
        # 2 + 5 + (3 + 10 + 20 + bert_hiden_size)= 40 + bert_hiden_size
        return features, (weekrep,daterep,timerep)
        
if __name__ == "__main__":
    model = ContextEncoder(8, 512, 0, 4)
    