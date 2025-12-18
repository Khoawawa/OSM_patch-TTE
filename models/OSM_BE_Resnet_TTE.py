import torch

from models.base.ContextEncoder import ContextEncoder
from models.base.LayerNormGRU import LayerNormGRU
from models.base.VisualEncoder import FiLm_ResnetEncoder, CA_ResnetEncoder, ViTEncoder, ResnetEncoder
from models.base.RegionEncoder import RegionEncoder
import torch.nn.functional as F
import torch.nn as nn
import math
import copy
batch_first = False
class MulT_TTE(torch.nn.Module):
    def __init__(self,
                 seq_hidden_dim, seq_layer,
                 decoder_layer,
                 bert_attention_heads,bert_hidden_size,pad_token_id,bert_hidden_layers,vocab_size=27300):
        super().__init__()
        self.context_encoder = ContextEncoder(bert_attention_heads,bert_hidden_size,pad_token_id,bert_hidden_layers,vocab_size) # trip specific encoder
        self.temporal_block = LayerNormGRU(input_dim=self.context_encoder.hidden_size, hidden_dim=seq_hidden_dim, num_layers=seq_layer)
        self.decoder = Decoder(d_model=seq_hidden_dim, N=decoder_layer)
        self.pool_attn = nn.Linear(seq_hidden_dim,1)
        self.mlp = nn.Sequential(
            nn.Linear(seq_hidden_dim + 33, seq_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(seq_hidden_dim, 1)
        )
        self.delta_mlp = nn.Sequential(
            nn.Linear(seq_hidden_dim + 33, seq_hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(seq_hidden_dim // 2, 1)
        )
    def attention_pooling(self, decoder, valid_mask):
        # (B,T,seq_hidden_dim)
        scores = self.pool_attn(decoder).squeeze(-1)  # (B,T)
        scores = scores.masked_fill(valid_mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(attn_weights.unsqueeze(1), decoder).squeeze(1)  # (B, seq_hidden_dim)
        return pooled
    
    def forward(self, input_, args):
        # visual input
        valid_mask = input_['valid_mask']  # (B,T)
        representation, loss_1, (weekrep,daterep,timerep,timene_summary) = self.context_encoder(input_, args)

        representation = representation if batch_first else representation.transpose(0,1).contiguous() # (T,B,Res + Ctx)
        hiddens, _ = self.temporal_block(representation, seq_lens = input_['lens'].long())
        device_type = "cuda"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            decoder = self.decoder(hiddens.float(), input_['lens'].long()).to(hiddens.dtype)
        decoder = decoder if batch_first else decoder.transpose(0,1).contiguous() # (B,T,seq_hidden_dim)
        # attention pooling
        pooled_decoder = self.attention_pooling(decoder, valid_mask) # (B,seq_hidden_dim)

        pooled_decoder_cong = torch.cat([pooled_decoder, weekrep[:,0], daterep[:,0], timerep[:,0], timene_summary], dim=-1) # (B,seq_hidden_dim + 33 + 1)

        t_base = self.mlp(pooled_decoder) # (B,1)
        t_delta = F.softplus(self.delta_mlp(pooled_decoder_cong)) # (B,1)

        t_obs = t_base + t_delta

        return t_obs, loss_1, t_delta

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            dropout=dropout,
            batch_first=False   # because your x is (T,B,D)
        )

    def forward(self, x, lens):
        device = lens.device
        max_len = x.size(0)
        mask = torch.arange(max_len, device=device)[None, :] >= lens[:, None]
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads=1, dropout=0.1):
        super().__init__()
        # self.norm_1 = nn.LayerNorm(d_model) #
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        # self.dropout_1 = nn.Dropout(dropout)    #
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        #self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)


    def forward(self, x, len):
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, len))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Decoder(nn.Module):
    def __init__(self, d_model, N=3, heads=1, dropout=0.1):
        super().__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, lens):
        for i in range(self.N):
            x = self.layers[i](x, lens)
        return self.norm(x)


if __name__ == "__main__":
    pass