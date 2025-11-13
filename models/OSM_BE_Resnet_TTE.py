import torch

from models.base.ContextEncoder import ContextEncoder
from models.base.LayerNormGRU import LayerNormGRU
from models.base.VisualEncoder import FiLm_ResnetEncoder, CA_ResnetEncoder
import torch.nn.functional as F
import torch.nn as nn
import math
import copy
batch_first = False
# from a abstract view point 
# there is 2 stream
# - visual stream
# - context stream
# every stream have their own block 
# then they are fed into a cross attention fusion block
# then go into mlp to extract the time#
class OSM_BER_TTE(torch.nn.Module):
    def __init__(self, adapter_hidden_dim,topk,use_precomputed,
                 seq_hidden_dim, seq_layer,
                 decoder_layer,
                 bert_attention_heads,bert_hidden_size,pad_token_id,bert_hidden_layers,vocab_size=27300):
        super().__init__()
        self.visual_encoder = CA_ResnetEncoder(adapter_hidden_dim,use_precomputed=use_precomputed)
        visual_out_dim = self.visual_encoder.output_dim # 384
        self.context_encoder = ContextEncoder(bert_attention_heads,bert_hidden_size,pad_token_id,bert_hidden_layers,vocab_size)
        self.temporal_block = LayerNormGRU(input_dim=visual_out_dim + self.context_encoder.hidden_size, hidden_dim=seq_hidden_dim, num_layers=seq_layer)
        self.decoder = Decoder(d_model=seq_hidden_dim, N=decoder_layer)
        self.mlp = nn.Sequential(
            nn.Linear(seq_hidden_dim + 33, seq_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(seq_hidden_dim, 1)
        )
    def forward(self, input_, args):
        # visual input
        patches = input_['patches']
        patch_ids = input_['patch_ids']
        valid_mask = input_['valid_mask']
        patch_center_gps = input_['patch_center_gps']
        diff = input_['offsets']
        # visual output
        visual_output = self.visual_encoder(patches,patch_ids,valid_mask,patch_center_gps,diff) # (B, T, 384)
        # context output
        ctx_output, loss_1, (weekrep,daterep,timerep) = self.context_encoder(input_, args)
        # temporal sendoff
        representation = torch.cat([visual_output, ctx_output], dim=-1) # (B,T,Res + Ctx)
        representation = representation if batch_first else representation.transpose(0,1).contiguous() # (T,B,Res + Ctx)
        hiddens, _ = self.temporal_block(representation, seq_lens = input_['lens'].long())
        assert not torch.isnan(representation).any(), "representation has nan"
        decoder = self.decoder(hiddens, input_['lens'].long()) # (T,B,seq_hidden_dim)
        decoder = decoder if batch_first else decoder.transpose(0,1).contiguous() # (B,T,seq_hidden_dim)
        if torch.isnan(decoder).sum() > 0:
            print("NaN detected in decoder, replacing with 0")
            print(torch.where(torch.isnan(decoder), 1, 0).sum())
            decoder = torch.nan_to_num(decoder, 0.0)
            
        assert torch.isnan(decoder).sum() == 0, "decoder has nan"
        # sum pooling
        decoder = decoder * valid_mask.unsqueeze(-1).float() # (B,T,seq_hidden_dim)
        pooled_decoder = decoder.sum(dim=1) # (B,seq_hidden_dim)
        # add back the weekrep, daterep, timerep for making model learn time of important events
        pooled_decoder = torch.cat([pooled_decoder, weekrep[:,0], daterep[:,0], timerep[:,0]], dim=-1) # (B,seq_hidden_dim + 33)
        output = self.mlp(pooled_decoder) # (B,1)
        return output, loss_1

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

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.attn_1 = nn.MultiheadAttention(embed_dim=d_model, dropout=dropout, num_heads=self.h)

    def forward(self, q, k, v, len):
        # perform linear operation and split into N heads
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        S = q.shape[0]
        mask = torch.arange(max(len)).unsqueeze(0) < torch.tensor(len).unsqueeze(1)
        mask = mask.to(q.device)
        attn_output, _ = self.attn_1(q, k, v, key_padding_mask=mask)
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
        self.norm_1 = nn.LayerNorm(d_model) #
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)    #
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        #self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)


    def forward(self, x, len):
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, x2, x2, len))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Decoder(nn.Module):
    def __init__(self, d_model, N=3, heads=1, dropout=0.1):
        super().__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, x, lens):
        for i in range(self.N):
            x = self.layers[i](x, lens)
        return self.norm(x)


if __name__ == "__main__":
    pass