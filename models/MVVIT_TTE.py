import torch
from models.VideoMae import ViTEncoder, ResnetEncoder
from models.ContextEncoder import ContextEncoder
from models.LayerNormGRU import LayerNormGRU
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
class Resnet_TTE(torch.nn.Module):
    def __init__(self, visual_out_dim, seq_hidden_dim, seq_layer, decoder_layer,
                 bert_attention_heads, bert_hiden_size, pad_token_id, bert_hidden_layers, v_query=True,vocab_size=27300, ca_head=8):
        super().__init__()
        self.visual_encoder = ResnetEncoder(visual_out_dim)
        self.context_encoder = ContextEncoder(bert_attention_heads, bert_hiden_size, pad_token_id, bert_hidden_layers, vocab_size)
        self.visual_output_dim = visual_out_dim
        self.context_output_dim = self.context_encoder.hidden_size
        if v_query:
            self.fusion_block = CrossAttention(dim_q=self.visual_output_dim,dim_kv=self.context_output_dim, num_heads=ca_head)
        else:
            self.fusion_block = CrossAttention(dim_q=self.context_output_dim,dim_kv=self.visual_output_dim, num_heads=ca_head)
        
        rnn_input = self.visual_output_dim if v_query else self.context_output_dim
        self.temporal_block = LayerNormGRU(input_dim=rnn_input, hidden_dim=seq_hidden_dim, num_layers=seq_layer)
        self.decoder = Decoder(d_model=seq_hidden_dim, N=decoder_layer)
        self.mlp = nn.Sequential(
            nn.Linear(seq_hidden_dim + 33, seq_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(seq_hidden_dim, 1)
        )
        
    def pooling_sum(self, hiddens, lens):
        lens = lens.to(hiddens.device)
        lens = torch.autograd.Variable(torch.unsqueeze(lens, dim=1), requires_grad=False)
        batch_size = range(hiddens.shape[0])
        for i in batch_size:
            hiddens[i, 0] = torch.sum(hiddens[i, :lens[i]], dim=0)
        return hiddens[list(batch_size), 0]
    
    def forward(self, input_, args):
        # input will be a dict
        # input['patches']: [B, T, C, H, W]
        # other thing for context just pass the input in
        visual_input = input_['patches'] # [B, T, C, H, W]
        patch_ids = input_['patch_ids']
        valid_mask = input_['valid_mask']
        context_input = input_
        lens = input_['lens']
        
        visual_encoded = self.visual_encoder(visual_input,patch_ids,valid_mask) # [B, T, D = visual_encoder.config.hidden_size]
        context_encoded, loss_1, (weekrep,daterep,timerep) = self.context_encoder(context_input,args) # [B, seq_len, D']
        # context_encoded: [B, T, D' = 64 + bert_hiden_size]
        cross_attn_output = self.fusion_block(visual_encoded, context_encoded) # [B, T, D]
        cross_attn_output = cross_attn_output if batch_first else cross_attn_output.transpose(0,1).contiguous() # [T,B,D]  
        hiddens, _ = self.temporal_block(cross_attn_output, seq_lens = lens.long())  # [T,B, seq_hidden_dim]
        decoder = self.decoder(hiddens, lens.long()) # [T,B, seq_hidden_dim]
        decoder = decoder if batch_first else decoder.transpose(0,1).contiguous() # [B, T, seq_hidden_dim]
        pooled_decoder = self.pooling_sum(decoder, lens.long())  # [B, seq_hidden_dim]
        pooled_hidden = torch.cat([pooled_decoder, weekrep[:, 0], daterep[:, 0], timerep[:, 0]], dim=-1) # [B, seq_hidden_dim + 3 + 10 + 20]
        output = self.mlp(pooled_hidden)
        output = args.scaler.inverse_transform(output) # [B, 1]
        return output, loss_1
    
class MMVIT_TTE(torch.nn.Module):
    def __init__(self, seq_hidden_dim, seq_layer, decoder_layer,
                 bert_attention_heads, bert_hiden_size, pad_token_id, bert_hidden_layers, v_query=True,vocab_size=27300, ca_head=8):
        super().__init__()
        self.visual_encoder = ViTEncoder()
        self.context_encoder = ContextEncoder(bert_attention_heads, bert_hiden_size, pad_token_id, bert_hidden_layers, vocab_size)
        self.visual_output_dim = self.visual_encoder.hidden_size
        self.context_output_dim = self.context_encoder.hidden_size
        if v_query:
            self.fusion_block = CrossAttention(dim_q=self.visual_output_dim,dim_kv=self.context_output_dim, num_heads=ca_head)
        else:
            self.fusion_block = CrossAttention(dim_q=self.context_output_dim,dim_kv=self.visual_output_dim, num_heads=ca_head)
        
        rnn_input = self.visual_output_dim if v_query else self.context_output_dim
        self.temporal_block = LayerNormGRU(input_dim=rnn_input, hidden_dim=seq_hidden_dim, num_layers=seq_layer)
        self.decoder = Decoder(d_model=seq_hidden_dim, N=decoder_layer)
        self.mlp = nn.Sequential(
            nn.Linear(seq_hidden_dim + 33, seq_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(seq_hidden_dim, 1)
        )
        
    def pooling_sum(self, hiddens, lens):
        lens = lens.to(hiddens.device)
        lens = torch.autograd.Variable(torch.unsqueeze(lens, dim=1), requires_grad=False)
        batch_size = range(hiddens.shape[0])
        for i in batch_size:
            hiddens[i, 0] = torch.sum(hiddens[i, :lens[i]], dim=0)
        return hiddens[list(batch_size), 0]
    
    def forward(self, input_, args):
        # input will be a dict
        # input['patches']: [B, T, C, H, W]
        # other thing for context just pass the input in
        visual_input = input_['patches'] # [B, T, C, H, W]
        context_input = input_
        lens = input_['lens']
        T = lens.max().item()
        
        visual_encoded = self.visual_encoder(visual_input,T,input_['mask'],args.device) # [B, T, D = visual_encoder.config.hidden_size]
        context_encoded, loss_1, (weekrep,daterep,timerep) = self.context_encoder(context_input,args) # [B, seq_len, D']
        # context_encoded: [B, T, D' = 64 + bert_hiden_size]
        cross_attn_output = self.fusion_block(visual_encoded, context_encoded) # [B, T, D]
        cross_attn_output = cross_attn_output if batch_first else cross_attn_output.transpose(0,1).contiguous() # [T,B,D]  
        hiddens, _ = self.temporal_block(cross_attn_output, seq_lens = lens.long())  # [T,B, seq_hidden_dim]
        decoder = self.decoder(hiddens, lens.long()) # [T,B, seq_hidden_dim]
        decoder = decoder if batch_first else decoder.transpose(0,1).contiguous() # [B, T, seq_hidden_dim]
        pooled_decoder = self.pooling_sum(decoder, lens.long())  # [B, seq_hidden_dim]
        pooled_hidden = torch.cat([pooled_decoder, weekrep[:, 0], daterep[:, 0], timerep[:, 0]], dim=-1) # [B, seq_hidden_dim + 3 + 10 + 20]
        output = self.mlp(pooled_hidden)
        output = args.scaler.inverse_transform(output) # [B, 1]
        return output, loss_1
    
class CrossAttention(torch.nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=8):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=dim_q, num_heads=num_heads, batch_first=True)
        self.kv_projection = torch.nn.Linear(dim_kv, dim_q)
        
    def forward(self, visual_feat, context_feat):
        # implement cross attention mechanism
        context_proj = self.kv_projection(context_feat)
        fused_feat, _ = self.attn(visual_feat, context_proj, context_proj)
        return fused_feat



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
        mask = torch.stack([torch.cat((torch.zeros(i), torch.ones(S - i)), 0) for i in len]).bool().to(k.device)
        attn_output, attn_output_weights = self.attn_1(q, k, v, key_padding_mask=mask)
        return attn_output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
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
        self.norm_1 = Norm(d_model) #
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)    #
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout) #
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
    model = MMVIT_TTE()
    pass