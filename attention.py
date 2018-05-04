import numpy as np
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F


# class SelfAttentiveEncoder(nn.Module):
#     # attention_d and attention_r is two hyperparameters which need to be manually modified
#     def __init__(self, in_channels, attention_d, attention_r):
#         super(SelfAttentiveEncoder, self).__init__()
#         self.ws1 = nn.Linear(in_channels, attention_d, bias=False)
#         self.ws2 = nn.Linear(attention_r, attention_r, bias= False)
#         self.tanh = nn.Tanh()
#         self.softmax = nn.Softmax()
#     def _initialize_weights(self, init_range = 0.1):
#         self.ws1.weight.data.uniform_(-init_range, init_range)
#         self.ws2.weight.data.uniform_(-init_range, init_range)
#     def forward(self, x): 
#         x = x.permute(1, 0, 2)    
#         size = x.size() # [batch, length, hidden_size * 2]
#         x = x.view(size[0] * size[1], -1) # [batch * length, hidden_size * 2]
#         x = self.ws1(x) # [batch * length, d]
#         x = self.tanh(x) 
#         x = self.ws2(x) # [batch * length, r]
#         alphas = x.view(size[0], size[1], -1) # [batch, length, r]
#         alphas = x.permute(0, 2, 1).contiguous() # [batch, r, length]
#         return torch.bmm(alphas, x)


class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_model)
        self.proj = Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = LayerNormalization(d_model)

        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)
    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)   # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)   # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)   # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=None) # attn_mask.repeat(n_head, 1, 1)

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1) 

        # project back to n_model
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)
        outputs = self.layer_norm(outputs)
        return outputs, attns


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, attn_dropout = 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(1)
    def forward(self, q, k, v, attn_mask=None):
        # k, q, v : [batch, length, hidden_size * 2(d_model)]
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        size = attn.size() # [batch, length(q), length(k)]
        attn = self.softmax(attn.view(size[0] * size[1], -1))
        attn = attn.view(size[0], size[1], -1)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v) # [batch, length(q), length(v)]
        return output, attn

class Linear(nn.Module):
    def __init__(self, d_in, d_out):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out)
        init.xavier_normal(self.linear.weight)
    def forward(self, x):
        return self.linear(x)
