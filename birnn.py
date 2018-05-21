import torch
import torch.nn as nn
import torch.nn.init as init


class LayerNorm(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-7):
        super(LayerNorm, self).__init__()

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


class BiRNN(nn.Module):

    def __init__(self, nIn, nHidden, rnn_type='gru', dropout=0):
        super(BiRNN, self).__init__()
        if rnn_type == 'gru':
            self.rnn = nn.GRU(nIn, nHidden, bidirectional=True,
                              dropout=dropout)
        else:
            self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True,
                               dropout=dropout)
        self.layernorm = LayerNorm(nHidden * 2)

    def forward(self, x):
        x = self.rnn(x)[0]
        x = self.layernorm(x)

        return x