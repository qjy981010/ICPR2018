import math
import torch.nn as nn
import torch.nn.init as init


class SelfAttention(nn.Module):
    def __init__(self, attention_size, batch_first=False):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first
        self.attention_weights = Parameter(torch.FloatTensor(attention_size))
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

        init.uniform(self.attention_weights.data, -0.005, 0.005)

    def get_mask(self, attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(attentions.size())).detach()

        if attentions.data.is_cuda:
            mask = mask.cuda()

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inputs, lengths):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.tanh(inputs.matmul(self.attention_weights))
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on the sentence lengths
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze()

        return representations, scores


class CRNN(nn.Module):
    """
    CRNN model

    Args:
        in_channels (int): input channel number，1 for grayscaled images，3 for rgb images
        out_channels (int): output channel number(class number), letters number in dataset
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        hidden_size = 256
        # CNN struct and parameters
        self.cnn_struct = ((64, ), (128, ), (256, 256), (512, 512), (512, ))
        self.cnn_paras = ((3, 1, 1), (3, 1, 1),
                          (3, 1, 1), (3, 1, 1), (2, 1, 0))
        # pooling layer struct
        self.pool_struct = ((2, 2), (2, 2), (2, 1), (2, 1), None)
        # add batchnorm layer or not
        self.batchnorm = (True, True, True, True, True)
        self.dropout = (0, 0, 0.1, 0.2, 0.2)
        self.cnn = self._get_cnn_layers()
        # output channel number of LSTM in pytorch is hidden_size *
        #     num_directions, num_directions=2 for bidirectional LSTM
        self.rnn1 = nn.GRU(self.cnn_struct[-1][-1], hidden_size,
                            bidirectional=True, dropout=0.2)
        self.rnn2 = nn.GRU(hidden_size*2, hidden_size,
                            bidirectional=True, dropout=0.2)
        # fully-connected
        self.fc = nn.Linear(hidden_size*2, out_channels)
        self._initialize_weights()

    def forward(self, x):   # input: height=32, width>=100
        x = self.cnn(x)   # batch, channel=512, height=1, width>=24
        x = x.squeeze(2)   # batch, channel=512, width>=24
        x = x.permute(2, 0, 1)   # width>=24, batch, channel=512
        x = self.rnn1(x)[0]   # length=width>=24, batch, channel=256*2
        x = self.rnn2(x)[0]   # length=width>=24, batch, channel=256*2
        l, b, h = x.size()
        x = x.view(l*b, h)   # length*batch, hidden_size*2
        x = self.fc(x)   # length*batch, output_size
        x = x.view(l, b, -1)   # length>=24, batch, output_size
        return x

    def _get_cnn_layers(self):
        cnn_layers = []
        in_channels = self.in_channels
        for i in range(len(self.cnn_struct)):
            for out_channels in self.cnn_struct[i]:
                cnn_layers.append(
                    nn.Conv2d(in_channels, out_channels, *(self.cnn_paras[i])))
                if self.batchnorm[i]:
                    cnn_layers.append(nn.BatchNorm2d(out_channels))
                cnn_layers.append(nn.ReLU(inplace=True))
                if self.dropout[i]:
                    cnn_layers.append(nn.Dropout(self.dropout[i]))
                in_channels = out_channels
            if (self.pool_struct[i]):
                cnn_layers.append(nn.MaxPool2d(self.pool_struct[i]))
        return nn.Sequential(*cnn_layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                m.bias.data.zero_()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             m.weight.data.normal_(0, 0.01)
    #             m.bias.data.zero_()
