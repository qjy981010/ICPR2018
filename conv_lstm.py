import torch.nn as nn
import torch
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      hidden_size: int thats the num of channels of the states, like hidden_size
    """

    def __init__(self, shape, input_chans, filter_size, hidden_size):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H,W
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.hidden_size = hidden_size
        # self.batch_size=batch_size
        # in this way the output has the same size
        self.padding = (filter_size-1) // 2
        self.conv = nn.Conv1d(self.input_chans + self.hidden_size,
                              4*self.hidden_size, self.filter_size, 1, self.padding)

    def forward(self, input, hidden_state):
        hidden, c = hidden_state  # hidden and c are images with several channels
        # print('hidden ',hidden.size())
        # print('c ',c.size())
        # print('input ',input.size())
        combined = torch.cat((input, hidden), 1)  # oncatenate in the channels
        # print('combined',combined.size())
        A = self.conv(combined)
        # it should return 4 tensors
        (ai, af, ao, ag) = torch.split(A, self.hidden_size, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f*c+i*g
        next_h = o*torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_size, self.shape).cuda(), requires_grad=False),
                Variable(torch.zeros(batch_size, self.hidden_size, self.shape).cuda(), requires_grad=False))##########


class CLSTM(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      hidden_size: int thats the num of channels of the states, like hidden_size
    """

    def __init__(self, shape, input_chans, filter_size, hidden_size, num_layers):
        super(CLSTM, self).__init__()

        self.shape = shape  # H,W
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        cell_list = []
        cell_list.append(CLSTM_cell(self.shape, self.input_chans,
                                    self.filter_size, self.hidden_size))  # the first
        # one has a different number of input channels

        for idcell in range(1, self.num_layers):
            cell_list.append(CLSTM_cell(
                self.shape, self.hidden_size, self.filter_size, self.hidden_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input, hidden_state=None):
        """
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            input is the tensor of shape seq_len,Batch,Chans,H,W
        """

        if hidden_state == None:
            hidden_state = self.init_hidden(input.size()[1])
        next_hidden = []  # hidden states(h and c)
        seq_len = input.size(0)

        for idlayer in range(self.num_layers):  # loop for every layer

            # hidden and c are images with several channels
            hidden_c = hidden_state[idlayer]
            all_output = []
            output_inner = []
            for t in range(seq_len):  # loop for every step
                # cell_list is a list with different conv_lstms 1 for every layer
                hidden_c = self.cell_list[idlayer](
                    input[t, ...], hidden_c)

                output_inner.append(hidden_c[0])

            next_hidden.append(hidden_c)
            input = torch.cat(output_inner, 0).view(
                input.size(0), *output_inner[0].size())  # seq_len,B,chans,H,W

        return input, next_hidden

    def init_hidden(self, batch_size):
        init_states = []  # this is a list of tuples
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states


########### Usage #######################################
if __name__ == '__main__':
    hidden_size = 10
    filter_size = 5
    batch_size = 10
    shape = 25  # H,W
    inp_chans = 3
    nlayers = 2
    seq_len = 4

    # If using this format, then we need to transpose in CLSTM
    input = Variable(torch.rand(seq_len, batch_size, inp_chans, shape))

    conv_lstm = CLSTM(shape, inp_chans, filter_size, hidden_size, nlayers)
    conv_lstm.apply(weights_init)

    # hidden_state = conv_lstm.init_hidden(batch_size)
    # print('hidden_h shape ', len(hidden_state))
    # print('hidden_h shape ', hidden_state[0][0].size())
    out = conv_lstm(input,)# hidden_state)
    print('out shape', out[0].size())
    print('len hidden ', len(out[1]))
    print('next hidden', out[1][0][0].size())
    print('convlstm dict', conv_lstm.state_dict().keys())


    L = torch.sum(out[0])
    L.backward()
