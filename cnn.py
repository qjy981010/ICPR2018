import torch
import torch.nn as nn
import torch.nn.init as init


class CNN(nn.Module):
    """
    """
    def __init__(self, in_chans, out_chans, kernel, stride=1, padding=0, dilate=1,
                 dim=2, numblocks=1, pool=False, res=False):
        super(CNN, self).__init__()
        self.res = res
        self.init_cnn = None
        self.pool = None
        if res and in_chans != out_chans:
            self.init_cnn = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, 1, 1),
                nn.BatchNorm2d(out_chans),
                nn.ReLU(inplace=True)
            )
            in_chans = out_chans
        layers = []
        for block_id in range(numblocks):
            if dim == 2:
                layers.append(nn.Conv2d(in_chans, out_chans, kernel, stride, padding, dilate))
            elif dim == 1:
                layers.append(nn.Conv1d(in_chans, out_chans, kernel, stride, padding, dilate))
            layers.append(nn.BatchNorm2d(out_chans))
            if block_id != numblocks - 1:
                layers.append(nn.ReLU(inplace=True))
            in_chans = out_chans
        if pool:
            self.pool = nn.MaxPool2d(pool)
        self.relu = nn.ReLU(inplace=True)
        self.cnns = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        if self.init_cnn is not None:
            x = self.init_cnn(x)
        if self.res:
            residual = x
        x = self.cnns(x)
        if self.res:
            x = x + residual
            del residual
        x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

