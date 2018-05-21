import torch.nn.init as init
import torch.nn as nn
from birnn import BiRNN
from cnn import CNN


class ResCR(nn.Module):
    """"""
    def __init__(self, nIn, nHidden, dropout=0.5):
        super().__init__()
        # self.cnn = nn.Sequential(
        #     nn.Conv1d(nIn, nIn, 3, 1, 3, 3),
        #     nn.BatchNorm1d(nIn),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(nIn, nIn, 3, 1, 1),
        #     nn.BatchNorm1d(nIn),
        # )
        # self.relu = nn.ReLU(inplace=True)
        self.birnn = BiRNN(nIn, nHidden, dropout=dropout)
        self.res = nIn == nHidden*2

    def forward(self, x):
        # residual = x
        # x = x.permute(1, 2, 0)
        # x = self.cnn(x)
        # x = x.permute(2, 0, 1)
        # x = x + residual
        # x = self.relu(x)
        if self.res: residual = x
        x = self.birnn(x)
        if self.res: x = x + residual
        return x


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

        self.cnn1 = CNN(in_channels, 64, 3, 1, 1, pool=(2, 2))
        self.cnn2 = CNN(64 , 128, 3, 1, 1, pool=(2, 2))
        self.cnn3 = CNN(128, 256, 3, 1, 1, numblocks=2, pool=(2, 1), res=False)
        self.cnn4 = CNN(256, 512, 3, 1, 1, numblocks=2, res=False)
        self.cnn5 = CNN(512, 512, 3, 1, (0, 1))
        self.cnn6 = CNN(512, 512, (2, 1), 1, (0, 1))

        # self.rescr1 = ResCR(512, 256)
        # self.rescr2 = ResCR(512, 256)
        # self.rescr3 = ResCR(512, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.cnn7 = CNN(512, 512, 3, 1, 3, 3, dim=1)
        self.cnn8 = CNN(512, 512, 3, 1, 1, dim=1)
        self.dropout2 = nn.Dropout(0.5)
        self.birnn1 = BiRNN(512, 256, dropout=0.5)
        self.birnn2 = BiRNN(512, 256, dropout=0.5)
        # self.birnn3 = BiRNN(512, 256, dropout=0.5)
        self.fc = nn.Linear(512, out_channels)
        self._initialize_weights()

    def forward(self, x):   # input: height=32, width>=100
        x = self.cnn1(x)   # batch, channel=512, height=1, width>=24
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        # x = self.dropout1(x)
        x = x.squeeze(2)   # batch, channel=512, width>=24
        x = self.cnn7(x)
        x = self.cnn8(x)
        # x = self.dropout2(x)
        x = x.permute(2, 0, 1)   # width>=24, batch, channel=512
        # x = self.rescr1(x)
        # x = self.rescr2(x)
        # x = self.rescr3(x)
        x = self.birnn1(x)
        x = self.birnn2(x)
        # x = self.birnn3(x)
        x = self.fc(x)
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
            elif isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                m.bias.data.zero_()

