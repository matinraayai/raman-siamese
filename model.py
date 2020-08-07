import torch
import torch.nn as nn
import torch.nn.init as init


def _weights_init(m: nn.Module):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight)


class NotSiamese2D(nn.Module):
    def __init__(self):
        super(NotSiamese2D, self).__init__()
        self.network = nn.Sequential(nn.Conv2d(1, 32,
                                               kernel_size=(5, 2),
                                               stride=(2, 1),
                                               padding_mode='zeros'),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d((2, 1)),
                                     nn.Dropout2d(0.2),
                                     nn.Conv2d(32, 64,
                                               kernel_size=(5, 1),
                                               stride=(2, 1),
                                               padding_mode='zeros'),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d((2, 1)),
                                     nn.Dropout2d(0.2),
                                     nn.Flatten(),
                                     nn.Linear(3456, 1024),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(1024, 2),
                                     nn.Softmax()
                                     )
        self.apply(_weights_init)

    def forward(self, x):
        return self.network(x)


class Siamese1D(nn.Module):
    def __init__(self):
        super(Siamese1D, self).__init__()
        self.network = nn.Sequential(nn.Conv1d(1, 16,
                                               kernel_size=21,
                                               padding_mode='zeros'),
                                     nn.BatchNorm1d(16),
                                     nn.LeakyReLU(inplace=True),
                                     nn.MaxPool1d(2, stride=2),
                                     nn.Conv1d(16, 32,
                                               kernel_size=11,
                                               padding_mode='zeros'),
                                     nn.BatchNorm1d(32),
                                     nn.LeakyReLU(inplace=True),
                                     nn.MaxPool1d(2, stride=2),
                                     nn.Conv1d(32, 64,
                                               kernel_size=5,
                                               padding_mode='zeros'),
                                     nn.BatchNorm1d(64),
                                     nn.LeakyReLU(inplace=True),
                                     nn.MaxPool1d(2, stride=2),
                                     nn.Flatten()
                                     )
        self.out = nn.Sequential(nn.Linear(6592, 1), nn.Sigmoid())
        self.apply(_weights_init)

    def forward(self, x1, x2):
        out1 = self.network(x1)
        out2 = self.network(x2)
        dis = torch.abs(out1 - out2)
        return self.out(dis)
