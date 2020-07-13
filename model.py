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
                                     nn.Linear(3456, 2048),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(2048, 1),
                                     nn.Sigmoid()
                                     )
        self.apply(_weights_init)

    def forward(self, x):
        return self.network(x)


class Siamese1D(nn.Module):
    def __init__(self):
        super(Siamese1D, self).__init__()
        self.network = nn.Sequential(nn.Conv2d(1, 32,
                                               kernel_size=(5, 2),
                                               stride=(2, 1),
                                               padding_mode='zeros'),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool1d(2),
                                     nn.BatchNorm1d(32),
                                     nn.Conv1d(32, 64,
                                               kernel_size=5,
                                               stride=2,
                                               padding_mode='zeros'),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool1d(2),
                                     nn.BatchNorm1d(64),
                                     nn.Flatten(),
                                     nn.Linear(3456, 4096),
                                     nn.Sigmoid()
                                     )
        self.out = nn.Sequential(nn.Linear(4096, 1), nn.Sigmoid())
        self.apply(_weights_init)

    def forward(self, x1, x2):
        out1 = self.network(x1)
        out2 = self.network(x2)
        dis = torch.abs(out1 - out2)
        return self.out(dis)
