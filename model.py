from abc import ABC

import torch
import torch.nn as nn
import torch.nn.init as init


def _weights_init(m: nn.Module):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        init.xavier_normal_(m.weight)


class SiameseDeepCid1D(nn.Module, ABC):
    def __init__(self):
        super(SiameseDeepCid1D, self).__init__()
        self.network = nn.Sequential(nn.Conv1d(1, 32,
                                               kernel_size=5,
                                               stride=2,
                                               padding_mode='zeros'),
                                     nn.BatchNorm1d(32),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool1d(2),
                                     nn.Conv1d(32, 64,
                                               kernel_size=5,
                                               stride=2,
                                               padding_mode='zeros'),
                                     nn.BatchNorm1d(64),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool1d(2),
                                     nn.Flatten()
                                     )
        self.out = nn.Sequential(nn.Linear(3456, 1), nn.Sigmoid())
        self.apply(_weights_init)

    def forward(self, x1, x2):
        out1 = self.network(x1)
        out2 = self.network(x2)
        dis = torch.abs(out1 - out2)
        return self.out(dis)


class Siamese1D(nn.Module, ABC):
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


def get_model(model_arch: str) -> nn.Module:
    if model_arch == 'deepcid-siamese':
        return SiameseDeepCid1D()
    elif model_arch == 'original-siamese':
        return Siamese1D()
    else:
        raise NotImplementedError("The requested model is not implemented.")