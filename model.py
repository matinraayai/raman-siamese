import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weight(module: nn.Module):
    return


class Siamese1D(nn.Module):
    def __init__(self, dropout_rate: float):
        super(Siamese1D, self).__init__()
        self.network = nn.Sequential(nn.Conv1d(1, 32,
                                               kernel_size=5,
                                               stride=2,
                                               padding_mode='zeros'),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool1d(2),
                                     nn.Dropout(dropout_rate),
                                     nn.Conv1d(32, 64,
                                               kernel_size=5,
                                               stride=2,
                                               padding_mode='zeros'),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool1d(2),
                                     nn.Dropout(dropout_rate),
                                     nn.Flatten(),
                                     nn.Linear(3456, 4096),
                                     nn.Sigmoid()
                                     )
        self.out = nn.Linear(4096, 1)

    def forward(self, x1, x2):
        out1 = self.network(x1)
        out2 = self.network(x2)
        dis = torch.abs(out1 - out2)
        return self.out(dis)
