import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from itertools import repeat
import collections.abc
import math
from functools import partial


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup

        hidden_channels = oup // ratio
        new_channels = hidden_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, hidden_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(hidden_channels, new_channels, dw_size, 1, dw_size // 2, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out
