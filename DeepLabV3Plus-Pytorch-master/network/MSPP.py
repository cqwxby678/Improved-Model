import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    '''深度可分离卷积层'''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False):
        super().__init__()
        # 自动计算填充
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:  # 处理(3,3)这样的元组
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        # 深度卷积 (逐通道卷积)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,  # 输出通道数等于输入通道数
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # 关键：分组数等于输入通道数
            bias=bias,
        )

        # 逐点卷积 (1x1卷积)
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,  # 1x1卷积核
            stride=1,
            padding=0,
            bias=bias,
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))


class MSinSPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=((3, 3), (5, 5), (7, 7))):
        super().__init__()
        hidden_channels = in_channels // 2

        # 计算拼接后的总通道数
        total_channels = in_channels + 12 * hidden_channels

        # 分支1: 3x3 (使用深度可分离卷积)
        self.conv1 = DepthwiseSeparableConv(in_channels, hidden_channels, kernel_size=kernel_sizes[0], stride=1)
        self.m1 = nn.MaxPool2d(
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=(kernel_sizes[0][0] // 2, kernel_sizes[0][1] // 2)
        )

        # 分支2: 5x5 (使用深度可分离卷积)
        self.conv2 = DepthwiseSeparableConv(in_channels, hidden_channels, kernel_size=kernel_sizes[1], stride=1)
        self.m2 = nn.MaxPool2d(
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=(kernel_sizes[1][0] // 2, kernel_sizes[1][1] // 2)
        )

        # 分支3: 7x7 (使用深度可分离卷积)
        self.conv3 = DepthwiseSeparableConv(in_channels, hidden_channels, kernel_size=kernel_sizes[2], stride=1)
        self.m3 = nn.MaxPool2d(
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=(kernel_sizes[2][0] // 2, kernel_sizes[2][1] // 2)
        )

        # 最后的1x1卷积保持普通卷积 (因为它已经是高效的逐点操作)
        self.conv1x1 = nn.Conv2d(total_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn_final = nn.BatchNorm2d(out_channels)
        self.act_final = nn.ReLU()

    def forward(self, x):
        # 分支1处理
        x1 = self.conv1(x)
        y1 = self.m1(x1)
        y2 = self.m1(y1)
        pool1_out = self.m1(y2)

        # 分支2处理
        x2 = self.conv2(x)
        y3 = self.m2(x2)
        y4 = self.m2(y3)
        pool2_out = self.m2(y4)

        # 分支3处理
        x3 = self.conv3(x)
        y5 = self.m3(x3)
        y6 = self.m3(y5)
        pool3_out = self.m3(y6)

        # 拼接所有特征
        features = torch.cat([
            x,  # 原始输入
            x1, y1, y2, pool1_out,  # 分支1
            x2, y3, y4, pool2_out,  # 分支2
            x3, y5, y6, pool3_out  # 分支3
        ], dim=1)

        # 通过最后的1x1卷积
        return self.act_final(self.bn_final(self.conv1x1(features)))