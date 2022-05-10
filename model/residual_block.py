import math

import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm


class _ResidualBlock(nn.Module):
    def __init__(self, d_feature, num_filters, kernel_size, dilation, dropout):
        super(_ResidualBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation  # 每个卷积核相当于 1+(k-1)*d 的大小

        self.conv1 = nn.Conv1d(
            in_channels=d_feature,
            out_channels=num_filters,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=d_feature,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.conv1 = weight_norm(self.conv1)
        self.conv2 = weight_norm(self.conv2)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = F.pad(x, (self.padding, 0))  # 在-1维度pad，左侧添加self.padding个0，右侧添加0个0，实现因果卷积
        x = self.dropout(self.relu(self.conv1(x)))
        x = F.pad(x, (self.padding, 0))
        x = self.dropout(self.relu(self.conv2(x)))
        x = x + residual
        return x


class ResidualBlock(nn.Module):
    def __init__(self, his_len, d_feature, num_filters, kernel_size, dropout, num_layers=None):
        super(ResidualBlock, self).__init__()
        dilation_factor = kernel_size

        # 如果 num_layers 没有被传递，就计算感受野能覆盖整个输入序列的 num_layers，
        # 由论文可知感受野的计算方式为(k− 1)d，d随网络层数指数级增长
        # (kernel_size-1)*(dilation_factor**(num_layers-1))==his_len-1 => num_layers
        if num_layers is None:
            num_layers = math.ceil(
                math.log((his_len - 1) / (kernel_size - 1), dilation_factor) + 1
            )

        self.residual_blocks_list = []
        for i in range(num_layers):
            dilation = dilation_factor ** i
            res_block = _ResidualBlock(
                d_feature, num_filters, kernel_size, dilation, dropout
            )
            self.residual_blocks_list.append(res_block)
        self.residual_blocks = nn.ModuleList(self.residual_blocks_list)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        x = x.permute(0, 2, 1)
        return x
