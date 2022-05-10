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
