import torch
from torch import nn

from model.residual_block import ResidualBlock


class TCN(nn.Module):
    def __init__(self, d_feature=7, his_len=96, pre_len=96, num_filters=3, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        self.pre_len = pre_len
        self.residual_blocks = ResidualBlock(his_len, d_feature, num_filters, kernel_size, dropout)

    def forward(self, x):
        x = self.residual_blocks(x)
        return x[:, -self.pre_len:, :]
