import math

from torch import nn

from model.residual_block import _ResidualBlock


class TCN(nn.Module):
    def __init__(self, d_feature=7, his_len=96, pre_len=96, num_filters=3, num_layers=2, dilation_factor=2,
                 kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        self.d_feature = d_feature
        self.his_len = his_len
        self.pre_len = pre_len

        self.kernel_size = kernel_size
        self.dilation_factor = dilation_factor
        self.num_filters = num_filters
        self.dropout = dropout

        # 如果 num_layers 没有被传递，就计算感受野能覆盖整个输入序列的 num_layers，
        # 由论文可知感受野的计算方式为(k− 1)d，d随网络层数指数级增长
        # (kernel_size-1)*(dilation_factor**(num_layers-1))==his_len-1 => num_layers
        if num_layers is None:
            num_layers = math.ceil(
                math.log((his_len - 1) / (kernel_size - 1), self.dilation_factor) + 1
            )
        self.num_layers = num_layers

        self.residual_blocks_list = []
        for i in range(self.num_layers):
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
