from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, Dropout, LeakyReLU, ReLU, Sigmoid, BatchNorm1d as BN
import numpy as np                         # import numpy
import torch

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(), Dropout(p=0.2))#,
        for i in range(1, len(channels))
    ])

class IGConv(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(IGConv, self).__init__(aggr='add', **kwargs)

        self.mlp1 = mlp1
        self.mlp2 = mlp2
        # self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp1)
        reset(self.mlp2)

    def update(self, aggr_out, x):
        if isinstance(x, tuple):
            tmp = torch.cat([x[1], aggr_out], dim=1)
            comb = self.mlp2(tmp)
            comb = (comb + x[1]) / 2
        else:
            tmp = torch.cat([x, aggr_out], dim=1)
            comb = self.mlp2(tmp)
            comb = (comb + x) / 2

        return comb

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.mul(x_j, edge_attr)
        agg = self.mlp1(tmp)
        return agg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1, self.mlp2)


class MDDNet(torch.nn.Module):
    def __init__(self, n_ch1, n_ch2):
        super(MDDNet, self).__init__()
        self.mlp1 = MLP([n_ch2, 256, 512])
        self.mlp2 = MLP([n_ch1+512, 1024])
        self.mlp2 = Seq(*[self.mlp2, Seq(Lin(1024, n_ch1)), BN(n_ch1), Dropout(0.2)])
        self.conv = IGConv(self.mlp1, self.mlp2)

    def forward(self, x0, edge_index, edge_attr):
        out = self.conv(x=x0, edge_index=edge_index, edge_attr=edge_attr)
        return out