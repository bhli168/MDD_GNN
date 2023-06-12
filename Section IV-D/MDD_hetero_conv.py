from typing import Dict, Optional

import torch
from torch_geometric.typing import NodeType, EdgeType, Adj

from collections import defaultdict

from torch import Tensor
from torch.nn import Module, ModuleDict
#from torch_geometric.nn.conv.hgt_conv import group
#from torch_geometric.nn.conv.han_conv import group
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, reset

def group(xs, q, k_lin):
    if len(xs) == 0:
        return None
    else:
        num_edge_types = len(xs)
        out = torch.stack(xs)
        attn_score = (q * torch.mean(nn.functional.leaky_relu(k_lin(out)), dim=1)).sum(-1)
        attn = F.softmax(attn_score, dim=0)
        out = torch.sum(attn.view(num_edge_types, 1, -1) * out, dim=0)
        return out
class MDDHeteroConv(Module):
    r"""A generic wrapper for computing graph convolution on heterogeneous
    graphs.
    This layer will pass messages from source nodes to target nodes based on
    the bipartite GNN layer given for a specific edge type.
    If multiple relations point to the same destination, their results will be
    aggregated according to :attr:`aggr`.
    In comparison to :meth:`torch_geometric.nn.to_hetero`, this layer is
    especially useful if you want to apply different message passing modules
    for different edge types.

    .. code-block:: python

        hetero_conv = HeteroConv({
            ('paper', 'cites', 'paper'): GCNConv(-1, 64),
            ('author', 'writes', 'paper'): SAGEConv((-1, -1), 64),
            ('paper', 'written_by', 'author'): GATConv((-1, -1), 64),
        }, aggr='sum')

        out_dict = hetero_conv(x_dict, edge_index_dict)

        print(list(out_dict.keys()))
        >>> ['paper', 'author']

    Args:
        convs (Dict[Tuple[str, str, str], Module]): A dictionary
            holding a bipartite
            :class:`~torch_geometric.nn.conv.MessagePassing` layer for each
            individual edge type.
        aggr (string, optional): The aggregation scheme to use for grouping
            node embeddings generated by different relations.
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`None`). (default: :obj:`"sum"`)
    """
    def __init__(self,
                 out_channels,
                 metadata,
                 convs: Dict[EdgeType, Module],
                 aggr: Optional[str] = "mean"):
        super().__init__()
        self.convs = ModuleDict({'__'.join(k): v for k, v in convs.items()})
        self.aggr = aggr
        self.k_lin = nn.ModuleDict()
        self.q = nn.ParameterDict()
        self.out_channels = out_channels
        if not isinstance(out_channels, dict):
            out_channels = {node_type: out_channels for node_type in metadata[0]}
        for node_type, out_channels in self.out_channels.items():
            self.k_lin[node_type] = nn.Linear(out_channels, out_channels)
            self.q[node_type] = nn.Parameter(torch.Tensor(1, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        # for conv in self.convs.values():
        #     conv.reset_parameters()
        reset(self.k_lin)
        glorot(self.q)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        *args_dict,
        **kwargs_dict,
    ) -> Dict[NodeType, Tensor]:
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding node feature
                information for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], Tensor]): A dictionary
                holding graph connectivity information for each individual
                edge type.
            *args_dict (optional): Additional forward arguments of invididual
                :class:`torch_geometric.nn.conv.MessagePassing` layers.
            **kwargs_dict (optional): Additional forward arguments of
                individual :class:`torch_geometric.nn.conv.MessagePassing`
                layers.
                For example, if a specific GNN layer at edge type
                :obj:`edge_type` expects edge attributes :obj:`edge_attr` as a
                forward argument, then you can pass them to
                :meth:`~torch_geometric.nn.conv.HeteroConv.forward` via
                :obj:`edge_attr_dict = { edge_type: edge_attr }`.
        """
        out_dict = defaultdict(list)
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type

            str_edge_type = '__'.join(edge_type)
            if str_edge_type not in self.convs:
                continue

            args = []
            for value_dict in args_dict:
                if edge_type in value_dict:
                    args.append(value_dict[edge_type])
                elif src == dst and src in value_dict:
                    args.append(value_dict[src])
                elif src in value_dict or dst in value_dict:
                    args.append(
                        (value_dict.get(src, None), value_dict.get(dst, None)))

            kwargs = {}
            for arg, value_dict in kwargs_dict.items():
                arg = arg[:-5]  # `{*}_dict`
                if edge_type in value_dict:
                    kwargs[arg] = value_dict[edge_type]
                elif src == dst and src in value_dict:
                    kwargs[arg] = value_dict[src]
                elif src in value_dict or dst in value_dict:
                    kwargs[arg] = (value_dict.get(src, None),
                                   value_dict.get(dst, None))

            conv = self.convs[str_edge_type]

            if src == dst:
                out = conv(x_dict[src], edge_index, *args, **kwargs)
            else:
                out = conv((x_dict[src], x_dict[dst]), edge_index, *args,
                           **kwargs)

            out_dict[dst].append(out)

        # for key, value in out_dict.items():
        #     out_dict[key] = group(value, self.aggr)
        for node_type, outs in out_dict.items():
            out = group(outs, self.q[node_type], self.k_lin[node_type]) ## for MDDTransfer, remove this line

            if out is None:
                out_dict[node_type] = None
                continue
            out_dict[node_type] = out

        return out_dict

        #return out_dict

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={len(self.convs)})'