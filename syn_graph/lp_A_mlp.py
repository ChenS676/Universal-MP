import torch
import torch.nn.functional as F
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data

import math

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Parameter

from torch_geometric.nn import inits
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import MLP
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import spmm


class SparseLinear(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.empty(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        inits.kaiming_uniform(self.weight, fan=self.in_channels,
                              a=math.sqrt(5))
        inits.uniform(self.in_channels, self.bias)

    def forward(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        # propagate_type: (weight: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, weight=self.weight,
                             edge_weight=edge_weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, weight_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if edge_weight is None:
            return weight_j
        else:
            return edge_weight.view(-1, 1) * weight_j

    def message_and_aggregate(self, adj_t: Adj, weight: Tensor) -> Tensor:
        return spmm(adj_t, weight, reduce=self.aggr)


class LINKX(torch.nn.Module):
    r"""The LINKX model from the `"Large Scale Learning on Non-Homophilous
    Graphs: New Benchmarks and Strong Simple Methods"
    <https://arxiv.org/abs/2110.14446>`_ paper.

    .. math::
        \mathbf{H}_{\mathbf{A}} &= \textrm{MLP}_{\mathbf{A}}(\mathbf{A})

        \mathbf{H}_{\mathbf{X}} &= \textrm{MLP}_{\mathbf{X}}(\mathbf{X})

        \mathbf{Y} &= \textrm{MLP}_{f} \left( \sigma \left( \mathbf{W}
        [\mathbf{H}_{\mathbf{A}}, \mathbf{H}_{\mathbf{X}}] +
        \mathbf{H}_{\mathbf{A}} + \mathbf{H}_{\mathbf{X}} \right) \right)

    .. note::

        For an example of using LINKX, see `examples/linkx.py <https://
        github.com/pyg-team/pytorch_geometric/blob/master/examples/linkx.py>`_.

    Args:
        num_nodes (int): The number of nodes in the graph.
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        num_layers (int): Number of layers of :math:`\textrm{MLP}_{f}`.
        num_edge_layers (int, optional): Number of layers of
            :math:`\textrm{MLP}_{\mathbf{A}}`. (default: :obj:`1`)
        num_node_layers (int, optional): Number of layers of
            :math:`\textrm{MLP}_{\mathbf{X}}`. (default: :obj:`1`)
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.0`)
    """
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        num_edge_layers: int = 1,
        num_node_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_layers = num_edge_layers

        self.edge_lin = SparseLinear(num_nodes, hidden_channels)

        if self.num_edge_layers > 1:
            self.edge_norm = BatchNorm1d(hidden_channels)
            channels = [hidden_channels] * num_edge_layers
            self.edge_mlp = MLP(channels, dropout=0., act_first=True)
        else:
            self.edge_norm = None
            self.edge_mlp = None

        channels = [in_channels] + [hidden_channels] * num_node_layers
        self.node_mlp = MLP(channels, dropout=0., act_first=True)

        self.cat_lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.cat_lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

        channels = [hidden_channels] * num_layers + [out_channels]
        self.final_mlp = MLP(channels, dropout=dropout, act_first=True)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.edge_lin.reset_parameters()
        if self.edge_norm is not None:
            self.edge_norm.reset_parameters()
        if self.edge_mlp is not None:
            self.edge_mlp.reset_parameters()
        self.node_mlp.reset_parameters()
        self.cat_lin1.reset_parameters()
        self.cat_lin2.reset_parameters()
        self.final_mlp.reset_parameters()

    def forward(
        self,
        x: OptTensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        """"""  # noqa: D419
        out = self.edge_lin(edge_index, edge_weight)

        if self.edge_norm is not None and self.edge_mlp is not None:
            out = out.relu_()
            out = self.edge_norm(out)
            out = self.edge_mlp(out)

        out = out + self.cat_lin1(out)

        if x is not None:
            x = self.node_mlp(x)
            out = out + x
            out = out + self.cat_lin2(x)

        return self.final_mlp(out.relu_())

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_nodes={self.num_nodes}, '
                f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels})')
        
# Define your graph data (example dummy data)
n_nodes = 100
edge_index = torch.randint(0, n_nodes, (2, 500))  # 500 edges
x = torch.randn(n_nodes, 16)  # Node features with 16 dimensions

data = Data(x=x, edge_index=edge_index)
data = train_test_split_edges(data)

# Define model parameters
in_channels = data.x.size(1)
hidden_channels = 32
out_channels = 16
num_layers = 3
dropout = 0.5

# Initialize the model
model = LINKX(
    num_nodes=n_nodes,
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    num_layers=num_layers,
    dropout=dropout
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.train_pos_edge_index)
    
    # Compute positive and negative scores
    pos_edge = data.train_pos_edge_index
    neg_edge = data.train_neg_edge_index
    
    pos_scores = (out[pos_edge[0]] * out[pos_edge[1]]).sum(dim=1)
    neg_scores = (out[neg_edge[0]] * out[neg_edge[1]]).sum(dim=1)
    
    # Compute loss
    loss = F.binary_cross_entropy_with_logits(
        torch.cat([pos_scores, neg_scores]),
        torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    )
    loss.backward()
    optimizer.step()
    return loss.item()

# Training loop
for epoch in range(100):
    loss = train()
    print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

# Evaluation function
def evaluate():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.test_pos_edge_index)
        pos_scores = (out[data.test_pos_edge_index[0]] * out[data.test_pos_edge_index[1]]).sum(dim=1)
        neg_scores = (out[data.test_neg_edge_index[0]] * out[data.test_neg_edge_index[1]]).sum(dim=1)
        
        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)
        
        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([pos_labels, neg_labels])
        
        
        print(f'AUC: {auc:.4f}')

# Evaluate the model
evaluate()
