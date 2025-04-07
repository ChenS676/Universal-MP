import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import argparse
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
import copy
import math
import torch
import wandb
from torch import Tensor
from torch.nn import BatchNorm1d, Parameter
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import inits
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import MLP
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import (
    to_networkx,
    train_test_split_edges,
    to_undirected,
    spmm
)
from torch_sparse import SparseTensor
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset
from graphgps.utility.utils import mvari_str2csv
from automorphism import run_wl_test_and_group_nodes, compute_automorphism_metrics
from syn_real.gnn_utils import (
    evaluate_hits,
    evaluate_auc,
    evaluate_mrr,
    get_root_dir,
    get_logger,
    get_config_dir,
    Logger,
    init_seed
)
from baselines.gnn_utils import save_emb 
from syn_real_generator import (
    extract_induced_subgraph,
    use_lcc,
    perturb_disjoint,
    generate_perturbed_graph
)
from baselines.gnn_utils import mlp_score
import torch_geometric.transforms as T
# python real_syn_automorphic.py --data_name Citeseer --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1 --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5 --batch_size 1024 
# python real_syn_automorphic.py --data_name Cora --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1 --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5 --batch_size 1024 
# python real_syn_automorphic.py --data_name ogbl-ddi --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1 --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5 --batch_size 1024 
# Cora
# intra ratio has no effect on Cora
# inter ratio has a big effect on Cora
# inter_ratios = [0.1] # Try also: 0.1–0.9
# intra_ratios = [0.5]    # Fixed intra ratio
# total_edges_list = [0.2, 1, 4, 7, 12, 18, 28]*250 # Will be scaled × 10^3



# Citeseer
# inter_ratios = [0.1] # Try also: 0.1–0.9
# intra_ratios = [0.5]    # Fixed intra ratio
# total_edges_list = [0.2, 1, 2, 3, 4, 5, 7, 8, 10, 14] * 1000


# DDI
# inter_ratios = [0.5]  # Try also: 0.1–0.9
# intra_ratios = [0.5]    
# total_edges_list =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 1


dir_path = get_root_dir()
log_print = get_logger('testrun', 'log', get_config_dir())
DATASET_PATH = '/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/baselines/dataset'

PT_LIST = [f"plots/Citeseer/processed_graph_inter0.5_intra0.5_edges1000_auto0.7200_norm1_0.7676.pt"]
    


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


# --- 1️⃣ Load Real-World Graph (Cora) ---
def load_real_world_graph(dataset_name="Cora"):
    """
    Load a real-world graph dataset (e.g., Cora) from PyTorch Geometric.
    Args:
        dataset_name (str): The dataset name (default: "Cora").
    Returns:
        Data: PyTorch Geometric Data object.
    """
    if dataset_name in ['Cora', 'Citeseer', 'PubMed']:
        dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
        data = dataset[0]  
        # data.x = torch.eye(data.num_nodes, dtype=torch.float)
        # data.x = torch.rand(data.num_nodes, data.num_nodes)
    elif dataset_name.startswith('ogbl'):
        data = extract_induced_subgraph()
        # dataset = PygLinkPropPredDataset(name=dataset_name, root='/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/syn_graph/dataset/')
        # data = dataset[0]
        data.x = torch.diag(torch.arange(data.num_nodes).float())
        # print(f"before data {data}")
        # # data, _, _ = use_lcc(data)
        # print(f"after lcc {data}")
        # print(data.x)
    return data


# --- 2️⃣ Create Disjoint Graph Copies & Merge ---
def create_disjoint_graph(data):
    """
    Creates two disjoint copies of a real-world graph (e.g., Cora).
    Args:
        data (Data): PyG Data object representing the original graph.
    Returns:
        Data: PyG Data object representing the new merged graph.
    """
    num_nodes = data.num_nodes
    G = to_networkx(data, to_undirected=True)
    G2 = nx.relabel_nodes(G, lambda x: x + num_nodes)
    merged_graph = nx.compose(G, G2)
    merged_edge_index = torch.tensor(list(merged_graph.edges)).mT

    if hasattr(data, "x") and data.x is not None:
        merged_x = torch.cat([data.x, data.x], dim=0)
    merged_data = Data(edge_index=merged_edge_index, num_nodes=2 * num_nodes)
    if hasattr(data, "x") and data.x is not None:
        merged_data.x = merged_x  
    # print(merged_data.x)
    return merged_data


# --- 3️⃣ Add Controllable Random Edges ---
def add_random_edges(graph_data, inter_ratio=0.5, intra_ratio=0.5, total_edges=1000):
    """
    Adds random edges between and within two graph copies in a controlled way.

    Args:
        graph_data (Data): The graph structure (PyG format).
        inter_ratio (float): Fraction of edges to add **between** the two graph copies.
        intra_ratio (float): Fraction of edges to add **within** each graph copy.
        total_edges (int): Total number of random edges to add.

    Returns:
        Data: Graph with additional edges.
    """
    num_nodes = graph_data.num_nodes // 2 
    inter_edges = int(total_edges * inter_ratio)
    intra_edges = total_edges - inter_edges  
    inter_edges_list = [
        (random.randint(0, num_nodes - 1), random.randint(num_nodes, 2 * num_nodes - 1))
        for _ in range(inter_edges)
    ]
    intra_edges_list = []
    for _ in range(intra_edges):
        copy = random.choice([0, 1]) 
        base_offset = num_nodes * copy 
        u, v = random.sample(range(base_offset, base_offset + num_nodes), 2)
        intra_edges_list.append((u, v))
        
    new_edges = torch.tensor(inter_edges_list + intra_edges_list, dtype=torch.long).T
    updated_edge_index = torch.cat([graph_data.edge_index, new_edges], dim=1)
    return Data(edge_index=updated_edge_index, num_nodes=graph_data.num_nodes, x=graph_data.x)


def plot_group_size_distribution(group_sizes, args, file_name):
    """ 
    Plots the group size distribution with log-log scaling.
    
    Parameters:
        group_sizes (list): Sizes of automorphism groups.
        args (argparse.Namespace): Arguments containing dataset name.
    """
    # Not readable
    # plt.figure()
    # plt.plot(group_sizes)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel("Group Index (log scale)")
    # plt.ylabel("Group Size (log scale)")
    # plt.title("Group Size Distribution (Log-Log Scale)")
    # plt.savefig(f'plots/{args.data_name}/group_size_{args.data_name}.png')
    # plt.close()

    plt.figure()
    plt.plot(np.log1p(group_sizes))
    plt.xlabel("Group Index (log scale)")
    plt.ylabel("Group Size (log scale)")
    plt.title("Group Size Distribution (Log-Log Scale)")
    plt.savefig(file_name)
    plt.close()
    

def plot_histogram_group_size(group_sizes, metrics_before, args):
    """ 
    Plots a histogram of group sizes.
    
    Parameters:
        group_sizes (list): Sizes of automorphism groups.
        metrics_before (dict): Dictionary containing WL test metrics.
        args (argparse.Namespace): Arguments containing dataset name.
    """
    plot_dir = f'plots/{args.data_name}'
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    counts, bins, _ = plt.hist(group_sizes, bins=20, edgecolor='black', alpha=0.75, density=True)
    counts = counts * 100 * np.diff(bins)
    plt.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', alpha=0.75)
    plt.xlabel("Group Size")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Group Sizes {metrics_before['A_r_norm_1']}")
    save_path = f'{plot_dir}/hist_group_size_{args.data_name}.png'
    plt.savefig(save_path)
    plt.close()
    # print(f"Saved to {save_path}")
    # print(f"Automorphism fraction before adding random edges: {metrics_before}")



def plot_graph_visualization(graph_data, node_labels, args, save_path):
    """ 
    Plots a general visualization of the graph using WL-based node coloring.
    
    Parameters:
        graph_data (torch_geometric.data.Data): The input graph data.
        node_labels (list or array): Node labels for coloring.
        args (argparse.Namespace): Arguments containing dataset name.
    """
    plt.figure(figsize=(6, 6))
    G = to_networkx(graph_data, to_undirected=True)
    nx.draw(G, node_size=10, font_size=8, cmap='Set1', node_color=node_labels, edge_color="gray")
    plt.title("Graph Visualization with WL-based Node Coloring")
    plt.savefig(save_path)
    plt.close()


def plot_histogram_group_size_log_scale(group_sizes, metrics_before, args, save_path):
    """ 
    Plots a histogram of group sizes with log scale on both axes.
    
    Parameters:
        group_sizes (list): Sizes of automorphism groups.
        metrics_before (dict): Dictionary containing WL test metrics.
        args (argparse.Namespace): Arguments containing dataset name.
    """

    plt.figure(figsize=(6, 4))
    counts, bins, _ = plt.hist(group_sizes, bins=20, edgecolor='black', alpha=0.75, density=True)
    counts = counts * 100 * np.diff(bins)
    plt.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', alpha=0.75)
    plt.yscale('log') 
    plt.xlabel("Group Size (log scale)")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"Histogram of Group Sizes {metrics_before['A_r_norm_1']}")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved to {save_path}")
    print(f"Automorphism fraction before adding random edges: {metrics_before}")
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default="Citeseer")
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='LINKX')
    parser.add_argument('--score_model', type=str, default='mlp_score')
    parser.add_argument('--pt_path', default=f"plots/Citeseer/processed_graph_inter0.5_intra0.5_edges1000_auto0.7200_norm1_0.7676.pt",
                        type=str)
    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_layers_predictor', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--gnnout_hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--eval_metric', type=str, default='AUC')
    
    ### train setting
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=20,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)

    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    parser.add_argument('--remove_edge_aggre', action='store_true', default=False)
    parser.add_argument('--name_tag', type=str, default='')
    parser.add_argument('--gin_mlp_layer', type=int, default=2)
    parser.add_argument('--gat_head', type=int, default=1)
    parser.add_argument('--cat_node_feat_mf', default=False, action='store_true')
    parser.add_argument('--cat_n2v_feat', default=False, action='store_true')
    parser.add_argument('--test_batch_size', type=int, default=1024 * 64) 
    parser.add_argument('--use_hard_negative', default=False, action='store_true')
    parser.add_argument('--wandb_log', default=False, action='store_true')
    parser.add_argument('--metric', type=str, default='AUC')
    parser.add_argument('--inter_ratio', type=float, required=False, help='Inter ratio', default=0.5)
    parser.add_argument('--intra_ratio', type=float, required=False, help='Intra ratio', default=0.5)
    parser.add_argument('--total_edges', type=int, required=False, help='Total edges', default=1000)
    args = parser.parse_args()
    # print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    # print('use_val_edge:', args.use_valedges_as_input)
    # print('use_hard_negative: ',args.use_hard_negative)
    # print(args)
    return args
    
    
def randomsplit(data, val_ratio: float = 0.05, test_ratio: float = 0.15):
    def removerepeated(ei):
        ei = to_undirected(ei)
        ei = ei[:, ei[0]<ei[1]]
        return ei
    data.num_nodes = data.x.shape[0]
    
    data = train_test_split_edges(data, test_ratio, test_ratio)
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    
    num_val = int(data.val_pos_edge_index.shape[1] * val_ratio/test_ratio)
    data.val_pos_edge_index = data.val_pos_edge_index[:, torch.randperm(data.val_pos_edge_index.shape[1])]
    split_edge['train']['edge'] = removerepeated(
        torch.cat((data.train_pos_edge_index, data.val_pos_edge_index[:, :-num_val]), dim=-1)).t()
    split_edge['valid']['edge'] = removerepeated(data.val_pos_edge_index[:, -num_val:]).t()
    split_edge['valid']['edge_neg'] = removerepeated(data.val_neg_edge_index).t()
    split_edge['test']['edge'] = removerepeated(data.test_pos_edge_index).t()
    split_edge['test']['edge_neg'] = removerepeated(data.test_neg_edge_index).t()
    return split_edge

    
    
def data2dict(data, splits, data_name) -> dict:
    #TODO test with all ogbl-datasets, start with collab
    if data_name in ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'ogbl-ddi']:
        datadict = {}
        datadict.update({'adj': data.adj_t})
        datadict.update({'train_pos': splits['train']['edge']})
        # datadict.update({'train_neg': splits['train']['edge_neg']})
        datadict.update({'valid_pos': splits['valid']['edge']})
        datadict.update({'valid_neg': splits['valid']['edge_neg']})
        datadict.update({'test_pos': splits['test']['edge']})
        datadict.update({'test_neg': splits['test']['edge_neg']})   
        datadict.update({'train_val': torch.cat([splits['valid']['edge'], splits['train']['edge']])})
        datadict.update({'x': data.x}) 
    else:
        raise ValueError('data_name not supported')
    return datadict


def train(model, score_func, train_pos, data, optimizer, batch_size):
    model.train()
    score_func.train()

    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0

    for perm in DataLoader(range(train_pos.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        num_nodes = data['x'].size(0)
        ######################### remove loss edges from the aggregation
        mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
        mask[perm] = 0
        train_edge_mask = train_pos[mask].transpose(1,0)
        train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)
        edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
        # x = data.x.to(train_pos.device)
        # adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)
        h = model(data.x, data.adj_t, data.edge_weight)
        edge = train_pos[perm].t()
        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        # Just do some trivial random sampling.
        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = score_func(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)
        optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    return total_loss / total_examples



@torch.no_grad()
def test_edge(score_func, input_data, h, batch_size):
    # input_data  = input_data.transpose(1, 0)
    # with torch.no_grad():
    preds = []
    for perm  in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].t()
        preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
    pred_all = torch.cat(preds, dim=0)
    return pred_all



def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    k_list = [1, 10, 20, 50, 100]
    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)
    # result_hit = {}
    for K in k_list:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])
    train_pred = torch.cat([pos_train_pred, neg_val_pred])
    train_true = torch.cat([torch.ones(pos_train_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])
    result_auc_train = evaluate_auc(train_pred, train_true)
    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)
    
    # result_mrr_train = evaluate_mrr( evaluator_mrr,  pos_train_pred, neg_val_pred.repeat(pos_train_pred.size(0), 1))
    result_mrr_val = evaluate_mrr( evaluator_mrr, pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1))
    result_mrr_test = evaluate_mrr( evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1))
    for k in result_mrr_val.keys():
        result[k] = (0, result_mrr_val[k], result_mrr_test[k])

    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])
    # print(result.keys())
    return result


@torch.no_grad()
@torch.no_grad()
def test(model, score_func, data, evaluation_edges, emb, evaluator_hit, evaluator_mrr, batch_size, use_valedges_as_input):
    model.eval()
    score_func.eval()
    # adj_t = adj_t.transpose(1,0)
    train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge = evaluation_edges
    if emb == None: x = data.x
    else: x = emb.weight
    h = model(x, data.adj_t.to(x.device))
    x1 = h
    x2 = torch.tensor(1)
    # print(h[0][:10])
    train_val_edge = train_val_edge.to(x.device)
    pos_valid_edge = pos_valid_edge.to(x.device) 
    neg_valid_edge = neg_valid_edge.to(x.device)
    pos_test_edge = pos_test_edge.to(x.device) 
    neg_test_edge = neg_test_edge.to(x.device)
    neg_valid_pred = test_edge(score_func, neg_valid_edge, h, batch_size)
    pos_valid_pred = test_edge(score_func, pos_valid_edge, h, batch_size)
    if use_valedges_as_input:
        print('use_val_in_edge')
        h = model(x, data.full_adj_t.to(x.device))
        x2 = h
    pos_test_pred = test_edge(score_func, pos_test_edge, h, batch_size)
    neg_test_pred = test_edge(score_func, neg_test_edge, h, batch_size)
    pos_train_pred = test_edge(score_func, train_val_edge, h, batch_size)
    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)
    
    # print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x1.cpu(), x2.cpu()]

    # visualize_predictions(pos_train_pred, neg_valid_pred,
    #                     pos_valid_pred, neg_valid_pred,
    #                     pos_test_pred, neg_test_pred)
    return result, score_emb

def run_training_pipeline(data, metrics, inter, intra, total_edges, args):
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    data = copy.deepcopy(data)
    edge_index = data.edge_index
    data.adj_t = SparseTensor.from_edge_index(
        data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)
    ).to_symmetric().coalesce()

    if hasattr(data, 'edge_weight'):
        if data.edge_weight != None:
            edge_weight = data.edge_weight.to(torch.float)
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
            train_edge_weight = split_edge['train']['edge_weight'].to(device)
            train_edge_weight = train_edge_weight.to(torch.float)
        else:
            train_edge_weight = None
    else:
        train_edge_weight = None

    # print(data, args.data_name)
    if data.edge_weight is None:
        edge_weight = torch.ones((data.edge_index.size(1), 1))
        print(f"custom edge_weight {edge_weight.size()} added for {args.data_name}")
    data = T.ToSparseTensor()(data)
    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        val_edge_index = to_undirected(val_edge_index)
        full_edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float)
        edge_weight = torch.cat([edge_weight, val_edge_weight], 0)
        A = SparseTensor.from_edge_index(full_edge_index, edge_weight.view(-1), [data.num_nodes, data.num_nodes])
        data.full_adj_t = A
        data.full_edge_index = full_edge_index
        # print(data.full_adj_t)
        # print(data.adj_t)
    data = data.to(device)
    data.edge_index = edge_index.to(device)
    data.edge_weight = edge_weight.to(device)
    data.adj_t = data.adj_t.to(device)
    split_edge = randomsplit(data)
    print("Dataset split:")
    for key1 in split_edge:
        for key2 in split_edge[key1]:
            print(key1, key2, split_edge[key1][key2].shape[0])

    # data = data2dict(data, split_edge, args.data_name)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    x = data.x
    train_pos = split_edge['train']['edge'].to(x.device)
    in_channels = x.size(1)
    n_nodes = x.size(0)
    
    pos_train_edge = split_edge['valid']['edge']
    pos_valid_edge = split_edge['valid']['edge']
    neg_valid_edge = split_edge['valid']['edge_neg']
    pos_test_edge = split_edge['test']['edge']
    neg_test_edge = split_edge['test']['edge_neg']

    idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)]
    train_val_edge = pos_train_edge[idx]
    pos_train_edge = pos_train_edge.to(device)
    
    model = LINKX(
        num_nodes=n_nodes,
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                    1, args.num_layers_predictor, args.dropout).to(device)

    # print(data)
    evaluation_edges = [train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge]

    eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    loggers = {
        key: Logger(args.runs) for key in [
            'Hits@1', 'Hits@10', 'Hits@20', 'Hits@50', 'Hits@100', 
            'MRR', 'AUC', 'AP', 'mrr_hit1', 'mrr_hit3', 'mrr_hit10', 
            'mrr_hit20', 'mrr_hit50', 'mrr_hit100'
        ]
    }

    # import itertools
    # hyperparams = {
    #     'batch_size': [2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12],
    #     'lr': [0.01, 0.001, 0.0001],
    # }
    # if args.data_name == 'Cora': 
    #     args.batch_size = 1024
    #     args.lr = 0.01
    # elif args.data_name == 'Citeseer':
    #     args.batch_size = 1024
    #     args.lr = 0.01
    # elif args.data_name == 'ogbl-ddi':
    #     args.batch_size = 2**5
    #     args.lr = 0.00001
    args.batch_size = 512
    args.lr = 0.001
    args.name_tag = (
        f'{args.data_name}_'
        f'{args.gnn_model}_'
        f'inter{inter:.2f}_'
        f'intra{intra:.2f}_'
        f'total{total_edges:.0f}_'
        f'Orbits_{metrics["Number of Unique Groups (C_auto)"]:.2f}_'
        f'Norm_{metrics["A_r_norm_1"]:.2f}_'
        f'ArScore_{metrics["automorphism_score"]:.2f}'
    )

    # for batch_size, lr in itertools.product(hyperparams['batch_size'], hyperparams['lr']):
    #     args.batch_size = batch_size
    #     args.lr = lr
    
    for run in range(args.runs):
        if args.wandb_log:
            wandb.init(
                project=f"{args.data_name}_",
                name=f"{args.data_name}_{args.gnn_model}_{args.score_model}_{args.name_tag}_{args.runs}"
            )
            wandb.config.update(args)
        print(f'#################################          Run {run}          #################################')
        seed = args.seed if args.runs == 1 else run
        print('seed:', seed)
        init_seed(seed)
        save_path = os.path.join(
            args.output_dir,
            f'lr{args.lr}_drop{args.dropout}_l2{args.l2}_numlayer{args.num_layers}_'
            f'numPredlay{args.num_layers_predictor}_numGinMlplayer{args.gin_mlp_layer}_'
            f'dim{args.hidden_channels}_best_run_{seed}'
        )
        model.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(score_func.parameters()),
            lr=args.lr,
            weight_decay=args.l2
        )
        best_valid, best_test, kill_cnt, step = 0, 0, 0, 0
        for epoch in range(1, args.epochs + 1):
            loss = train(model, score_func, train_pos, data, optimizer, args.batch_size)
            if epoch % args.eval_steps == 0:
                results_rank, score_emb= test(model, 
                                              score_func,
                                              data, 
                                              evaluation_edges,
                                              None, 
                                              evaluator_hit, 
                                              evaluator_mrr, 
                                              args.batch_size, 
                                              args.use_valedges_as_input)

                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)
                    wandb.log({'train_loss': loss}, step=epoch) if args.wandb_log else None
                    wandb.log({f"Metrics/{key}": result[-1]}, step=epoch) if args.wandb_log else None
                    step += 1
                best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max()
                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                    if args.save:
                        save_emb(score_emb, save_path)
                else:
                    kill_cnt += 1
                    if kill_cnt > args.kill_cnt:
                        print("Early Stopping!!")
                        break
        wandb.finish() if args.wandb_log else None
                
    result_all_run = {} 
    save_dict = {}
    for key in loggers.keys():
        if key in ['Hits@1', 'AUC', 'AP', 'MRR']:
            best_metric, best_valid_mean, mean_list, var_list, test_res = loggers[key].print_statistics()
            if key == eval_metric:
                best_metric_valid_str = best_metric
            if key == 'AUC':
                best_auc_valid_str = best_metric
            result_all_run[key] = [mean_list, var_list]
            save_dict[key] = test_res
            print(save_dict)
    print(best_metric_valid_str + ' ' + best_auc_valid_str)
    print(args.name_tag)
    mvari_str2csv(args.name_tag, save_dict, f'results/syn_{args.data_name}_{args.gnn_model}_tuned.csv')


def main():
    args = parse_args()
    init_seed(args.seed)

    if os.path.exists(f'plots/{args.data_name}') == False:
        os.makedirs(f'plots/{args.data_name}')

    csv_path = f'plots/{args.data_name}/_Node_Merging.csv'
    file_exists = os.path.isfile(csv_path)
    original_data = load_real_world_graph(args.data_name)
    perturb_disjoint(original_data, args, 0, 0, 0)
    
    disjoint_graph = create_disjoint_graph(original_data)
    disjoint_graph, metrics = perturb_disjoint(disjoint_graph, args, 0, 0, 0)
    run_training_pipeline(disjoint_graph, metrics, 0, 0, 0, args)
    
    if args.data_name == 'Cora':
        inter_ratios = [0.1]   
        intra_ratios =  [0.5]
        total_edges_list =  [0.2, 1, 4, 7, 12, 18, 20, 28] # 
        multi_factor = 250 

    elif args.data_name == 'Citeseer':
        # Citeseer
        inter_ratios = [0.1] # Try also: 0.1–0.9
        intra_ratios = [0.5]    # Fixed intra ratio
        total_edges_list = [0.2, 1, 2, 3, 4, 5, 7, 8, 10, 14]
        multi_factor = 1000 
        
    elif args.data_name == 'ogbl-ddi':
        # DDI
        inter_ratios = [0.5]  # Try also: 0.1–0.9
        intra_ratios = [0.5]    
        total_edges_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # np.round(np.arange(0, 40, 2), 2).tolist()# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
        multi_factor = 1
        
    for inter in inter_ratios:
        for intra in intra_ratios:
            for edge_factor in total_edges_list:
                total_edges = int(edge_factor * multi_factor)
                data, metrics = perturb_disjoint(disjoint_graph, args, inter, intra, total_edges)
                run_training_pipeline(data, metrics, inter, intra, total_edges, args)

if __name__ == "__main__":
    main()
