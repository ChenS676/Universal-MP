
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import math
from baselines.gnn_utils import get_root_dir, get_logger, get_config_dir, Logger, init_seed, save_emb
from baselines.gnn_utils import GCN, GAT, SAGE, GIN, MF, DGCNN, GCN_seal, SAGE_seal, DecoupleSEAL, mlp_score
# from logger import Logger
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.utils import to_networkx, to_undirected
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from baselines.gnn_utils  import evaluate_hits, evaluate_auc
from data_load import loaddataset
from torch_geometric.utils import negative_sampling
import os
from tqdm import tqdm 
from graphgps.utility.utils import mvari_str2csv, random_sampling_ogb
import torch
from torch_geometric.data import Data
import numpy as np
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
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

from sklearn.preprocessing import OneHotEncoder
dir_path = get_root_dir()
log_print = get_logger('testrun', 'log', get_config_dir())
DATASET_PATH = '/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/baselines/dataset'


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

class LINKX_WL(torch.nn.Module):
    r"""The LINKX model from the `"Large Scale Learning on Non-Homophilous
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
        wl_emb_dim: int = 0,
        num_wl: int = 0,
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

        channels = [in_channels+wl_emb_dim] + [hidden_channels] * num_node_layers 
        self.node_mlp = MLP(channels, dropout=0., act_first=True)
        self.wl_emb = nn.Embedding(num_wl, wl_emb_dim)
        self.cat_lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.cat_lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        channels = [hidden_channels] * num_layers + [out_channels]
        self.final_mlp = MLP(channels, dropout=dropout, act_first=True)
        

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
        wl_indices: Tensor,
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
        if x is not None and wl_indices is not None:
            wl_embedding = self.wl_emb(wl_indices)
            x = torch.cat((x, wl_embedding), dim=1)
            x = self.node_mlp(x)
            out = out + x
            out = out + self.cat_lin2(x) 
        
        return self.final_mlp(out.relu_())

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_nodes={self.num_nodes}, '
                f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels})')


def evaluate_mrr(evaluator, pos_val_pred, neg_val_pred):
    
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    # neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    
    mrr_output =  eval_mrr(pos_val_pred, neg_val_pred)

    valid_mrr = mrr_output['mrr_list'].mean().item()
    valid_mrr_hit1 = mrr_output['hits@1_list'].mean().item()
    valid_mrr_hit3 = mrr_output['hits@3_list'].mean().item()
    valid_mrr_hit10 = mrr_output['hits@10_list'].mean().item()

    valid_mrr_hit20 = mrr_output['hits@20_list'].mean().item()
    valid_mrr_hit50 = mrr_output['hits@50_list'].mean().item()
    valid_mrr_hit100 = mrr_output['hits@100_list'].mean().item()


    valid_mrr = round(valid_mrr, 4)
    # test_mrr = round(test_mrr, 4)
    valid_mrr_hit1 = round(valid_mrr_hit1, 4)
    valid_mrr_hit3 = round(valid_mrr_hit3, 4)
    valid_mrr_hit10 = round(valid_mrr_hit10, 4)

    valid_mrr_hit20 = round(valid_mrr_hit20, 4)
    valid_mrr_hit50 = round(valid_mrr_hit50, 4)
    valid_mrr_hit100 = round(valid_mrr_hit100, 4)
    
    results = {}
    results['mrr_hit1'] = valid_mrr_hit1
    results['mrr_hit3'] = valid_mrr_hit3
    results['mrr_hit10'] = valid_mrr_hit10

    results['MRR'] = valid_mrr

    results['mrr_hit20'] = valid_mrr_hit20
    results['mrr_hit50'] = valid_mrr_hit50
    results['mrr_hit100'] = valid_mrr_hit100
    print(f"provide {results.keys()} from the evaluate_mrr.")
    return results


def eval_mrr(y_pred_pos, y_pred_neg):
    '''
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
    '''
    # calculate ranks
    y_pred_pos = y_pred_pos.view(-1, 1)
    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    hits1_list = (ranking_list <= 1).to(torch.float)
    hits3_list = (ranking_list <= 3).to(torch.float)

    hits20_list = (ranking_list <= 20).to(torch.float)
    hits50_list = (ranking_list <= 50).to(torch.float)
    hits10_list = (ranking_list <= 10).to(torch.float)
    hits100_list = (ranking_list <= 100).to(torch.float)
    mrr_list = 1./ranking_list.to(torch.float)

    return { 'hits@1_list': hits1_list,
                'hits@3_list': hits3_list,
                'hits@20_list': hits20_list,
                'hits@50_list': hits50_list,
                'hits@10_list': hits10_list,
                'hits@100_list': hits100_list,
                'mrr_list': mrr_list}


def get_metric_score_citation2(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    k_list = [1, 10, 20, 50, 100]
    result = {}
    result_mrr_train = evaluate_mrr( evaluator_mrr,  pos_train_pred, neg_val_pred)
    result_mrr_val = evaluate_mrr( evaluator_mrr, pos_val_pred, neg_val_pred )
    result_mrr_test = evaluate_mrr( evaluator_mrr, pos_test_pred, neg_test_pred )
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in k_list:
        result[f'mrr_hit{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])
    return result

def plot_test_sequences(test_pred, test_true):
    """
    Plots test_pred as a line plot with transparent circles for positive and negative samples.
    """
    test_pred = test_pred.detach().cpu().numpy()  
    test_true = test_true.detach().cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(test_pred, marker='o', linestyle='-', label="Prediction Score", color='#7FCA85', alpha=0.7)
    plt.plot(test_true, marker='o', linestyle='-', label="True Score", color='#BDAED2', alpha=0.7)
    # Color true labels (1=Positive, 0=Negative)

    plt.xlabel("Sample Index")
    plt.ylabel("Prediction Score")
    plt.title("Test Predictions with True Labels")
    plt.legend()
    plt.savefig('plot_prediction.png')
    
def get_metric_score(dt_name, evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

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
    
    if dt_name == 'citation2':
        # result_mrr_train = evaluate_mrr( evaluator_mrr,  pos_train_pred, neg_val_pred.repeat(pos_train_pred.size(0), 1))
        result_mrr_val = evaluate_mrr( evaluator_mrr, pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1))
        result_mrr_test = evaluate_mrr( evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1))
        for k in result_mrr_val.keys():
            result[k] = (0, result_mrr_val[k], result_mrr_test[k])

    # result_auc = {}
    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])
    return result
     
def train_use_hard_negative(model, 
                            score_func, 
                            train_pos, 
                            data, 
                            emb, 
                            optimizer, 
                            batch_size, 
                            pos_train_weight, 
                            remove_edge_aggre, 
                            gnn_batch_size):
    model.train()
    score_func.train()
    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0
    # pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    if emb == None: 
        x = data.x
        emb_update = 0
    else: 
        x = emb.weight
        emb_update = 1
    for perm, perm_large in zip(DataLoader(range(train_pos.size(0)), batch_size,
                           shuffle=True),  DataLoader(range(train_pos.size(0)), gnn_batch_size,
                           shuffle=True)):
        optimizer.zero_grad()
        num_nodes = data.x.size(0)
        ######################### remove loss edges from the aggregation
        if remove_edge_aggre:
            mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
            mask[perm] = 0
            train_edge_mask = train_pos[mask].transpose(1,0)
            train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)
            if pos_train_weight != None:
                edge_weight_mask = pos_train_weight[mask]
                edge_weight_mask = torch.cat((edge_weight_mask, edge_weight_mask), dim=0).to(torch.float)
            else:
                edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
            adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)
        else:
            adj = data.adj_t 
             
        ###################
        # print(adj)
        h = model(x, adj)
        edge = train_pos[perm].t()
        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        edge_large = train_pos[perm_large].t()
        edge_large = torch.randint(0, num_nodes, edge_large.size(), dtype=torch.long, device=edge.device)
        with torch.no_grad():
            neg_out_gnn_large = score_func(h[edge_large[0]], h[edge_large[1]])
            neg_large_loss = -torch.log(1 - (torch.sigmoid(neg_out_gnn_large)) + 1e-15)
            edge = edge_large[:,torch.topk(neg_large_loss.squeeze(), batch_size)[1]]
        neg_out = score_func(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        if emb_update == 1: torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)
        optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    return total_loss / total_examples

def train(model, score_func, split_edge, train_pos, data, indices, emb, optimizer, batch_size, pos_train_weight, data_name, remove_edge_aggre):
    model.train()
    score_func.train()
    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0
    # pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    if emb == None: 
        x = data.x
        emb_update = 0
    else: 
        x = emb.weight
        emb_update = 1
    for perm in DataLoader(range(train_pos.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        num_nodes = data.x.size(0)
        ######################### remove loss edges from the aggregation
        if remove_edge_aggre:
            mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
            mask[perm] = 0
            train_edge_mask = train_pos[mask].transpose(1,0)
            train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)
            if pos_train_weight != None:
                edge_weight_mask = pos_train_weight[mask]
                edge_weight_mask = torch.cat((edge_weight_mask, edge_weight_mask), dim=0).to(torch.float)
            else:
                edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
            adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)
        else:
            adj = data.adj_t 
        ###################

        if indices is not None:
            h = model(indices, x, adj, data.edge_weight)
        else:
            h = model(data.x, adj, data.edge_weight)
        
        edge = train_pos[perm].t()
        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long,
                                device=h.device)
        neg_out = score_func(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        if emb_update == 1: torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)
        optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    return total_loss / total_examples

@torch.no_grad()
def test_edge(score_func, input_data, h, batch_size, mrr_mode=False, negative_data=None):
    preds = []
    if mrr_mode:
        source = input_data.t()[0]
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = negative_data.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            preds += [score_func(h[src], h[dst_neg]).squeeze().cpu()]
        pred_all = torch.cat(preds, dim=0).view(-1, 1000)
    else:
        for perm  in DataLoader(range(input_data.size(0)), batch_size):
            edge = input_data[perm].t()
            preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
        pred_all = torch.cat(preds, dim=0)
    return pred_all

@torch.no_grad()
def test_citation2(model, score_func, data, evaluation_edges, emb, evaluator_hit, evaluator_mrr, batch_size):
    model.eval()
    score_func.eval()
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
    neg_valid_pred = test_edge(score_func, pos_valid_edge, h, batch_size, mrr_mode=True, negative_data=neg_valid_edge)
    pos_valid_pred = test_edge(score_func, pos_valid_edge, h, batch_size)
    pos_test_pred = test_edge(score_func, pos_test_edge, h, batch_size)
    neg_test_pred = test_edge(score_func, pos_test_edge, h, batch_size, mrr_mode=True, negative_data=neg_test_edge)
    pos_train_pred = test_edge(score_func, train_val_edge, h, batch_size)
    pos_valid_pred = pos_valid_pred.view(-1)
    pos_test_pred =pos_test_pred.view(-1)
    pos_train_pred = pos_valid_pred.view(-1)
    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score_citation2(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x1.cpu(), x2.cpu()]
    return result, score_emb

def visualize_predictions(pos_train_pred, neg_train_pred, 
                          pos_valid_pred, neg_valid_pred, 
                          pos_test_pred, neg_test_pred):
  
    """
    Visualizes the distribution of positive and negative predictions for train, validation, and test sets.
    """
    plt.figure(figsize=(15, 5))
    
    datasets = [(pos_train_pred, neg_train_pred, 'Train'),
                (pos_valid_pred, neg_valid_pred, 'Validation'),
                (pos_test_pred, neg_test_pred, 'Test')]
    
    for i, (pos_pred, neg_pred, title) in enumerate(datasets):
        plt.subplot(1, 3, i + 1)
        sns.histplot(pos_pred, bins=50, kde=True, color='#7FCA85', stat='density', label='Positive')
        sns.histplot(neg_pred, bins=50, kde=True, color='#BDAED2', stat='density', label='Negative', alpha=0.6)
        plt.title(f'{title} Set')
        plt.xlabel('Prediction Score')
        plt.ylabel('Density')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('visual_grand.png')
    
    
@torch.no_grad()
def test(args, model, score_func, data, indices, evaluation_edges, emb, evaluator_hit, evaluator_mrr, batch_size, use_valedges_as_input):
    model.eval()
    score_func.eval()
    # adj_t = adj_t.transpose(1,0)
    train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge = evaluation_edges
    if emb == None: x = data.x
    else: x = emb.weight

    # h = model(data.x, data.adj_t, data.edge_weight)
    if indices is not None:
        h = model(indices, x, data.adj_t, data.edge_weight)
    else:
        h = model(data.x, data.adj_t, data.edge_weight)
                
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
    
    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    result = get_metric_score(args.data_name, evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x1.cpu(), x2.cpu()]

    # visualize_predictions(pos_train_pred, neg_valid_pred,
    #                     pos_valid_pred, neg_valid_pred,
    #                     pos_test_pred, neg_test_pred)
    return result, score_emb



def plot_hist_hash(n2v_emb):
    # Convert to numpy
    hash_ids = n2v_emb.numpy()
    # Optional: remap hash IDs to 0...N-1 for better axis readability
    _, hash_labels = np.unique(hash_ids, return_inverse=True)
    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(hash_labels, bins=len(np.unique(hash_labels)), color='skyblue', edgecolor='gray')
    plt.xlabel("Structural Role (unique hash ID)")
    plt.ylabel("Number of Nodes")
    plt.title("Histogram of Structural Roles in Graph")
    plt.tight_layout()
    plt.savefig('structural_role_distribution.png')

def structural_one_hot(n2v_emb):
    hash_ids = n2v_emb.numpy().reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    one_hot = encoder.fit_transform(hash_ids)
    # when unique id less than 1000
    return torch.tensor(one_hot, dtype=torch.float32)


def structural_learnable_embedding(n2v_emb, embedding_dim=16):
    unique_hashes, inverse_indices = torch.unique(n2v_emb, return_inverse=True)
    emb_layer = nn.Embedding(len(unique_hashes), embedding_dim)
    return emb_layer(inverse_indices)  # shape: (N_nodes, embedding_dim)


# def main(count, lr, l2, dropout):
def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='collab')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='LINKX')
    parser.add_argument('--score_model', type=str, default='mlp_score')
    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_layers_predictor', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--gnnout_hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--emb_dim', type=int, default=16)
    ### train setting
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--kill_cnt', dest='kill_cnt', default=50, type=int, help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2',		type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    # parser.add_argument('--metric', type=str, default='Hits@50')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    parser.add_argument('--remove_edge_aggre', action='store_true', default=False)
    parser.add_argument('--name_tag', type=str, default='')
    ####### gin
    parser.add_argument('--gin_mlp_layer', type=int, default=2)
    ######gat
    parser.add_argument('--gat_head', type=int, default=1)
    ######mf
    parser.add_argument('--cat_node_feat_mf', default=False, action='store_true')
    ##### n2v
    parser.add_argument('--cat_n2v_feat', default=False, action='store_true')
    parser.add_argument('--test_batch_size', type=int, default=1024 * 64) 
    parser.add_argument('--use_hard_negative', default=False, action='store_true')
    
    parser.add_argument('--cat_wl_feat', default=False, action='store_true')
    parser.add_argument('--wl_process', type=str)
    
    args = parser.parse_args()
    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print('use_val_edge:', args.use_valedges_as_input)
    print('cat_n2v_feat: ', args.cat_n2v_feat)
    print('use_hard_negative: ',args.use_hard_negative)
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    # dataset = PygLinkPropPredDataset(name=args.data_name, root=DATASET_PATH) 

    data, split_edge = loaddataset(name=args.data_name, use_valedges_as_input=False) #args.data_name
    edge_index = data.edge_index
    emb = None
    node_num = data.num_nodes

    args.name_tag = f"{args.data_name}_{args.gnn_model}_{args.score_model}_ep{args.epochs}_wl{args.cat_wl_feat}_pro{args.wl_process}"
    
    if hasattr(data, 'x'):
        if data.x != None:
            if len(data.x.shape) == 1:
                data.x = data.x.unsqueeze(1)
            x = data.x
            data.x = data.x.to(torch.float)
            
            if args.cat_wl_feat:
                print('cat wl embedding!!')
                wl_emb = torch.load(
                        os.path.join(
                            DATASET_PATH, 
                            'wl_label/'+ 
                            'ogbl-' + args.data_name+
                            '_wl_labels.pt'))

                if args.wl_process == 'norm':
                    normalized_wl = (wl_emb.float() - wl_emb.float().min()) / (wl_emb.float().max() - wl_emb.float().min())
                    normalized_wl = normalized_wl.unsqueeze(-1)
                    x = torch.cat([x, normalized_wl], dim=1)
                    indices = None
                    args.gnn_model = 'LINKX'
                elif args.wl_process == 'unique': 
                    args.gnn_model = 'LINKX_WL'
                    # wl_emb = wl_emb.unsqueeze(-1)
                    unique_hashes, indices = torch.unique(wl_emb, return_inverse=True)
                    wl_emb = nn.Embedding(num_embeddings=len(unique_hashes), embedding_dim=16)
                    # rff_features = random_fourier_features(normalized_wl, D=16)
                    indices = indices.to(device)
            else:
                wl_emb = None
                indices = None
                unique_hashes = None
                                    
            data.x = data.x.to(device)
            if args.data_name in ['ddi', 'ppa']:
                input_channel = data.in_channel
            else:
                input_channel = data.x.size(1) #data.x.shape[-1]
        else:
            emb = torch.nn.Embedding(node_num, args.hidden_channels).to(device)
            input_channel = args.hidden_channels
    else:
        emb = torch.nn.Embedding(node_num, args.hidden_channels).to(device)
        input_channel = args.hidden_channels
        
    if hasattr(data, 'edge_weight'):
        if data.edge_weight != None:
            edge_weight = data.edge_weight.to(torch.float)
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
            train_edge_weight = split_edge['train']['weight'].to(device)
            train_edge_weight = train_edge_weight.to(torch.float)
        else:
            train_edge_weight = None
    else:
        train_edge_weight = None
    
    print(data, args.data_name)
    if data.edge_weight is None:
        edge_weight = torch.ones((data.edge_index.size(1), 1))
        print(f"custom edge_weight {edge_weight.size()} added for {args.data_name}")
    data = T.ToSparseTensor()(data)
    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        val_edge_index = to_undirected(val_edge_index)
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float)
        edge_weight = torch.cat([edge_weight, val_edge_weight], 0)
        A = SparseTensor.from_edge_index(full_edge_index, edge_weight.view(-1), [data.num_nodes, data.num_nodes])
        data.full_adj_t = A
        data.full_edge_index = full_edge_index
        print(data.full_adj_t)
        print(data.adj_t)
    
    if args.data_name == 'citation2': 
        data.adj_t = data.adj_t.to_symmetric()
        if args.gnn_model == 'GCN':
            adj_t = data.adj_t.set_diag()
            deg = adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
            data.adj_t = adj_t
            
    n_nodes = data.x.size(0)
    data = data.to(device)
    if args.gnn_model == 'LINKX_WL':
        model = LINKX_WL(
        num_nodes=n_nodes,
        in_channels=input_channel,
        hidden_channels=args.hidden_channels,
        out_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        wl_emb_dim = 16,
        num_wl = len(unique_hashes),
    ).to(device)
        
    elif args.gnn_model == 'LINKX':
        model = LINKX(
        num_nodes=n_nodes,
        in_channels=input_channel,
        hidden_channels=args.hidden_channels,
        out_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    else:
        model = eval(args.gnn_model)(input_channel, args.hidden_channels,
                        args.hidden_channels, args.num_layers, args.dropout, 
                        mlp_layer=args.gin_mlp_layer, head=args.gat_head, 
                        node_num=node_num, cat_node_feat_mf=args.cat_node_feat_mf,  
                        data_name=args.data_name).to(device)

    score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                    1, args.num_layers_predictor, args.dropout).to(device)
    # train_pos = data['train_pos'].to(x.device)
    # eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
        'AUC': Logger(args.runs),
        'AP': Logger(args.runs),
        'mrr_hit3': Logger(args.runs),
        'mrr_hit1':  Logger(args.runs),
        'mrr_hit10':  Logger(args.runs),
        'mrr_hit20':  Logger(args.runs),
        'mrr_hit50':  Logger(args.runs),
        'mrr_hit100':  Logger(args.runs),
    } 

    if args.data_name =='collab':
        eval_metric = 'Hits@50'
    elif args.data_name =='ddi':
        eval_metric = 'Hits@20'
    elif args.data_name =='ppa':
        eval_metric = 'Hits@100'
    elif args.data_name =='citation2':
        eval_metric = 'MRR'

    if args.data_name != 'citation2':
        pos_train_edge = split_edge['train']['edge']
        pos_valid_edge = split_edge['valid']['edge']
        neg_valid_edge = split_edge['valid']['edge_neg']
        pos_test_edge = split_edge['test']['edge']
        neg_test_edge = split_edge['test']['edge_neg']
    else:
        source_edge, target_edge = split_edge['train']['source_node'], split_edge['train']['target_node']
        pos_train_edge = torch.cat([source_edge.unsqueeze(1), target_edge.unsqueeze(1)], dim=-1)
        source, target = split_edge['valid']['source_node'],  split_edge['valid']['target_node']
        pos_valid_edge = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1)
        neg_valid_edge = split_edge['valid']['target_node_neg'] 
        source, target = split_edge['test']['source_node'],  split_edge['test']['target_node']
        pos_test_edge = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1)
        neg_test_edge = split_edge['test']['target_node_neg']

    idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)]
    train_val_edge = pos_train_edge[idx]
    pos_train_edge = pos_train_edge.to(device)
    evaluation_edges = [train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge]
    best_valid_auc = best_test_auc = 2
    best_auc_valid_str = 2
    
    for run in range(args.runs):
        print('#################################          ', run, '          #################################')
        import wandb
        wandb.init(project=f"{args.data_name}_", name=f"{args.data_name}_{args.gnn_model}_{args.score_model}_{args.name_tag}_{args.runs}")
        wandb.config.update(args)
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)
        init_seed(seed)
        save_path = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)
        if emb != None:
            torch.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        score_func.reset_parameters()
        if emb != None:
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(score_func.parameters()) + list(emb.parameters() ),lr=args.lr, weight_decay=args.l2)
        else:
            optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(score_func.parameters()),lr=args.lr, weight_decay=args.l2)
        best_valid = 0
        kill_cnt = 0
        best_test = 0
        step = 0
        
        for epoch in range(1, 1 + args.epochs):
            print('epoch: ', epoch)
            if args.use_hard_negative:
                loss = train_use_hard_negative(model, score_func, pos_train_edge, data, emb, optimizer, args.batch_size, train_edge_weight, args.remove_edge_aggre, args.test_batch_size)
            else:
                loss = train(model, score_func, split_edge, pos_train_edge, data, indices, emb, optimizer, args.batch_size, train_edge_weight, args.data_name, args.remove_edge_aggre)
                wandb.log({'train_loss': loss}, step = epoch)
                step += 1
            if epoch % args.eval_steps == 0:
                if args.data_name == 'citation2':
                    results_rank, score_emb= test_citation2(model, score_func, data, indices, evaluation_edges, emb, evaluator_hit, evaluator_mrr, args.batch_size)
                else:
                    results_rank, score_emb= test(args, model, score_func, data, indices, evaluation_edges, emb, evaluator_hit, evaluator_mrr, args.batch_size, args.use_valedges_as_input)
                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)
                    wandb.log({f"Metrics/{key}": result[-1]}, step=step)
                    
                if epoch % args.log_steps == 0:
                    for key, result in results_rank.items():
                        
                        train_hits, valid_hits, test_hits = result
                        log_print.info(
                            f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')

                r = torch.tensor(loggers[eval_metric].results[run])
                best_valid_current = round(r[:, 1].max().item(),4)
                best_test = round(r[r[:, 1].argmax(), 2].item(), 4)

                print(eval_metric)
                log_print.info(f'best valid: {100*best_valid_current:.2f}%, '
                                f'best test: {100*best_test:.2f}%')
                
                if len(loggers['AUC'].results[run]) > 0:
                    r = torch.tensor(loggers['AUC'].results[run])
                    best_valid_auc = round(r[:, 1].max().item(), 4)
                    best_test_auc = round(r[r[:, 1].argmax(), 2].item(), 4)
                    
                    print('AUC')
                    log_print.info(f'best valid: {100*best_valid_auc:.2f}%, '
                                f'best test: {100*best_test_auc:.2f}%')
                
                print('---')
                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                    if args.save: save_emb(score_emb, save_path)
                else:
                    kill_cnt += 1
                    
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break
        for key in loggers.keys():
            if len(loggers[key].results[0]) > 0:
                print(key)
                loggers[key].print_statistics(run)
    
    result_all_run = {}
    save_dict = {}
    for key in loggers.keys():
        if len(loggers[key].results[0]) > 0:
            print(key)
            best_metric,  best_valid_mean, mean_list, var_list, test_res = loggers[key].print_statistics()
            if key == eval_metric:
                best_metric_valid_str = best_metric
                best_valid_mean_metric = best_valid_mean
            if key == 'AUC':
                best_auc_valid_str = best_metric
                best_auc_metric = best_valid_mean
            result_all_run[key] = [mean_list, var_list]
            save_dict[key] = test_res
    print(f"to results_ogb_gnn/{args.data_name}_lm_mrr.csv")
    print(f"with name {args.name_tag}.")
    mvari_str2csv(args.name_tag, save_dict, f'results_ogb_gnn/{args.data_name}_lm_mrr.csv')

    if args.runs == 1:
        print(str(best_valid_current) + ' ' + str(best_test) + ' ' + str(best_valid_auc) + ' ' + str(best_test_auc))
    else:
        print(str(best_metric_valid_str) +' ' +str(best_auc_valid_str))


if __name__ == "__main__":
    main()

"""from baselines.gnn_utils import get_root_dir, get_logger, get_config_dir, Logger, init_seed, save_emb
from baselines.gnn_utils import GCN, GAT, SAGE, GIN, MF, DGCNN, GCN_seal, SAGE_seal, DecoupleSEAL, mlp_score
# from logger import Logger
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.utils import to_networkx, to_undirected
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from baselines.gnn_utils  import evaluate_hits, evaluate_auc, evaluate_mrr
from data_load import loaddataset
from torch_geometric.utils import negative_sampling
import os
from tqdm import tqdm 
from graphgps.utility.utils import mvari_str2csv, random_sampling_ogb
import torch
from torch_geometric.data import Data
import numpy as np
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
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

from sklearn.preprocessing import OneHotEncoder
"""
# adopted from benchmarking/exist_setting_ogb: Run models on ogbl-collab, ogbl-ppa, and ogbl-citation2 under the existing setting.
# python gnn_ogb_heart.py  --use_valedges_as_input  --data_name ogbl-collab  --gnn_model GCN --hidden_channels 256 --lr 0.001 --dropout 0.  --num_layers 3 --num_layers_predictor 3 --epochs 9999 --kill_cnt 100  --batch_size 65536 
# OBGL-PPA,DDI, CITATION2, VESSEL, COLLAB
# basic idea is to replace diffusion operator in mpnn and say whether it works better in ogbl-collab and citation2
# and then expand to synthetic graph