import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import copy
import itertools
import time
from tqdm import tqdm
from torch_geometric.datasets import Planetoid, Amazon
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader
from ogb.linkproppred import PygLinkPropPredDataset
from torch_sparse import SparseTensor
import matplotlib.pyplot as plt
import seaborn as sns
import yaml 
from functools import partial
import torch_geometric.transforms as T
from metrics.metrics import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_sparse.matmul import spmm_max, spmm_mean, spmm_add
from torch_sparse import SparseTensor
from typing import Iterable, Final
from torch_geometric.utils import (negative_sampling,
                                   to_undirected, train_test_split_edges)
import pandas as pd
import wandb
import torch
from torch_sparse import SparseTensor
from baselines.gnn_utils import evaluate_hits, evaluate_auc, Logger, init_seed
from models.GNN import GNN
from model import (CN0LinkPredictor, 
                   CatSCNLinkPredictor, 
                   SCNLinkPredictor, 
                   IncompleteSCN1Predictor, 
                   CNhalf2LinkPredictor, 
                   CNResLinkPredictor, 
                   IncompleteCN1Predictor, 
                   CN2LinkPredictor)

from formatted_best_params import best_params_dict
from data_utils.graph_rewiring import apply_beltrami
import torch_sparse
from model import DropAdj, adjoverlap
import numpy as np

server = 'SDIL'

"""
python lp_cn.py --data_name ogbl-collab --xdp 0.25 --tdp 0.05 --pt 0.1 --preedp 0.0 --predp 0.3  --probscale 2.5 --proboffset 6.0 \
                            --alpha 1.05 --prelr 0.0037  --batch_size 65536  --ln --lnnn --predictor cn1 \
                            --epochs 2 --runs 1 --hidden_dim 128 --mplayers 1  --testbs 131072  --maskinput  \
                            --use_valedges_as_input  --res  --use_xlin  --tailact 
"""

# NCN predictor
class CNLinkPredictor(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
            lnfn(in_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0
        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        x = x + self.xlin(x)
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        xcns = [spmm_add(cn, x)]
        
        xij = self.xijlin(xi * xj)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns],
            dim=-1)
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])


# NCN with higher order neighborhood overlaps than NCN-2
class CN2LinkPredictor(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1):
        super().__init__()

        self.lins = nn.Sequential()

        self.register_parameter("alpha", nn.Parameter(torch.ones((3))))
        self.register_parameter("beta", nn.Parameter(torch.ones((1))))
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xcn1lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn2lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn4lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, out_channels))

    def forward(self, x, adj: SparseTensor, tar_ei, filled1: bool = False):
        spadj = adj.to_torch_sparse_coo_tensor()
        adj2 = SparseTensor.from_torch_sparse_coo_tensor(spadj @ spadj, False)
        cn1 = adjoverlap(adj, adj, tar_ei, filled1)
        cn2 = adjoverlap(adj, adj2, tar_ei, filled1)
        cn3 = adjoverlap(adj2, adj, tar_ei, filled1)
        cn4 = adjoverlap(adj2, adj2, tar_ei, filled1)
        xij = self.xijlin(x[tar_ei[0]] * x[tar_ei[1]])
        xcn1 = self.xcn1lin(spmm_add(cn1, x))
        xcn2 = self.xcn2lin(spmm_add(cn2, x))
        xcn3 = self.xcn2lin(spmm_add(cn3, x))
        xcn4 = self.xcn4lin(spmm_add(cn4, x))
        alpha = torch.sigmoid(self.alpha).cumprod(-1)
        x = self.lin(alpha[0] * xcn1 + alpha[1] * xcn2 * xcn3 +
                     alpha[2] * xcn4 + self.beta * xij)
        return x



# random split dataset
def randomsplit(dataset, val_ratio: float=0.05, test_ratio: float=0.10):
    def removerepeated(ei):
        ei = to_undirected(ei)
        ei = ei[:, ei[0]<ei[1]]
        return ei
    data = dataset[0]
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



def get_dataset(root: str, opt: dict, name: str, use_valedges_as_input: bool=False, load=None):
    if name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(root="dataset", name=name)
        split_edge = randomsplit(dataset)
        data = dataset[0]
        data.edge_index = to_undirected(split_edge["train"]["edge"].t())
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]
    elif name in ["Computers", "Photo"]:
        dataset = Amazon(root="dataset", name=name)
        split_edge = randomsplit(dataset)
        data = dataset[0]
        data.edge_index = to_undirected(split_edge["train"]["edge"].t())
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]
    else:
        dataset = PygLinkPropPredDataset(root="dataset", name=name)
        split_edge = dataset.get_edge_split()
        data = dataset[0]
        edge_index = data.edge_index
        try:
            data.num_nodes = data.x.shape[0]
        except:
            if hasattr(data, "num_nodes"):
                print(f"use default data.num_nodes: {data.num_nodes}.")
        
    data.edge_weight = None 
    data.adj_t = SparseTensor.from_edge_index(edge_index, 
                    sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    data.max_x = -1
      
    if name == "ogbl-ppa":
        data.x = torch.argmax(data.x, dim=-1).unsqueeze(-1).float()
        data.max_x = torch.max(data.x).item()
    elif name == "ogbl-ddi":
        data.x = torch.arange(data.num_nodes).unsqueeze(-1).float()
        data.max_x = data.max_x = -1 # data.num_nodes
    if load is not None:
        data.x = torch.load(load, map_location="cpu")
        data.max_x = -1
    
    print("dataset split ")
    for key1 in split_edge:
        for key2  in split_edge[key1]:
            print(key1, key2, split_edge[key1][key2].shape[0])

    # Use training + validation edges for inference on test set.
    if use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, 
                            sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
        # if opt['rewiring'] is not None:
        #     data.edge_index = full_edge_index.copy()
        #     data = rewire(data, opt, root)
    else:
        data.full_adj_t = data.adj_t
        # if opt['rewiring'] is not None:
        #     data = rewire(data, opt, root)
    return data, split_edge



def load_yaml_config(file_path):
    """Loads a YAML configuration file."""
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    

@torch.no_grad()
def test_edge(score_func, input_data, h, data, batch_size, mrr_mode=False, negative_data=None):
    preds = []
    if mrr_mode:
        source = input_data.t()[0]
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = negative_data.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            # src, dst_neg = source[perm], target_neg[perm]
            # DEBUG
            preds += [score_func(h,
                    data.adj_t,
                    edge).cpu()]
        pred_all = torch.cat(preds, dim=0).view(-1, 1000)
    else:
        for perm  in DataLoader(range(input_data.size(0)), batch_size):
            edge = input_data[perm].t()
            preds += [score_func(h,
                        data.adj_t,
                        edge).cpu()]
        pred_all = torch.cat(preds, dim=0)
    return pred_all

def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    k_list = [20, 50, 100]
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
    
    # result_auc = {}
    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])
    return result


def process_value(v):
    return (lambda x: x.tolist() if isinstance(x, torch.Tensor) else x)(v)


def save_parmet_tune(name_tag, metrics, root):
    csv_columns = ['Metric'] + list(metrics)
    try:
        Data = pd.read_csv(root)[:-1]
    except:
        Data = pd.DataFrame(None, columns=csv_columns)
        Data.to_csv(root, index=False)
    new_lst = [process_value(v) for k, v in metrics.items()]
    v_lst = [f'{name_tag}'] + new_lst
    new_df = pd.DataFrame([v_lst], columns=csv_columns)
    new_Data = pd.concat([Data, new_df])
    # best value
    highest_values = {}
    for column in new_Data.columns:
        try:
            highest_values[column] = new_Data[column].max()
        except:
            highest_values[column] = None
    # concat and save
    Best_list = ['Best'] + pd.Series(highest_values).tolist()[1:]
    # print(Best_list)
    Best_df = pd.DataFrame([Best_list], columns=Data.columns)
    upt_Data = pd.concat([new_Data, Best_df])
    upt_Data.to_csv(root,index=False)
    return upt_Data


                
def test_epoch(
               score_func, 
               data, 
               pos_encoding, 
               batch_size, 
               evaluation_edges, 
               evaluator_hit, 
               evaluator_mrr, 
               use_valedges_as_input):
    predictor.eval()

    # adj_t = adj_t.transpose(1,0)
    train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge = evaluation_edges
    if emb == None: x = data.x
    else: x = emb.weight
    h = data.x
        
    train_val_edge = train_val_edge.to(h.device)
    pos_valid_edge = pos_valid_edge.to(h.device) 
    neg_valid_edge = neg_valid_edge.to(h.device)
    pos_test_edge = pos_test_edge.to(h.device) 
    neg_test_edge = neg_test_edge.to(h.device)
    
    neg_valid_pred = test_edge(score_func, neg_valid_edge, h, data, batch_size)
    pos_valid_pred = test_edge(score_func, pos_valid_edge, h, data, batch_size)
    pos_test_pred = test_edge(score_func, pos_test_edge, h, data, batch_size)
    neg_test_pred = test_edge(score_func, neg_test_edge, h, data, batch_size)
    pos_train_pred = test_edge(score_func, train_val_edge, h, data, batch_size)
    
    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)
    
    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)

    return result
  
    
def train_epoch(
                predictor, 
                optimizer, 
                data, 
                pos_encoding, 
                splits, 
                batch_size):
    predictor.train()

    pos_train_edge = splits['train']['edge'].to(data.x.device)
    neg_train_edge = negative_sampling(
        data.edge_index.to(pos_train_edge.device),
        num_nodes=data.num_nodes,
        num_neg_samples=pos_train_edge.size(0)
    ).t().to(data.x.device)
    if emb == None: 
        x = data.x
        emb_update = 0
    else: 
        x = emb.weight
        emb_update = 1
        
    total_loss = total_examples = 0
    
    # indices = torch.randperm(pos_train_edge.size(0), device=pos_train_edge.device)
    # for start in tqdm(range(0, pos_train_edge.size(0), batch_size)):
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        edge = pos_train_edge[perm].t()
        pos_out = predictor.multidomainforward(data.x,
                                                data.adj_t,
                                                edge,
                                                cndropprobs=[])
        pos_loss = -F.logsigmoid(pos_out).mean()
        edge = neg_train_edge[perm].t()
        neg_out = predictor.multidomainforward(data.x,
                                                data.adj_t,
                                                edge,
                                                cndropprobs=[])

        neg_loss =  -F.logsigmoid(-neg_out).mean()
        loss = pos_loss + neg_loss
        total_loss += loss.item() *  batch_size
        total_examples += batch_size
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
           list(predictor.parameters()), 1.0
        )
        optimizer.step()
    return total_loss / total_examples

def merge_cmd_args(cmd_opt, opt):
  if cmd_opt['beltrami']:
    opt['beltrami'] = False
  if cmd_opt['function'] is not None:
    opt['function'] = cmd_opt['function']
  if cmd_opt['block'] is not None:
    opt['block'] = cmd_opt['block']
  if cmd_opt['attention_type'] != 'scaled_dot':
    opt['attention_type'] = cmd_opt['attention_type']
  if cmd_opt['self_loop_weight'] is not None:
    opt['self_loop_weight'] = cmd_opt['self_loop_weight']
  if cmd_opt['method'] is not None:
    opt['method'] = cmd_opt['method']
  if cmd_opt['step_size'] != 1:
    opt['step_size'] = cmd_opt['step_size']
  if cmd_opt['time'] != 1:
    opt['time'] = cmd_opt['time']
  if cmd_opt['epoch'] != 100:
    opt['epoch'] = cmd_opt['epoch']
  if not cmd_opt['not_lcc']:
    opt['not_lcc'] = False
  if cmd_opt['num_splits'] != 1:
    opt['num_splits'] = cmd_opt['num_splits']
  return opt

def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))

def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)

import random
def set_seed(seed=42):
    random.seed(seed)  # Фиксируем seed для стандартного random
    np.random.seed(seed)  # Фиксируем seed для NumPy
    torch.manual_seed(seed)  # Фиксируем seed для PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # Фиксируем seed для PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)  # Фиксируем seed для всех GPU
    torch.backends.cudnn.deterministic = True  # Опционально: делаем вычисления детерминированными
    torch.backends.cudnn.benchmark = False  # Отключаем автооптимизации для детерминированности

set_seed(999)
      
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=False,
                        default='yamls/cora/gcn.yaml',
                        help='The configuration file path.')
    ### MPLP PARAMETERS ###
    # dataset setting
    parser.add_argument('--data_name', type=str, default='Cora')
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--year', type=int, default=-1)

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_feature', type=str2bool, default='True', help='whether to use node features as input')
    parser.add_argument('--metric', type=str, default='Hits@50', help='main evaluation metric')

    parser.add_argument('--print_summary', type=str, default='')
  
    #NCNC params
    parser.add_argument('--use_valedges_as_input', action='store_true', help="whether to add validation edges to the input adjacency matrix of gnn")
    parser.add_argument('--epochs', type=int, default=40, help="number of epochs")
    parser.add_argument('--dataset', type=str, default="collab")
    parser.add_argument('--testbs', type=int, default=8192, help="batch size for test")
    parser.add_argument('--maskinput', action="store_true", help="whether to use target link removal")

    parser.add_argument('--mplayers', type=int, default=1, help="number of message passing layers")
    parser.add_argument('--nnlayers', type=int, default=3, help="number of mlp layers")
    parser.add_argument('--hidden_dim', type=int, default=32, help="hidden dimension")
    parser.add_argument('--ln', action="store_true", help="whether to use layernorm in MPNN")
    parser.add_argument('--lnnn', action="store_true", help="whether to use layernorm in mlp")
    parser.add_argument('--res', action="store_true", help="whether to use residual connection")
    parser.add_argument('--jk', action="store_true", help="whether to use JumpingKnowledge connection")
    parser.add_argument('--gnndp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--xdp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--tdp', type=float, default=0.3, help="dropout ratio of gnn")
    parser.add_argument('--gnnedp', type=float, default=0.3, help="edge dropout ratio of gnn")
    parser.add_argument('--predp', type=float, default=0.3, help="dropout ratio of predictor")
    parser.add_argument('--preedp', type=float, default=0.3, help="edge dropout ratio of predictor")
    parser.add_argument('--prelr', type=float, default=0.0003, help="learning rate of predictor")
    # detailed hyperparameters
    parser.add_argument('--beta', type=float, default=1)
    
    parser.add_argument('--splitsize', type=int, default=-1, help="split some operations inner the model. Only speed and GPU memory consumption are affected.")
    # parameters used to calibrate the edge existence probability in NCNC
    parser.add_argument('--probscale', type=float, default=5)
    parser.add_argument('--proboffset', type=float, default=3)
    parser.add_argument('--pt', type=float, default=0.5)
    parser.add_argument("--learnpt", action="store_true")

    # For scalability, NCNC samples neighbors to complete common neighbor. 
    parser.add_argument('--trndeg', type=int, default=-1, help="maximum number of sampled neighbors during the training process. -1 means no sample")
    parser.add_argument('--cndeg', type=int, default=-1)
    parser.add_argument("--depth", type=int, default=1, help="number of completion steps in NCNC")
    # gnn used, such as gin, gcn.

    parser.add_argument('--save_gemb', action="store_true", help="whether to save node representations produced by GNN")
    parser.add_argument('--load', type=str, help="where to load node representations produced by GNN")
    parser.add_argument("--loadmod", action="store_true", help="whether to load trained models")
    parser.add_argument("--savemod", action="store_true", help="whether to save trained models")
    
    parser.add_argument("--savex", action="store_true", help="whether to save trained node embeddings")
    parser.add_argument("--loadx", action="store_true", help="whether to load trained node embeddings")
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
    # not used in experiments
    parser.add_argument('--cnprob', type=float, default=0)
    
    # MY PARAMETERS
    parser.add_argument('--mlp_num_layers', type=int, default=3, help="Number of layers in MLP")
    parser.add_argument('--batch_size', type=int, default=65536)

    # optimizer
    
    # gcn
    parser.add_argument('--gcn', type=str2bool, default=False)
    parser.add_argument('--num_layers', type=int, default=3)
    
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=1)
    
    # ncnc decoder
    parser.add_argument('--predictor', type=str, default='nc1')
    parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    # depth, splitsize, probscale, proboffset, trndeg, tstdeg, pt, learnpt, alpha
    parser.add_argument('--tstdeg', type=int, default=-1, help="maximum number of sampled neighbors during the test process")
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument("--increasealpha", action="store_true")
    parser.add_argument('--gnnlr', type=float, default=0.01, help="learning rate of gnn")

    args = parser.parse_args()

    cmd_opt = vars(args)
    try:
      best_opt = best_params_dict[cmd_opt['data_name']]
      opt = {**cmd_opt, **best_opt}
      merge_cmd_args(cmd_opt, opt)
    except KeyError:
      opt = cmd_opt
    args.name_tag = f"{args.data_name}_beltrami{args.beltrami}_mlp_score_epochs{args.epochs}_runs{args.runs}"
    
    opt['beltrami'] = False

    init_seed(999)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    data, splits = get_dataset(opt['dataset_dir'], opt, opt['data_name'], opt['use_valedges_as_input'])
    edge_index = data.edge_index
    print(data)
    emb = None
    if hasattr(data, 'x'):
        if data.x != None:
            x = data.x
            data.x = data.x.to(torch.float)
            data.x = data.x.to(device)
            input_channel = data.x.size(1)
        else:
            emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
            input_channel = args.hidden_channels
    else:
        emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
        input_channel = args.hidden_channels
    
    if args.data_name == "ogbl-citation2":
        opt['metric'] = "MRR"
    if data.x is None:
        opt['use_feature'] = False
    
    if opt['beltrami']:
      print("Applying Beltrami")
      pos_encoding = apply_beltrami(data.to('cpu'), opt).to(device)
      opt['pos_enc_dim'] = pos_encoding.shape[1]
      print(f"pos encoding is {pos_encoding}")
      print(f"pos encoding shape is {pos_encoding.shape}")
      print(f"pos encoding type is {type(pos_encoding)}")
    else:
      pos_encoding = None

    print(data, args.data_name)
    if data.edge_weight is None:
        edge_weight = torch.ones((data.edge_index.size(1), 1))
        print(f"custom edge_weight {edge_weight.size()} added for {args.data_name}")
    data = T.ToSparseTensor()(data)
    data.edge_index = edge_index
    if args.use_valedges_as_input:
        val_edge_index = splits['valid']['edge'].t()
        val_edge_index = to_undirected(val_edge_index)
        full_edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float)
        edge_weight = torch.cat([edge_weight, val_edge_weight], 0)
        A = SparseTensor.from_edge_index(full_edge_index, edge_weight.view(-1), [data.num_nodes, data.num_nodes])
        data.full_adj_t = A
        data.full_edge_index = full_edge_index
        print(data.full_adj_t)
        print(data.adj_t)

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
        'AUC':Logger(args.runs),
        'AP':Logger(args.runs),
        'mrr_hit1':  Logger(args.runs),
        'mrr_hit3':  Logger(args.runs),
        'mrr_hit10':  Logger(args.runs),
        'mrr_hit20':  Logger(args.runs),
        'mrr_hit50':  Logger(args.runs),
        'mrr_hit100':  Logger(args.runs),
    }

    if args.data_name =='ogbl-collab':
        eval_metric = 'Hits@50'
    elif args.data_name =='ogbl-ddi':
        eval_metric = 'Hits@20'
    elif args.data_name =='ogbl-ppa':
        eval_metric = 'Hits@100'
    elif args.data_name =='ogbl-citation2':
        eval_metric = 'MRR'
    elif args.data_name in ['Cora', 'Pubmed', 'Citeseer']:
        eval_metric = 'Hits@100'

    if args.data_name != 'ogbl-citation2':
        pos_train_edge = splits['train']['edge']
        pos_valid_edge = splits['valid']['edge']
        neg_valid_edge = splits['valid']['edge_neg']
        pos_test_edge = splits['test']['edge']
        neg_test_edge = splits['test']['edge_neg']
    
    idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)]
    train_val_edge = pos_train_edge[idx]
    pos_train_edge = pos_train_edge.to(device)
    evaluation_edges = [train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge]
    best_valid_auc = best_test_auc = 2
    best_auc_valid_str = 2

    predictor_dict = {
        "cn0": CN0LinkPredictor,
        "catscn1": CatSCNLinkPredictor,
        "scn1": SCNLinkPredictor,
        "sincn1cn1": IncompleteSCN1Predictor,
        "cn1": CNLinkPredictor,
        "cn1.5": CNhalf2LinkPredictor,
        "cn1res": CNResLinkPredictor,
        "cn2": CN2LinkPredictor,
        "incn1cn1": IncompleteCN1Predictor
    }
    # predictor 
    predfn = predictor_dict[args.predictor]
    if args.predictor != "cn0":
        predfn = partial(predfn, cndeg=args.cndeg)
    if args.predictor in ["cn1", "incn1cn1", "scn1", "catscn1", "sincn1cn1"]:
        # cn1: CNLinkPredictor
        predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, 
                         twolayerlin=args.twolayerlin, beta=args.beta)
    if args.predictor == "incn1cn1":
        predfn = partial(predfn, depth=args.depth, splitsize=args.splitsize, scale=args.probscale, 
                         offset=args.proboffset, trainresdeg=args.trndeg, testresdeg=args.tstdeg, 
                         pt=args.pt, learnablept=args.learnpt, alpha=args.alpha)

    data = data.to(device)
    
    predictor = predfn(data.x.size(1), args.hidden_dim, 1, args.num_layers,
                           args.predp, args.preedp, args.lnnn).to(device)

    parameters = (
      [p for p in predictor.parameters() if p.requires_grad]
    )
    print_model_params(predictor)
    
    best_epoch = 0
    best_metric = 0
    best_results = None

    idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)]
    train_val_edge = pos_train_edge[idx]
    pos_train_edge = pos_train_edge.to(device)
    evaluation_edges = [train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge]
    
    # hyperparameter_search = {'hiddim': [64, 256], "gnndp": [0.0, 0.2, 0.5],
    #                         "xdp": [0.0, 0.3, 0.7], "tdp": [0.0, 0.2],
    #                         "gnnedp": [0.0], "predp": [0.0, 0.05], "preedp": [0.0, 0.4],
    #                         "batch_size": [256, 512, 1024, 2048], "gnnlr": [0.001, 0.0001], "prelr": [0.001, 0.0001]}
    # print_logger.info(f"hypersearch space: {hyperparameter_search}")
    
    hyperparameter_search = {"prelr": [0.01, 0.001, 0.003]} 
    print(f"hypersearch space: {hyperparameter_search}")
    tune_id = wandb.util.generate_id()
    tune_res = {}    
    for prelr, probescale, probeoffset in itertools.product(*hyperparameter_search.values()):
        args.prelr = prelr
        args.probescale = probescale
        args.probeoffset = probeoffset
        id = wandb.util.generate_id()
        tune_res[str(id)] = {}   
        tune_res[str(id)]['prelr'] = prelr
        
        # optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
        #     {'params': predictor.parameters(), 'lr': args.prelr}])
        optimizer = torch.optim.Adam(parameters, lr=prelr, weight_decay=0)
    
        print('#################################          ', 0, '          #################################')
        if opt['gcn']:
            name_tag = f"{args.data_name}_gcn_{server}_{args.runs}"
        else:
            name_tag = f"{args.data_name}_grand_{server}_{args.runs}"
            
        print(tune_res)
        time.sleep(20)
        wandb.init(project="GRAND4NC", name=f"{id}_gnn{args.gnnlr}_pre{args.prelr}_bs{args.batch_size}", config=opt)
        seed = 0
        print('seed: ', seed)
        init_seed(seed)

        best_valid = 0
        kill_cnt = 0
        best_test = 0
        step = 0

        for epoch in tqdm(range(1, opt['epochs'] + 1)):
            start_time = time.time()
            
            loss = train_epoch( predictor, 
                                optimizer, 
                                data, 
                                pos_encoding, 
                                splits, 
                                args.batch_size)
                
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            wandb.log({'train_loss': loss}, step = epoch)
            step += 1
            
            if epoch % args.eval_steps == 0:
                results = test_epoch(
                            predictor, 
                            data, 
                            pos_encoding, 
                            args.batch_size, 
                            evaluation_edges, 
                            evaluator_hit,
                            evaluator_mrr, 
                            args.use_valedges_as_input)
                    
                for key, result in results.items():
                    loggers[key].add_result(0, result)
                    wandb.log({f"Metrics/{key}": result[-1]}, step=epoch)
                    
                current_metric = results[eval_metric][2]
                if current_metric > best_metric:
                    best_epoch = epoch
                    best_metric = current_metric
                    best_results = results
                print(f"Epoch {epoch} completed in {time.time() - start_time:.2f}s")
                print(f"Current Best {current_metric}: {best_metric:.4f} (Epoch {best_epoch})")
            tune_res[str(id)][eval_metric] = best_metric
        save_parmet_tune(str(id), tune_res[str(id)], f'results_grand_gnn/tune{tune_id}_{args.data_name}_lm_mrr.csv')
    df = pd.DataFrame.from_dict(tune_res, orient='index')
    df.to_csv( f'results_grand_gnn/tune{tune_id}_{args.data_name}_lm_mrr.csv', index=False)  
    print(f"Training completed. Best {current_metric}: {best_metric:.4f} (Epoch {best_epoch})")

