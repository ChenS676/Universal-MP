import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from ogb.linkproppred import PygLinkPropPredDataset
from torch_sparse import SparseTensor
from torch_geometric import datasets
from torch_geometric.utils import (add_self_loops, degree,
                                   from_scipy_sparse_matrix, index_to_mask,
                                   is_undirected, negative_sampling,
                                   to_undirected, train_test_split_edges, coalesce)
from torch_geometric.transforms import (BaseTransform, Compose, ToSparseTensor,
                                        NormalizeFeatures, RandomLinkSplit,
                                        ToDevice, ToUndirected)
from torch_geometric.utils import add_self_loops, is_undirected, to_dense_adj, \
  dense_to_sparse, to_undirected
import torch_geometric.transforms as T
from torch_geometric.nn import Node2Vec
from data_utils.lcc import *
from data_utils.graph_rewiring import *
from torch_geometric.datasets import Planetoid, Amazon

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

CLUSTER_FILENAME = f"{ROOT_DIR}/data/ddi_features/clustering.txt"
PAGERANK_FILENAME = f"{ROOT_DIR}/data/ddi_features/pagerank.txt"
DEGREE_FILENAME = f"{ROOT_DIR}/data/ddi_features/degree.pkl"
CENTRALITY_FILENAME = f"{ROOT_DIR}/data/ddi_features/centrality.pkl"

# random split dataset
def randomsplit(dataset, val_ratio: float=0.10, test_ratio: float=0.2):
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
    data.edge_weight = None 
    
    data.adj_t = SparseTensor.from_edge_index(edge_index, 
                    sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    data.max_x = -1
    # if name == "ogbl-collab":
    #     data.edge_weight = data.edge_weight/2
        
    if name == "ogbl-ppa":
        data.x = torch.argmax(data.x, dim=-1).unsqueeze(-1).float()
        data.max_x = torch.max(data.x).item()
    elif name == "ogbl-ddi":
        data.x = torch.arange(data.num_nodes).unsqueeze(-1).float()
        data.max_x = data.num_nodes
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
        if opt['rewiring'] is not None:
            data.edge_index = full_edge_index.copy()
            data = rewire(data, opt, root)
    else:
        data.full_adj_t = data.adj_t
        if opt['rewiring'] is not None:
            data = rewire(data, opt, root)
    return data, split_edge
 
def rewire(data, opt, data_dir):
    rw = opt['rewiring']
    if rw == 'two_hop':
        data = get_two_hop(data)
    elif rw == 'gdc':
        data = apply_gdc(data, opt)
    # Didn't works
    elif rw == 'pos_enc_knn':
        data = apply_pos_dist_rewire(data, opt, data_dir)
    return data