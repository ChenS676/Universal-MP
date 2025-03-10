import argparse
import math
import random
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric import datasets
from torch_geometric.data import Data
from torch_geometric.transforms import (BaseTransform, Compose, ToSparseTensor,
                                        NormalizeFeatures, RandomLinkSplit,
                                        ToDevice, ToUndirected)
from torch_geometric.utils import (add_self_loops, degree,
                                   from_scipy_sparse_matrix, index_to_mask,
                                   is_undirected, negative_sampling,
                                   to_undirected, train_test_split_edges, coalesce)
from torch_sparse import SparseTensor    
from snap_dataset import SNAPDataset
from custom_dataset import SyntheticRandom, SyntheticRegularTilling, SyntheticDataset

from torch_geometric.data.collate import collate
from torch_geometric.transforms import RandomLinkSplit
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix



def get_dataset(root, name: str, use_valedges_as_input=False, year=-1):
    if name.startswith('ogbl-'):
        dataset = PygLinkPropPredDataset(name=name, root=root)
        data = dataset[0]
        """
            SparseTensor's value is NxNx1 for collab. due to edge_weight is |E|x1
            NeuralNeighborCompletion just set edge_weight=None
            ELPH use edge_weight
        """
        split_edge = dataset.get_edge_split()
        if name == 'ogbl-collab' and year > 0:  # filter out training edges before args.year
            data, split_edge = filter_by_year(data, split_edge, year)
        if name == 'ogbl-vessel':
            # normalize x, y, z coordinates  
            data.x[0, :] = torch.nn.functional.normalize(data.x[0, :], dim=0)
            data.x[1, :] = torch.nn.functional.normalize(data.x[1, :], dim=0)
            data.x[:, 2] = torch.nn.functional.normalize(data.x[:, 2], dim=0)
            
        if 'edge_weight' in data:
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
            # TEMP FIX: ogbl-collab has directed edges. adj_t.to_symmetric will
            # double the edge weight. temporary fix like this to avoid too dense graph.
            if name == "ogbl-collab":
                data.edge_weight = data.edge_weight/2
        
        if 'edge' in split_edge['train']:
            key = 'edge'
        else:
            key = 'source_node'
        print("-"*20)
        print(f"train: {split_edge['train'][key].shape[0]}")
        print(f"{split_edge['train'][key]}")
        print(f"valid: {split_edge['valid'][key].shape[0]}")
        print(f"test: {split_edge['test'][key].shape[0]}")
        print(f"max_degree:{degree(data.edge_index[0], data.num_nodes).max()}")
        
        data = ToSparseTensor(remove_edge_index=False)(data)
        data.adj_t = data.adj_t.to_symmetric()
        
        # Use training + validation edges for inference on test set.
        if use_valedges_as_input:
            val_edge_index = split_edge['valid']['edge'].t()
            full_edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
            data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, 
                                                    sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
            data.full_adj_t = data.full_adj_t.to_symmetric()
        else:
            data.full_adj_t = data.adj_t
        # make node feature as float
        if data.x is not None:
            data.x = data.x.to(torch.float)

        return data, split_edge

    pyg_dataset_dict = {
        'Cora': (datasets.Planetoid, {'name':'Cora'}),
        'Citeseer': (datasets.Planetoid, {'name':'Citeseer'}),
        'Pubmed': (datasets.Planetoid, {'name':'Pubmed'}),
        'CS': (datasets.Coauthor, {'name':'CS'}),
        'Physics': (datasets.Coauthor, {'name':'physics'}),
        'Computers': (datasets.Amazon, {'name':'Computers'}),
        'Photo': (datasets.Amazon, {'name':'Photo'}),
        'PolBlogs': (datasets.PolBlogs, {}),
        
        'musae-twitch':(SNAPDataset, {'name':'musae-twitch'}),
        'musae-github':(SNAPDataset, {'name':'musae-github'}),
        'musae-facebook':(SNAPDataset, {'name':'musae-facebook'}),
        
        'random-ERDOS_RENYI':(SyntheticRandom, {'name':'ERDOS_RENYI'}),
        'random-Tree': (SyntheticRandom, { 'name':'TREE'}),
        'random-Grid': (SyntheticRandom, { 'name':'GRID'}),
        'random-BA': (SyntheticRandom, { 'name':'BARABASI_ALBERT'}),

        'regulartilling-TRIANGULAR':(SyntheticRegularTilling, {'name':'TRIANGULAR'}),
        'regulartilling-HEXAGONAL':(SyntheticRegularTilling, {'name':'HEXAGONAL'}),
        'regularTilling-SQUARE_GRID':(SyntheticRegularTilling, {'name':'SQUARE_GRID'}),
        
        # TODO docment resource
        'syn-TRIANGULAR':(SyntheticDataset, {'name':'TRIANGULAR'}),
        'syn-GRID':(SyntheticDataset, {'name':'GRID'}),
        
    }

    if name in pyg_dataset_dict:
        dataset_class, kwargs = pyg_dataset_dict[name]
        dataset = dataset_class(root=root, transform=ToUndirected(), **kwargs)
        data, _, _ = collate(
                dataset[0].__class__,
                data_list=list(dataset),
                increment=True,
                add_batch=False,
            )
    else:
        data = load_unsplitted_data(root, name)
    return data, None



def load_unsplitted_data(root,name):
    # read .mat format files
    data_dir = root + '/{}.mat'.format(name)
    # print('Load data from: '+ data_dir)
    import scipy.io as sio
    net = sio.loadmat(data_dir)
    edge_index,_ = from_scipy_sparse_matrix(net['net'])
    data = Data(edge_index=edge_index,num_nodes = torch.max(edge_index).item()+1)
    if is_undirected(data.edge_index) == False: #in case the dataset is directed
        data.edge_index = to_undirected(data.edge_index)
    return data


def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


# random split dataset
def randomsplit(data, val_ratio: float=0.10, test_ratio: float=0.2):
    # coleasce edge_index
    def removerepeated(ei):
        ei = to_undirected(ei.t())
        ei = ei[:, ei[0]<ei[1]]
        return ei

    # double check and doc the resources
    transform = RandomLinkSplit(
        num_val=val_ratio,  
        num_test=test_ratio, 
        is_undirected=True, 
        split_labels=True   
    )

    train_data, val_data, test_data = transform(data)
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = removerepeated(train_data.pos_edge_label_index.t())
    split_edge['train']['edge_neg'] = removerepeated(train_data.neg_edge_label_index.t())
    
    # Validation edges
    num_val = int(val_data.pos_edge_label_index.shape[1])
    val_perm = torch.randperm(num_val)  
    split_edge['valid']['edge'] = removerepeated(val_data.pos_edge_label_index[:, val_perm].t())
    split_edge['valid']['edge_neg'] = removerepeated(val_data.neg_edge_label_index.t())

    # Test edges
    split_edge['test']['edge'] = removerepeated(test_data.pos_edge_label_index.t())
    split_edge['test']['edge_neg'] = removerepeated(test_data.pos_edge_label_index.t())

    construct_sparse_adj(split_edge['train']['edge'], data.num_nodes)
    plot_coo_matrix(split_edge['train']['edge_neg'], name='train_neg.png')
        
        
    data = train_test_split_edges(data, test_ratio, test_ratio)
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    num_val = int(data.val_pos_edge_index.shape[1] * val_ratio/test_ratio)
    data.val_pos_edge_index = data.val_pos_edge_index[:, torch.randperm(data.val_pos_edge_index.shape[1])]
    split_edge['train']['edge'] = removerepeated(torch.cat((data.train_pos_edge_index, data.val_pos_edge_index[:, :-num_val]), dim=-1)).t()
    split_edge['valid']['edge'] = removerepeated(data.val_pos_edge_index[:, -num_val:]).t()
    split_edge['valid']['edge_neg'] = removerepeated(data.val_neg_edge_index).t()
    split_edge['test']['edge'] = removerepeated(data.test_pos_edge_index).t()
    split_edge['test']['edge_neg'] = removerepeated(data.test_neg_edge_index).t()
    
    plot_coo_matrix(split_edge, num_nodes=data.num_nodes)
    return split_edge


def plot_coo_matrix(m: coo_matrix, name: str):
    """
    Plot the COO matrix.

    Parameters:
    - m: coo_matrix, input COO matrix
    - name: str, output file name
    """

    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='white')
    ax.plot(m.col, m.row, 's', color='black', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(name)
    return ax


def construct_sparse_adj(edge_index, num_node) -> coo_matrix:
    """
    Construct a sparse adjacency matrix from an edge index.

    Parameters:
    - edge_index: np.array or tuple, edge index
    """
    # Resource: https://stackoverflow.com/questions/22961541/python-matplotlib-plot-sparse-matrix-pattern

    if type(edge_index) == tuple:
        edge_index = np.concatenate([[edge_index[0].numpy()],
                                     [edge_index[1].numpy()]], axis=0)
    elif type(edge_index) != np.ndarray:
        edge_index.numpy()

    if edge_index.shape[0] > edge_index.shape[1]:
        edge_index = edge_index.T

    rows, cols = edge_index[0, :], edge_index[1, :]
    vals = np.ones_like(rows)
    m = coo_matrix((vals, (rows, cols)), shape=(num_node, num_node))
    return m



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_data_split(root, name: str, val_ratio, test_ratio, run=0):
    data_folder = Path(root) / name
    data_folder.mkdir(parents=True, exist_ok=True)
    
    data, _ = get_dataset(root, name)
    
    file_path = data_folder / f"split{run}_{int(100*val_ratio)}_{int(100*test_ratio)}.pt"
    if file_path.exists():
        split_edge = torch.load(file_path)
        print(f"load split edges from {file_path}")
    else:
        split_edge = randomsplit(data)
        torch.save(split_edge, file_path)
        print(f"save split edges to {file_path}")
    
    data.edge_index = to_undirected(split_edge["train"]["edge"].t())
    data.num_features = data.x.shape[0] if data.x is not None else 0
    print("-"*20)
    print(f"train: {split_edge['train']['edge'].shape[0]}")
    print(f"{split_edge['train']['edge'][:10,:]}")
    print(f"valid: {split_edge['valid']['edge'].shape[0]}")
    print(f"test: {split_edge['test']['edge'].shape[0]}")
    # print(f"max_degree:{degree(data.edge_index[0], data.num_nodes).max()}")
    return data, split_edge


def data_summary(name: str, data: Data, header=False, latex=False):
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    n_degree = data.adj_t.sum(dim=1).to(torch.float)
    avg_degree = n_degree.mean().item()
    degree_std = n_degree.std().item()
    max_degree = n_degree.max().long().item()
    density = num_edges / (num_nodes * (num_nodes - 1) / 2)
    if data.x is not None:
        attr_dim = data.x.shape[1]
    else:
        attr_dim = '-' # no attribute

    if latex:
        latex_str = ""
        if header:
            latex_str += r"""
            \begin{table*}[ht]
            \begin{center}
            \resizebox{0.85\textwidth}{!}{
            \begin{tabular}{lccccccc}
                \toprule
                \textbf{Dataset} & \textbf{\#Nodes} & \textbf{\#Edges} & \textbf{Avg. node deg.} & \textbf{Std. node deg.} & \textbf{Max. node deg.} & \textbf{Density} & \textbf{Attr. Dimension}\\
                \midrule"""
        latex_str += f"""
                \\textbf{{{name}}}"""
        latex_str += f""" & {num_nodes} & {num_edges} & {avg_degree:.2f} & {degree_std:.2f} & {max_degree} & {density*100:.4f}\% & {attr_dim} \\\\"""
        latex_str += r"""
                \midrule"""
        if header:
            latex_str += r"""
            \bottomrule
            \end{tabular}
            }
            \end{center}
            \end{table*}"""
        print(latex_str)
    else:
        print("-"*30+'Dataset and Features'+"-"*60)
        print("{:<10}|{:<10}|{:<10}|{:<15}|{:<15}|{:<15}|{:<10}|{:<15}"\
            .format('Dataset','#Nodes','#Edges','Avg. node deg.','Std. node deg.','Max. node deg.', 'Density','Attr. Dimension'))
        print("-"*110)
        print("{:<10}|{:<10}|{:<10}|{:<15.2f}|{:<15.2f}|{:<15}|{:<9.4f}%|{:<15}"\
            .format(name, num_nodes, num_edges, avg_degree, degree_std, max_degree, density*100, attr_dim))
        print("-"*110)



def initialize(data, method):
    if data.x is None:
        if method == 'one-hot':
            data.x = F.one_hot(torch.arange(data.num_nodes),num_classes=data.num_nodes).float()
            input_size = data.num_nodes
        elif method == 'trainable':
            node_emb_dim = 512
            emb = torch.nn.Embedding(data.num_nodes, node_emb_dim)
            data.emb = emb
            input_size = node_emb_dim
        else:
            raise NotImplementedError
    else:
        input_size = data.num_features
    return data, input_size

def initial_embedding(data, hidden_channels, device):
    embedding= torch.nn.Embedding(data.num_nodes, hidden_channels).to(device)
    torch.nn.init.xavier_uniform_(embedding.weight)
    return embedding


def create_input(data):
    if hasattr(data, 'emb') and data.emb is not None:
        x = data.emb.weight
    else:
        x = data.x
    return x


# adopted from "https://github.com/melifluos/subgraph-sketching/tree/main"
def filter_by_year(data, split_edge, year):
    """
    remove edges before year from data and split edge
    @param data: pyg Data, pyg SplitEdge
    @param split_edges:
    @param year: int first year to use
    @return: pyg Data, pyg SplitEdge
    """
    selected_year_index = torch.reshape(
        (split_edge['train']['year'] >= year).nonzero(as_tuple=False), (-1,))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
    train_edge_index = split_edge['train']['edge'].t()
    # create adjacency matrix
    new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight.unsqueeze(-1)
    return data, split_edge



def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


if __name__ == '__main__':
    dataset = 'ogbl-collab'
    dataset_dir = '/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/baselines/mplp/data'

    data, split_edge = get_data_split(dataset_dir, 'ogbl-collab', 0.15, 0.05, run=0)
    data, split_edge = get_data_split(dataset_dir, 'ogbl-vessel', 0.15, 0.05, run=0)
    data, split_edge = get_data_split(dataset_dir, 'ogbl-ppa', 0.15, 0.05, run=0)
    data, split_edge = get_data_split(dataset_dir, 'ogbl-ddi', 0.15, 0.05, run=0)
    data, split_edge = get_data_split(dataset_dir, 'ogbl-citation2', 0.15, 0.05, run=0)