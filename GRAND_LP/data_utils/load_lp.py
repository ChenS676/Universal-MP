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


############################################################################################################## 
# def get_dataset(root, name: str, opt: dict, use_valedges_as_input=False, device='cpu', year=-1):
#     if name.startswith('ogbl-'):
#         dataset = PygLinkPropPredDataset(name=name, root=root)

#         data = dataset[0]

#         # Idea from original repo: https://github.com/chuanqichen/cs224w/blob/main/ddi/gnn_augmented_node2vec.py
#         if name == 'ogbl-ddi':
#           data.x = get_features(data.num_nodes)
#           emb = Node2Vec(data.edge_index, opt['hidden_dim'] - data.x.size(1), walk_length=20,
#                       context_size=10, walks_per_node=10)
#           embeddings = emb()

#           data.x = torch.cat([data.x.to(device), embeddings.to(device)], dim=1)
#           dataset.data = data
#         else:
#           emb = None
        
#         if data.x is not None:
#           data.x = data.x.to(torch.float)
#         # TODO: CHECK (VERY LONGG)
#         # if opt['use_lcc']:
#         #   data, lcc, _ = use_lcc(data)
#         #   dataset.data = data
#           # print(data.x.shape, data.edge_index.shape)
#           # lcc = get_largest_connected_component(dataset)

#           # data.x = data.x[lcc]

#           # row, col = dataset.data.edge_index.numpy()
#           # edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
#           # data.edge_index = torch.tensor(remap_edges(edges, get_node_mapper(lcc)))
#           # print(data.x.shape, data.edge_index.shape)
        
#         split_edge = dataset.get_edge_split()

#         if name == 'ogbl-collab' and year > 0:
#             data, split_edge = filter_by_year(data, split_edge, year)
        
#         if name == 'ogbl-vessel':
#             data.x[:, 0] = torch.nn.functional.normalize(data.x[:, 0], dim=0)
#             data.x[:, 1] = torch.nn.functional.normalize(data.x[:, 1], dim=0)
#             data.x[:, 2] = torch.nn.functional.normalize(data.x[:, 2], dim=0)
          
#         if 'edge_weight' in data:
#             data.edge_weight = data.edge_weight.view(-1).to(torch.float)
#             if name == "ogbl-collab":
#                 data.edge_weight = data.edge_weight / 2

#         data = ToSparseTensor(remove_edge_index=False)(data)
#         data.adj_t = data.adj_t.to_symmetric()

#         if use_valedges_as_input:
#             val_edge_index = split_edge['valid']['edge_index'].t()
#             full_edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
#             data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, 
#                                                     sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
#             data.full_adj_t = data.full_adj_t.to_symmetric()
#             if opt['rewiring'] is not None:
#                 data.edge_index = full_edge_index.copy()
#                 data = rewire(data, opt, root)
#         else:
#             data.full_adj_t = data.adj_t
#             if opt['rewiring'] is not None:
#                 data = rewire(data, opt, root)
        
#         # if name != 'ogbl-ddi':    
#         #   data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
#         print(data)
#         return data, split_edge, emb

#     pyg_dataset_dict = {
#         'Cora': (datasets.Planetoid, {'name':'Cora'}),
#         'Citeseer': (datasets.Planetoid, {'name':'Citeseer'}),
#         'Pubmed': (datasets.Planetoid, {'name':'Pubmed'}),
#         'CS': (datasets.Coauthor, {'name':'CS'}),
#         'Physics': (datasets.Coauthor, {'name':'physics'}),
#         'Computers': (datasets.Amazon, {'name':'Computers'}),
#         'Photo': (datasets.Amazon, {'name':'Photo'}),
#         'PolBlogs': (datasets.PolBlogs, {}),
#     }

#     if name in pyg_dataset_dict:
#         dataset_class, kwargs = pyg_dataset_dict[name]
#         dataset = dataset_class(root=root, transform=ToUndirected(), **kwargs)
#         data, _, _ = collate(
#                 dataset[0].__class__,
#                 data_list=list(dataset),
#                 increment=True,
#                 add_batch=False,
#             )
#         undirected = data.is_undirected()
#         split_edge = get_edge_split(data,
#                    undirected,
#                    device,
#                    0.15,
#                    0.05,
#                    True,
#                    True)
#         data.adj_t = SparseTensor.from_edge_index(data.edge_index, 
#                                                   sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
#     else:
#         data = load_unsplitted_data(root, name)
    
#     if opt['rewiring'] is not None:
#         data = rewire(data, opt, root)
    
#     return data, split_edge, None

# from collections import Counter
# # For ogbl-ddi
# def get_features(n_nodes):
#     with open(PAGERANK_FILENAME, "r") as f:
#         contents = f.read()
#         pagerank_dict = ast.literal_eval(contents)
#     pagerank_vals = torch.FloatTensor(list(pagerank_dict.values())).reshape((n_nodes, 1))
    
#     # pagerank_vals = (pagerank_vals - pagerank_vals.mean()) / pagerank_vals.std() 
    
#     with open(CLUSTER_FILENAME, "r") as f:
#         contents = f.read()
#         clustering_dict = ast.literal_eval(contents)
#     cluster_vals = torch.FloatTensor(list(clustering_dict.values())).reshape((n_nodes, 1))
    
#     # cluster_vals = (cluster_vals - cluster_vals.mean()) / cluster_vals.std()
#     # # ERROR
#     # with open(DEGREE_FILENAME, "rb") as f:
#     #     degree_dict = pickle.load(f)
#     # print(list(degree_dict.values()))
#     # # degree_vals = torch.FloatTensor(list(degree_dict.values())).reshape((n_nodes, 1))
#     # # My:
#     # degree_vals = torch.FloatTensor([
#     #   sum(counter.values()) / len(counter.values()) if isinstance(counter, Counter) and len(counter) > 0 else 0
#     #   for counter in degree_dict.values()
#     # ]).reshape((n_nodes, 1))

#     # print(degree_vals)
#     # raise(0)
#     # with open(CENTRALITY_FILENAME, "rb") as f:
#     #     centrality_dict = pickle.load(f)
#     # # centrality_vals = torch.FloatTensor(list(centrality_dict.values())).reshape((n_nodes, 1))
#     # centrality_vals = [
#     #     sum(v.values()) / len(v) if isinstance(v, dict) and len(v) > 0 else 0
#     #     for v in centrality_dict.values()
#     # ]
#     # print(centrality_vals)
#     # raise(0)
#     ones = torch.ones((n_nodes, 1))
#     # features = torch.cat((ones, pagerank_vals, cluster_vals, centrality_vals, degree_vals), 1)
#     features = torch.cat((ones, pagerank_vals, cluster_vals, cluster_vals, cluster_vals), 1)
#     return features

# def get_edge_split(data: Data,
#                    undirected: bool,
#                    device: Union[str, int],
#                    val_pct: float,
#                    test_pct: float,
#                    include_negatives: bool,
#                    split_labels: bool):
#     transform = T.Compose([
#         T.NormalizeFeatures(),
#         T.ToDevice(device),
#         RandomLinkSplit(is_undirected=undirected,
#                         num_val=val_pct,
#                         num_test=test_pct,
#                         add_negative_train_samples=include_negatives,
#                         split_labels=split_labels),

#     ])
#     del data.adj_t, data.e_id, data.batch_size, data.n_asin, data.n_id
#     print(data)

#     train_data, val_data, test_data = transform(data)
#     return {'train': train_data, 'valid': val_data, 'test': test_data}

# def filter_by_year(data, split_edge, year):
#     """
#     remove edges before year from data and split edge
#     @param data: pyg Data, pyg SplitEdge
#     @param split_edges:
#     @param year: int first year to use
#     @return: pyg Data, pyg SplitEdge
#     """
#     selected_year_index = torch.reshape(
#         (split_edge['train']['year'] >= year).nonzero(as_tuple=False), (-1,))
#     split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
#     split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
#     split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
#     train_edge_index = split_edge['train']['edge'].t()
#     # create adjacency matrix
#     new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
#     new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
#     data.edge_index = new_edge_index
#     data.edge_weight = new_edge_weight.unsqueeze(-1)
#     return data, split_edge

# def load_unsplitted_data(root,name):
#     '''
#     This function is used to load and prepare graph data in the format ".mat" 
#     for later use in PyTorch Geometric library models
#     '''
#     # read .mat format files
#     data_dir = root + '/{}.mat'.format(name)
#     # print('Load data from: '+ data_dir)
#     net = sio.loadmat(data_dir)
#     edge_index,_ = from_scipy_sparse_matrix(net['net'])
#     data = Data(edge_index=edge_index,num_nodes = torch.max(edge_index).item()+1)
#     if is_undirected(data.edge_index) == False: #in case the dataset is directed
#         data.edge_index = to_undirected(data.edge_index)
#     return data
