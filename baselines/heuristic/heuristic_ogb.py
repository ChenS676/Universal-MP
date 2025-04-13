import sys
import math
from tqdm import tqdm
import random
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
import torch
from torch_sparse import spspmm
import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)
# adapted from https://github.com/jcatw/scnn
import torch
import random
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T 
from sklearn.preprocessing import normalize
import json
from torch_geometric.transforms import RandomLinkSplit
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
import matplotlib.pyplot as plt
import scipy.sparse as ssp
from typing import Dict
from torch_geometric.data import Dataset
import torch
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import scipy.sparse as ssp
from torch_geometric.utils import to_undirected, train_test_split_edges
from torch_geometric.data import Data
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from tqdm import tqdm
import argparse
from gnn_utils  import evaluate_hits, evaluate_auc, evaluate_mrr


def CN(A, edge_index, batch_size=100000):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src = edge_index[0, ind].cpu().numpy()
        dst = edge_index[1, ind].cpu().numpy()
    
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).squeeze()
        scores.append(cur_scores)
        # print('max cn: ', np.concatenate(scores, 0).max())
    return torch.FloatTensor(np.concatenate(scores, 0))


def AA(A, edge_index, batch_size=100000):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / np.log(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src = edge_index[0, ind].cpu().numpy()
        dst = edge_index[1, ind].cpu().numpy()
    
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    
    return torch.FloatTensor(scores)

def RA(A, edge_index, batch_size=100000):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / (A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src = edge_index[0, ind].cpu().numpy()
        dst = edge_index[1, ind].cpu().numpy()
    
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores)

def PPR(A, edge_index):
    # The Personalized PageRank heuristic score.
    # Need install fast_pagerank by "pip install fast-pagerank"
    # Too slow for large datasets now.
    from fast_pagerank import pagerank_power
    num_nodes = A.shape[0]
    src_index, sort_indices = torch.sort(edge_index[0])
    dst_index = edge_index[1, sort_indices]
    edge_index = torch.stack([src_index, dst_index])
    #edge_index = edge_index[:, :50]
    scores = []
    visited = set([])
    j = 0
    for i in tqdm(range(edge_index.shape[1])):
        if i < j:
            continue
        src = edge_index[0, i]
        personalize = np.zeros(num_nodes)
        personalize[src] = 1
        ppr = pagerank_power(A, p=0.85, personalize=personalize, tol=1e-7)
        j = i
        while edge_index[0, j] == src:
            j += 1
            if j == edge_index.shape[1]:
                break
        all_dst = edge_index[1, i:j]
        cur_scores = ppr[all_dst]
        if cur_scores.ndim == 0:
            cur_scores = np.expand_dims(cur_scores, 0)
        scores.append(np.array(cur_scores))

    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores)



def get_adj_matrix(edge_index, num_nodes):
    adj = ssp.csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                          shape=(num_nodes, num_nodes))
    return adj


def get_metric_score(evaluator_hit, pos_test_pred, neg_test_pred):

    result = {}
    k_list = [1, 10, 20, 50, 100]
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)
    # result_hit = {}
    for K in k_list:
        result[f'Hits@{K}'] = result_hit_test[f'Hits@{K}']

    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])
    result_auc_test = evaluate_auc(test_pred, test_true)
    
    # result_auc = {}
    result['AUC'] = result_auc_test['AUC']
    result['AP'] = result_auc_test['AP']
    return result

        

def main(args):
    dataset = PygLinkPropPredDataset(name=args.data_name)
    data = dataset[0]
    
    if args.data_name == 'ogbl-citation2':
        edge_index = data.edge_index 
    else:   
        edge_index = to_undirected(data.edge_index)
    num_nodes = data.num_nodes
    adj = get_adj_matrix(edge_index, num_nodes)
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    
    split_edge = dataset.get_edge_split()
    if args.data_name != 'ogbl-citation2':
        test_pos_edge = split_edge['test']["edge"].T
        test_neg_edge = split_edge['test']["edge_neg"].T
    else:
        source, target = split_edge['test']['source_node'],  split_edge['test']['target_node']
        test_pos_edge = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1)
        test_neg_edge = split_edge['test']['target_node_neg']
    
    # Compute heuristic scores
    test_pos_pred_CN = CN(adj, test_pos_edge)
    test_neg_pred_CN = CN(adj, test_neg_edge)

    test_pos_pred_RA = RA(adj, test_pos_edge)
    test_neg_pred_RA = RA(adj, test_neg_edge)

    test_pos_pred_AA = AA(adj, test_pos_edge)
    test_neg_pred_AA = AA(adj, test_neg_edge)

    # Evaluate heuristics
    if args.data_name == 'ogbl-citation2':
        CN_metric = evaluate_mrr( evaluator_mrr, test_pos_pred_CN, test_pos_pred_CN)
        RA_metric = evaluate_mrr( evaluator_mrr, test_pos_pred_RA, test_pos_pred_RA)
        AA_metric = evaluate_mrr( evaluator_mrr, test_pos_pred_AA, test_pos_pred_AA)
    else:
        CN_metric = get_metric_score(evaluator_hit, test_pos_pred_CN, test_neg_pred_CN)
        RA_metric = get_metric_score(evaluator_hit, test_pos_pred_RA, test_neg_pred_RA)
        AA_metric = get_metric_score(evaluator_hit, test_pos_pred_AA, test_neg_pred_AA)

    # Convert metric results into a DataFrame
    metrics_data = {
        "Metric": list(CN_metric.keys()),
        "CN": list(CN_metric.values()),
        "RA": list(RA_metric.values()),
        "AA": list(AA_metric.values())
    }

    df_metrics = pd.DataFrame(metrics_data)
    # Save results to CSV
    df_metrics.to_csv(f"{args.data_name}_heuristic.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='ogbl-ppa')
    args = parser.parse_args()
    main(args)
    
# import sys
# import math
# import numpy as np
# import pandas as pd
# import argparse
# import torch
# from tqdm import tqdm
# from torch_geometric.utils import to_undirected
# from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
# from gnn_utils import evaluate_hits, evaluate_auc, evaluate_mrr


# def repeat_experiments(adj, evaluator_hit, evaluator_mrr, test_pos_edge, test_neg_edge, data_name, num_runs=10):
#     results = {"Metric": []}
#     heuristics = {"CN": CN, "RA": RA, "AA": AA}
    
#     for heuristic_name, heuristic_func in heuristics.items():
#         all_results = []
#         for _ in range(num_runs):
#             test_pos_pred = heuristic_func(adj, test_pos_edge)
#             test_neg_pred = heuristic_func(adj, test_neg_edge)
            
#             if data_name == 'ogbl-citation2':
#                 metric_result = evaluate_mrr(evaluator_mrr, test_pos_pred, test_neg_pred)
#             else:
#                 metric_result = get_metric_score(evaluator_hit, test_pos_pred, test_neg_pred)
            
#             all_results.append(list(metric_result.values()))
            
#         all_results = np.array(all_results)
#         mean_results = np.mean(all_results, axis=0)
#         var_results = np.var(all_results, axis=0)
        
#         results[heuristic_name + "_Mean"] = mean_results
#         results[heuristic_name + "_Var"] = var_results
    
#     results["Metric"] = list(metric_result.keys())
#     df_metrics = pd.DataFrame(results)
#     df_metrics.to_csv(f"{data_name}_heuristic_results.csv", index=False)


# def main(args):
#     dataset = PygLinkPropPredDataset(name=args.data_name)
#     data = dataset[0]
#     edge_index = to_undirected(data.edge_index) if args.data_name != 'ogbl-citation2' else data.edge_index
#     num_nodes = data.num_nodes
#     adj = get_adj_matrix(edge_index, num_nodes)
#     evaluator_hit = Evaluator(name='ogbl-collab')
#     evaluator_mrr = Evaluator(name='ogbl-citation2')
    
#     split_edge = dataset.get_edge_split()
#     if args.data_name != 'ogbl-citation2':
#         test_pos_edge = split_edge['test']["edge"].T
#         test_neg_edge = split_edge['test']["edge_neg"].T
#     else:
#         source, target = split_edge['test']['source_node'], split_edge['test']['target_node']
#         test_pos_edge = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1)
#         test_neg_edge = split_edge['test']['target_node_neg']
    
#     repeat_experiments(adj, evaluator_hit, evaluator_mrr, test_pos_edge, test_neg_edge, args.data_name)
    

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='homo')
#     parser.add_argument('--data_name', type=str, default='ogbl-citation2')
#     args = parser.parse_args()
#     main(args)
