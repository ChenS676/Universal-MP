import os
import sys
# Cora, Pubmed, Citeseer, Photo, Computer

import math
from tqdm import tqdm
import numpy as np
import scipy.sparse as ssp
import torch

from torch_geometric.data import DataLoader
# adapted from https://github.com/jcatw/scnn
import torch
from torch_geometric.datasets import Planetoid
import pandas as pd
import scipy.sparse as ssp
import torch
from ogb.linkproppred import  Evaluator
import scipy.sparse as ssp
from torch_geometric.utils import to_undirected
from tqdm import tqdm
import argparse
from torch_geometric.utils import train_test_split_edges
from torch_geometric.datasets import Planetoid, Amazon
from gnn_utils  import evaluate_hits, evaluate_auc, evaluate_mrr

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
    k_list = [20, 50, 100]
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


# def main(args):
#     if args.data_name in ["Cora", "Citeseer", "Pubmed"]:
#         dataset = Planetoid(root="dataset", name=args.data_name)
#         split_edge = randomsplit(dataset)
#         data = dataset[0]
#         data.edge_index = to_undirected(split_edge["train"]["edge"].t())
#         edge_index = data.edge_index
#         data.num_nodes = data.x.shape[0]
#     elif args.data_name in ["Computers", "Photo"]:
#         dataset = Amazon(root="dataset", name=args.data_name)
#         split_edge = randomsplit(dataset)
#         data = dataset[0]
#         data.edge_index = to_undirected(split_edge["train"]["edge"].t())
#         edge_index = data.edge_index
#         data.num_nodes = data.x.shape[0]
        
#     num_nodes = data.num_nodes
#     adj = get_adj_matrix(edge_index, num_nodes)
#     evaluator_hit = Evaluator(name='ogbl-collab')
#     evaluator_mrr = Evaluator(name='ogbl-citation2')

#     test_pos_edge = split_edge['test']["edge"].T
#     test_neg_edge = split_edge['test']["edge_neg"].T

#     # Compute heuristic scores
#     test_pos_pred_CN = CN(adj, test_pos_edge)
#     test_neg_pred_CN = CN(adj, test_neg_edge)

#     test_pos_pred_RA = RA(adj, test_pos_edge)
#     test_neg_pred_RA = RA(adj, test_neg_edge)

#     test_pos_pred_AA = AA(adj, test_pos_edge)
#     test_neg_pred_AA = AA(adj, test_neg_edge)

#     # Evaluate heuristics
#     CN_mrr = evaluate_mrr( evaluator_mrr, test_pos_pred_CN, test_pos_pred_CN)
#     RA_mrr = evaluate_mrr( evaluator_mrr, test_pos_pred_RA, test_pos_pred_RA)
#     AA_mrr = evaluate_mrr( evaluator_mrr, test_pos_pred_AA, test_pos_pred_AA)

#     CN_metric = get_metric_score(evaluator_hit, test_pos_pred_CN, test_neg_pred_CN)
#     RA_metric = get_metric_score(evaluator_hit, test_pos_pred_RA, test_neg_pred_RA)
#     AA_metric = get_metric_score(evaluator_hit, test_pos_pred_AA, test_neg_pred_AA)

#     # Convert metric results into a DataFrame
#     metrics_data = {
#         "Metric": list(CN_metric.keys())+list(CN_mrr.keys()),
#         "CN": list(CN_metric.values())+ list(CN_mrr.values()),
#         "RA": list(RA_metric.values())+ list(RA_mrr.values()),
#         "AA": list(AA_metric.values())+ list(AA_mrr.values()),
#     }

#     df_metrics = pd.DataFrame(metrics_data)
#     # Save results to CSV
#     df_metrics.to_csv(f"{args.data_name}_heuristic.csv", index=False)

def main(args):
    results = []
    
    for _ in range(10):
        if args.data_name in ["Cora", "Citeseer", "Pubmed"]:
            dataset = Planetoid(root="dataset", name=args.data_name)
            split_edge = randomsplit(dataset)
            data = dataset[0]
            data.edge_index = to_undirected(split_edge["train"]["edge"].t())
            edge_index = data.edge_index
            data.num_nodes = data.x.shape[0]
        elif args.data_name in ["Computers", "Photo"]:
            dataset = Amazon(root="dataset", name=args.data_name)
            split_edge = randomsplit(dataset)
            data = dataset[0]
            data.edge_index = to_undirected(split_edge["train"]["edge"].t())
            edge_index = data.edge_index
            data.num_nodes = data.x.shape[0]

        num_nodes = data.num_nodes
        adj = get_adj_matrix(edge_index, num_nodes)
        evaluator_hit = Evaluator(name='ogbl-collab')
        evaluator_mrr = Evaluator(name='ogbl-citation2')

        test_pos_edge = split_edge['test']["edge"].T
        test_neg_edge = split_edge['test']["edge_neg"].T

        # Compute heuristic scores
        test_pos_pred_CN = CN(adj, test_pos_edge)
        test_neg_pred_CN = CN(adj, test_neg_edge)

        test_pos_pred_RA = RA(adj, test_pos_edge)
        test_neg_pred_RA = RA(adj, test_neg_edge)

        test_pos_pred_AA = AA(adj, test_pos_edge)
        test_neg_pred_AA = AA(adj, test_neg_edge)

        # Evaluate heuristics
        CN_mrr = evaluate_mrr(evaluator_mrr, test_pos_pred_CN, test_pos_pred_CN)
        RA_mrr = evaluate_mrr(evaluator_mrr, test_pos_pred_RA, test_pos_pred_RA)
        AA_mrr = evaluate_mrr(evaluator_mrr, test_pos_pred_AA, test_pos_pred_AA)

        CN_metric = get_metric_score(evaluator_hit, test_pos_pred_CN, test_neg_pred_CN)
        RA_metric = get_metric_score(evaluator_hit, test_pos_pred_RA, test_neg_pred_RA)
        AA_metric = get_metric_score(evaluator_hit, test_pos_pred_AA, test_neg_pred_AA)

        results.append({
            "CN": list(CN_metric.values()) + list(CN_mrr.values()),
            "RA": list(RA_metric.values()) + list(RA_mrr.values()),
            "AA": list(AA_metric.values()) + list(AA_mrr.values()),
        })
    
    # Compute mean and variance
    CN_results = np.array([res["CN"] for res in results])
    RA_results = np.array([res["RA"] for res in results])
    AA_results = np.array([res["AA"] for res in results])
    
    mean_results = {
        "Metric": list(CN_metric.keys()) + list(CN_mrr.keys()),
        "CN_mean": np.mean(CN_results, axis=0),
        "RA_mean": np.mean(RA_results, axis=0),
        "AA_mean": np.mean(AA_results, axis=0),
        "CN_var": np.var(CN_results, axis=0),
        "RA_var": np.var(RA_results, axis=0),
        "AA_var": np.var(AA_results, axis=0),
    }
    
    df_mean = pd.DataFrame(mean_results)
    df_mean.to_csv(f"{args.data_name}_heuristic_mean_variance.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='Citeseer')
    args = parser.parse_args()
    main(args)
