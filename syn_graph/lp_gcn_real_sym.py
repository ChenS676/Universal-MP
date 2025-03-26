# adopted from benchmarking/exist_setting_ogb: Run models on ogbl-collab, ogbl-ppa, and ogbl-citation2 under the existing setting.
# python gnn_ogb_heart.py  --use_valedges_as_input  --data_name ogbl-collab  --gnn_model GCN --hidden_channels 256 --lr 0.001 --dropout 0.  --num_layers 3 --num_layers_predictor 3 --epochs 9999 --kill_cnt 100  --batch_size 65536 
# OBGL-PPA,DDI, CITATION2, VESSEL, COLLAB
# basic idea is to replace diffusion operator in mpnn and say whether it works better in ogbl-collab and citation2
# and then expand to synthetic graph
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

# from logger import Logger
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.utils import to_networkx, to_undirected
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from baselines.gnn_utils  import evaluate_hits, evaluate_auc, evaluate_mrr
from torch_geometric.utils import negative_sampling
from tqdm import tqdm 
from graphgps.utility.utils import mvari_str2csv, random_sampling_ogb
import torch
from torch_geometric.data import Data
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
import networkx as nx
import random
from torch_geometric.datasets import Planetoid
from syn_random import (init_regular_tilling, 
                        init_pyg_regtil, 
                        RegularTilling, 
                        local_edge_rewiring,
                        nx2Data_split)
from baselines.gnn_utils import (get_root_dir, 
                                 get_logger, 
                                 get_config_dir, 
                                 Logger, 
                                 init_seed, 
                                 save_emb)
from baselines.gnn_utils import (GCN, 
                                 GAT, 
                                 SAGE, 
                                 GIN, 
                                 MF, 
                                 DGCNN, 
                                 GCN_seal, 
                                 SAGE_seal, 
                                 DecoupleSEAL, 
                                 mlp_score, 
                                 dot_product, 
                                 ChebGCN, 
                                 MixHopGCN)
from graph_generation import generate_graph, GraphType
from automorphism import run_wl_test_and_group_nodes, compute_automorphism_metrics

dir_path = get_root_dir()
log_print = get_logger('testrun', 'log', get_config_dir())


def save_new_results(loggers, data_name, num_node, file_name='test_results_0.25_0.5.csv'):
    new_data = []
    
    for key in loggers.keys():
        if key == 'AUC':
            if len(loggers[key].results[0]) > 0:
                print(key)
                best_valid, best_valid_mean, mean_list, var_list, test_res = loggers[key].print_statistics()

                # Prepare row data
                new_data.append([data_name, num_node, key, best_valid, best_valid_mean, mean_list, var_list, test_res])
        
        # Merge and save the new results with the old ones
    load_and_merge_data(new_data, data_name, num_node, file_name)


def load_and_merge_data(new_data, data_name, num_node, file_name='test_results.csv'):
    try:
        # Try to read the existing CSV file
        old_data = pd.read_csv(file_name)
        
        # Merge the new data (convert new_data to a DataFrame)
        new_data_df = pd.DataFrame(new_data, columns=['data_name', 'num_node', 'Metric', 'Best Valid', 'Best Valid Mean', 'Mean List', 'Variance List', 'Test Result'])
        
        # Concatenate only the necessary columns
        merged_data = pd.concat([old_data[['data_name', 'num_node', 'Metric', 'Best Valid', 'Best Valid Mean', 'Mean List', 'Variance List', 'Test Result']], new_data_df], ignore_index=True)
    
    except FileNotFoundError:
        # If the file doesn't exist, create a new DataFrame for the new data
        new_data_df = pd.DataFrame(new_data, columns=['data_name', 'num_node', 'Metric', 'Best Valid', 'Best Valid Mean', 'Mean List', 'Variance List', 'Test Result'])
        merged_data = new_data_df
    
    # Save the merged data back to the CSV file
    file_exists = os.path.exists(file_name)
    merged_data.to_csv(
        file_name,
        mode='a',                     # Append mode
        index=False,                  # Don't write row numbers
        header=not file_exists        # Write header only if file doesn't exist
    )
    # merged_data.to_csv(file_name, index=False)
    print(f'Merged data saved to {file_name}')
    
    
def train(model, score_func, split_edge, train_pos, data, emb, optimizer, batch_size, pos_train_weight, data_name, remove_edge_aggre):
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
        num_nodes = x.size(0)
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
        h = model(x, adj)
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
    
    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x1.cpu(), x2.cpu()]

    # visualize_predictions(pos_train_pred, neg_valid_pred,
    #                     pos_valid_pred, neg_valid_pred,
    #                     pos_test_pred, neg_test_pred)
    return result, score_emb


def evaluate_mrr(evaluator, pos_val_pred, neg_val_pred):
    
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    # neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    
    mrr_output =  eval_mrr(pos_val_pred, neg_val_pred)


    valid_mrr =mrr_output['mrr_list'].mean().item()
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

    
    return results['MRR']

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



def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    k_list = [1, 5, 50, 100]
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
    
    mrr_train =  evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred)
    mrr_valid =  evaluate_mrr(evaluator_mrr, pos_train_pred, neg_val_pred)
    mrr_test =  evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred)
    
    # result_auc = {}
    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])
    result['MRR'] =  (mrr_train, mrr_valid, mrr_test)
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



def get_graph_statistics(G, graph_name="Graph", perturbation="None"):
    """Calculate and return statistics of a NetworkX graph."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    
    # Compute degree statistics
    degrees = [deg for node, deg in G.degree()]
    avg_degree = sum(degrees) / num_nodes if num_nodes > 0 else 0
    min_degree = min(degrees) if degrees else None
    max_degree = max(degrees) if degrees else None

    # Create a dictionary with the statistics
    graph_key = f"{graph_name}_Perturbation_{perturbation}"
    
    stats = {
        "Graph Name": graph_key,
        "Number of Nodes": num_nodes,
        "Number of Edges": num_edges,
        "Density": density,
        "Average Degree": avg_degree,
        "Min Degree": min_degree,
        "Max Degree": max_degree,
    }

    return stats

def save_graph_statistics(stats, filename="graph_statistics.csv"):
    """Load existing data, merge new data, and save back to CSV."""
    # Convert new stats to DataFrame
    new_df = pd.DataFrame.from_dict(stats, orient='index')

    # Check if file exists, and load old data if present
    if os.path.exists(filename):
        old_df = pd.read_csv(filename, index_col=0)
        # Merge old and new data, ensuring uniqueness
        updated_df = pd.concat([old_df, new_df]).drop_duplicates()
    else:
        updated_df = new_df

    # Save the merged DataFrame back to CSV
    updated_df.to_csv(filename)
    print(f"Updated graph statistics saved to {filename}")


def create_disjoint_graph(data):
    """
    Creates two disjoint copies of a real-world graph (e.g., Cora).

    Args:
        data (Data): PyG Data object representing the original graph.

    Returns:
        Data: PyG Data object representing the new merged graph.
    """
    num_nodes = data.num_nodes

    # Convert PyG data to NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Create second copy with relabeled nodes
    G2 = nx.relabel_nodes(G, lambda x: x + num_nodes)

    # Merge both graphs
    merged_graph = nx.compose(G, G2)
    merged_edge_index = torch.tensor(list(merged_graph.edges)).T

    return Data(edge_index=merged_edge_index, num_nodes=2 * num_nodes)


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
    num_nodes = graph_data.num_nodes // 2  # Each copy has num_nodes/2 nodes

    # Compute the number of edges for each category
    inter_edges = int(total_edges * inter_ratio)
    intra_edges = total_edges - inter_edges  # Remaining edges go within each copy

    # --- Add Inter-Copy Edges ---
    inter_edges_list = [
        (random.randint(0, num_nodes - 1), random.randint(num_nodes, 2 * num_nodes - 1))
        for _ in range(inter_edges)
    ]

    # --- Add Intra-Copy Edges ---
    intra_edges_list = []
    for _ in range(intra_edges):
        copy = random.choice([0, 1])  # Choose whether to add in the first or second copy
        base_offset = num_nodes * copy  # Offset for second copy
        u, v = random.sample(range(base_offset, base_offset + num_nodes), 2)
        intra_edges_list.append((u, v))

    # Convert edges to PyTorch tensor
    new_edges = torch.tensor(inter_edges_list + intra_edges_list, dtype=torch.long).T

    # Merge with existing edges
    updated_edge_index = torch.cat([graph_data.edge_index, new_edges], dim=1)

    return Data(edge_index=updated_edge_index, num_nodes=graph_data.num_nodes)



def main():
    parser = argparse.ArgumentParser(description='homo')
    # TRIANGULAR = 1
    # HEXAGONAL = 2
    # SQUARE_GRID  = 3
    # KAGOME_LATTICE = 4
    parser.add_argument('--data_name', type=str, default='RegularTilling.TRIANGULAR')
    parser.add_argument('--N', type=str, help='number of the node in synthetic graph')
    parser.add_argument('--pr', type=float, default=0.3, help='percentage of perturbation of edges')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='ChebGCN')
    parser.add_argument('--score_model', type=str, 
                                            default='mlp_score', 
                                            choices=['mlp_score', 'dot_product'])
    ## gnn setting
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_layers_predictor', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--gnnout_hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)

    ### train setting
    parser.add_argument('--batch_size', type=int, default=2**12)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--kill_cnt', dest='kill_cnt', 
                                        default=20,    
                                        type=int,       
                                        help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2',		type=float,             
                                    default=0.0,			
                                    help='L2 Regularization for Optimizer')
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
    parser.add_argument('--test_batch_size', type=int, default=1024) 
    parser.add_argument('--use_hard_negative', default=False, action='store_true')

    args = parser.parse_args()
    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print('use_val_edge:', args.use_valedges_as_input)
    print('cat_n2v_feat: ', args.cat_n2v_feat)
    print('use_hard_negative: ',args.use_hard_negative)
    print(args)

    if args.data_name =='RegularTilling.TRIANGULAR':
        eval_metric = 'MRR'
        N = 4000
    if args.data_name =='RegularTilling.HEXAGONAL':
        eval_metric = 'MRR'
        N = 4000
    if args.data_name =='RegularTilling.SQUARE_GRID':
        eval_metric = 'MRR'
        N = 100
    if args.data_name =='RegularTilling.KAGOME_LATTICE':
        eval_metric = 'MRR'
        N = 100
    if args.N is not None:
        N = int(args.N)
        
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@5': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
        'AUC':Logger(args.runs),
        'AP':Logger(args.runs),
        'mrr_hit20':  Logger(args.runs),
        'mrr_hit50':  Logger(args.runs),
        'mrr_hit100':  Logger(args.runs),
    }
    
    
    init_seed(args.seed)
    dataset_name = "Cora"
    csv_path = f'{dataset_name}_Node_Merging.csv'
    file_exists = os.path.isfile(csv_path)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
    original_data = dataset[0]  # Extract the first graph in the dataset
    G = to_networkx(original_data, to_undirected=True)
    
    graph_stats = get_graph_statistics(G, graph_name=args.data_name, perturbation=args.pr)
    save_graph_statistics(graph_stats)
    
    sym_data = create_disjoint_graph(original_data)
    num_nodes = sym_data.num_nodes
    node_groups, node_labels = run_wl_test_and_group_nodes(sym_data.edge_index, num_nodes=num_nodes, num_iterations=30)
    metrics_before, _, _ = compute_automorphism_metrics(node_groups, num_nodes)
    metrics_before.update({'head': f'{dataset_name}_original'})
    print(f"Automorphism fraction before adding random edges: {metrics_before}")
    df = pd.DataFrame([metrics_before])
    df.to_csv(csv_path, mode='a', index=False, header=not file_exists)
    
    hyperparams = {
        'inter_ratio': [0.5],
        'intra_ratio': [0.1, 0.2, 0.3],
        'total_edges': [10**3],
    }

    for inter_ratio, intra_ratio, total_edgeds in itertools.product(hyperparams['inter_ratio'], hyperparams['intra_ratio'], hyperparams['total_edges']):
        inter_ratio = inter_ratio
        intra_ratio = intra_ratio
        total_edgeds = total_edgeds
        graph_data = add_random_edges(sym_data, inter_ratio=inter_ratio, intra_ratio=intra_ratio, total_edges=total_edgeds)

        # Compute Automorphism After Adding Random Edges
        node_groups, node_labels = run_wl_test_and_group_nodes(graph_data.edge_index, num_nodes=num_nodes, num_iterations=30)
        metrics_after, _, _ = compute_automorphism_metrics(node_groups, num_nodes)
        
        print(f"Automorphism fraction after adding random edges: {metrics_after}")
        metrics_after.update({'head': f'{dataset_name}_inter{inter_ratio}_intra{intra_ratio}_edges{total_edgeds}'})
        df = pd.DataFrame([metrics_after])
        df.to_csv(csv_path, mode='a', index=False, header=not file_exists)
        data = graph_data
        import pdb; pdb.set_trace()
        G = to_networkx(data, to_undirected=True)
        
        graph_stats = get_graph_statistics(G, graph_name=args.data_name, perturbation=args.pr)
        save_graph_statistics(graph_stats)

        for key, value in graph_stats.items():
            print(f"{key}: {value}")
    
        data, split_edge, G, pos = nx2Data_split(G, pos, True, 0.1, 0.1)

        for k, val in split_edge.items():
            print(k, 'pos_edge_index', val['pos_edge_label_index'].size())
        emb = None

        if hasattr(data, 'x'):
            if data.x != None:
                x = data.x
                data.x = data.x.to(torch.float)
                if args.cat_n2v_feat:
                    print('cat n2v embedding!!')
                    n2v_emb = torch.load(os.path.join(get_root_dir(), 'dataset', args.data_name+'-n2v-embedding.pt'))
                    data.x = torch.cat((data.x, n2v_emb), dim=-1)
                data.x = data.x.to(device)
                input_channel = data.x.size(1)
            else:
                emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
                input_channel = args.hidden_channels
        else:
            emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
            input_channel = args.hidden_channels
            
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
        
        print(data, args.data_name)
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
            print(data.full_adj_t)
            print(data.adj_t)
        data = data.to(device)

        if args.gnn_model == 'MixHopGCN':
            args.num_layers = 1
            
        model = eval(args.gnn_model)(input_channel, args.hidden_channels,
                        args.hidden_channels, args.num_layers, args.dropout, 
                        mlp_layer=args.gin_mlp_layer, head=args.gat_head, 
                        node_num=data.num_nodes, cat_node_feat_mf=args.cat_node_feat_mf,  
                        data_name=args.data_name).to(device)
        
        if args.gnn_model == 'MixHopGCN':
            args.hidden_channels = 3 * args.hidden_channels
        score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                        1, args.num_layers_predictor, args.dropout).to(device)

        print(data)
        print(split_edge)

        pos_train_edge = split_edge['train']['pos_edge_label_index']
        pos_valid_edge = split_edge['valid']['pos_edge_label_index']
        neg_valid_edge = split_edge['valid']['neg_edge_label_index']
        pos_test_edge = split_edge['test']['pos_edge_label_index']
        neg_test_edge = split_edge['test']['neg_edge_label_index']

        idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)]
        train_val_edge = pos_train_edge[idx]
        pos_train_edge = pos_train_edge.to(device)
        evaluation_edges = [train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge]
        best_valid_auc = best_test_auc = 2
        best_auc_valid_str = 2

        # hyperparams = {
        #     'batch_size': [512],
        #     'lr': [0.001]
        #     }
        args.batch_size = 512
        args.lr = 0.001

        print('#################################                    #################################')
        import wandb
        wandb.init(project=f"GRAND4_{args.data_name}_{args.gnn_model}_latest", name=f"{args.data_name}_{args.gnn_model}_bs{args.batch_size}_lr{args.lr}_perturb{args.pr}")
        wandb.config.update(args, allow_val_change=True)
        
        for run in range(args.runs):
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

                loss = train(model, score_func, split_edge, pos_train_edge, data, emb, optimizer, args.batch_size, train_edge_weight, args.data_name, args.remove_edge_aggre)
                wandb.log({'train_loss': loss}, step = epoch)
                step += 1
                    
                if epoch % args.eval_steps == 0:
                    results_rank, score_emb= test(model, score_func, data, evaluation_edges, emb, evaluator_hit, evaluator_mrr, args.batch_size, args.use_valedges_as_input)
                    for key, result in results_rank.items():
                        loggers[key].add_result(run, result)
                        wandb.log({f"Metrics/{key}": result[-1]}, step=epoch)
                        
                    if epoch % args.log_steps == 0:
                        for key, result in results_rank.items():
                            print(key)
                            
                            train_hits, valid_hits, test_hits = result
                            log_print.info(
                                f'Run: {run + 1:02d}, '
                                f'Epoch: {epoch:02d}, '
                                f'Loss: {loss:.4f}, '
                                f'Train: {100 * train_hits:.2f}%, '
                                f'Valid: {100 * valid_hits:.2f}%, '
                                f'Test: {100 * test_hits:.2f}%')

                    r = torch.tensor(loggers[eval_metric].results[run])
                    best_valid_current = round(r[:, 1].max().item(), 4)
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
        wandb.finish()
                                    
        for key in loggers.keys():
            if len(loggers[key].results[0]) > 0:
                print(key)
                best_valid, best_valid_mean, mean_list, var_list, test_res = loggers[key].print_statistics()
                print(best_valid)
                print(best_valid_mean)
                print(mean_list)
                print(var_list)
                print(test_res)
        save_new_results(loggers, f"{args.data_name}_{args.pr}", N, file_name=f'{args.data_name}_{args.gnn_model}_test_results_0.25_0.5.csv')


# Example usage function
def rewiring():
    perturb_ratio = [0, 0.1, 0.5, 0.7]
    N = 20
    node_size =150
    font_size = 100
    g_type = RegularTilling.TRIANGULAR
    G, _, _, pos = init_regular_tilling(N, g_type, seed=None)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    nx.draw(G, pos, node_size=node_size, font_size=font_size, node_color="gray", edge_color="gray")
    # plt.title(f"Original Triangular {G.number_of_edges()}")
    
    for pr in perturb_ratio:
        G_rewired, rewired_list = local_edge_rewiring(G, num_rewirings=int(pr * G.number_of_edges()), seed=None) # num_rewirting = int(pr * G.number_of_edges())

        node_colors = ["gray"] * len(G_rewired.nodes())
        highlight_nodes = rewired_list

        # Set selected nodes to green
        for i, node in enumerate(G_rewired.nodes()):
            if node in highlight_nodes:
                node_colors[i] = "green"

        # Draw the rewired graph
        
        plt.subplot(1, 2, 2)
        nx.draw(G_rewired, pos, node_size=node_size, font_size=font_size, node_color=node_colors, edge_color="gray")
        # plt.title(f"Rewired Triangular {G_rewired.number_of_edges()}")
        plt.savefig(f'rewired_{pr}.png')
        data_rewired, split_rewired, G_rewired, pos = nx2Data_split(G_rewired, pos, True, 0.25, 0.5)
    
    return data_rewired, split_rewired, G_rewired, pos



if __name__ == "__main__":
#     for i in range(100):
#         data = generate_graph(10, GraphType.TREE, seed=i)
#    rewiring()
   main()