import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import sys
import random
import itertools
import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import (
    to_networkx
)
from ogb.linkproppred import PygLinkPropPredDataset
from automorphism import run_wl_test_and_group_nodes, compute_automorphism_metrics
from baselines.data_utils.lcc import get_largest_connected_component, use_lcc
from syn_real.gnn_utils import (
    get_root_dir, 
    get_logger, 
    get_config_dir, 
    Logger, 
    init_seed
)
import matplotlib.pyplot as plt
import networkx as nx

from gnn_ogb_heart import init_seed
import torch
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


dir_path = get_root_dir()
log_print = get_logger('testrun', 'log', get_config_dir())
DATASET_PATH = '/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/baselines/dataset'


    
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
    elif dataset_name.startswith('ogbl'):
        data = extract_induced_subgraph()
        print(f"before data {data}")
        # dataset = PygLinkPropPredDataset(name=dataset_name, root='/pfs/work7/workspace/scratch/cc7738-kdd25/Universal-MP/syn_graph/dataset/')
        data, lcc_index, G = use_lcc(data)
        print(f"after lcc {data}")
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
    return Data(edge_index=updated_edge_index, num_nodes=graph_data.num_nodes)


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
    print(f"Saved to {save_path}")
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
    # print(f"Automorphism fraction before adding random edges: {metrics_before}")



def run_experiment(graph_data, args, inter_ratio, intra_ratio, total_edges):
    """
    Run the experiment with the given parameters.
    
    Parameters:
        graph_data (torch_geometric.data.Data): The input graph data.
        args (argparse.Namespace): Arguments containing dataset name.
        inter_ratio (float): Fraction of edges to add between the two graph copies.
        intra_ratio (float): Fraction of edges to add within each graph copy.
        total_edges (int): Total number of random edges to add.
    """
    # Add random edges to the graph
    if inter_ratio != 0 and intra_ratio != 0 and total_edges != 0:
        updated_graph_data = add_random_edges(graph_data, inter_ratio=inter_ratio, intra_ratio=intra_ratio, total_edges=total_edges)
    else:
        updated_graph_data = graph_data
    # Convert to NetworkX graph for visualization
    G = to_networkx(updated_graph_data, to_undirected=True)
    num_nodes = updated_graph_data.num_nodes
    node_groups, node_labels = run_wl_test_and_group_nodes(updated_graph_data.edge_index, num_nodes=num_nodes, num_iterations=30)
    
    metrics_after, num_nodes, group_sizes = compute_automorphism_metrics(node_groups, num_nodes)
    metrics_after.update({'head': f'{args.data_name}_inter{inter_ratio}_intra{intra_ratio}_edges{total_edges}'})
    csv_path = f'plots/{args.data_name}/_Node_Merging.csv'
    file_exists = os.path.isfile(csv_path)
    df = pd.DataFrame([metrics_after])
    df.to_csv(csv_path, mode='a', index=False, header=not file_exists)
    print(df)
    
    plot_group_size_distribution(group_sizes, args, f'plots/{args.data_name}/group_size_log1p{args.data_name}_inter{inter_ratio}_intra{intra_ratio}_edges{total_edges}.png')
    plot_histogram_group_size_log_scale(group_sizes, metrics_after, args, f'plots/{args.data_name}/hist_group_size_log_{args.data_name}_inter{inter_ratio}_intra{intra_ratio}_edges{total_edges}.png')
    plot_graph_visualization(updated_graph_data, node_labels, args,  f'plots/{args.data_name}/wl_test_{args.data_name}_vis_inter{inter_ratio}_intra{intra_ratio}_edges{total_edges}.png')
    print(f"Finished with inter_ratio={inter_ratio}, intra_ratio={intra_ratio}, total_edges={total_edges}")
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default="ogbl-ddi")
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--score_model', type=str, default='mlp_score')

    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_layers_predictor', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--gnnout_hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)

    ### train setting
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=2)
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

    parser.add_argument('--inter_ratio', type=float, required=False, help='Inter ratio', default=0.5)
    parser.add_argument('--intra_ratio', type=float, required=False, help='Intra ratio', default=0.5)
    parser.add_argument('--total_edges', type=int, required=False, help='Total edges', default=1000)
    args = parser.parse_args()
    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print('use_val_edge:', args.use_valedges_as_input)
    print('use_hard_negative: ',args.use_hard_negative)
    print(args)
    return args
    
    
def main():
    args = parse_args()
    init_seed(args.seed)

    if os.path.exists(f'plots/{args.data_name}') == False:
        os.makedirs(f'plots/{args.data_name}')
        
    csv_path = f'plots/{args.data_name}/_Node_Merging.csv'
    file_exists = os.path.isfile(csv_path)
    original_data = load_real_world_graph(args.data_name)

    run_experiment(original_data, args, 0, 0, 0)
    disjoint_graph = create_disjoint_graph(original_data)
    
    run_experiment(disjoint_graph, args, 0, 0, 0)
    inter_ratios = [0.5]  # Try also: 0.1–0.9
    intra_ratios = [0.5]    
    total_edges_list =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # [0.15, 1,  4.5, 8, 14, 40]  # Will be scaled × 10^3

    for inter in inter_ratios:
        for intra in intra_ratios:
            for edge_factor in total_edges_list:
                total_edges = int(edge_factor * 1)
                print(f"\n=== Running experiment: inter_ratio={inter}, intra_ratio={intra}, edge_factor={edge_factor} ===")
                run_experiment(disjoint_graph, args, inter, intra, total_edges)


def extract_induced_subgraph(dataset_name='ogbl-ddi', 
                            num_sampled_nodes=200, 
                            output_path='ddi_induced_subgraph.pt'):
    # Load the OGB dataset
    print("Loading dataset...")
    dataset = PygLinkPropPredDataset(name=dataset_name)
    data = dataset[0]

    # Randomly sample node indices
    num_nodes = data.num_nodes
    sampled_nodes = torch.randperm(num_nodes)[:num_sampled_nodes]

    # Extract the induced subgraph
    print(f"Extracting induced subgraph from {num_sampled_nodes} random nodes...")
    edge_index, _ = subgraph(sampled_nodes, data.edge_index, relabel_nodes=True)

    # Create the subgraph Data object
    subgraph_data = Data(
        edge_index=edge_index,
        num_nodes=sampled_nodes.size(0)
    )

    # Save the subgraph
    torch.save(subgraph_data, output_path)
    print(f"Saved induced subgraph with {subgraph_data.num_nodes} nodes and {subgraph_data.num_edges} edges to {output_path}")
    return subgraph_data

    
def extract_subgraph(dataset_name='ogbl-ddi', 
                    num_sampled_nodes=0, 
                    k_hop=2, 
                    output_path='ddi_subgraph.pt'):
    # Load the OGB dataset
    print("Loading dataset...")
    dataset = PygLinkPropPredDataset(name=dataset_name)
    data = dataset[0]

    # Randomly sample seed nodes
    num_nodes = data.num_nodes
    seed_nodes = torch.randperm(num_nodes)[:num_sampled_nodes]

    # Extract the k-hop subgraph
    print(f"Extracting {k_hop}-hop subgraph around {num_sampled_nodes} seed nodes...")
    subset, edge_index, _, _ = k_hop_subgraph(
        node_idx=seed_nodes,
        num_hops=k_hop,
        edge_index=data.edge_index,
        relabel_nodes=True
    )
    subgraph_data = Data(
        edge_index=edge_index,
        num_nodes=subset.size(0)
    )
    torch.save(subgraph_data, output_path)
    print(f"Saved subgraph with {subgraph_data.num_nodes} nodes and {subgraph_data.num_edges} edges to {output_path}")


if __name__ == "__main__":
    main()
    exit(-1)
    extract_induced_subgraph()

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
