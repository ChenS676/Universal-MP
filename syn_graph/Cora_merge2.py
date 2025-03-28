import os
import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx, to_networkx, sort_edge_index
from automorphism import run_wl_test_and_group_nodes, compute_automorphism_metrics, dataloader
import argparse
import pandas as pd

def validate_edge_index(edge_index, num_nodes):
    """Ensure edge_index is a valid tensor and contains valid node indices."""
    if edge_index is None or edge_index.numel() == 0:
        raise ValueError("edge_index is empty or None, cannot process graph.")
    
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    if edge_index.max() >= num_nodes or edge_index.min() < 0:
        raise ValueError(f"edge_index contains out-of-bound indices (0-{num_nodes-1}).")
    
    return edge_index

def merge_nodes_for_symmetry(edge_index, num_nodes, merge_ratio=0.2):
    """Enforces node automorphism by making groups of nodes structurally identical."""
    edge_index = validate_edge_index(edge_index, num_nodes)
    
    node_groups, _ = run_wl_test_and_group_nodes(edge_index, num_nodes=num_nodes, num_iterations=100)
    G = nx.Graph()
    G.add_edges_from(edge_index.numpy().T)

    sorted_groups = sorted(node_groups.values(), key=len, reverse=True)
    num_groups_to_merge = int(len(sorted_groups) * merge_ratio)
    selected_groups = sorted_groups[:num_groups_to_merge]

    for group in selected_groups:
        if len(group) < 2:
            continue

        merged_neighbors = set()
        for node in group:
            merged_neighbors.update(G.neighbors(node))
        
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if G.has_edge(group[i], group[j]):
                    G.remove_edge(group[i], group[j])

        for node in group:
            for neighbor in merged_neighbors:
                if neighbor not in group:
                    G.add_edge(node, neighbor)

    new_edge_index = from_networkx(G).edge_index
    return validate_edge_index(new_edge_index, num_nodes)

def duplicate_edges(edge_index, num_nodes, duplication_ratio=0.1):
    """Duplicates edges between nodes with similar neighborhood structures."""
    edge_index = validate_edge_index(edge_index, num_nodes)
    
    num_edges = edge_index.shape[1]
    num_duplications = int(num_edges * duplication_ratio)
    
    duplicate_indices = np.random.choice(num_edges, num_duplications, replace=False)
    duplicated_edges = edge_index[:, duplicate_indices]
    
    new_edge_index = torch.cat([edge_index, duplicated_edges], dim=1)
    return validate_edge_index(new_edge_index, num_nodes)

# CLI Arguments
parser = argparse.ArgumentParser(description='homo')
parser.add_argument('--data_name', type=str, default='ogbl-ppa')
args = parser.parse_args()

# Load Data
args.data_name = 'Cora'
data, num_nodes, edge_index = dataloader(args)
edge_index = validate_edge_index(edge_index, num_nodes)

csv_path = f'{args.data_name}_Node_Merging.csv'
file_exists = os.path.isfile(csv_path)

# Compute Automorphism Before Modification
node_groups, node_labels = run_wl_test_and_group_nodes(edge_index, num_nodes=num_nodes, num_iterations=100)
metrics_before, _, _ = compute_automorphism_metrics(node_groups, num_nodes)
print("Before Merging:", metrics_before)
metrics_before.update({'head': f'{args.data_name}_0'})
df = pd.DataFrame([metrics_before])
df.to_csv(csv_path, mode='a', index=False, header=not file_exists)

# Apply Node Merging
for merge_ratio in np.arange(0, 1.1, 0.1):
    new_edge_index = merge_nodes_for_symmetry(edge_index, num_nodes, merge_ratio=merge_ratio)
    
    node_groups, node_labels = run_wl_test_and_group_nodes(new_edge_index, num_nodes=num_nodes, num_iterations=100)
    metrics_after, _, _ = compute_automorphism_metrics(node_groups, num_nodes)
    print("After Merging:", metrics_after)

    metrics_after.update({'head': f'{args.data_name}_{merge_ratio}'})
    df = pd.DataFrame([metrics_after])
    df.to_csv(csv_path, mode='a', index=False, header=not file_exists)
