import os
import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx, to_networkx
from automorphism import run_wl_test_and_group_nodes, compute_automorphism_metrics, higher_order_wl
import argparse
from automorphism import dataloader
import pandas as pd

import networkx as nx
import random
from collections import Counter

import argparse
import os
import numpy as np
import pandas as pd

def duplicate_edges(G, num_duplicates=5):
    """
    Duplicate edges based on frequently occurring neighborhood structures.
    
    Parameters:
    G (nx.Graph): Input graph
    num_duplicates (int): Number of edges to duplicate
    
    Returns:
    G_new (nx.Graph): Graph with duplicated edges
    """
    G_new = G.copy()
    
    # Step 1: Identify frequently occurring neighborhood structures
    neighborhoods = {node: frozenset(G.neighbors(node)) for node in G.nodes()}
    neighborhood_counts = Counter(neighborhoods.values())
    
    # Step 2: Find nodes with similar neighborhoods
    similar_nodes = {}
    for node, nh in neighborhoods.items():
        similar_nodes.setdefault(nh, []).append(node)
    
    # Step 3: Duplicate edges to enforce similarity
    for nh, nodes in similar_nodes.items():
        if len(nodes) > 1:
            for _ in range(num_duplicates):
                u, v = random.sample(nodes, 2)
                if not G_new.has_edge(u, v):
                    G_new.add_edge(u, v)
    
    return G_new


# def merge_nodes_for_symmetry(edge_index, num_nodes, merge_ratio=0.2):
#     """
#     Merges nodes with similar WL labels by connecting them, enforcing structural equivalence.
    
#     Args:
#         edge_index (Tensor): The original edge index tensor.
#         num_nodes (int): The number of nodes in the graph.
#         merge_ratio (float): The fraction of node groups to merge (default 20%).

#     Returns:
#         new_edge_index (Tensor): Modified edge index with increased symmetry.
#     """
#     # Run WL test to group nodes by automorphic equivalence
#     node_groups, _ = run_wl_test_and_group_nodes(edge_index, num_nodes=num_nodes, num_iterations=100)

#     # Convert edge index to a networkx graph for easy manipulation
#     G = nx.Graph()
#     G.add_edges_from(edge_index.numpy().T)

#     # Sort groups by size (largest groups first)
#     sorted_groups = sorted(node_groups.values(), key=len, reverse=True)
    
#     # Select a subset of groups for merging
#     num_groups_to_merge = int(len(sorted_groups) * merge_ratio)
#     selected_groups = sorted_groups[:num_groups_to_merge]

#     # Enforce symmetry by adding edges between nodes in the same group
#     for group in selected_groups:
#         for i in range(len(group)):
#             for j in range(i + 1, len(group)):
#                 G.add_edge(group[i], group[j])  # Merge structurally similar nodes

#     # Convert back to edge index format
#     new_edge_index = from_networkx(G).edge_index

#     return new_edge_index

def merge_nodes_for_symmetry(edge_index, num_nodes, merge_ratio=0.2):
    """
    Enforces node automorphism by making groups of nodes structurally identical.

    Args:
        edge_index (Tensor): Edge index representation of the graph.
        num_nodes (int): Number of nodes in the graph.
        merge_ratio (float): Fraction of automorphic groups to merge.

    Returns:
        new_edge_index (Tensor): Modified edge index with increased automorphism.
    """
    # Step 1: Identify automorphic groups
    # node_groups, _ = run_wl_test_and_group_nodes(edge_index, num_nodes=num_nodes, num_iterations=100)
    node_groups, _ = run_wl_test_and_group_nodes(edge_index, num_nodes=num_nodes, num_iterations=100)
    
    # Convert edge index to NetworkX for easier modifications
    G = nx.Graph()
    G.add_edges_from(edge_index.numpy().T)

    # Step 2: Sort groups by size (prefer larger automorphic groups)
    sorted_groups = sorted(node_groups.values(), key=len, reverse=True)
    num_groups_to_merge = int(len(sorted_groups) * merge_ratio)
    selected_groups = sorted_groups[:num_groups_to_merge]

    # Step 3: Rewire nodes to enforce structural equivalence
    for group in selected_groups:
        if len(group) < 2:
            continue  # Skip small groups

        # Collect all unique neighbors
        merged_neighbors = set()
        for node in group:
            merged_neighbors.update(G.neighbors(node))

        # Remove existing edges within the group
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if G.has_edge(group[i], group[j]):
                    G.remove_edge(group[i], group[j])

        # Ensure all nodes in the group have the exact same neighbors
        for node in group:
            for neighbor in merged_neighbors:
                if neighbor not in group:  # Prevent self-loops
                    G.add_edge(node, neighbor)

    # Convert back to edge index format
    new_edge_index = from_networkx(G).edge_index
    return new_edge_index


def duplicate_edges(edge_index, num_nodes, duplication_ratio=0.1):
    """
    Duplicates edges between nodes with similar neighborhood structures.
    """
    num_edges = edge_index.shape[1]
    num_duplications = int(num_edges * duplication_ratio)
    
    # Select random edges to duplicate
    duplicate_indices = np.random.choice(num_edges, num_duplications, replace=False)
    duplicated_edges = edge_index[:, duplicate_indices]
    
    # Append duplicated edges to the original edge_index
    new_edge_index = np.hstack([edge_index, duplicated_edges])
    return new_edge_index

# TRIANGULAR = 1
# HEXAGONAL = 2
# SQUARE_GRID  = 3
# KAGOME_LATTICE = 4
parser = argparse.ArgumentParser(description='homo')
parser.add_argument('--data_name', type=str, default='ogbl-ppa')
args = parser.parse_args()  

# Example Usage
args.data_name = 'Cora'
data, num_nodes, edge_index = dataloader(args)
csv_path = f'{args.data_name}_Node_Merging.csv'
file_exists = os.path.isfile(csv_path)
# Compute automorphism before modification
node_groups, node_labels = run_wl_test_and_group_nodes(edge_index, num_nodes=num_nodes, num_iterations=100)
metrics_before, _, _ = compute_automorphism_metrics(node_groups, num_nodes)
print("Before Merging:", metrics_before)
metrics_before.update({'head': f'{args.data_name}_0'})
df = pd.DataFrame([metrics_before])
df.to_csv(csv_path, mode='a', index=False, header=not file_exists)

# Apply node merging
for merge_ratio in np.arange(0, 1.1, 0.1):
    new_edge_index = merge_nodes_for_symmetry(edge_index, num_nodes, merge_ratio=merge_ratio)

    # Compute automorphism after modification
    node_groups, node_labels = run_wl_test_and_group_nodes(new_edge_index, num_nodes=num_nodes, num_iterations=100)
    metrics_after, _, _ = compute_automorphism_metrics(node_groups, num_nodes)
    print("After Merging:", metrics_after)

    # Save results
    metrics_after.update({'head': f'{args.data_name}_{merge_ratio}'})
    df = pd.DataFrame([metrics_after])
    pd.DataFrame([metrics_after]).to_csv(csv_path, mode='a', index=False, header=not file_exists)

