import os
import torch
import networkx as nx
import numpy as np
import pandas as pd
from torch_geometric.utils import from_networkx, to_networkx, to_dense_adj, dense_to_sparse
from torch_geometric.nn import MessagePassing
from automorphism import run_wl_test_and_group_nodes, compute_automorphism_metrics, dataloader
import argparse
from torch_cluster import random_walk

def random_walk_diffusion(edge_index, num_nodes, walk_length=5, num_walks=10, alpha=0.85):
    """
    Performs random walk diffusion to enhance node similarity and increase automorphism.

    Args:
        edge_index (Tensor): Edge index representation of the graph.
        num_nodes (int): Number of nodes in the graph.
        walk_length (int): Number of steps in each random walk.
        num_walks (int): Number of walks per node.
        alpha (float): Restart probability in Personalized PageRank (PPR).

    Returns:
        new_edge_index (Tensor): Modified edge index with increased automorphism.
    """
    device = edge_index.device if torch.is_tensor(edge_index) else "cpu"
    edge_index = edge_index.to(device)
    
    # Perform random walks
    walks = random_walk(edge_index[0], edge_index[1], start=torch.arange(num_nodes).repeat(num_walks).to(device), walk_length=walk_length)
    
    # Compute node visit frequencies
    transition_matrix = torch.zeros((num_nodes, num_nodes), device=device)
    for walk in walks:
        for i in range(len(walk) - 1):
            transition_matrix[walk[i], walk[i+1]] += 1
    
    # Normalize transition matrix
    transition_matrix = transition_matrix / transition_matrix.sum(dim=1, keepdim=True)
    
    # Apply diffusion process (e.g., Personalized PageRank)
    ppr_matrix = (1 - alpha) * torch.linalg.inv(torch.eye(num_nodes, device=device) - alpha * transition_matrix)
    
    # Create new edges by linking nodes with high similarity in PPR space
    new_edges = []
    threshold = ppr_matrix.mean() + 2 * ppr_matrix.std()  # Adaptive threshold for new links
    print(f"Threshold: {threshold}, mean: {ppr_matrix.mean()}, std: {ppr_matrix.std()}")
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and ppr_matrix[i, j] > threshold:
                new_edges.append([i, j])
    
    # Convert new edges to edge index format
    new_edge_index = torch.tensor(new_edges, dtype=torch.long).T.to(device)
    new_edge_index = torch.cat([edge_index, new_edge_index], dim=1)  # Merge with original edges

    return new_edge_index


# TRIANGULAR = 1
# HEXAGONAL = 2
# SQUARE_GRID  = 3
# KAGOME_LATTICE = 4
parser = argparse.ArgumentParser(description='homo')
parser.add_argument('--data_name', type=str, default='ogbl-ppa')
args = parser.parse_args()

# Load Dataset
args.data_name = 'Cora'
data, num_nodes, edge_index = dataloader(args)
csv_path = f'{args.data_name}_RandomWalk_Diffusion.csv'
file_exists = os.path.isfile(csv_path)

# Compute automorphism before modification
node_groups, node_labels = run_wl_test_and_group_nodes(edge_index, num_nodes=num_nodes, num_iterations=100)
metrics_before, _, _ = compute_automorphism_metrics(node_groups, num_nodes)
print("Before Diffusion:", metrics_before)
metrics_before.update({'head': f'{args.data_name}_0'})
df = pd.DataFrame([metrics_before])
df.to_csv(csv_path, mode='a', index=False, header=not file_exists)

# Apply random walk diffusion-based graph augmentation
for alpha in np.arange(0.1, 1.0, 0.1):  # Sweeping over different restart probabilities
    new_edge_index = random_walk_diffusion(edge_index, num_nodes, walk_length=5, num_walks=10, alpha=alpha)

    # Compute automorphism after modification
    node_groups, node_labels = run_wl_test_and_group_nodes(new_edge_index, num_nodes=num_nodes, num_iterations=100)
    metrics_after, _, _ = compute_automorphism_metrics(node_groups, num_nodes)
    print(f"After Diffusion (alpha={alpha}):", metrics_after)

    # Save results
    metrics_after.update({'head': f'{args.data_name}_alpha{alpha}'})
    df = pd.DataFrame([metrics_after])
    pd.DataFrame([metrics_after]).to_csv(csv_path, mode='a', index=False, header=not file_exists)
