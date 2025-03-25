import networkx as nx
import numpy as np
import random
import os
from automorphism import run_wl_test_and_group_nodes, compute_automorphism_metrics
import pandas as pd 
import matplotlib.pyplot as plt
from automorphism import dataloader
import argparse
from torch_geometric.utils import to_networkx, from_networkx


def random_walk_rewire(G, p=0.1, steps=5):
    """
    Perturbs a fully connected graph by rewiring edges based on a random walk process.
    
    Parameters:
    G (networkx.Graph): Fully connected input graph.
    p (float): Probability of rewiring an edge at each step.
    steps (int): Number of perturbation steps.
    
    Returns:
    networkx.Graph: Perturbed graph with reduced automorphism.
    """
    nodes = list(G.nodes())
    
    for _ in range(steps):
        for u in nodes:
            if random.random() < p:
                # Perform a random walk
                v = random.choice(list(G.neighbors(u)))
                w = random.choice(nodes)
                
                # Ensure w is not already a neighbor and is not self-loop
                if w != u and not G.has_edge(u, w):
                    G.remove_edge(u, v)
                    G.add_edge(u, w)
    
    return G

parser = argparse.ArgumentParser(description='homo')
# TRIANGULAR = 1
# HEXAGONAL = 2
# SQUARE_GRID  = 3
# KAGOME_LATTICE = 4
parser.add_argument('--data_name', type=str, default='ogbl-ppa')
args = parser.parse_args()  
    
# Example usage
n = 1000  
args.data_name = 'Cora'
data, num_nodes, edge_index = dataloader(args)


G = to_networkx(data)
num_nodes = G.number_of_nodes()
edge_index = from_networkx(G).edge_index
node_groups, node_labels = run_wl_test_and_group_nodes(edge_index, num_nodes=num_nodes, num_iterations=100)
metrics, num_nodes, group_sizes = compute_automorphism_metrics(node_groups, num_nodes)
metrics.update({'data_name': f'{args.data_name}_0'})
print(metrics)
pd.DataFrame([metrics]).to_csv(f'GN_Perturbation.csv', index=False)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
nx.draw(G, with_labels=True, edge_color='gray')
plt.title("Original Fully Connected Graph")

for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
    perturbed_G = random_walk_rewire(G, p=p, steps=40)
    plt.subplot(1, 2, 2)
    nx.draw(perturbed_G, with_labels=True, edge_color='gray')
    plt.title(f"Perturbed Graph with p {p}")
    plt.savefig('random_walk_rewire.png')

    edge_index = from_networkx(perturbed_G).edge_index
    node_groups, node_labels = run_wl_test_and_group_nodes(edge_index, num_nodes=num_nodes, num_iterations=100)
    metrics, num_nodes, group_sizes = compute_automorphism_metrics(node_groups, num_nodes)
    metrics.update({'data_name': f'{args.data_name}_Perturbation_{p}'})
    print(metrics)
    csv_path = f'{args.data_name}_Perturbation.csv'
    file_exists = os.path.isfile(csv_path)
    pd.DataFrame([metrics]).to_csv(csv_path, mode='a', index=False, header=not file_exists)

