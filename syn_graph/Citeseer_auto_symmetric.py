import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.utils import from_networkx, to_networkx
import random
from automorphism import run_wl_test_and_group_nodes, compute_automorphism_metrics
import itertools 
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch 
    
# --- 1️⃣ Load Real-World Graph (Cora) ---
def load_real_world_graph(dataset_name="Cora"):
    """
    Load a real-world graph dataset (e.g., Cora) from PyTorch Geometric.
    Args:
        dataset_name (str): The dataset name (default: "Cora").
    Returns:
        Data: PyTorch Geometric Data object.
    """
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
    elif dataset_name.startswith('ogbl'):
        dataset = PygLinkPropPredDataset(name=dataset_name, root='/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/syn_graph/dataset/')
    data = dataset[0]  
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


# --- 4️⃣ Graph Visualization ---
def visualize_graph(data, title="Graph Visualization"):
    """
    Visualizes the graph using NetworkX and Matplotlib.

    Args:
        data (Data): PyTorch Geometric Data object.
        title (str): Title for the plot.
    """
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)  # Positioning algorithm
    nx.draw(G, pos, node_size=10, edge_color="gray", alpha=0.6, with_labels=False)
    plt.title(title)
    plt.savefig(title)


# --- 5️⃣ Compute Automorphism Fraction (Using WL-1) ---
def compute_wl_automorphism(edge_index, num_nodes, num_iterations=100):
    """
    Runs the 1-WL test to compute the fraction of automorphic nodes.

    Args:
        edge_index (Tensor): The edge index tensor representing the graph.
        num_nodes (int): Number of nodes in the graph.
        num_iterations (int): Number of WL iterations.

    Returns:
        float: The fraction of nodes that are automorphic.
    """
    from collections import Counter

    # Initialize node labels as unique IDs
    node_labels = torch.arange(num_nodes, dtype=torch.long)

    for _ in range(num_iterations):
        new_labels = {}
        for node in range(num_nodes):
            # Get neighbors
            neighbors = edge_index[1][edge_index[0] == node]
            neighbor_labels = [node_labels[n].item() for n in neighbors]
            new_labels[node] = hash(tuple(sorted([node_labels[node].item()] + neighbor_labels)))

        # Convert to tensor
        new_labels = torch.tensor([new_labels[n] for n in range(num_nodes)], dtype=torch.long)

        # Stop if labels don't change
        if torch.equal(new_labels, node_labels):
            break

        node_labels = new_labels

    # Count occurrences of each label
    label_counts = Counter(node_labels.tolist())

    # Compute automorphic fraction
    num_automorphic_nodes = sum(count for count in label_counts.values() if count > 1)
    automorphism_fraction = num_automorphic_nodes / num_nodes

    return automorphism_fraction


# --- 🚀 Main Execution ---
if __name__ == "__main__":
    dataset_name = "Citeseer"   #ogbl-ddi

    if os.path.exists(f'plots/{dataset_name}') == False:
        os.makedirs(f'plots/{dataset_name}')
        
    csv_path = f'plots/{dataset_name}/_Node_Merging.csv'
    file_exists = os.path.isfile(csv_path)
    original_data = load_real_world_graph(dataset_name)
    graph_data = create_disjoint_graph(original_data)
    num_nodes = graph_data.num_nodes
    node_groups, node_labels = run_wl_test_and_group_nodes(graph_data.edge_index, num_nodes=num_nodes, num_iterations=30)
    metrics_before, num_nodes, group_sizes = compute_automorphism_metrics(node_groups, num_nodes)
    metrics_before.update({'head': f'{dataset_name}_original'})
    df = pd.DataFrame([metrics_before])
    df.to_csv(csv_path, mode='a', index=False, header=not file_exists)

    plt.figure(figsize=(6, 6))
    G = to_networkx(graph_data, to_undirected=True)
    nx.draw(G, node_size=20, font_size=8, cmap='Set1', node_color=node_labels, edge_color="gray")
    plt.title("Graph Visualization with WL-based Node Coloring")
    plt.savefig(f'plots/{dataset_name}/wl_test_{dataset_name}_original.png')
    plt.figure()
    plt.plot(group_sizes)
    plt.savefig(f'plots/{dataset_name}/group_size_{dataset_name}.png')
    print(f"save to group_size_{dataset_name}_original.png")
        

    plt.figure(figsize=(6, 6))
    G = to_networkx(graph_data, to_undirected=True)
    nx.draw(G, node_size=20, font_size=8, cmap='Set1', node_color=node_labels, edge_color="gray")
    plt.title("Graph Visualization with WL-based Node Coloring")
    plt.savefig(f'plots/{dataset_name}/wl_test_{dataset_name}_original.png')


    # Plot histogram of group_sizes
    plt.figure(figsize=(6, 4))
    counts, bins, patches = plt.hist(group_sizes, bins=20, edgecolor='black', alpha=0.75, density=True)
    counts = counts * 100 * np.diff(bins)
    plt.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', alpha=0.75)

    plt.xlabel("Group Size")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Group Sizes {metrics_before['A_r_norm_1']}")
    plt.savefig(f'plots/{dataset_name}/hist_group_size_{dataset_name}.png')
    print(f"Saved to plots/group_size_{dataset_name}.png")
    print(f"Automorphism fraction before adding random edges: {metrics_before}")
    
    hyperparams = {
        'inter_ratio': [0.5],
        'intra_ratio': [0.5], #0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        'total_edges': [10**3, 10**4, 10**5],
    }

    for inter_ratio, intra_ratio, total_edgeds in itertools.product(hyperparams['inter_ratio'], hyperparams['intra_ratio'], hyperparams['total_edges']):
        inter_ratio = inter_ratio
        intra_ratio = intra_ratio
        total_edgeds = total_edgeds
        graph_data = add_random_edges(graph_data, inter_ratio=inter_ratio, intra_ratio=intra_ratio, total_edges=total_edgeds)

        # Compute Automorphism After Adding Random Edges
        node_groups, node_labels = run_wl_test_and_group_nodes(graph_data.edge_index, num_nodes=num_nodes, num_iterations=30)
        metrics_after, num_nodes, group_sizes  = compute_automorphism_metrics(node_groups, num_nodes)
        
        print(f"Automorphism fraction after adding random edges: {metrics_after}")
        metrics_after.update({'head': f'{dataset_name}_inter{inter_ratio}_intra{intra_ratio}_edges{total_edgeds}'})
        df = pd.DataFrame([metrics_after])
        df.to_csv(csv_path, mode='a', index=False, header=not file_exists)

        plt.figure()
        plt.plot(group_sizes)
        plt.savefig(f'plots/{dataset_name}/group_size_{dataset_name}_{inter_ratio}_{intra_ratio}_{total_edgeds}.png')

        # Visualiz  e with WL-based coloring
        plt.figure(figsize=(6, 6))
        G = to_networkx(graph_data, to_undirected=True)
        nx.draw(G, node_size=20, font_size=8, cmap='Set1', node_color=node_labels, edge_color="gray")
        plt.title("Graph Visualization with WL-based Node Coloring")
        plt.savefig(f'plots/{dataset_name}/wl_test_{dataset_name}_{inter_ratio}_{intra_ratio}_{total_edgeds}.png')
        plt.figure()
        plt.plot(group_sizes)
        plt.savefig(f'plots/{dataset_name}/group_size_{dataset_name}.png')

        # Plot histogram of group_sizes
        plt.figure(figsize=(6, 4))
        counts, bins, patches = plt.hist(group_sizes, bins=20, edgecolor='black', alpha=0.75, density=True)
        counts = counts * 100 * np.diff(bins)
        plt.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', alpha=0.75)

        plt.xlabel("Group Size")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Group Sizes {metrics_after['A_r_norm_1']}")
        plt.savefig(f'plots/{dataset_name}/hist_group_size_{dataset_name}_{inter_ratio}_{intra_ratio}_{total_edgeds}.png')