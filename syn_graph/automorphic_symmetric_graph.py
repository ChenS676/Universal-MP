import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx
import random
from automorphism import run_wl_test_and_group_nodes, compute_automorphism_metrics

# --- 1️⃣ Load Real-World Graph (Cora) ---
def load_real_world_graph(dataset_name="Cora"):
    """
    Load a real-world graph dataset (e.g., Cora) from PyTorch Geometric.
    
    Args:
        dataset_name (str): The dataset name (default: "Cora").
    
    Returns:
        Data: PyTorch Geometric Data object.
    """
    dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
    data = dataset[0]  # Extract the first graph in the dataset
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
    dataset_name = "Cora"  # Can be changed to "Citeseer", "PubMed", etc.

    # Load Real-World Graph
    original_data = load_real_world_graph(dataset_name)

    # Visualize Original Graph
    visualize_graph(original_data, title="Original Cora Graph")

    # Generate the Modified Graph
    graph_data = create_disjoint_graph(original_data)

    # Visualize Merged Graph Before Random Edge Addition
    visualize_graph(graph_data, title="Cora Graph with Two Disjoint Copies")

    # Compute Automorphism Before Adding Random Edges
    num_nodes = graph_data.num_nodes
    node_groups, node_labels = run_wl_test_and_group_nodes(graph_data.edge_index, num_nodes=num_nodes, num_iterations=30)
    metrics, _, _ = compute_automorphism_metrics(node_groups, num_nodes)
    
    print(f"Automorphism fraction before adding random edges: {metrics}")

    # Add Controlled Random Edges
    graph_data = add_random_edges(graph_data, inter_ratio=0.6, intra_ratio=0.4, total_edges=1000)

    # Visualize Graph After Adding Random Edges
    visualize_graph(graph_data, title="Cora Graph After Adding Controlled Random Edges")

    # Compute Automorphism After Adding Random Edges
    node_groups, node_labels = run_wl_test_and_group_nodes(graph_data.edge_index, num_nodes=num_nodes, num_iterations=30)
    metrics, _, _ = compute_automorphism_metrics(node_groups, num_nodes)
    print(f"Automorphism fraction after adding random edges: {metrics}")
