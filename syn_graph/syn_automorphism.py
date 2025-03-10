import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
import torch_geometric.transforms as T
from typing import Dict, Union
import matplotlib.pyplot as plt

def generate_symmetric_graph(N: int = 1000, 
                              p: float = 0.01, 
                              m: int = 2, 
                              graph_type: str = 'erdos_renyi', 
                              num_random_edges: int = 1000) -> Data:
    """
    Generate a synthetic nearly symmetric graph for link prediction.
    
    Args:
        N (int): Number of nodes in each base graph H.
        p (float): Probability of edge creation (for Erdős-Rényi graph).
        m (int): Number of edges to attach per new node (for Barabási-Albert graph).
        graph_type (str): Type of graph to generate ('erdos_renyi' or 'barabasi_albert').
        num_random_edges (int): Number of random edges to add.
    
    Returns:
        Data: A PyG Data object representing the generated graph.
    """
    if graph_type == 'erdos_renyi':
        H = nx.erdos_renyi_graph(N, p)
    elif graph_type == 'barabasi_albert':
        H = nx.barabasi_albert_graph(N, m)
    else:
        raise ValueError("Invalid graph_type. Choose 'erdos_renyi' or 'barabasi_albert'.")
    
    # Create two disjoint copies of H
    H1 = nx.relabel_nodes(H, lambda x: x)
    H2 = nx.relabel_nodes(H, lambda x: x + N)
    
    # Combine into one graph
    G = nx.union(H1, H2)
    
    # Add random edges
    all_possible_edges = list(nx.non_edges(G))
    random_edges = torch.randperm(len(all_possible_edges))[:num_random_edges]
    for idx in random_edges:
        u, v = all_possible_edges[idx]
        G.add_edge(u, v)
    
    # Convert to PyG format
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    data = Data(edge_index=edge_index, num_nodes=G.number_of_nodes())
    return data

def generate_symmetric_graph(N: int = 1000, 
                              p: float = 0.01, 
                              m: int = 2, 
                              graph_type: str = 'erdos_renyi', 
                              num_random_edges: int = 1000) -> Data:
    """
    Generate a synthetic nearly symmetric graph for link prediction.
    
    Args:
        N (int): Number of nodes in each base graph H.
        p (float): Probability of edge creation (for Erdős-Rényi graph).
        m (int): Number of edges to attach per new node (for Barabási-Albert graph).
        graph_type (str): Type of graph to generate ('erdos_renyi' or 'barabasi_albert').
        num_random_edges (int): Number of random edges to add.
    
    Returns:
        Data: A PyG Data object representing the generated graph.
    """
    if graph_type == 'erdos_renyi':
        H = nx.erdos_renyi_graph(N, p)
    elif graph_type == 'barabasi_albert':
        H = nx.barabasi_albert_graph(N, m)
    else:
        raise ValueError("Invalid graph_type. Choose 'erdos_renyi' or 'barabasi_albert'.")
    
    # Create two disjoint copies of H
    H1 = nx.relabel_nodes(H, lambda x: x)
    H2 = nx.relabel_nodes(H, lambda x: x + N)
    
    # Combine into one graph
    G = nx.union(H1, H2)
    
    # Add random edges
    all_possible_edges = list(nx.non_edges(G))
    random_edges = torch.randperm(len(all_possible_edges))[:num_random_edges]
    for idx in random_edges:
        u, v = all_possible_edges[idx]
        G.add_edge(u, v)
    
    # Convert to PyG format
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    data = Data(edge_index=edge_index, num_nodes=G.number_of_nodes())
    return data

def random_edge_split(data: Data,
                      undirected: bool,
                      device: Union[str, int],
                      val_pct: float,
                      test_pct: float,
                      split_labels: bool,
                      include_negatives: bool = False) -> Dict[str, Data]:
    """
    Split the edges into train, validation, and test sets for link prediction.
    """
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        RandomLinkSplit(is_undirected=undirected,
                        num_val=val_pct,
                        num_test=test_pct,
                        add_negative_train_samples=include_negatives,
                        split_labels=split_labels),
    ])
    train_data, val_data, test_data = transform(data)
    del train_data.neg_edge_label, train_data.neg_edge_label_index
    return {'train': train_data, 'valid': val_data, 'test': test_data}

def visualize_graph(graph: Data, filename: str = 'draw.png'):
    """Visualize and save the generated graph."""
    G = nx.Graph()
    edge_list = graph.edge_index.t().tolist()
    G.add_edges_from(edge_list)
    plt.figure(figsize=(10, 10))
    nx.draw(G, node_size=10, edge_color="gray", alpha=0.5)
    plt.savefig(filename)
    plt.close()

# Example usage
data = generate_symmetric_graph(N=1000, graph_type='erdos_renyi')
splits = random_edge_split(data, undirected=True, device='cpu', val_pct=0.25, test_pct=0.5, split_labels=True)
visualize_graph(data, 'draw.png')


# Implement a graph with perturbation ratio
