import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

from baselines.gnn_utils import get_root_dir, get_logger, get_config_dir, Logger, init_seed, save_emb
from baselines.gnn_utils import GCN, GAT, SAGE, GIN, MF, DGCNN, GCN_seal, SAGE_seal, DecoupleSEAL, mlp_score, dot_product

# from logger import Logger
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.utils import to_networkx, to_undirected
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from baselines.gnn_utils  import evaluate_hits, evaluate_auc, evaluate_mrr
import torch
from torch_geometric.data import Data
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
from syn_random import (init_regular_tilling, 
                        RegularTilling, 
                        local_edge_rewiring,
                        nx2Data_split)
import csv
from torch_geometric.utils import to_networkx, from_networkx
from graph_generation import generate_graph, GraphType
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import WLConv
from typing import Optional
from torch import Tensor
from torch_geometric.typing import Adj
import torch
from scipy.stats import qmc
from torch_geometric.utils import (
    degree,
    is_sparse,
    scatter,
    sort_edge_index,
    to_edge_index,
)
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class WLConv(torch.nn.Module):
    r"""The Weisfeiler Lehman (WL) operator from the `"A Reduction of a Graph
    to a Canonical Form and an Algebra Arising During this Reduction"
    <https://www.iti.zcu.cz/wl2018/pdf/wl_paper_translation.pdf>`_ paper.

    :class:`WLConv` iteratively refines node colorings according to:

    .. math::
        \mathbf{x}^{\prime}_i = \textrm{hash} \left( \mathbf{x}_i, \{
        \mathbf{x}_j \colon j \in \mathcal{N}(i) \} \right)

    Shapes:
        - **input:**
          node coloring :math:`(|\mathcal{V}|, F_{in})` *(one-hot encodings)*
          or :math:`(|\mathcal{V}|)` *(integer-based)*,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node coloring :math:`(|\mathcal{V}|)` *(integer-based)*
    """
    def __init__(self):
        super().__init__()
        self.hashmap = {}

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.hashmap = {}

    @torch.no_grad()
    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        r"""Runs the forward pass of the module."""
        if x.dim() > 1:
            assert (x.sum(dim=-1) == 1).sum() == x.size(0)
            x = x.argmax(dim=-1)  # one-hot -> integer.
        assert x.dtype == torch.long

        if is_sparse(edge_index):
            col_and_row, _ = to_edge_index(edge_index)
            col = col_and_row[0]
            row = col_and_row[1]
        else:
            edge_index = sort_edge_index(edge_index, num_nodes=x.size(0),
                                         sort_by_row=False)
            row, col = edge_index[0], edge_index[1]

        # `col` is sorted, so we can use it to `split` neighbors to groups:
        deg = degree(col, x.size(0), dtype=torch.long).tolist()

        out = []
        for node, neighbors in zip(x.tolist(), x[row].split(deg)):
            idx = hash(tuple([node] + neighbors.sort()[0].tolist()))
            if idx not in self.hashmap:
                self.hashmap[idx] = len(self.hashmap)
            out.append(self.hashmap[idx])

        return torch.tensor(out, device=x.device)

    def histogram(self, x: Tensor, batch: Optional[Tensor] = None,
                  norm: bool = False) -> Tensor:
        r"""Given a node coloring :obj:`x`, computes the color histograms of
        the respective graphs (separated by :obj:`batch`).
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_colors = len(self.hashmap)
        batch_size = int(batch.max()) + 1

        index = batch * num_colors + x
        out = scatter(torch.ones_like(index), index, dim=0,
                      dim_size=num_colors * batch_size, reduce='sum')
        out = out.view(batch_size, num_colors)

        if norm:
            out = out.to(torch.float)
            out /= out.norm(dim=-1, keepdim=True)

        return out


class WLConvMultiFeature(torch.nn.Module):
    def __init__(self):
        """Weisfeiler-Lehman convolution supporting multi-dimensional node features."""
        super().__init__()
        self.hashmap = {}

    def reset_parameters(self):
        """Resets hash storage."""
        self.hashmap = {}
    @torch.no_grad()
    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """
        Runs the forward pass of the Weisfeiler-Lehman update step.

        Args:
            x (Tensor): Node feature matrix of shape (num_nodes, num_features).
            edge_index (Tensor): Edge index tensor of shape (2, num_edges).

        Returns:
            Tensor: Updated node hash labels of shape (num_nodes,).
        """
        if x.dim() > 1:
            # Convert multi-dimensional features into hashable form
            x = [tuple(row.tolist()) for row in x]  # Convert each feature row to a tuple
        
        if is_sparse(edge_index):
            col_and_row, _ = to_edge_index(edge_index)
            col, row = col_and_row[0], col_and_row[1]
        else:
            edge_index = sort_edge_index(edge_index, num_nodes=len(x), sort_by_row=False)
            row, col = edge_index[0], edge_index[1]
        # Compute node degree
        deg = degree(col, len(x), dtype=torch.long).tolist()
        out = []
        for node, neighbors in zip(x, [x[row] for row in row.split(deg)]):
            # Hash the node's feature and its sorted neighbor features
            idx = hash((node, tuple(sorted(neighbors))))
            if idx not in self.hashmap:
                self.hashmap[idx] = len(self.hashmap)
            out.append(self.hashmap[idx])
        return torch.tensor(out, device=edge_index.device, dtype=torch.long)



class WLConvMultiFeature(torch.nn.Module):
    def __init__(self):
        """Weisfeiler-Lehman convolution supporting multi-dimensional node features."""
        super().__init__()
        self.hashmap = {}

    def reset_parameters(self):
        """Resets hash storage."""
        self.hashmap = {}

    @torch.no_grad()
    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """
        Runs the forward pass of the Weisfeiler-Lehman update step.

        Args:
            x (Tensor): Node feature matrix of shape (num_nodes, num_features).
            edge_index (Tensor): Edge index tensor of shape (2, num_edges).

        Returns:
            Tensor: Updated node hash labels of shape (num_nodes,).
        """
        if x.ndim > 1:
            # Convert multi-dimensional features into hashable form (tuple per node)
            x = [tuple(row.tolist()) for row in x]
        if is_sparse(edge_index):
            col_and_row, _ = to_edge_index(edge_index)
            col, row = col_and_row[0], col_and_row[1]
        else:
            edge_index = sort_edge_index(edge_index, num_nodes=len(x), sort_by_row=False)
            row, col = edge_index[0], edge_index[1]
        # Compute node degree
        deg = degree(col, len(x), dtype=torch.long).tolist()
        # Corrected neighbor feature extraction
        neighbors_per_node = [[] for _ in range(len(x))]
        for src, dst in zip(row.tolist(), col.tolist()):
            neighbors_per_node[dst].append(x[src])  # Collect features of neighbors
        out = []
        for node, neighbors in zip(x, neighbors_per_node): # O(N^avg_deg)
            # Hash the node's feature and its sorted neighbor features
            idx = hash((node, tuple(sorted(neighbors))))
            if idx not in self.hashmap:
                self.hashmap[idx] = len(self.hashmap)
            out.append(self.hashmap[idx])
        return torch.tensor(out, device=edge_index.device, dtype=torch.long)



class WLConvOptimized(torch.nn.Module):
    def __init__(self):
        """Weisfeiler-Lehman convolution optimized for multi-dimensional node features."""
        super().__init__()
        self.hashmap = {}

    def reset_parameters(self):
        """Resets hash storage."""
        self.hashmap = {}

    @torch.no_grad()
    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """
        Runs the forward pass of the Weisfeiler-Lehman update step (optimized).

        Args:
            x (Tensor): Node feature matrix of shape (num_nodes, num_features).
            edge_index (Tensor): Edge index tensor of shape (2, num_edges).

        Returns:
            Tensor: Updated node hash labels of shape (num_nodes,).
        """
        num_nodes = x.shape[0]

        if x.ndim > 1:
            # Convert multi-dimensional features into a hashable form (faster version)
            x = x.tolist()  # Avoid repeated tolist() calls
        else:
            x = x.tolist()

        # Convert edge_index to row, col format
        if is_sparse(edge_index):
            col_and_row, _ = to_edge_index(edge_index)
            col, row = col_and_row[0], col_and_row[1]
        else:
            edge_index = sort_edge_index(edge_index, num_nodes=num_nodes, sort_by_row=False)
            row, col = edge_index[0], edge_index[1]

        # Compute degree and initialize neighbor storage efficiently
        neighbors_per_node = [[] for _ in range(num_nodes)]
        
        # Use NumPy array for faster indexing
        x_array = np.array(x, dtype=object)  # Keeps features as objects for hashing
        
        # Efficient neighbor collection
        for src, dst in zip(row.cpu().numpy(), col.cpu().numpy()):
            neighbors_per_node[dst].append(x_array[src])
        # Faster hashing using NumPy broadcasting and tuple encoding
        out = np.empty(num_nodes, dtype=int)
        for i in range(num_nodes):
            # Hash the node's feature and its sorted neighbor features (avoiding Python loops)
            idx = hash((tuple([x_array[i]]), tuple(sorted(neighbors_per_node[i]))))

            if idx not in self.hashmap:
                self.hashmap[idx] = len(self.hashmap)

            out[i] = self.hashmap[idx]

        # Convert back to a tensor
        return torch.tensor(out, device=edge_index.device, dtype=torch.long)

def quasi_random_features(num_nodes, feature_dim=3, method='halton'):
    """
    Generates quasi-random node features using low-discrepancy sequences.
    
    Args:
        num_nodes (int): Number of nodes in the graph.
        feature_dim (int): Feature dimensionality.
        method (str): 'halton' (default) or 'sobol'.
    
    Returns:
        Tensor: (num_nodes, feature_dim) quasi-random features.
    """
    if method == 'halton':
        sampler = qmc.Halton(d=feature_dim, scramble=True)
    elif method == 'sobol':
        sampler = qmc.Sobol(d=feature_dim, scramble=True)
    else:
        raise ValueError("Method must be 'halton' or 'sobol'.")

    features = sampler.random(n=num_nodes)  # Generate quasi-random values
    return torch.tensor(features, dtype=torch.float)


def run_wl_test_and_group_nodes(edge_index, num_nodes, num_iterations=1000):
    """
    Runs the Weisfeiler-Lehman (WL) test and groups nodes with similar hashed labels.
    
    Args:
        edge_index (Tensor): The edge index tensor (2, |E|) representing the graph.
        num_nodes (int): The number of nodes in the graph.
        num_iterations (int): Number of WL iterations.
    
    Returns:
        node_groups (dict): Mapping from WL hashes to node sets.
        node_labels (Tensor): Final hashed labels for each node.
    """
    # wl = WLConvMultiFeature()  # Initialize the WL hashing layer
    wl = WLConvOptimized()  # Optimized version for multi-dimensional features
    # node_labels = torch.arange(num_nodes, dtype=torch.long)  # Initialize labels as unique integers
    # X = quasi_random_features(num_nodes, feature_dim=1)
    # X = X - X.mean(dim=0)  # Center features
    # node_labels = X / X.norm(dim=1, keepdim=True)  # Normalize features
    node_labels = np.ones(num_nodes)
    for _ in range(num_iterations):
        node_labels = wl(node_labels, edge_index)  
    # Group nodes based on final hashed values
    node_groups = {}
    for node, label in enumerate(node_labels.tolist()):
        if label not in node_groups:
            node_groups[label] = []
        node_groups[label].append(node)

    return node_groups, node_labels




def compute_inner_product_matrix(X):
    """
    Computes the pairwise inner product matrix for node features.

    Args:
        X (Tensor): Node feature matrix of shape (num_nodes, feature_dim).

    Returns:
        Tensor: Inner product matrix of shape (num_nodes, num_nodes).
    """
    X = X - X.mean(dim=0)  # Center features
    X = X / X.norm(dim=1, keepdim=True)  # Normalize features
    return torch.matmul(X, X.T)  # Compute pairwise inner products

def visualize_inner_product_matrix(H):
    """
    Visualizes the inner product matrix as a heatmap.

    Args:
        H (Tensor): Pairwise inner product matrix of shape (num_nodes, num_nodes).
    """
    plt.figure(figsize=(6,6))
    plt.imshow(H.numpy(), cmap='Blues', interpolation='nearest')
    plt.colorbar(label="Inner Product")
    plt.title("Pairwise Inner Product Matrix")
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")
    plt.savefig('inner_product_matrix.png')


def rewiring():
    perturb_ratio = [0, 0.1, 0.5, 0.7]
    N = 100
    node_size =150
    font_size = 100
    g_type = RegularTilling.TRIANGULAR
    G, _, _, pos = init_regular_tilling(N, g_type, seed=None)
    A = nx.to_numpy_array(G)  # Convert to adjacency matrix
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    nx.draw(G, pos, node_size=node_size, font_size=font_size, node_color="gray", edge_color="gray")
    # plt.title(f"Original Triangular {G.number_of_edges()}")
    # Define your automorphism function on synthetic graph 
    H_struct = compute_structural_similarity(A)

    # Visualize H_struct
    plt.figure(figsize=(6,6))
    plt.imshow(H_struct, cmap='Blue', interpolation='nearest')
    plt.colorbar(label="Structural Similarity")
    plt.title("H_struct for a Triangular Lattice Graph")
    plt.savefig('H_struct.png')
        
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



def compute_structural_similarity(A, k=6):
    """ Compute structural similarity matrix using Laplacian eigenvectors. """
    n = len(A)
    D = np.diag(A.sum(axis=1))  # Degree matrix
    L = D - A  # Unnormalized Laplacian
    _, eigvecs = eigsh(L, k=k, which='SM')  # Smallest k eigenvectors

    H_struct = np.dot(eigvecs, eigvecs.T)  # Similarity based on projection
    return H_struct

# Generate a 5x5 triangular lattice

def compute_automorphism_metrics(node_groups, num_nodes):
    """
    Computes numerical metrics for graph automorphism based on WL node grouping.

    Args:
        node_groups (dict): Dictionary mapping WL hash values to lists of node indices.
        num_nodes (int): Total number of nodes in the graph.

    Returns:
        dict: Automorphism metrics {A_r, C_auto, H_auto}
    """
    # Compute the size of each group (how many nodes share the same WL label)
    group_sizes = np.array([len(group) for group in node_groups.values()])
    # Automorphism Ratio (A_r)
    A_r = np.sum(group_sizes**2) / num_nodes**2
    # Number of Unique Groups (C_auto)
    C_auto = len(node_groups)
    # Automorphism Entropy (H_auto)
    p_i = group_sizes / num_nodes  # Probability of each group
    H_auto = -np.sum(p_i * np.log(p_i + 1e-9))  # Small epsilon to avoid log(0)

    return {
        "Automorphism Ratio (A_r)": A_r,
        "Number of Unique Groups (C_auto)": C_auto,
        "Automorphism Entropy (H_auto)": H_auto
    }

def dataloader(args):
    if args.data_name in ['ogbl-ddi', 'ogbl-collab', 'ogbl-ppa', 'ogbl-citation2']:
        dataset = PygLinkPropPredDataset(name=args.data_name, 
                                         root=os.path.abspath(os.path.join(get_root_dir(), f"dataset")))
        print(f"Dataset: {args.data_name}")
        print(f"Number of graphs: {len(dataset)}")

        data = dataset[0]
        print("data", data)
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        G = None 
        
    # elif args.data_name in ['RegularTilling.SQUARE_GRID', 
    #                       'RegularTilling.HEXAGONAL', 
    #                       'RegularTilling.TRIANGULAR', 
    #                       'RegularTilling.KAGOME_LATTICE']:
    #     N = 100
    #     G, _, _, pos = init_regular_tilling(N, eval(args.data_name), seed=None)
    #     data = from_networkx(G)
    #     num_nodes = G.number_of_nodes()
    #     edge_index = data.edge_index
    #     print(f"Dataset: {args.data_name}")
    #     print("data", data)
        
    # elif args.data_name in ['GraphType.TREE', 
    #                         'GraphType.RANDOM']:    
        
    #     G = generate_graph(10, eval(args.data_name), seed=0)
    #     data = from_networkx(G)
    #     num_nodes = G.number_of_nodes()
    #     print(f"Dataset: {args.data_name}")
    #     print("data", data)
        
    return G, num_nodes, edge_index


def process_graph(N, graph_type, pos=None, is_grid=False, label="graph"):
    if is_grid:
        G, _, _, pos = init_regular_tilling(N, RegularTilling.SQUARE_GRID, seed=None)
    else:
        G = generate_graph(N, graph_type, seed=0)
    
    # Draw the graph
    plt.figure(figsize=(12, 6))
    nx.draw(G, pos if is_grid else None, node_size=150, font_size=100, node_color="black", edge_color="gray")
    plt.savefig(f'{label}.png')
    
    # Process Graph with WL Test
    data = from_networkx(G)
    edge_index = data.edge_index
    node_groups, node_labels = run_wl_test_and_group_nodes(edge_index, num_nodes=G.number_of_nodes(), num_iterations=100)
    metrics = compute_automorphism_metrics(node_groups, G.number_of_nodes())
    metrics.update({'data_name': label})
    print(metrics)
    pd.DataFrame([metrics]).to_csv(f'{label}.csv', index=False)

    # Visualize with WL-based coloring
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos if is_grid else None, node_size=50, font_size=8, cmap='Set1', node_color=node_labels, edge_color="gray")
    plt.title("Graph Visualization with WL-based Node Coloring")
    plt.savefig(f'wl_test_{label}.png')



    
def test_automorphism():
    parser = argparse.ArgumentParser(description='homo')
    # TRIANGULAR = 1
    # HEXAGONAL = 2
    # SQUARE_GRID  = 3
    # KAGOME_LATTICE = 4
    parser.add_argument('--data_name', type=str, default='ogbl-citation2', choices=['ogbl-ddi', 'ogbl-ppa', 'ogbl-citation2', 'ogbl-collab'])
    args = parser.parse_args()    
    G, num_nodes, edge_index = dataloader(args)
    
    node_groups, node_labels = run_wl_test_and_group_nodes(edge_index, num_nodes=num_nodes, num_iterations=100)
    metrics = compute_automorphism_metrics(node_groups, num_nodes)
    metrics.update({'data_name': args.data_name})
    print(metrics)
    pd.DataFrame([metrics]).to_csv(f'{args.data_name}_alpha.csv', index=False)
    df = pd.DataFrame(node_labels.numpy(), columns=['node_labels'])
    df.to_csv(f'{args.data_name}_node_labels.csv', index=False)
    del node_labels, node_groups, metrics
    import IPython; IPython.embed()
    
    # Two Extreme Cases:
    process_graph(100, None, is_grid=True, label="G_low")  # Regular tiling case
    process_graph(10, GraphType.TREE, label="G_high")      # Tree case



if __name__ == "__main__":
    # DRAFT THE DATASET FROM THE SYNTHETIC GRAPH where their automophism should be 1 and for tree it should be very low
    test_automorphism()