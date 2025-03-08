import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
from matplotlib.figure import Figure
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.transforms import RandomLinkSplit
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected, coalesce, remove_self_loops
import torch_geometric.transforms as T
import networkx as nx 
from typing import Dict, List, Optional, Tuple, Any, Set


FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def calc_cn(ei: torch.Tensor, adj: 'scipy.sparse.csr_matrix', bs: int = 10000) -> torch.FloatTensor:
    """
    Computes the number of common neighbors for node pairs in `ei` using adjacency matrix `adj`.

    Input:
    - ei (torch.Tensor): Shape (2, E), where E is the number of edges.
    - adj ('scipy.sparse.csr_matrix'): Graph adjacency matrix in CSR format.
    - bs (int): Batch size for processing node pairs. Default is 10000.

    Output:
    - torch.FloatTensor: Tensor containing the number of common neighbors for each pair of nodes in `ei`.
    """
    ll = torch.utils.data.DataLoader(range(ei.size(1)), bs)
    sc = []

    for idx in ll:
        s, d = ei[0, idx].to('cpu'), ei[1, idx].to('cpu')
        cur_sc = np.array(adj[s].multiply(adj[d]).sum(axis=1)).flatten()
        sc.append(cur_sc)

    sc = np.concatenate(sc, axis=0)
    return torch.FloatTensor(sc)



def plot_cn_dist(tr_cn: torch.FloatTensor, te_cn: torch.FloatTensor) -> None:
    """
    Generates a density plot of the number of common neighbors for training and testing datasets.

    Input:
    - tr_cn (torch.FloatTensor): Tensor with common neighbors for the training dataset.
    - te_cn (torch.FloatTensor): Tensor with common neighbors for the testing dataset.

    Output:
    - None: Saves the plot as a PNG file.

    Description:
    This function uses Kernel Density Estimation (KDE) via Seaborn to visualize the distribution of common neighbors
    for both training and testing datasets.
    """
    # Convert tensors to NumPy arrays
    tr_cn = tr_cn.to('cpu').numpy()
    te_cn = te_cn.to('cpu').numpy()

    # Plot the distributions
    plt.figure(figsize=(10, 6))
    sns.kdeplot(tr_cn, label='Train', color='red', linewidth=2)
    sns.kdeplot(te_cn, label='Test', color='blue', linewidth=2)
    plt.xlabel('Common Neighbors')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Common Neighbors Distribution')
    plt.savefig(f'{FILE_PATH}/cn_dist.png')



def plot_color_graph(G: nx.Graph, pos: Optional[Dict[int, Tuple[float, float]]] = None, 
                        title: str = "Graph", node_size: int = 300, with_labels: bool = True) -> Figure:
    """
    Input:
    ----------
    G : nx.Graph
        The NetworkX graph to be plotted.
    pos : Optional[Dict[int, Tuple[float, float]]], default=None
        A dictionary of node positions, where keys are node indices and values are (x, y) coordinates.
    title : str, default="Graph"
        The title of the graph plot.
    node_size : int, default=300
        The size of the nodes in the plot.
    with_labels : bool, default=True
        Whether or not to display labels on the nodes.

    Output:
    -------
    Figure
        The matplotlib figure object representing the final graph plot.

    Description:
    -----------
        This function plots a colored graph where each node's color represents its label.
    """
    node_labels: Dict[int, Any] = nx.get_node_attributes(G, 'label')
    
    unique_labels: Set[Any] = set(node_labels.values())
    
    colors: List[str] = ['red', 'black']  # Add more colors if you have more labels
    
    color_map: Dict[Any, str] = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    
    node_color: List[str] = [color_map[node_labels[node]] for node in G.nodes]
    
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=with_labels, node_size=node_size, node_color=node_color, 
            edge_color='gray', font_size=10)
    plt.title(title)
    return plt





def plot_degree_histogram(G: nx.Graph) -> Figure:
    """
    Input:
    ----------
    G : nx.Graph
        The NetworkX graph whose degree distribution will be plotted.

    Output:
    -------
    Figure
        The matplotlib figure object representing the histogram plot.

    Description:
    -----------
        This function plots a histogram of the node degree frequency distribution for a graph.

    """
    
    degrees: List[int] = [val for (_, val) in G.degree()]
    
    # Get the unique degrees and their frequencies
    degree_counts: np.ndarray = np.bincount(degrees)
    degrees_unique: np.ndarray = np.arange(len(degree_counts))
    
    # Normalize the frequencies for colormap
    max_count: int = max(degree_counts)
    normalized_counts: np.ndarray = np.array(degree_counts) / max_count if max_count > 0 else degree_counts
    colors = cm.Blues(normalized_counts)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(degrees_unique, degree_counts, color=colors, align='center')
    
    ax.set_xlabel('Degree')
    ax.set_ylabel('Frequency')
    ax.set_title('Degree Distribution Histogram')
    ax.set_xlim(0, 10)
    
    return fig




def sort_edge_index(edge_index):
    """Sort the edge index in ascending order according to the source node index."""

    src_index, sort_indices = torch.sort(edge_index[:, 0])
    dst_index = edge_index[sort_indices, 1]
    edge_reindex = torch.stack([src_index, dst_index])
    return edge_reindex, sort_indices



def randomsplit(data, val_ratio: float = 0.05, test_ratio: float = 0.15):
    data.edge_index, _ = coalesce(
        data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    if not hasattr(data, 'x'):
        data.num_nodes = data.x.shape[0]

    transform = T.Compose([
            T.NormalizeFeatures(),
            RandomLinkSplit(
                is_undirected=True,
                num_val=val_ratio,
                num_test=test_ratio,
                add_negative_train_samples=True,
                split_labels=True
            )
        ])
    train_data, val_data, test_data = transform(data)
    splits = {'train': {}, 'valid': {}, 'test': {}}
    splits['train'] = train_data
    splits['valid'] = val_data
    splits['test'] = test_data
    for _, split in splits.items():
        split['x'] = data.x
    del data, train_data, val_data, test_data
    return splits



def load_synrandom(N: int, g_type: str, seed: int):
    raise NotImplementedError
    return 



def load_regulartilling(N: int, g_type: str, seed: int):
    raise NotImplementedError
    return



def loaddataset(name: str, nfeat_path=None):
    # TODO double check 
    #      1: 
    #      2:
    #      3:
    if name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(root="dataset", name=name)
        data = dataset[0]
        splits = randomsplit(data)
        data.edge_index = to_undirected(splits["train"]["pos_edge_label_index"])
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]
    else:
        dataset = PygLinkPropPredDataset(name=f'ogbl-{name}')
        data = dataset[0]
        if name == "ppa":
            data.x = torch.argmax(data.x, dim=-1).float()
            data.max_x = torch.max(data.x).item()
        elif name == "ddi":
            data.x = torch.diag(torch.arange(data.num_nodes).float())
            data.max_x = data.num_nodes
        splits = randomsplit(data)
        edge_index = data.edge_index
    data.edge_weight = None
    data.adj_t = SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    data.max_x = -1

    if nfeat_path:
        # comment ? 
        data.x = torch.load(nfeat_path, map_location="cpu")
    data.full_adj_t = data.adj_t
    return data, splits



if __name__ == "__main__":
    loaddataset("ddi", False)