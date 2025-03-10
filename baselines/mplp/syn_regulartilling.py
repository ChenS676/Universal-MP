"""
Utils for generating random graph. Adopted from https://raw.githubusercontent.com/JiaruiFeng/KP-GNN/main/datasets/graph_generation.py
"""
import os
import sys
import math
import random
from enum import Enum

import networkx as nx
import os.path as osp
import numpy as np
from typing import *
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import (to_undirected, 
                                coalesce, 
                                remove_self_loops,
                                from_networkx)
import scipy.sparse as sp
from syn_random import init_nodefeats, random_edge_split
from data_utils import plot_graph

"""
    Generates random graphs of different types of a given size.
    Some of the graph are created using the NetworkX library, for more info see
    https://networkx.github.io/documentation/networkx-1.10/reference/generators.html
"""



class RegularTilling(Enum):
    TRIANGULAR = 1
    HEXAGONAL = 2
    SQUARE_GRID  = 3
    KAGOME_LATTICE = 4
     
    
    
def triangular(N):
    """ Creates a m x k 2d grid triangular graph with N = m*k and m and k as close as possible """
    m = 1
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    G = nx.triangular_lattice_graph(m, N // m)
    pos =  nx.get_node_attributes(G, 'pos')
    return G, pos



def hexagonal(N):
    """ Creates a m x k 2d grid hexagonal graph with N = m*k and m and k as close as possible """
    m = 1
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    G = nx.hexagonal_lattice_graph(m, N // m)
    pos = nx.get_node_attributes(G, 'pos')
    return G, pos



def square_grid(M, N, seed) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    """
    Output:
    -------
    Tuple[nx.Graph, Dict[int, Tuple[float, float]]]
        A tuple containing the square grid graph and the node positions.

    Description:
    -----------
        This function generates a square grid graph with m rows and n columns.
        It assigns 'random edge weights' from Uniform distribution 
        and optional node labels based on heterophily or homophily settings.

    """
    np.random.seed(seed)
    num_nodes: int = M * N
    adj_matrix: np.ndarray = np.zeros((num_nodes, num_nodes), dtype=int)
    
    def node_id(x: int, y: int) -> int:
        return x * N + y
    
    for x in range(M):
        for y in range(N):
            current_id = node_id(x, y)
            
            # Right neighbor
            if y < N - 1:
                right_id = node_id(x, y + 1)
                adj_matrix[current_id, right_id] = 1
                adj_matrix[right_id, current_id] = 1
                
            # Down neighbor
            if x < M - 1:
                down_id = node_id(x + 1, y)
                adj_matrix[current_id, down_id] = 1
                adj_matrix[down_id, current_id] = 1

    pos: Dict[int, Tuple[float, float]] = {(x * N + y): (y, x) for x in range(M) for y in range(N)}

    G = nx.from_numpy_array(adj_matrix)
    
    for u, v in G.edges():
        G[u][v]['weight'] = random.uniform(0.1, 1.0)
    
    for node, position in pos.items():
        G.nodes[node]['pos'] = position
    
    return G, pos


def kagome_lattice(m, n, seed):
    """ Create a Kagome lattice and return its NetworkX graph and positions. """
    np.random.seed(seed)
    G = nx.Graph()
    pos = {}
    
    def node_id(x, y, offset):
        return 2 * (x * n + y) + offset
    
    for x in range(m):
        for y in range(n):
            # Two nodes per cell (offset 0 and 1)
            current_id0 = node_id(x, y, 0)
            current_id1 = node_id(x, y, 1)
            pos[current_id0] = (y, x)
            pos[current_id1] = (y + 0.5, x + 0.5)
            
            # Add nodes
            G.add_node(current_id0)
            G.add_node(current_id1)
            
            # Right and down connections
            if y < n - 1:
                right_id0 = node_id(x, y + 1, 0)
                right_id1 = node_id(x, y + 1, 1)
                G.add_edge(current_id0, right_id0)
                G.add_edge(right_id1, right_id0)
                G.add_edge(right_id0, current_id0)
                G.add_edge(right_id0, right_id1)
                
            if x < m - 1:
                down_id0 = node_id(x + 1, y, 0)
                down_id1 = node_id(x + 1, y, 1)
                G.add_edge(current_id0, down_id0)
                G.add_edge(current_id1, down_id1)
                G.add_edge(down_id0, current_id0)
                G.add_edge(down_id1, current_id1)
            
            # Diagonal connections
            if x < m - 1 and y < n - 1:
                diag_id0 = node_id(x + 1, y + 1, 0)
                diag_id1 = node_id(x + 1, y + 1, 1)
                G.add_edge(current_id1, diag_id0)
                G.add_edge(diag_id0, current_id1)
                G.add_edge(current_id1, diag_id1)
                G.add_edge(diag_id1, current_id1)
    
    return G, pos



def init_pyg_regtil(N: int, 
                    g_type: RegularTilling,
                    seed: int,
                    undirected = True,
                    val_pct = 0.05,
                    test_pct = 0.15,
                    split_labels = True, 
                    include_negatives = True) -> Tuple[Data, Data, Data]:
    
    G, _, _ = init_regular_tilling(N, g_type, seed)
    data = from_networkx(G)

    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    
    if undirected:
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        undirected = True
        
    data.x = init_nodefeats(G, 'random', int(np.log(N)) + 16)

    data = T.ToSparseTensor()(data)
    
    row, col, _ = data.adj_t.coo()
    data.edge_index = torch.stack([col, row], dim=0)
    data.edge_weight = data.adj_t.to_torch_sparse_csc_tensor().values()
    data.adj_t = sp.csr_matrix((data.edge_weight.cpu(), (data.edge_index[0].cpu(), data.edge_index[1].cpu())), 
                shape=(data.num_nodes, data.num_nodes))

    splits = random_edge_split(data,
                undirected,
                'cpu',
                val_pct, # val_pct = 0.15
                test_pct, # test_pct =  0.5,
                split_labels, # split_labels = True,
                include_negatives)  # include_negatives = False

    return data, splits


def init_regular_tilling(N, type=RegularTilling.SQUARE_GRID, seed=None):
    if type == RegularTilling.TRIANGULAR:
        G, pos = triangular(N)
    elif type == RegularTilling.HEXAGONAL:
        G, pos = hexagonal(N)
    elif type == RegularTilling.SQUARE_GRID:
        G, pos = square_grid(N, N, seed)
    elif type == RegularTilling.KAGOME_LATTICE:
        G, pos = kagome_lattice(N, N, seed)

    # generate adjacency matrix and nodes values
    nodes = list(G)
    random.shuffle(nodes)
    adj_matrix = nx.to_numpy_array(G, nodes)
    
    # draw the graph created
    plt.figure()
    try:
        plot = plot_graph(G, pos, title = f'{type.name} Graph')
    except:
        plot = plot_graph(G)

    plot.savefig('draw.png')

    return G, adj_matrix, type



if __name__ == '__main__':
    N = 6000
    for i, g_type in enumerate([
                             RegularTilling.TRIANGULAR, 
                             RegularTilling.HEXAGONAL, 
                             RegularTilling.SQUARE_GRID, 
                             RegularTilling.KAGOME_LATTICE, 
                             ]):
        
        data, splits = init_pyg_regtil(N, 
                g_type,
                i,
                undirected = True,
                val_pct = 0.15,
                test_pct = 0.05,
                split_labels = True, 
                include_negatives = True)
        import pdb; pdb.set_trace() 
        print(data)
        print(splits)