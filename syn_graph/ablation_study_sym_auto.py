import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx
import random
import pandas as pd


# --- 1️⃣ Load Real-World Graph (Cora) ---
def load_real_world_graph(dataset_name="Cora"):
    dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
    return dataset[0]  # Extract the first graph in the dataset


# --- 2️⃣ Create Disjoint Graph Copies ---
def create_disjoint_graph(data):
    num_nodes = data.num_nodes
    G = to_networkx(data, to_undirected=True)
    G2 = nx.relabel_nodes(G, lambda x: x + num_nodes)
    merged_graph = nx.compose(G, G2)
    merged_edge_index = torch.tensor(list(merged_graph.edges)).T
    return Data(edge_index=merged_edge_index, num_nodes=2 * num_nodes)


# --- 3️⃣ Add Controllable Random Edges ---
def add_random_edges(graph_data, inter_ratio=0.5, intra_ratio=0.5, total_edges=1000):
    num_nodes = graph_data.num_nodes // 2
    inter_edges = int(total_edges * inter_ratio)
    intra_edges = total_edges - inter_edges

    # Generate edges
    inter_edges_list = [
        (random.randint(0, num_nodes - 1), random.randint(num_nodes, 2 * num_nodes - 1))
        for _ in range(inter_edges)
    ]
    intra_edges_list = [
        (random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1))
        for _ in range(intra_edges // 2)
    ] + [
        (random.randint(num_nodes, 2 * num_nodes - 1), random.randint(num_nodes, 2 * num_nodes - 1))
        for _ in range(intra_edges // 2)
    ]

    new_edges = torch.tensor(inter_edges_list + intra_edges_list, dtype=torch.long).T
    updated_edge_index = torch.cat([graph_data.edge_index, new_edges], dim=1)
    return Data(edge_index=updated_edge_index, num_nodes=graph_data.num_nodes)


# --- 4️⃣ Compute Automorphism Fraction (WL-1 Test) ---
def compute_wl_automorphism(edge_index, num_nodes, num_iterations=100):
    from collections import Counter
    node_labels = torch.arange(num_nodes, dtype=torch.long)

    for _ in range(num_iterations):
        new_labels = {}
        for node in range(num_nodes):
            neighbors = edge_index[1][edge_index[0] == node]
            neighbor_labels = [node_labels[n].item() for n in neighbors]
            new_labels[node] = hash(tuple(sorted([node_labels[node].item()] + neighbor_labels)))

        new_labels = torch.tensor([new_labels[n] for n in range(num_nodes)], dtype=torch.long)
        if torch.equal(new_labels, node_labels):
            break
        node_labels = new_labels

    label_counts = Counter(node_labels.tolist())
    num_automorphic_nodes = sum(count for count in label_counts.values() if count > 1)
    return num_automorphic_nodes / num_nodes


# --- 5️⃣ Run Ablation Study ---
def ablation_study():
    dataset_name = "Cora"
    original_data = load_real_world_graph(dataset_name)
    graph_data = create_disjoint_graph(original_data)
    num_nodes = graph_data.num_nodes

    # Measure original automorphism fraction
    base_automorphism = compute_wl_automorphism(graph_data.edge_index, num_nodes)

    results = []

    # 🔹 Varying `inter_ratio`
    for inter_ratio in np.arange(0, 1.1, 0.2):
        modified_graph = add_random_edges(graph_data, inter_ratio=inter_ratio, intra_ratio=0.5, total_edges=1000)
        auto_frac = compute_wl_automorphism(modified_graph.edge_index, num_nodes)
        results.append(["inter_ratio", inter_ratio, auto_frac])

    # 🔹 Varying `intra_ratio`
    for intra_ratio in np.arange(0, 1.1, 0.2):
        modified_graph = add_random_edges(graph_data, inter_ratio=0.5, intra_ratio=intra_ratio, total_edges=1000)
        auto_frac = compute_wl_automorphism(modified_graph.edge_index, num_nodes)
        results.append(["intra_ratio", intra_ratio, auto_frac])

    # 🔹 Varying `total_edges`
    for total_edges in range(500, 3000, 500):
        modified_graph = add_random_edges(graph_data, inter_ratio=0.5, intra_ratio=0.5, total_edges=total_edges)
        auto_frac = compute_wl_automorphism(modified_graph.edge_index, num_nodes)
        results.append(["total_edges", total_edges, auto_frac])

    # Convert results to DataFrame
    df_results = pd.DataFrame(results, columns=["Factor", "Value", "Automorphism Fraction"])
    print(df_results)

    # 🔹 Plot Results
    plt.figure(figsize=(12, 5))

    for factor in ["inter_ratio", "intra_ratio", "total_edges"]:
        subset = df_results[df_results["Factor"] == factor]
        plt.plot(subset["Value"], subset["Automorphism Fraction"], label=factor)

    plt.axhline(y=base_automorphism, color='gray', linestyle="--", label="Baseline (Before Random Edges)")
    plt.xlabel("Parameter Value")
    plt.ylabel("Automorphism Fraction")
    plt.legend()
    plt.title("Ablation Study: Effect of Parameters on Automorphism Fraction")
    plt.savefig("Ablation Study_Effect of Parameters on Automorphism Fraction.pdf")


# 🚀 Run the Study
if __name__ == "__main__":
    ablation_study()
