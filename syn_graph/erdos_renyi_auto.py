import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import tqdm

# --- WL Automorphism Estimator ---
def wl_automorphic_node_fraction(G, num_iterations=3):
    """
    Run 1-WL refinement and compute the fraction of nodes that are not uniquely identifiable.
    """
    labels = {node: str(G.degree[node]) for node in G.nodes()}
    for _ in range(num_iterations):
        new_labels = {}
        for node in G.nodes():
            neighborhood = sorted([labels[neighbor] for neighbor in G.neighbors(node)])
            new_labels[node] = f"{labels[node]}_" + "_".join(neighborhood)
        label_counter = Counter(new_labels.values())
        label_map = {label: str(i) for i, label in enumerate(sorted(label_counter))}
        labels = {node: label_map[label] for node, label in new_labels.items()}
    
    label_counts = Counter(labels.values())
    num_automorphic = sum(count for count in label_counts.values() if count > 1)
    fraction = num_automorphic / G.number_of_nodes()
    return fraction

# --- Run Experiment ---
def run_automorphism_experiment(n=100, ps=np.linspace(0.05, 1.0, 20), num_seeds=5, output_csv="automorphism_vs_p.csv"):
    results = []

    for p in tqdm.tqdm(ps, desc="Edge Probabilities"):
        for seed in range(num_seeds):
            G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
            if not nx.is_connected(G):
                # Keep the largest connected component
                Gc = max(nx.connected_components(G), key=len)
                G = G.subgraph(Gc).copy()
                G = nx.convert_node_labels_to_integers(G)

            frac_auto = wl_automorphic_node_fraction(G)
            results.append({
                'n': n,
                'p': round(p, 3),
                'seed': seed,
                'automorphic_fraction': frac_auto
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✓ Results saved to {output_csv}")
    return df

# --- Plotting Function ---
def plot_automorphism_vs_p(df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='p', y='automorphic_fraction', ci='sd', marker='o')
    plt.xlabel("Erdős–Rényi Edge Probability (p)")
    plt.ylabel("Fraction of Automorphic Nodes (1-WL indistinguishable)")
    plt.title("Automorphism vs Edge Probability in ER Graphs")
    plt.tight_layout()
    plt.savefig("automorphism_vs_p.png", dpi=300)

    print("✓ Plot saved to automorphism_vs_p.png")

# --- Main ---
if __name__ == "__main__":
    df = run_automorphism_experiment(
        n=2708,
        ps=np.linspace(0.9, 1.0, 20),
        num_seeds=10,
        output_csv="automorphism_vs_p.csv"
    )
    plot_automorphism_vs_p(df)

