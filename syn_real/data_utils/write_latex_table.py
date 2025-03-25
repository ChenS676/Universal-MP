import pandas as pd

# Load the dataset
file_path = "baselines/data_utils/ogb_all_graph_metric_False.csv"  # Update with the correct file path
df = pd.read_csv(file_path)

# Mapping of required statistics to corresponding LaTeX labels
column_mapping = {
    "num_nodes": "#Nodes $|\\mathcal{V}|$",
    "num_edges": "#Edges $|\\mathcal{E}|$",
    "avg_deg": "Avg Deg (G)",
    "avg_deg2": "Avg Deg (G2)",
    "avg_cc": "Clustering",
    "avg_shortest_path": "Shortest Paths",
    "transitivity": "Transitivity",
    "degree_gini": "Deg Gini",
    "coreness_gini": "Coreness Gini",
    "deg_heterogeneity": "Heterogeneity",
    "power_law_estimate": "Power Law $\\alpha$",
}

# Selecting and renaming the required columns
df_selected = df[["name"] + list(column_mapping.keys())].rename(columns=column_mapping)

# Generating the LaTeX table content
latex_table_header = "        & " + " & ".join([f"\\textbf{{{name}}}" for name in df_selected['name']]) + " \\\\\n        \\midrule\n"

# Generating rows
latex_table_rows = "\n".join(
    f"        \\textbf{{{column}}} & " + " & ".join(map(lambda x: f"{x:.2f}", df_selected[column].values)) + " \\\\"
    for column in column_mapping.values()
)

# Full LaTeX table
latex_output = f"""
\\section{{Dataset Statistic}}
\\begin{{table*}}[ht]
    \\centering
    \\caption{{Statistics of standard benchmark graphs}}
    \\begin{{adjustbox}}{{width=\\textwidth}}
    \\begin{{tabular}}{{l{"c" * (len(df_selected) - 1)}}}
        \\toprule 
{latex_table_header}{latex_table_rows}
        \\bottomrule
    \\end{{tabular}}
    \\end{{adjustbox}}
    \\label{{tab:graph-stats}}
\\end{{table*}}
"""

# Save to a file or print
with open("dataset_statistics.tex", "w") as f:
    f.write(latex_output)

print("LaTeX table generated and saved as dataset_statistics.tex")
