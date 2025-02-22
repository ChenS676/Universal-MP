import os

def generate_latex_table(results_dict, datasets, metrics, caption="Results on link prediction benchmarks.", output_file="table.tex"):
    latex_code = """
    \begin{table*}[t]
    \vskip -0.1in
    \centering
    \caption{%s The format is average score $\pm$ standard deviation. OOM means out of GPU memory.}\label{tab:main_results}
    \vskip -0.05in
    \setlength{\tabcolsep}{2pt}
    \small{
    \begin{tabular}{l%s}
    \toprule
    & %s \\
    \midrule
    Metric & %s \\
    \midrule
    """ % (caption, 'c' * len(datasets), ' & '.join([f"\\textbf{{{d}}}" for d in datasets]), ' & '.join(metrics))
    
    for model, values in results_dict.items():
        scores = ' & '.join([f"${v[0]} {{\\scriptstyle \\pm {v[1]}}}$" if isinstance(v, tuple) else v for v in values])
        latex_code += f"\\textbf{{{model}}} & {scores} \\\n"
    
    latex_code += """
    \bottomrule
    \end{tabular}
    }
    \vskip -0.05in
    \end{table*}
    """
    
    with open(output_file, "w") as f:
        f.write(latex_code)
    
    return latex_code

# Example dictionary
results = {
    "CN": [(33.92, 0.46), (29.79, 0.90), (23.13, 0.15), (56.44, 0.00), (27.65, 0.00), (51.47, 0.00), (17.73, 0.00)],
    "AA": [(39.85, 1.34), (35.19, 1.33), (27.38, 0.11), (64.35, 0.00), (32.45, 0.00), (51.89, 0.00), (18.61, 0.00)],
    "RA": [(41.07, 0.48), (33.56, 0.17), (27.03, 0.35), (64.00, 0.00), (49.33, 0.00), (51.98, 0.00), (27.60, 0.00)]
}

datasets = ["Cora", "Citeseer", "Pubmed", "Collab", "PPA", "Citation2", "DDI"]
metrics = ["HR@100", "HR@100", "HR@100", "HR@50", "HR@100", "MRR", "HR@20"]

latex_table = generate_latex_table(results, datasets, metrics, output_file="table.tex")
print(f"LaTeX table saved to table.tex")
