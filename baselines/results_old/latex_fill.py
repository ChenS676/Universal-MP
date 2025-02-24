import pandas as pd

# Load the Excel file
file_path = "baselines/results/extracted_data_with_variance.xlsx"  # Change this to your actual file path
df = pd.read_excel(file_path, index_col=0)  # Assuming the first column is the index (Metric names)

# Define LaTeX table template
latex_template = """\\begin{table*}[t]
    \\vskip -0.1in
    \\centering
    \\caption{Results on link prediction benchmark datasets. Values are presented as the mean ± standard deviation. OOM indicates an out-of-memory (OOM) error on the GPU.}\\label{{tab:main_results}}
\\vskip -0.05in
\\setlength{{\\tabcolsep}}{{2pt}}
\\small{{
    \\begin{{tabular}}{{lccccccc}}
    \\toprule
         &
         \\textbf{{Cora}} &  
         \\textbf{{Citeseer}} & 
         \\textbf{{Pubmed}} &
         \\textbf{{Collab}} &
         \\textbf{{PPA}} &
         \\textbf{{Citation2}} 
         &\\textbf{{DDI}} 
         \\\\
\\midrule
          Metric &
          HR@100 &
          HR@100 & 
          HR@100 &
          HR@50 &
          HR@100 &
          MRR 
          &HR@20
         \\\\
         \\midrule
{}
         \\bottomrule
\\end{{tabular}}
}}
\\end{{table*}}
"""

# Function to format each row
def format_latex_row(metric, values):
    formatted_values = [f"${val:.2f} \\scriptstyle \\pm {std:.2f}$" if isinstance(val, (int, float)) else val 
                        for val, std in zip(values[::2], values[1::2])]
    return f"        \\textbf{{{metric}}} & " + " & ".join(formatted_values) + " \\\\\n"

# Convert DataFrame rows into LaTeX format
latex_rows = "".join(format_latex_row(metric, df.loc[metric].values) for metric in df.index)

# Fill the LaTeX template
latex_table = latex_template.format(latex_rows)

# Save to a .tex file
output_file = "output_table.tex"
with open(output_file, "w") as f:
    f.write(latex_table)

print(f"LaTeX table saved to {output_file}")
