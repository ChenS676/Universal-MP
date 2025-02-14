# rank_table
import re
import pandas as pd

# Load the LaTeX table file
file_path = "table_content.txt"
with open(file_path, "r", encoding="utf-8") as f:
    latex_content = f.readlines()

# Extract performance rows
performance_data = []
inside_table = False
headers = []

for i, line in enumerate(latex_content):
    if i >= 17 and i <=64:
        inside_table = True
    if inside_table:
        # Extract values, removing standard deviations
        cleaned_line = re.sub(r"\{\\scriptsize\$\\pm[^}]+\}", "", line)
        if '&' in cleaned_line:
            columns = []
            for col in re.split(r"&", cleaned_line):
                if col.strip():
                    col = re.sub(r'\\+|\s+$', '', col.strip())
                    
                    col = col.split("}")[-1]
                    print(col)
                    columns.append(col.strip())
                    
            # columns = [col.strip() for col in re.split(r"&", cleaned_line) if col.strip()]
            if len(columns) > 1:
                performance_data.append(columns)

# Convert to DataFrame
df = pd.DataFrame(performance_data)
columns = ["Model"] + [f"Metric_{i+1}" for i in range(df.shape[1] - 1)]
df.columns = columns

# Convert metrics to numeric
for col in df.columns[2:]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Rank values in each column
df_ranked = df.copy()
for col in df.columns[1:]:
    df_ranked[col] = df[col].rank(ascending=False, method="min")

# Calculate average rank
df_ranked["Average Rank"] = df_ranked.iloc[:, 2:].mean(axis=1)

# Save and display results
output_path = "processed_ranking.csv"
df_ranked.to_csv(output_path, index=False)
print(f"Processed table saved to {output_path}")
