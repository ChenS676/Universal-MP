# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the data
# file_path = "/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/syn_graph/tab1/tab1_data.csv"
# df = pd.read_csv(file_path, skiprows=1, header=None)

# # Rename columns
# column_names = [
#     "alpha_data_name", "num_node", "Metric", "Best Valid", "Best Valid Mean",
#     "Mean List", "Variance List", "Test Result"
# ]
# df.columns = column_names

# # Split "alpha_data_name" into "alpha" and "data_name"
# df[['alpha', 'data_name']] = df['alpha_data_name'].str.extract(r'([\d.]+)\s*(\w+)')
# df["alpha"] = df["alpha"].astype(float)

# # Extract Best Valid values and variance
# df["Best Valid Value"] = df["Best Valid"].str.extract(r'([\d.]+)').astype(float) / 100  # Convert to float between 0 and 1
# df["Variance"] = df["Best Valid"].str.extract(r'± ([\d.]+)').astype(float) / 100  # Convert to float between 0 and 1

# # Drop the original merged column
# df.drop(columns=["alpha_data_name"], inplace=True)

# # Group data by model type
# models = df["data_name"].unique()

# # Create the plot
# fig, ax = plt.subplots(figsize=(10, 6))

# # Define line styles and colors
# colors = plt.cm.get_cmap('tab10', len(models))
# dashed_models = {"ChebGCN", "LINKX", "GIN"}  # Models with dashed lines
# line_styles = {model: "--" if model in dashed_models else "-" for model in models}
# alpha_values = {model: 0.5 if model in dashed_models else 1.0 for model in models}  # Adjust opacity

# # Plot data with solid markers and error bars
# for idx, model in enumerate(models):
#     subset = df[df["data_name"] == model]
#     color = colors(idx, alpha=alpha_values[model])
    
#     # Plot the main trend line with markers
#     ax.plot(
#         subset["alpha"],
#         subset["Best Valid Value"],
#         linestyle=line_styles[model],
#         linewidth=2,
#         color=color,
#         label=model,
#         marker='o',
#         markersize=6,
#         markerfacecolor=colors(idx, alpha=1.0),
#         markeredgecolor='black',
#         markeredgewidth=0.8
#     )

#     # Add error bars with reduced transparency
#     ax.errorbar(
#         subset["alpha"],
#         subset["Best Valid Value"],
#         yerr=subset["Variance"],
#         fmt='none',
#         capsize=6,
#         elinewidth=1.5,
#         capthick=1.5,
#         color=colors(idx, alpha=0.3)
#     )

# # Formatting the plot
# ax.set_xlabel(r"$\alpha$", fontsize=16)
# ax.set_ylabel("AUC", fontsize=16)
# ax.set_xticks(df["alpha"].unique())
# ax.set_yticks(np.arange(0, 1.1, 0.1))  # Y-axis from 0 to 1 with float values
# ax.grid(True)
# ax.legend(fontsize=12, loc="lower right", bbox_to_anchor=(1, 0.5))  # Ensure legend is fully visible
# plt.tight_layout()

# plt.savefig('tab1_plot2.pdf')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict

# Define sample data (replace this with your actual dataset)
raw_data = [
    ("GCN", [0.1, 0.3, 0.5, 0.7, 0.9], [32.5, 67.5, 65.0, 82.5, 100.0], [20.58, 23.72, 17.48, 20.58, 0.0]),
    ("GAT", [0.1, 0.3, 0.5, 0.7, 0.9], [55.0, 77.5, 62.5, 92.5, 97.5], [32.91, 24.86, 24.3, 16.87, 7.91]),
    ("GIN", [0.1, 0.3, 0.5, 0.7, 0.9], [66.25, 70.5, 75.0, 97.5, 97.5], [16.72, 18.45, 11.79, 7.91, 7.91]),
    ("GraphSAGE", [0.1, 0.3, 0.5, 0.7, 0.9], [47.5, 75.0, 80.0, 87.5, 95.0], [27.51, 16.67, 22.97, 21.25, 15.81]),
    ("MixHopGCN", [0.1, 0.3, 0.5, 0.7, 0.9], [75.0, 80.0, 80.0, 82.5, 95.0], [28.87, 17.48, 10.54, 20.58, 15.81]),
    ("ChebGCN", [0.1, 0.3, 0.5, 0.7, 0.9], [65.0, 67.5, 66.25, 67.5, 67.5], [29.34, 21.89, 20.45, 21.89, 26.48]),
    ("LINKX", [0.1, 0.3, 0.5, 0.7, 0.9], [85.0, 87.0, 80.5, 90.0, 87.5], [15.81, 22.97, 7.91, 23.57, 17.68])
]

# Define new interpolated alpha values
new_alpha = np.arange(0.1, 1.0, 0.1)

# Create a new dictionary to store interpolated results
interpolated_data = defaultdict(dict)

# Perform interpolation for each model
for model, alpha, best_valid, variance in raw_data:
    f_best_valid = interp1d(alpha, best_valid, kind='linear', fill_value="extrapolate")
    f_variance = interp1d(alpha, variance, kind='linear', fill_value="extrapolate")

    interpolated_data[model]["alpha"] = new_alpha.tolist()
    interpolated_data[model]["best_valid"] = f_best_valid(new_alpha).tolist()
    interpolated_data[model]["variance"] = f_variance(new_alpha).tolist()

# Create the updated plot with error bars and reduced transparency for error bars
fig, ax = plt.subplots(figsize=(10, 6))

# Use 'tab10' colormap for distinguishable colors
colors = plt.cm.get_cmap('tab10', len(interpolated_data))

# Define different line styles and transparency settings
dashed_models = {"ChebGCN", "LINKX", "GIN"}  # Models that will have dashed lines
line_styles = {model: "--" if model in dashed_models else "-" for model in interpolated_data.keys()}
alpha_values = {model: 0.5 if model in dashed_models else 1.0 for model in interpolated_data.keys()}  # Reduce opacity for dashed lines

# Plot interpolated data with solid markers and error bars
for idx, (model, values) in enumerate(interpolated_data.items()):
    color = colors(idx)

    # Plot main lines
    ax.plot(
        values["alpha"],
        values["best_valid"],
        linestyle=line_styles[model],
        linewidth=2,
        color=color,
        label=model,
        marker='o',
        markersize=6,
        markerfacecolor=color,
        markeredgecolor='black',
        markeredgewidth=0.8
    )

    # Add error bars with reduced transparency
    ax.errorbar(
        values["alpha"],
        values["best_valid"],
        yerr=values["variance"],
        fmt='o',  # Markers only for error bars
        color=color,
        alpha=0.3,  # Reduced transparency for error bars
        capsize=6,
        elinewidth=2,
        capthick=2
    )

fontsize = 16
# Formatting the plot
ax.set_xlabel(r"$\alpha$", fontsize=fontsize)
ax.set_ylabel("AUC (/%)", fontsize=fontsize)
ax.set_xticks(new_alpha)
ax.set_yticks(np.arange(0, 101, 10))
ax.tick_params(axis='both', labelsize=fontsize) 
ax.grid(True)
ax.legend(fontsize=fontsize, loc="lower right")
plt.tight_layout()

plt.savefig('tab1_plot2.pdf')
