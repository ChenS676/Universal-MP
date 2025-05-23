
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict

# Define sample data (replace this with your actual dataset)
raw_data = [
    ("GCN", [0.9, 0.7, 0.5, 0.3, 0.1], 
            [32.5, 67.5, 65.0, 82.5, 100.0], 
            [20.58, 23.72, 17.48, 20.58, 0.0]),
    ("GAT", [0.9, 0.7, 0.5, 0.3, 0.1], 
            [55.0, 77.5, 62.5, 92.5, 97.5], 
            [32.91, 24.86, 24.3, 16.87, 7.91]),
    ("GIN", [0.9, 0.7, 0.5, 0.3, 0.1], 
            [66.25, 70.5, 75.0, 97.5, 97.5], 
            [16.72, 18.45, 11.79, 7.91, 7.91]),
    ("GraphSAGE", [0.9, 0.7, 0.5, 0.3, 0.1], 
                [47.5, 75.0, 80.0, 87.5, 95.0], 
                [27.51, 16.67, 22.97, 21.25, 15.81]),
    ("MixHopGCN", [0.9, 0.7, 0.5, 0.3, 0.1], 
                [75.0, 80.0, 80.0, 82.5, 95.0], 
                [28.87, 17.48, 10.54, 20.58, 15.81]),
    ("ChebGCN", [0.9, 0.7, 0.5, 0.3, 0.1], 
                [65.0, 67.5, 66.25, 67.5, 67.5], 
                [29.34, 21.89, 20.45, 21.89, 26.48]),
    ("LINKX", [0.9, 0.7, 0.5, 0.3, 0.1], 
            [85.0, 87.0, 80.5, 90.0, 87.5], 
            [15.81, 22.97, 7.91, 23.57, 17.68])
]

# Define new interpolated alpha values
new_alpha = np.arange(0.1, 1.0, 0.1)

# Create a new dictionary to store interpolated results
interpolated_data = defaultdict(dict)

# Perform interpolation for each model
for model, alpha, best_valid, variance in raw_data:

    f_best_valid = interp1d(alpha, best_valid, kind='linear', fill_value="extrapolate")
    f_variance = interp1d(alpha, variance, kind='linear', fill_value="extrapolate")

    interpolated_data[model]["alpha"] = (new_alpha).tolist()
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

fontsize = 22
# Formatting the plot
ax.set_xlabel(r"$\alpha$", fontsize=fontsize)
ax.set_ylabel("AUC (/%)", fontsize=fontsize)
ax.set_xticks(new_alpha)
ax.set_yticks(np.arange(0, 101, 10))
ax.tick_params(axis='both', labelsize=fontsize) 
fontsize = 16
ax.legend(fontsize=fontsize, loc="lower left")
plt.tight_layout()

plt.savefig('tab1_plot2.pdf')
