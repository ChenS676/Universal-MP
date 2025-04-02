# Metric,Hits@1,MRR,AUC,AP
# Cora_inter0.00_intra0.00_total0_Orbits_1007.00_Norm_0.72_ArScore_0.81,1.71 ± 1.69,5.45 ± 1.36,75.78 ± 8.23,77.43 ± 5.36
# Cora_inter0.10_intra0.50_total0_Orbits_1007.00_Norm_0.72_ArScore_0.81,2.94 ± 1.42,6.79 ± 1.82,76.79 ± 6.06,78.87 ± 3.96
# Cora_inter0.10_intra0.50_total500_Orbits_2343.00_Norm_0.69_ArScore_0.57,1.86 ± 1.32,5.55 ± 0.95,74.53 ± 10.47,76.62 ± 7.74
# Cora_inter0.10_intra0.50_total1000_Orbits_2754.00_Norm_0.66_ArScore_0.49,2.01 ± 0.90,4.59 ± 1.04,73.77 ± 10.17,75.77 ± 7.33
# Cora_inter0.10_intra0.50_total1500_Orbits_3100.00_Norm_0.64_ArScore_0.43,3.24 ± 1.43,5.87 ± 1.61,74.02 ± 9.52,75.73 ± 7.12
# Cora_inter0.10_intra0.50_total2000_Orbits_3358.00_Norm_0.61_ArScore_0.38,2.98 ± 1.33,5.84 ± 1.72,73.65 ± 9.06,75.93 ± 6.49
# Cora_inter0.10_intra0.50_total2500_Orbits_3630.00_Norm_0.58_ArScore_0.33,2.80 ± 1.87,6.00 ± 1.21,72.73 ± 10.47,75.07 ± 7.25
# Cora_inter0.10_intra0.50_total3000_Orbits_3834.00_Norm_0.56_ArScore_0.29,1.89 ± 0.57,5.06 ± 1.19,71.27 ± 11.76,73.76 ± 8.93
# Cora_inter0.10_intra0.50_total3500_Orbits_4010.00_Norm_0.54_ArScore_0.26,2.01 ± 1.10,4.50 ± 1.42,70.16 ± 12.47,72.42 ± 10.15
# Cora_inter0.10_intra0.50_total4000_Orbits_4196.00_Norm_0.51_ArScore_0.23,0.54 ± 0.86,4.36 ± 0.78,71.25 ± 11.29,73.60 ± 8.30
# Cora_inter0.10_intra0.50_total4500_Orbits_4380.00_Norm_0.49_ArScore_0.19,1.77 ± 0.91,4.81 ± 0.67,71.83 ± 10.43,73.98 ± 7.21
# Cora_inter0.10_intra0.50_total5000_Orbits_4520.00_Norm_0.46_ArScore_0.17,1.42 ± 0.59,3.76 ± 0.89,70.33 ± 10.40,72.25 ± 7.63
# Cora_inter0.10_intra0.50_total5500_Orbits_4629.00_Norm_0.43_ArScore_0.15,0.83 ± 0.29,2.67 ± 0.29,69.13 ± 10.77,71.14 ± 7.49
# Cora_inter0.10_intra0.50_total6000_Orbits_4719.00_Norm_0.41_ArScore_0.13,0.75 ± 0.64,3.53 ± 0.65,71.15 ± 7.93,73.42 ± 5.31
# Cora_inter0.10_intra0.50_total6500_Orbits_4801.00_Norm_0.39_ArScore_0.11,3.21 ± 1.10,5.79 ± 0.66,70.55 ± 7.75,73.10 ± 5.30
# Cora_inter0.10_intra0.50_total7000_Orbits_4862.00_Norm_0.38_ArScore_0.10,2.14 ± 0.62,4.79 ± 0.95,68.57 ± 10.97,71.22 ± 8.18


import pandas as pd
import matplotlib.pyplot as plt
import re

# Raw data as a multiline string
raw_data = """
Metric,Hits@1,MRR,AUC,AP
Cora_inter0.00_intra0.00_total0_Orbits_1007.00_Norm_0.72_ArScore_0.81,1.71 ± 1.69,5.45 ± 1.36,75.78 ± 8.23,77.43 ± 5.36
Cora_inter0.10_intra0.50_total0_Orbits_1007.00_Norm_0.72_ArScore_0.81,2.94 ± 1.42,6.79 ± 1.82,76.79 ± 6.06,78.87 ± 3.96
Cora_inter0.10_intra0.50_total500_Orbits_2343.00_Norm_0.69_ArScore_0.57,1.86 ± 1.32,5.55 ± 0.95,74.53 ± 10.47,76.62 ± 7.74
Cora_inter0.10_intra0.50_total1000_Orbits_2754.00_Norm_0.66_ArScore_0.49,2.01 ± 0.90,4.59 ± 1.04,73.77 ± 10.17,75.77 ± 7.33
Cora_inter0.10_intra0.50_total1500_Orbits_3100.00_Norm_0.64_ArScore_0.43,3.24 ± 1.43,5.87 ± 1.61,74.02 ± 9.52,75.73 ± 7.12
Cora_inter0.10_intra0.50_total2000_Orbits_3358.00_Norm_0.61_ArScore_0.38,2.98 ± 1.33,5.84 ± 1.72,73.65 ± 9.06,75.93 ± 6.49
Cora_inter0.10_intra0.50_total2500_Orbits_3630.00_Norm_0.58_ArScore_0.33,2.80 ± 1.87,6.00 ± 1.21,72.73 ± 10.47,75.07 ± 7.25
Cora_inter0.10_intra0.50_total3000_Orbits_3834.00_Norm_0.56_ArScore_0.29,1.89 ± 0.57,5.06 ± 1.19,71.27 ± 11.76,73.76 ± 8.93
Cora_inter0.10_intra0.50_total3500_Orbits_4010.00_Norm_0.54_ArScore_0.26,2.01 ± 1.10,4.50 ± 1.42,70.16 ± 12.47,72.42 ± 10.15
Cora_inter0.10_intra0.50_total4000_Orbits_4196.00_Norm_0.51_ArScore_0.23,0.54 ± 0.86,4.36 ± 0.78,71.25 ± 11.29,73.60 ± 8.30
Cora_inter0.10_intra0.50_total4500_Orbits_4380.00_Norm_0.49_ArScore_0.19,1.77 ± 0.91,4.81 ± 0.67,71.83 ± 10.43,73.98 ± 7.21
Cora_inter0.10_intra0.50_total5000_Orbits_4520.00_Norm_0.46_ArScore_0.17,1.42 ± 0.59,3.76 ± 0.89,70.33 ± 10.40,72.25 ± 7.63
Cora_inter0.10_intra0.50_total5500_Orbits_4629.00_Norm_0.43_ArScore_0.15,0.83 ± 0.29,2.67 ± 0.29,69.13 ± 10.77,71.14 ± 7.49
Cora_inter0.10_intra0.50_total6000_Orbits_4719.00_Norm_0.41_ArScore_0.13,0.75 ± 0.64,3.53 ± 0.65,71.15 ± 7.93,73.42 ± 5.31
Cora_inter0.10_intra0.50_total6500_Orbits_4801.00_Norm_0.39_ArScore_0.11,3.21 ± 1.10,5.79 ± 0.66,70.55 ± 7.75,73.10 ± 5.30
Cora_inter0.10_intra0.50_total7000_Orbits_4862.00_Norm_0.38_ArScore_0.10,2.14 ± 0.62,4.79 ± 0.95,68.57 ± 10.97,71.22 ± 8.18
"""

# Parse lines
lines = raw_data.strip().split("\n")
header = lines[0].split(",")
records = []

for line in lines[1:]:
    parts = line.split(",")
    metric = parts[0]
    auc = float(parts[3].split("±")[0].strip())
    match = re.search(r"ArScore_([\d.]+)", metric)
    arscore = float(match.group(1)) if match else None
    records.append((arscore, auc))

# Create DataFrame
df = pd.DataFrame(records, columns=["ArScore", "AUC"])

# Plot
plt.figure(figsize=(8, 5))
plt.plot(df["ArScore"], df["AUC"], marker='o', linestyle='-')
plt.xlabel("ArScore")
plt.ylabel("AUC")
plt.title("Relationship between AUC and ArScore")
plt.grid(True)
plt.tight_layout()
plt.savefig('syn_citeseer.pdf', format='pdf')
