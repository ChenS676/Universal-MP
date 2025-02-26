import json
from best_params import best_params_dict

output_file = "formatted_best_params.py"

root = '/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/GRAND_LP_chen'
with open(root + '/' + output_file, "w") as f:
    f.write("best_params_dict = {\n")
    for key, subdict in best_params_dict.items():
        f.write(f'    "{key}": {json.dumps(subdict, indent=None, separators=(",", ":")).replace("false", "False").replace("true", "True")},\n')
    f.write("}\n")

import csv
csv_output_file = "formatted_best_params.csv"
# Save to CSV
with open(root + '/' + csv_output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    header = ["Dataset"] + list(next(iter(best_params_dict.values())).keys())
    writer.writerow(header)
    for dataset, params in best_params_dict.items():
        row = [dataset] + [params.get(key, "") for key in header[1:]]
        writer.writerow(row)
