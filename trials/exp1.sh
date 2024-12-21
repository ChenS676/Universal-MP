#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=L40S


# execute your commands
cd /mnt/webscistorage/cc7738/ws_chen/Universal-MP/trials

# Array of model names
models=("Custom_GAT" "Custom_GCN" "GraphSAGE" "Custom_GIN" "LINKX")

# Iterate over each model and run the Python script in the background
for model in "${models[@]}"; do
    python linkx2.py --model "$model" &
done

# Optional: Wait for all background processes to complete before exiting the script
wait
