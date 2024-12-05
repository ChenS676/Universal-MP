#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=single
#SBATCH --mail-type=ALL
#SBATCH --output=/pfs/work7/workspace/scratch/cc7738-kdd25/Universal-MP/trials/
#SBATCH --error=/pfs/work7/workspace/scratch/cc7738-kdd25/Universal-MP/trials/error
#SBATCH --job-name=exp1


# execute your commands
cd /pfs/work7/workspace/scratch/cc7738-kdd25/Universal-MP/trials/

# Array of model names

models=("Custom_GAT" "Custom_GCN" "GraphSAGE" "Custom_GIN" "LINKX")

for model in "${models[@]}"; do

    python gcn2struc.py --model "$model" --h_key PPR
done

wait
