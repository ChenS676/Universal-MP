#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=accelerated

#SBATCH --job-name=ddi_automorphic
#SBATCH --output=logs/%x_%j.out      # Save output to logs/jobname_jobid.out
#SBATCH --error=logs/%x_%j.err       # Save errors to logs/jobname_jobid.err

#SBATCH --account=hk-project-pai00001   # specify the project group

#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cshao676@gmail.com
# Request GPU resources
#SBATCH --gres=gpu:1

source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate base

 
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18   
module load devel/cuda/11.8   
module load compiler/gnu/12
conda activate EAsF
 
echo ">>> .bashrc executed: Environment and modules are set up. <<<"
# Print loaded modules

echo ">>> Environment and modules set up successfully <<<"
echo "Job started at: $(date)"

# List of GNN models
# gnn_models=("GIN")

# Common parameters

cd /hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/syn_real


# GNN models to run
MODELS=("MixHopGCN") #   #"GIN" "SAGE" "ChebGCN" "MixHopGCN" "GAT"
DATA="Citeseer" #ogbl-ddi Citeseer

for ((i = 0; i < ${#MODELS[@]}; i+=2)); do
    MODEL1="${MODELS[i]}"
    MODEL2="${MODELS[i+1]}"

    echo "Launching $MODEL1 and ${MODEL2:-"(none)"} in parallel for dataset: $DATA"

    python real_syn_automorphic.py \
        --data_name "$DATA" \
        --gnn_model "$MODEL1" \
        --wandb_log \
        --epochs 100 \
        --runs 10 &

    # if [ -n "$MODEL2" ]; then
    #     python real_syn_automorphic.py \
    #         --data_name "$DATA" \
    #         --gnn_model "$MODEL2" \
    #         --wandb_log \
    #         --epochs 100 \
    #         --runs 10 &
    # fi

    wait  # Wait for both background jobs to finish
done

# python tune.py --data_name Cora --gnn_model GCN --wandb_log
# python tune.py --data_name Citeseer --gnn_model GCN --wandb_log
# python tune.py --data_name ogbl-ddi --gnn_model GCN --wandb_log
# python real_syn_automorphic.py --data_name Cora --gnn_model GCN --wandb_log --epochs 10 --wandb_log 
# python real_syn_automorphic.py --data_name Cora --gnn_model GAT --wandb_log  --epochs 10 --wandb_log 
# python real_syn_automorphic.py --data_name Cora --gnn_model GIN --wandb_log  --epochs 10 --wandb_log 
# python real_syn_automorphic.py --data_name Cora --gnn_model SAGE --wandb_log  --epochs 10 --wandb_log 
# python real_syn_automorphic.py --data_name Cora --gnn_model ChebGCN --wandb_log  --epochs 10 --wandb_log 
# python real_syn_automorphic.py --data_name Cora --gnn_model MixHopGCN --wandb_log  --epochs 10 --wandb_log 


# python real_syn_automorphic.py --data_name Citeseer --gnn_model GCN --wandb_log --epochs 10 --wandb_log 
# python real_syn_automorphic.py --data_name Citeseer --gnn_model GAT --wandb_log  --epochs 10 --wandb_log 
# python real_syn_automorphic.py --data_name Citeseer --gnn_model GIN --wandb_log  --epochs 10 --wandb_log 
# python real_syn_automorphic.py --data_name Citeseer --gnn_model SAGE --wandb_log  --epochs 10 --wandb_log 
# python real_syn_automorphic.py --data_name Citeseer --gnn_model ChebGCN --wandb_log  --epochs 10 --wandb_log 
# python real_syn_automorphic.py --data_name Citeseer --gnn_model MixHopGCN --wandb_log  --epochs 10 --wandb_log 


# python real_syn_automorphic.py --data_name ogbl-ddi --gnn_model GCN --wandb_log --epochs 100 --runs 10
# python real_syn_automorphic.py --data_name ogbl-ddi --gnn_model GAT --wandb_log  --epochs 100  --runs 10
# python real_syn_automorphic.py --data_name ogbl-ddi --gnn_model GIN --wandb_log  --epochs 100  --runs 10
# python real_syn_automorphic.py --data_name ogbl-ddi --gnn_model SAGE --wandb_log  --epochs 100  --runs 10
# python real_syn_automorphic.py --data_name ogbl-ddi --gnn_model ChebGCN --wandb_log  --epochs 100  --runs 10
# python real_syn_automorphic.py --data_name ogbl-ddi --gnn_model MixHopGCN --wandb_log  --epochs 100  --runs 10
