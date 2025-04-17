#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=501600mb
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --output=log/Universal_MPNN_%j.output
#SBATCH --error=error/Universal_MPNN_%j.error
#SBATCH --account=hk-project-pai00023  # Specify the project group
#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cshao676@gmail.com
#SBATCH --job-name=gnn_ppa

# Exit script on any error
set -e

# Load environment and dependencies
source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh
conda activate base

module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12

conda activate EAsF

# Change to the appropriate directory
cd /hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/baselines

echo ">>> Environment and modules are set up. <<<"

# Define GNN models and hyperparameters
gnn_models=("GCN" "SAGE") #"GCN" "SAGE" "GIN" "GAT"
data_name="ppa"
HIDDEN_DIM=256
LR=0.001
DROPOUT=0.0
N_LAYERS=3
N_PREDICTORS=3
EPOCHS=1000
EVAL_STEPS=5
KILL_CNT=200
BATCH_SIZE=65536  
RUNS=2
# Enable debug mode if needed (set DEBUG=1 to test a single model)
DEBUG=0

# Loop through models and execute training
for model in "${gnn_models[@]}"; do
    echo "------------------------------------------------------"
    echo "Running model: $model"
    echo "Start time: $(date)"

    CMD="python gnn_ogb_heart.py --data_name $data_name \
         --gnn_model $model --hidden_channels $HIDDEN_DIM --lr $LR --dropout $DROPOUT \
         --num_layers $N_LAYERS --num_layers_predictor $N_PREDICTORS --epochs $EPOCHS \
         --kill_cnt $KILL_CNT --batch_size $BATCH_SIZE --runs $RUNS --eval_steps $EVAL_STEPS" 

    echo "Executing: $CMD"
    time $CMD || { echo "Error: $model training failed"; exit 1; }

    echo "End time: $(date)"
    echo "------------------------------------------------------"

    # Exit early in debug mode
    if [ "$DEBUG" -eq 1 ]; then
        echo "Debug mode enabled, exiting after first model."
        break
    fi
done

echo ">>> All models completed successfully <<<"
echo "Job finished at: $(date)"


