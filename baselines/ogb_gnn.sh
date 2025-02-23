#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=accelerated


#SBATCH --output=log/Universal_MPNN_%j.output
#SBATCH --error=error/Universal_MPNN_%j.error
#SBATCH --account=hk-project-pai00023   # specify the project group

#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cshao676@gmail.com
#SBATCH --job-name=gnn_collab
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
cd /hkfs/work/workspace/scratch/cc7738-rebuttal 
cd /hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/baselines
 
echo ">>> .bashrc executed: Environment and modules are set up. <<<"
# Print loaded modules


# List of GNN models "GCN" "SAGE"
gnn_models=("GCN" "SAGE")
data_name="ogbl-collab"

# Loop through models and run training
for model in "${gnn_models[@]}"; do
    echo "------------------------------------------------------"
    echo "Running model: $model"
    echo "Start time: $(date)"

    CMD="python gnn_ogb_heart.py --use_valedges_as_input --data_name $data_name \
         --gnn_model $model --hidden_channels 256 --lr 0.001 --dropout 0.0 \
         --num_layers 3 --num_layers_predictor 3 --epochs 800 --kill_cnt 100 \
         --batch_size 65536"
    
    echo "Executing: $CMD"
    time $CMD || { echo "Error: $model training failed"; exit 1; }

    echo "End time: $(date)"
    echo "------------------------------------------------------"
done

echo ">>> All models completed successfully <<<"
echo "Job finished at: $(date)"

# TO DEBUG
# python gnn_ogb_heart.py  --use_valedges_as_input  --data_name ogbl-ddi  --gnn_model GCN --hidden_channels 256 --lr 0.001 --dropout 0.  --num_layers 3 --num_layers_predictor 3 --epochs 9999 --kill_cnt 100  --batch_size 65536 
# python gnn_ogb_heart.py  --use_valedges_as_input  --data_name ogbl-ddi  --gnn_model GIN --hidden_channels 256 --lr 0.001 --dropout 0.  --num_layers 3 --num_layers_predictor 3 --epochs 9999 --kill_cnt 100  --batch_size 65536 

# this performance is horrible
# python ddi_gnn.py --data_name ogbl-ddi --gnn_model GIN  --lr 0.01 --dropout 0.5  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 7 --eval_steps 1 --kill_cnt 100 --batch_size 65536