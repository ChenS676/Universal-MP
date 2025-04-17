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
#SBATCH --job-name=gnn_ddi
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

echo ">>> Environment and modules set up successfully <<<"
echo "Job started at: $(date)"

# List of GNN models
gnn_models=("GCN")

# Common parameters
DATA_NAME="ddi"
EPOCHS=1000
EVAL_STEPS=5
KILL_CNT=100
BATCH_SIZE=65536
HIDDEN_CHANNELS=256
RUNS=10

# Hyperparameters per model
declare -A LR=( ["GIN"]=0.001 ["GCN"]=0.01 ["SAGE"]=0.01 ["GAT"]=0.01 )
declare -A DROPOUT=( ["GIN"]=0 ["GCN"]=0.5 ["SAGE"]=0.5 ["GAT"]=0.5 )
declare -A NUM_LAYERS=( ["GIN"]=1 ["GCN"]=3 ["SAGE"]=3 ["GAT"]=3 )
declare -A NUM_LAYERS_PREDICTOR=( ["GIN"]=3 ["GCN"]=3 ["SAGE"]=3 ["GAT"]=3 )

for model in "${gnn_models[@]}"; do
    echo "------------------------------------------------------"
    echo "Running model: $model"
    echo "Start time: $(date)"
    CMD="python ddi_gnn.py --data_name $DATA_NAME --gnn_model $model --lr ${LR[$model]} --dropout ${DROPOUT[$model]} --num_layers ${NUM_LAYERS[$model]} --num_layers_predictor ${NUM_LAYERS_PREDICTOR[$model]} --hidden_channels $HIDDEN_CHANNELS --epochs $EPOCHS --eval_steps $EVAL_STEPS --kill_cnt $KILL_CNT --batch_size $BATCH_SIZE --runs $RUNS"
    
    echo "Executing: $CMD"
    eval $CMD || { echo "Error: $model training failed"; exit 1; }
    
    echo "End time: $(date)"
    echo "------------------------------------------------------"
done

echo ">>> All models completed successfully <<<"

# echo "Start time: $(date)"
# echo current command is: python ddi_gnn.py --data_name ogbl-ddi --gnn_model GIN  --lr 0.001 --dropout 0  --num_layers 1 --num_layers_predictor 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536
# python ddi_gnn.py --data_name ogbl-ddi --gnn_model GIN  --lr 0.001 --dropout 0  --num_layers 1 --num_layers_predictor 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536

# echo "Start time: $(date)"
# echo current command is: python ddi_gnn.py --data_name ogbl-ddi --gnn_model GCN  --lr 0.01 --dropout 0.5  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536 
# python ddi_gnn.py --data_name ogbl-ddi --gnn_model GCN  --lr 0.01 --dropout 0.5  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536 

# echo "Start time: $(date)"
# echo current command is: python ddi_gnn.py --data_name ogbl-ddi --gnn_model GCN  --lr 0.01 --dropout 0.5  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536 
# python ddi_gnn.py --data_name ogbl-ddi --gnn_model SAGE  --lr 0.01 --dropout 0.5  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536 


# echo "Start time: $(date)"
# echo current command is: python ddi_gnn.py --data_name ogbl-ddi --gnn_model GCN  --lr 0.01 --dropout 0.5  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536 
# python ddi_gnn.py --data_name ogbl-ddi --gnn_model GAT  --lr 0.01 --dropout 0.5  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536 
# echo "Start time: $(date)"