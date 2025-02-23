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

echo "Running command: time python  ogb_gnn.py  --data_name ppa  --gnn_model model --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024  --random_sampling"
echo "Start time: $(date)"


gnn_models=(GCN GIN SAGE GAT)

data_name="ogbl-collab"

gnn_models=("GIN" "GCN")

for model in "${gnn_models[@]}"; do
   time python gnn_ogb_heart.py --use_valedges_as_input --data_name "$data_name" \
       --gnn_model "$model" --hidden_channels 256 --lr 0.001 --dropout 0.0 \
       --num_layers 3 --num_layers_predictor 3 --epochs 9999 --kill_cnt 100 \
       --batch_size 65536
done

# TO DEBUG
# python gnn_ogb_heart.py  --use_valedges_as_input  --data_name ogbl-ddi  --gnn_model GCN --hidden_channels 256 --lr 0.001 --dropout 0.  --num_layers 3 --num_layers_predictor 3 --epochs 9999 --kill_cnt 100  --batch_size 65536 
# python gnn_ogb_heart.py  --use_valedges_as_input  --data_name ogbl-ddi  --gnn_model GIN --hidden_channels 256 --lr 0.001 --dropout 0.  --num_layers 3 --num_layers_predictor 3 --epochs 9999 --kill_cnt 100  --batch_size 65536 

# this performance is horrible
# python ddi_gnn.py --data_name ogbl-ddi --gnn_model GIN  --lr 0.01 --dropout 0.5  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 7 --eval_steps 1 --kill_cnt 100 --batch_size 65536