#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=accelerated
#SBATCH --job-name=gnn_plaintoid

#SBATCH --output=log/Universal_MPNN_%j.output
#SBATCH --error=error/Universal_MPNN_%j.error
#SBATCH --account=hk-project-pai00023   # specify the project group

#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cshao676@gmail.com

# Request GPU resources
#SBATCH --gres=gpu:1

# source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

# conda activate base

 
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18   
module load devel/cuda/11.8   
module load compiler/gnu/12
conda activate EAsF
cd /hkfs/work/workspace/scratch/cc7738-rebuttal 
cd /hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/syn_graph
 
echo ">>> .bashrc executed: Environment and modules are set up. <<<"
# Print loaded modules

echo "Start time: $(date)"

data_name='RegularTilling.KAGOME_LATTICE'


prs=(0.25 0.3 0.35 0.4 0.45) # 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7

for pr in "${prs[@]}"
do
    echo "Running with N=$N"
    python lp_gcn_syn.py --pr "$pr" 
done



