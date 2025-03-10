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

Ns=(90000 100000 120000 140000 160000 180000 200000) 
                      
for N in "${Ns[@]}"
do
    echo "Running N=$N"
    python lp_gcn_syn.py --N $N --data_name 'RegularTilling.KAGOME_LATTICE'

done




# Ns=(100 400 500 800)
# for N in "${Ns[@]}"
# do
#     echo "Running N=$N"
#     python lp_gcn_syn.py --N $N --data_name 'RegularTilling.SQUARE_GRID'

# done

