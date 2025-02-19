#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --mem=501600mb
#SBATCH --ntasks=152
#SBATCH --partition=cpuonly
#SBATCH --job-name=ogb_nx_stats
#SBATCH --nodes=192
#SBATCH --output=log/Universal_MPNN_%j.output
#SBATCH --error=error/Universal_MPNN_%j.error
#SBATCH --account=hk-project-pai00023   # specify the project group

#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cshao676@gmail.com

# Request GPU resources

source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate base

 
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18   
module load devel/cuda/11.8   
module load compiler/gnu/12
conda activate EAsF
cd /hkfs/work/workspace/scratch/cc7738-rebuttal 
cd /hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/baselines/data_utils
 
echo ">>> .bashrc executed: Environment and modules are set up. <<<"
# Print loaded modules

echo "Start time: $(date)"

data_name=(ppa vessel collab citation2 ddi) 

for data in "${data_name[@]}"; do
    time python ogb_data.py --data_name $data_name
done
