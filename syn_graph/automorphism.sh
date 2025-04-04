#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=dev_cpuonly
#SBATCH --job-name=alpha_clac

#SBATCH --output=log/Alpha_%j.output
#SBATCH --error=error/Alpha_%j.error
#SBATCH --account=hk-project-pai00023   # specify the project group

#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cshao676@gmail.com

# Request GPU resources

# source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate base
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

datasets=("Cora" "Citeseer" "Pubmed" "Computers" "Photo" "ogbl-ddi" "ogbl-collab" "ogbl-ppa" "ogbl-citation2")

# Loop through datasets in pairs
for ((i=0; i<${#datasets[@]}; i+=2)); do
    echo "Start training grand on ${datasets[i]}"
    python automorphism.py --data_name "${datasets[i]}" &

    if ((i+1 < ${#datasets[@]})); then
        echo "Start training grand on ${datasets[i+1]}"
        python automorphism.py --data_name "${datasets[i+1]}" &
    fi

    # Wait for the two jobs to finish before starting the next pair
    wait
done