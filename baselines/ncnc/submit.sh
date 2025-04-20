#!/bin/bash
#SBATCH --time=2:00:00
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
#SBATCH --job-name=test

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
cd /hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/baselines/ncnc/

echo ">>> Environment and modules are set up. <<<"

# Define GNN models and hyperparameters
# python ogb_linkx_tune.py --data_name ddi
# python ogb_linkx_tune.py --data_name ppa
# python ogb_linkx_tune.py --data_name collab

# python plaintoid_linkx_wl.py --data_name Cora --cat_wl_feat
# python plaintoid_linkx_wl.py --data_name Citeseer --cat_wl_feat
# python plaintoid_linkx_wl.py --data_name Pubmed --cat_wl_feat

# for data in Cora Citeseer Pubmed Photo Computers
# do
#   echo "python plaintoid_linkx_wl.py --data_name $data"
#   python plaintoid_linkx_wl.py --data_name $data

#   echo "python plaintoid_linkx_wl.py --data_name $data --cat_wl_feat --wl_process norm"
#   python plaintoid_linkx_wl.py --data_name $data --cat_wl_feat --wl_process norm 

#   echo "python plaintoid_linkx_wl.py --data_name $data --cat_wl_feat --wl_process unique"
#   python plaintoid_linkx_wl.py --data_name $data --cat_wl_feat --wl_process unique
# done

# python ogb_linkx.py --data_name ddi --epochs 100  --eval_steps 5
# python ogb_linkx.py --data_name ddi --epochs 9990  --eval_steps 20  --cat_wl_feat --wl_process norm 
#  python ogb_linkx.py --data_name ddi --epochs 9999  --eval_steps 20 --cat_wl_feat --wl_process unique

# python ogb_linkx.py --data_name collab --epochs 100  --eval_steps 5
# python ogb_linkx.py --data_name collab --cat_wl_feat --wl_process norm --epochs 9999  --eval_steps 5
# python ogb_linkx.py --data_name collab --cat_wl_feat --wl_process unique --epochs 9999 --eval_steps 5
python NeighborOverlap_eval.py   --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.05  --probscale 4.3 --proboffset 2.8  --alpha 1.0  --gnnlr 0.0043 --prelr 0.0024  --batch_size 1152  --ln --lnnn  --dataset Cora --model puregcn --hiddim 256 --mplayers 1  --testbs 8192 --maskinput  --jk  --use_xlin  --tailact --epochs 9999 --runs 2 --name_tag Cora_GCNCN1 --cat_wl_feat --wl_process norm --predictor fuse1

echo ">>> All models completed successfully <<<"
echo "Job finished at: $(date)"


