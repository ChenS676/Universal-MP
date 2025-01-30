#!/bin/bash

#SBATCH--time=1:00:00
#SBATCH--partition=dev_accelerated
#SBATCH--job-name=gcn4cn
#SBATCH--gres=gpu:1

#SBATCH--output=log/UniversalMPNN_%j.output
#SBATCH--error=error/UniversalMPNN_%j.error
#SBATCH--account=hk-project-pai00001#specifytheprojectgroup

#SBATCH--chdir=/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/trials

#Notificationsettings:
#SBATCH--mail-type=ALL
#SBATCH--mail-user=chen.shao2@kit.edu

#RequestGPUresources
source/hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

condaactivatess

#<<<condainitialize<<<
modulepurge
moduleloaddevel/cmake/3.18
moduleloaddevel/cuda/11.8
moduleloadcompiler/gnu/12


cd/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/trials
#Arrayofmodelnames



models=("Custom_GAT""Custom_GCN""GraphSAGE""Custom_GIN""LINKX")

formodelin"${models[@]}";do
echo"pythongcn2struc.py--model"$model"&"
pythongcn2struc.py--model"$model"--h_keyPPR&
done

wait
