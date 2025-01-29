#!/bin/bash

#SBATCH--time=4:00:00
#SBATCH--partition=accelerated
#SBATCH--job-name=gcn4cn
#SBATCH--gres=gpu:1

#SBATCH--output=log/UniversalMPNN_%j.output
#SBATCH--error=error/UniversalMPNN_%j.error
#SBATCH--account=hk-project-pai00001#specifytheprojectgroup

#SBATCH--chdir=/hkfs/work/workspace/scratch/cc7738-rebuttal/Universal-MP/trials

#Notificationsettings:
#SBATCH--mail-type=ALL
#SBATCH--output=/pfs/work7/workspace/scratch/cc7738-kdd25/Universal-MP/trials/
#SBATCH--error=/pfs/work7/workspace/scratch/cc7738-kdd25/Universal-MP/trials/error
#SBATCH--job-name=exp1


#executeyourcommands
cd/pfs/work7/workspace/scratch/cc7738-kdd25/Universal-MP/trials/

#Arrayofmodelnames
models=("Custom_GAT""Custom_GIN")#"LINKX""Custom_GCN""GraphSAGE"
nodefeat=("adjacency""one-hot""random""original")

formodelin"${models[@]}";do
fornodefeatin"${nodefeat[@]}";do
echo"pythongcn2struc.py--model"$model"--dataset"ddi"--nodefeat"$nodefeat"--h_key"CN""
pythongcn2struc.py--model"$model"--dataset"ddi"--nodefeat"$nodefeat"--h_key"CN"
done
done


wait

