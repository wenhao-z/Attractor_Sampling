#!/bin/bash
#SBATCH --export=ALL
#SBATCH -p cpu
##SBATCH -n 8
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=30G
#SBATCH --job-name=NetSampling

##SBATCH -p taising
##SBATCH --gres=gpu:4

###### Mail to me if there is some error
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=wenhao.zhang@pitt.edu

echo "Job starting"

# Go to working directory
cd /home/wenhaoz1/Projects/InfoAug_Corr/

module load matlab-9.5 

echo "Running Matlab"
matlab -nodisplay -nosplash <scanNetPars.m 1>log.txt 2>err.txt

echo "Job finished"
