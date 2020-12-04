#!/bin/bash -login
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --job-name=1D
#SBATCH --mem=10000M
#SBATCH --time=3:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14

module load languages/anaconda3/2019.10-3.7.4-tflow-2.1.0

cd $SLURM_SUBMIT_DIR
python PRE_TRAIN.py -c 0
python PRE_TRAIN.py -c 1
python PRE_TRAIN.py -c 2
python PRE_TRAIN.py -c 3
python PRE_TRAIN.py -c 4