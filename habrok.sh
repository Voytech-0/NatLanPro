#!/bin/bash
# JOB HEADERS HERE

#SBATCH --job-name=seq2seq_multihead
#SBATCH --time=30:00:00
#SBATCH --mem=10G
#SBATCH --gpus-per-node=1

module purge

module load Python/3.10.8-GCCcore-12.2.0

source $HOME/venvs/bachelor/bin/activate

python3 /home3/$USER/seq2seq_multihead.py
