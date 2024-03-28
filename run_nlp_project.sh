#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --job-name=nlp_runtest_job

# Load the module for Python 3.10.4 and PyTorch 1.12.1
pwd
module purge

module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
module load IPython/8.5.0-GCCcore-11.3.0

# Activate the virtual environment
source $HOME/nlp/bin/activate

# Run the commands for preprocessing
cd europarl-extract-master
./preprocess/preprocess_batch.sh corpora
python3 extract.py comparable -sl all -tl PL BG -i corpora/ -o corpora/ -s corpora/europarl_statements.csv -al -c speaker
python3 make_csv.py
cd ..
cd src
python3 preprocessing.py
python3 feat_engineering.py

# Run the commands for training
ipython -c "%run seq2seq.ipynb"