#!/usr/bin/env bash

set -e
echo "Creating Python 3.10 conda env in $DATA/venvs/nano_rl"
mkdir -p $DATA/venvs
module load Anaconda3
source activate base
conda create -p $DATA/venvs/nano_rl python=3.10 -y
echo "Activating Conda environment"
conda activate $DATA/venvs/nano_rl

# echo "Cloning PIRL repository to $DATA/repos"
# mkdir $DATA/repos
# cd $_
# git clone --recursive git@github.com:OllyK/Physics-Informed-Reinforcement-Learning.git

echo "Installing PIRL repository requirements to conda env"
cd $DATA/repos/Physics-Informed-Reinforcement-Learning
pip install -r requirements.txt

echo "Installing deflector-gym submodule"
cd deflector-gym 
pip install .

echo "Making directory for Slurm logs in $DATA/nano_rl_logs"
mkdir -p $DATA/nano_rl_logs

conda deactivate
echo "Finished!"

