#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=nano_rl_logs/nano_rl_original_%A.out
#SBATCH --error=nano_rl_logs/nano_rl_original_%A.err
#SBATCH --gres=gpu:v100:2
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=17
#SBATCH --account=dtce-schmidt
#SBATCH --partition=short
#SBATCH --job-name=nano_rl_original
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=dtce0085@ox.ac.uk

export DATA_S=/data/dtce-schmidt/dtce0085
export CONPREFIX=$DATA_S/.conda/pirl

which python
set -e
module load Mamba
which python
# source activate base
source activate $CONPREFIX
conda list
which python
nvidia-smi
ulimit -n 65535
ulimit -u 65535
cd $DATA_S/PIRL/Physics-Informed-Reinforcement-Learning
python main.py --data_dir $DATA_S/PIRL/run --ri_1 3.5 --ri_2 4.2 --reward_mode original

