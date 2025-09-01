#! /bin/bash
#SBATCH --job-name=pirl_original_1-5_3-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17
#SBATCH --time=3:00:00
#SBATCH --output=pirl_2ri_logs/%x_%A.out
#SBATCH --error=pirl_2ri_logs/%x_%A.err
#SBATCH --gres=gpu:h100:2
#SBATCH --mem-per-gpu=31G
#SBATCH --account=dtce-schmidt
#SBATCH --partition=short
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
python main.py --data_dir $DATA_S/PIRL/run --ri_1 1.5 --ri_2 3.5 --reward_mode original

