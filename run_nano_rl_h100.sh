#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=nano_rl_logs/nano_rl_%A.out
#SBATCH --error=nano_rl_logs/nano_rl_%A.err
#SBATCH --gres=gpu:h100:2
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=20
#SBATCH --account=dtce-schmidt
#SBATCH --partition=short
#SBATCH --job-name=nano_rl
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=<insert your email here>

set -e
module load Anaconda3
source activate base
conda activate $DATA/venvs/nano_rl
which python
nvidia-smi
ulimit -n 65535
ulimit -u 65535

# Preserve Ray's session logs (raylet/gcs/dashboard_agent) for post-mortem even on failure,
# since /tmp is per-job and wiped at job end. These reveal the real cause behind the masked
# "Failed to start the dashboard" / "Unable to register worker with raylet" errors.
RAY_LOG_DEST="${SLURM_SUBMIT_DIR:-.}/ray_logs_${SLURM_JOB_ID}"
trap 'mkdir -p "$RAY_LOG_DEST"; cp -r /tmp/ray/session_*/logs "$RAY_LOG_DEST"/ 2>/dev/null || true' EXIT

cd $DATA/repos/Physics-Informed-Reinforcement-Learning
python main.py --data_dir $DATA/repos/Physics-Informed-Reinforcement-Learning/run

