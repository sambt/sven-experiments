#!/bin/bash
#SBATCH --partition=iaifi_gpu_priority
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/rebuttal-%j.out

# Rebuttal scans use the uv env (has torch_optimizer + kfac-pytorch for the
# Shampoo / K-FAC baselines), NOT the jax conda env.
set -e
REPO=/n/home11/sambt/iaifi/sv3
VENV=$REPO/.venv/bin/python
cd "$REPO"

config=$1
config_name=$(basename "$config" .yaml)
shift

"$VENV" run.py --config-name "$config_name" "$@"
