#!/bin/bash
#SBATCH --partition=iaifi_gpu_priority,iaifi_gpu,gpu
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --job-name=optimizer_profile
#SBATCH --output=slurm_logs/optimizer_profile-%j.out
# Memory / step-time profile, every architecture serially on ONE exclusively reserved node
# (co-tenant jobs skew launch-bound timings). Resumable: finished configurations are skipped.
#   PROFILE_CONFIGS="profile_mnist profile_cifar"   subset of configs
#   PROFILE_ARGS="profile.num_steps=3 ..."           extra hydra overrides (smoke tests)
set -u; set -f
cd /n/home11/sambt/iaifi/sv3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
CONFIGS=${PROFILE_CONFIGS:-"profile_toy_1d profile_polynomial profile_mnist profile_mnist_width profile_nanogpt profile_nanogpt_width profile_cifar"}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1; echo "node $(hostname) start $(date)"
for c in $CONFIGS; do
  echo "################ $c ($(date +%T)) ################"
  .venv/bin/python -m experiments.optimizer_profile --config-name $c print_config=false ${PROFILE_ARGS:-} 2>&1 | grep -v "it/s\|s/it\|^$"
done
echo "################ all done $(date) ################"
