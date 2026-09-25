#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --job-name=bench_shard
#SBATCH --output=slurm_logs/bench_shard-%j.out
# Sharding benchmark on a gpu_test MIG slice (A100 3g.20gb). Fixed total work
# per setting; varies NPROC and per-process thread count. Same CPU allocation
# (8 cores) as the production launcher.
set -u
cd /n/home/anon/sven-experiments
PY=.venv/bin/python
OUT=bench/results_${SLURM_JOB_ID}.jsonl
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "cpus: $(nproc)  SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
for wl in toy1d mnist; do
  ep=$([ $wl = toy1d ] && echo 4 || echo 2)
  for th in 0 1; do
    for np in 1 2 4 8; do
      echo "=== $wl threads=$th nproc=$np ($(date +%T)) ==="
      $PY bench/bench_sharding.py --workload $wl --nproc $np --n-runs 8 --epochs $ep --threads $th --out $OUT
    done
  done
done
for th in 0 1; do
  for np in 1 2; do
    echo "=== cifar threads=$th nproc=$np ($(date +%T)) ==="
    $PY bench/bench_sharding.py --workload cifar --nproc $np --n-runs 2 --epochs 1 --threads $th --out $OUT
  done
done
echo "=== done $(date +%T) ==="
