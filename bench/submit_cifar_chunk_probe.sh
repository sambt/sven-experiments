#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --job-name=cifar_chunk_probe
#SBATCH --output=slurm_logs/cifar_chunk_probe-%j.out
set -u; cd /n/home/anon/sven-experiments; OUT=bench/cifar_chunk_probe_${SLURM_JOB_ID}.jsonl
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run() { echo "=== $* ($(date +%T)) ==="; PYTHONPATH=. .venv/bin/python bench/cifar_chunk_probe.py "$@" --out $OUT 2>&1 | grep -E "RESULT|Error|error|Traceback"; }
run --capture hooks
run --chunk-numel 4194304      # 2^22: 4 groups (current)
run --chunk-numel 8388608      # 2^23: 2 groups
run --chunk-numel 16777216     # 2^24: 1 group = full (B,P) Jacobian
run --chunk-numel 16777216 --k 128
echo "=== done $(date +%T) ==="
