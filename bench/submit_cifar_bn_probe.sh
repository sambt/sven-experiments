#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --job-name=cifar_bn_probe
#SBATCH --output=slurm_logs/cifar_bn_probe-%j.out
set -u; cd /n/home11/sambt/iaifi/sv3; PY=.venv/bin/python; OUT=bench/cifar_bn_probe_${SLURM_JOB_ID}.jsonl
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run() { echo "=== $* ($(date +%T)) ==="; PYTHONPATH=. $PY bench/cifar_bn_probe.py "$@" --out $OUT 2>&1 | grep -E "RESULT|Error|Traceback|error" ; }
run --mode adam --lr 0.01
run --mode hooks_frozen   --k 128 --lr 0.1 --rtol 1e-4     # current scan best
run --mode chunked_batch  --k 128 --lr 0.1 --rtol 1e-4
run --mode chunked_batch  --k 64  --lr 1.0 --rtol 1e-3     # old classic best settings
run --mode chunked_frozen --k 64  --lr 1.0 --rtol 1e-3
run --mode classic        --k 64  --lr 1.0 --rtol 1e-3     # pre-Gram pipeline (randomized_v2 SVD)
echo "=== done $(date +%T) ==="
