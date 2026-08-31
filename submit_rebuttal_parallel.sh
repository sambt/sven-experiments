#!/bin/bash
#SBATCH --partition=iaifi_gpu_priority
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=slurm_logs/rebuttalp-%j.out

# Intra-GPU parallel scan: run NPROC processes on one GPU, each covering a
# disjoint 1/NPROC slice of the grid (via the scan's n_shards/shard_id).
# Small MLPs don't saturate the GPU, so this gives a near-NPROC speedup until
# compute-bound. Safe: disjoint shards write disjoint run_ids (dedup + no races).
#
# Usage:   NPROC=6 sbatch submit_rebuttal_parallel.sh <config.yaml> [hydra overrides...]
# Example: NPROC=8 sbatch submit_rebuttal_parallel.sh experiments/configs/rebuttal_baselines_toy_1d_scan.yaml
#          NPROC=4 sbatch submit_rebuttal_parallel.sh experiments/configs/rebuttal_overparam_mnist_scan.yaml n_data=5000
set -u
REPO=/n/home11/sambt/iaifi/sv3
PY=$REPO/.venv/bin/python
cd "$REPO"

config=$1
config_name=$(basename "$config" .yaml)
shift

N=${NPROC:-6}
echo "[parallel] $config_name : $N shards on 1 GPU  (extra overrides: $*)"

pids=()
for s in $(seq 0 $((N-1))); do
    "$PY" run.py --config-name "$config_name" +n_shards=$N +shard_id=$s "$@" \
        > "slurm_logs/${config_name}_shard${s}_of${N}_${SLURM_JOB_ID:-local}.out" 2>&1 &
    pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[parallel] done (fail=$fail)"
exit $fail
