#!/bin/bash
# GPT-2-small comparison: one FULL GPU per shard (Sven runs are ~33GB & ~9h, so the
# N-procs-per-GPU sharding is unsuitable). Splits the config by mode so Sven runs
# get dedicated GPUs and baselines pack a few per shard, each shard as its own job.
set -u
cd /n/home11/sambt/iaifi/sv3
CFG=experiments/configs/exp_gpt2_small_comparison.yaml
CN=$(basename "$CFG" .yaml)
sub() { # mode n_shards shard_id
  sbatch --parsable --partition=iaifi_gpu_priority --time=12:00:00 --nodes=1 \
    --gres=gpu:1 --cpus-per-task=8 --mem=64G --output=slurm_logs/gpt2-%j.out \
    --wrap="cd $(pwd); .venv/bin/python run.py --config-name $CN mode=$1 +n_shards=$2 +shard_id=$3"
}
echo "== Sven runs (3): one GPU each =="
for i in 0 1 2; do echo "  sven shard $i -> $(sub svd 3 $i)"; done
echo "== baselines (12 runs): 6 shards, ~2 runs each =="
for i in 0 1 2 3 4 5; do echo "  std shard $i -> $(sub standard 6 $i)"; done
