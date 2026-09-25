#!/bin/bash
#SBATCH --partition=lab_gpu_priority,lab_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=probe_campaign_a100
#SBATCH --output=/n/labstore01/LABS/anon_lab/Users/anon/sv3_campaign_scratch/probe_results/slurm/probe_job1-%j.out
# JOB 1: full A100-80GB. Parts (a) environment truth, (b) CIFAR capture x empty_cache x
# allocator, (c) ResNet co-tenancy, (d) MLP sharding plateau, (e) nanoGPT co-tenancy.
# Ordered most-valuable-first; every measurement is written as one JSON line the moment
# it finishes, so a wall-clock timeout still leaves data.
set -u
SCRATCH=/n/labstore01/LABS/anon_lab/Users/anon/sv3_campaign_scratch
export PROBE_SNAPSHOT=$SCRATCH/probe_snapshot
RESULTS=$SCRATCH/probe_results
cd "$PROBE_SNAPSHOT"
echo "node $(hostname)  job $SLURM_JOB_ID  start $(date -Is)"
cat "$PROBE_SNAPSHOT/PROBE_SNAPSHOT_HEADS.txt"
# 1:45:00 minus a 7-minute margin so the driver stops cleanly instead of being killed
BUDGET=$(( $(date +%s) + 5880 ))
/n/home/anon/sven-experiments/.venv/bin/python \
  "$PROBE_SNAPSHOT/bench/probe_campaign/probe_driver.py" \
  --parts a,b,c,d,e --results "$RESULTS" --snapshot "$PROBE_SNAPSHOT" \
  --device-class a100_80gb --budget-epoch "$BUDGET"
echo "end $(date -Is)"
/n/home/anon/sven-experiments/.venv/bin/python \
  "$PROBE_SNAPSHOT/bench/probe_campaign/analyse.py" --results "$RESULTS" || true
