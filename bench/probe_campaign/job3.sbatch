#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=probe_campaign_mig_b
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Users/sambt/sv3_campaign_scratch/probe_results/slurm/probe_job3-%j.out
# JOB 3: part (b) -- CIFAR capture x empty_cache x PYTORCH_CUDA_ALLOC_CONF -- on a MIG
# 3g.20gb slice, because all 32 iaifi A100-80GB are allocated and JOB 1 is stuck behind
# the QOS node cap.  Two questions this slice CAN answer today:
#   * the empty_cache (C-T3) and expandable_segments effects on step time, which are
#     properties of the code and the allocator, not of the GPU's size;
#   * whether CIFAR Sven fits a 20 GB slice at all -- chunked 0.25 already reserves
#     18.5 GB of 20 GB here, so the allocator setting decides it. Full capture (23 GB
#     allocated) must OOM; recording that is the point.
# Only 1 process at a time, so this cannot disturb JOB 2 on another slice of the node.
set -u
SCRATCH=/n/holystore01/LABS/iaifi_lab/Users/sambt/sv3_campaign_scratch
export PROBE_SNAPSHOT=$SCRATCH/probe_snapshot
RESULTS=$SCRATCH/probe_results
cd "$PROBE_SNAPSHOT"
echo "node $(hostname)  job $SLURM_JOB_ID  start $(date -Is)"
BUDGET=$(( $(date +%s) + 2100 ))
/n/home11/sambt/iaifi/sv3/.venv/bin/python \
  "$PROBE_SNAPSHOT/bench/probe_campaign/probe_driver.py" \
  --parts b --results "$RESULTS" --snapshot "$PROBE_SNAPSHOT" \
  --device-class mig_3g20gb --budget-epoch "$BUDGET"
echo "end $(date -Is)"
