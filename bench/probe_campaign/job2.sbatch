#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=probe_campaign_mig
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Users/sambt/sv3_campaign_scratch/probe_results/slurm/probe_job2-%j.out
# JOB 2: one a100_3g.20gb MIG slice (gpu_test starts within seconds, 2 jobs per user).
# CIFAR Sven chunked 0.25 on the slice, then parts (d) and (e) -> the A100-equivalence
# factor of a MIG slice per workload class.
# gpu_test rejects >= 8 cores per GPU, and part (d) needs 12 single-threaded processes,
# so the job holds 2 slices (12/2 = 6 cores each) and pins every process to slice 0 --
# the measurement is still "N processes sharing ONE 3g.20gb slice".
set -u
export CUDA_VISIBLE_DEVICES=0
SCRATCH=/n/holystore01/LABS/iaifi_lab/Users/sambt/sv3_campaign_scratch
export PROBE_SNAPSHOT=$SCRATCH/probe_snapshot
RESULTS=$SCRATCH/probe_results   # shared with JOB 1; records carry device_class=mig_3g20gb
cd "$PROBE_SNAPSHOT"
echo "node $(hostname)  job $SLURM_JOB_ID  start $(date -Is)"
cat "$PROBE_SNAPSHOT/PROBE_SNAPSHOT_HEADS.txt"
BUDGET=$(( $(date +%s) + 3240 ))
/n/home11/sambt/iaifi/sv3/.venv/bin/python \
  "$PROBE_SNAPSHOT/bench/probe_campaign/probe_driver.py" \
  --parts a,cifarslice,d,e --results "$RESULTS" --snapshot "$PROBE_SNAPSHOT" \
  --device-class mig_3g20gb --budget-epoch "$BUDGET"
echo "end $(date -Is)"
/n/home11/sambt/iaifi/sv3/.venv/bin/python \
  "$PROBE_SNAPSHOT/bench/probe_campaign/analyse.py" --results "$RESULTS" || true
