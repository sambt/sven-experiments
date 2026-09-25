#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --job-name=p5_timing
#SBATCH --output=slurm_logs/phase5/timing-%j.out
#
# ONE scan's Phase-5 timing pass: every selected (method, configuration) of that scan run
# BACK TO BACK on one GPU, one process at a time (NPROC 1), with nothing else of ours on
# the GPU. Submit it with `bench/submit_timing_phase5.sh` (dry run by default); it is not a
# worker-pool job and must not be launched by `tools/launch_campaign.py`.
#
# WHY SERIAL, AND WHY ALL OF ONE SCAN IN ONE JOB. `EXPERIMENTS.md section 1.5`
# measured an identical MNIST-Adam configuration at 1.13 ms a step on a quiet node and
# 4.58 ms on a node at 4/4 GPUs: a launch-bound MLP step time is set by HOST CPU load, not
# by the GPU, so a claim-queue pool on the OTHER GPUs of the same node corrupts these
# numbers without ever touching this one. A claim queue is the wrong tool here -- there is
# no throughput to win, only isolation to protect. Keeping one scan's methods in one job
# means the whole comparison that gets plotted shares one machine, so a slow node makes
# the bars uniformly slow instead of scrambling their order.
#
# CONTAMINATION IS MEASURED, NOT ASSUMED: `bench/calibrate_step.py` runs at the start and
# again at the end of the job, and both JSON lines go into this log (`[calib] {...}`).
# `--exclusive` (EXCLUSIVE=1 in the submit script) removes the co-tenants outright, at the
# cost of holding a whole node slot of the QOS -- the 2026-09-17 exclusive timing job sat
# in QOSMaxNodePerUserLimit for 13 h -- so it is opt-in, and the calibration is what makes
# the non-exclusive default auditable.
#
# Environment (all set by the submit script; every one of them has a default here so the
# script also runs by hand):
#   SCAN     the parent scan name, e.g. mnist_scan_ce (this job runs ${SCAN}_timing)
#   SNAP     deploy snapshot to run from (EXPERIMENTS.md section 12: never the live tree)
#   JSON     bench/best_configs.json, the SELECTION (schema 2, tools/select_best.py)
#   ROOT     SV3_RESULTS_ROOT
#   PY       python
#   ONLY     optional: run only methods whose name matches this (grep -E)
set -u
set -f                      # hydra list overrides contain [...]: never let bash glob them

SCAN=${SCAN:?SCAN (the parent scan name) must be set}
SNAP=${SNAP:?SNAP (the deploy snapshot) must be set}
JSON=${JSON:?JSON (bench/best_configs.json) must be set}
ROOT=${ROOT:?ROOT (SV3_RESULTS_ROOT) must be set}
PY=${PY:-${SV3_REPO:-$HOME/sven-experiments}/.venv/bin/python}
ONLY=${ONLY:-}
CALIB_STEPS=${CALIB_STEPS:-1000}

# C-T1/C-T3 and EXPERIMENTS.md section 12 "Execution": the snapshot and nothing but the snapshot,
# one thread, expandable segments, and `empty_cache` left at its new default of false (the
# per-step empty_cache made CIFAR Sven 4.5x slower, 841 -> 187 ms/step, and was the sole
# source of its step-time variance).
export PYTHONPATH="$SNAP:$SNAP/sven"
export SV3_RESULTS_ROOT="$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# hydra writes an `outputs/` tree into the cwd: keep it in the scratch tree
WORKDIR=${WORKDIR:-${SV3_SCRATCH:-$HOME/scratch/sven}/campaign/phase5_timing}
mkdir -p "$WORKDIR" && cd "$WORKDIR" || exit 2

echo "################ phase-5 timing: $SCAN ################"
echo "job       ${SLURM_JOB_ID:-none}   $(date -Is)"
echo "host      $(hostname)"
echo "exclusive ${SLURM_JOB_NAME:-?}    cpus_on_node=${SLURM_CPUS_ON_NODE:-?}"
echo "snapshot  $SNAP"
echo "results   $ROOT"
echo "selection $JSON"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "loadavg   $(cat /proc/loadavg)"
# every process on this node's GPUs, ours or anyone's: the co-tenants the calibration is
# there to explain
echo "-- other processes on this GPU (should be empty) --"
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null

echo "== calibration BEFORE ($(date +%T)) =="
"$PY" "$SNAP/bench/calibrate_step.py" --tag "start:$SCAN" --steps "$CALIB_STEPS" \
  || echo "[calib] FAILED at start"

# The selection file is the single source of truth for what gets timed; the overrides are
# read out of it verbatim (tools/select_best.py has already proven each one expands to
# exactly the five seed run_ids of the configuration it names). `++`, not `=`: no scan
# config declares `svd_info`, and the MNIST ones declare `checkpoints_svd` but not
# `checkpoints`, so a plain `=` override is refused by hydra's struct mode. None of the
# three enters `grid.run_hash`, so a timing run keeps the scan's run_id AND its run hash.
TIMING_OVERRIDES="++svd_info=none ++checkpoints=none ++checkpoints_svd=null"

PLAN=$("$PY" - "$JSON" "$SCAN" <<'PYEOF'
import json, sys
path, scan = sys.argv[1], sys.argv[2]
payload = json.load(open(path))
if payload.get("schema") != 2:
    sys.exit(f"{path} is not a schema-2 selection file (run tools/select_best.py)")
entry = payload["scans"].get(scan)
if entry is None:
    sys.exit(f"{scan} is not in {path} (have: {', '.join(sorted(payload['scans']))})")
rows = sorted(entry["methods"].values(),
              key=lambda s: -(s.get("mean_wall_time_s") or 0.0))   # slowest first
for sel in rows:
    if not sel.get("verified"):
        print(f"SKIP|{sel['method']}|unverified overrides: {sel.get('verify_error')}")
        continue
    print(f"RUN|{sel['method']}|{sel['overrides']}")
PYEOF
) || { echo "cannot read the selection: $PLAN"; exit 2; }

# the loop runs in a subshell (it is the right-hand side of a pipe), so it keeps no state
# for afterwards: the per-method `exit` lines in this log are the record, and
# bench/check_timing_join.py is what decides whether the pass is complete and consistent.
echo "$PLAN" | while IFS='|' read -r kind method rest; do
  [ -n "${kind:-}" ] || continue
  if [ "$kind" = SKIP ]; then
    echo "=== $SCAN / $method: SKIPPED ($rest) ==="; continue
  fi
  if [ -n "$ONLY" ] && ! printf '%s' "$method" | grep -Eq "$ONLY"; then continue; fi
  echo "=== $SCAN / $method  ($(date +%T)) ==="
  echo "    overrides: $TIMING_OVERRIDES $rest"
  t0=$(date +%s)
  "$PY" "$SNAP/run.py" --config-name "${SCAN}_timing" print_config=false \
      $TIMING_OVERRIDES $rest
  rc=$?
  echo "    exit $rc after $(( $(date +%s) - t0 ))s"
  [ "$rc" -eq 0 ] || echo "    [error] $SCAN / $method exited $rc"
done

echo "== calibration AFTER ($(date +%T)) =="
"$PY" "$SNAP/bench/calibrate_step.py" --tag "end:$SCAN" --steps "$CALIB_STEPS" \
  || echo "[calib] FAILED at end"
echo "loadavg   $(cat /proc/loadavg)"
echo "################ done: $SCAN  $(date -Is) ################"
echo "Next: $PY <repo>/bench/check_timing_join.py $SCAN   (same run_id => same final val loss)"
