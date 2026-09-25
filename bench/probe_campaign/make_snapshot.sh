#!/bin/bash
# Freeze BOTH repos at HEAD into the probe snapshot, then copy the probe scripts in.
# ~10 agents are editing the working tree while the probe jobs run, so the jobs must not
# import anything from /n/home/anon/sven-experiments -- probe_run.py enforces that at runtime.
#   bash bench/probe_campaign/make_snapshot.sh
set -euo pipefail
SV3=/n/home/anon/sven-experiments
SCRATCH=/n/labstore01/LABS/anon_lab/Users/anon/sv3_campaign_scratch
SNAP=$SCRATCH/probe_snapshot
RESULTS=$SCRATCH/probe_results

rm -rf "$SNAP"
mkdir -p "$SNAP" "$RESULTS/slurm"
git -C "$SV3" archive --format=tar HEAD | tar -x -C "$SNAP"
mkdir -p "$SNAP/sven"
git -C "$SV3/sven" archive --format=tar HEAD | tar -x -C "$SNAP/sven"
{ echo "sv3 HEAD  $(git -C "$SV3" rev-parse HEAD)"
  echo "sven HEAD $(git -C "$SV3/sven" rev-parse HEAD)"
  date -Is; } > "$SNAP/PROBE_SNAPSHOT_HEADS.txt"

# The probe scripts themselves are untracked, so git archive does not carry them.
mkdir -p "$SNAP/bench/probe_campaign"
cp "$SV3"/bench/probe_campaign/*.py "$SV3"/bench/probe_campaign/*.sbatch \
   "$SV3"/bench/probe_campaign/*.sh "$SNAP/bench/probe_campaign/"

cat "$SNAP/PROBE_SNAPSHOT_HEADS.txt"
echo "snapshot: $SNAP  ($(du -sh "$SNAP" | cut -f1))"
echo "results:  $RESULTS"
