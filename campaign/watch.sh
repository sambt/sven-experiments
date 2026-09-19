#!/bin/bash
# Block until something needs attention, then print why and exit (the orchestrating session is re-invoked on exit).
# Reasons: fewer than 2 gpu_test jobs alive, a pool log alarm, oom/error markers or failed attempts, or a heartbeat.
R=/n/holystore01/LABS/iaifi_lab/Users/sambt/sven_experiments
S=/n/holystore01/LABS/iaifi_lab/Users/sambt/sv3_campaign_scratch/logs/campaign
HEARTBEAT=${1:-7200}; t0=$(date +%s); base_err=${2:-0}
while true; do
  sleep 300
  mig=$(squeue -u $USER -h -p gpu_test -o %i | wc -l)
  [ "$mig" -lt 2 ] && { echo "REASON: only $mig gpu_test job(s) alive"; break; }
  err=$(ls $R/*/done 2>/dev/null | grep -c -E "\.(oom|error)$"); att=$(ls $R/*/attempts 2>/dev/null | grep -c .)
  [ $((err+att)) -gt "$base_err" ] && { echo "REASON: oom/error markers=$err attempts=$att (baseline $base_err)"; break; }
  al=$(grep -l -E "failed runner process\(es\): [1-9]|\[poisoned\]" $S/*2c6faf59-*.out 2>/dev/null | wc -l)
  [ "$al" -gt 0 ] && { echo "REASON: $al pool log(s) with alarms"; break; }
  [ $(( $(date +%s) - t0 )) -ge "$HEARTBEAT" ] && { echo "REASON: heartbeat"; break; }
done
/n/home11/sambt/iaifi/sv3/campaign/monitor.sh
