#!/bin/bash
# Compact campaign health check: per-scan done-marker statuses, live claims, failed attempts, pool-log alarms.
R=/n/labstore01/LABS/anon_lab/Users/anon/sven_experiments
S=/n/labstore01/LABS/anon_lab/Users/anon/sv3_campaign_scratch/logs
date
printf "%-42s %6s %6s %5s %5s %6s %5s\n" scan ok div oom err claim att
for d in $R/*/; do n=$(basename $d); [ -d $d/done ] || continue
  l=$(ls $d/done); c() { echo "$l" | grep -c "\.$1\$"; }
  printf "%-42s %6s %6s %5s %5s %6s %5s\n" $n $(c ok) $(c diverged) $(c oom) $(c error) $(ls $d/claims 2>/dev/null | wc -l) $(ls $d/attempts 2>/dev/null | wc -l)
done
echo "--- jobs:"; squeue -u $USER -h -o "%T %P" | sort | uniq -c
echo "--- alarms in pool logs (this snapshot):"
grep -l -E "failed runner process\(es\): [1-9]|\[poisoned\]|Traceback" $S/campaign/*$(basename $(cat /n/home/anon/sven-experiments/campaign/CURRENT_SNAPSHOT) | cut -c1-8)-*.out 2>/dev/null | head
grep -h -E "^\[pool\].*(failed|crash)" $S/campaign/*$(basename $(cat /n/home/anon/sven-experiments/campaign/CURRENT_SNAPSHOT) | cut -c1-8)-*.out 2>/dev/null | grep -v " 0 failed" | tail -5
