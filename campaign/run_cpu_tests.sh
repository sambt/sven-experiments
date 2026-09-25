#!/bin/bash
# Run a command (usually pytest) as a short CPU SLURM job and print its log.
# Use this for anything heavy (> ~2 GB RAM or > ~2 min): the interactive node that hosts the
# development session has 4 cores / 16 GB shared by every agent.
#
# Usage:  campaign/run_cpu_tests.sh [-C dir] <command...>
#   campaign/run_cpu_tests.sh .venv/bin/python -m pytest tests/ -x -q
#   campaign/run_cpu_tests.sh -C sven ../.venv/bin/python -m pytest tests/test_torch_gram.py -q
set -u
REPO=/n/home/anon/sven-experiments
DIR=$REPO
if [ "${1:-}" = "-C" ]; then DIR=$(cd "$REPO" && cd "$2" && pwd); shift 2; fi
mkdir -p "$REPO/slurm_logs/devtests"
LOG=$(mktemp "$REPO/slurm_logs/devtests/devtest-XXXXXX.out")
CMD=$(printf '%q ' "$@")
rc=1
for part in test shared test shared serial_requeue; do
  # `test` cannot be combined with other partitions and allows only 5 submitted jobs per user,
  # so alternate: test (starts in seconds) -> shared -> ...
  if sbatch --wait --quiet -p $part -c 4 --mem=12G -t 00:40:00 -J devtest -o "$LOG" \
       --wrap "cd $DIR && export OMP_NUM_THREADS=4 PYTHONPATH=$REPO:\${PYTHONPATH:-} && $CMD"; then
    rc=0; break
  else
    rc=$?
    # the wrapped command failing also lands here: if the job ran, the log is non-empty -> stop
    if [ -s "$LOG" ]; then break; fi
    sleep 5
  fi
done
cat "$LOG"
echo "[run_cpu_tests] exit=$rc log=$LOG"
exit $rc
