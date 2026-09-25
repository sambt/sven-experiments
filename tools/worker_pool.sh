#!/bin/bash
# One SLURM job (or one bare machine) working a campaign WORK LIST from the claim queue.
#
#   tools/worker_pool.sh <snapshot> <work-items-file> [--label NAME] [--dry-run]
#                        [--until <epoch>|+<seconds>]
#
# For every visible GPU it starts a POOL. Each pool walks the work items IN ORDER; for
# item `config | overrides | NPROC` it launches NPROC runner processes pinned to its own
# GPU, waits for all of them, then moves to the next item. Because the runner takes its
# work from the claim queue (`scheduler=claims`), every pool of every job may work the
# same item at the same time, and an item whose runs are all done returns in seconds --
# so the list is a priority order, not an assignment.
#
# Work-items file (one item per line; `#` comments and blank lines ignored):
#
#     # config_name            | hydra overrides                     | NPROC
#     toy_1d_scan              | mode=svd                            | 12
#     toy_1d_scan              | mode=standard optimizers_standard=[Adam,SGD] | 12
#
# WALL CLOCK: a pool never STARTS an item within WORKER_RESERVE_S (default 900 s) of the
# job's end, and exits 0 instead. The deadline comes from --until, WORKER_DEADLINE_EPOCH,
# WORKER_TIME_BUDGET (seconds from now) or SLURM's own $SLURM_JOB_END_TIME. That is what
# gives a chained job control back BEFORE slurm's SIGTERM: a batch shell whose foreground
# child is SIGTERMed never runs the line after it (bash defers the signal and then exits
# 128+15), so a self-renewing job that relies on the wall clock alone can never renew.
#
# Environment (EXPERIMENTS.md section 12 "Stage 1 contracts / Execution"): PYTHONPATH into the
# snapshot, SV3_RESULTS_ROOT, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
# OMP_NUM_THREADS=1. Overridable from outside: SV3_RESULTS_ROOT, WORKER_PY,
# WORKER_LOG_ROOT, WORKER_SCHEDULER_OVERRIDE (set empty to drop `scheduler=claims`),
# WORKER_RESERVE_S / WORKER_DEADLINE_EPOCH / WORKER_TIME_BUDGET (the wall clock above).
#
# Exit status: 0 when every runner process exited 0 -- including the case where every
# item was already finished (an empty queue is success, not failure) and the case where
# the pools wound down at the wall clock. Non-zero only if a runner process crashed.
# SIGTERM/SIGINT are forwarded to every runner and stop the pools from starting further
# items (SLURM sends SIGTERM at the wall clock).
set -u
set -f          # hydra list overrides contain [...]: never let bash glob them

usage() { sed -n '2,37p' "$0"; }
die() { echo "[pool] ERROR: $*" >&2; exit 2; }

SNAP=""; ITEMS=""; LABEL=""; DRY=0; UNTIL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL=${2:?--label needs a value}; shift 2 ;;
    --until) UNTIL=${2:?--until needs an epoch or +seconds}; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option '$1'" ;;
    *) if [ -z "$SNAP" ]; then SNAP=$1; elif [ -z "$ITEMS" ]; then ITEMS=$1;
       else die "unexpected argument '$1'"; fi; shift ;;
  esac
done
[ -n "$SNAP" ] && [ -n "$ITEMS" ] || { usage; exit 2; }

SNAP=$(cd "$SNAP" 2>/dev/null && pwd) || die "snapshot '$SNAP' not found"
[ -f "$ITEMS" ] || die "work-items file '$ITEMS' not found"
ITEMS=$(cd "$(dirname "$ITEMS")" && pwd)/$(basename "$ITEMS")
[ -f "$SNAP/run.py" ] || die "no run.py in the snapshot $SNAP"
[ -e "$SNAP/.deploy_complete" ] || echo "[pool] WARNING: $SNAP has no .deploy_complete marker (incomplete export?)"

PY=${WORKER_PY:-${SV3_REPO:-$HOME/sven-experiments}/.venv/bin/python}
[ -x "$PY" ] || die "python '$PY' is not executable (set WORKER_PY)"
# `++`, not `=`: NO scan config declares `scheduler` (the default lives in
# grid.resolve_scan_settings and tests/test_configs.py forbids the key in a config), so
# hydra's struct mode rejects a plain `scheduler=claims` with "Could not override
# 'scheduler' ... not in struct" and every runner process of every campaign job exits 1
# before it trains anything. `++key=value` overrides the key if it is there and appends
# it if it is not, so it is correct either way.
SCHED=${WORKER_SCHEDULER_OVERRIDE-++scheduler=claims}
JOBID=${SLURM_JOB_ID:-local.$$}
[ -n "$LABEL" ] || LABEL=$(basename "$ITEMS" .txt)
# Logs go to the scratch tree ($SV3_SCRATCH), NOT to $HOME: one log per (job x pool x
# item x NPROC) plus a hydra dir each is O(10^4) files for the full plan.
# WORKER_LOG_ROOT still overrides it (the tests point it at a temp dir).
LOG_ROOT=${WORKER_LOG_ROOT:-${SV3_SCRATCH:-$HOME/scratch/sven}/campaign/logs}
LOGDIR=$LOG_ROOT/$LABEL.$JOBID

# --- the wall clock (see the header): the epoch after which no NEW item is started ----
# Every value is checked BEFORE arithmetic (`set -u` makes $(( x + abc )) an unbound
# variable error, which would bury the reason). An explicit value that is not a number is
# operator error and fatal; a $SLURM_JOB_END_TIME that is not an epoch (older slurm
# exports a formatted date) only costs the wind-down and must never fail a real job.
is_uint() { case "${1:-}" in ''|*[!0-9]*) return 1 ;; *) return 0 ;; esac; }
RESERVE_S=${WORKER_RESERVE_S:-900}
is_uint "$RESERVE_S" || die "WORKER_RESERVE_S must be an integer, got '$RESERVE_S'"
DEADLINE=0
if [ -n "$UNTIL" ]; then
  case "$UNTIL" in
    +*) is_uint "${UNTIL#+}" || die "--until +<seconds> must be a number, got '$UNTIL'"
        DEADLINE=$(( $(date +%s) + ${UNTIL#+} )) ;;
    *)  is_uint "$UNTIL" || die "--until must be a unix epoch or +<seconds>, got '$UNTIL'"
        DEADLINE=$UNTIL ;;
  esac
elif [ -n "${WORKER_DEADLINE_EPOCH:-}" ]; then
  is_uint "$WORKER_DEADLINE_EPOCH" \
    || die "WORKER_DEADLINE_EPOCH must be a unix epoch, got '$WORKER_DEADLINE_EPOCH'"
  DEADLINE=$WORKER_DEADLINE_EPOCH
elif [ -n "${WORKER_TIME_BUDGET:-}" ]; then
  is_uint "$WORKER_TIME_BUDGET" \
    || die "WORKER_TIME_BUDGET must be a number of seconds, got '$WORKER_TIME_BUDGET'"
  DEADLINE=$(( $(date +%s) + WORKER_TIME_BUDGET ))
elif [ -n "${SLURM_JOB_END_TIME:-}" ]; then
  if is_uint "$SLURM_JOB_END_TIME"; then
    DEADLINE=$SLURM_JOB_END_TIME
  else
    echo "[pool] WARNING: \$SLURM_JOB_END_TIME='$SLURM_JOB_END_TIME' is not a unix epoch;"
    echo "[pool]          no wall-clock wind-down (pass --until <epoch> to get one)"
  fi
fi
STOP_AT=0
[ "$DEADLINE" -gt 0 ] && STOP_AT=$((DEADLINE - RESERVE_S))

# ---------------------------------------------------------------------------
# Environment: the snapshot, and nothing but the snapshot
# ---------------------------------------------------------------------------
# An inherited PYTHONPATH is dropped on purpose -- it is how the live working tree
# would sneak back in. `sven` is editable-installed from the live tree, but its
# finder is APPENDED to sys.meta_path (after the path finder), so PYTHONPATH wins;
# the assertion below is what proves it, per job, instead of trusting that.
export PYTHONPATH=$SNAP:$SNAP/sven
export SV3_RESULTS_ROOT=${SV3_RESULTS_ROOT:-$SNAP/experiment_results}
# Both spellings: this torch prints "PYTORCH_CUDA_ALLOC_CONF is deprecated, use
# PYTORCH_ALLOC_CONF instead" for the old name (it still honours it -- the Gate-1 CIFAR
# Sven run measured 200 ms/step, the fast expandable-segments path, not the 841 ms of a
# default allocator with per-step empty_cache). Exporting the new name too means the
# setting the probe measured survives the version that stops honouring the old one:
# `expandable_segments` is what halves peak reserved memory and what makes CIFAR Sven
# fit a 19.6 GB MIG slice at all (EXPERIMENTS.md section 1.5).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-$PYTORCH_CUDA_ALLOC_CONF}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export SV3_SNAPSHOT=$SNAP
# the snapshot's identity is the point of the snapshot: no job may write __pycache__
# into the shared, immutable tree it imports from.
export PYTHONDONTWRITEBYTECODE=1
cd "$SNAP" || die "cannot cd into the snapshot"

echo "[pool] snapshot     $SNAP"
echo "[pool] items        $ITEMS"
echo "[pool] results root $SV3_RESULTS_ROOT"
echo "[pool] host         $(hostname)   job $JOBID   logs $LOGDIR"
if [ "$STOP_AT" -gt 0 ]; then
  echo "[pool] wall clock   no new item after $(date -Is -d "@$STOP_AT") (job end $(date -Is -d "@$DEADLINE"), reserve ${RESERVE_S}s)"
else
  echo "[pool] wall clock   none (no --until / \$SLURM_JOB_END_TIME): items run until the queue drains or SIGTERM"
fi
[ -f "$SNAP/DEPLOY_INFO.json" ] && sed -e 's/^/[pool] deploy /' "$SNAP/DEPLOY_INFO.json" | head -6

# ---------------------------------------------------------------------------
# Assertion: python must resolve BOTH packages inside the snapshot
# ---------------------------------------------------------------------------
# `find_spec` is used rather than a real import so this costs milliseconds instead of
# a cold `import torch`; it walks exactly the finders/sys.path the runner will walk
# (cwd = the snapshot here, script dir = the snapshot there, so sys.path[0] matches).
"$PY" - <<'PYEOF' || die "the snapshot guard failed -- refusing to run"
import importlib.util, os, sys
snap = os.path.realpath(os.environ["SV3_SNAPSHOT"])
bad = []
for name in ("experiments", "sven"):
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:                       # a broken install must not look OK
        print(f"[pool] {name}: find_spec raised {exc!r}")
        bad.append(name)
        continue
    origin = os.path.realpath(spec.origin) if (spec and spec.origin) else None
    print(f"[pool] resolves    {name:12s}= {origin}")
    if origin is None or not origin.startswith(snap + os.sep):
        bad.append(name)
if bad:
    print(f"[pool] ABORT: {', '.join(bad)} resolve OUTSIDE the snapshot {snap}")
    sys.exit(1)
PYEOF

# ---------------------------------------------------------------------------
# Work items
# ---------------------------------------------------------------------------
ITEM_CFG=(); ITEM_OVR=(); ITEM_NP=()
lineno=0
while IFS= read -r raw || [ -n "$raw" ]; do
  lineno=$((lineno+1))
  line=${raw%%#*}                                     # strip comments
  line=$(printf '%s' "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
  [ -z "$line" ] && continue
  IFS='|' read -r c o n <<<"$line"
  c=$(printf '%s' "${c:-}" | tr -d '[:space:]')
  o=$(printf '%s' "${o:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
  n=$(printf '%s' "${n:-}" | tr -d '[:space:]')
  [ -n "$c" ] || die "$ITEMS:$lineno: empty config name in '$raw'"
  case "$n" in ''|*[!0-9]*) die "$ITEMS:$lineno: NPROC must be a positive integer, got '$n'";; esac
  [ "$n" -ge 1 ] || die "$ITEMS:$lineno: NPROC must be >= 1, got '$n'"
  ITEM_CFG+=("$c"); ITEM_OVR+=("$o"); ITEM_NP+=("$n")
done < "$ITEMS"
NITEMS=${#ITEM_CFG[@]}
[ "$NITEMS" -gt 0 ] || die "$ITEMS holds no work items"

# The items file's own header records the snapshot it was written FOR (launch_campaign.py
# stamps it). Print it, and shout if it is not the snapshot we are running: the same
# run_id under two snapshots has two run hashes, so the C-R3 rule would have the two jobs
# retire each other's finished results to _stale/ and re-run them.
ITEMS_SNAP=$(sed -n 's/^# snapshot //p' "$ITEMS" | head -1)
if [ -n "$ITEMS_SNAP" ]; then
  echo "[pool] items for   $ITEMS_SNAP"
  sed -n -e 's/^# sv3  /[pool] items sv3  /p' -e 's/^# sven /[pool] items sven /p' "$ITEMS"
  if [ "$ITEMS_SNAP" != "$SNAP" ]; then
    echo "[pool] WARNING: this items file was written for ANOTHER snapshot ($ITEMS_SNAP)."
    echo "[pool]          Two snapshots on one work list means every run has two run"
    echo "[pool]          hashes and each job retires the other's results to _stale/."
  fi
fi

echo "[pool] $NITEMS work item(s):"
for i in $(seq 0 $((NITEMS-1))); do
  printf '[pool]   %2d  NPROC=%-3s %-42s %s\n' "$i" "${ITEM_NP[$i]}" "${ITEM_CFG[$i]}" "${ITEM_OVR[$i]}"
done

# ---------------------------------------------------------------------------
# Visible GPUs -> one pool each (MIG UUIDs included)
# ---------------------------------------------------------------------------
DEVICES=()
if command -v nvidia-smi >/dev/null 2>&1; then
  while IFS= read -r l; do
    case "$l" in
      *MIG*UUID:*) DEVICES+=("$(printf '%s' "$l" | sed -E 's/.*UUID: *([^)]*)\).*/\1/')") ;;
    esac
  done < <(nvidia-smi -L 2>/dev/null)
  if [ ${#DEVICES[@]} -eq 0 ]; then                   # not a MIG node: whole GPUs
    while IFS= read -r l; do
      case "$l" in
        GPU*UUID:*) DEVICES+=("$(printf '%s' "$l" | sed -E 's/.*UUID: *([^)]*)\).*/\1/')") ;;
      esac
    done < <(nvidia-smi -L 2>/dev/null)
  fi
fi
# CUDA_VISIBLE_DEVICES is a filter on that list: numeric entries index it (what SLURM
# exports), `GPU-`/`MIG-` entries select by UUID. UUIDs are what we then re-export, so
# a pool can never be aliased onto another pool's device.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  SEL=()
  IFS=',' read -r -a want <<<"$CUDA_VISIBLE_DEVICES"
  for w in "${want[@]}"; do
    w=$(printf '%s' "$w" | tr -d '[:space:]')
    [ -z "$w" ] && continue
    case "$w" in
      ''|*[!0-9]*) SEL+=("$w") ;;                      # a UUID (or a name we pass through)
      *) if [ "$w" -lt ${#DEVICES[@]} ] 2>/dev/null; then SEL+=("${DEVICES[$w]}"); else SEL+=("$w"); fi ;;
    esac
  done
  DEVICES=("${SEL[@]}")
fi
if [ ${#DEVICES[@]} -eq 0 ]; then
  echo "[pool] WARNING: no GPU visible -- running ONE pool with the inherited device setting"
  DEVICES=("")
fi
echo "[pool] ${#DEVICES[@]} pool(s): ${DEVICES[*]}"

if [ "$DRY" = 1 ]; then
  echo "[pool] --dry-run: the commands that WOULD run"
  for d in $(seq 0 $((${#DEVICES[@]}-1))); do
    for i in $(seq 0 $((NITEMS-1))); do
      hdir=${WORKER_HYDRA_RUN_DIR-$LOGDIR/hydra/<tag>}
      printf '[pool]   gpu%s x%s  CUDA_VISIBLE_DEVICES=%s %s %s/run.py --config-name %s %s %s %s\n' \
        "$d" "${ITEM_NP[$i]}" "${DEVICES[$d]}" "$PY" "$SNAP" "${ITEM_CFG[$i]}" "$SCHED" \
        "${hdir:+hydra.run.dir=$hdir}" "${ITEM_OVR[$i]}"
    done
  done
  exit 0
fi

mkdir -p "$LOGDIR" || die "cannot create $LOGDIR"
RUNDIR=$LOGDIR/.run; mkdir -p "$RUNDIR"
STOPFILE=$RUNDIR/STOP; PIDFILE=$RUNDIR/pids
: > "$PIDFILE"
rm -f "$STOPFILE"

TERMED=0
on_term() {
  TERMED=1
  echo "[pool] caught SIGTERM/SIGINT -- no further items; forwarding to running runners"
  : > "$STOPFILE"
  while IFS= read -r p; do [ -n "$p" ] && kill -TERM "$p" 2>/dev/null; done < "$PIDFILE"
  for p in "${POOLS[@]:-}"; do [ -n "$p" ] && kill -TERM "$p" 2>/dev/null; done
}
trap on_term TERM INT

run_pool() {                       # run_pool <slot> <device>
  local slot=$1 dev=$2 i p np cfg ovr log rc fails=0 t0 t1
  for i in $(seq 0 $((NITEMS-1))); do
    if [ -e "$STOPFILE" ]; then
      echo "[pool$slot] stopping before item $i (stop requested)"
      break
    fi
    if [ "$STOP_AT" -gt 0 ] && [ "$(date +%s)" -ge "$STOP_AT" ]; then
      # A graceful wind-down, NOT a failure: the caller (a --chain job) gets control
      # back with exit 0 while it still has time to reconcile and resubmit.
      echo "[pool$slot] stopping before item $i: within ${RESERVE_S}s of the wall clock"
      : > "$RUNDIR/winddown"
      break
    fi
    cfg=${ITEM_CFG[$i]}; ovr=${ITEM_OVR[$i]}; np=${ITEM_NP[$i]}
    local pids=()
    t0=$SECONDS
    for p in $(seq 0 $((np-1))); do
      tag=$(printf 'item%02d' "$i")_${cfg}_gpu${slot}_p${p}
      log=$LOGDIR/$tag.log
      # Keep hydra's own per-run directory OUT of the snapshot: the snapshot is shared
      # by every job of the campaign and its identity is the point, so nothing may write
      # into it. `WORKER_HYDRA_RUN_DIR=` (empty) restores hydra's default.
      hydra_dir=${WORKER_HYDRA_RUN_DIR-$LOGDIR/hydra/$tag}
      hydra_arg=""
      [ -n "$hydra_dir" ] && hydra_arg="hydra.run.dir=$hydra_dir"
      (
        [ -n "$dev" ] && export CUDA_VISIBLE_DEVICES="$dev"
        echo "[runner] $(date -Is) host=$(hostname) dev=${CUDA_VISIBLE_DEVICES:-unset} item=$i proc=$p"
        echo "[runner] $PY $SNAP/run.py --config-name $cfg $SCHED $hydra_arg $ovr"
        exec "$PY" "$SNAP/run.py" --config-name "$cfg" $SCHED $hydra_arg $ovr
      ) > "$log" 2>&1 &
      pids+=($!)
      echo "$!" >> "$PIDFILE"
    done
    for p in "${pids[@]}"; do
      wait "$p"; rc=$?                 # capture BEFORE any other command runs
      if [ "$rc" -ne 0 ]; then
        fails=$((fails+1))
        echo "[pool$slot] item $i ($cfg): a runner exited $rc -- see $LOGDIR"
      fi
    done
    t1=$((SECONDS-t0))
    echo "[pool$slot] item $i done in ${t1}s: $cfg $ovr (NPROC=$np, failures so far $fails)"
  done
  echo "$fails" > "$RUNDIR/fails.gpu$slot"
}

POOLS=()
for d in $(seq 0 $((${#DEVICES[@]}-1))); do
  run_pool "$d" "${DEVICES[$d]}" &
  POOLS+=($!)
done

for p in "${POOLS[@]}"; do
  while :; do
    wait "$p" 2>/dev/null; wrc=$?
    # a trapped signal makes `wait` return >128 while the pool is still alive
    if [ "$wrc" -gt 128 ] && kill -0 "$p" 2>/dev/null; then continue; fi
    break
  done
done

FAILS=0
# `set -f` is on (hydra list overrides), so iterate the slots rather than globbing.
for d in $(seq 0 $((${#DEVICES[@]}-1))); do
  f=$RUNDIR/fails.gpu$d
  [ -s "$f" ] || continue
  FAILS=$((FAILS + $(cat "$f")))
done
echo "[pool] finished: ${#DEVICES[@]} pool(s), $NITEMS item(s), $FAILS failed runner process(es)"
if [ -e "$RUNDIR/winddown" ]; then
  echo "[pool] wound down at the wall clock with items left: relaunch (or let the chain renew) to continue"
fi
if [ "$TERMED" = 1 ]; then
  echo "[pool] terminated by signal"
  exit 143
fi
[ "$FAILS" -eq 0 ] || exit 1
exit 0
