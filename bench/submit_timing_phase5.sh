#!/bin/bash
# Submit the Phase-5 timing pass: ONE job per scan, every selected method of that scan run
# serially on one GPU (NPROC 1). DRY RUN BY DEFAULT -- it prints the jobs and the exact
# sbatch commands and submits nothing until you pass --submit.
#
#   bench/submit_timing_phase5.sh                        # dry run, all seven scans
#   bench/submit_timing_phase5.sh mnist_scan_ce          # dry run, one scan
#   bench/submit_timing_phase5.sh --submit               # submit
#   bench/submit_timing_phase5.sh --submit --exclusive   # ... on whole nodes
#
# Options / environment:
#   --submit            actually sbatch (default: print only)
#   --exclusive         add `--exclusive`: no co-tenant of any kind on the node. Costs a
#                       whole node slot of the QOS and cannot start while anyone else
#                       holds a GPU on the node (the 2026-09-17 exclusive timing job sat
#                       in QOSMaxNodePerUserLimit for 13 h), so it is opt-in. Without it,
#                       the calibration lines in the log are what make the pass auditable.
#   --scan NAME         repeatable; same as a positional argument
#   --only REGEX        run only methods matching this (passed through as $ONLY)
#   SNAP=...            deploy snapshot (default $SV3_DEPLOY_SNAPSHOT, else the newest
#                       complete one under the deploy base)
#   JSON=...            selection file (default <repo>/bench/best_configs.json)
#   ROOT=...            SV3_RESULTS_ROOT (default: the selection file's results_root)
#   TIME=...            wall clock per job (default 36:00:00)
#
# Prerequisite: `tools/select_best.py` has written a schema-2 bench/best_configs.json from
# a RECONCILED results root, and the snapshot carries the `<scan>_timing` configs. Both are
# checked below and a failure is fatal, because a timing pass against the wrong selection
# produces numbers for a configuration nobody plots.
set -u
set -f

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-$REPO/.venv/bin/python}
JSON=${JSON:-$REPO/bench/best_configs.json}
DEPLOY_BASE=${DEPLOY_BASE:-${SV3_SCRATCH:-$HOME/scratch/sven}/deploy}
LOG_DIR=${LOG_DIR:-${SV3_SCRATCH:-$HOME/scratch/sven}/campaign/logs/phase5}
TIME=${TIME:-36:00:00}
SUBMIT=0; EXCLUSIVE=${EXCLUSIVE:-0}; ONLY=${ONLY:-}; SCANS=""

die() { echo "[timing] ERROR: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    --exclusive) EXCLUSIVE=1; shift ;;
    --scan) SCANS="$SCANS ${2:?--scan needs a name}"; shift 2 ;;
    --only) ONLY=${2:?--only needs a regex}; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    -*) die "unknown option '$1'" ;;
    *) SCANS="$SCANS $1"; shift ;;
  esac
done

[ -f "$JSON" ] || die "no selection file $JSON -- run tools/select_best.py first"
[ -x "$PY" ] || die "python '$PY' is not executable (set PY)"

# --- the snapshot (CONTRACTS.md: campaign processes never run the live tree) -----------
SNAP=${SNAP:-${SV3_DEPLOY_SNAPSHOT:-}}
if [ -z "$SNAP" ]; then
  # `set +f` inside the subshell only: globbing is off in this script (hydra list
  # overrides contain `[...]`), and with it off `$DEPLOY_BASE/*/` stays a literal star
  # and every snapshot looks missing.
  SNAP=$(set +f; ls -1dt "$DEPLOY_BASE"/*/ 2>/dev/null | while read -r d; do
           [ -e "$d/.deploy_complete" ] && { echo "${d%/}"; break; }; done)
fi
[ -n "$SNAP" ] || die "no deploy snapshot found under $DEPLOY_BASE (run tools/deploy_snapshot.sh, or set SNAP)"
[ -e "$SNAP/.deploy_complete" ] || die "$SNAP is not a complete snapshot (no .deploy_complete)"
[ -f "$SNAP/run.py" ] || die "$SNAP has no run.py"
[ -f "$SNAP/bench/calibrate_step.py" ] \
  || die "$SNAP has no bench/calibrate_step.py -- commit bench/ and re-run tools/deploy_snapshot.sh"

# --- what to time ---------------------------------------------------------------------
read -r ROOT_FROM_JSON SCANS_FROM_JSON <<EOF
$("$PY" - "$JSON" <<'PYEOF'
import json, sys
p = json.load(open(sys.argv[1]))
if p.get("schema") != 2:
    sys.exit("not a schema-2 selection file (run tools/select_best.py)")
print(p.get("results_root") or "-", " ".join(sorted(p["scans"])))
PYEOF
)
EOF
[ -n "${SCANS_FROM_JSON:-}" ] || die "cannot read $JSON"
[ -n "$(echo "$SCANS" | tr -d ' ')" ] || SCANS=$SCANS_FROM_JSON
ROOT=${ROOT:-$ROOT_FROM_JSON}
[ "$ROOT" != "-" ] || die "no results root: set ROOT (the selection file carries none)"
# the selection may have resolved the root through the repo's experiment_results symlink;
# the workers must get the real path, not one under the 95 G NFS home
ROOT=$(cd "$ROOT" 2>/dev/null && pwd -P) || die "results root $ROOT does not exist"

echo "[timing] snapshot  $SNAP"
echo "[timing] selection $JSON"
echo "[timing] results   $ROOT"
echo "[timing] scans     $SCANS"
echo "[timing] mode      $([ "$SUBMIT" = 1 ] && echo SUBMIT || echo 'dry run (nothing is submitted)')$([ "$EXCLUSIVE" = 1 ] && echo '  +exclusive')"

# Report the selection each job will time, and refuse a selection that is not ready: a
# method whose overrides did not verify, or a scan whose grid is still unfinished (a
# configuration with 3 of 5 seeds is already `eligible`, so a half-finished grid extension
# can win outright).
"$PY" - "$JSON" $SCANS <<'PYEOF'
import json, sys
p = json.load(open(sys.argv[1]))
bad = False
for scan in sys.argv[2:]:
    e = p["scans"].get(scan)
    if e is None:
        print(f"[timing]   {scan}: NOT IN THE SELECTION"); bad = True; continue
    ms = e["methods"]
    secs = sum((s.get("mean_wall_time_s") or 0) * len(s["model_seeds"]) for s in ms.values())
    print(f"[timing]   {scan:38s} {len(ms):2d} method(s), "
          f"{sum(len(s['model_seeds']) for s in ms.values()):3d} run(s), "
          f"~{secs / 3600:5.1f} h serial (at the scans' own sharded step times)")
    for m, s in sorted(ms.items()):
        if not s.get("verified"):
            print(f"[timing]     {m}: UNVERIFIED overrides ({s.get('verify_error')})"); bad = True
        if s["n_ok"] < s["n_expected"]:
            print(f"[timing]     {m}: selected on {s['n_ok']}/{s['n_expected']} seeds"); bad = True
    if e.get("n_missing"):
        print(f"[timing]     {e['n_missing']} run(s) of this grid are unfinished"); bad = True
if bad:
    print("[timing] re-run tools/select_best.py once `tools/reconcile.py --all "
          "campaign/plan_campaign.yaml` is clean; ALLOW_INCOMPLETE=1 overrides")
sys.exit(1 if bad else 0)
PYEOF
rc=$?
if [ "$rc" -ne 0 ]; then
  [ "${ALLOW_INCOMPLETE:-0}" = 1 ] \
    || die "the selection is not ready to time (see above); ALLOW_INCOMPLETE=1 overrides"
  echo "[timing] ALLOW_INCOMPLETE=1: timing an incomplete selection anyway"
fi

mkdir -p "$LOG_DIR"
extra=""; [ "$EXCLUSIVE" = 1 ] && extra="--exclusive"
for scan in $SCANS; do
  [ -f "$SNAP/experiments/configs/${scan}_timing.yaml" ] \
    || die "$SNAP has no experiments/configs/${scan}_timing.yaml (re-deploy the snapshot)"
  set -- sbatch --parsable $extra -t "$TIME" \
      --job-name="p5_timing.$scan" \
      --output="$LOG_DIR/timing-$scan-%j.out" \
      --export="ALL,SCAN=$scan,SNAP=$SNAP,JSON=$JSON,ROOT=$ROOT,PY=$PY,ONLY=$ONLY" \
      "$SNAP/bench/timing_phase5.sbatch"
  if [ "$SUBMIT" = 1 ]; then
    jid=$("$@") || die "sbatch failed for $scan"
    echo "[timing] $jid  $scan"
  else
    echo "[timing] \$ $*"
  fi
done
[ "$SUBMIT" = 1 ] || echo "[timing] re-run with --submit to submit"
