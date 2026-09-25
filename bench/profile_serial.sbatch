#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --job-name=optimizer_profile
#SBATCH --output=slurm_logs/profile/profile-%j.out
#
# Memory / step-time profile: every architecture serially on ONE exclusively reserved node
# (a co-tenant job skews launch-bound step times by up to 2x -- gpu.probe.md measured the
# same MNIST-Adam step at 1.13 ms on a quiet node and 4.58 ms on a node at 4/4 GPUs).
# Resumable: a configuration whose JSON already exists is skipped, so a job that hits the
# wall clock is continued by submitting it again.
#
#   sbatch bench/profile_serial.sbatch                             # all 7 configs -> v3
#   PROFILE_CONFIGS="profile_mnist profile_cifar" sbatch ...        # a subset
#   SV3_PROFILE_ROOT=/path/to/tmp_root PROFILE_SMOKE=1 sbatch ...   # smoke, 3 steps
#
# Environment (every one has a default, so the script also runs by hand):
#   SNAP               deploy snapshot to run from (default $SV3_DEPLOY_SNAPSHOT, else the
#                      newest complete one under $DEPLOY_BASE, else the live repo with a
#                      warning). CONTRACTS.md: campaign processes never import the live
#                      tree -- ~6 agents edit it while a 2 h job runs.
#   SV3_PROFILE_ROOT   absolute results root (default: `profile.output_dir` from the
#                      configs = profile_results_v3). REQUIRED when running from a
#                      snapshot, else a relative root would write into the frozen export.
#   PROFILE_CONFIGS    which configs to profile
#   PROFILE_SMOKE=1    preset of PROFILE_ARGS that walks every configuration in a few
#                      steps each: proves the code path and the 720-way expansion without
#                      measuring anything (~16 min for all of them on a MIG slice).
#   PROFILE_ARGS       extra hydra overrides, appended after the smoke preset
#   CALIB_STEPS        bench/calibrate_step.py steps at start and end (0 = skip)
#
# WHY v3 EXISTS. The 2026-09-17 profile (profile_results_v2) got TWO settings wrong: Sven's
# per-step torch.cuda.empty_cache(), which made full-capture Sven up to 4.5x slower on
# CIFAR (841 -> 187 ms/step) and was the sole source of its step-time variance, and the
# missing expandable_segments allocator. Both are now as the campaign ran them:
# `empty_cache` is passed explicitly False (experiments.optimizer_profile.EMPTY_CACHE) and
# the allocator setting below matches tools/worker_pool.sh. The BatchNorm policy is NOT a
# difference -- every v2 CIFAR Gram record has meta.freeze_norm_stats false, i.e. batch
# statistics, like the scans; `bn_mode_of` only stops the post-C-E2 profiler default from
# freezing them. v2 is FROZEN as the before-table -- never write there.
#
# COMPLETENESS. After a clean full pass this script writes `_profile_complete` into the
# results root; analysis/profile_helpers.profile_status also accepts a full per-config
# count, which is what certifies the 2026-09-20 pass (job 47396284 runs from deploy
# snapshot 118156ae, which predates the sentinel).
#
# IF THIS JOB WILL NOT START (`QOSMaxNodePerUserLimit`, `Resources`): --exclusive holds a
# whole node slot of the QOS and cannot start while anyone holds a GPU on the node (the
# 2026-09-17 exclusive timing job waited 13 h). The fallback is ONE non-exclusive GPU with
# the calibration microbenchmark bracketing the pass, which this script already records:
#   scancel <job>; sbatch --oversubscribe -c 8 bench/profile_serial.sbatch
# and then believe the numbers only if the two `[calib]` lines agree to within a few
# percent (they are what says whether the machine changed underneath the pass).
set -u
set -f                      # hydra list overrides contain [...]: never let bash glob them

REPO=${SV3_REPO:-$HOME/sven-experiments}
DEPLOY_BASE=${DEPLOY_BASE:-${SV3_SCRATCH:-$HOME/scratch/sven}/deploy}
PY=${PY:-$REPO/.venv/bin/python}
CONFIGS=${PROFILE_CONFIGS:-"profile_toy_1d profile_polynomial profile_mnist profile_mnist_width profile_nanogpt profile_nanogpt_width profile_cifar"}
CALIB_STEPS=${CALIB_STEPS:-1000}

# --- the snapshot (or, explicitly, the live tree) --------------------------------------
# SNAP=live is the only way to get the working tree: the default picks the newest complete
# snapshot, and silently profiling code that is one commit behind is exactly the mistake
# this whole re-run exists to correct.
SNAP=${SNAP:-${SV3_DEPLOY_SNAPSHOT:-}}
if [ -z "$SNAP" ]; then
  SNAP=$(set +f; ls -1dt "$DEPLOY_BASE"/*/ 2>/dev/null | while read -r d; do
           [ -e "$d/.deploy_complete" ] && { echo "${d%/}"; break; }; done)
fi
if [ "$SNAP" = live ] || [ "$SNAP" = "$REPO" ]; then
  echo "[profile] WARNING: SNAP=$SNAP -- running the LIVE working tree $REPO, which other"
  echo "[profile] WARNING: agents may edit mid-job. Fine for a smoke, never for v3."
  CODE=$REPO
elif [ -n "$SNAP" ] && [ -e "$SNAP/.deploy_complete" ]; then
  CODE=$SNAP
else
  echo "[profile] ERROR: '$SNAP' is not a complete deploy snapshot. Run"
  echo "[profile] ERROR:   tools/deploy_snapshot.sh   (or pass SNAP=live deliberately)"
  exit 2
fi
[ -f "$CODE/experiments/optimizer_profile.py" ] || { echo "[profile] ERROR: $CODE has no experiments/optimizer_profile.py"; exit 2; }

# --- environment: the snapshot and nothing but the snapshot ----------------------------
# PYTHONPATH exactly as tools/worker_pool.sh sets it: the editable `sven` install points at
# the live tree and its finder is APPENDED to sys.meta_path, so PYTHONPATH wins.
export PYTHONPATH="$CODE:$CODE/sven"
# both spellings: this torch deprecates PYTORCH_CUDA_ALLOC_CONF in favour of PYTORCH_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-$PYTORCH_CUDA_ALLOC_CONF}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
# ABSOLUTE, and in the REPO, not in $CODE or the cwd: `profile.output_dir` in the configs is
# relative, and this job runs with cwd = a scratch workdir and code = a frozen snapshot, so
# leaving it relative would scatter the results next to neither. analysis/profile_helpers
# resolves ROOT_V3 to exactly this path.
export SV3_PROFILE_ROOT=${SV3_PROFILE_ROOT:-$REPO/profile_results_v3}

# hydra writes an `outputs/` tree into the cwd: keep it out of the frozen snapshot
WORKDIR=${WORKDIR:-${SV3_SCRATCH:-$HOME/scratch/sven}/campaign/profile}
mkdir -p "$WORKDIR" && cd "$WORKDIR" || exit 2

ARGS=${PROFILE_ARGS:-}
if [ "${PROFILE_SMOKE:-0}" = 1 ]; then
  # A pass over every configuration that measures nothing: 2 warm-up + 3 measured steps on
  # 2 staged batches. It still builds every wrapper/optimizer, so an unsupported
  # combination, an OOM wall or a broken import shows up exactly where the real run
  # would hit it -- for a fraction of the time.
  ARGS="profile.warmup_steps=2 profile.num_steps=3 profile.min_steps=3 profile.max_seconds=10 profile.max_warmup_seconds=30 profile.staged_batches=2 $ARGS"
  echo "[profile] SMOKE: $ARGS"
  case "${SV3_PROFILE_ROOT:-}" in
    ""|*profile_results_v3*) echo "[profile] ERROR: a smoke run must set SV3_PROFILE_ROOT to a TEMP root (not v3)"; exit 2 ;;
  esac
fi

echo "################ optimizer profile ################"
echo "job       ${SLURM_JOB_ID:-none}   $(date -Is)"
echo "host      $(hostname)   cpus_on_node=${SLURM_CPUS_ON_NODE:-?}"
echo "code      $CODE"
echo "results   $SV3_PROFILE_ROOT"
echo "configs   $CONFIGS"
echo "alloc     PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "loadavg   $(cat /proc/loadavg)"
echo "-- other processes on this GPU (should be empty under --exclusive) --"
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null

if [ "$CALIB_STEPS" != 0 ] && [ -f "$CODE/bench/calibrate_step.py" ]; then
  echo "== calibration BEFORE ($(date +%T)) =="
  "$PY" "$CODE/bench/calibrate_step.py" --tag start:profile --steps "$CALIB_STEPS" \
    || echo "[calib] FAILED at start"
fi

rc_all=0
for c in $CONFIGS; do
  echo "################ $c ($(date +%T)) ################"
  "$PY" -m experiments.optimizer_profile --config-name "$c" print_config=false \
      hydra.run.dir="$WORKDIR/hydra/\${now:%Y-%m-%d_%H-%M-%S}_$c" $ARGS 2>&1 \
    | grep -v "it/s\|s/it\|^$"
  rc=${PIPESTATUS[0]}
  echo "    exit $rc"
  [ "$rc" -eq 0 ] || rc_all=1
done

if [ "$CALIB_STEPS" != 0 ] && [ -f "$CODE/bench/calibrate_step.py" ]; then
  echo "== calibration AFTER ($(date +%T)) =="
  "$PY" "$CODE/bench/calibrate_step.py" --tag end:profile --steps "$CALIB_STEPS" \
    || echo "[calib] FAILED at end"
fi
echo "loadavg   $(cat /proc/loadavg)"

# The completeness signal analysis/profile_helpers.profile_status looks for: only after a
# clean pass over the FULL config set, and never for a smoke (whose root is a temp one).
DEFAULT_CONFIGS="profile_toy_1d profile_polynomial profile_mnist profile_mnist_width profile_nanogpt profile_nanogpt_width profile_cifar"
if [ "$rc_all" -eq 0 ] && [ "${PROFILE_SMOKE:-0}" != 1 ] && [ "$CONFIGS" = "$DEFAULT_CONFIGS" ]; then
  n=$(set +f; ls -1 "$SV3_PROFILE_ROOT"/*/*.json 2>/dev/null | wc -l)
  printf '%s\njob %s\nconfigs %s\nn_json %s\n' "$(date -Is)" "${SLURM_JOB_ID:-none}" \
      "$CONFIGS" "$n" > "$SV3_PROFILE_ROOT/_profile_complete"
  echo "wrote     $SV3_PROFILE_ROOT/_profile_complete ($n json)"
fi

echo "################ all done $(date -Is)  rc=$rc_all ################"
echo "Next: the four analysis/notebooks/profiling/profile_*.ipynb notebooks. profile_helpers.results_root()"
echo "      switches to v3 by itself once the pass is COMPLETE (720 of 720, or the"
echo "      _profile_complete file above); a partial root is announced and skipped."
exit $rc_all
