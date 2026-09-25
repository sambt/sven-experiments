#!/bin/bash
# End-to-end smoke check of the refactored runner (C-R2): scan() -> expand_grid ->
# shard -> dedup -> execute, 1 epoch of toy_1d_scan on CPU, all six families.
#
# Covers what tests/test_grid.py cannot (it is torch-free): the families actually
# build and train, every family writes its jsonl + npz, a second pass skips every
# run, and `+n_shards=2 +shard_id=1` runs exactly the odd indices of the grid.
#
#   bash tests/golden/scan_smoke.sh [workdir]
#
# THE REAL RESULTS ROOT IS NEVER TOUCHED: scan() writes to the hard-coded RELATIVE
# path "experiment_results", so every run.py invocation below is `cd`'d into
# $WORK/work, which holds its own empty experiment_results/, and hydra.run.dir is
# pinned under $WORK too. Nothing in the command line resolves into the repo.
set -u
REPO=${SV3_REPO:-$HOME/sven-experiments}
PY=$REPO/.venv/bin/python
WORK=${1:-$(mktemp -d "${TMPDIR:-/tmp}/grid_smoke.XXXXXX")}
export HYDRA_FULL_ERROR=1

mkdir -p "$WORK" || exit 2      # so the guard below sees a real path, never ""
case "$(cd "$WORK" && pwd)" in
  "$REPO"|"$REPO"/*) echo "refusing to run inside the repo: $WORK"; exit 2 ;;
esac
echo "workdir: $WORK"

OV="device=cpu num_epochs=1 model_seeds=[1000] dataset.n_train=256 dataset.n_val=128
    k_values=[2] lrs=[0.1] rtol=[1e-3]
    lrs_standard=[1e-3] optimizers_standard=[Adam,LBFGS,PolyakSGD]
    lrs_lbfgs=[0.5] lbfgs_max_iter=[1] lbfgs_history_size=[2]
    lrs_jd=[1e-3] lrs_hig=[0.1] tau_hig=[1e-4] +svd_spectra_every=1"

run() {  # run <label> <extra overrides...>
  local label=$1; shift
  echo "=================== $label ==================="
  ( cd "$WORK/work" && "$PY" "$REPO/run.py" --config-name toy_1d_scan mode=all $OV "$@" \
      hydra.run.dir="$WORK/hydra/$label" ) 2>&1 | tail -25
  echo "--- files after $label:"
  ls -1 "$WORK/work/experiment_results/toy_1d_scan/" 2>/dev/null
}

rm -rf "$WORK/work" "$WORK/hydra"
mkdir -p "$WORK/work/experiment_results"
run pass1
run pass2                                  # everything must be skipped
rm -rf "$WORK/work" && mkdir -p "$WORK/work/experiment_results"
run shard1of2 +n_shards=2 +shard_id=1      # exactly the odd indices of the grid

echo "=================== record check ==================="
WORK=$WORK "$PY" - <<'EOF'
import glob, json, os
d = os.path.join(os.environ["WORK"], "work/experiment_results/toy_1d_scan")
for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
    r = json.load(open(p))
    print(os.path.basename(p))
    print("   keys:", ",".join(r))
    print("   train:", r["losses"].get("train"), "val:", r["losses"].get("val"),
          "n_params:", r.get("n_params"), "n_train:", r.get("n_train"))
print("npz:", sorted(os.path.basename(x) for x in glob.glob(os.path.join(d, "diag", "*.npz"))))
EOF
echo "=================== the real results root is untouched ==================="
ls -1 $REPO/experiment_results/toy_1d_scan/*.jsonl | wc -l
