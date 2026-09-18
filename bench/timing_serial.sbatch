#!/bin/bash
#SBATCH --partition=iaifi_gpu_priority,iaifi_gpu,gpu
#SBATCH --exclusive
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --job-name=timing_serial_exclusive  # overridden per group below
#SBATCH --output=slurm_logs/timing_serial-%j.out
# ONE exclusively reserved node, ONE GPU, every (scan, optimizer, best-setting) timing
# configuration run back to back (all seeds inside each run). No co-tenant of any
# kind, so wall times are clean; run_id dedup makes it resumable.
# Configs come from bench/best_configs.json (bench/select_best_configs.py); the
# tag|override lines are produced by the same rule as submit_timing_runs.sh.
set -u; set -f
cd /n/home11/sambt/iaifi/sv3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python; JSON=bench/best_configs.json
SCANS=${SCANS:-"toy_1d_scan polynomial_scan mnist_scan_labelRegression mnist_scan_ce exp_nanogpt_speedrun"}
# GROUP=long runs only configs whose estimated serial time (sharded per-run time x seeds)
# is >= LONG_H hours; GROUP=short the rest; GROUP=all (default) everything. Two nodes,
# one long job + one short job, finish in about the time of the longest config set.
GROUP=${GROUP:-all}; LONG_H=${LONG_H:-1.0}
echo "group=$GROUP (threshold ${LONG_H} h)"
# RESELECT=1: regenerate bench/best_configs.json from the current scans first (the
# selection rule lives in analysis/scan_analysis.py; RERUNS_NEEDED item 2). Runs on the
# reserved node before the timing loop so timed and plotted configs cannot drift apart.
if [ "${RESELECT:-0}" = 1 ]; then
  echo "== reselecting best configs ($(date +%T)) =="
  PYTHONPATH=. $PY bench/select_best_configs.py $SCANS || { echo "reselect FAILED"; exit 1; }
fi
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; echo "node $(hostname), cpus $SLURM_CPUS_ON_NODE, start $(date)"
for scan in $SCANS; do
  echo "################ $scan ($(date +%T)) ################"
  $PY - "$scan" "$JSON" "$GROUP" "$LONG_H" <<'PYEOF' | while read -r line; do
import json, sys
scan, path, group, long_h = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
def num(v):
    f = float(v); return str(int(f)) if f == int(f) and abs(f) >= 1 else f"{f:g}"
for method, cfg, val, n_seeds, t_sharded in json.load(open(path))[scan]:
    est_h = t_sharded * n_seeds / 3600
    if group == 'long' and est_h < long_h: continue
    if group == 'short' and est_h >= long_h: continue
    if method == 'Sven':
        print(f"svd|mode=svd k_values=[{num(cfg['k'])}] lrs=[{cfg['lr']:g}] rtol=[{cfg['rtol']:g}]")
    elif method == 'LBFGS':
        print(f"LBFGS|mode=standard optimizers_standard=[LBFGS] lrs_lbfgs=[{cfg['lr']:g}] lbfgs_max_iter=[{num(cfg['lbfgs_max_iter'])}] lbfgs_history_size=[{num(cfg['lbfgs_history_size'])}]")
    elif method == 'PolyakSGD':
        print("PolyakSGD|mode=standard optimizers_standard=[PolyakSGD]")
    elif method == 'HIG':
        print(f"HIG|mode=hig lrs_hig=[{cfg['lr']:g}] tau_hig=[{cfg['tau']:g}]")
    elif method.startswith('JD_'):
        print(f"{method}|mode=jd lrs_jd=[{cfg['lr']:g}] aggregators_jd=[{method[3:]}] inner_optimizers_jd=[{cfg.get('inner_optimizer') or 'Adam'}]")
    elif method in ('AdamW', 'Muon', 'MuonW') and cfg.get('weight_decay') is not None:
        wd = float(cfg['weight_decay'])
        print(f"{method}|mode=standard optimizers_standard=[{method}] lrs_standard=[{cfg['lr']:g}] weight_decays=[{wd:g}]")
    else:
        print(f"{method}|mode=standard optimizers_standard=[{method}] lrs_standard=[{cfg['lr']:g}]")
PYEOF
    tag=${line%%|*}; ov=${line#*|}
    echo "=== $scan / $tag  ($(date +%T)) ==="
    $PY run.py --config-name ${scan}_timing print_config=false $ov 2>&1 | grep -E "^SVD:|^Standard:|^LBFGS:|^PolyakSGD:|^HIG:|^JD:|\[error\]|\[skip\]|Traceback|Error|Scan complete"
  done
done
echo "################ all done $(date) ################"
