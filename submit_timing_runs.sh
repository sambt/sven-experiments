#!/bin/bash
# Standalone (NPROC=1) timing runs of the best (optimizer, setting) per headline scan.
# Selection = analysis.scan_analysis best_sven()/best_baseline() (seed-mean final val
# loss), written to bench/best_configs.json by bench/select_best_configs.py.
# Usage: ./submit_timing_runs.sh <scan> [<scan> ...]      DRY=1 to print only
#        EXCLUSIVE=1 ...   -> sbatch --exclusive: whole node, so no co-tenant job (ours or
#                             anyone's) can slow a launch-bound run; use for the final timing pass.
set -u; set -f
cd /n/home/anon/sven-experiments
JSON=bench/best_configs.json
for scan in "$@"; do
  cfg=experiments/configs/${scan}_timing.yaml
  [ -f "$cfg" ] || { echo "missing $cfg"; exit 1; }
  echo "== $scan -> experiment_results/${scan}_timing/ =="
  .venv/bin/python - "$scan" "$JSON" <<'PY' | while read -r line; do
import json, sys, math
scan, path = sys.argv[1], sys.argv[2]
for method, cfg, *_ in json.load(open(path))[scan]:
    def num(v):
        f = float(v); return str(int(f)) if f == int(f) and abs(f) >= 1 else f"{f:g}"
    if method == 'Sven':
        print(f"svd_k{num(cfg['k'])}_lr{cfg['lr']:g}_rtol{cfg['rtol']:g}|mode=svd k_values=[{num(cfg['k'])}] lrs=[{cfg['lr']:g}] rtol=[{cfg['rtol']:g}]")
    elif method == 'LBFGS':
        print(f"LBFGS_lr{cfg['lr']:g}_mi{num(cfg['lbfgs_max_iter'])}_hs{num(cfg['lbfgs_history_size'])}|mode=standard optimizers_standard=[LBFGS] lrs_lbfgs=[{cfg['lr']:g}] lbfgs_max_iter=[{num(cfg['lbfgs_max_iter'])}] lbfgs_history_size=[{num(cfg['lbfgs_history_size'])}]")
    elif method == 'PolyakSGD':
        print(f"PolyakSGD|mode=standard optimizers_standard=[PolyakSGD]")
    elif method == 'HIG':
        print(f"HIG_lr{cfg['lr']:g}_tau{cfg['tau']:g}|mode=hig lrs_hig=[{cfg['lr']:g}] tau_hig=[{cfg['tau']:g}]")
    elif method.startswith('JD_'):
        agg = method[3:]; inner = cfg.get('inner_optimizer') or 'Adam'
        print(f"JD_{agg}_lr{cfg['lr']:g}|mode=jd lrs_jd=[{cfg['lr']:g}] aggregators_jd=[{agg}] inner_optimizers_jd=[{inner}]")
    elif method in ('AdamW', 'Muon', 'MuonW') and cfg.get('weight_decay') is not None:
        wd = float(cfg['weight_decay'])   # AdamW's run_id always carries its wd (default 0.01)
        print(f"{method}_lr{cfg['lr']:g}_wd{wd:g}|mode=standard optimizers_standard=[{method}] lrs_standard=[{cfg['lr']:g}] weight_decays=[{wd:g}]")
    else:
        print(f"{method}_lr{cfg['lr']:g}|mode=standard optimizers_standard=[{method}] lrs_standard=[{cfg['lr']:g}]")
PY
    tag=${line%%|*}; ov=${line#*|}
    if [ "${DRY:-0}" = 1 ]; then echo "  [dry] NPROC=1 ${scan}_timing $ov"; continue; fi
    extra=""; [ "${EXCLUSIVE:-0}" = 1 ] && extra="--exclusive"
    jid=$(NPROC=1 sbatch --parsable $extra --job-name="${scan}_timing:$tag" submit_rebuttal_parallel.sh "$cfg" $ov)
    echo "  $jid  $tag  ($ov)"
  done
done
