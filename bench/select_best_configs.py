"""Write bench/best_configs.json: best (optimizer, setting) per headline scan, chosen
exactly as the notebooks do (scan_analysis best_sven / best_baseline: seed-mean of the
last finite validation loss). Consumed by submit_timing_runs.sh.
    PYTHONPATH=. .venv/bin/python bench/select_best_configs.py [scan ...]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis'))
os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis'))
from scan_analysis import load_scan
SCANS = sys.argv[1:] or ['toy_1d_scan', 'polynomial_scan', 'mnist_scan_labelRegression', 'mnist_scan_ce', 'exp_nanogpt_speedrun']
path = '../bench/best_configs.json'
out = json.load(open(path)) if os.path.exists(path) else {}
def clean(cfg):
    return {k: (None if isinstance(v, float) and np.isnan(v) else (v if isinstance(v, str) else float(v))) for k, v in cfg.items() if k != 'optimizer'}
for name in SCANS:
    s = load_scan(name, name, '/tmp/_plots')
    rows = []
    cfg = s.best_sven(); r = s.sven_rows(**cfg)
    rows.append(['Sven', clean(cfg), float(r['final_val_loss'].mean()), int(len(r)), float(r['total_time'].mean())])
    for opt in s.baselines:
        cfg = s.best_baseline(opt)
        if cfg is None: continue
        r = s.baseline_rows(cfg)
        rows.append([opt, clean(cfg), float(r['final_val_loss'].mean()), int(len(r)), float(r['total_time'].mean())])
    out[name] = rows
    print(name, '->', [(m, c) for m, c, *_ in rows])
json.dump(out, open(path, 'w'), indent=1)
