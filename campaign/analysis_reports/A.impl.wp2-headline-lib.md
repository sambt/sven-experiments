**WP2 (library half) — done.** Full test suite green, notebook executed in place with 0 error cells.

### Files created (no existing file touched)
- `/n/home/anon/sven-experiments/analysis/headline.py` (1352 l) — selection→runs matching, confirmation/data-seed tables, paired vs Sven, budget wrappers, efficiency + calibration + time-to-target, ranking summary, md/LaTeX export.
- `/n/home/anon/sven-experiments/analysis/headline_tables.ipynb` (27 cells, executed, 0 errors, outputs kept).
- `/n/home/anon/sven-experiments/tests/test_headline.py` (665 l, 20 tests, synthetic scan+confirm+timing via `SV3_RESULTS_ROOT`).
- Generated: `analysis/tables/` (138 `.md`/`.tex`, 229 KB), `analysis/plots_v2/headline/{paired_vs_sven,best_of_n}.pdf`.

### Verified by running
- `pytest tests/test_headline.py` → **20 passed**; `pytest tests/` → **966 passed, 24 skipped**; `test_analysis_offline.py` → 33 passed.
- `assert_no_test_selection()` → `85 (scan, method) selections checked`; it raises when a test key is injected (pinned by test).
- `check_selection_reproduces` all 7 scans: `ok=True`, max rel diff **≤2.7e-16**.
- Independent raw-JSONL recompute (plain `json`, no analysis layer) matches: CIFAR-CE Sven `1.42385 ± 0.0125961`, test 1.41151, acc 0.52976, train_eval 0.818639; toy per-data-seed 1.051e-07/1.252e-06/1.684e-07; nanoGPT Sven 2.40388 s/epoch, 615.447 MB, 20.86 ms/step. Cold (`use_cache=False`) read identical to cached.
- **Calibration (p5_timing logs, all 7 jobs present)**: start→end median drift **−3.06 % … +3.63 %** (worst: nanoGPT 0.5974→0.6191 ms), 0 contaminated at the 10 % threshold, loadavg 2.1–9.8. **No host-load contamination — MLP step times are quotable.**

### Headline (confirmation seeds, `finished/attempted` = 5/5 or 15/15 unless noted)
| scan | top-3 (val) | Sven rank | Sven val | test | acc | gap | s/epoch (fastest) |
|---|---|---|---|---|---|---|---|
| Toy 1D | HIG > **Sven** > SOAP | 2/15 | 5.09e-7±1.5e-6 | 5.36e-7 | – | +77 % | 1.31 (SGD 0.51) |
| Polynomial | HIG > **Sven** > JD | 2/15 | 0.1388±0.055 | 0.1360 | – | +27 % | 1.33 (SGDm 0.53) |
| MNIST label-reg | MuonW > HIG > **Sven** | 3/14 | 0.05328±7e-4 | 0.05363 | 96.7 % | +2.2 % | 3.74 (SGD 1.33) |
| MNIST CE | MuonW > Muon > HIG | 4/14 | 0.1134±0.0051 | 0.1128 | 96.6 % | −1.4 % | 3.72 (SGD 1.28) |
| CIFAR label-reg | MuonW > SOAP > Muon | 9/11 | 0.4838±0.011 | 0.4938 | **69.4 %** | +0.7 % | 62.9 (SGD 2.75) |
| CIFAR CE | MuonW > L-BFGS > SGD | 9/11 | 1.424±0.013 | 1.412 | **53.0 %** | +1.5 % | 63.0 (SGD 2.70) |
| nanoGPT | AdamW > **Sven** > MuonW | 2/5 (3/5 test) | 1.715±0.008 | 1.977 | – | −0.5 % | 2.40 (AdamW 0.91) |

Full per-scan tables: `analysis/tables/confirmation_<scan>.md`.

### Deviations
1. `tabulate` is absent and the venv has no `pip`: `headline.to_markdown` falls back to its own renderer (pinned by test). **All WPs using `df.to_markdown()` will hit this.**
2. `paired.py` refuses test metrics by design, so paired differences on `final_test_*` live in `headline.paired_outcome_vs_sven` (same ddof/t-interval/sign convention).
3. Own `run_id` join for standalone times (task-permitted) instead of `attach_standalone_times`; `timing_join_report` reproduces its deviation check.
4. Added beyond the plan: `train_eval` (eval-mode train loss), `elig_conf` (eligibility re-asked on fresh seeds), `val_conf_ds0`/`gap_val_same_instance`.
5. Targets pre-declared as median-method confirmation val loss × {2, 1, 0.5}; 0.5× is unreached by everyone on nanoGPT (reported as `n_never`, not hidden).

### Open issues / notes
- **Correction for WP4b/WP5/paper**: CIFAR-CE Sven is **9th/11 on confirmation** val (10th/11 on tuning — SGDm degrades to 1.431±0.494 on fresh seeds). Plan §4 says 10th/11.
- **Correction**: CIFAR-CE standalone epoch times are Sven 63.0 s vs baselines **2.70–12.38 s** (decision 3 says "5–18 s").
- Decision-3 numbers confirmed exactly (69.4/73.4/76.3/77.9; train_eval 0.0674; CE 53.0, train_eval 0.819).
- Polynomial **L-BFGS is not eligible on confirmation** (2/15, 13 diverged) → flagged; its paired interval is ±2.6. MNIST SOAP is eligible but unstable (label-reg 0.581±0.438).
- Budget: Sven's grid is 180 pts → **115 distinct** trajectories (toy, 0.64), 160→106 (MNIST-CE, 0.66), but **1.00 on both CIFAR** (truncation always binds).
- Orchestrator: `analysis/tables/` (138 files) is **not** gitignored — commit as paper artefacts or add to `analysis/.gitignore`. `headline.HEADLINE_SCANS` duplicates `tools/select_best.HEADLINE_SCANS`. I use WP1's new `style.method_label` via `getattr` fallback, so ordering of WP landings is safe.