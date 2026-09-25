## TRACK analysis-offline — final report

### Files created
- `/n/home11/sambt/iaifi/sv3/analysis/repair_legacy.py` (C-A1), `analysis/budget.py` (C-A4), `analysis/paired.py` (C-A6), `tests/test_analysis_offline.py`, `analysis/legacy_repair/{8 scans}.parquet + corrections.csv + .gitignore` (outputs; `_cache/` ignored).

### Files changed
- `analysis/profile_helpers.py` — `cycle_mean()`, `_timing()` (stored `cycle_mean` else **recomputed from `raw`**), `step_ms/wall_ms/capture_ms/solve_ms` switched to it, new reference column **`step_ms_steady`**, `_CACHE_VERSION = 2` in the cache signature, `STATUS_NOTE` (adds `nonfinite`) replacing three status-keyed dicts that would `KeyError`, `plot_raw_steps` draws both means, x-label "Amortised step time".
- `experiments/optimizer_profile.py` — `CYCLE = 10`, `cycle_mean()`, `summarize()` gains `cycle_mean`/`cycle` (keeps `steady_mean`); after the measured steps and after `result["memory"]`, `status = "nonfinite"` if any parameter is non-finite (timings still recorded).
- `make_plots.sh` — uses `$REPO/.venv/bin/jupyter` with a preflight check + fix hint.
- `pyproject.toml` + `uv.lock` — dev group gains `nbconvert>=7.17.1` (via `uv add --dev --no-sync`; lock diff is additions only, torch untouched).

### Acceptance tests (observed: `32 passed, 1 warning in 4.14s` with `SV3_CHECK_REAL_SCANS=1`; `27 passed, 5 skipped` by default)
- **C-T2 real numbers**: `methods__SOAP__B64__kf1.json` → `step_ms == 6.14 ± 0.005`, `step_ms_steady == 5.56 ± 0.005`; `gram_hooks` → `10.40 ± 0.05` (both); and `step_ms == cycle_mean(raw.step_ms)`. Full root: 720 configs, 0 NaN, median +0.27 %, p95 +15 %, max +64 %.
- `cycle_mean` = last-80 % / whole-cycle mean, phase-independent; the runner's copy (ast-extracted, torch-free) is bit-identical and its `summarize` exports it; `nonfinite` is placed after `result["memory"]` and before `except _Infeasible`; cache version invalidates a stale frame; `nonfinite` renders in `method_table`/`plot_method_bars`/`plot_heatmap`.
- **C-A1**: `batch_weights`, `n_val_window` (batch-size sweep pins `n_val` to 9993–10000), `repair_curve` (10/4 → 2.2 vs recorded 3.0), synthetic `repair_scan` (new columns, nothing written to the results root), the equal-weight assertion fires on a mismatch, a bad `n_val` becomes an error row, `resolve_n_val` precedence; real `cifar10_resnet_ce_scan` ≈ 0.17/0.71/1.1 %.
- **C-A4**: `best_of_n` vs brute force for all n, with/without replacement, minimize/maximize; `trajectory_table` on synthetic duplicates; real scans reproduce Fable **exactly**: 12.0/18, 11.35/18, 22.40/32, 16.0/32 (effective 48 / 45.4 / 89.6 / 64 vs 4).
- **C-A6**: mean −3, std √2.5, half-width 2.7764·√0.5, unpaired seed dropped, n=1 → NaN std, `SEED_SPREAD_LABEL` says "std over seaeds"→`± 1 std over seeds` and never "confidence"; both modules reject a test metric.
- **nbconvert**: recorded in `[dependency-groups]`, importable, `.venv/bin/jupyter nbconvert --version` = 7.17.1; `make_plots.sh` has no bare `jupyter`.

### C-A1 corrections of the final validation loss (rel. %, `eq_check_max = 0.0` — bit-exact reproduction of every stored curve, 0 errors, `n_val = 10000` everywhere)
| scan | runs | median | p95 | max | Fable |
|---|---|---|---|---|---|
| toy_1d_scan | 734 | 0.064 | 0.139 | 0.98 | .07/.14/**.34** |
| polynomial_scan | 731 | 0.097 | 0.331 | 1.70 | .10/.38/1.7 |
| mnist_scan_ce | 1040 | 0.416 | 0.480 | 0.78 | .41/.48/.79 |
| mnist_scan_labelRegression | 999 | 0.278 | 0.457 | 0.60 | .25/.45/.58 |
| cifar10_resnet_ce_scan | 290 | 0.170 | 0.707 | 1.08 | .17/.71/1.1 |
| cifar10_resnet_scan_labelRegression | 290 | 0.105 | 1.73 | **24.3** | — |
| rebuttal_batchsize_polynomial_scan | 2496 | 0.185 | 2.06 | 8.5 | .15/2.3/9.0 |
| rebuttal_overparam_polynomial_scan | 2160 | 0.195 | 2.52 | 24.6 | .19/3.3/**33** |

Tail maxima differ from Fable's (toy 0.98 vs 0.34, overparam 24.6 vs 33): I score every run with a finite stored and repaired final value (`n_scored` column), Fable's subset is unstated.

### Deviations from CONTRACTS.md
1. `uv sync --inexact --active` was **not** run: its dry run wanted to uninstall+reinstall the editable `sven`, which would break other agents mid-test. Used the documented fallback `uv pip install --python .venv/bin/python nbconvert==7.17.1` (17 pure additions, same versions as the lock). `uv.lock` is correct, the venv matches it.
2. Edited 8 docstring lines in `experiments/optimizer_profile.py` outside `summarize()`/the status check — they describe exactly those two changes ("use the steady-state mean" was now wrong).
3. `repair_legacy` points `style.load_results(cache_dir=…)` at `analysis/legacy_repair/_cache` so no byte is written under the results root (cold loads instead).

### For the integrator
- New API: `profile_helpers.cycle_mean(values, cycle=10)`, `CYCLE`, `STATUS_NOTE`, `_CACHE_VERSION`, column `step_ms_steady`; `optimizer_profile.cycle_mean(values, cycle=CYCLE)`, `summarize` keys `cycle_mean`/`cycle`. The two `cycle_mean` bodies are **duplicated on purpose** (analysis must stay torch-free); `test_profiler_and_analysis_cycle_mean_agree` guards drift.
- `repair_legacy`: `repair_scan(name, results_root=None, n_val=None, workers=16, check=5, rtol=2e-5, cache_dir=None, verbose=True) -> DataFrame`, `repair_run`, `repair_curve`, `batch_weights`, `n_batches`, `n_val_window`, `resolve_n_val`, `resolve_root`, `correction_summary`, `write`, `load_repair`, `LEGACY_N_VAL`, `HEADLINE`, `DEFAULT_SCANS`, `OUT_DIR`, CLI `main`. Join on `run_id`; curve column `val_ew`, scalar `final_val_loss_ew`.
- `budget`: `trajectory_table(df, metric, seed_col, within=('lr',), sig=None)`, `best_of_n(values, n, minimize, replace)`, `best_of_n_curve(df, …, diverged='worst'|'drop'|float)`, `plot_best_of_n(curves, ax)`.
- `paired`: `SEED_SPREAD_LABEL`, `interval_label(level, n)`, `paired_runs`, `paired_difference(...) -> dict`, `paired_table(df, rows, reference)`, `fmt_difference`, `plot_paired(tbl, ax)`. Rows are `analysis_helpers.config_table`/`best_per_method` rows.
- **Call sites I do not own that must change**: `analysis/profile_overview.ipynb` has two markdown cells saying step times are the "**steady-state mean** … first 20 % and any >3 MAD spike are dropped" and "steady-state step time" — now false; they must say the amortised/cycle mean (and the `x Adam` column moved, e.g. SOAP 1.37 → 1.45). A notebook cell for `budget`/`paired` and the legacy-repair table is still to be added (I did not execute or edit notebooks). Anyone adding `status: "nonfinite"` handling elsewhere should reuse `profile_helpers.STATUS_NOTE`.
- Unfixed, not mine: `n_val` is absent from every pre-2026-09-17 record, so C-R4 must keep writing it; `LEGACY_N_VAL` covers only the eight scans I ran — any other legacy scan needs `n_val=` or a new entry.