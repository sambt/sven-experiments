All findings verified and fixed; everything re-run green.

# WP1 fixer — all 4 review findings confirmed and fixed

## Files changed (all WP1-owned)
`analysis/style.py`, `analysis/analysis_helpers.py`, `analysis/scan_analysis.py`, `tests/test_analysis_core.py`; minimal edits in `analysis/{cifar,finetune,critbatch,nanogpt}_analysis.ipynb`; 14 notebooks re-executed in place. Scratch (gitignored): `agent_lab/wp1/{exec_guarded.sh,nb_snapshot.txt}`. `make_plots.sh` untouched. Nothing committed; results read-only apart from `_cache/`.

## F1 (high) — CONFIRMED, wired
`SEED_SPREAD_LABEL` really was unused by WP1 code: all four `fill_between` calls and the `clipped_yerr` bars were unlabelled. Added `style.seed_spread_label(plain=False)` (single source: lazy-imports `paired.SEED_SPREAD_LABEL`; a top-level import is a cycle `scan_analysis→paired→analysis_helpers→scan_analysis`, so `paired.py` — possibly WP2's — stays untouched) and `style.band_legend(ax)`: one idempotent grey proxy patch, no data, verified not to move limits on linear or log axes. Called once per axes, **last in the legend**, by `plot_best_curves`, `plot_k_sweep`, `plot_sensitivity`, `plot_knob_curves`, `plot_time_summary`, `plot_time_vs_k` and `analysis_helpers.errorbar_seeds`. Real scan (`mnist_scan_ce`): `plot_best_curves legend: ['Sven (η=0.5, k=32, rtol=0.3)', …, '$\pm$ 1 std over seeds']`; `time_summary legend: ['$\pm$ 1 std over seeds']`. Hard-coded literals replaced by `seed_spread_label(plain=True)` → printed headers now read `mean ± 1 std over seeds` (cifar c4, finetune c7); nanogpt c5 is a *markdown* heading (no constant possible) — wording harmonised only.

## F2 (medium) — CONFIRMED, fixed
Reproduced: legacy `mnist_scan_ce` frame → `expected_run_ids = 1610` for 1040 runs, `missing_configs = 122 configs / 610 runs` (fresh grid points). `load_results` now stamps `_results_root` on every frame (also on the cache-hit path, so pickles stay compatible both ways; registered in `PROVENANCE_COLUMNS`, never a config column); `style.frame_results_root(df, scan)` reads it back per scan; `expected_run_ids` prefers explicit root → frame root → process default. Same class of bug fixed in `style.load_diagnostics` (a foreign-root row read the fresh npz). After: legacy → `expected 0, missing_configs 0`; fresh → `1610`, unchanged.

## F3 (medium) — CONFIRMED, fixed at the helper
All four `label=m` sites (overparam c4, batchsize c4, critbatch c5, finetune c5) go through `errorbar_seeds`, so the mapping went there (`label` that is a method key → `method_label`; anything else passes through) — no edits to WP4a/WP4b's phase-B notebooks. The two printed tables the reviewer named now use `method_label(r.method)` (column widened to 18: "Stochastic L-BFGS" is 17 chars). Remaining raw keys are the machine `method` column beside `summary_table`'s `label` (by design).

## F4 (medium) — CONFIRMED, marked provisional
Header cell of `cifar_analysis` now states the CIFAR-CE numbers are provisional and must be re-run after `tools/select_best.py`. Scan is at **753 runs** (742 at the implementer's run); the 9 extension configs carry 1–2 of 5 seeds, `attempted=5`, **all `eligible=False`**, so today's table legitimately still shows the old pick (Sven val 1.4028 ± 0.0207, acc 0.533). Two of them already beat it on partial seeds: k=128/lr=0.5/rtol=0.3 → 1.3262 (1 seed), k=128/lr=0.1/rtol=0.1 → 1.3581 (2 seeds). **Expect the CIFAR-CE Sven pick and decision-3's "Sven 53.0%" to change.** `bench/best_configs.json` is still `generated_at 03:58` (pre-extension).

## Verified by running
* `pytest tests/` → **1025 passed, 33 skipped** (137.7 s); `tests/test_analysis_core.py` → 31 passed, 6 skipped. New: `test_errorbar_seeds_shows_display_names`, `test_expected_run_ids_follow_the_frames_own_root`, `test_diagnostics_come_from_the_rows_own_root`; `test_seed_spread_label_is_the_one_band_legend` now asserts the label is in the **legend** of five helpers + `errorbar_seeds`, once each, and absent with `band=False`.
* Warning sweep over all 43 fresh dirs + 3 legacy loads: **0 AVERAGED-OVER, 0 other `[style]` warnings, 0 exceptions**, methods incl. `SGDm`.
* Notebooks (guarded runner, one sbatch): 14/15 re-executed, **0 error cells, 0 unexecuted**, 7–36 s each (~225 s). Only flagged lines are my reporter's regex hitting the `n_missing` table header.

## Deviations / open items
* `polynomial_analysis` was **skipped by the concurrency guard** — WP2 modified it at 19:45 mid-batch. It is clean (0 errors) from the earlier pass; WP2 re-runs it in phase B. For the same reason I did **not** run `NO_PROFILES=1 ./make_plots.sh` (it would clobber WP2's live notebook); the guarded runner covered the identical set.
* `bench/best_configs.json`'s `label` field is pre-C-B7 (`"LBFGS"`, `"JD_UPGrad"`, key `SVD`): phase-B packages should call `style.method_label(entry['method'])`, not read `label`.
* Legend order: the band entry is last in `scan_analysis` helpers, first in notebook `errorbar_seeds` loops (no end-hook).