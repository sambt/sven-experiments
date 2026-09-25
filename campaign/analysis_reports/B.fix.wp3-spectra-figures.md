**WP3 spectra-figures — fixer report**

## Files changed
`/n/home/anon/sven-experiments/analysis/spectra_figs.py` (927→1203 l), `/n/home/anon/sven-experiments/analysis/spectra_analysis.ipynb` (42→43 cells), `/n/home/anon/sven-experiments/analysis/comparisons.ipynb` (22→23 cells), `/n/home/anon/sven-experiments/tests/test_spectra_figs.py` (456→659 l, 23→32 tests). `analysis/sv_diagnostics.py` **not touched** in this round (the implementer's noise-floor fix stands; it is still the one shared-helper fix to list). Backups: `/n/home/anon/sv3_wp3_scratch/fixB/*.ipynb.bak`. No results written.

| notebook | code cells | error cells | runtime |
|---|---|---|---|
| `spectra_analysis.ipynb` | 26 | **0** | ~105 s |
| `comparisons.ipynb` | 12 | **0** | ~45 s |

`tests/` green on a CPU job: **1210 passed, 33 skipped** (136 s). (The implementer's "31 tests" was a miscount — the file had 23; it now has 32.)

## All 7 findings reproduced; all 7 fixed
1. **F1 x-axis (high) — real.** Measured the schedule on all 7 diag passes: `{1: 1000, 20: N}`, toy 1,262 logged of 6,240 steps, so index-fraction 0.50→step 630 (10.1 %), 0.85→step 2,440 (39 %); MNIST 1,731 of 15,620, nanoGPT 1,220 of 5,400 — the seven curves were on different axes. Now `x = svs_step / n_steps` and a new **`smooth_steps(step, y, window)`** averages over a fixed number of *steps*. Layout fixed (`subplots_adjust(right=0.60)`, explicit font sizes, one-line title; the y-label no longer overlaps the title). **The figure's claim reverses**: toy's jump is at 1–25 % of training, not 80 %. Discarded batch residual at 1/10/25/50/100 % of training, printed (seed mean): toy 9.8e-7 / 1.5e-5 / **4.6e-2** / 8.1e-2 / 9.5e-2; MNIST-CE 4.8e-1 / 1.7e-1 / 1.3e-1 / 1.0e-1 / 7.4e-2; polynomial flat ~4e-3; CIFAR-labelreg 1e-8→1.2e-5; CIFAR-CE 1e-8→5.3e-6; MNIST-labelreg and nanoGPT on the 1e-8 floor throughout.
2. **F2 wrong lr (high) — real.** `best_sven(k=128)` → lr 0.5/rtol 0.3 vs selection of record lr 0.1/rtol 0.01; used/B at lr 0.5 is 0.6080→0.2291, at lr 0.1 0.9717→**0.5586**. Every lr in `comparisons.ipynb` now comes from `headline.selection_methods(scan)['SVD']`; the live best-on-disk is printed beside it with a `** differs **` flag (only CIFAR-CE differs), and `headline.freshness_report()` + the PROVISIONAL banner were added (now 777 tuning records, 37 post-selection, 5 live claims). The report's "0.97→0.56" is now what the notebook prints (0.5586, not 0.5577 — a 4th seed landed).
3. **F3 hidden single seed (medium) — real** (5 entries, `ents[0]`). `plot_probe_spectra` annotates "seed 1000 (1 of 5 cached)" in every panel, names the seed in the suptitle, and warns if a caller does not choose; the notebook passes the seed explicitly. §9.1 markdown states it.
4. **F4 grid seed counts (medium) — real.** toy lr=0.01: k=8/16/32 × rtol=1e-6 are **1/5** (4 diverged), × 1e-5 **4/5**; CIFAR-CE lr=0.1: rtol 0.03/0.1/0.3 are 4/4, 4/4, 4/4 with 1–2 of 5 not landed. `used_rank_grid` now returns `n_records`/`n_diverged`/`n_no_curve`/`counts`, keeps all-diverged cells as NaN rows, and the heatmap prints `n_seeds/attempted` under and red-hatches every short cell, with a new table of short cells and a line naming the absent CIFAR-CE `k=64, rtol≥0.03` cells (the extension ran `k=[128]` only).
5. **F5 tie counted as win (medium) — real.** New `low4_table` / `low4_verdict` give the paired per-seed difference with a 95 % t-interval. **dist_init: Sven's mean is smallest on 2 of 4 scans, resolved on 0** — polynomial 7.524 vs HIG 8.120 (−0.597 ± 0.72, 4/5 seeds lower, *not* resolved), MNIST-labelreg 12.48 vs HIG 12.52 (−0.042 ± 0.20, 3/5, not resolved); larger on toy (+1.56 ± 1.76, not resolved) and MNIST-CE (+1.79 ± 1.00, resolved). **param_norm: smallest on 1 of 4, resolved on 1** (polynomial −0.590 ± 0.33, 5/5 seeds).
6. **F6 width claim (medium) — real.** Measured widths **593 / 673 / 512 / 512**, never 27,562. Intro table corrected; new `probe_widths()` prints rows/P/width/ckpts/methods/seeds/`n_used`(20)/`n_cached_any_tag`(24 for MNIST-labelreg) under the same filters `probe_metrics` uses.
7. **F7 unmarked float64 floor (medium) — real.** Toy resolved rank **14→45 of 593**, cond 4.4e11→6.75e11 against a 1e12 ceiling, raw σ_min/σ_max 1.3e-17, σ_32 unresolved at every checkpoint. `probe_metrics` gained `floor`/`sigma_b`/`sigma_b_resolved`; `plot_probe_metric` draws the ceiling on `cond` and the floor on `σ_B/σ_1`, rings unresolved points, and (new `near_decades=6`) omits a line the data never approach so polynomial/MNIST panels stay readable. A resolution table backs it in-notebook.

## Unchanged key numbers (re-verified)
`min(k, rtol-rank) == num_nonzero_svs` on every logged step of all 7 scans. rtol binds on 5 of 7; batch residual kept first→final: toy 92.08→91.53 %, poly 92.25→99.59 %, MNIST-labelreg 100→100, MNIST-CE **31.0→95.2 %** (10.0→2.8 SVs), CIFAR-labelreg 100→99.999, CIFAR-CE 100→99.999 (128→117), nanoGPT 100→100. Probe `top_k` at the selected k: toy 100.000 %, **polynomial 58.820 %** (`in_span` 99.490 %), MNIST-labelreg 99.868 %, MNIST-CE 96.092 %. Step-0 between-method spread 0.0e+00 on all four probe scans. Seed spread of final used rank 0 % (nanoGPT) → **68.7 % (MNIST-CE)**, CIFAR-labelreg 24.2 %.

## Deviations / open issues
* `scipy.stats` is now imported by `spectra_figs` (same convention as the frozen `paired.py`).
* CIFAR-CE remains live-provisional in both notebooks; no CIFAR-CE number is hard-coded and the diag pass still matches today's pick.
* Probe spectra for both CIFAR scans and nanoGPT still absent (cost-bound); skipped cleanly.
* Both notebooks are unchanged in figure filenames — 36 files under `plots_v2/spectra_analysis/`, 16 under `plots_v2/comparisons/`.

## For `make_plots.sh` (orchestrator)
Uncomment `# WP3 spectra    : spectra_analysis` in the `NEW=( … )` block; `comparisons` is already in `SCANS`.