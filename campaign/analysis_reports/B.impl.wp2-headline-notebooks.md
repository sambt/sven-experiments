**WP2 phase B (notebooks) — done.** All six notebooks execute in place with **0 error cells, 0 unexecuted cells**; `pytest tests/` → **1155 passed, 33 skipped**.

## Files
**New:** `/n/home11/sambt/iaifi/sv3/analysis/headline_figs.py` (≈620 l) — the only new code; `/n/home11/sambt/iaifi/sv3/tests/test_headline_figs.py` (24 tests, synthetic root, imports `test_headline`'s fixture). Scratch generators (gitignored): `agent_lab/wp2/build_mlp_notebooks.py`, `build_baselines_notebook.py`.
**Rewritten:** `analysis/{toy_1d,polynomial,mnist,mnist_labelRegression}_analysis.ipynb` (38 cells / 24 code each), `analysis/baselines_analysis.ipynb` (35 / 22). **Re-executed:** `analysis/headline_tables.ipynb` (28 / 19) — refreshes `analysis/tables/` (140 files) against the now-767-run CIFAR-CE scan. `headline.py` **unchanged** (no bug found). Figures: 14 PDF+PNG per MLP notebook, 10 for baselines, under `analysis/plots_v2/{toy_1d,polynomial,mnist_ce,mnist_labelRegression,baselines}/`.

## Per notebook (runtime, sequential, one sbatch)
toy 25 s · polynomial 25 s · mnist-labelreg 25 s · mnist-CE 26 s · baselines 42 s · headline_tables ~90 s.

## Structure (every notebook) and the key numbers
§2 summary = `headline.confirmation_view` (val/test/acc/train-eval, `finished/attempted`, selected config, tuning beside) + freshness banner; §3 curves on **steps / examples / synchronised standalone `train_times`** (2×3: top-6+Sven, then all 14–15); §4 paired ±*t*; §5 budget; §6 efficiency + **timing-join** + time-to-target; §7 robustness; §8 Sven landscape + grid edges; §9 data-seed replicates; §10 spectra pointer; §11 computed closing numbers.

| | toy | poly | MNIST-LR | MNIST-CE |
|---|---|---|---|---|
| Sven rank (val / test / acc, confirm) | 2/15, 2/15 | 2/15, 2/15 | 3/14, 3/14, 4/14 | 4/14, 4/14, 5/14 |
| best method | HIG 1.77e-9 | HIG 0.084 | MuonW 0.0498 | MuonW 0.1046 |
| Sven val / ×best | 5.09e-7 / 287 | 0.1388 / 1.65 | 0.0533 / 1.07 | 0.1134 / 1.08 |
| selected config | k32 lr0.01 rtol1e-4 | k16 lr0.5 rtol0.03 | k64 lr0.5 rtol1e-3 | k32 lr0.5 rtol0.3 |
| **grid edge** | **lr LOW edge** | none | none | **rtol HIGH edge** |
| paired: Sven better/worse/inconclusive | 10/0/4 | 12/1/1 | 4/1/8 | 8/2/3 |
| s/epoch (×fastest) | 1.31 (2.56×SGD) | 1.33 (2.53×) | 3.74 (2.81×) | 3.72 (2.92×) |
| ms/step, peak MB | 3.40, 18.9 | 3.44, 18.9 | 4.03, 31.4 | 4.00, 31.4 |
| grid pts → distinct | 180→115 (.64) | 144→89 (.62) | 128→60 (.475) | 160→106 (.66) |
| **Sven diverged (wide / recorded)** | **194/900 / 0** | 4/720 / 0 | 1/640 / 0 | 0/800 / 0 |
| worst method | KFAC .475 | L-BFGS .907 | KFAC 1.00 | KFAC 1.00 |
| optimism (same-instance) | −63.4 % | +0.6 % | +2.2 % | −1.4 % |
| instance/seed spread (median) | 0.59 | **2.63** | — | — |

Other results the figures carry: toy Sven divergences are **entirely at rtol ≤ 1e-4 and k ≥ 4**, rising with lr (67 % at rtol 1e-6, lr ≥ 0.5) — the heatmaps reproduce `EXPERIMENTS.md` §7 exactly. Cross-GPU: only SGD/SGDm reproduce (~2e-7); Sven **1.26e-3** (MNIST-LR) / 3.1e-1 raw max on MNIST-CE, SOAP has one standalone-only divergence at **204×** — no bit-reproducibility claimed anywhere. On the time axis HIG wins on steps but needs **1150 s vs Sven 60 s** on MNIST (why that axis is log). baselines §2 prints all seven scans: CIFAR-LR 9/11/9/11/9/11, **CIFAR-CE 9/11 val, 11/11 test acc**, nanoGPT 2/5 val, 3/5 test.

## Deviations
1. New plotting/table code went to `headline_figs.py` (not `headline.py`, per the freeze). `scan_matrix`, `divergence_by_method`, `sven_divergence_grid`, `grid_edges`, `instance_spread`, `curve_axis`, `closing_numbers` are the only new derivations; every headline number is fetched from `headline.py`.
2. **`budget.plot_best_of_n` labels lines with the raw key** (`LBFGS`, `JD_UPGrad`) — a C-B7 violation in frozen `budget.py`. Worked around with `hf.relabel_methods(ax)`; **WP1 should fix `budget.py` to use `style.method_label`.**
3. Curves come from the **confirmation** pass (not the tuning scan) so a curve and the table above it describe the same runs; toy/polynomial curves pool 15 runs over 3 instances (§9 decomposes it).
4. PNG copies at 150 dpi (PDF keeps `savefig.dpi=300`); 22–26 MB per plot dir.
5. Dropped from the old notebooks: `summary_table` bars (tuning-seed only), the sharded-scan wall-time/efficiency cells, the batch-resolved diag traces, and §5 spectra (one pointer cell, per task).
6. baselines_analysis was turned into the **cross-scan** view of the four MLP scans (matrices + 4×3 curve grids) and prints no CIFAR rank in prose — it computes them via `hl.ranking_summary()` under a 7-scan freshness gate.

## Open issues
* CIFAR-CE is **still PROVISIONAL** (767/785 runs, 5 live claims, 27 post-selection records; 5 `p2_cifar_ce_rtol` jobs running). `headline_tables` and `baselines_analysis` print the banner and compute every CIFAR-CE cell, so re-running `tools/select_best.py` + re-executing both updates them. None of my notebooks hard-codes a CIFAR number.
* ~36 stale PDFs per scan dir from the pre-rewrite notebooks (`sv_rank_*`, `sv_spectra_*`, `train_loss_k_sweep_*`, `total_time_*`) — `plots_v2/` is untracked, so this is disk clutter only; leaving them in case WP3 reuses the names.
* `analysis/tables/` (140 files) is still not gitignored — orchestrator decision.

## make_plots.sh (I did not edit it)
Uncomment **`headline_tables`** in the `NEW=(...)` list. The other five are already in `SCANS` / `STUDIES` and need no change.