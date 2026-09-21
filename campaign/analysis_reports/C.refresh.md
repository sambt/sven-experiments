**Refresh complete — commit `040f271` on `robustness-campaign`.**

## 1. State verified from disk
- `tools/reconcile.py --all --list p2_cifar_ce_rtol --list p2_cifar_fig5_selected`: **0 runs to do**, svd 45/45 and 15/15 `ok`, 0 diverged/oom/error/stale/claimed-live.
- `bench/best_configs.json` vs `6e7fc72~1` (sorted-key dump): **only** `scans.cifar10_resnet_ce_scan` changed, and inside it only the `SVD` method — the other six scans and the ten other CIFAR-CE methods are byte-identical. Extra top-level keys = bumped `generated_at` + new `selection_provenance`.
- New pick's passes: `_confirm`/`_timing`/`_diag` hold **5 runs each** beside the old pick's 5. `headline.selected_runs` matches on hyperparameter *values*, so all four CIFAR-CE passes resolve to `svd_bs128_k128_lr0.5_rtol0.3_…`, n=5 (seeds 4000-4004; 4100-4104 confirm). **No mixing.**
- `profile_helpers.profile_status(ROOT_V3)` = `(True, 720, 720)`, all 7 config dirs full (no sentinel; count rule).

## 2. CIFAR-CE Sven, old → new (confirmation seeds, 5/5, 0 diverged)
| | old `k128 lr0.1 rtol0.01` | new `k128 lr0.5 rtol0.3` |
|---|---|---|
| val | 1.42385 ± 0.0126 (rank **9**/11) | **1.35523 ± 0.0160** (rank **9**/11) |
| test loss | 1.41151 ± 0.0132 (rank 9) | **1.33908 ± 0.0164** (rank **8**) |
| test acc | 53.0% ± 0.81 (rank **11**/11) | **58.2% ± 0.53** (rank **10**/11) |
| train-in-eval | 0.8186 | **0.1586** |
| cost | 177.3 ms/step, 22.96 GB | 177.6 ms/step, 22.96 GB (unchanged, as Gram requires) |

## 3. Fig 5 at the selected config (`k=128, lr=0.5, rtol=1e-3`)
15/15 `ok`, **0 divergences**. val / test-acc at pf 1→0.05: `0.471 → 0.503 → 0.876 → 2.40 → 4.94` and `69.1% → 68.8% → 67.8% → 25.6% → 19.2%`. Cost: 187.5 ms/step, 22965 MB at pf=1; slowest pf=0.5 at 453.2 ms (×2.42); memory only ×0.51 at pf=0.05 while the step is ×1.71. Legacy `k=64, lr=1` keeps its 2/15 divergences (pf 0.05, 0.1) — the re-run **separates** "masking hurts" (it does) from "lr=1 is unstable" (it is: all 5 headline-grid Sven divergences sit there). Notebook shows both, selected first, read out of the selection file.

## 4. Profile v2 → v3 (headline)
CIFAR `full J` **842.0 → 188.0 ms (×0.22)**, `chunked` 470.2 → 207.9 (×0.44), classic 1126.2 → 1040.7; nanoGPT hooks 83.2 → 59.5 (×0.72); MNIST classic 15.5 → 10.1, `full J` 19.4 → 14.3; toy/poly within 2%. **Every memory ratio 1.00.** Baseline control median v3/v2 = 0.993/1.010/0.969/0.999/1.008; 5 of 63 set points >10%, each named (2 L-BFGS regime switches, 1 divergence, Muon MNIST ×0.85 / CIFAR ×1.20 — allowed, `expandable_segments` is process-wide). 8 status flips accounted for in a new markdown paragraph; none is a cost result. Fig-5 overlay now opens on v3 (masked 432-504 vs 193 ms unmasked — same direction as records; memory agrees to 0.05%).

## 5. Execution / tests
0 error cells, 0 unexecuted code cells in `cifar_analysis`, `headline_tables`, `baselines_analysis`, `comparisons`, `spectra_analysis`, `legacy_vs_fresh`, `nanogpt_analysis` (added — it printed the moved selection timestamp) and the 4 profile notebooks. `freshness_report` **clean** over 7 scans × 4 passes; PROVISIONAL banner gone. `analysis/tables/` regenerated (38 files). **1225 passed, 33 skipped, 0 failed** (excluding the 4 `test_paper_assets_*` files owned by the parallel paper workflow; re-run after its edit settled: 242 passed, 1 skipped).

## 6. Files
Committed: `EXPERIMENTS.md` (new §1.5 v2→v3 subsection, re-pointed §5 selection row + closed rtol bullet, rewritten §3.3, §9.1 resolved, §9.3 re-derived), `campaign/ANALYSIS_PLAN.md` (§7.3 CE facts marked superseded), `campaign/grid_counts.md`, `analysis/{headline,large_figs,spectra_figs,profile_helpers}.py`, 11 notebooks, `analysis/tables/`, 3 test files.

## 7. Open
- `campaign/CAMPAIGN_STATUS.md` not touched — the orchestrator owns it and another agent committed `005346c` mid-session; needs a status line for this refresh.
- `rtol=0.3` is **still** `EDGE-HIGH` (top of the extended set) — accepted, not extended further.
- Orphaned `profile_results_v2/_profile_cache.pkl` still in a read-only root (unchanged decision).
- Paper workflow must re-run `python -m paper_assets` to pick up these numbers.