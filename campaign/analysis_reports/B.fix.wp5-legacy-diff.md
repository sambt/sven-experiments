## Files changed (WP5-owned only; nothing committed, results roots untouched)

`analysis/legacy_diff.py` (951→1219 l) · `analysis/legacy_vs_fresh.ipynb` (34→40 cells) · `analysis/WHAT_CHANGED.md` · `tests/test_legacy_diff.py` (20→33 tests) · regenerated `analysis/tables/legacy/` (15 md + 15 tex; 2 new) and `analysis/plots_v2/legacy_vs_fresh/` (2 figures, unchanged code). **No shared/frozen helper touched; no bug found in one.**

**Notebook `legacy_vs_fresh`: 40 cells, 21 code, 0 error cells, 0 unexecuted, 54 s.** `pytest tests/` → **1187 passed, 33 skipped**.

## Findings: 5 confirmed and fixed, 1 fixed with its premise corrected

**F1 (high, aggregate pools ranks-only) — CONFIRMED.** New `rank_change_tally(summary)` groups on the `ranks_only` flag `sven_rank_summary` already writes, so the sentence cannot drift; cell 34 and WHAT_CHANGED §1 now print both groups and never their sum. **Same target: 15 points (6 headline + 9 swept), 7 better / 5 unchanged / 3 worse, Σ −2, mean −0.13. Ranks only (additive cubic): 11 points, 5/3/3, Σ −16, mean −1.45 — 89 % of the pooled total.** Headline same-target: 2 better, 3 unchanged, 1 worse (CIFAR-LR +3). All three large gains (−6, −6, −5) are additive-cubic comparisons. The old "−0.69 places" is retired.

**F2 (tuning-seed fallback undisclosed) — CONFIRMED.** `diff_table` now writes `fresh_basis`; `diff_view` marks both rank cells `*` and adds a `fresh rank basis` column; `fallback_rows()` + cell 14 print the substitution. Polynomial L-BFGS now reads `fresh rank 11/14 *`, `14/15 *`, basis `tuning (2/15 confirmation runs finished: not eligible)` — ranked on 0.2395, printing 0.4855. **1 of 85 headline rows; Sven is `confirm` on all 7, asserted in the notebook.** WHAT_CHANGED §Scope and §4(ii) say so. Two fixture tests pin it (SGD now finishes 1/5 confirmation seeds).

**F3 (`grid_grew=False` hardcoded) — CONFIRMED.** `sven_only_table` now computes both grids plus new `grid_config_strings()`, adding `n_configs_leg/fresh`, `grid_grew`, `fresh_config_in_legacy_grid`. **κ study 3 → 42 configurations, and the fresh winner (k=32, lr=0.75, κ=3) is not on the legacy grid, so its "+0.9 %" compares two searches** (now `grid`-caused and stated in the cell). mnist_paramfrac 19→20, toy_1d_microbatch 24→24 — matching the review. **6 of 10 ablations had identical grids, so CIFAR Fig-5's +59.8 % on an unchanged 5-point grid is not a budget artefact.**

**F4 (intersection-only scope) — CONFIRMED.** New `scan_inventory()` + `LEGACY_ONLY_DISPOSITION` (quoted from `EXPERIMENTS.md` §10) and notebook §1.1b: **27 both / 7 legacy-only (674 runs) / 16 fresh-only (all `_confirm`/`_diag`/`_timing` passes — the fresh campaign added no new scan).** WHAT_CHANGED §Scope carries the table: **four legacy figures now rest on nothing** — critical batch size (2×210 runs, user-cut), CIFAR κ (no fresh CIFAR κ data), CIFAR-**CE** param-fraction (Fig-5 replaces label regression only), small-N fine-tuning (240 runs parked, legacy trained BN on 250–2000 images).

**F5 (abstract ellipsis hid "faster") — CONFIRMED verbatim against `sven_submission.pdf`.** §4 now quotes the whole sentence and splits four claims. New notebook §3.6 computes `hl.time_to_target_table`: **Sven needs fewer epochs than Adam on 3 of 3 regression scans (4.27 vs 10.0 toy; 2.27 vs 3.73 poly; 1.0 vs 1.4 MNIST-LR) but less wall time on only 1 of 3 (5.6 vs 5.7 s; 3.0 vs 2.2 s = 1.39×; 3.7 vs 2.1 s = 1.81×).** "Converging faster" is a per-step claim; "at a fraction of the wall-time cost" is not tested here at all.

**F6 (nanoGPT `split`) — gap real, PREMISE REJECTED.** The review says both roots hold "the same 871 validation blocks" and legacy "did NOT select on the examples it reported". Measured (new `nanogpt_split_check`, notebook §3.5): **legacy `val[0]`=4.368765 equals fresh `test[0]`=4.368757 (1.8e−6), while fresh `val[0]`=4.362548 differs by 1.4e−3; `n_val` 871 both sides, `n_train` 7,842→6,971 = exactly `n_test`.** So the legacy validation blocks are the fresh **test** split, and since legacy carried no `n_test`/`test` field it *did* select and report on one and the same held-out set. The wording gap is real, so `_SPLIT_CAUSE` is now per dataset: nanoGPT gets `testsplit_carved` with that measured text; `split` no longer mentions Shakespeare; quarantined GPT-2 gets no split cause (never diffed). Ranks unaffected (2/5).

## Deviations / open issues

* **WHAT_CHANGED is ~1,880 prose words + 29 table lines (≈2.5 pp)** — the implementer's declared over-length, grown by ~280 words by F1/F4/F5. Trim candidates: §3's predicted-vs-observed table, §5.
* Swept-study ranks still use final val loss, not the rebuttal's time-to-target (stated in both).
* CIFAR-CE remains PROVISIONAL (767 runs, 27 after the selection, 5 live claims); every CIFAR-CE cell recomputes — **re-run after `tools/select_best.py`**.
* `tests/test_spectra_figs.py::test_plot_probe_metric_marks_the_float64_resolution_limit` failed once mid-session and passed in isolation and on re-run — WP3's file, being edited concurrently; **not** caused by this package (zero `legacy_diff` references).

**make_plots.sh:** uncomment the placeholder `# WP5 legacy diff: legacy_vs_fresh` → `legacy_vs_fresh` in `NEW=()`.