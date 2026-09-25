WP1 complete. Final report:

---

# WP1 (housekeeping) — done, all acceptance criteria met

## Files changed (all WP1-owned)
* `analysis/style.py` — column registries, method registry, 2 shared-helper bug fixes
* `analysis/scan_analysis.py` — display names at every label site; `summary_table` gained a `label` column
* `analysis/analysis_helpers.py` — re-exports `method_label`
* `make_plots.sh` — `NEW=()` list for the other packages, `NO_PROFILES=1` mode
* `tests/test_analysis_core.py` — +218 lines (4 new tests, 1 extended fixture, 1 new real-scan test, 1 default-root fix)
* notebooks (minimal edits, then executed in place): `cifar_analysis`, `kappa_analysis`, `critbatch_analysis`, `finetune_analysis`, `toy_1d_analysis`, `polynomial_analysis`, `mnist_analysis`, `mnist_analysis_labelRegression` (+10 more re-executed)
* scratch (gitignored): `agent_lab/wp1/{collect_warnings,fix_notebooks,nb_report}.py`, `exec_nbs.sh`, `gate.sh` — re-runnable

## 1. Column registration (found empirically, not guessed)
Loaded **all 43 dirs of `experiment_results/` + 3 legacy dirs** through `load_results → add_derived → config_columns/config_table`, `average_over_seeds`, `Scan.configs`, `attach_standalone_times`. Complete warning set was:
`test`(43) `train_eval_final`(43) `val_best`(43) `val_final`(43) `effective_loader_seed`(42) `val_best_index`(42) `val_last3_mean`(42) `test_acc`(23) `ckpt_init_file`(13) `checkpoint_policy`(3); plus `no colour registered for 'SGDm'` in 28 dirs.
→ 7 added to `OUTCOME_COLUMNS` (+`time_excl_first_epoch`, 6 `standalone_*`); 10 to `PROVENANCE_COLUMNS` (`effective_loader_seed, checkpoint_policy, ckpt_init_file, ckpt_error, train_eval_size, eval_every_steps, svd_spectra_schedule, svd_summary, muon_variant, muon_rule`). **Nothing added to `HPARAM_COLUMNS`.**
Re-run after the fix: **0 AVERAGED-OVER warnings, 0 colour warnings, 0 exceptions** across all 46 dirs.

## 2. Method registry
`METHOD_COLORS['SGDm']='#8FC99C'` (light green, as AdamW is to Adam). New `style.METHOD_LABELS`/`method_label()`: `LBFGS→"Stochastic L-BFGS"` (C-B7), `SGDm→"SGD + momentum"`, `JD→"JD (UPGrad)"`, `PolyakSGD→"Polyak SGD"`; wired into `plot_best_curves`, `plot_k_sweep`, `plot_time_summary`, `plot_time_comparison`, `plot_efficiency`, `summary_table`. **Machine keys unchanged** (`method` column, dict keys, colour lookup) — only what a reader sees. `paired.SEED_SPREAD_LABEL` already existed; pinned by test.

## 3. Two genuine bugs in shared helpers (minimal fixes, both tested)
* `style.clipped_band`: `peak_gpu_mem_mb` is bit-identical across seeds → `v.mean()` lands 3.6e-15 **below** `v.min()`, the clip put the lower edge above the mean, `clipped_yerr` returned −3.6e-15 and `ax.bar` raised `'yerr' must not contain negative values`. **This killed the whole wall-time/memory section of both MNIST notebooks.** Edges now pinned to the mean's side.
* `style.average_over_seeds`: a column that is a string for some optimizers and NaN for the rest (`gram_capture`, `muon_variant`, `muon_rule`) took the NaN as its first value → `np.mean` on strings → `TypeError: ... not compatible with add.reduce` on `mnist_scan_ce`, `mnist_scan_labelRegression`, `rebuttal_overparam_mnist_scan`. No notebook calls it today, but WP2/WP4 might.

## 4. Notebook execution (all in place, outputs kept)

| notebook | cells (code) | err before → after | s |
|---|---|---|---|
| toy_1d_analysis | 35 (21) | 0→0 | 164 |
| polynomial_analysis | 35 (21) | 0→0 | 81 |
| mnist_analysis | 35 (21) | **1→0** | 50 |
| mnist_analysis_labelRegression | 35 (21) | **1→0** | 34 |
| comparisons | 19 (9) | 0→0 | 16 |
| baselines / batchsize / overparam | 8/5/7 | 0→0 | 9/5/8 |
| critbatch / finetune | 6/8 | 0→0 (was no data) | 5/5 |
| nanogpt / cifar / kappa | 9/7/5 | 0→0 | 6/15/5 |
| microbatch / paramfrac | 10/10 | 0→0 | 10/10 |

**Totals: 0 error cells, 0 unexecuted cells, 0 `[style]` lines, 0 stderr** (all 15). `NO_PROFILES=1 ./make_plots.sh` → `done: 15 notebook(s)`, exit 0.

## 5. Tests
`tests/` full suite: **1000 passed, 33 skipped** (122 s). New: `test_schema2_columns_are_registered`, `test_average_over_seeds_tolerates_a_partly_string_column`, `test_clipped_band_never_crosses_the_mean`, `test_every_campaign_method_has_a_colour`, `test_method_labels`, `test_seed_spread_label_is_the_one_band_legend`.
Opt-in real-scan checks (`SV3_CHECK_REAL_SCANS=1`, 6 passed): legacy `toy_1d_scan` 734→148, `mnist_scan_ce` 1040→208 groups (identical to auto-detection); fresh `toy_1d_scan` 1770→354, `mnist_scan_ce` 1610→322, `cifar10_resnet_ce_scan` 744→152, `mnist_scan_ce_confirm` 70→14 — column grouping **== run_id grouping** in every case.

## Deviations from the plan
1. `test_real_scan_grouping_unchanged` defaulted its root to `experiment_results` — which is now the *fresh* root, where its auto-detection oracle is invalid by design (it returned 1610 groups for 1610 runs). Repointed to `experiment_results_legacy_2026-09-18` and added the fresh-root variant above. It was opt-in, so this never failed CI.
2. Added `SGDm` to the hand-written `BASELINES` lists in the 4 headline notebooks. Without it SGDm was silently dropped from every plot despite being in the records. WP2 owns those notebooks in phase B — revert if unwanted.
3. `critbatch`/`finetune` kept in `make_plots.sh` (plan allowed excluding them) reading the legacy root via an explicit `LEGACY_ROOT` + a header cell listing the legacy caveats (test-as-validation, batch-weighted means, train-mode BN, F3 leak for finetune).

## Open issues / notes for the other packages
* **`cifar10_resnet_ce_scan` is a MOVING TARGET.** The decision-5 rtol extension is running *now* (5 `P2.a100.*_cifar_*` jobs); the scan grew 744→745 runs while I worked, and `missing_configs` still lists 4 configs / 20 runs (`k128, lr{0.1,0.5}, rtol{0.03,0.1,0.3}`). WP2/WP4b must re-run `cifar_analysis` + CIFAR-CE tables after it finishes and after `tools/select_best.py`. The slim cache invalidates automatically.
* `optimizer_profile` (decision 4) is PENDING in the queue → `profile_results_v3/` is coming; profile notebooks untouched by me (their owner's call), but they do pick up `clipped_band`/`method_label` from `style.py`.
* **Add your notebook to the `NEW=()` block in `make_plots.sh`** (commented placeholders for `headline_tables`, `spectra_analysis`, `gpt2_analysis`, `legacy_vs_fresh`) in the same commit that adds it.
* `summary_table` now has a `label` column between `method` and `config` — use it for paper tables; `baselines_analysis`'s explicit `cols` list does not show it.
* `analysis_helpers.OUTCOMES` was deliberately **not** extended with the runner's `val_final`/`test`/`train_eval_final`, so `config_table`'s column set is unchanged. WP2 may want those seed-averaged — add them there rather than re-deriving.
* `campaign/run_cpu_tests.sh` still documents "4 cores / 16 GB" (node is now 8/32) and caps jobs at `-c 4 --mem=12G`; that sufficed for every notebook here. I hit `QOSMaxSubmitJobPerUserLimit` once — keep concurrent devtest jobs low.
* Nothing committed; `experiment_results/` untouched except its `_cache/` pickles.