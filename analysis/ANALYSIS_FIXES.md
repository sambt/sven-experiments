# Analysis fixes — audit of `analysis/` (2026-09-17)

> Paths in this document predate the 2026-09-21 reorganisation: the helper modules are now
> in `analysis/lib/` and the notebooks in `analysis/notebooks/<group>/`
> (`analysis/README.md`). Nothing else here changed — a module or notebook named by its bare
> name still means the same file.

Working checklist from the full audit of every analysis script and notebook. Fix one
piece at a time; tick items off here and note what was decided.

Status: `[ ]` open · `[~]` in progress · `[x]` done (with date + one-line note)

**After any fix the notebooks must be re-executed** — saved outputs and PDFs are stale
until then.

## Already done (before this list)

- [x] Sensitivity plots (`plot_sensitivity`): one point = one config, seed mean ± std band,
  third axis pinned at the best config's value (`third='min'` = old envelope).
- [x] `kappa_analysis` / `cifar_analysis` param-fraction plot: seed means + error bars
  (`analysis_helpers.seed_mean_best`) instead of single-run `idxmin`.
- [x] SV spectra x-axis normalised by `k` (= B), not by the stored array width; rtol line drawn.
- [x] Deterministic tie-break in `Scan.configs` (smallest k, largest rtol) — makes `best_sven`
  agree with the standalone timing runs; Sven now has a standalone bar in every scan.
- [x] `epoch_axis` no longer falls back to plotting epoch index as seconds.
- [x] Global optimizer colours: `style.METHOD_COLORS` / `method_color()`, used by
  `scan_analysis`, `profile_helpers`, and all study notebooks.
- [x] `summary_table` crash on string-valued hparams (JD / HIG).

---

## A. Wrong or misleading results

- [x] **A1. Study notebooks report the luckiest single run, not a config.**
  `analysis_helpers.best_per_method()` is `idxmin` over individual runs. Feeds every
  table/plot in `baselines`, `batchsize`, `overparam`, `finetune`, `nanogpt`, `cifar`
  (incl. the overparam "Sven rank k/11" table). Scans have 2–5 seeds. Select by seed
  mean, show seed spread (as `scan_analysis` does).
  - **Done 2026-09-17.** `analysis_helpers.config_table()` groups runs into configs (all scalar hparam columns), seed mean/std/min + `n_seeds`/`n_diverged`; `best_per_method()` / `seed_mean_best()` now return best *config* rows (same column names, now seed means). All six notebooks print mean ± std and draw clipped seed error bars (`errorbar_seeds`). `finetune_analysis` edited but not executed (data missing locally, B16).
- [x] **A2. `nanogpt_analysis` plots one run's curve per optimizer**, no seed band; its
  summary table reports sharded-scan wall time although standalone timings are loaded
  two cells later.
  - **Done 2026-09-17.** Curves are seed mean + clipped band of the best config (`config_runs` + `seed_band`); summary now `scan_analysis.summary_table` with `standalone_time_s` beside the sharded time. Ordering changed: AdamW 1.760 ± 0.012, Sven 1.769 ± 0.019 (was Sven 1.742 vs AdamW 1.750 from single best runs).
- [x] **A3. `critbatch_analysis`**: target = 1.2 × single best run; steps-to-target is min
  over configs *and* seeds; runs that never reach target silently dropped; "steps" is
  actually epochs; only 2 seeds.
  - **Done 2026-09-17.** `epochs_to_target_table()`: target = 1.2 × best seed-mean final val loss; a config counts only if ALL seeds reach it; seed-mean epochs ± spread; unreached cells drawn as hollow markers. Axis says epochs. **Still open:** a McCandlish plot wants optimizer *steps* (epochs × N/B), which needs N or steps/epoch in the results (baselines carry no `n_steps`); and 2 seeds is thin.
- [x] **A4. Diverged runs handled two ways.** `scan_analysis.final()` = last finite value;
  `analysis_helpers._final()` = `curve[-1]`. 48 / 89 / 30 / 97 runs in the four headline
  scans are NaN under one rule and finite under the other → same scan can rank
  differently in study vs scan notebooks.
  - **Done 2026-09-17.** DECISION: diverged = failed. One rule, `style.is_diverged()` (non-finite end, or val end > 10x val start -- the latter added under B19), used by both stacks; runs flagged `diverged`, left out of seed means and curves, counted in `n_diverged`. Configs ranked: eligible (> half of seeds finished) → **fewest diverged seeds** → seed mean (without the middle key, LBFGS's "best" became configs that diverge on 2/5 seeds). `load_scan(drop_diverged=)` removed. Effect: LBFGS best config unchanged in toy / polynomial / MNIST label-reg; **MNIST-CE LBFGS best is now lr=0.5, max_iter=1, hist=2, which has no standalone timing run (timed: max_iter=2) → needs one timing rerun.**
- [x] **A5. Seed bands / error bars go ≤ 0 on log axes** (arithmetic mean ± std) in
  `plot_best_curves`, `plot_k_sweep`, best-per-method bar chart, microbatch/paramfrac
  plots. Toy 1D: train band ≤ 0 for 12 methods (the vertical slabs); bar std ≥ mean for 6.
  One convention everywhere.
  - **Done 2026-09-17.** DECISION: arithmetic seed mean; band = ±1 std with the lower edge clipped at the lowest seed. One implementation: `style.clipped_band` / `clipped_yerr`, via `scan_analysis.seed_band`. Applied to `plot_best_curves`, `plot_k_sweep`, `plot_sensitivity`, the best-per-method bar chart (4 notebooks), all study notebooks, and the microbatch/paramfrac cells (those two notebooks cannot currently run — see B19).
- [x] **A6. `ddof=0` in microbatch/paramfrac notebooks**, `ddof=1` everywhere else.
  - **Done 2026-09-17.** Fixed together with A5 (`ddof=1` in the microbatch/paramfrac cells).
- [x] **A7. Shard-inflated times still used** in: `summary_table` `total_time_s`,
  `plot_time_summary`, `plot_time_vs_k`, `plot_efficiency`, and the first wall-time
  convergence plots (section 3). Use standalone timings or label as sharded.
  - **Done 2026-09-17.** Standalone timings are used wherever they exist: `plot_best_curves(time_key='auto')` (x label says standalone / sharded), `plot_time_summary(source='auto')` (seed mean + clipped error bar, y label says which; returns the source), `summary_table` has `sharded_time_s` AND `standalone_time_s`. The duplicated standalone curves in section 6 are gone (section 3 now IS the standalone plot). `plot_time_vs_k` / `plot_efficiency` can only use scan times (no standalone k sweep -- RERUNS_NEEDED 3) and are labelled "sharded scan".
- [x] **A8. `plot_time_vs_k` averages over all rtols and seeds**; docstring says "seed-mean".
  - **Done 2026-09-17.** `plot_time_vs_k(rtol='best')`: seed mean +/- clipped spread per (k, lr) at ONE rtol (the best config's, in the title). Result on toy: flat in k -- the Gram `eigh` is B x B whatever k is.
- [x] **A9. `comparisons.ipynb` §5 uses `lr = max(sven_lrs)` (= 1.0) for every dataset** —
  arbitrary, near-unstable, not stated on the plot.
  - **Done 2026-09-17.** `comparisons` s5 uses each dataset's best lr at k = B (`best_sven(k=B)`), shown in the legend and title.
- [x] **A10. Model-selection metric inconsistent**: `baselines`/`batchsize`/`overparam`
  select on final *train* loss, the rest on final *val* loss. MNIST-CE best Sven differs
  by loss (k=4, rtol=1e-2) vs accuracy (k=8, rtol=1e-4). Selection and reporting share
  the validation split (no test set).
  - **Done 2026-09-17.** DECISION: **final validation loss is the one selection metric in every
    notebook** (`baselines`, `batchsize`, `overparam` switched from train loss on request); each
    notebook declares `METRIC`; `summary_table` shows the train loss (and accuracy) alongside.
    The switch changes the overparam story: on polynomial LBFGS is now best at every N (Sven
    was, on train loss) and on MNIST Sven is best only at P/N >= 5.5. MNIST-CE loss-vs-accuracy
    disagreement stays (we select on loss). No test split: RERUNS_NEEDED 8.
- [x] **A11. SV-spectra tails are survivorship-biased** (averaged only over steps whose rank
  reached that index; pinned just above rtol). Proper fix needs untruncated SVs logged
  in a rerun.
  - **Done 2026-09-17.** Optimizer fixed: `SvenGram.step` / `SvenGramReg.step` log the full B-vector spectrum before the k / rtol cut (69 Gram tests pass); classic randomized path documented as unable to. `plot_epoch_spectra` warns while fed truncated spectra. Needs the k = B slice rerun: RERUNS_NEEDED 1.

## B. Broken or stale

- [x] **B12. `paramfrac_analysis` cells 11–14 (MNIST) cannot run**: `datasets['MNIST']`
  commented out; `df.columns([...])` is a TypeError; last cell has no `savefig`. Neither
  `microbatch_analysis` nor `paramfrac_analysis` has saved outputs.
  - **Done 2026-09-17.** Both notebooks rewritten from scratch on `scan_analysis` (see B19); the broken MNIST cells are gone and all four datasets (toy, polynomial, MNIST-CE, MNIST label-reg) are in.
- [x] **B13. `baselines_analysis` source vs outputs disagree**: source loads
  `polynomial_scan` / `mnist_scan_labelRegression`, saved outputs came from
  `rebuttal_baselines_*` (toy scale differs 100×: 5.9e-8 vs 3e-6).
  - **Done 2026-09-17.** Rewritten on `scan_analysis`: loads the four headline scans (the `rebuttal_baselines_*` runs now live there), `summary_table` over ALL optimizers present, seed-mean bars with clipped error bars, `METRIC = final_train_loss` in one place (A10 still open). Plots to `plots_v2/baselines/`. New result: **HIG beats Sven on toy (9.3e-8 vs 8.0e-7) and polynomial (2.3e-3 vs 1.0e-2) train loss**; Sven is mid-pack on MNIST.
- [x] **B14. `comparisons.ipynb` outputs predate current data** (659 vs 714 runs).
  - **Done 2026-09-17.** Re-executed in place with `make_plots.sh comparisons` (714 / 711 / 979 / 1020 runs, inline figures). Every other notebook is still stale until `./make_plots.sh` is run -- see the last item of this file.
- [x] **B15. `make_plots.sh` calls nonexistent `analysis/profile_analysis.py`** and the old
  `profile_results/` directory.
  - **Done 2026-09-17.** Rewritten: `make_plots.sh` now re-executes the analysis notebooks in place with `jupyter nbconvert --execute --inplace` (all, `ONLY_SCANS=1`, or named ones). The old `profile_analysis.py` / `profile_results/` targets no longer exist; the profile notebooks read `profile_results_v2/` via `profile_helpers`.
- [x] **B16. Six referenced scans missing from local `experiment_results/`**:
  `rebuttal_baselines_toy_1d_scan`, `cifar10_resnet_paramFrac_scan_labelReg`,
  `rebuttal_fig5_cifar_paramfrac_scan`, `exp_finetune_cifar_smallN`,
  `cifar10_resnet_kappaScan_labelReg`, `cifar10_resnet_ce_kappaScan`.
  Notebooks skip silently or fail.
  - **Done 2026-09-17.** Cannot be fixed from this checkout: needs a sync from the cluster or a rerun. Tracked as RERUNS_NEEDED.md item 7 (the notebooks already skip a missing directory with a printed `missing ...` line). `rebuttal_baselines_toy_1d_scan` is no longer referenced (B13).
- [x] **B17. `FOCUS_RTOL` defined in all four scan notebooks, never used.**
  - **Done 2026-09-17.** Removed from the four scan notebooks (`FOCUS_LR` alone remains, with a comment saying what it is for).
- [x] **B18. `profile_helpers.load_profiles` weak cache key** (count + max mtime); in-place
  re-runs undetected. `style._dir_signature` is the robust version.
  - **Done 2026-09-17.** `load_profiles` now keys its cache on `style._dir_signature` (name, size, mtime of every file); verified the second load hits the cache.

- [x] **B19. `microbatch_analysis` / `paramfrac_analysis` cannot run on the current data at all.**
  Their hard-coded slices do not exist: the four scans contain only k=32, rtol=1e-3
  (notebooks ask for k=16 or rtol=1e-2) → `IndexError` on the first plot cell. Also they
  import `seaborn` (unused; not installed in the local env). The paramfrac scans are
  missing seeds (97 / 86 runs of 100). Supersedes C21.
  - **Done 2026-09-17.** Rewritten (`microbatch_analysis`, `paramfrac_analysis`): no hard-coded slices -- the lr is `knob_ref_lr` = best seed-mean val loss at the reference value (mb=1 / f=1), shared by every line; new `scan_analysis` helpers `plot_knob_curves` / `plot_knob_summary` / `knob_table`; seaborn dropped; plots to `plots_v2/{microbatch,paramfrac}/`. Tables carry `n_seeds` / `n_diverged` / `n_missing` (no result file). Sub-decision, extending A4: a run is also **diverged if its final val loss is > 10x its pre-training val loss** (`style.is_diverged`, `DIVERGED_FACTOR`) -- the MNIST paramfrac runs at f <= 0.25 end at 1e7..1e15 without a NaN. Verified it changes no headline best config; it flags 47 finite blow-ups on toy (Sven at lr >= 0.5, k >= 4; KFAC), 5 / 5 / 3 on the others.
- [~] **B20. MNIST-CE LBFGS needs a standalone timing rerun** for its best config under the
  A4 rule (lr=0.5, max_iter=1, history=2); until then it has no standalone bar / curve.
  `bench/select_best_configs.py` should be re-run with the new ranking for all scans.
  - **Done 2026-09-17.** Not an analysis fix: needs 5 timing runs on the cluster. Moved to RERUNS_NEEDED.md item 2 (kept here so the checklist stays complete).

## C. Consistency

- [x] **C19. `BASELINES` in scan notebooks omits AdamW, HIG, JD_UPGrad** (present in scans
  and timing runs). Include, or exclude with a stated reason.
  - **Done 2026-09-17.** `BASELINES` in the four scan notebooks now lists all 12 optimizers (still filtered against what the scan contains). Note: AdamW is bit-identical to Adam in every headline scan because the grid only has weight_decay = 0 -- it overlays Adam; a real AdamW comparison needs a weight-decay grid (RERUNS_NEEDED 10).
- [x] **C20. Three helper stacks**: `scan_analysis` + profile helpers → `plots_v2/`;
  `analysis_helpers` and two legacy notebooks with inline `add_derived_columns` →
  `plots/` (which also holds ~25 old directories). Pick one root.
  - **Done 2026-09-17.** Two stacks remain by design -- `scan_analysis` (scan-shaped data) and `analysis_helpers` (config tables over arbitrary study grids), the latter now built on the former (`seed_band`, `is_diverged`, `method_color` re-exported). The legacy inline `add_derived_columns` notebooks are gone (B19). Every notebook writes to `plots_v2/<name>/`; `plots/` is now legacy output only (~25 old directories) and can be deleted once nothing in the paper links to it.
- [x] **C21. Hard-coded slices in `microbatch_analysis` / `paramfrac_analysis`** unrelated to
  best configs (toy k=16, lr=0.05; polynomial lr=0.5 in one, 0.1 in the other);
  `param_fracs` from unsorted `.unique()` → arbitrary legend/colour order.
  - **Done 2026-09-17** with B19 (no hard-coded slices any more).
- [x] **C22. Naming**: "Toy 1D" / "1D" / "1D Regression"; "MNIST" = label regression in
  `comparisons`, ambiguous elsewhere; "SVD" vs "Sven"; axis labels `final_train_loss` /
  "best final train loss" / "Final Val Loss".
  - **Done 2026-09-17.** `style.DATASET_TITLES` is the one spelling of every dataset (used by all notebooks, incl. `comparisons` legends: "1D" -> "Toy 1D", "MNIST" -> "MNIST (label reg.)"); `style.metric_label()` is the one axis label per metric ("Final validation loss (seed mean)" etc.). The `SVD` optimizer id is mapped to "Sven" at load time everywhere.
- [x] **C23. `legend_below` uses the first panel's handles only** → methods / OOM markers
  only in later panels get no entry. `chunk_fraction_sweeps`,
  `batchsize_capture_vs_solve` have no legend.
  - **Done 2026-09-17.** `legend_below` collects handles from every axes of the figure; `chunk_fraction_sweeps` and `batchsize_capture_vs_solve` now have legends.
- [x] **C24. "Peak memory / SGD" heatmap silently falls back to Adam/AdamW** when SGD absent.
  - **Done 2026-09-17.** `rel_mem` is peak_mb / SGD only (no Adam/AdamW fallback); the profile data has SGD for every architecture, so no cell goes blank.
- [x] **C25. Hard-coded parameter counts** (593, 673, 27562, 11,181,642) in
  `overparam_analysis` / `finetune_analysis`.
  - **Done 2026-09-17.** `generic_scan._scan_facts` now writes `n_params`, `n_train`, `n_val` on every record (also what RERUNS_NEEDED 6 needs); `analysis_helpers.n_params_of` / `Scan.n_params` read it and fall back to the old constant WITH a printed warning for records made before 2026-09-17.
- [x] **C26. `k` coloured viridis in k-sweep / SVs-used plots, default cycle in sensitivity
  plots.**
  - **Done 2026-09-17.** `plot_sensitivity` colours k with viridis like the k-sweep and SVs-used plots (rtol / lr line families keep the default cycle).
- [x] **C27. "Font family 'arial' not found" spam** on every figure; add a fallback list in
  `set_style`.
  - **Done 2026-09-17.** `set_style` uses `font.family = sans-serif` with the fallback list Arial / Helvetica / Liberation Sans / DejaVu Sans; no more findfont spam.

## Final step, after every code fix

- [x] **Re-execute all notebooks in place**: `./make_plots.sh` (from the repo root) -- run 2026-09-17
  after the section-C fixes: 19 notebooks, 193 figures, no error cells. Two things it caught
  and that were fixed on the spot: `cifar_analysis` referenced a loop variable outside the
  loop when its paramfrac data is missing; `finetune_analysis` did not catch its missing
  data directory (now prints `missing ...` and skips like the others). Remaining printed
  notes are all expected: `missing-data` (RERUNS 7), `spectra-truncated` (RERUNS 1),
  `no-standalone` for MNIST-CE LBFGS (RERUNS 2 / B20), `[n_params]` fallback (RERUNS 10).

- [x] **E30. Backend columns split configurations.** `analysis_helpers.config_columns` keyed
  configs on `gram_capture` / `gram_chunk_numel` too; the CIFAR label-reg scan captured some
  seeds of the same Sven config `chunked` and others `full`, so those configs appeared as
  3+2 / 4+1 seed fragments -- "40 missing runs", mostly ineligible. Found by the cluster
  instance (290/290 files present). Fixed 2026-09-17: backend columns excluded from the
  config identity; `cifar_analysis` re-executed; RERUNS_NEEDED item 4 corrected.

## D. Open questions raised by the fixed plots (not fixes -- decisions)

- [ ] **D28. "Final" = last epoch rewards oscillating optimizers.** With every plot honest,
  HIG's train loss on toy 1D swings by ~3 decades between consecutive epochs and happens to
  land at 1e-7 on epoch 20 -- which is why `baselines_analysis` ranks it above Sven
  (now on val loss: 3.1e-7 vs 3.0e-6). The same applies to any last-epoch metric. Options: (a) keep
  last-epoch and say so; (b) score by the mean over the last few epochs; (c) score by the
  best epoch (optimistic). (b) is the usual choice; it is one function
  (`style.final_value`) if wanted. Not changed without a decision.
- [x] **D29. `AdamW` duplicates `Adam` in every headline scan.** Root cause was the experiment
  code, not the grid: `build_standard_optimizer` forced `weight_decay=0.0` on AdamW. Fixed
  2026-09-17 -- AdamW now runs with its PyTorch default 0.01 unless the config says otherwise
  (`experiment_utils.resolve_weight_decay`); AdamW run_ids carry `_wd`; `weight_decay` is in
  the analysis config key. **Needs the AdamW reruns in RERUNS_NEEDED item 0.** Muon (PyTorch
  default wd 0.1, also forced to 0): a `MuonW` variant was added instead of changing Muon --
  same split as Adam / AdamW; runs listed in item 0.
