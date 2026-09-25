# Analysis plan for the fresh campaign results (written 2026-09-20)

Handoff document: readable with no context from the session that produced it. Read first:
`campaign/CAMPAIGN_STATUS.md` (what ran, where), `campaign/CONTRACTS.md` (record schema 2, layout, binding
decisions), `analysis/ANALYSIS_FIXES.md` (analysis conventions), `CHANGES_NEEDED.md` §1 and §3.9 (C-A1..A6, C-L4, C-Z1).

## 0. State of play

**Data (all under `experiment_results/` = `/n/holystore01/LABS/iaifi_lab/Users/sambt/sven_experiments`, schema 2):**
* 21 campaign scans incl. the extension round — reconcile clean (`tools/reconcile.py --all campaign/plan_campaign.yaml`).
* `exp_gpt2_small_comparison` — 29 runs, 1 seed (5 baseline runs still finishing on 09-20 16:00; check reconcile on `campaign/plan_gpt2.yaml`).
* Result-dependent passes for the 7 headline scans: `<scan>_timing` (standalone, logging off), `<scan>_diag`
  (best config of every method, `checkpoints: log` / `epochs` on CIFAR, dense spectra), `<scan>_confirm`
  (5 fresh model seeds base+100..104; toy/polynomial additionally 3 data seeds).
* Selection of record: `bench/best_configs.json` from `tools/select_best.py` (full binding rule: eligible ->
  fewest diverged -> seed-mean final VAL loss; agrees 85/85 with `analysis/scan_analysis.py`). `tools/reconcile.py`'s
  quick table omits the fewest-diverged tier (6/85 differ) — never quote it.
* Legacy results (read-only): `experiment_results_legacy_2026-09-18/`; legacy example-weighted repair in
  `analysis/legacy_repair/` (regenerate the parquet files with `analysis/repair_legacy.py`; `corrections.csv` is committed).
* Profiles: `profile_results_v2/` (repaired offline by `cycle_mean`; NOT re-measured with `empty_cache` off — Sven's
  CIFAR step is now ~190 ms vs the profiled 845 ms, see §5).

**Code (verified 2026-09-20 by executing `toy_1d_analysis`, `cifar_analysis`, `overparam_analysis` on copies: 0 error cells):**
loaders read schema 2, manifests, status, `finished/attempted`, test outcome columns, timing join. NOT yet done:
everything below.

**Known facts the analysis must respect**
* Selection uses validation only; test metrics are outcomes (`scan_analysis.configs` / `analysis_helpers.config_table`
  assert this). Report the CONFIRMATION seeds for headline numbers, not the tuning seeds (F25).
* Diverged = failed (`status == diverged` or the 10x rule); show `finished/attempted` on every table.
* Scans ran mostly on MIG A100-40GB slices, timing/diag/confirm partly on A100-80GB: Muon (bf16 Newton-Schulz), L-BFGS,
  SOAP/Shampoo/HIG and CIFAR runs are NOT bit-reproducible across GPU types (`bench/check_timing_join.py`); Sven and
  plain first-order MLP runs are. Timing claims use `<scan>_timing` only; check the calibration lines in the timing job
  logs (`sv3_campaign_scratch/logs/`, `bench/calibrate_step.py`) for host-load contamination before quoting MLP step times.
* KFAC dies deterministically on MNIST (cusolver eigh) -> `diverged` records; HIG never ran on CIFAR (parked, infeasible).
* `k = B` is a natural maximum, not a grid edge. CIFAR-CE Sven `rtol` still sits on its high edge (accepted, ~25 GPU-h to extend).

## 1. Housekeeping (short; blocks everything else)

Owner files: `analysis/style.py`, `analysis/analysis_helpers.py`, `analysis/scan_analysis.py`, notebooks (in place).
1. Register the new schema-2 columns so the "[style] column ... is being AVERAGED OVER" warnings disappear:
   outcomes (`val_final`, `val_best`, `val_best_index`, `val_last3_mean`, `test`, `test_acc`, `train_eval_final`) and
   provenance (`effective_loader_seed`, `ckpt_init_file`, `checkpoint_policy`, `ckpt_error`, ...). Do NOT add them to
   the hyperparameter allow-list.
2. Stale scan references: `cifar_analysis.ipynb` still reads the cut 1-seed `cifar10_resnet_paramFrac_scan_labelReg`
   -> point Fig-5 at `rebuttal_fig5_cifar_paramfrac_scan` (3 seeds); `kappa_analysis` -> MNIST only (CIFAR kappa cut);
   `critbatch_analysis`, `finetune_analysis` -> no fresh data: make them load the legacy root explicitly and say so in a
   header cell, or exclude them from `make_plots.sh`.
3. Method registry: colours/labels for `SGDm`; "Stochastic L-BFGS" label (C-B7); legends "+/- 1 std over seeds" via
   `analysis/paired.SEED_SPREAD_LABEL` (C-A6).
4. Run `./make_plots.sh` end to end (all 19 notebooks, in place; needs a compute node — cold loads are ~20 s per scan
   on Lustre), triage every error/warning cell, fix, re-run until clean. Plots go to `analysis/plots_v2/`.
Acceptance: all notebooks execute with 0 error cells and no "[style] ... AVERAGED OVER" warning; `tests/` green.

## 2. Headline numbers (the tables that go in the paper)

New helper module `analysis/headline.py` (+ tests) and one new notebook `analysis/headline_tables.ipynb`; then cells in
the five headline notebooks + `nanogpt_analysis`.
1. **Confirmation table** per scan: for each method's selected config (`bench/best_configs.json`) the mean +/- std
   over the 5 fresh seeds in `<scan>_confirm` of final val loss, final TEST loss, test accuracy (classification),
   `finished/attempted`; tuning-seed numbers beside them (the gap = selection optimism). Toy/polynomial: additionally
   across the 3 data seeds (F27) — report mean over data seeds and the between-instance spread.
2. **Paired differences vs Sven** (`analysis/paired.py`): per method, mean difference with t-interval, paired by model
   seed (same init and data order), on confirmation seeds.
3. **Tuning-budget disclosure** (`analysis/budget.py`): distinct-trajectory table and best-of-n curves per method.
4. **Efficiency**: steps / examples / synchronised training time (`train_times`) as separate axes; time-to-target at
   pre-declared targets (e.g. val loss of the median method's final value, and 2x / 0.5x of it), counting runs that
   never reach it; wall-time from `<scan>_timing` via `attach_standalone_times`; peak memory from `losses.peak_gpu_mem_mb`.
5. **Ranking summary** across scans (one table: rank of Sven and each baseline per scan, val and test).
Acceptance: numbers reproduce from a clean cache; selection never touches test (assertion holds); every table shows
`finished/attempted`; a unit test pins the confirmation-table computation on a synthetic fixture.

## 3. Spectrum and mechanism figures (F17, F19, C-A3, C-L4)

1. From `<scan>_diag` (no new code in the runner): full-width spectra over training for the best Sven config with the
   k cut, the rtol line and the float32 noise floor (`sv_noise_floor`); `utr` plots (|u_i . r| vs index: is the residual
   in the truncated tail?); `update_norm` / `resid_norm` / `sv_min_kept` vs step; rank actually used vs k.
2. `analysis/ckpt_tools.py` (C-L4, NEW, torch; heavy parts via `campaign/run_cpu_tests.sh` or a GPU job): load a
   checkpoint into the model (Hydra config from `{scan}/configs/`), reconstruct any step's batch with
   `sampler.batch_indices_for_run`, float64 Jacobian spectrum on (a) a FIXED probe set shared by all optimizers of a scan
   and (b) the full training pool for toy/polynomial (10,000 x ~600) and a few thousand examples for MNIST. Output: spectra
   along the trajectories of Sven AND the baselines (Adam, Muon, HIG) from the `log` checkpoints; distance from
   initialisation and parameter norms per optimizer (Codex Low 4).
   Test: on a tiny MLP the tool's spectrum equals `torch.linalg.svdvals` of the explicit Jacobian; the reloaded
   checkpoint reproduces the recorded val loss.
3. Figures: spectrum evolution on the probe set, Sven vs Adam vs HIG (toy, polynomial, MNIST label-reg).
Acceptance: figures regenerate from `make_plots.sh`; tool has tests; nothing reads the legacy truncated spectra.

## 4. Reviewer-specific figures

| reviewer point | data | deliverable |
|---|---|---|
| R1 crux: dataset-level overparameterisation P > N | `rebuttal_overparam_{toy_1d,polynomial,mnist}_scan` (extended lr grids) | loss and time-to-target vs P/N per method, fixed val/test sets (C-D3 makes them comparable now); Sven's rank vs P/N; note MNIST top point is N=50000 |
| R1 Q2: Fig-5 in main text | `rebuttal_fig5_cifar_paramfrac_scan` (3 seeds) + `profile_results_v2` param_fraction study | val/test loss AND accuracy vs param fraction with seed bands, `actual_param_fraction`, peak memory, step time; state that masks are resampled every step. Caveat: masked Sven was ~1.9x SLOWER per step in the smoke run (full capture does not get cheaper with a mask) — report honestly |
| R1: kappa = 1 vs 2 | `mnist_kappaScan_labelRegression` (210 runs, matched effective step 2 lr / kappa) | loss vs effective step for kappa 1/2/3, k = 32 (truncation binds) and k = 64 |
| R1: transformers | `exp_nanogpt_speedrun` (+timing/diag/confirm), `exp_gpt2_small_comparison` | nanoGPT: val/test vs steps and vs time, confirmation table. GPT-2: NEW notebook `gpt2_analysis.ipynb` — val/test vs step (`val_step`, `eval_step_idx`), per-lr Sven curves, wall-clock; framing = one seed, one epoch, k = B = 16, Sven 5.20 vs Muon 3.77 / AdamW 3.93 at 3x the time: a negative scaling data point, with the untuned k/B caveat |
| R2 Q1: batch-size sensitivity | `rebuttal_batchsize_polynomial_scan` (L-BFGS grid cut per O5, extended lrs) | best loss vs batch size per method; Sven k/B fractions |
| R2: CE classification, larger models | `mnist_scan_ce`, both CIFAR scans | headline tables (section 2). CIFAR: Sven is 9th/11 (label-reg) and 10th/11 (CE) under the corrected evaluation — state plainly; show val AND test accuracy |
| R2: Muon comparison | all scans (`muon_variant` recorded) | note the grouping rule (hidden matrices + flattened convs -> Muon, embeddings/head/1-D -> AdamW, `match_rms_adamw`); MuonW is now the strongest baseline on MNIST/CIFAR |
| R3 / R1: wall-time and memory | `<scan>_timing`, `profile_results_v2` | time tables from standalone runs; cost statement per backend (Gram: B x B eigh + capture, independent of k) |
| micro-batch / param-fraction (MLP) | 8 ablation scans | refresh `microbatch_analysis`, `paramfrac_analysis` |

## 5. Legacy-vs-fresh diff, docs, optional re-profile

1. `analysis/legacy_vs_fresh.ipynb` + a markdown table `analysis/WHAT_CHANGED.md`: per scan and method, legacy best
   (example-weighted repaired val loss from `analysis/legacy_repair/`, legacy selection) vs fresh (val, test,
   confirmation), rank changes, and the attributed cause (test-set selection removed; BN evaluation fixed; true
   polynomial; Muon grouping; extended grids; failure records — e.g. HIG's legacy half-crashed grid). Polynomial is a
   different target: compare ranks only.
2. Docs (C-Z1): `EXPERIMENTS.md` (seeds, splits, evaluation protocol, capture mode `full` + `empty_cache` off, new
   scans, cut scans, extension round, passes), `CLAUDE.md`/paper cost text (Gram backend cost is a B x B eigh plus
   capture, independent of k; O(kN|D|) describes only the classic randomized path), `analysis/RERUNS_NEEDED.md` launch log
   with both SHAs per phase, README pointers to `campaign/`.
3. OPTIONAL (needs a user go, ~2 h on an exclusive A100 node): re-run the optimizer profile
   (`bench/profile_serial.sbatch`) at the campaign code — `empty_cache` off changes Sven's profile numbers by up to 4.5x
   on CIFAR, so the existing profile tables are pessimistic for Sven's full-capture variants.

## 6. Orchestration

* One agent (or one small workflow) per work package; file ownership keeps them disjoint:
  WP1 housekeeping (style/helpers/notebook fixes) -> must land first;
  then in parallel: WP2 (`analysis/headline.py`, `headline_tables.ipynb`, headline notebooks),
  WP3 (`analysis/ckpt_tools.py`, `sv_diagnostics.py`, `comparisons.ipynb`),
  WP4a (overparam/batchsize/kappa/microbatch/paramfrac notebooks), WP4b (CIFAR + Fig-5 + nanoGPT + new GPT-2 notebook),
  WP5 (legacy diff + docs).
  Each: implement -> adversarial review (numbers recomputed independently from the jsonl for a sample) -> fix.
* Subagents on opus. The workflow engine runs (CPUs - 2) agents per workflow: on a small node launch one workflow per WP.
* Heavy work (notebook execution, torch) through `campaign/run_cpu_tests.sh` or on the compute node directly; never on a
  login node. Results roots are read-only for analysis; caches go to `experiment_results/_cache/` (guarded).
* Final gate: `./make_plots.sh` clean, `tests/` green, a reviewer agent spot-checks 10 table numbers against raw records.

## 7. Decisions (user, 2026-09-20) — all settled, nothing launched yet
1. **Headline = confirmation seeds**, with the tuning-seed numbers shown beside them (selection optimism visible).
2. **GPT-2: report as is** (one seed, one epoch, k = B = 16; Sven 5.20 vs Muon 3.77 / AdamW 3.93 at ~3x the time per
   run). k / batch-size exploration is deferred.
3. **CIFAR: report as is, no further investigation now.** Confirmation-seed facts to state: label-regression — Sven
   test acc 69.4% vs Adam 73.4 / MuonW 76.3 / SOAP 77.9, Sven fits the train subset well (train_eval 0.067) but
   generalises worse; cross-entropy — Sven 53.0% vs 73-78%, an optimisation failure (train_eval 0.82); 62 s/epoch vs
   5-18 s. BatchNorm was ON (`bn_mode: batch`: batch statistics in training, one running-stat update per step, eval
   with running stats) for every optimizer. The legacy "Sven ~ Adam on CIFAR" came from evaluating Sven with
   validation-batch statistics.
   **SUPERSEDED for cross-entropy (2026-09-21), by decision 5 below:** the `rtol` extension moved the CIFAR-CE Sven
   pick to k=128 / lr=0.5 / rtol=0.3 and the passes were re-run, so the CE facts to state are now **test acc 58.2%**
   vs 73-78%, val 1.3552, test loss 1.3391, train_eval **0.159** (still an optimisation failure, but far less extreme
   than the 0.82 above), 63.1 s/epoch. The label-regression facts and the BatchNorm paragraph are unchanged.
   EXPERIMENTS.md §5 and §9.3 carry the numbers of record.
4. **Re-profile: YES.** Re-run the optimizer step-time / memory profile (`bench/profile_serial.sbatch`, all
   `experiments/configs/profile_*.yaml`, ~2 h on one exclusive A100-80GB node) on the campaign code, because the
   09-17 profile was measured with the per-step `torch.cuda.empty_cache()` that made full-capture Sven up to 4.5x
   slower. Before launching: make sure `experiments/optimizer_profile.py` builds Sven with `empty_cache=False` (now the
   optimizer default - verify the profiler does not override it) and exports
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; write to a NEW root `profile_results_v3/` (keep v2 for the
   before/after table); run from a deploy snapshot; then point `analysis/profile_helpers.RESULTS_ROOT` / the four
   profile notebooks at v3 and add a v2-vs-v3 comparison cell. Exclusive nodes do not start under contention: submit
   and let it wait, or fall back to a non-exclusive GPU with the calibration microbenchmark
   (`bench/calibrate_step.py`) recorded at start and end.
5. **CIFAR-CE Sven rtol extension: YES.** `cifar10_resnet_ce_scan.yaml`: add `rtol` 0.03, 0.1, 0.3 (the optimum sits on
   the 1e-2 high edge; MNIST-CE prefers 0.1-0.3). Additive only: 2 k x 5 lr x 3 rtol x 5 seeds = +150 runs at ~0.45 h =
   ~70 GPU-h if the full lr grid is kept; to stay near the ~25 GPU-h quoted, restrict the new rtol values to the
   lrs around the optimum via a separate override item (`lrs=[0.05,0.1,0.5] k_values=[128]` -> 45 runs, ~20 GPU-h) —
   DEFAULT: the restricted 45-run version as an extra plan item, not a config-grid change, so counts/goldens do not
   move. Afterwards: `tools/select_best.py`, regenerate `campaign/plan_phase5.yaml`, and re-run timing / diag /
   confirm for CIFAR-CE Sven only if the selected config changed.

### Launch checklist for 4 and 5 (when the user says go)
* Both need GPUs but are independent of the analysis work packages (WP1-WP5 can start in parallel; WP2/WP4b
  re-read CIFAR-CE numbers at the end, and WP5's profile tables wait for v3).
* Item 5: add the plan item (or config values), tests green, commit, `tools/deploy_snapshot.sh`, launch with
  `tools/launch_campaign.py ... --snapshot <new> --list <list> --submit`; the queue must be empty of jobs from other
  snapshots for the same list (launcher guard).
* Item 4: one sbatch; results in `profile_results_v3/`.
