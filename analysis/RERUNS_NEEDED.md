# Runs needed to make the analysis robust and complete

Companion to `ANALYSIS_FIXES.md`. Everything the analysis code can fix has been fixed there;
this is the list of things that only new (or re-)runs can fix, found while doing so.
Numbers are as of 2026-09-17 from the local `experiment_results/`.

Ordered by how much of the story depends on them. Each item says what to run, roughly how
big it is, and which plot/table it unblocks.

## Cluster handoff (read first)

Everything below is ready to submit; nothing needs further analysis-side work. In order of
payoff:

1. **Item 0 -- AdamW / MuonW** (new baselines; `experiments/configs/*.yaml` already list `MuonW`,
   and `weight_decays` now defaults to the optimizer's own wd). Delete the old
   `*_optimAdamW_mseed*.jsonl` (+ `diag/*.npz`) files that have no `_wd` in the name, then
   resubmit each scan config: the dedup check skips every other run.
2. **Item 1 -- SV spectra**: the `sven` package must be at the commit with the full-spectrum
   logging (`SvenGram.step` records all B singular values). Delete the `k = B` Sven files of
   the four headline scans and resubmit with `k_values: [B]`.
3. **Item 2 -- timing**: regenerate `bench/best_configs.json` from the current
   `analysis/lib/scan_analysis.py` (`Scan.best_sven` / `best_baseline`; rule: eligible -> fewest
   diverged seeds -> seed-mean final val loss -> smallest k, largest rtol) and run
   `submit_timing_runs.sh` for every scan; at minimum MNIST-CE LBFGS and the new AdamW/MuonW.
4. Items 4-5 (missing seeds, more seeds) as budget allows.

After the runs: sync `experiment_results/` back and run `./make_plots.sh` from the repo root.
The analysis needs no code change for any of these; it prints a note wherever data is
still missing.

---

## 0. AdamW with weight decay ON: rerun AdamW in every scan (ANALYSIS_FIXES D29)

**Why.** `build_standard_optimizer` passed `weight_decay=0.0` explicitly to every optimizer,
overriding AdamW's PyTorch default of 0.01 -- so every "AdamW" run in the headline scans,
nanoGPT and critical-batch studies is bit-identical to Adam.

**Fix, done (experiment code).** `weight_decays` in a config now defaults to `[None]` =
"the optimizer's own default" (`experiment_utils.resolve_weight_decay`: AdamW 0.01,
everything else 0.0); an explicit number is still honoured. AdamW's `run_id` now always
carries `_wd<value>`, so the new runs do not collide with -- and get skipped as -- the old
wd=0 files. `submit_timing_runs.sh` passes the wd of the selected AdamW config.
Analysis: `weight_decay` is part of the baseline config key, so old wd=0 and new wd=0.01
AdamW runs are separate configurations.

**What to rerun** (4 lrs x seeds, one AdamW config each; delete the old
`*_optimAdamW_mseed*` files without `_wd` first, or they stay in the tables as an
Adam duplicate):

| scan | AdamW runs to redo | note |
|---|---|---|
| `toy_1d_scan`, `polynomial_scan`, `mnist_scan_ce`, `mnist_scan_labelRegression` | 20 each | + 5 standalone timing runs each once the best AdamW config is known |
| `exp_nanogpt_speedrun` | 12 | + 3 timing runs; this is the study where AdamW vs Sven is a tie |
| `exp_critbatch_nanogpt` | 36 | |
| `rebuttal_batchsize_polynomial_scan`, `rebuttal_overparam_*` | none | already have an explicit wd=0.01 AdamW alongside wd=0 |

**Muon -> `MuonW` (decided).** Muon's PyTorch default is `weight_decay=0.1` and it, too, was
forced to 0. Rather than change Muon, a `MuonW` optimizer was added (`experiment_utils`:
Muon at wd 0.1 on the 2-D parameters, its AdamW side-optimizer at AdamW's default 0.01 on
the rest -- everything at its own default), mirroring Adam / AdamW: the existing Muon
(wd = 0) runs stay valid and both variants are compared. `MuonW` is in `optimizers_standard`
of every scan config that has Muon, in `BASELINES`, and in `style.METHOD_COLORS`.

| scan | MuonW runs to add |
|---|---|
| `toy_1d_scan`, `polynomial_scan`, `mnist_scan_ce`, `mnist_scan_labelRegression` | 20 each (+ 5 timing) |
| `exp_nanogpt_speedrun` | 12 (+ 3 timing) |
| `exp_finetune_cifar_smallN`, `mnist_scan_brier`, `exp_gpt2_small_comparison` | if those studies are kept |
| `rebuttal_batchsize_polynomial_scan`, `rebuttal_overparam_*` | 4 lrs x seeds x N/B values each -- note these configs sweep `weight_decays: [0.0, 0.01]`, which MuonW would inherit; set `weight_decays: [null]` there for a MuonW-at-default run, or accept the sweep |

## 1. Singular-value spectra: rerun the `k = B` Sven slice with full-spectrum logging

**Why.** `SvenGram.step` / `SvenGramReg.step` recorded only the singular values above
`rtol * sigma_0` (the ones it inverted), so every stored spectrum ends exactly at rtol and
the tail of the per-epoch average is a survivorship average over the steps whose rank reached
that index (ANALYSIS_FIXES A11). The plots cannot show the part of the spectrum below rtol,
which is the part that says *why* rtol matters.

**Fix, done.** `sven/sven/opt/sven.py` now logs the full B-vector `sigma` (all
eigenvalues of the Gram matrix, before the k / rtol cut) in `svd_info["svs"]`;
`num_nonzero_svs` is unchanged (still the count used). Cost: B floats per saved step, same
as before at full rank. The classic (randomized-SVD) path cannot do this -- `pinv()` never
sees the values below rtol -- and its docstring now says so; all scans use the Gram path.
`sv_diagnostics.plot_epoch_spectra` prints a warning while it is fed truncated spectra.

**What to rerun.** Only the runs the spectrum plots read: Sven at `k = B`, every lr and rtol,
all seeds, in the four headline scans (the SV-rank and SVs-used plots read `num_nonzero_svs`
and are fine already):

| scan | k = B | lrs | rtols | seeds | runs | ~standalone time / run |
|---|---|---|---|---|---|---|
| `toy_1d_scan` | 32 | 4 | 3 | 5 | 60 | 85 s |
| `polynomial_scan` | 32 | 4 | 3 | 5 | 60 | 85 s |
| `mnist_scan_ce` | 64 | 4 | 4 | 5 | 80 | 230 s |
| `mnist_scan_labelRegression` | 64 | 4 | 4 | 5 | 80 | 240 s |

280 runs, ~14 GPU-hours sharded. A minimal set is `lr in {FOCUS_LR, 0.1, best lr}` at all
rtols (the lrs the scan notebooks and `comparisons.ipynb` actually plot), ~half of that.

**How.** `generic_scan` skips a run whose `{run_id}.jsonl` already exists, so either
(a) delete the `{run_id}.jsonl` + `diag/{run_id}.npz` of those runs first and rerun the
scan config restricted to `k_values=[B]` (same seeds => same trajectory; the timing reruns
reproduced the scan to 0.00e+00 relative deviation, so this is a pure replacement), or
(b) run into `experiment_results/<scan>_spectra/` and point `sv_diagnostics` there.
(a) is simpler and keeps one source of truth. Use `svd_info: full`, `svd_spectra_every: 20`
(the current defaults).

## 2. Standalone timing: regenerate the selection, rerun what changed (ANALYSIS_FIXES B20)

**Why.** Standalone (one-run-per-GPU) timings exist only for the *best* config of each
method, selected by `bench/select_best_configs.py` on the cluster. The selection rule has
changed twice (deterministic tie-break; diverged = failed with the fewest-diverged-first
ranking), so the timed config must be re-derived from `scan_analysis.best_*` and any changed
config re-timed.

**Known so far** (every other method in every scan still matches its timing run). This is
ANALYSIS_FIXES.md B20 -- the only headline-scan gap that a rerun, not the analysis, has to close:

| scan | method | timed config | best config now |
|---|---|---|---|
| `mnist_scan_ce` | LBFGS | lr=0.5, max_iter=2, history=2 | **lr=0.5, max_iter=1, history=2** |

5 runs (`std_bs64_mlp_width32_lr0.5_optimLBFGS_mi1_hs2_lsstrong_wolfe_mseed300{0..4}_lseed3000`
into `mnist_scan_ce_timing/`). Until then MNIST-CE has no standalone bar or wall-time curve for
LBFGS. `nanogpt` (`exp_nanogpt_speedrun_timing/`) is complete.

`bench/select_best_configs.py` is not in this checkout; whatever it does, it must call
`Scan.best_sven()` / `best_baseline()` from the current `scan_analysis.py` so the timed
configs and the plotted ones cannot drift apart again.

## 3. Sven wall time vs k, honestly

**Why.** `sven_walltime_vs_k.pdf` uses the sharded scan's times (the only ones that exist
for a k sweep) and is labelled so. With the Gram backend the `eigh` is B x B whatever k is,
so the plot is flat and only says "no k dependence"; that is a real result but the sharded
noise (+/- 3 s on 126 s) is as big as any effect.

**What to run, if the plot is kept.** Standalone Sven at the best (lr, rtol) over all k
(6 ks x 5 seeds = 30 runs per scan; 8 ks for MNIST), into the `<scan>_timing/` dirs.
Otherwise drop the plot and state the result in text.

## 4. Missing seeds (result file never written)

`n_missing` in the tables counts seeds with no result file (crash, time limit, never
submitted). Runs to (re)submit, by scan:

| scan | missing | where |
|---|---|---|
| ~~`cifar10_resnet_scan_labelRegression`~~ | ~~40~~ **0** | **Correction 2026-09-17: complete (290/290).** The 40 were an analysis artefact: `gram_capture` (chunked vs full, a memory-layout choice) was part of the config key and split the k=128 Sven configs into seed fragments. Fixed in `analysis_helpers.config_columns`. |
| `rebuttal_batchsize_polynomial_scan` | 24 | spread over configs; 104 configs ineligible, mostly LBFGS/KFAC divergence, not missing runs |
| `polynomial_scan` | 9 | KFAC |
| `toy_1d_scan` | 6 | HIG (5), KFAC (1) |
| `mnist_paramfrac_labelreg_scan` | 13 | f <= 0.25 at lr >= 0.5 (with the blow-ups, f <= 0.25 has no eligible config) |
| `mnist_paramfrac_ce_scan` | 7 | same pattern |
| `toy_1d_paramfrac_scan` / `polynomial_paramfrac_scan` | 3 / 14 | f <= 0.25 at lr >= 0.5 |
| `rebuttal_overparam_mnist_scan` | 3 | |
| `mnist_scan_labelRegression` | 1 | HIG |

Counts are against the grid size. (`config_table`'s `n_missing` only sees configs that
have at least one result file; a config with none is invisible to it, which is why its
paramfrac numbers are lower than the ones above.)

The paramfrac gaps are almost certainly crashes of the same blow-up that the 10x rule
flags; re-running them will just add `n_diverged`. Fine -- but then the f <= 0.25 points
need a smaller lr in the grid (0.01, 0.02) to exist at all.

## 5. Too few seeds where the result is a tie

| study | seeds | what it decides |
|---|---|---|
| `exp_nanogpt_speedrun` | 3 | AdamW 1.760 +/- 0.012 vs Sven 1.769 +/- 0.019: a tie at 3 seeds. 5+ seeds (2 more per config, 32 runs) to say anything. |
| `exp_critbatch_mnist` / `_nanogpt` | 2 | error bars on 2 seeds are not error bars; 5 seeds = 126 more runs per study |
| `mnist_microbatch_ce_scan`, `mnist_paramfrac_ce_scan` | 3 | the CE knob sweeps; 2 more seeds = 56 + 30 runs |

## 6. Critical batch: log what a McCandlish plot needs

The critical-batch notebooks plot *epochs* to target vs batch size; the standard quantity
is optimizer *steps* (= epochs x N/B), and the number of examples processed. Neither N nor
the steps per epoch is in the result records for the baselines (Sven has
`svd_summary.n_steps`). Add `n_train` and `steps_per_epoch` to every run's record in
`generic_scan._write_run` (trivial, no rerun needed for the *value* -- but the existing
records lack them, so either backfill from the config or rerun with item 5).

## 7. Result directories referenced by notebooks but not in this checkout (ANALYSIS_FIXES B16)

Sync from the cluster (or rerun): `rebuttal_baselines_toy_1d_scan` (no longer needed --
`baselines_analysis` now reads the headline scans), `cifar10_resnet_paramFrac_scan_labelReg`
(5 runs), `rebuttal_fig5_cifar_paramfrac_scan` (15), `exp_finetune_cifar_smallN` (240),
`cifar10_resnet_kappaScan_labelReg` (5), `cifar10_resnet_ce_kappaScan` (3). The CIFAR
paramfrac / kappa studies are 1 seed x few configs and would need item-4/5 treatment anyway.

## 8. A held-out test split

Every "best" is selected on the validation loss and reported on the same validation loss
(ANALYSIS_FIXES A10). With 4 lrs x 6-8 ks x 3-4 rtols per method the optimism is small, but
a reviewer can ask. Cheapest fix: for the best config of each method only, evaluate the
final model on the test split (MNIST/CIFAR have one; toy/polynomial: draw a fresh sample).
That is an evaluation pass on saved checkpoints if any exist, otherwise a rerun of the
~13 x 5 best-config runs per scan with `save_final=True` / a test-set evaluation added to
the loop.

## 9. After any rerun

`./make_plots.sh` (repo root) re-executes every notebook in place; the slim caches in
`experiment_results/_cache/` invalidate themselves on any file change, and so does the
profile cache. If runs were replaced in place (item 1), nothing else needs clearing.

## 10. Smaller things noticed

* **AdamW == Adam in every headline scan** until item 0 is run.
* **`n_params` / `n_train` / `n_val` are on records only from 2026-09-17** (C25): older
  records fall back to hard-coded P (593 / 673 / 27 562 / 11 181 642) with a printed
  warning. Any rerun (items 1, 4, 5) fixes this for the runs it touches; a full backfill
  would mean re-running everything, which is not worth it for a constant.

* `FOCUS_LR` in the scan notebooks (0.05 toy, 0.1 polynomial / MNIST-CE, 0.5 MNIST label-reg)
  is not the best lr on toy (0.1) or MNIST-CE (0.5). Not a rerun -- a notebook constant --
  but the SV section then shows a non-optimal lr; pick one convention (best lr, or a fixed
  lr and say why).
* `rebuttal_batchsize_polynomial_scan`: LBFGS diverges on 439 of its runs and KFAC on 41;
  at B >= 128 several methods have no eligible config. If batch-size scaling of the
  baselines matters, they need a lower-lr grid at large B.
* The scan-time inflation is uneven (Sven shared its GPU with LBFGS/KFAC shards): any new
  scan intended for timing should run `NPROC=1` from the start rather than be re-timed.

---

## Launch log -- 2026-09-17 (cluster), `submit_reruns_2026-09-17.sh`

64 jobs (SLURM 46944006-46944091; full list in `slurm_logs/submit_reruns_2026-09-17.log`). Superseded
runs were **moved aside, not deleted**: `<scan>/_adamw_wd0/` (the wd=0 AdamW files, items 0) and
`<scan>/_spectra_truncated/` (the k = B Sven files, item 1), each with its `diag/`; the loader reads only
top-level `*.jsonl`, so dedup re-runs them. Re-running the launcher is safe (dedup; move-asides idempotent).

| item | what was launched |
|---|---|
| 0 | `mode=standard optimizers_standard=[AdamW,MuonW(,KFAC on toy/poly)]` on the 4 headline scans; nanoGPT `mode=standard` (config now 5 seeds, so AdamW/MuonW x 5 and Muon/SOAP seeds 5003-4); critbatch-nanoGPT `mode=standard` (AdamW x 5 seeds). Overparam / batch-size scans untouched (explicit wd sweep already there; no MuonW added -- decision left open). |
| 1 | `mode=svd k_values=[B]` on the 4 headline scans (toy/poly one job each, MNIST per seed) with sven `ca8742b` (full-spectrum logging; editable install verified). |
| 2 | `timing_serial_RERUNS` (46944091): exclusive node, `RESELECT=1` regenerates `bench/best_configs.json` with the current `scan_analysis` rule, then times every best config with dedup (only the changed ones actually run: MNIST-CE LBFGS mi=1, AdamW-wd / MuonW everywhere, nanoGPT seeds 5003-4). `--dependency=afterany` on the 20 headline + nanoGPT jobs. `timing_serial.sbatch` now passes `weight_decays=[wd]` for AdamW/Muon/MuonW. |
| 4 | Audit against the cluster's files: the CIFAR label-reg "40 missing" and batch-size "24 missing" were a stale local sync (both grids are complete here); the real gaps are deterministic failures: KFAC 0/20 on both MNIST scans (eigh), 19/20 toy, 11/20 poly; HIG never finishes at lr >= 0.5 (toy, MNIST label-reg). Resubmitted only where a file can appear: KFAC on toy/poly, HIG at lr <= 0.1 on toy (5) and MNIST label-reg (1). JD is 20/20 everywhere. |
| 5 | 5 seeds now in `exp_nanogpt_speedrun`, `exp_critbatch_nanogpt`, `exp_critbatch_mnist`, `mnist_microbatch_ce_scan`, `mnist_paramfrac_ce_scan` (configs edited; new seeds submitted, per seed for critbatch). |
| 7 | CIFAR ablations submitted for the first time, one run per job at NPROC=1 (~2 h each, full-J capture): kappa label-reg (5), kappa CE (3), paramfrac label-reg (5), paramfrac CE (6), Fig-5 (15 = 5 fractions x 3 seeds). **CE set point re-derived** from the BN-fixed headline scan: k=128, lr=0.1, rtol=1e-2 (was the pre-Gram k=64 / 1.0 / 1e-3; label-reg's k=64 / 1.0 / 1e-3 was confirmed). Still 1 seed for kappa/paramfrac (the configs' choice; a 3-5 seed version is a one-line edit + resubmit). `exp_finetune_cifar_smallN` and GPT-2 not launched (separate decision). |

Decided 2026-09-17 (user): **item 3 dropped** -- under the hooks-based Gram backend the eigenproblem is
B x B whatever k is, so truncation does not change wall time; the flat plot is stated in text. **Item 8
(held-out test evaluation) noted, deferred** -- needs a test-set pass at the end of the training loop for the
best configs; not now. **Launched in part 2** (`submit_reruns_2026-09-17_part2.sh`): `exp_finetune_cifar_smallN`
(4 N x {svd, standard}), MuonW at its default wd=0.1 in the three overparam scans and the batch-size scan
(`weight_decays=[0.1]` explicit), and GPT-2-small via `./submit_gpt2.sh` (partitions widened). Still open:
item 6 backfill, the lower-lr grid for paramfrac f <= 0.25.
After everything finishes: sync `experiment_results/`, then `./make_plots.sh`.

---

# Launch log -- the robustness campaign, 2026-09-18 .. 2026-09-20

**Everything above this line is historical.** It was written on 2026-09-17 against the results
that are now frozen read-only at `experiment_results_legacy_2026-09-18/`. The campaign below
supersedes items 0-7 and 10 (new baselines at their own weight decay, full-spectrum logging,
standalone timing, missing/extra seeds, `n_params`/`n_train`/`steps_per_epoch` on every
record, the CIFAR ablations); item 8, "a held-out test split", is **done** -- three splits with
fixed sizes are now the foundation of every scan, selection uses validation only and test is
an outcome. What each scan actually ran is in `EXPERIMENTS.md`; the plan and the decisions are
in `campaign/CONTRACTS.md` and `campaign/CAMPAIGN_STATUS.md`.

Results root: `/n/labstore01/LABS/anon_lab/Users/anon/sven_experiments` (repo symlink
`experiment_results`), started EMPTY on 2026-09-18 ~20:55 EDT. Pool logs under
`/n/labstore01/LABS/anon_lab/Users/anon/sv3_campaign_scratch/logs/`. Every phase ran from
an exported snapshot, `/n/labstore01/LABS/anon_lab/Users/anon/sv3_deploy/<sv3sha8>_<svensha8>/`,
never from the working tree; `git_dirty` is `false` on all 24,824 records (19:10 EDT; the
total still rises with the in-flight row 5). Both repos on branch
`robustness-campaign`. SHAs and job ids below are read back from the records' provenance
blocks, not from the launch commands.

The **date** column is the span of the phase's own records (`start_time` of the first to
`end_time` of the last), converted to **EDT** throughout -- the records store UTC, which is
EDT + 4 h. `EXPERIMENTS.md`'s phase table gives the same spans in UTC; if the two ever
disagree, one of them has been hand-edited.

| # | date (EDT) | phase | snapshot (sv3 + sven) | SLURM jobs | runs | outcome |
|---|---|---|---|---|---|---|
| 1 | 09-18 20:02 -> 09-19 22:10 | main campaign: 6 headline scans + nanoGPT (P0), 3 overparam + Fig-5 + batch-size (P1), kappa + 8 micro-batch/param-fraction scans (P3) | `2c6faf59` + `203a4e61` | CIFAR Sven 47080825/26/27/28/29/32; CIFAR baselines 47080834/35; nanoGPT 47080839; MIG combined list 47080843/44 (all four MLP headline scans) then, after the 03:55 reorder, 47143141/45 (`all_mlp_mig_v2` = P1 -> P3 -> MNIST HIG last); MLP A100 overflow 47080846/48/49/50 and 47274379/80 (the MNIST-CE HIG tail); Fig-5 47080858/60 -- **21 jobs with records** | **15,735** | COMPLETE, reconcile clean, 0 oom/error. MIG jobs hit the 12 h wall twice (expected, resubmitted with the same command). 47080843/44 were cancelled at 03:55 on 09-19 because MNIST HIG at NPROC 6 was blocking the MIG lane ahead of P1; ~48 partial HIG runs lost, stale claims expired after 10 min and were retaken automatically. Report: `campaign/reconcile_2026-09-19.txt` |
| 2 | 09-19 23:34 -> 09-20 03:48 | C-B3 grid-extension round: additive points on 9 scans, every one closing an edge the reconcile flagged | `62e5105e` + `203a4e61` | MIG 47322040/42; CIFAR baselines 47322047/52; MLP headline overflow 47322064/67/70 (toy, polynomial, MNIST-CE); P1 overflow 47322076/79/82 (the three overparam + batch-size scans) -- 10 jobs | **+7,480** | COMPLETE, reconcile clean. Additive only: existing `run_id`s unchanged, the 15,735 finished runs deduped out by their done markers, **0 `_stale/` directories created**. In-plan total 23,215 |
| 3 | 09-20 01:49 -> 09-20 19:13 | `exp_gpt2_small_comparison` (re-admitted by the user 09-19 22:40): 1 seed, 1 epoch = 13,125 steps at B = 16, k = B = 16, step-based evaluation every 500 steps | `e5b6fb77` + `203a4e61` | 47330243-65 -- 10 jobs, NPROC 1, one A100-80GB per run | **29** | COMPLETE: 29 of 29 `ok`, 0 diverged, reconcile `0 run(s) to do` (the last SOAP run, lr 3e-3, finished 09-20 19:13 EDT; an earlier pass of this log recorded it as still in flight). Smoke first (GREEN: Sven 2.52 s/step = 9.2 h/run, 36,050 MB peak; AdamW 2.8 h, Muon 2.9 h, SOAP 4.1 h), then Sven's lrs were extended to [0.02,0.05,0.1,0.5,1.0] because lr 0.5 was unstable in the smoke |
| 4 | 09-20 04:00 -> 09-20 10:39 | phase 5, the three result-dependent passes over all 7 headline scans: `<scan>_timing` (425), `<scan>_diag` (425), `<scan>_confirm` (725) | `b8fadc6f` + `203a4e61` | MLP MIG 47337921 (diag) / 47337922 (confirm); heavy A100 47337923/24 (CIFAR + nanoGPT diag) and 47337927/30 (CIFAR + nanoGPT confirm); MLP A100 47337931 (MNIST-CE diag) / 47337933 (MNIST confirm); **one serial timing job per scan**, 47337934 (CIFAR-CE), 47337935 (CIFAR-label-reg), 47337936 (nanoGPT), 47337937 (MNIST-CE), 47337938 (MNIST-label-reg), 47337939 (polynomial), 47337941 (toy) -- 15 jobs | **1,575** | COMPLETE, reconcile clean. Launched only after the extension round reconciled clean and `tools/select_best.py` + `tools/gen_phase5_plan.py` had been re-run. Timing jobs were non-exclusive with `bench/calibrate_step.py` at the start and end of each job so host-load contamination is detectable after the fact |
| 5 | 09-20 18:37 -> IN FLIGHT | `p2_cifar_ce_rtol`: the approved CIFAR-CE Sven `rtol` extension, **off-grid by design** -- `mode=svd k_values=[128] lrs=[0.05,0.1,0.5] rtol=[0.03,0.1,0.3]` as a plan item rather than a config-grid edit, so run counts, `campaign/grid_counts.md` and `tests/golden/` do not move (committed as `f0f89b2`) | `f0f89b24` + `203a4e61` | 47394881-47394885 -- 5 jobs so far (launched by the GPU-items track) | **45** (7 done at 19:21) | running. `cifar10_resnet_ce_scan`: expected 740 -> 785, 747 on disk at 19:21 (all `ok`, none diverged under either definition); reconcile read `35 run(s) to do; incomplete: cifar10_resnet_ce_scan` at 19:05, and that number falls as the runs land. Every count for this one scan is a snapshot until it finishes; every other scan is final. Purely additive: no existing `run_id` or `run_hash` moves. **If it changes the selected CIFAR-CE Sven configuration**, re-run `select_best.py` -> `gen_phase5_plan.py` -> the three passes, for CIFAR-CE Sven only |

**Verified after the fact (2026-09-20 18:00-19:10 EDT, this analysis phase):**

* `tools/reconcile.py --all campaign/plan_campaign.yaml` -> `21 scan(s), 0 run(s) to do;
  incomplete: none`; 23,215 expected, 21,813 `ok`, 1,402 `diverged`, **0 `oom`, 0 `error`, 0
  `started-only`, 0 `stale-hash`, 0 `jsonl-only`, 0 `never-started`**. (Re-run at 19:05 after
  row 5 landed its first records: `35 run(s) to do; incomplete: cifar10_resnet_ce_scan`,
  expected 785, 745 on disk. Nothing else moved.)
* **Two failure counts, and they are not interchangeable.** The 1,402 above is
  `status == "diverged"`, the lifecycle count that decides retries. The analysis definition
  (`analysis/style.is_diverged`: recorded, or non-finite, or final val > 10x `val[0]`) counts
  **2,553** over the same 21 scans, and that is the one selection and every table use. Of the
  1,402 recorded, 1,147 carry a `diverged_at_step` (`DivergedError`) and **275 do not** -- 243
  K-FAC `_LinAlgError` and 32 Sven `RuntimeError` from the masked-Gram guard, both mapped to
  `diverged` by `_classify_failure`, which is why `status: "error"` is 0. Per-scan and
  per-method figures for both counts: `EXPERIMENTS.md` §7.
* `tools/reconcile.py --all campaign/plan_phase5.yaml` -> `21 scan(s), 0 run(s) to do`;
  1,575 runs across the 21 companion directories -- of which **20 are `diverged`** (21 under
  the wide rule), 13 of them `polynomial_scan_confirm`'s 15 L-BFGS runs. The passes are not
  failure-free; see `EXPERIMENTS.md` §4.
* `tools/reconcile.py --all campaign/plan_gpt2.yaml` -> `1 scan(s), 0 run(s) to do; incomplete:
  none`; 29 expected, **29 `ok`**, 0 diverged (re-run 19:12 EDT; it read 28 `ok` / 1
  `claimed-live` an hour earlier).
* `bench/check_timing_join.py` -> **425 timing runs, 425 joined onto their scan by `run_id`
  AND `run_hash`, 0 missing.** Trajectories: nanoGPT bit-identical (median 0.00e+00), toy
  4.5e-07, polynomial 8.7e-09, MNIST-CE 5.7e-05, MNIST-label-reg 8.3e-03, CIFAR 2.5-3.1e-02
  (relative, median). Deviations are confined to methods that are not bit-reproducible across
  GPU types -- Muon's bf16 Newton-Schulz, L-BFGS's `strong_wolfe` line search, SOAP / Shampoo /
  HIG decompositions at 1e-9-scale losses, CIFAR's cuDNN kernel selection -- because the scans
  ran mostly on A100-40GB MIG slices and the passes on A100-80GB. Hashes match, so these are
  the same experiments.
* `bench/best_configs.json` (schema 2, rule `full`, generated 2026-09-20T03:58:14-0400) agrees
  with `analysis/lib/scan_analysis.py` on all **85** (scan, method) picks; `tools/reconcile.py`'s
  quick table differs on 6 of the 85 because it omits the fewest-diverged tier.
* No `_stale/` and no `attempts/` directory exists under any of the 43 result directories.

**Still open after the campaign** (tracked in `campaign/ANALYSIS_PLAN.md` §7, not here):

1. **CIFAR-CE Sven `rtol` sat on its top edge** (1e-2) — **now being extended**, see row 5 of
   the table above. Afterwards: re-run `select_best.py`, regenerate `plan_phase5.yaml`, and
   re-run timing/diag/confirm for CIFAR-CE Sven **only if its selected configuration changed**.
2. **`toy_1d_scan` Sven lr is on the new bottom edge** (0.01 after the extension round).
   `k:EDGE-HIGH` on six scans is `k = B`, a method boundary, and is deliberately not extended.
3. **The Fig-5 set point was never re-derived.** The scan ran (15/15) on the pre-Gram classic
   k = 64 / lr = 1.0 / rtol = 1e-3 while the config still carries its
   `SET POINT, STILL TENTATIVE` marker; the BN-fixed headline optimum is k = 128 / lr = 0.5 /
   rtol = 1e-3. Re-running it at the headline set point is 15 runs, ~5.5 GPU-h.
4. **`profile_results_v2` needs re-measuring** at the campaign code into `profile_results_v3/`:
   it was profiled with the per-step `torch.cuda.empty_cache()` that is now off, which makes it
   up to 4.5x pessimistic for Sven's full-capture variants. Approved; ~2 h on one exclusive
   A100-80GB node.
5. **Parked, config edits standing:** `exp_finetune_cifar_smallN` (408 runs; all eight
   `p3_finetune` items in `campaign/plan_campaign.yaml` are `enabled: false`) and the 200
   enumerated JD + HIG runs on the two CIFAR configs (`p3_cifar_jd_hig`, all items disabled;
   HIG on ResNet18 needs a full per-sample Jacobian, ~50 GPU-h per scan).
6. **One timing record is not its scan's trajectory:** `mnist_scan_labelRegression`, SOAP at
   lr 0.01, mseed 3001, ends at 0.901 in the scan and 184.8 in the timing pass. Take step times
   from `<scan>_timing` and loss values from the scan or the confirmation pass -- never loss
   values from a timing record.

After any rerun: `tools/reconcile.py`, then `tools/select_best.py --require-complete`, then
`tools/gen_phase5_plan.py`, then `./make_plots.sh` (the slim caches under
`experiment_results/_cache/` invalidate themselves on any file change).
