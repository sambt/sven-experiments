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
   `analysis/scan_analysis.py` (`Scan.best_sven` / `best_baseline`; rule: eligible -> fewest
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
