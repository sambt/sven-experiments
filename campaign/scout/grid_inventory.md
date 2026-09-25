Read: `CHANGES_NEEDED.md` (full), all 37 non-profile configs + `dataset/`, `model/` groups, `generic_scan.py`, `experiment_utils.process_hparam_config`, all 9 launchers, `EXPERIMENTS.md`, `analysis/RERUNS_NEEDED.md`. **Verified by running:** grid arithmetic via a script mirroring `generic_scan.py:417-462, 605-612, 683-689, 761-766, 842-846, 905` (`/tmp/claude-66176/.../scratchpad/{count,delta2}.py`); `DRY=1` runs of all four `submit_fresh_suite.sh` groups and both rerun launchers; `squeue`; `sinfo`; per-scan top-level `*.jsonl` counts; `best_configs.json`.

## 1. Per-config grid inventory (exact, from configs)

Counts = runs **including all seeds**, for every family the config carries. "Sven" = `svd_grid`; families with `†` are carried in the config but never launched by any launcher.

| config | model · n_params | dataset | ep | B | seeds | Sven | first-order | 2nd-order | LBFGS | Polyak | JD | HIG | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `toy_1d_scan` | MLP 1→[16×3]→1 · 593 | toy 1-D | 20 | 32 | 5 | 360 | Adam/AdamW/SGD/RMSprop/Muon/MuonW = 120 | SOAP/Shampoo/KFAC = 60 | 135 | 5 | 20 | 80 | **780** |
| `polynomial_scan` | MLP 6→[16×3]→1 · 673 | rand. poly | 20 | 32 | 5 | 360 | 120 | 60 | 135 | 5 | 20 | 80 | **780** |
| `mnist_scan_labelRegression` | MLP 784→[32×3]→10 · 27,562 | MNIST | 20 | 64 | 5 | 640 | 120 | 60 | 135 | 5 | 20 | 80 | **1060** |
| `mnist_scan_ce` | same · 27,562 | MNIST | 20 | 64 | 5 | 640 | 120 | 60 | 135 | 5 | 20 | 80 | **1060** |
| `cifar10_resnet_scan_labelRegression` | ResNet18 · 11,181,642 | CIFAR-10 | 20 | 128 | 5 | 90 | Adam/SGD/RMSprop = 60 | — | 135 | 5 | 20† | 80† | **290** (390 w/ JD+HIG) |
| `cifar10_resnet_ce_scan` | ResNet18 · 11,181,642 | CIFAR-10 | 20 | 128 | 5 | 90 | 60 | — | 135 | 5 | 20† | 80† | **290** (390) |
| `toy_1d_microbatch_scan` | 593 | toy | 20 | 32 | 5 | 120 (mb 1–32 × 4 lr) | — | — | — | — | — | — | **120** |
| `toy_1d_paramfrac_scan` | 593 | toy | 20 | 32 | 5 | 100 (pf 5 × 4 lr) | — | — | — | — | — | — | **100** |
| `polynomial_microbatch_scan` | 673 | poly | 20 | 32 | 5 | 120 | — | — | — | — | — | — | **120** |
| `polynomial_paramfrac_scan` | 673 | poly | 20 | 32 | 5 | 100 | — | — | — | — | — | — | **100** |
| `mnist_microbatch_labelreg_scan` / `_ce_scan` | 27,562 | MNIST | 20 | 64 | 5 | 140 each (mb 7 × 4 lr) | — | — | — | — | — | — | **140** ea |
| `mnist_paramfrac_labelreg_scan` / `_ce_scan` | 27,562 | MNIST | 20 | 64 | 5 | 100 each | — | — | — | — | — | — | **100** ea |
| `mnist_kappaScan_labelRegression` | 27,562 | MNIST | 20 | 64 | 5 | 15 (κ 1,2,3 × 1 lr) | — | — | — | — | — | — | **15** |
| `cifar10_resnet_kappaScan_labelReg` | 11.18M | CIFAR | 20 | 128 | **1** | 5 (κ 1,1.5,2,2.5,3) | — | — | — | — | — | — | **5** |
| `cifar10_resnet_ce_kappaScan` | 11.18M | CIFAR | 20 | 128 | **1** | 3 (κ 1,2,3) | — | — | — | — | — | — | **3** |
| `cifar10_resnet_paramFrac_scan_labelReg` | 11.18M | CIFAR | 20 | 128 | **1** | 5 | — | — | — | — | — | — | **5** |
| `cifar10_resnet_ce_paramFrac_scan` | 11.18M | CIFAR | 20 | 128 | **1** | 6 | — | — | — | — | — | — | **6** |
| `rebuttal_overparam_toy_1d_scan` | 593 | toy, B=N | 200 | N | 5 | 180 | 180 (wd sweep: AdamW/Muon/MuonW ×2) | 60 | 135 | 5 | — | — | **560 ×4 N = 2240** |
| `rebuttal_overparam_polynomial_scan` | 673 | poly, B=N | 200 | N | 5 | 180 | 180 | 60 | 135 | 5 | — | — | **560 ×4 = 2240** |
| `rebuttal_overparam_mnist_scan` | 27,562 | MNIST sub | 20 | 64 | 5 | 60 | 180 | 60 | 135 | 5 | — | — | **440 ×6 = 2640** |
| `rebuttal_batchsize_polynomial_scan` | 673 | poly | 20 | 6 values | 5 | 360 | 1080 | 360 | **810** | 30 | — | — | **2640** |
| `rebuttal_fig5_cifar_paramfrac_scan` | 11.18M | CIFAR | 20 | 128 | 3 | 15 | — | — | — | — | — | — | **15** |
| `exp_nanogpt_speedrun` | nanoGPT · 826,368 | shakespeare | 50 | 64 | 5 | 20 | AdamW/Muon/MuonW/SOAP = 80 | — | — | — | — | — | **100** |
| `exp_critbatch_nanogpt` | 826,368 | shakespeare | 40 | 6 values | 5 | 120 | AdamW 90 | — | — | — | — | — | **210** |
| `exp_critbatch_mnist` | 27,562 | MNIST | 20 | 7 values | 5 | 105 | Adam 105 | — | — | — | — | — | **210** |
| `exp_finetune_cifar_smallN` | ResNet18-pretrained · 11.18M | CIFAR sub | 30 | 64 | 3 | 12 | AdamW/SGD/Muon/MuonW = 48 | — | — | — | — | — | **60 ×4 N = 240** |
| `exp_gpt2_small_comparison` | GPT-2-small untied · ~163M | FineWeb-edu | 1 | 16 | 1 | 3 | AdamW/Muon/MuonW/SOAP = 16 | — | — | — | — | — | **19** |

**Total for a full relaunch as the configs stand: 15,968 runs** (15,768 excluding the never-run CIFAR JD/HIG). Inheriting configs (`defaults: [<parent>, _self_]`, no overrides — pure aliases for a separate results dir): `toy_1d_scan_timing`, `polynomial_scan_timing`, `mnist_scan_ce_timing`, `mnist_scan_labelRegression_timing`, `exp_nanogpt_speedrun_timing`. **There is no `cifar10_*_timing.yaml`** — phase-5 timing on CIFAR needs two new configs.

## 2. Launcher groups (DRY=1 verified)

All go through `submit_rebuttal_parallel.sh`: `--partition=iaifi_gpu_priority,iaifi_gpu,gpu`, `--time=12:00:00`, 1 node, 1 GPU, 8 CPUs, 48 GB, NPROC processes each running `specs[shard_id::n_shards]`.

| group | configs | job split | NPROC | jobs |
|---|---|---|---|---|
| `headline` | 6 | toy/poly: 4 jobs each (svd / FIRST / SECOND / LBFGS); MNIST: svd **per seed** (5) + 3 or 2 standard; CIFAR: svd per seed + LBFGS per seed + 1 CORE_FIRST | 6 / 4 / 2 | **45** |
| `ablations` | 13 | one Sven-only job per config | 6 / 4 / 2 | **13** |
| `rebuttal` | 5 | overparam: 4 jobs × each n_data (14 n_data values); batchsize: 3 + LBFGS **per batch size** (6); fig5: 1 | 6 / 4 / 2 | **66** |
| `tier3` | 5 | nanoGPT/critbatch: svd+standard; finetune: 2 × 4 n_data; critbatch_mnist: 1 | 2 / 4 | **13** |
| `submit_gpt2.sh` | 1 | `--wrap`, 3 Sven shards + 6 standard shards, own 64 GB/12 h | 1 | **9** |
| `submit_reruns_2026-09-17.sh` | — | one-off; `aside()` moves superseded jsonl+npz into `<scan>/_adamw_wd0/`, `_spectra_truncated/`; chains `bench/timing_serial.sbatch` via `--dependency=afterany` | 1–4 | **63** |
| `..._part2.sh` | — | finetune + MuonW backfill + `submit_gpt2.sh` | 2–6 | **23** |
| `bench/timing_serial.sbatch` | 5 `_timing` configs | ONE `--exclusive` job, 36 h, all 54 (method, best-config) pairs serially, `RESELECT=1` re-derives `best_configs.json` | 1 | **1** |
| `submit_timing_runs.sh` | same | one job per (scan, method) instead, `EXCLUSIVE=1` optional | 1 | 54 |
| `bench/relaunch_cifar_k128.sh` | 2 CIFAR | 2 scans × 5 seeds × 3 lrs, one run per job | 1 | 30 |

**Full relaunch = 137 SLURM jobs** (headline+ablations+rebuttal+tier3) **+ 9 GPT-2 + 1–54 timing**.

**Compute facts (sinfo, verified):** `iaifi_gpu` and `iaifi_gpu_priority` are the *same* 8 nodes `holygpu8a271xx–274xx`, each **4× A100-SXM4-80GB**, 64 CPUs, 503 GB, **TIMELIMIT 3-00:00:00**. `gpu` adds A100-80GB and A10 nodes, also 3 days. `gpu_test` is **A100 MIG `3g.20gb` slices, 8 per node, 12 h, several nodes idle** — fine for toy/poly/MNIST MLPs, marginal for CIFAR (`gram_capture: full` materialises a 128×11.18M fp32 Jacobian ≈ 5.7 GB per shard), useless for GPT-2 (~33 GB). The 12 h in the launchers is self-imposed, not a partition limit — the single biggest lever for "fewer, longer jobs". `squeue` right now: **9 running `wrap` jobs = `submit_gpt2.sh`** (2 h in; these carry the F4 val⊂train contamination) and `timing_serial_RERUNS` pending on `QOSMaxNodePerUserLimit`. No `exp_finetune` jobs are running (its dir already holds 240 jsonl, i.e. complete and F3-contaminated).

## 3. Growth under CHANGES_NEEDED (run-count deltas)

| change | delta | note |
|---|---|---|
| **C-B2** SGDm | **+568** | only the 11 scans that launch SGD: +20 each headline/CIFAR, +80/+80/+120 overparam, +120 batchsize, +48 finetune |
| **C-B1** MuonW in `FIRST` | **+0** | `MuonW` is already in every relevant `optimizers_standard`; `submit_fresh_suite.sh:49` just omits it — a launcher bug, not a grid change. On CIFAR, Muon/MuonW are absent from the config entirely (O6 default: keep them out) |
| **C-B4** AdamW lr × wd {0, .01, .1} | **+280** | +40 each on toy/poly/MNIST×2; **+60 each on the two CIFAR scans, where AdamW is currently absent** |
| **C-X1** κ retuning | **≈ +160** | MNIST 15 → ~120 (3 κ × 4 lr × 2 truncation slices × 5 seeds); CIFAR-labelReg 5 → ~40, CIFAR-CE 3 → ~24 (1 seed) |
| **C-X2** critbatch `k_fractions {0.25,0.5,1.0}` | **+450** | MNIST Sven 105 → 315; nanoGPT Sven 120 → 360 |
| **O5** LBFGS in batch-size scan | **−720** | 810 → 90 (6 B × 3 lr × 5 seeds, `max_iter`/`history` fixed at headline best) |
| **C-B3** edge extensions, ≤2 half-decade rounds | **≈ +400–700** | per (method, scan, round): 5 runs for a first-order lr (HIG: ×4 τ = 20; Sven: ×k×rtol, e.g. CIFAR 1 lr × 2 k × 3 rtol × 5 = 30/round ≈ 60 GPU-h) |
| **Phase 5** three passes | **≈ +1,500** | ~81 (scan, method) best configs × 5 seeds × {timing, diagnostics, confirmation} ≈ 1,215, **+300** for toy/poly × 2 extra data seeds. Diagnostics add `checkpoints: log` + dense spectra; timing needs 2 new CIFAR `_timing` configs |
| **net** | **≈ 18,600–18,900 runs** | 15,968 − 720 + 568 + 280 + 160 + 450 ≈ **16,700** for the scans, + extensions + phase 5 |

## 4. Stale / not part of the paper suite

| config | status |
|---|---|
| `mnist_microbatch_scan.yaml` | **stale.** `EXPERIMENTS.md:236` lists it under "Not re-run" (legacy, `lrs: [100.0]`); in no launcher; no results dir. Superseded by `mnist_microbatch_{ce,labelreg}_scan` |
| `rebuttal_mnist_batchk_probe.yaml` | **stale.** `EXPERIMENTS.md:239` "a one-off batch/k diagnostic that already served its purpose"; in no launcher; no results dir |
| `mnist_scan_brier.yaml` | **optional, 960 runs.** In no launcher; no results dir. `RERUNS_NEEDED.md:70` treats it as conditional ("if those studies are kept"). Cheap to keep, but it is a second full MNIST headline grid — recommend an explicit keep/drop decision |
| `exp_gpt2_small_comparison.yaml` | **decide.** 19 runs but ~9 h each and 33 GB; blocked on C-D2 rebuild; currently running with contaminated tokens (O1) |
| `submit_paper_all.sh`, `submit_rebuttal_all.sh`, `submit_rebuttal.sh`, `submit_seed_sweep.sh`, `submit_experiment.sh` | **superseded.** `submit_rebuttal_all.sh` references three configs that no longer exist (`rebuttal_baselines_{toy_1d,polynomial,mnist}_scan.yaml`); `submit_experiment.sh` activates a conda `jax` env. Delete or clearly mark them, or the campaign will re-submit the wrong thing |
| `profile_*.yaml` (7) | out of scope per brief; `profile_results/`, `profile_results_v2/` are repaired **offline** by C-T2 |

## 5. Inconsistencies the campaign must resolve

1. **No launcher ever runs `mode=jd` or `mode=hig`.** `submit_fresh_suite.sh:53-66` only emits `mode=svd` and `mode=standard`, yet JD (20/scan) and HIG (80/scan) are in `best_configs.json` for toy/poly/MNIST×2 and in the cost table of `CHANGES_NEEDED.md:438`. They exist only because of earlier ad-hoc launches. A "full relaunch" as written **silently drops 400 runs**.
2. **MNIST-CE and CIFAR lose whole baseline families.** `submit_fresh_suite.sh:79` uses `CORE_FIRST` and omits the `$SECOND` job for `mnist_scan_ce`, so AdamW, Muon, MuonW, SOAP, Shampoo and KFAC are never submitted — although the config lists all 11 and `best_configs.json` has SOAP/Shampoo for that scan. CIFAR (`:87`) gets `CORE_FIRST` only, so AdamW/Muon/SOAP/Shampoo/KFAC have **never** run on ResNet18. C-B4 forces AdamW onto CIFAR; decide the rest deliberately.
3. **CIFAR ablations are 1-seed** (`model_seeds: [4000]` in all four kappa/paramFrac configs) while `rebuttal_fig5_cifar_paramfrac_scan` uses 3 and the headline uses 5. Decision §1 ("seed-mean selection", O3 "5 seeds") is inconsistent with them; `submit_reruns_2026-09-17.sh` splits them one-run-per-job at NPROC=1.
4. **κ grids disagree between the two CIFAR configs** (`[1,1.5,2,2.5,3]` vs `[1,2,3]`) and `EXPERIMENTS.md:107-109` documents 5 values for both.
5. **Set points in the ablation configs are stale/contradictory.** `cifar10_resnet_kappaScan_labelReg.yaml:114` and `rebuttal_fig5_cifar_paramfrac_scan.yaml:883` say "tentative: pre-Gram classic best… re-derive from the BN-fixed headline scan" (k=64, lr=1.0, rtol=1e-3), while `EXPERIMENTS.md:98-99,108-109,151` documents k=128, lr=0.1, rtol=1e-4 for the same scans. Every phase-4 config must be re-pointed at phase-3 outputs before launch.
6. **`EXPERIMENTS.md` drift beyond F35:** 10 seeds (configs: 5) for 12 scans; 3 seeds for the MNIST-CE knob sweeps (configs: 5); `[chunked]` for every CIFAR row (configs: `gram_capture: full`); `k∈{16,32}` / `k∈{32,64}` for microbatch/paramfrac (configs: a single `k_values`); `rtol 1e-2` for `polynomial_microbatch_scan` (config: `1e-3`); `mnist_microbatch_ce_scan` "k·frac∈{.25,.5,1}" (config: `k_fractions: [1.0]`); `exp_nanogpt_speedrun` "k∈{32,64}" (config: `[64]`); `exp_finetune` omits MuonW.
7. **`exp_critbatch_nanogpt.yaml:242` comment says `k = B/2` but sets `k_fractions: [1.0]`** — and `EXPERIMENTS.md:184-185` documents k=B/2 for both critbatch scans. The published "critical batch at fixed rank fraction" claim is wrong about the value actually run. C-X2 fixes this by sweeping all three fractions.
8. **`weight_decays` semantics differ across scans.** Headline/ablation configs leave it `[None]` (AdamW→0.01, MuonW→0.1); overparam and batch-size configs set `[0.0, 0.01]`, which doubles AdamW/Muon/MuonW and forces MuonW to wd 0.01 rather than its default — hence the explicit `weight_decays=[0.1]` patch in `submit_reruns_..._part2.sh`. C-B4 should make one rule.
9. **`rebuttal_overparam_polynomial_scan` / `_toy_1d` set `dataset.n_val: 10000` but `n_train: ${n_data}`**, and the toy/poly datasets take one `seed` for all splits — exactly F22/C-D3. Note `toy_1d_regression.yaml` already has `n_test: 10000` that nothing evaluates (F5); `random_polynomial.yaml` has no `n_test` at all.
10. **Sharding vs. dedup ordering.** `_shard_skip()` (`generic_scan.py:366-369`) increments a counter over the *whole* family grid before the file-existence check, so a resubmission reproduces the same assignment only if the grid is byte-identical — any config edit (an added lr, C-B3) reshuffles shard membership. C-R2's `expand_grid` must preserve ordering explicitly, and `n_shards`/`shard_id` must land in the manifest (they are passed as `+n_shards=` hydra additions today and are recorded nowhere).
11. **`exp_critbatch_mnist` is launched with no `mode` override** (`submit_fresh_suite.sh:124`), so one job runs Sven *and* Adam in the same shard — the only place the "never let Sven wait behind a baseline" rule (`:17-19`) is broken.
12. `bench/timing_serial.sbatch` hard-codes `SCANS` to the 5 scans with `_timing` configs; `submit_reruns_2026-09-17.sh` chains it with `GROUP=all` on a 36 h `--exclusive` reservation, which is what currently sits in `QOSMaxNodePerUserLimit` — an exclusive whole-node request competes with the 2-node cap for the rest of the campaign.