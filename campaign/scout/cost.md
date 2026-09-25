## 1. Per-scan totals (measured; all numbers from reading every top-level `*.jsonl`)

Schema note (**verified**): timing lives inside `losses` — `total_time`, `epoch_times`, `avg_epoch_time`, `avg_batch_time_train/_val`, `peak_gpu_mem_mb`. **`n_shards`, `host`, GPU name, `status` and SLURM job id are absent from every one of the 15,329 records** (C-R3 is therefore unverifiable against history; NPROC below is read off the launchers, not the data). `total_time` present on 100% of records; it agrees with `sum(epoch_times)` to a median 1e-4 relative.

`nproc` = launcher value (`submit_fresh_suite.sh:15,70-125`; the 2026-09-17 CIFAR-ablation reruns and `rebuttal_fig5` ran one run per job, so nproc=1). `gpu_h` = proc_h/nproc is the doc's convention and is **optimistic** — see §2.

| scan | runs | proc_h | nproc | gpu_h(doc conv.) | n_div | max run (min) | first mtime | last mtime | span_h | active_h |
|---|---|---|---|---|---|---|---|---|---|---|
| cifar10_resnet_ce_scan | 290 | 215.2 | 2 | 107.6 | 0 | 255.7 | 09-11 16:02 | 09-16 08:56 | 112.9 | 22 |
| cifar10_resnet_scan_labelRegression | 290 | 214.6 | 2 | 107.3 | 9 | 249.5 | 09-11 18:29 | 09-16 01:42 | 103.2 | 16 |
| rebuttal_batchsize_polynomial_scan | 2496 | 287.5 | 6 | 47.9 | 502 | 81.3 | 09-15 03:16 | 09-18 13:03 | 81.8 | 21 |
| mnist_scan_labelRegression | 999 | 183.7 | 4 | 45.9 | 100 | 179.5 | 09-11 15:03 | 09-18 03:02 | 156.0 | 36 |
| mnist_scan_ce | 1040 | 168.5 | 4 | 42.1 | 35 | 111.2 | 09-11 16:55 | 09-18 03:23 | 154.5 | 30 |
| rebuttal_overparam_mnist_scan | 2477 | 163.0 | 4 | 40.8 | 300 | 43.0 | 09-14 22:54 | 09-18 12:41 | 85.8 | 18 |
| mnist_paramfrac_ce_scan | 92 | 43.6 | 4 | 10.9 | 6 | 46.9 | 09-14 18:01 | 09-18 08:32 | 86.5 | 12 |
| mnist_microbatch_labelreg_scan | 140 | 42.5 | 4 | 10.6 | 0 | 24.3 | 09-14 17:14 | 09-15 03:35 | 10.3 | 11 |
| polynomial_scan | 731 | 40.8 | 6 | 6.8 | 94 | 20.3 | 09-11 14:42 | 09-18 02:39 | 156.0 | 8 |
| mnist_paramfrac_labelreg_scan | 87 | 40.1 | 4 | 10.0 | 12 | 45.6 | 09-14 18:01 | 09-15 04:00 | 10.0 | 11 |
| mnist_microbatch_ce_scan | 140 | 35.3 | 4 | 8.8 | 0 | 29.2 | 09-14 17:19 | 09-18 08:20 | 87.0 | 10 |
| toy_1d_scan | 734 | 34.5 | 6 | 5.7 | 95 | 19.7 | 09-11 14:42 | 09-18 02:40 | 156.0 | 7 |
| exp_critbatch_nanogpt | 210 | 32.2 | 2 | 16.1 | 0 | 58.4 | 09-15 05:12 | 09-18 07:37 | 74.4 | 12 |
| rebuttal_fig5_cifar_paramfrac_scan* | 15 | 27.3 | 1 | 27.3 | 0 | 150.5 | 09-18 10:10 | 09-18 13:10 | 3.0 | 4 |
| rebuttal_overparam_polynomial_scan | 2160 | 26.1 | 6 | 4.3 | 155 | 2.1 | 09-14 20:16 | 09-18 12:23 | 88.1 | 5 |
| rebuttal_overparam_toy_1d_scan | 2160 | 25.3 | 6 | 4.2 | 117 | 2.1 | 09-14 17:30 | 09-18 12:20 | 90.8 | 5 |
| exp_critbatch_mnist | 210 | 23.2 | 4 | 5.8 | 1 | 36.1 | 09-15 05:02 | 09-18 06:10 | 73.1 | 6 |
| toy_1d_microbatch_scan | 120 | 16.7 | 6 | 2.8 | 3 | 129.4 | 09-14 13:31 | 09-14 16:17 | 2.8 | 3 |
| cifar10_resnet_paramFrac_scan_labelReg* | 5 | 10.8 | 1 | 10.8 | 0 | 183.9 | 09-18 08:51 | 09-18 10:23 | 1.5 | 3 |
| cifar10_resnet_kappaScan_labelReg* | 5 | 9.2 | 1 | 9.2 | 0 | 113.1 | 09-18 07:10 | 09-18 07:45 | 0.6 | 1 |
| cifar10_resnet_ce_paramFrac_scan* | 6 | 8.9 | 1 | 8.9 | 0 | 114.1 | 09-18 09:00 | 09-18 10:33 | 1.5 | 2 |
| mnist_scan_labelRegression_timing | 60 | 8.8 | 1 | 8.8 | 0 | 40.4 | 09-16 23:44 | 09-17 06:22 | 6.6 | 8 |
| mnist_scan_ce_timing | 60 | 7.8 | 1 | 7.8 | 0 | 41.3 | 09-17 02:47 | 09-17 12:45 | 10.0 | 10 |
| exp_nanogpt_speedrun | 100 | 7.6 | 2 | 3.8 | 0 | 9.7 | 09-15 04:13 | 09-18 05:26 | 73.2 | 5 |
| mnist_kappaScan_labelRegression | 15 | 6.6 | 4 | 1.6 | 0 | 27.2 | 09-14 17:12 | 09-14 18:30 | 1.3 | 2 |
| toy_1d_paramfrac_scan | 97 | 4.4 | 6 | 0.7 | 14 | 3.1 | 09-14 16:11 | 09-14 16:56 | 0.8 | 1 |
| polynomial_paramfrac_scan | 86 | 4.3 | 6 | 0.7 | 3 | 3.4 | 09-14 16:21 | 09-14 17:06 | 0.8 | 2 |
| cifar10_resnet_ce_kappaScan* | 2 | 3.6 | 1 | 3.6 | 0 | 108.6 | 09-18 08:33 | 09-18 08:34 | 0.01 | 1 |
| exp_finetune_cifar_smallN* | 240 | 3.5 | 2 | 1.8 | 24 | 13.6 | 09-18 11:19 | 09-18 12:49 | 1.5 | 2 |
| polynomial_microbatch_scan | 120 | 3.4 | 6 | 0.6 | 0 | 1.8 | 09-14 16:11 | 09-14 16:45 | 0.6 | 1 |
| polynomial_scan_timing | 65 | 1.0 | 1 | 1.0 | 1 | 9.9 | 09-16 23:51 | 09-17 00:54 | 1.1 | 2 |
| exp_nanogpt_speedrun_timing | 12 | 1.0 | 1 | 1.0 | 0 | 8.8 | 09-17 04:27 | 09-17 05:16 | 0.8 | 2 |
| toy_1d_scan_timing | 65 | 0.9 | 1 | 0.9 | 0 | 1.6 | 09-16 23:02 | 09-16 23:49 | 0.8 | 1 |
| **exp_gpt2_small_comparison\*** | **0** | **0** | — | — | — | — | — | — | — | — |

`*` = the "not local" set of §4.3. **Total: 15,329 runs, 1702 process-h**; local subtotal **1638.7 process-h**, not-local **63.3** (GPT-2 zero). Full scan×family table (runs / proc_h / median / p90 / max min / n_div per family) is in `scan_family.csv` and `scan_family_compact.csv`; family rollup: Sven 5586 runs/915 h, LBFGS 3530/480 h, HIG 214/125 h, Shampoo 500/77 h, all first-order combined 4503/74 h, SOAP+KFAC 896/21 h, JD 100/11 h.

**GPT-2 status (verified)**: 9 jobs RUNNING (`squeue`, names `wrap`, logs `slurm_logs/gpt2-*.out`), 2.2–3.1 h in, 12 h limit, still at `0%` on the first run of a 1-epoch/13,125-step grid; **zero records written**. One job pending: `timing_serial_RERUNS` (36 h limit) blocked on `QOSMaxNodePerUserLimit`.

## 2. vs. CHANGES_NEEDED §4.3 — and the one real discrepancy

| entry | doc runs/proc_h | measured | Δ |
|---|---|---|---|
| toy_1d_scan | 714 / 34 | 734 / 34.5 | +20 runs |
| polynomial_scan | 711 / 41 | 731 / 40.7 | +20 |
| mnist_scan_ce | 1020 / 169 | 1040 / 168.5 | +20 |
| mnist_scan_labelRegression | 979 / 184 | 999 / 183.7 | +20 |
| cifar ce / labelReg | 290 / 215 each | 290 / 215.2, 290 / 214.6 | exact |
| micro-batch (4) | 464 / 86 | 520 / 97.9 | +56 / +12 h |
| param-fraction (4) | 323 / 80 | 362 / 92.5 | +39 / +12 h |
| mnist_kappaScan | 15 / 7 | 15 / 6.6 | ok |
| rebuttal_batchsize_poly | 2376 / 285 | 2496 / 287.5 | +120 |
| rebuttal_overparam_mnist | 2357 / 161 | 2477 / 163.0 | +120 |
| rebuttal_overparam toy+poly | 4160 / 50 | 4320 / 51.4 | +160 |
| **exp_critbatch mnist+nanogpt** | **168 / 28** | **420 / 55.4** | **+252 runs / +27 h** |
| exp_nanogpt_speedrun | 48 / 4 | 100 / 7.6 | +52 / +3.6 h |
| standalone timing (5 dirs) | 262 / 19 | 262 / 19.5 | exact |
| subtotal local | ~1580 | **1638.7** | +59 h |

The doc's per-scan process-hours are accurate; every Δ is 2026-09-17/18 reruns landing after it was written (`+20` per headline scan = the AdamW-wd re-run after `_adamw_wd0` was moved aside). Its sub-claims check out exactly: batch-size LBFGS **810 runs / 237.7 h / 439 diverged** (doc: 810/238/439); HIG on MNIST **66.7 h (80 runs, CE) + 50.0 h (39 runs, labelReg)**.

**The doc's GPU-hour column is the discrepancy.** Dividing process-h by NPROC assumes sharding is free. The controlled benchmark it cites (`bench/results_46028086.jsonl`, **read + recomputed**) measures throughput vs NPROC on one A100: toy1d 952→1052→1232→1362 runs/h for NPROC 1/2/4/8 (**1.10 / 1.29 / 1.43x**), MNIST 471→518→609→670 (**1.10 / 1.29 / 1.42x**). `submit_fresh_suite.sh:16` claims "2–3.5x throughput per GPU"; the file it cites says 1.3–1.4x. So **~490 GPU-h is an underestimate; the defensible figure is the measured allocation, below.**

**SLURM accounting (verified, `sacct` day-by-day 09-09..09-18; needs narrow windows or it errors "Too wide of a date range")**: 442 GPU jobs, **795 GPU-job-hours** of A100-80GB allocation over a 233 h window → **average concurrency 3.4, peak 17** (not 8: each lab partition caps 2 *nodes* × 4 GPUs = 8 jobs, and both partitions count separately). Median queue wait **4.27 h** (p75 6.98, max 15.7). **168 job-hours lost to 14 TIMEOUTs** (11 of them CIFAR Sven at the 12 h wall before the split-by-seed fix) + 17 h cancelled. Delivered ratio: **2.14 process-h per allocated GPU-h**. Partition split: lab_gpu_priority 255 jobs/485 h, lab_gpu 170/289 h, gpu 12/19.6 h, gpu_test 5/1.2 h.

## 3. Measured sharding inflation (sharded parent / standalone timing, joined on `run_id`)

`sharding_inflation.csv`, 239 matched pairs. Ratio of `total_time`, and of `avg_batch_time_train` in brackets.

| workload (parent NPROC) | Sven | LBFGS | HIG | JD | Shampoo | SOAP | KFAC | first-order | all (median) |
|---|---|---|---|---|---|---|---|---|---|
| toy_1d (6) | 1.51 [1.31] | 3.03 [2.99] | 2.63 [2.56] | 1.70 | 4.14 [4.10] | 2.11 | 2.04 | 1.48–1.93 [~1.0] | **1.73** (p25 1.53, p75 2.35) |
| polynomial (6) | 1.24 [1.06] | 2.21 | 2.02 | 1.66 | 4.11 | 2.15 | 2.04 | 1.49–2.03 [~1.0] | **1.69** (1.51–2.11) |
| mnist_ce (4) | 1.43 [1.37] | 1.38 | 1.00 | 1.02 | 1.02 | 0.89 | — | 1.01–1.06 [0.70–0.95] | **1.04** (1.01–1.08) |
| mnist_labelReg (4) | 1.39 | 1.48 | **4.22** | 1.01 | 1.02 | 0.89 | — | **0.64–0.95** [0.34–0.64] | **0.98** (0.66–1.08) |
| nanoGPT (2) | 1.11 | — | — | — | — | 1.01 | — | Muon 1.03 | **1.03** |

Readings: on the 6-way MLP scans, GPU-bound methods (Shampoo 4.1x, LBFGS 2.2–3.0x, HIG 2.0–2.6x, KFAC/SOAP ~2x) lose most of the nominal 6x, while cheap first-order runs lose almost nothing per batch (batch ratio ≈1.0; the 1.5x total is fixed setup + eval). **Effective throughput gain = NPROC/inflation ≈ 3.5x at NPROC=6 and ~3.9x at NPROC=4 for the homogeneous production jobs** — much better than the mixed-load bench, because the launcher already splits Sven / first-order / second-order / LBFGS into separate jobs. **NanoGPT at NPROC=2 gains only ~1.9x and CIFAR has no timing dir at all: there is no measurement of two concurrent Sven-ResNet runs on one A100.** Sven-CIFAR is 630 ms/batch and 8.2–23 GB peak — assume ~1.0–1.2x gain there, i.e. the two CIFAR scans cost ~215 GPU-h *each*, not 107.

Two results are not physical and mark the standalone reference as unreliable: **MNIST-labelReg first-order ratios of 0.64 (batch-time 0.34)** and **HIG 4.22**. The timing set was produced by two `--exclusive` serial jobs (`bench/timing_serial.sbatch`, jobs 46783849 `timing_serial_LONG` 13:31:43 and 46783850 `timing_serial_SHORT` 06:00:09 — 19.5 job-h, matching the 19.5 process-h in the five timing dirs). `OMP_NUM_THREADS=1` *is* set there, so threads are not the cause; the likely mechanism for launch-bound runs (1.3 ms/batch) is GPU clock/power state on an otherwise-idle exclusive GPU. **Any new timing pass (C-T1/phase 5) must lock clocks or co-run a calibration load, or these ratios stay uninterpretable.**

## 4. Heaviest runs — what sets the minimum job length

Top 20 by wall time are **all CIFAR Sven**, 4.12–4.26 h, `bs128 k64 lr0.1 rtol{1e-3,1e-4} gram_bnbatch`, all 5 seeds of both CIFAR scans (`per_run.csv`). Distribution: 375 runs >1 h, 143 >2 h, **20 >4 h, max 4.26 h**. Heaviest per scan:

| scan | family | h | run_id (abridged) |
|---|---|---|---|
| cifar10_resnet_ce_scan | Sven | 4.26 | `svd_bs128_k64_lr0.1_rtol0.001…gram_bnbatch` |
| cifar10_resnet_scan_labelRegression | Sven | 4.16 | same settings, mseed4000 |
| cifar10_resnet_paramFrac_scan_labelReg | Sven | 3.07 | `k64_lr1.0_rtol0.001_pf0.5_elementwise` |
| mnist_scan_labelRegression | HIG | 2.99 | `hig_bs64_w32_lr0.1_tau1e-4` |
| rebuttal_fig5_cifar_paramfrac_scan | Sven | 2.51 | `k64_lr1.0_pf0.5_elementwise` |
| toy_1d_microbatch_scan | Sven | 2.16 | `k32_lr1.0_mb1` (median in that scan 2.0 **min**) |
| mnist_scan_ce | HIG | 1.85 | `lr1.0_tau1e-4` |
| rebuttal_batchsize_polynomial_scan | LBFGS | 1.36 | `bs8_lr1.0_mi3_hs10_strong_wolfe` |

Implication: **no single run exceeds 4.3 h, so a 12 h job holds ≥2 of the worst back to back; nothing needs a >12 h reservation**, and the 11 CIFAR TIMEOUTs were caused by packing 18 Sven runs into one job, not by any run being long. Peak memory (verified): CIFAR Sven `full` capture **22,965 MB** → will *not* fit a `gpu_test` MIG slice (`nvidia_a100_3g.20gb`, 20 GB, 12 nodes × 8 slices, 12 h limit); everything else does (MNIST ≤481 MB, toy/poly ~22 MB, nanoGPT ≤2.4 GB, CIFAR baselines 1.2 GB, fine-tune 1.3 GB).

## 5. Diverged-run cost (recoverable by C-R1's early stop)

Divergence here = non-finite final val/train, or final val > 10× val[0]. **1631 diverged runs, 512 process-h — 30% of the entire campaign.** `rebuttal_batchsize_polynomial_scan` 232/288 h (81%), `rebuttal_overparam_mnist_scan` 69 h (42%), `mnist_scan_labelRegression` 52 h (28%), `polynomial_scan` 20 h (50%), `toy_1d_scan` 12 h (36%). By family: LBFGS 936 diverged / **389 h**, KFAC 72/4.8 h, Sven 358/16.7 h.

## 6. Data-quality caveats

1. **No `n_shards`, `host`, GPU name, `status`, job id or git SHA on any record** — sharding factor, node identity and failure classification had to come from the launchers and `sacct`. C-R3/C-R4 are the fix; without them no future audit of this kind is possible from the records alone.
2. **`total_time` is 100% present** and self-consistent; no unreadable files, no JSON errors, **no multi-line jsonl** (1 record per file, exactly).
3. **Move-aside dirs excluded** (jsonl counts, read-only listing): `cifar10_resnet_{ce_scan,scan_labelRegression}/_frozen_bn` 90 each; `{toy_1d,polynomial,mnist_scan_ce,mnist_scan_labelRegression}/_spectra_truncated` 60/60/80/80 and `/_adamw_wd0` 20 each; `exp_critbatch_nanogpt/_adamw_wd0` 36, `exp_nanogpt_speedrun/_adamw_wd0` 12; `*_timing/_shared_node` 55/55/50/50/12. That is **~770 additional completed runs** whose cost is *not* in the 1702 h — real historical spend, superseded results.
4. `_backup_2026-08-31` (54 scans) and `_backup_2026-09-11` (29 scans) were not opened at all; they hold at least one more full campaign's worth of spend.
5. **Cross-contaminated mtime spans.** `span_h` is inflated wherever a 09-17/18 rerun touched an old scan (toy/poly/mnist headline all show 156 h spans but only 7–36 *active* hours). Use `active_h` (distinct hours with ≥1 write) as the end-to-end proxy: the four MLP headline scans were 7–36 active hours each, the two CIFAR scans 16–22, batch-size 21.
6. Shard-fill efficiency (process-h / (NPROC × job-h), labelled jobs only) is **0.79–1.00 for every scan whose jobs carry `--job-name`** — shard balance was good; the >1 values (toy 6.0, poly 8.0, mnist_labelReg 1.7) are attribution gaps: the first wave (09-11, 45 jobs, **129 job-h**) and the 09-08 wave (15 jobs, 74 h) predate `--job-name` and appear as `submit_rebuttal_parallel.sh` / `wrap` (204 unattributable job-h).
7. `cifar10_resnet_ce_kappaScan` has 2 of 3 grid points (config `kappa: [1,2,3]`, 1 seed); the other three CIFAR ablations and `rebuttal_fig5` (5 pf × 3 seeds) and `exp_finetune_cifar_smallN` (4 n_data × 60) are **complete** — the not-local set is essentially done at 63.3 process-h, well under the doc's 150–250 GPU-h guess, but `exp_finetune` carries F3 and GPT-2 has produced nothing.

**Artifacts** (all under `/tmp/claude-66176/-n-home-anon-sven-experiments/30b30b16-e05d-4e46-8a25-371ea6181950/scratchpad/`): `per_run.parquet` (15,329 × 49 — the per-run dataframe) and `per_run.csv`; `scan_family.csv`, `scan_family_compact.csv`, `scan_totals.csv`, `sharding_inflation.csv`, `sacct_gpu_jobs.csv`, `sacct_uniq.psv`; script `measure_cost.py`; raw output `report.txt`, `report2.txt`. Nothing in the repo or in `experiment_results` was written, and no SLURM job was touched.