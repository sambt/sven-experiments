All three jobs completed. Final report:

---

## Jobs

| job | id | where | elapsed | state |
|---|---|---|---|---|
| JOB 1 (a,b,c,d,e) | **47037513** | `iaifi_gpu_priority`, holygpu8a27303, 1× A100-SXM4-80GB, 12 cores | 33:07 | COMPLETED 0:0 |
| JOB 2 (a, CIFAR-slice, d, e) | **47037629** | `gpu_test`, holygpu7c26202, MIG 3g.20gb | 16:21 | COMPLETED 0:0 |
| JOB 3 (part b on a slice) | **47038158** | `gpu_test`, holygpu7c26203, MIG 3g.20gb | 12:08 | COMPLETED 0:0 |

3 of 3 jobs, **≈0.9 A100-GPU-hours** of the ~12 allowed. 296 measured processes in 76 groups. JOB 1 sat in `QOSMaxNodePerUserLimit` (the 9 GPT-2 jobs hold 2 nodes in each iaifi QOS) and started at 13:36, 17 min after submission; JOB 3 was added because all 32 iaifi A100s were allocated and a MIG slice can answer the `empty_cache`/allocator axes.

**Files** (all under `/n/home11/sambt/iaifi/sv3/bench/probe_campaign/`, nothing else touched): `probe_plan.py` (measurement plan + `choose_best_b`), `probe_run.py` (one measurement, snapshot guard, `empty_cache` monkeypatch, file barrier), `probe_driver.py`, `analyse.py`, `make_snapshot.sh`, `job1.sbatch`, `job2.sbatch`, `job3.sbatch`, `test_probe_campaign.py`.
Snapshot `…/sv3_campaign_scratch/probe_snapshot` (sv3 `558463c`, sven `ca8742b`); results `…/sv3_campaign_scratch/probe_results/` (`jsonl/`, `logs/`, `env/`, `profiler/`, `slurm/`, `progress.log`, `best_b.json`). One command: `.venv/bin/python bench/probe_campaign/analyse.py`. Tests: **`28 passed in 2.61s`**. Every measurement recorded `sven`/`experiments` resolving inside the snapshot ("snapshot provenance: OK for every measurement").

## The headline: `empty_cache` is a 4.5x speedup on CIFAR Sven

CIFAR ResNet18 Sven, B=128, batch-stat BN, A100-80GB, wall median over 75 steps:

| capture | empty_cache | PYTORCH_CUDA_ALLOC_CONF | ms/step | p10–p90 | peak alloc | peak **reserved** |
|---|---|---|---|---|---|---|
| full | **off** | default | **186.5** | 185.8–187.5 | 22 976 MB | 32 833 MB |
| full | **off** | expandable | **186.7** | 185.9–187.6 | 22 976 MB | **23 325 MB** |
| full | on | expandable | 586.3 | 552–647 | 22 976 | 23 387 |
| full | on | default | **841.4** | 516–1159 | 22 976 | 33 074 |
| cf0.5 | off | expandable | 191.9 | 191–197 | 10 014 | **10 215** |
| cf0.5 | on | default | 586.9 | 440–740 | 10 015 | 23 176 |
| cf0.25 | off | expandable | 205.4 | 204–206 | 6 921 | **7 363** |
| cf0.25 | on | default | 467.3 | 353–581 | 6 921 | 18 495 |

`empty_cache` off is **0.437x** the step time averaged over the other axes, and 841→186 ms on the production setting. My `empty_cache=on` number (841 ms) reproduces the archived `profile_results_v2` value (856 ms) to 2%, so the harness is calibrated. `empty_cache` was also the entire source of step-time variance (p10–p90 spans 643 ms with it on, 1.7 ms with it off). C-T3 is described in the spec as "measure the difference once"; **it is worth roughly 310 GPU-hours on the two headline CIFAR scans alone** (Sven is ~200 of each scan's 215 process-h; ÷4.5 ⇒ ~44 h each).

`expandable_segments:True` is free on time once `empty_cache` is off (186.5 vs 186.7 ms) and **halves peak reserved** (0.54x, 21 561→11 661 MB mean). With `empty_cache` *on* it interacts: 1.43x faster on the A100, 1.6x **slower** on a MIG slice. Take both changes together, never `expandable` alone.

Chunked capture only ever helped because it gave `empty_cache` less to churn: with it off, `full` (186.5) beats cf0.5 (192.0) and cf0.25 (205.0). **Do not change `gram_capture`** — and `full` is literally the chunked code path with one group (`sven/sven/nn/gram_wrapper.py:178`), G accumulated in float64 either way, so switching would have been numerics-safe but is unnecessary.

## Decision table: co-tenancy (wall time, barrier-synchronised)

T = (processes that ran) / inflation, i.e. standalone-runs-worth of work per GPU per wall-second.

| device | workload | NPROC | ms/proc | inflation | **T** | peak resvd |
|---|---|---|---|---|---|---|
| A100-80GB | CIFAR Sven full, ec-off | 1 / 2 / 3 | 186.7 / 381.3 / 385.0 | 1.00 / 2.04 / 2.06 | 1.00 / **0.98** / 0.97 | 32 833 MB (3rd proc OOMs) |
| A100-80GB | CIFAR Adam | 1 / 4 | 24.0 / 25.0 | 1.00 / 1.04 | 1.00 / **3.85** | 325 MB |
| A100-80GB | toy-1D Sven (hooks) | 1/4/6/8/12 | 11.8/12.1/13.2/16.3/19.7 | 1.00/1.03/1.12/1.38/1.67 | 1.0/3.90/5.35/5.78/**7.18** | 23 MB |
| A100-80GB | MNIST Sven (hooks) | 1/4/6/8/12 | 11.1/13.0/16.2/20.4/29.3 | 1.00/1.17/1.45/1.84/2.63 | 1.0/3.41/4.13/4.36/**4.56** | 25 MB |
| A100-80GB | MNIST Adam | 1/4/6/8/12 | 4.58/4.71/4.54/3.93/4.58 | ≈1.00 | up to **12.0** (CPU-bound) | 25 MB |
| A100-80GB | MNIST LBFGS(3) | 1/4/6/8/12 | 16.4/42.5/60.9/80.2/115.8 | 1.00/2.60/3.72/4.90/7.08 | 1.0/1.54/1.61/1.63/**1.69** | 29 MB |
| A100-80GB | nanoGPT Sven | 1 / 2 / 3 | 71.7 / 79.0 / 92.5 | 1.00/1.10/1.29 | 1.0/1.82/**2.33** | 663 MB |
| MIG 3g.20gb | CIFAR Sven full | 1 | — | — | — | **OOM** (needs 22.4 GB of 19.62) |
| MIG 3g.20gb | CIFAR Sven cf0.5 ec-off | 1 | 419.1 | — | — | 10 215 MB |
| MIG 3g.20gb | toy-1D Sven | 1/4/6/8/12 | 3.5/7.2/9.6/11.6/18.0 | 1.00/2.06/2.73/3.29/5.11 | 1.0/1.95/2.20/**2.43**/2.35 | 23 MB |
| MIG 3g.20gb | MNIST Sven | 1/4/6/8/12 | 4.1/9.2/13.1/17.4/26.8 | 1.00/2.26/3.21/4.28/6.57 | 1.0/1.77/**1.87**/1.87/1.83 | 25 MB |
| MIG 3g.20gb | MNIST Adam | 1/4/6/8/12 | 1.13/1.73/1.87/2.63/4.10 | 1.00/1.52/1.65/2.32/3.62 | 1.0/2.63/**3.64**/3.44/3.32 | 25 MB |
| MIG 3g.20gb | MNIST LBFGS(3) | 1/4/6/8/12 | 31.9/40.5/58.7/78.0/117.9 | 1.00/1.27/1.84/2.44/3.69 | 1.0/3.16/**3.27**/3.27/3.25 | 29 MB |
| MIG 3g.20gb | nanoGPT Sven | 1 / 2 / 3 | 35.0 / 71.3 / 105.6 | 1.00/2.04/3.02 | 1.0/**0.98**/0.99 | 663 MB |

## Recommendations

**CIFAR capture mode:** keep `gram_capture: full`, set **`empty_cache=False`** and **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**. 186.7 ms/step, 23.3 GB reserved.

**NPROC per workload class — A100-80GB:** CIFAR Sven **1** (T=0.98 at 2; the old NPROC=2 was only ever filling allocator stalls); CIFAR/first-order baselines **4+**; toy/poly/MNIST Sven-hooks **12** (still rising — go higher if cores allow); MNIST Adam and other cheap first-order **= cores available**; LBFGS(max_iter≥2) **4–6** (genuinely GPU-bound, T plateaus at 1.6); nanoGPT Sven **3**. Keep `cpus-per-task ≥ NPROC`, not NPROC+2 — cores, not the GPU, are the limit for the cheap MLPs.

**NPROC on a MIG 3g.20gb slice:** MLP Sven **6**, toy-1D Sven **8**, MNIST Adam **6**, LBFGS(3) **6–8**, nanoGPT **1**, CIFAR Sven **1** at `cf0.5` (full capture does not fit).

**MIG is usable for CIFAR Sven** — the scout ruled it out on the 23 GB full-capture peak, but `cf0.5` + `expandable_segments` needs only 10.2 GB reserved and runs at 419 ms (2.2x the A100). A-100-equivalents per slice, GPU-bound workloads only: CIFAR Sven 0.71–0.75, MNIST LBFGS3 0.51.

## Surprises

1. **CUDA-event timing is unusable for co-tenancy.** MNIST-Adam `step_ms` *fell* from 1.01 to 0.22 ms going from NPROC 1 to 6 while wall time rose 1.13→1.87 ms: on a time-sliced GPU the wait before `e0` executes is invisible to the events. Using `step_ms` would have reported T=28 instead of 3.6. Every NPROC/sharding decision must use `wall_ms`; `analyse.py` flags `EVENT-TIME-UNUSABLE` when wall/event > 1.5.
2. **Host CPU, not the GPU, sets the cheap-MLP step time.** Identical MNIST-Adam config: 1.13 ms on the quiet `gpu_test` node vs **4.58 ms** on the 4/4-occupied iaifi node. That is the mechanism behind the cost scout's "not physical" 0.34–0.64 sharded/standalone ratios, and it means C-T1/phase-5 timing runs must pin the node or co-run a calibration load. Cross-node MLP step times are not comparable.
3. **MPS: the binary does exist** at `/usr/bin/nvidia-cuda-mps-control` on GPU nodes (the scout could only check a CPU node), but Compute Mode is `Default` and `GresTypes=gpu` only, so it stays out of scope — for the prerequisites, not for a missing binary.
4. `gpu_test` nodes are A100-SXM4-**40GB** parents sliced 8 ways (19.62 GB usable per slice), not 80GB. Slice SM clock stayed pinned at 1410 MHz; the A100 idles at 210 MHz and reaches 1410 under load.

## Open issues for the integrator (files I do not own)

* **C-T3 should default `empty_cache=False` for every run, not only timing/profile runs.** CONTRACTS.md specifies `empty_cache: bool = True`; the measured 4.5x on CIFAR (and the variance collapse) makes `True` the wrong default. Keep it reachable for reproducing legacy runs.
* Export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the launchers — free, halves reserved memory, and it is what makes CIFAR Sven fit a MIG slice.
* `submit_fresh_suite.sh:15-16`'s "2–3.5x throughput per GPU" and CIFAR `NPROC=2` are both wrong; replace with the table above.
* **C-T2**: switching `analysis/profile_helpers.py` to `cycle_mean` of `step_ms` is fine for single-process profiles but must not be used for NPROC decisions (see surprise 1). Same for `bench/bench_sharding.py`.
* The profiler builds Sven with `track_svd_info=False` while `generic_scan.py:523` sets it `True`, so every Sven number here omits the per-step `sigma_full.cpu().numpy()` + `count_nonzero().item()` syncs that production pays. C-L1's `log_this_step` is what removes them; worth one measurement as part of C-L1's acceptance.

**Gaps:** the `empty_cache` axis was measured on CIFAR only — part (d) ran stock (`ec` on), so the MLP effect is unmeasured (expected small at 21 MB peaks). Part (c) ran in `full/ec-off/default` (32.8 GB) because part (b)'s strict speed ordering preferred 186.5 over 186.7 ms; with `expandable` (23.3 GB) 3 co-tenants would fit, but since NPROC 2 already gives T=0.98 there is nothing to gain. I fixed that tie-break (`SPEED_TIE_TOL = 3%`, then least reserved memory) in `probe_plan.choose_best_b`, with a test. NPROC > 12 was not tested and A100 toy-1D/MNIST Sven and Adam were still improving at 12. `env/mig_3g20gb/clocks_under_load.csv` interleaves samples from JOB 2 and JOB 3 (two nodes, same `device_class` label).