# Sharded execution: assessment

**Premise corrections first** (all verified by running commands today):

| Claim in task | Reality |
|---|---|
| lab_gpu and lab_gpu_priority "share 4 nodes" | **Disjoint** node sets: `lab_gpu` = gpunode8a[27102-27103,27202-27203]; `lab_gpu_priority` = gpunode8a[27302-27303,27402-27403]. Each partition has its own QOS with `MaxTRESPU node=2` → **2+2 nodes = 16 A100-80GB simultaneously**, not 8. |
| "jobs there start almost immediately (max priority)" | Median submit→start over 2026-09-14…18: **lab_gpu 269 min, lab_gpu_priority 257 min, gpu 172 min** (n=154/226/9); p90 = 10–11 h. `PriorityTier=4` for both lab partitions (gpu=3), `PreemptMode=OFF`, `OverSubscribe=NO`. Priority buys queue position, not GPUs. |
| Sharding bench was on MIG | `slurm_logs/bench_shard-46028086.out:2` = **A100-SXM4-40GB**, 8 cores (`diag_env-46029087.out`: `Cpus_allowed_list: 8-15`, `nproc=1` is an `OMP_NUM_THREADS=1` artifact, affinity is really 8). |
| "2–3.5x throughput per GPU at NPROC 6/4" | Only true against a `threads=0` (torch default 8 threads, oversubscribed) baseline. Production exports `OMP_NUM_THREADS=1`. **Correct comparison below.** |
| fleet is available | Right now: 275 A100-80GB across 69 usable nodes, **272 allocated, 3 free, 0 idle nodes**. |

## 1. Static `specs[shard_id::n_shards]` under heterogeneous run lengths

Code: `generic_scan.py:363-368`, one shared `_run_idx` counter consumed at six call sites (486, 626, 698, 775, 854, 908), advancing *before* the dedup check. Round-robin over a counter that walks seeds → families → grid means neighbouring indices are near-identical configs, so for a **fresh** large grid the slices are statistically balanced.

Measured (293 jobs with complete shard-log sets, span>10 min; imbalance = Σ_shards(job_end − shard_last_write)/(N·span), from log mtimes + `sacct` Start):

| set | n | mean wasted GPU share | median | worst |
|---|---|---|---|---|
| fresh, N=2 | 58 | 0.029 | 0.002 | 0.24 |
| fresh, N=4 | 108 | 0.042 | 0.011 | 0.30 |
| fresh, N=6 | 83 | 0.040 | 0.010 | 0.21 |
| **resubmitted** (any shard had `[skip]`) | 43 | **0.20** | 0.13 | **0.57** |

So intra-job skew on a fresh grid costs only ~4% — *not* the problem. The problems are the two structural ones:

* **Resubmission after timeout is severely imbalanced.** Dedup is per-run-id and the slice is static, so a resubmitted job's remaining work is arbitrarily distributed. e.g. job 43906468 `rebuttal_baselines_mnist_scan`: shard finish times 0.08, 0.09, 3.86, 5.54 h → 57% of the GPU-job idle; 44439960/44439955: 54%. **37 of 293 jobs (12.6%) hit TIMEOUT, every one with all shards incomplete** — so every timeout produces one of these.
* **Campaign-level concurrency, not per-GPU efficiency, is the binding loss.** Per launch wave (from `sacct` Start/End, all sharded jobs): average concurrent single-GPU jobs was 1.0–10.8 with peaks of 16–17. Best waves: 46407 (89 jobs, 285 GPU-h over 26.5 h span → **avg 10.8, peak 16**), 46031 (avg 9.2, peak 17), 46944 (avg 8.4, peak 11). Typical waves ran at **2–6 GPUs**. Peak 16–17 confirms both lab QOS grant 2 nodes each plus `gpu` overflow.

Also: heterogeneity is extreme in the CIFAR jobs where each shard holds 1–2 runs of ~2.5 h (`rebuttal_fig5_cifar_paramfrac_scan_shard0_of1_46944089.out`: 20 epochs in 2:28, epoch time 10.7 min for epochs 1–11 then 4 min). With `runs=[2,1,1,1]` (job 43201277) skew is 26% and unavoidable under a static slice.

## 2. Option comparison

Throughput metric T = standalone-run-seconds completed per wall-second per GPU held. Recomputed from `bench/results_46028086.jsonl` at the production `threads=1` baseline. `runs_per_hour` in that file is **startup-contaminated** (`n_runs=8` for every setting, so at N=8 each shard runs exactly one run) — the defensible number is N / per-run inflation:

| workload | N=2 | N=4 | N=8 | runs/h ratio at N=8 |
|---|---|---|---|---|
| toy1d Sven | 1.78 | 2.10 | 2.45 | 1.43 |
| toy1d Adam | 1.19 | 1.41 | 1.57 | — |
| MNIST Sven | 1.75 | 2.14 | 2.56 | 1.42 |
| MNIST Adam | 1.35 | 1.77 | 2.06 | — |
| CIFAR ResNet (chunked, tiny N) | 2.00 | — | — | 1.00 |

Startup is cheap and amortises: torch import ≈ 4 s, MNIST dataset load 8.0 s, CIFAR 8.7 s (`load_s_max`) — <1% of a multi-hour shard.

**A. Status quo.** T ≈ 2.0–2.2 per GPU on mixed MLP grids at N=4–6 (not 4–6), ≈1.0 for ResNet. Realised campaign concurrency 2–11 GPUs. Cost: 12.6% timeout rate, ~20% waste on every resubmission, per-job queue wait 4.3 h median.

**B. Whole-node 4-GPU jobs + dynamic claim queue.** Per-GPU T is *identical* to A at equal NPROC (same silicon, same contention); the gain is entirely in GPUs-held × duration and in eliminating tail/resubmit idle. `--nodes=1 --gres=gpu:4 --time=3-00:00:00` (partition MaxTime is 3 days on all three, verified) gives 4 nodes × 4 GPUs = **16 GPUs held for 72 h = 1,152 GPU-h per allocation window**, which is the whole campaign. Acquisition latency, from co-tenant `TIME_LEFT` today: gpunode8a27102 ≈ 6 h, 27103 ≈ 7.3 h, 27303 ≈ 18 h, 27402 ≈ 20 h, 27202/27302/27403 ≈ 3 days (mgerdes/arghya hold 3-day jobs). So realistically **2 whole nodes in ~7 h, a 3rd in ~18–20 h**; the 4th may take 3 days. That is no worse than the observed p75 single-GPU wait (7–8 h) and then holds 4 GPUs instead of 1.

**C. CUDA MPS — unavailable, do not plan on it.** `/etc/slurm/slurm.conf` has `GresTypes=gpu` only (no `mps`, no `shard`), so SLURM cannot allocate MPS. `nvidia-cuda-mps-control` is absent from this CPU node's PATH and `/usr/bin` (`nvidia-smi` is also absent here, so this is *not* conclusive for GPU nodes — the probe must re-check on a GPU node). A user-started MPS daemon additionally needs the GPU in EXCLUSIVE_PROCESS mode, which a shared-node allocation cannot set. Treat as out of scope; on a whole-node allocation it is worth one 5-minute test.

**D. gpu_test MIG — underused and genuinely useful.** 12 nodes × 8× `a100_3g.20gb`; 6 idle right now. QOS `gpu_test`: `cpu=64,gres/gpu=8,mem=512G`, `MaxJobsPU=2`, `MaxSubmitPU=2`, 12 h. **Median observed wait 0.2 min** (verified). Memory is ample: measured peaks are toy1d 19 MB, MNIST 20 MB, CIFAR-chunked 8.2 GB — a 20 GB slice fits everything except full-capture ResNet. A `3g.20gb` slice is ~40% of an A100's SMs, but MLP runs are launch-bound so the loss is small. One job of `--gres=gpu:8 --cpus-per-task=56` = **8 more GPU-equivalents, essentially instantly**, at the cost of resubmitting every 12 h (only 2 jobs may be submitted at a time → use a self-resubmitting chain).

**E. CPU cores are *not* the current bottleneck.** `submit_rebuttal_parallel.sh:6` requests `--cpus-per-task=8` for up to NPROC=6; `num_workers` is never set anywhere in `experiments/` (grep: no hits) so every loader runs in-process and each shard needs ≈1 core. 8 cores for 6 shards is adequate. The evidence that CPU is not the limit: per-run inflation at N=8 is 3.1–3.3x with 8 cores for 8 single-threaded processes, i.e. **superlinear contention with cores to spare** — the serialisation is GPU time-slicing / context switch, not cores. Consequence: the "16 cores per GPU → ~16 processes" hope is wrong; returns plateau near N=6–8 (4→8 buys only ~+20% aggregate while doubling per-run latency and therefore timeout risk). Keep `cpus-per-task = NPROC+2`.

## 3. Recommended architecture

| workload class | where | job shape | NPROC/GPU | expected T (run-s per wall-s per GPU) |
|---|---|---|---|---|
| tiny MLP first-order (toy/poly/MNIST baselines, overparam, batchsize) | gpu_test MIG chain + whole-node pool | 8-GPU gpu_test job, 12 h, self-resubmitting | 6 | 1.5–1.8 (MIG: ×~0.8 → 1.2–1.5) |
| MLP Sven / HIG / LBFGS / SOAP | whole-node lab pool | `--gres=gpu:4`, 3 d | 6 | 2.2–2.4 |
| nanoGPT | whole-node pool | same | 2 | ~1.3 (unmeasured — probe) |
| ResNet18 Sven (full capture, 23 GB, ~2.5 h/run) | whole-node pool | same | **1** | 1.0 |
| ResNet baselines (Adam ~24 ms/step) | whole-node pool | same | 3–4 (80 GB fits) | 1.8–2.2 (probe) |
| timing runs (C-T1, phase 5) | `--exclusive` on lab, unchanged | `bench/timing_serial.sbatch` | 1 | 1.0, must stay exclusive |

**Honest campaign cost.** `CHANGES_NEEDED.md:434-451` divides process-h by NPROC, i.e. it assumes *linear* sharding. Re-deflating the 1,580 process-h by the measured aggregate factors (2.2 for the N=6 rows, 2.1 for N=4, 1.0–1.5 for N=2/1) gives **≈925 GPU-h for the local scans, not 490**; plus 200–400 for the non-local scans, plus test/`train_eval`/checkpoints/dense-spectra overhead and the C-B1–B4/C-X1–X2 grid growth, minus the LBFGS batch-size saving. Plan on **1,100–1,500 GPU-h**. At 16 lab GPUs + 8 gpu_test slices that is **3–4 days of pure compute**, 6–10 days with gates and development.

## 4. Dynamic claim queue: cost and risk

Engineering is small *because C-R2 is already on the critical path*: once `expand_grid(rcfg) -> [RunSpec]` plus `execute(spec)` exists, the queue is a 40-line replacement for `_shard_skip`. `O_EXCL` create of `{scan}/claims/{run_id}.started` on Lustre is atomic (coherent metadata locking, unlike NFS) and the volume (~15,000 creates over days) is negligible.

Risks, ranked: (1) **stale claims** — recover by writing `$SLURM_JOB_ID` into the claim and reclaiming when `squeue -j` says the job is gone, not by mtime age; (2) it collides with C-R1's `{run_id}.started` marker, whose semantics are "crashed run" — use two distinct names or the reconcile logic will mis-report; (3) claim-then-crash before the jsonl leaves an `oom`/`error` gap that only `tools/reconcile.py` will catch, so reconcile becomes mandatory rather than advisory; (4) it removes the "disjoint shards write disjoint run_ids, so it is race-free" guarantee that the comment at `generic_scan.py:361-362` relies on — every writer must now be idempotent; (5) worker-per-GPU pinning via `CUDA_VISIBLE_DEVICES` inside one allocation is trivial but the pool must respect per-class NPROC (ResNet 1, MLP 6) — run **two pools** (a long-run pool on 1 GPU, a short-run pool on 3) rather than one heterogeneous queue.

Compared with static shards plus resubmission, the queue buys: no timeout-resubmission waste (≈20% of every resubmitted job), no tail idle at the end of a scan, no need to hand-split grids by seed or batch size (`submit_fresh_suite.sh:73,81`), and longest-first scheduling to kill the CIFAR `runs=[2,1,1,1]` skew. **Do it** — but only as the second step, after C-R2 lands; keep static shards working as the fallback so the campaign is never blocked on the queue.

## 5. One-hour GPU probe (precise design)

Get one `--gres=gpu:4 --nodes=1 --cpus-per-task=64 --time=1:00:00 -p lab_gpu_priority,lab_gpu` job. Then, in order:

1. **Environment truth (2 min):** `nvidia-smi -L`, `nvidia-smi -q -d COMPUTE` (compute mode), `which nvidia-cuda-mps-control`, `grep Cpus_allowed_list /proc/self/status`, `nvidia-smi --query-gpu=memory.total`. Settles option C.
2. **Fix `bench_sharding.py` first:** set `--n-runs = 6 × nproc` so every setting runs ≥6 runs per shard, and report both steady-state per-run inflation *and* runs/h. Without this the whole benchmark family is startup-biased.
3. **NPROC sweep on the real launcher, not the bench** (25 min): `toy_1d_scan` and `mnist_scan_labelRegression` Sven at NPROC ∈ {1,4,6,8,12} on separate GPUs of the same node, `cpus-per-task` unconstrained, 2 epochs, fixed 24-run slice. Records the plateau and whether 12 is ever worth it.
4. **ResNet co-tenancy on 80 GB** (20 min): 1 vs 2 vs 3 concurrent full-capture Sven ResNet18 runs, 2 epochs, log `peak_gpu_mem_mb` and per-epoch time. Confirms or kills NPROC>1 for ResNet at 80 GB (the earlier loss was measured with 40 GB in view).
5. **ResNet-baseline co-tenancy** (5 min): 4 concurrent Adam ResNet18 — the untested case that dominates `cifar10_resnet_*_scan` run *counts*.
6. **Lustre claim-file probe** (3 min): 8 processes racing `O_EXCL` on 2,000 ids in `experiment_results/_probe_claims/`; assert exactly-once and record ops/s.
7. **Startup cost at scale** (3 min): time `import torch` + `CIFAR10Dataset()` on a cold node; if dataset load × NPROC × jobs is material, add a shared-memory cache.

Deliverable: a table of (workload, NPROC, per-run inflation, aggregate T, peak memory) that replaces the `2-3.5x` note at `submit_fresh_suite.sh:15-16`, plus a yes/no on MPS and on ResNet co-tenancy.

**Files:** `/n/home/anon/sven-experiments/submit_rebuttal_parallel.sh`, `/n/home/anon/sven-experiments/experiments/experiment_code/generic_scan.py:359-380`, `/n/home/anon/sven-experiments/bench/bench_sharding.py:44-46`, `/n/home/anon/sven-experiments/bench/results_46028086.jsonl`, `/n/home/anon/sven-experiments/slurm_logs/diag_env-46029087.out`, `/n/home/anon/sven-experiments/slurm_logs/bench_shard-46028086.out`, `/n/home/anon/sven-experiments/bench/timing_serial.sbatch`. Scratch analysis: `/tmp/claude-66176/-n-home-anon-sven-experiments/30b30b16-e05d-4e46-8a25-371ea6181950/scratchpad/{res.json,sacct2.psv,waits.psv}`. Nothing in the repo or `experiment_results` was modified; no SLURM jobs touched.