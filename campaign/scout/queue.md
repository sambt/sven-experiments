## 1. Queue waits, per partition actually used

`sacct` records the **allocated** partition for started jobs, so the column below is where a job *ran*, not what was requested (a comma-list submission that wins shows up as a single partition). 719 jobs, 2026-08-28 → now, all QOS=`normal` (the caps come from the *partition* QOS). Jobs that never started are excluded; I found no dependency-held jobs (the one pending job has `Dependency=(null)`, and 130/131 comma-list submissions were `CANCELLED` while pending, median 11 min after submit → deliberate supersede, not dependency).

| allocated partition | n started | med wait | p75 | p90 | max | started <60 s |
|---|---|---|---|---|---|---|
| lab_gpu_priority | 380 | **2.24 h** | 6.33 h | 10.1 h | 26.6 h | 50 |
| lab_gpu | 170 | **4.47 h** | 8.15 h | 10.8 h | 15.6 h | 2 |
| gpu | 12 | 2.88 h | 6.71 h | 10.4 h | 10.8 h | 0 |
| gpu_test | 10 | **3 s** | 4 s | 10 s | 18 s | 10 |

Wait is driven almost entirely by **burst size** (jobs grouped by submit minute, median of each burst's worst wait):

| burst size | bursts | median worst-wait | p90 worst-wait |
|---|---|---|---|
| 1 | 10 | 4 s | 1.1 h |
| 2–5 | 16 | 7.5 min | 9.0 h |
| 6–20 | 13 | 5.6 h | 12.3 h |
| 21–60 | 7 | 6.7 h | 10.2 h |
| 84 (one burst, 2026-09-14 08:59) | 1 | 8.7 h med / **15.0 h max** | — |

That 84-job burst drained at a steady **5.6 job-starts/hour for 15 h** — exactly 16 slots ÷ 2.9 h mean job length. This is the single most useful throughput number I measured.

## 2. Concurrency and how the `node=2` cap really behaves

Reconstructed from Start/End intervals:

- **Peak ever: 17 concurrent GPU jobs** (2026-09-11 17:24) across 7 distinct nodes.
- `lab_gpu_priority` alone: peak **8 jobs on exactly 2 distinct nodes** (`gpunode8a27302`, `gpunode8a27402`), 2026-09-02 12:20.
- `lab_gpu` alone: peak **8 jobs on exactly 2 distinct nodes** (`gpunode8a27202`, `gpunode8a27203`), 2026-09-14 13:28.

So the cap is **2 distinct nodes, not 2 jobs and not 2 GPUs** — 8 single-GPU jobs pack onto 2 four-GPU nodes fine.

**The two caps are independent — verified.** At 2026-09-14 13:28:30 the user had **16 single-GPU jobs running on 4 lab nodes simultaneously**: `27202`×4 + `27203`×4 under `lab_gpu`, `27303`×4 + `27403`×4 under `lab_gpu_priority`. So 2 nodes in each QOS = **16 A100-80GB is genuinely reachable**. Confirmed structurally: `sacctmgr` shows two separate QOS each with `MaxTRESPU node=2`.

Sustained (time-weighted mean concurrent jobs): **3.18 over the 18-day window, 5.05 over the last 7 days, 7.25 over the last 4 days.**

## 3. The `gpu` partition and fairshare

Barely usable. Only **12 allocations in 3 weeks**, peak 4 concurrent, time-weighted mean **0.12 jobs**. Right now: **131/132 GPUs in use, 0 fully-free nodes, 224 pending**. Fairshare: `sshare -U` → `anon_lab/anon  NormShares=0.00778  EffectvUsage=0.0257  FairShare=0.1015` — usage is ~3.3× share, so priority there is weak and getting weaker as you run. Treat `gpu` as **+0–4 opportunistic GPUs**, never as capacity you plan on.

## 4. Outcomes

707 GPU jobs: COMPLETED 514, CANCELLED 153 (133 never started; 17 killed mid-run at median 3.0 h), **TIMEOUT 34**, FAILED 6, OUT_OF_MEMORY 1, RUNNING 10, PENDING 1.

TIMEOUTs are concentrated: `submit_rebuttal_parallel.sh` 20, `cifar10_resnet_scan_labelRegression` 5, `cifar10_resnet_ce_scan` 5, `mnist_scan_ce`/`mnist_scan_labelRegression`/`rebuttal_batchsize_polynomial_scan` 1 each. All 34 sat at elapsed 12:01 — the 12 h wall. **CIFAR and the parallel launcher are the only timeout-risky things.**

Elapsed of COMPLETED jobs that requested 12 h (n=483): **median 0.53 h, p75 2.38 h, p90 5.46 h, max 11.23 h; only 1% exceeded 0.9×limit.** You are requesting ~20× the median need, which costs you backfill priority for nothing.

Cost model for re-run sizing (COMPLETED >1 min, GPU-hours): total **863 GPU-h completed / 1421 GPU-h started** in 18 days. Heaviest: `cifar10_resnet_ce_scan` med 5.51 h/job, `wrap` (gpt2) med 8.25 h, `cifar10_resnet_scan_labelRegression` 3.03 h, `rebuttal_batchsize_polynomial_scan` 3.28 h, `rebuttal_fig5_cifar_paramfrac_scan` 1.81 h, `mnist_scan_ce` 1.50 h; toy_1d/polynomial are **0.05–0.25 h** (i.e. free).

## 5. Whole-node / exclusive requests

Four in history, all `Exclusive=NODE, gres/gpu=4`:

| job | submitted | wait | outcome |
|---|---|---|---|
| `timing_serial_LONG` 46783849 | 09-16 19:13 | **1.2 min** | COMPLETED 13:31:43 on 27403 |
| `timing_serial_SHORT` 46783850 | 09-16 19:13 | **3.2 min** | COMPLETED 06:00:09 on 27303 |
| `optimizer_profile` 46868711 | 09-17 12:14 | **4 s** | COMPLETED 02:00:16 on 27403 |
| `timing_serial_exclusive` 46783764 | 09-16 19:11 | never started | CANCELLED |
| `timing_serial_RERUNS` 46944091 | 09-17 22:20 | **pending 13.4 h** | `Reason=QOSMaxNodePerUserLimit` |

Exclusive is instant *if* an lab node is fully free, and impossible otherwise. **Right now 0 of 8 lab nodes and 0 of 36 `gpu` nodes are fully free.** It also burns one of your two node slots on a single job.

## 6. Current state (as of ~11:45, 2026-09-18)

Nine RUNNING nanoGPT/GPT-2 scan jobs, `sbatch --wrap`, one config each (`[0/1]` progress bar), 12 h limit, `StdOut=/n/home/anon/sven-experiments/slurm_logs/gpt2-469444{59..67}.out`:

- `lab_gpu`: 46944459/64/65 — all on `gpunode8a27203` (1 node, 3 GPUs)
- `lab_gpu_priority`: 46944460/61/63 on `gpunode8a27302`, 46944462 on `gpunode8a27402` (2 nodes, 4 GPUs) → **at cap**
- `gpu`: 46944466 on `gpunode8a22302`, 46944467 on `gpunode8a22303`

Started 08:33–09:15, hard walls 20:33–21:15. Historical gpt2 `wrap` jobs ran med 8.25 h / max 9.39 h → **expect these to finish ~16:45–18:40 today**. Visible configs include SVD bs=16 k=16 rtol=0.001 at lr 0.1 / 0.5 / 1.0.

Also: `timing_serial_RERUNS` 46944091 pending (`bench/timing_serial.sbatch`, exclusive, 1-12:00:00, blocked on the node cap since last night — it will not start while you hold 2 priority nodes and nothing is fully free), plus one CPU `interactive` on `shared`.

Contention on the 8 lab nodes is heavy and long-lived: `another_user` **47 pending jobs / 141 pending GPUs** (3-GPU, 16–24 h each) and holds 27102×3, 27103×3, 27303×3, 27402×3; `mgerdes` 10 pending and holds 4 GPUs on **3-day** walltimes; `arghya` 2 GPUs (3 d, 1.75 d elapsed); `tshelley2002` 1 GPU (3 d, 2.3 d elapsed); `kfraser` 10 pending. Every lab node is at 4/4 or 3/4.

## 7. Recommended planning numbers

**Undocumented levers I found that change the arithmetic:**

| partition | nodes | GPUs | node cap | MaxTime | preempt | free now | pending |
|---|---|---|---|---|---|---|---|
| lab_gpu + lab_gpu_priority | 4+4 | 32 A100-80 | **2 each (independent)** | 3 d | off, tier 4 | 2 | 58 |
| **lab_gpu_requeue** | same 8 | 32 A100-80 | **none (QoS=N/A)** | **7 d** | REQUEUE, tier 3, GraceTime=0 | 2 | — |
| gpu | 36 | 132 A100-80 | none | 3 d | off, tier 3 | **1** | 224 |
| gpu_h200 | 20 up | 80 H200 | none | 3 d | **off**, tier 3 | **1** | 224 |
| gpu_requeue | 432 | mixed A40/A100/H200 | none | 3 d | REQUEUE, tier 2 | many | **3626** |
| **gpu_test** | 10 up | 80 MIG `a100_3g.20gb` (20 GB) | `MaxJobsPU=2`, `MaxSubmitPU=2`, `MaxTRESPU gres/gpu=8` | **12 h** | off, tier 4 | **50** | 1 |

**Sustained concurrent A100-equivalents for a multi-day campaign:**

- **Pessimistic: 6.** Equals the measured 7-day average (5.05) plus a little. This is what you get if another_user's 141-GPU backlog and the three 3-day holds keep the lab nodes at 4/4.
- **Expected: 10–11.** 7–8 from the two lab QOS (you hold 7 right now; the 16 ceiling only materialises when 4 lab nodes have free GPUs) + 1 opportunistic on `gpu` + **~2.5 A100-equiv from gpu_test**. Matches the measured 4-day average of 7.25 plus gpu_test, which you have essentially never used.
- **Optimistic: 18–20.** 16 lab (demonstrated for hours on 09-14) + 2–4 on `gpu`. Requires the other five lab users to be quiet; do not budget on it.
- **gpu_test adds ~2.5 A100-equiv at zero queue wait** (measured 3–18 s), but only through **2 jobs**, so each job must drive 4 MIG devices itself (`CUDA_VISIBLE_DEVICES` fan-out inside one 12 h allocation). 20 GB/slice and no NVLink — fine for toy_1d, polynomial, small-MLP MNIST (which are 0.05–0.25 h/config anyway), wrong for CIFAR-ResNet or nanoGPT.
- **`lab_gpu_requeue` is the biggest unexploited lever: no node cap on your own lab's 32 A100s, 7-day walltime.** It could plausibly double the expected number. Cost: `PreemptMode=REQUEUE` with `GraceTime=0`, preempted by tier-4 `lab_gpu*` jobs — i.e. by your own labmates whenever they queue. Only worth it if the runner appends results per-config and resumes; you have exactly **one** historical job there (`_validate2_rq.sbatch`, 54 s), so it is untested at scale and should be piloted before the campaign, not during.

**On sharding, from the evidence:** the `node=2` cap is on distinct nodes, so 8 single-GPU jobs already saturate your ceiling in each QOS — **packing configs into fewer jobs does not raise the GPU ceiling.** Its real value is different and still large: (a) burst waits scale hard with job count (84 jobs → 15 h tail; 1 job → 4 s), so 16 long jobs each consuming an internal config queue pay one ~2–4 h wait instead of 84 fresh ones; (b) holding a GPU across configs stops other users grabbing the free GPUs on nodes you already occupy — the mechanism that has you at 7 of 16 right now. Against it: 34 jobs already died at the 12 h wall, so a multi-config shard **must** append per-config results and be resumable or a timeout costs the whole shard. Also stop requesting 12 h when the median completed job is 0.53 h, and drop `--exclusive` entirely (it consumes a whole node slot and cannot start today).

*Verified by running:* all `sacct`/`squeue`/`sinfo`/`sshare`/`sprio`/`sacctmgr`/`scontrol` numbers and the Python interval reconstruction (`/tmp/claude-66176/-n-home-anon-sven-experiments/30b30b16-e05d-4e46-8a25-371ea6181950/scratchpad/an.py`, `an2.py`, data in `jobs.psv`). *Read only, not executed:* the nine `gpt2-*.out` log headers used to identify current jobs. *Inference, not measurement:* the "expected 10–11" and "optimistic 18–20" planning figures, and the claim that `lab_gpu_requeue` would lift the ceiling (structurally supported by `QoS=N/A`, but never exercised at scale by this user).