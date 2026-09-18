# Cloud GPU rental decision memo

**Prepared 2026-09-18. All prices marked LIVE were read off the provider's own page or API on 2026-09-18.**

---

## 0. Bottom line

1. **Vast.ai is not a viable A100-80GB source today.** The entire platform has **13 rentable verified A100-80GBs with driver >= 570**, only 6 of them in the US, and **zero** 8x A100-80 or 8x H100 nodes. RunPod is both cheaper and vastly more available for 80 GB cards.
2. **Vast.ai is 3-4x cheaper than RunPod for the cards workload A actually needs** — A100-**40**GB at $0.336-0.538/GPU-hr (RunPod no longer sells A100-40 or V100 at all).
3. **RunPod spot is real and ~53% off**, putting A100-**80** at an estimated ~$0.56/hr — the best 80 GB price anywhere, and enough to run both workloads on one provider. One undocumented behaviour blocks this (see 5.1) and must be tested today.
4. **Workload B should not go on an A100-80 at list price.** B is a plain transformer with no fp64, so the weak-fp64 48 GB cards (RTX 6000 Ada, L40S) are cheap for it. A100-80/H100 on *spot* are equally cheap and finish faster.
5. **Skip L40S / A6000 / A40 / RTX 4090 for workload A.** Weak fp64 *and* weak bandwidth is the worst combination in the table — they cost more per unit of work than renting an A100-80.
6. Budget **$200-350** for the recommended mix; **$480-670** for the conservative all-RunPod-on-demand version.

---

## 1. Verified prices

### 1.1 RunPod — LIVE from https://www.runpod.io/pricing, observed 2026-09-18

$/GPU-hr. Spot column is **estimated**, not published (see 5.1).

| GPU | Community | Secure | Est. spot (Comm, -53%) |
|---|---|---|---|
| A100 PCIe 80GB | **1.19** | 1.59 | ~0.56 |
| A100 SXM 80GB | **1.39** | 1.59 | ~0.66 |
| H100 PCIe | 1.99 | 2.89 | ~0.94 |
| H100 SXM | 2.69 | 3.49 | ~1.27 |
| H100 NVL | 2.59 | 3.19 | ~1.22 |
| H200 | 3.59 | 4.59 | ~1.70 |
| B200 | 5.98 | 6.79 | ~2.83 |
| B300 | 6.94 | 7.89 | — |
| L40S | 0.79 | 1.09 | ~0.37 |
| L40 | 0.69 | 0.82 | — |
| RTX 6000 Ada | 0.74 | 0.84 | ~0.35 |
| RTX Pro 6000 | 1.69 | 2.09 | — |
| RTX A6000 | 0.33 | 0.53 | ~0.16 |
| A40 | 0.35 | 0.49 | — |
| RTX 5090 | 0.69 | 0.99 | ~0.33 |
| RTX 4090 | 0.34 | 0.74 | ~0.16 |
| RTX 3090 | 0.22 | 0.50 | — |
| RTX A5000 | 0.16 | 0.27 | — |
| L4 | 0.44 | 0.49 | — |

**RunPod no longer lists A100-40GB or V100 in any tier.**

### 1.2 Vast.ai — LIVE from its public marketplace API, observed 2026-09-18

Queried `PUT https://console.vast.ai/api/v0/search/asks/` (no auth needed). Filtered to `verification == "verified"` **and** `driver_vers >= 570000000`, normalised to $/GPU-hr (`dph_total / num_gpus`). "Min bid" is the floor for an interruptible bid, also per GPU.

| Pool | Offers | **Total GPUs available** | Cheapest | Median | Min bid | Notes |
|---|---|---|---|---|---|---|
| A100-80 SXM4 | 5 | **13** | 1.028 | 1.028 | 0.667-1.00 | 1x Taiwan; 2x/4x Mississippi @ 1.615-1.696 |
| A100-80 PCIe | 0 | **0** | — | — | — | none at all |
| **A100-40 SXM4** | 13 | 26 | **0.336** | 0.672 | 0.333 | cheapest = California, reliability 0.92 |
| A100-40 PCIe | 5 | 9 | 0.431 | 0.831 | 0.427 | then a jump to $0.80 (Japan) |
| **V100-32GB** | 8 | 28 | **0.107** (8x box) | 0.242 | 0.107 | all US/BG; Minnesota box has 7.7 Gb/s up |
| RTX 5090 | 118 | **454** | 0.402 | 0.553 | 0.293 | enormous headroom; many in China |
| RTX 4090 | 21 | — | 0.390 | 0.536 | 0.160 | |
| RTX 6000 Ada | 5 | — | 0.496 | 0.696 | 0.267 | 48 GB |
| L40S | 9 | 17 | 0.535 | 0.668 | 0.200 | |
| RTX A6000 | 3 | — | 0.469 | 0.604 | 0.240 | |
| H100 SXM | 3 | — | 3.139 | 4.162 | 2.600 | **no 8x H100 rentable** |
| H100 NVL | 6 | — | 2.722 | 2.851 | 0.397 | 94 GB |
| H200 / H200 NVL | 1 / 2 | — | 4.740 / 3.936 | — | 4.539 / 1.467 | |
| A40 | 1 | — | 1.136 (unverified) | — | 0.400 | |

Other live observations from the same query:
- **Only 1 A100 offer on the entire platform is datacenter-tier** (`hosting_type == 1`). Across a 373-offer sample of the datacenter-class GPUs: 336 `hosting_type 0` (community), 37 `hosting_type 1` (datacenter). Verification split: 201 verified, 96 unverified, 76 **de**verified.
- Per-host storage $0.133-0.867/GB/month; egress $0.67-53.33/TB. These vary 6x and 80x between hosts — check before renting.
- The two Swedish A100-80 offers ($1.055) run driver **535.230.02** and are excluded by the >= 570 filter.

### 1.3 Alternative providers — LIVE own pricing pages, observed 2026-09-18

| GPU | Lambda | Verda (ex-DataCrunch) | Hyperstack | Crusoe | TensorDock | Thunder | Nebius | DigitalOcean |
|---|---|---|---|---|---|---|---|---|
| A100-80 SXM | 2.79 (8x only) | 1.74 (**spot 0.87**) | 1.60 | 2.30 | 1.80 | **1.09** | n/o | n/o |
| A100-80 PCIe | n/o | n/o | **1.35 (spot 1.08)** | 2.00 | 1.50 | — | n/o | n/o |
| A100-40 | 1.99 | 1.26 (**spot 0.63**) | n/o | n/o | — | n/o | n/o | n/o |
| H100 PCIe | 3.29 | n/o | 2.50 (spot 2.00) | n/o | n/o | 3.20 | n/o | n/o |
| H100 SXM | 4.29 (8x: 3.99) | 3.35 (spot 1.67) | 3.20 | 3.90 | 2.25 | n/o | 3.85 | 4.41 |
| H200 | n/o | 4.37 (spot 2.19) | 3.99 | 4.29 | n/o | n/o | 4.50 | 4.47 |
| L40S | n/o | 1.45 (spot 0.73) | "L40" 1.00 | 1.50 | "L40" 0.95 | "L40" 0.79 | 1.55-1.82 | 1.57 |
| RTX A6000 | 1.09 | 0.60 (spot 0.30) | 0.50 (spot 0.40) | n/o | 0.45 | 0.35 | n/o | n/o |
| V100 | 0.79 (VRAM unconfirmed) | 0.18 (**16GB**) | n/o | n/o | 0.17 (**16GB**) | n/o | n/o | n/o |

**Beats RunPod's A100 baseline:** Thunder Compute $1.09 (undercuts Community $1.19 and Secure $1.59); Hyperstack spot $1.08; Verda spot $0.87. **Nothing beats RunPod H100 PCIe Community $1.99** on a true PCIe part — closest is Hyperstack spot $2.00.

**Self-serve with a card, no KYC, confirmed:** Lambda, Hyperstack, DigitalOcean, TensorDock, Crusoe (standard tiers), Verda. **Harder gate:** Nebius (mandatory bank card; top tiers sales-only). **Unconfirmed:** Thunder Compute.
**8x self-serve confirmed:** Lambda (8x A100-80 SXM @ $2.79/GPU ≈ $22.3/node-hr; 8x H100 SXM @ $3.99/GPU ≈ $31.9/node-hr), Crusoe, DigitalOcean, Thunder Compute ($25.60/hr for 8x H100 PCIe).
**Egress explicitly free:** Lambda, Hyperstack, Crusoe, TensorDock, Thunder, RunPod. **Not stated:** Verda, Nebius.

Caveats: Thunder's $1.09 single-GPU A100 is inconsistent with the $1.49/GPU implied by its 4x A100 bundle ($5.96/hr) — **verify before committing**. "L40" != "L40S" at Hyperstack/TensorDock/Thunder. **Nebius has dropped A100 entirely** and has an announced 17-21% rise on H100/H200/B200 effective 2026-10-01 (search-derived, unverified). DataCrunch -> Verda rebrand confirmed by a 301 redirect; may mean a fresh account/ToS.

**True V100-32GB exists only on Vast.ai.** Every V100 SKU on all eight conventional providers is 16 GB.

---

## 2. Memory: which capture mode each card can run

Corrected figures: **chunked capture = ~6.9 GB allocated / ~18.5 GB reserved. Full capture = 23 GB allocated / 33 GB reserved.** Reserved is what must fit in VRAM, so reserved is the binding constraint.

| VRAM | Cards | Chunked (18.5 GB res.) | Full (33 GB res.) |
|---|---|---|---|
| 16 GB | V100-16, L4 | **NO** (18.5 > 16) | NO |
| 24 GB | RTX 4090, 3090, A5000 | **tight** — 77% occupancy | NO |
| 32 GB | RTX 5090, **V100-32** | yes, 58% | NO (33 > 32) |
| 40 GB | **A100-40** | yes, 46% | **tight** — 83% |
| 48 GB | L40S, A6000, 6000 Ada, A40 | yes, 39% | yes, 69% |
| 80 GB+ | A100-80, H100, H200 | yes | yes |

Consequences:
- **16 GB cards are out entirely** — this is why the cheap 16 GB V100s at Verda/TensorDock ($0.17-0.18) are useless and only Vast's 32 GB V100s are interesting.
- **24 GB cards (4090) can only do chunked, at 77% occupancy.** Feasible but fragmentation-prone; set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and expect to raise the chunk count. Do not plan a 60-hour campaign on this margin.
- **A100-40 runs chunked comfortably and full capture only barely (83%).** Treat A100-40 as a chunked-capture-only card in planning.
- **Workload B's 33 GB peak means >= 48 GB in practice.** On a 40 GB card that is 83% occupancy with no headroom — do not put B on an A100-40.

---

## 3. fp64 arithmetic and price/performance for workload A

### 3.1 The fp64 term

fp64 Gram cost = `2 * B^2 * P` = `2 * 128^2 * 11.2e6` = **3.67e11 FLOP/step**.

Assuming an A100 DGEMM achieves ~12 TFLOP/s on this tall-skinny shape (FP64 tensor cores, 19.5 peak, derated for shape): **31 ms, or ~8% of a 400 ms step.** The other ~369 ms is the jacrev capture, which writes a 128 x 11.2M fp32 Jacobian (5.7 GB) per step and is therefore substantially HBM-bandwidth-bound.

Because the split between "bandwidth-bound" and "compute-bound" capture is the single biggest modelling uncertainty, every figure below is **bracketed by two independent models**:
- **Roofline:** capture time scales as 65% / HBM bandwidth + 35% fixed.
- **DLPerf:** capture time scales as 1 / DLPerf, using max unshared DLPerf per model measured live on Vast today (A100-80 = 129, A100-40 = 95, H100 SXM = 293, H200 = 382, V100-32 = 26, 5090 = 219, 4090 = 107, 6000 Ada = 113, L40S = 90, A6000 = 55).

Achieved fp64 assumed: A100 12, H100 PCIe 30, H100 SXM / H200 45, V100 7.0, 5090 1.40, 4090 1.10, L40S 1.20, A6000 1.05 TFLOP/s.

### 3.2 Table — workload A, $ per A100-80-equivalent hour

| GPU | VRAM | capture mode | fp64 ms/step | slowdown vs A100-80 | best $/hr LIVE | **$/A100-eq-hr** | spot $/hr | spot $/eq-hr |
|---|---|---|---|---|---|---|---|---|
| **V100-32 SXM2** (Vast 8x) | 32G | chunked | 52 | **2.24 - 4.71** | 0.107 | **0.24 - 0.50** | 0.107 | 0.24 - 0.50 |
| **A100-40 SXM** (Vast) | 40G | chunked | 31 | **1.19 - 1.33** | 0.336 | **0.40 - 0.45** | 0.333 | 0.40 - 0.44 |
| RTX 5090 (Vast) | 32G | chunked | **262** | 1.20 - 1.66 | 0.402 | 0.48 - 0.67 | 0.400 | 0.48 - 0.66 |
| A100-40 SXM rel 0.97 (Vast) | 40G | chunked | 31 | 1.19 - 1.33 | 0.538 | 0.64 - 0.72 | 0.533 | 0.63 - 0.71 |
| RTX 4090 (Vast) | 24G | chunked (tight) | **334** | 1.95 - 2.37 | 0.390 | 0.76 - 0.92 | 0.160 | 0.31 - 0.38 |
| A100-80 SXM (Vast TW) | 80G | both | 31 | 1.00 | 1.028 | 1.03 | 1.000 | 1.00 |
| L40S (Vast) | 48G | both | **306** | 2.09 - 2.50 | 0.535 | 1.12 - 1.34 | 0.200 | 0.42 - 0.50 |
| A100-80 PCIe (RunPod C) | 80G | both | 31 | 1.03 - 1.05 | 1.190 | 1.23 - 1.24 | ~0.562 | **0.58 - 0.59** |
| RTX A6000 (Vast) | 48G | both | **350** | 2.79 - 3.04 | 0.469 | 1.31 - 1.43 | 0.240 | 0.67 - 0.73 |
| A100-80 SXM (RunPod C) | 80G | both | 31 | 1.00 | 1.390 | 1.39 | ~0.657 | 0.66 |
| H100 PCIe (RunPod C) | 80G | both | 12 | 0.57 - 0.95 | 1.990 | 1.14 - 1.90 | ~0.940 | 0.54 - 0.90 |
| H100 SXM (RunPod C) | 80G | both | 8 | 0.43 - 0.71 | 2.690 | 1.15 - 1.91 | ~1.271 | 0.54 - 0.90 |
| H200 (RunPod C) | 141G | both | 8 | 0.33 - 0.60 | 3.590 | 1.19 - 2.15 | ~1.696 | 0.56 - 1.02 |

### 3.3 Reading the table

- **Are 24 GB consumer cards viable for A?** Memory-wise yes, but only in chunked mode at 77% occupancy. Performance-wise the fp64 Gram goes from 8% of the step on an A100 to **~40% on a 4090** (334 ms of a 731-949 ms step). The 4090 still lands at $0.76-0.92/A100-eq-hr on-demand — worse than an A100-40 and no better than a RunPod A100-80. **Verdict: usable as overflow capacity, never as the primary plan.** The 5090 is the better consumer card (1792 GB/s, fp64 1.64) at $0.48-0.67.
- **Skip L40S, A6000, A40 entirely for A.** Weak fp64 *and* weak bandwidth. At $1.12-1.43/A100-eq-hr they cost more per unit of work than renting an actual A100-80. (Their on-demand prices look cheap; the fp64 penalty eats the discount.)
- **H100 / H200 are a trap for A.** H100 **PCIe** has the same 2039 GB/s as an A100-80 SXM, so under the bandwidth-bound model it is only ~5% faster for 1.7x the price. Even under the favourable DLPerf model it never beats an A100-40 on $/work. Buy H100 for *wall clock* on B, never for $/work on A.
- **Estimated H100/H200 spot ($0.54-0.90/eq-hr) is competitive**, but rests on an unverified spot discount.

### 3.4 Confidence — read this before acting on the V100 number

Confidence in the slowdown estimates, highest to lowest:

- **A100-40: high.** Same SM count, clocks and fp64 path as an A100-80; only HBM2 (1555) vs HBM2e (2039) differs. Both models agree within 12% (1.19x vs 1.33x). The $0.40-0.45/A100-eq-hr figure is robust.
- **RTX 5090 / 4090: medium.** fp64 rate is well known (1/64 of fp32) and the fp64 term dominates the uncertainty, which makes the estimate *more* stable, not less. DLPerf flatters them because it is a mixed/fp16 benchmark and our capture is fp32/TF32 — so treat the slow end of each range (1.66x, 2.37x) as the planning number.
- **H100 PCIe / SXM, H200: low-medium.** The 0.57-0.95x spread on H100 PCIe *is* the bandwidth-vs-compute-bound question, unresolved. Do not commit to H100 for A on the strength of the optimistic end.
- **V100-32: LOW. This is the weakest number in the memo, and it drives the cheapest option.** The two models disagree by more than 2x (2.24x vs 4.71x), so $/A100-eq-hr is only bounded to $0.24-0.50. Specific reasons to distrust the optimistic end:
  - **No TF32.** An A100 runs the convolution and matmul path on TF32 tensor cores at 156 TFLOP/s; a V100 has no TF32 at all and falls back to 15.7 TFLOP/s fp32 FMA. My roofline model scales only with HBM bandwidth (900 vs 2039 GB/s) plus a flat 1.25x fudge for this, which almost certainly understates the hit if any part of capture is compute-bound. The DLPerf ratio (129/26 = 5.0x) is the empirical signal and it is much worse.
  - **No bf16.** Irrelevant if capture is fp32 throughout, but it removes any fallback if you wanted to trade precision for speed.
  - **torch.func / jacrev on sm_70.** The vmap-of-grad path is far less exercised and less optimised on Volta than on Ampere; silent fallbacks to slow kernels are plausible and would not show up in either model.
  - **PCIe vs SXM2 is *not* a concern here.** Both V100-32 variants run 900 GB/s HBM2 (only the V100S PCIe-32 is 1134), and NVLink is irrelevant because the runs are independent single-GPU jobs. The Vast offers in question are 8x SXM2 boxes.
  - **fp64 is the one bright spot**: V100 has genuine 7.0-7.8 TFLOP/s fp64, so the Gram matmul costs only ~52 ms — 6x better than a 5090 despite the V100 being five years older. That is precisely why it is worth measuring rather than dismissing.
  - **Net:** V100-32 is *potentially* the cheapest capacity on the table at $0.24/A100-eq-hr and *potentially* no better than an A100-40 at $0.50. A 30-minute measurement resolves a swing of roughly $100-150 on the campaign. **Do not build the plan on V100 until measured; A100-40 at $0.40-0.45 is the robust cheap pick.**

---

## 4. Per-workload recommendation

### 4.1 Workload A — CIFAR Sven (200-300 A100-hours)

| Option | Cost for 200-300 A100-h | Notes |
|---|---|---|
| **Vast A100-40 SXM @ $0.336-0.538** | **$80-215** | Recommended. Chunked capture only. Assemble 5-8 GPUs from several offers. |
| Vast V100-32 8x box @ $0.107 | $48-151 | Cheapest *if* the step time confirms; see 3.4. Chunked only. Solves C too. |
| Vast RTX 5090 @ $0.402 | $96-200 | 454-GPU pool = unlimited overflow. Chunked only (32 GB). |
| RunPod A100-80 PCIe spot @ ~$0.56 | $116-177 | If spot survives preemption; 80 GB so it also runs full capture. |
| RunPod A100-80 PCIe on-demand @ $1.19 | $246-372 | The no-thinking-required option. |
| Vast L40S / A6000 | $223-430 | **Do not.** |

### 4.2 Workload B — GPT-2-small (75-90 A100-hours, 33 GB peak, so >= 48 GB)

**Important: B is a plain transformer with no fp64 at all, so the fp64 penalty that rules out the 48 GB cards for A does not apply to B.** Scaling by DLPerf alone:

| Option | slowdown | $/hr | $/A100-eq-hr | Cost for 75-90 A100-h | GPU-hours needed |
|---|---|---|---|---|---|
| **Vast RTX 6000 Ada 48GB @ $0.496** | 1.14 | 0.496 | **0.57** | **$43-51** | 86-103 |
| RunPod A100-80 PCIe **spot** ~$0.56 | 1.03 | 0.56 | 0.58 | $44-53 | 77-93 |
| RunPod H100 PCIe **spot** ~$0.94 | 0.59 | 0.94 | 0.55 | $42-50 | **44-53** |
| Vast L40S 48GB @ $0.535 | 1.43 | 0.535 | 0.77 | $58-69 | 107-129 |
| RunPod RTX 6000 Ada Comm $0.74 | 1.14 | 0.74 | 0.84 | $63-76 | 86-103 |
| RunPod A100-80 PCIe on-demand $1.19 | 1.03 | 1.19 | 1.23 | $92-111 | 77-93 |
| RunPod H100 PCIe on-demand $1.99 | 0.59 | 1.99 | 1.17 | $88-106 | 44-53 |

**Recommendation for B: RunPod A100-80 PCIe (spot if it survives the 5.1 test, else on-demand), 2 cards for ~45-55 h.** It is within $10 of the cheapest option, gives 80 GB of headroom over a 33 GB peak, keeps B on the same provider and SSH path as everything else, and needs only ~2 concurrent cards. **If the 60-hour wall clock turns out to be binding, switch B to H100 PCIe** — same cost, roughly half the GPU-hours (44-53 vs 77-93). **If you want to save ~$50 and accept 48 GB + a third machine, 2x RTX 6000 Ada on Vast at $0.496 is the cheapest credible option.** Do **not** put B on an A100-40 (83% occupancy, no headroom).

### 4.3 Workload C — MLP overflow (optional, kernel-launch-bound)

What matters is $/hour per (GPU + ~8 fast CPU cores), not GPU throughput.
- **Best: the Vast 8x V100-32 Minnesota box, $0.86/hr for all 8 GPUs, 9-10 effective cores/GPU, 7.7 Gb/s uplink.** ~$20 for a full day, and it doubles as the workload-A test bed.
- Alternatives: Vast RTX 3090/A5000 class at $0.16-0.22/hr; RunPod RTX A5000 Community $0.16. Check `cpu_cores_effective / num_gpus >= 8` in the offer before renting — several cheap offers give only 8-12 cores for the whole box.
- Do **not** pay for A100s here; the GPU is idle.

### 4.4 Workload D — timing/benchmark runs

Stays on the university A100s. Out of scope. **Note:** keep all *timing-sensitive* measurements on the cluster's A100s. Marketplace hosts vary in CPU, PCIe generation, disk bandwidth and whether the GPU is shared (`gpu_frac < 1`), so step times measured there are not comparable to the cluster's and must not enter the paper's timing tables.

---

## 5. Hidden costs and gotchas

### 5.1 The one thing that must be tested before committing to spot

**RunPod's docs never state whether a spot preemption is a "stop" (volume disk at `/workspace` survives) or a "terminate" (volume disk wiped).** The general rules are documented — stopping preserves `/workspace` and clears the container disk; terminating "permanently deletes all data not stored in a network volume" — but which one preemption triggers is not stated anywhere. **The preemption warning period is also undocumented**; third-party sources give 5 s, 30 s and 5 min, mutually contradictory and none traceable to an official page. RunPod's own blog says only "Spot instances can be interrupted without notice."

Mitigation: attach a **Global Volume** (region-independent, `$0.09/GB/month` + IOPS, "any Pod, anywhere, can mount it at startup") or rsync out on a loop. A Global Volume is the better choice over a network volume, which pins the pod to one datacenter. Caveat: at $0 balance a global volume "is flagged and permanently deleted after 15 days".

### 5.2 RunPod

- **Proxied SSH cannot do rsync.** The docs state the basic `ssh <pod-id>@ssh.runpod.io` method "does not support commands like SCP (Secure Copy Protocol) or SFTP" — so no rsync and no port forwarding. You **must** rent an instance with a public IP and sshd in-container. Secure Cloud pods "always have a public IP address"; on **Community Cloud you must pass `supportPublicIp: true` at creation** ("If null, the Pod might not have a public IP"). Direct TCP maps a dynamic external port to internal 22, exposed as `RUNPOD_TCP_PORT_22`; rsync then works via `rsync -e "ssh -p <port>"` (inferred from the documented scp mechanism — the docs show scp, not rsync, explicitly).
- **Filter by CUDA version, not driver.** There is no driver-version filter. The API takes `allowedCudaVersions`, accepting `13.0, 12.9, 12.8, 12.7, 12.6, 12.5, 12.4, 12.3, 12.2, 12.1, 12.0, 11.8`. Pass `["12.8","12.9","13.0"]` to guarantee a CUDA-12.8-capable host. The UI has the equivalent filter.
- **The container disk is wiped on stop — and that is where `uv` installs by default.** Put the venv and the repos under `/workspace` (volume disk) or every stop costs you a full `uv sync`. Container disk "Lost on stop/restart"; `/workspace` "Retained until Pod deleted".
- **tmux is not preinstalled.** `apt install tmux` in the bootstrap. RunPod documents and recommends it ("Use TMUX for any process that takes longer than a few minutes"), but notes "TMUX sessions don't persist across Pod restarts".
- **Stopped pods keep billing disk at double the running rate**: volume disk $0.10/GB/month running, **$0.20 stopped**. Container disk is not billed when stopped (it is wiped). Network volume $0.07/GB/month (first TB, $0.05 beyond), unaffected by pod state, and **locked to its datacenter**.
- **Zero balance is destructive.** "Pods are automatically stopped when your account balance reaches $0" — and pods **without** a network volume are then "terminated, and their data can't be recovered". Keep the balance funded.
- **Billing is per second** for compute and storage, **with no fees for data ingress or egress**. You must hold "at least one hour's worth of credits for your selected configuration to deploy an on-demand instance" (~$13 for an 8x A100 box).
- Base images look like `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. You will be installing your own torch 2.9.1+cu128 via `uv` anyway; pick any recent CUDA 12.8 image and let uv own the env.
- No documented restriction on outbound SSH/rsync ports, or on long-running processes.

### 5.3 Vast.ai

- **Interruptible is benign for this workload.** Being outbid **pauses** the instance, it does not destroy it: "an interruptible instance can be abruptly paused at any time if another user places a higher bid", pausing "stops all processes that are running", and crucially **"You can still transfer data off a stopped instance."** Since your workers lose at most ~1 h of in-flight work and results are files on disk, **bid at or slightly above `min_bid` and accept eviction.**
- **You do not need a persistent volume on Vast — but you do need frequent rsync-out**, because (a) a paused instance keeps billing storage and cannot be restarted on demand, and (b) recovering data from a paused instance is a manual step you do not want on the critical path. Rsync results out every 10-15 minutes.
- **Vast charges bandwidth in both directions**: "You are charged bandwidth prices for every byte sent or received to or from the instance, regardless of what state it is in." Observed $0.67-53.33/TB depending on host. For 60 GB out that is $0.04-3.20 — negligible *unless* you land on the $53/TB Romanian host. Check `internet_up_cost_per_tb` before renting.
- **Storage bills while the instance merely exists**, stopped or not: "Stopping an instance does not avoid storage costs." Observed $0.133-0.867/GB/month.
- **Filter aggressively.** Recommended filter set: `verification == "verified"`, `reliability > 0.97`, `driver_vers >= 570000000`, `inet_up > 500` Mb/s, `direct_port_count > 0`, `gpu_frac == 1.0` (exclude shared/partitioned GPUs), and geolocation in US or EU.
  - The cheapest A100-40 ($0.336) has **reliability 0.9195**, and the $0.427 one has **0.8625**. I would pay $0.538 for the Utah host at 0.9744 for a multi-day campaign.
  - Most cheap 5090s are in **China** — bad for rsync back to a US university cluster, and possibly blocked.
  - 76 of 373 sampled offers are **de**verified. Do not rent those.
- **SSH:** Vast connects you to a **tmux session by default** and advises against disabling it. scp/sftp are supported. `direct_port_count > 0` and `static_ip` indicate direct TCP ports are available.
- **No 8x A100-80 and no 8x H100** are rentable today (verified tier). Vast cannot supply a single 8-GPU 80 GB box.

### 5.4 Driver / CUDA / torch pinning

- Sources today state CUDA 12.8 wants **driver R570+**. CUDA minor-version compatibility *may* let cu128 wheels run on 535-series drivers for Ampere, but do not bet a 60-hour campaign on it. Keeping the >= 570 filter costs almost nothing on Vast (nearly all offers now run 580-595); it excludes only the two Swedish A100-80s (535.230.02).
- **Blackwell (RTX 5090) genuinely requires >= 570** — there is no minor-version-compatibility escape there.
- **torch 2.9.1+cu128 still supports V100 (sm_70).** Volta is dropped only from **2.11** cu128/cu129 binaries, to allow a cuDNN 9.15.1 update ([pytorch#172351](https://github.com/pytorch/pytorch/issues/172351)). Your pin is safe — **do not let anyone bump torch past 2.10 while V100s are in the plan.**

### 5.5 Availability risk for an 8x node at short notice

- **Vast: no.** Zero rentable verified 8x A100-80 or 8x H100 today.
- **RunPod: probably, unverified.** Instant Clusters cover A100/H100/H200/B200 at 2-8 *nodes* (16-64 GPUs), "deploy in minutes"; a single 8-GPU pod is an ordinary pod with no separate quota step beyond the account spend limit. But **no doc guarantees capacity**, and there is no SLA for on-demand or spot on either cloud. Verify in the console before planning around it.
- **Guaranteed-capacity fallback: Lambda**, 8x A100-80 SXM at $2.79/GPU-hr ($22.3/node-hr), self-serve, no KYC. Roughly 2x the price of RunPod Community; worth it only if a single box is operationally essential.
- Note that the recommended plan deliberately does **not** need an 8x box: workload A's runs are independent, so 3-4 smaller machines are fine. That is the cheaper trade, at the cost of more orchestration.

---

## 6. Recommended rental plan

Target: **300-450 A100-equivalent hours over ~60 hours of wall clock**, starting ~36 hours from now (i.e. ~2026-09-20 early). That is **5.0-7.5 concurrent A100-equivalents**.

### 6.1 Recommended (balanced)

| Role | Resource | Hours | Cost |
|---|---|---|---|
| Workload A | Vast, 5-8x **A100-40 SXM** @ $0.34-0.54, verified, rel > 0.97, US | ~55 | $95-215 |
| Workload B | RunPod, 2x **A100-80 PCIe** Community, public IP, `allowedCudaVersions 12.8+` | ~50 | $56 (spot) - $119 (on-demand) |
| Workload C | Vast, 1x 8x **V100-32** box @ $0.86/hr total | ~24 | ~$21 |
| Overflow | Vast **RTX 5090** @ $0.40 as needed (454-GPU pool) | — | $0-80 |
| Storage + egress | RunPod ~$2-4, Vast ~$8, egress ~$1 | — | ~$11-13 |
| | | **Total** | **$185-450, expect ~$250-330** |

### 6.2 Cheaper but riskier

All-Vast, interruptible bids at `min_bid`, accepting eviction:
- A on 2x 8x **V100-32** boxes ($0.107/GPU-hr) = $95 for 55 h -> 16 GPUs, 3.4-7.1 A100-equivalents **(range reflects the V100 uncertainty in 3.4 — this variant is only cheap if the measurement comes in at the optimistic end)**
- plus **RTX 5090** overflow at $0.29-0.40 bid
- B on 2x Vast **RTX 6000 Ada** 48 GB @ $0.496 = $50
- **Total ~$165-345.** Risks: eviction churn, no 80 GB card anywhere in the plan, marketplace hosts in 3-5 locations, and the V100 step time unmeasured.

### 6.3 Safer

- RunPod **Secure** Cloud, 8x A100-80 SXM @ $1.59/GPU-hr = $12.72/hr x 60 h = **$763**. One box, guaranteed public IP, T3/T4 datacenter, 480 A100-hours, covers A and B, no marketplace variance.
- Or split: 4x A100-80 Secure (60 h) + 2x H100 PCIe Secure (50 h) = **$671**.
- Or RunPod **Community** on-demand 8x A100-80 SXM = **$667**.
- Or, if a single guaranteed 8x box is essential: **Lambda 8x A100-80 SXM, $1,339** for 60 h.

### 6.4 What I would actually do

Spend $3 and 90 minutes on the tests in section 7 first. Then: **workload A on Vast A100-40s, workload B on RunPod A100-80 PCIe, workload C on the Vast V100 box, RTX 5090s as elastic overflow. Expect $250-330.** Escalate to 6.3 only if the tests expose a blocker (spot wipes data *and* on-demand A100-80 capacity is short, or Vast's A100-40 hosts prove flaky).

---

## 7. What to test / reserve TODAY (total burn < $5)

Ordered by value. Items 1-3 are the ones that actually change the plan.

1. **Measure workload A's step time on a Vast A100-40** (~$0.34/hr, California, driver 595, 1 hour). Run `uv sync` with torch 2.9.1+cu128, confirm `torch.cuda` initialises, run workload A in **chunked** mode, and instrument the fp64 Gram separately with `torch.cuda.Event` so you get the fp64 share as a measured number rather than my 8% estimate. **This single measurement validates the whole cheap-A100-40 thesis.**
2. **Measure workload A on the Vast 8x V100-32 Minnesota box** ($0.86/hr for the box, 30 min). My estimate spans 2.24-4.71x and that spread is worth $100-150 on the campaign. Confirm chunked capture fits in 32 GB (18.5 GB reserved should leave room) and that the torch.func/jacrev path does not fall back to slow kernels on sm_70. If it lands under ~2.5x, this is the cheapest capacity available and also solves workload C.
3. **Test RunPod spot preemption semantics** (a few cents). Launch the cheapest spot pod, write a sentinel file to `/workspace`, terminate/preempt it, and check whether the volume disk survived. This resolves the undocumented behaviour in 5.1 and is worth ~$150-400 on the plan (spot vs on-demand for both A and B).
4. **Validate the whole operational path end to end on one RunPod A100-80 PCIe Community pod** (~$1.19, 30 min), created **with `supportPublicIp: true`** and **`allowedCudaVersions: ["12.8","12.9","13.0"]`**: direct-TCP SSH from the cluster, `apt install tmux`, `uv sync` into `/workspace`, a real `rsync` of ~5 GB back to the cluster with throughput measured, and confirm 80 GB cards are actually available in the region you want. Also confirms B's memory headroom.
5. **Fund both RunPod and Vast today** (see section 8) so neither is the blocker on a weekend.
6. **Reserve nothing long-term** until items 1-3 land. Every option in this memo is hourly and self-serve; there is no reservation that helps a 60-hour job, and Vast's reserved/prepaid discounts (1-6 months) are irrelevant at this timescale.

Optional, if time permits: spot-check **Thunder Compute's $1.09 A100-80** claim (cheapest A100-80 list price found, but internally inconsistent with their 4x bundle) and **Hyperstack's $1.08 A100 spot**. Either would beat RunPod Community, but neither is needed for the plan to work.

---

## 8. Account setup lead times

**Nothing here should block a weekend rental.** Both primary providers are same-hour, self-serve, credit-card.

- **RunPod** — prepaid credits. "Start with as little as $10" (prepaid *cards* need $100/transaction). **Default spending limit $80/hour across all resources**, which increases automatically with account history — ample for an 8x A100 box at ~$11-13/hr, so **no quota request is needed for the recommended plan**. KYC is required only for crypto payments. Larger clusters: email help@runpod.io. Credits are non-refundable once deposited. Fund ~$400 to be safe.
- **Vast.ai** — needs a verified credit card and verified email; prepaid credit with optional auto-billing of any negative balance. Minimum deposit is not documented. No KYC found for ordinary rentals. Fund ~$200.
- **If a fallback is needed:** Lambda, Hyperstack, DigitalOcean, TensorDock, Crusoe and Verda are all credit-card-and-go with no KYC found today. **Avoid Nebius** for a weekend start (mandatory billing setup, A100 discontinued, sales gate on top tiers). Thunder Compute's onboarding gate is genuinely unconfirmed — do not depend on it.
- The only real lead-time risk is **capacity, not paperwork**: RunPod 8x A100-80 availability is unverified and unguaranteed. Mitigate by not needing an 8x box (section 6.1) or by holding Lambda as the guaranteed fallback.

---

## 9. Provenance

**Verified live on 2026-09-18** (own pages/APIs, fetched during this research):
runpod.io/pricing · docs.runpod.io/pods/pricing · /pods/overview · /pods/manage-pods · /pods/storage/types · /storage/network-volumes · /storage/globalstore · /accounts-billing/billing · /get-started/billing-information · /pods/configuration/use-ssh · /api-reference/pods/POST/pods · /api-reference/pods/PATCH/pods/podId · /instant-clusters · /tips-and-tricks/tmux · console.vast.ai/api/v0/search/asks/ (live marketplace, all quantitative Vast data incl. DLPerf, reliability, driver versions, storage and egress rates) · vast.ai/pricing · vast.ai/article/Rental-Types · docs.vast.ai billing FAQ · docs.vast.ai jupyter-ssh FAQ · lambda.ai/service/gpu-cloud · verda.com/pricing (via datacrunch.io 301) · hyperstack.cloud/gpu-pricing · nebius.com/prices · crusoe.ai/cloud/pricing · digitalocean.com/pricing/gpu-droplets · tensordock.com/cloud-gpus.html · thundercompute.com/pricing

**Older, inferred or unverified — flagged in place:**
- RunPod spot **prices** are not published anywhere; the ~53% discount is extrapolated from one blog example (spot A6000 $0.232 vs on-demand $0.491, runpod.io/blog/spot-vs-on-demand-instances-runpod). Every "est. spot" figure inherits that uncertainty.
- RunPod spot **preemption warning** and **whether preemption wipes the volume disk**: undocumented. Third-party figures (5 s / 30 s / 5 min) conflict and are unattributable.
- All **slowdown factors** in section 3 are models (roofline + DLPerf), not measurements. See 3.4 for per-GPU confidence; the V100 figure is the weakest.
- **Achieved fp64 throughputs** are assumptions derated from vendor peaks, not benchmarks.
- Nebius's **17-21% price rise on 2026-10-01** is search-derived and unverified.
- Thunder Compute's **$1.09 A100-80** is from its live page but internally inconsistent with its 4x A100 bundle price.
- Aggregators (computeprices.com, cloud-gpus.com, gpus.io) produced no usable cross-check — two failed to render, and the third returned marketplace-floor figures not comparable to list prices. **No aggregator data is used in this memo.**

Raw data and scripts: `vast_live2.json`, `vast_query.py`, `vast2.py`, `vast3.py`, `vast4.py`, `dlperf.py`, `perf.py`, `sens.py`, `final.py` in this directory.
