# Campaign status — resume here

Living log for the robustness campaign (`CHANGES_NEEDED.md`). Newest entries at the bottom of each section.
A new session should read: this file, `campaign/CONTRACTS.md`, then `campaign/scout/*.md` as needed.

## Fixed points
* **Deadline:** 5 days from 2026-09-18 (target Wed 2026-09-23). No scope trims were accepted by the user; if the
  date is at risk, say so early and propose cuts.
* **Branches:** `robustness-campaign` in both `sv3` and `sven/` (off `rebuttals` @ cfd1220 and
  `stochastic_params` @ ca8742b). The orchestrator commits; development agents never do.
* **Subagents run on opus** (user request: preserve Fable usage).
* **User decisions 2026-09-18:** let the 9 in-flight GPT-2 jobs finish (output quarantined: val = first 200 train
  batches), stale `timing_serial_RERUNS` cancelled; CIFAR 5 headline seeds, kappa/param-fraction ablations stay
  1 seed, Fig-5 3 seeds; polynomial = variance-normalised monomials; external GPUs wanted — compare RunPod vs
  Vast.ai, use the cluster fully first, consider cheaper GPUs for suitable work; the orchestrator operates rented
  machines over SSH; development + GPU probe approved; **every scan launch needs the user's explicit go**.
* **Execution design:** long-lived single-GPU worker jobs pulling from a claim-file queue (lanes per workload
  class), gpu_test MIG lane for MLP work, scans run from an exported snapshot (never the live working tree),
  each scan on one GPU type. CIFAR capture mode (full vs chunked) is decided by the probe.
* **Reordered phases:** grid extensions run right after each headline scan reconciles, before ablation set
  points and phase 5, so phase 5 runs once.

## Scope decision 2026-09-18 (evening): CLUSTER ONLY, headline first
User rejected renting GPUs (too expensive) and asked to prioritise the headline scans and cut follow-ups by
reviewer relevance (rev1-3.md, REBUTTALS.md). Priority list below CONFIRMED by the user the same evening, with one change: NO AdamW/MuonW weight-decay grid (single torch-default wd everywhere; C-B4 dropped). In-flight GPT-2 jobs cancelled at the user's request (4 contaminated baseline records exist in the legacy dir).
* P0: six headline scans (+ added baselines) and `exp_nanogpt_speedrun` (R1 "no transformers", ~4 GPU-h).
* P1: `rebuttal_overparam_{toy_1d,polynomial,mnist}` (R1 crux, P>N), `rebuttal_fig5_cifar_paramfrac_scan` 3 seeds
  (R1 Q2; user: necessary), `rebuttal_batchsize_polynomial_scan` with the O5 LBFGS cut (R2 Q1).
* P2: standalone timing of headline best configs (R3 wall-time), one grid-extension round on headline.
* P3: `mnist_kappaScan` with lr retune (R1 kappa gap), MLP microbatch/paramfrac (toy + MNIST label-reg first),
  `exp_finetune_cifar_smallN` (R2 Q3), diagnostics + confirmation seeds for headline scans, polynomial data seeds.
* CUT: `exp_critbatch_{mnist,nanogpt}` (user), GPT-2-small, both CIFAR kappa scans, the two 1-seed CIFAR paramFrac
  scans (Fig-5 covers them), `mnist_scan_brier`, second extension round, C-X2, C-E4 off the critical path, C-L4 later.
Estimated ~520-600 GPU-h (A100 lane 280-360, MIG lane ~240) -> fits the cluster in ~2-3 days of compute.

## Timeline
| when | what | state |
|---|---|---|
| Fri 09-18 pm | scouting (8 agents), plan, user decisions | done |
| Fri 09-18 pm | Stage 0: 10 parallel dev tracks + GPU probe + cloud pricing research | running (workflow `sven-campaign-stage0`) |
| Sat 09-19 | Stage 1 integration in `generic_scan.py` (R1/R3/R4/S1-S3/L2/L3/E1 wiring, claims loop, results root), configs, launcher, reconcile | pending |
| Sat 09-19 night | Stage 2 full-diff review, Gate 1 GPU smoke matrix on gpu_test | pending |
| Sun 09-20 am | Phase 2 pilot (toy, polynomial) -> Gate 2 user review; MNIST/overparam/batch-size start right after | pending |
| Sun-Tue | CIFAR headline -> extensions -> set points -> ablations, nanoGPT, critbatch, GPT-2 | pending |
| Tue 09-22 | Phase 5: timing (cluster A100 only), diagnostics, confirmation seeds | pending |
| Wed 09-23 | notebooks, legacy-vs-fresh diff, docs | pending |

## Log
* 09-18 13:45 — branches created; scaffolding committed (558463c); Stage 0 workflow launched; cloud research
  agent launched. Probe results will land in
  `/n/holystore01/LABS/iaifi_lab/Users/sambt/sv3_campaign_scratch/probe_results/`.
