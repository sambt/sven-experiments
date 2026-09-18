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
* 09-18 ~16:30 — **GPU probe done** (jobs 47037513 A100-80, 47037629/47038158 MIG; report in the stage-0 workflow
  journal, scripts in `bench/probe_campaign/`, results in `.../sv3_campaign_scratch/probe_results/`, one-command
  table: `.venv/bin/python bench/probe_campaign/analyse.py`). Decisions: CIFAR Sven keeps `gram_capture: full` with
  **`empty_cache=False` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** = 186.7 ms/step (was 841; the per-step
  `torch.cuda.empty_cache()` was the whole slowdown and all of the variance), 23.3 GB reserved, NPROC=1 (2 co-tenants
  give T=0.98). `empty_cache=False` becomes the default for ALL runs. NPROC on A100-80: MLP Sven 12, cheap first-order
  = cores, LBFGS 4-6, nanoGPT Sven 3, CIFAR baselines 4+. NPROC on a MIG slice: MLP Sven 6-8, Adam 6, LBFGS 6-8,
  nanoGPT 1, CIFAR Sven 1 at chunked 0.5 (419 ms/step, 10 GB). MLP step time is set by host CPU load (1.13 ms on a
  quiet gpu_test node vs 4.58 ms on a full iaifi node) -> timing runs must control for node load; use wall time, never
  CUDA events, for co-tenancy decisions. CIFAR Sven run is now ~25-30 min -> headline CIFAR Sven ~80 GPU-h total.
* 09-18 ~16:40 — Stage 0: all 10 implementers finished; adversarial reviews in progress (real catches so far: 0-d
  device tensors in `svd_info` crash the runner on GPU; sparse `svs` vs per-step `_split_diagnostics` seam;
  `n_train > 50,000` silently clamps -> overparam MNIST top point must become N=50000). Nothing committed yet.
* 09-18 ~19:15 — Stage 0 committed for 8/10 tracks (sv3: grid a4ef395, optim d9f9a58, data 5c96eeb, loops 14c49b9,
  claims-prov 5efa1bd, ckpt-sampler 6361496; sven: nn f963f18, opt 203a4e6). analysis-core / analysis-offline still in
  review+fix (workflows w3tpbpmse, w4cwpfbcs). NOTE: a workflow runs only (CPUs-2) agents at once = 2 on this 4-CPU
  interactive job -> launch ONE WORKFLOW PER TRACK for parallelism.
* 09-18 ~19:20 — Stage 1 launched as three workflows: runner integrator (w8one9rnb: training wiring -> lifecycle ->
  2-lens review -> fix), configs (wmc0k1wky), launcher+reconcile (wnakx85ev). Contracts for them are in
  campaign/CONTRACTS.md "Stage 1 contracts". After they finish: commit, Stage 2 = full-diff review + Gate 1 GPU smoke
  matrix on gpu_test (needs a deploy snapshot), then ask the user for the go on the pilot (toy + polynomial).
* 09-18 ~20:55 — **Legacy results frozen** at the user's request: `sven_experiments` renamed to
  `/n/holystore01/LABS/iaifi_lab/Users/sambt/sven_experiments_legacy_2026-09-18` (repo symlink
  `experiment_results_legacy_2026-09-18`, write bits removed except `_cache/`), new EMPTY `sven_experiments`
  behind the unchanged `experiment_results` symlink. Legacy analysis: `SV3_RESULTS_ROOT=../experiment_results_legacy_2026-09-18`.
  All 10 Stage-0 tracks committed (analysis-core 4767461, analysis-offline 40675bd).
* 09-18 ~21:05 — **STANDING GO from the user:** run the Gate-1 smoke tests as soon as Stage 1 is committed; once they
  pass, launch the six headline scans together (toy_1d_scan + polynomial_scan double as the pilot; the user reviews
  their tables while MNIST/CIFAR run). This go covers the HEADLINE scans only - nanoGPT and P1-P3 still need a go.
* 09-18 ~21:15 — **STANDING GO widened by the user:** nanoGPT counts as headline; after a green smoke matrix queue
  EVERYTHING that does not depend on scan results, headline + nanoGPT first in every work list. Independent (queue
  immediately, in this order): six headline scans + exp_nanogpt_speedrun -> rebuttal_overparam_{toy_1d,polynomial,mnist},
  rebuttal_fig5_cifar_paramfrac_scan, rebuttal_batchsize_polynomial_scan -> mnist_kappaScan, toy/MNIST micro-batch and
  param-fraction scans, exp_finetune_cifar_smallN (these use set points fixed in their configs from the legacy
  headline results: if the fresh headline optimum moves materially, rerun them - cheap, hash dedup handles it).
  Result-dependent (WAIT): standalone timing of best configs, diagnostics (log checkpoints + dense spectra) and
  confirmation seeds of selected configs, polynomial data-seed replicates of selected configs.
* 09-18 ~21:30 — User: CIFAR fine-tune -> extension phase (out of the launch plan); MNIST kappa stays. Standing go
  reconfirmed: after Stage 1 commit + green smoke, queue all result-independent scans, headline + nanoGPT first.
