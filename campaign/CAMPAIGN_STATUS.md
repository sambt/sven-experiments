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
* 09-18 18:35 EDT — **Stage 1 committed**: configs 0ac47a3, launcher 334e14a, runner integration 28abad5 (sv3 full
  suite 723 passed; sven 198 passed). Gate-1 smoke workflow launched (wn486mfkz: deploy snapshot -> smoke matrix via the
  real launcher on gpu_test + one A100 into `.../sv3_campaign_scratch/smoke_<sha>/` -> verification (a)-(j) -> fixes
  if needed -> independent audit with GO/NO-GO). The smoke agent may commit bug fixes on robustness-campaign.
  NEXT (standing go): on GO -> consolidate the MIG work into ONE ordered list (headline -> overparam/batch-size ->
  kappa/microbatch/paramfrac; gpu_test allows only 2 jobs), enable Fig-5 at k=64/lr=1/rtol=1e-3, commit, re-deploy,
  `tools/launch_campaign.py campaign/plan_campaign.yaml` dry run, then `--submit` P0 first, then P1, P3.
  Interactive job ends 22:19 EDT: MIG jobs last 12 h and need ONE resubmit tomorrow morning (same command).

## CAMPAIGN LAUNCHED — Fri 2026-09-18 20:05 EDT
* Gate 1: smoke matrix GREEN, independent audit **GO** (`campaign/stage1_reports/gate1.{smoke,audit}.md`). The smoke run
  caught a launch-killing bug (`scheduler=claims` rejected by Hydra struct mode -> every runner exited 1), fixed in f29298c.
* **Production snapshot:** `/n/holystore01/LABS/iaifi_lab/Users/sambt/sv3_deploy/2c6faf59_203a4e61` (sv3 2c6faf59,
  sven 203a4e61, git_dirty false). ALL campaign jobs must use this snapshot (`--snapshot <path>`); if code or configs
  change, the launcher refuses to mix snapshots - wait for the queue to drain or cancel first.
* Results root: `/n/holystore01/LABS/iaifi_lab/Users/sambt/sven_experiments` (repo symlink `experiment_results`).
  Pool logs: `/n/holystore01/LABS/iaifi_lab/Users/sambt/sv3_campaign_scratch/logs/campaign/`.
* Jobs (names carry list + snapshot sha): CIFAR Sven x6 47080825-32; CIFAR baselines x2 47080834-35; nanoGPT x2
  47080839-40; MIG combined list `all_mlp_mig` x2 47080843-44 (12 h, ends Sat ~08:00 EDT -> RESUBMIT the same command
  until reconcile is clean); MLP A100 overflow P0 x4 47080846-50, P1 x2 47080867/69, P3 x1 47080871; Fig-5 x2 47080858/60.
* Launch / resubmit command (dry run without `--submit`):
  `.venv/bin/python tools/launch_campaign.py campaign/plan_campaign.yaml --snapshot <SNAP> --list <name> [--n-jobs N] --submit`
  Progress: `.venv/bin/python tools/reconcile.py --all campaign/plan_campaign.yaml --no-best` (add `--phase P0`).
* First-hour watch list (auditor): pool log "results root" = sven_experiments; CIFAR Sven ~190 ms/step (>250 = allocator
  env lost); `[poisoned]` lines / non-empty `attempts/`; KFAC on MNIST dies deterministically in cusolver eigh -> recorded
  as `diverged` (known from legacy, accepted); Shampoo (~35 min/run) and HIG (~22 min/run) are the slow MLP items on MIG.
* WAITING on results (not launched): standalone timing of best configs, diagnostics + confirmation seeds, polynomial
  data-seed replicates (needs `data_seed` in `result_id_fields` first). Deferred: CIFAR fine-tune (extension phase).
* Log locations (all under `/n/holystore01/LABS/iaifi_lab/Users/sambt/sv3_campaign_scratch/`):
  job-level pool log `logs/campaign/<jobname>-<jobid>.out`; per-process runner logs
  `logs/<jobname>.<jobid>/item<NN>_<scan>_gpu<g>_p<k>.log` (+ `hydra/`); work-item files and launch records
  `work/campaign/<list>.items.txt|.launched.json`; smoke roots `smoke_*`; probe `probe_results/`.
  Dev-test CPU job logs: repo `slurm_logs/devtests/`.
* 09-18 20:20 EDT — user will quit and RESUME THIS CONVERSATION in a new 2-day CPU job; then monitor, resubmit the
  MIG list every 12 h and keep the A100 queue fed. For workflow parallelism ask for >= 8 cores (cap = CPUs-2 per workflow).
* 09-19 01:27 EDT check-in: toy_1d_scan 1080/1080 and polynomial_scan 1200/1200 COMPLETE (77 / 154 diverged records, 0 oom/error); MNIST CE 1210/1360, label-reg 1211/1360; CIFAR label-reg Sven 59; 5 A100 + 2 MIG jobs running, 14 pending on the node cap; no alarms.
* 09-19 03:55 EDT: MNIST HIG (~2.25 h/run at NPROC 6 on MIG) was blocking the MIG lane ahead of P1. Cancelled MIG jobs 47080843/44 (lost ~48 partial HIG runs; stale claims expire after 10 min) and resubmitted list `all_mlp_mig_v2` = P1 -> P3 -> MNIST HIG last (NPROC 4): jobs 47143141/45, end ~15:55 EDT -> resubmit `--list all_mlp_mig_v2` then. Headline status: toy/poly complete; MNIST scans complete except HIG (label-reg 26/150, CE 0/150); CIFAR label-reg Sven ~90/90 soon, CE started.
* 09-19 05:56 EDT check-in: P1 moving fast on MIG (overparam toy/poly 1540/2080 each, MNIST overparam 1450/2400, batch-size 1230/1680); CIFAR label-reg 98, CE 30; 9 A100 jobs running (overflow jobs now on MNIST HIG), 10 pending; 0 oom/error, no alarms.
* 09-19 07:57 EDT check-in: overparam toy/poly COMPLETE (2080 each), MNIST overparam 2104/2400, batch-size 1590/1680, nanoGPT 85/140, CIFAR label-reg 249/555, CE 227/655; 9 A100 running, 10 pending; 0 oom/error. The 48 claims in mnist_scan_labelRegression are stale leftovers of cancelled job 47080843 (taken over automatically when a worker reaches HIG).
* 09-19 09:57 EDT check-in: ALL MLP scans complete except MNIST HIG (label-reg 26/150, CE 0/150, now running on MIG at NPROC 4): toy, polynomial, both MNIST (minus HIG), 3 overparam scans, batch-size, kappa, 8 micro-batch/param-fraction scans, nanoGPT 140/140. CIFAR label-reg 320/555, CE 262/655. 0 oom/error anywhere.
* 09-19 11:58 EDT check-in: CIFAR label-reg 393/555, CE 293/655; MNIST label-reg HIG 60/150; 10 A100 + 2 MIG running, 7 pending; 0 oom/error.
* 09-19 13:58 EDT check-in: ALL 240 CIFAR Sven runs COMPLETE (6 jobs exited cleanly, 0 failed processes). Open: CIFAR baselines (label-reg 415/555, CE 365/655), Fig-5 2/15 (2 jobs running), MNIST HIG (label-reg ~106/150, CE 0/150; MIG + 4 A100 overflow jobs). MIG jobs end ~15:55 -> resubmit all_mlp_mig_v2.
* 09-19 16:00 EDT: MIG jobs hit the 12 h wall (TIMEOUT, expected); resubmitted all_mlp_mig_v2. mnist_scan_labelRegression COMPLETE (1360); open: MNIST-CE HIG 12/150, CIFAR label-reg 415/555, CE 437/655, Fig-5 8/15.
* 09-19 18:00 EDT check-in: Fig-5 COMPLETE (15/15). Open: MNIST-CE HIG 68/150, CIFAR label-reg 447/555, CE 495/655 (LBFGS + late baseline items). 6 A100 + 2 MIG running, none pending; 0 oom/error.
* 09-19 20:00 EDT check-in (24 h after launch): open = MNIST-CE HIG 126/150, CIFAR label-reg 532/555, CE 615/655. 4 A100 + 2 MIG running; 0 oom/error.

## ALL QUEUED SCANS COMPLETE — Sat 2026-09-19 22:11 EDT (26 h after launch)
`tools/reconcile.py --all` over 21 scans: **0 runs to do, 0 oom/error, 0 retries** (report: `campaign/reconcile_2026-09-19.txt`).
Headline (seed-mean final VAL loss, best config per method): toy HIG 4.7e-9 < Sven 4.8e-7 < SOAP 4.3e-6 < Adam 1.7e-5;
polynomial (new true degree-4 target) HIG 0.071 < Sven 0.118 < Polyak 0.136 < SOAP 0.150 < Adam 0.185; MNIST label-reg
MuonW 0.0500, HIG 0.0508, Sven 0.0522, SGD 0.0534, Adam 0.0568; MNIST-CE MuonW 0.104, Muon 0.106, HIG 0.114, SGD 0.122,
Sven 0.123, Adam 0.144; CIFAR label-reg MuonW 0.346, SOAP 0.352, Muon 0.365, AdamW 0.395, Adam 0.399, ..., Sven 0.480 (9th of 11);
CIFAR-CE MuonW 0.689, LBFGS 1.01, AdamW 1.09, ..., Sven 1.40 (10th of 11); nanoGPT AdamW 1.714, Sven 1.724, Muon/MuonW/SOAP 1.82.
Edge optima still open: toy Sven (lr 0.05 low edge, rtol 1e-4 low edge), toy HIG lr low edge, CIFAR LBFGS lr=2 high edge,
CIFAR-CE SGD lr=1 high edge, KFAC low edges, overparam/batch-size baseline lr=0.1 high edges (those scans kept the old
1e-4..1e-1 grid - deliberate deviation in configs.impl.md; extension is cheap, <2 GPU-h).
NEXT (needs the user): review tables; decide the extension round; then timing / diagnostics / confirmation passes and the
schema-2 analysis notebooks.
* 09-19 ~22:40 EDT — USER GO (standing, overnight): launch the grid-extension round, then the result-dependent passes (timing/diagnostics/confirmation), and GPT-2-small (after a one-run GPU smoke). Agents: ext-configs, phase5-tooling running; GPT-2 agent starts after ext-configs lands.
* 09-19 23:35 EDT: extension round launched from NEW snapshot /n/holystore01/LABS/iaifi_lab/Users/sambt/sv3_deploy/62e5105e_203a4e61 (sv3 62e5105e; +7,480 additive runs, finished runs dedup out, no _stale created): MIG all_mlp_mig 47322040/42, CIFAR baselines 47322047/52, MLP overflow P0 x3, P1 x3. gpt2-prep agent started.
* 09-20 00:00 EDT: phase-5 tooling committed (tools/select_best.py -> bench/best_configs.json; tools/gen_phase5_plan.py -> campaign/plan_phase5.yaml; bench/submit_timing_phase5.sh; *_diag/_confirm configs). Decisions: selection uses the FULL binding rule (eligible -> fewest diverged -> seed mean); CIFAR diag keeps checkpoints: epochs (~100 GB, fine); no float64 switch exists -> skipped. TODO when gpt2-prep lands: classify the 14 new configs in tests/test_configs.py. Launch passes only after the extension round reconciles clean: select_best.py --require-complete, gen_phase5_plan.py, deploy, launch.
* 09-20 01:55 EDT: GPT-2 smoke GREEN (Sven 2.52 s/step = 9.2 h/run, 35 GiB; AdamW 2.8 h, Muon 2.9 h, SOAP 4.1 h); Sven lrs extended to [0.02,0.05,0.1,0.5,1.0] (smoke: lr 0.5 unstable). GPT-2 launched (29 runs, ~122 GPU-h, 10 jobs) from snapshot /n/holystore01/LABS/iaifi_lab/Users/sambt/sv3_deploy/e5b6fb77_203a4e61 (extension round keeps running from /n/holystore01/LABS/iaifi_lab/Users/sambt/sv3_deploy/62e5105e_203a4e61; separate scans/lists).
* 09-20 04:05 EDT: extension round COMPLETE (reconcile clean, 23,2xx runs). Final selection written (bench/best_configs.json, full binding rule; CIFAR 'incomplete 100' = parked JD/HIG, polynomial LBFGS selected on 3/5 seeds). Phase-5 launched from snapshot /n/holystore01/LABS/iaifi_lab/Users/sambt/sv3_deploy/b8fadc6f_203a4e61: diag+confirm MIG 47337921/22, heavy A100 x4, MLP A100 x2; timing 7 per-scan serial jobs (non-exclusive, calibration microbenchmark at start/end for contamination check). GPT-2 running (10 jobs, snapshot /n/holystore01/LABS/iaifi_lab/Users/sambt/sv3_deploy/e5b6fb77_203a4e61).
* 09-20 07:02 EDT check-in: diag+confirm DONE for toy, polynomial, both MNIST; CIFAR diag/confirm ~50-55 of 55 each; timing done for toy/poly/nanoGPT, MNIST timing 6/70 (HIG/Shampoo/JD slow, serial), CIFAR timing ~21/55; nanoGPT diag/confirm pending in the heavy lists; GPT-2 5/29 done, 10 running. 18 jobs running, 0 oom/error.
* 09-20 10:05 EDT check-in: diag + confirm COMPLETE for all 7 headline scans; timing complete for toy/poly/nanoGPT/both CIFAR, MNIST timing 33/70 and 19/70 (serial heavy methods); GPT-2 10/29. bench/check_timing_join.py: nanoGPT bit-identical, medians ~1e-8 (toy/poly); deviations only for hardware-sensitive methods (Muon bf16 Newton-Schulz up to 25% on polynomial, LBFGS line search, SOAP/Shampoo/HIG decompositions at 1e-9 losses, CIFAR cuDNN 1-3% median) = scans ran on MIG A100-40 slices, timing on A100-80: cross-hardware nondeterminism, not a lifecycle bug; timing validity unaffected. Worth a sentence in the paper: Muon/LBFGS seed-level results are not bit-reproducible across GPU types.
* 09-20 13:07 EDT check-in: ALL timing/diag/confirm passes complete. GPT-2 20/29 done, 5 jobs running (remaining: mostly slow runs), 0 oom/error.
* 09-20 16:10 EDT: GPT-2 24/29 done (all 5 Sven runs + 19 baselines; 5 baseline runs in flight). Sven val by lr: 0.02->6.32, 0.05->6.04, 0.1->5.20 (interior optimum, test 5.11), 0.5->7.90, 1.0->10.2; baselines so far: Muon 3.77, MuonW 3.85, AdamW 3.93, SOAP 3.96 (1 seed, 1 epoch, 9.3 h per Sven run vs ~3 h).
* 09-20 ~16:40 EDT: analysis plan written: campaign/ANALYSIS_PLAN.md (5 work packages + orchestration + open decisions). User will resume on a compute node to orchestrate it. Verified beforehand: toy/cifar/overparam notebooks execute on the fresh results with 0 error cells.
* 09-20 ~17:10 EDT: user decisions recorded in campaign/ANALYSIS_PLAN.md section 7 (confirmation seeds as headline; GPT-2 and CIFAR reported as is; re-profile YES -> profile_results_v3; CIFAR-CE Sven rtol extension YES, restricted 45-run version by default). NOTHING launched - waiting for the user's go from a compute node.

## ANALYSIS PHASE — started Sun 2026-09-20 18:25 EDT (8-core compute node, user said execute campaign/ANALYSIS_PLAN.md)
* Contracts + file ownership: `campaign/ANALYSIS_CONTRACTS.md`. GPT-2 reconciled complete (29/29; one job still winding down).
* Phase A workflow `sven-analysis-phaseA` (wrmmewxm3), 5 tracks each implement -> adversarial review (numbers recomputed
  from raw records) -> fix: WP1 housekeeping, WP2 headline library (`analysis/headline.py`, `headline_tables.ipynb`),
  WP3 `analysis/ckpt_tools.py` + cached probe-set spectra, WP5 docs, GPU items (CIFAR-CE Sven rtol extension 45 runs +
  re-profile into `profile_results_v3/`; this agent may commit its own files and submit exactly those jobs).
* Phase B (after A, one owner per notebook set): WP2 headline notebooks, WP3 spectrum figures, WP4a reviewer MLP
  notebooks, WP4b CIFAR/nanoGPT/new GPT-2 notebook, WP5 legacy-vs-fresh diff. Phase C: make_plots.sh clean, tests green,
  10-number spot check; after the GPU items: re-select, phase-5 reruns for CIFAR-CE Sven if its pick changed, profile v3 notebooks.
* 09-20 ~20:15 EDT — **Analysis phase A committed (94ee715)**; reports in `campaign/analysis_reports/A.*`. Reviews recomputed
  numbers from raw records and caught real errors (accuracy sign in paired tables, pooled-vs-same-instance optimism, dataset
  cache collision across data seeds in ckpt_tools, stale doc facts). Corrections to the plan's numbers: CIFAR-CE Sven 53.0% vs
  56.8-77.9% (9th/11 on val, 11th/11 on test acc), 63 s/epoch vs 2.7-12.4 s; Sven diverges on 688/7,825 on-grid runs (low
  rtol); MNIST Sven is NOT bit-reproducible across MIG-40GB vs A100-80GB; timing calibration clean (max drift 3.6%);
  checkpoints total 243 GB. GPU items: CIFAR-CE rtol extension jobs 47394881-85 running from snapshot f0f89b24 (partial
  seeds already beat the old pick: rtol 0.3 -> 1.33 vs 1.40); re-profile job 47396284 pending (exclusive) -> profile_results_v3/.
  After the rtol jobs: `tools/select_best.py cifar10_resnet_ce_scan --out <side file>` and SPLICE the one key into
  bench/best_configs.json (running it plainly would overwrite the 7-scan file), then gen_phase5_plan + rerun CIFAR-CE Sven
  timing/diag/confirm if the pick changed, then re-execute cifar_analysis + headline_tables.
* 09-20 ~20:20 EDT — Phase B workflow `sven-analysis-phaseB` (wu7rezhmf): WP2 headline notebooks, WP3 spectrum figures,
  WP4a reviewer MLP notebooks, WP4b CIFAR/nanoGPT/new GPT-2 notebook, WP5 legacy-vs-fresh diff (implement -> review -> fix).
* 09-20 ~21:55 EDT — **Analysis phase B committed (4d133ea)**; reports `campaign/analysis_reports/B.*`. 23 notebooks
  (19 old + headline_tables, spectra_analysis, gpt2_analysis, legacy_vs_fresh) registered in make_plots.sh. Key results
  (confirmation seeds): Sven rank by val = toy 2/15, polynomial 2/15 (HIG first on both), MNIST label-reg 3/14, MNIST-CE
  4/14 (MuonW first), nanoGPT 2/5, CIFAR label-reg 9/11, CIFAR-CE 9/11 (11/11 on test acc), GPT-2 last (5.20 vs 3.77).
  Honest caveats found by the reviews: at an EQUAL small tuning budget (n=8) Sven ranks 15/15 on toy, 4/15 poly, 10/15 and
  8/15 on MNIST (its larger grid is partly redundant: 47-66% distinct trajectories); epochs-to-target Sven is 2nd-4th,
  not first; Sven needs fewer epochs than Adam on 3/3 regression scans but less WALL TIME on only 1/3; overparam: only 1 of
  14 rank-1/2 gaps is significant (mostly ties), Sven's divergence rate is 2nd worst on polynomial overparam; kappa differs
  at 2 of 3 matched effective steps (k=32); distance-from-init advantage NOT resolved; legacy->fresh Sven rank on
  same-target points: 7 better / 5 unchanged / 3 worse (CIFAR label-reg +3 worse). Fig-5 ran at the legacy set point
  (k=64, lr=1.0) not the fresh selection (k=128, lr=0.5) -> rerun launched in phase C.
* 09-20 ~22:00 EDT — Phase C workflow `sven-analysis-phaseC` (w0r80pzm0): cifar-followup (wait rtol ext 42/45 -> re-select
  CIFAR-CE by SPLICING -> rerun CIFAR-CE Sven timing/diag/confirm if changed -> Fig-5 rerun at selected config, 15 runs ->
  re-execute dependent notebooks), profile-v3 (job 47396284 running -> switch notebooks to v3, v2-vs-v3 table), then an
  independent final gate (tests, make_plots on a copy, >= 12 spot checks, cross-document consistency, honesty audit) + fixer.
* 09-20 ~22:40 EDT — USER (going to bed): let phase C finish; then update the ICLR manuscript (`iclr_manuscript/`, its own
  Overleaf-synced git repo - never commit/push there) in a NEW tex file copied from `iclr2026_conference.tex`, all edits in
  BLUE: new results + baselines (consider polynomial/MNIST/nanoGPT as headline), reworked SV-spectrum discussion, Gram-trick
  main-text mention + full appendix, all reviewer-driven studies (mostly appendix), new profiling, anything else needed;
  concise, in the existing style. Rules: `campaign/PAPER_CONTRACTS.md` (numbers only via generated macros/tables from
  `analysis/paper_assets/`, blue markup, claims discipline). Workflow `sven-paper-update` (wbuzu9mp7) launched in parallel
  with phase C: plan (`campaign/PAPER_PLAN.md`) -> 4 asset modules + Gram appendix + experiment-details drafts ->
  integrate into `iclr_manuscript/iclr2026_conference_v2.tex` -> compile -> 3-lens review -> fix. AFTER BOTH workflows
  finish: re-run `cd analysis && ../.venv/bin/python -m paper_assets`, recompile, check the refreshed CIFAR-CE / Fig-5 /
  profile-v3 numbers flow into the PDF, final read-through.
* 09-21 07:35 EDT — The compute node FAILED overnight (interactive job 47390824 NODE_FAIL after 5 h 18 min); both workflows
  died mid-flight. All GPU work had finished first: CIFAR-CE rtol extension, re-selection (6e7fc72: Sven -> k=128, lr=0.5,
  rtol=0.3), CIFAR-CE Sven timing/diag/confirm reruns, Fig-5 rerun at the selected config (1b7b61d, 15 runs), profile v3.
  Session resumed on a LOGIN node (heavy work only via campaign/run_cpu_tests.sh). Relaunched: `sven-analysis-phaseC-resume`
  (w1nq68ygk: notebook refresh + profile v3 switch -> final gate -> fixer) and the paper workflow resumed from its cache
  (wc9dsdk4f, script now at campaign/workflows/sven-paper-update.js; plan + Gram appendix + experiment-details drafts were
  cached, the 4 asset agents rerun, then integrate -> review -> fix). Blueprint: campaign/PAPER_PLAN.md.
* 09-21 ~09:15 EDT — **ANALYSIS COMPLETE (phase C + final gate).** Refresh 040f271, gate fixes ad863fe / 4118687; reports
  `campaign/analysis_reports/C.*`. Gate: tests 1225 passed; 23/23 notebooks 0 error cells on an isolated copy, no
  PROVISIONAL banner, committed tables reproduce byte-identically; 21 numbers recomputed from raw records (17 ok, 4 wrong ->
  fixed); all 8 findings real and fixed. Final CIFAR-CE Sven (new pick k=128, lr=0.5, rtol=0.3; confirmation seeds):
  val 1.355 (9/11), test 1.339 (8/11), test acc 58.2% (10/11), train-eval 0.159 (was 0.82: the optimisation failure was the
  rtol edge), 178 ms/step. Fig-5 at the selected config (k=128, lr=0.5): 0 divergences; test acc 69.1/68.8/67.8/25.6/19.2%
  at param fraction 1/0.5/0.25/0.1/0.05; masking makes the step SLOWER (x2.4 at pf 0.5) and saves memory only at small pf.
  Profile v2->v3: CIFAR full-J Sven 842 -> 188 ms, chunked 470 -> 208, nanoGPT hooks 83 -> 60; memory ~unchanged.
  Reproducibility: same-GPU reruns are bit-identical for MLP scans + nanoGPT, NOT for CIFAR; cross-GPU (MIG-40 vs A100-80)
  Sven deviates 6e-9 (poly) to 1.4e-2 median (MNIST-CE). Settled record total 24,894. Open by design: rev2 Q2b (Sven on
  hidden layers only) was never run - state as a non-answer; campaign-wide checkpoint verification only sampled (diag
  passes); remaining grid edges: rtol high edge on MNIST-CE and CIFAR-CE (accepted).
  Paper workflow (wc9dsdk4f) still running; it must re-run `python -m paper_assets` after these commits.
* 09-21 ~12:05 EDT — **MANUSCRIPT v2 DRAFT COMPLETE.** `iclr_manuscript/iclr2026_conference_v2.tex` (copy of the original,
  edits in blue via \new{} / newtext; 18 appendix fragments in `iclr_manuscript/sections_v2/`; figures
  `figures_iclr/` (45 PDFs), tables `tables_v2/` (55), number macros `numbers_v2*.tex` (1,352 macros), new bib
  `references_v2.bib`; change log + author decisions in `iclr_manuscript/CHANGES_v2.md`). Proof PDF:
  `iclr_manuscript/iclr2026_conference_v2_PROOF.pdf` (96 pages; main text ends on page 9; 0 LaTeX errors, 0 undefined
  refs/cites, 0 overfull > 10pt). Nothing committed in the Overleaf repo. Regenerate everything with
  `cd analysis && ../.venv/bin/python -m paper_assets` then `bash agent_lab/paper_build/build.sh` (local TeX Live 2018
  lacks algpseudocode/bbm/nicefrac -> out-of-repo shim in agent_lab/paper_build/shim; Overleaf is authoritative for the
  page count). Workflow reports: `campaign/paper_reports/` (3 adversarial reviews: 28 high findings, all dispositioned by
  the fixer). Asset package committed in sv3 (1c09159). Assets re-generated after the final analysis gate: only
  timestamps changed.
