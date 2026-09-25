**Files changed** (owned by WP5; nothing committed, no result directory touched):
- `/n/home/anon/sven-experiments/EXPERIMENTS.md` — full rewrite, 730 lines, 12 sections
- `/n/home/anon/sven-experiments/README.md` — rewrite, 178 lines
- `/n/home/anon/sven-experiments/analysis/RERUNS_NEEDED.md` — appended the campaign launch log (+81 lines); items 0–7/10 marked historical, item 8 marked done
- No repo-root `CLAUDE.md` exists (`ls CLAUDE.md` → no such file); per the task I did **not** create one.

**Verified by running**
- `tools/reconcile.py --all campaign/plan_campaign.yaml --no-best` → `21 scan(s), 0 run(s) to do; incomplete: none`; **23,215 expected / 21,813 ok / 1,402 diverged / 0 oom / 0 error / 0 started-only / 0 stale-hash / 0 never-started**. Re-run at 18:42 EDT: `40 run(s) to do; incomplete: cifar10_resnet_ce_scan` (expected 740→**785**) — the GPU track launched `p2_cifar_ce_rtol` (commit `f0f89b2`) mid-task; documented as in-flight in all three files.
- `plan_phase5.yaml` → 21 companions, **1,575 runs** = 425 timing + 425 diag + 725 confirm, 0 to do. `plan_gpt2.yaml` → 29 expected, **28 ok, 1 claimed-live** (SOAP lr 3e-3).
- `tools/select_best.py --compare-analysis --print` → `85 selection(s), 0 unverified, 0 error(s)`, **`no disagreements`** with `scan_analysis.py`, **6 disagreements with reconcile**.
- `bench/check_timing_join.py` → **425/425 joined by run_id AND run_hash, 0 missing**; medians 0.00e+00 (nanoGPT), 8.7e-09 (poly), 4.5e-07 (toy), 5.7e-05 (MNIST-CE), 8.3e-03 (MNIST-LR), 2.5e-02/3.1e-02 (CIFAR).
- Provenance read from all records: **sv3 `2c6faf59`(15,735) / `62e5105e`(7,480) / `e5b6fb77`(28) / `b8fadc6f`(1,575), sven `203a4e61` throughout, `git_dirty` false everywhere**; job ids mapped to scans (21/10/10/15 jobs). No `_stale/` or `attempts/` dir under any of the 43 result dirs. `pytest tests/test_configs.py` → **364 passed, 10 skipped**.

**Factual corrections (old → new)**
1. "**28 experiments · 10 families · params 593→11.18M**" → 21 in-plan scans + GPT-2 + 21 phase-5 companions; exact counts per scan with `ok/attempted`.
2. "**10 seeds**" on toy/poly/MNIST/microbatch/paramfrac → **5** (`model_seeds` are 5 everywhere; Fig-5 3, GPT-2 1). CIFAR "5 seeds" was right.
3. CIFAR "**[chunked, batch-stat BN]**" → `gram_capture: **full**` + `bn_mode: batch`; `chunked` is used by **no** campaign scan (probe: full 186.5 ms beats cf0.5 192.0 with `empty_cache` off).
4. "runs **permanently absent**, no result file written" → every failure is a **record** with `status: diverged`; 0 oom/error/poisoned. Per-method divergence table replaces the "~6/~10/~20/~34 missing" estimates.
5. "**SOAP high-LR CUDA deadlock**, ~3 points absent" → did not recur; all 40 SOAP runs per MNIST scan have records.
6. "K-FAC ~20 missing on MNIST" → **40/40 diverged on both** MNIST scans (cusolver `eigh`), hence **no eligible K-FAC config** and 14 (not 15) methods in `best_configs.json` there.
7. "`cifar10_resnet_ce_kappaScan` missing κ=1 and 1.5" → that scan is **cut**; `mnist_kappaScan_labelRegression` is **210/210 with κ∈{1,2,3}**, grid `3κ × 7lr × 2k × 5seeds` (matched effective steps 0.25/0.5/1.0), not `κ∈{1,1.5,2,2.5,3}, k=64, lr=.5`.
8. Fig-5 "k=128, lr=.1, rtol=1e-4, 3 seeds" → **k=64, lr=1.0, rtol=1e-3** (what ran); flagged as the un-closed tentative set point.
9. `svd_spectra_every: 20` → `svd_spectra_schedule: {dense_first: 200, every: 20}` (diag: 1000/20). `sv_min` retired for schema 2.
10. `steps_per_epoch` — **MNIST 781, CIFAR 351** (grid_counts' 782/352 use `ceil`, sampler uses `drop_last=True`).
11. Cost: **added** the C-Z1/F28 statement (capture + one `M×M` `eigh`, independent of `k`; `O(kN|D|)` = classic randomized path only; `full` capture materialises `(B,P)`) to both EXPERIMENTS.md §1.5 and README.
12. Added: three splits with sizes, evaluation protocol (val-for-selection / test-as-outcome / example-weighted / eval-mode / `eval_batch_size` 2048·256·16 / `train_eval_size` 10000·1000·200), `bn_mode` policy, `derive_loader_seed` pairing, Muon grouping rule + `match_rms_adamw`, no-wd-sweep, dedup/claims, the polynomial redefinition (variance-normalised monomials, 210 terms; legacy = `AdditiveCubicDataset`, additive, 185 terms, **absolute losses not comparable**), the three passes, selection rule + Sven's picks and edges, cut/parked lists, GPU non-reproducibility, reproduce recipe.

**Paper-text locations I cannot edit** (listed in EXPERIMENTS.md §9): `iclr_manuscript/iclr2026_conference.tex` **90, 290, 387, 846** (`O(kN|D|)` / "a factor of k over SGD"), `WorkingNotes/main.tex:253` (`O(kdp)`), and **`iclr2026_conference.tex:692`** — "As with MNIST, Sven achieves a similar loss to the baseline optimizers" on CIFAR, contradicted by Sven 9th/11 and 10th/11.

**Deviations** — none from the plan. Two judgement calls: I did not create a `CLAUDE.md` (none exists), and I kept result *numbers* out of EXPERIMENTS.md (WP2 owns them), citing only what §7 of the plan settled.

**Notes for other WPs / orchestrator**
- **WP2/WP4b:** use the **status-only** vs **`is_diverged`** distinction — `scan_analysis` counts more (CIFAR-CE **11 vs 0**, RMSprop 8 of them; poly 260 vs 244; MNIST-LR 149 vs 129). Sven's post-extension picks moved: toy **2.873e-07** (k32/lr0.01/rtol1e-4), poly **0.10948** (k16/lr0.5/rtol3e-2) — `CAMPAIGN_STATUS`'s 09-19 table (4.8e-07 / 0.118) is pre-extension, don't quote it.
- **WP2/WP4a (timing):** one timing record is **not** its scan's trajectory — `mnist_scan_labelRegression` SOAP lr 0.01 mseed 3001, 0.901 (scan) vs **184.8** (timing). Take step times from `_timing`, loss values from scan/confirm.
- **WP1:** `bn_mode` is recorded `frozen` for svd rows and `batch` for standard rows on MLP/nanoGPT/GPT-2 (no running stats, so cosmetic) — any group-by `bn_mode` splits those scans in two.
- **Orchestrator:** `tests/test_configs.py:722` comment cites "EXPERIMENTS.md:108-109" for the Fig-5 set point; that line reference is now stale (the content lives in §3.3 and §9.1). Also: the Fig-5 scan ran on the tentative set point, so `test_fig5_setpoint_is_flagged_tentative_exactly_while_it_is_tentative` passes for the wrong reason — a re-point is 15 runs / ~5.5 GPU-h. `campaign/CIFAR_CE_RTOL_SNAPSHOT` (untracked) appeared from the GPU track.