## WP4a — reviewer MLP studies (overparam / batch size / κ / micro-batch / param-fraction)

**Files created:** `/n/home/anon/sven-experiments/analysis/reviewer_figs.py` (new, ~580 lines: per-arm binding-rule selection `arm_table`, `rank_table`, `divergence_table`, `edge_optima`, `scan_census`, `reach_epoch/reach_table`, `median_method_target`, `fastest_to_target`, `knob_table`, `kappa_table`, `matched_step_table`, `plot_arm/plot_knob`, `savefig` PDF+PNG); `/n/home/anon/sven-experiments/tests/test_reviewer_figs.py` (25 tests, synthetic tmp-root scans only).
**Files rewritten:** `analysis/{overparam,batchsize,kappa,microbatch,paramfrac}_analysis.ipynb`. Builders (gitignored, re-runnable): `agent_lab/wp4a/build_*.py`, `nbbuild.py`, `orig/`.
**Shared helpers: untouched** — no genuine bug found. `make_plots.sh`: **no change needed**, all five are already in `STUDIES`.

| notebook | cells (code) | errors | runtime | figures (PDF+PNG) |
|---|---|---|---|---|
| overparam | 25 (14) | **0** | 37.5 s | 5 |
| batchsize | 19 (10) | **0** | 16.5 s | 5 |
| kappa | 15 (8) | **0** | 12.9 s | 4 |
| microbatch | 21 (11) | **0** | 35.9 s | 12 |
| paramfrac | 21 (11) | **0** | 33.1 s | 12 |

`pytest tests/` → **1131 passed, 33 skipped (149 s)**. Raw-JSONL spot-checks reproduce: MNIST-overparam N=2500 Sven **0.124305**, batchsize B=64 Sven **0.119666**, κ k=64 matched step 0.25 → **0.05417448 / 0.05417450 / 0.05417452**.

### Key numbers

**Overparam (R1 crux).** Sven's validation rank per arm: MNIST **1,1,1,2,2,2** at P/N 11.0→0.55; polynomial **1,2,2,4** at 4.0→0.5; toy **2,4,3,3** — *no* P/N dependence there. Mean rank over/under P/N=1: 1.25/2.00 (MNIST), 1.50/3.00 (poly), **1.75 vs 2.67 pooled** (8 vs 6 arms). Legacy "consistently 2nd-fastest for P/N>1" **does not reproduce**: under legacy-style selection Sven is 1st on *epochs* in all four toy arms (23.6–27.4 vs L-BFGS 28.4–43, Adam 57–64) but **4/3/4/5 by seconds**, flat across the boundary; on the corrected polynomial only 3 of 13 methods reach 1e-3 at P/N=4, nobody below P/N=1. Optimisation ≠ fitting: MNIST N=10,000 Sven is 1st on val, **10th on `train_eval`**; no method reaches train 1e-3 on MNIST even at P/N=11. Sven diverges (wide rule) 111/720 toy, 252/1800 poly, 1/360 MNIST — **0 recorded `status: diverged`**. 14 edge optima, 11 Sven; `rank_eff ≈ 12` explains the toy low-k edge.

**Batch size (R2 Q1).** Sven ranks **3,1,1,1,5,4** at B=8…256; its own loss varies only 1.30× (Adam 1.45×, MuonW 2.02×). 20 epochs = 25,000 steps at B=8 vs 780 at B=256 (constant examples). No k/B<1 arm exists (k=B pinned) — answered via `rtol`-induced effective rank: **8.0/10.6/17.2/27.3/82.3/126** = k_eff/B **1.00/0.66/0.54/0.43/0.64/0.49**. Sven 10.4–18.0 ms/step vs Adam 2.7–4.0, L-BFGS 21–23; peak mem 18.7→22.8 MB vs flat 18.4. L-BFGS (O5 shape mi=3/hs=2) has **no eligible config below B=64** (23/25 diverge at B=8); Sven's divergence *rises* with B (2/80→13/80).

**κ (R1).** At **k=B=64** matched effective step 2η/κ, relative spread over κ is **7.8e-7** (step 0.25) and **1.4e-3** (0.5) — κ is exactly an lr rescaling. At **k=32** it is **6.8% / 3.5% / 11.0%** — truncation breaks the identity. κ=1 fails only untruncated (8 of 10 divergences at k=B); at k=32 κ=1 is the *most* step-tolerant. Mechanism: at k=B, κ=3 keeps only 60–63 of 64 directions.

**Micro-batch.** μB=1→B costs 8% (MNIST-CE), 2% (MNIST-labelreg), **2.1×** (poly), **225×** (toy). Rank falls exactly B/μB. **Step time ≤16% better; peak memory unchanged (31.363 MB at every μB on MNIST).** 3 diverged of 520.

**Param fraction.** MNIST-CE f=1→0.1: 0.1323→0.1471 (acc 96.19→95.63%); poly flat to f=0.25 then 1.342; toy 2.0e-6→3.5e-2. **Masked step ~2× slower** (toy 19.4 vs 9.5 ms) and memory *rises* (MNIST 31.4→48.0 MB at f=0.75, 1.53×). 39/400 recorded raises, all η≥0.5.

### Deviations / open issues
1. No PDF+PNG helper existed (`Scan.save` is PDF-only, frozen) → added `rf.savefig` and rewired the 8 inherited knob figures through it.
2. Kept §1–§4 of microbatch/paramfrac verbatim; appended §5–§9.
3. No `freshness_report` banner: none of these 13 scans is in `best_configs.json` or flagged (only CIFAR-CE is).
4. These scans have **no `_timing` pass** — every second is the co-tenancy-inflated scan clock, labelled as such; conclusions ride on epochs/steps/examples.
5. MNIST-labelreg paramfrac's data-chosen reference lr (0.5) wipes out f≤0.25 (0/5) → blank cells; an η=0.1 companion panel would show that trend.
6. Open grid edge worth closing: MNIST-overparam Sven selects k=64 **and** η=1.0 (both grid tops) at N=2500/5000.
7. Deleted two superseded figures (`overparam/best_valloss_vs_PoverN.pdf`, `kappa/vs_kappa.pdf`).