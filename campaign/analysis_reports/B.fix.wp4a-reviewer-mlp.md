## WP4a fix round — all 13 findings verified against the raw jsonl; all 13 real, all fixed

**Files changed** `analysis/reviewer_figs.py` (+9 helpers, 731→1073 lines: `paired_seed_diff`, `neighbour_gaps`, `monotonicity_table`, `reach_count_table`, `divergence_vs_reference`, `fixed_knob_cost`, `rank_vs_cap`, `eff_step_coverage`, `matched_step_paired`, plus `arm_ticks`/`figure_legend`); `tests/test_reviewer_figs.py` 25→**38 tests**; rewritten + re-executed `analysis/{overparam,batchsize,kappa,microbatch,paramfrac}_analysis.ipynb`. Builders `agent_lab/wp4a/build_{overparam,batchsize,kappa,knobs}.py` (gitignored). **Shared helpers and `make_plots.sh` untouched** (`git diff --stat` empty; no genuine bug found; all five notebooks already in `STUDIES`). Nothing committed.

| notebook | cells (code) | errors | runtime | figs (PDF+PNG) |
|---|---|---|---|---|
| overparam | 27 (15) | **0** | 40.0 s | 5 |
| batchsize | 22 (12) | **0** | 18.5 s | **6** (+`sven_cost_fixed_rtol`) |
| kappa | 15 (8) | **0** | 13.7 s | 4 |
| microbatch | 21 (11) | **0** | 34.9 s | 12 |
| paramfrac | 21 (11) | **0** | 33.8 s | 12 |

`pytest tests/` → **1209 passed, 33 skipped** (134 s, via `run_cpu_tests.sh`).

### What the corrected numbers now say

**Steps axis (F1, was "every method improves monotonically").** Computed: **1 of 13** methods is monotone in steps (K-FAC). Best at smallest *B* (25,000 steps): Sven, SOAP, Muon, PolyakSGD, RMSprop, SGD, Shampoo, K-FAC; best at **largest** *B* (780 steps): Adam, AdamW, MuonW; middle (*B*=64): L-BFGS, SGD+momentum. *(Reviewer's "not one of 13" and "Muon/SOAP best at largest B" were both slightly off; the notebook's claim was wrong either way.)* Sven 0.1231→0.1346→0.1197→0.1175→0.1063→0.1038.

**Sven step time (F4).** Confound confirmed. At **fixed rtol** the step rises with *B*: 1e-3 → 9.67/10.37/10.55/12.67/16.35/**17.41 ms** (1.80×, monotone); 1e-2 → 1.79×; 1e-4 → 1.84×; 1e-5 → 17.23→32.31 (1.88×). rtol 1e-5 costs **1.78–2.26×** the 1e-3 step at fixed *B*, and the *B*=8 arm is the only one that selects it. Peak memory is rtol-**independent** and monotone 18.67→22.82 MB.

**Divergence honesty (F2).** Per scan: polynomial Sven **14.00%** (252/1800) vs SGD 12.50%, SGDm 11.25% — Sven is **2nd worst of 13**; toy 15.42% vs SGD 15.62%/SGDm 20.00% (4th worst); MNIST 0.28% (12th worst, vs K-FAC 52.92%, L-BFGS 27.78%). Sven's grid is **11.25×** larger on the synthetic tasks. `divergence_vs_PoverN` also had Sven plotted twice (duplicate legend entry) — fixed, baselines now use `method_color`.

**Selection-metric vs outcome + ties (F5, F9).** Printed `_std` columns; `neighbour_gaps` shows **1 of 14** rank-1/rank-2 gaps clears 95% (toy *N*=150 vs Adam, *t*=−2.84). Every headline arm is a tie: MNIST *N*=2500 −PolyakSGD *t*=−0.13, −SGD −0.44; *N*=5000 −SGD −0.83; *N*=10000 −SGD −0.99; poly *N*=170 −SGDm −0.40. Losses are real: *N*=40000 −MuonW *t*=+4.39; poly *N*=675 −L-BFGS *t*=+11.08. Batchsize: *B*=16 significant (−4.34); *B*=32 nearest rival −1.59; *B*=64 −0.07. **Test ranks** added: MNIST **4,1,2,2,4,3**; pooled **2.12 vs 3.17** (val 1.75 vs 2.67).

**Reach counts (F3).** Computed per arm: poly 3/13 at *P/N*=4, 1/13 at 2, **0** below 1; MNIST 0/13 everywhere; toy 5/13 in all arms. §5 and §8 now read off the same cell.

**κ (F6, F8).** At *k*=32, **2 of 3** matched steps differ (0.25: κ1−κ2 *t*=+3.72; 1.0: *t*=−4.39); **eff step 0.5 is pure noise** (0/3 pairs, all *p*>0.47, per-seed std 0.0035–0.0038 > the 3.5% spread) — the "inconsistent direction ⇒ mechanism" inference dropped. Coverage table: κ1≤3.0, κ2≤1.5, **κ3≤1.0 with 0 runs above** → its ceiling is flagged `GRID EDGE`.

**Rank law (F7).** `rank_vs_cap`: MNIST-labelreg 7/7 exact; toy & poly 4/6 (11.06 & 27.25 rtol-limited, toy non-monotone 11.55 at μB=2); **MNIST-CE 2/7**. Toy loss **non-monotone**, 200× at μB=*B*, 225× peak at μB=16.

**Paramfrac (F13).** 39 recorded = **32** masked-Gram guard (toy 1/poly 12/CE 9/labelreg 10) **+ 7** non-finite training loss (poly 3, labelreg 4); 70 wide; all at η≥0.5.

**Figures (F10, F11).** Retitled `sven_rank_vs_PoverN` (+test-rank series); one short y-label per row, `arm_ticks` (fixed locator, `min_log_sep=0.12`), `figure_legend` below the axes — the overlapping row labels, colliding `3×10⁰4×10⁰` ticks and the on-data frameless legend (whose "unlabelled orange marker" was a SOAP point showing through) are gone.

**Open issues** no `_timing` pass on any of these 13 scans (all seconds are the co-tenancy-inflated scan clock); MNIST-overparam Sven still selects η=1 at *N*=2500/5000 (extendable edge; *k*=64=*B* is not); MNIST-labelreg paramfrac η=0.5 reference still blanks *f*≤0.25.