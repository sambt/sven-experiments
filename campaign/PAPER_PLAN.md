# PAPER_PLAN — blueprint for the ICLR revision (written 2026-09-20)

Binding companion to `campaign/PAPER_CONTRACTS.md` (read that first: blue markup, numbers-as-macros,
claims discipline, LaTeX build, never commit in `iclr_manuscript/`). This document decides
**what the revised paper claims, where every sentence goes, and which asset backs it**.
Every other paper agent follows it; where it is silent, `PAPER_CONTRACTS.md` governs.

Sources of record for every number below: `bench/best_configs.json` (selection),
`analysis/tables/*.md` (exported today), `campaign/analysis_reports/{A,B}.fix.*.md` (corrected
numbers + caveats), `analysis/WHAT_CHANGED.md`, `EXPERIMENTS.md`, `rev{1,2,3}.md`,
`REBUTTALS.md` §5–6. **Numbers printed in this plan are the values on 2026-09-20 22:30 EDT and
are for orientation only** — no agent types one into the tex. They exist so a reviewer can tell
whether a macro came out wrong.

**Three inputs are still moving** (parallel workflow): CIFAR-CE Sven re-selection (already moved
to `k=128, lr=0.5, rtol=0.3` at 22:06 — the confirmation/timing/diag passes for it may not have
re-run yet), the Fig-5 re-run at the selected configuration, and `profile_results_v3/`
(578/720 configurations, **no `cifar_resnet18` directory yet**). Everything that touches them is
marked **[MOVING]** and must be produced by `analysis/paper_assets/`, never typed.

---

## 1. Headline framing — DECIDED

**Main results = random polynomial regression + MNIST (label regression) + nanoGPT
(tiny-shakespeare).** 1D regression moves to Appendix E (all-seed results) and keeps its full
table and curves there.

*Justification (the two sentences that go in the response letter, not the paper):* 1D regression
is a 593-parameter fit whose seed variance swamps its mean — Sven's confirmation value is
`5.09e-07 ± 1.49e-06` and its tuning→confirmation same-instance gap is −63 % — so it cannot carry
a headline claim, while polynomial / MNIST / nanoGPT span synthetic regression, a real
classification dataset and a transformer language model, which is exactly the coverage R1 ("no
transformers, no sequence tasks") and R2 ("more widely used models") said was missing. On those
three Sven's standing is both stable and defensible across seeds and splits (2nd of 15, 3rd of
14, 2nd of 5, all on confirmation seeds), each against the full 15-method field that now includes
every baseline the reviewers asked for.

**The story the paper tells** (this is the spine; every section serves it):
Sven is a principled over-parametrised natural-gradient step that (i) beats every first-order and
every Kronecker-factored baseline on final loss on all four MLP scans, (ii) reaches a fixed target
in fewer steps than Adam but not in less wall time, (iii) ties the best baseline on a transformer
LM at small scale, (iv) is not first anywhere — HIG leads the synthetics and MuonW the
classification scans, and (v) does not yet work at ResNet/GPT-2 scale, where its cost is dominated
by the Jacobian capture, not by the eigendecomposition. The Gram backend is the reason (i)–(iii)
are measurable at all, and the reason (v) is a capture problem rather than an algebra problem.

---

## 2. CLAIMS MAP

### 2.1 Claims the revised paper makes

Legend — **asset**: id from §5. **fn**: the analysis function of record. Caveats are **binding**:
the claim may not appear without them.

| # | where | claim | asset · fn | number today | caveat it must carry |
|---|---|---|---|---|---|
| C1 | abstract, §2 | Sven takes the Moore–Penrose pseudoinverse of the **per-sample scalar-residual** Jacobian `M ∈ R^{B×P}`, one row per sample, independent of output dimension | — (theory) | — | none |
| C2 | abstract, §2.3, App. D | Under the **Gram backend** a step costs the capture plus one `B×B` eigendecomposition, **independent of k**; `O(kN|D|)` describes only the classic randomized path, which no experiment uses | T7, F3 · `profile_helpers.method_table`, `scaling_table` | flat step time vs `k` in the timing pass; MNIST 4.03 ms/step vs Adam 1.13 | must name the capture mode (`hooks` vs `full`); `k`/`rtol` decide how many eigenpairs are *inverted*, not the cost |
| C3 | abstract, §4, App. D, Q | With the **hooks** capture, peak memory is at parity with Adam on MLPs and 1.9× on nanoGPT; the dense **full** capture that batch-statistics normalisation forces costs 47× | T1, F3, T7 · `headline.efficiency_table`, `profile_helpers.method_table` | MNIST 31.4 MB vs Adam 31.7 (0.99×); nanoGPT 615 vs 330 MB; CIFAR 22,960 vs 492 MB | the parity is **hooks-capture only**; ResNet18 needs `full` because batch-statistics BN cannot go through hooks |
| C4 | abstract, §4.1, T1 | Sven beats Adam, AdamW, SGD, SGD+momentum and RMSprop on final validation **and** test loss on all four MLP scans, with paired per-seed differences that clear a 95 % t-interval | T1, T5, F1 · `headline.paired_vs_sven`, `confirmation_table` | polynomial: Sven −0.099 vs Adam, CI [−0.135, −0.063]; 13 of 14 baselines worse, all significant | confirmation seeds; `finished/attempted` in the table; polynomial pools 3 data seeds |
| C5 | abstract, §4.1 | Sven is 2nd of 15 (polynomial), 3rd of 14 (MNIST-LR), 4th of 14 (MNIST-CE), 2nd of 15 (1D), 2nd of 5 (nanoGPT) on confirmation-seed validation loss | T1, T4 · `headline.ranking_summary`, `rank_matrix` | 0.1388 / 0.05328 / 0.1134 / 5.09e-07 / 1.715 | **Sven is never first**: HIG 1st on both synthetics, MuonW 1st on both MNIST scans and both CIFAR scans, AdamW 1st on nanoGPT |
| C6 | §4.1, T1 | On nanoGPT Sven and AdamW are **tied**: paired difference +0.0022, CI [−0.023, +0.027] | T1, F1 · `headline.paired_vs_sven`, `large_figs.curve_band` | Sven 1.715 ± 0.008 (ppl 5.56), AdamW 1.713 ± 0.018 | a tie, not a win; Sven costs 2.6× the wall time (120.2 s vs 45.7 s) |
| C7 | §4.1, App. E | Sven reaches a pre-declared target in **fewer epochs** than Adam on 3 of 3 MLP regression scans, and in **less wall time on 1 of 3**; on nanoGPT AdamW is ahead on both axes | T2, F1 · `headline.time_to_target_table`, `epochs_to_target` | median-method target: toy 4.7 vs 11.9 ep / 6.1 vs 6.7 s; poly 7.3 vs 10.2 ep / 9.7 vs 5.9 s; MNIST-LR 9.2 vs 12.7 ep / 34.4 vs 18.7 s; nanoGPT 21.6 vs 18.4 ep / 51.9 vs 16.8 s | must report **runs that never reach** the target (poly: Adam 5 of 15, Sven 1 of 15; MNIST-LR Adam 2 of 5, Sven 0) and say wall time is from the standalone timing pass |
| C8 | §4.1, App. Q | Sven's step is 2.9–3.6× Adam's on MLP/transformer scans and 27× on ResNet18 | T1, T7, F3 · `headline.efficiency_table` | 3.40 / 3.44 / 4.03 / 20.9 / 176.9 ms per step (toy/poly/MNIST/nanoGPT/CIFAR) vs Adam 1.05 / 1.06 / 1.13 / 7.22 / 6.50 | timing pass only (`<scan>_timing`), never a scan clock; 9–10 baseline configurations are cheaper per step and 3–5 dearer |
| C9 | §4.2 (spectra) | The rank actually used is `min(k, rtol-rank)` on **every logged step of all seven scans**; `rtol` is the binding cut on 5 of 7 | F2, T8 · `spectra_figs.mechanism_table`, `used_rank`, `rtol_rank` | exact identity, 7/7 scans | this retires the paper's "best k is a large fraction of B ⇒ many significant directions" inference: the polynomial pick is `k=16` of `B=32` **with** `rtol=0.03` |
| C10 | §4.2, App. O | The kept directions carry almost all of the batch residual, and the fraction **rises** over training on every scan: polynomial 92.3→99.6 %, MNIST-CE 31.0→95.2 % while the used rank falls 10.0→2.8 | F2, T8 · `spectra_figs.energy_in_top`, `cumulative_energy`, `plot_energy_capture` | toy 92.1→91.5, poly 92.3→99.6, MNIST-LR 100→100, MNIST-CE 31.0→95.2, CIFAR-LR 100→99.999, CIFAR-CE 100→99.999, nanoGPT 100→100 | these are **online float32 per-batch Gram** spectra; the probe-set numbers (C11) are the float64 cross-check and they disagree on polynomial |
| C11 | §4.2, App. O | On a fixed float64 probe set the top-`k` directions capture 100.000 % (toy), **58.82 %** (polynomial, `k=16`), 99.868 % (MNIST-LR), 96.09 % (MNIST-CE) of the probe residual | F9, T8 · `spectra_figs.probe_energy`, `probe_metrics`, `ckpt_tools.load_spectra` | polynomial `in_span` 99.49 % at the same checkpoint | polynomial is the one task where truncation at the *selected* `k` discards real signal (41 %) and still wins — state it, do not hide it; probe widths are 593/673/**512**/512 rows, not the parameter count |
| C12 | §4.2, App. O | Spectra along Sven's trajectory and along Adam's/HIG's are identical at step 0 (shared initialisation) and separate over training; Sven's distance from initialisation is **not** resolvably smaller | F9, T9 · `spectra_figs.low4_table`, `low4_verdict` | `dist_init`: Sven smallest on 2 of 4 scans, resolved on **0**; `param_norm` smallest on 1 of 4, resolved on 1 (polynomial −0.590 ± 0.33) | the min-norm story is **not** confirmed empirically; say so |
| C13 | §4.2, App. J | On CIFAR-10 `k` binds rather than `rtol`: mean used rank 112.5/128 (label-reg) and 123.3/128 (CE) | F7, T8 · `spectra_figs.used_rank_grid`, `large_figs.sven_rank_used` | 87.9 % / 96.3 % of `k` | [MOVING] CE re-selection |
| C14 | §4, App. C | Protocol: three fixed splits per dataset, selection on **validation only**, test as an outcome, 5 tuning + 5 confirmation seeds (+3 data seeds on the synthetics), identical initialisation and data order across optimizers at fixed model seed | T3, T10 · `headline.assert_no_test_selection`, `confirmation_table` | 85 of 85 selections reproduce from the binding rule | every table prints `finished/attempted` under `style.is_diverged` (wider than the recorded status) |
| C15 | §4, App. F | Selection optimism is visible, not absorbed: median tuning→confirmation gap is −1.8 % (poly, same instance) to +1.7 % (MNIST-LR); Sven's is +0.6 % / +2.2 % | T6 · `headline.selection_optimism_table` | Sven same-instance: toy −63.4 %, poly +0.64 %, MNIST-LR +2.16 %, MNIST-CE −1.36 % | the pooled column is **instance variation**, not optimism — label both |
| C16 | §5 (limitations), App. F | Sven's tuning grid is larger than the baselines' and partly redundant; at an equal budget of n=8 trials it is **15/15 on 1D**, 4/15 on polynomial, 10/15 on MNIST-LR, 8/15 on MNIST-CE | T6, F4 · `headline_figs.equal_budget_table`, `budget.best_of_n_curve`, `budget.trajectory_table` | distinct-trajectory fraction 0.475–0.664 for Sven vs 1.0 for the single-knob baselines | this is a point **against** Sven and is reported as such; on the three losing scans it never matches the leader at any budget its grid allows |
| C17 | §5, App. P | Sven diverges when `rtol` is at the bottom of the grid and the risk grows with lr: 688 of ~7,870 on-grid Sven runs (8.7 %) over 16 of 21 grids, and **zero** on MNIST-CE, CIFAR-CE, nanoGPT and two micro-batch scans | F10, T11 · `headline_figs.campaign_divergence`, `divergence_pattern`, `sven_divergence_grid` | toy 194/900 (all at `rtol ≤ 1e-4`, none at 1e-3 or 1e-2); poly 4/720; MNIST-LR 1/640 | **[MOVING]** count grows as CIFAR-CE lands; per-scan rate, not a single headline number; Sven is 2nd-worst of 13 on the polynomial P/N sweep (14.0 % vs SGD 12.5 %) on an 11.25× larger grid |
| C18 | App. G | Across the P/N boundary Sven is 1st or 2nd at 12 of 19 swept points and first on MNIST at the three most over-parametrised ones, but **only 1 of 14 rank-1/rank-2 gaps clears 95 %** | F5, T12 · `reviewer_figs.rank_table`, `neighbour_gaps`, `paired_seed_diff` | MNIST test ranks 4,1,2,2,4,3; pooled mean rank Sven 2.12 vs best baseline 3.17 (test), 1.75 vs 2.67 (val); the one resolved gap is toy N=150 vs Adam (t = −2.84) | most headline arms are **ties**; the loss at `N=40000` is a real loss to MuonW (t = +4.39); R1's "positive result in this regime" is therefore answered *directionally*, not decisively |
| C19 | App. G | Time-to-target in the P>N regime: reach counts are low and must be printed — polynomial 3 of 13 methods reach at P/N=4, 1 of 13 at 2, **0 below 1**; MNIST 0 of 13 everywhere; toy 5 of 13 in all arms | F5, T12 · `reviewer_figs.reach_count_table`, `reach_table` | as printed | the P/N scans have **no timing pass**: every second on them is a co-tenancy-inflated scan clock, and the figure must say so |
| C20 | App. H | Sven's best loss is achieved at the **smallest** batch size and degrades monotonically with B; 1 of 13 methods (K-FAC) is monotone in steps; only B=16 clears significance against the nearest rival | F6, T13 · `reviewer_figs.arm_table`, `monotonicity_table`, `neighbour_gaps` | Sven 0.1231→0.1346→0.1197→0.1175→0.1063→0.1038 across B=8…256; B=16 t=−4.34, B=32 −1.59, B=64 −0.07 | at fixed `rtol` Sven's step time rises 1.80× from B=8 to B=256, and peak memory 18.67→22.82 MB; the apparent "flat cost" is an artefact of the selected `rtol` changing with B |
| C21 | App. M | With signed residuals **κ=1 no longer fails**: 210 of 210 runs complete at κ ∈ {1,2,3}; at matched effective step `2·lr/κ` and `k=32` the κ choice changes the outcome at 2 of 3 matched steps | F8, T14 · `reviewer_figs.matched_step_paired`, `kappa_table`, `eff_step_coverage` | eff step 0.25: κ1−κ2 t=+3.72; 1.0: t=−4.39; **0.5 is pure noise** (all p>0.47) | 10 of 210 blow up under the wide rule, 9 of them outside the matched steps; κ=3's grid stops at effective step 1.0 (`GRID EDGE`) |
| C22 | App. K | Parameter masking does **not** buy time: masked Sven is 1.9–2.2× slower per step than unmasked at 0.51× memory, because the full capture does not get cheaper with a mask | F7, T15 · `large_figs.plot_paramfrac_cost`, `paramfrac_table` | 178.3 ms/step unmasked vs 342–389 masked | **[MOVING]** re-run at the selected configuration; the step-time overlay only switches on when `profile_helpers.results_root()` returns v3 |
| C23 | App. K | Quality holds down to a parameter fraction of ~0.5 and collapses below it | F7, T15 · `large_figs.paramfrac_view`, `plot_paramfrac_quality` | **[MOVING]** at the old set point (k=64, lr=1.0): f=1 val 0.668 / 74.1 % acc, f=0.5 0.767 / 72.9 %, f≤0.25 collapses to 10.0–17.8 % (chance = 10 %) on 2/3, 2/3, 3/3 seeds, 2 diverged | the old set point was **not** the selected configuration and both blow-ups sit at its single lr=1.0; after the re-run, state the threshold as a macro and keep the "one fixed configuration, not re-tuned per f" note |
| C24 | App. J | On CIFAR-10/ResNet18 Sven is **9th of 11** on both losses; label regression is a generalisation failure (train-eval 0.067, the 3rd-lowest of 11, test accuracy 69.4 % vs 76.3–77.9 %), cross-entropy an optimisation failure (train-eval 0.819, the highest of 11) | F7, T16 · `large_figs.optimisation_view`, `generalisation_gaps`, `headline.confirmation_table` | LR: val 0.4838 ± 0.0105, acc 69.42 %; CE **[MOVING]** was val 1.424 ± 0.013, acc 53.0 %, last of 11 on test accuracy | 62.9 / 63.0 s per epoch = 23× SGD; BatchNorm was `bn_mode: batch` for **every** optimizer; say which seed set |
| C25 | App. L | GPT-2-small is a **negative** scaling data point: Sven's validation loss is 39 % above the best baseline at 3.2× the time per run | F11, T17 · `large_figs.gpt2_summary_view`, `gpt2_lr_table` | Sven val 5.198 (ppl 181) / test 5.107 vs Muon 3.741, MuonW 3.846, AdamW 3.925, SOAP 3.935; 9.25 h/run, 2,517 ms/step, 36,050 MB | one seed, one epoch, `k = B = 16` untuned; the lr optimum is **interior** (0.1 of 0.02–1.0), so lr is not the explanation — `k/B` is the knob that was never explored |
| C26 | App. I | On MNIST with cross-entropy Sven is 4th of 14 and within 0.9 % of MuonW; the spectrum becomes sharply hierarchical within the first epochs | T1, F2, F9 · `headline.confirmation_table`, `spectra_figs.plot_spectra_over_training` | Sven 0.1134 ± 0.0051, acc 96.6 %; MuonW 0.1046; used rank 10.0→2.8 | the old single-seed CE figure is retired; CE is also the scan with the largest seed spread of the final used rank (68.7 %) |
| C27 | App. R | Selection-level results reproduce; **seed-level** results for Muon (bf16 Newton–Schulz), L-BFGS (line search), SOAP/Shampoo/HIG and every CIFAR run are not bit-reproducible across GPU types | T18 · `headline.timing_join_report` | median relative deviation: nanoGPT 0.0, polynomial 8.7e-9, MNIST-LR 8.3e-3 (max 2.0e+2 on one SOAP run) | Sven's MNIST timing twin deviates 1.26e-3 / 3.80e-2, so "Sven is bit-reproducible" must **not** be claimed; the MNIST confirmation pass is GPU-heterogeneous (45 MIG / 25 A100-80GB) |
| C28 | §3 (related work) | The reviewer-requested baselines are now run, not just cited: AdamW, SOAP, Shampoo, K-FAC, Muon and MuonW under a stated grouping rule; K-FAC has **no eligible configuration** on either MNIST scan (40/40 runs fail in `linalg.eigh` on the rank-deficient Kronecker Fisher) | T1, T3 · `headline.load_selection`, `headline_figs.divergence_by_method` | K-FAC 40/40 on both MNIST scans | this is on-message for §2.3 (K-FAC exists in the over-parametrised regime only via damping) but must be stated as a measurement, not a rhetorical point |
| C29 | §2, App. D | The Gram update is the truncated-SVD update: leading singular values agree to ~5e-07; below ~1e-2·σ_max the float32 Gram, not the method, sets the floor | App. D text · `ckpt_tools` §7 verification | toy online-vs-offline Gram 2.4e-04 at the 1e-6·σ_max threshold, ≤5e-07 above 1e-2·σ_max | do **not** repeat the older "exact to 9e-7" phrasing unqualified |
| C30 | §5 | Sven is a complementary tool, not a replacement; scientific-computing losses that decompose over conditions are the natural target | — | — | keep; drop nothing, add no new promise |

### 2.2 Claims in the current manuscript that the fresh data do not support

| # | current text (line in `iclr2026_conference.tex`) | why it fails | replacement direction |
|---|---|---|---|
| X1 | abstract: "competitive with leading baselines such as Adam, Muon, and K-FAC" (72) | K-FAC has no eligible configuration on either MNIST scan; MuonW beats Sven on four scans | name the field and the standing: beats all first-order and Kronecker-factored baselines on the MLP scans, 2nd/3rd/4th, ties AdamW on nanoGPT, behind HIG and MuonW |
| X2 | abstract: "incurring only a moderate computational overhead relative to SGD… traditional natural gradient methods scale quadratically" (72); intro "overhead is only a factor of k" (90); §2.3 `O(kN|D|)` (290); §5 (387); App. I (846) | the Gram backend's cost is capture + one `B×B` eigh, **independent of k**; `O(kN|D|)` is the classic randomized path only | the C2 statement, in all five places, backend-qualified |
| X3 | abstract: "an optimized implementation that keeps memory usage on-par with standard baselines under mild restrictions on model architecture" (72) | true for the **hooks** capture (MNIST 0.99×, nanoGPT 1.9×) and false for the **full** capture the ResNet needs (47×) | C3, with the capture mode named and the ResNet exception in the same sentence |
| X4 | intro: "Sven outperforms Adam and other standard first-order methods on our regression tasks in both convergence speed and final validation loss" (90) | "convergence speed" is per-step only; wall-clock goes the other way on 2 of 3 | C4 + C7, with "in fewer steps" and the wall-time table cited |
| X5 | Fig. 1 caption: "converges faster per epoch and to a lower final loss than all standard first-order methods, remaining competitive with LBFGS despite significantly lower wall-time cost. Sven performs comparably to HIG" (328) | stochastic L-BFGS diverges on 204 of 225 polynomial runs and is not eligible on its confirmation seeds (2 of 15); **HIG beats Sven** on both synthetics | new caption: the field, the seed band, HIG first, and L-BFGS's failure fraction printed |
| X6 | §4.1: "In the simplest cases … Sven significantly outperforms the standard optimizer baselines"; "For MNIST, Sven matches but does not outperform Adam, though all optimizers perform roughly the same" (370–372) | the first half survives against first-order methods only; the MNIST claim is now a resolved ordering (Sven 3rd, beats Adam by a significant paired margin) | rewrite around T1: significant against first-order, behind HIG/MuonW, MNIST is not a wash |
| X7 | §4.1: "In all cases, the best setting for k is a substantial fraction of (or equal to) the batch size, indicating that there are often a large number of significant directions" (374) | `k=B` is the pick on 6 of 7 scans but the **polynomial pick is k=16 of B=32 with rtol=0.03**, and the used rank is `min(k, rtol-rank)` with `rtol` binding on 5 of 7 | C9: separate the two knobs; report the used rank, not the cap |
| X8 | §4.1: "Improvement begins to saturate around k ~ B/2" (375) | measured at one conservative `rtol`; the plateau is `rtol`-dependent, as the same paragraph half-admits later | re-derive the k-sweep at the **selected** `rtol` per scan (F12) and state the saturation point per scan as a macro, or drop the universal claim |
| X9 | §4.1 spectra paragraph + footnote 8 ("averaging … with respect to the number of batches for which singular value j passes the rtol threshold") (377) | legacy spectra were **truncated at `rtol`** and their tails are survivorship averages; fresh records log all `B` singular values before the cut | C10/C11 from full-width spectra; delete the survivorship footnote; keep the hierarchy contrast but re-measure it |
| X10 | App. E: "Sven performs about as well as the related baselines (LBFGS, HIG) and SGD, with Adam and JD lagging somewhat behind" + footnote "this is likely just because we plot results from one model seed" (683) | one seed; the 14-method confirmation table exists | C26 |
| X11 | App. F: "As with MNIST, Sven achieves a similar loss to the baseline optimizers" (692) | 9th of 11 on both CIFAR scans | C24, plainly |
| X12 | App. F: "The validation loss trajectories … behave essentially the same across a wide range of f, even down to 5 %" + the overparametrisation speculation (692) | under the corrected evaluation f ≤ 0.25 collapses to near chance accuracy with 2 of 15 runs diverged | C23, with the threshold as a macro and the lr confound named |
| X13 | App. G: "we use κ = 2 because we found optimization to be unstable with κ = 1 … would crash within a few epochs due to NaNs"; "we had only one successful run with κ = 1 for 1D regression" (713, 717) | with signed residuals the κ<2 NaN at zero residual is gone: 210/210 complete | C21: κ=2 remains the default, κ=1 is now runnable, and the matched-effective-step comparison replaces the anecdote |
| X14 | App. H: the three-backend profiling story and all its numbers (743–843) | the campaign used the exact **Gram** backend; the v2 profile was measured with a per-step `torch.cuda.empty_cache()` that cost full-capture Sven up to 4.5× | keep the randomized-SVD algorithms as background (they are the classic path), lead with the Gram backend, and regenerate every number from `profile_results_v3` |
| X15 | App. D: "we ran 10 model seeds per hyperparameter setting"; "Figure 1 shows trajectories from the (arbitrarily chosen) numerically smallest seed" (631, 646) | 5 tuning + 5 confirmation seeds; all main figures are seed means ± 1 std | C14; no single-seed curve appears anywhere in the revision |
| X16 | §4 datasets: the polynomial definition (342–346, 617–621) | the target was redefined: **all** 210 monomials of total degree ≤ 4 in 6 variables, factors multiplied, coefficients variance-normalised by `N(0,1)/sqrt(E[m²])` | restate the definition; no absolute polynomial loss from the NeurIPS version may be re-quoted |
| X17 | §4: "All runs are trained for 20 epochs" (354) | nanoGPT is 50 epochs, GPT-2 is one 13,125-step epoch, the P/N synthetics are 200 full-batch epochs | per-scan epochs in App. C |
| X18 | App. F: "We use the 'v2' variant of randomized SVD … to avoid issues around taking the SVD of a k×P matrix" (690) | every CIFAR run used the Gram backend with `gram_capture: full` | re-point to App. D |

---

## 3. SECTION-BY-SECTION EDIT PLAN (main text)

Target: **≤ 9 pages**. The current main text ends on page 9 (`sec:conc` p9, `sec:code` p9), i.e.
it is already full — every addition below is paid for by a named deletion. Blue markup per
`PAPER_CONTRACTS.md`; unchanged text stays byte-identical.

### Preamble
Add, in blue-free code: `\usepackage{xcolor}` is already loaded; define
`\newcommand{\new}[1]{\textcolor{blue}{#1}}` and
`\newenvironment{newtext}{\color{blue}}{}`; `\input{numbers_v2}`;
`\bibliography{example_paper,references_v2}`. Keep `\jdt` (unused). **Add no `\usepackage` line
at all**: `siunitx` is not installed on the build machine (checked), all number formatting happens
in Python, and three packages the file already loads (`algpseudocode`, `bbm`, `nicefrac`) are
themselves missing locally — see the build note in §6 and risk 10.

### Abstract (rewrite ~40 % of it, blue)
Keep sentences 1–3 (decomposition, pseudoinverse, truncated SVD). Replace the cost sentence with
C2, the "competitive with" sentence with C5+C6, and the memory sentence with C3. Add one clause
for the negative results ("and we report plainly where it does not: ResNet18/CIFAR-10 and
GPT-2-small"). Keep the scientific-computing closer. **Length: unchanged (0.55 pp).**

### §1 Introduction (0.90 pp)
* Paragraph 1–2: unchanged.
* Paragraph 3 ("We focus on small-scale regression experiments…"): rewrite in blue. New content:
  the cost statement (C2), the memory statement (C3), the claim that the remaining barrier is the
  **capture under normalisation layers**, not the algebra, and a forward pointer to App. D.
  Delete "even within current constraints, Sven outperforms Adam and other standard first-order
  methods … in both convergence speed and final validation loss" → C4/C7 wording.
* **Paper-organisation paragraph: rewrite entirely (blue).** New text names §2 (update rule,
  dimensionality, Gram implementation), §3 (related work incl. the baselines now run), §4
  (polynomial, MNIST, nanoGPT + spectra), §5 (conclusion + limitations), then groups the
  appendices: theory (A, B), implementation (C, D), results (E–N), mechanism (O), robustness and
  reproducibility (P, R), cost (Q).
* Keep the forthcoming-application sentence.

### §2 Methodology (2.90 pp; net +0.05)
* **Trim (pays for the additions):** the two-part footnote on Appendices A/B (lines 106–108) to
  one sentence; the natural-gradient recap in §2.2 (lines 255–262) by ~1/3 — the functional
  picture is App. A's job.
* **NEW blue paragraph after Eq. (5) — dimensionality (R3's central ask).** `B` = (micro)batch
  size, `P` = number of parameters, `N` = `|D|` = training-set size, `M ∈ R^{B×P}` with row α the
  gradient of the **scalar** effective residual `R_eff^α = (ℓ^α)^{κ/2}`, `M⁺ ∈ R^{P×B}`. State
  explicitly that a multi-output model (MNIST's ten logits, a language model's vocabulary) still
  gives one row per sample, because the residual is the per-sample loss, not the output vector —
  this is exactly R3's Q2. One sentence on the two notions of over-parametrisation: `P > B` (every
  scan) and `P > N` (App. G), and which claims use which.
* **Eq. (9) rewritten (blue)** to expose `R_eff` so κ visibly changes the decomposition rather
  than the loss value (R3's Q3), with the pointer to App. B's new derivation subsection.
* **§2.1 κ paragraph (blue edit):** replace "κ=2 avoids pathologies" with the signed-residual
  statement (C21) — scalar-MSE rows are `sign(r)|r|^κ`, the sign cancels in the pseudoinverse,
  κ=1 completes, κ=2 stays the default, and App. M measures the difference at matched effective
  step.
* **§2.3 cost paragraph (blue rewrite):** C2 in full — the classic randomized path is `O(kN|D|)`;
  the Gram path is the capture plus one `B×B` eigendecomposition, independent of `k`; `rtol` then
  `k` are applied to the eigenpairs; at ResNet scale the binding constraint is the capture's
  memory, not the decomposition. **NEW short blue paragraph** introducing the Gram implementation
  by name (one `B×B` eigh of `G = J Jᵀ` accumulated in float64; `hooks` capture for MLPs and
  transformers, dense `full` capture where batch-statistics normalisation forbids hooks) with the
  measured consequences (C3, C8) and a pointer to App. D. Keep Algorithm 1 unchanged.

### §3 Related Work (1.15 pp; net −0.05)
* Natural-gradient paragraph: add one blue sentence that K-FAC cannot resolve the
  over-parametrised singularity (it still inverts a rank-deficient `P×P` block-Kronecker Fisher
  and exists only via damping — biased, not min-norm), with the measured 40/40 MNIST failure as
  the empirical footnote (R3's Q5, C28).
* Second-order paragraph: add one blue sentence naming SOAP, Shampoo, Muon/MuonW and AdamW as
  **baselines in §4** with the grouping rule pointer, and that MuonW is the strongest baseline on
  the classification scans. Compress the Levenberg–Marquardt sentence.
* Jacobian/pseudoinverse paragraph: unchanged for EGN; compress the JD and HIG sentences by one
  clause each; add a blue clause that HIG is first on both synthetic scans in §4 — the honest
  reading of "mechanistically closest precursor".
* **NEW one-sentence blue addition** on LoRA (R2's Q3): Sven's truncation is a low-rank
  restriction of the *Jacobian row space* per step, not a low-rank re-parameterisation of the
  weights; the fine-tuning application is future work (**no fine-tuning claim** — that scan is
  parked).

### §4 Experiments (2.65 pp)
Structure: intro + datasets (0.26) → protocol (0.22) → baselines (0.12) → **F1** (0.62) →
**T1** (0.40) → results prose (0.30) → **F2** (0.30) → spectra prose (0.21) → **F3** (0.22).

* **Datasets (blue edits):** polynomial redefined (X16); MNIST label regression kept with the
  raw-output definition; **nanoGPT** added (tiny-shakespeare characters, 4 layers / 4 heads /
  128 d, block 128, 826,368 parameters, 80/10/10 by position); pointers to CIFAR-10 (App. J) and
  GPT-2-small (App. L). 1D regression appears as one sentence pointing to App. E.
* **NEW blue protocol paragraph (C14):** three fixed splits; selection on validation only under a
  stated rule (eligible → fewest diverged → seed-mean final validation loss); test metrics as
  outcomes; 5 tuning seeds, then the selected configuration re-run on **5 fresh confirmation
  seeds** (plus 3 data seeds on the synthetics) which is what every number in the paper reports;
  identical initialisation and data order across optimizers at fixed seed, so a paired comparison
  is meaningful; divergence counted, never dropped, with `finished/attempted` in every table.
* **Baselines (blue rewrite of the current list):** 15 methods on the MLP scans — Adam, AdamW,
  SGD, SGD+momentum, RMSprop, Muon, MuonW, SOAP, Shampoo, K-FAC, stochastic L-BFGS, Polyak SGD,
  JD (UPGrad), HIG, Sven — with the Muon grouping rule in one sentence (embeddings and the output
  head to AdamW, hidden projections to Muon, conv kernels flattened, `match_rms_adamw` so one lr
  means the same effective step) and the reduced fields on CIFAR (11) and the LM scans (5), with
  the reasons (HIG refuses batch-statistics BN; K-FAC and JD were not run on ResNet).
* **F1 caption** replaces X5: field, seed band, HIG first on polynomial, the AdamW tie on
  nanoGPT, and the note that the wall-time row uses the standalone timing pass.
* **Results prose (blue rewrite of §4.1 ¶1):** C4, C5, C6, C7 in that order, each with its
  caveat. Two sentences maximum on 1D (pointer to App. E). One sentence on CIFAR/GPT-2 pointing
  to J and L — the negatives are named in the main text, not buried.
* **§4.2 NEW subsection "What the singular values do" (blue, replaces the old §4.1 ¶2–3):**
  C9 (used rank is `min(k, rtol-rank)`, `rtol` binds on 5 of 7), C10 (kept-energy rises over
  training; MNIST-CE's collapse to ~3 directions), C11 (the float64 probe cross-check and
  polynomial's 58.8 %), one sentence of C12 (the min-norm story is not resolved), one sentence of
  C13 (on the ResNet `k` binds). Delete the survivorship footnote (X9) and the "not immediately
  clear why additional singular values are beneficial" speculation — it is now measured.
* **F3** (cost and memory across five model scales) with the Gram story in its caption.

### §5 Conclusion (0.60 pp)
* ¶1: blue edits — replace the "factor of k" sentence with C2, and "performance appears to
  saturate at around k ~ B/2" with the C9/X8 wording.
* Merge ¶2 and ¶4 (both are "scaling is future work"); keep the toolkit framing and the
  scientific-computing paragraph.
* **NEW blue limitations paragraph** (5–6 sentences, not apologetic, no hedging): never first
  (HIG, MuonW); wall-clock cost 2.9–27× Adam and better time-to-target on 1 of 3 regression
  scans; larger tuning grid, and at n=8 trials Sven is not ahead (C16); ResNet18/CIFAR-10 and
  GPT-2-small are negative (C24, C25) and the `k/B` ratio there was never tuned; divergence at
  small `rtol` (C17); the min-norm mechanism is not empirically resolved (C12).
* §6 Code Availability + LLM declaration: unchanged.

---

## 4. APPENDIX PLAN

Existing letters keep their order; new appendices are inserted where they belong and the whole
block is re-lettered by LaTeX. Lengths are estimates in pages (pp).

| id | appendix | status | what changes | assets | length |
|---|---|---|---|---|---|
| A | Natural gradients and functional gradient descent | unchanged | nothing (black, byte-identical) | — | 2.0 |
| B | Gradient descent and functional analysis | **+ new subsection** | NEW blue §B.4 "From the Sven step to the Gauss–Newton/natural-gradient step": the R3-Q4 derivation of Eq. (16) from (8)+(15) — `δθ = −η(MᵀM)⁻¹MᵀR`, `∂L/∂θ = 2MᵀR`, hence `δθⁱ = −(η/2)Σ_j[(MᵀM)⁻¹]^{ij}∂L/∂θ^j` — plus two sentences on why the κ generalisation leaves the identity intact | — | 3.0 + 0.3 |
| C | Experiment details | **heavy update** | datasets (redefined polynomial with variance-normalised monomials; MNIST; CIFAR-10; tiny-shakespeare; FineWeb-edu/GPT-2 BPE), split sizes table, models and parameter counts (593 / 673 / 27,562 / 826,368 / 11,181,642 / 163,109,376), the evaluation protocol (one `evaluate()`, example/token-weighted, eval mode, no buffer mutation, `train_eval` subset), `bn_mode` policy, seeding and data-order derivation, the full per-method hyperparameter grids for all 15 methods on all scans, the Muon grouping rule, weight-decay defaults (not swept), epochs per scan, and the failure-accounting definition | T3, T10 | 2.0 → **3.5** |
| **D** | **Gram-trick implementation (NEW)** | new | the memory-optimised Sven in full: the identity `G = J Jᵀ`, float64 accumulation, one `eigh`, `rtol` cut **then** rank cap `k`, and why the update is identical to the truncated-SVD pipeline (C29 tolerances); **Algorithm 3** (Gram Sven step); the three capture modes (`hooks` = one weighted backward, `full` = `jacrev` over all parameters, `chunked`) with the rule for which model needs which and why batch-statistics BN forbids hooks; the cost decomposition (capture vs solve vs apply) and the memory table; the `k`-independence of the decomposition and what that does to the step-time-vs-`k` curve; the allocator note (`empty_cache` off, expandable segments) as a measurement caveat, not a tuning tip; what remains to be engineered (per-sample gradients under normalisation layers) | F3, T7, T19, Alg. 3 | **2.5** |
| E | Additional results and all-seed plots | **update + absorb 1D** | confirmation-seed curves with ±1 std bands for all four MLP scans **and** nanoGPT (replacing the single-seed figures); the full per-method confirmation table per scan; the ranking summary across all seven scans; the 1D regression scan in full (it leaves the main text here); the data-seed replicate table for the two synthetics; the `k` sweep at each scan's **selected** `rtol` (X8) | F12, F13, T4, T5, T20 | 1.0 → **3.0** |
| **F** | **Tuning budget and paired comparisons (NEW)** | new | R1's implicit and the reviews' explicit fairness objection: distinct-trajectory counts per method, best-of-n curves, the equal-budget table at n=8 (C16), paired per-seed differences vs Sven with 95 % t-intervals for every scan and metric, and the selection-optimism table (C15). Written as a disclosure, not a defence | F4, T5, T6 | **2.0** |
| **G** | **Dataset-level over-parameterisation, P > N (NEW)** | new | R1's crux: loss, test-rank and time-to-target vs P/N for 13 methods on three tasks (toy P=593, polynomial P=673 full-batch; MNIST P=27,562 subsampled to N=2.5k…50k); Sven's rank vs P/N with the significance verdict (C18); reach counts (C19); the divergence-rate comparison at matched P/N; the note that validation and test sets are fixed as N varies, which is what makes the arms comparable, and that MNIST's top point is N=50,000 | F5, T12 | **2.0** |
| **H** | **Batch-size sensitivity (NEW)** | new | R2's Q1: best loss vs B for 13 methods; which method is best at which B (only 1 of 13 is monotone in steps); Sven's selected `k/B` and `rtol` per B; the fixed-`rtol` cost decomposition that explains the apparent flat cost (C20); Sven's effective rank vs B | F6, T13 | **1.5** |
| I | Classification with cross-entropy (MNIST) | **update** | the 14-method confirmation table and confirmation-seed curves replace the single-seed figure; accuracy alongside loss; the CE spectrum transition re-measured (C26, C10) with the used-rank collapse and its seed spread; the label-regression vs CE spectrum comparison kept but regenerated from full-width spectra | F2, F9, T1 | 0.75 → **1.5** |
| J | CIFAR-10 / ResNet18 | **update, both losses** | both scans, 11 methods, validation **and** test loss and accuracy, per-epoch and per-wall-time curves, the optimisation-vs-generalisation decomposition (C24), Sven's `(lr, rtol)` landscape with ineligible and partial cells marked, the used-rank-vs-`k` finding (C13), the cost bars (63 s/epoch, 177 ms/step, 23 GB), the BatchNorm policy statement, and the plain verdict that Sven does not work here yet | F7, T16 | 0.75 → **2.5** |
| **K** | **Parameter fraction at ResNet scale (Fig-5) (NEW, split out of J)** | new | R1's Q2 answered in full: quality vs f with seed bands and `finished/attempted`, accuracy as well as loss, `actual_param_fraction`, peak memory and step time, the statement that the mask is resampled every step, the collapse threshold (C23) and the cost finding that masking is **slower** (C22). **Decision: appendix, not main text** — see §6 note | F7, T15 | **1.5** |
| **L** | **Transformers: nanoGPT and GPT-2-small (NEW)** | new | nanoGPT details (architecture, corpus, splits, 50 epochs, 5 seeds) with the confirmation table, the paired AdamW tie (C6), the lr sensitivity and the cost (2.40 s/epoch, 615 MB); then GPT-2-small as a clearly labelled negative (C25) — one seed, one epoch, `k=B=16`, val/test vs step for all five methods, the interior lr optimum, 9.25 h/run, and the `k/B` caveat as the named next experiment | F1, F11, T17 | **2.0** |
| M | Varying κ | **update** | the matched-effective-step design (`2·lr/κ` with three shared steps), κ ∈ {1,2,3} × two `k` arms, the paired verdict at each matched step (C21), the coverage table with κ=3's grid edge, and the retirement of the κ=1-crashes anecdote (X13) | F8, T14 | 0.5 → **1.25** |
| N | Decreasing memory requirements: micro-batching and parameter batching | **update** | the theory subsections unchanged; results regenerated on the four MLP micro-batch and four param-fraction scans with seed bands, the rank law `used = min(k, rtol-rank)` checked against the micro-batch cap (exact on 7 of 7 MNIST-LR arms, 4 of 6 on toy/poly, 2 of 7 on MNIST-CE), the non-monotone toy loss, the measured cost of both knobs, and the honest statement that neither realises a saving under current autograd — now with the masked-Gram failure count (32 raises, all at lr ≥ 0.5) | F14, T21 | 1.0 → **2.0** |
| **O** | **Singular-value spectra: deep dive (NEW)** | new | the mechanism appendix: full-width online spectra over training for all seven scans with the `k` cut, the `rtol` line and the float32 noise floor; `|uᵢ·r|` vs index; kept-energy and discarded-residual trajectories; used rank vs step and its seed spread; the used-rank grid over `(k, rtol)` with short cells hatched; then the **offline float64 probe-set** spectra along Sven's, Adam's and HIG's trajectories (identical at step 0, separating later), condition number against the float64 resolution ceiling, `σ_B/σ_1` against the floor, effective rank, distance from initialisation and parameter norm with the paired verdicts (C12) | F2, F9, T8, T9 | **3.0** |
| **P** | **Robustness and divergence accounting (NEW)** | new | the two failure definitions (recorded status vs the wider analysis rule) and why the wider one governs everything; per-scan and per-method `finished/attempted`; Sven's divergence pattern in `(rtol, lr)` (C17) with the scans where it is zero; K-FAC's deterministic MNIST failure; L-BFGS's 204/225 polynomial failure and its 2-of-15 confirmation fraction; the masked-Gram guard; the phase-5 passes' own 21 divergences | F10, T11 | **1.5** |
| Q | Memory and time profiling | **update to v3** | lead with the Gram backend (App. D) and keep the two randomized-SVD algorithms as the classic path they describe; regenerate every number from `profile_results_v3`: step time and peak memory vs `k`, batch size, micro-batch, parameter fraction and parameter count for `gram_hooks` / `gram_full` / `gram_chunked` / classic and 12 baselines across five architectures; the capture-vs-solve phase breakdown; the Pareto view; a v2-vs-v3 provenance line stating that the earlier profile carried a per-step cache flush worth up to 4.5× on full-capture Sven **[MOVING]** | F3, F15, T7, T19 | 2.5 → **3.5** |
| **R** | **Reproducibility notes (NEW)** | new | what is fixed and what is not: split seeds, model seeds, derived loader seeds, the confirmation-seed offset, the `run_id`/`run_hash` dedup idea (without repo internals), the timing/diagnostics/confirmation pass design, the cross-GPU non-reproducibility table (C27) with the named methods, the one SOAP trajectory that differs materially between passes, the GPU-heterogeneity note on the MNIST confirmation pass, and the code links. **Anonymous**: no cluster, host, path or scheduler names | T18 | **1.0** |

Appendix total ≈ 40 pp (unlimited). Nothing from the current appendices is deleted outright
except the survivorship footnote (X9) and the single-seed figures they justify.

---

## 5. ASSET INVENTORY

Paths are relative to `iclr_manuscript/`. **module** = which file of `analysis/paper_assets/`
owns it. Panels are single-column width used 3-across at `0.32\linewidth` unless stated;
2-across at `0.49\linewidth`; full width at `\linewidth`. Sven is black
(`style.METHOD_COLORS`), names come from `style.method_label`, bands are mean ± 1 std over seeds
labelled `paired.SEED_SPREAD_LABEL`.

### 5.1 Figures

| id | output path | layout | module | where | functions of record | caption's key message |
|---|---|---|---|---|---|---|
| F1 | `figures_iclr/main/headline_curves.pdf` | 2×3 @0.32 | main | **main §4** | `headline_figs.curve_figure`, `plot_curves`, `confirm_runs`, `standalone_epoch_times`; `large_figs.curve_band` for nanoGPT | validation loss vs epoch (top) and vs standalone wall time (bottom) for polynomial / MNIST-LR / nanoGPT, 15 / 14 / 5 methods, confirmation seeds ± 1 std. **Replaces Fig. 1** (`figures_v2/{toy_1d,polynomial,mnist}_SVDrand/{val_loss_best,wall_time_vs_val_loss_best}.pdf`). Loss values come from the confirmation pass, per-epoch times from `<scan>_timing` — never a timing run's loss |
| F2 | `figures_iclr/spectra/spectrum_truncation.pdf` | 1×3 @0.32 | spectra | **main §4.2** | `spectra_figs.plot_spectra_over_training`, `plot_energy_capture`, `plot_rank_used`, `mechanism_table` | (a) full-width spectrum evolution (polynomial) with the `k` cut, `rtol` line and float32 floor; (b) kept-residual energy vs training progress for poly / MNIST-LR / MNIST-CE / nanoGPT; (c) rank actually used vs step for all seven scans. **Replaces Fig. 2's bottom-left panel** (`figures_v2/comparisons/sv_spectra_lines_combined.pdf`) |
| F3 | `figures_iclr/main/cost_memory.pdf` | 1×2 @0.49 | main | **main §4** | `profile_helpers.plot_sweep`, `scaling_table`, `method_table`; `headline.efficiency_table` | step time (left) and peak memory (right) vs parameter count over five architectures (593 → 163 M) for Sven-Gram-hooks, Sven-Gram-full, Adam, MuonW, HIG, L-BFGS. The Gram story in one picture: cost flat in `k`, memory set by the capture mode. **New** |
| F4 | `figures_iclr/reviewer/budget.pdf` | 1×2 @0.49 | reviewer | App. F | `budget.plot_best_of_n`, `best_of_n_curve`; `headline_figs.equal_budget_table`, `relabel_methods` | best-of-n validation loss vs n per method on polynomial and MNIST-LR, with the n=8 equal-budget marker. Sven's advantage shrinks or inverts at equal budget |
| F5 | `figures_iclr/reviewer/overparam.pdf` | 2×3 @0.32 | reviewer | App. G | `reviewer_figs.plot_arm`, `rank_table`, `reach_table`, `divergence_vs_reference`, `arm_ticks`, `figure_legend` | best loss vs P/N (toy, polynomial, MNIST), Sven's val and test rank vs P/N, time-to-target vs P/N with reach counts, divergence rate vs P/N. Sven improves toward the over-parametrised end but most gaps are ties |
| F6 | `figures_iclr/reviewer/batchsize.pdf` | 2×2 @0.49 | reviewer | App. H | `reviewer_figs.plot_arm`, `arm_table`, `fixed_knob_cost`, `effective_rank` | best loss vs B per method; Sven's effective rank vs B; step time and peak memory vs B at **fixed** `rtol`; divergence vs B |
| F7 | `figures_iclr/large/cifar.pdf`, `cifar_landscape.pdf`, `fig5_quality.pdf`, `fig5_cost.pdf` | 2×2 @0.49; 1×2; 1×2; 1×2 | large | App. J, K | `large_figs.plot_curves`, `optimisation_view`, `plot_cost_bars`, `plot_sven_landscape`, `sven_rank_used`, `plot_paramfrac_quality`, `plot_paramfrac_cost` | CIFAR: val/test loss and accuracy vs epoch and vs time; Sven's `(lr, rtol)` landscape with partial cells annotated `f/a`; Fig-5: quality vs f with hollow reduced-n points, and cost vs f (memory down, step time up). **[MOVING]** |
| F8 | `figures_iclr/reviewer/kappa.pdf` | 1×3 @0.32 | reviewer | App. M | `reviewer_figs.plot_knob`, `matched_step_table`, `matched_step_paired`, `eff_step_coverage` | loss vs **effective step** (not lr) for κ ∈ {1,2,3} at `k=32` and `k=64`; the matched-step spread; used rank vs effective step |
| F9 | `figures_iclr/spectra/probe_spectra.pdf`, `online_spectra.pdf`, `probe_metrics.pdf` | 2×2 @0.49; 2×2; 2×3 @0.32 | spectra | App. O | `spectra_figs.plot_probe_spectra`, `plot_probe_energy_profile`, `plot_utr_over_training`, `plot_energy_profile`, `plot_norms`, `plot_probe_metric`, `plot_rank_used`, `used_rank_grid` | probe-set float64 spectra along Sven / Adam / HIG trajectories at four checkpoints, one named seed per panel; `|uᵢ·r|` profiles; condition number vs the float64 ceiling; `σ_B/σ_1` vs the floor; distance from initialisation and parameter norm |
| F10 | `figures_iclr/reviewer/divergence.pdf` | 1×2 @0.49 | reviewer | App. P | `headline_figs.plot_divergence_grid`, `sven_divergence_grid`, `divergence_by_method`, `divergence_pattern` | Sven's divergence fraction over the `(rtol, lr)` grid per scan, and per-method failure rates with `finished/attempted`. **[MOVING]** |
| F11 | `figures_iclr/large/gpt2.pdf` | 1×2 @0.49 | large | App. L | `large_figs.plot_gpt2_curves`, `gpt2_step_curve`, `gpt2_lr_table`, `add_run_cost` | validation loss vs optimisation step for the five methods (26 evaluation points), and Sven's per-lr curves showing the interior optimum. Labelled a negative result in the caption |
| F12 | `figures_iclr/main/k_sweeps.pdf` | 1×4 @0.24 | main | App. E | `headline_figs.plot_curves` at fixed `(lr, rtol)`; `selected_sven`, `grid_edges` | validation loss vs epoch over `k` at each scan's **selected** `rtol` — the corrected version of the old Fig. 2 top row (X8) |
| F13 | `figures_iclr/main/allseed_curves.pdf` | 3×3 @0.32 | main | App. E | `headline_figs.curve_figure` (epoch, time, train) | confirmation-seed mean ± 1 std for 1D / polynomial / MNIST-LR on three axes. **Replaces Fig. 3** (`figures_v2/*_allSeeds/*`) |
| F14 | `figures_iclr/reviewer/knobs.pdf` | 2×2 @0.49 | reviewer | App. N | `reviewer_figs.plot_knob`, `knob_table`, `rank_vs_cap` | loss vs micro-batch size and vs parameter fraction on the four MLP scans, with the rank law overlaid and the cost of each knob |
| F15 | `figures_iclr/large/profile_*.pdf` (6 files: `methods`, `k_sweep`, `batchsize`, `scaling`, `phases`, `pareto`) | mixed @0.49 / `\linewidth` | large | App. Q | `profile_helpers.plot_method_bars`, `plot_sweep`, `plot_heatmap`, `plot_phase_bars`, `plot_pareto`, `plot_steadiness`, `compare_table`, `sven_change_table` | the full v3 profile: per-method step time and memory per architecture; flat step time vs `k`; scaling in B, micro-batch, parameter fraction and `P`; capture vs solve vs apply; the memory/time Pareto front. **[MOVING]** |

Figures retired without replacement: none — every existing figure is either regenerated, moved to
an appendix, or superseded by a strictly larger version of itself.

### 5.2 Tables (all `\input` from `tables_v2/`, booktabs, generated)

| id | path | module | where | functions | content |
|---|---|---|---|---|---|
| T1 | `tables_v2/headline_confirm.tex` | main | **main §4** | `headline.confirmation_view`, `efficiency_table`, `epochs_to_target` | Sven + 7 baselines × {polynomial, MNIST-LR, nanoGPT}: final val, final test, test accuracy, `finished/attempted`, epochs to the median-method target, s/epoch, peak MB. **The main-text results table; new** |
| T2 | `tables_v2/time_to_target.tex` | main | main §4 (compact) + App. E (full) | `headline.time_to_target_table`, `targets_for` | epochs / steps / examples / standalone seconds to three targets, with `n_reached` and `n_never_reached` |
| T3 | `tables_v2/protocol.tex` | main | App. C | `headline.load_selection`; `EXPERIMENTS.md` §1 as prose source | per scan: dataset, loss, split sizes, model and `P`, batch size, epochs, seeds, methods, `finished/attempted` |
| T4 | `tables_v2/ranking.tex` | main | main §4 or App. E | `headline.ranking_summary`, `rank_matrix` | rank of every method on every scan, validation and test |
| T5 | `tables_v2/confirmation_<scan>.tex` (7) | main | App. E, I, J, L | `headline.confirmation_table`, `confirmation_view`, `paired_vs_sven`, `paired_outcome_vs_sven` | the full per-scan confirmation tables and the paired differences vs Sven with t-intervals and `sven_better` counts |
| T6 | `tables_v2/budget.tex`, `equal_budget.tex`, `optimism.tex` | main | App. F | `budget.trajectory_table`, `headline_figs.equal_budget_table`, `headline.selection_optimism_table` | distinct trajectories per method; rank at n=8; selection optimism same-instance vs pooled |
| T7 | `tables_v2/profile_methods.tex` | large | App. Q + App. D | `profile_helpers.method_table`, `add_relative` | step time, ×Adam, peak memory, ×SGD, capture ms, solve+apply ms, p90/p10 per architecture. **[MOVING]** |
| T8 | `tables_v2/spectra_mechanism.tex` | spectra | main §4.2 (2 columns) + App. O (full) | `spectra_figs.mechanism_table`, `energy_in_top`, `probe_energy`, `used_rank_grid` | per scan: selected `k`, `rtol`, which cut binds, used rank first→final, kept energy first→final, probe top-`k` energy, seed spread of the final rank |
| T9 | `tables_v2/spectra_low4.tex` | spectra | App. O | `spectra_figs.low4_table`, `low4_verdict`, `probe_widths` | distance from initialisation and parameter norm, Sven vs Adam vs HIG, paired with 95 % intervals and a resolved/not-resolved verdict; probe-set dimensions |
| T10 | `tables_v2/grids.tex` | main | App. C | `headline.record_hparams`, `load_selection` | the full hyperparameter grid per method per scan, and the selected configuration with its grid-edge flags |
| T11 | `tables_v2/divergence.tex` | reviewer | App. P | `headline_figs.campaign_divergence`, `divergence_by_method`, `divergence_pattern`; `reviewer_figs.divergence_table` | per scan and per method: recorded vs wide failure counts, `finished/attempted`, and Sven's pattern in `(rtol, lr)`. **[MOVING]** |
| T12 | `tables_v2/overparam.tex` | reviewer | App. G | `reviewer_figs.arm_table`, `rank_table`, `reach_count_table`, `neighbour_gaps` | per P/N arm: best loss per method, Sven's val and test rank, the nearest-rival t statistic, reach counts |
| T13 | `tables_v2/batchsize.tex` | reviewer | App. H | `reviewer_figs.arm_table`, `monotonicity_table`, `fixed_knob_cost` | per B: best loss per method, Sven's selected `k/B` and `rtol`, step time and memory at fixed `rtol` |
| T14 | `tables_v2/kappa.tex` | reviewer | App. M | `reviewer_figs.kappa_table`, `matched_step_table`, `matched_step_paired`, `eff_step_coverage` | κ × effective step × `k`: loss, paired difference, coverage and grid edges |
| T15 | `tables_v2/fig5.tex` | large | App. K | `large_figs.paramfrac_table`, `paramfrac_view`, `paramfrac_config` | per f: val/test loss and accuracy, `finished/attempted`, `actual_param_fraction`, peak memory, step time, and the fixed-configuration note. **[MOVING]** |
| T16 | `tables_v2/cifar.tex` | large | App. J | `large_figs.optimisation_view`, `generalisation_gaps`, `cost_view`, `run_counts` | both CIFAR scans, 11 methods: val, test, val/test accuracy, train-eval, s/epoch, ms/step, peak MB, `finished/attempted`. **[MOVING]** |
| T17 | `tables_v2/transformers.tex` | large | App. L | `headline.confirmation_table` (nanoGPT), `large_figs.gpt2_best_per_method`, `gpt2_lr_table`, `gpt2_summary_view` | nanoGPT confirmation table + GPT-2 per-method and per-lr table with cost |
| T18 | `tables_v2/reproducibility.tex` | main | App. R | `headline.timing_join_report`, `timing_join_view`, `pass_gpus`, `calibration_report` | per scan: median and max relative deviation between the scan and its timing twin, which methods deviate, GPU types per pass, calibration drift |
| T19 | `tables_v2/gram_cost.tex` | large | App. D | `profile_helpers.method_table`, `sven_change_table`, `compare_table` | Sven's four backends × five architectures: step time, capture share, memory, and the `k`-independence check. **[MOVING]** |
| T20 | `tables_v2/data_seeds.tex` | main | App. E | `headline.data_seed_table` | the two synthetics across three data seeds: between-instance spread vs within-instance seed std |
| T21 | `tables_v2/knobs.tex` | reviewer | App. N | `reviewer_figs.knob_table`, `rank_vs_cap`, `divergence_table` | micro-batch and parameter-fraction arms: loss, used rank vs cap, cost, failures |

### 5.3 Number-macro groups

`numbers_v2.tex` `\input`s the four module files. **Macro names contain letters only** (TeX
forbids digits in command names): `\num<Scan><Method><Quantity>`, with scan keys
`Toy`, `Poly`, `MnistLR`, `MnistCE`, `CifarLR`, `CifarCE`, `Nanogpt`, `GptTwo`, `FigFive`,
`Overparam`, `Batchsize`, `Kappa`, `Microbatch`, `Paramfrac`, `Prof`; method keys `Sven`, `Adam`,
`AdamW`, `MuonW`, `Hig`, `Soap`, `Lbfgs`, `Best`, `Median`; quantity keys `Val`, `Test`, `Acc`,
`Rank`, `NMethods`, `Fin`, `Att`, `Epochs`, `Steps`, `WallS`, `MsStep`, `MemMb`, `Ppl`,
`TrainEval`, `Energy`, `UsedRank`, `DivFrac`, `Ci`, `GapRel`.

| group | file | module | contents (≈ count) |
|---|---|---|---|
| G1 | `numbers_v2_main.tex` | main | headline losses / test / accuracy / ranks / field sizes for the 7 scans; paired differences and intervals vs Sven; time-to-target epochs and seconds; step times, wall times and peak memory; selection-optimism percentages; equal-budget ranks; the campaign totals actually quoted (≈ 120 macros) |
| G2 | `numbers_v2_reviewer.tex` | reviewer | P/N ranks, t statistics and reach counts; batch-size levels and the fixed-`rtol` cost ratios; κ matched-step statistics and coverage; micro-batch / parameter-fraction levels and the rank-law hit rate; divergence fractions per scan (≈ 90) |
| G3 | `numbers_v2_large.tex` | large | both CIFAR scans' levels, accuracies, train-eval, costs and ranks; Fig-5 per-f levels and the collapse threshold; nanoGPT and GPT-2 levels, perplexities, costs; every profile-v3 number quoted in App. D and Q, plus the v2→v3 ratio (≈ 100) |
| G4 | `numbers_v2_spectra.tex` | spectra | used rank and kept energy first→final per scan; probe top-`k` energies and `in_span`; rtol-vs-`k` binding flags; condition numbers and resolution limits; `dist_init` / `param_norm` paired values and verdicts; probe-set dimensions (≈ 70) |

### 5.4 `analysis/paper_assets/` — the one command

```
analysis/paper_assets/
  __init__.py      exports build(); no side effects on import
  __main__.py      CLI: python -m paper_assets [--only main,reviewer,large,spectra]
                        [--figures-only|--tables-only|--numbers-only] [--dry-run] [--check]
  _common.py       MANUSCRIPT (= repo/iclr_manuscript), FIG_DIR, TABLE_DIR, NUM_DIR,
                   save_fig() (PDF + a PNG twin for eyeballing, 0.32/0.49/1.0 widths, the
                   manuscript's font sizes), booktabs emitter, Macro() emitter with
                   name validation (letters only, no collisions, sorted output, a comment
                   header naming the source function per macro), and a provenance block
                   (selection timestamp, results root, profile root, freshness flags)
  main.py          F1 F3 F12 F13 · T1 T2 T3 T4 T5 T10 T18 T20 · G1
  reviewer.py      F4 F5 F6 F8 F10 F14 · T6 T11 T12 T13 T14 T21 · G2
  large.py         F7 F11 F15 · T7 T15 T16 T17 T19 · G3
  spectra.py       F2 F9 · T8 T9 · G4
```

Rules: **no number is computed in `paper_assets`** — it calls the analysis functions named in
§5.1–5.3 and formats the result. Every module refuses to write and exits non-zero if
`headline.freshness_report()` flags a scan it depends on **unless** `--allow-provisional` is
passed, in which case it stamps `PROVISIONAL` into the provenance block and into the caption of
every affected asset. `--check` re-runs everything into a temporary directory and diffs, so the
integrator can tell whether a refresh moved the paper. Tests: `tests/test_paper_assets.py` —
macro-name validation, no-digit rule, collision detection, the booktabs emitter on a synthetic
frame, and a smoke test that each module's asset list is non-empty and every declared output path
is written (figures stubbed).

---

## 6. PAGE BUDGET

**Measured, not estimated** (I compiled the current `iclr2026_conference.tex` in a scratch copy):
§2 starts p2, §3 p5, §4 p6, Fig. 1 p7, Fig. 2 p8, §5 and §6 p9, references begin **about 60 %
down page 9**, appendices from p12, 27 pages total. So the body occupies **≈ 8.6 pages** and the
real slack before the 9-page ceiling is **≈ 0.4 page**.

| block | now | new | delta | how |
|---|---|---|---|---|
| title + abstract | 0.55 | 0.55 | 0 | rewritten, same length |
| §1 Introduction | 0.85 | 0.90 | +0.05 | organisation paragraph grows; cost sentence shrinks |
| §2 Methodology | 2.85 | 2.90 | +0.05 | −0.35 (appendix footnote, NGD recap) +0.20 (dimensionality) +0.20 (Gram + cost) |
| §3 Related Work | 1.20 | 1.15 | −0.05 | +3 sentences, −4 clauses |
| §4 Experiments | 2.40 | 2.65 | +0.25 | F1 0.62 + T1 0.40 + F2 0.30 + F3 0.22 + prose 1.11 (was 2 figures 1.15 + prose 1.25) |
| §5 Conclusion | 0.60 | 0.60 | 0 | two paragraphs merged pays for the limitations paragraph |
| §6 Code + LLM | 0.15 | 0.15 | 0 | — |
| **total (body)** | **8.60** | **8.90** | **+0.30** | leaves ~0.1 page of margin; the drop order below is not optional if a float lands badly |

**Local build note (decided).** This machine's TeX Live 2018 is missing `algpseudocode` /
`algorithmicx`, `bbm` and `nicefrac`, all three of which the current file loads; the committed
`.log` shows the real (Overleaf) build has them. **Do not vendor substitutes into
`iclr_manuscript/`** — a local `algpseudocode.sty` would shadow the real package on Overleaf and
silently change how the user's paper renders. Instead the integrator compiles in a scratch copy
with `TEXINPUTS=<shimdir>:` where `<shimdir>` lives **outside** the manuscript repo and holds
three-line stubs (`algpseudocode` mapping `\State/\If/\Else/\EndIf/\For/\EndFor/\Require/\Comment/
\Return` onto the vendored `algorithmic.sty`; `bbm` providing `\mathbbm`; an empty `nicefrac`).
Verified: the shimmed build completes all four passes with **0 undefined references or citations**.
Caveat: the shim's spacing is not the real package's — the shimmed build is 26 pages against the
real 27 and moves individual floats by a page — so **local page counts are indicative, and the
0.4-page slack is the safety margin for that difference**. The authoritative page count is
Overleaf's; report the local number together with the page of `sec:code`.

The three stubs, verified (write them under the agent's own scratch directory, then
`cd <scratchcopy> && TEXINPUTS="<shimdir>:" pdflatex -interaction=nonstopmode iclr2026_conference_v2`
… `bibtex` … ×2):

```latex
% algpseudocode.sty  -- LOCAL BUILD SHIM, never inside iclr_manuscript/
\NeedsTeXFormat{LaTeX2e}\ProvidesPackage{algpseudocode}[2026/09/20 local shim]
\RequirePackage{algorithmic}
\let\State\STATE  \let\If\IF      \let\Else\ELSE     \let\EndIf\ENDIF
\let\For\FOR      \let\EndFor\ENDFOR \let\While\WHILE \let\EndWhile\ENDWHILE
\let\Require\REQUIRE \let\Ensure\ENSURE \let\Comment\COMMENT
\providecommand{\Return}{\textbf{return}~}
% bbm.sty
\NeedsTeXFormat{LaTeX2e}\ProvidesPackage{bbm}[2026/09/20 local shim]
\providecommand{\mathbbm}[1]{\mathbf{#1}}
% nicefrac.sty  -- empty stub (the paper loads it but uses no \nicefrac)
\NeedsTeXFormat{LaTeX2e}\ProvidesPackage{nicefrac}[2026/09/20 local shim]
```

**Drop order if it overflows** (first to go first): F3's memory panel (keep step time only, 0.22 →
0.12); T1's `s/epoch` and `peak MB` columns (they are in App. Q anyway, 0.40 → 0.32); F2's panel
(c) (it is App. O's F9 too, 0.30 → 0.21); T2's main-text compact form (appendix only). Never drop
the protocol paragraph, the dimensionality paragraph or the limitations paragraph — they are the
answers to R3 and the honesty requirement.

**Fig-5 main-vs-appendix — DECIDED: appendix (K), with two sentences and a pointer in §4.**
R1 asked for the main text on the premise that it is the paper's most promising memory result.
Under the corrected protocol it is not a memory *or* time result: masking makes the step 1.9–2.2×
slower at 0.51× memory, quality collapses below f ≈ 0.5, the study is one fixed configuration on
3 seeds with 2 of 15 runs diverged, and its configuration was not the selected one (re-run in
flight). Promoting it to the main text at the cost of the protocol or spectrum material would
overstate it. The §4 sentences say exactly that and point to K, and K opens by naming R1's
request and answering all four parts of it (seeds, accuracy, peak memory, step time, mask
resampling).

---

## 7. RISKS

1. **[MOVING] CIFAR-CE re-selection already changed the configuration** (`lr 0.1 → 0.5`,
   `rtol 0.01 → 0.3`) and the three passes are re-running now: at 22:33 EDT each of
   `cifar10_resnet_ce_scan_{confirm,timing,diag}` held exactly **1 of 5** records at the new
   configuration (56 records each). Every CIFAR-CE number in §2 of this plan is stale — the
   level, the rank, the accuracy and the train-eval diagnosis may all move, and the CE row of
   the spectra tables with them. *Mitigation:* macros only; `large.py` and
   `main.py` gate on `headline.freshness_report()`; App. J's prose is written so the rank
   sentence survives a level change, and the integrator re-runs `python -m paper_assets` after
   phase C closes and re-reads J, P and T16.
2. **[MOVING] Fig-5 re-run may not reproduce the collapse.** The re-run is under way at the
   selected configuration (`k=128, lr=0.5, rtol=1e-3`, 3 of 15 records at 22:33, 5 claims live),
   which removes exactly the `lr = 1.0` confound both old blow-ups sat at — so the collapse
   threshold is likely to move and may disappear. *Mitigation:* C23's threshold is a macro; App. K
   states the confound and reports whichever threshold the re-run gives; the cost finding (C22) is
   independent of it. The old 15 runs stay on disk under their own `run_id`s, so
   `large_figs.paramfrac_config` must be asked which configuration it is reading.
3. **[MOVING] `profile_results_v3` is 578/720 and contains no `cifar_resnet18` directory.** If
   CIFAR never lands in v3, App. Q and F3's CIFAR point must either stay on v2 (contaminated by
   the per-step cache flush) or be sourced from the timing pass (177 ms/step, 22,960 MB — measured
   in training, which is the better number anyway). *Mitigation:* F3's CIFAR point comes from
   `headline.efficiency_table` (timing pass) in both cases; App. Q prints
   `profile_helpers.profile_status()` and a per-row provenance flag; `large.py` refuses to mix v2
   and v3 rows in one table.
4. **9-page limit.** The body already occupies ~8.6 of the 9 pages and LaTeX page breaks are not
   linear in words. *Mitigation:* the drop order in §6, and the integrator compiles and reports
   the page of `sec:code` after every asset change.
5. **Over-claiming.** HIG is first on both synthetics, MuonW on four scans, and Sven is never
   first. The reviewers will check. *Mitigation:* §2's caveat column is binding; the review lens
   for the paper agents should be "find a sentence whose caveat is missing".
6. **Polynomial target redefinition.** Any absolute polynomial number from the NeurIPS version is
   now wrong. *Mitigation:* X16; `main.py` emits no polynomial macro that is not regenerated; App.
   C states the definition without narrating the change.
7. **Ineligible cells printed as means.** Polynomial L-BFGS rests on 2 of 15 confirmation runs and
   K-FAC has no eligible MNIST configuration. *Mitigation:* every table prints
   `finished/attempted` and the tuning-seed basis marker; `headline.confirmation_view` already
   carries both.
8. **GPU heterogeneity inside a pass.** The MNIST confirmation pass is 45 MIG / 25 A100-80GB (Sven
   on 80GB for label regression, on MIG for CE). *Mitigation:* `headline_figs.gpu_homogeneity_line`
   in App. R and a footnote on T1.
9. **Blue markup leaking.** `\color{blue}` inside a float or a `tabular` can colour rules and
   escape the group. *Mitigation:* `\new{...}` (= `\textcolor`) for anything inside a table or
   caption; `newtext` only around whole paragraphs at text level; never wrap a `figure`/`table`
   environment; colour captions with `\new{}` around the caption text.
10. **The build machine cannot compile the paper as shipped.** `algpseudocode`/`algorithmicx`,
    `bbm` and `nicefrac` are absent from this TeX Live 2018 (`siunitx` too, so it may not be
    added); `xcolor`, `booktabs` and `subcaption` are present. *Mitigation:* the out-of-repo
    `TEXINPUTS` shim in §6 — verified to build the current file cleanly — and no new
    `\usepackage` line in the v2 file. Never commit a shim into `iclr_manuscript/`.
11. **`analysis/tables/*.tex` are not publication-quality** (they come from a markdown-ish
    fallback renderer; `tabulate` is absent). *Mitigation:* `paper_assets._common` emits its own
    booktabs; nothing under `analysis/tables/` is `\input` by the manuscript.
12. **New citations.** `references_v2.bib` needs at least: nanoGPT / tiny-shakespeare, GPT-2
    (Radford et al.), FineWeb-edu, LoRA (Hu et al.), the Muon scaling/`match_rms_adamw` reference,
    `torch.func`/`functorch`, and SGD-with-momentum. Every `\cite` must resolve — the build gate
    is "no undefined references or citations".
13. **Anonymity.** App. C, Q and R are the dangerous ones (hardware, hosts, paths, scheduler).
    Say "a single 40 GB or 80 GB data-centre GPU per run"; name no cluster, partition, path or
    repository other than the existing anonymised links.
14. **Divergence totals drift.** `campaign_divergence` reported 7,865 → 7,867 → ~7,870 on-grid
    Sven runs within an hour. *Mitigation:* quote per-scan fractions, and take the total from the
    macro, never from prose in another document.

---

## 8. SEQUENCING

1. **paper-assets-core** — `analysis/paper_assets/{__init__,__main__,_common}.py` +
   `tests/test_paper_assets.py`. Must land before the four module agents (they import `_common`).
2. **In parallel:** `main.py` (F1 F3 F12 F13, T1–T5 T10 T18 T20, G1) · `reviewer.py` (F4 F5 F6 F8
   F10 F14, T6 T11–T14 T21, G2) · `large.py` (F7 F11 F15, T7 T15–T17 T19, G3) · `spectra.py`
   (F2 F9, T8 T9, G4). Each: implement → run → paste the macro file's head and the figure list
   into its report.
3. **In parallel with 2:** the two prose drafts that do not depend on assets —
   App. D (Gram implementation, incl. Algorithm 3) and App. C (experiment details), plus
   `references_v2.bib`.
4. **integrate** — copy `iclr2026_conference.tex` → `iclr2026_conference_v2.tex`, apply §3 and §4
   with blue markup, `\input{numbers_v2}`, wire every asset, compile the four-pass sequence,
   report the page of `sec:code`.
5. **review** (3 lenses: claims-vs-caveats against §2, numbers-vs-macros against the analysis
   functions, LaTeX/anonymity/page-count) → **fix**.
6. **after phase C closes:** re-run `cd analysis && ../.venv/bin/python -m paper_assets`,
   recompile, re-read App. J, K, P, Q and every `[MOVING]` claim.
