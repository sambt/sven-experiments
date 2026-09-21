# What the robustness fixes changed

`CHANGES_NEEDED.md` §4.2 phase 7 and §6 ask for the differences from the legacy tables in
writing. This is that account. Every number here is computed by `analysis/legacy_vs_fresh.ipynb`
(module `analysis/legacy_diff.py`, tables exported to `analysis/tables/legacy/`, figures to
`analysis/plots_v2/legacy_vs_fresh/`); re-executing that notebook regenerates all of them.

**How the two sides are made comparable.** Legacy = `experiment_results_legacy_2026-09-18/`,
read-only. Its validation losses averaged per-batch means, so every legacy number below is the
**example-weighted repair** (`analysis/repair_legacy.py` → `analysis/legacy_repair/*.parquet`),
which is exact: the equal-weight mean of the same per-batch blocks reproduces the stored curve to
< 2e-5 on all 21 repaired scans. The correction is 0.06–0.42 % median on the headline scans and
8.6 % median / 53 % max on the masked CIFAR Fig-5 runs. The legacy *selection* is the binding rule
(`CHANGES_NEEDED.md` §1) applied to legacy data; the fresh numbers are not re-selected at all but
read from `bench/best_configs.json` through `analysis/headline.py`. As a gate, this module's own
rule reproduces **84 of 85** selections exactly; the single exception is CIFAR-CE Sven, whose
scan is still growing (the rtol extension), and the notebook asserts that any such drift is
confined to the scans `headline.freshness_report()` flags as provisional.

**Scope.** 27 scan directories exist in both roots: 7 headline (full per-method diff), 4 swept
studies (ranked within each axis value), 10 Sven-only ablations (configuration + level only),
6 excluded (5 standalone timing passes, and GPT-2, whose legacy corpus had validation as a prefix
of training — a quarantine, not a comparison).

**Seven legacy scans are outside that intersection and vanish from every table below** (notebook
§1.1b; dispositions quoted from `EXPERIMENTS.md` §10):

| legacy scan | runs | now rests on |
|---|---|---|
| `exp_critbatch_mnist`, `exp_critbatch_nanogpt` | 210 each | cut by the user — **nothing**; the critical-batch figure has no replacement |
| `cifar10_resnet_kappaScan_labelReg`, `cifar10_resnet_ce_kappaScan` | 5, 2 | cut (1 seed) — the κ study is MNIST-only now, so **no fresh CIFAR κ data** |
| `cifar10_resnet_paramFrac_scan_labelReg` | 5 | `rebuttal_fig5_cifar_paramfrac_scan` (3 seeds, in both roots, §5) |
| `cifar10_resnet_ce_paramFrac_scan` | 6 | **nothing** — Fig-5 replaces label regression only |
| `exp_finetune_cifar_smallN` | 240 | parked (408 runs); its legacy runs trained BatchNorm on 250–2000 images, the defect C-E2 names — **the fine-tuning claim is unsupported** |

So **four legacy figures rest on nothing** and must be dropped or re-run, not re-quoted. Nothing
is lost the other way: all 16 fresh-only directories are `_confirm` / `_diag` / `_timing` passes
of the headline scans, which the legacy protocol had no equivalent of.

**Ranks are given in two fields, and on a stated basis.** The fresh campaign added baselines
(SGDm everywhere; AdamW/Muon/MuonW/SOAP on CIFAR; KFAC on MNIST), so "n-th" is ambiguous. Below,
*rank* means rank among the methods **both** campaigns ran; the full-field rank is given where it
matters. A fresh rank is a rank on the **confirmation** mean, except where that configuration is
not eligible on the confirmation seeds, where the rule falls back to the tuning-seed loss and the
table marks the cell `*` with its basis in its own column (1 of 85 headline rows: polynomial
L-BFGS, 2 of 15 confirmation runs finished). **Sven is eligible on all seven**, so nothing said
about Sven here rests on that fallback.

---

## 1. Sven's rank, legacy → fresh

| scan | legacy | fresh | change | Sven val legacy → fresh |
|---|---|---|---|---|
| Toy 1D | 2/14 | 2/14 | = | 2.97e-06 → 5.09e-07 |
| Random Polynomial | 8/14 | **2/14** | **−6** | ranks only (additive cubic) |
| MNIST (label reg.) | 4/13 | 3/13 | −1 | 0.0504 → 0.0533 |
| MNIST (CE) | 5/13 | 4/13 | −1 | 0.1131 → 0.1134 |
| CIFAR-10 (label reg.) | **2/6** | **5/6** | **+3** | 0.4107 → 0.4838 (+17.8 %) |
| CIFAR-10 (CE) *(provisional)* | 5/6 | 5/6 | = | 1.415 → 1.424 |
| nanoGPT (tiny-shakespeare) | 2/5 | 2/5 | = | 1.760 → 1.715 |

**The aggregate must be split by target, and is then flat.** Of the 26 comparable points (the 7
above plus 19 axis points of the swept studies), **11 are polynomial** — ranks on the legacy
*additive cubic*, a different objective function — and they carry **89 % of the pooled total**, so
a pooled "−0.69 places on average" is mostly a statement about which function was fitted
(`legacy_diff.rank_change_tally`, from the same `ranks_only` flag the table prints):

| group | points | better | unchanged | worse | Σ Δrank | mean Δrank |
|---|---|---|---|---|---|---|
| same target | 15 (6 headline + 9 swept) | 7 | 5 | 3 | −2 | **−0.13** |
| ranks only, additive cubic | 11 (1 headline + 10 swept) | 5 | 3 | 3 | −16 | −1.45 |

**Where the two campaigns fitted the same function Sven's rank barely moved** — mean −0.13
places, and on the headline scans 2 better, 3 unchanged, 1 worse (the worse one being CIFAR
label regression, +3). All three large gains (−6, −6, −5) are additive-cubic comparisons. In the
**full** fresh field Sven is 2nd/15 (toy), 2nd/15 (polynomial), 3rd/14 (MNIST-LR), 4th/14
(MNIST-CE), 9th/11 (CIFAR-LR), 9th/11 on validation and 11th/11 on test accuracy (CIFAR-CE,
provisional), 2nd/5 (nanoGPT).

Swept studies (rank within each axis value): MNIST P/N sweep 2,1,3,4,3 → 1,1,1,2,2 (Sven
improves at every n_data); polynomial P/N 2,2,2,2 → 1,2,2,4; toy P/N 2,2,1,4 → 2,4,3,3;
polynomial batch size 9,6,7,1,2,1 → 3,1,1,1,5,4. The rebuttal's claim of a *uniform* 2nd place in
the overparametrised regime does not survive as stated, but Sven is 1st or 2nd at 12 of the 19 axis
points, and the metric here (final validation loss) is not the metric the rebuttal used
(wall-clock to train loss < 1e-3).

## 2. The causes, and the five that could be tested

Each row of every per-scan table carries codes from a closed vocabulary: `split`,
`testsplit_carved`, `ew`, `bn`, `target`, `fixedval`, `muon`, `adamw_wd`, `grid`, `failures`,
`droplast`, `newbase` (`legacy_diff.CAUSES` spells them out; they are assigned from facts — the
dataset, the method's absence from the legacy root, the two grid sizes on each side — never from
prose). Five are testable:

* **CIFAR BatchNorm (`bn`) — confirmed, and it is the biggest single change.** On label
  regression Sven moves **+17.8 %** on a grid that did *not* change (18 → 18 configurations), so
  the move cannot be a tuning-budget effect; the two baselines the rule still puts at the same
  learning rate move +1.0 % (Adam) and +1.7 % (SGD). Sven was the only optimizer evaluated in
  train mode on validation-batch statistics. On cross-entropy Sven moves only +0.6 %: it was
  already 5th of 6 there, so the BatchNorm artefact was what made "Sven ≈ Adam on CIFAR" look
  true on the regression task, not on the classification one.
* **MNIST split (`split`) — confirmed.** For Adam, SGD, RMSprop and Shampoo on both MNIST scans
  the rule picks the **same learning rate on 8 of 8** (method, scan) pairs although every one of
  those grids doubled; levels shift by 0.8–18.5 % (median 7.3 %), which is the difference between
  two validation sets, and ranks move by at most 3 places — the field moving around them.
* **Failures were invisible (`failures`) — confirmed.** On the legacy root 16 seed-runs of an
  *observed* configuration left no record at all: HIG lost 6 of them and recorded 4
  divergences, KFAC lost 10 and recorded 12 — and on legacy toy HIG recorded **no failure of any
  kind** while 5 of its runs were simply absent. Legacy toy HIG is 35 runs over 8
  configurations; the fresh scan is 210/210 over 42 configurations with 11 recorded divergences.
  A configuration whose every seed crashed cannot even be counted without a manifest, which is
  what C-R2 added.
* **AdamW was Adam (`adamw_wd`) — confirmed.** In the legacy rebuttal scans, which still hold
  both generations, **394 of 394** paired `wd = 0` AdamW runs are bit-identical to Adam at the
  same lr and seed; at `wd = 0.01` none are.
* **nanoGPT is not a `split` story (`testsplit_carved`) — measured.** The character corpus never
  had a test set: it was 90/10 by position and is now 80/10/10. Legacy `val[0] = 4.368765` equals
  fresh `test[0] = 4.368757` (1.8e−6) while fresh `val[0] = 4.362548` differs by 1.4e−3, `n_val`
  is unchanged and `n_train` fell by exactly `n_test` (7,842 → 6,971). So **the legacy validation
  blocks are the fresh *test* split**: legacy did select and report on one held-out set, but the
  legacy number is comparable with the fresh **test** number, not the fresh validation one. Sven
  is 2nd of 5 either way.

Not separately testable: grids grew (toy HIG 8 → 42 configurations, Sven 72 → 180; MNIST-CE Sven
128 → 160), and Muon's regrouping plus `match_rms_adamw` made MuonW the strongest baseline on
both MNIST and both CIFAR scans. Grids are now counted on the Sven-only ablations too, and one
cannot be read as a measurement: the **κ study's legacy grid is 3 configurations against 42** and
the fresh winner (k=32, lr=0.75, κ=3) is not on it, so its "+0.9 %" compares two searches. Six of
the ten ablations were searched over identical grids, so their level changes — CIFAR Fig-5's
+59.8 % on an unchanged 5-point grid included — are not search-budget artefacts.

## 3. Predicted vs observed

| the critiques predicted | what happened |
|---|---|
| Polynomial results describe an additive cubic, not a degree-4 target (C-T1) | Ranks are not preserved: Sven 8th → 2nd, KFAC 13th → 7th. Losses are incomparable and are printed as `n/a`. |
| Sven's CIFAR parity is a BatchNorm artefact (C-T2) | Confirmed: +17.8 % on label regression at an unchanged grid, 2nd/6 → 5th/6 (9th/11 in the full field). |
| Using the test set for selection inflates the reported numbers (C-T4) | Levels shift a few per cent on MNIST with the selected lr unchanged; the *ranking* is unaffected. The optimism is now visible instead as the tuning → confirmation gap (`headline.py`). |
| Missing runs beat recorded divergences (C-T5) | Confirmed, and it favoured HIG and KFAC — the two methods that lost runs silently. |
| "AdamW" duplicates Adam (C-T7) | Confirmed on 394 paired runs in the legacy rebuttal scans. |
| Batch-weighted metrics distort batch-size comparisons (C-T8) | Median correction 0.19 % on the batch-size scan but 8.5 % max, and 8.6 % median on the masked CIFAR runs. |
| Tuning budget favours Sven (C-M1) | The fresh grids narrow the gap but do not close it (Sven 180 vs 8 configurations for most baselines on toy); `headline.budget_table` is the disclosure. |

## 4. Which claims survive, and which must change

The abstract's regression sentence reads, verbatim (`sven_submission.pdf`): *"On regression
tasks, Sven significantly outperforms standard first-order methods including Adam, **converging
faster and** to a lower final loss, while remaining competitive with LBFGS **at a fraction of the
wall-time cost**."* Those are four separable claims and they do not all land the same way.

**Survive.** (i) *"…to a lower final loss"* — Sven beats Adam, SGD, RMSprop and AdamW on toy
(2nd vs 8th), polynomial (2nd vs 10th), MNIST label-regression (3rd vs 8th) and MNIST-CE (4th vs
9th), on fresh confirmation seeds with a held-out split. (ii) *"competitive with LBFGS"* —
stochastic L-BFGS is last or near-last on the two synthetic scans and diverges on 204 of 225
polynomial runs; on polynomial it finishes only 2 of 15 confirmation runs, so it is not eligible
there at all and its rank is taken on its tuning seeds (§Scope). (iii) The nanoGPT ordering is
unchanged (AdamW 1st, Sven 2nd of 5). (iv) `REBUTTALS.md`'s P/N story survives in direction:
Sven is 1st or 2nd at 12 of the 19 swept points, and first on MNIST at the three most
overparametrised ones.

**Must be requalified.** (v) *"converging faster"* is true **per step and false in wall-clock**,
and the abstract does not say which. At the median-method target on the confirmation seeds
(`headline.time_to_target_table`, notebook §3.6) Sven needs fewer epochs than Adam on 3 of 3
regression scans (toy 4.27 vs 10.0; polynomial 2.27 vs 3.73; MNIST-LR 1.0 vs 1.4) but less wall
time on only **1 of 3** (toy 5.6 vs 5.7 s; polynomial 3.0 vs 2.2 s = 1.39×; MNIST-LR 3.7 vs
2.1 s = 1.81×). The claim must say "in fewer steps" and cite WP2's efficiency table for the
wall-clock, which runs the other way. (vi) *"at a fraction of the wall-time cost"* of L-BFGS is
**not tested here at all** — this document compares levels and ranks, not cost — and belongs to
WP2's table too.

**Must change.** (i) **CIFAR.** `REBUTTALS.md` §1's "the single true dataset-overparam case
(ResNet18/CIFAR) merely matches baselines at higher wall-time" is wrong: under a correct
BatchNorm policy Sven is 9th of 11 on label regression (test accuracy 69.4 % vs 77.9 % for SOAP)
and, provisionally, 9th of 11 on validation and last on test accuracy on cross-entropy, at
63 s/epoch against 2.7–12.4 s (WP2's efficiency table). (ii) **"Outperforms standard first-order
methods" needs qualifying**: Muon with weight decay beats Sven on both MNIST scans and both CIFAR
scans, and HIG beats it on toy and polynomial. Sven is never first on any headline scan.
(iii) **Every polynomial number in the paper must be re-derived**: the published target was an
additive cubic. (iv) `REBUTTALS.md` §4.6's "consistently 2nd-fastest" and the 1e-8–1e-11
polynomial train losses are legacy-target, legacy-metric statements. (v) The Fig-5 masking
numbers move: legacy Sven at `param_fraction = 1` reads 0.418 and the fresh run 0.668 (+59.8 %),
the same BatchNorm story with an 8.6 % example-weighting correction on top.

## 5. What this comparison cannot say

One legacy instance (toy and polynomial were a single data draw; the fresh confirmation pools
three), different seed counts (5 tuning vs 5 tuning **and** 5 confirmation), different validation
data on every scan, and different GPU types between the scans (mostly MIG A100-40GB) and the
confirmation/timing passes (A100-80GB) — nothing here is a bit-level comparison. A rank change of
one place is often a few per cent of loss between methods whose seed bands overlap; the paired
differences in `analysis/headline.py` are where that is quantified. **CIFAR-CE is provisional**
until the rtol extension lands and `tools/select_best.py` is re-run; re-executing the notebook
updates every CIFAR-CE number in it.
