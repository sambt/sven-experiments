## Configs track — fixer report

### Findings: 7 accepted, 1 accepted as its stated alternative

**F1 (high, CIFAR-CE `lrs_standard`) — fixed.** Confirmed against the 290 legacy jsonl: RMSprop's optimum **is** the 1e-1 top edge (seed-mean final val 2.1002/1.3324/1.3725/**1.1983** at 1e-4…1e-1); Adam interior (1e-2), SGD interior (1e-3). Added `1.0` (+40 runs, ~0.5 GPU-h), rewrote the false "keeps SGD/SGDm off the top edge" comment, added `("cifar10_resnet_ce_scan","lrs_standard",[3e-1,1.0])` to `MUST_CONTAIN`. **Deviation from the suggested fix:** *not* added to label-reg — measured there, nothing sits on the top edge (RMSprop 0.4157 at 1e-2 vs 0.9331 at 1e-1; SGD 1/5 seeds finite at 1e-2, 0/5 at 1e-1), and `match_rms_adamw` moves Muon's optimum *down*. Both configs now state their own evidence.

**F2 (high, `lrs_lbfgs`) — fixed on both scans.** Confirmed: label-reg per-lr means 3.0635/0.7855/**0.7534**, best shape mi3/hs5 0.7010→**0.6991**; CE's best shape mi2/hs5 1.3533→**1.3201** is also on the edge (the per-lr mean is best at 0.5 only because the mi3 arms blow up 1.78→3.04). Added `2.0` to both (+45 runs each, ~1.7 GPU-h each), pinned in `MUST_CONTAIN`. One point, not two: 0.3%/2.5% gain across the last half-decade, and lr>1 under `strong_wolfe` over-relaxes.

**F3 (kappa) — grid fixed, not just the comment.** `lrs: [0.125,0.25,0.375,0.5,0.75,1.0,1.5]` (210 runs, +60, +0.6 GPU-h) realises **three** effective steps `2*lr/kappa` ∈ {0.25,0.5,1.0} at *all three* kappas instead of one; keeps legacy lr 0.5. Test now asserts the matched set exactly (`KAPPA_MATCHED_STEPS`), replacing `assert matched`.

**F4 (HIG rationale) — fixed.** Confirmed: `mnist_scan_ce` has 20/20 finite HIG runs at lr 0.5 and 20/20 at 1.0 (0.1735 / 0.2023 vs 0.1035 at 0.05). Corrected in 4 configs, `test_hig_grid_is_shifted_down_not_grown`, and `_explain_mnist_scan_ce`.

**F5 (`strip_post_legacy`) — fixed** with `EXPECTED_BN_TOKEN = {scan: (token, carrier_families)}`; strips only the expected token, flags any other, and requires `n_bn == n_carry`. Proved by re-running the CIFAR case three ways: `OK as-is (90/90)`, `BAD bn_mode deleted (n_bn=0 expected=90)`, `BAD bn_mode: frozen (565 unexpected tokens)` — the old code printed `OK` for both failures.

**F6 (Fig-5) — guarded as far as my files reach.** New `test_fig5_setpoint_is_flagged_tentative_exactly_while_it_is_tentative`: marker present ⟹ values are exactly the classic triple; marker gone ⟹ values changed. The *launch* half needs `p1_cifar_fig5` set `enabled: false` — launcher's file, see below.

**F7 (CIFAR jd/hig) — took the stated alternative, rejected deletion.** `plan_campaign.yaml` parks `p3_cifar_jd_hig` explicitly "needs a user decision"; deleting the keys would make revival a config edit against a pending decision. Instead the table states them as `(20)`/`(80)` (parser extended: `(N)` = described, unbudgeted), pinned by `PARKED_MODES`, plus a config comment naming the `mode=all` consequence, plus open item 4.

**F8 (parked accounting) — fixed.** Row is `P3-parked`; totals split **15735 in-plan + 408 parked = 16143**; storage ~107 GB described / **~89 GB in-plan**; test asserts all three numbers and that only `PARKED` scans say "parked".

### Verified by running
- `.venv/bin/python -m pytest tests/test_configs.py tests/test_grid.py tests/test_tools_launch.py -q` → **`379 passed, 10 skipped in 28.53s`**
- `campaign/run_cpu_tests.sh … pytest tests/ -q` → **`685 passed, 29 skipped, 27 warnings in 102.28s`**, `exit=0`
- `campaign/run_cpu_tests.sh … tests/golden/legacy_grid_equiv.py` → **`ALL EQUIVALENT`**, `exit=0`; CIFAR `655 order_identical=True (90 ids carry _bnbatch)`, kappa `210`, finetune `(90 ids carry _bnfrozen)`. **No re-freeze needed** — the two FREEZE scans still enumerate 1080 / 1680.

### Files changed
`experiments/configs/{cifar10_resnet_ce_scan, cifar10_resnet_scan_labelRegression, mnist_kappaScan_labelRegression, toy_1d_scan, polynomial_scan, mnist_scan_ce, mnist_scan_labelRegression}.yaml`; `tests/golden/legacy_grid_equiv.py`; `tests/test_configs.py`; `campaign/grid_counts.md` (rewritten).

**`tests/test_grid.py` (integrator's) — 6 more minimal edits:** RE-FREEZING docstring `15 -> 210`; INVENTORY comment gains the CIFAR-lbfgs delta; `cifar10_resnet_scan_labelRegression` lbfgs `135 → 180`; `mnist_kappaScan_labelRegression` svd `150 → 210`; kappa token counts `50/100 → 70/140`; `_explain_mnist_scan_ce` docstring.

### Open questions for the orchestrator
1. **CIFAR-label-reg Sven `lrs` edge still accepted** (open item 1). Baseline edges on both CIFAR scans are now measured and extended; Sven is the only knowingly-accepted edge there (+60 runs ≈ 30 GPU-h to close).
2. **`p1_cifar_fig5` is enabled by omission** — a P1 launch today burns ~5.5 GPU-h on the tentative set point and yields a wrong Fig-5. Needs the launcher to park it.
3. P1 rebuttal lr grids still unextended (~+1,900 runs, <2 GPU-h).
4. CIFAR jd/hig: budget ~100 GPU-h or decide JD/HIG stays on the MLP scans (then the three keys can go).

### Notes for other tracks
- **launcher:** re-read `campaign/grid_counts.md` — CIFAR-CE **655**, CIFAR-label-reg **555**, kappa **210**; in-plan total **15735**. `test_the_plan_covers_every_run_grid_counts_says_exists` passes unchanged. Please set `enabled: false` on `p1_cifar_fig5`.
- **integrator:** the 6 edits above are all counts/docstrings; no assertion was loosened.