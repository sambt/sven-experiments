All ten findings verified and fixed, notebooks re-executed clean, full suite green.

**WP2 phase-B fix — all 10 reviewer findings were real; all 10 fixed.** `pytest tests/` → **1210 passed, 33 skipped**; 6 notebooks re-executed in place, **0 error cells, 0 unexecuted** each. Frozen helpers untouched (`git diff` on `headline.py`, `budget.py`, `style.py`, `paired.py`, `analysis_helpers.py`, `scan_analysis.py`, `make_plots.sh` is empty — **no bug fix needed in a shared module**).

## Files
**Changed:** `/n/home11/sambt/iaifi/sv3/analysis/headline_figs.py` (772→1190 l), `/n/home11/sambt/iaifi/sv3/tests/test_headline_figs.py` (24→34 tests), `analysis/{toy_1d,polynomial,mnist,mnist_labelRegression}_analysis.ipynb` (38 cells / 24 code, ~31 s each), `analysis/baselines_analysis.ipynb` (36 / 23, 48 s), `analysis/headline_tables.ipynb` (28→**29** cells / 20 code, 29 s; re-exported `analysis/tables/`, now against **778** CIFAR-CE runs). Scratch generators (gitignored): `agent_lab/wp2/build_{mlp_notebooks,baselines_notebook}.py`, `run_nbs.sh`, `check_nb.py`.

## Per finding
1. **high, budget claim (baselines cell 34).** Real — reproduced from `analysis/tables/budget_best_of_n_*`. New `hf.equal_budget_table` + `closing_numbers` keys; a method with <n points is read at its own grid size. **At n=8: Sven ranks 15/15 (toy, 14 better, needs n=28 to match SOAP), 4/15 (poly), 10/15 (MNIST-LR), 8/15 (MNIST-CE); on the three losers it never matches the leader at any budget its grid allows.** Bullet rewritten: the distinct-trajectory count (0.475–0.664) is a point *for* the objection.
2. **high, "ahead of that field" on steps.** Real. New `sven_rank_epochs_to_target` / `n_reached_target` / `ttt_fewest_epochs_method`: **Sven is 2/13, 4/13, 4/11, 4/11**, behind HIG 2.2, HIG 3.53, Muon 6.0, Polyak SGD 3.6 epochs. All five closing cells rewritten.
3. **high, ms/step ordering.** Real. Replaced with computed neighbours: Sven 3.40/3.44/4.03/4.00 ms, **9–10 configs cheaper, 3–5 dearer**; below it SOAP/L-BFGS/SOAP/SOAP, above it Shampoo/Shampoo/L-BFGS/JD, dearest JD 6.3/5.6 and Shampoo 128/129 ms. L-BFGS is *cheaper* than Sven on 2 of 4 (not "well below" it anywhere).
4. **medium, CIFAR hard-coded in headline_tables.** Real. New code cell 27 computes it; §7 markdown carries no CIFAR digits. Prints: **Sven 53.0% vs 56.8–77.9% (weak two SGDm 56.8, Polyak 63.2; other eight 68.6–77.9), train-eval 0.819 vs 0.0046–0.651 (highest of 11: True), 63.0 s/epoch vs 2.70–12.4**, config `k=128, lr=0.1, rtol=0.01`, `PROVISIONAL=True`.
5. **medium, banner + silent NaN row.** Real. `provisional_banner` now prints the 3-step order (re-select → **re-run `_confirm`/`_timing`/`_diag` for any method whose config moved** → re-execute); same text in headline_tables cell 5. New `hf.missing_confirmation` shouts on a 0-row match instead of yielding a NaN row that `ranking_summary` sorts last.
6. **medium, phantom log-axis segment.** Real (`transform([0.])` → −1000). `plot_curves` now uses `set_xscale('log', nonpositive='mask')`; **visually confirmed** on the re-rendered MNIST-LR time panel — curves start at their first measurement. Test pins `x[0]==0.0` *and* a non-finite transform.
7. **medium, `bit_reproduced`.** Real. New `hf.timing_join_view` renames it **`reproduced_within_0.001`**; the tolerance is printed above each table and the Sven line says "which is NOT bit-identity".
8. **medium, MNIST confirm passes not GPU-homogeneous.** Real. New `hf.pass_gpus` / `gpu_homogeneity_line`, called from `summary_cell` and the baselines loader. **MNIST-LR 45 MIG / 25 80GB (Sven on 80GB, MuonW & HIG on MIG); MNIST-CE 38/32 (Sven MIG); toy, poly, both CIFAR homogeneous.** MNIST-CE also prints the computed size of the effect: **Sven moves ≤3.8e-2 relative = 0.00431 absolute vs the smallest "significant" paired difference 0.00664** → same order, flagged.
9. **medium, templated divergence prose.** Real. New `hf.divergence_pattern`: **toy 194/900 (rtol ≤1e-4 only, monotone in lr, worst 20/30 at rtol 1e-6 lr 0.5), poly 4/720, MNIST-LR 1/640, MNIST-CE "NO divergence anywhere ... 0 of 800".**
10. **medium, "688 of 7,825".** Real and moving. New `hf.campaign_divergence` over the 22 tuning grids (excludes `_confirm`/`_timing`/`_diag`): **688 of 7,867 (8.7%), 39 recorded, 16 grids affected** — 7,865 an hour earlier, 7,867 now.

## Deviations
* Every new derivation lives in `headline_figs.py`; `headline.py` untouched. `budget.plot_best_of_n`'s raw-key labels are still worked around with `hf.relabel_methods` — **WP1 should fix `budget.py` to use `style.method_label`**.
* `closing_numbers` gained `bon=`, `df=`, `campaign=` (pass `False` to skip the 22-scan load, ~4 s warm).
* Notebook count: headline_tables 28→29 cells (the computed CIFAR-CE cell).

## Open issues
* CIFAR-CE still **PROVISIONAL** (778/785 records, 5 live claims, 38 post-selection). No CIFAR number is hard-coded anywhere in my six notebooks now; after `tools/select_best.py` the banner names the passes to re-run first.
* `analysis/tables/` (141 files) still not gitignored — orchestrator decision.
* ~36 stale pre-rewrite PDFs per `plots_v2/<scan>/` dir (untracked; disk clutter only).

## make_plots.sh (not edited)
Uncomment **`headline_tables`** in `NEW=(...)`; the other five are already in `SCANS`/`STUDIES`.