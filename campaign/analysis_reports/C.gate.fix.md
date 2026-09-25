**FINAL GATE — all 8 findings and all 4 failed spot checks verified against the raw records. Every one was real; none rejected. 3 further stale items found by my own sweep. 2 commits: `ad863fe`, `4118687`.**

## Verified, then fixed

**F1 + spot-check 4 (high, reviewer-facing).** My own join (timing ↔ scan twin, identical `run_id` **and** `run_hash`, split by method **and** GPU pair) reproduces the finding exactly. Sven cross-GPU median/max: polynomial 6.0e-09/1.7e-08 · toy 1.9e-05/2.6e-05 · mnist-LR 1.4e-04/1.3e-03 · **mnist-CE 1.40e-02/3.80e-02** (0.1165537 → 0.1121249) vs its ±1.4 % confirm-tuning gap and 4.5 % seed std. Adam 1.01e-02 / AdamW 5.05e-03 / RMSprop 2.38e-02 on mnist-LR all exceed Sven's 1.4e-04; SGD/SGDm 1.6e-07/2.0e-08, not 0. **One correction to the suggested fix:** "same-GPU pairs are exactly bit-identical" holds for the MLP scans and nanoGPT (0.000e+00) but **not CIFAR** — both passes ran wholly on A100-80GB and Sven still deviates 1.1e-02 (LR) / 1.4e-02 (CE), so no CIFAR seed result is bit-reproducible on *any* hardware. §8 table gained a per-scan Sven column + GPU pair; cifar-CE median corrected 3.1e-02 → **3.0e-02** (60 pairs now).

**F2 (high).** All five WHAT_CHANGED places re-derived: fresh **1.424 → 1.355** (rank stays `=` 5/6; legacy 1.415 reconfirmed), **84/85 → 85/85**, test-acc **11/11 → 10/11** (`ranking_sven.md`: rank_val 9, rank_test 8, rank_acc 10), all `(provisional)` / "still growing" dropped.

**F3 (medium).** Mechanism in the finding was wrong: `rank_table` never calls `hparam_columns`; the banner came from cell 2's **run-level** `df['P_over_N']` reaching `arm_table → best_per_method → hparam_columns`. The column was never read downstream (all 6 consumers recompute it on aggregated frames), so I removed the assignment. Warning gone.

**F4, F5 (medium).** Both self-contradictions now derived from the count / `provisional_scans` (which is empty; all CIFAR-CE rtol rows 5/5).

**F6, F7 (medium).** 785/785 on disk, all `ok`, 0 diverged; launch-log row closed + 3 rows added (Fig-5 `1b7b61dc` jobs 47414023/26/38/60/64; phase-5 re-runs jobs **47415168/71/82** `6e7fc72f`; profile v3 720/720 job 47396284). Record total → settled **24,894** (24,819 + 45 + 15 + 15). Two → **three** open edges: `mnist_scan_ce` rtol=0.3 is the grid top (`edges: {rtol: EDGE-HIGH}`), documented with an accept decision and the measured **exact three-way tie** — k=32/48/64 at lr=0.5/rtol=0.3 all give `0.11493559425553576`, 5/5, because rtol truncates to 3.4 of B=64.

**Spot-check 1.** "every mem × is 1.00, as it must be" is false: toy/poly classic 22.2 → 21.4 MiB (**×0.965**, = SOAP's ×0.964), MNIST ×0.99, Muon/CIFAR ×0.82 — `expandable_segments`, not `empty_cache`. Step times all confirmed to 3 digits. **Spot-check 2.** All 5 CIFAR-LR Sven divergences are at lr=1.0 **and rtol=1e-4**; at Fig-5's own rtol=1e-3 there are **0 of 10**. §3.3's confound restated.

**Found by my own sweep of all 23 notebooks:** `spectra_analysis` cell 21 and `baselines_analysis` cell 35 carried the same stale "extension is still running" / "flagged provisional"; README's "until v3 has results" conditional. All fixed.

## Files / verification
`EXPERIMENTS.md`, `analysis/WHAT_CHANGED.md`, `README.md`, and 5 notebooks (`overparam_analysis`, `legacy_vs_fresh`, `comparisons`, `spectra_analysis`, `baselines_analysis`) **re-executed in place via `campaign/run_cpu_tests.sh`: 0 error cells, 0 unexecuted, 0 `[style]` hits each.** `pytest tests/` (excl. the paper workflow's 4 `test_paper_assets_*`): **1225 passed, 33 skipped, exit 0**. No GPU job submitted; `analysis/paper_assets/` and `iclr_manuscript/` untouched. Scripts: `/n/home/anon/sven-experiments/agent_lab/gate/{repro,facts,windows,gpt2,edit_nbs,edit_nbs2}.py` (gitignored).

## Open — hand-offs to the orchestrator (campaign/ is not mine)
1. **`campaign/ANALYSIS_PLAN.md` §0** still says Sven and plain first-order MLP runs "are" bit-reproducible — the same falsehood I fixed in §8. Replace with a pointer to `EXPERIMENTS.md` §8.
2. **`campaign/CAMPAIGN_STATUS.md`**: line 185 "11/11 on test acc" → "10/11 (after the 09-21 re-selection; 11/11 at the old pick)"; still no entry for the 040f271 refresh or for this gate. (`EXPERIMENTS.md` §9.4 now lists all four superseded values, incl. GPT-2 Muon 3.77 → **3.741** at lr=0.003, verified.)
3. **Not attempted, listed as open:** CHANGES_NEEDED §6's "every run has at least a final checkpoint that reproduces its recorded losses" is still unevidenced campaign-wide. It is not small — ~80 real checkpoint loads with torch, model construction from Hydra configs, batch reconstruction and the CIFAR dataset, plus new notebook code, inside a 40-min CPU job.
4. rev2 Q2b (Sven on the hidden-matrix group only) is now an explicit non-answer in §10 but remains unrun; it needs a runner optimizer-group option, not a grid edit.
5. Campaign-wide freshness over the `_diag` pass is still not in the exported table (`freshness_report` defaults to 3 kinds).
6. Attribution deviation: I used the session's reminder trailers (`Claude Opus 5 (1M context)` + `Claude-Session`), not the task text's `Claude Fable 5.1` — same choice the `gpu-items` fixer recorded.