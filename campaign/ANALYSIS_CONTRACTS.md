# Analysis-phase contracts (2026-09-20) — binding for every analysis agent

The plan is `campaign/ANALYSIS_PLAN.md` (read it fully, incl. section 0 "Known facts" and section 7 "Decisions").
Background: `campaign/CAMPAIGN_STATUS.md`, `campaign/CONTRACTS.md` (record schema 2 and results layout),
`analysis/ANALYSIS_FIXES.md` (analysis conventions already decided), `CHANGES_NEEDED.md` §1 (binding selection rule).

## Ground rules
* Repo `/n/home11/sambt/iaifi/sv3`, branch `robustness-campaign`. **Never commit, stash, reset, checkout or switch
  branches** — the orchestrator commits. Several agents work in this tree at the same time: edit ONLY the files your
  work package owns (listed in your task). Need a change elsewhere? Put new code in your own module; a genuine bug in a
  shared helper may get a MINIMAL fix, which you must list in your report.
* Results are READ-ONLY: `experiment_results/` (fresh, schema 2), `experiment_results_legacy_2026-09-18/` (legacy),
  `profile_results_v2/`. Never write, move or delete there (the loader cache under `experiment_results/_cache/` is the
  one allowed write). Never submit or cancel SLURM jobs unless your task says so.
* Python `/n/home11/sambt/iaifi/sv3/.venv/bin/python`; Jupyter `.venv/bin/jupyter`. This is an 8-core / 32 GB compute
  node shared by ~6 agents: run one notebook / one test file at a time locally; anything long (> ~5 min) or
  memory-hungry goes through `campaign/run_cpu_tests.sh <cmd...>` (sbatch --wait on a CPU partition, prints the log).
  Cold-loading a scan from Lustre takes ~20 s and is cached afterwards. `import torch` is slow (1-2 min cold).
* Notebook edits: edit the `.ipynb` JSON programmatically or with a notebook-aware tool, keep cell order and the
  existing style (`sys.path.insert(0, '.')`, `set_style()`, figures saved under `plots_v2/<name>/`), execute IN PLACE
  with `cd analysis && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace <nb> --ExecutePreprocessor.timeout=3600`
  and confirm 0 error cells. Keep outputs in the notebook.
* Binding analysis conventions: selection = seed-mean final VALIDATION loss under the full rule (eligible -> fewest
  diverged -> seed mean), implemented in `analysis/scan_analysis.py` and `tools/select_best.py` ->
  `bench/best_configs.json` (the selection of record; never `tools/reconcile.py`'s quick table). Test metrics are
  OUTCOMES, never selection inputs. Diverged = failed (status or 10x rule), excluded from means and counted;
  every table shows `finished / attempted`. Seed band = mean +/- 1 std (ddof=1) labelled "+/- 1 std over seeds".
  Headline numbers = CONFIRMATION seeds (`<scan>_confirm`), tuning-seed numbers shown beside them.
  Colours `style.METHOD_COLORS` (Sven black), names `style.DATASET_TITLES`.
* Tests: CPU pytest in `tests/` for every new helper (synthetic fixtures; no dependence on the real results for unit
  tests). Keep the full suite green.
* Final report (your return value, dense markdown): files created/changed; what you verified by running (paste the
  decisive output lines / key numbers); deviations; open issues; notes for other work packages.

## File ownership
| WP | owns |
|---|---|
| WP1 housekeeping | `analysis/style.py`, `analysis/analysis_helpers.py`, `analysis/scan_analysis.py`, `make_plots.sh`, minimal stale-reference fixes in existing notebooks, `tests/test_analysis_core.py` |
| WP2 headline | NEW `analysis/headline.py`, NEW `analysis/headline_tables.ipynb`, NEW `tests/test_headline.py`; phase B: `toy_1d_analysis`, `polynomial_analysis`, `mnist_analysis`, `mnist_analysis_labelRegression`, `baselines_analysis` notebooks |
| WP3 spectra | NEW `analysis/ckpt_tools.py`, NEW `tools/compute_ckpt_spectra.py`, NEW `tests/test_ckpt_tools.py`; phase B: `analysis/sv_diagnostics.py`, `comparisons.ipynb`, NEW `spectra_analysis.ipynb` |
| WP4a reviewer MLP | phase B: `overparam_analysis`, `batchsize_analysis`, `kappa_analysis`, `microbatch_analysis`, `paramfrac_analysis` notebooks, NEW `analysis/reviewer_figs.py` |
| WP4b reviewer large | phase B: `cifar_analysis`, `nanogpt_analysis`, NEW `gpt2_analysis.ipynb`, NEW `analysis/large_figs.py` |
| WP5 docs + diff | `EXPERIMENTS.md`, `README.md`, `analysis/RERUNS_NEEDED.md`, repo `CLAUDE.md` if present; phase B: NEW `analysis/legacy_vs_fresh.ipynb`, NEW `analysis/WHAT_CHANGED.md`, NEW `analysis/legacy_diff.py` |
| GPU items | `campaign/plan_campaign.yaml` (one new list), `experiments/optimizer_profile.py`, `bench/profile_serial.sbatch`, `experiments/configs/profile_*.yaml`, `analysis/profile_helpers.py`, the four `profile_*.ipynb` |
