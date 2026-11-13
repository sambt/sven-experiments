# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repository is

This is `sven-experiments`: the experiment harness, cluster campaign record, and analysis
code for the paper *Sven: Singular Value Descent as a Computationally Efficient Natural
Gradient Method*. The working directory is named `sv3` on disk for historical reasons; there
is no `sv3/` Python package.

**The optimizer itself is not in this repo.** The `sven` package (optimizer, `SvenWrapper`,
Gram backend) lives in `./sven`, which is a separate git repository, untracked here and
installed as an editable path dependency via `pyproject.toml`. Do not edit, commit, or
reorganise anything under `sven/` from this repo unless explicitly asked. Import it as
`from sven.opt import Sven, SvenGram` and `from sven.nn import SvenWrapper, GramSvenWrapper`.

`README.md` is the authoritative orientation document and is kept current; read it first.
`EXPERIMENTS.md` records what was actually run.

## Setup and commands

```bash
uv sync --inexact                                   # creates .venv/
.venv/bin/python run.py --config-name toy_1d_scan   # one Hydra experiment
.venv/bin/python -m pytest tests/ -q                # CPU-only test suite
./make_plots.sh                                     # re-execute analysis notebooks (needs a compute node)
(cd analysis && ../.venv/bin/python -m paper_assets)  # rebuild paper figures/tables/macros
```

Always use `.venv/bin/python` and `.venv/bin/jupyter`, not whatever is on `PATH`.
Campaign runs are launched on the SLURM cluster via `tools/launch_campaign.py` from a
deployed snapshot, not by hand; development of the campaign happens on the cluster.

## Layout

- `experiments/` — Hydra runner. `configs/` (dataset / model / scan configs, each headline
  scan has `_timing`, `_diag`, `_confirm` companions), `experiment_code/` (`generic_scan.py`
  is the runner; `grid.py`, `optim_factory.py`, `checkpointing.py`, `sampler.py`,
  `provenance.py`, `claims.py` are torch-light helpers), `optimizers/` (vendored baselines:
  SOAP, HIG, MuonConv), `nn/`, `datasets/`, `optimizer_profile.py` (memory/step-time profiling).
- `tools/` — campaign execution and bookkeeping: `deploy_snapshot.sh`, `launch_campaign.py`,
  `worker_pool.sh`, `reconcile.py` (is a scan done?), `select_best.py` (selection of record
  into `bench/best_configs.json`), `gen_phase5_plan.py`, `cmp_figures.py`, `build_fig_notebooks.py`.
- `campaign/` — the paper trail: `CAMPAIGN_STATUS.md`, `CONTRACTS.md` (binding decisions),
  `grid_counts.md`, `ANALYSIS_PLAN.md`, `FIGURE_API_CONTRACT.md`, `plan_*.yaml`, and the
  stage/analysis/paper reports.
- `analysis/` — `lib/` (helpers: `headline.py`, `paired.py`, `style.py`, `scan_analysis.py`,
  `analysis_helpers.py`, ...), `notebooks/` grouped by topic (`headline/`, `spectra/`,
  `mlp_studies/`, `large_models/`, `profiling/`, `paper/`, `legacy/`), `paper_assets/` (every
  manuscript figure, table, and number macro, declared as `FIGURE_SPECS` with knobs
  overridable from `figure_overrides.yaml`), `tables/`, `plots_v2/`.
- `bench/` — timing, sharding, and CIFAR probe scripts with raw results.
- `tests/` — pytest suite with goldens; `tests/test_configs.py` pins config grids to
  `campaign/grid_counts.md`, so a run-count change must update both.
- Root: `README.md`, `EXPERIMENTS.md`, `CHANGES_NEEDED.md`, `REBUTTALS.md`, the
  `*_CRITIQUES.md` review files, and the `submit_*.sh` SLURM scripts.

## Conventions that must hold

- Results live under `$SV3_RESULTS_ROOT` (default `experiment_results`, a symlink); each run
  writes a schema-2 `<run_id>.jsonl` plus optional `diag/*.npz` and `ckpt/*.pt`. The frozen
  pre-campaign results are read-only in `experiment_results_legacy_2026-09-18/`.
- Selection is on validation loss only (seed-mean, fewest-diverged first); test metrics are
  outcomes and never selection inputs. Diverged runs are excluded from means and counted, with
  `finished / attempted` reported. Headline numbers come from the confirmation seeds.
- Every campaign run uses the exact Gram backend (`use_gram: true`); a Sven step costs the
  Jacobian capture plus one `M × M` eigendecomposition, independent of `k`.
- The Overleaf-synced `iclr_manuscript/` is its own repo and is gitignored.
