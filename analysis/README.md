# analysis/ — layout

Reorganised 2026-09-21: the notebooks are grouped under `notebooks/`, the shared
helper modules live in `lib/`.  Nothing about the analysis itself changed — the
conventions are still `ANALYSIS_FIXES.md` and `campaign/ANALYSIS_CONTRACTS.md`, and
the outputs are still written to the same places.

```
analysis/
  notebooks/        the 23 notebooks, grouped -- see notebooks/README.md for the index
    headline/       the seven headline scans + the paper's tables
    spectra/        singular values: what the truncation keeps and discards
    mlp_studies/    the reviewer studies on the MLP tasks (P>N, batch size, kappa, ...)
    large_models/   CIFAR-10/ResNet18, nanoGPT, GPT-2
    profiling/      memory and step time (profile_results_v3/)
    legacy/         legacy vs fresh: what the robustness fixes changed
  lib/              the shared helper modules the notebooks import
  paper_assets/     the paper's figures/tables/number macros (python -m paper_assets)
  plots_v2/         figures the notebooks write (one directory per notebook)
  tables/           markdown/LaTeX tables the notebooks export
  ckpt_spectra/     cached checkpoint Jacobian spectra (regenerable, gitignored)
  legacy_repair/    offline example-weighted curves for the legacy results
  plots/            the pre-campaign figures, kept for reference
```

## Running

```bash
./make_plots.sh                    # re-execute every notebook in place
./make_plots.sh kappa_analysis     # one, by name (no group, no .ipynb)
./make_plots.sh mlp_studies        # one whole group
cd analysis && ../.venv/bin/python -m paper_assets    # the paper's assets
```

Paths are relative to `analysis/`, not to the notebook: the first cell of every
notebook walks up to this directory, chdir's here and puts `lib/` on `sys.path`.  So
`../experiment_results` (the results root, overridable with `$SV3_RESULTS_ROOT`),
`plots_v2/<name>/` and `tables/` mean the same thing from every notebook, and
`import style` works in Jupyter, under `nbconvert`, and from `tests/`.

## lib/

| module | what it does |
|---|---|
| `style.py` | results root, `load_results`, method colours and names, the shared plot style |
| `scan_analysis.py` | the per-scan analysis: grids, the selection rule, curves |
| `analysis_helpers.py` | shared helpers for the study notebooks (completeness, config columns) |
| `headline.py` | the paper's headline tables for the seven scans |
| `headline_figs.py` | figures and tables for the headline notebooks |
| `reviewer_figs.py` | the reviewer-facing MLP studies, in one place |
| `large_figs.py` | CIFAR / nanoGPT / GPT-2 figures and tables |
| `spectra_figs.py` | the mechanism figures: what the truncation keeps and throws away |
| `sv_diagnostics.py` | singular-value diagnostics from the `_diag` passes |
| `ckpt_tools.py` | offline checkpoint tools: reload a trained run and take its Jacobian |
| `paired.py` | paired differences between two configurations, matched by model seed |
| `budget.py` | tuning-budget disclosure from the existing scans |
| `profile_helpers.py` | loading and plotting for the memory / step-time profile |
| `legacy_diff.py` | legacy results against the fresh campaign |
| `repair_legacy.py` | example-weighted validation curves for the legacy results, offline |

`tests/` imports these the same way the notebooks do (`analysis/lib` on `sys.path`);
the CPU suite for them is `tests/test_analysis_core.py`, `test_headline*.py`,
`test_reviewer_figs.py`, `test_large_figs.py`, `test_spectra_figs.py`,
`test_ckpt_tools.py`, `test_profile_helpers.py`, `test_legacy_diff.py` and
`test_paper_assets_*.py`.

## Documents here

* `ANALYSIS_FIXES.md` — the analysis conventions and the decisions behind them.
* `WHAT_CHANGED.md` — what the robustness campaign changed relative to the legacy results.
* `RERUNS_NEEDED.md` — the gaps: what is still legacy-only or missing data.
