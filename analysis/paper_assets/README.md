# paper_assets — the paper's figures, tables and number macros

`cd analysis && ../.venv/bin/python -m paper_assets` rebuilds everything the manuscript uses:
figures to `iclr_manuscript/figures_iclr/<group>/<name>.pdf` (with a PNG twin and a provenance
sidecar), tables to `iclr_manuscript/tables_v2/*.tex`, and number macros to
`iclr_manuscript/numbers_v2.tex`. `--only <module>`, `--figures-only`, `--numbers-only` and
`--dry-run` narrow it. `tools/cmp_figures.py` tells whether a rebuild changed a figure or only
its timestamp.

| module | owns |
|---|---|
| `main.py` | the main-text figures and tables: headline curves, cost/memory, k sweeps, hyper-parameter landscape, paired differences, rank heat-map |
| `reviewer.py` | the reviewer studies on the MLP tasks: over-parameterisation, batch size, κ, micro-batch / parameter fraction, tuning budget, divergence |
| `large.py` | CIFAR-10, Fig. 5, nanoGPT, GPT-2, and the profile figures |
| `spectra.py` | the singular-value figures: spectra along training, norms, used rank, probe sets |
| `campaign.py` | the campaign-level numbers (run counts, GPU-hours) |
| `figspec.py` | the figure registry and the override layer (read `figure_overrides.yaml`) |
| `notebook.py` | the `pf` API `analysis/notebooks/paper/` uses to draw, edit, save and pin |
| `common.py`, `_common.py` | shared conventions: paths, `save_fig`, provenance records, the macro book |

## Conventions

* **Numbers are never typed by hand.** Every campaign-derived number in the prose is a macro
  from `numbers_v2.tex` (`\num<Scan><Method><Quantity>`, documented in the file's header) and
  every results table is `\input` from `tables_v2/`. One command regenerates all of them.
* Figures follow the manuscript's visual language: single-column-width panels used three
  across at `0.32\linewidth`, log-y validation loss vs epoch and vs wall time, Sven black,
  `style.METHOD_COLORS`, `style.method_label` names, seed band = mean ± 1 std over seeds.
  Headline numbers come from the confirmation seeds; wall time from the standalone
  `<scan>_timing` runs. The analysis conventions are `EXPERIMENTS.md` §12.
* Cosmetic choices made in the paper notebooks are pinned in `figure_overrides.yaml` and
  replayed by the CLI, so **the override file is part of how the paper looks** — read it before
  concluding a figure's builder produces what you see.

## The figure contract

Every figure is a `FIGURE_SPECS` entry whose `draw(ctx, opts)` writes nothing, so the same
builder can be driven from the CLI and from a notebook. A module that adds or changes a figure
keeps to the following.

## What your module must expose

```python
FIGURE_SPECS = figspec.check_defaults({
    'headline_curves': figspec.FigureSpec(
        draw=_headline_curves,          # (ctx, opts) -> (fig, meta);  NEVER writes
        doc='F1: validation loss vs epoch and vs standalone train time, three tasks',
        defaults={'methods': 'top5', 'aspect': 0.86, 'panel_legend': True, ...},
    ),
    ...
})

def context(root=None, reload=False):   # the data object your draw functions take
    ...                                 # cached per root; cheap to call twice
```

* **The key is the output stem**: `figures_iclr/<GROUP>/<key>.pdf`. Every figure a module's
  `build()` writes must appear exactly once, under that name.
* **`draw(ctx, opts) -> (fig, meta)` never writes, never closes the figure.** `meta` is a
  dict which MUST carry:
  * `provenance` — the `common.provenance(...)` record the figure is saved with today
    (identical content: same functions, scans, reads, notes, extras). Alternatively set
    `FigureSpec.provenance=lambda ctx, meta: record` if your module computes it outside the
    builder (that is how `large` does it today) — one of the two, never neither.
  * `axes` — what `plt.subplots` returned (the 2-D array, or your own nested list). This is
    what the notebook hands the user to edit and what the generic cosmetics act on.
  * plus anything your `build()` needs downstream, unchanged (the methods drawn per scan,
    the frame that feeds macros, ...). Keep those keys and their meanings.
* **`build()` keeps its signature and its report**, and becomes the only save path: iterate
  `FIGURE_SPECS`, call `figspec.draw_figure(name, ctx=ctx)` (or `spec.draw(ctx, opts)` +
  `figspec.apply_opts`) and save through `common.save_fig`, exactly as now — same order, same
  stdout lines, same report keys, same `dry_run` behaviour, same skip-on-no-data behaviour.
* **Options.** At the top of each builder: `opts` arrives resolved (defaults < persisted
  overrides < call), so just read it: `methods = opts['methods']`. Do not call
  `figure_opts` yourself inside the builder — `figspec.draw_figure` did it.

## Which knobs to expose in `defaults`

Lift the cosmetics that are hardcoded in your builders **and that a person would plausibly
want to change**, at minimum:

* content: which methods / scans / panels are drawn (`methods`, `scans`, `top_n`, and any
  `all_methods`-style flag that currently exists as a function argument);
* geometry: `fraction` (panel width as a fraction of `\linewidth`), `aspect`, `extra_h`,
  `ncol`/`nrow` where the layout is a grid;
* the legend: whether panels get their own, its `loc`, `fontsize`, `ncol`, frame;
* line weights (`lw`, `sven_lw`), marker size, colour-map name, annotation on/off, and the
  clip limits of any `_clip_outliers`-style truncation.

Rules:
* **Defaults must equal today's hardcoded values.** With an empty override file the figure
  must come out byte-identical (modulo the PDF's embedded `/CreationDate`).
* Do **not** name a knob `figsize`, `xlabel`, `ylabel`, `title`, `xlim`, `ylim`, `xscale`,
  `yscale`, `legend`, `suptitle`, `grid`, `tick_labelsize` or `rc` — those are the generic
  cosmetics `figspec.apply_opts` applies to every figure after your builder returns.
  `check_defaults` raises if you do.
* Keep `defaults` flat and JSON/YAML-representable (str, number, bool, list, dict of those).
  A knob whose value is a python object cannot be pinned from a notebook.
* Document each knob in one short phrase in `doc` or as a comment next to `defaults`, in the
  module's existing voice.

## Adding a figure

Write the builder as `draw(ctx, opts) -> (fig, meta)` with `meta` carrying `provenance` and
`axes`, add a `FigureSpec` to the module's `FIGURE_SPECS`, and regenerate the paper notebooks
with `.venv/bin/python tools/build_fig_notebooks.py`. The new figure then appears in
`pf.figures()`, in its group's notebook and in the CLI build. `tests/test_paper_assets_<module>.py`
checks that every figure draws without writing, returns `provenance` and `axes`, and that its
defaults round-trip through `figspec.figure_opts`.
