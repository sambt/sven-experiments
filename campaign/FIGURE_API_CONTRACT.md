# Figure API contract (2026-09-21) — binding for every `paper_assets` figure module

## Why

The user wants to **make and edit the paper's figures in notebooks**: "the plot that goes into
fig 1 of the paper I want to have all the baselines on it and change some things about legend
placement, axis labels, aspect ratios, etc. Easier to do myself than describe to you."

Today a builder draws *and* writes in one call, so the only way to change a legend position is
to edit the builder and rebuild the group. The fix is structural and already written:
`analysis/paper_assets/figspec.py` (registry + override layer, **do not edit**) and
`analysis/paper_assets/notebook.py` (the notebook API, **do not edit**). Your job is to make
your module implement the contract below, **without changing a single pixel of the committed
figures** while the override file is empty.

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

* **The key is the output stem**: `figures_iclr/<GROUP>/<key>.pdf`. Every figure your
  `build()` writes today must appear exactly once, under the name it writes today.
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

## Ground rules

* Repo `/n/home/anon/sven-experiments`, branch `robustness-campaign`. Python `.venv/bin/python`.
  **Never** `git commit`/`add`/`stash`/`checkout`/`switch` — the orchestrator commits.
* **You own exactly two files**: `analysis/paper_assets/<your module>.py` and
  `tests/test_paper_assets_<your module>.py`. Do not edit `common.py`, `figspec.py`,
  `notebook.py`, another module, the notebooks, or anything under `analysis/lib/` (a genuine
  bug there: report it, do not fix it).
* Results roots and `profile_results_v*/` are READ-ONLY. `iclr_manuscript/` is an
  Overleaf-synced git repo: writing the figures it holds is what the build does, but **never**
  run git in it, and never touch its `tables_v2/`, `sections_v2/` or `.tex` files.
* Heavy work (a group rebuild, a notebook, a long test) goes through
  `campaign/run_cpu_tests.sh <cmd...>` (sbatch --wait, prints the log). This node is shared.

## How you prove it (required in your report)

1. `campaign/run_cpu_tests.sh .venv/bin/python -m pytest tests/test_paper_assets_<mod>.py -q`
   green, plus the new tests you added: that every figure in `FIGURE_SPECS` draws without
   writing, that `draw` returns `provenance` and `axes`, that the defaults round-trip through
   `figspec.figure_opts`, and that at least one knob visibly changes the figure (e.g. a
   different method count in `meta`).
2. **The byte-identity check.** A pre-snapshot of your group's PDFs is at
   `/tmp/claude-66176/-n-home-anon-sven-experiments/30b30b16-e05d-4e46-8a25-371ea6181950/scratchpad/figs_pre/<group>/`.
   Rebuild your group's figures only —
   `campaign/run_cpu_tests.sh bash -c 'cd analysis && ../.venv/bin/python -m paper_assets --only <mod> --figures-only'`
   — then compare with `.venv/bin/python tools/cmp_figures.py <group>`, which reports a
   difference only when the content differs (it scrubs `/CreationDate`). **Every figure must
   come out identical**, and the provenance sidecars must be unchanged too (the tool checks
   them). Paste the tool's summary line.
3. Say explicitly which knobs you exposed per figure, and anything you deliberately left
   hardcoded because exposing it would have changed the output.
