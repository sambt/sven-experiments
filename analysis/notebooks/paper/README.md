# The paper's figures, editable

Four notebooks, one per figure group, that draw the manuscript's figures with the same
builders `python -m paper_assets` uses — so a change made here is a change to the paper,
not a lookalike.

| notebook | figures |
|---|---|
| `fig_main.ipynb` | the main text: **Fig. 1 = `headline_curves`**, `cost_memory`, `k_sweeps`, `hparam_landscape`, the all-seed appendix versions |
| `fig_reviewer.ipynb` | the reviewer studies: `overparam`, `batchsize`, `kappa`, `knobs`, `budget`, `divergence` |
| `fig_large.ipynb` | CIFAR, Fig. 5, nanoGPT, GPT-2, and the six profile figures |
| `fig_spectra.ipynb` | the singular-value figures: spectra along training, norms, used rank, probe sets |

## The loop

```python
import paper_assets.notebook as pf

pf.figures()                     # every paper figure: group, knobs, what is pinned
pf.info('headline_curves')       # this figure's knobs and their current values

f = pf.make('headline_curves', methods='all')   # draws it, writes NOTHING
f.show(zoom=2)                                  # magnified on screen only
f.ax(0).set_ylabel('Validation loss')           # edit the matplotlib objects
f.fig.set_size_inches(5.5, 2.6)
f.save()                                        # PDF + PNG twin + provenance sidecar
```

`f.axes` is the layout the builder made (a 2-D grid for most figures, a 1-D row for
some), and `f.flat` / `f.ax(i)` are the panels in row-major order whatever the layout —
`f.ax(0)` is the top-left one.

**Pin** a choice to make the paper build agree with the notebook:

```python
pf.pin('headline_curves', methods='all', legend={'loc': 'lower left', 'ncol': 6})
pf.rebuild('headline_curves')      # draw with the pinned options and save
pf.unpin('headline_curves')        # back to the builder's default
```

Pinned options are stored in `analysis/paper_assets/figure_overrides.yaml` and read by
**both** the notebooks and `python -m paper_assets`, so a full rebuild replays your
version instead of reverting it. The file is committed on purpose: it is part of how the
paper looks. You can hand-edit it; a live kernel picks the change up at the next
`pf.make()`.

## Two kinds of option

* **The figure's own knobs** — what it draws (methods, scans, panels), geometry
  (`fraction` = panel width as a fraction of `\linewidth`, `aspect`), line weights,
  legend placement, annotations. `pf.info(<name>)` lists them with defaults.
* **The generic cosmetics**, available on every figure: `figsize`, `xlabel`, `ylabel`,
  `title`, `xlim`, `ylim`, `xscale`, `yscale`, `legend`, `suptitle`, `grid`,
  `tick_labelsize`, `rc`. Each accepts one value for all panels, a list by panel
  position, or a dict keyed by panel index — `ylabel={0: 'Validation loss'}`,
  `legend={'loc': 'lower left', 'ncol': 6, 'fontsize': 5}`, `rc={'lines.linewidth': 1.4}`.

## Size on the page

The figures are drawn at their true size: a 0.32-linewidth panel is 1.76 in wide with
7 pt type, which is why they look small inline and why `show(zoom=...)` magnifies the
screen copy only. To judge the type size, open the PNG twin under
`agent_lab/paper_assets/<group>/` after saving — it is written at the physical size.

## Where the pieces are

* `analysis/paper_assets/<group>.py` — the builders, one `FIGURE_SPECS` entry per figure.
* `analysis/paper_assets/figspec.py` — the registry and the override layer.
* `analysis/paper_assets/notebook.py` — the `pf` API these notebooks use.
* `campaign/FIGURE_API_CONTRACT.md` — the contract a builder has to keep (read this before
  adding a figure).
* `tools/cmp_figures.py` — did a rebuild change a figure, or only its timestamp?

## Adding a figure

Write the builder in the right `paper_assets` module as `draw(ctx, opts) -> (fig, meta)`
with `meta` carrying `provenance` and `axes`, add a `FigureSpec` to that module's
`FIGURE_SPECS`, and regenerate these notebooks with
`cd analysis && ../.venv/bin/python ../agent_lab/paper_nb/build_fig_notebooks.py`. The
new figure then appears in `pf.figures()`, in its group's notebook and in the CLI build.
