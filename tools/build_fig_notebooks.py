"""Generate analysis/notebooks/paper/fig_<group>.ipynb from the live figure registry.

Run it after adding or changing a figure in a `paper_assets` module:

    .venv/bin/python tools/build_fig_notebooks.py

One notebook per figure group, one section per figure, with that figure's REAL knobs and
docstring in the cells -- which is why this reads the registry instead of hardcoding a
table.  Executing a generated notebook draws every figure of the group and writes
nothing: the `save()` / `pin()` lines are there, commented, for the person editing.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]                      # tools/ -> the repo root
OUT = REPO / 'analysis' / 'notebooks' / 'paper'

sys.path.insert(0, str(REPO / 'analysis' / 'lib'))
sys.path.insert(0, str(REPO / 'analysis'))
os.environ.setdefault('PAPER_ASSETS_KEEP_BACKEND', '1')

from paper_assets import figspec as F      # noqa: E402

BOOTSTRAP = """# This notebook lives in analysis/notebooks/paper/, but every analysis path is
# relative to analysis/ (plots_v2/, tables/, ../experiment_results), so run from there
# with the helper modules in analysis/lib/ on the path.
_A = os.getcwd()
while not all(os.path.isdir(os.path.join(_A, d)) for d in ('lib', 'notebooks')):
    _A, _prev = os.path.dirname(_A), _A
    if _A == _prev:
        raise RuntimeError('analysis/ not found above ' + os.getcwd())
os.chdir(_A)
sys.path.insert(0, _A)
sys.path.insert(0, os.path.join(_A, 'lib'))
"""

GROUP_INTRO = {
    'main': """# Fig. 1 and the other main-text figures

The figures the main text carries: the headline convergence panels (**Fig. 1 =
`headline_curves`**), the cost/memory scaling figure, Sven's `k` sweeps and the
hyperparameter landscape, plus the all-seed appendix versions.

`headline_curves` is the one to reach for first: `methods` chooses the field it draws
(the front of the field, or every baseline), and the legend, labels and aspect ratio are
ordinary knobs.""",
    'reviewer': """# The reviewer-study figures (appendix)

One figure per study the NeurIPS reviews asked for: dataset-level over-parameterisation
(*P > N*), batch-size sensitivity, the residual exponent kappa, the memory knobs
(micro-batching and parameter fraction), the tuning-budget disclosure and the divergence
accounting.""",
    'large': """# CIFAR, nanoGPT, GPT-2 and the profile figures

The large-model figures: CIFAR-10/ResNet18 (curves, cost, the optimisation-vs-
generalisation gap, the Sven landscape, the rank actually used), Fig. 5's parameter
fraction at scale, nanoGPT, GPT-2-small, and the six figures of the `profile_results_v3`
study.""",
    'spectra': """# The singular-value figures

What the truncation keeps and what it discards: the spectra along training, the
update/residual norms, the used rank, and the probe-set spectra and metrics for every
scan.""",
}

HOWTO = """## How this works

```python
f = pf.make('<figure>')          # draws it, writes NOTHING
f.show(zoom=2)                   # magnified screen copy (the PDF keeps its page size)
f.ax(0).set_ylabel('...')        # edit the matplotlib objects however you like
f.fig.set_size_inches(5.5, 2.4)  # f.axes = the builder's layout, f.flat = every panel
f.save()                         # writes iclr_manuscript/figures_iclr/<group>/<figure>.pdf,
                                 # the PNG twin and the provenance sidecar -- exactly as
                                 # `python -m paper_assets` does
```

Two kinds of option:

* **the figure's own knobs** — what it draws (methods, scans, panels), its geometry
  (`fraction`, `aspect`), line weights, legend placement. `pf.info('<figure>')` lists them
  with their defaults.
* **the generic cosmetics**, on every figure: `figsize`, `xlabel`, `ylabel`, `title`,
  `xlim`, `ylim`, `xscale`, `yscale`, `legend`, `suptitle`, `grid`, `tick_labelsize`,
  `rc`. Each takes a value for all panels, a list by panel position, or a dict keyed by
  panel index — e.g. `ylabel={0: 'Validation loss'}`, `legend={'loc': 'lower left',
  'ncol': 6, 'fontsize': 5}`.

Pass either to `pf.make(...)` for a one-off, or **pin** it so the paper build draws it
that way too:

```python
pf.pin('headline_curves', methods='all', aspect=0.95)   # -> paper_assets/figure_overrides.yaml
pf.rebuild('headline_curves')                           # draw with the pinned options + save
pf.unpin('headline_curves')                             # back to the builder's default
```

Pinned options live in `analysis/paper_assets/figure_overrides.yaml`, which
`python -m paper_assets` reads, so a full rebuild reproduces your version instead of
reverting it. Nothing in this notebook writes until you call `save()` or `rebuild()`, so
re-running it top to bottom is safe.

**A note on size.** These are drawn at their true size on the page (a 0.32-linewidth
panel is 1.76 in wide, 7 pt type), so they look small inline; `show(zoom=...)` only
magnifies the screen copy. To judge the type size, look at the PNG twin under
`agent_lab/paper_assets/<group>/` after saving — it is written at the physical size."""


#: extra cells for particular figures, appended after the standard pair.  Keyed by figure
#: name; each entry is a list of (kind, text) with kind in {'md', 'code'}.
EXTRAS = {
    'headline_curves': [
        ('md', """### Fig. 1 with every baseline

The committed default draws the front of the field plus Sven (`methods='top'`,
`top_n=5`), because five curves per 1.76-inch panel is what stays readable; the
all-methods version of the same figure is the separate appendix figure
`headline_curves_all`.

The cell below draws **Fig. 1 itself with every baseline**, which is what you want if the
main-text figure should carry the whole field. Nothing is written until you `save()`; to
make `python -m paper_assets` draw it that way as well, pin it."""),
        ('code', """# `panel_legend=False` drops the per-panel key (15 entries do not fit in a 1.76 in
# panel) and `figure_legend_methods=True` puts ONE method key under the figure instead --
# without the second flag the figure would have no method key at all.  ncol=6 is what
# fits 16 entries across 5.28 in; ncol=8 clips them at both edges.
f_all = pf.make('headline_curves', methods='all',
                panel_legend=False, figure_legend_methods=True, figure_legend_ncol=6,
                aspect=0.9, extra_h=0.85)
f_all.show(zoom=2.5)"""),
        ('code', """# keep it: pins `methods` (and the legend that makes room for 15 curves) for the
# notebook AND the CLI, then writes the paper PDF.
# pf.pin('headline_curves', methods='all', panel_legend=False,
#        figure_legend_methods=True, figure_legend_ncol=6, aspect=0.9, extra_h=0.85)
# pf.rebuild('headline_curves')
#
# and to go back to the committed version:
# pf.unpin('headline_curves'); pf.rebuild('headline_curves')"""),
    ],
}

#: nbformat >= 4.5 wants a stable id per cell (it warns on every execute without one).
#: Deterministic, so regenerating a notebook does not churn every id.
_IDS = {'n': 0}


def _cell_id():
    _IDS['n'] += 1
    return f'cell-{_IDS["n"]:03d}'


def md(text):
    return {'cell_type': 'markdown', 'id': _cell_id(), 'metadata': {},
            'source': [l + '\n' for l in text.split('\n')][:-1] + [text.split('\n')[-1]]}


def code(text):
    lines = text.split('\n')
    src = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    return {'cell_type': 'code', 'id': _cell_id(), 'execution_count': None,
            'metadata': {}, 'outputs': [], 'source': src}


def _example(name, spec):
    """A commented edit-and-save cell using this figure's own knob names."""
    knobs = sorted(spec.defaults)
    shown = ', '.join(f'{k}={spec.defaults[k]!r}' for k in knobs[:3])
    lines = [f"# --- edit {name} -------------------------------------------------------",
             f"# its own knobs: {', '.join(knobs) if knobs else '(none)'}"]
    if shown:
        lines.append(f"# f = pf.make('{name}', {shown})")
    lines += [f"# f = pf.make('{name}', aspect=0.95, legend={{'loc': 'lower left', 'ncol': 6}},",
              f"#             ylabel={{0: 'Validation loss'}})",
              "# f.ax(0).set_xlabel('Epoch')               # ... or edit the objects directly",
              "# f.show(zoom=2)",
              "# f.save()                                   # write the paper PDF",
              f"# pf.pin('{name}', aspect=0.95)              # ... and make the CLI agree"]
    return code('\n'.join(lines))


def notebook_for(group, specs):
    cells = [md(GROUP_INTRO.get(group, f'# The {group} figures')), md(HOWTO),
             code("%matplotlib inline\nimport os\nimport sys\n\n" + BOOTSTRAP
                  + "\nimport paper_assets.notebook as pf\n\n"
                  f"pf.figures({group!r})")]
    for name, spec in specs:
        doc = (spec.doc or '').strip()
        knobs = '\n'.join(f'* `{k}` — default `{spec.defaults[k]!r}`'
                          for k in sorted(spec.defaults))
        cells.append(md(f'## `{name}`\n\n{doc}\n\n'
                        + (f'Its own knobs:\n\n{knobs}\n' if knobs else '')))
        cells.append(code(f"f = pf.make('{name}')\nf.show(zoom=2)"))
        cells.append(_example(name, spec))
        for kind, text in EXTRAS.get(name, ()):
            cells.append(md(text) if kind == 'md' else code(text))
    cells.append(md('## What is pinned\n\nEverything this notebook has pinned, i.e. what '
                    '`python -m paper_assets` will replay. `pf.unpin(<name>)` drops one, '
                    '`pf.unpin()` drops them all.'))
    cells.append(code('pf.pinned()'))
    return {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                       'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'}},
        'nbformat': 4, 'nbformat_minor': 5}


def main():
    reg = F.registry(reload=True)
    if not reg:
        raise SystemExit('no FIGURE_SPECS declared yet -- nothing to generate')
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for group in sorted({s.group for s in reg.values()}):
        specs = sorted(((n, s) for n, s in reg.items() if s.group == group))
        _IDS['n'] = 0
        path = OUT / f'fig_{group}.ipynb'
        path.write_text(json.dumps(notebook_for(group, specs), indent=1,
                                   ensure_ascii=False) + '\n')
        written.append((path, len(specs)))
        print(f'{path.relative_to(REPO)}: {len(specs)} figure(s)')
    return written


if __name__ == '__main__':
    main()
