"""F2, F9, T8, T9 and macro group G4: what Sven's singular values do.

The singular-value story of the revision, in one re-runnable module.  Two measurements
carry it and they are DIFFERENT objects -- :mod:`spectra_figs` documents the distinction
at length and every asset here states which one it shows:

* **online, per batch** (``<scan>_diag/diag/*.npz``): all ``B`` singular values of that
  step's batch Jacobian, logged by the optimizer before the ``k`` / ``rtol`` cut, in
  float32 with a ``sqrt(eps)*sigma_max`` Gram round-off floor.  This is the measurement
  the mechanism claims (C9, C10, C13) are made on.
* **offline, on a fixed float64 probe set** (``analysis/ckpt_spectra/``): the same rows
  for every optimizer, seed and checkpoint, so trajectories are comparable and the rank
  axis runs to ``min(n_rows, P)`` rather than to ``B``.  This is where the truncation is
  visible (C11) and where the min-norm question is answered (C12).

Nothing here computes a statistic: every number comes from :mod:`spectra_figs`,
:mod:`sv_diagnostics`, :mod:`ckpt_tools` or :mod:`headline`, and this module lays it out
at print size, formats it and records where it came from.  Run it with::

    cd analysis && ../.venv/bin/python -m paper_assets.spectra

Every figure is declared in :data:`FIGURE_SPECS` with its own knobs and drawn by a
function that writes NOTHING (``campaign/FIGURE_API_CONTRACT.md``), so the same builder
serves ``python -m paper_assets`` and a notebook::

    import paper_assets.notebook as pf
    f = pf.make('spectrum_truncation', scans=['polynomial_scan'])   # draws, saves nothing
    f.axes[1].set_ylabel('Discarded')                               # ... then edit it
    f.save()

:func:`build` is the only save path; it iterates :data:`FIGURE_SPECS` and hands each
figure to :func:`common.save_fig`.

Outputs (``campaign/PAPER_PLAN.md`` section 5):

===========================  ===========================================================
``spectrum_truncation.pdf``  F2, main section 4.2: 1x3 at ``0.32\\linewidth``
``online_spectra.pdf``       F9, App. O: full-width spectra over training, 7 scans
``online_utr.pdf``           F9, App. O: ``|u_i.r|`` profiles, 7 scans
``online_mechanism.pdf``     F9, App. O: kept / discarded energy, used rank, seed spread
``online_norms.pdf``         F9, App. O: applied update and residual norms, 7 scans
``mnist_lr_vs_ce.pdf``       F9, App. I: label regression vs cross-entropy, one model
``used_rank_grid.pdf``       F9, App. O: which cut binds over the ``(k, rtol)`` grid
``probe_spectra.pdf``        F9, App. O: probe spectra along 4 optimizers (polynomial)
``probe_spectra_all.pdf``    F9, App. O: the same for all 4 cached scans
``probe_metrics.pdf``        F9, App. O: effective rank, condition number, sigma_B/sigma_1
``probe_metrics_all.pdf``    F9, App. O: the same over all 4 cached scans
``probe_norms.pdf``          F9, App. O: distance from init and parameter norm (C12)
``probe_energy.pdf``         F9, App. O: cumulative PROBE residual vs directions kept
``spectra_mechanism.tex``    T8 (full) + ``spectra_mechanism_main.tex`` (main text)
``spectra_probe_energy.tex`` T8: probe top-``k`` energy at several ``k``
``spectra_low4.tex``         T9: dist-from-init / parameter norm, paired, with verdicts
``spectra_probe_widths.tex`` T9: the probe sets' dimensions
``numbers_v2_spectra.tex``   G4
===========================  ===========================================================
"""
from __future__ import annotations

import contextlib
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as c
from . import figspec

import ckpt_tools as ct          # noqa: E402  (common puts analysis/ on sys.path)
import headline as hl            # noqa: E402
import spectra_figs as sf        # noqa: E402
import style                     # noqa: E402
import sv_diagnostics as sv      # noqa: E402

MODULE = 'spectra'
#: figures of this module live in ``iclr_manuscript/figures_iclr/spectra/``
GROUP = 'spectra'

#: scan -> the letters a macro name uses (``\num<Scan>Sven<Quantity>``; TeX forbids
#: digits in a command name, so every key is letters only)
SCAN_KEY = {
    'toy_1d_scan': 'Toy',
    'polynomial_scan': 'Poly',
    'mnist_scan_labelRegression': 'MnistLR',
    'mnist_scan_ce': 'MnistCE',
    'cifar10_resnet_scan_labelRegression': 'CifarLR',
    'cifar10_resnet_ce_scan': 'CifarCE',
    'exp_nanogpt_speedrun': 'Nanogpt',
}

#: short names for a 7-entry legend on a 1.76 in panel (``style.DATASET_TITLES`` are the
#: full names and are used in every table and panel title)
SCAN_SHORT = {
    'toy_1d_scan': 'Toy 1D',
    'polynomial_scan': 'Polynomial',
    'mnist_scan_labelRegression': 'MNIST-LR',
    'mnist_scan_ce': 'MNIST-CE',
    'cifar10_resnet_scan_labelRegression': 'CIFAR-LR',
    'cifar10_resnet_ce_scan': 'CIFAR-CE',
    'exp_nanogpt_speedrun': 'nanoGPT',
}

#: One colour per SCAN, used by every figure in this module.  Deliberately not
#: :data:`style.METHOD_COLORS`: those encode the optimizer, and a panel that draws seven
#: scans of ONE optimizer (Sven) must not reuse another method's hue.  Black is left to
#: Sven in the probe figures, where the lines really are methods.
SCAN_COLORS = {
    'toy_1d_scan': '#4C72B0',
    'polynomial_scan': '#C44E52',
    'mnist_scan_labelRegression': '#DD8452',
    # a dark neutral rather than the seaborn brown that pairs with the orange above: the
    # two MNIST scans are the ONE comparison the revision draws as two lines in the same
    # panel (same model, same B, same splits, different loss), and orange against brown
    # at 0.32 linewidth is not a distinction a reader can make.  Nothing here is Sven's
    # black -- the probe figures are the ones whose lines are methods.
    'mnist_scan_ce': '#4A4A4A',
    'cifar10_resnet_scan_labelRegression': '#55A868',
    'cifar10_resnet_ce_scan': '#8172B3',
    'exp_nanogpt_speedrun': '#64B5CD',
}
#: line styles, so two scans that both sit exactly on 1.0 (MNIST-LR and nanoGPT keep
#: 100 % of the batch residual at every step) stay distinguishable
SCAN_LS = {
    'toy_1d_scan': '-', 'polynomial_scan': '-',
    'mnist_scan_labelRegression': '--', 'mnist_scan_ce': '-',
    'cifar10_resnet_scan_labelRegression': '-.', 'cifar10_resnet_ce_scan': '-',
    'exp_nanogpt_speedrun': (0, (1, 1.2)),
}

#: the scan whose full-width spectrum evolution is F2's panel (a): the only headline scan
#: where ``k < B`` on an MLP, i.e. the only one whose figure shows a truncation at all
F2_SPECTRUM_SCAN = 'polynomial_scan'
#: the four scans F2's energy panel used to draw (the main text's three headline scans
#: plus MNIST-CE).  It now draws ALL of them: on a linear "kept" axis five of the seven
#: superimposed at 1.0 and CIFAR-CE -- the one scan where the kept energy falls
#: substantially, and the caveat Sec. 4.2 points at this panel for -- was not in the
#: figure at all.  Panel (b) is now the DISCARDED fraction on a log axis, where every
#: scan's movement is visible and the pinned ones sit legibly at the bottom.
F2_ENERGY_SCANS = ('polynomial_scan', 'mnist_scan_labelRegression', 'mnist_scan_ce',
                   'exp_nanogpt_speedrun')
#: the two scans the 2x3 probe-metric figure shows in full (the rest are in the 4x4)
PROBE_DEEP_SCANS = ('polynomial_scan', 'mnist_scan_ce')
#: the probe-metric columns App. O reports, in panel order
PROBE_METRICS = ('eff_rank', 'cond', 'sigma_B_over_1')
#: floor for the "energy DISCARDED" panel: below this the cut throws away nothing that
#: float32 can see, and a log axis needs a bottom
DISCARD_FLOOR = 1e-8
#: a kept-energy trajectory within this of 1.0 at the end of training is reported as
#: "pinned at 100%": what the cut discards there is below what the table prints and, on
#: every such scan, below 1e-4 of the residual.  The classification the prose quotes
#: (how many scans the cut bites on, and which way the kept energy moves there) is
#: derived from this threshold rather than asserted.
ENERGY_PINNED_TOL = 1e-4
#: the moving average of every online curve, in OPTIMIZER STEPS (``sf.smooth_steps``);
#: the dense spectra schedule is not uniform, so a sample window would smooth the late
#: part of a curve 20x harder than the early part
SMOOTH_STEPS = 60
#: fractions of training the discarded-energy table reports
PROGRESS_MARKS = (0.01, 0.1, 0.25, 0.5, 1.0)


def _name_list(names):
    """``'A'`` / ``'A and B'`` / ``'A, B and C'`` -- an English list for a prose macro."""
    items = [c.latex_escape(str(n)) for n in names]
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    return ', '.join(items[:-1]) + ' and ' + items[-1]


def _sci(value, sig=2, dashes='--'):
    """``0.000345`` -> ``$3.5\\times 10^{-4}$``.

    ``common.fmt_sig`` keeps a decimal spelling down to 1e-4, which is right for a loss
    and wrong for a round-off floor: ``0.00035`` in a table column reads as a rounding
    artefact rather than as sqrt(eps).
    """
    if not c.finite(value):
        return c.Raw(dashes)
    v = float(value)
    if v == 0:
        return '0'
    exp = math.floor(math.log10(abs(v)))
    mant = v / 10 ** exp
    return c.Raw(f'${mant:.{max(sig - 1, 0)}f}\\times 10^{{{exp}}}$')


def _thousands(value, dashes='--'):
    """``10000`` -> ``10{,}000`` -- a five-digit count in a table or a sentence."""
    if not c.finite(value):
        return c.Raw(dashes)
    return c.Raw(f'{int(round(float(value))):,}'.replace(',', '{,}'))


def _rank_cell(value, dashes='--'):
    """A seed-mean rank at one decimal, so 64.0 and 63.6 line up in a column."""
    if not c.finite(value):
        return c.Raw(dashes)
    return c.Raw(f'{float(value):.1f}')


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass
class Data:
    """Everything the module reads, loaded once.

    ``diags`` / ``arrays`` are the online side (:func:`spectra_figs.sven_diag`,
    :func:`spectra_figs.diag_arrays`), ``probe`` / ``low4`` / ``probe_energy`` the offline
    side.  ``provisional`` is the set of scans whose selection can still move
    (``headline.freshness_report``, plus any scan whose ``_diag`` pass ran a configuration
    the selection no longer names): every asset that touches one is stamped.
    """
    root: str | None = None
    diags: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)
    seed_target: dict = field(default_factory=dict)
    mech: pd.DataFrame = field(default_factory=pd.DataFrame)
    grids: dict = field(default_factory=dict)
    probe_methods: dict = field(default_factory=dict)
    probe: dict = field(default_factory=dict)
    widths: pd.DataFrame = field(default_factory=pd.DataFrame)
    low4: pd.DataFrame = field(default_factory=pd.DataFrame)
    probe_energy: pd.DataFrame = field(default_factory=pd.DataFrame)
    rank_law_ok: bool = False
    rank_law_steps: int = 0
    provisional: tuple = ()
    notes: list = field(default_factory=list)

    @property
    def scans(self):
        return list(self.diags)

    def title(self, scan):
        return self.diags[scan].title if scan in self.diags else hl.scan_title(scan)

    def is_provisional(self, scans):
        scans = [scans] if isinstance(scans, str) else scans
        return sorted(set(scans) & set(self.provisional))


def _diag_seed_target(scan, results_root=None):
    """How many seeds the ``_diag`` pass INTENDS, from the other methods in the pass.

    Sven's spectra rest on the seeds that landed at the selected configuration (2 of 5 on
    CIFAR-CE while its re-selection re-runs).  The denominator of that ``finished/attempted``
    cannot come from Sven's own rows, so it is taken as the largest per-method seed count
    in the same pass -- the pass's own design, read from the pass.
    """
    try:
        df = hl.load(scan, 'diag', results_root=results_root, verbose=False)
    except Exception:
        return 0
    if not len(df) or 'model_seed' not in df:
        return 0
    return int(df.groupby('optimizer')['model_seed'].nunique().max())


def load(root=None, scans=None, verbose=True):
    """Read every input once and report what is provisional (see :class:`Data`)."""
    scans = list(scans or hl.HEADLINE_SCANS)
    data = Data(root=root)
    prov = set(c.provisional_scans())
    try:
        rep = hl.freshness_report(scans=scans, kinds=('', 'confirm', 'timing', 'diag'),
                                 results_root=root)
        prov |= set(rep.attrs.get('provisional_scans', []))
    except Exception as exc:                        # pragma: no cover - IO dependent
        data.notes.append(f'freshness_report failed: {exc}')
    for scan in scans:
        try:
            d = sf.sven_diag(scan, results_root=root, verbose=False)
        except (FileNotFoundError, ValueError) as exc:
            data.notes.append(f'{scan}: no usable Sven diag pass ({exc})')
            continue
        data.diags[scan] = d
        data.arrays[scan] = [sf.diag_arrays(r, results_root=root)
                             for _, r in d.rows.iterrows()]
        data.seed_target[scan] = _diag_seed_target(scan, results_root=root)
        if not d.selection_matches:
            prov.add(scan)
            data.notes.append(d.selection_note)
    data.provisional = tuple(sorted(prov & set(scans)))
    data.mech = sf.mechanism_table(data.diags.values(), results_root=root)
    # the independent check that the saved spectrum IS the one the step used: the rank law
    # min(k, rtol-rank) recomputed here against the optimizer's own num_nonzero_svs
    ok, n = True, 0
    for arrs in data.arrays.values():
        for a in arrs:
            if a.get('nnz_logged') is None:
                ok = False
                continue
            ok &= bool(np.array_equal(a['used_rank'], a['nnz_logged'].astype(int)))
            n += int(len(a['step']))
    data.rank_law_ok, data.rank_law_steps = bool(ok and n), n
    for scan, d in data.diags.items():
        try:
            g = sf.used_rank_grid(scan, lr=d.lr, results_root=root)
        except Exception as exc:                    # pragma: no cover - IO dependent
            data.notes.append(f'{scan}: used_rank_grid failed ({exc})')
            continue
        if len(g):
            data.grids[scan] = g
    for scan in scans:
        try:
            methods = sf.probe_methods(scan)
        except Exception:
            methods = []
        if not methods:
            continue
        data.probe_methods[scan] = methods
        data.probe[scan] = sf.probe_metrics(scan, methods=methods, results_root=root)
    if data.probe_methods:
        data.widths = sf.probe_widths(list(data.probe_methods))
        data.low4 = sf.low4_table(data.probe)
        frames = []
        for scan, methods in data.probe_methods.items():
            if 'Sven' not in methods or scan not in data.diags:
                continue
            d = data.diags[scan]
            ks = sorted({8, 16, 32, int(d.k), int(d.B)})
            tbl = sf.probe_energy(scan, ks=ks)
            if not len(tbl):
                continue
            tbl.insert(0, 'scan', scan)
            tbl.insert(1, 'k_selected', int(d.k))
            tbl.insert(2, 'B', int(d.B))
            frames.append(tbl)
        data.probe_energy = (pd.concat(frames, ignore_index=True) if frames
                             else pd.DataFrame())
    if verbose:
        print(f'{len(data.diags)} diag passes '
              f'({sum(len(a) for a in data.arrays.values())} Sven runs), '
              f'{len(data.probe)} scans with cached probe spectra')
        print(f'rank law min(k, rtol-rank) == num_nonzero_svs on '
              f'{data.rank_law_steps} logged steps: {data.rank_law_ok}')
        if data.provisional:
            print(f'PROVISIONAL: {", ".join(data.provisional)}')
        for note in data.notes:
            print(f'  note: {note}')
    return data


# ---------------------------------------------------------------------------
# Small plotting helpers (layout only -- no statistic is computed here)
# ---------------------------------------------------------------------------
def _panels(n, ncol, fraction, aspect=c.DEFAULT_ASPECT, extra_h=0.0, sharex=False,
            sharey=False):
    """A grid of ``n`` axes drawn at ``fraction`` of ``\\linewidth`` each, extras removed."""
    import matplotlib.pyplot as plt

    c.set_paper_style(fraction)
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, sharex=sharex, sharey=sharey,
                             figsize=c.figsize(ncol, nrow, fraction, aspect, extra_h))
    axes = np.atleast_1d(np.asarray(axes)).ravel()
    for ax in axes[n:]:
        ax.remove()
    return fig, list(axes[:n])


def _step_colorbar(fig, axes, norm, cmap='plasma', lo=0.15, hi=1.0,
                   label='Optimizer step'):
    """Colourbar matching :func:`spectra_figs.progress_colors`' clipped colormap.

    ``sv_diagnostics.epoch_colorbar`` places its own axes with ``fig.add_axes``, which
    fights ``constrained_layout`` (:func:`common.set_paper_style` turns it on); this asks
    the layout engine for the space instead.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap

    base = plt.get_cmap(cmap)
    clipped = LinearSegmentedColormap.from_list(
        f'{base.name}_clipped', base(np.linspace(lo, hi, 256)))
    sm = ScalarMappable(cmap=clipped, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=list(axes), fraction=0.03, pad=0.012, aspect=40)
    cb.set_label(label)
    cb.ax.tick_params(labelsize=plt.rcParams['ytick.labelsize'])
    return cb


def _progress_curve(d, arrays, key, smooth=SMOOTH_STEPS, transform=None):
    """``(training progress, seed-mean curve)`` for one :func:`spectra_figs.diag_arrays` key.

    The x axis is ``svs_step / n_steps``, never the index of the logged point: the dense
    schedule logs every step up to 1,000 and every 20th afterwards, so index fraction 0.5
    is 10 % of training on toy (B.fix WP3 finding 1).
    """
    step, mean, std, n = sf.seed_stack(d.rows, key, arrays=arrays)
    if step is None:
        return None, None, None
    if transform is not None:
        mean = transform(mean)
    xs, ys = sf.smooth_steps(step, mean, smooth)
    denom = float(d.n_steps or (xs[-1] if len(xs) else 1) or 1)
    return xs / denom, ys, n


def _scan_line(ax, scan, x, y, **kw):
    kw.setdefault('color', SCAN_COLORS.get(scan, '0.3'))
    kw.setdefault('ls', SCAN_LS.get(scan, '-'))
    kw.setdefault('lw', 1.3)
    kw.setdefault('label', SCAN_SHORT.get(scan, scan))
    return ax.plot(x, y, **kw)


def _legend_below(fig, handles, labels, ncol=4, **kw):
    """A figure-level legend UNDER the panels, with the layout engine making room.

    ``common.legend_below`` anchors at ``y=0`` in figure coordinates, which
    ``constrained_layout`` (on, from :func:`common.set_paper_style`) does not know about,
    so the legend lands on top of the bottom row's x labels.  matplotlib's ``outside``
    locations are the ones constrained_layout reserves space for.
    """
    kw.setdefault('frameon', False)
    return fig.legend(handles, labels, ncol=ncol, loc='outside lower center', **kw)


def _scan_legend(fig, scans, ncol=4, lw=1.3, **kw):
    """One figure-level legend of scan colours under a row of panels."""
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=SCAN_COLORS.get(s, '0.3'), ls=SCAN_LS.get(s, '-'),
                      lw=lw) for s in scans]
    labels = [SCAN_SHORT.get(s, s) for s in scans]
    return _legend_below(fig, handles, labels, ncol=ncol, **kw)


def _method_legend(fig, methods, ncol=None, band=True, **kw):
    """A figure-level optimizer legend, plus the seed-band entry the panels promise."""
    from matplotlib.patches import Patch

    handles, labels = c.method_handles(methods)
    if band:
        handles.append(Patch(color='0.45', alpha=0.25))
        labels.append(style.seed_spread_label())
    return _legend_below(fig, handles, labels, ncol=ncol or len(labels), **kw)


def _rtol_tex(rtol, min_exp=-3):
    """``0.0001`` -> ``$10^{-4}$``, ``0.03`` -> ``$0.03$`` -- a panel title is 1.4 in wide.

    ``1e-06`` costs five characters as a tick label and three as a power of ten, which is
    the difference between a readable and an overlapping rtol axis on the (k, rtol) grids.
    """
    exp = math.log10(float(rtol))
    if abs(exp - round(exp)) < 1e-9 and round(exp) <= min_exp:
        return rf'$10^{{{int(round(exp))}}}$'
    return rf'${rtol:g}$'


#: short y labels for the probe-metric panels: ``spectra_figs.PROBE_METRIC_LABELS``
#: spells each quantity out, which is right in a notebook and twice too long for a
#: 1.4 in panel whose neighbour is 0.02 in away
SHORT_METRIC_LABELS = {
    'eff_rank': r'Effective rank $e^{H(\sigma)}$',
    'cond': r'$\sigma_{\max} / \sigma_{\min}$',
    'sigma_B_over_1': r'$\sigma_B / \sigma_1$',
    'rank': 'Resolved rank',
    'proj_frac': r'$\|P_U r\|^2 / \|r\|^2$',
    'dist_init': r'$\|\theta_t - \theta_0\|$',
    'param_norm': r'$\|\theta_t\|$',
}


def _step_endpoints(ax, steps, loc='lower left', cmap='plasma', pad=0.03, bbox=True):
    """Name the FIRST and LAST step in the colours they are drawn in.

    Every ``progress_colors`` panel builds its own ``LogNorm`` from its own step range,
    and the seven scans run 5,400-15,620 steps, so ONE shared colourbar would be wrong
    for six of seven panels.  Two annotations per panel are exact, cost no layout space
    and say the same thing: which end of the colour scale is early.

    ``bbox`` puts a translucent white patch behind each label.  There is no corner that
    is free on all seven scans -- a decaying spectrum leaves the upper right empty while
    CIFAR-CE's and nanoGPT's are nearly flat and fill it -- so the label is placed in the
    emptiest corner FOR THAT PANEL and the patch keeps it readable where a line still
    passes under it.
    """
    steps = np.asarray(steps, dtype=float)
    if not steps.size:
        return None
    # hi=0.80 for the TEXT: the top of plasma is a pale yellow that is unreadable on
    # white, while the lines themselves keep the full range
    colors, _ = sf.progress_colors(np.maximum(steps[[0, -1]], 1), cmap=cmap, hi=0.80)
    left = 'left' in loc
    lower = 'lower' in loc
    x = pad if left else 1 - pad
    ys = (pad, pad + 0.09) if lower else (1 - pad - 0.09, 1 - pad)
    texts = [(f'step {int(steps[0]):,}', colors[0]),
             (f'step {int(steps[-1]):,}', colors[-1])]
    kw = {'bbox': dict(boxstyle='square,pad=0.12', fc='white', ec='none', alpha=0.85)} \
        if bbox else {}
    for (text, color), y in zip(texts if lower else texts[::-1], ys[::-1]):
        ax.annotate(text, xy=(x, y), xycoords='axes fraction', color=color,
                    ha='left' if left else 'right',
                    # anchored to the far side in each direction, so the pair stays
                    # INSIDE the axes: a bottom-anchored label at y=0.97 overflows into
                    # the panel title
                    va='bottom' if lower else 'top', zorder=6, **kw)
    return ax


def _emptiest_corner(ax, frac=0.30, default='lower left'):
    """Which corner of ``ax`` the fewest drawn points fall in (data coordinates).

    The step labels and the ``k`` label have to go somewhere, and on these seven scans
    "somewhere" differs per panel: the toy spectrum drops nine decades and leaves the
    whole right-hand side empty, CIFAR-CE's barely drops and leaves only the bottom.
    Counting the points each corner box actually contains decides it from the data
    instead of from an assumption about the shape of a spectrum.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    log_y = ax.get_yscale() == 'log'
    log_x = ax.get_xscale() == 'log'

    def _norm(v, lo, hi, logscale):
        v = np.asarray(v, dtype=float)
        if logscale:
            with np.errstate(divide='ignore', invalid='ignore'):
                v, lo, hi = np.log10(np.abs(v)), math.log10(abs(lo)), math.log10(abs(hi))
        span = (hi - lo) or 1.0
        return (v - lo) / span

    xs, ys = [], []
    for line in ax.get_lines():
        xd, yd = line.get_xdata(), line.get_ydata()
        if len(xd) != len(yd) or not len(xd):
            continue
        xs.append(_norm(xd, x0, x1, log_x))
        ys.append(_norm(yd, y0, y1, log_y))
    if not xs:
        return default
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if not xs.size:
        return default
    counts = {}
    for name, xsel, ysel in (('lower left', xs < frac, ys < frac),
                             ('lower right', xs > 1 - frac, ys < frac),
                             ('upper left', xs < frac, ys > 1 - frac),
                             ('upper right', xs > 1 - frac, ys > 1 - frac)):
        counts[name] = int(np.count_nonzero(xsel & ysel))
    # ties go to the earlier entry, i.e. the bottom of the panel, where no spectrum
    # figure draws a reference-line label
    return min(counts, key=lambda k: (counts[k], list(counts).index(k)))


#: legend labels a frozen helper spells in a way mathtext renders badly
#: (``resolution ceiling $1/1e-12$`` -> ``1/1e - 12``)
_LEGEND_RENAME = {
    r'resolution ceiling $1/1e-12$': r'float64 ceiling $10^{12}$',
    r'float64 probe floor (1e-12$\,\sigma_1$)': r'float64 floor $10^{-12}\,\sigma_1$',
}


def _axes_legend(fig, ncol=None, extra_handles=(), extra_labels=(), **kw):
    """One figure legend from EVERY panel's handles, de-duplicated by label.

    The probe-metric panels add their own entries when the data need them -- the float64
    resolution ceiling, the hollow marker for a ``sigma_B`` outside the resolved rank --
    and those must reach the legend, which a hand-built method key cannot know about.
    """
    seen = {}
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        for label, handle in zip(labels, handles):
            seen.setdefault(_LEGEND_RENAME.get(label, label), handle)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    for handle, label in zip(extra_handles, extra_labels):
        seen.setdefault(label, handle)
    # Sven first, whatever order the panels drew it in (it is drawn LAST so that its
    # black line is on top, which would otherwise put it last in the key too)
    order = sorted(seen, key=lambda lab: (lab != style.method_label('Sven'),
                                          list(seen).index(lab)))
    return _legend_below(fig, [seen[lab] for lab in order], order,
                         ncol=ncol or min(len(seen), 5), **kw)


def _probe_method_order(methods):
    """Sven last, so its black line is drawn ON TOP of the baselines it sits under.

    On MNIST label regression Sven and HIG differ by 0.3 % on both norms, and whichever
    is drawn second is the only one visible.
    """
    return [m for m in methods if m != 'Sven'] + (['Sven'] if 'Sven' in methods else [])


def _panel_title(d, scan=None, short=True, rtol=False, lines=1):
    """``MNIST-CE, $k/B=32/64$`` -- the configuration ON DISK, in one short line.

    A title on a 1.4 in panel has room for about 24 characters at the panel font size;
    the full spelling with the rtol needs 42 and overflows into the neighbouring panel.
    The rtol is drawn as a labelled line in the spectrum panels and is a column of
    ``tables_v2/spectra_mechanism.tex`` for every scan.

    ``lines=2`` breaks after the scan name.  The one-line form is still ~1.1 in wide at
    the 4-column panel size and the RIGHTMOST panel's title is then clipped by the figure
    edge (``save_fig`` writes the figure at its declared physical size, with no
    ``bbox_inches='tight'`` that would silently change the width on the page), so every
    grid wider than three columns asks for the two-line form.
    """
    name = SCAN_SHORT.get(scan or d.scan, d.title) if short else d.title
    tail = f', rtol {_rtol_tex(d.rtol)}' if rtol else ''
    sep = '\n' if lines > 1 else ', '
    return f'{name}{sep}$k/B = {d.k}/{d.B}${tail}'


def _annotate_cuts(ax, d, arrays, width, show_k=True, color='0.30', k_loc='bottom',
                   avoid=None):
    """Label the rtol line, the recorded Gram float32 floor and the ``k`` cut in place.

    A legend for the three reference lines needs ~1.6 in of width and these panels are
    1.4 in wide, so each line is labelled where it is drawn (and the floor's VALUE is a
    macro and a table column rather than a legend entry).  The two horizontal labels are
    staggered in x and sit on opposite sides of their lines: on four of the seven scans
    rtol and the recorded floor are within a factor of three of each other, and two
    labels at the same height on a ten-decade axis land on top of one another.  The rtol
    label starts just to the RIGHT of the ``k`` line, which is drawn full height and
    would otherwise strike through it.

    ``avoid`` is the corner :func:`_step_endpoints` was given, so the round-off-floor
    label -- which sits near the bottom of the axis on four of the seven scans, exactly
    where the step labels are -- goes to the other side of the panel rather than on top
    of them.  ``k_loc='top'`` moves the ``k`` label to the top of its line, which is what
    the narrow 4-column panels need: at the bottom it lands on the step labels.
    """
    floor = float(np.nanmedian(np.concatenate([a['noise_rel'] for a in arrays])))
    x_rtol = (d.k / max(width, 1) + 0.04) if d.truncates else 0.26
    # the floor label sits opposite the step labels: right-aligned at the panel edge when
    # they are on the left, left-aligned in from the middle when they are on the right
    floor_right = 'left' in (avoid or '')
    for xf, y, va, dy, ha, text in ((x_rtol, d.rtol, 'bottom', 1.5, 'left',
                                     r'rtol$\,\sigma_{\max}$'),
                                    (0.99 if floor_right else 0.34, floor, 'top', -1.5,
                                     'right' if floor_right else 'left',
                                     r'$\sqrt{\epsilon}\,\sigma_{\max}$')):
        ax.annotate(text, xy=(min(max(xf, 0.02), 0.99) * width, y), xytext=(0, dy),
                    textcoords='offset points', ha=ha, va=va, color=color,
                    clip_on=False, zorder=5)
    if show_k and d.truncates:
        # at the BOTTOM of the k line: the top of a spectrum panel is where the rtol
        # label and the leading singular values are.  At the top, the label goes to the
        # LEFT of the line, because the rtol label always starts just to its right --
        # on the two scans whose rtol is 0.3 that line is within a decade of sigma_max
        # and the two labels would otherwise be printed on top of each other.
        top = k_loc == 'top'
        ax.annotate(f'$k={d.k}$',
                    xy=(d.k - 1, ax.get_ylim()[1] if top else ax.get_ylim()[0]),
                    xytext=(-1.5 if top else 1.5, -1 if top else 1),
                    textcoords='offset points',
                    ha='right' if top else 'left', va='top' if top else 'bottom',
                    color='#C44E52', clip_on=False, zorder=5,
                    bbox=(dict(boxstyle='square,pad=0.10', fc='white', ec='none',
                               alpha=0.85) if top else None))
    return floor


# ---------------------------------------------------------------------------
# The option layer: the knobs every draw function reads (see figspec.py)
# ---------------------------------------------------------------------------
def _panels_for(opts, n, ncol=None):
    """:func:`_panels` driven by a figure's geometry knobs.

    Every spec carries ``ncol`` / ``fraction`` / ``aspect`` / ``extra_h`` / ``sharex`` /
    ``sharey``, so the panel grid of any figure can be changed from a notebook without
    touching a builder.  ``ncol`` given here is a grid whose width follows from the
    content (one column per metric, per method, ...) and the knob then only overrides it.
    """
    return _panels(n, int(ncol or opts['ncol']), float(opts['fraction']),
                   aspect=float(opts['aspect']), extra_h=float(opts['extra_h']),
                   sharex=bool(opts['sharex']), sharey=bool(opts['sharey']))


def _scans_drawn(data, opts, pool=None):
    """The scans a figure draws: its ``scans`` knob, filtered to what was actually read.

    An empty ``scans`` means "every scan in ``pool``" (the default for the appendix
    figures, which are per-scan grids).  A named scan that is not in ``pool`` is dropped
    rather than raised on, which is what the builders did before the knob existed: a
    figure over the headline scans must still draw on a results root where one of them
    has no diag pass yet.
    """
    pool = list(data.scans if pool is None else pool)
    want = [str(s) for s in (opts.get('scans') or ())]
    return [s for s in want if s in pool] if want else pool


def _step_label_loc(ax, opts, default='lower left'):
    """Which corner :func:`_step_endpoints` writes in: a fixed one, or ``'auto'``."""
    loc = opts.get('step_label_loc') or 'auto'
    if loc == 'auto':
        return _emptiest_corner(ax, default=opts.get('step_label_fallback') or default)
    return loc


def _require_probe(data, name, scans=None):
    """Refuse a probe figure with no offline cache, in one sentence instead of a KeyError.

    ``build()`` skips these six figures when nothing is cached (:data:`PROBE_FIGURES`);
    a notebook asking for one by name deserves to be told why it cannot be drawn rather
    than to trip over an empty dict three frames down.
    """
    if not data.probe_methods:
        raise RuntimeError(
            f'{name}: no cached probe spectra under {ct.SPECTRA_DIR} -- this figure is '
            f'the OFFLINE float64 measurement and needs '
            f'tools/compute_ckpt_spectra.py <scan> to have run')
    if scans is not None and not scans:
        raise RuntimeError(
            f'{name}: none of the scans it draws has a probe cache; cached: '
            f'{sorted(data.probe_methods)}')
    return True


# ---------------------------------------------------------------------------
# F2 -- main text, section 4.2
# ---------------------------------------------------------------------------
def _spectrum_truncation(data, opts):
    """F2: the singular-value story in three panels at ``0.32\\linewidth``.

    (a) the full-width batch spectrum of the polynomial scan over training, with the ``k``
    cut, the ``rtol`` line and the RECORDED Gram float32 floor -- everything below that
    floor is round-off, not structure; (b) the fraction of the batch residual the cut
    keeps, against training progress, for the three headline scans and MNIST-CE; (c) the
    number of directions actually inverted, ``min(k, rtol-rank)``, for all seven scans.
    """
    scan = (opts['spectrum_scan'] if opts['spectrum_scan'] in data.diags
            else data.scans[0])
    scans = _scans_drawn(data, opts)
    floor = float(opts['discard_floor'])
    fig, axes = _panels_for(opts, 3)
    a, b, cc = axes

    # (a) spectrum evolution, one seed (a seed mean would draw a matrix no step ever had)
    d = data.diags[scan]
    arr = data.arrays[scan][:1]
    norm, handles = sf.plot_spectra_over_training(
        a, d, arrays=arr, limit=int(opts['limit']), lw=float(opts['lw']),
        cmap=opts['cmap'], show_rtol=bool(opts['show_rtol']),
        show_noise=bool(opts['show_noise']), show_k=bool(opts['show_k']))
    lo, hi = a.get_ylim()
    pad_lo, pad_hi = opts['spectrum_headroom']
    a.set_ylim(lo / pad_lo, hi * pad_hi)
    a.set_xlabel('SV index $i$')
    a.set_ylabel(r'$\sigma_i / \sigma_{\max}$')
    a.set_title(f'(a) {SCAN_SHORT.get(scan, scan)} spectrum')
    width = int(arr[0]['width'])
    _annotate_cuts(a, d, arr, width, show_k=bool(opts['show_k']))
    steps = np.asarray(arr[0]['step'])
    colors, _ = sf.progress_colors(np.maximum(steps[[0, -1]], 1), cmap=opts['cmap'])
    # The late-training colour is a pale yellow on this colormap, which is close to
    # illegible as text on white at print size, so the two step labels are drawn in a
    # dark hue with a small coloured marker carrying the colormap information instead.
    if opts['annotate_steps']:
        for frac, step, colour in ((0.10, steps[0], colors[0]),
                                   (0.02, steps[-1], colors[-1])):
            a.plot([0.035], [frac + 0.012], marker='o', ms=2.2, color=colour,
                   transform=a.transAxes, clip_on=False, zorder=8)
            a.annotate(f'step {int(step):,}', xy=(0.065, frac), xycoords='axes fraction',
                       color='0.15', ha='left', va='bottom')

    # (b) residual energy DISCARDED by the cut (k AND rtol), vs training progress.  Every
    # scan is drawn: on a linear "kept" axis the five scans that keep ~100% lay on top of
    # one another and the one scan whose kept energy falls substantially (CIFAR-CE) was
    # missing from the panel the main text cites for exactly that caveat.
    for s in scans:
        x, y, _ = _progress_curve(data.diags[s], data.arrays[s], 'frac_used',
                                  smooth=opts['smooth'])
        if x is None:
            continue
        _scan_line(b, s, x, np.maximum(1.0 - np.asarray(y, dtype=float), floor),
                   lw=opts['scan_lw'])
    b.set_yscale('log')
    b.set_ylim(floor / 2, 1.4)
    b.set_xlim(0, 1)
    b.set_xlabel('Training progress')
    b.set_ylabel('Batch residual discarded')
    b.set_title('(b) what the cut discards')

    # (c) directions actually inverted, all scans
    for s in scans:
        x, y, _ = _progress_curve(data.diags[s], data.arrays[s], 'used_rank',
                                  smooth=opts['smooth'])
        if x is not None:
            _scan_line(cc, s, x, y, lw=opts['scan_lw'])
    cc.set_yscale('log')
    cc.set_xlim(0, 1)
    cc.set_xlabel('Training progress')
    cc.set_ylabel(r'SVs inverted $\min(k,\,$rtol$)$')
    cc.set_title('(c) rank actually used')

    if opts['show_legend']:
        _scan_legend(fig, scans, ncol=opts['legend_ncol'], lw=opts['scan_lw'],
                     **opts['legend_kw'])
    prov = c.provenance(
        functions=['spectra_figs.sven_diag', 'spectra_figs.plot_spectra_over_training',
                   'spectra_figs.diag_arrays', 'spectra_figs.seed_stack',
                   'spectra_figs.smooth_steps', 'spectra_figs.energy_in_top',
                   'spectra_figs.used_rank'],
        scans=[hl.dir_name(s, 'diag') for s in scans],
        reads=[c.results_root()], provisional=data.is_provisional(scans),
        note=(f'(a) {scan} one seed, online per-batch Gram spectra; (b) frac_used = '
              f'energy in min(k, rtol-rank) directions of THIS batch; (c) all '
              f'{len(scans)} scans, seed mean, {opts["smooth"]}-step moving average'))
    return fig, {'axes': axes, 'provenance': prov, 'scan': scan, 'scans': scans}


# ---------------------------------------------------------------------------
# F9 -- appendix O
# ---------------------------------------------------------------------------
def _online_spectra(data, opts):
    """Full-width online spectra over training for every scan, with the three cuts."""
    scans = _scans_drawn(data, opts)
    fig, axes = _panels_for(opts, len(scans))
    norm = None
    for ax, scan in zip(axes, scans):
        d, arr = data.diags[scan], data.arrays[scan][:1]
        norm, _ = sf.plot_spectra_over_training(
            ax, d, arrays=arr, limit=int(opts['limit']), lw=float(opts['lw']),
            cmap=opts['cmap'], show_rtol=bool(opts['show_rtol']),
            show_noise=bool(opts['show_noise']), show_k=bool(opts['show_k']))
        lo, hi = ax.get_ylim()
        # a decade of headroom above sigma_max, which is where the k label goes: with the
        # 1.5x of the notebook version it lands on the rtol label on the two scans whose
        # selected rtol is 0.3
        pad_lo, pad_hi = opts['spectrum_headroom']
        ax.set_ylim(lo / pad_lo, hi * pad_hi)
        ax.set_title(_panel_title(d, scan, short=opts['short_titles'],
                                  lines=opts['title_lines']))
        ax.set_xlabel('SV index $i$')
        ax.set_ylabel(r'$\sigma_i / \sigma_{\max}$')
        width = int(arr[0]['width'])
        # deterministic, and collision-free by construction on all seven scans: the two
        # step labels own the lower LEFT, the round-off-floor label the right-hand edge,
        # the rtol label the space just right of the k line at its own height, and the k
        # label the empty band above sigma_max.  The bottom of the k line is NOT free --
        # "step 15,600" is half a panel wide and the k line stands at k/B = 0.5.
        loc = _step_label_loc(ax, opts)
        _annotate_cuts(ax, d, arr, width, show_k=bool(opts['show_k']), avoid=loc,
                       k_loc=opts['k_loc'])
        if opts['annotate_steps']:
            _step_endpoints(ax, arr[0]['step'], loc=loc, cmap=opts['cmap'])
    prov = c.provenance(
        functions=['spectra_figs.plot_spectra_over_training', 'spectra_figs.diag_arrays',
                   'spectra_figs.noise_floor_rel'],
        scans=[hl.dir_name(s, 'diag') for s in scans], reads=[c.results_root()],
        provisional=data.is_provisional(scans),
        note='one seed per panel (each line is a different batch; a seed mean would draw '
             'a matrix no step ever had)')
    return fig, {'axes': axes, 'provenance': prov, 'scans': scans, 'norm': norm}


def _online_utr(data, opts):
    """``|u_i.r| / ||U^T r||`` over training: where the batch residual actually sits."""
    scans = _scans_drawn(data, opts)
    fig, axes = _panels_for(opts, len(scans))
    norm = None
    for ax, scan in zip(axes, scans):
        d, arr = data.diags[scan], data.arrays[scan][:1]
        norm = sf.plot_utr_over_training(ax, d, arrays=arr, limit=int(opts['limit']),
                                         lw=float(opts['lw']), cmap=opts['cmap'],
                                         show_k=bool(opts['show_k'])) or norm
        lo, hi = ax.get_ylim()
        pad_lo, pad_hi = opts['spectrum_headroom']
        ax.set_ylim(lo / pad_lo, hi * pad_hi)
        ax.set_title(_panel_title(d, scan, short=opts['short_titles'],
                                  lines=opts['title_lines']))
        ax.set_xlabel('SV index $i$')
        ax.set_ylabel(r'$|u_i^{\top} r| / \|U^{\top} r\|$')
        if opts['show_k'] and d.truncates:
            ax.annotate(f'$k={d.k}$', xy=(d.k - 1, ax.get_ylim()[1]), xytext=(1.5, -1),
                        textcoords='offset points', ha='left', va='top',
                        color='#C44E52', clip_on=False, zorder=5)
        if opts['annotate_steps']:
            _step_endpoints(ax, arr[0]['step'], loc=_step_label_loc(ax, opts),
                            cmap=opts['cmap'])
    prov = c.provenance(
        functions=['spectra_figs.plot_utr_over_training', 'spectra_figs.diag_arrays'],
        scans=[hl.dir_name(s, 'diag') for s in scans], reads=[c.results_root()],
        provisional=data.is_provisional(scans),
        note='online, one seed per panel; the companion of the spectrum figure -- the '
             'spectrum says how hard a direction is to move, this says how much '
             'residual is in it')
    return fig, {'axes': axes, 'provenance': prov, 'scans': scans, 'norm': norm}


def _online_mechanism(data, opts):
    """Kept and discarded batch residual, the used rank, and its seed spread."""
    scans = _scans_drawn(data, opts)
    floor = float(opts['discard_floor'])
    fig, axes = _panels_for(opts, 4)
    keep, disc, rank, seeds = axes

    for s in scans:
        x, y, _ = _progress_curve(data.diags[s], data.arrays[s], 'frac_used',
                                  smooth=opts['smooth'])
        if x is None:
            continue
        _scan_line(keep, s, x, y, lw=opts['scan_lw'])
        _scan_line(disc, s, x, np.clip(1.0 - y, floor, None), lw=opts['scan_lw'])
        xr, yr, _ = _progress_curve(data.diags[s], data.arrays[s], 'used_rank',
                                    smooth=opts['smooth'])
        _scan_line(rank, s, xr, yr, lw=opts['scan_lw'])
    keep.set_ylim(0, 1.06)
    keep.set_xlim(0, 1)
    keep.set_xlabel('Training progress')
    keep.set_ylabel('Batch residual kept')
    keep.set_title('(a) energy in the inverted directions')
    disc.axhline(floor, color='0.6', lw=0.8, ls=':', zorder=0)
    disc.annotate(f'floor $10^{{{int(round(math.log10(floor)))}}}$ '
                  f'(nothing discarded)', xy=(0.02, floor), xytext=(0, 2),
                  textcoords='offset points', ha='left', va='bottom', color='0.35')
    disc.set_yscale('log')
    disc.set_xlim(0, 1)
    disc.set_xlabel('Training progress')
    disc.set_ylabel('Batch residual discarded')
    disc.set_title('(b) what the cut discards')
    rank.set_yscale('log')
    rank.set_xlim(0, 1)
    rank.set_xlabel('Training progress')
    rank.set_ylabel(r'SVs inverted $\min(k,\,$rtol$)$')
    rank.set_title('(c) rank actually used')

    # (d) the seed spread the seed-mean panels hide: per-seed used rank on the scan whose
    # final rank varies most over seeds (MNIST-CE, 68.7 % relative spread)
    spread = _rank_spread(data)
    worst = opts['spread_scan'] or (spread.iloc[0]['scan'] if len(spread) else scans[0])
    d = data.diags[worst]
    for a, seed in zip(data.arrays[worst], d.seeds):
        ys, idx = sv.smooth(a['used_rank'].astype(float), int(opts['seed_smooth']))
        seeds.plot(np.asarray(a['step'])[idx] / max(d.n_steps, 1), ys,
                   lw=float(opts['seed_lw']), label=f'seed {seed}')
    seeds.axhline(d.k, color='#C44E52', lw=1.0, ls=':', label=f'$k={d.k}$')
    seeds.set_xlim(0, 1)
    seeds.set_xlabel('Training progress')
    seeds.set_ylabel('SVs inverted')
    seeds.set_title(f'(d) {SCAN_SHORT.get(worst, worst)} per seed (no averaging)')
    seeds.legend(**opts['seed_legend_kw'])

    if opts['show_legend']:
        _scan_legend(fig, scans, ncol=opts['legend_ncol'], lw=opts['scan_lw'],
                     **opts['legend_kw'])
    prov = c.provenance(
        functions=['spectra_figs.seed_stack', 'spectra_figs.smooth_steps',
                   'spectra_figs.diag_arrays', 'sv_diagnostics.smooth'],
        scans=[hl.dir_name(s, 'diag') for s in scans], reads=[c.results_root()],
        provisional=data.is_provisional(scans),
        note=f'panels (a)-(c) seed mean, {opts["smooth"]}-step moving average; (d) the '
             f'{len(data.arrays.get(worst, []))} seeds of {worst} drawn separately')
    return fig, {'axes': axes, 'provenance': prov, 'scans': scans,
                 'spread_scan': worst}


def _online_norms(data, opts):
    """Applied update norm and batch residual norm over training, per scan."""
    scans = _scans_drawn(data, opts)
    fig, axes = _panels_for(opts, len(scans))
    for ax, scan in zip(axes, scans):
        sf.plot_norms(ax, data.diags[scan], arrays=data.arrays[scan],
                      smooth=int(opts['smooth']), band=bool(opts['band']))
        ax.set_title(_panel_title(data.diags[scan], scan,
                                  short=opts['short_titles'],
                                  lines=opts['title_lines']))
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    lw = float(opts['legend_lw'])
    if opts['show_legend']:
        _legend_below(fig, [Line2D([], [], color=style.method_color('Sven'), lw=lw),
                            Line2D([], [], color='#DD8452', lw=lw),
                            Patch(color='0.45', alpha=0.25)],
                      [r'$\|\Delta\theta\|$ (applied, incl. $\eta$)', r'$\|r\|$',
                       style.seed_spread_label()], ncol=opts['legend_ncol'],
                      **opts['legend_kw'])
    prov = c.provenance(
        functions=['spectra_figs.plot_norms', 'spectra_figs.seed_stack'],
        scans=[hl.dir_name(s, 'diag') for s in scans], reads=[c.results_root()],
        provisional=data.is_provisional(scans),
        note='update_norm is the APPLIED change (learning rate included); seed mean '
             '+/- 1 std')
    return fig, {'axes': axes, 'provenance': prov, 'scans': scans}


#: the two scans the label-regression vs cross-entropy comparison contrasts, and the
#: fractions of training its spectrum panel samples
LR_VS_CE = ('mnist_scan_labelRegression', 'mnist_scan_ce')
LR_VS_CE_MARKS = (0.0, 0.02, 0.1, 0.5, 1.0)


def _progress_band(d, arrays, key, smooth=SMOOTH_STEPS):
    """``(progress, mean, mean-std, mean+std)`` for one :func:`diag_arrays` key.

    :func:`_progress_curve`'s companion for the panels that promise a seed band: the
    same x axis (optimizer step over the scan's own step count) and the same moving
    average applied to all three curves, so the band cannot drift off its mean.
    """
    step, mean, std, n = sf.seed_stack(d.rows, key, arrays=arrays)
    if step is None:
        return (None,) * 4
    xs, ys = sf.smooth_steps(step, mean, smooth)
    _, lo = sf.smooth_steps(step, mean - std, smooth)
    _, hi = sf.smooth_steps(step, mean + std, smooth)
    denom = float(d.n_steps or (xs[-1] if len(xs) else 1) or 1)
    return xs / denom, ys, lo, hi


def _lr_vs_ce(data, opts):
    """MNIST label regression against cross-entropy: the same model, two spectra.

    The revision's replacement for the single-seed LR-vs-CE spectrum panel of App. I.
    Both scans share the architecture, the batch size (``B=64``) and the splits, so the
    two spectra are directly comparable index by index -- and they are the paper's
    cleanest evidence that the loss, not the data, sets the hierarchy: the CE spectrum
    collapses onto a handful of directions within the first epochs while the
    label-regression spectrum stays flat and keeps all ``B`` of them.
    """
    scans = _scans_drawn(data, opts)
    marks = [float(f) for f in opts['marks']]
    fsize = float(opts['annot_fontsize'])
    a_lo, a_hi = opts['alpha_range']
    fig, axes = _panels_for(opts, 3)
    spec, rank, keep = axes
    caps = []
    for scan in scans:
        d, arrs = data.diags[scan], data.arrays[scan]
        a = arrs[0]
        steps = np.asarray(a['step'], dtype=float)
        color = SCAN_COLORS.get(scan, '0.3')
        # (a) the spectrum at matched FRACTIONS of training, not at matched steps: the
        # two scans run the same number of epochs but MNIST-CE logs 15,620 steps
        n_steps = float(d.n_steps or steps[-1] or 1)
        for frac in marks:
            j = int(np.argmin(np.abs(steps - frac * n_steps)))
            spec.plot(np.arange(a['svs_rel'].shape[1]), a['svs_rel'][j],
                      color=color, ls=SCAN_LS.get(scan, '-'), lw=float(opts['lw']),
                      alpha=a_lo + (a_hi - a_lo) * frac,
                      label=SCAN_SHORT.get(scan, scan) if frac == marks[-1]
                      else None)
        spec.axhline(d.rtol, color=color, lw=0.9, ls=':', zorder=0)
        spec.annotate(f'rtol {_rtol_tex(d.rtol)}', xy=(0.99, d.rtol), xytext=(0, 1.5),
                      textcoords='offset points',
                      xycoords=spec.get_yaxis_transform(), ha='right', va='bottom',
                      color=color, fontsize=fsize)
        # (b), (c) the consequence, over the seeds
        for ax, key in ((rank, 'used_rank'), (keep, 'frac_used')):
            x, y, lo, hi = _progress_band(d, arrs, key, smooth=opts['smooth'])
            if x is None:
                continue
            ax.fill_between(x, lo, hi, color=color, alpha=float(opts['band_alpha']),
                            lw=0)
            _scan_line(ax, scan, x, y, lw=opts['scan_lw'])
        rank.axhline(d.k, color=color, lw=0.9, ls=':', zorder=0)
        rank.annotate(f'$k={d.k}$', xy=(0.01, d.k), xytext=(0, 1.5),
                      textcoords='offset points', xycoords=rank.get_yaxis_transform(),
                      ha='left', va='bottom', color=color, fontsize=fsize)
        caps.append(int(d.k))
    spec.set_yscale('log')
    spec.set_xlabel('SV index $i$')
    spec.set_ylabel(r'$\sigma_i / \sigma_{\max}$')
    spec.set_title('(a) spectrum, faint $\\to$ late')
    for ax, label, title in ((rank, r'SVs inverted $\min(k,\,$rtol$)$',
                              '(b) rank actually used'),
                             (keep, 'Batch residual kept', '(c) energy kept')):
        ax.set_xlim(0, 1)
        ax.set_xlabel('Training progress')
        ax.set_ylabel(label)
        ax.set_title(title)
    rank.set_yscale('log')
    if caps:
        # headroom above the higher cap, so the two "k = ..." labels sit under their own
        # dotted lines instead of in the panel title
        rank.set_ylim(top=max(caps) * float(opts['rank_headroom']))
    keep.set_ylim(0, 1.06)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=SCAN_COLORS.get(s, '0.3'),
                      ls=SCAN_LS.get(s, '-'), lw=float(opts['scan_lw'])) for s in scans]
    if opts['show_legend']:
        _legend_below(fig, handles + [Patch(color='0.45', alpha=0.25)],
                      [SCAN_SHORT.get(s, s) for s in scans]
                      + [style.seed_spread_label()],
                      ncol=opts['legend_ncol'], **opts['legend_kw'])
    prov = c.provenance(
        functions=['spectra_figs.diag_arrays', 'spectra_figs.seed_stack',
                   'spectra_figs.smooth_steps'],
        scans=[hl.dir_name(s, 'diag') for s in scans], reads=[c.results_root()],
        provisional=data.is_provisional(scans),
        note='(a) one seed per scan, the spectrum at '
             f'{", ".join(f"{f:g}" for f in marks)} of training (a seed mean '
             'would draw a matrix no step ever had); (b), (c) seed mean +/- 1 std over '
             f'the diag seeds, {opts["smooth"]}-step moving average; both scans share the '
             'architecture, B and the splits, so the SV index axes are comparable')
    return fig, {'axes': axes, 'provenance': prov, 'scans': scans}


def _grid_heatmap(ax, g, d, fontsize=5.0, cmap='viridis', mark_selected=True):
    """One ``(k, rtol)`` grid of ``used / k``, with short cells hatched and counted.

    A cell can rest on one seed while its neighbour rests on five (Sven diverges at the
    bottom of every ``rtol`` grid), so ``n_seeds/attempted`` is printed inside every short
    cell and the cell is hatched; a cell whose every run diverged prints ``0/attempted``
    in red, and a cell the grid never contained is simply absent.
    """
    from matplotlib.patches import Rectangle

    full = int(g['n_records'].max())
    piv = g.pivot(index='k', columns='rtol', values='used_frac_k').sort_index(
        ascending=False)
    nsd = g.pivot(index='k', columns='rtol', values='n_seeds').reindex(
        index=piv.index, columns=piv.columns)
    att = g.pivot(index='k', columns='rtol', values='n_records').reindex(
        index=piv.index, columns=piv.columns)
    ax.imshow(piv.to_numpy(), vmin=0, vmax=1, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(piv.columns)), [_rtol_tex(v) for v in piv.columns],
                  rotation=45, ha='right')
    ax.set_yticks(range(len(piv.index)), [int(i) for i in piv.index])
    ax.set_xlabel('rtol')
    ax.set_ylabel('$k$')
    values, counts, atts = piv.to_numpy(), nsd.to_numpy(), att.to_numpy()
    for yi in range(len(piv.index)):
        for xi in range(len(piv.columns)):
            v, ns, at = values[yi, xi], counts[yi, xi], atts[yi, xi]
            if not np.isfinite(v):
                if np.isfinite(at):
                    ax.text(xi, yi, f'0/{int(at)}', ha='center', va='center',
                            fontsize=fontsize, color='#C44E52')
                continue
            short = np.isfinite(ns) and int(ns) < full
            ax.text(xi, yi, f'{v:.2f}' + (f'\n{int(ns)}/{int(at)}' if short else ''),
                    ha='center', va='center', fontsize=fontsize,
                    color='w' if v < 0.6 else 'k')
            if short:
                ax.add_patch(Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False,
                                       hatch='///', edgecolor='#C44E52', lw=0.8))
    cols = np.asarray(piv.columns, dtype=float)
    if mark_selected and np.isclose(d.rtol, cols).any() and d.k in list(piv.index):
        ax.plot(int(np.argmin(np.abs(cols - d.rtol))), list(piv.index).index(d.k),
                marker='s', ms=9, mfc='none', mec='#C44E52', mew=1.4)
    return ax


def _used_rank_grid(data, opts):
    """Which cut binds over the tuning ``(k, rtol)`` grid, at the selected learning rate."""
    scans = _scans_drawn(data, opts, pool=[s for s in data.scans if s in data.grids])
    fig, axes = _panels_for(opts, len(scans))
    for ax, scan in zip(axes, scans):
        d = data.diags[scan]
        _grid_heatmap(ax, data.grids[scan], d, fontsize=float(opts['cell_fontsize']),
                      cmap=opts['cmap'], mark_selected=bool(opts['mark_selected']))
        ax.set_title(f'{SCAN_SHORT.get(scan, d.title)}\n'
                     f'SVs used / $k$ at $\\eta={d.lr:g}$')
    prov = c.provenance(
        functions=['spectra_figs.used_rank_grid', 'sv_diagnostics.rank_per_epoch'],
        scans=[hl.dir_name(s) for s in scans], reads=[c.results_root()],
        provisional=data.is_provisional(scans),
        note='final-epoch mean num_nonzero_svs / k over the TUNING grid at the selected '
             'lr; red square = the selected configuration, red hatch = fewer seeds than '
             'the grid intended (n_seeds/attempted printed)')
    return fig, {'axes': axes, 'provenance': prov, 'scans': scans}


def _probe_seed(scan):
    """The lowest cached model seed of a scan -- named in the panel, never implicit."""
    ents = ct.load_spectra(sf.probe_scan_name(scan), n_probe=-1, epochs_only=False)
    return min(int(e['model_seed']) for e in ents) if ents else None


def _probe_spectra(data, opts):
    """Probe-set float64 spectra along each optimizer's trajectory, one named seed."""
    _require_probe(data, 'probe_spectra')
    scan = opts['scan'] or next((s for s in PROBE_DEEP_SCANS if s in data.probe_methods),
                                next(iter(data.probe_methods), None))
    if scan not in data.probe_methods:
        raise RuntimeError(f'probe_spectra: no probe cache for scan {scan!r}; cached: '
                           f'{sorted(data.probe_methods)}')
    methods = data.probe_methods[scan]
    seed = _probe_seed(scan)
    d = data.diags.get(scan)
    fig, axes = _panels_for(opts, len(methods))
    steps = None
    for ax, method in zip(axes, methods):
        sf.plot_probe_spectra(ax, scan, method, seed=seed, limit=int(opts['limit']),
                              lw=float(opts['lw']), cmap=opts['cmap'],
                              annotate_seed=False)
        ax.set_title(style.method_label(method))
        ax.set_xlabel('SV index $i$')
        if steps is None:
            ents = ct.load_spectra(sf.probe_scan_name(scan), method=method,
                                   model_seed=seed)
            steps = np.asarray(ents[0]['step']) if ents else None
        if steps is not None and opts['annotate_steps']:
            _step_endpoints(ax, steps, loc=_step_label_loc(ax, opts),
                            cmap=opts['cmap'])
        if d is not None and opts['show_k'] and style.canonical_method(method) == 'Sven':
            # the whole point of the probe set: Sven inverts at most k of these
            # min(rows, P) directions
            ax.axvline(d.k, color='#C44E52', lw=float(opts['k_lw']), zorder=0)
            ax.annotate(f'$k={d.k}$', xy=(d.k, ax.get_ylim()[0]), xytext=(1.5, 1),
                        textcoords='offset points', ha='left', va='bottom',
                        color='#C44E52')
    for ax in axes:
        if not ax.get_subplotspec().is_first_col():
            ax.set_ylabel('')
    # which scan and seed this is belongs to the FIGURE, not inside the first panel: at
    # the top left of a probe-spectrum panel it sits on the leading singular values and
    # on the k line
    if opts['scan_seed_title']:
        fig.suptitle(f'{SCAN_SHORT.get(scan, scan)}, model seed {seed}', color='0.25')
    prov = c.provenance(
        functions=['spectra_figs.plot_probe_spectra', 'ckpt_tools.load_spectra'],
        scans=[sf.probe_scan_name(scan)],
        reads=[str(ct.SPECTRA_DIR)],
        note=f'{scan}, model seed {seed} of '
             f'{len(ct.load_spectra(sf.probe_scan_name(scan)))} cached trajectories; the '
             f'SAME probe rows at every checkpoint and for every optimizer, float64')
    return fig, {'axes': axes, 'provenance': prov, 'scan': scan, 'seed': seed,
                 'methods': list(methods)}


def _probe_spectra_all(data, opts):
    """The probe-spectrum figure over every cached scan x optimizer."""
    _require_probe(data, 'probe_spectra_all')
    scans = _scans_drawn(data, opts, pool=list(data.probe_methods))
    ncol = int(opts['ncol']) or max(len(data.probe_methods[s]) for s in scans)
    fig, axes = _panels_for(opts, len(scans) * ncol, ncol=ncol)
    seeds = {}
    for i, scan in enumerate(scans):
        seeds[scan] = _probe_seed(scan)
        d = data.diags.get(scan)
        for j, method in enumerate(data.probe_methods[scan]):
            ax = axes[i * ncol + j]
            sf.plot_probe_spectra(ax, scan, method, seed=seeds[scan],
                                  limit=int(opts['limit']), lw=float(opts['lw']),
                                  cmap=opts['cmap'], annotate_seed=False)
            ax.set_title(f'{SCAN_SHORT.get(scan, scan)} / {style.method_label(method)}')
            ax.set_xlabel('SV index $i$')
            if j:
                ax.set_ylabel('')
            if (d is not None and opts['show_k']
                    and style.canonical_method(method) == 'Sven'):
                ax.axvline(d.k, color='#C44E52', lw=float(opts['k_lw']), zorder=0)
            if not j and opts['annotate_steps']:
                ents = ct.load_spectra(sf.probe_scan_name(scan), method=method,
                                       model_seed=seeds[scan])
                if ents:
                    _step_endpoints(ax, np.asarray(ents[0]['step']),
                                    loc=_step_label_loc(ax, opts), cmap=opts['cmap'])
    prov = c.provenance(
        functions=['spectra_figs.plot_probe_spectra', 'ckpt_tools.load_spectra'],
        scans=[sf.probe_scan_name(s) for s in scans],
        reads=[str(ct.SPECTRA_DIR)],
        note='one named seed per scan: ' + ', '.join(f'{s}={v}' for s, v in seeds.items()))
    return fig, {'axes': axes, 'provenance': prov, 'scans': scans, 'seeds': seeds}


def _probe_panel(ax, data, scan, metric, title=None, logy=None, marker='.', band=True,
                 cond_pad=5.0):
    """One :func:`spectra_figs.plot_probe_metric` panel, sized for the page.

    Three deviations from the notebook version, all of them about a 1.4 in panel: a short
    y label (:data:`SHORT_METRIC_LABELS`), Sven drawn last so its black line is visible
    where it coincides with a baseline, and -- for the condition number -- a y range set
    from the seed MEANS and the resolution ceiling rather than from the seed bands, which
    on the synthetics span nine decades and would flatten every curve into a line.
    """
    logy = (metric not in ('eff_rank', 'rank', 'proj_frac')) if logy is None else logy
    table = data.probe[scan]
    sf.plot_probe_metric(ax, table, metric, marker=marker, logy=logy, band=band,
                         methods=_probe_method_order(data.probe_methods[scan]))
    ax.set_ylabel(SHORT_METRIC_LABELS.get(metric,
                                          sf.PROBE_METRIC_LABELS.get(metric, metric)))
    if title:
        ax.set_title(title)
    if metric == 'cond' and metric in table:
        means = table.groupby(['method', 'step'])[metric].mean()
        floor = float(table['floor'].iloc[0])
        if len(means) and means.max() > 0:
            ax.set_ylim(max(means.min() / cond_pad, 1.0),
                        max(means.max() * cond_pad, 1.0 / floor * 1.5))
    return ax


def _probe_metric_grid(data, opts, scans, metrics, prov, transpose=False, logy=None):
    """The shared body of the three probe-metric grids: one panel per (scan, metric).

    ``transpose`` lays the scans out along the columns and the metrics down the rows,
    which is what the norms figure wants (two quantities, four scans).
    """
    outer = metrics if transpose else scans
    inner = scans if transpose else metrics
    ncol = int(opts['ncol']) or len(inner)
    fig, axes = _panels_for(opts, len(outer) * len(inner), ncol=ncol)
    for i, row in enumerate(outer):
        for j, col in enumerate(inner):
            scan, metric = (col, row) if transpose else (row, col)
            ax = axes[i * len(inner) + j]
            # the scan is named once per row -- or once per column, transposed
            named = (i if transpose else j) == 0
            _probe_panel(ax, data, scan, metric, logy=logy, marker=opts['marker'],
                         band=bool(opts['band']), cond_pad=float(opts['cond_pad']),
                         title=SCAN_SHORT.get(scan, scan) if named else None)
    if opts['show_legend']:
        _axes_legend(fig, ncol=opts['legend_ncol'], **opts['legend_kw'])
    return fig, {'axes': axes, 'provenance': prov, 'scans': list(scans),
                 'metrics': list(metrics)}


def _probe_metrics(data, opts):
    """Effective rank, condition number and ``sigma_B/sigma_1`` along four trajectories."""
    _require_probe(data, 'probe_metrics')
    scans = _scans_drawn(data, opts, pool=list(data.probe))
    _require_probe(data, 'probe_metrics', scans)
    metrics = list(opts['metrics'])
    prov = c.provenance(
        functions=['spectra_figs.probe_metrics', 'spectra_figs.plot_probe_metric',
                   'spectra_figs.effective_rank', 'ckpt_tools.load_spectra'],
        scans=[sf.probe_scan_name(s) for s in scans], reads=[str(ct.SPECTRA_DIR)],
        note='seed mean +/- 1 std over 5 cached seeds; cond is taken over the resolved '
             'part only and cannot exceed 1/floor, and a hollow marker means index B is '
             'outside the resolved rank')
    return _probe_metric_grid(data, opts, scans, metrics, prov)


def _probe_metrics_all(data, opts):
    """The same probe metrics plus the reachable residual fraction, every cached scan."""
    _require_probe(data, 'probe_metrics_all')
    scans = _scans_drawn(data, opts, pool=list(data.probe))
    _require_probe(data, 'probe_metrics_all', scans)
    metrics = list(opts['metrics'])
    prov = c.provenance(
        functions=['spectra_figs.probe_metrics', 'spectra_figs.plot_probe_metric',
                   'spectra_figs.effective_rank'],
        scans=[sf.probe_scan_name(s) for s in scans], reads=[str(ct.SPECTRA_DIR)],
        note='"Resolved rank" counts sigma_i above floor*sigma_max: on toy it is 14 of '
             '593 at initialisation, which is why that scan\'s condition number and '
             'sigma_B/sigma_1 are threshold artefacts and are marked as such')
    return _probe_metric_grid(data, opts, scans, metrics, prov)


def _probe_norms(data, opts):
    """Distance from initialisation and parameter norm: the min-norm claim, unconfirmed."""
    _require_probe(data, 'probe_norms')
    scans = _scans_drawn(data, opts, pool=list(data.probe))
    _require_probe(data, 'probe_norms', scans)
    metrics = list(opts['metrics'])
    prov = c.provenance(
        functions=['spectra_figs.probe_metrics', 'spectra_figs.plot_probe_metric',
                   'spectra_figs.low4_table'],
        scans=[sf.probe_scan_name(s) for s in scans], reads=[str(ct.SPECTRA_DIR)],
        note='the paired per-seed verdict is in tables_v2/spectra_low4.tex: Sven has the '
             'smallest mean distance from initialisation on 2 of 4 scans and the paired '
             '95 % interval excludes zero on none of them')
    return _probe_metric_grid(data, opts, scans, metrics, prov,
                              transpose=True, logy=bool(opts['logy']))


def _probe_energy(data, opts):
    """Cumulative PROBE residual vs directions kept -- the honest truncation figure."""
    _require_probe(data, 'probe_energy')
    scans = _scans_drawn(data, opts,
                         pool=[s for s in data.probe_methods
                               if 'Sven' in data.probe_methods[s]])
    _require_probe(data, 'probe_energy', scans)
    fig, axes = _panels_for(opts, len(scans))
    for ax, scan in zip(axes, scans):
        d = data.diags.get(scan)
        k = int(d.k) if d is not None else None
        seed = _probe_seed(scan)
        sf.plot_probe_energy_profile(ax, scan, k=(k if opts['show_k'] else None),
                                     limit=int(opts['limit']), seed=seed,
                                     cmap=opts['cmap'])
        ax.set_title(f'{SCAN_SHORT.get(scan, scan)}' + (f' ($k={k}$)' if k else ''))
        ax.set_ylabel('Probe residual kept')
        ax.set_xlabel('Directions kept $i$')
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        ents = ct.load_spectra(sf.probe_scan_name(scan), method='Sven', model_seed=seed)
        if ents and opts['annotate_steps']:
            # a cumulative curve rises left to right, so the free corner is the lower
            # right on three scans and the upper left on the one whose step-0 profile
            # already saturates -- let the data say which
            _step_endpoints(ax, np.asarray(ents[0]['step']),
                            loc=_step_label_loc(ax, opts), cmap=opts['cmap'])
        # the k line needs no label here: the panel title carries the value
    prov = c.provenance(
        functions=['spectra_figs.plot_probe_energy_profile', 'spectra_figs.probe_energy',
                   'spectra_figs.cumulative_energy'],
        scans=[sf.probe_scan_name(s) for s in scans], reads=[str(ct.SPECTRA_DIR)],
        note='the x axis runs to min(n_rows, P), not to B: a k that keeps ~100 % of a '
             '32-row batch residual keeps much less of the probe residual')
    return fig, {'axes': axes, 'provenance': prov, 'scans': scans}


# ---------------------------------------------------------------------------
# The registry: one entry per output stem, its knobs and their defaults
# ---------------------------------------------------------------------------
#: knobs every figure of this module has, so a notebook can change the panel grid of any
#: of them without knowing which builder drew it (see :func:`_panels_for`)
_GEOMETRY = {'ncol': 3, 'fraction': 0.32, 'aspect': c.DEFAULT_ASPECT, 'extra_h': 0.0,
             'sharex': False, 'sharey': False}


def _geom(**over):
    """:data:`_GEOMETRY` with this figure's values -- the grid it is drawn at today."""
    out = dict(_GEOMETRY)
    out.update(over)
    return out


FIGURE_SPECS = figspec.check_defaults({
    'spectrum_truncation': figspec.FigureSpec(
        draw=_spectrum_truncation,
        doc='F2, main 4.2: (a) the polynomial batch spectrum with the k / rtol / '
            'round-off cuts, (b) what the cut discards, (c) the rank inverted',
        defaults=dict(
            _geom(ncol=3, fraction=0.32, aspect=1.00, extra_h=0.30),
            spectrum_scan=F2_SPECTRUM_SCAN,     # whose spectrum panel (a) draws
            scans=[],                           # panels (b), (c); empty = every scan
            limit=40,                           # steps drawn in panel (a)
            lw=0.55,                            # one spectrum line
            scan_lw=1.3,                        # one scan's curve in (b), (c)
            cmap='plasma',                      # colour by optimizer step
            spectrum_headroom=[6.0, 1.4],        # y room below / above the spectrum
            smooth=SMOOTH_STEPS,                # moving average, in optimizer steps
            discard_floor=DISCARD_FLOOR,        # bottom of the log "discarded" axis
            show_k=True, show_rtol=True, show_noise=True,   # the three cut lines
            annotate_steps=True,                # the two "step N" labels in (a)
            show_legend=True, legend_ncol=4, legend_kw={})),
    'online_spectra': figspec.FigureSpec(
        draw=_online_spectra,
        doc='F9, App. O: the online per-batch spectrum over training, one panel per scan',
        defaults=dict(
            _geom(ncol=4, fraction=0.25, aspect=0.95),
            scans=[], limit=40, lw=0.6, cmap='plasma',
            spectrum_headroom=[4.0, 6.0],
            show_k=True, show_rtol=True, show_noise=True,
            k_loc='top',                        # where the "k=" label sits on its line
            annotate_steps=True, step_label_loc='lower left',
            short_titles=True, title_lines=2)),
    'online_utr': figspec.FigureSpec(
        draw=_online_utr,
        doc='F9, App. O: |u_i.r| / ||U^T r|| over training -- where the residual sits',
        defaults=dict(
            _geom(ncol=4, fraction=0.25, aspect=0.95),
            scans=[], limit=40, lw=0.6, cmap='plasma',
            spectrum_headroom=[3.0, 1.5],
            show_k=True,
            annotate_steps=True, step_label_loc='auto',
            short_titles=True, title_lines=2)),
    'online_mechanism': figspec.FigureSpec(
        draw=_online_mechanism,
        doc='F9, App. O: kept / discarded batch residual, the used rank, and its seed '
            'spread on the scan where it is widest',
        defaults=dict(
            _geom(ncol=2, fraction=0.49, aspect=0.78, extra_h=0.26),
            scans=[], scan_lw=1.3, smooth=SMOOTH_STEPS, discard_floor=DISCARD_FLOOR,
            spread_scan='',                     # empty = the widest seed spread
            seed_smooth=25, seed_lw=1.0,        # panel (d), per seed, no averaging
            seed_legend_kw={'ncol': 2, 'handlelength': 1.0, 'labelspacing': 0.15,
                            'fontsize': 6.5},
            show_legend=True, legend_ncol=7, legend_kw={})),
    'online_norms': figspec.FigureSpec(
        draw=_online_norms,
        doc='F9, App. O: applied update norm and batch residual norm, one panel per scan',
        defaults=dict(
            _geom(ncol=4, fraction=0.25, aspect=0.95),
            scans=[], smooth=25, band=True,     # band = the seed spread
            short_titles=True, title_lines=2,
            show_legend=True, legend_ncol=3, legend_lw=1.4, legend_kw={})),
    'mnist_lr_vs_ce': figspec.FigureSpec(
        draw=_lr_vs_ce,
        doc='F9, App. I: MNIST label regression against cross-entropy -- same model, '
            'same B, same splits, two spectra',
        defaults=dict(
            _geom(ncol=3, fraction=0.32, aspect=0.85, extra_h=0.30),
            scans=list(LR_VS_CE), marks=list(LR_VS_CE_MARKS),   # fractions of training
            lw=1.1, scan_lw=1.3, alpha_range=[0.30, 1.00],      # faint = early
            smooth=SMOOTH_STEPS, band_alpha=0.20, annot_fontsize=6.0,
            rank_headroom=1.9,                  # room above k for its label
            show_legend=True, legend_ncol=3, legend_kw={})),
    'used_rank_grid': figspec.FigureSpec(
        draw=_used_rank_grid,
        doc='F9, App. O: which cut binds over the tuning (k, rtol) grid, per scan',
        defaults=dict(
            _geom(ncol=3, fraction=0.33, aspect=1.05),
            scans=[], cmap='viridis', cell_fontsize=5.0,
            mark_selected=True)),               # the red square on the selected cell
    'probe_spectra': figspec.FigureSpec(
        draw=_probe_spectra,
        doc='F9, App. O: offline float64 probe spectra along each optimizer, one scan '
            'and one named seed',
        defaults=dict(
            _geom(ncol=2, fraction=0.49, aspect=0.82, sharey=True),
            scan='',                            # empty = the first PROBE_DEEP_SCANS hit
            limit=12, lw=1.3, cmap='plasma',
            show_k=True, k_lw=1.1,
            annotate_steps=True, step_label_loc='auto',
            scan_seed_title=True)),             # "<scan>, model seed N" over the panels
    'probe_spectra_all': figspec.FigureSpec(
        draw=_probe_spectra_all,
        doc='F9, App. O: the probe-spectrum figure over every cached scan x optimizer',
        defaults=dict(
            _geom(ncol=0, fraction=0.25, aspect=0.92),
            scans=[],                           # empty = every scan with a probe cache
            limit=12, lw=1.3, cmap='plasma',
            show_k=True, k_lw=1.0,
            annotate_steps=True, step_label_loc='lower left')),
    'probe_metrics': figspec.FigureSpec(
        draw=_probe_metrics,
        doc='F9, App. O: effective rank, condition number and sigma_B/sigma_1 along the '
            'trajectories of two scans',
        defaults=dict(
            _geom(ncol=0, fraction=0.32, aspect=0.88, extra_h=0.26),
            scans=list(PROBE_DEEP_SCANS), metrics=list(PROBE_METRICS),
            marker='.', band=True, cond_pad=5.0,
            show_legend=True, legend_ncol=6, legend_kw={})),
    'probe_metrics_all': figspec.FigureSpec(
        draw=_probe_metrics_all,
        doc='F9, App. O: the same probe metrics plus the resolved rank, every cached scan',
        defaults=dict(
            _geom(ncol=0, fraction=0.25, aspect=0.92, extra_h=0.24),
            scans=[], metrics=['rank'] + list(PROBE_METRICS),
            marker='.', band=True, cond_pad=5.0,
            show_legend=True, legend_ncol=6, legend_kw={})),
    'probe_norms': figspec.FigureSpec(
        draw=_probe_norms,
        doc='F9, App. O: distance from initialisation and parameter norm (C12), scans '
            'across the columns',
        defaults=dict(
            _geom(ncol=0, fraction=0.25, aspect=0.92, extra_h=0.24),
            scans=[], metrics=['dist_init', 'param_norm'],
            marker='.', band=True, cond_pad=5.0, logy=False,
            show_legend=True, legend_ncol=5, legend_kw={})),
    'probe_energy': figspec.FigureSpec(
        draw=_probe_energy,
        doc='F9, App. O: cumulative PROBE residual against directions kept, with k',
        defaults=dict(
            _geom(ncol=4, fraction=0.25, aspect=1.0),
            scans=[], limit=7, cmap='plasma', show_k=True,
            annotate_steps=True, step_label_loc='auto',
            step_label_fallback='lower right')),
})


# ---------------------------------------------------------------------------
# Derived views the tables and macros share
# ---------------------------------------------------------------------------
def _rank_spread(data):
    """Seed spread of the FINAL used rank per scan, largest first.

    The seed-mean panels hide it and it is the biggest of the online numbers: MNIST-CE's
    final used rank varies by ~69 % over seeds, nanoGPT's not at all.
    """
    rows = []
    for scan, arrs in data.arrays.items():
        vals = np.array([a['used_rank'][-1] for a in arrs], dtype=float)
        mean = float(vals.mean())
        rows.append({'scan': scan, 'mean': mean,
                     'std': float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                     'rel': (float(vals.std(ddof=1)) / abs(mean)
                             if len(vals) > 1 and mean else 0.0),
                     'n_seeds': len(vals)})
    out = pd.DataFrame(rows)
    return (out.sort_values('rel', ascending=False).reset_index(drop=True) if len(out)
            else out)


def _discard_table(data):
    """Batch residual DISCARDED at fixed fractions of TRAINING, per scan.

    Read at fractions of training, never of the logged-point index: the dense schedule
    logs every step to 1,000 and every 20th afterwards, so the two axes disagree by up to
    5x (B.fix WP3 finding 1 -- toy's jump is at 1-25 % of training, not at 80 %).
    """
    rows = []
    for scan in data.scans:
        x, y, _ = _progress_curve(data.diags[scan], data.arrays[scan], 'frac_used')
        if x is None:
            continue
        disc = np.clip(1.0 - y, DISCARD_FLOOR, None)
        rec = {'scan': scan}
        for mark in PROGRESS_MARKS:
            rec[mark] = float(disc[np.argmin(np.abs(x - mark))])
        rows.append(rec)
    return pd.DataFrame(rows)


def _probe_at_selected_k(data, when='last'):
    """Probe-set top-``k`` energy at each scan's SELECTED ``k``, one row per scan."""
    if not len(data.probe_energy):
        return pd.DataFrame()
    pe = data.probe_energy
    sel = pe[(pe['when'] == when) & (pe['k'] == pe['k_selected'])]
    return sel.reset_index(drop=True)


def _probe_endpoints(data, method='Sven'):
    """First / final checkpoint values of the probe metrics for one optimizer."""
    rows = []
    for scan, tbl in data.probe.items():
        sub = tbl[tbl['method'] == method]
        if not len(sub):
            continue
        first = sub[sub['step'] == sub['step'].min()]
        last = sub[sub['step'] == sub['step'].max()]
        rows.append({
            'scan': scan, 'n_rows': int(sub['n_rows'].iloc[0]),
            'n_params': int(sub['n_params'].iloc[0]),
            'width': int(min(sub['n_rows'].iloc[0], sub['n_params'].iloc[0])),
            'rank_first': float(first['rank'].mean()),
            'rank_final': float(last['rank'].mean()),
            'eff_rank_first': float(first['eff_rank'].mean()),
            'eff_rank_final': float(last['eff_rank'].mean()),
            'cond_final': float(last['cond'].mean()),
            'sigma_b': int(sub['sigma_b'].iloc[0]),
            'sigma_b_over_1_final': float(last['sigma_B_over_1'].mean()),
            'sigma_b_resolved': bool(sub['sigma_b_resolved'].all()),
            'floor': float(sub['floor'].iloc[0]),
            'proj_frac_final': float(last['proj_frac'].mean()),
            'n_ckpts': int(sub['step'].nunique()),
            'n_seeds': int(last['model_seed'].nunique()),
        })
    return pd.DataFrame(rows)


def _step0_spread(data):
    """Largest between-method disagreement at step 0, per scan.

    Every optimizer of a scan starts from the same checkpoint and the probe rows are
    identical, so this must be 0: it is the check that the trajectories are comparable.
    """
    rows = []
    for scan, tbl in data.probe.items():
        at0 = tbl[tbl['step'] == 0].groupby('method')[
            ['sigma_max', 'probe_loss', 'param_norm']].mean()
        rows.append({'scan': scan,
                     'spread': float((at0.max() - at0.min()).abs().max())
                     if len(at0) else np.nan,
                     'n_methods': int(len(at0))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T8 / T9
# ---------------------------------------------------------------------------
def _binds_cell(row):
    """``rtol`` / ``$k$`` / ``neither`` as a LaTeX cell (``mechanism_table``'s verdict)."""
    text = str(row['binds'])
    return c.Raw({'rtol': 'rtol', 'k': '$k$'}.get(text, r'neither ($k=B$)'))


def _binds_first(d, row):
    """Which cut binds at the FIRST logged step: ``k`` / ``rtol`` / ``neither``.

    :func:`spectra_figs.mechanism_table`'s ``binds`` is the verdict at the last logged
    step, and the two ends disagree wherever the rtol-rank falls during training -- which
    is the whole mechanism story -- so the paper needs both.
    """
    if not c.finite(row['used_first']):
        return 'unknown'
    at_cap = float(row['used_first']) >= int(d.k) - 0.5
    if at_cap:
        return 'neither' if int(d.k) >= int(d.B) else 'k'
    return 'rtol'


def _binds_moved(data):
    """The scans whose binding cut is not the same at the first and the last logged step,
    named for the caption, or ``no scan`` -- never asserted by hand."""
    mech = data.mech.set_index('scan')
    moved = [SCAN_SHORT.get(s, s) for s in data.scans
             if _binds_first(data.diags[s], mech.loc[data.diags[s].title])
             != {'rtol': 'rtol', 'k': 'k'}.get(
                 str(mech.loc[data.diags[s].title]['binds']), 'neither')]
    return ', '.join(moved) if moved else 'no scan'


def table_mechanism(data, name='spectra_mechanism'):
    """T8 (full): what the ``k`` / ``rtol`` cut does on every scan.

    Columns: the configuration on disk, which cut binds, the rank actually inverted and
    the batch residual kept at the first and last logged step, the probe-set top-``k``
    energy where a probe cache exists, the recorded Gram float32 floor, and the seed basis.
    """
    mech = data.mech.set_index('scan')
    probe = _probe_at_selected_k(data)
    spread = _rank_spread(data).set_index('scan')
    disc = _discard_table(data)
    disc = disc.set_index('scan') if len(disc) else disc
    rows = []
    for scan in data.scans:
        d = data.diags[scan]
        m = mech.loc[d.title]
        pe = probe[probe['scan'] == scan] if len(probe) else probe
        rows.append({
            'Scan': SCAN_SHORT.get(scan, d.title),
            c.Raw('$k$ / $B$'): c.Raw(f'${d.k}$ / ${d.B}$'),
            'rtol': c.Raw(_rtol_tex(d.rtol)),
            'Binds': _binds_cell(m),
            c.Raw(r'Used, first $\to$ final'): c.Raw(
                f'{_rank_cell(m["used_first"])} $\\to$ {_rank_cell(m["used_final"])}'),
            c.Raw(r'Kept, first $\to$ final'): c.Raw(
                f'{c.fmt_pct(m["energy_used_first"], 2)} $\\to$ '
                f'{c.fmt_pct(m["energy_used_final"], 2)}'),
            'Discarded, final': (_sci(float(disc.loc[scan, 1.0]))
                                 if len(disc) and scan in disc.index else c.Raw('--')),
            'Probe top-$k$': (c.fmt_pct(float(pe['top_k'].iloc[0]), 2) if len(pe)
                              else c.Raw('--')),
            'Rank spread': (c.fmt_pct(float(spread.loc[scan, 'rel']), 1)
                            if scan in spread.index else c.Raw('--')),
            'Seeds': c.fmt_counts(int(m['n_seeds']), data.seed_target.get(scan, 0)),
        })
    prov = c.provenance(
        functions=['spectra_figs.mechanism_table', 'spectra_figs.energy_in_top',
                   'spectra_figs.used_rank', 'spectra_figs.probe_energy',
                   'spectra_figs.noise_floor_rel', 'spectra_figs.seed_stack'],
        scans=[hl.dir_name(s, 'diag') for s in data.scans], reads=[c.results_root()],
        provisional=data.is_provisional(data.scans))
    path = c.write_table(
        name, rows, align=['l', 'r', 'r', 'c', 'c', 'c', 'r', 'r', 'r', 'c'], fit=True,
        provenance_record=prov,
        caption=('What the truncation does, per scan. "Used" is the number of directions '
                 'actually inverted, min(k, rtol-rank), at the first and last logged '
                 'step; "Kept" the fraction of that batch\'s residual they carry and '
                 '"Discarded" the remainder at the end of training; "Probe top-k" the '
                 'same fraction measured on the fixed float64 probe set at the same k. '
                 '"Rank spread" is the relative seed std of the final used rank. Seed '
                 'means over the diag pass.'),
        label='tab:spectra-mechanism',
        notes=['online per-batch Gram spectra (float32 model); the probe column is the '
               'offline float64 measurement and is a different object',
               '"Binds" is read at the LAST logged step; the verdict at the FIRST step '
               f'is the macro num<Scan>SvenBindsFirst and it DIFFERS on {_binds_moved(data)}'
               ', so a sentence about which cut binds must say when',
               'Seeds = the diag-pass seeds at the selected configuration / the seeds '
               'the pass intends',
               'the recorded Gram round-off floor is \\numSpectraNoiseFloorRel times '
               'sigma_max on every scan, and \\numSpectraRtolBelowFloor of the selected '
               'rtol values lies BELOW it -- worth a clause in the caption',
               'the "Discarded" column is floored at 1e-8, the point below which the cut '
               'throws away nothing float32 can see'])
    return path, rows


def table_mechanism_main(data, name='spectra_mechanism_main'):
    """T8 (main text): the same table, six columns wide."""
    mech = data.mech.set_index('scan')
    rows = []
    for scan in data.scans:
        d = data.diags[scan]
        m = mech.loc[d.title]
        rows.append({
            'Scan': d.title,
            c.Raw('$k$ / $B$'): c.Raw(f'${d.k}$ / ${d.B}$'),
            'rtol': c.Raw(f'${d.rtol:g}$'),
            'Binds': _binds_cell(m),
            c.Raw(r'Used first $\to$ final'): c.Raw(
                f'{_rank_cell(m["used_first"])} $\\to$ '
                f'{_rank_cell(m["used_final"])}'),
            c.Raw(r'Kept first $\to$ final'): c.Raw(
                f'{c.fmt_pct(m["energy_used_first"], 1)} $\\to$ '
                f'{c.fmt_pct(m["energy_used_final"], 1)}'),
        })
    prov = c.provenance(
        functions=['spectra_figs.mechanism_table'],
        scans=[hl.dir_name(s, 'diag') for s in data.scans], reads=[c.results_root()],
        provisional=data.is_provisional(data.scans))
    return c.write_table(
        name, rows, align=['l', 'r', 'r', 'c', 'c', 'c'], fit=True,
        provenance_record=prov,
        caption=('The rank Sven actually inverts is min(k, rtol-rank), and rtol is the '
                 'binding cut on five of the seven scans. "Kept" is the fraction of the '
                 'batch residual those directions carry, at the first and last logged '
                 'step.'),
        label='tab:spectra-mechanism-main'), rows


def table_probe_energy(data, name='spectra_probe_energy'):
    """T8 companion: probe-set top-``k`` energy at several ``k``, first and last checkpoint."""
    if not len(data.probe_energy):
        return None, []
    pe = data.probe_energy
    rows = []
    for scan in data.probe_methods:
        sub = pe[pe['scan'] == scan]
        if not len(sub):
            continue
        d = data.diags.get(scan)
        sub = sub.assign(_when=sub['when'].map({'first': 0, 'last': 1}).fillna(2))
        for _, r in sub.sort_values(['_when', 'k']).iterrows():
            rows.append({
                'Scan': data.title(scan),
                'Checkpoint': 'first' if r['when'] == 'first' else 'last',
                'Step': c.fmt_int(r['step']),
                '$k$': c.Raw(f'{int(r["k"])}' + (r'\,$^{\ast}$'
                                                 if d is not None and int(r['k']) == int(d.k)
                                                 else '')),
                'Top-$k$ energy': c.fmt_pct(r['top_k'], 3),
                'Seed std': c.fmt_pct(r['top_k_std'], 3),
                'Reachable': c.fmt_pct(r['in_span'], 3),
                'Seeds': c.fmt_int(r['n_seeds']),
            })
    prov = c.provenance(
        functions=['spectra_figs.probe_energy', 'spectra_figs.cumulative_energy',
                   'ckpt_tools.load_spectra'],
        scans=[sf.probe_scan_name(s) for s in data.probe_methods],
        reads=[str(ct.SPECTRA_DIR)])
    return c.write_table(
        name, rows, align=['l', 'l', 'r', 'r', 'r', 'r', 'r', 'c'],
        provenance_record=prov,
        caption=('Residual energy inside the top-k PROBE directions (offline, float64). '
                 '"Reachable" is ||P_U r||^2/||r||^2, the part of the probe residual the '
                 'Jacobian column space can reach at all -- below 1 only where the probe '
                 'set has more rows than the model has parameters. An asterisk marks the '
                 'selected k.'),
        label='tab:spectra-probe-energy',
        notes=['offline float64 Jacobian on a fixed probe set: a different object from '
               'the per-batch online spectra']), rows


def table_low4(data, name='spectra_low4'):
    """T9: distance from initialisation and parameter norm, paired, with the verdict."""
    if not len(data.low4):
        return None, []
    metric_name = {'dist_init': r'$\|\theta_t-\theta_0\|$', 'param_norm': r'$\|\theta_t\|$'}
    rows = []
    for _, r in data.low4.iterrows():
        half = r['paired_half']
        rows.append({
            'Scan': r['scan'],
            'Quantity': c.Raw(metric_name.get(r['metric'], r['metric'])),
            'Sven': c.fmt_pm(r['Sven'], r['Sven_std'], sig=4),
            'Best other': style.method_label(r['best_other']),
            c.Raw('Its mean'): c.fmt_sig(r['best_other_mean'], 4),
            'Rank': c.fmt_rank(r['rank'], r['n_methods']),
            'Paired diff.': c.fmt_pm(r['paired_mean'], r['paired_std']),
            '95% interval': (c.fmt_ci(r['paired_mean'] - half, r['paired_mean'] + half)
                                if c.finite(half) else c.Raw('--')),
            'Seeds lower': c.fmt_counts(r['n_ref_lower'], r['n_paired']),
            'Verdict': str(r['verdict']),
        })
    prov = c.provenance(
        functions=['spectra_figs.low4_table', 'spectra_figs.low4_verdict',
                   'spectra_figs.probe_metrics', 'paired.paired_difference'],
        scans=[sf.probe_scan_name(s) for s in data.probe], reads=[str(ct.SPECTRA_DIR)])
    return c.write_table(
        name, rows, align=['l', 'l', 'r', 'l', 'r', 'c', 'r', 'c', 'c', 'l'], fit=True,
        provenance_record=prov,
        caption=('Does Sven stay closer to its initialisation, or reach a smaller-norm '
                 'solution? At the last cached checkpoint, Sven against the best of the '
                 'other three optimizers, paired by model seed (same initialisation, same '
                 'probe rows). A rank of 1 whose paired 95 \\% interval covers zero is not '
                 'a win, and the verdict column says which is which.'),
        label='tab:spectra-low4',
        notes=['paired t-interval over 5 seeds; "not resolved" = the interval covers '
               'zero']), rows


def table_probe_widths(data, name='spectra_probe_widths'):
    """T9 companion: the probe sets' dimensions, measured rather than assumed."""
    if not len(data.widths):
        return None, []
    ends = _probe_endpoints(data).set_index('scan')
    step0 = _step0_spread(data).set_index('scan')
    rows = []
    for _, r in data.widths.iterrows():
        scan = r['scan']
        e = ends.loc[scan] if scan in ends.index else None
        rows.append({
            'Scan': data.title(scan),
            'Probe rows': _thousands(r['n_rows']),
            '$P$': _thousands(r['n_params']),
            'Spectrum width': _thousands(r['width'] if np.isscalar(r['width'])
                                         else min(r['width'])),
            'Resolved rank, final': (_rank_cell(e['rank_final']) if e is not None
                                     else c.Raw('--')),
            'Checkpoints': c.fmt_int(r['n_ckpts']),
            'Optimizers': c.fmt_int(r['methods']),
            'Seeds': c.fmt_int(r['seeds']),
            'Step-0 spread': (c.fmt_sig(float(step0.loc[scan, 'spread']), 2)
                              if scan in step0.index else c.Raw('--')),
        })
    prov = c.provenance(
        functions=['spectra_figs.probe_widths', 'spectra_figs.probe_metrics',
                   'ckpt_tools.load_spectra'],
        scans=[sf.probe_scan_name(s) for s in data.probe_methods],
        reads=[str(ct.SPECTRA_DIR)])
    return c.write_table(
        name, rows, align=['l', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r'], fit=True,
        provenance_record=prov,
        caption=('The cached probe sets. The spectrum width is min(probe rows, P) and only '
                 'one of the two bounds is ever active: on MNIST the probe set is 512 rows '
                 'against 27{,}562 parameters, so the spectrum resolves 512 directions, '
                 'not 27k. "Resolved rank" counts sigma_i above the float64 threshold. '
                 '"Step-0 spread" is the largest disagreement between the four optimizers '
                 'at the shared initialisation, which must be zero for the trajectories '
                 'to be comparable.'),
        label='tab:spectra-probe-widths'), rows


# ---------------------------------------------------------------------------
# G4 -- the number macros
# ---------------------------------------------------------------------------
def macros(data):
    """Macro group G4: every spectrum number the revision quotes.

    ``\\num<Scan>Sven<Quantity>`` with letters only, one per number, each carrying the
    analysis function it came from as a comment.  A scan whose inputs are still refreshing
    marks its macros ``PROVISIONAL``.
    """
    m = c.Macros(module=MODULE)
    mech = data.mech.set_index('scan')
    spread = _rank_spread(data).set_index('scan')
    disc = _discard_table(data)
    disc = disc.set_index('scan') if len(disc) else disc
    probe_sel = _probe_at_selected_k(data)
    ends = _probe_endpoints(data).set_index('scan')
    src_mech = 'spectra_figs.mechanism_table'
    src_probe = 'spectra_figs.probe_energy'

    # --- the two structural facts the section rests on -------------------------------
    m.add('numSpectraScans', len(data.scans), source=src_mech,
          note='scans with a dense-spectra diag pass')
    m.add('numSpectraRankLawSteps', _thousands(data.rank_law_steps),
          source='spectra_figs.used_rank vs the optimizer num_nonzero_svs',
          note='logged steps on which min(k, rtol-rank) == num_nonzero_svs')
    binds = mech['binds'].value_counts()
    m.add('numSpectraRtolBinds', int(binds.get('rtol', 0)), source=src_mech,
          note='scans on which rtol is the binding cut')
    m.add('numSpectraKBinds', int(binds.get('k', 0)) + int(binds.get('neither ($k=B$)', 0)),
          source=src_mech, note='scans on which rtol never binds (k = B, or k binds)')
    m.add('numSpectraNoiseFloorRel', _sci(float(np.nanmedian(mech['noise_rel']))),
          source='spectra_figs.noise_floor_rel',
          note='median recorded Gram float32 floor, sqrt(eps)*sigma_max')
    m.add('numSpectraNeitherBinds', int(binds.get('neither ($k=B$)', 0)),
          source=src_mech,
          note='scans on which NEITHER cut binds at the last step (k = B and the whole '
               'resolved spectrum clears rtol*sigma_1) -- what numSpectraKBinds counts '
               'when no scan is k-bound, which is why the prose must not call it "k binds"')
    m.add('numSpectraKBindsOnly', int(binds.get('k', 0)), source=src_mech,
          note='scans on which the rank cap k is the binding cut at the last step')
    below = int((pd.to_numeric(mech['rtol']) < pd.to_numeric(mech['noise_rel'])).sum())
    m.add('numSpectraRtolBelowFloor', below, source=src_mech,
          note='scans whose selected rtol lies BELOW the recorded Gram round-off floor, '
               'so the rtol cut is made inside the noise')
    below_names = _name_list(mech.index[(pd.to_numeric(mech['rtol'])
                                        < pd.to_numeric(mech['noise_rel'])).values])
    if below_names:                     # an empty list must not reach a macro
        m.add('numSpectraRtolBelowFloorScans', c.Raw(below_names),
              source=src_mech, note='which scans those are')
    # --- how the kept energy moves, counted rather than asserted ---------------------
    # The prose used to say the cut "bites" on two scans and explain the three pinned
    # ones by an rtol below the round-off floor.  Both are wrong: the cut discards
    # measurable energy on FOUR scans (rising on two, falling on two) and the pinned
    # scans all have rtol ABOVE the floor.  These five counts are the honest version.
    e_first = pd.to_numeric(mech['energy_used_first'], errors='coerce')
    e_final = pd.to_numeric(mech['energy_used_final'], errors='coerce')
    pinned = e_final >= 1.0 - ENERGY_PINNED_TOL
    bites = ~pinned & e_final.notna()
    m.add('numSpectraEnergyPinned', int(pinned.sum()), source=src_mech,
          note=f'scans whose kept energy stays within {ENERGY_PINNED_TOL:g} of 100%: '
               'the cut discards nothing measurable there')
    pinned_names = _name_list(mech.index[pinned.values])
    if pinned_names:
        m.add('numSpectraEnergyPinnedScans', c.Raw(pinned_names),
              source=src_mech, note='which scans those are')
    m.add('numSpectraCutBites', int(bites.sum()), source=src_mech,
          note='scans on which the cut discards a measurable share of the residual')
    m.add('numSpectraEnergyRises', int((bites & (e_final > e_first)).sum()),
          source=src_mech, note='of those, the ones whose kept energy rises over training')
    m.add('numSpectraEnergyFalls', int((bites & (e_final < e_first)).sum()),
          source=src_mech, note='of those, the ones whose kept energy falls')
    if pinned.any():
        m.add('numSpectraPinnedDiscardMax', _sci(float((1.0 - e_final[pinned]).max())),
              source=src_mech,
              note='the largest fraction of the residual discarded on any pinned scan '
                   '-- the reason they read as 100%')

    for scan in data.scans:
        key = SCAN_KEY.get(scan)
        if key is None:
            continue
        d = data.diags[scan]
        row = mech.loc[d.title]
        prov = bool(scan in data.provisional)
        for suffix, value, sig, source, note in (
                # `KCap`, not `K`: this is the RANK CAP, which C9 must keep separate from
                # the rank actually used, and `large.py` already owns
                # ``\numCifar{LR,CE}SvenK`` for App. J -- two \newcommand lines with the
                # same name in the files numbers_v2.tex inputs together is a build error.
                ('SvenKCap', d.k, 3, 'headline.selection_methods',
                 'selected k, the rank cap (see SvenUsedRank* for the rank used)'),
                ('SvenB', d.B, 3, 'record batch_size', 'batch size'),
                ('SvenUsedRankFirst', row['used_first'], 3, src_mech,
                 'directions inverted at the first logged step'),
                ('SvenUsedRankFinal', row['used_final'], 3, src_mech,
                 'directions inverted at the last logged step'),
                ('SvenEnergyFirst', 100 * row['energy_used_first'], 3, src_mech,
                 'percent of the batch residual kept, first logged step'),
                ('SvenEnergyFinal', 100 * row['energy_used_final'], 3, src_mech,
                 'percent of the batch residual kept, last logged step'),
                ('SvenEnergyTopKFinal', 100 * row['energy_top_k_final'], 3, src_mech,
                 'percent kept by the top-k cut alone, last logged step'),
                ('SvenDiagSeeds', row['n_seeds'], 2, src_mech,
                 'Sven diag seeds at the selected configuration')):
            if c.finite(value):
                m.add(f'num{key}{suffix}', float(value), source=source, note=note,
                      sig=sig, provisional=prov)
        m.add(f'num{key}SvenRtol', c.Raw(f'{d.rtol:g}'),
              source='headline.selection_methods', note='selected rtol', provisional=prov)
        m.add(f'num{key}SvenBinds', {'rtol': 'rtol', 'k': 'k'}.get(str(row['binds']),
                                                                   'neither'),
              source=src_mech, note='which cut binds at the last logged step',
              provisional=prov)
        # the same verdict at the FIRST logged step: the two ends disagree on CIFAR-10
        # label regression (all B directions in use at step 0, rtol binding by the end),
        # so a claim about "which cut binds" needs both ends or it is wrong at one of them
        m.add(f'num{key}SvenBindsFirst', _binds_first(d, row),
              source=src_mech, note='which cut binds at the first logged step',
              provisional=prov)
        if scan in spread.index:
            m.add(f'num{key}SvenUsedRankSpread',
                  100 * float(spread.loc[scan, 'rel']), source=src_mech, sig=3,
                  note='relative seed std of the final used rank, percent',
                  provisional=prov)
        if len(disc) and scan in disc.index:
            m.add(f'num{key}SvenDiscardFinal', float(disc.loc[scan, 1.0]),
                  source='spectra_figs.seed_stack + smooth_steps', sig=2,
                  note='batch residual discarded at the end of training '
                       f'(floor {DISCARD_FLOOR:g})', provisional=prov)
        if len(probe_sel):
            pe = probe_sel[probe_sel['scan'] == scan]
            if len(pe):
                m.add(f'num{key}SvenProbeEnergy', 100 * float(pe['top_k'].iloc[0]),
                      source=src_probe, sig=4,
                      note='percent of the PROBE residual in the top-k directions, '
                           'last checkpoint')
                m.add(f'num{key}SvenProbeSpan', 100 * float(pe['in_span'].iloc[0]),
                      source=src_probe, sig=4,
                      note='percent of the probe residual the Jacobian column space '
                           'can reach')
        if scan in ends.index:
            e = ends.loc[scan]
            m.add(f'num{key}SvenProbeRows', _thousands(e['n_rows']),
                  source='spectra_figs.probe_widths', note='probe-set rows')
            m.add(f'num{key}SvenProbeWidth', _thousands(e['width']),
                  source='spectra_figs.probe_widths',
                  note='probe spectrum width, min(rows, P) -- MEASURED from svals.shape')
            for suffix, value, sig, note in (
                    ('SvenProbeRankFirst', e['rank_first'], 3,
                     'numerically resolved rank at the first checkpoint'),
                    ('SvenProbeRankFinal', e['rank_final'], 3,
                     'numerically resolved rank at the last checkpoint'),
                    ('SvenProbeEffRankFinal', e['eff_rank_final'], 3,
                     'effective rank exp(H(sigma)) at the last checkpoint'),
                    ('SvenProbeCondFinal', e['cond_final'], 3,
                     'condition number over the resolved part, last checkpoint'),
                    ('SvenProbeProjFinal', 100 * e['proj_frac_final'], 4,
                     'percent of the probe residual inside the column space')):
                if c.finite(value):
                    m.add(f'num{key}{suffix}', float(value),
                          source='spectra_figs.probe_metrics', note=note, sig=sig)

    # --- the probe-set resolution ceiling and the shared-initialisation check --------
    if len(ends):
        m.add('numSpectraProbeFloor',
              c.Raw(f'$10^{{{int(round(math.log10(float(ends["floor"].iloc[0]))))}}}$'),
              source='spectra_figs.probe_metrics',
              note='relative singular value below which a float64 probe spectrum is '
                   'round-off')
        m.add('numSpectraProbeCeiling',
              c.Raw(f'$10^{{{int(round(-math.log10(float(ends["floor"].iloc[0]))))}}}$'),
              source='spectra_figs.probe_metrics',
              note='1 / floor: the largest condition number the probe set can resolve')
    step0 = _step0_spread(data)
    if len(step0):
        m.add('numSpectraStepZeroSpread',
              c.Raw(f'${float(step0["spread"].max()):g}$'),
              source='spectra_figs.probe_metrics',
              note='largest between-optimizer disagreement at the shared initialisation')
        m.add('numSpectraProbeScans', len(step0), source='spectra_figs.probe_widths',
              note='scans with cached probe-set trajectories')

    # --- C12: the min-norm story, unconfirmed ---------------------------------------
    if len(data.low4):
        for metric, tag in (('dist_init', 'DistInit'), ('param_norm', 'ParamNorm')):
            sub = data.low4[data.low4['metric'] == metric]
            if not len(sub):
                continue
            small = sub[sub['rank'] == 1]
            m.add(f'numSpectraSven{tag}Smallest', len(small),
                  source='spectra_figs.low4_table',
                  note=f'scans of {len(sub)} on which Sven has the smallest mean {metric}')
            m.add(f'numSpectraSven{tag}Scans', len(sub),
                  source='spectra_figs.low4_table', note='scans compared')
            m.add(f'numSpectraSven{tag}Resolved',
                  int(small['resolved'].sum()) if len(small) else 0,
                  source='spectra_figs.low4_table',
                  note='of those, the paired 95 % interval excluding zero')
            best = sub.loc[sub['paired_mean'].idxmin()]
            m.add(f'numSpectraSven{tag}BestScan', str(best['scan']),
                  source='spectra_figs.low4_table',
                  note='scan with the most negative paired difference')
            m.add(f'numSpectraSven{tag}BestDiff', float(best['paired_mean']),
                  source='spectra_figs.low4_table', sig=3,
                  note='its paired difference (Sven minus the best other optimizer)')
            if c.finite(best['paired_half']):
                m.add(f'numSpectraSven{tag}BestHalf', float(best['paired_half']),
                      source='spectra_figs.low4_table', sig=2,
                      note='half-width of its paired 95 % t-interval')
    return m


# ---------------------------------------------------------------------------
# Collision guard
# ---------------------------------------------------------------------------
_NEWCOMMAND = re.compile(r'\\newcommand\{\\([A-Za-z]+)\}')


def macro_collisions(path=None):
    """Macro names this module shares with another module's already-written file.

    Four modules write four macro files that the manuscript ``\\input``s together, so a
    name defined twice is a LaTeX error at build time rather than a silent overwrite.
    This reports it at generation time instead.
    """
    mine = Path(path or c.NUM_DIR / f'numbers_v2_{MODULE}.tex')
    names = set(_NEWCOMMAND.findall(mine.read_text())) if mine.is_file() else set()
    out = {}
    for other in sorted(c.NUM_DIR.glob('numbers_v2_*.tex')):
        if other == mine:
            continue
        shared = names & set(_NEWCOMMAND.findall(other.read_text()))
        if shared:
            out[other.name] = sorted(shared)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
#: the figures that rest on the OFFLINE probe cache (``analysis/ckpt_spectra/``).
#: :func:`build` skips them when nothing is cached -- that skip is the only reason the
#: distinction between them and the online figures ever existed as two tuples.
PROBE_FIGURES = ('probe_spectra', 'probe_spectra_all', 'probe_metrics',
                 'probe_metrics_all', 'probe_norms', 'probe_energy')
TABLES = (table_mechanism, table_mechanism_main, table_probe_energy, table_low4,
          table_probe_widths)

#: :func:`load` is expensive (every diag pass, plus the cached checkpoint spectra), so
#: the draw functions share one :class:`Data` per results root
_CONTEXT: dict = {}


def _ctx_key(root):
    return str(root) if root else ''


def context(root=None, reload=False, verbose=True):
    """The :class:`Data` every draw function takes, read once per results root.

    The module half of the figure contract (``campaign/FIGURE_API_CONTRACT.md``): cheap
    to call twice, so a notebook can draw one figure after another without re-reading the
    diag passes.  ``reload=True`` re-reads, which is what to do after a pass finishes.
    """
    key = _ctx_key(root)
    if reload or key not in _CONTEXT:
        _CONTEXT[key] = load(root=root, verbose=verbose)
    return _CONTEXT[key]


def _draw_and_save(name, spec, data, png=True):
    """Draw one figure with its resolved options and write it: the module's ONE save path.

    The save happens inside the same rcParams state as the draw on purpose.
    :func:`common.set_paper_style` (which :func:`_panels` calls) sets ``savefig.bbox``,
    ``savefig.pad_inches`` and ``pdf.fonttype`` -- Type 42, not matplotlib's default Type
    3 -- and ``fig.savefig`` reads those at save time, so restoring the rcParams between
    drawing and saving would change the bytes that reach the page.
    """
    import matplotlib.pyplot as plt

    opts = figspec.figure_opts(name, spec.defaults)
    rc = opts.get('rc') or {}
    with plt.rc_context(rc) if rc else contextlib.nullcontext():
        fig, meta = spec.draw(data, opts)
        figspec.apply_opts(fig, meta.get('axes'), opts)
        return c.save_fig(fig, name, GROUP, provenance_record=meta['provenance'],
                          png=png)


def build(root=None, figures=True, tables=True, numbers=True, data=None, verbose=True,
          dry_run=False):
    """Build F2, F9, T8, T9 and G4.  Returns a report dict.

    ``dry_run`` reports what would be written and writes nothing, which is the module
    protocol ``__main__`` calls every module with (``main``, ``reviewer``, ``large`` and
    ``campaign`` take the same keyword).  Before this keyword existed the whole spectra
    layer aborted with a ``TypeError`` on every ``python -m paper_assets`` run.
    """
    if verbose:
        c.banner(MODULE, 'F2 F9 (figures) - T8 T9 (tables) - G4 (numbers)')
    report = {'module': MODULE, 'figures': [], 'tables': [], 'macros': 0,
              'n_macros': 0, 'provisional': [], 'notes': [], 'status': ''}
    if dry_run:
        # the FIGURE NAMES, which are the PDF stems: the dry run used to report the
        # builders' function names (`fig_online_utr.pdf`) and the build then wrote
        # `online_utr.pdf`, so its file list named thirteen files that never existed
        report['figures'] = [f'{name}.pdf' for name in FIGURE_SPECS]
        report['tables'] = [f'{fn.__name__}.tex' for fn in TABLES]
        report['status'] = 'dry run'
        return report
    if data is None:
        data = load(root=root, verbose=verbose)
        _CONTEXT[_ctx_key(root)] = data          # so a notebook reuses this read
    if not data.diags:
        raise RuntimeError('no diag pass could be read; nothing to build')
    if data.provisional and not c.allow_provisional():
        raise RuntimeError(f'provisional scans {data.provisional} and '
                           f'PAPER_ASSETS_STRICT=1')
    report.update(provisional=list(data.provisional), notes=list(data.notes))
    if figures:
        for name, spec in FIGURE_SPECS.items():
            if name in PROBE_FIGURES and not data.probe_methods:
                report['notes'].append(f'{name}: no cached probe spectra')
                continue
            pdf, png = _draw_and_save(name, spec, data)
            report['figures'].append(str(pdf))
            if verbose:
                print(f'  fig  {pdf.relative_to(c.MANUSCRIPT)}   (png: {png})')
    if tables:
        for fn in TABLES:
            path, rows = fn(data)
            if path is None:
                report['notes'].append(f'{fn.__name__}: no input')
                continue
            report['tables'].append(str(path))
            if verbose:
                print(f'  tab  {path.relative_to(c.MANUSCRIPT)}   ({len(rows)} rows)')
    if numbers:
        book = macros(data)
        prov = c.provenance(
            functions=['spectra_figs.mechanism_table', 'spectra_figs.probe_energy',
                       'spectra_figs.probe_metrics', 'spectra_figs.low4_table',
                       'spectra_figs.probe_widths', 'spectra_figs.used_rank_grid'],
            scans=([hl.dir_name(s, 'diag') for s in data.scans]
                   + [sf.probe_scan_name(s) for s in data.probe_methods]),
            reads=[c.results_root(), str(ct.SPECTRA_DIR)],
            provisional=list(data.provisional))
        path = book.write(provenance=prov)
        report['macros'] = report['n_macros'] = len(book)
        report['numbers_file'] = str(path)
        collisions = macro_collisions(path)
        report['macro_collisions'] = collisions
        if verbose:
            print(f'  num  {path.relative_to(c.MANUSCRIPT)}   ({len(book)} macros, '
                  f'{len(book.provisional)} provisional)')
            if collisions:
                print(f'  ** macro name collisions with another module: {collisions}')
    return report


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description='Build the spectra assets (F2 F9 T8 T9 G4).')
    ap.add_argument('--figures-only', action='store_true')
    ap.add_argument('--tables-only', action='store_true')
    ap.add_argument('--numbers-only', action='store_true')
    ap.add_argument('--root', default=None, help='results root override')
    args = ap.parse_args(argv)
    only = (args.figures_only, args.tables_only, args.numbers_only)
    report = build(root=args.root,
                   figures=args.figures_only or not any(only),
                   tables=args.tables_only or not any(only),
                   numbers=args.numbers_only or not any(only))
    print(f'\n{len(report["figures"])} figures, {len(report["tables"])} tables, '
          f'{report["macros"]} macros'
          + (f'; PROVISIONAL: {", ".join(report["provisional"])}'
             if report['provisional'] else ''))
    return 0


if __name__ == '__main__':      # pragma: no cover
    sys.exit(main())
