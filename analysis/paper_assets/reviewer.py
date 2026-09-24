"""``paper_assets.reviewer`` -- the reviewer-study assets of the ICLR revision.

Owns (``campaign/PAPER_PLAN.md`` section 5.4):

=====  ==========================================  ==========================
asset  file                                        appendix
=====  ==========================================  ==========================
F4     ``figures_iclr/reviewer/budget.pdf``         F  (tuning budget)
F5     ``figures_iclr/reviewer/overparam.pdf``      G  (P > N)
F5     ``figures_iclr/reviewer/overparam_losses.pdf``   G (train / val / test, as carried)
F6     ``figures_iclr/reviewer/batchsize.pdf``      H  (batch size)
F8     ``figures_iclr/reviewer/kappa.pdf``          M  (kappa)
F10    ``figures_iclr/reviewer/divergence.pdf``     P  (robustness)
F10b   ``figures_iclr/reviewer/divergence_grids.pdf``    P (per-grid maps)
F14    ``figures_iclr/reviewer/knobs.pdf``          N  (micro-batch / param fraction)
T6     ``tables_v2/{budget,equal_budget}.tex``       F  (``optimism.tex``: main)
T11    ``tables_v2/divergence.tex``                 P
T12    ``tables_v2/{overparam,overparam_loss}.tex`` G
T13    ``tables_v2/{batchsize,batchsize_loss}.tex`` H
T14    ``tables_v2/kappa.tex``                      M
T21    ``tables_v2/knobs.tex``                      N
G2     ``numbers_v2_reviewer.tex``                  (every number App. F-H, M, N, P quotes)
=====  ==========================================  ==========================

**No number is computed here.**  Every value comes from the analysis functions of record
--- :mod:`reviewer_figs` (per-arm selection, ranks, reach tables, paired gaps, the kappa
and knob tables, divergence rates), :mod:`budget` and :mod:`headline`/:mod:`headline_figs`
(trajectory counts, best-of-n, equal budget, selection optimism, campaign-wide divergence)
--- and is only *formatted* here.  Figures are drawn from those same frames.

Two deliberate deviations, both cosmetic:

* the line widths of :func:`reviewer_figs.plot_arm` and :func:`budget.plot_best_of_n`
  (2.2--2.6 pt) are sized for an 8x6 in notebook panel; on a 1.76 in panel they cover the
  data, so the artists are re-scaled after the fact by :func:`_thin` --- the plotted
  values are untouched;
* a figure-level legend is placed by :func:`_legend_below` (constrained layout) rather
  than :func:`reviewer_figs.figure_legend` (which calls ``tight_layout``).

**The figures follow the registry contract** (``campaign/FIGURE_API_CONTRACT.md``):
:data:`FIGURE_SPECS` names all eight, each ``draw(ctx, opts) -> (fig, meta)`` writing
nothing, so one panel can be redrawn and re-tuned from ``analysis/notebooks/paper/``
without rebuilding the appendix::

    import paper_assets.notebook as pf
    pf.info('batchsize')                       # its knobs and what is pinned
    f = pf.make('batchsize', cmap='cividis')   # draws, writes nothing
    f.save()                                   # PDF + PNG + provenance sidecar

``ctx`` is the :class:`_Data` object :func:`context` returns: it memoises both the scan
frames and the per-section analysis frames (``ctx.overparam()``, ``ctx.batchsize()``,
...), which is what lets any single figure be drawn on its own while :func:`build` still
computes each section exactly once.

Run it alone with ``cd analysis && ../.venv/bin/python -m paper_assets.reviewer``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:                                   # the module is ``common.py``; the plan calls it ``_common``
    from paper_assets import common as C
except ImportError:                    # pragma: no cover
    from paper_assets import _common as C

from paper_assets import figspec       # the registry + the notebooks' override layer

import analysis_helpers as ah          # noqa: E402  (C puts analysis/ on sys.path)
import budget                          # noqa: E402
import headline as hl                  # noqa: E402
import headline_figs as hf             # noqa: E402
import paired                          # noqa: E402
import reviewer_figs as rf             # noqa: E402
import style                           # noqa: E402

MODULE = 'reviewer'
GROUP = 'reviewer'                     # figures_iclr/<GROUP>/
SVEN = rf.SVEN

# ---------------------------------------------------------------------------
# The scans this module reads (EXPERIMENTS.md sections 3.1-3.5)
# ---------------------------------------------------------------------------
#: App. G -- dataset-level over-parameterisation, one scan per task
OVERPARAM_SCANS = (('toy_1d', 'rebuttal_overparam_toy_1d_scan'),
                   ('polynomial', 'rebuttal_overparam_polynomial_scan'),
                   ('mnist_labelreg', 'rebuttal_overparam_mnist_scan'))
#: App. H -- batch-size sensitivity (R2's Q1), on the polynomial task
BATCHSIZE_SCAN = 'rebuttal_batchsize_polynomial_scan'
#: App. P -- the sweeps whose Sven grids F10/F10b show beside the headline grids: their
#: shared learning-rate axes reach values that break several methods, so they are where
#: the diverged fraction gets high
DIVERGENCE_SWEEP_SCANS = tuple(s for _, s in OVERPARAM_SCANS) + (BATCHSIZE_SCAN,)
#: App. M -- kappa at matched effective step
KAPPA_SCAN = 'mnist_kappaScan_labelRegression'
#: App. N -- the two memory knobs, four MLP scans each
MICROBATCH_SCANS = (('toy_1d', 'toy_1d_microbatch_scan'),
                    ('polynomial', 'polynomial_microbatch_scan'),
                    ('mnist_labelreg', 'mnist_microbatch_labelreg_scan'),
                    ('mnist_ce', 'mnist_microbatch_ce_scan'))
PARAMFRAC_SCANS = (('toy_1d', 'toy_1d_paramfrac_scan'),
                   ('polynomial', 'polynomial_paramfrac_scan'),
                   ('mnist_labelreg', 'mnist_paramfrac_labelreg_scan'),
                   ('mnist_ce', 'mnist_paramfrac_ce_scan'))
#: App. F -- the tuning grids whose budget is disclosed (the four MLP headline scans)
BUDGET_SCANS = ('toy_1d_scan', 'polynomial_scan', 'mnist_scan_labelRegression',
                'mnist_scan_ce')
#: App. F -- the two panels of F4 (the scans whose best-of-n curves are drawn)
BUDGET_PANELS = ('polynomial_scan', 'mnist_scan_labelRegression')

#: ``rtol`` is a code identifier, and the manuscript sets it as ``\texttt{rtol}`` in every
#: one of its 14 mentions.  In a matplotlib label LaTeX is not available, so the same name
#: is set in mathtext typewriter; the earlier ``\`rtol'`` spelling is LaTeX's quoting
#: convention and matplotlib printed the backtick and the apostrophe literally.
RTOL_TEX = r'\texttt{rtol}'          # goes into a table cell, header or caption
RTOL_FIG = r'$\mathtt{rtol}$'        # goes into an axis label, title or legend entry

#: task key -> the plan's macro scan key (section 5.3)
TASK_KEY = {'toy_1d': 'Toy', 'polynomial': 'Poly', 'mnist_labelreg': 'MnistLR',
            'mnist_ce': 'MnistCE'}
SCAN_TASK = {'toy_1d_scan': 'toy_1d', 'polynomial_scan': 'polynomial',
             'mnist_scan_labelRegression': 'mnist_labelreg',
             'mnist_scan_ce': 'mnist_ce'}

#: reader-facing names for the scan directories these appendices quote.  No underscores:
#: they end up inside macro bodies and table cells.
SCAN_LABELS = {
    'toy_1d_scan': 'Toy 1D (tuning grid)',
    'polynomial_scan': 'Polynomial (tuning grid)',
    'mnist_scan_labelRegression': 'MNIST label reg. (tuning grid)',
    'mnist_scan_ce': 'MNIST CE (tuning grid)',
    'cifar10_resnet_scan_labelRegression': 'CIFAR-10 label reg.',
    'cifar10_resnet_ce_scan': 'CIFAR-10 CE',
    'exp_nanogpt_speedrun': 'nanoGPT',
    'exp_gpt2_small_comparison': 'GPT-2 small',
    'rebuttal_overparam_toy_1d_scan': 'Toy 1D, $P/N$ sweep',
    'rebuttal_overparam_polynomial_scan': 'Polynomial, $P/N$ sweep',
    'rebuttal_overparam_mnist_scan': 'MNIST, $P/N$ sweep',
    'rebuttal_batchsize_polynomial_scan': 'Polynomial, batch-size sweep',
    'rebuttal_fig5_cifar_paramfrac_scan': 'CIFAR-10, parameter fraction',
    'mnist_kappaScan_labelRegression': r'MNIST, $\kappa$ sweep',
    'toy_1d_microbatch_scan': 'Toy 1D, micro-batch',
    'polynomial_microbatch_scan': 'Polynomial, micro-batch',
    'mnist_microbatch_labelreg_scan': 'MNIST label reg., micro-batch',
    'mnist_microbatch_ce_scan': 'MNIST CE, micro-batch',
    'toy_1d_paramfrac_scan': 'Toy 1D, param. fraction',
    'polynomial_paramfrac_scan': 'Polynomial, param. fraction',
    'mnist_paramfrac_labelreg_scan': 'MNIST label reg., param. fraction',
    'mnist_paramfrac_ce_scan': 'MNIST CE, param. fraction',
}

#: one colour per SCAN, for the figures whose lines are scans and not methods (App. P and
#: the two knob families).  Never a method colour: :data:`style.METHOD_COLORS` owns those.
SCAN_COLORS = {
    'toy_1d': '#1B7837', 'polynomial': '#762A83', 'mnist_labelreg': '#B35806',
    'mnist_ce': '#2166AC',
}
TASK_COLORS = SCAN_COLORS
GRID_COLORS = {
    'rebuttal_overparam_polynomial_scan': '#762A83',
    'toy_1d_scan': '#1B7837',
    'rebuttal_overparam_toy_1d_scan': '#7FBC41',
    'rebuttal_batchsize_polynomial_scan': '#B35806',
    'polynomial_scan': '#D6604D',
    'mnist_scan_labelRegression': '#2166AC',
    'mnist_scan_ce': '#4393C3',
    'mnist_kappaScan_labelRegression': '#542788',
    'toy_1d_paramfrac_scan': '#8073AC',
    'polynomial_paramfrac_scan': '#E08214',
    'mnist_paramfrac_labelreg_scan': '#35978F',
    'mnist_paramfrac_ce_scan': '#01665E',
    'rebuttal_overparam_mnist_scan': '#053061',
}
#: kappa = 2 is the default and is drawn black
KAPPA_COLORS = {1: '#C44E52', 2: '#000000', 3: '#4C72B0'}

_FALLBACK_COLORS = ('#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860')


def _task_title(key):
    return style.DATASET_TITLES.get(key, key)


def _scan_label(name):
    return SCAN_LABELS.get(name, name.replace('_', ' '))


def _grid_color(name, i=0):
    return GRID_COLORS.get(name, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])


# ---------------------------------------------------------------------------
# Macro-name spelling: TeX forbids digits in a command name
# ---------------------------------------------------------------------------
_DIGIT_WORDS = ('Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight',
                'Nine')
#: short names for the values these appendices index macros by, so the macro a sentence
#: quotes reads as English (``\numBatchsizeSvenValBSixteen``).  Anything not here is
#: spelled digit by digit, which is ugly but never ambiguous.
_VALUE_WORDS = {
    0.05: 'Twentieth', 0.1: 'Tenth', 0.25: 'Quarter', 0.5: 'Half', 0.75: 'ThreeQuarter',
    1.0: 'One', 2.0: 'Two', 3.0: 'Three', 4.0: 'Four', 8.0: 'Eight', 16.0: 'Sixteen',
    32.0: 'ThirtyTwo', 48.0: 'FortyEight', 64.0: 'SixtyFour', 128.0: 'OneTwentyEight',
    256.0: 'TwoFiftySix', 1.5: 'OneAndHalf', 0.125: 'Eighth', 0.375: 'ThreeEighth',
}


def _word(value):
    """A value as a letters-only macro-name fragment (``16`` -> ``Sixteen``)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ''.join(ch for ch in str(value) if ch.isalpha()) or 'X'
    if v in _VALUE_WORDS:
        return _VALUE_WORDS[v]
    text = f'{v:g}'
    out = []
    for ch in text:
        if ch.isdigit():
            out.append(_DIGIT_WORDS[int(ch)])
        elif ch == '.':
            out.append('P')
        elif ch == '-':
            out.append('Neg')
        elif ch in 'eE':
            out.append('E')
        elif ch == '+':
            continue
        else:
            out.append(ch)
    return ''.join(out)


def _join(items, last=' and '):
    """``a, b and c`` -- for a macro whose body is a short list of names.

    Scan labels contain commas of their own ("MNIST CE, micro-batch"), so the separator
    becomes a semicolon as soon as one of the items does.
    """
    items = [str(i) for i in items]
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    sep = '; ' if any(',' in i for i in items) else ', '
    return sep.join(items[:-1]) + last + items[-1]


def _numbered(values, fmt='{:g}'):
    """``0.25, 0.5, 1`` -- a swept grid as a printable list."""
    return ', '.join(fmt.format(float(v)) for v in values)


def _tex_num(value, sig=3):
    """A number as a math BODY (no ``$``), with a power of ten written as one.

    ``rtol`` spans 1e-5 .. 1e-1 and ``f'{1e-5:g}'`` is ``1e-05``, which is what a table
    must not print.
    """
    v = _num(value)
    if not np.isfinite(v):
        return '--'
    if v > 0:
        e = np.log10(v)
        if abs(e - round(e)) < 1e-12 and not (-4 < round(e) < 4):
            return f'10^{{{int(round(e))}}}'
    body = str(C.fmt_sig(v, sig))
    return body[1:-1] if body.startswith('$') and body.endswith('$') else body


def _tex_list(values, sig=3):
    """``$10^{-5}$, $10^{-4}$, 0.001`` -- a swept grid of small numbers."""
    return ', '.join(f'${_tex_num(v, sig)}$' for v in values)


# ---------------------------------------------------------------------------
# Small frame helpers
# ---------------------------------------------------------------------------
def _num(value):
    """A python float from a possibly-numpy / possibly-missing cell."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return v


def _maybe(M, name, value, **kw):
    """Add a macro only where the number exists.

    Used for the cells a study legitimately leaves empty -- the MNIST label-regression
    parameter-fraction sweep has no finished run at all below $f = 0.5$ at its reference
    learning rate, and that absence is reported as a count, not as a loss.
    """
    if value is None:
        return None
    if C.is_number(value) and not C.finite(value):
        return None
    return M.add(name, value, **kw)


def _row(frame, **eq):
    """The first row of ``frame`` matching every ``column=value``, or ``None``."""
    m = pd.Series(True, index=frame.index)
    for col, val in eq.items():
        if col not in frame.columns:
            return None
        if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
            m &= np.isclose(pd.to_numeric(frame[col], errors='coerce'), float(val))
        else:
            m &= frame[col] == val
    sub = frame[m]
    return None if sub.empty else sub.iloc[0]


def _sven(frame):
    return _row(frame, method=SVEN)


def _sorted_unique(series):
    return sorted(float(v) for v in pd.to_numeric(series, errors='coerce').dropna().unique())


# ---------------------------------------------------------------------------
# Figure plumbing
# ---------------------------------------------------------------------------
def _thin(ax, sven_lw=1.25, other_lw=0.75, ms=2.2, cap_ms=1.3):
    """Re-scale the artists of a notebook-sized plot helper to print scale.

    :func:`reviewer_figs.plot_arm` and :func:`budget.plot_best_of_n` hard-code 2.2-2.6 pt
    lines and 3 pt caps (right for an 8x6 in panel, opaque on a 1.76 in one).  Only the
    widths change; nothing plotted moves.
    """
    black = {'#000000', 'k', 'black', '#000'}
    for line in ax.get_lines():
        if line.get_marker() in ('_', '|'):        # an error-bar cap
            line.set_markersize(cap_ms)
            line.set_markeredgewidth(0.5)
            continue
        colour = line.get_color()
        is_sven = isinstance(colour, str) and colour.lower() in black
        line.set_linewidth(sven_lw if is_sven else other_lw)
        if line.get_marker() not in ('', 'None', None, ' '):
            line.set_markersize(ms * (1.2 if is_sven else 1.0))
    for con in ax.containers:
        lines = getattr(con, 'lines', None)
        if not lines:
            continue
        parts = list(lines) + [None, None, None]
        for cap in (parts[1] or ()):
            cap.set_markersize(cap_ms)
            cap.set_markeredgewidth(0.5)
        for bar in (parts[2] or ()):
            bar.set_linewidth(0.55)
    return ax


def _grid(ax, which='both'):
    ax.grid(True, which='major', ls='--', lw=0.35, alpha=0.45)
    return ax


def _axes_handles(axes):
    """``(handles, labels)`` over ``axes`` in order, deduplicated by label.

    The same collection :func:`reviewer_figs.figure_legend` does (including the one
    seed-band entry :func:`style.band_legend` adds), without its ``tight_layout`` call --
    the paper figures use constrained layout.
    """
    seen, handles, labels = set(), [], []
    for ax in np.atleast_1d(np.asarray(axes, dtype=object)).ravel():
        hs, ls = ax.get_legend_handles_labels()
        for h, l in zip(hs, ls):
            if not l or l.startswith('_') or l in seen:
                continue
            seen.add(l)
            handles.append(h)
            labels.append(l)
    return handles, labels


def _legend_below(fig, handles, labels, ncol=5, height=0.42, fontsize=None, **kw):
    """One legend under the panels, with the space for it taken out of the layout.

    ``height`` is in inches (it is the ``extra_h`` the figure was sized with); anything
    else goes straight to ``fig.legend`` through :func:`common.legend_below`, so a
    notebook can move it or give it a frame.
    """
    w, h = fig.get_size_inches()
    frac = min(0.45, float(height) / max(h, 1e-6))
    engine = fig.get_layout_engine()
    if engine is not None:
        try:
            engine.set(rect=(0.0, frac, 1.0, 1.0 - frac))
        except (AttributeError, TypeError):        # pragma: no cover - older mpl
            pass
    kw['ncol'] = ncol
    if fontsize is not None:
        kw['fontsize'] = fontsize
    return C.legend_below(fig, handles, labels, y=0.0, **kw)


def _figure_legend(fig, handles, labels, spec):
    """:func:`_legend_below` driven by a figure's ``legend_below`` option dict.

    The keys are ``ncol``, ``height``, ``fontsize`` and whatever ``fig.legend`` takes
    (``loc``, ``frameon``, ``bbox_to_anchor``, ...), plus ``remove: True`` for "no shared
    legend at all" -- the same spelling the generic ``legend`` cosmetic uses, and the only
    one that works through the override layer, which MERGES a dict-valued option into the
    default rather than replacing it.
    """
    kw = dict(spec or {})
    if not kw or kw.pop('remove', False):
        return None
    return _legend_below(fig, handles, labels, ncol=kw.pop('ncol', 5),
                         height=kw.pop('height', 0.42),
                         fontsize=kw.pop('fontsize', None), **kw)


def _panel_legend(ax, spec):
    """A per-panel legend from an option dict; ``remove: True`` leaves the panel bare."""
    kw = dict(spec or {})
    if not kw or kw.pop('remove', False):
        return None
    return ax.legend(**kw)


def _subplots(opts, nrow=None, ncol=None, aspect=None, extra_h=None):
    """``plt.subplots`` at the size the page gives these panels.

    ``fraction`` is the panel width as a fraction of ``\\linewidth`` and drives
    :func:`common.figsize`; ``style_fraction`` is the separate key
    :func:`common.set_paper_style` looks the font sizes up by (a 3-panel row is 1/3 wide
    and set at the 0.32 sizes).  ``squeeze=False``, so ``axes[i][j]`` works even when a
    notebook asks for a single row.
    """
    import matplotlib.pyplot as plt

    figspec.paper_style(opts['style_fraction'])
    nrow = int(opts['nrow'] if nrow is None else nrow)
    ncol = int(opts['ncol'] if ncol is None else ncol)
    size = C.figsize(ncol, nrow, opts['fraction'],
                     aspect=(opts['aspect'] if aspect is None else aspect),
                     extra_h=(opts['extra_h'] if extra_h is None else extra_h))
    return plt.subplots(nrow, ncol, figsize=size, squeeze=False)


def _pn_marker(ax, opts):
    """The dotted $P/N = 1$ guide the App. G panels carry (``mark_pn_one``)."""
    if opts.get('mark_pn_one'):
        ax.axvline(1.0, ls=':', c='0.55', lw=0.6, zorder=0)
    return ax


def _proxy(label, **kw):
    from matplotlib.lines import Line2D
    kw.setdefault('color', 'k')
    kw.setdefault('lw', 0.9)
    return Line2D([], [], label=label, **kw)


# ---------------------------------------------------------------------------
# Loading (one frame per scan, reused by every section)
# ---------------------------------------------------------------------------
class _Data:
    """Lazy, memoised scan frames plus the provenance list of what was read.

    Also the ``ctx`` every ``draw(ctx, opts)`` takes.  Each appendix's analysis frames
    used to be computed inside its ``build_*``, which meant a single figure could not be
    drawn without rebuilding the section; they are memoised accessors here instead
    (:meth:`overparam` ... :meth:`divergence`), so ``build()`` still computes each one
    once and a notebook can draw one panel on its own.  The provenance record a section's
    assets are written with is memoised the same way (:meth:`prov`), so a figure and the
    tables beside it keep quoting one record.
    """

    def __init__(self, root=None):
        self.root = root
        self._frames = {}
        self._cache = {}
        self.scans = []

    def __call__(self, name):
        if name not in self._frames:
            df = ah.add_derived(style.load_results(name, results_root=self.root))
            self._frames[name] = df
            self.scans.append(name)
        return self._frames[name]

    def drop(self, name=None):
        """Forget scan frames -- one, or (``name=None``) every one AND the section cache.

        ``build()`` calls this between sections and depends on it to keep the peak
        footprint at one appendix's worth of frames: the section frames hold references
        to the scan frames, so dropping only the latter would free nothing.  A cached
        context is therefore empty but perfectly usable afterwards -- the next
        ``ctx.kappa()`` re-loads its scan.  ``self.scans`` (what the provenance records
        was read) is kept on purpose.
        """
        if name is None:
            self._frames.clear()
            self._cache.clear()
        else:
            self._frames.pop(name, None)

    # -- the per-section analysis frames, each computed at most once ---------
    def _memo(self, key, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def overparam(self):
        """App. G: ``{task: {best, ranks, gaps, reach, div, ...}}`` (three scans)."""
        return self._memo('overparam', lambda: _overparam_frames(self))

    def overparam_ranks(self):
        """App. G: Sven's rank on every arm of every task, as one frame."""
        return self._memo('overparam_ranks', lambda: _sven_rank_rows(self.overparam()))

    def batchsize(self):
        """App. H: the batch-size sweep's selection, ranks, costs and divergence."""
        return self._memo('batchsize', lambda: _batchsize_frames(self))

    def kappa(self):
        """App. M: the kappa table at matched effective step, paired and covered."""
        return self._memo('kappa', lambda: _kappa_frames(self))

    def knobs(self):
        """App. N: ``{knob: {task: ...}}`` for micro-batching and parameter masking."""
        return self._memo('knobs', lambda: _knob_frames(self))

    def budget(self):
        """App. F: best-of-n curves, trajectory counts and the equal-budget table."""
        return self._memo('budget', lambda: _budget_frames(self.root))

    def divergence(self):
        """App. P: the campaign-wide divergence accounting and the per-grid maps."""
        return self._memo('divergence', lambda: _divergence_frames(self.root))

    def prov(self, section):
        """The provenance record ``section``'s figures and tables are written with."""
        return self._memo(f'prov:{section}', lambda: _PROVENANCE[section](self))


# ---------------------------------------------------------------------------
# The module's context: one _Data per results root, so drawing two figures from a
# notebook loads each scan once (see campaign/FIGURE_API_CONTRACT.md).
# ---------------------------------------------------------------------------
_CONTEXTS: dict = {}


def context(root=None, reload=False):
    """The :class:`_Data` this module's ``draw`` functions take, cached per ``root``.

    Cheap to call twice.  ``reload=True`` throws the cached one away (after a scan has
    been re-selected on disk, say).  Note that :meth:`_Data.drop` empties a context
    without invalidating it -- the cache still hands out the same object, which simply
    re-loads what it is asked for next.
    """
    key = str(root) if root is not None else ''
    if reload:
        _CONTEXTS.pop(key, None)
    if key not in _CONTEXTS:
        _CONTEXTS[key] = _Data(root)
    return _CONTEXTS[key]


# ---------------------------------------------------------------------------
# F5 / F5b / T12 / macros -- App. G, dataset-level over-parameterisation (P > N)
# ---------------------------------------------------------------------------
def _overparam_frames(data):
    """Everything App. G quotes, per task, from the functions of record."""
    out = {}
    for key, scan in OVERPARAM_SCANS:
        df = data(scan)
        P = ah.n_params_of(df, what=scan)
        best = rf.arm_table(df, 'n_data')
        best['P_over_N'] = P / best['n_data']
        ranks = {q: rf.rank_table(best, 'n_data', q)
                 for q in ('final_val_loss', 'final_test_loss', 'final_train_eval')}
        gaps = rf.neighbour_gaps(df, best, 'n_data')
        targets = rf.median_method_target(best, 'final_train_eval', 1.0, arm='n_data')
        reach = rf.reach_table(df, best, 'n_data', targets, which='train_eval')
        counts = rf.reach_count_table(df, 'n_data', 1e-3, which='train_eval')
        div = rf.divergence_table(df, 'n_data')
        dvr = rf.divergence_vs_reference(df, reference=SVEN)
        arms = sorted(best['n_data'].unique())
        out[key] = {
            'scan': scan, 'df': df, 'P': int(P), 'best': best, 'ranks': ranks,
            'gaps': gaps, 'targets': targets, 'reach': reach, 'reach_counts': counts,
            'div': div, 'dvr': dvr, 'arms': [float(a) for a in arms],
            'pn': sorted(P / float(a) for a in arms),
        }
    return out


def _sven_rank_rows(over):
    """Sven's rank on every arm of every task (validation, test, train_eval)."""
    rows = []
    for key, d in over.items():
        for n in sorted(d['best']['n_data'].unique()):
            r_val = _row(d['ranks']['final_val_loss'], method=SVEN, n_data=n)
            if r_val is None:
                continue
            r_test = _row(d['ranks']['final_test_loss'], method=SVEN, n_data=n)
            r_train = _row(d['ranks']['final_train_eval'], method=SVEN, n_data=n)
            arm = d['best'][d['best']['n_data'] == n].sort_values('final_val_loss')
            rows.append({
                'task': key, 'n_data': float(n), 'P': d['P'],
                'P_over_N': d['P'] / float(n),
                'rank_val': _num(r_val['rank']), 'n_methods': _num(r_val['n_methods']),
                'rank_test': _num(None if r_test is None else r_test['rank']),
                'rank_train': _num(None if r_train is None else r_train['rank']),
                'sven_val': _num(r_val['value']), 'counts': r_val['counts'],
                'winner': arm['display'].iloc[0] if len(arm) else '--',
            })
    return pd.DataFrame(rows)


def _reach_ratio(d):
    """Sven's epochs-to-target divided by the arm's median-method epochs-to-target.

    Scale-free, so the three tasks (200 full-batch epochs on the synthetics, 20 on MNIST)
    fit one panel.  ``all_reached`` is Sven's own flag: hollow markers mark an arm where
    some Sven seed never got there.
    """
    reach = d['reach']
    rows = []
    for n in sorted(reach['n_data'].unique()):
        arm = reach[reach['n_data'] == n]
        sven = _row(arm, method=SVEN)
        ep = pd.to_numeric(arm['epochs'], errors='coerce')
        med = float(np.nanmedian(ep[np.isfinite(ep)])) if np.isfinite(ep).any() else np.nan
        rows.append({'n_data': float(n), 'P_over_N': d['P'] / float(n),
                     'sven_epochs': _num(None if sven is None else sven['epochs']),
                     'median_epochs': med,
                     'ratio': (_num(sven['epochs']) / med
                               if sven is not None and med and np.isfinite(med)
                               else np.nan),
                     'all_reached': bool(sven is not None and sven['all_reached']),
                     'n_reached': int(0 if sven is None else sven['n_reached']),
                     'n_runs': int(0 if sven is None else sven['n_runs'])})
    return pd.DataFrame(rows)


#: the outcome each F5b row draws, and the axis label it carries.  A knob lists the
#: quantities; the wording stays here so the option itself is a flat list of column names
_OUTCOME_LABELS = {
    'final_val_loss': 'Final val. loss',
    'final_test_loss': 'Final test loss',
    'final_train_eval': 'Final train loss (subset)',
}


def _overparam_keys(over, opts):
    """The tasks F5/F5b draw: the ``tasks`` option, in the scan list's order."""
    return [k for k, _ in OVERPARAM_SCANS if k in over and k in opts['tasks']]


def _fig_overparam(ctx, opts):
    over, ranks = ctx.overparam(), ctx.overparam_ranks()
    fig, axes = _subplots(opts)
    keys = _overparam_keys(over, opts)

    for j, key in enumerate(keys):
        d = over[key]
        ax = axes[0][j]
        rf.plot_arm(ax, d['best'], 'P_over_N', 'final_val_loss', logx=True, logy=True)
        _thin(ax, sven_lw=opts['sven_lw'], other_lw=opts['other_lw'],
              ms=opts['thin_ms'])
        _pn_marker(ax, opts)
        rf.arm_ticks(ax, d['pn'], fmt=opts['arm_tick_fmt'],
                     min_log_sep=opts['min_log_sep'])
        ax.set_xlabel('$P/N$')
        ax.set_ylabel('Final val. loss' if j == 0 else '')
        ax.set_title(_task_title(key))
        _grid(ax)

    # (1,0) Sven's rank, validation (selected on) and test (outcome)
    ax = axes[1][0]
    for key in keys:
        sub = ranks[ranks['task'] == key].sort_values('P_over_N')
        c = TASK_COLORS[key]
        ax.plot(sub['P_over_N'], sub['rank_val'], '-o', color=c, lw=opts['lw'],
                ms=opts['ms'], label=_task_title(key))
        ax.plot(sub['P_over_N'], sub['rank_test'], '--s', color=c, lw=opts['overlay_lw'],
                ms=opts['overlay_ms'], alpha=0.8, mfc='none')
    _pn_marker(ax, opts)
    ax.set_xscale('log')
    ax.invert_yaxis()
    top = int(np.nanmax(np.r_[ranks['rank_val'].to_numpy(), ranks['rank_test'].to_numpy()]))
    ax.set_yticks(range(1, top + 1, 1 if top <= 6 else 2))
    ax.set_xlabel('$P/N$')
    ax.set_ylabel("Sven's rank: val, test")
    _panel_legend(ax, opts['panel_legend'])
    _grid(ax)

    # (1,1) time to the per-arm median-method target, relative to that median
    ax = axes[1][1]
    for key in keys:
        t = _reach_ratio(over[key])
        c = TASK_COLORS[key]
        ax.plot(t['P_over_N'], t['ratio'], '-o', color=c, lw=opts['lw'], ms=opts['ms'],
                label=_task_title(key))
        miss = t[~t['all_reached']]
        if len(miss) and opts['mark_partial_reach']:
            ax.plot(miss['P_over_N'], miss['ratio'], 'o', color=c, ms=3.4, mfc='white',
                    mew=0.7, zorder=4)
    ax.axhline(1.0, ls='-', c='0.55', lw=0.5, zorder=0)
    _pn_marker(ax, opts)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$P/N$')
    ax.set_ylabel('Sven epochs / median method')
    _grid(ax)

    # (1,2) divergence rate over the whole arm, Sven vs the cleanest baseline
    ax = axes[1][2]
    for key in keys:
        d = over[key]
        c = TASK_COLORS[key]
        sv = d['div'][d['div']['method'] == SVEN].sort_values('n_data')
        ax.plot([d['P'] / n for n in sv['n_data']], sv['frac'], '-o', color=c,
                lw=opts['lw'], ms=opts['ms'], label=_task_title(key))
        # the WORST baseline, not the cleanest: the honest comparison is whether Sven
        # sits above the noisiest method of the field (C17), and a "best baseline" line
        # is 0 on every arm of every scan and says nothing
        other = d['div'][d['div']['method'] != SVEN]
        hi = other.groupby('n_data')['frac'].max().reset_index().sort_values('n_data')
        ax.plot([d['P'] / n for n in hi['n_data']], hi['frac'], '--', color=c,
                lw=opts['overlay_lw'], alpha=0.8)
    _pn_marker(ax, opts)
    ax.set_xscale('log')
    ax.set_xlabel('$P/N$')
    ax.set_ylabel('diverged: Sven, worst baseline')
    _grid(ax)

    handles, labels = _axes_handles(axes[0])
    _figure_legend(fig, handles, labels, opts['legend_below'])
    return fig, {'axes': axes, 'provenance': ctx.prov('overparam'), 'tasks': keys,
                 'style_fraction': opts['style_fraction']}


def _fig_overparam_outcomes(ctx, opts):
    over = ctx.overparam()
    fig, axes = _subplots(opts)
    keys = _overparam_keys(over, opts)
    exclude = opts.get('exclude') or {}
    for i, q in enumerate(opts['metrics']):
        for j, key in enumerate(keys):
            d = over[key]
            ax = axes[i][j]
            # a method dropped from one task's panels (a list, or {task: [methods]}):
            # SOAP on MNIST sits a decade above the field and would set the y range
            drop = set(exclude if isinstance(exclude, (list, tuple))
                       else list(exclude.get('*', ())) + list(exclude.get(key, ())))
            methods = [m for m in ah.method_order(d['best']['method'].unique())
                       if m not in drop] if drop else None
            rf.plot_arm(ax, d['best'], 'P_over_N', q, methods=methods, logx=True,
                        logy=True)
            _thin(ax, sven_lw=opts['sven_lw'], other_lw=opts['other_lw'],
                  ms=opts['thin_ms'])
            _pn_marker(ax, opts)
            rf.arm_ticks(ax, d['pn'], fmt=opts['arm_tick_fmt'],
                         min_log_sep=opts['min_log_sep'])
            ax.set_xlabel('$P/N$' if i == len(opts['metrics']) - 1 else '')
            ax.set_ylabel(_OUTCOME_LABELS.get(q, q) if j == 0 else '')
            if i == 0:
                ax.set_title(_task_title(key))
            _grid(ax)
    handles, labels = _axes_handles(axes[0])
    _figure_legend(fig, handles, labels, opts['legend_below'])
    return fig, {'axes': axes, 'provenance': ctx.prov('overparam'), 'tasks': keys,
                 'metrics': list(opts['metrics']),
                 'style_fraction': opts['style_fraction']}


def _gap_cell(gap):
    """``-2.84*`` -- a paired $t$ statistic, starred where the interval excludes zero."""
    if gap is None:
        return '--'
    t = C.fmt_sig(_num(gap['t']), 3)
    return C.Raw(f'{t}$^{{*}}$') if bool(gap['significant']) else str(t)


def _table_overparam(over, ranks, provenance):
    """T12 -- Sven per arm, with the nearest-rival t statistic and the reach counts."""
    rows = []
    for key, _ in OVERPARAM_SCANS:
        if key not in over:
            continue
        d = over[key]
        rows.append(C.span_row(f"{_task_title(key)}  ($P = {d['P']}$)"))
        for _, r in ranks[ranks['task'] == key].sort_values('P_over_N',
                                                            ascending=False).iterrows():
            n = r['n_data']
            gap = d['gaps'][(d['gaps']['n_data'] == n)]
            above = gap[gap['rival_rank'] < gap['rank']].sort_values('rival_rank')
            below = gap[gap['rival_rank'] > gap['rank']].sort_values('rival_rank')
            above = above.iloc[0] if len(above) else None
            below = below.iloc[0] if len(below) else None
            rc = _row(d['reach_counts'], n_data=n)
            rows.append({
                '$N$': C.fmt_int(n),
                '$P/N$': C.fmt_sig(r['P_over_N'], 3),
                'Sven val.': C.fmt_pm(r['sven_val'],
                                      _num(_row(d['best'], method=SVEN,
                                                n_data=n)['final_val_loss_std'])),
                'fin./att.': r['counts'],
                'Rank (val)': C.fmt_rank(r['rank_val'], r['n_methods']),
                'Rank (test)': C.fmt_rank(r['rank_test'], r['n_methods']),
                'Best method': r['winner'],
                'Rival above': ('--' if above is None else above['rival_display']),
                '$t$ (above)': _gap_cell(above),
                'Rival below': ('--' if below is None else below['rival_display']),
                '$t$ (below)': _gap_cell(below),
                'Reach $10^{-3}$': ('--' if rc is None else
                                    f"{int(rc['n_reached'])}/{int(rc['n_field'])}"),
            })
    path = C.write_table('overparam', rows, provenance_record=provenance, fit=True,
                         caption=('Dataset-level over-parameterisation. Per $P/N$ arm: the '
                                  'configuration selected on validation loss inside that '
                                  'arm, Sven\'s rank on validation (selected on) and on '
                                  'test (an outcome), the paired-by-seed $t$ statistic '
                                  'against the method immediately above and immediately '
                                  'below it in the arm\'s ordering, and how many of the '
                                  'field reach a full training loss of $10^{-3}$ on every '
                                  'seed of some configuration. A starred $t$ is one whose '
                                  '95\\% $t$-interval excludes zero; every unstarred '
                                  'placement is a statistical tie.'),
                         label='tab:overparam',
                         notes=('rank is over the methods with an eligible configuration '
                                'in that arm',
                                'reviewer_figs.arm_table / rank_table / neighbour_gaps / '
                                'reach_count_table'))
    return path


def _table_overparam_loss(over, provenance):
    """T12b -- every method's selected validation loss, per arm, one block per task."""
    parts = []
    for key, _ in OVERPARAM_SCANS:
        if key not in over:
            continue
        d = over[key]
        arms = sorted(d['best']['n_data'].unique())
        cols = ['Method'] + [f"$P/N={d['P'] / float(a):.3g}$" for a in arms]
        rows = []
        for m in ah.method_order(sorted(d['best']['method'].unique())):
            sub = d['best'][d['best']['method'] == m]
            rec = {'Method': style.method_label(m)}
            for a in arms:
                cell = _row(sub, n_data=a)
                rec[f"$P/N={d['P'] / float(a):.3g}$"] = (
                    '--' if cell is None else C.fmt_sig(_num(cell['final_val_loss']), 3))
            rows.append(rec)
        parts.append(C.booktabs(
            rows, columns=cols, fit=True,
            caption=(f"{_task_title(key)} ($P={d['P']}$): the selected configuration's "
                     'seed-mean final validation loss per method and $P/N$ arm. '
                     '"--" = no eligible configuration in that arm.'),
            label=f'tab:overparam-loss-{key.replace("_", "-")}'))
    path = C.TABLE_DIR / 'overparam_loss.tex'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(parts))
    C.write_provenance(path, provenance)
    return path


def _prov_overparam(ctx=None):
    return C.provenance(
        functions=('reviewer_figs.arm_table', 'reviewer_figs.rank_table',
                   'reviewer_figs.neighbour_gaps', 'reviewer_figs.median_method_target',
                   'reviewer_figs.reach_table', 'reviewer_figs.reach_count_table',
                   'reviewer_figs.divergence_table',
                   'reviewer_figs.divergence_vs_reference', 'reviewer_figs.plot_arm',
                   'analysis_helpers.n_params_of'),
        scans=[s for _, s in OVERPARAM_SCANS], note='App. G (P > N): F5, F5b, T12')


def build_overparam(data, M, want, report):
    prov = data.prov('overparam')
    over = data.overparam()
    ranks = data.overparam_ranks()

    if want['figures']:
        _figures('overparam', data, report)
    if want['tables']:
        report['tables'].append(str(_table_overparam(over, ranks, prov)))
        report['tables'].append(str(_table_overparam_loss(over, prov)))
    if not want['numbers']:
        return over

    src = 'reviewer_figs.rank_table / arm_table / neighbour_gaps'
    M.add('numOverparamNTasks', len(over), source=src)
    M.add('numOverparamNArms', len(ranks), source=src)
    M.add('numOverparamNMethods',
          int(max(int(d['df']['method'].nunique()) for d in over.values())), source=src)
    M.add('numOverparamNTopTwo', int((ranks['rank_val'] <= 2).sum()), source=src)
    M.add('numOverparamNFirst', int((ranks['rank_val'] == 1).sum()), source=src)
    M.add('numOverparamNFirstTest', int((ranks['rank_test'] == 1).sum()), source=src)
    for col, tag in (('rank_val', 'Val'), ('rank_test', 'Test')):
        over_pn = ranks[ranks['P_over_N'] > 1][col]
        under_pn = ranks[ranks['P_over_N'] < 1][col]
        if len(over_pn):
            M.add(f'numOverparamSvenRank{tag}Over', float(over_pn.mean()), source=src,
                  note='pooled mean rank at P/N > 1')
        if len(under_pn):
            M.add(f'numOverparamSvenRank{tag}Under', float(under_pn.mean()), source=src,
                  note='pooled mean rank at P/N < 1')
        # the strongest baseline's pooled mean rank, for the sentence that compares the
        # two (C18: "pooled mean rank Sven 2.12 vs best baseline 3.17 on test")
        best_rank, best_name = _baseline_mean_rank(over, col)
        if np.isfinite(best_rank):
            M.add(f'numOverparamBestRank{tag}', best_rank, source=src,
                  note='pooled mean rank over all arms of the strongest baseline')
            M.add(f'numOverparamBestRank{tag}Who', C.Raw(best_name), source=src)

    # the arm-by-arm rank strings the prose quotes verbatim
    for key, d in over.items():
        tag = TASK_KEY[key]
        sub = ranks[ranks['task'] == key].sort_values('P_over_N', ascending=False)
        M.add(f'numOverparam{tag}RanksVal',
              C.Raw(_numbered(sub['rank_val'], '{:.0f}')), source=src,
              note='Sven rank per arm, most over-parameterised first')
        M.add(f'numOverparam{tag}RanksTest',
              C.Raw(_numbered(sub['rank_test'], '{:.0f}')), source=src)
        M.add(f'numOverparam{tag}Pn', C.Raw(_numbered(sub['P_over_N'], '{:.3g}')),
              source=src, note='the P/N arms, most over-parameterised first')
        M.add(f'numOverparam{tag}NMethods', int(sub['n_methods'].max()), source=src)
        M.add(f'numOverparam{tag}P', d['P'], source='analysis_helpers.n_params_of')
        rc = d['reach_counts']
        M.add(f'numOverparam{tag}ReachMax', int(rc['n_reached'].max()),
              source='reviewer_figs.reach_count_table',
              note='most methods reaching train_eval < 1e-3 in any arm')
        M.add(f'numOverparam{tag}ReachField', int(rc['n_field'].max()),
              source='reviewer_figs.reach_count_table')
        dvr = d['dvr']
        sv = _row(dvr, method=SVEN)
        M.add(f'numOverparam{tag}SvenDivFrac', C.fmt_pct(_num(sv['frac']), 2),
              source='reviewer_figs.divergence_vs_reference')
        M.add(f'numOverparam{tag}SvenDivRank', int(dvr.attrs['reference_rank']),
              source='reviewer_figs.divergence_vs_reference',
              note='worst-first rank of Sven among the methods of this scan')
        M.add(f'numOverparam{tag}SvenDivN', int(sv['n_runs']),
              source='reviewer_figs.divergence_vs_reference')
        M.add(f'numOverparam{tag}NMethodsDiv', int(len(dvr)), source=src)
        worse = dvr.attrs.get('above_reference') or []
        M.add(f'numOverparam{tag}DivWorse',
              C.Raw(_join(worse) if worse else 'no method'),
              source='reviewer_figs.divergence_vs_reference',
              note='methods with a higher wide-rule divergence rate than Sven')
        ratio = pd.to_numeric(dvr.loc[dvr['method'] != SVEN, 'grid_ratio_vs_reference'],
                              errors='coerce')
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if len(ratio):
            M.add(f'numOverparam{tag}GridRatio', float(1.0 / np.median(ratio)),
                  source='reviewer_figs.divergence_vs_reference',
                  note="Sven's grid size divided by the MEDIAN baseline grid of this scan")

    gaps = pd.concat([d['gaps'].assign(task=k, P_over_N=d['P'] / d['gaps']['n_data'])
                      for k, d in over.items()], ignore_index=True)
    top = gaps[(gaps['rank'] <= 2) & (gaps['rival_rank'] > gaps['rank'])]
    M.add('numOverparamNGaps', int(len(top)), source='reviewer_figs.neighbour_gaps',
          note='rank-1/rank-2 placements with a beaten rival')
    M.add('numOverparamNGapsSig', int(top['significant'].sum()),
          source='reviewer_figs.neighbour_gaps',
          note='of those, the gaps whose 95% paired t-interval excludes 0')
    if int(top['significant'].sum()):
        won = top[top['significant']].iloc[0]
        M.add('numOverparamGapSigWhere',
              C.Raw(f"{_task_title(won['task'])} at $P/N={won['P_over_N']:.3g}$ "
                    f"against {won['rival_display']}"),
              source='reviewer_figs.neighbour_gaps')
        M.add('numOverparamGapSigT', _num(won['t']),
              source='reviewer_figs.neighbour_gaps')
    lost = gaps[(gaps['rank'] > 1) & (gaps['rival_rank'] < gaps['rank'])
                & gaps['significant']]
    M.add('numOverparamNLossSig', int(len(lost)), source='reviewer_figs.neighbour_gaps',
          note='arms where a rival beats Sven by a resolved paired margin')
    if len(lost):
        w = lost.sort_values('t', ascending=False).iloc[0]
        M.add('numOverparamLossSigWhere',
              C.Raw(f"{_task_title(w['task'])} at $N={int(w['n_data'])}$ against "
                    f"{w['rival_display']}"), source='reviewer_figs.neighbour_gaps')
        M.add('numOverparamLossSigT', _num(w['t']), source='reviewer_figs.neighbour_gaps')
        M.add('numOverparamLossSigList', C.Raw(_join(
            [f"{_task_title(r['task'])} at $N={int(r['n_data'])}$ against "
             f"{r['rival_display']} ($t={_num(r['t']):.3g}$)"
             for _, r in lost.sort_values('t', ascending=False).iterrows()])),
              source='reviewer_figs.neighbour_gaps',
              note='every arm where a rival beats Sven by a resolved paired margin')
    return over


def _baseline_mean_rank(over, col):
    """``(mean rank, display name)`` of the strongest baseline over all arms.

    The comparison C18 makes: Sven's pooled mean rank against the best a baseline
    achieves.  A method absent from an arm (no eligible configuration there) contributes
    nothing to its own mean, which flatters it -- so this is the most generous reading.
    """
    metric = 'final_val_loss' if col == 'rank_val' else 'final_test_loss'
    per_method = {}
    for d in over.values():
        r = d['ranks'][metric]
        for m, sub in r[r['method'] != SVEN].groupby('method'):
            per_method.setdefault(m, []).extend(
                pd.to_numeric(sub['rank'], errors='coerce').tolist())
    means = {m: float(np.nanmean(v)) for m, v in per_method.items() if len(v)}
    if not means:
        return float('nan'), '--'
    who = min(means, key=means.get)
    return means[who], style.method_label(who)


# ---------------------------------------------------------------------------
# F6 / T13 / macros -- App. H, batch-size sensitivity (R2's Q1)
# ---------------------------------------------------------------------------
def _batchsize_frames(data):
    df = data(BATCHSIZE_SCAN)
    best = rf.arm_table(df, 'batch_size')
    out = {
        'df': df, 'best': best, 'Bs': _sorted_unique(df['batch_size']),
        'ranks_val': rf.rank_table(best, 'batch_size', 'final_val_loss'),
        'ranks_test': rf.rank_table(best, 'batch_size', 'final_test_loss'),
        'gaps': rf.neighbour_gaps(df, best, 'batch_size'),
        'mono': rf.monotonicity_table(best, 'total_steps', 'final_val_loss'),
        'div': rf.divergence_table(df, 'batch_size'),
        'step': rf.fixed_knob_cost(df, 'batch_size', 'rtol', 'avg_batch_time_train',
                                   scale=1e3),
        'mem': rf.fixed_knob_cost(df, 'batch_size', 'rtol', 'peak_gpu_mem_mb'),
    }
    sv = df[df['method'] == SVEN].copy()
    sv['rank_eff'] = sv.apply(rf.effective_rank, axis=1)
    ok = sv[~sv['diverged'].astype(bool) & ~sv['failed'].astype(bool)]
    rank = (ok.groupby(['batch_size', 'rtol'])
            .agg(rank_eff=('rank_eff', 'mean'), n=('run_id', 'size')).reset_index())
    rank['rank_eff_over_B'] = rank['rank_eff'] / rank['batch_size']
    out['rank_by_rtol'] = rank
    out['piv_step'] = out['step'].pivot(index='rtol', columns='batch_size', values='value')
    out['piv_mem'] = out['mem'].pivot(index='rtol', columns='batch_size', values='value')
    return out


def _fig_batchsize_loss(ctx, opts):
    """F6 as carried: the batch-size sweep on validation loss alone, one panel.

    The used-rank, cost and divergence panels of `batchsize` are retired from the
    paper -- the rank law, the profiling appendix and the robustness appendix each
    already say what those panels said -- and stay available on that figure.
    """
    b = ctx.batchsize()
    fig, axes = _subplots(opts)
    Bs = [B for B in b['Bs'] if not opts['batch_sizes'] or B in opts['batch_sizes']]
    best = b['best'][b['best']['batch_size'].isin(Bs)] if opts['batch_sizes'] \
        else b['best']
    ax = axes[0][0]
    rf.plot_arm(ax, best, 'batch_size', 'final_val_loss', logy=True)
    _thin(ax, sven_lw=opts['sven_lw'], other_lw=opts['other_lw'], ms=opts['thin_ms'])
    ax.set_xscale('log', base=2)
    rf.arm_ticks(ax, Bs, fmt=opts['arm_tick_fmt'])
    ax.set_xlabel('batch size $B$')
    ax.set_ylabel('Final validation loss')
    _grid(ax)
    handles, labels = _axes_handles([ax])
    _figure_legend(fig, handles, labels, opts['legend_below'])
    return fig, {'axes': axes, 'provenance': ctx.prov('batchsize'), 'Bs': Bs,
                 'style_fraction': opts['style_fraction']}


def _fig_batchsize(ctx, opts):
    import matplotlib.pyplot as plt

    b = ctx.batchsize()
    fig, axes = _subplots(opts)
    Bs = [B for B in b['Bs'] if not opts['batch_sizes'] or B in opts['batch_sizes']]
    best = b['best'][b['best']['batch_size'].isin(Bs)] if opts['batch_sizes'] \
        else b['best']

    ax = axes[0][0]
    rf.plot_arm(ax, best, 'batch_size', 'final_val_loss', logy=True)
    _thin(ax, sven_lw=opts['sven_lw'], other_lw=opts['other_lw'], ms=opts['thin_ms'])
    ax.set_xscale('log', base=2)
    rf.arm_ticks(ax, Bs, fmt=opts['arm_tick_fmt'])
    ax.set_xlabel('batch size $B$')
    ax.set_ylabel('Final validation loss')
    ax.set_title('Selected config. per method')
    _grid(ax)

    ax = axes[0][1]
    rank_by_rtol = b['rank_by_rtol']
    rtols = sorted(rank_by_rtol['rtol'].unique())
    cmap = plt.get_cmap(opts['cmap'])
    for i, rt in enumerate(rtols):
        s = rank_by_rtol[rank_by_rtol['rtol'] == rt].sort_values('batch_size')
        ax.plot(s['batch_size'], s['rank_eff'], '-o', lw=opts['rtol_lw'],
                ms=opts['rtol_ms'], color=cmap(i / max(1, len(rtols) - 1)),
                label=f'{RTOL_FIG} $={_tex_num(rt)}$')
    if opts['cap_line']:
        ax.plot(Bs, Bs, ls='--', c='0.45', lw=0.8, label='$k=B$ (the cap)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    rf.arm_ticks(ax, Bs, fmt=opts['arm_tick_fmt'])
    ax.set_xlabel('batch size $B$')
    ax.set_ylabel('singular values kept / step')
    ax.set_title("Sven's used rank vs the cap")
    _panel_legend(ax, opts['rank_legend'])
    _grid(ax)

    ax = axes[1][0]
    for i, rt in enumerate(rtols):
        s = b['step'][b['step']['rtol'] == rt].sort_values('batch_size')
        ax.plot(s['batch_size'], s['value'], '-o', lw=opts['rtol_lw'],
                ms=opts['rtol_ms'], color=cmap(i / max(1, len(rtols) - 1)),
                label=f'{RTOL_FIG} $={_tex_num(rt)}$')
    sel = best[best['method'] == SVEN].sort_values('batch_size')
    ax.plot(sel['batch_size'], sel['step_s'] * 1e3, 's--', lw=1.2, ms=2.6, color='k',
            label='selected config.')
    ax.set_xscale('log', base=2)
    rf.arm_ticks(ax, Bs, fmt=opts['arm_tick_fmt'])
    ax.set_xlabel('batch size $B$')
    ax.set_ylabel('ms per optimizer step')
    ax.set_title(f"Sven's cost at fixed {RTOL_FIG}")
    _grid(ax)
    if opts['memory_axis']:
        ax2 = ax.twinx()
        mem = b['mem'].groupby('batch_size')['value'].mean().reset_index()
        ax2.plot(mem['batch_size'], mem['value'], ':', lw=0.8,
                 color=opts['memory_color'], label='peak memory')
        ax2.set_ylabel('peak memory (MB)', color=opts['memory_color'])
        ax2.tick_params(axis='y', colors=opts['memory_color'],
                        labelsize=opts['memory_labelsize'])
    _panel_legend(ax, opts['cost_legend'])

    ax = axes[1][1]
    for m in ah.method_order(sorted(b['div']['method'].unique())):
        s = b['div'][b['div']['method'] == m].sort_values('batch_size')
        if opts['only_diverging_methods'] and s['n_diverged'].sum() == 0:
            continue
        ax.plot(s['batch_size'], s['frac'], '-o', color=style.method_color(m),
                lw=opts['sven_lw'] if m == SVEN else opts['other_lw'],
                ms=opts['ms'] if m == SVEN else opts['overlay_ms'],
                label=style.method_label(m), zorder=3 if m == SVEN else 1)
    ax.set_xscale('log', base=2)
    rf.arm_ticks(ax, Bs, fmt=opts['arm_tick_fmt'])
    ax.set_xlabel('batch size $B$')
    ax.set_ylabel('fraction diverged (wide rule)')
    ax.set_title('Divergence, wide rule')
    _grid(ax)

    handles, labels = _axes_handles([axes[0][0]])
    _figure_legend(fig, handles, labels, opts['legend_below'])
    return fig, {'axes': axes, 'provenance': ctx.prov('batchsize'),
                 'batch_sizes': list(Bs), 'rtols': [float(r) for r in rtols],
                 'style_fraction': opts['style_fraction']}


def _table_batchsize(b, provenance):
    rows = []
    for B in b['Bs']:
        sv = _row(b['best'], method=SVEN, batch_size=B)
        rv = _row(b['ranks_val'], method=SVEN, batch_size=B)
        rt = _row(b['ranks_test'], method=SVEN, batch_size=B)
        arm = b['best'][b['best']['batch_size'] == B].sort_values('final_val_loss')
        gap = b['gaps'][b['gaps']['batch_size'] == B].sort_values('rival_rank')
        near = gap.iloc[0] if len(gap) else None
        ad = _row(b['best'], method='Adam', batch_size=B)
        rows.append({
            '$B$': C.fmt_int(B),
            'steps/epoch': C.fmt_int(_num(None if sv is None else sv['steps_per_epoch'])),
            'Sven val.': C.fmt_pm(_num(None if sv is None else sv['final_val_loss']),
                                  _num(None if sv is None else sv['final_val_loss_std'])),
            'fin./att.': ('--' if sv is None else sv['counts']),
            'Rank (val)': ('--' if rv is None else C.fmt_rank(rv['rank'], rv['n_methods'])),
            'Rank (test)': ('--' if rt is None else C.fmt_rank(rt['rank'], rt['n_methods'])),
            'Best method': arm['display'].iloc[0] if len(arm) else '--',
            'Sven $(k,\\mathrm{rtol})$':
                ('--' if sv is None else
                 C.Raw(f"$({_num(sv['k']):.0f},\\ {_tex_num(sv['rtol'])})$")),
            'used rank': C.fmt_sig(_num(None if sv is None else sv['rank_eff']), 3),
            'Sven ms/step': C.fmt_sig(_num(None if sv is None else sv['step_s']) * 1e3, 3),
            'Adam ms/step': C.fmt_sig(_num(None if ad is None else ad['step_s']) * 1e3, 3),
            'Sven MB': C.fmt_sig(_num(None if sv is None else sv['peak_gpu_mem_mb']), 4),
            'nearest rival $t$': ('--' if near is None else C.fmt_sig(_num(near['t']), 3)),
        })
    parts = [C.booktabs(
        rows, fit=True,
        caption=('Batch-size sensitivity on random-polynomial regression (20 epochs at '
                 'every $B$). Per batch size: the configuration selected inside that arm, '
                 "Sven's rank on validation and test, its selected rank cut and the rank "
                 'the update actually used, the measured step time beside Adam\'s, and the '
                 'paired $t$ statistic against its nearest rival. Times are the scan\'s own '
                 'clock (many runs per GPU), so they are upper bounds.'),
        label='tab:batchsize')]
    # the fixed-rtol cost decomposition: the confound behind the "flat cost" reading
    cost_rows = []
    piv = b['piv_step']
    for rt in piv.index:
        row = {RTOL_TEX: C.Raw(f'${_tex_num(rt)}$')}
        for B in piv.columns:
            row[f'$B={int(B)}$'] = C.fmt_sig(_num(piv.loc[rt, B]), 3)
        first, last = _num(piv.loc[rt].iloc[0]), _num(piv.loc[rt].iloc[-1])
        row['ratio'] = C.fmt_sig(last / first, 3) if first else '--'
        cost_rows.append(row)
    mem = b['piv_mem']
    mrow = {RTOL_TEX: C.Raw(f'peak MB (any {RTOL_TEX})')}
    for B in mem.columns:
        mrow[f'$B={int(B)}$'] = C.fmt_sig(_num(mem[B].mean()), 4)
    first, last = _num(mem.iloc[:, 0].mean()), _num(mem.iloc[:, -1].mean())
    mrow['ratio'] = C.fmt_sig(last / first, 3) if first else '--'
    cost_rows.append(mrow)
    parts.append(C.booktabs(
        cost_rows, midrules=(len(cost_rows) - 1,), fit=True,
        caption=(f"Sven's cost at FIXED {RTOL_TEX}: milliseconds per optimizer step, and "
                 'peak memory in the last row. The per-arm selected configuration changes '
                 f'{RTOL_TEX} with $B$, which is what makes its cost curve look flat; at '
                 f'fixed {RTOL_TEX} the step time rises monotonically with $B$ and the '
                 f'memory is {RTOL_TEX}-independent.'),
        label='tab:batchsize-cost'))
    path = C.TABLE_DIR / 'batchsize.tex'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(parts))
    C.write_provenance(path, provenance)
    return path


def _table_batchsize_loss(b, provenance):
    cols = ['Method'] + [f'$B={int(B)}$' for B in b['Bs']] + ['best at', 'monotone']
    rows = []
    mono = b['mono'].set_index('method')
    for m in ah.method_order(sorted(b['best']['method'].unique())):
        sub = b['best'][b['best']['method'] == m]
        rec = {'Method': style.method_label(m)}
        for B in b['Bs']:
            cell = _row(sub, batch_size=B)
            rec[f'$B={int(B)}$'] = ('--' if cell is None
                                    else C.fmt_sig(_num(cell['final_val_loss']), 3))
        if m in mono.index:
            r = mono.loc[m]
            at = sub.sort_values('final_val_loss')
            rec['best at'] = (f"$B={int(_num(at['batch_size'].iloc[0]))}$" if len(at)
                              else '--')
            rec['best at'] = C.Raw(rec['best at'])
            rec['monotone'] = 'yes' if bool(r['monotone']) else 'no'
        else:
            rec['best at'], rec['monotone'] = '--', '--'
        rows.append(rec)
    path = C.write_table('batchsize_loss', rows, columns=cols, fit=True,
                         provenance_record=provenance,
                         caption=('Selected seed-mean final validation loss per method and '
                                  'batch size, with the batch size each method is best at '
                                  'and whether its loss is monotone along the steps axis '
                                  '(20 epochs buy more steps as $B$ shrinks).'),
                         label='tab:batchsize-loss')
    return path


def _prov_batchsize(ctx=None):
    return C.provenance(
        functions=('reviewer_figs.arm_table', 'reviewer_figs.rank_table',
                   'reviewer_figs.neighbour_gaps', 'reviewer_figs.monotonicity_table',
                   'reviewer_figs.divergence_table', 'reviewer_figs.fixed_knob_cost',
                   'reviewer_figs.effective_rank', 'reviewer_figs.plot_arm'),
        scans=(BATCHSIZE_SCAN,), note='App. H (batch size): F6, T13')


def build_batchsize(data, M, want, report):
    prov = data.prov('batchsize')
    b = data.batchsize()
    if want['figures']:
        _figures('batchsize', data, report)
    if want['tables']:
        report['tables'].append(str(_table_batchsize(b, prov)))
        report['tables'].append(str(_table_batchsize_loss(b, prov)))
    if not want['numbers']:
        return b

    src = 'reviewer_figs.arm_table / rank_table / monotonicity_table'
    Bs = b['Bs']
    M.add('numBatchsizeNArms', len(Bs), source=src)
    M.add('numBatchsizeNMethods', int(b['df']['method'].nunique()), source=src)
    M.add('numBatchsizeGrid', C.Raw(_numbered(Bs, '{:.0f}')), source=src)
    sv = b['ranks_val'][b['ranks_val']['method'] == SVEN].sort_values('batch_size')
    M.add('numBatchsizeSvenVals', C.Raw(_numbered(sv['value'], '{:.4g}')), source=src,
          note='Sven final validation loss, B = 8 .. 256')
    M.add('numBatchsizeSvenRanks', C.Raw(_numbered(sv['rank'], '{:.0f}')), source=src)
    for B in (Bs[0], Bs[-1]):
        r = _row(b['best'], method=SVEN, batch_size=B)
        w = _word(B)
        M.add_pm(f'numBatchsizeSvenValB{w}', _num(r['final_val_loss']),
                 _num(r['final_val_loss_std']), source=src)
        M.add(f'numBatchsizeSvenMsStepB{w}', _num(r['step_s']) * 1e3, source=src)
        M.add(f'numBatchsizeSvenMemMbB{w}', _num(r['peak_gpu_mem_mb']), source=src)
        rr = _row(b['ranks_val'], method=SVEN, batch_size=B)
        M.add(f'numBatchsizeSvenRankB{w}', int(_num(rr['rank'])), source=src)
        ad = _row(b['best'], method='Adam', batch_size=B)
        if ad is not None:
            M.add(f'numBatchsizeAdamMsStepB{w}', _num(ad['step_s']) * 1e3, source=src)
    best_arm = sv.sort_values('rank').iloc[0]
    worst_arm = sv.sort_values('rank').iloc[-1]
    M.add('numBatchsizeSvenBestB', int(_num(best_arm['batch_size'])), source=src)
    M.add('numBatchsizeSvenBestRank', int(_num(best_arm['rank'])), source=src)
    M.add('numBatchsizeSvenWorstB', int(_num(worst_arm['batch_size'])), source=src)
    M.add('numBatchsizeSvenWorstRank', int(_num(worst_arm['rank'])), source=src)
    vals = pd.to_numeric(sv['value'], errors='coerce')
    M.add('numBatchsizeSvenSpread', float(vals.max() / vals.min()), source=src,
          note='ratio of Sven worst to best validation loss over the 32x batch range')
    mono = b['mono']
    M.add('numBatchsizeNMonotone', int(mono['monotone'].sum()),
          source='reviewer_figs.monotonicity_table',
          note='methods monotone on the steps axis')
    M.add('numBatchsizeNMonotoneField', int(len(mono)),
          source='reviewer_figs.monotonicity_table')
    who = sorted(mono.loc[mono['monotone'], 'display'])
    M.add('numBatchsizeMonotoneWho', C.Raw(_join(who) if who else 'no method'),
          source='reviewer_figs.monotonicity_table')
    piv = b['piv_step']
    for rt in piv.index:
        row = piv.loc[rt]
        w = _word(rt)
        M.add(f'numBatchsizeStepRatioRtol{w}',
              float(_num(row.iloc[-1]) / _num(row.iloc[0])),
              source='reviewer_figs.fixed_knob_cost',
              note=f'Sven step time B={int(row.index.min())} -> {int(row.index.max())} '
                   f'at fixed rtol={float(rt):g}')
    lo = piv.index.min()
    ref = 1e-3 if 1e-3 in set(piv.index) else piv.index.max()
    ratio = piv.loc[lo] / piv.loc[ref]
    M.add('numBatchsizeRtolCostMin', float(ratio.min()),
          source='reviewer_figs.fixed_knob_cost',
          note=f'cost of rtol={float(lo):g} relative to rtol={float(ref):g} at fixed B')
    M.add('numBatchsizeRtolCostMax', float(ratio.max()),
          source='reviewer_figs.fixed_knob_cost')
    mem = b['piv_mem']
    M.add('numBatchsizeMemMbMin', float(mem.iloc[:, 0].mean()),
          source='reviewer_figs.fixed_knob_cost')
    M.add('numBatchsizeMemMbMax', float(mem.iloc[:, -1].mean()),
          source='reviewer_figs.fixed_knob_cost')
    gaps = b['gaps']
    firsts = gaps[(gaps['rank'] == 1) & (gaps['rival_rank'] > 1)]
    M.add('numBatchsizeNFirst', int(len(firsts)), source='reviewer_figs.neighbour_gaps')
    M.add('numBatchsizeNFirstSig', int(firsts['significant'].sum()),
          source='reviewer_figs.neighbour_gaps')
    for _, r in gaps.sort_values('batch_size').iterrows():
        if r['rival_rank'] <= r['rank']:
            continue
        name = f"numBatchsizeGapTB{_word(r['batch_size'])}"
        if name in M:
            continue
        M.add(name, _num(r['t']), source='reviewer_figs.neighbour_gaps',
              note=f"Sven vs {r['rival_display']}, paired by seed")
    sel = b['best'][b['best']['method'] == SVEN].sort_values('batch_size')
    M.add('numBatchsizeSvenRtols', C.Raw(_tex_list(sel['rtol'])), source=src,
          note='the rtol the per-arm selection picks, B = 8 .. 256')
    M.add('numBatchsizeSvenUsedRanks', C.Raw(_numbered(sel['rank_eff'], '{:.1f}')),
          source='reviewer_figs.effective_rank')
    return b


# ---------------------------------------------------------------------------
# F8 / T14 / macros -- App. M, kappa at matched effective step
# ---------------------------------------------------------------------------
def _kappa_frames(data):
    df = data(KAPPA_SCAN)
    tbl = rf.kappa_table(df)
    out = {
        'df': df, 'tbl': tbl,
        'matched': tbl.attrs.get('shared_eff_steps') or [],
        'm_val': rf.matched_step_table(tbl, 'final_val_loss'),
        'm_test': rf.matched_step_table(tbl, 'final_test_loss'),
        'paired': rf.matched_step_paired(df, 'final_val_loss'),
        'coverage': rf.eff_step_coverage(df),
        'ks': _sorted_unique(tbl['k']),
        'B': int(_num(df['batch_size'].dropna().iloc[0])),
    }
    return out


def _fig_kappa(ctx, opts):
    kp = ctx.kappa()
    fig, axes = _subplots(opts)
    tbl, B = kp['tbl'], kp['B']
    ks = [k for k in kp['ks'] if not opts['ks'] or k in opts['ks']]
    kappas = [kap for kap in sorted(tbl['kappa'].unique())
              if not opts['kappas'] or int(kap) in opts['kappas']]
    ls_for = {k: (opts['full_ls'] if int(k) == B else opts['trunc_ls']) for k in ks}
    default_lw, other_lw = opts['default_kappa_lw'], opts['other_lw']

    ax = axes[0][0]
    for k in ks:
        for kap in kappas:
            s = tbl[(tbl['k'] == k) & (tbl['kappa'] == kap)].sort_values('eff_step')
            if s.empty:
                continue
            style.band_legend(ax)
            ax.errorbar(s['eff_step'], s['final_val_loss'],
                        yerr=style.clipped_yerr(s['final_val_loss'],
                                                s['final_val_loss_std'],
                                                s['final_val_loss_min']),
                        marker='o', ms=opts['ms'], capsize=opts['capsize'],
                        elinewidth=opts['elinewidth'],
                        lw=default_lw if int(kap) == 2 else other_lw, ls=ls_for[k],
                        color=KAPPA_COLORS[int(kap)])
    if opts['mark_matched_steps']:
        for e in kp['matched']:
            ax.axvline(e, ls=':', c='0.6', lw=0.6, zorder=0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'effective step $2\eta/\kappa$')
    ax.set_ylabel('Final validation loss')
    _grid(ax)

    ax = axes[0][1]
    for k in ks:
        s = kp['m_val'][kp['m_val']['k'] == k].sort_values('eff_step')
        ax.plot(s['eff_step'], s['rel_spread'], marker='o', ms=opts['spread_ms'],
                lw=opts['lw'], ls=ls_for[k], color='k')
    ax.set_xscale('log')
    ax.set_yscale('log')
    rf.arm_ticks(ax, kp['matched'], fmt=opts['arm_tick_fmt'])
    ax.set_xlabel(r'matched effective step')
    ax.set_ylabel(r'spread over $\kappa$')
    _grid(ax)

    ax = axes[0][2]
    for k in ks:
        for kap in kappas:
            s = tbl[(tbl['k'] == k) & (tbl['kappa'] == kap)
                    & (tbl['n_diverged'] == 0)].sort_values('eff_step')
            if s.empty:
                continue
            ax.plot(s['eff_step'], s['rank_eff'], marker='o', ms=opts['ms'],
                    lw=opts['rank_lw'], ls=ls_for[k], color=KAPPA_COLORS[int(kap)])
    ax.set_xscale('log')
    ax.set_xlabel(r'effective step $2\eta/\kappa$')
    ax.set_ylabel('singular values kept / step')
    _grid(ax)

    handles = [_proxy(rf'$\kappa={int(k)}$' + (' (default)' if int(k) == 2 else ''),
                      color=KAPPA_COLORS[int(k)],
                      lw=default_lw if int(k) == 2 else other_lw)
               for k in kappas]
    handles += [_proxy(f'$k={int(k)}$' + (' ($=B$)' if int(k) == B else ''),
                       color='0.35', ls=ls_for[k]) for k in ks]
    handles += [h for h in _axes_handles(axes)[0]
                if getattr(h, 'get_label', lambda: '')() == style.seed_spread_label()]
    labels = [h.get_label() for h in handles]
    _figure_legend(fig, handles, labels, opts['legend_below'])
    return fig, {'axes': axes, 'provenance': ctx.prov('kappa'),
                 'ks': list(ks), 'kappas': [int(k) for k in kappas],
                 'style_fraction': opts['style_fraction']}


def _table_kappa(kp, provenance):
    parts = []
    rows = []
    for _, r in kp['m_val'].sort_values(['k', 'eff_step']).iterrows():
        rec = {'$k$': C.fmt_int(r['k']),
               r'$2\eta/\kappa$': C.fmt_sig(_num(r['eff_step']), 3)}
        for kap in (1, 2, 3):
            col = f'kappa={kap}'
            rec[rf'$\kappa={kap}$'] = (C.fmt_sig(_num(r[col]), 4) if col in r.index
                                       else '--')
            ncol = f'n(kappa={kap})'
            rec[f'$n_{{{kap}}}$'] = (r[ncol] if ncol in r.index else '--')
        rec['rel. spread'] = C.fmt_pct(_num(r['rel_spread']), 2)
        rows.append(rec)
    parts.append(C.booktabs(
        rows,
        caption=(r'$\kappa$ at matched effective step $2\eta/\kappa$ on MNIST label '
                 r'regression. In the untruncated solve the $\kappa$ update is exactly '
                 r'$2/\kappa$ times the $\kappa=2$ update, so a $\kappa$ sweep at fixed '
                 r'$\eta$ is a learning-rate sweep in disguise; the grid realises three '
                 r'effective steps at all three $\kappa$. $n_\kappa$ is finished/attempted.'),
        label='tab:kappa'))

    prows = []
    for _, r in kp['paired'].sort_values(['k', 'eff_step']).iterrows():
        a, b = str(r['pair']).replace('kappa', '').split('-')
        prows.append({
            '$k$': C.fmt_int(r['k']),
            r'$2\eta/\kappa$': C.fmt_sig(_num(r['eff_step']), 3),
            'pair': C.Raw(rf'$\kappa_{{{a}}} - \kappa_{{{b}}}$'),
            '$n$': C.fmt_int(r['n']),
            'mean diff.': C.fmt_sig(_num(r['mean']), 3),
            '$t$': C.fmt_sig(_num(r['t']), 3),
            '$p$': C.fmt_sig(_num(r['p']), 2),
            '95\\%': 'yes' if bool(r['significant']) else 'no',
        })
    parts.append(C.booktabs(
        prows, fit=True,
        caption=(r'Paired-by-seed differences between the $\kappa$ at each matched '
                 r'effective step (same five model seeds on both sides, so the '
                 r'initialisation and data order cancel). The mean is $\kappa_a-\kappa_b$ '
                 r'in final validation loss; the last column is whether the 95\% '
                 r'$t$-interval excludes zero.'),
        label='tab:kappa-paired'))

    crows = []
    top = _num(kp['coverage'].attrs.get('top_matched'))
    for _, r in kp['coverage'].iterrows():
        at_edge = np.isfinite(_num(r['max_eff_step'])) and int(r['n_above_matched']) == 0
        crows.append({
            r'$\kappa$': C.fmt_int(r['kappa']),
            'effective steps': C.fmt_int(r['n_eff_steps']),
            'smallest': C.fmt_sig(_num(r['min_eff_step']), 3),
            'largest': C.fmt_sig(_num(r['max_eff_step']), 3),
            'runs above the top matched step': C.fmt_int(r['n_above_matched']),
            'ceiling': 'grid edge' if at_edge else 'measured',
        })
    parts.append(C.booktabs(
        crows, fit=True,
        caption=(r'Effective-step coverage. The learning-rate list is shared, so the '
                 r'effective step $2\eta/\kappa$ is not: the largest one in the grid falls '
                 r'like $1/\kappa$. Where no run sits above the top matched effective step '
                 f'(${C.fmt_sig(top, 3)}$) the ceiling is a property of the grid, not of '
                 r'the method.'),
        label='tab:kappa-coverage'))
    path = C.TABLE_DIR / 'kappa.tex'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(parts))
    C.write_provenance(path, provenance)
    return path


def _prov_kappa(ctx=None):
    return C.provenance(
        functions=('reviewer_figs.kappa_table', 'reviewer_figs.matched_step_table',
                   'reviewer_figs.matched_step_paired',
                   'reviewer_figs.eff_step_coverage', 'reviewer_figs.paired_seed_diff'),
        scans=(KAPPA_SCAN,), note='App. M (kappa): F8, T14')


def build_kappa(data, M, want, report):
    prov = data.prov('kappa')
    kp = data.kappa()
    if want['figures']:
        _figures('kappa', data, report)
    if want['tables']:
        report['tables'].append(str(_table_kappa(kp, prov)))
    if not want['numbers']:
        return kp

    df, tbl = kp['df'], kp['tbl']
    src = 'reviewer_figs.kappa_table'
    M.add('numKappaNRuns', int(len(df)), source=src)
    M.add('numKappaNRecorded', int((df['status'] == style.STATUS_DIVERGED).sum()),
          source=src, note='records the runner marked diverged (status field)')
    M.add('numKappaNDivWide', int(df['diverged'].astype(bool).sum()), source=src,
          note='style.is_diverged: the wider analysis rule')
    M.add('numKappaValues', C.Raw(_numbered(sorted(tbl['kappa'].unique()), '{:.0f}')),
          source=src)
    M.add('numKappaB', kp['B'], source=src)
    M.add('numKappaKs', C.Raw(_numbered(kp['ks'], '{:.0f}')), source=src)
    M.add('numKappaMatchedSteps', C.Raw(_numbered(kp['matched'], '{:g}')), source=src,
          note='effective steps realised by every kappa')
    M.add('numKappaNMatched', len(kp['matched']), source=src)
    M.add('numKappaNSeeds', int(df['model_seed'].nunique()), source=src)

    mp = kp['paired']
    k_trunc = min(kp['ks']) if kp['ks'] else None
    for k in kp['ks']:
        sub = mp[mp['k'] == k]
        tag = 'Trunc' if k == k_trunc else 'Full'
        M.add(f'numKappaNSig{tag}', int(sub['significant'].sum()),
              source='reviewer_figs.matched_step_paired',
              note=f'significant kappa pairs at k={int(k)}')
        M.add(f'numKappaNStepsSig{tag}',
              int(len({float(e) for e in sub.loc[sub['significant'], 'eff_step']})),
              source='reviewer_figs.matched_step_paired',
              note=f'matched steps with at least one significant pair at k={int(k)}')
        sp = pd.to_numeric(kp['m_val'][kp['m_val']['k'] == k]['rel_spread'],
                           errors='coerce')
        if np.isfinite(sp).any():
            M.add(f'numKappaSpreadMax{tag}', float(np.nanmax(sp)),
                  source='reviewer_figs.matched_step_table',
                  note=f'largest relative spread over kappa at k={int(k)}')
        stable = kp['m_val'][(kp['m_val']['k'] == k)
                             & (pd.to_numeric(kp['m_val']['eff_step'],
                                              errors='coerce') < 1.0)]
        sp2 = pd.to_numeric(stable['rel_spread'], errors='coerce')
        if np.isfinite(sp2).any():
            M.add(f'numKappaSpreadStable{tag}', float(np.nanmax(sp2)),
                  source='reviewer_figs.matched_step_table',
                  note='largest relative spread below effective step 1')
    for _, r in mp[mp['k'] == k_trunc].iterrows():
        a, b = str(r['pair']).replace('kappa', '').split('-')
        name = f"numKappaT{_word(a)}{_word(b)}Eff{_word(r['eff_step'])}"
        if name in M:
            continue
        M.add(name, _num(r['t']), source='reviewer_figs.matched_step_paired',
              note=f"kappa{a} - kappa{b} at effective step {_num(r['eff_step']):g}, "
                   f"k={int(_num(r['k']))}")
    noise = [float(e) for e in sorted(set(mp['eff_step']))
             if not bool(mp[np.isclose(mp['eff_step'], e)]['significant'].any())]
    M.add('numKappaNoiseSteps', C.Raw(_numbered(noise, '{:g}') if noise else 'none'),
          source='reviewer_figs.matched_step_paired',
          note='matched effective steps where no kappa pair is resolved')
    cov = kp['coverage']
    for _, r in cov.iterrows():
        w = _word(r['kappa'])
        M.add(f'numKappaMaxEffStep{w}', _num(r['max_eff_step']),
              source='reviewer_figs.eff_step_coverage')
        M.add(f'numKappaNAboveMatched{w}', int(r['n_above_matched']),
              source='reviewer_figs.eff_step_coverage')
    div_cells = df[df['diverged'].astype(bool)]
    if len(div_cells):
        M.add('numKappaDivLrMin', float(pd.to_numeric(div_cells['lr']).min()),
              source=src, note='smallest learning rate at which any run diverges')
        eff = 2.0 * pd.to_numeric(div_cells['lr']) / pd.to_numeric(div_cells['kappa'])
        n_out = int(sum(not any(np.isclose(e, m) for m in kp['matched']) for e in eff))
        M.add('numKappaNDivOutsideMatched', n_out, source=src,
              note='divergences that sit outside the matched effective steps')
        for kap in sorted(df['kappa'].unique()):
            sub = df[df['kappa'] == kap]
            M.add(f'numKappaNDiv{_word(kap)}', int(sub['diverged'].astype(bool).sum()),
                  source=src)
    best = tbl.loc[pd.to_numeric(tbl['final_val_loss'], errors='coerce').idxmin()]
    M.add('numKappaBestVal', _num(best['final_val_loss']), source=src)
    M.add('numKappaBestWhere',
          C.Raw(rf"$k={int(_num(best['k']))}$, $\kappa={int(_num(best['kappa']))}$, "
                rf"$2\eta/\kappa={_num(best['eff_step']):g}$"), source=src)
    return kp


# ---------------------------------------------------------------------------
# F14 / T21 / macros -- App. N, micro-batching and parameter masking
# ---------------------------------------------------------------------------
KNOBS = (('microbatch_size', 1.0, MICROBATCH_SCANS, 'micro-batch size $\\mu B$'),
         ('param_fraction', 1.0, PARAMFRAC_SCANS, 'parameter fraction $f$'))
KNOB_TAG = {'microbatch_size': 'Microbatch', 'param_fraction': 'Paramfrac'}


def _ref_lr(df, knob, ref):
    """The learning rate the knob sweep is read at: the best one at the reference value.

    :func:`scan_analysis.knob_ref_lr` with a frame instead of a ``Scan`` (it takes the
    same binding rule through :func:`analysis_helpers.best_per_method`, so it is the same
    number, without loading the scan twice).
    """
    sub = df[np.isclose(pd.to_numeric(df[knob], errors='coerce'), float(ref))]
    best = ah.best_per_method(sub)
    row = _row(best, method=SVEN)
    return None if row is None else _num(row['lr'])


def _knob_frames(data):
    out = {}
    for knob, ref, scans, _ in KNOBS:
        per = {}
        for key, name in scans:
            df = data(name)
            lr = _ref_lr(df, knob, ref)
            tbl = rf.knob_table(df, knob)
            at_lr = tbl[np.isclose(pd.to_numeric(tbl['lr'], errors='coerce'), lr)] \
                if lr is not None else tbl.iloc[:0]
            at_lr = at_lr.sort_values(knob)
            B = int(_num(df['batch_size'].dropna().iloc[0]))
            cap = (lambda v, B=B: B / v) if knob == 'microbatch_size' else None
            chk = rf.rank_vs_cap(at_lr, knob, cap) if cap is not None else None
            per[key] = {'scan': name, 'df': df, 'lr': lr, 'tbl': tbl, 'at_lr': at_lr,
                        'B': B, 'rank_cap': chk,
                        'values': _sorted_unique(at_lr[knob]),
                        'n_recorded': int((df['status'] == style.STATUS_DIVERGED).sum()),
                        'n_wide': int(df['diverged'].astype(bool).sum()),
                        'n_runs': int(len(df)),
                        'div_lrs': sorted(float(v) for v in
                                          df.loc[df['diverged'].astype(bool),
                                                 'lr'].dropna().unique())}
        out[knob] = per
    return out


def _knob_levels(per, knob):
    """Every knob value swept anywhere in a family (the tick list of its axes)."""
    out = set()
    for d in per.values():
        out.update(d.get('values') or ())
    return sorted(out)


def _rel(series, ref_value):
    v = pd.to_numeric(series, errors='coerce')
    return v / ref_value if ref_value and np.isfinite(ref_value) else v * np.nan


def _knob_ref_row(d, knob, ref):
    return _row(d['at_lr'], **{knob: ref})


def _fig_knobs(ctx, opts):
    knobs = ctx.knobs()
    fig, axes = _subplots(opts)
    xlab = {k: lab for k, _, _, lab in KNOBS}
    tasks = list(opts['tasks'])
    drawn = []

    for j, (knob, ref, scans, _) in enumerate(KNOBS):
        ax = axes[0][j]
        for key, _name in scans:
            d = knobs[knob].get(key)
            if d is None or d['at_lr'].empty or key not in tasks:
                continue
            r0 = _knob_ref_row(d, knob, ref)
            if r0 is None:
                continue
            base = _num(r0['final_val_loss'])
            s = d['at_lr']
            ax.plot(s[knob], _rel(s['final_val_loss'], base), '-o', lw=opts['lw'],
                    ms=opts['ms'], color=SCAN_COLORS[key], label=_task_title(key))
            drawn.append((knob, key))
        if opts['reference_line']:
            ax.axhline(1.0, ls='-', c='0.6', lw=0.5, zorder=0)
        ax.set_yscale('log')
        if knob == 'microbatch_size':
            ax.set_xscale('log', base=2)
            rf.arm_ticks(ax, _knob_levels(knobs[knob], knob), fmt=opts['arm_tick_fmt'])
        ax.set_xlabel(xlab[knob])
        ax.set_ylabel('val. loss / reference' if j == 0 else '')
        ax.set_title(f'Quality vs {"micro-batching" if j == 0 else "parameter masking"}')
        _grid(ax)

    def cost_panel(ax, col, ylabel, title):
        # both knobs on one axis, each scaled to its own maximum, relative to the
        # reference setting: solid = micro-batching, dashed = parameter masking
        for knob, ref, scans, _ in KNOBS:
            ls = '-' if knob == 'microbatch_size' else '--'
            for key, _name in scans:
                d = knobs[knob].get(key)
                if d is None or d['at_lr'].empty or key not in tasks:
                    continue
                r0 = _knob_ref_row(d, knob, ref)
                if r0 is None or col not in d['at_lr']:
                    continue
                base = _num(r0[col])
                s = d['at_lr']
                x = pd.to_numeric(s[knob], errors='coerce')
                x = x / x.max() if knob == 'microbatch_size' else x
                ax.plot(x, _rel(s[col], base), ls=ls, marker='o', lw=opts['time_lw'],
                        ms=opts['time_ms'], color=SCAN_COLORS[key])
        if opts['reference_line']:
            ax.axhline(1.0, ls='-', c='0.6', lw=0.5, zorder=0)
        ax.set_xscale('log')
        # a linear ratio axis: everything sits between ~0.9 and ~2.4, where a log
        # axis only makes the ticks harder to read
        if opts['cost_logy']:
            ax.set_yscale('log')
        ax.set_xlabel(r'knob / its maximum ($\mu B/B$ or $f$)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _grid(ax)

    if opts['bottom_left'] == 'memory':
        cost_panel(axes[1][0], 'peak_mem_mb', 'peak memory / reference',
                   'Peak memory: neither knob saves much')
    else:                                   # the used-rank panel, kept as an option
        ax = axes[1][0]
        for key, _name in MICROBATCH_SCANS:
            d = knobs['microbatch_size'].get(key)
            if d is None or d['rank_cap'] is None or d['rank_cap'].empty \
                    or key not in tasks:
                continue
            chk = d['rank_cap'].sort_values('microbatch_size')
            ax.plot(chk['microbatch_size'], chk['rank_used'], '-o', lw=opts['lw'],
                    ms=opts['ms'], color=SCAN_COLORS[key], label=_task_title(key))
            if opts['cap_line']:
                ax.plot(chk['microbatch_size'], chk['cap'], ':', lw=opts['cap_lw'],
                        color=SCAN_COLORS[key])
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        rf.arm_ticks(ax, _knob_levels(knobs['microbatch_size'], 'microbatch_size'),
                     fmt=opts['arm_tick_fmt'])
        ax.set_xlabel(xlab['microbatch_size'])
        ax.set_ylabel('singular values kept / step')
        ax.set_title(r'Used rank vs the cap $B/\mu B$ (dotted)')
        _grid(ax)

    cost_panel(axes[1][1], 'step_s', 'ms per step / reference', 'Neither knob buys time')

    handles, labels = _axes_handles([axes[0][0], axes[0][1]])
    if opts['family_proxies']:
        handles += [_proxy(r'micro-batch ($\mu B$)', color='0.35', ls='-'),
                    _proxy('parameter fraction ($f$)', color='0.35', ls='--')]
        labels += [h.get_label() for h in handles[-2:]]
    _figure_legend(fig, handles, labels, opts['legend_below'])
    return fig, {'axes': axes, 'provenance': ctx.prov('knobs'), 'drawn': drawn,
                 'style_fraction': opts['style_fraction']}


def _table_knobs(knobs, provenance):
    parts = []
    for knob, ref, scans, lab in KNOBS:
        rows = []
        for key, _name in scans:
            d = knobs[knob].get(key)
            if d is None or d['at_lr'].empty:
                continue
            rows.append(C.span_row(
                f"{_task_title(key)}  ($B = {d['B']}$, $\\eta = {d['lr']:g}$)"))
            chk = None if d['rank_cap'] is None else d['rank_cap'].set_index(knob)
            for _, r in d['at_lr'].iterrows():
                v = _num(r[knob])
                rec = {
                    'knob': C.fmt_sig(v, 3),
                    'fin./att.': r['counts'],
                    'val. loss': C.fmt_pm(_num(r['final_val_loss']),
                                          _num(r['final_val_loss_std'])),
                    'test loss': C.fmt_sig(_num(r['final_test_loss']), 3),
                    'test acc.': (C.fmt_pct(_num(r['final_test_acc']), 1)
                                  if np.isfinite(_num(r['final_test_acc'])) else '--'),
                    'used rank': C.fmt_sig(_num(r['rank_eff']), 3),
                    'ms/step': C.fmt_sig(_num(r['step_s']) * 1e3, 3),
                    'peak MB': C.fmt_sig(_num(r['peak_mem_mb']), 4),
                }
                if chk is not None and v in chk.index:
                    rec['cap'] = C.fmt_sig(_num(chk.loc[v, 'cap']), 3)
                if knob == 'param_fraction' and 'actual_param_fraction' in r.index:
                    rec['measured $f$'] = C.fmt_sig(_num(r['actual_param_fraction']), 3)
                rows.append(rec)
        if not rows:
            continue
        head = ('Micro-batching. The Jacobian is captured in $B/\\mu B$ chunks, so the '
                'rank the update can use is capped at $B/\\mu B$; the used rank is '
                '$\\min(k,\\ \\text{\\texttt{rtol}-rank},\\ B/\\mu B)$, which is why it '
                'equals the '
                'cap only where the cap is the binding constraint.'
                if knob == 'microbatch_size' else
                'Parameter masking. A random fraction $f$ of the parameters is updated, '
                'resampled every step; "measured $f$" is the fraction the mask really hit '
                '(elementwise masking of a 593-parameter network cannot hit 0.1 exactly). '
                'A row with no finished run is the sweep\'s own answer at that $f$, not a '
                'missing measurement.')
        parts.append(C.booktabs(
            rows, fit=True,
            caption=(f'{head} One fixed Sven configuration per scan, read at the learning '
                     f'rate selected at the reference setting ($\\mu B=1$ / $f=1$) rather '
                     f'than re-tuned per setting. Times and memory are the scan\'s own '
                     f'measurements, so they are upper bounds (many runs shared each GPU); '
                     # one spelling of the seed band, from the analysis layer's constant
                     f'the seed band is {paired.SEED_SPREAD_LABEL}.'),
            label=f'tab:knobs-{KNOB_TAG[knob].lower()}'))
    path = C.TABLE_DIR / 'knobs.tex'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(parts))
    C.write_provenance(path, provenance)
    return path


def _prov_knobs(ctx=None):
    return C.provenance(
        functions=('reviewer_figs.knob_table', 'reviewer_figs.rank_vs_cap',
                   'reviewer_figs.plot_knob', 'analysis_helpers.best_per_method'),
        scans=[s for _, s in MICROBATCH_SCANS] + [s for _, s in PARAMFRAC_SCANS],
        note='App. N (micro-batch / parameter fraction): F14, T21')


def build_knobs(data, M, want, report):
    prov = data.prov('knobs')
    knobs = data.knobs()
    if want['figures']:
        _figures('knobs', data, report)
    if want['tables']:
        report['tables'].append(str(_table_knobs(knobs, prov)))
    if not want['numbers']:
        return knobs

    src = 'reviewer_figs.knob_table'
    for knob, ref, scans_list, _ in KNOBS:
        tag = KNOB_TAG[knob]
        per = knobs[knob]
        M.add(f'num{tag}NScans', len(per), source=src)
        tot_runs = sum(d['n_runs'] for d in per.values())
        tot_rec = sum(d['n_recorded'] for d in per.values())
        tot_wide = sum(d['n_wide'] for d in per.values())
        M.add(f'num{tag}NRuns', tot_runs, source=src)
        M.add(f'num{tag}NRecorded', tot_rec, source=src,
              note='records the runner marked diverged over the four scans')
        M.add(f'num{tag}NDivWide', tot_wide, source=src,
              note='style.is_diverged over the four scans')
        lrs = sorted({v for d in per.values() for v in d['div_lrs']})
        if lrs:
            M.add(f'num{tag}DivLrMin', float(min(lrs)), source=src,
                  note='smallest learning rate at which any run diverges')
        n_match = n_cells = 0
        for key, d in per.items():
            k = TASK_KEY[key]
            if d['at_lr'].empty:
                continue
            M.add(f'num{tag}{k}Lr', d['lr'], source='analysis_helpers.best_per_method',
                  note='the learning rate selected at the reference setting')
            M.add(f'num{tag}{k}Values', C.Raw(_numbered(d['values'], '{:g}')), source=src)
            r0 = _knob_ref_row(d, knob, ref)
            if r0 is not None:
                M.add_pm(f'num{tag}{k}ValRef', _num(r0['final_val_loss']),
                         _num(r0['final_val_loss_std']), source=src,
                         note='at the reference setting of the knob')
                M.add(f'num{tag}{k}MsStepRef', _num(r0['step_s']) * 1e3, source=src)
                M.add(f'num{tag}{k}MemMbRef', _num(r0['peak_mem_mb']), source=src)
                # the far end of the knob: the most aggressive setting that still has a
                # finished run.  On MNIST label regression the parameter-fraction sweep
                # has none below f = 0.5 at its reference learning rate, and that is
                # reported as a count of empty arms rather than as a loss.
                far = d['at_lr'].sort_values(knob,
                                             ascending=(knob == 'param_fraction'))
                ok = far[np.isfinite(pd.to_numeric(far['final_val_loss'],
                                                   errors='coerce'))]
                blank = int(len(far) - len(ok))
                M.add(f'num{tag}{k}NBlank', blank, source=src,
                      note='knob settings with no finished run at the reference lr')
                edge = ok.iloc[0] if len(ok) else None
                if edge is not None:
                    M.add(f'num{tag}{k}Edge', _num(edge[knob]), source=src,
                          note='the most aggressive setting with a finished run')
                    _maybe(M, f'num{tag}{k}ValEdge', _num(edge['final_val_loss']),
                           source=src, note=f'at {knob} = {_num(edge[knob]):g}')
                    _maybe(M, f'num{tag}{k}ValEdgeRatio',
                           _num(edge['final_val_loss']) / _num(r0['final_val_loss']),
                           source=src,
                           note='loss at the far end of the knob / at the reference')
                    if _num(r0['step_s']):
                        _maybe(M, f'num{tag}{k}MsStepEdgeRatio',
                               _num(edge['step_s']) / _num(r0['step_s']), source=src)
                    if _num(r0['peak_mem_mb']):
                        _maybe(M, f'num{tag}{k}MemEdgeRatio',
                               _num(edge['peak_mem_mb']) / _num(r0['peak_mem_mb']),
                               source=src)
            if d['rank_cap'] is not None and len(d['rank_cap']):
                chk = d['rank_cap']
                M.add(f'num{tag}{k}RankLawHits', int(chk.attrs['n_match']),
                      source='reviewer_figs.rank_vs_cap',
                      note='cells where the used rank equals the cap to 1%')
                M.add(f'num{tag}{k}RankLawCells', int(len(chk)),
                      source='reviewer_figs.rank_vs_cap')
                n_match += int(chk.attrs['n_match'])
                n_cells += int(len(chk))
        if n_cells:
            M.add(f'num{tag}RankLawHits', n_match, source='reviewer_figs.rank_vs_cap')
            M.add(f'num{tag}RankLawCells', n_cells, source='reviewer_figs.rank_vs_cap')
    return knobs


# ---------------------------------------------------------------------------
# F4 / T6 / macros -- App. F, the tuning budget
# ---------------------------------------------------------------------------
def _budget_frames(root=None):
    out = {}
    for scan in BUDGET_SCANS:
        curves, at_n = hl.best_of_n_table(scan, results_root=root)
        out[scan] = {'curves': curves, 'at_n': at_n,
                     'traj': hl.budget_table(scan, results_root=root),
                     'equal': hf.equal_budget_table(curves, n=hf.BUDGET_EQUAL_N),
                     'closing': hf._equal_budget(curves, n=hf.BUDGET_EQUAL_N)}
    return out


def _fig_budget(ctx, opts):
    bud = ctx.budget()
    panels = [s for s in opts['panels'] if s in bud]
    fig, axes = _subplots(opts, ncol=len(panels) or 1)
    for ax, scan in zip(axes[0], panels):
        curves = bud[scan]['curves']
        methods = [m for m in curves['method'].unique()
                   if not opts['methods'] or hl.method_key(m) in opts['methods']]
        methods = ([m for m in methods if hl.method_key(m) == SVEN]
                   + sorted(m for m in methods if hl.method_key(m) != SVEN))
        budget.plot_best_of_n(curves, ax, methods=methods)
        _thin(ax, sven_lw=opts['sven_lw'], other_lw=opts['other_lw'],
              ms=opts['thin_ms'])
        if opts['equal_budget_line']:
            ax.axvline(hf.BUDGET_EQUAL_N, ls=':', c=opts['equal_budget_color'], lw=0.8)
        if opts['annotate_equal_budget']:
            ax.annotate(f'$n={hf.BUDGET_EQUAL_N}$', (hf.BUDGET_EQUAL_N, 0.02),
                        xycoords=('data', 'axes fraction'),
                        fontsize=opts['annotation_fontsize'],
                        color=opts['equal_budget_color'], ha='left', va='bottom')
        ax.set_title(hl.scan_title(scan))
        ax.set_xlabel('tuning budget $n$ (random draws)')
        ax.set_ylabel('expected best val. loss')
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        ax.grid(True, which='major', ls='--', lw=0.35, alpha=0.45)
        ax.grid(False, which='minor')
    handles, labels = C.method_handles(
        [hl.method_key(m) for m in bud[panels[0]]['curves']['method'].unique()
         if not opts['methods'] or hl.method_key(m) in opts['methods']],
        lw=opts['legend_lw']) if panels else ([], [])
    order = np.argsort([0 if l == style.method_label(SVEN) else 1 for l in labels],
                       kind='stable')
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    _figure_legend(fig, handles, labels, opts['legend_below'])
    return fig, {'axes': axes, 'provenance': ctx.prov('budget'), 'panels': panels,
                 'methods': labels, 'style_fraction': opts['style_fraction']}


def _table_budget(bud, provenance):
    """T6a -- grid points against distinct trajectories, per method and scan."""
    methods = []
    for scan in BUDGET_SCANS:
        methods.extend(hl.method_key(m) for m in bud[scan]['traj']['method'])
    methods = ([SVEN] + sorted({m for m in methods if m != SVEN}))
    rows = []
    for m in methods:
        rec = {'Method': style.method_label(m)}
        for scan in BUDGET_SCANS:
            t = bud[scan]['traj'].copy()
            t['key'] = [hl.method_key(x) for x in t['method']]
            r = _row(t, key=m)
            title = hl.scan_title(scan)
            if r is None:
                rec[title] = '--'
                continue
            rec[title] = C.Raw(f"{_num(r['distinct']):.3g} / {int(r['grid_points'])}")
        rows.append(rec)
    return C.write_table(
        'budget', rows, provenance_record=provenance,
        caption=('Tuning budget disclosure: distinct trajectories out of grid points, per '
                 f'method and scan. A Sven grid point whose {RTOL_TEX}-rank stays below $k$ '
                 'is '
                 'the identical trajectory, so the search a grid really performs is '
                 'smaller than the grid; a single-knob baseline has one trajectory per '
                 'point. Counts are the mean over $(\\text{seed},\\eta)$ groups scaled to '
                 'the whole grid.'),
        label='tab:budget',
        notes=('budget.trajectory_table via headline.budget_table',))


def _table_equal_budget(bud, provenance):
    """T6b -- every method's expected best loss at a budget of n = 8 trials."""
    rows = []
    for scan in BUDGET_SCANS:
        eq = bud[scan]['equal']
        c = bud[scan]['closing']
        sven = _row(eq, method=SVEN)
        i = list(eq['method']).index(SVEN) + 1 if SVEN in list(eq['method']) else np.nan
        rows.append({
            'Scan': hl.scan_title(scan),
            '$n$': C.fmt_int(eq.attrs.get('n', hf.BUDGET_EQUAL_N)),
            'Sven expected best': C.fmt_sig(_num(None if sven is None
                                                 else sven['expected_best']), 4),
            "Sven's rank at $n$": C.fmt_rank(i, len(eq)),
            'Best method at $n$': eq.iloc[0]['display'] if len(eq) else '--',
            'its value': C.fmt_sig(_num(eq.iloc[0]['expected_best']) if len(eq)
                                   else float('nan'), 4),
            'Sven grid': C.fmt_int(_num(None if sven is None else sven['grid_points'])),
            '$n$ Sven needs to match it': (
                C.fmt_int(c['sven_n_to_match_best_at_equal_n'])
                if c.get('sven_n_to_match_best_at_equal_n') else 'never'),
        })
    return C.write_table(
        'equal_budget', rows, provenance_record=provenance, fit=True,
        caption=(f'Equal tuning budget. Expected best seed-mean validation loss when '
                 f'$n={hf.BUDGET_EQUAL_N}$ configurations are drawn uniformly at random '
                 f'from each method\'s grid (exact order statistics of the finite '
                 f'population; a configuration no seed finished is scored at the worst '
                 f'seed mean in the scan, because a random search would still have spent a '
                 f'trial on it). A method whose grid is smaller than $n$ is read at its own '
                 f'grid size. The last column is the budget Sven needs before its expected '
                 f'best matches the best method\'s value at $n={hf.BUDGET_EQUAL_N}$.'),
        label='tab:equal-budget',
        notes=('headline_figs.equal_budget_table; budget.best_of_n_curve',))


def _table_optimism(opt, provenance):
    """T6c -- tuning-to-confirmation gap: selection optimism, and instance variation.

    **Not called by** :func:`build_budget`: ``main`` owns ``tables_v2/optimism.tex`` (see
    the comment there).  Kept so App. F can be given its own rendering if the integrator
    decides the appendix wants one, and so the columns App. F's prose refers to are on
    record next to the macros that quote them.
    """
    rows = []
    for _, r in opt.iterrows():
        rows.append({
            'Scan': r['scan'],
            'methods': C.fmt_int(r['n_methods']),
            'data seeds': C.fmt_int(r['n_data_seeds']),
            'median (same instance)': C.fmt_pct_value(_num(r['median_same_%']), 1,
                                                      signed=True),
            'Sven (same instance)': C.fmt_pct_value(_num(r['sven_same_%']), 1, signed=True),
            'worse than tuning': C.fmt_int(r['n_worse_same']),
            'median (pooled)': C.fmt_pct_value(_num(r['median_pooled_%']), 1, signed=True),
            'Sven (pooled)': C.fmt_pct_value(_num(r['sven_pooled_%']), 1, signed=True),
        })
    return C.write_table(
        'optimism', rows, provenance_record=provenance, fit=True,
        caption=('Selection optimism, as the relative change from the tuning pass to the '
                 'confirmation pass. The same-instance columns re-run the selected '
                 'configuration on fresh model seeds and the same problem instance: that '
                 'gap IS selection optimism. The pooled columns also cross the two extra '
                 'data seeds of the two synthetic tasks, so most of what they show there is '
                 'problem-instance variation, not optimism. Positive = worse on '
                 'confirmation.'),
        label='tab:optimism',
        notes=('headline.selection_optimism_table',))


def _prov_budget(ctx=None):
    return C.provenance(
        functions=('headline.budget_table', 'headline.best_of_n_table',
                   'headline.selection_optimism_table', 'budget.trajectory_table',
                   'budget.best_of_n_curve', 'budget.plot_best_of_n',
                   'headline_figs.equal_budget_table'),
        scans=BUDGET_SCANS, note='App. F (tuning budget): F4, T6',
        provisional=C.provisional_scans(list(BUDGET_SCANS)))


def build_budget(data, M, want, report):
    root = data.root
    prov = data.prov('budget')
    bud = data.budget()
    if want['figures']:
        _figures('budget', data, report)
    if want['tables']:
        report['tables'].append(str(_table_budget(bud, prov)))
        report['tables'].append(str(_table_equal_budget(bud, prov)))
    # ``tables_v2/optimism.tex`` is NOT written here.  The plan's T6 row lists it under
    # App. F with this module, but its section 5.2 assigns the file to ``main``, which
    # implements it as a view of the confirmation tables it already holds
    # (``main._t6_optimism``).  Both modules writing one path made the file's contents
    # depend on which module ran last -- so main keeps it, and this module quotes the same
    # ``headline.selection_optimism_table`` frame through its macros only.
    opt = hl.selection_optimism_table(results_root=root)
    if not want['numbers']:
        return bud

    M.add('numBudgetEqualN', hf.BUDGET_EQUAL_N,
          source='headline_figs.BUDGET_EQUAL_N',
          note='the smallest grid any baseline with a swept learning rate has')
    sven_fracs = []
    for scan in BUDGET_SCANS:
        tag = TASK_KEY[SCAN_TASK[scan]]
        t = bud[scan]['traj'].copy()
        t['key'] = [hl.method_key(m) for m in t['method']]
        sv = _row(t, key=SVEN)
        if sv is not None:
            M.add(f'numBudget{tag}SvenDistinct', _num(sv['distinct']),
                  source='budget.trajectory_table')
            M.add(f'numBudget{tag}SvenGrid', int(sv['grid_points']),
                  source='budget.trajectory_table')
            M.add(f'numBudget{tag}SvenDistinctFrac', C.fmt_pct(_num(sv['distinct_frac']), 1),
                  source='budget.trajectory_table')
            sven_fracs.append(_num(sv['distinct_frac']))
        eq = bud[scan]['equal']
        names = list(eq['method'])
        if SVEN in names:
            M.add(f'numBudget{tag}SvenRankEqual', names.index(SVEN) + 1,
                  source='headline_figs.equal_budget_table',
                  note=f'rank at a budget of {hf.BUDGET_EQUAL_N} trials')
        M.add(f'numBudget{tag}NMethodsEqual', int(len(eq)),
              source='headline_figs.equal_budget_table')
        if len(eq):
            M.add(f'numBudget{tag}BestEqual', C.Raw(str(eq.iloc[0]['display'])),
                  source='headline_figs.equal_budget_table')
        c = bud[scan]['closing']
        if c.get('sven_best_of_n') is not None:
            M.add(f'numBudget{tag}SvenBestOfN', float(c['sven_best_of_n']),
                  source='headline_figs.equal_budget_table')
        n_match = c.get('sven_n_to_match_best_at_equal_n')
        M.add(f'numBudget{tag}SvenNToMatch',
              C.Raw(C.fmt_int(n_match) if n_match else 'never'),
              source='headline_figs._equal_budget',
              note='budget at which Sven matches the best method at the equal budget')
    if sven_fracs:
        M.add('numBudgetSvenDistinctFracMin', C.fmt_pct(min(sven_fracs), 1),
              source='budget.trajectory_table')
        M.add('numBudgetSvenDistinctFracMax', C.fmt_pct(max(sven_fracs), 1),
              source='budget.trajectory_table')
    for _, r in opt.iterrows():
        key = {v: k for k, v in
               {t: hl.scan_title(s) for s, t in SCAN_TASK.items()}.items()}
        tag = None
        for scan, task in SCAN_TASK.items():
            if hl.scan_title(scan) == r['scan']:
                tag = TASK_KEY[task]
        if tag is None:
            continue
        M.add(f'numOptimism{tag}Median', C.fmt_pct_value(_num(r['median_same_%']), 1,
                                                         signed=True),
              source='headline.selection_optimism_table', note='same instance')
        M.add(f'numOptimism{tag}Sven', C.fmt_pct_value(_num(r['sven_same_%']), 1,
                                                       signed=True),
              source='headline.selection_optimism_table', note='same instance')
    return bud


# ---------------------------------------------------------------------------
# F10 / F10b / T11 / macros -- App. P, robustness and divergence accounting
# ---------------------------------------------------------------------------
def _divergence_frames(root=None):
    """The campaign-wide count (macros and the parked T11); the per-method counts on the
    four headline grids; and Sven's (rtol, eta) maps on those plus the sweeps."""
    camp = hf.campaign_divergence(results_root=root)
    per_method, patterns, grids = {}, {}, {}
    for scan in BUDGET_SCANS:
        df = hl.load(scan, '', results_root=root)
        per_method[scan] = hf.divergence_by_method(df)
        patterns[scan] = hf.divergence_pattern(df, index='rtol', columns='lr')
        # a grid with NO divergence still draws (all zeros) and still prints its axis
        grids[scan] = hf.sven_divergence_grid(df, index='rtol', columns='lr')
        if not patterns[scan].get('rtol_grid'):
            sv = df[df['method'] == SVEN]
            patterns[scan]['rtol_grid'] = _sorted_unique(sv['rtol']) if len(sv) else []
    for scan in DIVERGENCE_SWEEP_SCANS:
        df = hl.load(scan, '', results_root=root)
        grids[scan] = hf.sven_divergence_grid(df, index='rtol', columns='lr')
    return {'campaign': camp, 'per_method': per_method, 'patterns': patterns,
            'grids': grids}


def _divergence_grids(dv, opts, key='scans'):
    """The grids a figure draws: the ``key`` knob, in its order, of what was read."""
    return [s for s in opts[key] if dv['grids'].get(s, (None,))[0] is not None]


def _fig_divergence(ctx, opts):
    dv = ctx.divergence()
    fig, axes = _subplots(opts)

    ax = axes[0][0]
    for i, scan in enumerate(_divergence_grids(dv, opts, 'rtol_scans')):
        frac, counts = dv['grids'][scan]
        n = np.nan_to_num(counts.values.astype(float))
        bad = np.nan_to_num(frac.values.astype(float)) * n
        by_rtol = pd.Series(bad.sum(axis=1) / np.where(n.sum(axis=1) > 0, n.sum(axis=1), 1),
                            index=[float(v) for v in frac.index]).sort_index()
        # the sweeps (P/N, batch size) are dashed: their lr axes are shared with the
        # baselines and reach values that break everything, unlike the tuning grids
        ls = opts['sweep_ls'] if scan in DIVERGENCE_SWEEP_SCANS else '-'
        ax.plot(by_rtol.index, by_rtol.to_numpy(), ls, marker='o', lw=opts['lw'],
                ms=opts['ms'], color=_grid_color(scan, i), label=_scan_label(scan))
    ax.set_xscale('log')
    ax.set_xlabel(f'{RTOL_FIG} (the relative singular-value cut)')
    ax.set_ylabel('fraction of Sven runs diverged')
    ax.set_title(f'Divergence lives at the bottom of the {RTOL_FIG} grid')
    _grid(ax)

    ax = axes[0][1]
    scans = [s for s in opts['scans'] if s in dv['per_method']]
    methods = []
    for scan in scans:
        methods.extend(dv['per_method'][scan]['method'].tolist())
    order = ([SVEN] + sorted({m for m in methods if m != SVEN}))
    for y, m in enumerate(order):
        for i, scan in enumerate(scans):
            t = dv['per_method'][scan]
            r = _row(t, method=m)
            if r is None:
                continue
            off = (i - (len(scans) - 1) / 2) * opts['scan_offset']
            ax.plot([_num(r['frac_diverged'])], [y + off], 'o', ms=opts['dot_ms'],
                    color=_grid_color(scan, i), mfc=_grid_color(scan, i),
                    label=_scan_label(scan) if y == 0 else None)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([style.method_label(m) for m in order],
                       fontsize=opts['method_labelsize'])
    ax.invert_yaxis()
    ax.set_xlabel('fraction of the grid that diverged')
    ax.set_title('Per method, on the four MLP grids')
    ax.grid(axis='x', ls='--', lw=0.35, alpha=0.45)
    _panel_legend(ax, opts['panel_legend'])

    handles, labels = _axes_handles([axes[0][0]])
    _figure_legend(fig, handles, labels, opts['legend_below'])
    return fig, {'axes': axes, 'provenance': ctx.prov('divergence'),
                 'scans': scans, 'grids': _divergence_grids(dv, opts, 'rtol_scans'),
                 'methods': order,
                 'style_fraction': opts['style_fraction']}


def _fig_divergence_grids(ctx, opts):
    import matplotlib.pyplot as plt

    dv = ctx.divergence()
    figspec.paper_style(opts['style_fraction'])
    top = _divergence_grids(dv, opts)
    n = len(top)
    ncol = int(opts['ncol'])
    nrow = int(np.ceil(n / ncol)) or 1
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=C.figsize(ncol, nrow, opts['fraction'],
                                               aspect=opts['aspect'],
                                               extra_h=opts['extra_h']))
    for i, scan in enumerate(top):
        ax = axes[i // ncol][i % ncol]
        frac, counts = dv['grids'][scan]
        if frac is None:
            ax.axis('off')
            continue
        hf.plot_divergence_grid(frac, ax, counts=counts, title=_scan_label(scan),
                                xlabel=r'$\eta$', ylabel=RTOL_FIG)
        # the colour already carries the fraction; at 1.8 in per panel the cell has room
        # for one line, so it keeps the COUNTS (the denominators are the point of the
        # table) and drops the percentage the helper prints above them
        for text in ax.texts:
            if opts['counts_only']:
                text.set_text(text.get_text().split('\n')[-1])
            text.set_fontsize(opts['cell_fontsize'])
        ax.set_yticklabels([f'${_tex_num(v)}$' for v in frac.index])
        ax.tick_params(labelsize=opts['panel_ticksize'])
        ax.title.set_fontsize(opts['title_fontsize'])
        ax.xaxis.label.set_fontsize(opts['label_fontsize'])
        ax.yaxis.label.set_fontsize(opts['label_fontsize'])
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    return fig, {'axes': axes, 'provenance': ctx.prov('divergence'), 'grids': top,
                 'style_fraction': opts['style_fraction']}


def _table_divergence(dv, provenance):
    camp = dv['campaign']
    rows = []
    for _, r in camp.iterrows():
        rows.append({
            'Tuning grid': _scan_label(r['scan']),
            'Sven runs': C.fmt_int(r['n_runs']),
            'diverged (wide rule)': C.fmt_int(r['n_diverged']),
            'recorded by the runner': C.fmt_int(r['n_recorded']),
            'rate': C.fmt_pct(_num(r['frac_diverged']), 2),
        })
    rows.append({
        'Tuning grid': C.Raw(r'\textbf{all grids}'),
        'Sven runs': C.fmt_int(camp.attrs['n_runs']),
        'diverged (wide rule)': C.fmt_int(camp.attrs['n_diverged']),
        'recorded by the runner': C.fmt_int(camp.attrs['n_recorded']),
        'rate': C.fmt_pct(_num(camp.attrs['frac_diverged']), 2),
    })
    parts = [C.booktabs(
        rows, midrules=(len(rows) - 1,),
        caption=('Sven\'s failures over every tuning grid of the campaign. "Recorded" is '
                 'the runner\'s own lifecycle count (the step raised or produced a '
                 'non-finite batch loss); "wide rule" is the analysis definition that '
                 'governs selection and every table in this paper (recorded, or a '
                 'non-finite final value, or a finite blow-up to more than ten times the '
                 'initial validation loss). Confirmation, timing and diagnostic passes '
                 're-run an already selected configuration and are not part of any search, '
                 'so they are not counted here.'),
        label='tab:divergence')]

    mrows = []
    methods = []
    for scan in BUDGET_SCANS:
        methods.extend(dv['per_method'][scan]['method'].tolist())
    for m in [SVEN] + sorted({x for x in methods if x != SVEN}):
        rec = {'Method': style.method_label(m)}
        for scan in BUDGET_SCANS:
            r = _row(dv['per_method'][scan], method=m)
            rec[hl.scan_title(scan)] = (
                '--' if r is None
                else C.Raw(f"{int(r['n_diverged'])}/{int(r['n_runs'])}"))
        mrows.append(rec)
    parts.append(C.booktabs(
        mrows, fit=True,
        caption=('Failures per method on the four MLP tuning grids, as '
                 'diverged/attempted under the wide rule. The grids are not the same size '
                 f'(Sven sweeps a {RTOL_TEX} axis the baselines do not have, and that axis '
                 'deliberately includes values known to break), so the counts are read '
                 'with their denominators. K-FAC has no eligible configuration on either '
                 'MNIST scan.'),
        label='tab:divergence-methods'))

    prows = []
    for scan, pat in dv['patterns'].items():
        if not pat.get('n_runs'):
            continue
        prows.append({
            'Tuning grid': _scan_label(scan),
            'Sven runs': C.fmt_int(pat['n_runs']),
            'diverged': C.fmt_int(pat['n_diverged']),
            f'{RTOL_TEX} affected': C.Raw('none' if not pat.get('rtol_affected') else
                                          _tex_list(pat['rtol_affected'])),
            f'{RTOL_TEX} swept': C.Raw('--' if not pat.get('rtol_grid') else
                                       _tex_list(sorted(pat['rtol_grid']))),
            r'$\eta$ affected': C.Raw('none' if not pat.get('lr_affected') else
                                      _tex_list(pat['lr_affected'])),
            # 'no' would read as a measurement on a grid with nothing to measure
            'rate rises with $\\eta$': (('yes' if pat.get('rises_with_lr') else 'no')
                                       if pat['n_diverged'] else '--'),
            f'confined to low {RTOL_TEX}': (('yes' if pat.get('confined_to_low_rtol')
                                             else 'no') if pat['n_diverged'] else '--'),
        })
    if prows:
        parts.append(C.booktabs(
            prows, fit=True,
            caption=('Where Sven diverges. The mechanism is visible in the grid: a loose '
                     f'{RTOL_TEX} inverts near-noise Gram directions, and the risk grows '
                     'with the learning rate. It is a per-grid statement, not a property of '
                     'the method: on several scans the count is zero at every '
                     f'{RTOL_TEX} and every $\\eta$.'),
            label='tab:divergence-pattern'))
    path = C.TABLE_DIR / 'divergence.tex'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(parts))
    C.write_provenance(path, provenance)
    return path


def _prov_divergence(ctx=None):
    root = None if ctx is None else ctx.root
    return C.provenance(
        functions=('headline_figs.campaign_divergence',
                   'headline_figs.divergence_by_method',
                   'headline_figs.divergence_pattern',
                   'headline_figs.sven_divergence_grid',
                   'headline_figs.plot_divergence_grid'),
        scans=hf.on_grid_scans(root),
        note='App. P (robustness): F10, F10b, T11. The campaign-wide denominator is every '
             'ON-GRID (search) run: the confirmation, timing and diagnostic passes re-run '
             'an already selected configuration and are excluded.'
             + (' A scan it reads is still refreshing, so the denominator can move.'
                if C.provisional_scans() else ''),
        provisional=C.provisional_scans())


def build_divergence(data, M, want, report):
    prov = data.prov('divergence')
    dv = data.divergence()
    if want['figures']:
        _figures('divergence', data, report)
    if want['tables']:
        report['tables'].append(str(_table_divergence(dv, prov)))
    if not want['numbers']:
        return dv

    camp = dv['campaign']
    src = 'headline_figs.campaign_divergence'
    prov_flag = bool(C.provisional_scans())
    M.add('numDivSvenNRuns', int(camp.attrs['n_runs']), source=src,
          note='Sven runs on the campaign tuning grids', provisional=prov_flag)
    M.add('numDivSvenNDiverged', int(camp.attrs['n_diverged']), source=src,
          provisional=prov_flag)
    M.add('numDivSvenNRecorded', int(camp.attrs['n_recorded']), source=src,
          provisional=prov_flag)
    M.add('numDivSvenDivFrac', C.fmt_pct(_num(camp.attrs['frac_diverged']), 1),
          source=src, provisional=prov_flag)
    M.add('numDivNGrids', int(camp.attrs['n_scans']), source=src, provisional=prov_flag)
    M.add('numDivNGridsAffected', int(camp.attrs['n_scans_affected']), source=src,
          provisional=prov_flag)
    zero = [_scan_label(s) for s in camp[camp['n_diverged'] == 0]['scan']]
    M.add('numDivSvenZeroGrids', C.Raw(_join(zero) if zero else 'no grid'), source=src,
          note='grids where Sven has no failure under either definition',
          provisional=prov_flag)
    worst = camp.iloc[0] if len(camp) else None
    if worst is not None:
        M.add('numDivSvenWorstGrid', C.Raw(_scan_label(worst['scan'])), source=src)
        M.add('numDivSvenWorstN', int(worst['n_diverged']), source=src)
        M.add('numDivSvenWorstOf', int(worst['n_runs']), source=src)
        M.add('numDivSvenWorstFrac', C.fmt_pct(_num(worst['frac_diverged']), 1),
              source=src)
    for scan in BUDGET_SCANS:
        tag = TASK_KEY[SCAN_TASK[scan]]
        r = _row(camp, scan=scan)
        if r is not None:
            M.add(f'numDiv{tag}SvenNDiverged', int(r['n_diverged']), source=src)
            M.add(f'numDiv{tag}SvenNRuns', int(r['n_runs']), source=src)
            M.add(f'numDiv{tag}SvenDivFrac', C.fmt_pct(_num(r['frac_diverged']), 2),
                  source=src)
        pat = dv['patterns'].get(scan) or {}
        if pat.get('rtol_affected'):
            M.add(f'numDiv{tag}SvenRtols',
                  C.Raw(_tex_list(pat['rtol_affected'])),
                  source='headline_figs.divergence_pattern',
                  note='the rtol values at which Sven diverges on this grid')
        t = dv['per_method'][scan]
        M.add(f'numDiv{tag}NMethods', int(len(t)),
              source='headline_figs.divergence_by_method')
        worst_m = t.iloc[0] if len(t) else None
        if worst_m is not None:
            M.add(f'numDiv{tag}WorstMethod', C.Raw(str(worst_m['display'])),
                  source='headline_figs.divergence_by_method')
            M.add(f'numDiv{tag}WorstMethodFrac',
                  C.fmt_pct(_num(worst_m['frac_diverged']), 1),
                  source='headline_figs.divergence_by_method')
        allbad = t[t['n_configs_all_bad'] == t['n_configs']]
        if len(allbad):
            M.add(f'numDiv{tag}NoEligible',
                  C.Raw(_join(sorted(allbad['display']))),
                  source='headline_figs.divergence_by_method',
                  note='methods whose every grid point lost every seed')
    return dv


# ---------------------------------------------------------------------------
# The figure registry (campaign/FIGURE_API_CONTRACT.md)
# ---------------------------------------------------------------------------
#: the provenance record each appendix's assets are written with, by section.  Memoised
#: through :meth:`_Data.prov`, so the figure and the tables beside it quote one record.
_PROVENANCE = {
    'budget': _prov_budget, 'overparam': _prov_overparam,
    'batchsize': _prov_batchsize, 'kappa': _prov_kappa, 'knobs': _prov_knobs,
    'divergence': _prov_divergence,
}

#: the legend strip every figure of this module carries under its panels.  ``height`` is
#: the inches :func:`common.figsize` was given as ``extra_h``; the rest goes to
#: ``fig.legend``, and ``{}`` drops the legend altogether.
_LEGEND = {'loc': 'lower center', 'frameon': False}

FIGURE_SPECS = figspec.check_defaults({
    # ---------------- App. F ----------------
    'budget': figspec.FigureSpec(
        draw=_fig_budget,
        doc='F4: best-of-n tuning curves on two grids, with the equal-budget mark',
        defaults={
            'panels': list(BUDGET_PANELS),   # the scans drawn, one panel each
            'methods': [],                   # [] = every method on the grid, Sven first
            'nrow': 1,                       # the column count follows `panels`
            'fraction': 0.49,                # panel width / \linewidth
            'style_fraction': 0.49,          # the font-size set (common.set_paper_style)
            'aspect': 0.82,                  # panel height / width
            'extra_h': 0.62,                 # inches added for the legend strip
            'sven_lw': 1.3,                  # _thin: Sven's line
            'other_lw': 0.7,                 # _thin: every other method
            'thin_ms': 2.2,                  # _thin: marker size
            'equal_budget_line': True,       # the dotted n = BUDGET_EQUAL_N rule
            'annotate_equal_budget': True,   # ... and the "$n=8$" label beside it
            'annotation_fontsize': 5.4,
            'equal_budget_color': 'crimson',
            'legend_lw': 0.9,                # the width of the legend's own handles
            'legend_below': dict(_LEGEND, ncol=5, height=0.62, fontsize=5.4),
        }),
    # ---------------- App. G ----------------
    'overparam': figspec.FigureSpec(
        draw=_fig_overparam,
        doc='F5: the P/N sweep -- selected val. loss per task, then rank, reach, divergence',
        defaults={
            'tasks': [k for k, _ in OVERPARAM_SCANS],   # which tasks are drawn
            'nrow': 2, 'ncol': 3,
            'fraction': 1 / 3, 'style_fraction': 0.32,
            'aspect': 0.86, 'extra_h': 0.50,
            'lw': 1.0, 'ms': 2.4,            # the per-task summary lines (row 2)
            'overlay_lw': 0.7,               # their dashed second series (test, worst)
            'overlay_ms': 2.0,
            'sven_lw': 1.25, 'other_lw': 0.75, 'thin_ms': 2.2,   # _thin, row 1
            'arm_tick_fmt': '{:.2g}',        # the arm ticks of row 1
            'min_log_sep': 0.12,             # ... and how close two labels may sit
            'mark_pn_one': True,             # the dotted P/N = 1 guide
            'mark_partial_reach': True,      # hollow marker where a Sven seed never got there
            'panel_legend': {'fontsize': 5.0, 'loc': 'upper left', 'handlelength': 1.1},
            'legend_below': dict(_LEGEND, ncol=5, height=0.50, fontsize=5.2),
        }),
    'overparam_losses': figspec.FigureSpec(
        draw=_fig_overparam_outcomes,
        doc=('F5, as carried: the P/N sweep on the training subset, validation and '
             'test loss, one row each and one column per task. Replaces the two-row '
             '`overparam` (whose summary row -- rank, reach, divergence -- is retired '
             'from the paper) and the former test/train `overparam_outcomes`.'),
        defaults={
            'tasks': [k for k, _ in OVERPARAM_SCANS],
            'metrics': ['final_train_eval', 'final_val_loss', 'final_test_loss'],
            'exclude': {'mnist_labelreg': ['SOAP']},   # per task, or a flat list
            'nrow': 3, 'ncol': 3,
            'fraction': 1 / 3, 'style_fraction': 0.32,
            'aspect': 0.80, 'extra_h': 0.50,
            'sven_lw': 1.25, 'other_lw': 0.75, 'thin_ms': 2.2,
            'arm_tick_fmt': '{:.2g}', 'min_log_sep': 0.12,
            'mark_pn_one': True,
            'legend_below': dict(_LEGEND, ncol=5, height=0.50, fontsize=5.2),
        }),
    # ---------------- App. H ----------------
    'batchsize_loss': figspec.FigureSpec(
        draw=_fig_batchsize_loss,
        doc=('F6, as carried: selected validation loss against batch size, every method '
             'tuned inside its arm -- the first panel of `batchsize`.'),
        defaults={
            'batch_sizes': [],
            'nrow': 1, 'ncol': 1,
            'fraction': 0.55, 'style_fraction': 0.49,
            'aspect': 0.66, 'extra_h': 0.55,
            'sven_lw': 1.25, 'other_lw': 0.75, 'thin_ms': 2.2,
            'arm_tick_fmt': '{:.0f}',
            'legend_below': dict(_LEGEND, ncol=4, height=0.55, fontsize=5.2),
        }),
    'batchsize': figspec.FigureSpec(
        draw=_fig_batchsize,
        doc='F6: batch-size sweep -- selection, used rank vs the cap, cost, divergence',
        defaults={
            'batch_sizes': [],               # [] = every arm in the scan
            'nrow': 2, 'ncol': 2,
            'fraction': 0.49, 'style_fraction': 0.49,
            'aspect': 0.80, 'extra_h': 0.55,
            'cmap': 'viridis',               # the rtol colour ramp
            'rtol_lw': 0.9, 'rtol_ms': 2.2,  # one line per rtol (panels 2 and 3)
            'sven_lw': 1.25, 'other_lw': 0.75, 'thin_ms': 2.2,
            'ms': 2.4, 'overlay_ms': 2.0,    # divergence panel: Sven / the others
            'only_diverging_methods': True,  # drop a method that never diverged
            'cap_line': True,                # the dashed k = B cap
            'memory_axis': True,             # the twinned peak-memory curve
            'memory_color': '#B35806',
            'memory_labelsize': 5.6,
            'arm_tick_fmt': '{:.0f}',
            'rank_legend': {'fontsize': 5.4, 'loc': 'upper left', 'handlelength': 1.1,
                            'ncol': 1},
            'cost_legend': {'fontsize': 5.4, 'loc': 'upper left', 'handlelength': 1.1},
            'legend_below': dict(_LEGEND, ncol=5, height=0.55, fontsize=5.6),
        }),
    # ---------------- App. M ----------------
    'kappa': figspec.FigureSpec(
        draw=_fig_kappa,
        doc='F8: kappa at matched effective step -- loss, spread over kappa, used rank',
        defaults={
            'ks': [],                        # [] = every rank cut in the scan
            'kappas': [],                    # [] = every kappa in the scan
            'nrow': 1, 'ncol': 3,
            'fraction': 1 / 3, 'style_fraction': 0.32,
            'aspect': 0.92, 'extra_h': 0.42,
            'ms': 2.0, 'spread_ms': 2.2,
            'capsize': 1.2, 'elinewidth': 0.5,
            'default_kappa_lw': 1.2,         # kappa = 2 is the default and drawn thicker
            'other_lw': 0.8,
            'lw': 1.0,                       # the spread panel
            'rank_lw': 0.9,                  # the used-rank panel
            'full_ls': '-',                  # k = B (the untruncated solve)
            'trunc_ls': '--',                # k < B
            'mark_matched_steps': True,       # the dotted matched-effective-step rules
            'arm_tick_fmt': '{:g}',
            'legend_below': dict(_LEGEND, ncol=6, height=0.42, fontsize=5.4),
        }),
    # ---------------- App. N ----------------
    'knobs': figspec.FigureSpec(
        draw=_fig_knobs,
        doc='F14: micro-batching and parameter masking -- quality (top), and peak '
            'memory and step time relative to the reference setting (bottom); '
            "``bottom_left='rank'`` restores the used-rank-vs-cap panel.",
        defaults={
            'tasks': [k for k, _ in MICROBATCH_SCANS],   # which scans of each family
            'bottom_left': 'memory',         # or 'rank'
            'cost_logy': False,              # the two ratio panels: linear y
            'nrow': 2, 'ncol': 2,
            'fraction': 0.49, 'style_fraction': 0.49,
            'aspect': 0.80, 'extra_h': 0.50,
            'lw': 1.0, 'ms': 2.4,
            'time_lw': 0.9, 'time_ms': 2.0,  # the step-time panel
            'cap_line': True, 'cap_lw': 0.7,  # the dotted B/uB cap
            'reference_line': True,          # the "= the reference setting" rule at 1
            'family_proxies': True,          # the two line-style entries in the legend
            'arm_tick_fmt': '{:.0f}',
            'legend_below': dict(_LEGEND, ncol=3, height=0.50, fontsize=5.6),
        }),
    # ---------------- App. P ----------------
    'divergence': figspec.FigureSpec(
        draw=_fig_divergence,
        doc='F10: where Sven diverges -- by rtol on the headline grids and the sweeps, '
            'and by method on the headline grids',
        defaults={
            'rtol_scans': list(BUDGET_SCANS) + list(DIVERGENCE_SWEEP_SCANS),  # left panel
            'scans': list(BUDGET_SCANS),      # right panel (per method)
            'sweep_ls': '--',                 # line style of the sweeps in the left panel
            'nrow': 1, 'ncol': 2,
            'fraction': 0.49, 'style_fraction': 0.49,
            'aspect': 0.86, 'extra_h': 0.62,
            'lw': 1.0, 'ms': 2.4,
            'dot_ms': 2.6,                    # the per-method dots
            'scan_offset': 0.16,              # how far apart a method's four grids sit
            'method_labelsize': 5.2,
            'panel_legend': {},               # the shared strip names every scan
            'legend_below': dict(_LEGEND, ncol=4, height=0.62, fontsize=5.2),
        }),
    'divergence_grids': figspec.FigureSpec(
        draw=_fig_divergence_grids,
        doc='F10b: Sven\'s (rtol, eta) divergence maps on the headline grids, one panel each',
        defaults={
            # panels, in this order (rows follow `ncol`): the headline grids, then the
            # sweeps (P/N, batch size), where the diverged fraction gets high
            'scans': list(BUDGET_SCANS) + list(DIVERGENCE_SWEEP_SCANS),
            'ncol': 3,
            'fraction': 1 / 3, 'style_fraction': 0.32,
            'aspect': 0.95, 'extra_h': 0.0,   # no legend strip: the maps are annotated
            'counts_only': True,              # keep the counts, drop the helper's percent
            'cell_fontsize': 4.4,
            'panel_ticksize': 4.6,
            'title_fontsize': 5.4,
            'label_fontsize': 5.6,
        }),
})

#: which figures each section owns, in the order ``build()`` writes them
SECTION_FIGURES = {
    'budget': ('budget',),
    'overparam': ('overparam', 'overparam_losses'),
    'batchsize': ('batchsize', 'batchsize_loss'),
    'kappa': ('kappa',),
    'knobs': ('knobs',),
    'divergence': ('divergence', 'divergence_grids'),
}


def _draw(name, ctx, **call_opts):
    """Draw one figure and write NOTHING.  ``(fig, meta)``.

    Deliberately not :func:`figspec.draw_figure`: that resolves the name through the
    registry, which imports every other figure module, and a rebuild of this group has
    no reason to depend on those.  The two do the same thing -- resolve the options,
    draw inside the figure's ``rc``, then apply the generic cosmetics.
    """
    import matplotlib.pyplot as plt

    spec = FIGURE_SPECS[name]
    opts = figspec.figure_opts(name, spec.defaults, **call_opts)
    with plt.rc_context(opts.get('rc') or {}):
        fig, meta = spec.draw(ctx, opts)
        meta = dict(meta or {})
        figspec.apply_opts(fig, meta.get('axes'), opts)
    meta['options'] = opts
    return fig, meta


def _save(name, ctx):
    """Draw one figure and write it: the manuscript PDF, the PNG twin, the sidecar."""
    fig, meta = _draw(name, ctx)
    record = meta.get('provenance')
    if record is None:                      # pragma: no cover - a bug in the draw fn
        raise RuntimeError(f'{name}: no provenance record; a paper figure is never '
                           f'written without one (campaign/FIGURE_API_CONTRACT.md)')
    # the draw ran inside an rc_context, so the paper rcParams it set are gone again by
    # now -- and savefig reads pdf.fonttype (42, never Type 3) and savefig.bbox off the
    # LIVE ones, so they go back before the write
    figspec.paper_style(meta.get('style_fraction', 0.32))
    pdf, _png = C.save_fig(fig, name, GROUP, provenance_record=record)
    return pdf


def _figures(section, ctx, report):
    """Write ``section``'s figures, in order, and record them in the report."""
    for name in SECTION_FIGURES[section]:
        report['figures'].append(str(_save(name, ctx)))
    return report


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------
SECTIONS = ('budget', 'overparam', 'batchsize', 'kappa', 'knobs', 'divergence')


def build(root=None, dry_run=False, figures=True, tables=True, numbers=True,
          only=None, verbose=True):
    """Build every reviewer-study asset.  Returns the report ``__main__`` prints."""
    want = {'figures': bool(figures), 'tables': bool(tables), 'numbers': bool(numbers)}
    sections = [s for s in SECTIONS if only is None or s in set(only)]
    report = {'module': MODULE, 'figures': [], 'tables': [], 'n_macros': 0,
              'sections': sections, 'status': ''}
    if dry_run:
        report['status'] = 'dry run'
        report['figures'] = [str(C.fig_path(n, GROUP)) for s in SECTIONS
                             for n in SECTION_FIGURES[s]]
        report['tables'] = [str(C.TABLE_DIR / f'{n}.tex') for n in
                            ('budget', 'equal_budget', 'overparam',
                             'overparam_loss', 'batchsize', 'batchsize_loss', 'kappa',
                             'knobs', 'divergence')]
        return report
    if verbose:
        C.banner(MODULE, f'sections={sections} build={want}')
    prov = C.provisional_scans()
    if prov and not C.allow_provisional():
        report['status'] = f'refusing to write: provisional {prov}'
        return report

    M = C.Macros(module=MODULE)
    data = context(root, reload=True)       # a build starts from nothing cached
    # one section at a time, dropping its frames (and the section cache that holds them)
    # before the next: the peak footprint is one appendix's worth of scans
    if 'budget' in sections:
        build_budget(data, M, want, report)
        data.drop()
    if 'overparam' in sections:
        build_overparam(data, M, want, report)
        data.drop()
    if 'batchsize' in sections:
        build_batchsize(data, M, want, report)
        data.drop()
    if 'kappa' in sections:
        build_kappa(data, M, want, report)
        data.drop()
    if 'knobs' in sections:
        build_knobs(data, M, want, report)
        data.drop()
    if 'divergence' in sections:
        # the campaign-wide count reads all 21 grids; give it the headroom
        hl.clear_cache()
        build_divergence(data, M, want, report)
    if want['numbers'] and len(M):
        # the divergence section's macros are campaign-wide (``campaign_divergence`` reads
        # every on-grid scan), so the macro file's provenance has to list those too --
        # otherwise \numDivSvenNRuns looks as though it came from the 17 scans above
        div_scans = list(hf.on_grid_scans(root)) if 'divergence' in sections else []
        path = M.write(provenance=C.provenance(
            functions=('see the per-macro comments',),
            scans=sorted(set(data.scans) | set(BUDGET_SCANS) | set(div_scans)),
            note='G2: every number App. F, G, H, M, N and P quote',
            provisional=C.provisional_scans()))
        report['macros'] = str(path)
        report['n_macros'] = len(M)
    if verbose:
        print(f"[{MODULE}] figures={len(report['figures'])} "
              f"tables={len(report['tables'])} macros={report['n_macros']}")
        for p in report['figures'] + report['tables']:
            print('   ', p)
    return report


if __name__ == '__main__':
    build()
