"""``paper_assets.main`` -- the headline assets of the ICLR revision.

Owns (``campaign/PAPER_PLAN.md`` section 5):

* **F1** ``figures_iclr/main/headline_curves.pdf`` -- validation loss vs epoch (top) and
  vs standalone synchronised training time (bottom) for the three headline tasks
  (random polynomial, MNIST label regression, nanoGPT), confirmation seeds, +/- 1 std.
  The main-text panels draw the front of the field plus Sven; the all-methods version of
  the same figure is ``headline_curves_all.pdf`` for the appendix.
* **F3** ``figures_iclr/main/cost_memory.pdf`` -- step time and peak memory against
  parameter count over five architectures, measured in the standalone timing pass.
* **F12** ``figures_iclr/main/k_sweeps.pdf`` + ``hparam_landscape.pdf`` -- Sven's
  ``k`` sweep at each scan's SELECTED ``rtol`` (the corrected version of the old Fig. 2
  top row) and the final-loss landscape over ``k`` and over ``rtol``.
* **F13** ``figures_iclr/main/allseed_curves.pdf`` + ``allseed_curves_ce_lm.pdf`` --
  the all-methods / all-seeds appendix versions for every headline scan, 1D regression
  and MNIST-CE included, each with a ``_nosoap`` twin that drops SOAP.
* **T1--T5, T10, T18, T20** in ``tables_v2/``, and the **G1** macro group in
  ``numbers_v2_main.tex``.

Nothing here computes a number: every value comes from :mod:`headline`,
:mod:`headline_figs` or :mod:`scan_analysis` and is only formatted.  Run it with::

    cd analysis && ../.venv/bin/python -m paper_assets.main

Every figure above is declared in :data:`FIGURE_SPECS` and split into a *draw* half
(``_headline_curves(ctx, opts) -> (fig, meta)``, which writes nothing) and a *save* half
(:func:`build`, through :func:`common.save_fig`), so the same builder can be driven from
``analysis/notebooks/paper/`` -- see ``campaign/FIGURE_API_CONTRACT.md`` and
:mod:`paper_assets.notebook`.  Each figure's own knobs (which methods, which scans, the
geometry, the legends, the line weights) are in its ``defaults``; the generic cosmetics
(``xlabel``, ``ylim``, ``legend``, ...) come free from :func:`figspec.apply_opts`.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import headline as hl
import headline_figs as hf
import paired
import scan_analysis as sa
import style

from . import common as C
from . import figspec

GROUP = 'main'

#: the three tasks the revised main text leads with (PAPER_PLAN section 1)
HEADLINE_MAIN = ('polynomial_scan', 'mnist_scan_labelRegression', 'exp_nanogpt_speedrun')

#: the four MLP/regression scans whose Sven hyperparameter sweeps are re-derived
SWEEP_SCANS = ('toy_1d_scan', 'polynomial_scan', 'mnist_scan_labelRegression',
               'mnist_scan_ce')

#: the seven scans the ranking figures span -- the same field as T4 (`_t4_ranking`)
RANK_SCANS = ('toy_1d_scan', 'polynomial_scan', 'mnist_scan_labelRegression',
              'mnist_scan_ce', 'cifar10_resnet_scan_labelRegression',
              'cifar10_resnet_ce_scan', 'exp_nanogpt_speedrun')

#: App. E's all-seed figure: three scans x two axes
ALLSEED_SCANS = ('toy_1d_scan', 'polynomial_scan', 'mnist_scan_labelRegression')
#: and the two the second all-seed figure covers, so every headline scan has one
ALLSEED_SCANS_B = ('mnist_scan_ce', 'exp_nanogpt_speedrun')

#: the methods the main-text results table lists beside Sven.  The leaders (HIG on the
#: synthetics, MuonW on the classification scans) and the first-order field the paper
#: claims to beat; the full field is in the per-scan appendix tables (T5).
MAIN_TABLE_METHODS = ('Sven', 'HIG', 'MuonW', 'Muon', 'SOAP', 'AdamW', 'Adam', 'SGD')

#: how many methods the front-of-the-field panels of F1 draw, Sven always included
MAIN_PANEL_TOP_N = 5

#: scan -> the letters a macro name uses for it (no digits: TeX forbids them)
SCAN_KEY = {
    'toy_1d_scan': 'Toy',
    'polynomial_scan': 'Poly',
    'mnist_scan_labelRegression': 'MnistLR',
    'mnist_scan_ce': 'MnistCE',
    'cifar10_resnet_scan_labelRegression': 'CifarLR',
    'cifar10_resnet_ce_scan': 'CifarCE',
    'exp_nanogpt_speedrun': 'Nanogpt',
}

#: The scans this module emits MACROS for.  One macro, one owner: a name defined in two
#: of the four ``numbers_v2_*.tex`` files makes LaTeX abort, so the ownership follows the
#: EXPLICIT half of ``PAPER_PLAN`` section 5.3 wherever the two group descriptions
#: overlap.  G3 (``paper_assets.large``) names "both CIFAR scans' levels, accuracies,
#: train-eval, costs and ranks" and "nanoGPT and GPT-2 levels, perplexities, costs", so
#: all three of those scans' numbers come from ``numbers_v2_large.tex``; G1's generic
#: "the 7 scans" yields to it.  The three scans still appear in this module's cross-scan
#: FIGURES and TABLES (F1, T1--T5, T18), which ``large`` does not produce --
#: :func:`common.write_numbers_index` re-checks the split on every build and prints a
#: clash if either side changes its mind.
MACRO_SCANS = ('toy_1d_scan', 'polynomial_scan', 'mnist_scan_labelRegression',
               'mnist_scan_ce')

#: the scans whose macros another module owns, and which one (for the report / a reader)
MACRO_DEFERRED = {'exp_nanogpt_speedrun': 'large', 'cifar10_resnet_ce_scan': 'large',
                  'cifar10_resnet_scan_labelRegression': 'large'}

#: individual macro families this module leaves to another module, and why.  Checked on
#: every build by :func:`common.write_numbers_index`, which prints a clash if either side
#: starts emitting the other's names.
MACRO_FAMILIES_DEFERRED = {
    'num<Scan>SvenK': 'spectra (T8: "per scan: selected k, rtol, which cut binds")',
    'num<Scan>SvenRtol': 'spectra (T8)',
    'numNanogpt*': 'large (G3: nanoGPT levels, perplexities, costs)',
    'numCifar*': 'large (G3: both CIFAR scans levels, accuracies, costs, ranks)',
}

#: method -> the letters a macro name uses for it
METHOD_KEY = {
    'Sven': 'Sven', 'Adam': 'Adam', 'AdamW': 'AdamW', 'SGD': 'Sgd',
    'SGDm': 'Sgdm', 'RMSprop': 'Rmsprop', 'Muon': 'Muon', 'MuonW': 'MuonW',
    'SOAP': 'Soap', 'Shampoo': 'Shampoo', 'KFAC': 'Kfac', 'LBFGS': 'Lbfgs',
    'PolyakSGD': 'Polyak', 'JD': 'Jd', 'HIG': 'Hig',
}

#: the methods a per-scan macro block is emitted for (the ones the prose quotes): Sven,
#: the first-order reference the paper claims to beat, and the two leaders.  Every other
#: method's number is in the per-scan tables (T5) -- a macro exists because a SENTENCE
#: quotes it, not because the number exists.
MACRO_METHODS = ('Sven', 'Adam', 'AdamW', 'MuonW', 'HIG')

#: the methods whose per-step cost and time-to-target the prose quotes
MACRO_COST_METHODS = ('Sven', 'Adam', 'AdamW')

#: the methods whose paired difference against Sven the prose quotes
MACRO_PAIRED_METHODS = ('Adam', 'AdamW')

#: the five methods the Introduction and the Conclusion name as "first-order": no
#: preconditioner beyond a diagonal second-moment estimate, and in particular NOT the
#: orthogonalizing family (Muon / MuonW), which uses gradient information only but is
#: read as its own class.  ``num<Scan>NFirstOrderResolved`` of ``num<Scan>NFirstOrder``
#: is what lets those two sentences state the resolution instead of asserting it.
FIRST_ORDER_METHODS = ('Adam', 'AdamW', 'SGD', 'SGDm', 'RMSprop')

#: the scans the abstract's "Sven's step is N-Mx Adam's" range is taken over: every MLP
#: and LM scan (the ResNet's 27x is quoted separately, as its own number, in App. J)
STEP_RATIO_SCANS = ('toy_1d_scan', 'polynomial_scan', 'mnist_scan_labelRegression',
                    'mnist_scan_ce', 'exp_nanogpt_speedrun')

#: the target whose epochs / seconds the paper quotes (``headline.TARGET_NAMES``)
MAIN_TARGET = hl.TARGET_NAMES[1.0]

_SWEEP_CMAP = 'viridis'

#: panel titles / axis labels short enough for a 1.76 in panel.  ``style.DATASET_TITLES``
#: is the one spelling for a table or a prose sentence; a 0.32-width panel title of
#: "nanoGPT (tiny-shakespeare)" is clipped, so the corpus is named in the caption instead.
SHORT_TITLE = {'exp_nanogpt_speedrun': 'nanoGPT',
               'mnist_scan_labelRegression': 'MNIST (label reg.)',
               'polynomial_scan': 'Random polynomial'}
SHORT_XLABEL = {'time': 'Standalone train time (s)'}

#: legends that sit over data get a white frame rather than a different (worse) position:
#: with four to eight ordered lines per panel there is no corner that is empty in every
#: panel, and a semi-transparent frame keeps both the key and the curve readable.
_LEGEND_FRAME = dict(frameon=True, framealpha=0.88, edgecolor='none', borderpad=0.25)


def _short_title(scan):
    return SHORT_TITLE.get(scan, hl.scan_title(scan))


def _short_xlabel(ax, versus):
    label = SHORT_XLABEL.get(versus)
    if label:
        ax.set_xlabel(label)
    return ax.get_xlabel()


def _clip_outliers(ax, note_xy=(0.97, 0.95), factor=20.0, headroom=50.0,
                   note_fontsize=4.4):
    """Truncate a panel's y axis at ``factor`` x the median pre-training loss, and SAY SO.

    On the all-methods figures one diverging method (SOAP on MNIST label regression
    spikes to 6e9 in the training loss) compresses thirteen other curves into the bottom
    tenth of the panel.  Clipping is only honest if it is announced, so the panel is
    annotated whenever the limit actually bites; the run is still counted as diverged in
    every table, and its final value is in the confirmation table.
    """
    firsts = []
    for line in ax.get_lines():
        y = np.asarray(line.get_ydata(), dtype=float)
        y = y[np.isfinite(y)]
        if y.size:
            firsts.append(float(y[0]))
    if not firsts:
        return False
    # 20x the median pre-training loss, and only when the panel currently spans more
    # than another 1.7 decades above that (``headroom``) -- i.e. only for a genuine
    # blow-up (SOAP's 6e9 training spike), never to trim a first-epoch transient.
    top = float(factor) * float(np.median(firsts))
    lo, hi = ax.get_ylim()
    if not (np.isfinite(top) and top > 0 and hi > float(headroom) * top):
        return False
    ax.set_ylim(lo, top)
    ax.text(*note_xy, 'axis truncated', transform=ax.transAxes, ha='right', va='top',
            fontsize=note_fontsize, color='0.35')
    return True


def _in_columns(j, ncol, where):
    """Does panel column ``j`` of ``ncol`` carry a label?

    ``where`` is the value of an ``xlabel_columns`` / ``ylabel_columns`` knob:
    ``'all'``, ``'none'``, ``'first'``, ``'middle'``, ``'last'``, or an explicit list of
    column indices.
    """
    if isinstance(where, (list, tuple)):
        return j in {int(v) for v in where}
    return {'all': True, 'none': False, 'first': j == 0,
            'middle': j == ncol // 2, 'last': j == ncol - 1}[str(where)]


def _exclude_for(exclude, scan):
    """The methods ``exclude`` drops from ``scan``.

    A flat list applies to every scan (``exclude=['SOAP']``); a dict applies per scan
    (``exclude={'mnist_scan_labelRegression': ['SOAP']}``), where the key ``'*'`` means
    every scan and a scan the dict does not name loses nothing.
    """
    if isinstance(exclude, dict):
        return list(exclude.get('*', ())) + list(exclude.get(scan, ()))
    return list(exclude)


def _resolve_methods(ctx, scan, methods, top_n=MAIN_PANEL_TOP_N, exclude=()):
    """Which methods a panel draws: ``'top'``, ``'all'``, or an explicit list.

    ``'top'`` is the front of the field plus Sven (the main-text panels of F1),
    ``'all'`` is every method with a selected configuration, best first.  Whatever comes
    out, a method with no confirmation runs on this scan is dropped -- asking for one is
    not an error, there is simply nothing to draw.  ``exclude`` then drops named methods
    (any spelling :func:`style.canonical_method` knows, flat or per scan -- see
    :func:`_exclude_for`), which is how the SOAP-free variants of F13 are drawn without
    listing the other fifteen.
    """
    drop = {style.canonical_method(m) for m in _exclude_for(exclude, scan)}
    if isinstance(methods, str):
        if methods == 'top':
            wanted = ctx.main_panel_methods(scan, top_n=top_n)
        elif methods == 'all':
            wanted = ctx.order(scan)
        else:
            raise ValueError(f"methods must be 'top', 'all' or a list, not {methods!r}")
    else:
        wanted = list(methods)
    return [m for m in wanted
            if m in ctx.runs(scan) and style.canonical_method(m) not in drop]


# ---------------------------------------------------------------------------
# Shared context: every table, figure and macro reads the same frames once
# ---------------------------------------------------------------------------
class Ctx:
    """The analysis tables of record, computed once per build.

    Loading a pass from Lustre costs ~20 s cold, and T1, F1, the macros and the
    time-to-target table all want the same three frames, so they are memoised here
    rather than recomputed per asset (which is also the only way the numbers in the
    table and the curve in the figure are guaranteed to be the same numbers).
    """

    def __init__(self, root=None, scans=hl.HEADLINE_SCANS):
        self.root = root
        self.scans = tuple(scans)
        self.payload = hl.load_selection()
        self._conf, self._eff, self._ttt, self._runs = {}, {}, {}, {}
        self._times, self._paired, self._scan_obj, self._tune = {}, {}, {}, {}
        self.provisional = set(C.provisional_scans())
        self.ranking = hl.ranking_summary(scans=self.scans, payload=self.payload,
                                          results_root=root)

    # -- per-scan tables -------------------------------------------------
    def conf(self, scan):
        if scan not in self._conf:
            self._conf[scan] = hl.confirmation_table(scan, payload=self.payload,
                                                     results_root=self.root)
        return self._conf[scan]

    def eff(self, scan):
        if scan not in self._eff:
            self._eff[scan] = hl.efficiency_table(scan, payload=self.payload,
                                                  results_root=self.root)
        return self._eff[scan]

    def ttt(self, scan):
        if scan not in self._ttt:
            self._ttt[scan] = hl.time_to_target_table(scan, payload=self.payload,
                                                      results_root=self.root)
        return self._ttt[scan]

    def paired(self, scan, metric=hl.SELECTION_METRIC):
        """The paired table against Sven: on validation loss (the selection metric)
        or, through :func:`headline.paired_outcome_vs_sven`, on a test OUTCOME of the
        configurations validation already chose."""
        key = scan if metric == hl.SELECTION_METRIC else (scan, metric)
        if key not in self._paired:
            if metric == hl.SELECTION_METRIC:
                self._paired[key] = hl.paired_vs_sven(scan, payload=self.payload,
                                                      results_root=self.root)
            else:
                self._paired[key] = hl.paired_outcome_vs_sven(
                    scan, metric=metric, payload=self.payload, results_root=self.root)
        return self._paired[key]

    def runs(self, scan):
        if scan not in self._runs:
            self._runs[scan] = hf.confirm_runs(scan, payload=self.payload,
                                               results_root=self.root)
        return self._runs[scan]

    def times(self, scan):
        if scan not in self._times:
            self._times[scan] = hf.standalone_epoch_times(scan, payload=self.payload,
                                                          results_root=self.root)
        return self._times[scan]

    def tune(self, scan):
        if scan not in self._tune:
            self._tune[scan] = hl.load(scan, '', results_root=self.root)
        return self._tune[scan]

    def scan_obj(self, scan):
        """A :class:`scan_analysis.Scan` over the TUNING grid -- what the Sven sweeps need."""
        if scan not in self._scan_obj:
            self._scan_obj[scan] = sa.Scan(
                name=scan, title=hl.scan_title(scan),
                plot_dir=C.LAB / 'scan_plots' / scan,
                df=self.tune(scan).copy(), results_root=self.root)
        return self._scan_obj[scan]

    # -- derived lookups -------------------------------------------------
    def order(self, scan):
        """Methods best-first on confirmation validation loss."""
        return hf.method_ranking(scan, table=self.conf(scan))

    def main_panel_methods(self, scan, top_n=MAIN_PANEL_TOP_N):
        """The front of the field plus Sven -- what a main-text panel draws."""
        order = [m for m in self.order(scan) if m in self.runs(scan)]
        top = order[:int(top_n)]
        if hl.SVEN_LABEL in order and hl.SVEN_LABEL not in top:
            top.append(hl.SVEN_LABEL)
        return top

    def row(self, scan, method, table='conf'):
        tbl = {'conf': self.conf, 'eff': self.eff}[table](scan)
        sub = tbl[tbl['method'] == method]
        return sub.iloc[0] if len(sub) else None

    def rank(self, scan, method, column='rank_val'):
        sub = self.ranking[(self.ranking['scan'] == scan)
                           & (self.ranking['method'] == method)]
        if not len(sub):
            return np.nan, np.nan
        return float(sub.iloc[0][column]), float(sub.iloc[0]['n_methods'])

    def target_row(self, scan, method, target=MAIN_TARGET):
        t = self.ttt(scan)
        sub = t[(t['target'] == target) & (t['method'] == method)]
        return sub.iloc[0] if len(sub) else None

    def is_provisional(self, *scans):
        return bool(self.provisional.intersection(scans))


#: one :class:`Ctx` per results root, so a notebook that draws six figures reads the
#: three frames once (``Ctx`` itself memoises per scan, but only within one instance)
_CONTEXT: dict = {}


def context(root=None, reload=False):
    """The :class:`Ctx` this module's builders take, cached per results root.

    Cheap to call twice, which is what lets :func:`build` and a notebook share one set of
    loaded frames.  ``reload=True`` throws the cached one away (after a re-run moved a
    pass on disk).
    """
    key = '' if root is None else str(root)
    if reload or key not in _CONTEXT:
        _CONTEXT[key] = Ctx(root=root)
    return _CONTEXT[key]


# ---------------------------------------------------------------------------
# Formatting helpers local to this module
# ---------------------------------------------------------------------------
def _short_config(sel_or_config):
    """A compact statement of a selected configuration for a narrow table column.

    ``headline.config_label`` is the full one (and is what the per-scan appendix tables
    print); the main-text table has room for the learning rate, the weight decay when it
    is not zero, and Sven's ``k`` / ``rtol``.
    """
    text = sel_or_config if isinstance(sel_or_config, str) else ''
    bits, extra = [], []
    for part in text.split(', '):
        if '=' not in part:
            continue
        name, _, value = part.partition('=')
        if name == 'lr':
            bits.insert(0, rf'$\eta$={value}')
        elif name == 'k':
            bits.append(f'$k$={value}')
        elif name == 'rtol':
            bits.append(f'rtol={value}')
        elif name == 'weight_decay':
            bits.append(f'wd={value}')
        elif name in ('max_iter', 'history_size', 'tau', 'kappa'):
            extra.append(f'{name.replace("_", " ")}={value}')
    if not bits and not extra:
        return C.Raw('--')
    return C.Raw(', '.join(bits + extra[:1]))


def _name_list(names):
    """``'A'`` / ``'A and B'`` / ``'A, B and C'`` -- an English list for a prose macro.

    The names are method display labels (``style.method_label``), so they are escaped for
    LaTeX exactly as a table cell would be.
    """
    items = [C.latex_escape(str(n)) for n in names]
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    return ', '.join(items[:-1]) + ' and ' + items[-1]


def _counts_cell(row, prefix='conf'):
    cell = C.fmt_counts(row.get(f'fin_{prefix}'), row.get(f'att_{prefix}'))
    if not row.get(f'elig_{prefix}', True):
        return C.Raw(f'{cell}$^{{\\dagger}}$')
    return cell


def _val_cell(mean, std, best=False, sig=4):
    """The confirmation-seed validation cell, bold when the block's lowest is within
    reach of it.

    The WHOLE value goes bold, band included.  ``\\textbf`` does not reach into math
    mode, so the ``\\pm`` -- the one math-mode token in the cell -- is emboldened on
    its own with ``\\boldsymbol`` (amsmath, already loaded by the manuscript);
    without that the symbol would sit at the normal weight between two bold numbers.
    """
    cell = C.fmt_pm(mean, std, sig)
    if not best:
        return cell
    head, sep, tail = str(cell).partition(' $\\pm$ ')
    if not sep:                                   # no band: just the mean
        return C.Raw(f'\\textbf{{{head}}}')
    return C.Raw(f'\\textbf{{{head} $\\boldsymbol{{\\pm}}$ {tail}}}')


def _best_val_methods(ctx, scan, methods):
    """Every method of one block whose validation loss is the lowest or ties it.

    "Ties" means the gap to the block's lowest mean is no larger than the two rows'
    seed bands added in quadrature, ``sqrt(s_best^2 + s_i^2)`` -- a DISPLAY rule over
    the two numbers the row already prints, not a test: the paper's significance claim
    is the paired, per-seed comparison of ``headline.paired_vs_sven`` (C6), and this
    convention exists so the table does not silently award a win the text calls a tie.

    A method whose confirmation pass is NOT eligible (the dagger: more than half its
    runs failed, so the mean beside it is a mean over the survivors) cannot win the
    block -- comparing a 2-seed survivor mean against a 5-seed one would be the
    selection bug this table exists to expose.
    """
    vals = {}
    for m in methods:
        c = ctx.row(scan, m)
        if c.get('elig_conf', True) and C.finite(c['val_conf']):
            std = c.get('val_conf_std')
            vals[m] = (float(c['val_conf']),
                       float(std) if C.finite(std) else 0.0)
    if not vals:
        return set()
    best = min(vals, key=lambda m: vals[m][0])
    low, low_std = vals[best]
    return {m for m, (v, sd) in vals.items()
            if v - low <= math.sqrt(low_std ** 2 + sd ** 2)}


def _epochs_cell(ctx, scan, method):
    """Epochs to the median-method target, with the reach count when it is not all runs."""
    r = ctx.target_row(scan, method)
    if r is None or not C.finite(r['epochs']):
        return C.Raw('never')
    cell = C.fmt_sig(r['epochs'], 3)
    if not bool(r['all_reached']):
        cell = C.Raw(f'{cell} ({int(r["n_reached"])}/{int(r["n_runs"])})')
    return cell


def _n_epochs(ctx, scan):
    df = ctx.tune(scan)
    v = pd.to_numeric(df.get('num_epochs'), errors='coerce').dropna()
    return int(v.max()) if len(v) else np.nan


def _scalar(ctx, scan, column):
    df = ctx.tune(scan)
    v = pd.to_numeric(df.get(column), errors='coerce').dropna()
    return float(v.iloc[0]) if len(v) else np.nan


# ---------------------------------------------------------------------------
# F1 -- the headline convergence figure
# ---------------------------------------------------------------------------
def _panel(ctx, ax, scan, versus, methods, title=None, lw=1.1, sven_lw=2.0,
           show_ylabel=True, show_xlabel=True):
    drawn = hf.plot_curves(scan, ax, which='val', versus=versus, methods=methods,
                           runs=ctx.runs(scan),
                           times=ctx.times(scan) if versus == 'time' else None,
                           lw=lw, sven_lw=sven_lw, quiet=True,
                           payload=ctx.payload, results_root=ctx.root)
    if title:
        ax.set_title(title, pad=2.0)
    if not show_ylabel:
        ax.set_ylabel('')
    if not show_xlabel:
        ax.set_xlabel('')
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.grid(which='major', ls=':', alpha=0.35)
    ax.grid(which='minor', ls=':', alpha=0.18)
    return drawn


def _headline_curves(ctx, opts):
    """F1: one row per x axis (vs epoch, vs standalone train time) x the headline tasks.

    Draws only; :func:`build` saves.  ``opts['methods']`` is what makes this the same
    builder for the main-text figure (the front of the field plus Sven) and for
    ``headline_curves_all`` (every method that has a selected configuration).
    """
    import matplotlib.pyplot as plt

    figspec.paper_style(opts['font_fraction'])
    scans = list(opts['scans'])
    # epoch on top, standalone synchronised training time below (F1's two axes in
    # PAPER_CONTRACTS.md); see _ALLSEED_COLS for why the top axis is epochs
    rows = list(opts['versus'])
    nrow, ncol = len(rows), len(scans)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=C.figsize(ncol, nrow, opts['fraction'],
                                               aspect=opts['aspect'],
                                               extra_h=opts['extra_h']),
                             squeeze=False)
    used, per_scan = [], {}
    for j, scan in enumerate(scans):
        methods = _resolve_methods(ctx, scan, opts['methods'], opts['top_n'],
                                   exclude=opts['exclude'])
        per_scan[scan] = methods
        for i, versus in enumerate(rows):
            # the x label goes on the MIDDLE column only: "Synchronised training time
            # (s), standalone" is wider than a 1.76 in panel and three of them collide
            _panel(ctx, axes[i][j], scan, versus, methods,
                   title=(_short_title(scan) if (i == 0 and opts['panel_titles'])
                          else None),
                   lw=opts['lw'], sven_lw=opts['sven_lw'],
                   show_ylabel=_in_columns(j, ncol, opts['ylabel_columns']),
                   show_xlabel=_in_columns(j, ncol, opts['xlabel_columns']))
        used.extend(methods)
        if opts['panel_legend']:
            # A PER-PANEL legend.  With a single shared legend the reader cannot tell
            # which of the nine named methods are the five drawn in a given panel -- and
            # the fields differ between panels -- so each panel names its own curves.
            h, lab = C.method_handles(methods)
            leg = axes[0][j].legend(h, lab, **opts['panel_legend_kw'])
            leg.get_frame().set_linewidth(opts['panel_legend_frame_lw'])
    order = [m for m in hl.method_order(dict.fromkeys(used)) if m in set(used)]
    if opts['figure_legend_methods']:
        handles, labels = C.method_handles(order)
    else:
        handles, labels = [], []
    handles.append(plt.Rectangle((0, 0), 1, 1, fc='0.45', alpha=0.25, lw=0))
    labels.append(style.seed_spread_label())
    leg_ncol = opts['figure_legend_ncol']
    if leg_ncol is None:                       # as many rows as columns, 3 to 6 wide
        leg_ncol = min(6, max(3, (len(labels) + 1) // 2))
    fig.legend(handles, labels,
               **figspec.merge_kw(opts['figure_legend_kw'], ncol=leg_ncol))
    rec = C.provenance(
        functions=['headline_figs.plot_curves', 'headline_figs.confirm_runs',
                   'headline_figs.standalone_epoch_times',
                   'headline.confirmation_table', 'scan_analysis.seed_band'],
        scans=[hl.dir_name(s, k) for s in scans for k in ('confirm', 'timing')],
        note=('validation loss vs epoch and vs the standalone timing pass\'s synchronised '
              'training time; confirmation seeds, mean +/- 1 std'),
        methods_drawn={s: v for s, v in per_scan.items()},
        provisional=sorted(ctx.provisional.intersection(scans)))
    return fig, {'axes': axes, 'provenance': rec, 'per_scan': per_scan}


# ---------------------------------------------------------------------------
# F3 -- cost and memory across five architectures
# ---------------------------------------------------------------------------
#: the five architectures F3 spans, smallest first, and where each one's cost is measured
F3_SCANS = ('toy_1d_scan', 'polynomial_scan', 'mnist_scan_labelRegression',
            'exp_nanogpt_speedrun', 'cifar10_resnet_scan_labelRegression')
F3_METHODS = ('Sven', 'Adam', 'MuonW', 'HIG', 'LBFGS', 'SGD')
#: the architecture family behind each F3 point, for the capture-mode note
ARCH_NAME = {'toy_1d_scan': 'MLP', 'polynomial_scan': 'MLP',
             'mnist_scan_labelRegression': 'MLP', 'exp_nanogpt_speedrun': 'nanoGPT',
             'cifar10_resnet_scan_labelRegression': 'ResNet18'}
#: the short task names used in F3's top-axis MLP tick labels
F3_SHORT = {'toy_1d_scan': '1D', 'polynomial_scan': 'poly',
            'mnist_scan_labelRegression': 'MNIST'}


def _cluster_params(scans, ctx, decades=0.5):
    """Group scans whose parameter counts are within ``decades`` of each other on a log
    axis, so the top axis of F3 gets one tick per visually distinct position."""
    got = sorted(((float(_scalar(ctx, s, 'n_params')), s) for s in scans
                  if C.finite(_scalar(ctx, s, 'n_params'))))
    out = []
    for P, s in got:
        if out and abs(math.log10(P) - math.log10(out[-1][-1][0])) <= decades:
            out[-1].append((P, s))
        else:
            out.append([(P, s)])
    return [[s for _P, s in group] for group in out]


def _cost_memory(ctx, opts):
    """F3: step time (left) and peak memory (right) vs the parameter count.

    Both come from ``headline.efficiency_table``, i.e. from the standalone ``_timing``
    pass -- one selected configuration at a time, alone on the GPU.  That is the only
    wall clock in the campaign that is a statement about the optimizer, and for the ResNet
    it is also the only measurement that exists (``profile_results_v3`` has no
    ``cifar_resnet18`` directory, PAPER_PLAN risk 3).
    """
    import matplotlib.pyplot as plt

    figspec.paper_style(opts['font_fraction'])
    scans = list(opts['scans'])
    methods = list(opts['methods'])
    fig, axes = plt.subplots(1, 2, figsize=C.figsize(2, 1, opts['fraction'],
                                                     aspect=opts['aspect'],
                                                     extra_h=opts['extra_h']),
                             squeeze=False)
    rows, capture = [], {}
    for scan in scans:
        eff = ctx.eff(scan)
        P = _scalar(ctx, scan, 'n_params')
        timing = hl.load(scan, 'timing', results_root=ctx.root)
        sven = timing[timing['optimizer'] == hl.SVEN_METHOD]
        modes = sorted({str(v) for v in sven.get('gram_capture', pd.Series()).dropna()})
        capture[scan] = modes
        for _, r in eff.iterrows():
            if r['method'] not in methods:
                continue
            rows.append({'scan': scan, 'P': P, 'method': r['method'],
                         'ms_per_step': r['ms_per_step'],
                         'mem': r['peak_gpu_mem_mb'],
                         'gram_capture': '/'.join(modes)})
    frame = pd.DataFrame(rows)
    # The architectures behind the x positions, named once at the top of the figure, and
    # read off the plotted data rather than typed: the caption quotes the endpoints of
    # this axis as macros, so the two must come from the same place.  The two MLP scans
    # sit 0.06 of a decade apart, so they share one tick.
    groups = {}
    for scan in scans:
        P = _scalar(ctx, scan, 'n_params')
        if not C.finite(P):
            continue
        groups.setdefault(ARCH_NAME.get(scan, scan), []).append(float(P))
    arch = []
    for family, ps in groups.items():
        label = family
        if family == 'MLP':
            tasks = [s for s in scans if ARCH_NAME.get(s) == 'MLP']
            # one tick per decade-cluster of MLPs: 1D+poly together, MNIST on its own
            for cluster in _cluster_params(tasks, ctx, opts['cluster_decades']):
                names = ', '.join(F3_SHORT.get(s, s) for s in cluster)
                arch.append((float(np.mean([_scalar(ctx, s, 'n_params')
                                            for s in cluster])),
                             f'MLP\n({names})'))
            continue
        arch.append((float(np.mean(ps)), label))
    arch.sort()
    for ax, col, label in ((axes[0][0], 'ms_per_step', 'Time per optimizer step (ms)'),
                           (axes[0][1], 'mem', 'Peak GPU memory (MB)')):
        for m in methods:
            sub = frame[frame['method'] == m].dropna(subset=['P', col]).sort_values('P')
            if not len(sub):
                continue
            is_sven = m == hl.SVEN_LABEL
            ax.plot(sub['P'], sub[col], marker='o' if is_sven else 's',
                    ms=opts['sven_ms'] if is_sven else opts['ms'],
                    color=style.method_color(m),
                    lw=opts['sven_lw'] if is_sven else opts['lw'],
                    zorder=5 if is_sven else 2)
        # C3: the parity is a property of the CAPTURE, not of the algebra -- the ResNet's
        # dense `full` capture is the one point that is not at parity.  Stated once per
        # panel rather than per marker: the two MLP scans are 0.06 of a decade apart and
        # per-point labels overprint each other and the neighbouring curves.
        modes = {}
        for scan in scans:
            for mode in capture.get(scan, ()):
                family = ARCH_NAME.get(scan, scan)
                if family not in modes.setdefault(mode, []):
                    modes[mode].append(family)
        if modes and opts['capture_note']:
            note = '; '.join(f'{m}: ' + ', '.join(v) for m, v in modes.items())
            ax.text(0.98, 0.02, f'Sven capture -- {note}', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=opts['capture_note_fontsize'],
                    color='0.2',
                    bbox=dict(facecolor='white', alpha=0.85, lw=0, pad=0.8))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Parameters $P$')
        ax.set_ylabel(label)
        ax.grid(which='both', ls=':', alpha=0.3)
        top = ax.secondary_xaxis('top')
        top.set_xticks([p for p, _ in arch])
        top.set_xticklabels([t for _, t in arch], fontsize=opts['arch_tick_fontsize'])
        top.tick_params(length=1.6, pad=1.0)
    handles, labels = C.method_handles(
        [m for m in methods if m in set(frame['method'])])
    fig.legend(handles, labels, **opts['figure_legend_kw'])
    rec = C.provenance(
        functions=['headline.efficiency_table', 'headline.run_efficiency'],
        scans=[hl.dir_name(s, 'timing') for s in scans],
        note=('step time and peak memory of each method\'s SELECTED configuration, '
              'measured in the standalone timing pass (one run per GPU); the label under '
              'a Sven point is its Gram capture mode, which is what sets the memory'),
        architectures={s: _scalar(ctx, s, 'n_params') for s in scans},
        gram_capture=capture,
        provisional=sorted(ctx.provisional.intersection(scans)))
    return fig, {'axes': axes, 'provenance': rec, 'frame': frame}


# ---------------------------------------------------------------------------
# F12 -- Sven's hyperparameter sweeps, at the SELECTED setting of the other knobs
# ---------------------------------------------------------------------------
def _sweep_curves(ctx, ax, scan, axis='k', which='val', cmap=_SWEEP_CMAP,
                  cmap_range=(0.05, 0.9), lw=1.0, selected_lw=1.9, band_alpha=0.15):
    """Validation curves across one Sven axis with the other two pinned at the selection.

    This is ``scan_analysis.plot_k_sweep``'s quantity (same ``sven_rows`` /
    ``seed_band`` / ``epoch_axis``) drawn at panel size and, crucially, at the scan's
    **selected** ``rtol`` rather than one conservative value -- the correction X8 asks
    for.  Returns the values actually drawn.
    """
    import matplotlib.pyplot as plt

    obj = ctx.scan_obj(scan)
    chosen, _sel = hf.selected_sven(scan, payload=ctx.payload)
    values = {'k': obj.ks, 'rtol': obj.rtols}[axis]
    colors = plt.get_cmap(cmap)(np.linspace(cmap_range[0], cmap_range[1],
                                            max(len(values), 1)))
    drawn = []
    for i, v in enumerate(values):
        cfg = {a: chosen.get(a) for a in sa.SVEN_CONFIG}
        cfg[axis] = v
        rows = obj.sven_rows(**cfg)
        mean, lower, upper = sa.seed_band(rows, which)
        if mean is None:
            continue
        x = sa.epoch_axis(rows, mean, which)
        n = min(len(x), len(mean))
        sel = np.isclose(float(v), float(chosen.get(axis, np.nan)))
        ax.plot(x[:n], mean[:n], color=colors[i], lw=selected_lw if sel else lw,
                zorder=5 if sel else 2)
        if len(rows) > 1:
            ax.fill_between(x[:n], lower[:n], upper[:n], color=colors[i],
                            alpha=band_alpha, lw=0)
        drawn.append(float(v))
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation loss')
    ax.grid(which='both', ls=':', alpha=0.3)
    ax.set_title(_short_title(scan), pad=2.0)
    return drawn, chosen, values, colors


def _k_sweeps(ctx, opts):
    """F12: the ``k`` sweep at each scan's selected ``(lr, rtol)``."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figspec.paper_style(opts['font_fraction'])
    scans = list(opts['scans'])
    fig, axes = plt.subplots(1, len(scans),
                             figsize=C.figsize(len(scans), 1, opts['fraction'],
                                               aspect=opts['aspect'],
                                               extra_h=opts['extra_h']),
                             squeeze=False)
    info = {}
    for j, scan in enumerate(scans):
        drawn, chosen, values, colors = _sweep_curves(
            ctx, axes[0][j], scan, opts['axis'], which=opts['which'],
            cmap=opts['cmap'], cmap_range=opts['cmap_range'], lw=opts['lw'],
            selected_lw=opts['selected_lw'], band_alpha=opts['band_alpha'])
        info[scan] = {'k_drawn': drawn, 'selected': {k: (None if v is None else float(v))
                                                     for k, v in chosen.items()}}
        if opts['selected_note']:
            axes[0][j].text(*opts['selected_note_xy'],
                            rf'$\eta={chosen["lr"]:g}$, rtol$={chosen["rtol"]:g}$',
                            transform=axes[0][j].transAxes,
                            fontsize=opts['selected_note_fontsize'], va='bottom')
        if j:
            axes[0][j].set_ylabel('')
        ks = [int(v) for v in values]
        handles = [Line2D([], [], color=colors[i], lw=opts['legend_handle_lw'])
                   for i in range(len(ks))]
        if opts['panel_legend']:
            axes[0][j].legend(handles, [f'$k$={v}' for v in ks],
                              **opts['panel_legend_kw'])
    rec = C.provenance(
        functions=['headline_figs.selected_sven', 'scan_analysis.Scan.sven_rows',
                   'scan_analysis.seed_band', 'scan_analysis.epoch_axis'],
        scans=list(scans),
        note=('validation loss vs epoch over k with lr and rtol pinned at the SELECTED '
              'configuration of each scan (the thick curve is the selected k); the old '
              'figure swept k at one conservative rtol'),
        selected=info)
    return fig, {'axes': axes, 'provenance': rec, 'info': info}


def _hparam_landscape(ctx, opts):
    """Sven's final-loss landscape: vs ``k`` (one line per ``rtol``) and vs ``rtol``
    (one line per ``k``), both at the selected learning rate, with the selected point
    ringed and ineligible configurations left out of the line but marked.

    This is the figure behind C9 / X8: the two knobs are separated, and the saturation
    point is read off the curve at the rtol that was actually selected.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figspec.paper_style(opts['font_fraction'])
    scans = list(opts['scans'])
    rows = [tuple(r) for r in opts['rows']]
    fig, axes = plt.subplots(len(rows), len(scans),
                             figsize=C.figsize(len(scans), len(rows), opts['fraction'],
                                               aspect=opts['aspect'],
                                               extra_h=opts['extra_h']),
                             squeeze=False)
    info = {}
    for j, scan in enumerate(scans):
        obj = ctx.scan_obj(scan)
        chosen, _sel = hf.selected_sven(scan, payload=ctx.payload)
        cfg = obj.configs(obj.sven[obj.sven['lr'] == chosen['lr']], sa.SVEN_CONFIG)
        cfg = cfg[cfg['eligible']]
        saturate = {}
        for i, (x_axis, line_axis) in enumerate(rows):
            ax = axes[i][j]
            lines = sorted(cfg[line_axis].dropna().unique())
            colors = plt.get_cmap(opts['cmap'])(
                np.linspace(opts['cmap_range'][0], opts['cmap_range'][1],
                            max(len(lines), 1)))
            for t, v in enumerate(lines):
                sub = cfg[np.isclose(cfg[line_axis].astype(float), float(v))]
                sub = sub.sort_values(x_axis)
                if not len(sub):
                    continue
                sel = np.isclose(float(v), float(chosen.get(line_axis, np.nan)))
                ax.plot(sub[x_axis].astype(float), sub['score'], marker='o',
                        ms=opts['ms'], color=colors[t],
                        lw=opts['selected_lw'] if sel else opts['lw'],
                        zorder=5 if sel else 2)
                if sel and x_axis == 'k':
                    # the saturation point the paper may quote: the smallest k within
                    # 5% (``saturation_factor``) of the best score on the SELECTED
                    # rtol line.  This feeds num<Scan>SvenKSaturate, so changing the
                    # knob changes a number the prose quotes.
                    best = float(sub['score'].min())
                    hit = sub[sub['score'] <= opts['saturation_factor'] * best]
                    if len(hit):
                        saturate['k_within_5pct'] = float(hit.iloc[0][x_axis])
                        saturate['best_score_on_selected_rtol'] = best
            if opts['mark_selected']:
                at = cfg[np.isclose(cfg['k'].astype(float), float(chosen['k']))
                         & np.isclose(cfg['rtol'].astype(float),
                                      float(chosen['rtol']))]
                ax.plot([float(chosen[x_axis])], [float(at['score'].iloc[0])],
                        marker='*', ms=opts['star_ms'], mfc='none', mec='k',
                        mew=opts['star_mew'], zorder=8)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel({'k': 'Rank cap $k$', 'rtol': 'Relative tolerance'}[x_axis])
            if j == 0:
                ax.set_ylabel('Final val. loss\n(seed mean)')
            ax.grid(which='both', ls=':', alpha=0.3)
            if i == 0:
                ax.set_title(_short_title(scan), pad=2.0)
            handles = [Line2D([], [], color=colors[t], lw=opts['legend_handle_lw'])
                       for t in range(len(lines))]
            labels = [(f'$k$={int(v)}' if line_axis == 'k' else f'{v:g}')
                      for v in lines]
            if opts['panel_legend']:
                ax.legend(handles, labels,
                          title=('rtol' if line_axis == 'rtol' else None),
                          **opts['panel_legend_kw'])
        info[scan] = {'selected': {k: (None if v is None else float(v))
                                   for k, v in chosen.items()},
                      'n_eligible_at_selected_lr': int(len(cfg)), **saturate}
    rec = C.provenance(
        functions=['scan_analysis.Scan.configs', 'headline_figs.selected_sven'],
        scans=list(scans),
        note=('seed-mean final validation loss over k at fixed rtol (top) and over rtol '
              'at fixed k (bottom), both at the selected learning rate; the star is the '
              'selected configuration and ineligible configurations are dropped'),
        landscape=info)
    return fig, {'axes': axes, 'provenance': rec, 'info': info}


# ---------------------------------------------------------------------------
# F13 -- the all-methods / all-seeds appendix figures
# ---------------------------------------------------------------------------
#: F13's two axes, per ``PAPER_PLAN`` section 5.1 (``curve_figure`` "(epoch, time)").
#: The x axis is the EPOCH index, not the optimizer step: the batch size differs between
#: scans, the paper's time-to-target claim (C7) is quoted in epochs, and
#: ``PAPER_CONTRACTS.md`` names "log-y validation loss vs epoch and vs wall time" as the
#: manuscript's visual language.  ``versus='step'`` is available from
#: :func:`headline_figs.plot_curves` for a per-scan notebook, but no paper figure uses it.
#: The training-loss column is a knob away -- ``columns=[['val', 'epoch'],
#: ['val', 'time'], ['train', 'epoch']]`` puts it back.
_ALLSEED_COLS = (('val', 'epoch'), ('val', 'time'))


def _allseed_curves(ctx, opts):
    """Every method's selected configuration, confirmation seeds, on two axes.

    One row per scan, one column per ``(which, versus)`` pair.  This is the figure that
    retires every single-seed curve in the current manuscript (X15): the line is the seed
    mean and the band is +/- 1 std over the confirmation seeds.
    """
    import matplotlib.pyplot as plt

    figspec.paper_style(opts['font_fraction'])
    scans = list(opts['scans'])
    columns = [tuple(c) for c in opts['columns']]
    nrow, ncol = len(scans), len(columns)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=C.figsize(ncol, nrow, opts['fraction'],
                                               aspect=opts['aspect'],
                                               extra_h=opts['extra_h']),
                             squeeze=False)
    used, clipped = [], []
    for i, scan in enumerate(scans):
        methods = _resolve_methods(ctx, scan, opts['methods'], opts['top_n'],
                                   exclude=opts['exclude'])
        used.extend(methods)
        for j, (which, versus) in enumerate(columns):
            ax = axes[i][j]
            hf.plot_curves(scan, ax, which=which, versus=versus, methods=methods,
                           runs=ctx.runs(scan),
                           times=ctx.times(scan) if versus == 'time' else None,
                           lw=opts['lw'], sven_lw=opts['sven_lw'], quiet=True,
                           payload=ctx.payload, results_root=ctx.root)
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
            ax.grid(which='both', ls=':', alpha=0.3)
            _short_xlabel(ax, versus)
            if opts['clip_outliers'] and _clip_outliers(
                    ax, note_xy=opts['clip_note_xy'], factor=opts['clip_factor'],
                    headroom=opts['clip_headroom'],
                    note_fontsize=opts['clip_note_fontsize']):
                clipped.append((scan, which, versus))
            if j == 0:
                ax.set_ylabel(f'{_short_title(scan)}\n' + ax.get_ylabel())
            elif which == columns[0][0]:
                # the same quantity as column 0, already named there: repeating the
                # label only steals width from the panels
                ax.set_ylabel('')
            if i < nrow - 1:
                ax.set_xlabel('')
    order = [m for m in hl.method_order(dict.fromkeys(used)) if m in set(used)]
    handles, labels = C.method_handles(order)
    handles.append(plt.Rectangle((0, 0), 1, 1, fc='0.45', alpha=0.25, lw=0))
    labels.append(style.seed_spread_label())
    fig.legend(handles, labels, **opts['figure_legend_kw'])
    rec = C.provenance(
        functions=['headline_figs.plot_curves', 'headline_figs.confirm_runs',
                   'headline_figs.standalone_epoch_times', 'scan_analysis.seed_band'],
        scans=[hl.dir_name(s, k) for s in scans for k in ('confirm', 'timing')],
        note=('all methods with a selected configuration, confirmation seeds, '
              'mean +/- 1 std; validation loss vs epoch and vs standalone '
              'synchronised training time'
              + (f"; {', '.join(opts['exclude'])} excluded" if opts['exclude'] else '')),
        axis_truncated=[list(c) for c in clipped],
        provisional=sorted(ctx.provisional.intersection(scans)))
    return fig, {'axes': axes, 'provenance': rec, 'order': order}


# ---------------------------------------------------------------------------
# F14/F15 -- the ranking summary (T4) as a picture
# ---------------------------------------------------------------------------
#: which family each headline scan belongs to, for the strip plot's marker shapes.
#: Three shapes is what a 1.76 in panel can tell apart; seven would need colour, and
#: colour is already spent on the method.
RANK_FAMILY = {
    'toy_1d_scan': 'MLP', 'polynomial_scan': 'MLP',
    'mnist_scan_labelRegression': 'MLP', 'mnist_scan_ce': 'MLP',
    'cifar10_resnet_scan_labelRegression': 'ResNet18',
    'cifar10_resnet_ce_scan': 'ResNet18', 'exp_nanogpt_speedrun': 'transformer',
}
RANK_FAMILY_MARKER = {'MLP': 'o', 'ResNet18': 's', 'transformer': '^'}

#: the two rankings T4 prints in one cell, as separate panels
_RANK_COLS = ('val', 'test')
_RANK_TITLE = {'val': 'Validation loss', 'test': 'Test loss'}


def _rank_frame(ctx, scans, which):
    """``{(method, scan): (rank, n)}`` plus the field size of each scan.

    ``which`` is ``'val'`` or ``'test'``; a method with no finite rank on a scan is
    simply absent, which is the dash of T4 (not run there, or no finite confirmation
    value).
    """
    col = f'rank_{which}'
    r = ctx.ranking
    out, n_by = {}, {}
    for scan in scans:
        sub = r[r['scan'] == scan]
        if not len(sub):
            continue
        n_by[scan] = int(sub['n_methods'].max())
        for _, row in sub.iterrows():
            if C.finite(row[col]):
                out[(row['method'], scan)] = int(row[col])
    return out, n_by


def _rank_norm(rank, n):
    """Rank on a 0 (best of the field) to 1 (worst) scale.

    The field size runs from 5 (nanoGPT) to 15 (the synthetics), so a raw rank is not
    comparable across scans -- 9th of 11 is worse than 4th of 14 -- and every figure
    here colours and positions by this instead, keeping the raw rank as the annotation.
    """
    return 0.0 if n <= 1 else (float(rank) - 1.0) / (float(n) - 1.0)


def _rank_methods(ctx, scans, methods, sort, which='val'):
    """The row order both panels share: best median normalised rank at the top.

    A shared order is the point -- the val and test panels are read against each other,
    and a panel that sorted itself would move a method between them for no reason.
    """
    ranks, n_by = _rank_frame(ctx, scans, which)
    present = {m for m, _ in ranks}
    if isinstance(methods, str):
        if methods != 'all':
            raise ValueError(f"methods must be 'all' or a list, not {methods!r}")
        wanted = [m for m in hl.method_order(dict.fromkeys(present)) if m in present]
    else:
        wanted = [m for m in methods if m in present]
    if sort == 'median':
        def key(m):
            vals = [_rank_norm(ranks[(m, s)], n_by[s]) for s in scans
                    if (m, s) in ranks]
            return (float(np.median(vals)) if vals else 2.0, m)
    elif sort == 'sven':                    # Sven first, then the paper's method order
        key = lambda m: (style.canonical_method(m) != 'Sven', wanted.index(m))
    elif sort == 'field':                   # the order the rest of the paper uses
        key = wanted.index
    else:
        raise ValueError(f"sort must be 'median', 'sven' or 'field', not {sort!r}")
    return sorted(wanted, key=key)


def _rank_panel_axes(opts, ncol, nrow=1):
    import matplotlib.pyplot as plt

    figspec.paper_style(opts['font_fraction'])
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=C.figsize(ncol, nrow, opts['fraction'],
                                               aspect=opts['aspect'],
                                               extra_h=opts['extra_h']),
                             squeeze=False)
    return fig, axes


def _rank_heatmap(ctx, opts):
    """F14: T4 as a grid -- methods down, scans across, colour = normalised rank.

    One panel per ranking (validation, test), the same row order in both.  The raw rank
    stays printed in the cell, so this carries everything the table does; what it adds
    is that a row can be read at a glance and that the colour is comparable between
    columns of different field size.
    """
    import matplotlib.pyplot as plt

    scans = list(opts['scans'])
    columns = list(opts['columns'])
    order = _rank_methods(ctx, scans, opts['methods'], opts['sort'])
    fig, axes = _rank_panel_axes(opts, len(columns))
    # the dash of T4 (not run on that scan, or no finite confirmation value)
    cmap = plt.get_cmap(opts['cmap']).with_extremes(bad=opts['missing_color'])
    info = {}
    for j, which in enumerate(columns):
        ax = axes[0][j]
        ranks, n_by = _rank_frame(ctx, scans, which)
        grid = np.full((len(order), len(scans)), np.nan)
        for i, m in enumerate(order):
            for k, s in enumerate(scans):
                if (m, s) in ranks:
                    grid[i, k] = _rank_norm(ranks[(m, s)], n_by[s])
        ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0.0, vmax=1.0,
                  aspect='auto', interpolation='nearest')
        for i, m in enumerate(order):
            for k, s in enumerate(scans):
                if (m, s) not in ranks:
                    ax.text(k, i, '--', ha='center', va='center',
                            fontsize=opts['annot_fontsize'], color='0.45')
                    continue
                v = grid[i, k]
                # the label has to stay legible at both ends of the colormap
                ax.text(k, i, f'{ranks[(m, s)]:d}', ha='center', va='center',
                        fontsize=opts['annot_fontsize'],
                        color=('white' if v < opts['dark_below'] else 'black'))
        ax.set_xticks(range(len(scans)))
        # the field size goes INLINE, not on a second line: rotated two-line ticks
        # overlap their neighbours at this panel width
        ax.set_xticklabels([TABLE_SCAN_SHORT.get(s, _short_title(s))
                            + (f' ({n_by[s]})' if opts['show_field_size'] else '')
                            for s in scans], rotation=opts['xtick_rotation'],
                           ha=('right' if opts['xtick_rotation'] else 'center'),
                           rotation_mode=('anchor' if opts['xtick_rotation'] else None))
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([style.method_label(m) for m in order] if j == 0 else [])
        if j == 0:
            for lab, m in zip(ax.get_yticklabels(), order):
                if style.canonical_method(m) == 'Sven':
                    lab.set_fontweight('bold')
        ax.set_title(_RANK_TITLE.get(which, which))
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        info[which] = {'n_cells': int(np.isfinite(grid).sum()),
                       'n_missing': int(np.isnan(grid).sum())}
    if opts['colorbar']:
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axes[0].tolist(),
                          **opts['colorbar_kw'])
        cb.set_ticks([0.0, 1.0])
        cb.set_ticklabels(['best', 'worst'])
        cb.outline.set_visible(False)
    rec = C.provenance(
        functions=['headline.ranking_summary', 'headline.rank_matrix'],
        scans=[hl.dir_name(s, 'confirm') for s in scans],
        note=('T4 as a heatmap: colour is the rank normalised by the field size of its '
              'own scan (0 = best, 1 = worst), the printed number is the raw rank'),
        panels=list(columns), order=list(order), cells=info,
        provisional=sorted(ctx.provisional.intersection(scans)))
    return fig, {'axes': axes, 'provenance': rec, 'order': order, 'info': info}


def _rank_strip(ctx, opts):
    """F15: one row per method, one marker per scan, x = normalised rank.

    What the grid cannot show: how TIGHT a method is.  Sven's markers sit together at
    the good end on the four MLP scans and on nanoGPT and then jump to the bad end on
    the two ResNet18 scans, which is the paper's own account of where the method stands
    and is invisible in a table of ranks.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    scans = list(opts['scans'])
    columns = list(opts['columns'])
    order = _rank_methods(ctx, scans, opts['methods'], opts['sort'])
    fig, axes = _rank_panel_axes(opts, len(columns))
    fams = []
    for j, which in enumerate(columns):
        ax = axes[0][j]
        ranks, n_by = _rank_frame(ctx, scans, which)
        for i, m in enumerate(order):
            y = len(order) - 1 - i          # best method at the TOP
            ax.axhline(y, color='0.9', lw=0.5, zorder=0)
            vals = []
            for s in scans:
                if (m, s) not in ranks:
                    continue
                v = _rank_norm(ranks[(m, s)], n_by[s])
                vals.append(v)
                fam = RANK_FAMILY.get(s, 'MLP')
                fams.append(fam)
                ax.plot(v, y, RANK_FAMILY_MARKER.get(fam, 'o'),
                        color=style.method_color(m),
                        ms=opts['marker_size'],
                        mew=opts['marker_edge_lw'], mec='white', zorder=3,
                        alpha=opts['marker_alpha'])
            if vals and opts['median_tick']:
                ax.plot([float(np.median(vals))] * 2,
                        [y - opts['median_half_height'], y + opts['median_half_height']],
                        color='0.25', lw=opts['median_lw'], zorder=4)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xlim(*opts['xlim_norm'])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([style.method_label(m) for m in reversed(order)]
                           if j == 0 else [])
        if j == 0:
            for lab, m in zip(ax.get_yticklabels(), reversed(order)):
                if style.canonical_method(m) == 'Sven':
                    lab.set_fontweight('bold')
        ax.set_xlabel(opts['xlabel_text'])
        ax.set_title(_RANK_TITLE.get(which, which))
        ax.grid(axis='x', ls=':', alpha=0.4)
        ax.tick_params(axis='y', length=0)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
    if opts['figure_legend']:
        seen = [f for f in dict.fromkeys(fams)]
        handles = [Line2D([], [], ls='', marker=RANK_FAMILY_MARKER.get(f, 'o'),
                          color='0.35', ms=opts['marker_size']) for f in seen]
        fig.legend(handles, seen, **opts['figure_legend_kw'])
    rec = C.provenance(
        functions=['headline.ranking_summary', 'headline.rank_matrix'],
        scans=[hl.dir_name(s, 'confirm') for s in scans],
        note=('T4 as a strip plot: one marker per scan at the rank normalised by that '
              "scan's field size (0 = best, 1 = worst), the tick is the method's median"),
        panels=list(columns), order=list(order),
        provisional=sorted(ctx.provisional.intersection(scans)))
    return fig, {'axes': axes, 'provenance': rec, 'order': order}


def _rank_defaults(**over):
    """The knobs F14 and F15 share: both read the same matrix and differ in how they
    draw a cell of it."""
    out = dict(
        # -- content ----------------------------------------------------------
        scans=list(RANK_SCANS),         # one column (heatmap) / marker (strip) per scan
        methods='all',                  # 'all' or an explicit list
        columns=list(_RANK_COLS),       # one panel per ranking: 'val' and/or 'test'
        sort='median',                  # 'median' (best median first), 'sven', 'field'
        # -- geometry ---------------------------------------------------------
        fraction=0.40,
        aspect=1.05,
        extra_h=0.30,
        font_fraction=0.32,
    )
    out.update(over)
    return out


# ---------------------------------------------------------------------------
# F16 -- the paired per-seed differences against Sven, as a figure (T5b's rows)
# ---------------------------------------------------------------------------
def _paired_diffs(ctx, opts):
    """One panel per scan: every baseline's paired difference against Sven.

    x is the baseline, sorted so Sven's largest win is leftmost; y is the mean
    within-pair difference ``Sven - method`` in final validation loss (negative = Sven
    lower) with its 95 % t-interval as the bar.  A filled marker is a resolved
    difference (the interval excludes zero), a hollow one a tie.  The fraction of
    pairs Sven won is printed at each point: it carries the pairing count too, since
    every scan attempted the same seeds, so ``2/2`` on a 15-seed scan says that only
    two pairs finished.  An interval wider than the panel is clipped and its end drawn
    as an arrow, so one diverging baseline cannot flatten the others.
    """
    import matplotlib.pyplot as plt

    figspec.paper_style(opts['font_fraction'])
    scans = list(opts['scans'])
    ncol = int(opts['ncol'])
    nrow = int(math.ceil(len(scans) / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=C.figsize(ncol, nrow, opts['fraction'],
                                               aspect=opts['aspect'],
                                               extra_h=opts['extra_h']),
                             squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(scans):]:
        ax.remove()
    info = {}
    seen = []                                   # methods in order of first appearance
    for j, scan in enumerate(scans):
        ax = flat[j]
        pv = (ctx.paired(scan, opts['metric']).sort_values('mean')
              .reset_index(drop=True))
        seen.extend(m for m in pv['method'] if m not in seen)
        n_pairs_max = int(pv['n'].max()) if len(pv) else 0
        # the visible range follows the TYPICAL difference, not the largest: on the
        # MNIST scans one SOAP at -0.5 would otherwise flatten thirteen methods within
        # +/- 0.05 into a line.  Whatever falls outside is clipped and arrowed.
        reach = (pv['mean'].abs() + pv['half_width']).to_numpy(dtype=float)
        span = float(np.nanpercentile(reach, opts['clip_percentile'])) if len(pv) else 1.0
        lim = max(span * float(opts['clip_factor']), 1e-12)
        ax.axhspan(-lim, 0, color=opts['win_shade'], lw=0, zorder=0)
        ax.axhline(0, color='0.3', lw=0.7, zorder=1)
        clipped = []
        for x, r in pv.iterrows():
            color = style.method_color(r['method'])
            lo, hi = float(r['ci_low']), float(r['ci_high'])
            lo_c, hi_c = max(lo, -lim), min(hi, lim)
            ax.plot([x, x], [lo_c, hi_c], color=color, lw=opts['ci_lw'], zorder=2,
                    solid_capstyle='butt')
            for end, y, clip in ((lo, lo_c, lo < -lim), (hi, hi_c, hi > lim)):
                if clip:
                    ax.plot(x, y, marker='v' if end < 0 else '^', ms=opts['arrow_ms'],
                            color=color, zorder=3, clip_on=False)
                    clipped.append((r['method'], end))
            resolved = bool(r.get('significant'))
            ax.plot(x, float(r['mean']), 'o', ms=opts['marker_ms'], color=color,
                    mfc=color if resolved else 'white', mew=opts['marker_mew'],
                    zorder=4)
        ax.set_ylim(-lim * opts['ylim_pad'], lim * opts['ylim_pad'])
        ax.set_xlim(-0.6, len(pv) - 0.4)
        ax.set_xticks(range(len(pv)))
        # the pairs-won count rides on the tick label ("SGD 15/15"): at fourteen
        # methods per 1.4 in panel there is no room beside the points.  With
        # ``xtick_labels='ratio'`` the name is dropped from the tick (the colour key in
        # the legend strip carries it) and only "15/15" remains.
        ratio = [f"{int(w)}/{int(n)}" for w, n in zip(pv['sven_better'], pv['n'])]
        if opts['xtick_labels'] == 'none':
            # no tick text at all: the colour key names the baseline and the corner
            # note gives the seed count
            labels = [''] * len(pv)
            ax.text(0.97, 0.96, f"$N={n_pairs_max}$ seeds", transform=ax.transAxes,
                    ha='right', va='top', fontsize=opts['pairs_fontsize'])
        elif opts['xtick_labels'] == 'ratio':
            labels = ratio
        elif opts['xtick_labels'] == 'won':
            # the numerator alone, with the scan's pair count printed once in the
            # corner; a baseline with FEWER finished pairs than that keeps its "w/n"
            labels = [f"{int(w)}" if int(n) == n_pairs_max else r_
                      for w, n, r_ in zip(pv['sven_better'], pv['n'], ratio)]
            ax.text(0.97, 0.96, f"$N={n_pairs_max}$ seeds", transform=ax.transAxes,
                    ha='right', va='top', fontsize=opts['pairs_fontsize'])
        else:
            labels = [hl.display_name(m) + (f"  {r}" if opts['show_ratio'] else '')
                      for m, r in zip(pv['method'], ratio)]
        rot = float(opts['xtick_rotation'])
        ax.set_xticklabels(labels, rotation=rot, ha='center' if rot == 0 else 'right',
                           rotation_mode='anchor', fontsize=opts['xtick_fontsize'])
        ax.tick_params(axis='x', length=0)
        ax.set_title(_short_title(scan))
        if j % ncol == 0:
            ax.set_ylabel(opts['ylabel_text'])
        ax.grid(axis='y', ls=':', alpha=0.35)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        info[scan] = {'n_methods': int(len(pv)), 'pairs_max': n_pairs_max,
                      'clipped': [[m, float(v)] for m, v in clipped]}
    if opts['figure_legend']:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [Line2D([], [], ls='', marker='o', color='0.3', ms=opts['marker_ms']),
                   Line2D([], [], ls='', marker='o', color='0.3', mfc='white',
                          ms=opts['marker_ms'], mew=opts['marker_mew']),
                   Line2D([], [], color='0.3', lw=opts['ci_lw']),
                   Patch(color=opts['win_shade'])]
        labels = ['resolved (95% interval excludes 0)', 'not resolved',
                  '95% t-interval', 'Sven lower']
        if opts['method_legend']:
            labels[0] = 'resolved (interval excludes 0)'
        kw = dict(opts['figure_legend_kw'])
        if opts['method_legend']:
            # the colour key, one dot per baseline, ahead of the mark key.  A legend
            # fills column by column, so each block is padded with blank entries to a
            # whole number of columns and the mark key stays together at the right.
            rows = int(opts['legend_rows'])
            blank = lambda: Line2D([], [], ls='')
            pad = lambda n: (-n) % rows
            key_h = [Line2D([], [], ls='', marker='o', ms=opts['marker_ms'],
                            color=style.method_color(m)) for m in seen]
            key_l = [hl.display_name(m) for m in seen]
            handles = (key_h + [blank() for _ in range(pad(len(key_h)))]
                       + handles + [blank() for _ in range(pad(len(handles)))])
            labels = (key_l + [''] * pad(len(key_l))
                      + labels + [''] * pad(len(labels)))
            kw['ncol'] = len(handles) // rows
        fig.legend(handles, labels, **kw)
    outcome = opts['metric'] != hl.SELECTION_METRIC
    rec = C.provenance(
        functions=(['headline.paired_outcome_vs_sven'] if outcome
                   else ['headline.paired_vs_sven', 'paired.paired_table']),
        scans=[hl.dir_name(s, 'confirm') for s in scans],
        note=(f'paired per-seed differences Sven - method in {opts["metric"]}, '
              'mean with 95% t-interval; a filled marker is an interval excluding zero; '
              'the label is pairs Sven won / pairs finished; intervals wider than '
              'clip_factor x the largest |mean| are clipped and arrowed'),
        panels=info, provisional=sorted(ctx.provisional.intersection(scans)))
    return fig, {'axes': axes, 'provenance': rec, 'info': info}


# ---------------------------------------------------------------------------
# The figure registry -- what `build()` writes and what a notebook can change
# ---------------------------------------------------------------------------
#: the per-panel key of F1's main-text version.  Kept out of the rcParams defaults
#: because at 4.3 pt it is smaller than anything else on the page and is a deliberate
#: choice, not a style.
_F1_PANEL_LEGEND = dict(loc='upper right', fontsize=4.3, handlelength=1.0,
                        handletextpad=0.4, borderpad=0.25, labelspacing=0.18,
                        borderaxespad=0.3, framealpha=0.82, fancybox=False,
                        edgecolor='0.8')

#: the figure-level legend strip every curve figure in this group carries
_LEGEND_STRIP = dict(loc='outside lower center', fontsize=6.0, handlelength=1.2,
                     columnspacing=0.8)


def _headline_defaults(**over):
    """F1's knobs.  ``headline_curves`` and ``headline_curves_all`` are the SAME builder
    with a different ``methods`` / weight / legend set, so the two differ only by
    ``over``."""
    out = dict(
        # -- content ----------------------------------------------------------
        scans=list(HEADLINE_MAIN),      # one column per scan, left to right
        methods='top',                  # 'top' (front of the field + Sven), 'all', or
                                        # an explicit list of method names
        top_n=MAIN_PANEL_TOP_N,         # how many the 'top' field holds, Sven aside
        exclude=[],                     # methods to drop from whatever that chose: a
                                        # list for every panel, or {scan: [names]} for
                                        # one panel ('*' = every panel)
        versus=['epoch', 'time'],       # one panel row per x axis
        # -- geometry ---------------------------------------------------------
        fraction=0.32,                  # panel width as a fraction of \linewidth
        aspect=0.86,                    # panel height / panel width
        extra_h=0.38,                   # inches added for the legend strip
        font_fraction=0.32,             # which panel width the type is sized for
        # -- labels -----------------------------------------------------------
        panel_titles=True,              # the task name over the top row
        xlabel_columns='middle',        # 'all' / 'none' / 'first' / 'middle' / 'last'
        ylabel_columns='first',         # or an explicit list of column indices
        # -- lines ------------------------------------------------------------
        lw=1.1,
        sven_lw=1.9,
        # -- legends ----------------------------------------------------------
        panel_legend=True,              # each panel names its own curves
        panel_legend_kw=dict(_F1_PANEL_LEGEND),
        panel_legend_frame_lw=0.3,
        figure_legend_methods=False,    # does the strip repeat the method key?
        figure_legend_ncol=None,        # None = as many rows as columns, 3..6 wide
        figure_legend_kw=dict(_LEGEND_STRIP),
    )
    out.update(over)
    return out


def _allseed_defaults(scans, **over):
    """F13's knobs.  One spec per scan group, everything else shared; the SOAP-free
    variants are this with ``exclude``."""
    out = dict(
        scans=list(scans),              # one panel row per scan
        columns=[list(c) for c in _ALLSEED_COLS],   # [which, versus] per column
        methods='all',                  # 'all' / 'top' / an explicit list
        top_n=MAIN_PANEL_TOP_N,         # only consulted when methods == 'top'
        exclude=[],                     # method names to drop from whatever that chose:
                                        # a list for every row, or {scan: [names]}
        # two columns across the same \linewidth as the old three: wider panels at
        # (very nearly) the row height they had, so each one is rectangular
        fraction=0.48,
        aspect=0.55,
        extra_h=0.70,
        lw=0.85,
        sven_lw=1.8,
        # the announced y-axis truncation (see _clip_outliers): the limit is
        # clip_factor x the median pre-training loss, applied only when the panel
        # spans another clip_headroom above that
        clip_outliers=True,
        clip_factor=20.0,
        clip_headroom=50.0,
        clip_note_xy=[0.97, 0.95],
        clip_note_fontsize=4.4,
        figure_legend_kw=dict(_LEGEND_STRIP, ncol=5),
        font_fraction=0.32,
    )
    out.update(over)
    return out


FIGURE_SPECS = figspec.check_defaults({
    'headline_curves': figspec.FigureSpec(
        draw=_headline_curves,
        doc=('F1: validation loss vs epoch (top) and vs standalone synchronised train '
             'time (bottom) for the three headline tasks, confirmation seeds, +/- 1 std. '
             "Set methods='all' to put the whole field on it; the legend strip, the "
             'per-panel key, the aspect ratio and the axis labels are all knobs.'),
        defaults=_headline_defaults(),
    ),
    'headline_curves_all': figspec.FigureSpec(
        draw=_headline_curves,
        doc=('F1, appendix version: the same panels with EVERY method that has a '
             'selected configuration, one shared method key instead of a per-panel one.'),
        defaults=_headline_defaults(methods='all', lw=0.9, extra_h=0.62,
                                    panel_legend=False, figure_legend_methods=True),
    ),
    'cost_memory': figspec.FigureSpec(
        draw=_cost_memory,
        doc=('F3: step time (left) and peak GPU memory (right) against the parameter '
             'count over five architectures, from the standalone timing pass.'),
        defaults=dict(
            scans=list(F3_SCANS),           # smallest parameter count first
            methods=list(F3_METHODS),       # the curves drawn, Sven emphasised
            fraction=0.49,
            aspect=0.80,
            extra_h=0.34,
            font_fraction=0.49,
            ms=2.6,                         # marker size, baselines / Sven
            sven_ms=3.4,
            lw=1.0,
            sven_lw=1.8,
            capture_note=True,              # the "Sven capture -- ..." box per panel
            capture_note_fontsize=4.8,
            arch_tick_fontsize=4.4,         # the architecture names on the top axis
            cluster_decades=0.5,            # MLPs within this many decades share a tick
            figure_legend_kw=dict(loc='outside lower center', ncol=6, fontsize=6.2,
                                  handlelength=1.2, columnspacing=0.8),
        ),
    ),
    'k_sweeps': figspec.FigureSpec(
        draw=_k_sweeps,
        doc=("F12 top: Sven's k sweep at each scan's SELECTED (lr, rtol); the thick "
             'curve is the selected k.'),
        defaults=dict(
            scans=list(SWEEP_SCANS),
            axis='k',                       # which Sven axis is swept ('k' or 'rtol')
            which='val',                    # 'val' or 'train' loss
            fraction=0.245,
            aspect=1.05,
            extra_h=0.42,
            font_fraction=0.32,
            cmap=_SWEEP_CMAP,               # one colour per swept value
            cmap_range=[0.05, 0.9],
            lw=1.0,
            selected_lw=1.9,                # the selected value's own weight
            band_alpha=0.15,                # the +/- 1 std seed band
            selected_note=True,             # the "eta=..., rtol=..." corner note
            selected_note_xy=[0.03, 0.03],
            selected_note_fontsize=5.4,
            panel_legend=True,
            legend_handle_lw=1.4,
            panel_legend_kw=dict(_LEGEND_FRAME, fontsize=4.8, ncol=2, loc='upper right',
                                 handlelength=1.0, labelspacing=0.15,
                                 columnspacing=0.6),
        ),
    ),
    'hparam_landscape': figspec.FigureSpec(
        draw=_hparam_landscape,
        doc=("F12 bottom: Sven's final-loss landscape over k at fixed rtol (top row) and "
             'over rtol at fixed k (bottom row), at the selected lr; the star is the '
             'selected configuration.'),
        defaults=dict(
            scans=list(SWEEP_SCANS),
            rows=[['k', 'rtol'], ['rtol', 'k']],    # [x axis, one line per] per row
            fraction=0.245,
            aspect=1.0,
            extra_h=0.30,
            font_fraction=0.32,
            cmap=_SWEEP_CMAP,
            cmap_range=[0.05, 0.9],
            ms=2.4,
            lw=0.9,
            selected_lw=1.8,
            mark_selected=True,             # the ringed star on the selected point
            star_ms=7,
            star_mew=0.9,
            # the smallest k within this factor of the best score on the selected rtol
            # line is num<Scan>SvenKSaturate -- a number the prose quotes
            saturation_factor=1.05,
            panel_legend=True,
            legend_handle_lw=1.2,
            panel_legend_kw=dict(_LEGEND_FRAME, fontsize=4.4, ncol=2, loc='best',
                                 handlelength=0.9, labelspacing=0.12,
                                 columnspacing=0.5, title_fontsize=4.4),
        ),
    ),
    'allseed_curves': figspec.FigureSpec(
        draw=_allseed_curves,
        doc=('F13: every method, confirmation seeds, on two axes (val vs epoch, val vs '
             'standalone time) for 1D regression, the polynomial and MNIST label '
             'regression.'),
        defaults=_allseed_defaults(ALLSEED_SCANS),
    ),
    'allseed_curves_ce_lm': figspec.FigureSpec(
        draw=_allseed_curves,
        doc='F13, continued: the same two axes for MNIST-CE and nanoGPT.',
        defaults=_allseed_defaults(ALLSEED_SCANS_B),
    ),
    'allseed_curves_ce': figspec.FigureSpec(
        draw=_allseed_curves,
        doc=('App. I: the MNIST cross-entropy all-seed trajectories alone. '
             '`allseed_curves_ce_lm` paired this row with nanoGPT so that every headline '
             'scan had an all-seed figure, but nanoGPT has its own in the transformers '
             'appendix and the MNIST-CE appendix is about MNIST-CE.'),
        defaults=_allseed_defaults(('mnist_scan_ce',), extra_h=0.62, exclude=['SOAP']),
    ),
    'allseed_curves_nosoap': figspec.FigureSpec(
        draw=_allseed_curves,
        doc=('F13 without SOAP: the same panels as `allseed_curves`, minus the one '
             'method whose seed spread sets the y range of the MNIST label-regression '
             'row. Everything else is drawn identically, so the two are comparable.'),
        defaults=_allseed_defaults(ALLSEED_SCANS, exclude=['SOAP']),
    ),
    'allseed_curves_ce_lm_nosoap': figspec.FigureSpec(
        draw=_allseed_curves,
        doc='F13 continued, without SOAP: `allseed_curves_ce_lm` minus SOAP.',
        defaults=_allseed_defaults(ALLSEED_SCANS_B, exclude=['SOAP']),
    ),
    'rank_heatmap': figspec.FigureSpec(
        draw=_rank_heatmap,
        doc=('F14: T4 (`tables_v2/ranking.tex`) as a grid -- every method on every '
             'scan, validation (left) and test (right). Colour is the rank normalised '
             'by its own scan\'s field size, so the columns are comparable; the '
             'printed number is the raw rank, so nothing the table says is lost.'),
        defaults=_rank_defaults(
            # two panels at half the linewidth each = a figure exactly \linewidth wide,
            # so \includegraphics[width=\linewidth] shows it 1:1 and the type is the
            # size paper_style drew it at
            fraction=0.5,
            aspect=0.95,
            # -- cells ------------------------------------------------------
            cmap='viridis_r',           # dark = best; reversed so good reads heavy
            missing_color='0.93',       # the dash cells (not run / no finite value)
            dark_below=0.45,            # normalised rank under this gets white text
            annot_fontsize=5.0,
            show_field_size=True,       # "MNIST-LR (14)": the field size of that scan
            xtick_rotation=45,          # seven task names do not fit a 2.2 in panel flat
            # -- key --------------------------------------------------------
            colorbar=True,
            colorbar_kw=dict(fraction=0.045, pad=0.03, aspect=28),
        ),
    ),
    'paired_diffs': figspec.FigureSpec(
        draw=_paired_diffs,
        doc=('F16, App. G: the paired per-seed differences against Sven (T5b) as one '
             'panel per scan -- mean and 95% interval per baseline, filled when '
             'resolved, with the pairs-won count at each point.'),
        defaults=dict(
            scans=list(RANK_SCANS),         # one panel per scan
            metric=hl.SELECTION_METRIC,     # or 'final_test_loss' (an outcome)
            ncol=4,
            fraction=0.25,
            aspect=1.25,                    # the rotated tick labels take ~a third of
                                            # the cell, so the panel needs the height
            extra_h=0.30,                   # the legend strip
            font_fraction=0.32,
            # -- the y range ------------------------------------------------
            clip_percentile=75,             # visible range = clip_factor x this
            clip_factor=2.0,                # percentile of |mean| + half-width
            ylim_pad=1.08,
            # -- marks --------------------------------------------------------
            marker_ms=3.6,
            marker_mew=0.9,
            ci_lw=1.1,
            arrow_ms=3.0,
            show_ratio=True,                # "pairs Sven won / pairs finished", on
                                            # the tick label
            xtick_labels='names',           # 'names' (+ ratio), 'ratio' alone, 'won'
                                            # (the numerator) or 'none'; the last two
                                            # print N seeds in the corner
            pairs_fontsize=5.0,             # the "N = 15 seeds" corner note ('won')
            method_legend=False,            # a colour key per baseline in the strip
            legend_rows=3,                  # ... laid out in this many rows
            xtick_rotation=60,
            xtick_fontsize=4.8,
            win_shade='0.94',               # the y < 0 half-plane: Sven lower
            ylabel_text='Sven $-$ method',  # ... in final validation loss (caption)
            figure_legend=True,
            figure_legend_kw=dict(_LEGEND_STRIP, ncol=4),
        ),
    ),
    'paired_diffs_main': figspec.FigureSpec(
        draw=_paired_diffs,
        doc=('F16, main-text form: the same paired differences without the two CIFAR-10 '
             'panels -- one row of five, so the comparison the headline paragraph makes '
             'sits beside it; App. G keeps the seven-panel version.'),
        defaults=dict(
            scans=[s for s in RANK_SCANS if not s.startswith('cifar10')],
            metric=hl.SELECTION_METRIC,
            ncol=5,
            fraction=0.2,
            aspect=0.95,                    # no tick text to make room for
            extra_h=0.62,                   # the strip holds the colour key too
            font_fraction=0.32,
            clip_percentile=75, clip_factor=2.0, ylim_pad=1.08,
            marker_ms=3.2, marker_mew=0.8, ci_lw=1.0, arrow_ms=2.8,
            show_ratio=True,
            xtick_labels='none',            # bare ticks; N seeds in the corner
            pairs_fontsize=5.0,
            method_legend=True,
            legend_rows=3,
            xtick_rotation=90, xtick_fontsize=4.4,
            win_shade='0.94',
            ylabel_text='Sven $-$ method',
            figure_legend=True,
            figure_legend_kw=dict(_LEGEND_STRIP, fontsize=5.2, columnspacing=1.0),
        ),
    ),
    'paired_diffs_main_test': figspec.FigureSpec(
        draw=_paired_diffs,
        doc=('F16, main-text form on TEST loss: the same paired differences without the two CIFAR-10 '
             'panels -- one row of five, so the comparison the headline paragraph makes '
             'sits beside it; App. G keeps the seven-panel version.'),
        defaults=dict(
            scans=[s for s in RANK_SCANS if not s.startswith('cifar10')],
            metric='final_test_loss',   # an outcome of the val-selected configs
            ncol=5,
            fraction=0.2,
            aspect=0.95,                    # no tick text to make room for
            extra_h=0.62,                   # the strip holds the colour key too
            font_fraction=0.32,
            clip_percentile=75, clip_factor=2.0, ylim_pad=1.08,
            marker_ms=3.2, marker_mew=0.8, ci_lw=1.0, arrow_ms=2.8,
            show_ratio=True,
            xtick_labels='none',            # bare ticks; N seeds in the corner
            pairs_fontsize=5.0,
            method_legend=True,
            legend_rows=3,
            xtick_rotation=90, xtick_fontsize=4.4,
            win_shade='0.94',
            ylabel_text='Sven $-$ method',
            figure_legend=True,
            figure_legend_kw=dict(_LEGEND_STRIP, fontsize=5.2, columnspacing=1.0),
        ),
    ),
    'rank_strip': figspec.FigureSpec(
        draw=_rank_strip,
        doc=('F15: the same ranking as a strip plot -- one row per method, one marker '
             'per scan at its normalised rank, the tick at the method\'s median. This '
             'is the one that shows how tight or spread a method is across tasks, '
             'which the grid and the table cannot.'),
        defaults=_rank_defaults(
            aspect=1.15,
            extra_h=0.42,
            # -- markers ----------------------------------------------------
            marker_size=3.4,
            marker_alpha=0.95,
            marker_edge_lw=0.4,
            median_tick=True,           # the black tick at the method's median
            median_lw=1.1,
            median_half_height=0.32,
            # -- axes -------------------------------------------------------
            xlim_norm=[-0.06, 1.06],
            xlabel_text='Normalised rank  (0 = best of field)',
            # -- key --------------------------------------------------------
            figure_legend=True,         # the three model-family marker shapes
            figure_legend_kw=dict(_LEGEND_STRIP, ncol=3),
        ),
    ),
})

#: how :func:`build` turns a builder's ``meta`` into ``figure_info[name]``, the dict
#: :func:`_macros` reads.  A figure absent here feeds no macro (F1's all-methods twin
#: draws the same methods as F1, and F13 feeds nothing).
_FIGURE_INFO = {
    'headline_curves': lambda meta: {s: list(v)
                                     for s, v in meta['per_scan'].items()},
    'cost_memory': lambda meta: meta['frame'].to_dict('records'),
    'k_sweeps': lambda meta: meta['info'],
    'hparam_landscape': lambda meta: meta['info'],
}


def _draw_and_save(name, ctx):
    """Draw one figure and write it -- this module's ONE save path.

    The save happens inside the same ``rc_context`` as the draw on purpose: the builder
    calls :func:`figspec.paper_style`, and half of what that sets (``pdf.fonttype``, the
    sans-serif stack, ``savefig.bbox``) is read by ``savefig``, not by the drawing calls.
    """
    import matplotlib.pyplot as plt

    spec = FIGURE_SPECS[name]
    opts = figspec.figure_opts(name, spec.defaults)
    with plt.rc_context(opts.get('rc') or {}):
        fig, meta = spec.draw(ctx, opts)
        figspec.apply_opts(fig, meta.get('axes'), opts)
        pdf, png = C.save_fig(fig, name, GROUP, meta['provenance'])
    return pdf, png, meta


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _t1_headline(ctx, name='headline_confirm', compact=False):
    """T1: the main-text results table -- Sven and seven baselines on the three
    headline tasks, on the confirmation seeds with the tuning seeds beside."""
    rows, groups_of, midrules = [], [], []
    for scan in HEADLINE_MAIN:
        conf, eff = ctx.conf(scan), ctx.eff(scan)
        present = [m for m in MAIN_TABLE_METHODS if m in set(conf['method'])]
        groups_of.append((scan, len(present)))
        # the task names a BLOCK, not a column: a repeated 26-character task name is
        # 1 in of a 5.5 in text block spent on nothing.  Every block but the first is
        # ruled off from the one above it; the first already sits under the header rule.
        if rows:
            midrules.append(len(rows))
        rows.append(C.span_row(hl.scan_title(scan)))
        eff_by = {r['method']: r for _, r in eff.iterrows()}
        best = _best_val_methods(ctx, scan, present)
        for m in present:
            c = ctx.row(scan, m)
            e = eff_by.get(m)
            row = {
                'Method': hl.display_name(m),
                'Configuration': _short_config(c['config']),
                'Val. loss': _val_cell(c['val_conf'], c['val_conf_std'],
                                       best=(m in best)),
                'Val. (tune)': C.fmt_sig(c['val_tune'], 4),
                'Test loss': C.fmt_pm(c['test_conf'], c.get('test_conf_std'), 4),
                'Test acc.': (C.fmt_pct(c.get('acc_conf'), 1)
                              if C.finite(c.get('acc_conf')) else C.Raw('--')),
                'fin./att.': _counts_cell(c),
                'Ep. to tgt.': _epochs_cell(ctx, scan, m),
            }
            if not compact:
                row['s/ep.'] = C.fmt_sig(e['epoch_s'], 3) if e is not None else C.Raw('--')
                row['MB'] = C.fmt_sig(e['peak_gpu_mem_mb'], 4) if e is not None \
                    else C.Raw('--')
            rows.append(row)
    notes = [
        'confirmation seeds (5 fresh model seeds; the two synthetics also carry 3 data '
        'seeds, so 15 runs); mean +/- 1 std over seeds',
        'the target is the median method\'s confirmation-seed final validation loss '
        '(headline.TARGET_MULTIPLIERS); a bracket gives reached/attempted when not '
        'every run got there',
        's/epoch and peak memory come from the standalone timing pass, one run per GPU',
        'dagger on fin./att. = more than half the confirmation runs failed, so the mean '
        'beside it is a mean over the survivors',
        'bold = the lowest confirmation-seed validation loss of that task block, and '
        'every method within sqrt(s_best^2 + s_i^2) of it -- a display convention over '
        'the printed mean and band, not a significance test (that is the paired '
        'comparison of T4); a daggered row is not eligible to win a block',
    ]
    rec = C.provenance(
        functions=['headline.confirmation_table', 'headline.confirmation_view',
                   'headline.efficiency_table', 'headline.time_to_target_table',
                   'headline.epochs_to_target'],
        scans=[hl.dir_name(s, k) for s in HEADLINE_MAIN
               for k in ('', 'confirm', 'timing')],
        note='T1, the main-text results table',
        methods_per_scan=dict(groups_of),
        provisional=sorted(ctx.provisional.intersection(HEADLINE_MAIN)))
    return C.write_table(name, rows, provenance_record=rec, fit=True,
                         align=['l', 'p{0.95in}'] + ['r'] * (6 if compact else 8),
                         midrules=midrules, notes=notes,
                         caption=('Headline results on the confirmation seeds. Sven is '
                                  'second on the polynomial scan, third on MNIST and '
                                  'tied with AdamW on nanoGPT.'),
                         label='tab:headline' + ('compact' if compact else ''))


def _t2_time_to_target(ctx, name='time_to_target', scans=None, compact=False):
    """T2: epochs / steps / examples / standalone seconds to each pre-declared target."""
    scans = scans or HEADLINE_MAIN
    methods = MAIN_TABLE_METHODS if compact else None
    rows = []
    targets_seen = {}
    for scan in scans:
        t = ctx.ttt(scan)
        targets_seen[scan] = t.attrs['targets']
        for label in (t['target'].unique() if not compact else [MAIN_TARGET]):
            sub = t[t['target'] == label]
            if methods is not None:
                sub = sub[sub['method'].isin(methods)]
            sub = sub.sort_values('epochs', na_position='last')
            if not len(sub):
                continue
            target_value = float(sub.iloc[0]['target_value'])
            rows.append(C.span_row(
                C.Raw(f'{C.latex_escape(hl.scan_title(scan))}, target '
                      f'{C.latex_escape(label)} = {C.fmt_sig(target_value, 3)}')))
            for _, r in sub.iterrows():
                rows.append({
                    'Method': hl.display_name(r['method']),
                    'Epochs': C.fmt_pm(r['epochs'], r['epochs_std'], 3),
                    'Steps': C.fmt_sig(r['steps'], 3),
                    'Examples': C.fmt_sig(r['examples'], 3),
                    'Alone (s)': C.fmt_sig(r['standalone_s'], 3),
                    'Reached': C.fmt_counts(r['n_reached'], r['n_runs']),
                    'Never': C.fmt_int(r['n_never_reached']),
                    'Div.': C.fmt_int(r['n_diverged']),
                })
    notes = [
        'epochs from the confirmation runs; seconds = epochs x the standalone timing '
        'pass\'s seconds per epoch for the same configuration',
        '"Never" = the run trained to the end without reaching the target; "Diverged" = '
        'the run has no trajectory to ask. The two are different facts and are not added',
        'on the two synthetics one pooled target is applied to three random problem '
        'instances, so compare methods at a target, not targets with each other',
    ]
    rec = C.provenance(
        functions=['headline.time_to_target_table', 'headline.targets_for',
                   'headline.target_reference', 'headline.epochs_to_target'],
        scans=[hl.dir_name(s, k) for s in scans for k in ('confirm', 'timing')],
        note='T2' + (' (compact, main text)' if compact else ' (full, appendix)'),
        targets={s: {k: float(v) for k, v in t.items()}
                 for s, t in targets_seen.items()},
        provisional=sorted(ctx.provisional.intersection(scans)))
    return C.write_table(name, rows, provenance_record=rec, fit=True,
                         align=['l'] + ['r'] * 7, notes=notes,
                         caption='Time to a pre-declared target.',
                         label='tab:ttt' + ('main' if compact else 'full'))


def _t3_protocol(ctx, name='protocol'):
    """T3: the protocol table -- one row per scan, what was actually run."""
    rows = []
    for scan in ctx.scans:
        df = ctx.tune(scan)
        conf = ctx.conf(scan)
        sels = hl.selection_methods(scan, ctx.payload)
        n_ds = int(pd.to_numeric(conf['n_data_seeds'], errors='coerce').max() or 0)
        rows.append({
            'Task': hl.scan_title(scan),
            'Loss': {'mse': 'squared error', 'label_regression': 'label regression',
                     'ce': 'cross-entropy', 'lm_ce': 'token cross-entropy'}.get(
                         str(df['loss'].dropna().iloc[0]) if 'loss' in df.columns
                         else '', '--'),
            # the split sizes live in App. C's "Datasets and Splits" table, not here
            C.Raw('$P$'): C.fmt_int(_scalar(ctx, scan, 'n_params')),
            C.Raw('$B$'): C.fmt_int(_scalar(ctx, scan, 'batch_size')),
            'Epochs': C.fmt_int(_n_epochs(ctx, scan)),
            'Seeds': C.Raw('5+5' + (f'$\\times{n_ds}$' if n_ds > 1 else '')),
            'Methods': C.fmt_int(len(sels)),
            'Grid runs': C.fmt_int(len(df)),
            'Confirm fin./att.': C.fmt_counts(
                pd.to_numeric(conf['fin_conf'], errors='coerce').sum(),
                pd.to_numeric(conf['att_conf'], errors='coerce').sum()),
        })
    notes = [
        'selection uses the seed-mean final VALIDATION loss only (eligible -> fewest '
        'diverged -> seed mean); test metrics are outcomes of the configuration '
        'validation chose',
        '"5 + 5" = five tuning seeds, then the selected configuration re-run on five '
        'fresh confirmation seeds, which is what every number in the paper reports',
        'a diverged run is counted, never dropped, under the wider analysis rule '
        '(recorded status, a non-finite curve, or a final validation loss more than 10x '
        'its pre-training value)',
    ]
    rec = C.provenance(
        functions=['headline.load_selection', 'headline.confirmation_table',
                   'headline.assert_no_test_selection', 'style.is_diverged'],
        scans=[hl.dir_name(s, k) for s in ctx.scans for k in ('', 'confirm')],
        note='T3, the protocol table for App. C',
        n_selections_checked=hl.assert_no_test_selection(payload=ctx.payload,
                                                         scans=ctx.scans),
        provisional=sorted(ctx.provisional))
    return C.write_table(name, rows, provenance_record=rec, notes=notes, fit=True,
                         align=['l', 'l'] + ['c'] * 7,
                         caption='The protocol, as run.', label='tab:protocol')


#: short scan names for a table HEADER (a 5.5 in page cannot take seven full titles)
TABLE_SCAN_SHORT = {
    'toy_1d_scan': '1D reg.', 'polynomial_scan': 'Poly.',
    'mnist_scan_labelRegression': 'MNIST-LR', 'mnist_scan_ce': 'MNIST-CE',
    'cifar10_resnet_scan_labelRegression': 'CIFAR-LR',
    'cifar10_resnet_ce_scan': 'CIFAR-CE', 'exp_nanogpt_speedrun': 'nanoGPT',
}


def _t4_ranking(ctx, name='ranking'):
    """T4: rank of every method on every scan, validation and test.

    Methods down the side and scans across the top, with the two ranks in one cell as
    ``validation/test``.  The transpose (scans down, 15 method columns) needs 17 columns
    and overflows a 5.5 in page by 1.6 in even at ``\\tiny``; this shape is eight columns
    wide and carries the same content.
    """
    ranking = ctx.ranking
    n_by_scan = {s: int(ranking[ranking['scan'] == s]['n_methods'].max())
                 for s in ctx.scans}
    methods = sorted({m for m in ranking['method']},
                     key=lambda m: (m != hl.SVEN_LABEL, m))
    columns = ['Method'] + [TABLE_SCAN_SHORT.get(s, hl.scan_title(s))
                            for s in ctx.scans]
    rows = [{'Method': C.Raw(r'\emph{field size} $n$'),
             **{TABLE_SCAN_SHORT.get(s, hl.scan_title(s)): C.fmt_int(n_by_scan[s])
                for s in ctx.scans}}]
    for m in methods:
        row = {'Method': hl.display_name(m)}
        for s in ctx.scans:
            sub = ranking[(ranking['scan'] == s) & (ranking['method'] == m)]
            col = TABLE_SCAN_SHORT.get(s, hl.scan_title(s))
            if not len(sub):
                row[col] = C.Raw('--')
                continue
            rv, rt = sub.iloc[0]['rank_val'], sub.iloc[0]['rank_test']
            if not C.finite(rv):
                row[col] = C.Raw('--')
            elif C.finite(rt):
                row[col] = C.Raw(f'{int(rv)}/{int(rt)}')
            else:
                row[col] = C.fmt_int(rv)
        rows.append(row)
    notes = [
        'each cell is the rank on the confirmation-seed mean final VALIDATION loss / the '
        'rank on the final TEST loss, out of the field size n in the first row',
        'a dash means the method was not run on that scan (HIG refuses '
        'batch-statistics normalisation; K-FAC has no eligible MNIST configuration) or '
        'produced no finite confirmation value',
    ]
    rec = C.provenance(
        functions=['headline.ranking_summary', 'headline.rank_matrix',
                   'headline.confirmation_table'],
        scans=[hl.dir_name(s, 'confirm') for s in ctx.scans],
        note='T4, the ranking summary',
        provisional=sorted(ctx.provisional))
    return C.write_table(name, rows, columns=columns, provenance_record=rec, fit=True,
                         align=['l'] + ['r'] * (len(columns) - 1), notes=notes,
                         midrules={1}, caption='Rank of every method on every scan, '
                                                'validation / test.',
                         label='tab:ranking')


def _t5_confirmation(ctx, scan):
    """T5: the full per-scan confirmation table."""
    conf = ctx.conf(scan)
    has_acc = pd.to_numeric(conf.get('acc_conf'), errors='coerce').notna().any()
    has_tre = pd.to_numeric(conf.get('treval_conf'), errors='coerce').notna().any()
    has_ds = bool(conf.attrs.get('data_seeds')) and len(conf.attrs['data_seeds']) > 1
    rows = []
    for _, r in conf.iterrows():
        row = {'Method': hl.display_name(r['method']),
               'Selected configuration': C.latex_escape(r['config']),
               'Val. loss': C.fmt_pm(r['val_conf'], r['val_conf_std'], 4),
               'Test loss': C.fmt_pm(r.get('test_conf'), r.get('test_conf_std'), 4)}
        # a wrapping p-column carries the configuration; the numeric columns stay right
        # aligned, so the table is as wide as its NUMBERS and not as its longest string
        if has_acc:
            row['Test acc.'] = (C.fmt_pct(r.get('acc_conf'), 2)
                                if C.finite(r.get('acc_conf')) else C.Raw('--'))
        if has_tre:
            row['Train (eval)'] = C.fmt_pm(r.get('treval_conf'),
                                           r.get('treval_conf_std'), 4)
        row['fin./att.'] = _counts_cell(r)
        row['Val. (tune)'] = C.fmt_pm(r['val_tune'], r['val_tune_std'], 4)
        row['tune fin./att.'] = C.fmt_counts(r['fin_tune'], r['att_tune'])
        if has_ds:
            row['Betw. inst.'] = C.fmt_sig(r.get('val_conf_dsspread'), 3)
            row['Optimism'] = C.fmt_pct(r.get('gap_val_same_instance_rel'), 1,
                                        signed=True)
        else:
            row['Optimism'] = C.fmt_pct(r.get('gap_val_rel'), 1, signed=True)
        rows.append(row)
    notes = [
        f'{hl.scan_title(scan)}: confirmation seeds, mean +/- 1 std over seeds, '
        f'sorted by validation loss -- the row order is the ranking',
        '"Optimism" is (confirmation - tuning)/tuning on the problem instance the '
        'tuning ran on; positive means the tuning seeds flattered the configuration',
    ]
    if has_ds:
        notes.append('"Between-instance" is the std of the three per-data-seed means: '
                     'variation of the random problem, not the seed band')
    rec = C.provenance(
        functions=['headline.confirmation_table', 'headline.confirmation_view'],
        scans=[hl.dir_name(scan, k) for k in ('', 'confirm')],
        note=f'T5 confirmation table for {scan}',
        provisional=sorted(ctx.provisional.intersection({scan})))
    return C.write_table(f'confirmation_{scan}', rows, provenance_record=rec, fit=True,
                         align=['l', 'p{1.15in}'] + ['r'] * 10, notes=notes,
                         caption=f'{hl.scan_title(scan)}: every method\'s selected '
                                 f'configuration on the confirmation seeds.',
                         label=f'tab:confirm{SCAN_KEY.get(scan, scan)}')


def _t5_paired(ctx, scan):
    """T5b: the paired per-seed differences against Sven, with 95% t-intervals."""
    tbl = ctx.paired(scan)
    rows = []
    for _, r in tbl.iterrows():
        rows.append({
            'Method': hl.display_name(r['method']),
            C.Raw('Sven $-$ method'): C.fmt_sig(r['mean'], 3),
            '95\\% interval': C.fmt_ci(r['ci_low'], r['ci_high'], 3),
            'Pairs': C.fmt_int(r['n']),
            'Unpaired': C.fmt_int(r.get('n_unpaired')),
            'Sven better': C.fmt_counts(r['sven_better'], r['n']),
            'Resolved': C.Raw('yes' if bool(r.get('significant')) else 'no'),
        })
    notes = [
        f'{hl.scan_title(scan)}: the difference is taken WITHIN a pair '
        f'({tbl.attrs.get("pair_on", "model_seed")}), i.e. at the same initialisation '
        f'and data order, so the seed-to-seed variation is removed',
        'negative means Sven has the lower validation loss; the interval is a Student-t '
        'interval on the mean difference, not the seed band',
        '"Unpaired" counts the seeds only one side finished -- a diverged run has no '
        'value and is not imputed',
    ]
    rec = C.provenance(
        functions=['headline.paired_vs_sven', 'paired.paired_table',
                   'paired.paired_difference'],
        scans=[hl.dir_name(scan, 'confirm')],
        note=f'T5 paired differences vs Sven for {scan}: {tbl.attrs.get("sign", "")}',
        provisional=sorted(ctx.provisional.intersection({scan})))
    return C.write_table(f'paired_{scan}', rows, provenance_record=rec, fit=True,
                         align=['l'] + ['r'] * 6, notes=notes,
                         caption=f'{hl.scan_title(scan)}: paired per-seed differences '
                                 f'against Sven.',
                         label=f'tab:paired{SCAN_KEY.get(scan, scan)}')


#: the grid axes T10 reports, in the order a row lists them
_GRID_AXES = ('lr', 'weight_decay', 'k', 'rtol', 'kappa', 'lbfgs_max_iter',
              'lbfgs_history_size', 'tau', 'polyak_max_lr', 'polyak_eps',
              'aggregator', 'inner_optimizer')
_GRID_LABEL = {'lr': r'$\eta$', 'weight_decay': 'wd', 'k': '$k$', 'rtol': 'rtol',
               'kappa': r'$\kappa$', 'lbfgs_max_iter': 'max iter',
               'lbfgs_history_size': 'history', 'tau': r'$\tau$',
               'polyak_max_lr': 'max lr', 'polyak_eps': 'eps',
               'aggregator': 'aggregator', 'inner_optimizer': 'inner'}


def _grid_rows(ctx, scan):
    """Per method: the grid actually swept and how many distinct configurations it has.

    Read off the tuning records rather than a plan file, so what the table prints is
    what ran.  ``style.config_key`` counts configurations (a run_id without its seed
    suffix), which is the same identity the selection used.
    """
    df = ctx.tune(scan)
    sels = hl.selection_methods(scan, ctx.payload)
    out = []
    for raw in hl.method_order(sels):
        m = hl.method_key(raw)
        sub = df[df['optimizer'].map(hl.method_key) == m]
        if not len(sub):
            continue
        chosen = hl.record_hparams(sels[raw]) or {}
        bits, edges = [], []
        for axis in _GRID_AXES:
            if axis not in sub.columns:
                continue
            vals = sub[axis].dropna().unique()
            if len(vals) < 2:
                continue
            try:
                order = sorted(float(v) for v in vals)
                shown = ', '.join(f'{v:g}' for v in order)
            except (TypeError, ValueError):
                order, shown = None, ', '.join(sorted(str(v) for v in vals))
            bits.append(f'{_GRID_LABEL.get(axis, axis)} $\\in$ {{{shown}}}')
            # App. C-continued promises "a flag where that configuration sits at the edge
            # of its own grid".  An edge selection is not a defect, but it bounds the
            # result -- CIFAR-CE's Sven selection is at the MAXIMUM of both its rtol and
            # its lr axis -- so the table says so instead of leaving the reader to
            # compare the "Selected" column against the "Grid swept" column by eye.
            if order is None or axis not in chosen or chosen[axis] is None:
                continue
            try:
                got = float(chosen[axis])
            except (TypeError, ValueError):
                continue
            if got == order[0]:
                edges.append((axis, 'min'))
            elif got == order[-1]:
                edges.append((axis, 'max'))
        n_cfg = len({style.config_key(r) for r in sub['run_id']})
        label = hl.config_label(sels[raw])
        out.append({'method': m, 'display': hl.display_name(m),
                    'grid': '; '.join(bits) or 'no swept hyperparameter',
                    'n_configs': n_cfg, 'n_runs': len(sub),
                    'selected': label,
                    'edges': edges,
                    'selected_marked': _mark_edges(label, edges)})
    return out


def _EDGE_MARK(where):
    return r'$^{\downarrow}$' if where == 'min' else r'$^{\uparrow}$'


def _mark_edges(label, edges):
    """``'k=128, lr=0.5, rtol=0.3'`` with an arrow on every value at an axis endpoint.

    ``$^\\uparrow$`` = the largest value that axis was swept over, ``$^\\downarrow$`` =
    the smallest.  Returned as LaTeX (the caller writes it with ``C.Raw``).
    """
    where = dict(edges)
    parts = []
    for part in str(label).split(', '):
        name, _, value = part.partition('=')
        cell = C.latex_escape(part)
        if name in where:
            cell = cell + _EDGE_MARK(where[name])
        parts.append(cell)
    return ', '.join(parts)


def _t10_grids(ctx, name='grids'):
    """T10: the hyperparameter grid per method per scan.

    Emitted as one compact ``methods x scans`` matrix of grid sizes (which fits on a page
    without ``longtable``, and no new package may be loaded) plus one per-scan table with
    the grid itself and the selected configuration.
    """
    written = []
    per_scan = {}
    for scan in ctx.scans:
        per_scan[scan] = _grid_rows(ctx, scan)
    methods = sorted({r['method'] for rows in per_scan.values() for r in rows},
                     key=lambda m: (m != hl.SVEN_LABEL, m))
    columns = ['Method'] + [TABLE_SCAN_SHORT.get(s, hl.scan_title(s))
                            for s in ctx.scans]
    rows = []
    for m in methods:
        row = {'Method': hl.display_name(m)}
        for scan in ctx.scans:
            hit = [r for r in per_scan[scan] if r['method'] == m]
            row[TABLE_SCAN_SHORT.get(scan, hl.scan_title(scan))] = (
                C.fmt_int(hit[0]['n_configs']) if hit else C.Raw('--'))
        rows.append(row)
    rec = C.provenance(
        functions=['headline.load_selection', 'headline.config_label',
                   'style.config_key'],
        scans=list(ctx.scans), note='T10 grid sizes (distinct configurations per method)',
        provisional=sorted(ctx.provisional))
    written.append(C.write_table(
        name, rows, columns=columns, provenance_record=rec, fit=True,
        align=['l'] + ['r'] * len(ctx.scans),
        notes=['the number of DISTINCT configurations of each method on each scan '
               '(a run_id without its seed suffix); a blank means the method was not '
               'run there',
               'Sven sweeps k x lr x rtol, most baselines a learning rate alone -- the '
               'budget disclosure in App. F reads these against best-of-n curves'],
        caption='Grid size per method per scan.', label='tab:gridsizes'))
    for scan in ctx.scans:
        rows = [{'Method': r['display'], 'Grid swept': C.Raw(r['grid']),
                 'Configs': C.fmt_int(r['n_configs']),
                 'Runs': C.fmt_int(r['n_runs']),
                 'Selected': C.Raw(r['selected_marked'])}
                for r in per_scan[scan]]
        n_edge = sum(1 for r in per_scan[scan] if r['edges'])
        rec = C.provenance(
            functions=['headline.load_selection', 'headline.config_label',
                       'headline.record_hparams', 'style.config_key'],
            scans=[scan], note=f'T10 grid for {scan}',
            edge_selections={r['method']: r['edges'] for r in per_scan[scan]
                             if r['edges']})
        written.append(C.write_table(
            f'grid_{scan}', rows, provenance_record=rec, fit=True,
            align=['l', 'p{2.2in}', 'r', 'r', 'p{1.1in}'],
            notes=[f'{hl.scan_title(scan)}: the grid as the records carry it, and the '
                   f'configuration the validation loss selected',
                   'a selected value carrying $\\uparrow$ ($\\downarrow$) is the largest '
                   '(smallest) value that axis was swept over, so the result is bounded '
                   f'by the grid on that axis; {n_edge} of {len(rows)} methods here have '
                   'at least one such axis',
                   'weight decay is at each optimizer\'s own default and is not swept '
                   'except where a value appears above'],
            caption=f'{hl.scan_title(scan)}: hyperparameter grids.',
            label=f'tab:grid{SCAN_KEY.get(scan, scan)}'))
    return written, per_scan


def _t18_reproducibility(ctx, name='reproducibility'):
    """T18: did the standalone timing pass reproduce the scan's trajectories?"""
    rows, detail = [], {}
    for scan in ctx.scans:
        tj = hl.timing_join_report(scan, payload=ctx.payload, results_root=ctx.root)
        if not len(tj):
            continue
        med = pd.to_numeric(tj['median_rel_dev'], errors='coerce')
        mx = pd.to_numeric(tj['max_rel_dev_raw'], errors='coerce')
        tol = tj.attrs.get('tol', 1e-3)
        moved = tj[~tj['bit_reproduced'].astype(bool)]
        worst = tj.loc[mx.idxmax()] if mx.notna().any() else None
        eff = ctx.eff(scan)
        gpus = sorted({str(g) for g in eff.get('gpu_timing', pd.Series()).dropna()})
        sven = tj[tj['method'] == hl.SVEN_LABEL]
        detail[scan] = {'n_methods': int(len(tj)),
                        'n_not_reproduced': int(len(moved)),
                        'median_of_medians': float(med.median()) if med.notna().any()
                        else None,
                        'max_rel_dev_raw': float(mx.max()) if mx.notna().any() else None,
                        'worst_method': (str(worst['method']) if worst is not None
                                         else None),
                        'sven_median_rel_dev': (float(sven.iloc[0]['median_rel_dev'])
                                                if len(sven) else None),
                        'sven_max_rel_dev': (float(sven.iloc[0]['max_rel_dev'])
                                             if len(sven) else None)}
        rows.append({
            'Task': TABLE_SCAN_SHORT.get(scan, hl.scan_title(scan)),
            'Methods': C.fmt_int(len(tj)),
            'Median dev.': C.fmt_sig(med.median(), 2),
            'Max dev.': C.fmt_sig(mx.max(), 2),
            'Worst': (hl.display_name(worst['method']) if worst is not None
                      else C.Raw('--')),
            C.Raw(f'Within ${tol:g}$'): C.fmt_counts(len(tj) - len(moved), len(tj)),
            'Sven median': C.fmt_sig(sven.iloc[0]['median_rel_dev'], 2) if len(sven)
            else C.Raw('--'),
            'Timing GPU': C.latex_escape((', '.join(gpus) if gpus else '--')
                                         .replace('NVIDIA ', '')),
        })
    notes = [
        'the standalone timing run of a configuration shares its run_id, seed, '
        'initialisation and data order with its scan twin; the deviation is in the final '
        'validation loss',
        'GPU kernels are not bit-exact across device types and the two passes partly ran '
        'on different ones, so selection-level results reproduce while seed-level ones '
        'need not -- Muon (bf16 Newton-Schulz), L-BFGS (line search), SOAP/Shampoo/HIG '
        'and every CIFAR run deviate',
        'Sven\'s own MNIST twin deviates too, so "Sven is bit-reproducible" is not '
        'claimed',
    ]
    calib = hl.calibration_report(scans=ctx.scans)
    if len(calib):
        notes.append(f'the fixed-step calibration probe drifted at most '
                     f'{calib.attrs.get("max_abs_drift", float("nan")):.1%} between the '
                     f'start and the end of a timing pass')
    rec = C.provenance(
        functions=['headline.timing_join_report', 'headline_figs.timing_join_view',
                   'headline.efficiency_table', 'headline.calibration_report'],
        scans=[hl.dir_name(s, k) for s in ctx.scans for k in ('', 'timing')],
        note='T18, the reproducibility table for App. R',
        detail=detail, provisional=sorted(ctx.provisional))
    return C.write_table(name, rows, provenance_record=rec, fit=True,
                         align=['l'] + ['r'] * 6 + ['l'], notes=notes,
                         caption='Scan against its standalone timing twin.',
                         label='tab:repro'), detail


def _t20_data_seeds(ctx, name='data_seeds'):
    """T20: between-instance against within-instance spread on the two synthetics."""
    rows, detail, midrules = [], {}, []
    for scan in ('toy_1d_scan', 'polynomial_scan'):
        spread = hf.instance_spread(scan, table=ctx.conf(scan))
        if spread.empty:
            continue
        if rows:                        # rule off every block but the first
            midrules.append(len(rows))
        rows.append(C.span_row(hl.scan_title(scan)))
        for _, r in spread.iterrows():
            rows.append({
                'Method': hl.display_name(r['method']),
                'Val. (pooled)': C.fmt_sig(r['val_conf'], 4),
                'Val. (inst. mean)': C.fmt_sig(r['val_conf_dsmean'], 4),
                'Betw.-inst. std': C.fmt_sig(r['val_conf_dsspread'], 3),
                'Within-inst. std': C.fmt_sig(r['val_conf_seedspread'], 3),
                'Ratio': C.fmt_sig(r['ratio_instance_over_seed'], 3),
                # the tuning-instance value is read against the tuning seeds, which
                # this table does not carry: it lives in the optimism table instead
            })
        sven = spread[spread['method'] == hl.SVEN_LABEL]
        detail[scan] = {
            'data_seeds': [int(s) for s in spread.attrs.get('data_seeds', [])],
            'median_ratio': float(pd.to_numeric(spread['ratio_instance_over_seed'],
                                                errors='coerce').median()),
            'sven_ratio': (float(sven.iloc[0]['ratio_instance_over_seed'])
                           if len(sven) else None)}
    notes = [
        'each data seed is a DIFFERENT random problem, so the std of the three '
        'per-instance means is variation of the target, not of the optimizer\'s luck',
        'a ratio above 1 says the choice of instance moves the answer more than the seed '
        'does, and no single-instance number should then be quoted without it',
    ]
    rec = C.provenance(
        functions=['headline.data_seed_table', 'headline_figs.instance_spread',
                   'headline.confirmation_table'],
        scans=[hl.dir_name(s, 'confirm') for s in ('toy_1d_scan', 'polynomial_scan')],
        note='T20, the data-seed replicate table', detail=detail)
    return C.write_table(name, rows, provenance_record=rec, fit=True,
                         align=['l'] + ['r'] * 5, midrules=midrules, notes=notes,
                         caption='Data-seed replicates on the two synthetic tasks.',
                         label='tab:dataseeds'), detail


def _t6_optimism(ctx, name='optimism'):
    """The selection-optimism table (C15).  ``paper_assets.reviewer`` owns App. F's
    budget tables; this one is built here because it is a pure
    :func:`headline.selection_optimism_table` view of the confirmation tables this
    module already holds, and the integrator needs it for the protocol discussion."""
    tbl = hl.selection_optimism_table(tables={s: ctx.conf(s) for s in ctx.scans},
                                      scans=ctx.scans, payload=ctx.payload,
                                      results_root=ctx.root)
    rows = []
    for _, r in tbl.iterrows():
        rows.append({
            'Task': C.latex_escape(r['scan']),
            'Methods': C.fmt_int(r['n_methods']),
            'Data seeds': C.fmt_int(r['n_data_seeds']),
            'Median (inst.)': C.fmt_pct_value(r['median_same_%'], 1, True),
            'Sven (inst.)': C.fmt_pct_value(r['sven_same_%'], 1, True),
            'Worse': C.fmt_counts(r['n_worse_same'], r['n_methods']),
            'Median (pool)': C.fmt_pct_value(r['median_pooled_%'], 1, True),
            'Sven (pool)': C.fmt_pct_value(r['sven_pooled_%'], 1, True),
        })
    notes = [
        'the tuning-instance column IS selection optimism: fresh model seeds, same data, '
        'same everything else. Positive means the tuning seeds flattered the selection',
        'the pooled column also carries the two problem instances the tuning never saw '
        'on the synthetics, where it has a different magnitude and (for 1D) the opposite '
        'sign; on the five scans without data seeds the two columns are identical',
    ]
    rec = C.provenance(
        functions=['headline.selection_optimism_table', 'headline.confirmation_table'],
        scans=[hl.dir_name(s, k) for s in ctx.scans for k in ('', 'confirm')],
        note='selection optimism (C15)', provisional=sorted(ctx.provisional))
    return C.write_table(name, rows, provenance_record=rec, fit=True,
                         align=['l'] + ['r'] * 7, notes=notes,
                         caption='Selection optimism: confirmation against tuning.',
                         label='tab:optimism'), tbl


# ---------------------------------------------------------------------------
# G1 -- the macro group
# ---------------------------------------------------------------------------
def _macros(ctx, figure_info=None):
    """Every number the abstract / intro / results / conclusion quotes."""
    M = C.Macros(module='main')
    figure_info = figure_info or {}

    def key(scan):
        return SCAN_KEY[scan]

    # --- field sizes and the protocol -----------------------------------
    M.add('numNMethodsMlp', len(hl.selection_methods('polynomial_scan', ctx.payload)),
          source='headline.selection_methods')
    # The prose enumerates the baselines, so the count it is checked against must be the
    # baseline count and not the field size (which includes Sven).
    M.add('numNBaselinesMlp',
          len(hl.selection_methods('polynomial_scan', ctx.payload)) - 1,
          source='headline.selection_methods minus Sven')
    M.add('numNMethodsMnist',
          len(hl.selection_methods('mnist_scan_ce', ctx.payload)),
          source='headline.selection_methods')
    M.add('numNMethodsNanogpt',
          len(hl.selection_methods('exp_nanogpt_speedrun', ctx.payload)),
          source='headline.selection_methods')
    M.add('numNSelections',
          hl.assert_no_test_selection(payload=ctx.payload, scans=ctx.scans),
          source='headline.assert_no_test_selection')
    M.add('numNScans', len(ctx.scans), source='headline.HEADLINE_SCANS')

    # --- the ranking figure's caption (F14) ------------------------------
    # The caption states how far the validation and test rankings agree, which is what
    # licenses reading the two panels as one result; it is a count over the same frame
    # the figure draws, not a number typed into the caption.
    rk = ctx.ranking.dropna(subset=['rank_val', 'rank_test'])
    agree = rk['rank_val'] == rk['rank_test']
    gap = (rk['rank_val'] - rk['rank_test']).abs()
    M.add('numRankCells', int(len(rk)), source='headline.ranking_summary')
    M.add('numRankCellsAgree', int(agree.sum()),
          source='headline.ranking_summary (rank_val == rank_test)')
    M.add('numRankCellsDiffer', int((~agree).sum()),
          source='headline.ranking_summary (rank_val != rank_test)')
    M.add('numRankCellsAdjacent', int(((~agree) & (gap == 1)).sum()),
          source='headline.ranking_summary (|rank_val - rank_test| == 1)')
    M.add('numRankCellsWide', int(((~agree) & (gap > 1)).sum()),
          source='headline.ranking_summary (|rank_val - rank_test| > 1)')
    M.add('numRankCellsDifferNanogpt',
          int((~agree & (rk['scan'] == 'exp_nanogpt_speedrun')).sum()),
          source='headline.ranking_summary (nanoGPT disagreements)')
    # --- the extent of F3 (fig:cost_memory), so its caption cannot drift --------------
    # The caption used to claim the figure spans up to "163 million" parameters, which is
    # GPT-2-small: that model has no standalone timing pass (one epoch, one seed, no
    # companion passes), so it is not in F3 at all.  The endpoints are now read off the
    # scans the figure actually draws.
    f3_params = {s: _scalar(ctx, s, 'n_params') for s in F3_SCANS}
    f3_finite = {s: float(p) for s, p in f3_params.items() if C.finite(p)}
    if f3_finite:
        lo_scan = min(f3_finite, key=f3_finite.get)
        hi_scan = max(f3_finite, key=f3_finite.get)
        M.add('numCostMemoryNArchs', len(f3_finite),
              source='paper_assets.main.F3_SCANS',
              note='model scales drawn in fig:cost_memory')
        # Thousands-separated integers, not scientific notation: these two are quoted in
        # a figure caption as running prose ("spanning 593 to 11,181,642 parameters").
        M.add('numCostMemoryPMin', C.Raw(f'{int(f3_finite[lo_scan]):,}'.replace(',', '{,}')),
              source='records n_params',
              note=f'smallest parameter count drawn in fig:cost_memory ({lo_scan})')
        M.add('numCostMemoryPMax', C.Raw(f'{int(f3_finite[hi_scan]):,}'.replace(',', '{,}')),
              source='records n_params',
              note=f'largest parameter count drawn in fig:cost_memory ({hi_scan})')
        M.add('numCostMemoryPMaxArch',
              C.latex_escape(ARCH_NAME.get(hi_scan, hi_scan)),
              source='paper_assets.main.ARCH_NAME',
              note='the architecture at the right-hand end of fig:cost_memory')
    M.add('numNTuningSeeds', 5, source='EXPERIMENTS.md protocol (records: 5 model seeds)')
    M.add('numNConfirmSeeds', 5, source='headline.load(scan, "confirm") model seeds')

    # --- per-scan blocks -------------------------------------------------
    for scan in MACRO_SCANS:
        s, prov = key(scan), ctx.is_provisional(scan)
        conf, eff = ctx.conf(scan), ctx.eff(scan)
        n_meth = int(ctx.ranking[ctx.ranking['scan'] == scan]['n_methods'].max())
        M.add(f'num{s}NMethods', n_meth, source='headline.ranking_summary',
              provisional=prov)
        M.add(f'num{s}Params', _scalar(ctx, scan, 'n_params'), sig=9,
              source='records n_params')
        M.add(f'num{s}Batch', _scalar(ctx, scan, 'batch_size'), source='records batch_size')
        M.add(f'num{s}Epochs', _n_epochs(ctx, scan), source='records num_epochs')
        best = conf.iloc[0]
        M.add(f'num{s}BestMethod', C.latex_escape(hl.display_name(best['method'])),
              source='headline.confirmation_table (row 0)', provisional=prov)
        M.add_pm(f'num{s}BestVal', best['val_conf'], best['val_conf_std'], sig=4,
                 source='headline.confirmation_table', provisional=prov)
        med = float(pd.to_numeric(conf['val_conf'], errors='coerce').median())
        M.add(f'num{s}MedianVal', med, sig=4,
              source='headline.target_reference (median method)', provisional=prov)
        eff_by = {r['method']: r for _, r in eff.iterrows()}
        adam_ms = eff_by.get('Adam', {}).get('ms_per_step') if 'Adam' in eff_by else None
        for m in MACRO_METHODS:
            mk = METHOD_KEY.get(m, m)
            r = ctx.row(scan, m)
            if r is None:
                continue
            M.add_pm(f'num{s}{mk}Val', r['val_conf'], r['val_conf_std'], sig=4,
                     source='headline.confirmation_table', provisional=prov)
            if C.finite(r.get('test_conf')):
                M.add(f'num{s}{mk}Test', r['test_conf'], sig=4,
                      source='headline.confirmation_table', provisional=prov)
            if C.finite(r.get('acc_conf')):
                M.add(f'num{s}{mk}Acc', 100 * float(r['acc_conf']), sig=3,
                      source='headline.confirmation_table (percent)', provisional=prov)
            rank, n = ctx.rank(scan, m)
            if C.finite(rank):
                M.add(f'num{s}{mk}Rank', rank, source='headline.ranking_summary',
                      provisional=prov)
            M.add(f'num{s}{mk}Fin', r['fin_conf'], source='headline.confirmation_table',
                  provisional=prov)
            M.add(f'num{s}{mk}Att', r['att_conf'], source='headline.confirmation_table',
                  provisional=prov)
            e = eff_by.get(m) if m in MACRO_COST_METHODS else None
            if e is not None:
                for col, q, sig in (('ms_per_step', 'MsStep', 3),
                                    ('peak_gpu_mem_mb', 'MemMb', 4),
                                    ('wall_s', 'WallS', 3),
                                    ('epoch_s', 'EpochS', 3)):
                    if C.finite(e[col]):
                        M.add(f'num{s}{mk}{q}', e[col], sig=sig,
                              source='headline.efficiency_table (standalone timing pass)',
                              provisional=prov)
                if m != 'Adam' and C.finite(adam_ms) and C.finite(e['ms_per_step']):
                    M.add(f'num{s}{mk}MsStepVsAdam', e['ms_per_step'] / float(adam_ms),
                          sig=3, source='headline.efficiency_table (ratio)',
                          provisional=prov)
            t = ctx.target_row(scan, m) if m in MACRO_COST_METHODS else None
            if t is not None and C.finite(t['epochs']):
                M.add(f'num{s}{mk}TargetEpochs', t['epochs'], sig=3,
                      source='headline.time_to_target_table (median-method target)',
                      provisional=prov)
                if C.finite(t['standalone_s']):
                    M.add(f'num{s}{mk}TargetSecs', t['standalone_s'], sig=3,
                          source='headline.time_to_target_table (standalone seconds)',
                          provisional=prov)
                M.add(f'num{s}{mk}TargetReached', t['n_reached'],
                      source='headline.time_to_target_table', provisional=prov)
                M.add(f'num{s}{mk}TargetRuns', t['n_runs'],
                      source='headline.time_to_target_table', provisional=prov)
        # C8's caveat: how many baseline configurations are cheaper per step than Sven's
        ms = pd.to_numeric(eff['ms_per_step'], errors='coerce')
        sven_ms = ms[eff['method'] == hl.SVEN_LABEL]
        if len(sven_ms) and C.finite(sven_ms.iloc[0]):
            other = ms[eff['method'] != hl.SVEN_LABEL].dropna()
            M.add(f'num{s}NCheaperThanSven', int((other < sven_ms.iloc[0]).sum()),
                  source='headline.efficiency_table (ms_per_step)', provisional=prov)
            M.add(f'num{s}NDearerThanSven', int((other > sven_ms.iloc[0]).sum()),
                  source='headline.efficiency_table (ms_per_step)', provisional=prov)
        # target value itself
        tt = ctx.ttt(scan)
        tv = tt[tt['target'] == MAIN_TARGET]['target_value']
        if len(tv):
            M.add(f'num{s}Target', float(tv.iloc[0]), sig=4,
                  source='headline.targets_for (median method)', provisional=prov)
        # --- paired differences against Sven ----------------------------
        pv = ctx.paired(scan)
        if len(pv):
            n_worse = int((pv['mean'] < 0).sum())
            M.add(f'num{s}NWorseThanSven', n_worse, source='headline.paired_vs_sven',
                  provisional=prov)
            M.add(f'num{s}NResolvedWorse',
                  int(((pv['mean'] < 0) & pv['significant']).sum()),
                  source='headline.paired_vs_sven (95% t-interval)', provisional=prov)
            M.add(f'num{s}NPairedMethods', len(pv), source='headline.paired_vs_sven',
                  provisional=prov)
            # --- the five first-order baselines, separately ------------------
            # The Introduction and the Conclusion claim a win over exactly these five.
            # The claim holds on the seed means everywhere; the RESOLUTION does not, so
            # the count is generated rather than asserted (on MNIST label regression
            # only RMSprop's margin clears the interval).
            fo = pv[pv['method'].isin(FIRST_ORDER_METHODS)]
            if len(fo):
                M.add(f'num{s}NFirstOrder', len(fo),
                      source='headline.paired_vs_sven (FIRST_ORDER_METHODS)',
                      provisional=prov)
                M.add(f'num{s}NFirstOrderWorse', int((fo['mean'] < 0).sum()),
                      source='headline.paired_vs_sven', provisional=prov)
                M.add(f'num{s}NFirstOrderResolved',
                      int(((fo['mean'] < 0) & fo['significant']).sum()),
                      source='headline.paired_vs_sven (95% t-interval)',
                      provisional=prov)
            # --- who is ahead of Sven, and whether the lead is resolved ------
            # Section 4.1 used to name one leader where two or three methods are ahead.
            ahead = pv[pv['mean'] > 0].sort_values('mean', ascending=False)
            if len(ahead):
                M.add(f'num{s}AheadList',
                      C.Raw(_name_list(ahead['display'])),
                      source='headline.paired_vs_sven (paired mean > 0)',
                      provisional=prov,
                      note='every method with a lower paired mean loss than Sven')
                M.add(f'num{s}NAhead', len(ahead),
                      source='headline.paired_vs_sven', provisional=prov)
                res = ahead[ahead['significant'].astype(bool)]
                unres = ahead[~ahead['significant'].astype(bool)]
                if len(res):
                    M.add(f'num{s}AheadResolvedList', C.Raw(_name_list(res['display'])),
                          source='headline.paired_vs_sven', provisional=prov,
                          note='methods ahead of Sven by a resolved paired margin')
                M.add(f'num{s}NAheadResolved', len(res),
                      source='headline.paired_vs_sven', provisional=prov)
                if len(unres):
                    M.add(f'num{s}AheadUnresolvedList',
                          C.Raw(_name_list(unres['display'])),
                          source='headline.paired_vs_sven', provisional=prov,
                          note='methods ahead of Sven within seed noise (interval '
                               'covers zero)')
                M.add(f'num{s}NAheadUnresolved', len(unres),
                      source='headline.paired_vs_sven', provisional=prov)
            leaders = [conf.iloc[0]['method']]
            for m in list(MACRO_PAIRED_METHODS) + [x for x in leaders
                                                   if x not in MACRO_PAIRED_METHODS
                                                   and x != hl.SVEN_LABEL]:
                sub = pv[pv['method'] == m]
                if not len(sub):
                    continue
                mk = METHOD_KEY.get(m, m)
                r = sub.iloc[0]
                M.add(f'num{s}Paired{mk}', r['mean'], sig=3,
                      source='headline.paired_vs_sven (Sven - method)', provisional=prov)
                M.add(f'num{s}Paired{mk}CiLo', r['ci_low'], sig=3,
                      source='headline.paired_vs_sven', provisional=prov)
                M.add(f'num{s}Paired{mk}CiHi', r['ci_high'], sig=3,
                      source='headline.paired_vs_sven', provisional=prov)
                M.add(f'num{s}Paired{mk}Pairs', r['n'],
                      source='headline.paired_vs_sven', provisional=prov)
                M.add(f'num{s}Paired{mk}SvenBetter', r['sven_better'],
                      source='headline.paired_vs_sven', provisional=prov)
        # --- selection optimism -----------------------------------------
        sven = conf[conf['method'] == hl.SVEN_LABEL]
        if len(sven) and C.finite(sven.iloc[0].get('gap_val_same_instance_rel')):
            M.add(f'num{s}SvenOptimism',
                  C.fmt_pct(sven.iloc[0]['gap_val_same_instance_rel'], 1, signed=True),
                  source='headline.confirmation_table (gap_val_same_instance_rel)',
                  provisional=prov)
        # --- Sven's selected learning rate ------------------------------
        # `num<Scan>SvenK` and `num<Scan>SvenRtol` belong to G4: PAPER_PLAN's T8 is
        # "per scan: selected k, rtol, which cut binds, ...", so the two rank knobs are
        # defined in numbers_v2_spectra.tex and are not repeated here.  The learning
        # rate is not part of that table, so it stays with the headline numbers.
        chosen, _sel = hf.selected_sven(scan, payload=ctx.payload)
        if chosen.get('lr') is not None:
            M.add(f'num{s}SvenLr', float(chosen['lr']), sig=3,
                  source='bench/best_configs.json via headline_figs.selected_sven',
                  provisional=prov)

    # --- the perplexity the LM section quotes ----------------------------
    # nanoGPT's own levels, perplexities and costs are G3's (see MACRO_DEFERRED): they
    # are defined in numbers_v2_large.tex and are NOT repeated here.

    # --- cross-scan statements the abstract makes ------------------------
    sven_ranks = {s: ctx.rank(s, hl.SVEN_LABEL)[0] for s in ctx.scans}
    finite_ranks = [v for v in sven_ranks.values() if C.finite(v)]
    M.add('numSvenBestRank', min(finite_ranks), source='headline.ranking_summary')
    M.add('numSvenWorstRank', max(finite_ranks), source='headline.ranking_summary')
    M.add('numSvenNeverFirst', int(sum(1 for v in finite_ranks if v == 1)),
          source='headline.ranking_summary (count of scans where Sven is first)')
    # the step-time ratio the abstract quotes spans every MLP and LM scan, nanoGPT
    # included: it is a cross-scan aggregate, so it does not collide with G3's
    # per-nanoGPT macros (MACRO_DEFERRED)
    ms_ratios = []
    for scan in STEP_RATIO_SCANS:
        eff = ctx.eff(scan)
        by = {r['method']: r['ms_per_step'] for _, r in eff.iterrows()}
        if C.finite(by.get(hl.SVEN_LABEL)) and C.finite(by.get('Adam')):
            ms_ratios.append(by[hl.SVEN_LABEL] / by['Adam'])
        elif C.finite(by.get(hl.SVEN_LABEL)) and C.finite(by.get('AdamW')):
            ms_ratios.append(by[hl.SVEN_LABEL] / by['AdamW'])
    if ms_ratios:
        M.add('numSvenStepVsAdamMin', min(ms_ratios), sig=3,
              source='headline.efficiency_table (ms_per_step ratio over the MLP/LM scans)')
        M.add('numSvenStepVsAdamMax', max(ms_ratios), sig=3,
              source='headline.efficiency_table (ms_per_step ratio over the MLP/LM scans)')

    # --- the two failures the related-work paragraph names ---------------
    for scan, label in (('mnist_scan_ce', 'MnistCE'), ('mnist_scan_labelRegression',
                                                       'MnistLR')):
        df = ctx.tune(scan)
        sub = df[df['optimizer'].map(hl.method_key) == 'KFAC']
        if len(sub):
            bad = int((sub['diverged'] | sub['failed']).sum())
            M.add(f'num{label}KfacFailed', bad,
                  source='headline.load(scan) diverged|failed rows, optimizer=KFAC')
            M.add(f'num{label}KfacRuns', len(sub),
                  source='headline.load(scan) rows, optimizer=KFAC')
    df = ctx.tune('polynomial_scan')
    sub = df[df['optimizer'].map(hl.method_key) == 'LBFGS']
    if len(sub):
        M.add('numPolyLbfgsDiverged', int((sub['diverged'] | sub['failed']).sum()),
              source='headline.load(polynomial_scan) diverged|failed, optimizer=LBFGS')
        M.add('numPolyLbfgsRuns', len(sub),
              source='headline.load(polynomial_scan) rows, optimizer=LBFGS')
    # Fig. 1's caption used to explain L-BFGS's absence from the polynomial panel with
    # the TUNING-grid divergence rate above, which reads as if the method could not be
    # run.  It has a selected configuration and confirmation runs; it is simply not in
    # the five drawn.  These three macros are the honest reason.
    conf_poly = ctx.conf('polynomial_scan')
    lb = conf_poly[conf_poly['method'].map(hl.method_key) == 'LBFGS']
    if len(lb):
        rank, _n = ctx.rank('polynomial_scan', 'LBFGS')
        src = 'headline.confirmation_table'
        if C.finite(rank):
            M.add('numPolyLbfgsRank', rank, source='headline.ranking_summary')
        M.add('numPolyLbfgsFin', lb.iloc[0]['fin_conf'], source=src)
        M.add('numPolyLbfgsAtt', lb.iloc[0]['att_conf'], source=src)

    # --- the k / rtol saturation the corrected sweep gives ---------------
    for scan, rec in (figure_info.get('hparam_landscape') or {}).items():
        s = key(scan)
        if C.finite(rec.get('k_within_5pct')):
            M.add(f'num{s}SvenKSaturate', rec['k_within_5pct'],
                  source='paper_assets.main hparam_landscape (smallest k within 5 per '
                         'cent of the best score on the selected rtol line)')
    return M


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------
def build(root=None, figures=True, tables=True, numbers=True, dry_run=False):
    """Build every asset this module owns.  Returns a report dict."""
    C.banner(GROUP, f'figures={figures} tables={tables} numbers={numbers}')
    ctx = context(root=root)
    report = {'figures': [], 'tables': [], 'n_macros': 0,
              'provisional': sorted(ctx.provisional), 'status': '',
              'macros_deferred_to': dict(MACRO_DEFERRED),
              'macro_families_deferred': dict(MACRO_FAMILIES_DEFERRED)}
    if ctx.provisional:
        report['status'] = f'PROVISIONAL: {sorted(ctx.provisional)}'
    if dry_run:
        report['status'] = (report['status'] + ' (dry run)').strip()
        report['would_write'] = {
            'figures': list(FIGURE_SPECS),
            'tables': ['headline_confirm', 'headline_confirm_compact',
                       'time_to_target', 'time_to_target_main', 'protocol', 'ranking',
                       'optimism', 'grids', 'reproducibility', 'data_seeds']
            + [f'confirmation_{s}' for s in ctx.scans]
            + [f'paired_{s}' for s in ctx.scans]
            + [f'grid_{s}' for s in ctx.scans]}
        return report

    figure_info = {}
    if figures:
        # one pass over the registry, in declaration order: the figures come out in the
        # order they always have, and a figure added to FIGURE_SPECS is built here
        # without touching build()
        for name in FIGURE_SPECS:
            pdf, _png, meta = _draw_and_save(name, ctx)
            report['figures'].append(str(pdf))
            if name in _FIGURE_INFO:
                figure_info[name] = _FIGURE_INFO[name](meta)

    if tables:
        report['tables'].append(str(_t1_headline(ctx)))
        report['tables'].append(str(_t1_headline(ctx, 'headline_confirm_compact',
                                                 compact=True)))
        report['tables'].append(str(_t2_time_to_target(ctx, 'time_to_target')))
        report['tables'].append(str(_t2_time_to_target(ctx, 'time_to_target_main',
                                                       compact=True)))
        report['tables'].append(str(_t3_protocol(ctx)))
        report['tables'].append(str(_t4_ranking(ctx)))
        path, _ = _t6_optimism(ctx)
        report['tables'].append(str(path))
        for scan in ctx.scans:
            report['tables'].append(str(_t5_confirmation(ctx, scan)))
            report['tables'].append(str(_t5_paired(ctx, scan)))
        grids, _ = _t10_grids(ctx)
        report['tables'].extend(str(p) for p in grids)
        (path, detail) = _t18_reproducibility(ctx)
        report['tables'].append(str(path))
        report['reproducibility'] = detail
        (path, detail) = _t20_data_seeds(ctx)
        report['tables'].append(str(path))

    if numbers:
        M = _macros(ctx, figure_info)
        path = M.write(provenance=C.provenance(
            functions=['headline.confirmation_table', 'headline.efficiency_table',
                       'headline.ranking_summary', 'headline.paired_vs_sven',
                       'headline.time_to_target_table',
                       'headline.assert_no_test_selection',
                       'headline_figs.selected_sven'],
            scans=[hl.dir_name(s, k) for s in ctx.scans
                   for k in ('', 'confirm', 'timing')],
            note='G1: every number the abstract / intro / results / conclusion quotes',
            provisional=sorted(ctx.provisional)))
        report['n_macros'] = len(M)
        report['macros'] = str(path)
        report['macro_names'] = sorted(M.values)
    return report


if __name__ == '__main__':
    import json as _json
    print(_json.dumps(build(), indent=2, default=str)[:4000])
