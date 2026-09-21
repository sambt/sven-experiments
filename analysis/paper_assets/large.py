"""Paper assets for the LARGE models and the cost profile (``PAPER_PLAN`` section 5, module
``large``): F7, F11, F15 - T7, T15, T16, T17, T19 - macro group G3.

Scope, and where each number comes from:

===========  =========================================================================
CIFAR-10     both objectives (label regression = one-hot MSE, and cross-entropy),
/ ResNet18   11 methods: curves vs epoch (confirmation seeds) and vs synchronised
             training time (the standalone ``{scan}_timing`` pass, which re-runs the
             TUNING seeds), validation / test accuracy, the train-vs-validation gap,
             the results table, Sven's ``(lr, rtol)`` landscape including the ``rtol``
             extension, the rank actually inverted, and the cost bars.  App. J.
Fig 5        the parameter-fraction study at ResNet scale: quality and cost against
             the ACTUAL parameter fraction, every Sven configuration that is on disk,
             with the one the headline scan SELECTED marked.  App. K.
nanoGPT      the confirmation table with the paired difference against Sven, the
             learning-rate sensitivity and the cost.  App. L.
GPT-2 small  one seed, one epoch: validation / test loss vs optimisation step for the
             best lr per method and for all five Sven lrs, the per-lr table, wall
             clock and peak memory.  Framed as a single-seed negative result.  App. L.
profile v3   step time and peak memory per method and architecture, the flat step
             time in ``k``, the batch-size / width scaling, the capture-vs-solve
             split, and the chunking Pareto front.  App. D and Q.
===========  =========================================================================

Three rules this module follows, because three of its inputs were still moving while the
paper was written (``PAPER_PLAN`` section 7):

1. **Nothing is cached and nothing is typed.**  Every figure, cell and macro is computed
   from :mod:`headline`, :mod:`large_figs` and :mod:`profile_helpers` at build time, and
   the Sven configuration of every scan is read from ``bench/best_configs.json`` through
   :func:`headline.selection_methods`.  Re-running ``python -m paper_assets.large`` after
   ``tools/select_best.py`` moves every affected asset by itself -- the CIFAR-CE
   re-selection (``lr`` 0.1 -> 0.5, ``rtol`` 0.01 -> 0.3) needed no edit here.
2. **Fig 5 shows what is on disk.**  :func:`large_figs.paramfrac_groups` splits the scan
   by ``(k, lr, rtol)`` and marks which group is the selected configuration, so the
   figure and the table grow as the re-run lands instead of describing a configuration
   that is no longer the pick.  Points that rest on fewer seeds than the scan intended
   are drawn hollow and carry their ``finished/attempted``.
3. **One profile root per table.**  :data:`PROFILE_PREFERENCE` prefers
   ``profile_results_v3`` -- the pass measured with Sven's per-step
   ``torch.cuda.empty_cache()`` OFF, which is the measurement of record -- and falls back
   to v2 only if v3 holds nothing at all.  A partial v3 is USED and stamped
   ``PROVISIONAL`` rather than silently replaced by the contaminated v2 numbers, and the
   two roots are never mixed inside one table.  Whether a root covers the ResNet is a
   property of the root, not an assumption: ``profile_info['has_cifar']`` is read at build
   time, every profile table carries :func:`Inputs.profile_note` as a note, and when the
   ResNet is missing the cost point falls back to :func:`headline.efficiency_table`
   (measured in training, 5 standalone runs).  As of the completed v3 (720/720) the
   ResNet IS profiled, so App. D and Q quote v3 for all five architectures.

The figures follow the module contract of :mod:`paper_assets.figspec`
(``campaign/FIGURE_API_CONTRACT.md``): :data:`FIGURE_SPECS` maps the PDF stem to a
:class:`figspec.FigureSpec` whose ``draw(ctx, opts)`` returns ``(fig, meta)`` and writes
nothing, :func:`context` is the :class:`Inputs` those draw functions take, and
:func:`build` is the one save path.  The cosmetics that used to be hard-coded in a
builder -- which methods, scans and architectures are drawn, the panel geometry, the
shared legend, line weights, the seed-shortfall note and the headroom -- are the
figure's ``defaults``, so they can be pinned from ``analysis/notebooks/paper/`` without
editing anything here; the defaults ARE what the paper was built with.

Conventions are inherited, not re-invented: colours :data:`style.METHOD_COLORS` (Sven
black), display names :func:`style.method_label`, seed band = mean +/- 1 std over seeds
(:data:`paired.SEED_SPREAD_LABEL`) with the lower edge clipped, ``finished/attempted`` on
every table row, and page-size figures / macro validation / booktabs from
:mod:`paper_assets.common`.
"""
from __future__ import annotations

import contextlib
import json
import os
from functools import cached_property
from pathlib import Path

import matplotlib
matplotlib.use('Agg', force=False)
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

from . import common as C                # noqa: E402
from . import figspec                    # noqa: E402

import analysis_helpers as ah            # noqa: E402
import headline as hl                    # noqa: E402
import large_figs as lf                  # noqa: E402
import paired                            # noqa: E402
import profile_helpers as ph             # noqa: E402
import reviewer_figs as rf               # noqa: E402
import scan_analysis as sa               # noqa: E402
import style                             # noqa: E402

#: figures of this module live in ``iclr_manuscript/figures_iclr/large/``
GROUP = 'large'

#: the two CIFAR headline scans, in report order, with the title the paper uses
CIFAR = {'cifar10_resnet_scan_labelRegression': style.DATASET_TITLES['cifar_labelreg'],
         'cifar10_resnet_ce_scan': style.DATASET_TITLES['cifar_ce']}
NANOGPT = 'exp_nanogpt_speedrun'
GPT2 = lf.GPT2_SCAN
FIG5 = lf.FIG5_SCAN

#: macro scan keys (``PAPER_PLAN`` section 5.3), per scan directory
SCAN_KEY = {'cifar10_resnet_scan_labelRegression': 'CifarLR',
            'cifar10_resnet_ce_scan': 'CifarCE',
            NANOGPT: 'Nanogpt', GPT2: 'GptTwo', FIG5: 'FigFive'}

#: CIFAR-10 has ten classes, so a uniform guess scores 0.1.  This is the DATASET's
#: definition (not a measurement), and it is the reference line on Fig 5's accuracy panel
#: -- below it a configuration has not learned anything.
CIFAR_CHANCE_ACC = 1.0 / 10.0

#: Which profile root a cost number comes from.  v3 is the measurement of record: v2 ran
#: Sven with a per-step ``torch.cuda.empty_cache()`` (up to 4.5x slower for the full
#: capture) and without ``expandable_segments``, so its Sven step times are not the cost
#: of the algorithm.  ``profile_helpers.results_root`` stays on v2 until v3 is COMPLETE,
#: which is the right default for an exploratory notebook and the wrong one for the paper:
#: here a PARTIAL v3 is preferred and stamped, because a contaminated number is worse than
#: a provisional one.  ``$SV3_PROFILE_ROOT`` still wins over both.
PROFILE_PREFERENCE = (ph.ROOT_V3, ph.ROOT_V2)

#: Sven backends in report order, and the macro key of each.
SVEN_BACKENDS = {'gram_hooks': 'Hooks', 'gram_full': 'Full',
                 'gram_chunked': 'Chunked', 'classic': 'Classic'}

#: architecture -> macro key, for the profile macros
ARCH_KEY = {'toy_1d': 'Toy', 'polynomial': 'Poly', 'mnist': 'Mnist', 'nanogpt': 'Nanogpt',
            'mnist_width': 'MnistWidth', 'nanogpt_width': 'NanogptWidth',
            'cifar_resnet18': 'Cifar'}

#: the four curve quantities a CIFAR figure shows, and the panel title of each
CIFAR_PANELS = (('val', 'Validation loss'), ('test_acc', 'Test accuracy'))

#: a sweep panel stays legible only with a few lines on it; these are the ones a cost
#: claim in the paper is about
SWEEP_METHODS = list(SVEN_BACKENDS) + ['Adam', 'SGD']

#: architectures the batch-size sweep figure shows.  ``polynomial`` is left out on
#: purpose: it is the same 2-layer MLP family as ``toy_1d`` (673 against 593 parameters)
#: and its curve lies on top of it, so the fifth panel would cost every other panel a
#: fifth of its width for no new information.  Its numbers are in T7 and the macros.
SWEEP_ARCHS = ('toy_1d', 'mnist', 'nanogpt', 'cifar_resnet18')


# ---------------------------------------------------------------------------
# Little helpers
# ---------------------------------------------------------------------------
_UNITS = ('Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine',
          'Ten', 'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen',
          'Seventeen', 'Eighteen', 'Nineteen')
_TENS = {2: 'Twenty', 3: 'Thirty', 4: 'Forty', 5: 'Fifty', 6: 'Sixty', 7: 'Seventy',
         8: 'Eighty', 9: 'Ninety'}


def words(n):
    """``25 -> 'TwentyFive'`` -- an integer as a letters-only macro-name fragment.

    TeX forbids digits in a command name, so a per-level macro (``\\numFigFiveValPctFive``)
    has to spell its level out.  Mechanical, so no level is hand-named.
    """
    n = int(round(float(n)))
    if n < 0:
        return 'Minus' + words(-n)
    if n < 20:
        return _UNITS[n]
    if n < 100:
        return _TENS[n // 10] + (_UNITS[n % 10] if n % 10 else '')
    if n == 100:
        return 'Hundred'
    return 'N' + ''.join(_UNITS[int(d)] for d in str(n))


def frac_token(f):
    """``0.25 -> 'PctTwentyFive'`` -- the macro fragment for a parameter fraction."""
    return 'Pct' + words(round(float(f) * 100))


def _num(value):
    """``value`` as a float, or NaN (never an exception, never a string)."""
    v = pd.to_numeric(value, errors='coerce')
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def _thousands(value):
    """``163109376 -> '163,109,376'`` -- a count a reader can read, as a macro body.

    A parameter count formatted by :func:`paper_assets.common.fmt_sig` would come out
    ``$1.63\\times 10^{8}$``, which is not how a model size is quoted in prose.
    """
    v = _num(value)
    return '' if not np.isfinite(v) else f'{int(round(v)):,}'


def _text(value):
    """A string-valued macro body, safe for LaTeX text mode.

    A value that already CONTAINS LaTeX (a method label such as ``Sven (Gram, full $J$)``)
    passes through; anything else is escaped, because a plain ``_`` (``profile_results_v3``)
    is a hard error in text mode and :func:`paper_assets.common.check_raw` refuses it.
    """
    s = str(value)
    return C.Raw(s) if ('$' in s or '\\' in s) else C.latex_escape(s)


def _scan_n_params(inp, scan):
    """The parameter count the records of ``scan`` report, or NaN."""
    for kind in ('confirm', 'timing', ''):
        try:
            df = hl.load(scan, kind or 'confirm', results_root=inp.results_root)
        except Exception:
            continue
        if df is None or 'n_params' not in getattr(df, 'columns', ()):
            continue
        got = pd.to_numeric(df['n_params'], errors='coerce').dropna()
        if len(got):
            return float(got.iloc[0])
    return float('nan')


def _grid_runner_up(grid):
    """The best non-selected eligible cell of a Sven grid, and the selected cell's own
    seed mean and spread.

    Used by App. J to say how close the choice was.  The selected cell is the one
    ``large_figs.mark_selected`` flagged (which reads ``bench/best_configs.json``), so a
    re-selection moves this by itself.  ``None`` when the grid has fewer than two
    eligible cells.
    """
    if grid is None or not len(grid):
        return None
    ok = grid
    if 'eligible' in ok.columns:
        ok = ok[ok['eligible'].astype(bool)]
    ok = ok.dropna(subset=['val']).sort_values('val')
    if len(ok) < 2:
        return None
    sel = ok[ok['selected'].astype(bool)] if 'selected' in ok.columns else ok.iloc[:0]
    best = sel.iloc[0] if len(sel) else ok.iloc[0]
    rest = ok[~ok.index.isin([best.name])]
    if not len(rest):
        return None
    nxt = rest.iloc[0]
    bits = [f'{ax}={float(nxt[ax]):g}' for ax in ('k', 'lr', 'rtol')
            if ax in nxt.index and pd.notna(nxt[ax])]
    return {'label': ', '.join(bits), 'val': float(nxt['val']),
            'best_val': float(best['val']),
            'best_std': float(best.get('val_std', float('nan')))}


def _name_list(names):
    """``'A'`` / ``'A and B'`` / ``'A, B and C'`` -- an English list for a prose macro."""
    items = [C.latex_escape(str(n)) for n in names]
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    return ', '.join(items[:-1]) + ' and ' + items[-1]


def _row_of(tbl, method='Sven', col='method'):
    """The row of ``tbl`` whose ``col`` is ``method`` (``None`` if there is none)."""
    if tbl is None or not len(tbl) or col not in tbl.columns:
        return None
    hit = tbl[tbl[col].astype(str) == method]
    return hit.iloc[0] if len(hit) else None


def _rank(tbl, column, method='Sven', ascending=True):
    """``(rank, n)`` of ``method`` in ``tbl[column]`` -- ties take the lower rank."""
    v = pd.to_numeric(tbl[column], errors='coerce')
    n = int(np.isfinite(v).sum())
    mask = tbl['method'].astype(str) == method
    if not mask.any() or not n:
        return float('nan'), n
    r = v.rank(method='min', ascending=ascending)[mask].iloc[0]
    return (float(r) if np.isfinite(r) else float('nan')), n


def _fs(fraction=0.49, delta=0.0):
    """The font size :func:`paper_assets.common.set_paper_style` uses at ``fraction``."""
    return C._FONT_FOR_FRACTION.get(round(float(fraction), 2), 7.0) + delta


def _shared_legend(fig, ncol=4, methods_first=True, loc='outside lower center',
                   fontsize=None):
    """One figure-level legend under the panels, deduped over every axes.

    ``loc='outside lower center'`` is what makes constrained layout reserve the strip;
    a plain ``lower center`` legend would sit on top of the bottom row's x labels.
    """
    seen = {}
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            seen.setdefault(label, handle)
    if not seen:
        return None
    labels = list(seen)
    if methods_first:   # the seed-band entry last, where a reader looks for a note
        band = paired.SEED_SPREAD_LABEL
        labels = [l for l in labels if l != band] + [l for l in labels if l == band]
    return fig.legend([seen[l] for l in labels], labels, ncol=ncol, loc=loc,
                      frameon=False, **({} if fontsize is None
                                        else {'fontsize': fontsize}))


def _method_legend(fig, methods, ncol=4, band=True, extra=(),
                   loc='outside lower center', fontsize=None):
    """One figure-level method legend with CLEAN names, under the panels.

    Not the per-axes handles: :func:`large_figs.plot_curves` labels each line with its own
    seed count, which is the right thing on a panel and produces two entries for the same
    method in a shared legend when two scans have different counts (CIFAR-CE's Sven rests
    on fewer confirmation seeds than the label-regression scan's while the re-selection
    lands).  The counts belong on the panel (:func:`_seed_shortfall_note`) and in the
    table's ``fin/att`` column; the legend is the colour key.
    """
    from matplotlib.patches import Patch
    handles, labels = C.method_handles(methods)
    if band:
        handles.append(Patch(facecolor='0.45', alpha=0.25, edgecolor='none'))
        labels.append(paired.SEED_SPREAD_LABEL)
    for handle, label in extra:
        handles.append(handle)
        labels.append(label)
    return fig.legend(handles, labels, ncol=ncol, loc=loc, frameon=False,
                      **({} if fontsize is None else {'fontsize': fontsize}))


def _seed_shortfall_note(ax, drawn, where=(0.02, 0.03)):
    """Name, on the panel, any method drawn on fewer seeds than the rest.

    A band over 2 of 5 seeds is not the same statement as a band over 5, and the figure
    has to say which it is where the curve is, not only in the table beside it.
    """
    counts = {m: int(n) for m, n in (drawn or {}).items() if n}
    if not counts:
        return None
    most = max(counts.values())
    short = [f'{style.method_label(m)}: {n} of {most} seeds'
             for m, n in counts.items() if n < most]
    if not short:
        return None
    return ax.text(*where, '\n'.join(short), transform=ax.transAxes, fontsize=5.4,
                   va='bottom', ha='left', color='0.25')


def plot_curves_positive(ax, runs_by_method, key, versus='time', time_key='train_times',
                         band=True, lw_sven=1.4, lw=0.85):
    """:func:`large_figs.plot_curves` for a LOG axis: the same curves without the ``x = 0``
    point.

    The validation / test curves start at the UNTRAINED model, which sits at epoch 0 and
    therefore at ``t = 0``; on a log time axis that point cannot be drawn, and leaving it
    in the data draws a spurious horizontal lead-in from the left spine to the first real
    evaluation (an untrained accuracy plotted as if it had been reached instantly).  A
    linear time axis is not an option either: Sven's epoch is 23x SGD's on the ResNet, so
    every baseline would be squeezed into the left few per cent of the panel.

    Everything numeric comes from the same functions of record as ``plot_curves``
    (:func:`large_figs.curve_band` for the seed band, :func:`large_figs.curve_axis` for
    the axis); only the first sample is dropped.  Returns ``{method: n_seeds}``.
    """
    drawn = {}
    banded = False
    for method, rows in runs_by_method.items():
        mean, lower, upper, n = lf.curve_band(rows, key)
        if mean is None:
            continue
        x = lf.curve_axis(rows, mean, key, versus, time_key)
        if x is None:
            continue
        m = len(x)
        x = np.asarray(x[:m], dtype=float)
        y = np.asarray(mean[:m], dtype=float)
        keep = np.isfinite(x) & (x > 0)
        if not keep.any():
            continue
        color = style.method_color(method)
        ax.plot(x[keep], y[keep], color=color,
                lw=lw_sven if method == 'Sven' else lw,
                label=style.method_label(method),
                zorder=3 if method == 'Sven' else 2)
        if band and n > 1:
            lo = np.asarray(lower[:m], dtype=float)[keep]
            hi = np.asarray(upper[:m], dtype=float)[keep]
            ax.fill_between(x[keep], lo, hi, color=color, alpha=0.18, lw=0,
                            zorder=(3 if method == 'Sven' else 2) - 0.5)
            banded = True
        drawn[method] = n
    if banded:
        style.band_legend(ax)
    ax.set_xscale('log')
    ax.set_xlabel(sa.axis_label(versus, time_key))
    ax.set_ylabel(lf.curve_label(key))
    if not lf.is_accuracy(key):
        ax.set_yscale('log')
    return drawn


def _has_study(df, arch, study):
    """Whether the profile frame holds any row of ``study`` for ``arch``."""
    return bool(len(df[(df.arch == arch) & (df.study == study)]))


def _methods_with_data(df, arch, study, methods):
    """``methods`` restricted to the ones that actually have a row here.

    :func:`profile_helpers.plot_sweep` draws (and labels) a line for every method it is
    given, so a backend the sweep never ran -- the chunked capture has no ``k`` sweep --
    would otherwise get a legend entry with nothing behind it.
    """
    have = set(df[(df.arch == arch) & (df.study == study)]['method'])
    return [m for m in methods if m in have]


def selected_lrs(inp, scan=None):
    """``{canonical method: selected lr}`` from the selection of record.

    Read at build time, never typed: a re-selection moves the ring on the figure and the
    star in the table by itself.
    """
    out = {}
    try:
        sels = hl.selection_methods(scan or NANOGPT, inp.payload)
    except KeyError:        # no selection for this scan: the table loses its star, not
        return out          # its rows (every level in it is a scan quantity)
    for method, sel in sels.items():
        lr = (sel.get('hparams') or {}).get('lr')
        if lr is not None and C.finite(_num(lr)):
            out[hl.method_key(method)] = _num(lr)
    return out


def _drop_axes_legends(fig):
    """Remove per-axes legends: these figures carry ONE legend under the panels."""
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()


def _headroom(ax, frac=0.06):
    """Give a panel a little room above (and below) its data.

    :func:`profile_helpers.plot_sweep` sets no y margin and parks an "not measured"
    marker on the top spine, so the highest line of a sweep panel can be drawn ON the
    frame and its markers clipped in half -- visible in the first build of
    ``profile_batchsize`` (the ``full`` capture on Toy 1D) and ``profile_k_sweep``.
    Multiplicative on a log axis, additive on a linear one.
    """
    lo, hi = ax.get_ylim()
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return ax
    if ax.get_yscale() == 'log' and lo > 0:
        span = (hi / lo) ** frac
        ax.set_ylim(lo / span, hi * span)
    else:
        span = (hi - lo) * frac
        ax.set_ylim(lo - span, hi + span)
    return ax


def profile_root(prefer=None):
    """The profile root to read, and what is known about its completeness.

    Returns ``(root, info)`` with ``info`` = ``{name, complete, n_found, n_expected,
    per_config, has_cifar, provisional}``.  See :data:`PROFILE_PREFERENCE` for why this
    is not :func:`profile_helpers.results_root`.
    """
    env = os.environ.get('SV3_PROFILE_ROOT')
    order = [p for p in (env, prefer, *PROFILE_PREFERENCE) if p]
    root = Path(next((p for p in order if ph.has_profiles(p)), PROFILE_PREFERENCE[0]))
    complete, n, expected = ph.profile_status(root)
    per_config = {d.name: sum(1 for _ in d.glob('*.json'))
                  for d in sorted(root.glob('*')) if d.is_dir()}
    info = {'name': root.name, 'root': str(root),
            'complete': bool(complete), 'n_found': int(n), 'n_expected': int(expected),
            'per_config': per_config,
            'has_cifar': any(k.startswith('profile_cifar') for k in per_config),
            'provisional': not bool(complete),
            'protocol': profile_protocol(root)}
    return root, info


#: the measurement-protocol keys App. Q's opening paragraph describes.  It used to
#: describe the retired v2 profile (15 staged batches, 5 warm-up steps, 10 measured);
#: every record of ``profile_results_v3`` says 8 / 10 / 50, so the sentence is generated
#: from the records and a re-profile updates it.
PROTOCOL_KEYS = ('staged_batches', 'warmup_steps', 'num_steps', 'min_steps',
                 'max_seconds')

#: the leading fraction of a measured series ``profile_helpers.cycle_mean`` treats as
#: start-up and drops.  It is hard-coded there (``a[int(0.2 * a.size):]``); if that
#: changes, this must change with it -- ``numProfAmortPct`` is the number App. Q quotes.
PROFILE_STARTUP_FRACTION = 0.2


def profile_protocol(root):
    """The measurement protocol the records of ``root`` agree on.

    Returns ``{key: value}`` for every key of :data:`PROTOCOL_KEYS` that every record
    carries with the SAME value, plus ``n_records`` and ``disagree`` (the keys on which
    the records differ, which must never reach the paper as a single number).
    """
    seen, n = {k: set() for k in PROTOCOL_KEYS}, 0
    for f in sorted(Path(root).glob('*/*.json')):
        try:
            block = (json.loads(f.read_text()).get('profile') or {})
        except (OSError, ValueError):
            continue
        n += 1
        for k in PROTOCOL_KEYS:
            if block.get(k) is not None:
                seen[k].add(block[k])
    out = {'n_records': n, 'disagree': sorted(k for k, v in seen.items() if len(v) > 1)}
    for k, v in seen.items():
        if len(v) == 1:
            out[k] = v.pop()
    return out


# ---------------------------------------------------------------------------
# Inputs: every analysis call this module makes, once per build
# ---------------------------------------------------------------------------
class Inputs:
    """Lazily loaded inputs, shared by the figures, the tables and the macros.

    One object per build, so a scan is loaded once and a figure cannot end up describing
    a different selection from the table beside it.
    """

    def __init__(self, root=None):
        self.results_root = root
        self.payload = hl.load_selection()
        self.notes = []

    # -- freshness ---------------------------------------------------------
    @cached_property
    def provisional(self):
        """Tags for everything that can still move under this build (for the stamps)."""
        tags = list(C.provisional_scans(tuple(CIFAR) + (NANOGPT,)))
        # `freshness_report` fires on a LIVE claim or a post-selection tuning record, which
        # is the "the selection can still move" signal.  It does not fire on a pass that is
        # simply short: after the CIFAR-CE re-selection its confirmation and timing passes
        # re-ran from scratch, and for a while every number came from 3 of the 5 seeds with
        # nothing in flight.  A level from 3 seeds is provisional too, so check the Sven row
        # of each pass as well.
        for scan in (*CIFAR, NANOGPT):
            row = _row_of(self.conf[scan], 'Sven')
            if row is not None:
                fin, att = _num(row.get('fin_conf')), _num(row.get('att_conf'))
                if np.isfinite(fin) and np.isfinite(att) and fin < att:
                    tags.append(f'{scan}: Sven on {fin:.0f} of {att:.0f} confirmation '
                                f'seeds')
            cost = self.timing_cost(scan)
            fin, att = cost.get('fin', np.nan), cost.get('att', np.nan)
            if np.isfinite(fin) and np.isfinite(att) and fin < att:
                tags.append(f'{scan}: Sven on {fin:.0f} of {att:.0f} timing runs')
        if self.fig5_incomplete:
            tags.append(f'{FIG5}: selected configuration on '
                        f'{self.fig5_selected["cfg"]["n_seeds"]} of '
                        f'{self.fig5_target_seeds} seeds')
        if self.profile_info['provisional']:
            tags.append(f'{self.profile_info["name"]}: '
                        f'{self.profile_info["n_found"]}/'
                        f'{self.profile_info["n_expected"]} configurations')
        return tags

    def scan_provisional(self, scan):
        return any(str(scan) in tag for tag in self.provisional)

    # -- CIFAR / nanoGPT ---------------------------------------------------
    @cached_property
    def conf(self):
        return {s: hl.confirmation_table(s, self.payload, self.results_root)
                for s in (*CIFAR, NANOGPT)}

    @cached_property
    def eff(self):
        return {s: hl.efficiency_table(s, self.payload, self.results_root)
                for s in (*CIFAR, NANOGPT)}

    @cached_property
    def runs_conf(self):
        return {s: lf.selected_runs_by_method(s, 'confirm', self.payload,
                                              self.results_root)
                for s in (*CIFAR, NANOGPT)}

    @cached_property
    def runs_time(self):
        return {s: lf.selected_runs_by_method(s, 'timing', self.payload,
                                              self.results_root)
                for s in (*CIFAR, NANOGPT)}

    @cached_property
    def grid(self):
        return {s: lf.mark_selected(lf.sven_grid(s, self.results_root), s, self.payload)
                for s in (*CIFAR, NANOGPT)}

    @cached_property
    def rank_used(self):
        return {s: lf.sven_rank_used(s, 'diag', self.results_root, self.payload)
                for s in CIFAR}

    @cached_property
    def gaps(self):
        return {s: lf.generalisation_gaps(self.conf[s]) for s in CIFAR}

    @cached_property
    def paired_nanogpt(self):
        return hl.paired_vs_sven(NANOGPT, payload=self.payload,
                                 results_root=self.results_root)

    @cached_property
    def nanogpt_scan(self):
        """The nanoGPT TUNING scan with the derived columns -- the lr-sensitivity input.

        The confirmation pass holds one configuration per method, so the sensitivity of a
        method to its learning rate can only come from the scan that tuned it.
        """
        return ah.add_derived(style.load_results(NANOGPT,
                                                 results_root=self.results_root))

    @cached_property
    def nanogpt_lr(self):
        """Best configuration per (method, lr) on the nanoGPT scan, selected on
        validation loss -- :func:`reviewer_figs.arm_table` with ``lr`` as the arm."""
        return rf.arm_table(self.nanogpt_scan, 'lr')

    # -- Fig 5 -------------------------------------------------------------
    @cached_property
    def fig5_groups(self):
        """Every Sven configuration of the Fig-5 scan, the SELECTED one first."""
        return lf.paramfrac_groups(results_root=self.results_root, payload=self.payload)

    @cached_property
    def fig5_selected(self):
        sel = next((g for g in self.fig5_groups if g['is_selected']), None)
        if sel is None:      # the re-run has not started: report the scan as it stands
            self.notes.append(
                f'{FIG5}: NO run at the selected configuration '
                f'{self.fig5_groups[0]["headline"]["label"]}; on disk '
                f'{[g["cfg"]["label"] for g in self.fig5_groups]}')
            return self.fig5_groups[0]
        return sel

    @cached_property
    def fig5_target_seeds(self):
        """How many seeds a Fig-5 point is meant to have: the most any configuration of
        the scan reached (3 as launched).  Computed, so the re-run can change it."""
        return max((int(g['cfg']['n_seeds']) for g in self.fig5_groups), default=0)

    @cached_property
    def fig5_incomplete(self):
        sel = self.fig5_selected
        return (not sel['is_selected']
                or int(sel['cfg']['n_seeds']) < self.fig5_target_seeds
                or len(sel['cfg']['fractions']) < max(
                    (len(g['cfg']['fractions']) for g in self.fig5_groups), default=0))

    @cached_property
    def fig5_profiles(self):
        """The profile's own ``param_fraction`` study for the ResNet, or an empty frame.

        Restricted to the mask granularities the SCAN ran, so the overlay compares like
        with like; absent from v3 (no ``cifar_resnet18`` directory), in which case the
        panels show the records only.
        """
        p = self.profiles
        if not len(p) or 'cifar_resnet18' not in set(p['arch']):
            return pd.DataFrame()
        modes = set(self.fig5_selected['table']['mask_mode']) | {'none'}
        return p[(p.arch == 'cifar_resnet18') & (p.study == 'param_fraction')
                 & (p.method == 'gram_full') & p.status.isin(ph.TIMED)
                 & p.mask_mode.isin(modes)]

    # -- GPT-2 -------------------------------------------------------------
    @cached_property
    def gpt2(self):
        return lf.gpt2_frame(self.results_root)

    @cached_property
    def gpt2_best(self):
        return lf.gpt2_best_per_method(self.gpt2)

    @cached_property
    def gpt2_lr(self):
        return lf.gpt2_lr_table(self.gpt2)

    # -- the profile -------------------------------------------------------
    @cached_property
    def _profile(self):
        root, info = profile_root()
        return root, info, ph.load_profiles(root)

    @property
    def profile_root(self):
        return self._profile[0]

    @property
    def profile_info(self):
        return self._profile[1]

    @property
    def profiles(self):
        return self._profile[2]

    @cached_property
    def profile_archs(self):
        return [a for a in ph.ARCH_ORDER if a in set(self.profiles.get('arch', []))]

    @cached_property
    def profile_widths(self):
        return [a for a in ('mnist_width', 'nanogpt_width')
                if a in set(self.profiles.get('arch', []))]

    @cached_property
    def profile_compare(self):
        """v2 -> v3 per configuration, for the one table that is allowed to hold both."""
        if str(self.profile_root) == str(ph.ROOT_V2):
            return pd.DataFrame()
        return ph.compare_profiles(ph.ROOT_V2, self.profile_root)

    def profile_note(self):
        """The provenance sentence every profile asset carries."""
        info = self.profile_info
        bits = [f'profile root {info["name"]}',
                f'{info["n_found"]}/{info["n_expected"]} configurations'
                + ('' if info['complete'] else ', PARTIAL')]
        if not info['has_cifar']:
            bits.append('no ResNet18 profile in this root: the CIFAR cost point comes '
                        'from the standalone timing pass')
        return '; '.join(bits)

    # -- cost of one method, from the timing pass --------------------------
    def timing_cost(self, scan, method='Sven'):
        row = _row_of(self.eff[scan], method)
        if row is None:
            return {}
        return {'epoch_s': _num(row.get('epoch_s')),
                'ms_per_step': _num(row.get('ms_per_step')),
                'peak_mb': _num(row.get('peak_gpu_mem_mb')),
                'wall_s': _num(row.get('wall_s')),
                'fin': _num(row.get('fin_timing')), 'att': _num(row.get('att_timing'))}


# ---------------------------------------------------------------------------
# The figures.  Every draw function is ``(ctx, opts) -> (fig, meta)`` and writes
# NOTHING: the only save path is :func:`common.save_fig`, reached either from
# :func:`build` or from the notebook API (``campaign/FIGURE_API_CONTRACT.md``).  What
# used to be a hard-coded cosmetic is a key of the figure's ``defaults`` below, and the
# DEFAULT IS WHAT THE BUILDER DID, so an empty override file redraws the paper exactly.
# ---------------------------------------------------------------------------
def _scan_items(opts):
    """``[(scan, title)]`` for the ``scans`` knob -- the CIFAR pair, in report order."""
    return [(s, CIFAR.get(s, str(s))) for s in (opts.get('scans') or CIFAR)]


#: panel key -> the title fragment the CIFAR curve figures use (:data:`CIFAR_PANELS`)
_PANEL_TITLES = dict(CIFAR_PANELS)


def _curve_panels(opts):
    """``[(key, panel title)]`` for the ``panels`` knob of a curve figure."""
    keys = opts.get('panels') or [k for k, _t in CIFAR_PANELS]
    return [(k, _PANEL_TITLES.get(k, lf.curve_label(k))) for k in keys]


def _only_methods(runs_by_method, methods=None):
    """``runs_by_method`` restricted to ``methods`` (all of them when that is empty).

    Returns the dict itself when nothing is asked for, so the default path is the object
    the figure drew from before there were options at all.
    """
    if not methods:
        return runs_by_method
    return {m: rows for m, rows in runs_by_method.items() if m in methods}


def _style_fraction(opts):
    """The ``fraction`` :func:`common.set_paper_style` is called with.

    Usually the panel width itself; ``profile_batchsize`` is the exception (its panels
    are a fifth of the line wide but its type is sized for 0.49), hence the own knob.
    """
    f = opts.get('style_fraction')
    return float(f if f is not None else opts['fraction'])


def _title_fs(opts):
    """The panel-title font size: the knob, or :func:`_fs` at the figure's fraction."""
    fs = opts.get('title_fontsize')
    return float(fs) if fs is not None else _fs(_style_fraction(opts))


def _legend_kw(opts):
    """The figure-level legend's placement knobs, as ``fig.legend`` keywords."""
    kw = {'ncol': int(opts.get('legend_ncol', 4)),
          'loc': opts.get('legend_loc', 'outside lower center')}
    if opts.get('legend_fontsize') is not None:
        kw['fontsize'] = opts['legend_fontsize']
    return kw


# ---------------------------------------------------------------------------
# F7a / F7b: CIFAR curves, per epoch and per second
# ---------------------------------------------------------------------------
def _cifar_curves(inp, opts):
    """Validation loss and test accuracy vs epoch (confirmation) or vs time (timing).

    One row per objective, 11 methods, seed band = mean +/- 1 std.  The time axis is the
    standalone timing pass, which re-runs the TUNING seeds: the panel title says which
    seeds it is, because the table beside it reports the confirmation seeds and on
    CIFAR-CE the two differ by more than the seed band.
    """
    versus = opts['versus']
    runs = inp.runs_time if versus == 'time' else inp.runs_conf
    scans, panels = _scan_items(opts), _curve_panels(opts)
    C.set_paper_style(_style_fraction(opts))
    fig, axes = plt.subplots(len(scans), len(panels), squeeze=False,
                             figsize=C.figsize(len(panels), len(scans),
                                               fraction=opts['fraction'],
                                               aspect=opts['aspect'],
                                               extra_h=opts['extra_h']))
    fs = _title_fs(opts)
    drawn = {}
    for i, (scan, title) in enumerate(scans):
        by_method = _only_methods(runs[scan], opts['methods'])
        for j, (key, panel) in enumerate(panels):
            ax = axes[i][j]
            if versus == 'time':
                d = plot_curves_positive(ax, by_method, key, versus='time',
                                         band=opts['band'], lw_sven=opts['lw_sven'],
                                         lw=opts['lw_time'])
            else:
                d = lf.plot_curves(ax, by_method, key, versus=versus, band=opts['band'],
                                   lw_sven=opts['lw_sven'], lw=opts['lw'])
            drawn[(scan, key)] = d
            if opts['seed_note']:
                _seed_shortfall_note(ax, d, where=(opts['note_xy_acc']
                                                   if lf.is_accuracy(key)
                                                   else opts['note_xy_loss']))
            ax.set_title(f'{title}: {panel.lower()}', fontsize=fs)
    methods = list(dict.fromkeys(m for scan, _t in scans
                                 for m in _only_methods(runs[scan], opts['methods'])))
    _method_legend(fig, methods, band=opts['legend_band'], **_legend_kw(opts))
    return fig, {'axes': axes,
                 'drawn': {f'{k[0]}|{k[1]}': v for k, v in drawn.items()},
                 'seeds': {s: lf.seed_note(_only_methods(runs[s], opts['methods']))
                           for s, _t in scans}}


def _cifar_gap(inp, opts):
    """The train-vs-validation gap: ``losses.train_eval`` against the validation loss.

    A method below the diagonal fits the training subset and generalises worse; a method
    far to the right has not optimised at all.  The two CIFAR objectives are one of each,
    which is why the two panels share their axes convention.
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter
    scans = _scan_items(opts)
    C.set_paper_style(_style_fraction(opts))
    fig, axes = plt.subplots(1, len(scans), squeeze=False,
                             figsize=C.figsize(len(scans), 1, fraction=opts['fraction'],
                                               aspect=opts['aspect'],
                                               extra_h=opts['extra_h']))
    fs = _title_fs(opts)
    methods = []
    for ax, (scan, title) in zip(axes[0], scans):
        g = inp.gaps[scan].dropna(subset=['train_eval', 'val'])
        methods.extend(g['method'])
        # colour is the method (style.METHOD_COLORS, Sven black) and the legend below is
        # the key: a text label per point is unreadable at this size, which is what the
        # first draft of this panel showed
        ax.scatter(g['train_eval'], g['val'], s=opts['marker_size'], zorder=3,
                   edgecolor='none', c=[style.method_color(m) for m in g['method']])
        lo = float(np.nanmin([g['train_eval'].min(), g['val'].min()])) * opts['pad_lo']
        hi = float(np.nanmax([g['train_eval'].max(), g['val'].max()])) * opts['pad_hi']
        if opts['diagonal']:
            ax.plot([lo, hi], [lo, hi], color='0.6', lw=0.8, ls='--', zorder=1,
                    label='no generalisation gap')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('Train loss, eval mode')
        ax.set_ylabel('Validation loss')
        ax.set_title(title, fontsize=fs)
    extra = []
    if opts['diagonal']:
        extra.append((Line2D([], [], color='0.6', lw=0.8, ls='--'),
                      'no generalisation gap'))
    _method_legend(fig, list(dict.fromkeys(methods)), band=opts['legend_band'],
                   extra=extra, **_legend_kw(opts))
    return fig, {'axes': axes}


def _cifar_landscape(inp, opts):
    """Sven's ``(lr, rtol)`` landscape per ``k``, per objective, on ONE colour scale.

    Blank + ``--`` where nothing ran, ``f/a`` under a cell with fewer runs than seeds,
    ``*`` and a dashed edge where a cell is not eligible, red ring on the selection of
    record.  The CIFAR-CE grid is ragged on purpose (the ``rtol`` extension covers three
    learning rates at ``k=128`` only) and a cell that was never run must not be filled in.
    """
    scans = _scan_items(opts)
    C.set_paper_style(_style_fraction(opts))
    ks = sorted({float(k) for s, _t in scans
                 for k in inp.grid[s]['k'].dropna().unique()})
    fig, axes = plt.subplots(len(scans), len(ks), squeeze=False,
                             figsize=C.figsize(len(ks), len(scans),
                                               fraction=opts['fraction'],
                                               aspect=opts['aspect']))
    fs = _title_fs(opts)
    marks = {}
    for i, (scan, title) in enumerate(scans):
        g = inp.grid[scan]
        xs, ys = lf.landscape_axes(g, 'lr', 'rtol')
        vmin, vmax = lf.landscape_scale(g, 'val')
        for j, k in enumerate(ks):
            ax = axes[i][j]
            sub = g[np.isclose(pd.to_numeric(g['k'], errors='coerce'), k)]
            if not len(sub):
                ax.set_visible(False)
                continue
            im, mat, inelig, partial = lf.plot_sven_landscape(
                ax, sub, x='lr', y='rtol', xs=xs, ys=ys, vmin=vmin, vmax=vmax)
            marks[f'{scan}|k={k:g}'] = {'partial': int(partial.sum()),
                                        'ineligible': int(inelig.sum())}
            ax.set_title(f'{title}, $k={int(k)}$',
                         fontsize=fs)
            if j:
                ax.set_ylabel('')
        fig.colorbar(im, ax=axes[i], label=opts['cbar_label'])
    return fig, {'axes': axes, 'cells': marks}


def _cifar_cost(inp, opts):
    """Seconds per epoch and peak GPU memory of the selected configuration, standalone.

    Both from ``{scan}_timing``: the only pass that had the GPU to itself.  Log axes,
    because Sven is more than an order of magnitude from the first-order methods on both.
    """
    scans = _scan_items(opts)
    C.set_paper_style(_style_fraction(opts))
    fig, axes = plt.subplots(len(scans), 2, squeeze=False,
                             figsize=C.figsize(2, len(scans), fraction=opts['fraction'],
                                               aspect=opts['aspect']))
    fs = _title_fs(opts)
    for i, (scan, title) in enumerate(scans):
        lf.plot_cost_bars(axes[i], inp.eff[scan], log=opts['log'])
        for ax in axes[i]:
            ax.set_title(title, fontsize=fs)
            for text in ax.get_xticklabels():
                text.set_fontsize(opts['xtick_fontsize'])
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
    _shared_legend(fig, **_legend_kw(opts))
    return fig, {'axes': axes}


def _cifar_rank_used(inp, opts):
    """The rank Sven actually inverted, per epoch, against its ``k`` cap.

    ``svd_summary.num_nonzero_svs_epoch`` from the ``{scan}_diag`` pass: the number of
    singular values that survived the ``rtol`` cut, averaged over the steps of an epoch.
    At the cap, ``k`` is the binding constraint; below it, ``rtol`` is.
    """
    scans = _scan_items(opts)
    C.set_paper_style(_style_fraction(opts))
    fig, axes = plt.subplots(1, len(scans), squeeze=False,
                             figsize=C.figsize(len(scans), 1, fraction=opts['fraction'],
                                               aspect=opts['aspect']))
    fs = _title_fs(opts)
    stats = {}
    for ax, (scan, title) in zip(axes[0], scans):
        ru = inp.rank_used[scan]
        if not len(ru):
            ax.set_visible(False)
            continue
        for _, sub in ru.groupby('run_id'):
            ax.plot(sub['epoch'], sub['rank_used'], color=style.method_color('Sven'),
                    alpha=opts['alpha'], lw=opts['lw'])
        k = _num(ru['k'].dropna().iloc[0])
        ax.axhline(k, color=opts['k_color'], ls='--', lw=0.9, label=f'$k={int(k)}$')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Rank inverted')
        ax.set_ylim(0, max(k, float(ru['rank_used'].max())) * opts['ylim_pad'])
        ax.set_title(f'{title} ({ru["run_id"].nunique()} seeds)',
                     fontsize=fs)
        ax.legend(fontsize=opts['legend_fontsize'], loc=opts['legend_loc'])
        stats[scan] = {'k': k, 'mean': float(ru['rank_used'].mean()),
                       'frac': float(ru['rank_used'].mean() / k) if k else float('nan'),
                       'n_seeds': int(ru['run_id'].nunique())}
    return fig, {'axes': axes, 'rank_used': stats}


# ---------------------------------------------------------------------------
# F7c / F7d: Fig 5
# ---------------------------------------------------------------------------
#: line style per Fig-5 configuration; the SELECTED one is always the solid line
FIG5_STYLES = (('-', 'o'), ('--', 's'), (':', '^'), ('-.', 'D'))


def _fig5_styles(opts):
    return [tuple(pair) for pair in (opts.get('styles') or FIG5_STYLES)]


def _fig5_quality(inp, opts):
    """Loss and accuracy against the ACTUAL parameter fraction, all configurations on disk.

    Solid = the configuration the headline label-regression scan selected; every other
    configuration is dashed and labelled as not selected.  A point with fewer seeds than
    the scan intended is hollow and carries its ``finished/attempted``; the chance line on
    the accuracy panel is what "collapse" means.
    """
    from matplotlib.lines import Line2D
    C.set_paper_style(_style_fraction(opts))
    fig, axes = plt.subplots(1, 2, figsize=C.figsize(2, 1, fraction=opts['fraction'],
                                                     aspect=opts['aspect'],
                                                     extra_h=opts['extra_h']))
    fs = _title_fs(opts)
    styles = _fig5_styles(opts)
    counts = {}
    target = inp.fig5_target_seeds
    for i, (g, (ls, marker)) in enumerate(zip(inp.fig5_groups, styles)):
        counts[g['cfg']['label']] = lf.plot_paramfrac_quality(
            axes, g['table'], ls=ls, marker=marker, label_suffix='',
            hollow_label=False, annotate_counts=False)
        # ``plot_paramfrac_quality``'s own annotation is finished/ATTEMPTED, i.e. it marks
        # a point some of whose runs diverged (the hollow marker) but not one whose third
        # seed has not landed yet -- which is the state the Fig-5 re-run is in.  Annotate
        # against the number of seeds the scan reached at its fullest instead.
        if not opts['seed_note']:
            continue
        for ax, key in zip(axes, ('val', 'val_acc')):
            for _, r in g['table'].iterrows():
                n = int(_num(r['finished']))
                v = _num(r[key])
                if np.isfinite(v) and n and n < target:
                    ax.annotate(f'{n} seed' + ('s' if n > 1 else ''),
                                (_num(r['actual']), v), fontsize=opts['note_fontsize'],
                                color='0.3', xytext=(4, -9),
                                textcoords='offset points')
    if opts['chance_line']:
        axes[1].axhline(CIFAR_CHANCE_ACC, color='0.55', lw=0.8, ls=':')
    for ax in axes:                     # room for the seed-count note at f = 1
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo * opts['xlim_pad_lo'], hi * opts['xlim_pad_hi'])
    axes[0].set_title('Loss', fontsize=fs)
    axes[1].set_title('Accuracy', fontsize=fs)
    _drop_axes_legends(fig)
    # colour = which split, line style = which Sven configuration: the two are separate
    # dimensions and a merged auto-legend printed eight entries for four things
    ms = opts['legend_marker_size']
    handles = [(Line2D([], [], color='C0', lw=1.2), 'validation'),
               (Line2D([], [], color='C1', lw=1.2), 'test')]
    for g, (ls, marker) in zip(inp.fig5_groups, styles):
        handles.append((Line2D([], [], color='0.35', lw=1.2, ls=ls, marker=marker,
                               ms=ms),
                        f'{g["cfg"]["label"]} ({g["role"]})'))
    handles += [(Line2D([], [], color='0.35', lw=0, marker='o', ms=4.5, mfc='none',
                        mew=1.2), 'some seeds diverged'),
                (Line2D([], [], color='0.55', lw=0.8, ls=':'), 'chance (10 classes)')]
    C.legend_below(fig, [h for h, _l in handles], [l for _h, l in handles],
                   bbox_to_anchor=None, **_legend_kw(opts))
    return fig, {'axes': axes, 'counts': {k: v.to_dict('records')
                                          for k, v in counts.items()}}


def _fig5_cost(inp, opts):
    """Peak memory and time per step against the actual parameter fraction.

    The records are the measurement of record (the campaign's own code).  The profile's
    ``param_fraction`` study is overlaid where the chosen root has one; its step-time
    overlay is drawn only when that root is v3, because v2's per-step ``empty_cache()``
    reverses the SIGN of the masking cost.
    """
    C.set_paper_style(_style_fraction(opts))
    fig, axes = plt.subplots(1, 2, figsize=C.figsize(2, 1, fraction=opts['fraction'],
                                                     aspect=opts['aspect'],
                                                     extra_h=opts['extra_h']))
    fs = _title_fs(opts)
    is_v3 = str(inp.profile_root) == str(ph.ROOT_V3)
    step_times = is_v3 if opts['profile_step_times'] is None \
        else bool(opts['profile_step_times'])
    prof = inp.fig5_profiles if opts['profile_overlay'] else pd.DataFrame()
    for i, (g, (ls, marker)) in enumerate(zip(inp.fig5_groups, _fig5_styles(opts))):
        lf.plot_paramfrac_cost(axes, g['table'], ls=ls, marker=marker,
                               record_label=f'records, {g["role"]} ({g["cfg"]["label"]})',
                               profiles=prof if (i == 0 and len(prof)) else None,
                               profile_step_times=step_times)
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    axes[0].set_title('Peak GPU memory', fontsize=fs)
    axes[1].set_title('Time per step', fontsize=fs)
    _shared_legend(fig, **_legend_kw(opts))
    return fig, {'axes': axes, 'profile_overlay': int(len(prof)),
                 'overlay_is_v3': is_v3}


# ---------------------------------------------------------------------------
# F11b: nanoGPT (App. L)
# ---------------------------------------------------------------------------
def _nanogpt(inp, opts):
    """nanoGPT in three panels: vs epoch, vs wall time, and vs the learning rate.

    The first two are the appendix version of the main-text panel (F1) with the same
    conventions -- confirmation seeds for the loss, the standalone ``{scan}_timing`` pass
    for the clock, seed band = mean +/- 1 std.  The third is the lr sensitivity the
    curves cannot show: the best configuration per (method, lr) on the TUNING scan, with
    the selection of record ringed.  Sven's grid is one point wide in ``k`` and ``rtol``
    here, so its line IS its lr sensitivity.
    """
    C.set_paper_style(_style_fraction(opts))
    fig, axes = plt.subplots(1, 3, figsize=C.figsize(3, 1, fraction=opts['fraction'],
                                                     aspect=opts['aspect'],
                                                     extra_h=opts['extra_h']))
    fs = _title_fs(opts)
    conf = _only_methods(inp.runs_conf[NANOGPT], opts['methods'])
    lf.plot_curves(axes[0], conf, 'val', versus='epoch', band=opts['band'],
                   lw_sven=opts['lw_sven'], lw=opts['lw'])
    axes[0].set_title('vs epoch (confirmation)', fontsize=fs)
    plot_curves_positive(axes[1], _only_methods(inp.runs_time[NANOGPT], opts['methods']),
                         'val', versus='time', band=opts['band'],
                         lw_sven=opts['lw_sven'], lw=opts['lw_time'])
    axes[1].set_title('vs time (standalone)', fontsize=fs)

    best = inp.nanogpt_lr
    ax = axes[2]
    rf.plot_arm(ax, best, 'lr', 'final_val_loss', logx=True, logy=True, label_col='method')
    picked = []
    for method, lr in (selected_lrs(inp) if opts['mark_selected'] else {}).items():
        # the selection file names methods at RECORD level ('SVD'); a table's `method`
        # column is the canonical key ('Sven'), so match through `headline.method_key`
        row = best[(best['method'].map(hl.method_key) == method)
                   & np.isclose(pd.to_numeric(best['lr'], errors='coerce'), lr)]
        if not len(row):
            continue
        ax.scatter(row['lr'], row['final_val_loss'], s=opts['ring_size'],
                   facecolors='none', edgecolors=style.method_color(method), lw=1.0,
                   zorder=5)
        picked.append({'method': method, 'lr': lr})
    lrs = sorted(set(pd.to_numeric(best['lr'], errors='coerce').dropna()))
    # the grid spans 1e-5 ... 1 and the methods' grids are offset from each other, so
    # ticking at the ARM values (reviewer_figs.arm_ticks, used where a knob has four
    # levels) runs the labels into one another: decades only here
    from matplotlib.ticker import LogLocator, NullFormatter
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Best validation loss')
    ax.set_title('lr sensitivity (scan)', fontsize=fs)
    for a in axes:
        leg = a.get_legend()
        if leg is not None:
            leg.remove()
        if opts['headroom']:
            _headroom(a, opts['headroom'])
    from matplotlib.lines import Line2D
    extra = []
    if opts['mark_selected']:
        ring = Line2D([], [], marker='o', ls='none', mfc='none', mec='0.35', ms=5)
        extra.append((ring, 'selected configuration'))
    _method_legend(fig, list(conf), band=opts['legend_band'], extra=extra,
                   **_legend_kw(opts))
    return fig, {'axes': axes, 'selected_lrs': picked, 'n_lrs': len(lrs),
                 'seeds': lf.seed_note(conf)}


# ---------------------------------------------------------------------------
# F11: GPT-2 small
# ---------------------------------------------------------------------------
def _gpt2(inp, opts):
    """Validation loss vs optimisation step: the best lr per method, and all Sven lrs.

    One seed, one epoch, 26 step evaluations, so there is no band anywhere on this figure
    and the caption says so.  The right panel is the evidence that Sven's learning-rate
    optimum is INTERIOR to its sweep: the level is not an lr failure.
    """
    C.set_paper_style(_style_fraction(opts))
    fig, axes = plt.subplots(1, 2, figsize=C.figsize(2, 1, fraction=opts['fraction'],
                                                     aspect=opts['aspect'],
                                                     extra_h=opts['extra_h']))
    fs = _title_fs(opts)
    best = inp.gpt2[inp.gpt2['run_id'].isin(inp.gpt2_best['run_id'])]
    lf.plot_gpt2_curves(axes[0], best, 'val_step', lw_sven=opts['lw_sven'],
                        lw=opts['lw'])
    sven = inp.gpt2[inp.gpt2['method'] == 'Sven'].sort_values('lr')
    lrs = sorted(sven['lr'].unique())
    cmap = plt.get_cmap(opts['cmap'])
    lf.plot_gpt2_curves(
        axes[1], sven, 'val_step', lw_sven=opts['lr_lw'], lw=opts['lr_lw'],
        label_fn=lambda r: f"lr={r['lr']:g}",
        color_fn=lambda r: cmap(opts['cmap_lo'] + opts['cmap_span']
                                * lrs.index(r['lr']) / max(len(lrs) - 1, 1)))
    axes[0].set_title('Best lr per method', fontsize=fs)
    axes[1].set_title(f'Sven, all {len(lrs)} learning rates '
                      f'($k=B={int(_num(sven["k"].iloc[0]))}$)',
                      fontsize=fs)
    for ax in axes:
        ax.set_yscale('log')
        ax.legend(fontsize=opts['legend_fontsize'], ncol=opts['legend_ncol'])
    return fig, {'axes': axes, 'n_lrs': len(lrs)}


def _gpt2_lr(inp, opts):
    """Final validation and test loss against the learning rate, one line per method.

    26 of the 29 runs are baselines on six learning rates each; every point is one run.
    The panel exists to show where each method's optimum sits inside its own sweep.
    """
    C.set_paper_style(_style_fraction(opts))
    panels = [tuple(p) for p in opts['panels']]
    fig, axes = plt.subplots(1, len(panels),
                             figsize=C.figsize(len(panels), 1, fraction=opts['fraction'],
                                               aspect=opts['aspect'],
                                               extra_h=opts['extra_h']))
    for ax, (col, name) in zip(np.atleast_1d(axes), panels):
        for method, sub in inp.gpt2_lr.groupby('method', sort=False):
            sub = sub.sort_values('lr')
            ax.plot(sub['lr'], pd.to_numeric(sub[col], errors='coerce'), 'o-',
                    ms=opts['marker_size'], color=style.method_color(method),
                    lw=opts['lw_sven'] if method == 'Sven' else opts['lw'],
                    label=style.method_label(method))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel(name)
    _shared_legend(fig, **_legend_kw(opts))
    return fig, {'axes': axes}


# ---------------------------------------------------------------------------
# F15: the v3 profile
# ---------------------------------------------------------------------------
def _profile_methods(inp, opts):
    """Step time relative to Adam and peak memory relative to SGD, methods x architectures.

    A heat map rather than bars: it is the one view in which the Gram backends can be read
    against twelve baselines on four architectures at once, and a cell with no measurement
    (OOM / infeasible) says which it was instead of going blank.
    """
    C.set_paper_style(_style_fraction(opts))
    archs = list(opts['archs'] or inp.profile_archs)
    panels = [tuple(p) for p in opts['panels']]
    # full width, no colour bars and the method names only once: at 0.49 x 2 the long
    # method labels plus two colour bars left ~0.2 in per cell and the annotations
    # overlapped into each other
    fig, axes = plt.subplots(1, len(panels),
                             figsize=C.figsize(1, 1, fraction=opts['fraction'],
                                               aspect=opts['aspect']),
                             gridspec_kw={'wspace': opts['wspace']})
    fs = _title_fs(opts)
    for ax, (value, title) in zip(np.atleast_1d(axes), panels):
        ph.plot_heatmap(inp.profiles, ax, value=value, archs=archs,
                        fmt=opts['fmt'])
        ax.set_title(title, fontsize=fs)
        for text in ax.get_xticklabels():
            text.set_fontsize(opts['tick_fontsize'])
        for text in ax.get_yticklabels():
            text.set_fontsize(opts['tick_fontsize'])
    if opts['share_method_labels'] and len(panels) > 1:
        axes[1].set_yticklabels([])
    return fig, {'axes': axes, 'archs': archs}


def _profile_k_sweep(inp, opts):
    """Step time against the rank cap ``k``: flat, because the Gram solve is a ``B x B``
    eigendecomposition whatever ``k`` is.

    This is the picture behind the cost claim of section 2.3: ``k`` and ``rtol`` decide how
    many eigenpairs are INVERTED, not how much work the step does.
    """
    C.set_paper_style(_style_fraction(opts))
    archs = [a for a in (opts['archs'] or inp.profile_archs)
             if _has_study(inp.profiles, a, 'k')]
    ncols = int(opts['ncols']) if opts['ncols'] else (2 if len(archs) <= 4 else 3)
    nrows = int(np.ceil(len(archs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=C.figsize(ncols, nrows, fraction=opts['fraction'],
                                               aspect=opts['aspect'],
                                               extra_h=opts['extra_h']))
    fs = _title_fs(opts)
    for ax, arch in zip(axes.ravel(), archs):
        ph.plot_sweep(inp.profiles, arch, 'k', 'k_fraction', ax, value='step_ms',
                      methods=_methods_with_data(inp.profiles, arch, 'k',
                                                 list(opts['methods'])))
        ax.set_title(ph.ARCH_TITLES.get(arch, arch), fontsize=fs)
        if opts['headroom']:
            _headroom(ax, opts['headroom'])
    for ax in axes.ravel()[len(archs):]:
        ax.set_visible(False)
    _drop_axes_legends(fig)
    _shared_legend(fig, **_legend_kw(opts))
    return fig, {'axes': axes, 'archs': archs}


def _profile_batchsize(inp, opts):
    """Step time (top) and peak memory (bottom) against the batch size ``B``.

    ``B`` is the side of the Gram matrix and the number of Jacobian rows, so it is the one
    knob that moves both; the dotted line on the memory panels is the analytic Jacobian
    size, i.e. what the ``full`` capture must materialise.
    """
    C.set_paper_style(_style_fraction(opts))
    allowed = set(opts['archs'] or SWEEP_ARCHS)
    archs = [a for a in inp.profile_archs
             if a in allowed and _has_study(inp.profiles, a, 'batch_size')]
    fraction = (opts['fraction'] if opts['fraction'] is not None
                else 1.0 / max(len(archs), 1))
    fig, axes = plt.subplots(2, len(archs), squeeze=False,
                             figsize=C.figsize(len(archs), 2, fraction=fraction,
                                               aspect=opts['aspect'],
                                               extra_h=opts['extra_h']))
    fs = _title_fs(opts)
    analytic = 'analytic_jac_mb' if opts['analytic'] else None
    for j, arch in enumerate(archs):
        methods = _methods_with_data(inp.profiles, arch, 'batch_size',
                                     list(opts['methods']))
        ph.plot_sweep(inp.profiles, arch, 'batch_size', 'B', axes[0][j], value='step_ms',
                      methods=methods)
        ph.plot_sweep(inp.profiles, arch, 'batch_size', 'B', axes[1][j], value='peak_mb',
                      methods=methods, analytic=analytic)
        axes[0][j].set_title(ph.ARCH_TITLES.get(arch, arch), fontsize=fs)
        axes[0][j].set_xlabel('')          # the column's x axis is labelled once, below
        axes[1][j].set_title('')
        for ax in (axes[0][j], axes[1][j]):
            if opts['headroom']:
                _headroom(ax, opts['headroom'])
            for text in ax.get_yticklabels():
                text.set_fontsize(opts['ytick_fontsize'])
    _drop_axes_legends(fig)
    _shared_legend(fig, **_legend_kw(opts))
    return fig, {'axes': axes, 'archs': archs}


def _profile_scaling(inp, opts):
    """Step time and peak memory against the parameter count, over the two width sweeps.

    The MLP sweep runs to a width no ResNet reaches; the nanoGPT sweep is the transformer
    version of the same question.  Read with the fitted exponents in the App. Q table:
    the Gram backends grow with the CAPTURE, not with the decomposition.
    """
    C.set_paper_style(_style_fraction(opts))
    widths = list(opts['archs'] or inp.profile_widths)
    if not widths:
        return None, {'axes': None, 'archs': []}
    fig, axes = plt.subplots(2, len(widths), squeeze=False,
                             figsize=C.figsize(len(widths), 2, fraction=opts['fraction'],
                                               aspect=opts['aspect'],
                                               extra_h=opts['extra_h']))
    fs = _title_fs(opts)
    analytic = 'analytic_jac_mb' if opts['analytic'] else None
    for j, arch in enumerate(widths):
        methods = _methods_with_data(inp.profiles, arch, 'width', list(opts['methods']))
        ph.plot_sweep(inp.profiles, arch, 'width', 'n_params', axes[0][j],
                      value='step_ms', methods=methods)
        ph.plot_sweep(inp.profiles, arch, 'width', 'n_params', axes[1][j],
                      value='peak_mb', methods=methods, analytic=analytic)
        for ax in (axes[0][j], axes[1][j]):
            ax.set_xscale('log')
            if opts['headroom']:
                _headroom(ax, opts['headroom'])
        axes[0][j].set_title(ph.ARCH_TITLES.get(arch, arch), fontsize=fs)
        axes[1][j].set_title('')
    _drop_axes_legends(fig)
    _shared_legend(fig, **_legend_kw(opts))
    return fig, {'axes': axes, 'archs': widths}


def _profile_phases(inp, opts):
    """Capture against solve+apply, per Sven backend and architecture.

    The whole cost argument in one panel: the hatched part (one ``B x B``
    eigendecomposition plus the update) barely moves between architectures, while the
    solid part -- getting the Jacobian rows out of autograd -- is what grows.
    """
    C.set_paper_style(_style_fraction(opts))
    archs = list(opts['archs'] or inp.profile_archs)
    fig, ax = plt.subplots(figsize=C.figsize(1, 1, fraction=opts['fraction'],
                                             aspect=opts['aspect']))
    ph.plot_phase_bars(inp.profiles, ax, archs=archs)
    for text in ax.get_xticklabels():
        text.set_fontsize(opts['xtick_fontsize'])
    ax.set_ylabel('Step time (ms)')
    ax.legend(fontsize=opts['legend_fontsize'], ncol=opts['legend_ncol'],
              loc=opts['legend_loc'])
    # which group is which architecture: plot_phase_bars advances x by 1 per drawn bar
    # and by a further 0.8 between architectures, so the group centres follow from the
    # counts it drew (its own tick positions are the bars, not the groups)
    d = inp.profiles[(inp.profiles.study == 'methods')
                     & (inp.profiles.family == 'sven')
                     & inp.profiles.status.isin(ph.TIMED)]
    x = 0.0
    for arch in archs:
        n = sum(1 for m in ph.SVEN
                if len(d[(d.arch == arch) & (d.method == m)]))
        if n and opts['arch_labels']:
            ax.text(x + (n - 1) / 2.0, 1.02, ph.ARCH_TITLES.get(arch, arch),
                    transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                    fontsize=opts['arch_label_fontsize'])
        x += n + opts['group_gap']
    return fig, {'axes': ax, 'archs': archs}


#: architectures the chunking front is drawn for, in preference order.  The ResNet is the
#: one where it matters: chunking the capture is what makes 23 GB fit in a third of that.
PARETO_ARCHS = ('cifar_resnet18', 'nanogpt', 'mnist', 'toy_1d')


def _chunk_front_legend():
    """``[(handle, label)]`` for :func:`_plot_chunk_front`'s own marks."""
    from matplotlib.lines import Line2D
    return [(Line2D([], [], color=ph.color('gram_chunked'), marker='D', ms=3.2, lw=1.2),
             'Gram, chunked (front)'),
            (Line2D([], [], color=ph.color('gram_full'), marker='s', ms=4, ls='none'),
             ph.label('gram_full')),
            (Line2D([], [], color=ph.color('gram_hooks'), marker='o', ms=4, ls='none'),
             ph.label('gram_hooks')),
            (Line2D([], [], color=ph.color('Adam'), marker='*', ms=5, ls='none'),
             'Adam')]


def _plot_chunk_front(ax, df, arch, annotate=True, grid=True):
    """Peak memory against step time along the chunk-fraction sweep, endpoints named.

    :func:`profile_helpers.plot_pareto` annotates every point, which at print size writes
    five overlapping labels on top of a nearly vertical front.  Here only the two ends are
    named -- the whole content of the trade-off is what the cheapest-memory end costs in
    time -- and the other backends are single reference markers.
    """
    d = df[(df.arch == arch) & (df.study == 'chunk_fraction')
           & df.status.isin(ph.TIMED)].sort_values('chunk_fraction')
    if len(d):
        ax.plot(d['peak_mb'], d['step_ms'], '-D', ms=3.2, lw=1.2,
                color=ph.color('gram_chunked'))
        for _, r in ((d.iloc[[0, -1]] if len(d) > 1 else d) if annotate
                     else d.iloc[:0]).iterrows():
            groups = _num(r.get('n_groups'))
            note = (f"$f$={_num(r['chunk_fraction']):g}"
                    + (f'\n({int(groups)} grp)' if np.isfinite(groups) else ''))
            ax.annotate(note, (r['peak_mb'], r['step_ms']), fontsize=5.4,
                        textcoords='offset points', xytext=(4, 2), color='0.25')
    ref = df[(df.arch == arch) & (df.study == 'methods') & df.status.isin(ph.TIMED)]
    for method, marker, size in (('gram_full', 's', 4.5), ('gram_hooks', 'o', 4.5),
                                 ('classic', '^', 4.5), ('Adam', '*', 6.0)):
        r = ref[ref.method == method]
        if len(r):
            ax.plot(r['peak_mb'], r['step_ms'], marker, ms=size,
                    color=ph.color(method), mec='k', mew=0.4)
    from matplotlib.ticker import NullFormatter
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('Peak GPU memory (MB)')
    ax.set_ylabel('Step time (ms)')
    if grid:
        ax.grid(True, which='both', ls='--', alpha=0.35)
    return d


def _profile_pareto(inp, opts):
    """The chunked capture's memory / time trade-off, against the other backends.

    Chunking the Jacobian capture into ``n`` groups is the knob that makes a model fit at
    all; the front shows what each factor of memory costs in time, and where the hooks
    capture sits relative to all of it.
    """
    C.set_paper_style(_style_fraction(opts))
    have = [a for a in inp.profile_archs if _has_study(inp.profiles, a, 'chunk_fraction')]
    # two panels at most: the figure is included at \linewidth, so a third would shrink
    # every panel below the size its fonts were chosen for
    prefer = list(opts['archs'] or PARETO_ARCHS)
    archs = ([a for a in prefer if a in have] or list(have))[:int(opts['max_panels'])]
    fig, axes = plt.subplots(1, len(archs), squeeze=False,
                             figsize=C.figsize(len(archs), 1, fraction=opts['fraction'],
                                               aspect=opts['aspect'],
                                               extra_h=opts['extra_h']))
    fs = _title_fs(opts)
    for ax, arch in zip(axes[0], archs):
        _plot_chunk_front(ax, inp.profiles, arch, annotate=opts['annotate_ends'],
                          grid=opts['front_grid'])
        ax.set_title(ph.ARCH_TITLES.get(arch, arch), fontsize=fs)
    _drop_axes_legends(fig)
    marks = _chunk_front_legend()
    C.legend_below(fig, [h for h, _l in marks], [l for _h, l in marks],
                   bbox_to_anchor=None, **_legend_kw(opts))
    return fig, {'axes': axes, 'archs': archs}


# ---------------------------------------------------------------------------
# The registry: the figure name IS the PDF stem under figures_iclr/large/
# ---------------------------------------------------------------------------
#: the geometry every panelled figure shares, with the CIFAR pair's values
_CIFAR_CURVE_DEFAULTS = {
    'versus': 'epoch',          # 'epoch' (confirmation pass) or 'time' (timing pass)
    'scans': None,              # which CIFAR scans, one row each (default: both)
    'panels': None,             # which curve quantities, one column each
    'methods': None,            # which methods (default: every one the pass has)
    'band': True,               # the seed band
    'seed_note': True,          # name a method drawn on fewer seeds than the rest
    'note_xy_acc': [0.02, 0.03],
    'note_xy_loss': [0.35, 0.03],
    'lw': 0.85, 'lw_time': 0.85, 'lw_sven': 1.4,
    'fraction': 0.49, 'aspect': 0.85, 'extra_h': 0.62, 'style_fraction': None,
    'title_fontsize': None,
    'legend_ncol': 4, 'legend_loc': 'outside lower center', 'legend_fontsize': None,
    'legend_band': True,        # the seed-band entry in the shared legend
}

FIGURE_SPECS = figspec.check_defaults({
    'cifar_curves_epoch': figspec.FigureSpec(
        draw=_cifar_curves,
        doc='F7a: CIFAR-10/ResNet18, validation loss and test accuracy vs EPOCH '
            '(confirmation seeds), one row per objective.',
        defaults=dict(_CIFAR_CURVE_DEFAULTS, versus='epoch'),
    ),
    'cifar_curves_time': figspec.FigureSpec(
        draw=_cifar_curves,
        doc='F7b: the same curves vs standalone TRAINING TIME (the {scan}_timing pass), '
            'on a log time axis with the untrained point dropped.',
        defaults=dict(_CIFAR_CURVE_DEFAULTS, versus='time'),
    ),
    'cifar_gap': figspec.FigureSpec(
        draw=_cifar_gap,
        doc='F7: the train-vs-validation gap per method, one panel per objective '
            '(``diagonal`` draws the no-gap line, ``marker_size`` is the scatter size).',
        defaults={
            'scans': None, 'diagonal': True, 'marker_size': 16,
            'pad_lo': 0.75, 'pad_hi': 1.35,          # axis padding around the data
            'fraction': 0.49, 'aspect': 0.92, 'extra_h': 0.60, 'style_fraction': None,
            'title_fontsize': None,
            'legend_ncol': 4, 'legend_loc': 'outside lower center',
            'legend_fontsize': None, 'legend_band': False,
        },
    ),
    'cifar_landscape': figspec.FigureSpec(
        draw=_cifar_landscape,
        doc="F7: Sven's (lr, rtol) landscape per k and per objective, one colour scale "
            'per objective.',
        defaults={
            'scans': None, 'cbar_label': '$\\log_{10}$ val loss',
            'fraction': 0.49, 'aspect': 0.95, 'style_fraction': None,
            'title_fontsize': None,
        },
    ),
    'cifar_cost': figspec.FigureSpec(
        draw=_cifar_cost,
        doc='F7: seconds per epoch and peak GPU memory from the standalone timing pass, '
            'one row per objective.',
        defaults={
            'scans': None, 'log': True,              # log axes (Sven is 10x+ away)
            'xtick_fontsize': 5.6,
            'fraction': 0.49, 'aspect': 0.95, 'style_fraction': None,
            'title_fontsize': None,
            'legend_ncol': 3, 'legend_loc': 'outside lower center',
            'legend_fontsize': None,
        },
    ),
    'cifar_rank_used': figspec.FigureSpec(
        draw=_cifar_rank_used,
        doc='F7: the rank Sven actually inverted per epoch (one line per seed) against '
            'its k cap.',
        defaults={
            'scans': None, 'lw': 0.9, 'alpha': 0.55, 'k_color': 'C3',
            'ylim_pad': 1.08,
            'fraction': 0.49, 'aspect': 0.85, 'style_fraction': None,
            'title_fontsize': None,
            'legend_fontsize': 5.8, 'legend_loc': 'lower left',
        },
    ),
    'fig5_quality': figspec.FigureSpec(
        draw=_fig5_quality,
        doc='F7c: loss and accuracy against the ACTUAL parameter fraction, every Sven '
            'configuration on disk (``styles`` is one [linestyle, marker] per '
            'configuration, the selected one first).',
        defaults={
            'styles': [list(p) for p in FIG5_STYLES],
            'chance_line': True,                     # the 10-class chance accuracy
            'seed_note': True, 'note_fontsize': 5.2,
            'xlim_pad_lo': 0.85, 'xlim_pad_hi': 1.5,
            'legend_marker_size': 3.0,
            'fraction': 0.49, 'aspect': 0.9, 'extra_h': 0.52, 'style_fraction': None,
            'title_fontsize': None,
            'legend_ncol': 3, 'legend_loc': 'outside lower center',
            'legend_fontsize': None,
        },
    ),
    'fig5_cost': figspec.FigureSpec(
        draw=_fig5_cost,
        doc="F7d: peak memory and time per step against the actual parameter fraction; "
            "``profile_overlay`` overlays the profile's own param_fraction study and "
            "``profile_step_times`` (None = only when the root is v3) its step times.",
        defaults={
            'styles': [list(p) for p in FIG5_STYLES],
            'profile_overlay': True, 'profile_step_times': None,
            'fraction': 0.49, 'aspect': 0.9, 'extra_h': 0.42, 'style_fraction': None,
            'title_fontsize': None,
            'legend_ncol': 2, 'legend_loc': 'outside lower center',
            'legend_fontsize': None,
        },
    ),
    'nanogpt': figspec.FigureSpec(
        draw=_nanogpt,
        doc='F11b: nanoGPT vs epoch, vs standalone wall time and vs the learning rate, '
            'with the selection of record ringed (``mark_selected``).',
        defaults={
            'methods': None, 'band': True, 'mark_selected': True, 'ring_size': 44,
            'lw': 0.9, 'lw_time': 0.85, 'lw_sven': 1.4,
            'headroom': 0.04,                        # room above the highest line; 0 = off
            'fraction': 0.32, 'aspect': 0.95, 'extra_h': 0.42, 'style_fraction': None,
            'title_fontsize': None,
            'legend_ncol': 4, 'legend_loc': 'outside lower center',
            'legend_fontsize': None, 'legend_band': True,
        },
    ),
    'gpt2': figspec.FigureSpec(
        draw=_gpt2,
        doc='F11: GPT-2 small, validation loss vs optimisation step -- the best lr per '
            "method, and all of Sven's lrs on the ``cmap`` ramp.  One seed, no band.",
        defaults={
            'lw': 0.9, 'lw_sven': 1.4, 'lr_lw': 1.1,
            'cmap': 'viridis', 'cmap_lo': 0.08, 'cmap_span': 0.82,
            'fraction': 0.49, 'aspect': 0.9, 'extra_h': 0.30, 'style_fraction': None,
            'title_fontsize': None,
            'legend_fontsize': 5.4, 'legend_ncol': 1,   # per panel, not a shared legend
        },
    ),
    'gpt2_lr': figspec.FigureSpec(
        draw=_gpt2_lr,
        doc='F11: GPT-2 final loss against the learning rate, one line per method; '
            '``panels`` is [column, axis label] per panel.',
        defaults={
            'panels': [['val', 'Validation loss'], ['test', 'Test loss']],
            'marker_size': 2.6, 'lw': 0.9, 'lw_sven': 1.4,
            'fraction': 0.49, 'aspect': 0.9, 'extra_h': 0.26, 'style_fraction': None,
            'legend_ncol': 5, 'legend_loc': 'outside lower center',
            'legend_fontsize': None,
        },
    ),
    'profile_methods': figspec.FigureSpec(
        draw=_profile_methods,
        doc='F15: step time relative to Adam and peak memory relative to SGD, methods x '
            'architectures, as a heat map; ``panels`` is [column, title] per panel.',
        defaults={
            'archs': None,
            'panels': [['rel_time', 'Step time / Adam'],
                       ['rel_mem', 'Peak memory / SGD']],
            'fmt': '{:.3g}', 'tick_fontsize': 6.5, 'wspace': 0.04,
            'share_method_labels': True,             # method names on the left panel only
            'fraction': 1.0, 'aspect': 0.82, 'style_fraction': None,
            'title_fontsize': None,
        },
    ),
    'profile_k_sweep': figspec.FigureSpec(
        draw=_profile_k_sweep,
        doc='F15: step time against the rank cap k, one panel per architecture that has '
            'a k sweep (``ncols`` None = 2 panels wide up to four architectures).',
        defaults={
            'archs': None, 'methods': list(SVEN_BACKENDS), 'ncols': None,
            'headroom': 0.06,
            'fraction': 0.49, 'aspect': 0.82, 'extra_h': 0.34, 'style_fraction': None,
            'title_fontsize': None,
            'legend_ncol': 4, 'legend_loc': 'outside lower center',
            'legend_fontsize': None,
        },
    ),
    'profile_batchsize': figspec.FigureSpec(
        draw=_profile_batchsize,
        doc='F15: step time (top) and peak memory (bottom) against the batch size B; '
            '``fraction`` None spreads the columns over the full line width.',
        defaults={
            'archs': list(SWEEP_ARCHS), 'methods': list(SWEEP_METHODS),
            'analytic': True,                        # the analytic Jacobian-size line
            'headroom': 0.06, 'ytick_fontsize': 5.8,
            'fraction': None, 'aspect': 0.95, 'extra_h': 0.46, 'style_fraction': 0.49,
            'title_fontsize': 6.4,
            'legend_ncol': 5, 'legend_loc': 'outside lower center',
            'legend_fontsize': None,
        },
    ),
    'profile_scaling': figspec.FigureSpec(
        draw=_profile_scaling,
        doc='F15: step time and peak memory against the parameter count over the width '
            'sweeps (skipped when the profile root has none).',
        defaults={
            'archs': None, 'methods': list(SWEEP_METHODS),
            'analytic': True, 'headroom': 0.06,
            'fraction': 0.49, 'aspect': 0.82, 'extra_h': 0.40, 'style_fraction': None,
            'title_fontsize': None,
            'legend_ncol': 5, 'legend_loc': 'outside lower center',
            'legend_fontsize': None,
        },
    ),
    'profile_phases': figspec.FigureSpec(
        draw=_profile_phases,
        doc='F15: capture against solve+apply per Sven backend and architecture, with '
            'the architecture names above their bar groups (``arch_labels``).',
        defaults={
            'archs': None, 'xtick_fontsize': 5.6,
            'arch_labels': True, 'arch_label_fontsize': 6.4, 'group_gap': 0.8,
            'fraction': 1.0, 'aspect': 0.46, 'style_fraction': None,
            'title_fontsize': None,
            'legend_ncol': 2, 'legend_loc': 'upper left', 'legend_fontsize': 6.5,
        },
    ),
    'profile_pareto': figspec.FigureSpec(
        draw=_profile_pareto,
        doc="F15: the chunked capture's memory / time front against the other backends; "
            '``archs`` is the preference order and ``max_panels`` how many fit.',
        defaults={
            'archs': list(PARETO_ARCHS), 'max_panels': 2,
            'annotate_ends': True, 'front_grid': True,
            'fraction': 0.49, 'aspect': 0.95, 'extra_h': 0.40, 'style_fraction': None,
            'title_fontsize': None,
            'legend_ncol': 4, 'legend_loc': 'outside lower center',
            'legend_fontsize': None,
        },
    ),
})

#: which analysis functions produced each figure (written into its provenance)
FIG_FUNCTIONS = {
    'cifar_curves_epoch': ('large_figs.selected_runs_by_method', 'large_figs.plot_curves',
                           'large_figs.curve_band', 'large_figs.seed_note'),
    'cifar_curves_time': ('large_figs.selected_runs_by_method', 'large_figs.plot_curves',
                          'large_figs.curve_axis', 'large_figs.seed_note'),
    'cifar_gap': ('headline.confirmation_table', 'large_figs.generalisation_gaps'),
    'cifar_landscape': ('large_figs.sven_grid', 'large_figs.mark_selected',
                        'large_figs.plot_sven_landscape', 'large_figs.landscape_scale'),
    'cifar_cost': ('headline.efficiency_table', 'large_figs.plot_cost_bars'),
    'cifar_rank_used': ('large_figs.sven_rank_used',),
    'fig5_quality': ('large_figs.paramfrac_groups', 'large_figs.paramfrac_table',
                     'large_figs.plot_paramfrac_quality'),
    'fig5_cost': ('large_figs.paramfrac_groups', 'large_figs.plot_paramfrac_cost',
                  'profile_helpers.load_profiles'),
    'nanogpt': ('large_figs.selected_runs_by_method', 'large_figs.plot_curves',
                'reviewer_figs.arm_table', 'reviewer_figs.plot_arm',
                'headline.selection_methods'),
    'gpt2': ('large_figs.gpt2_frame', 'large_figs.gpt2_best_per_method',
             'large_figs.plot_gpt2_curves', 'large_figs.gpt2_step_curve'),
    'gpt2_lr': ('large_figs.gpt2_lr_table',),
    'profile_methods': ('profile_helpers.load_profiles', 'profile_helpers.plot_heatmap',
                        'profile_helpers.add_relative'),
    'profile_k_sweep': ('profile_helpers.plot_sweep',),
    'profile_batchsize': ('profile_helpers.plot_sweep',),
    'profile_scaling': ('profile_helpers.plot_sweep', 'profile_helpers.scaling_table'),
    'profile_phases': ('profile_helpers.plot_phase_bars',),
    'profile_pareto': ('profile_helpers.plot_pareto',),
}

#: ``meta`` keys that are drawing state rather than provenance: ``axes`` is what the
#: notebook hands the user to edit and ``options`` is what ``figspec.draw_figure`` adds.
#: Everything else a draw function returns is a fact about the inputs and belongs in the
#: sidecar, which is how ``build()`` wrote it before the split.
_META_NOT_PROVENANCE = ('axes', 'options', 'provenance')


def _provenance_for(name):
    """The :func:`common.provenance` record ``name``'s figure is saved with.

    This was inline in :func:`build`; as a :class:`figspec.FigureSpec` callable the
    notebook's save path writes the same sidecar as the CLI, which is the whole point of
    there being one save path.  A profile figure (and ``fig5_cost``, which overlays the
    profile) records the root it read.
    """
    def provenance(inp, meta):
        extra = {k: v for k, v in (meta or {}).items()
                 if k not in _META_NOT_PROVENANCE}
        return C.provenance(functions=FIG_FUNCTIONS.get(name, ()),
                            scans=_figure_scans(name),
                            reads=([inp.profile_info['root']]
                                   if 'profile' in name or name == 'fig5_cost' else []),
                            provisional=inp.provisional, **extra)
    return provenance


# the record depends on the figure's NAME (its functions, its scans, whether it reads a
# profile root), which a FigureSpec only learns when it is registered
for _name, _spec in FIGURE_SPECS.items():
    _spec.provenance = _provenance_for(_name)
del _name, _spec


#: one :class:`Inputs` per results root, so a notebook that draws six figures loads each
#: scan once (and ``build()`` refreshes it, because a CLI build must re-read the disk)
_CONTEXTS: dict = {}


def context(root=None, reload=False):
    """The :class:`Inputs` this module's draw functions take, cached per results root."""
    key = None if root is None else str(root)
    if reload or key not in _CONTEXTS:
        _CONTEXTS[key] = Inputs(root)
    return _CONTEXTS[key]


# ---------------------------------------------------------------------------
# T16: CIFAR-10 / ResNet18, both objectives
# ---------------------------------------------------------------------------
def table_cifar(inp):
    """Both CIFAR scans, 11 methods: quality, the train-eval diagnosis and the cost.

    Quality is the confirmation pass (5 fresh seeds); the cost columns are the standalone
    timing pass, which is the only honest wall clock.  ``fin/att`` is printed for both, so
    a row that rests on fewer runs than the scan intended says so.
    """
    rows, midrules = [], set()
    for scan, title in CIFAR.items():
        tbl = inp.conf[scan]
        eff = inp.eff[scan].set_index('method')
        runs = inp.runs_conf[scan]
        if rows:
            midrules.add(len(rows))
        for i, r in tbl.iterrows():
            method = r['method']
            e = eff.loc[method] if method in eff.index else None
            # the validation accuracy is a per-epoch curve, not a column of the
            # confirmation table: take the last entry of the same runs' seed mean
            val_acc = np.nan
            if method in runs:
                mean, _lo, _hi, _n = lf.curve_band(runs[method], 'val_acc')
                if mean is not None:
                    val_acc = float(mean[-1])
            rows.append({
                'Objective': title if i == 0 else '',
                'Method': r.get('display', method),
                'Configuration': r['config'],
                'Val': C.fmt_pm(r['val_conf'], r['val_conf_std'], 4),
                'Test': C.fmt_pm(r['test_conf'], r['test_conf_std'], 4),
                'Val acc': C.fmt_pct(val_acc),
                'Test acc': C.fmt_pct(r.get('acc_conf')),
                'Train (eval)': C.fmt_sig(r.get('treval_conf'), 4),
                's/epoch': C.fmt_sig(_num(e['epoch_s']) if e is not None else np.nan),
                'ms/step': C.fmt_sig(_num(e['ms_per_step']) if e is not None else np.nan),
                'Peak MB': C.fmt_int(_num(e['peak_gpu_mem_mb'])
                                     if e is not None else np.nan),
                'fin/att': C.fmt_counts(r['fin_conf'], r['att_conf']),
            })
    notes = ['quality: 5 fresh confirmation seeds of the selected configuration; cost: '
             'the standalone timing pass (one run per GPU, logging off), which re-runs '
             'the tuning seeds',
             'every optimizer ran with batch-statistics BatchNorm']
    if inp.scan_provisional('cifar10_resnet_ce_scan'):
        notes.append('PROVISIONAL: the cross-entropy re-selection is still landing')
    return rows, {'midrules': sorted(midrules), 'notes': notes,
                  'functions': ('headline.confirmation_table', 'headline.efficiency_table',
                                'large_figs.optimisation_view', 'large_figs.curve_band')}


# ---------------------------------------------------------------------------
# T15: Fig 5
# ---------------------------------------------------------------------------
def table_fig5(inp):
    """Fig 5's table: per parameter fraction, quality and cost, per configuration.

    ``f (actual)`` is the fraction the mask really covered, which is what the figure's x
    axis uses.  The cost columns come from the same runs as the quality columns (this scan
    has no standalone timing pass), so they are comparable across fractions and not
    against a standalone number.
    """
    rows, midrules = [], set()
    for g in inp.fig5_groups:
        if rows:
            midrules.add(len(rows))
        tbl, cfg = g['table'], g['cfg']
        for i, r in tbl.iterrows():
            # a single-seed point has no spread to report: `paramfrac_table` records 0.0
            # for it (one value, ddof=1 undefined), and printing "0.524 +/- 0" would read
            # as a measured zero spread rather than as one run
            many = int(_num(r['finished'])) > 1
            rows.append({
                'Configuration': (f'{cfg["label"]} ({g["role"]})' if i == 0 else ''),
                'f': C.fmt_sig(r['param_fraction'], 2),
                'f actual': C.fmt_sig(r['actual'], 3),
                'Val': C.fmt_pm(r['val'], r['val_std'] if many else np.nan, 4),
                'Test': C.fmt_pm(r['test'], r['test_std'] if many else np.nan, 4),
                'Val acc': C.fmt_pct(r['val_acc']),
                'Test acc': C.fmt_pct(r['test_acc']),
                'Peak MB': C.fmt_int(r['peak_mem_mb']),
                'ms/step': C.fmt_sig(r['ms_per_step'], 4),
                'ms/step x full': C.fmt_sig(r.get('ms_per_step_vs_full'), 3),
                'fin/att': C.fmt_counts(r['finished'], r['attempted']),
                'diverged': C.fmt_int(r['n_diverged']),
            })
    sel = inp.fig5_selected
    notes = [lf.MASK_NOTE.replace('$f$', '$f$'),
             f'one fixed Sven configuration per block, not re-tuned per $f$; '
             f'{sel["cfg"]["n_seeds"]} seed(s) per point at the selected configuration',
             f'chance accuracy is {100 * CIFAR_CHANCE_ACC:.0f}\\% (10 classes)']
    if inp.fig5_incomplete:
        notes.append('PROVISIONAL: the re-run at the selected configuration is still '
                     'landing')
    return rows, {'midrules': sorted(midrules), 'notes': notes,
                  'functions': ('large_figs.paramfrac_groups', 'large_figs.paramfrac_table',
                                'large_figs.paramfrac_view', 'large_figs.paramfrac_config')}


# ---------------------------------------------------------------------------
# T17: the transformers
# ---------------------------------------------------------------------------
def table_nanogpt(inp):
    """nanoGPT: the confirmation table with perplexity, the paired difference against
    Sven, and the standalone cost."""
    tbl = inp.conf[NANOGPT]
    eff = inp.eff[NANOGPT].set_index('method')
    pair = inp.paired_nanogpt.set_index('method') if len(inp.paired_nanogpt) else None
    rows = []
    for _, r in tbl.iterrows():
        e = eff.loc[r['method']] if r['method'] in eff.index else {}
        val = _num(r['val_conf'])
        test = _num(r['test_conf'])
        p = pair.loc[r['method']] if (pair is not None and r['method'] in pair.index) \
            else None
        rows.append({
            'Method': r.get('display', r['method']),
            'Configuration': r['config'],
            'Val': C.fmt_pm(val, r['val_conf_std'], 4),
            'Val ppl': C.fmt_sig(np.exp(val) if np.isfinite(val) else np.nan, 4),
            'Test': C.fmt_pm(test, r['test_conf_std'], 4),
            'Test ppl': C.fmt_sig(np.exp(test) if np.isfinite(test) else np.nan, 4),
            'Sven - this': ('--' if p is None else C.fmt_sig(p['mean'], 2)),
            '95% t-interval': ('--' if p is None
                               else C.fmt_ci(p['ci_low'], p['ci_high'], 2)),
            's/epoch': C.fmt_sig(_num(e.get('epoch_s')) if len(e) else np.nan),
            'ms/step': C.fmt_sig(_num(e.get('ms_per_step')) if len(e) else np.nan),
            'Peak MB': C.fmt_int(_num(e.get('peak_gpu_mem_mb')) if len(e) else np.nan),
            'fin/att': C.fmt_counts(r['fin_conf'], r['att_conf']),
        })
    notes = ['quality on 5 fresh confirmation seeds; cost from the standalone timing pass',
             'Sven $-$ this: the paired per-seed difference in final validation loss '
             '(negative = Sven lower); the interval is a Student-$t$ interval on the mean '
             'difference, not the seed band']
    return rows, {'notes': notes,
                  'functions': ('headline.confirmation_table', 'headline.paired_vs_sven',
                                'headline.efficiency_table')}


def table_nanogpt_lr(inp):
    """nanoGPT lr sensitivity: the best configuration per (method, lr) on the TUNING scan.

    One block per method, ordered by learning rate, with the selection of record starred.
    The loss columns are seed means over the scan's 5 seeds (the scan clock is co-tenanted,
    so no wall time is printed here -- T17a has the standalone cost).
    """
    best = inp.nanogpt_lr
    if not len(best):
        return [], {}
    picked = selected_lrs(inp)
    d = best.assign(_key=best['method'].map(hl.method_key))
    order = hl.method_order(sorted(set(d['_key'])))
    rows, midrules = [], set()
    for method in order:
        sub = d[d['_key'] == method].sort_values('lr')
        if not len(sub):
            continue
        if rows:
            midrules.add(len(rows))
        first = True
        for _, r in sub.iterrows():
            star = '$\\star$' if np.isclose(_num(r['lr']),
                                            picked.get(method, np.nan)) else ''
            rows.append({
                'Method': (hl.display_name(method) if first else ''),
                'lr': C.fmt_sig(r['lr'], 2) + (f' {star}' if star else ''),
                'Val': C.fmt_pm(r['final_val_loss'], r.get('final_val_loss_std'), 4),
                'Val ppl': C.fmt_sig(r.get('val_ppl'), 4),
                'Test': C.fmt_sig(r.get('final_test_loss'), 4),
                'Train (eval)': C.fmt_sig(r.get('final_train_eval'), 4),
                'Rank used': C.fmt_sig(r.get('rank_eff'), 3),
                'Peak MB': C.fmt_int(r.get('peak_gpu_mem_mb')),
                'fin/att': C.fmt_counts(r.get('finished'), r.get('attempted')),
            })
            first = False
    notes = ['best configuration per (method, learning rate), selected on validation '
             'loss over the tuning scan\'s 5 seeds; $\\star$ = the selection of record',
             'Sven\'s grid is one point wide in $k$ and \\texttt{rtol} on this scan, so '
             'its column IS its learning-rate sensitivity']
    return rows, {'midrules': sorted(midrules), 'notes': notes,
                  'functions': ('reviewer_figs.arm_table', 'headline.selection_methods',
                                'analysis_helpers.add_derived')}


def table_gpt2(inp):
    """GPT-2 small: the best learning rate per method, with what the run cost.

    One seed and one epoch per cell, so every number is a single run; the table is
    labelled that way and no spread is printed for it.
    """
    best = inp.gpt2_best.sort_values('val')
    wall = pd.to_numeric(best['wall_h'], errors='coerce')
    fastest = float(np.nanmin(wall)) if np.isfinite(wall).any() else np.nan
    leader = _num(best['val'].min())
    rows = []
    for _, r in best.iterrows():
        val = _num(r['val'])
        rows.append({
            'Method': r['display'],
            'Best lr': C.fmt_sig(r['lr'], 2),
            'Val': C.fmt_sig(val, 4),
            'Val ppl': C.fmt_sig(r['val_ppl'], 4),
            'Test': C.fmt_sig(r['test'], 4),
            'vs best': C.fmt_pct((val / leader - 1) if np.isfinite(leader) and leader
                                 else np.nan, signed=True),
            'Wall (h)': C.fmt_sig(r['wall_h'], 3),
            'x fastest': C.fmt_sig(_num(r['wall_h']) / fastest, 3),
            'ms/step': C.fmt_int(r['ms_per_step']),
            'Peak MB': C.fmt_int(r['peak_gpu_mem_mb']),
            'lrs (fin/att)': C.fmt_counts(r['finished'], r['attempted']),
        })
    facts = inp.gpt2.iloc[0]
    notes = [f'one model seed, one epoch of {int(_num(facts["steps_per_epoch"])):,} steps '
             f'at $B={int(_num(facts["batch_size"]))}$, evaluated every '
             f'{int(_num(facts["eval_every_steps"]))} steps: every cell is a single run',
             'the learning rate is selected on validation loss; the test column is an '
             'outcome']
    return rows, {'notes': notes,
                  'functions': ('large_figs.gpt2_frame', 'large_figs.gpt2_best_per_method',
                                'large_figs.gpt2_summary_view', 'large_figs.add_run_cost')}


def table_gpt2_lr(inp):
    """Every (method, learning rate) of the GPT-2 comparison -- the lr-sensitivity grid."""
    rows, midrules = [], set()
    last = None
    for _, r in inp.gpt2_lr.iterrows():
        if last is not None and r['method'] != last:
            midrules.add(len(rows))
        rows.append({
            'Method': r['display'] if r['method'] != last else '',
            'lr': C.fmt_sig(r['lr'], 2),
            'Val': C.fmt_sig(r['val'], 4),
            'Test': C.fmt_sig(r['test'], 4),
            'Wall (h)': C.fmt_sig(r['wall_h'], 3),
            'ms/step': C.fmt_int(r['ms_per_step']),
            'Peak MB': C.fmt_int(r['peak_gpu_mem_mb']),
            'Status': str(r['status']),
        })
        last = r['method']
    return rows, {'midrules': sorted(midrules),
                  'notes': ['one run per row (one seed, one epoch)'],
                  'functions': ('large_figs.gpt2_lr_table',)}


# ---------------------------------------------------------------------------
# T7 / T19: the profile
# ---------------------------------------------------------------------------
def table_profile_methods(inp):
    """Every method at each architecture's set point: step time, memory, phases, spread.

    One block per architecture, Sven's backends first.  ``x Adam`` and ``x SGD`` are the
    ratios :func:`profile_helpers.add_relative` computes at the same architecture, batch
    size and width; ``Capture`` and ``Solve+apply`` exist for the Sven backends only,
    because only they have the two phases.
    """
    rows, midrules = [], set()
    for arch in inp.profile_archs:
        tbl = ph.method_table(inp.profiles, arch)
        if rows:
            midrules.add(len(rows))
        for i, r in tbl.iterrows():
            rows.append({
                'Architecture': ph.ARCH_TITLES.get(arch, arch) if i == 0 else '',
                'Method': str(r['Method']),
                'Step (ms)': C.fmt_sig(r['Step (ms)'], 4),
                'x Adam': C.fmt_sig(r['x Adam'], 3),
                'Peak (MB)': C.fmt_sig(r['Peak mem (MB)'], 4),
                'x SGD': C.fmt_sig(r['x SGD'], 3),
                'Capture (ms)': C.fmt_sig(r['Capture (ms)'], 3),
                'Solve+apply (ms)': C.fmt_sig(r['Solve+apply (ms)'], 3),
                'p90/p10': C.fmt_sig(r['p90/p10'], 3),
                'Note': str(r['Note'] or ''),
            })
    return rows, {'midrules': sorted(midrules),
                  'notes': [inp.profile_note(),
                            'step time is the amortised mean over the last 80\\% of the '
                            'measured steps, truncated to whole 10-step cycles, so a '
                            'periodic refresh is counted the right number of times',
                            'one run per configuration on one data-centre GPU; p90/p10 is '
                            'the spread of the per-step times within that run'],
                  'functions': ('profile_helpers.method_table',
                                'profile_helpers.add_relative',
                                'profile_helpers.load_profiles')}


def table_gram_cost(inp):
    """Sven's four backends at every architecture: where the step time goes, and the
    ``k``-independence check.

    ``Capture share`` is the fraction of the step spent getting the Jacobian rows out of
    autograd; ``k`` spread is the ratio of the largest to the smallest step time over the
    whole ``k`` sweep at that architecture, i.e. the measurement behind "the cost does not
    depend on ``k``".
    """
    prof = inp.profiles
    rows, midrules = [], set()
    for arch in inp.profile_archs:
        block = prof[(prof.arch == arch) & (prof.study == 'methods')]
        ksweep = prof[(prof.arch == arch) & (prof.study == 'k')
                      & prof.status.isin(ph.TIMED)]
        if rows:
            midrules.add(len(rows))
        first = True
        for method in SVEN_BACKENDS:
            r = block[block.method == method]
            if not len(r):
                continue
            r = r.iloc[0]
            ks = ksweep[ksweep.method == method]
            step = _num(r['step_ms'])
            capture = _num(r['capture_ms'])
            lo = _num(ks['step_ms'].min()) if len(ks) else np.nan
            hi = _num(ks['step_ms'].max()) if len(ks) else np.nan
            rows.append({
                'Architecture': ph.ARCH_TITLES.get(arch, arch) if first else '',
                'Backend': ph.label(method),
                'Step (ms)': C.fmt_sig(step, 4),
                'x Adam': C.fmt_sig(r['rel_time'], 3),
                'Capture (ms)': C.fmt_sig(capture, 3),
                'Capture share': C.fmt_pct(capture / step if step else np.nan, 0),
                'Solve+apply (ms)': C.fmt_sig(r['solve_ms'], 3),
                'Peak (MB)': C.fmt_sig(r['peak_mb'], 4),
                # 'x SGD' and 'k range' are deliberately NOT columns here: eleven
                # columns overran \textwidth by ~270 pt and the \resizebox that made
                # the float fit put the body text near 6 pt.  Both are in App. Q's
                # tab:profile_methods and tab:profile_scaling at full size.
                'k spread': C.fmt_sig(hi / lo if lo else np.nan, 3),
            })
            first = False
    notes = [inp.profile_note(),
             'the Gram solve is one $B \\times B$ eigendecomposition whatever $k$ is: the '
             '$k$ spread column is the largest / smallest step time over that '
             'architecture\'s whole $k$ sweep',
             'capture modes: hooks = one weighted backward pass; full J = the dense '
             'per-sample Jacobian; chunked = the same in parameter groups; classic = the '
             'randomized-SVD path no experiment uses',
             'peak memory relative to SGD, and the $k$ range each spread is taken over, '
             'are in Tables~\\ref{tab:profile_methods} and~\\ref{tab:profile_scaling}']
    return rows, {'midrules': sorted(midrules), 'notes': notes,
                  'functions': ('profile_helpers.method_table',
                                'profile_helpers.load_profiles',
                                'profile_helpers.sven_change_table')}


def table_profile_change(inp):
    """v2 -> v3 for Sven's backends: the one table that is allowed to hold two roots.

    It exists because the older profile measured Sven with a per-step
    ``torch.cuda.empty_cache()``, which is a property of the measurement and not of the
    optimizer.  Memory is expected NOT to move (the allocator returns blocks at a
    different time, the live set is the same), which is what makes the step-time column a
    measurement of the fix.
    """
    cmp = inp.profile_compare
    if not len(cmp):
        return [], {'notes': ['no comparison: only one profile root has results'],
                    'functions': ('profile_helpers.compare_profiles',)}
    tbl = ph.sven_change_table(cmp)
    rows = []
    for _, r in tbl.iterrows():
        rows.append({
            'Architecture': ph.ARCH_TITLES.get(r['arch'], r['arch']),
            'Backend': str(r['Method']),
            'Step (ms) v2': C.fmt_sig(r['step ms v2'], 4),
            'Step (ms) v3': C.fmt_sig(r['step ms v3'], 4),
            'Step v3/v2': C.fmt_sig(r['step v3/v2'], 3),
            'Peak (MB) v2': C.fmt_sig(r['peak MB v2'], 4),
            'Peak (MB) v3': C.fmt_sig(r['peak MB v3'], 4),
            'Mem v3/v2': C.fmt_sig(r['mem v3/v2'], 3),
        })
    base = ph.baseline_control(cmp)
    notes = [inp.profile_note()]
    if len(base):
        worst = base.loc[base['median v3/v2'].sub(1).abs().idxmax()]
        notes.append(f'control: the baselines, which never called empty\\_cache, move by a '
                     f'median factor of {float(worst["median v3/v2"]):.2f} at worst '
                     f'({ph.ARCH_TITLES.get(worst["arch"], worst["arch"])}), so the Sven '
                     f'column is the fix and not the machine')
    return rows, {'notes': notes,
                  'functions': ('profile_helpers.compare_profiles',
                                'profile_helpers.sven_change_table',
                                'profile_helpers.baseline_control')}


def table_profile_scaling(inp):
    """How step time and peak memory grow with the parameter count, per method.

    ``exponent`` is the slope of $\\log$ value against $\\log P$ over the upper half of the
    sweep -- the asymptotic regime -- and ``first failure`` is the smallest model at which
    a method had no measurement at all.
    """
    rows, midrules = [], set()
    for arch in inp.profile_widths:
        for value, label in (('step_ms', 'step time'), ('peak_mb', 'peak memory')):
            tbl = ph.scaling_table(inp.profiles, arch, value)
            if rows:
                midrules.add(len(rows))
            for i, r in tbl.iterrows():
                rows.append({
                    'Sweep': (f'{ph.ARCH_TITLES.get(arch, arch)}, {label}'
                              if i == 0 else ''),
                    'Method': str(r['Method']),
                    'At smallest P': str(r['smallest P']),
                    'Largest timed P': str(r['largest timed P']),
                    'At largest': str(r['at largest']),
                    'Growth': C.fmt_sig(r['growth'], 3),
                    'Exponent in P': C.fmt_sig(r['exponent'], 2),
                    'First failure': str(r['first failure'] or ''),
                })
    return rows, {'midrules': sorted(midrules),
                  'notes': [inp.profile_note(),
                            'the exponent is fitted over the upper half of the sweep'],
                  'functions': ('profile_helpers.scaling_table',
                                'profile_helpers.fit_exponent')}


TABLES = (
    ('cifar', table_cifar),
    ('fig5', table_fig5),
    ('transformers', table_nanogpt),
    ('transformers_nanogpt_lr', table_nanogpt_lr),
    ('transformers_gpt2', table_gpt2),
    ('transformers_gpt2_lr', table_gpt2_lr),
    ('profile_methods', table_profile_methods),
    ('gram_cost', table_gram_cost),
    ('profile_v2_v3', table_profile_change),
    ('profile_scaling', table_profile_scaling),
)

TABLE_CAPTIONS = {
    'cifar': 'T16: CIFAR-10 / ResNet18, both objectives, 11 methods.',
    'fig5': 'T15: the parameter-fraction study at ResNet scale (Fig 5, R1 Q2).',
    'transformers': 'T17a: nanoGPT (tiny-shakespeare), confirmation seeds.',
    'transformers_nanogpt_lr': 'T17a2: nanoGPT, every learning rate of the tuning scan.',
    'transformers_gpt2': 'T17b: GPT-2 small, best learning rate per method.',
    'transformers_gpt2_lr': 'T17c: GPT-2 small, every learning rate.',
    'profile_methods': 'T7: the v3 profile, every method at each set point.',
    'gram_cost': 'T19: Sven\'s four backends -- capture, solve, memory, k-independence.',
    'profile_v2_v3': 'T7b: what the corrected profile changed (v2 -> v3).',
    'profile_scaling': 'T7c: growth of step time and memory with the parameter count.',
}


# ---------------------------------------------------------------------------
# G3: the number macros
# ---------------------------------------------------------------------------
def macros(inp):
    """Macro group G3: every number App. D, J, K, L and Q quote.

    Names are ``num<Scan><Method><Quantity>`` with letters only
    (:class:`paper_assets.common.Macros` validates and refuses a non-finite value), and
    each carries the analysis function it came from as a comment in the file.
    """
    book = C.Macros(module='large')
    skipped = []

    def add(name, value, **kw):
        """Record one macro, or SKIP it when the number does not exist.

        :meth:`paper_assets.common.Macros.add` refuses a NaN outright, which is the right
        rule for the file (no ``nan`` may reach the paper) and the wrong one for the
        build: while the CIFAR-CE pass and the Fig-5 re-run are landing, a handful of
        quantities have no value yet, and one of them must not take the other hundred
        macros down with it.  A skipped name is reported at the end of the build, so a
        macro the prose needs and does not have is visible rather than silently ``nan``.
        """
        if value is None \
                or (C.is_number(value) and not C.finite(value)) \
                or (isinstance(value, str) and not str(value).strip()):
            skipped.append(name)
            return None
        return book.add(name, value, **kw)

    def add_pm(base, mean, std, **kw):
        """``\\num<base>`` and ``\\num<base>Std`` -- the mean and the seed spread."""
        add(base, mean, **kw)
        if C.finite(std):
            add(base + 'Std', std, **{**kw, 'sig': 2})
        return base

    # ---- CIFAR, both objectives ----------------------------------------
    for scan, title in CIFAR.items():
        key = SCAN_KEY[scan]
        prov = inp.scan_provisional(scan)
        tbl, eff = inp.conf[scan], inp.eff[scan]
        src = 'headline.confirmation_table'
        sven = _row_of(tbl, 'Sven')
        best = tbl.iloc[0]
        val = pd.to_numeric(tbl['val_conf'], errors='coerce')
        acc = pd.to_numeric(tbl['acc_conf'], errors='coerce')
        treval = pd.to_numeric(tbl['treval_conf'], errors='coerce')
        rank_val, n_meth = _rank(tbl, 'val_conf')
        rank_acc, n_acc = _rank(tbl, 'acc_conf', ascending=False)
        add(f'num{key}NMethods', n_meth, source=src, provisional=prov)
        add(f'num{key}Name', _text(title), source='style.DATASET_TITLES')
        add(f'num{key}BestName', _text(best['display']), source=src, provisional=prov)
        add(f'num{key}BestVal', _num(best['val_conf']), source=src, provisional=prov, sig=4)
        # Two different quantities, and App. J needs both spelled apart: ``BestAcc`` is
        # the FIELD MAXIMUM accuracy (whoever holds it), ``BestNameAcc`` is the accuracy
        # OF the loss-best method named by ``BestName``.  They differ on both CIFAR
        # objectives -- MuonW wins the loss while SOAP (LR) and Muon (CE) hold the top
        # accuracy -- so quoting ``BestAcc`` next to ``BestName`` attributed a number to
        # the wrong method.
        add(f'num{key}BestAcc', 100 * _num(acc.max()), source=src, provisional=prov,
            note='the highest test accuracy in the field, whichever method holds it')
        add(f'num{key}BestNameAcc', 100 * _num(best['acc_conf']), source=src,
            provisional=prov,
            note='the test accuracy of the method with the lowest validation loss '
                 '(num<key>BestName)')
        add(f'num{key}BestNameTrainEval', _num(best['treval_conf']), source=src,
            provisional=prov, sig=4,
            note='the training-subset loss of the loss-best method: on cross-entropy it '
                 'is WORSE than Sven\'s, which is why train-eval cannot diagnose that '
                 'scan as an optimisation failure')
        if C.finite(_num(acc.max())):
            add(f'num{key}BestAccName',
                _text(style.method_label(tbl.loc[acc.idxmax(), 'method'])),
                source=src, provisional=prov,
                note='the method that holds the highest test accuracy')
        if sven is not None:
            add_pm(f'num{key}SvenVal', _num(sven['val_conf']),
                   _num(sven['val_conf_std']), source=src, provisional=prov, sig=4)
            add(f'num{key}SvenTest', _num(sven['test_conf']), source=src, provisional=prov, sig=4)
            add_pm(f'num{key}SvenAcc', 100 * _num(sven['acc_conf']),
                   100 * _num(sven['acc_conf_std']), source=src, provisional=prov)
            add(f'num{key}SvenTrainEval', _num(sven['treval_conf']), source=src,
                provisional=prov, sig=4)
            add(f'num{key}SvenConfig', _text(sven['config']),
                source='headline.config_label', provisional=prov)
            add(f'num{key}SvenFin', _num(sven['fin_conf']), source=src, provisional=prov)
            add(f'num{key}SvenAtt', _num(sven['att_conf']), source=src, provisional=prov)
            add(f'num{key}SvenRank', rank_val, source=src, provisional=prov)
            add(f'num{key}SvenRankAcc', rank_acc, source=src, provisional=prov)
            add(f'num{key}SvenGapRel',
                100 * (_num(sven['val_conf']) / _num(best['val_conf']) - 1),
                source=src, provisional=prov)
        add(f'num{key}MedianTrainEval', float(np.nanmedian(treval)), source=src,
            provisional=prov, sig=4)
        others = acc[tbl['method'].astype(str) != 'Sven'].dropna()
        if len(others):
            add(f'num{key}BaselineAccMin', 100 * float(others.min()), source=src,
                provisional=prov)
            add(f'num{key}BaselineAccMax', 100 * float(others.max()), source=src,
                provisional=prov)
        # cost, from the standalone timing pass
        esrc = 'headline.efficiency_table'
        cost = inp.timing_cost(scan)
        e = eff.set_index('method')
        epoch = pd.to_numeric(eff['epoch_s'], errors='coerce')
        mem = pd.to_numeric(eff['peak_gpu_mem_mb'], errors='coerce')
        if cost:
            add(f'num{key}SvenWallS', cost['epoch_s'], source=esrc, provisional=prov,
                note='seconds per epoch, standalone timing pass')
            add(f'num{key}SvenMsStep', cost['ms_per_step'], source=esrc, provisional=prov)
            add(f'num{key}SvenMemMb', cost['peak_mb'], source=esrc, provisional=prov,
                sig=5)
            add(f'num{key}SvenFinTiming', cost['fin'], source=esrc, provisional=prov)
            add(f'num{key}SvenAttTiming', cost['att'], source=esrc, provisional=prov)
            fastest = float(np.nanmin(epoch)) if np.isfinite(epoch).any() else np.nan
            leanest = float(np.nanmin(mem)) if np.isfinite(mem).any() else np.nan
            if np.isfinite(fastest):
                add(f'num{key}FastestName',
                    _text(style.method_label(eff.loc[epoch.idxmin(), 'method'])),
                    source=esrc, provisional=prov)
            add(f'num{key}FastestWallS', fastest, source=esrc, provisional=prov)
            add(f'num{key}LeanestMemMb', leanest, source=esrc, provisional=prov, sig=5)
            add(f'num{key}SvenWallVsBest', cost['epoch_s'] / fastest, source=esrc,
                provisional=prov)
            add(f'num{key}SvenMemVsBest', cost['peak_mb'] / leanest, source=esrc,
                provisional=prov)
            # Same-source companions for the ratios above.  The Limitations paragraph
            # compared Sven's TIMING-pass step against Adam's PROFILE set point, which
            # is a cross-source ratio (7.5x instead of the like-for-like 27x); App. Q
            # forbids that unless the source of each is named.  These four macros make
            # every quoted CIFAR ratio available from one pass.
            for meth, mk in (('Adam', 'Adam'), ('MuonW', 'MuonW')):
                if meth not in e.index:
                    continue
                r = e.loc[meth]
                add(f'num{key}{mk}MsStep', _num(r.get('ms_per_step')), source=esrc,
                    provisional=prov, note='standalone timing pass, same pass as Sven')
                add(f'num{key}{mk}MemMb', _num(r.get('peak_gpu_mem_mb')), source=esrc,
                    provisional=prov, sig=5)
                if C.finite(_num(r.get('ms_per_step'))):
                    add(f'num{key}SvenMsStepVs{mk}',
                        cost['ms_per_step'] / _num(r['ms_per_step']), source=esrc,
                        provisional=prov,
                        note=f'Sven / {meth} per-step cost, both from the timing pass')
                if C.finite(_num(r.get('peak_gpu_mem_mb'))):
                    add(f'num{key}SvenMemVs{mk}',
                        cost['peak_mb'] / _num(r['peak_gpu_mem_mb']), source=esrc,
                        provisional=prov,
                        note=f'Sven / {meth} peak memory, both from the timing pass')
                if C.finite(_num(r.get('epoch_s'))):
                    add(f'num{key}SvenWallVs{mk}',
                        cost['epoch_s'] / _num(r['epoch_s']), source=esrc,
                        provisional=prov,
                        note=f'Sven / {meth} seconds per epoch, both from the timing pass')
        # the grid: divergences and the rank actually inverted
        d = lf.grid_divergence_counts(inp.grid[scan])
        add(f'num{key}SvenDivN', d['diverged'], source='large_figs.grid_divergence_counts',
            provisional=prov)
        add(f'num{key}SvenDivRecorded', d['recorded'],
            source='large_figs.grid_divergence_counts', provisional=prov)
        add(f'num{key}SvenDivFrac', 100 * _num(d['rate']),
            source='large_figs.grid_divergence_counts', provisional=prov,
            note='per cent of the Sven runs on this grid that have a record')
        add(f'num{key}SvenGridPoints', int(len(inp.grid[scan])),
            source='large_figs.sven_grid', provisional=prov)
        add(f'num{key}Params', _thousands(_num(_scan_n_params(inp, scan))),
            source='the run records (n_params)', provisional=prov)
        # The runner-up cell of Sven's own grid, so the appendix can say how close the
        # selection was: on cross-entropy the selection sits on the extended rtol edge
        # and the next cell is inside one seed std of it, which bounds that result.
        runner = _grid_runner_up(inp.grid[scan])
        if runner is not None:
            gsrc = 'large_figs.sven_grid (seed means over the tuning grid)'
            add(f'num{key}RunnerUpConfig', _text(runner['label']), source=gsrc,
                provisional=prov)
            add(f'num{key}RunnerUpVal', _num(runner['val']), source=gsrc, sig=4,
                provisional=prov)
            add(f'num{key}SelectedGridVal', _num(runner['best_val']), source=gsrc, sig=4,
                provisional=prov,
                note='the selected cell\'s own seed-mean on the TUNING grid, which is '
                     'what the runner-up is comparable with')
            if C.finite(_num(runner['best_std'])) and _num(runner['best_std']):
                add(f'num{key}RunnerUpStdGap',
                    abs(_num(runner['val']) - _num(runner['best_val']))
                    / _num(runner['best_std']), source=gsrc, sig=2, provisional=prov,
                    note='the gap to the runner-up in units of the selected cell\'s '
                         'seed std')
        ru = inp.rank_used[scan]
        if len(ru):
            k = _num(ru['k'].dropna().iloc[0])
            add(f'num{key}SvenK', k, source='large_figs.sven_rank_used', provisional=prov)
            add(f'num{key}SvenUsedRank', float(ru['rank_used'].mean()),
                source='large_figs.sven_rank_used', provisional=prov)
            add(f'num{key}SvenUsedRankFrac', 100 * float(ru['rank_used'].mean()) / k,
                source='large_figs.sven_rank_used', provisional=prov,
                note='mean rank inverted as a per cent of the k cap')
            add(f'num{key}SvenUsedRankSeeds', int(ru['run_id'].nunique()),
                source='large_figs.sven_rank_used', provisional=prov)

    # ---- Fig 5 ----------------------------------------------------------
    key = SCAN_KEY[FIG5]
    sel, prov = inp.fig5_selected, inp.fig5_incomplete
    cfg, tbl = sel['cfg'], sel['table']
    fsrc = 'large_figs.paramfrac_table'
    add(f'num{key}NConfigs', len(inp.fig5_groups), source='large_figs.paramfrac_groups')
    add(f'num{key}SvenConfig', _text(cfg['label']), source='large_figs.paramfrac_config',
        provisional=prov)
    add(f'num{key}Seeds', int(cfg['n_seeds']), source='large_figs.paramfrac_config',
        provisional=prov)
    add(f'num{key}TargetSeeds', int(inp.fig5_target_seeds),
        source='large_figs.paramfrac_config')
    add(f'num{key}NFractions', len(cfg['fractions']),
        source='large_figs.paramfrac_config', provisional=prov)
    add(f'num{key}ChanceAcc', 100 * CIFAR_CHANCE_ACC, source='dataset definition')
    add(f'num{key}Epochs', _num(cfg.get('num_epochs')),
        source='large_figs.paramfrac_config')
    for _, r in tbl.iterrows():
        token = frac_token(r['param_fraction'])
        add(f'num{key}Val{token}', _num(r['val']), source=fsrc, provisional=prov)
        add(f'num{key}Acc{token}', 100 * _num(r['test_acc']), source=fsrc,
            provisional=prov)
        add(f'num{key}MemMb{token}', _num(r['peak_mem_mb']), source=fsrc, sig=5,
            provisional=prov)
        add(f'num{key}MsStep{token}', _num(r['ms_per_step']), source=fsrc,
            provisional=prov)
        add(f'num{key}Fin{token}', int(r['finished']), source=fsrc, provisional=prov)
        add(f'num{key}Att{token}', int(r['attempted']), source=fsrc, provisional=prov)
    # where quality holds, and where it collapses -- computed, never assumed
    hold, collapse = paramfrac_threshold(tbl)
    if np.isfinite(hold):
        add(f'num{key}HoldFrac', hold, source='paper_assets.large.paramfrac_threshold',
            provisional=prov,
            note='smallest fraction whose test accuracy is within 10% (relative) of f=1')
    if np.isfinite(collapse):
        add(f'num{key}CollapseFrac', collapse,
            source='paper_assets.large.paramfrac_threshold', provisional=prov,
            note='largest fraction below the hold point, i.e. where quality has gone')
    cost = paramfrac_cost_summary(tbl)
    for name, value, sig in (('MsStepFull', cost['full_ms'], 4),
                             ('MsStepMaskedMin', cost['masked_ms_min'], 4),
                             ('MsStepMaskedMax', cost['masked_ms_max'], 4),
                             ('SlowMin', cost['slow_min'], 3),
                             ('SlowMax', cost['slow_max'], 3),
                             ('MemFracMin', cost['mem_frac_min'], 3),
                             ('MemMbFull', cost['full_mb'], 5)):
        if np.isfinite(value):
            add(f'num{key}{name}', value, source=fsrc, provisional=prov)
    div, runs, _cells = lf.divergence_lrs(FIG5, results_root=inp.results_root)
    add(f'num{key}DivN', div, source='large_figs.divergence_lrs', provisional=prov)
    add(f'num{key}Runs', runs, source='large_figs.divergence_lrs', provisional=prov,
        note='EVERY run on disk, over ALL configurations of the sweep -- not the runs '
             'behind the quoted quality/cost numbers (see RunsSelected)')
    # The appendix used to quote ``Runs`` (both configurations, 30) beside "3 seeds per
    # point at 5 fractions", which is 15.  The study rests on the selected configuration,
    # so that count gets its own macro and the other configuration its own.
    att = pd.to_numeric(tbl.get('attempted'), errors='coerce')
    if att is not None and np.isfinite(att).any():
        n_sel = int(np.nansum(att))
        add(f'num{key}RunsSelected', n_sel, source=fsrc, provisional=prov,
            note='runs at the SELECTED configuration: the ones every quoted number uses')
        if C.finite(runs) and int(runs) > n_sel:
            add(f'num{key}RunsOther', int(runs) - n_sel, source=fsrc, provisional=prov,
                note='runs at the earlier configuration, also drawn in the figure')

    # ---- nanoGPT --------------------------------------------------------
    key = SCAN_KEY[NANOGPT]
    tbl, eff = inp.conf[NANOGPT], inp.eff[NANOGPT]
    src = 'headline.confirmation_table'
    sven, best = _row_of(tbl, 'Sven'), tbl.iloc[0]
    rank_val, n_meth = _rank(tbl, 'val_conf')
    rank_test, _ = _rank(tbl, 'test_conf')
    add(f'num{key}NMethods', n_meth, source=src)
    add(f'num{key}BestName', _text(best['display']), source=src)
    add(f'num{key}BestVal', _num(best['val_conf']), source=src, sig=4)
    if sven is not None:
        add_pm(f'num{key}SvenVal', _num(sven['val_conf']),
                    _num(sven['val_conf_std']), source=src, sig=4)
        add(f'num{key}SvenPpl', float(np.exp(_num(sven['val_conf']))), source=src,
            note='exp of the validation loss (per-token cross-entropy)')
        add(f'num{key}SvenTest', _num(sven['test_conf']), source=src, sig=4)
        add(f'num{key}SvenTestPpl', float(np.exp(_num(sven['test_conf']))), source=src)
        add(f'num{key}SvenRank', rank_val, source=src)
        add(f'num{key}SvenRankTest', rank_test, source=src)
        add(f'num{key}SvenConfig', _text(sven['config']), source='headline.config_label')
        add(f'num{key}SvenFin', _num(sven['fin_conf']), source=src)
        add(f'num{key}SvenAtt', _num(sven['att_conf']), source=src)
        add(f'num{key}SvenGapRel',
            100 * (_num(sven['val_conf']) / _num(best['val_conf']) - 1), source=src)
    adamw = _row_of(tbl, 'AdamW')
    if adamw is not None:
        add_pm(f'num{key}AdamWVal', _num(adamw['val_conf']),
                    _num(adamw['val_conf_std']), source=src, sig=4)
    pair = _row_of(inp.paired_nanogpt, 'AdamW')
    if pair is not None:
        psrc = 'headline.paired_vs_sven'
        add(f'num{key}PairedAdamW', _num(pair['mean']), source=psrc,
            note='Sven - AdamW in final validation loss, paired by model seed')
        add(f'num{key}PairedAdamWCiLow', _num(pair['ci_low']), source=psrc)
        add(f'num{key}PairedAdamWCiHigh', _num(pair['ci_high']), source=psrc)
        add(f'num{key}PairedAdamWN', int(_num(pair['n'])), source=psrc)
        add(f'num{key}PairedAdamWResolved',
            _text('resolved' if bool(pair['significant']) else 'a tie'), source=psrc,
            note='whether the 95% paired interval for AdamW excludes zero -- it does '
                 'NOT, so no sentence may call the nanoGPT ordering resolved')
    pv = inp.paired_nanogpt
    if pv is not None and len(pv):
        # who is ahead of Sven here, and whether the lead is resolved (see
        # paper_assets.main for the same block on the perceptron scans)
        ahead = pv[pd.to_numeric(pv['mean'], errors='coerce') > 0] \
            .sort_values('mean', ascending=False)
        if len(ahead):
            psrc = 'headline.paired_vs_sven (paired mean > 0)'
            add(f'num{key}AheadList', C.Raw(_name_list(ahead['display'])), source=psrc)
            add(f'num{key}NAhead', len(ahead), source=psrc)
            res = ahead[ahead['significant'].astype(bool)]
            unres = ahead[~ahead['significant'].astype(bool)]
            add(f'num{key}NAheadResolved', len(res), source=psrc)
            add(f'num{key}NAheadUnresolved', len(unres), source=psrc)
            if len(res):
                add(f'num{key}AheadResolvedList', C.Raw(_name_list(res['display'])),
                    source=psrc)
            if len(unres):
                add(f'num{key}AheadUnresolvedList', C.Raw(_name_list(unres['display'])),
                    source=psrc)
    esrc = 'headline.efficiency_table'
    e = eff.set_index('method')
    for method, mkey in (('Sven', 'Sven'), ('AdamW', 'AdamW')):
        if method in e.index:
            r = e.loc[method]
            add(f'num{key}{mkey}WallS', _num(r['epoch_s']), source=esrc,
                note='seconds per epoch, standalone')
            add(f'num{key}{mkey}MsStep', _num(r['ms_per_step']), source=esrc)
            add(f'num{key}{mkey}MemMb', _num(r['peak_gpu_mem_mb']), source=esrc, sig=4)
            add(f'num{key}{mkey}TotalWallS', _num(r['wall_s']), source=esrc, sig=4)
    if 'Sven' in e.index and 'AdamW' in e.index:
        add(f'num{key}SvenWallVsAdamW',
            _num(e.loc['Sven', 'wall_s']) / _num(e.loc['AdamW', 'wall_s']), source=esrc)
        add(f'num{key}SvenMemVsAdamW',
            _num(e.loc['Sven', 'peak_gpu_mem_mb'])
            / _num(e.loc['AdamW', 'peak_gpu_mem_mb']), source=esrc)
    d = lf.grid_divergence_counts(inp.grid[NANOGPT])
    add(f'num{key}SvenDivN', d['diverged'], source='large_figs.grid_divergence_counts')
    add(f'num{key}SvenDivRecorded', d['recorded'],
        source='large_figs.grid_divergence_counts')

    # lr sensitivity on the TUNING scan (T17a2 / the third panel of the nanoGPT figure).
    # The same quantities as GPT-2's block below, so App. L can make the same statement
    # about both transformers.
    lsrc = 'reviewer_figs.arm_table'
    lr_tbl = inp.nanogpt_lr
    add(f'num{key}NLrs', int(pd.to_numeric(lr_tbl['lr'], errors='coerce').nunique()),
        source=lsrc, note='distinct learning rates on the nanoGPT grid')
    sven_lr = lr_tbl[lr_tbl['method'].map(hl.method_key) == 'Sven'].sort_values('lr')
    if len(sven_lr):
        v = pd.to_numeric(sven_lr['final_val_loss'], errors='coerce')
        i = int(np.nanargmin(v.to_numpy()))
        add(f'num{key}SvenNLrs', int(len(sven_lr)), source=lsrc)
        add(f'num{key}SvenLrMin', _num(sven_lr['lr'].min()), source=lsrc, sig=2)
        add(f'num{key}SvenLrMax', _num(sven_lr['lr'].max()), source=lsrc, sig=2)
        add(f'num{key}SvenLrOptimum',
            'interior' if 0 < i < len(sven_lr) - 1
            else ('low edge' if i == 0 else 'high edge'), source=lsrc,
            note='where the best learning rate sits inside the sweep')
        add(f'num{key}SvenValWorstLr', float(np.nanmax(v.to_numpy())), source=lsrc, sig=4)
        add(f'num{key}SvenValSpreadLr',
            float(np.nanmax(v.to_numpy()) / np.nanmin(v.to_numpy())), source=lsrc,
            note='worst / best validation loss over the learning-rate grid')

    # ---- GPT-2 small ----------------------------------------------------
    key = SCAN_KEY[GPT2]
    gsrc = 'large_figs.gpt2_best_per_method'
    best_tbl = inp.gpt2_best.sort_values('val')
    sven = _row_of(best_tbl, 'Sven')
    lead = best_tbl.iloc[0]
    facts = inp.gpt2.iloc[0]
    add(f'num{key}NMethods', int(len(best_tbl)), source=gsrc)
    add(f'num{key}NRuns', int(len(inp.gpt2)), source='large_figs.gpt2_frame')
    add(f'num{key}Params', _thousands(facts['n_params']), source='the run records',
        note='parameters of GPT-2 small as the records report them')
    add(f'num{key}Steps', _thousands(facts['steps_per_epoch']), source='the run records',
        note='optimizer steps in the single epoch')
    add(f'num{key}Batch', _num(facts['batch_size']), source='the run records')
    add(f'num{key}Evals', len((facts.get('losses') or {}).get('eval_step_idx') or []),
        source='the run records')
    add(f'num{key}BestName', _text(lead['display']), source=gsrc)
    add(f'num{key}BestVal', _num(lead['val']), source=gsrc, sig=4)
    add(f'num{key}BestPpl', _num(lead['val_ppl']), source=gsrc, sig=4)
    total_h = float(pd.to_numeric(inp.gpt2['wall_h'], errors='coerce').sum())
    add(f'num{key}TotalGpuH', total_h, source='large_figs.add_run_cost')
    add(f'num{key}PerRunGpuH', total_h / max(len(inp.gpt2), 1),
        source='large_figs.add_run_cost')
    if sven is not None:
        add(f'num{key}SvenVal', _num(sven['val']), source=gsrc, sig=4)
        add(f'num{key}SvenPpl', _num(sven['val_ppl']), source=gsrc, sig=4)
        add(f'num{key}SvenTest', _num(sven['test']), source=gsrc, sig=4)
        add(f'num{key}SvenGapRel', 100 * (_num(sven['val']) / _num(lead['val']) - 1),
            source=gsrc, note='per cent above the best baseline on validation loss')
        add(f'num{key}SvenRank', _rank(best_tbl, 'val')[0], source=gsrc)
        add(f'num{key}SvenLr', _num(sven['lr']), source=gsrc, sig=2)
        add(f'num{key}SvenK', _num(sven['k']), source=gsrc,
            note='k = B on this scan: the ratio was never tuned')
        add(f'num{key}SvenWallH', _num(sven['wall_h']), source=gsrc)
        add(f'num{key}SvenMsStep', _num(sven['ms_per_step']), source=gsrc, sig=4)
        add(f'num{key}SvenMemMb', _num(sven['peak_gpu_mem_mb']), source=gsrc, sig=5)
        add(f'num{key}SvenWallVsBest',
            _num(sven['wall_h']) / _num(lead['wall_h']), source=gsrc)
        add(f'num{key}SvenMemVsBest',
            _num(sven['peak_gpu_mem_mb']) / _num(lead['peak_gpu_mem_mb']), source=gsrc)
        sven_lrs = inp.gpt2_lr[inp.gpt2_lr['method'] == 'Sven'].sort_values('lr')
        v = pd.to_numeric(sven_lrs['val'], errors='coerce')
        i = int(np.nanargmin(v.to_numpy()))
        add(f'num{key}SvenNLrs', int(len(sven_lrs)), source='large_figs.gpt2_lr_table')
        add(f'num{key}SvenLrMin', _num(sven_lrs['lr'].min()),
            source='large_figs.gpt2_lr_table', sig=2)
        add(f'num{key}SvenLrMax', _num(sven_lrs['lr'].max()),
            source='large_figs.gpt2_lr_table', sig=2)
        add(f'num{key}SvenLrOptimum',
            'interior' if 0 < i < len(sven_lrs) - 1
            else ('low edge' if i == 0 else 'high edge'),
            source='large_figs.gpt2_lr_table',
            note='where the best learning rate sits inside the sweep')
        add(f'num{key}SvenValWorstLr', float(np.nanmax(v)),
            source='large_figs.gpt2_lr_table', sig=4)

    # ---- the profile ----------------------------------------------------
    info = inp.profile_info
    pprov = bool(info['provisional'])
    add('numProfRoot', _text(info['name']), source='paper_assets.large.profile_root')
    add('numProfNConfigs', info['n_found'], source='profile_helpers.profile_status',
        provisional=pprov)
    add('numProfNExpected', info['n_expected'], source='profile_helpers.profile_status')
    add('numProfComplete', 'complete' if info['complete'] else 'partial',
        source='profile_helpers.profile_status', provisional=pprov)
    add('numProfNArchs', len(inp.profile_archs), source='profile_helpers.load_profiles',
        provisional=pprov)
    # the measurement protocol, read off the records rather than described from memory
    proto = info.get('protocol') or {}
    psrc = 'paper_assets.large.profile_protocol (every record of the profile root)'
    add('numProfNRecords', proto.get('n_records'), source=psrc, provisional=pprov)
    for pkey, name in (('staged_batches', 'numProfStagedBatches'),
                       ('warmup_steps', 'numProfWarmupSteps'),
                       ('num_steps', 'numProfMeasuredSteps'),
                       ('min_steps', 'numProfMinSteps'),
                       ('max_seconds', 'numProfMaxSeconds')):
        if pkey in proto:
            add(name, int(proto[pkey]), source=psrc, provisional=pprov,
                note=f'profile.{pkey}, unanimous over {proto.get("n_records")} records')
    if 'num_steps' in proto:
        tail = int(proto['num_steps'] - int(PROFILE_STARTUP_FRACTION
                                            * proto['num_steps']))
        add('numProfAmortSteps', (tail // ph.CYCLE) * ph.CYCLE or tail,
            source='profile_helpers.cycle_mean over profile.num_steps',
            provisional=pprov,
            note='the step time the profile reports is the mean over this many steps: '
                 'the tail of the measured series, truncated to whole cycles')
        add('numProfAmortPct', 100 * (1 - PROFILE_STARTUP_FRACTION),
            source='profile_helpers.cycle_mean', provisional=pprov)
    if proto.get('disagree'):
        print(f'[large] ** profile protocol is NOT unanimous on {proto["disagree"]} -- '
              f'App. Q must not quote those as single numbers')
    prof = inp.profiles
    msrc = 'profile_helpers.method_table'
    for arch in inp.profile_archs:
        akey = ARCH_KEY.get(arch, arch.replace('_', '').title())
        block = prof[(prof.arch == arch) & (prof.study == 'methods')]
        for method, mkey in SVEN_BACKENDS.items():
            r = block[block.method == method]
            if not len(r):
                continue
            r = r.iloc[0]
            if r['status'] not in ph.TIMED:
                continue
            step, capture = _num(r['step_ms']), _num(r['capture_ms'])
            add(f'numProf{akey}{mkey}MsStep', step, source=msrc, provisional=pprov)
            add(f'numProf{akey}{mkey}MemMb', _num(r['peak_mb']), source=msrc, sig=4,
                provisional=pprov)
            if np.isfinite(_num(r['rel_time'])):
                add(f'numProf{akey}{mkey}VsAdam', _num(r['rel_time']), source=msrc,
                    provisional=pprov)
            if np.isfinite(_num(r['rel_mem'])):
                add(f'numProf{akey}{mkey}VsSgd', _num(r['rel_mem']), source=msrc,
                    provisional=pprov)
            if np.isfinite(capture) and step:
                add(f'numProf{akey}{mkey}CaptureFrac', 100 * capture / step, source=msrc,
                    provisional=pprov, note='per cent of the step spent in the capture')
            ks = prof[(prof.arch == arch) & (prof.study == 'k')
                      & (prof.method == method) & prof.status.isin(ph.TIMED)]
            if len(ks) > 1:
                lo, hi = _num(ks['step_ms'].min()), _num(ks['step_ms'].max())
                add(f'numProf{akey}{mkey}KSpread', hi / lo, source='profile_helpers.plot_sweep',
                    provisional=pprov,
                    note='largest / smallest step time over the whole k sweep')
                add(f'numProf{akey}{mkey}KMin', _num(ks['k'].min()),
                    source='profile_helpers.load_profiles', provisional=pprov)
                add(f'numProf{akey}{mkey}KMax', _num(ks['k'].max()),
                    source='profile_helpers.load_profiles', provisional=pprov)
        for method, mkey in (('Adam', 'Adam'), ('SGD', 'Sgd')):
            r = block[block.method == method]
            if len(r) and r.iloc[0]['status'] in ph.TIMED:
                add(f'numProf{akey}{mkey}MsStep', _num(r.iloc[0]['step_ms']), source=msrc,
                    provisional=pprov)
                add(f'numProf{akey}{mkey}MemMb', _num(r.iloc[0]['peak_mb']), source=msrc,
                    sig=4, provisional=pprov)
    # scaling exponents of the width sweeps
    for arch in inp.profile_widths:
        akey = ARCH_KEY.get(arch, arch.replace('_', '').title())
        for value, vkey in (('step_ms', 'StepExp'), ('peak_mb', 'MemExp')):
            tbl = ph.scaling_table(prof, arch, value).set_index('Method')
            for method, mkey in SVEN_BACKENDS.items():
                name = ph.label(method)
                if name not in tbl.index:
                    continue
                exp = _num(tbl.loc[name, 'exponent'])
                if np.isfinite(exp):
                    add(f'numProf{akey}{mkey}{vkey}', exp,
                        source='profile_helpers.scaling_table', sig=2, provisional=pprov)
    # what the correction changed
    cmp = inp.profile_compare
    if len(cmp):
        chg = ph.sven_change_table(cmp)
        ratios = pd.to_numeric(chg['step v3/v2'], errors='coerce')
        if np.isfinite(ratios).any():
            i = int(np.nanargmin(ratios.to_numpy()))
            add('numProfSvenStepRatioMin', float(ratios.min()),
                source='profile_helpers.sven_change_table', provisional=pprov,
                note='the largest correction: v3 / v2 step time for a Sven backend')
            add('numProfSvenStepRatioMinName',
                _text(f'{chg.iloc[i]["Method"]} on '
                      f'{ph.ARCH_TITLES.get(chg.iloc[i]["arch"], chg.iloc[i]["arch"])}'),
                source='profile_helpers.sven_change_table', provisional=pprov)
            add('numProfSvenStepSpeedupMax', 1.0 / float(ratios.min()),
                source='profile_helpers.sven_change_table', provisional=pprov,
                note='how much faster the corrected pass measures that configuration')
        mem = pd.to_numeric(chg['mem v3/v2'], errors='coerce')
        if np.isfinite(mem).any():
            add('numProfSvenMemRatioMax', float(np.nanmax(np.abs(mem - 1)) + 1),
                source='profile_helpers.sven_change_table', provisional=pprov,
                note='the largest memory move between the passes: the fix changes time, '
                     'not the live set')
        base = ph.baseline_control(cmp)
        if len(base):
            worst = base.loc[base['median v3/v2'].sub(1).abs().idxmax()]
            add('numProfBaselineRatioWorst', _num(worst['median v3/v2']),
                source='profile_helpers.baseline_control', provisional=pprov,
                note='median v3/v2 of the baselines at the worst architecture (control)')
    book.skipped = skipped
    return book


def paramfrac_threshold(tbl, rel=0.10, acc_key='test_acc'):
    """``(hold, collapse)``: where quality still holds down to, and where it has gone.

    ``hold`` is the SMALLEST parameter fraction whose accuracy is within ``rel``
    (relative) of the unmasked ``f = 1`` accuracy; ``collapse`` is the largest fraction
    below it, i.e. the first level at which the study lost the result.  Computed rather
    than written down, because the Fig-5 re-run at the selected configuration can move it
    (and the old threshold sat at a learning rate that is no longer the pick).  NaN when
    the scan has no ``f = 1`` point yet.
    """
    if tbl is None or not len(tbl) or acc_key not in tbl.columns:
        return float('nan'), float('nan')
    d = tbl.sort_values('actual')
    acc = pd.to_numeric(d[acc_key], errors='coerce').to_numpy(dtype=float)
    frac = pd.to_numeric(d['actual'], errors='coerce').to_numpy(dtype=float)
    ref = acc[np.isclose(frac, 1.0)]
    if not len(ref) or not np.isfinite(ref[0]):
        return float('nan'), float('nan')
    ok = np.isfinite(acc) & (acc >= (1.0 - rel) * ref[0])
    if not ok.any():
        return float('nan'), float('nan')
    hold = float(frac[ok].min())
    below = frac[np.isfinite(acc) & (frac < hold)]
    return hold, (float(below.max()) if len(below) else float('nan'))


def paramfrac_cost_summary(tbl):
    """Fig 5's cost finding as numbers: masking makes the step slower, not faster.

    ``slow_min`` / ``slow_max`` are the step time at a masked fraction over the step time
    at ``f = 1``; ``mem_frac_min`` is the smallest peak memory as a fraction of the
    unmasked one.  The full-Jacobian capture does not get cheaper with an element mask, so
    the only saving is memory.
    """
    out = {k: float('nan') for k in ('full_ms', 'full_mb', 'masked_ms_min',
                                     'masked_ms_max', 'slow_min', 'slow_max',
                                     'mem_frac_min')}
    if tbl is None or not len(tbl):
        return out
    d = tbl.sort_values('actual')
    full = d[np.isclose(pd.to_numeric(d['actual'], errors='coerce'), 1.0)]
    masked = d[pd.to_numeric(d['actual'], errors='coerce') < 1.0]
    if not len(full):
        return out
    out['full_ms'] = _num(full.iloc[0]['ms_per_step'])
    out['full_mb'] = _num(full.iloc[0]['peak_mem_mb'])
    if len(masked):
        ms = pd.to_numeric(masked['ms_per_step'], errors='coerce').dropna()
        mb = pd.to_numeric(masked['peak_mem_mb'], errors='coerce').dropna()
        if len(ms):
            out['masked_ms_min'] = float(ms.min())
            out['masked_ms_max'] = float(ms.max())
            if out['full_ms']:
                out['slow_min'] = float(ms.min()) / out['full_ms']
                out['slow_max'] = float(ms.max()) / out['full_ms']
        if len(mb) and out['full_mb']:
            out['mem_frac_min'] = float(mb.min()) / out['full_mb']
    return out


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------
def build(root=None, dry_run=False, figures=True, tables=True, numbers=True):
    """Build every asset this module owns.  Returns the report ``__main__`` prints."""
    inp = context(root, reload=True)     # a CLI build always re-reads the disk
    C.banner('large', f'CIFAR-10/ResNet18, Fig 5, nanoGPT, GPT-2 small, '
                      f'profile {inp.profile_info["name"]}')
    report = {'figures': [], 'tables': [], 'n_macros': 0, 'notes': [],
              'provisional': inp.provisional, 'status': ''}
    if inp.provisional:
        print('[large] PROVISIONAL inputs:')
        for tag in inp.provisional:
            print(f'    - {tag}')
        if not C.allow_provisional():
            report['status'] = 'refused (PAPER_ASSETS_STRICT)'
            return report
    print(f'[large] profile: {inp.profile_note()}')

    if dry_run:
        report['figures'] = [f'{n}.pdf' for n in FIGURE_SPECS]
        report['tables'] = [f'{n}.tex' for n, _ in TABLES]
        report['status'] = 'dry run'
        return report

    if figures:
        for name, spec in FIGURE_SPECS.items():
            # defaults < figure_overrides.yaml (what a notebook pinned) < nothing here:
            # the CLI passes no call options on purpose, so a rebuild replays the file
            opts = figspec.figure_opts(name, spec.defaults)
            rc = opts.get('rc') or {}
            # the draw function's own `set_paper_style` is what `savefig` writes the PDF
            # with (pdf.fonttype 42, no bbox), so the save has to happen where the draw
            # left the rcParams -- an rc_context around the draw alone would hand savefig
            # the matplotlib defaults back.
            with (plt.rc_context(rc) if rc else contextlib.nullcontext()):
                fig, meta = spec.draw(inp, opts)
                meta = dict(meta or {})
                if fig is None:
                    print(f'[large] {name}: nothing to draw -- skipped')
                    report['notes'].append(f'{name}: skipped (no data)')
                    continue
                figspec.apply_opts(fig, meta.get('axes'), opts)
                rec = spec.provenance(inp, meta)
                pdf, png = C.save_fig(fig, name, GROUP, provenance_record=rec)
            report['figures'].append(str(pdf.relative_to(C.MANUSCRIPT)))
            print(f'[large] figure {pdf.relative_to(C.MANUSCRIPT)}   (png {png})')

    if tables:
        for name, fn in TABLES:
            rows, meta = fn(inp)
            if not rows:
                print(f'[large] table {name}: no rows -- skipped')
                report['notes'].append(f'table {name}: skipped (no rows)')
                continue
            rec = C.provenance(functions=meta.get('functions', ()),
                               scans=_table_scans(name),
                               reads=[inp.profile_info['root']] if 'profile' in name
                                     or name == 'gram_cost' else [],
                               provisional=inp.provisional,
                               notes=list(meta.get('notes', ())))
            path = C.write_table(name, rows, provenance_record=rec,
                                 caption=TABLE_CAPTIONS.get(name),
                                 label=f'tab:{name}',
                                 notes=meta.get('notes', ()),
                                 midrules=meta.get('midrules', ()))
            report['tables'].append(str(path.relative_to(C.MANUSCRIPT)))
            print(f'[large] table  {path.relative_to(C.MANUSCRIPT)}  '
                  f'({len(rows)} rows)')

    if numbers:
        book = macros(inp)
        rec = C.provenance(functions=('headline.confirmation_table',
                                      'headline.efficiency_table',
                                      'headline.paired_vs_sven',
                                      'large_figs.paramfrac_table',
                                      'large_figs.gpt2_best_per_method',
                                      'large_figs.grid_divergence_counts',
                                      'large_figs.sven_rank_used',
                                      'profile_helpers.method_table',
                                      'profile_helpers.scaling_table',
                                      'profile_helpers.sven_change_table'),
                           scans=sorted(SCAN_KEY),
                           reads=[inp.profile_info['root']],
                           provisional=inp.provisional,
                           n_macros=len(book))
        path = book.write(provenance=rec)
        report['n_macros'] = len(book)
        skipped = list(getattr(book, 'skipped', ()))
        report['skipped_macros'] = skipped
        print(f'[large] macros {path.relative_to(C.MANUSCRIPT)}  ({len(book)} macros, '
              f'{len(book.provisional)} provisional)')
        if skipped:
            print(f'[large] {len(skipped)} macro(s) have no value on this build and were '
                  f'NOT written: {", ".join(skipped)}')

    report['notes'].extend(inp.notes)
    report['status'] = 'ok' + (' (PROVISIONAL)' if inp.provisional else '')
    return report


def _figure_scans(name):
    if name.startswith('cifar'):
        return list(CIFAR)
    if name.startswith('fig5'):
        return [FIG5]
    if name.startswith('nanogpt'):
        return [NANOGPT]
    if name.startswith('gpt2'):
        return [GPT2]
    if name.startswith('profile'):
        return []
    return []


def _table_scans(name):
    return {'cifar': list(CIFAR), 'fig5': [FIG5], 'transformers': [NANOGPT],
            'transformers_nanogpt_lr': [NANOGPT],
            'transformers_gpt2': [GPT2], 'transformers_gpt2_lr': [GPT2]}.get(name, [])


if __name__ == '__main__':          # python -m paper_assets.large
    build()
