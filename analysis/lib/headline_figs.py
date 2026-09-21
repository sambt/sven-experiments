"""Figures and tables for the WP2 phase-B notebooks.

The five notebooks this module serves --- ``toy_1d_analysis``, ``polynomial_analysis``,
``mnist_analysis``, ``mnist_analysis_labelRegression`` and ``baselines_analysis`` ---
report the campaign's headline numbers, and those numbers have exactly one
implementation: :mod:`headline`.  Nothing here recomputes a confirmation mean, a paired
difference, a budget curve, a time-to-target row or a timing number; every such table is
fetched from ``headline`` and only *arranged* here.

What this module does own is the three things ``headline`` has no reason to:

* **curve panels on the three x-axes** (optimizer steps, examples processed, and
  *synchronised training time* from the standalone ``{scan}_timing`` pass), drawn from
  the CONFIRMATION pass so a curve and the table above it describe the same runs;
* **the divergence map of a tuning grid** --- what fraction of each method's grid blew
  up, under both the recorded ``status`` count and the wider analysis rule
  (:func:`style.is_diverged`), and where Sven's divergences sit in ``(rtol, lr, k)``;
* **the Sven (k, lr, rtol) landscape**, with the selected point marked and every grid
  edge named, including ``k = B``, which is a natural maximum rather than an edge.

Conventions are the binding ones (``campaign/ANALYSIS_CONTRACTS.md``): selection is
seed-mean final VALIDATION loss, test metrics are outcomes, diverged = failed and is
counted, a band is +/- 1 std over seeds, display names come from
:func:`style.method_label`, and colours from :data:`style.METHOD_COLORS`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import analysis_helpers as ah   # noqa: F401  (re-exported convenience for the notebooks)
import budget
import headline as hl
import paired
import scan_analysis as sa
import style

__all__ = [
    'save', 'relabel_methods', 'provisional_banner', 'summary_cell', 'ranking_line',
    'confirm_runs', 'standalone_epoch_times', 'method_ranking', 'curve_axis',
    'plot_curves', 'curve_figure', 'CURVE_AXES',
    'divergence_by_method', 'sven_divergence_grid', 'plot_divergence_grid',
    'sven_grid_table', 'selected_sven', 'grid_edges', 'plot_selected_marker',
    'instance_spread', 'plot_instance_spread', 'closing_numbers', 'scan_matrix',
    'missing_confirmation', 'pass_gpus', 'gpu_homogeneity_line', 'timing_join_view',
    'on_grid_scans', 'campaign_divergence', 'divergence_pattern',
    'equal_budget_table', 'BUDGET_EQUAL_N',
]

#: the three x-axes every convergence figure is drawn on (F21 / C-A5).  ``'step'`` is
#: the epoch axis in optimizer-step units --- the same shape, in the unit a reader can
#: compare across batch sizes; ``'time'`` is the SYNCHRONISED training time of the
#: standalone timing pass, not the sharded scan's wall clock.
CURVE_AXES = ('step', 'examples', 'time')

#: which per-epoch series the ``'time'`` axis integrates.  ``train_times`` is the
#: synchronised per-epoch TRAINING time (C-T1): evaluation, a large and identical
#: overhead for every method, is out of it.
TIME_KEY = 'train_times'
TIME_FALLBACK = 'epoch_times'

#: resolution of the PNG copy of a figure.  The PDF is the paper artefact and keeps the
#: global ``savefig.dpi``; the PNG is a preview, and a 17 x 9 inch panel at 300 dpi is a
#: 2 MB file nobody looks at.
PNG_DPI = 150

#: the tuning budget at which the best-of-n curves are read against each other in the
#: closing numbers.  Eight is the smallest grid any baseline with a swept learning rate
#: has in this suite (Polyak SGD has one point and JD six), so at n = 8 every method is
#: either drawing from its whole grid or from eight of a larger one -- the equal-budget
#: comparison the grid sizes otherwise prevent.
BUDGET_EQUAL_N = 8

_AXIS_LABELS = {
    'time': 'Synchronised training time (s), standalone',
    'wall': 'Wall time (s), standalone',
}


def axis_label(versus):
    """The x label of one of :data:`CURVE_AXES`."""
    return _AXIS_LABELS.get(versus) or sa.axis_label(versus)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save(fig, where, stem, **kw):
    """Save ``fig`` as PDF **and** PNG and return the two paths.

    ``where`` is either a :class:`scan_analysis.Scan` --- then its own ``save`` helper
    (and so its ``plot_dir``) is used --- or a directory.
    """
    kw.setdefault('bbox_inches', 'tight')
    if hasattr(where, 'save'):
        # the Scan's own helper for the PDF (the paper artefact), then a lighter PNG
        out = [where.save(fig, f'{stem}.pdf')]
        fig.savefig(Path(where.plot_dir) / f'{stem}.png', dpi=PNG_DPI, **kw)
        return out + [Path(where.plot_dir) / f'{stem}.png']
    d = Path(where)
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ('pdf', 'png'):
        path = d / f'{stem}.{ext}'
        fig.savefig(path, **({**kw, 'dpi': PNG_DPI} if ext == 'png' else kw))
        out.append(path)
    return out


def relabel_methods(ax, **legend_kw):
    """Rewrite a legend whose entries start with a raw optimizer KEY.

    :func:`budget.plot_best_of_n` labels its lines ``"<method> (<n> pts)"`` with the
    record-level key, so a reader sees ``LBFGS`` / ``JD_UPGrad`` instead of
    "Stochastic L-BFGS" / "JD (UPGrad)" (C-B7).  ``budget.py`` is frozen in this phase,
    so the fix lives here: the key is mapped through :func:`style.method_label` and the
    legend redrawn.  Handles with no method key are left alone.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    new = []
    for text in labels:
        head, sep, tail = text.partition(' (')
        label = style.method_label(head)
        new.append(label + sep + tail)
    return ax.legend(handles, new, **legend_kw)


# ---------------------------------------------------------------------------
# (a) the results-summary cell
# ---------------------------------------------------------------------------
_freshness_memo = {}


def provisional_banner(scan, results_root=None, force=False):
    """Print a PROVISIONAL banner when ``scan`` still has runs in flight.

    ``headline.freshness_report`` compares every record's mtime and every live claim
    against the instant ``bench/best_configs.json`` was written: a tuning-grid record
    newer than the selection, or a claim with no record yet, means the selected
    configuration can still move and every table for that scan is provisional.  Returns
    ``True`` when the banner was printed.
    """
    key = str(style.resolve_results_root(results_root))
    if force or key not in _freshness_memo:
        _freshness_memo[key] = hl.freshness_report(results_root=results_root)
    fresh = _freshness_memo[key]
    sub = fresh[fresh['scan'] == scan]
    if sub.empty or not bool(sub['provisional'].any()):
        print(f'[fresh] {scan}: no runs in flight; every table below is final with '
              f"respect to the selection written {fresh.attrs['selection_generated_at']}")
        return False
    live = int(sub['n_claims_live'].sum())
    after = int(sub.loc[sub['selection_input'], 'n_after_selection'].sum())
    print('!' * 78)
    print(f'PROVISIONAL -- {hl.scan_title(scan)} ({scan}) is still moving: {live} live '
          f'claim(s), {after} tuning-grid record(s) newer than the selection of record '
          f"({fresh.attrs['selection_generated_at']}).")
    print('To bring this notebook up to date, in this order (ANALYSIS_PLAN decision 5):')
    print('  1. tools/select_best.py  -- rewrite bench/best_configs.json;')
    print(f'  2. for every method whose selected configuration MOVED, re-run the '
          f'{hl.dir_name(scan, "confirm")}/ and {hl.dir_name(scan, "timing")}/ (and '
          f'{hl.dir_name(scan, "diag")}/) passes for it -- the existing passes contain '
          f'no run of a configuration that was not selected when they were generated, '
          f'so its confirmation row would read 0/0 finished with an empty mean;')
    print('  3. re-execute this notebook.  The selected configuration, its rank and '
          'every number below can change.')
    print('!' * 78)
    return True


def missing_confirmation(table, quiet=False):
    """The methods whose SELECTED configuration has no run in the confirmation pass.

    Zero matched records is not a small number, it is the absence of a number, and
    :func:`headline.confirmation_table` reports it as ``0/0 finished`` with a NaN mean --
    which :func:`headline.ranking_summary` then sorts last, so the method silently appears
    at the bottom of the ranking instead of erroring.  That is exactly what a re-selection
    produces if the ``_confirm`` pass is not re-run with it (see
    :func:`provisional_banner`), so the condition is checked and shouted about rather than
    left to be read out of a table.
    """
    if not len(table) or 'att_conf' not in table.columns:
        return []
    bad = table[(pd.to_numeric(table['att_conf'], errors='coerce').fillna(0) == 0)
                & (pd.to_numeric(table['fin_conf'], errors='coerce').fillna(0) == 0)]
    names = list(bad['display'])
    if names and not quiet:
        print('!' * 78)
        for _, r in bad.iterrows():
            print(f"NO CONFIRMATION RUNS for {r['display']} at its selected "
                  f"configuration ({r['config']}): 0 records matched in "
                  f"{hl.dir_name(table.attrs.get('scan', ''), 'confirm')}/.  Its "
                  f"confirmation mean is EMPTY, not bad -- it must not be ranked.  "
                  f"Either the selection moved after the pass ran (re-run the pass) or "
                  f"the match is broken.")
        print('!' * 78)
    return names


def pass_gpus(scan, kind='confirm', payload=None, results_root=None, runs=None):
    """Which GPU type each method's selected-configuration runs used, in one pass.

    Every rank, every paired difference and every selection-optimism number on a scan is
    built from one pass, and a pass is only a clean comparison BETWEEN methods if they all
    ran on the same device: GPU kernels are not bit-exact across device types, and the
    same notebooks measure that deviation at up to 4e-2 relative for Sven.  The
    confirmation passes of toy, polynomial and both CIFAR scans are homogeneous; the two
    MNIST ones are not, so this is checked and printed rather than assumed.

    ``runs`` optionally supplies the ``{method: rows}`` mapping (e.g. the frame
    :func:`confirm_runs` already built) instead of reading the pass again.

    Returns one row per (method, gpu) with ``n_runs``; ``attrs['homogeneous']`` is whether
    the whole pass ran on one device type and ``attrs['gpus']`` the device totals.
    """
    if runs is None:
        if not hl.has_pass(scan, kind, results_root=results_root):
            out = pd.DataFrame(columns=['method', 'display', 'gpu', 'n_runs'])
            out.attrs.update(scan=scan, kind=kind, homogeneous=None, gpus={})
            return out
        runs = {raw: rws for raw, (rws, _rep)
                in hl.method_runs(scan, kind, payload=payload,
                                  results_root=results_root).items()}
    rows = []
    for raw, rws in runs.items():
        if not len(rws) or 'gpu_name' not in rws.columns:
            continue
        for gpu, n in rws['gpu_name'].dropna().value_counts().items():
            rows.append({'method': hl.method_key(raw), 'display': hl.display_name(raw),
                         'gpu': str(gpu), 'n_runs': int(n)})
    out = pd.DataFrame(rows, columns=['method', 'display', 'gpu', 'n_runs'])
    totals = (out.groupby('gpu')['n_runs'].sum().sort_values(ascending=False).to_dict()
              if len(out) else {})
    out.attrs.update(scan=scan, kind=kind, gpus=totals,
                     homogeneous=(len(totals) <= 1 if totals else None))
    return out


def gpu_homogeneity_line(scan, kind='confirm', payload=None, results_root=None,
                         gpus=None, runs=None, verbose=True):
    """Print (and return) the one-line device-homogeneity verdict for a pass."""
    g = gpus if gpus is not None else pass_gpus(scan, kind, payload=payload,
                                                results_root=results_root, runs=runs)
    totals = g.attrs.get('gpus') or {}
    if not totals:
        if verbose:
            print(f'  GPU types in the {kind} pass: not recorded')
        return g
    if len(totals) == 1:
        if verbose:
            name, n = next(iter(totals.items()))
            print(f'  GPU: all {n} {kind} runs on {name} -- one device type, so a rank '
                  f'or a paired difference here compares like with like')
        return g
    if verbose:
        print(f'  GPU: the {kind} pass is NOT homogeneous -- '
              + ', '.join(f'{n} run(s) on {name}' for name, n in totals.items())
              + '.  A comparison BETWEEN methods that ran on different device types '
                'carries the cross-GPU deviation of section 6 on top of the seed spread:')
        for m, sub in g.groupby('display', sort=True):
            per = ', '.join(f'{int(r.n_runs)}x {r.gpu}' for r in sub.itertuples())
            print(f'    {m:<20s} {per}')
    return g


def summary_cell(scan, payload=None, results_root=None, print_table=True):
    """The notebook's top table (plan 2.1): confirmation seeds beside the tuning seeds.

    Returns ``(table, view)``: the full :func:`headline.confirmation_table` frame and the
    rendered :func:`headline.confirmation_view`.  Validation is what selected the
    configuration; the test loss / accuracy columns are OUTCOMES of that choice, and
    ``finished/attempted`` comes from each pass's manifest.
    """
    tbl = hl.confirmation_table(scan, payload=payload, results_root=results_root)
    view = hl.confirmation_view(tbl)
    if print_table:
        hl.print_table(view, f'{hl.scan_title(scan)} ({scan}) -- selected configuration '
                             f'of every method on the CONFIRMATION seeds, tuning seeds '
                             f'beside it; {paired.SEED_SPREAD_LABEL}')
        if tbl.attrs.get('data_seeds'):
            print(f"  data-seed replicates: {tbl.attrs['data_seeds']} "
                  f"(the tuning scan ran on {min(tbl.attrs['data_seeds'])} only)")
        missing = set(missing_confirmation(tbl))
        for _, r in tbl[~tbl['elig_conf'].astype(bool)].iterrows():
            if r['display'] in missing:
                continue          # already shouted about: no runs at all, not "0 of 5"
            print(f"  NOT ELIGIBLE on the confirmation seeds: {r['display']} "
                  f"({int(r['fin_conf'])}/{int(r['att_conf'])} finished, "
                  f"{int(r['div_conf'])} diverged) -- its mean is over the survivors only")
        # which device each method's confirmation runs used: a rank is only a clean
        # comparison if they all used the same one, and on the MNIST scans they did not
        gpu_homogeneity_line(scan, 'confirm', payload=payload, results_root=results_root)
    return tbl, view


def ranking_line(scan, summary=None, payload=None, results_root=None):
    """Sven's rank on this scan, printed and returned as a dict."""
    summary = (summary if summary is not None
               else hl.ranking_summary(scans=(scan,), payload=payload,
                                       results_root=results_root))
    row = summary[(summary['scan'] == scan) & (summary['method'] == hl.SVEN_LABEL)]
    if row.empty:
        print(f'{scan}: no Sven selection')
        return {}
    r = row.iloc[0]
    n = int(r['n_methods'])
    out = {'n_methods': n, 'rank_val': r['rank_val'], 'rank_test': r['rank_test'],
           'rank_acc': r['rank_acc'], 'val_conf': r['val_conf'],
           'test_conf': r['test_conf'], 'acc_conf': r['acc_conf']}
    bits = [f"{int(r['rank_val'])}/{n} on validation loss"]
    if np.isfinite(r['rank_test']):
        bits.append(f"{int(r['rank_test'])}/{n} on test loss")
    if np.isfinite(r['rank_acc']):
        bits.append(f"{int(r['rank_acc'])}/{n} on test accuracy")
    print(f'Sven on {hl.scan_title(scan)}, confirmation seeds: ' + ', '.join(bits))
    best = summary[summary['scan'] == scan].sort_values('rank_val').iloc[0]
    print(f"  best method by validation loss: {best['display']} "
          f"({best['val_conf']:.4g}) vs Sven {r['val_conf']:.4g}")
    out['best_method'] = best['display']
    out['best_val'] = float(best['val_conf'])
    return out


# ---------------------------------------------------------------------------
# (b) curves on three axes, for every method
# ---------------------------------------------------------------------------
def confirm_runs(scan, payload=None, results_root=None, kind='confirm'):
    """``{canonical method: rows}`` --- the runs of each method's SELECTED configuration
    in the confirmation pass.

    Matching goes through :func:`headline.selected_runs` (hyperparameter values, not the
    run_id string), so it is the same configuration the confirmation table reports.
    """
    out = {}
    for raw, (rows, _rep) in hl.method_runs(scan, kind, payload=payload,
                                            results_root=results_root).items():
        if len(rows):
            out[hl.method_key(raw)] = rows
    return out


def standalone_epoch_times(scan, payload=None, results_root=None, key=TIME_KEY):
    """``{canonical method: per-epoch time array}`` from ``{scan}_timing``.

    One selected configuration at a time, alone on a GPU: the scan's own and the
    confirmation pass's wall times are inflated by shard co-tenancy, so a
    convergence-vs-time axis may only use these.  ``key`` defaults to the synchronised
    per-epoch *training* time; a pass that did not record it falls back to
    ``epoch_times`` (wall clock, evaluation included) and the fallback is reported in
    the returned frame's ``attrs``.
    """
    out, used = {}, {}
    if not hl.has_pass(scan, 'timing', results_root=results_root):
        print(f'[times] no {hl.dir_name(scan, "timing")}/ -- the time axis is unavailable')
        return out
    for raw, (rows, _rep) in hl.method_runs(scan, 'timing', payload=payload,
                                            results_root=results_root).items():
        m = hl.method_key(raw)
        mean, _ = sa.seed_curve(rows, key)
        which = key
        if mean is None:
            mean, _ = sa.seed_curve(rows, TIME_FALLBACK)
            which = TIME_FALLBACK
        if mean is not None:
            out[m] = np.asarray(mean, dtype=float)
            used[m] = which
    fell_back = sorted(m for m, w in used.items() if w != key)
    if fell_back:
        print(f'[times] {len(fell_back)} method(s) have no {key!r}; using '
              f'{TIME_FALLBACK!r}: {fell_back}')
    return out


def method_ranking(scan, table=None, payload=None, results_root=None):
    """Method keys ordered by CONFIRMATION-seed validation loss, best first."""
    tbl = (table if table is not None
           else hl.confirmation_table(scan, payload=payload, results_root=results_root))
    ordered = tbl.sort_values('val_conf', kind='mergesort', na_position='last')
    return list(ordered['method'])


def curve_axis(rows, curve, which='val', versus='step', epoch_times=None):
    """x-values for a seed-mean ``curve`` on one of :data:`CURVE_AXES`.

    ``val`` / ``val_acc`` / ``test`` / ``train_eval`` are recorded once before training
    starts, so they are one entry longer than the per-epoch time series and must start
    at x = 0; getting this wrong shifts every validation curve by an epoch.  Returns
    None when the axis cannot be built, so the caller skips the curve instead of drawing
    it in the wrong unit.
    """
    if versus != 'time':
        return sa.epoch_axis(rows, curve, which=which, versus=versus)
    if epoch_times is None or not len(epoch_times):
        return None
    x = np.cumsum(np.asarray(epoch_times, dtype=float))
    if len(curve) == len(x) + 1:          # pre-training entry at index 0
        x = np.concatenate([[0.0], x])
    return x[:len(curve)]


def plot_curves(scan, ax, which='val', versus='step', methods=None, runs=None,
                times=None, band=True, lw=1.5, sven_lw=3.0, alpha=0.95, logx=None,
                payload=None, results_root=None, quiet=False):
    """Seed-mean confirmation curves of every method's selected configuration.

    Sven is drawn last, black and thick (:data:`style.METHOD_COLORS`); the seed band is
    +/- 1 std with the lower edge clipped for the log axis, and gets ONE legend entry
    (:func:`style.band_legend`).  Returns the methods actually drawn.
    """
    runs = runs if runs is not None else confirm_runs(scan, payload, results_root)
    if versus == 'time' and times is None:
        times = standalone_epoch_times(scan, payload, results_root)
    order = methods if methods is not None else list(runs)
    order = [m for m in order if m in runs]
    order = [m for m in order if m != hl.SVEN_LABEL] + \
            ([hl.SVEN_LABEL] if hl.SVEN_LABEL in order else [])
    drawn, banded, skipped = [], False, []
    for m in order:
        rows = runs[m]
        mean, lower, upper = sa.seed_band(rows, which)
        if mean is None:
            skipped.append(f'{m} (no usable {which} curve)')
            continue
        x = curve_axis(rows, mean, which, versus,
                       epoch_times=(times or {}).get(m) if versus == 'time' else None)
        if x is None:
            skipped.append(f'{m} (no {versus} axis)')
            continue
        n = min(len(x), len(mean))
        is_sven = m == hl.SVEN_LABEL
        ax.plot(x[:n], mean[:n], color=style.method_color(m),
                lw=sven_lw if is_sven else lw, alpha=1.0 if is_sven else alpha,
                zorder=5 if is_sven else 2, label=style.method_label(m))
        if band and len(rows) > 1:
            ax.fill_between(x[:n], lower[:n], upper[:n], color=style.method_color(m),
                            alpha=0.18, lw=0, zorder=(4.5 if is_sven else 1.5))
            banded = True
        drawn.append(m)
    if banded:
        style.band_legend(ax)
    if skipped and not quiet:
        print(f'  [curves] {which} vs {versus}: not drawn -- ' + ', '.join(skipped))
    ax.set_xlabel(axis_label(versus))
    ax.set_ylabel({'train': 'Train loss', 'val': 'Validation loss',
                   'val_acc': 'Validation accuracy', 'test': 'Test loss',
                   'test_acc': 'Test accuracy',
                   'train_eval': 'Train loss (eval mode)'}.get(which, which))
    if not which.endswith('_acc'):
        ax.set_yscale('log')
    # the time axis is logarithmic by default: the standalone cost of the selected
    # configurations spans a factor of ~20 within a scan (on MNIST, HIG needs 1150 s
    # against Sven's 60), and on a linear axis the slowest method flattens every other
    # curve against the y axis.  The step / examples axes stay linear, so the pre-training
    # point at x = 0 is visible somewhere in every figure.
    #
    # `nonpositive='mask'` is load-bearing.  Every validation curve starts with the
    # UNTRAINED model at x = 0 (see `curve_axis`), and matplotlib's default for a log
    # scale is `nonpositive='clip'`, which maps 0 to log10 = -1000 instead of dropping it:
    # the segment from the pre-training point to epoch 1 then enters the axes at the
    # left-hand edge and is drawn as a flat line at the epoch-1 loss, i.e. as data at
    # times the method was not yet measured at.  Masking leaves the first drawn point
    # where the first measurement is.
    if (versus == 'time') if logx is None else logx:
        ax.set_xscale('log', nonpositive='mask')
    ax.grid(which='both', ls='--', alpha=0.4)
    return drawn


def curve_figure(scan, which='val', top_n=6, order=None, runs=None, times=None,
                 axes_list=CURVE_AXES, figsize=(17.5, 9.5), payload=None,
                 results_root=None, fig=None):
    """The convergence figure: two rows x the three x-axes.

    Row 1 shows the ``top_n`` methods by confirmation validation loss plus Sven, so the
    race at the front is readable; row 2 shows EVERY method the scan selected, so
    nothing is hidden by the choice of six.  Returns ``(fig, axes, info)``.
    """
    import matplotlib.pyplot as plt

    runs = runs if runs is not None else confirm_runs(scan, payload, results_root)
    times = times if times is not None else standalone_epoch_times(scan, payload,
                                                                   results_root)
    order = order if order is not None else method_ranking(scan, payload=payload,
                                                           results_root=results_root)
    order = [m for m in order if m in runs]
    top = [m for m in order[:top_n]]
    if hl.SVEN_LABEL in order and hl.SVEN_LABEL not in top:
        top = top + [hl.SVEN_LABEL]
    if fig is None:
        fig, axes = plt.subplots(2, len(axes_list), figsize=figsize, squeeze=False)
    else:
        axes = np.asarray(fig.axes).reshape(2, len(axes_list))
    for j, versus in enumerate(axes_list):
        plot_curves(scan, axes[0][j], which=which, versus=versus, methods=top,
                    runs=runs, times=times, quiet=(j > 0))
        plot_curves(scan, axes[1][j], which=which, versus=versus, methods=order,
                    runs=runs, times=times, lw=1.1, alpha=0.75, quiet=True)
    axes[0][0].legend(fontsize=8, ncol=2, framealpha=0.95)
    axes[1][0].legend(fontsize=7, ncol=3, framealpha=0.95)
    n_seeds = max((len(r) for r in runs.values()), default=0)
    axes[0][0].set_title(f'Front of the field: best {len(top)} by confirmation '
                         f'validation loss', fontsize=11, loc='left')
    axes[1][0].set_title(f'All {len(order)} methods with a selected configuration',
                         fontsize=11, loc='left')
    for ax in axes.ravel():           # the global style is sized for a single panel
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.tick_params(labelsize=9)
    fig.tight_layout()
    info = {'top': top, 'all': order, 'n_runs_per_method': n_seeds,
            'axes': list(axes_list), 'which': which}
    return fig, axes, info


# ---------------------------------------------------------------------------
# (e) the standalone timing pass against the scan
# ---------------------------------------------------------------------------
def timing_join_view(tj, columns=None):
    """:func:`headline.timing_join_report` with its reproducibility column NAMED.

    The report's ``bit_reproduced`` is ``max_rel_dev <= attrs['tol']`` (1e-3 by default)
    and no one-sided divergence, i.e. "the standalone run reproduced the scan's final
    validation loss to within a tenth of a per cent".  That is not bit-identity, and no
    MLP run in this campaign is bit-reproducible across GPU types -- on ``toy_1d_scan``
    the column is True for 14 of 15 methods at deviations down to 1e-5, which a reader
    takes for bit-identity if the threshold is not on the page.  So the notebook view
    renames the column after the tolerance it actually tested.
    """
    tol = tj.attrs.get('tol')
    out = tj if columns is None else tj[[c for c in columns if c in tj.columns]]
    out = out.copy()
    if 'bit_reproduced' in out.columns and tol is not None:
        out = out.rename(columns={'bit_reproduced': f'reproduced_within_{tol:g}'})
    out.attrs.update(tj.attrs)
    return out


# ---------------------------------------------------------------------------
# (f) robustness: what fraction of a grid blew up, and where
# ---------------------------------------------------------------------------
def scan_matrix(per_scan, column, index='display', scans=None, order=None):
    """A methods x scans matrix of one column of a per-scan table.

    ``per_scan`` is ``{scan: table}`` (confirmation tables, efficiency tables, divergence
    tables --- anything with one row per method).  Rows are ordered Sven first, then
    alphabetically, and a method missing from a scan is left blank rather than filled.
    """
    frames = []
    for scan in (scans or list(per_scan)):
        tbl = per_scan.get(scan)
        if tbl is None or not len(tbl) or column not in tbl.columns:
            continue
        s = tbl.drop_duplicates(subset=[index]).set_index(index)[column]
        s.name = hl.scan_title(scan)
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    if order is None:
        names = list(out.index)
        sven = [n for n in names if n == style.method_label(hl.SVEN_LABEL)]
        order = sven + sorted(n for n in names if n not in sven)
    return out.reindex(order)


def _method_series(df):
    """The per-run optimizer name, whichever column the frame carries.

    ``analysis_helpers.add_derived`` writes a ``method`` column (and is what
    :func:`headline.load` applies); a frame straight out of
    :func:`scan_analysis.load_scan` has only the record's ``optimizer``.  Both spell some
    optimizers differently from the analysis key (``SVD``, ``JD_UPGrad``), so the result
    is canonicalised either way.
    """
    col = 'method' if 'method' in df.columns else 'optimizer'
    return df[col].map(hl.method_key)


def divergence_by_method(df, sort=True):
    """Per method: how much of its tuning grid failed, under BOTH counts.

    ``n_recorded`` is the lifecycle count (``status == 'diverged'``: the run raised or
    produced a non-finite batch loss).  ``n_diverged`` is the analysis count
    (:func:`style.is_diverged`: recorded, or a non-finite final value, or a finite
    blow-up to more than 10x the initial validation loss) --- the one that governs
    selection and every table, and the wider of the two.  Quoting only the first is how
    "Sven never diverges" gets written down: on ``toy_1d_scan`` Sven has 0 recorded and
    194 wide.

    ``n_configs_all_bad`` is the number of grid POINTS that lost every seed --- a
    configuration a tuning search would have had to discard entirely.
    """
    d = df.copy()
    d['_bad'] = d['diverged'].astype(bool) | d['failed'].astype(bool)
    d['_status'] = d.apply(style.status_of, axis=1)
    d['_cfg'] = d['run_id'].map(style.config_key)
    d['_method'] = _method_series(d)
    rows = []
    for raw, g in d.groupby('_method', sort=False):
        per_cfg = g.groupby('_cfg')['_bad'].agg(n_bad='sum', n='size')
        rows.append({
            'method': hl.method_key(raw),
            'display': hl.display_name(raw),
            'n_runs': int(len(g)),
            'n_recorded': int((g['_status'] == style.STATUS_DIVERGED).sum()),
            'n_diverged': int(g['diverged'].sum()),
            'n_failed': int(g['failed'].sum()),
            'frac_diverged': float(g['_bad'].mean()),
            'n_configs': int(len(per_cfg)),
            'n_configs_any_bad': int((per_cfg['n_bad'] > 0).sum()),
            'n_configs_all_bad': int((per_cfg['n_bad'] == per_cfg['n']).sum()),
        })
    out = pd.DataFrame(rows)
    out['frac_configs_all_bad'] = out['n_configs_all_bad'] / out['n_configs']
    if sort:
        out = out.sort_values(['frac_diverged', 'n_diverged'], ascending=False)
    out = out.reset_index(drop=True)
    out.attrs['counts'] = ('n_recorded = status field (lifecycle); n_diverged = '
                           'style.is_diverged (analysis, wider: + non-finite final '
                           'value + >10x blow-up)')
    return out


_campaign_memo = {}


def on_grid_scans(results_root=None):
    """Every TUNING-grid directory under the results root, in name order.

    ``{scan}_confirm`` / ``_timing`` / ``_diag`` re-run one already-selected
    configuration, so they are not part of any search and must not enter a grid count;
    everything else in the root is a grid.
    """
    root = Path(style.resolve_results_root(results_root))
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and not p.name.startswith('_')
                  and not p.name.endswith(('_confirm', '_timing', '_diag')))


def campaign_divergence(method=hl.SVEN_LABEL, scans=None, results_root=None,
                        verbose=False):
    """Campaign-wide: how much of ``method``'s ON-GRID work blew up, per scan.

    ``EXPERIMENTS.md`` section 7 quotes "688 of its 7,825 on-grid runs" for Sven, and that
    denominator moves: the CIFAR-CE ``rtol`` extension of plan decision 5 was still landing
    runs while these notebooks were written.  Quoting a number that changes under the
    notebook is how a paper acquires a wrong exact count, so it is computed here from the
    same records and the same rule (``style.is_diverged``, via
    :func:`divergence_by_method`) instead.

    Returns one row per scan (``n_runs``, ``n_diverged``, ``n_recorded``,
    ``frac_diverged``) with the totals in ``attrs``.  Memoised per (method, root).
    """
    key = (method, str(style.resolve_results_root(results_root)),
           tuple(scans) if scans else None)
    if key in _campaign_memo:
        return _campaign_memo[key]
    rows = []
    for scan in (scans if scans is not None else on_grid_scans(results_root)):
        df = hl.load(scan, '', results_root=results_root)
        if not len(df):
            continue
        sub = df[_method_series(df) == method]
        if not len(sub):
            continue
        d = divergence_by_method(sub).set_index('method')
        if method not in d.index:
            continue
        r = d.loc[method]
        rows.append({'scan': scan, 'n_runs': int(r['n_runs']),
                     'n_diverged': int(r['n_diverged']),
                     'n_recorded': int(r['n_recorded']),
                     'frac_diverged': float(r['frac_diverged'])})
    out = pd.DataFrame(rows).sort_values('n_diverged', ascending=False)
    out = out.reset_index(drop=True)
    out.attrs['method'] = method
    out.attrs['n_scans'] = int(len(out))
    out.attrs['n_runs'] = int(out['n_runs'].sum()) if len(out) else 0
    out.attrs['n_diverged'] = int(out['n_diverged'].sum()) if len(out) else 0
    out.attrs['n_recorded'] = int(out['n_recorded'].sum()) if len(out) else 0
    out.attrs['n_scans_affected'] = int((out['n_diverged'] > 0).sum()) if len(out) else 0
    out.attrs['frac_diverged'] = (out.attrs['n_diverged'] / out.attrs['n_runs']
                                  if out.attrs['n_runs'] else np.nan)
    _campaign_memo[key] = out
    if verbose:
        print(f"{hl.display_name(method)} on-grid campaign-wide: "
              f"{out.attrs['n_diverged']} of {out.attrs['n_runs']} runs diverged under "
              f"the analysis rule ({out.attrs['frac_diverged']:.1%}), "
              f"{out.attrs['n_recorded']} of them recorded as such by the runner; "
              f"{out.attrs['n_scans_affected']} of {out.attrs['n_scans']} tuning grids "
              f"affected")
    return out


def divergence_pattern(df, method=hl.SVEN_LABEL, index='rtol', columns='lr'):
    """Where a method's divergences sit on two grid axes, as a statement or its absence.

    The mechanistic story (loose ``rtol`` inverts near-noise Gram directions, and the risk
    grows with the learning rate) is true of ``toy_1d_scan``, where 194 of Sven's 900 runs
    blow up.  It is NOT a property of every grid: Sven has 0 of 800 on ``mnist_scan_ce``,
    1 of 640 on ``mnist_scan_labelRegression`` and 4 of 720 on ``polynomial_scan``, and a
    sentence asserting the pattern under a table of zeros costs the section its whole
    point.  So the sentence is built from the counts: ``summary`` is a plain-English
    verdict and the columns behind it are returned for the prose to quote.
    """
    frac, counts = sven_divergence_grid(df, index=index, columns=columns, method=method)
    out = {'method': method, 'axes': (index, columns), 'any': False,
           'n_runs': 0, 'n_diverged': 0}
    if frac is None:
        out['summary'] = f'no {hl.display_name(method)} runs on this grid'
        return out
    n = np.nan_to_num(counts.values.astype(float))
    bad = np.nan_to_num(frac.values.astype(float)) * n
    out['n_runs'] = int(n.sum())
    out['n_diverged'] = int(round(bad.sum()))
    out['any'] = out['n_diverged'] > 0
    rows = []
    for i, iv in enumerate(frac.index):
        for j, cv in enumerate(frac.columns):
            if bad[i, j] > 0:
                rows.append({index: float(iv), columns: float(cv),
                             'frac': float(frac.values[i, j]),
                             'n_diverged': int(round(bad[i, j])), 'n': int(n[i, j])})
    cells = pd.DataFrame(rows)
    out['cells'] = cells
    if not out['any']:
        out['summary'] = (f'NO divergence anywhere on this grid: 0 of {out["n_runs"]} '
                          f'{hl.display_name(method)} runs, at every {index} and every '
                          f'{columns}')
        out[f'{index}_affected'] = []
        out[f'{columns}_affected'] = []
        return out
    idx_bad = sorted(cells[index].unique())
    col_bad = sorted(cells[columns].unique())
    out[f'{index}_affected'] = idx_bad
    out[f'{columns}_affected'] = col_bad
    out[f'{index}_grid'] = [float(v) for v in frac.index]
    out[f'{columns}_grid'] = [float(v) for v in frac.columns]
    worst = cells.sort_values('frac', ascending=False).iloc[0]
    out['worst_cell'] = (f'{index}={float(worst[index]):g}, '
                         f'{columns}={float(worst[columns]):g}: '
                         f'{int(worst["n_diverged"])}/{int(worst["n"])} '
                         f'({float(worst["frac"]):.0%})')
    by_col = (cells.groupby(columns)['n_diverged'].sum()
              / pd.Series({float(c): n[:, j].sum()
                           for j, c in enumerate(frac.columns)}))
    out[f'frac_by_{columns}'] = {float(k): round(float(v), 4)
                                 for k, v in by_col.dropna().items()}
    monotone = bool(len(by_col.dropna()) > 1
                    and (np.diff(by_col.dropna().to_numpy()) >= 0).all())
    out[f'rises_with_{columns}'] = monotone
    # "at or near the bottom of the grid" means: the affected values ARE the lowest
    # len(idx_bad) values of the swept axis, and they are not the whole axis
    grid_sorted = sorted(out[f'{index}_grid'])
    out[f'confined_to_low_{index}'] = bool(
        len(idx_bad) < len(grid_sorted)
        and set(np.round(idx_bad, 12)) <= set(np.round(grid_sorted[:len(idx_bad)], 12)))
    out['summary'] = (
        f'{out["n_diverged"]} of {out["n_runs"]} {hl.display_name(method)} runs diverge, '
        f'at {index} ' + ', '.join(f'{v:g}' for v in idx_bad)
        + f' (of {", ".join(f"{v:g}" for v in out[f"{index}_grid"])}) and {columns} '
        + ', '.join(f'{v:g}' for v in col_bad)
        + ('; the fraction rises monotonically with ' + columns if monotone else ''))
    return out


def sven_divergence_grid(df, index='rtol', columns='lr', method=hl.SVEN_LABEL):
    """``(fraction, counts)`` pivots of a method's divergences over two grid axes.

    The fraction is NaN where the grid has no point at all (the CIFAR-CE extension round
    is a partial rectangle), which the heatmap leaves blank rather than filling with 0.
    """
    d = df[_method_series(df) == method].copy()
    if d.empty or index not in d.columns or columns not in d.columns:
        return None, None
    d['_bad'] = (d['diverged'].astype(bool) | d['failed'].astype(bool)).astype(float)
    frac = d.pivot_table(index=index, columns=columns, values='_bad', aggfunc='mean')
    counts = d.pivot_table(index=index, columns=columns, values='_bad', aggfunc='size')
    return frac, counts


def plot_divergence_grid(frac, ax, counts=None, cmap='Reds', vmax=1.0, title=None,
                         xlabel=None, ylabel=None, fmt='{:.0%}'):
    """Heatmap of a divergence-fraction pivot, with the fraction printed in each cell."""
    values = frac.values.astype(float)
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, aspect='auto', cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(frac.columns)))
    ax.set_xticklabels([f'{c:g}' for c in frac.columns], rotation=45)
    ax.set_yticks(range(len(frac.index)))
    ax.set_yticklabels([f'{i:g}' for i in frac.index])
    ax.set_xlabel(xlabel or frac.columns.name)
    ax.set_ylabel(ylabel or frac.index.name)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isfinite(v):
                ax.text(j, i, '--', ha='center', va='center', fontsize=8, color='0.5')
                continue
            n = int(counts.values[i, j]) if counts is not None else None
            text = fmt.format(v) + ('' if n is None else f'\n{int(round(v * n))}/{n}')
            ax.text(j, i, text, ha='center', va='center', fontsize=7.5,
                    color='white' if v > 0.55 * vmax else 'black')
    if title:
        ax.set_title(title, fontsize=11)
    return im


# ---------------------------------------------------------------------------
# (g) the Sven hyperparameter landscape
# ---------------------------------------------------------------------------
def sven_grid_table(scan_obj, metric=None):
    """Every Sven grid point: seed mean / std of the selection metric, seeds finished,
    diverged, eligibility --- straight from :meth:`scan_analysis.Scan.configs`, which is
    the binding selection rule (eligible -> fewest diverged -> seed mean)."""
    return scan_obj.configs(scan_obj.sven, sa.SVEN_CONFIG, metric)


def selected_sven(scan, payload=None):
    """``(chosen, selection)`` for Sven on ``scan`` from ``bench/best_configs.json``.

    ``chosen`` is ``{k, lr, rtol}`` as the RECORDS spell them; ``selection`` is the raw
    entry (its ``overrides`` are the verified statement of what ran).  The selection
    file's own ``label`` field is pre-C-B7 and is never displayed.
    """
    sel = hl.selection_methods(scan, payload)[hl.SVEN_METHOD]
    hp = hl.record_hparams(sel)
    chosen = {a: hp.get(a) for a in sa.SVEN_CONFIG}
    return chosen, sel


def grid_edges(scan_obj, chosen):
    """Where the selected Sven configuration sits on each swept axis.

    ``k = B`` is the uncapped-rank configuration --- a natural maximum, not a grid edge
    that wants extending --- and is labelled as such; any other value at the end of its
    axis is a real edge and is called one.
    """
    axes = {'k': scan_obj.ks, 'lr': scan_obj.sven_lrs, 'rtol': scan_obj.rtols}
    rows = []
    for name, values in axes.items():
        v = chosen.get(name)
        if v is None or not values:
            continue
        v = float(v)
        lo, hi = float(min(values)), float(max(values))
        if np.isclose(v, hi) and name == 'k' and int(v) == scan_obj.B:
            where = 'k = B (uncapped rank: a natural maximum, not a grid edge)'
        elif np.isclose(v, hi):
            where = 'HIGH EDGE of the grid'
        elif np.isclose(v, lo):
            where = 'LOW EDGE of the grid'
        else:
            where = 'interior'
        rows.append({'axis': name, 'selected': v, 'n_points': len(values),
                     'grid': ', '.join(f'{x:g}' for x in values), 'position': where})
    out = pd.DataFrame(rows)
    out.attrs['on_edge'] = [r['axis'] for r in rows if 'EDGE' in r['position']]
    return out


def plot_selected_marker(ax, pivot, chosen, row='k', col='lr', **kw):
    """Ring the selected configuration's cell in a ``(row, col)`` heatmap."""
    try:
        i = list(pivot.index).index(float(chosen[row]))
        j = list(pivot.columns).index(float(chosen[col]))
    except (ValueError, KeyError, TypeError):
        return None
    kw.setdefault('edgecolor', 'white')
    kw.setdefault('facecolor', 'none')
    kw.setdefault('lw', 2.4)
    import matplotlib.patches as mpatches
    patch = mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, **kw)
    ax.add_patch(patch)
    ax.plot([j], [i], marker='*', ms=11, color='white', mec='black', mew=0.6, zorder=6)
    return patch


# ---------------------------------------------------------------------------
# (h) data-seed replicates: between-instance vs between-seed spread
# ---------------------------------------------------------------------------
def instance_spread(scan, table=None, payload=None, results_root=None):
    """Between-instance against between-seed spread on the data-seed replicates.

    Toy and polynomial re-run each selected configuration on 3 *different random
    problems* (F27).  ``between_instance`` is the std of the three per-instance MEANS ---
    how much the answer depends on which problem was drawn --- and ``within_seed`` the
    seed band averaged over the instances.  A ratio above 1 says the choice of instance
    moves the answer more than the seed does, and no single-instance number should then
    be quoted without it.  Both columns come from
    :func:`headline.confirmation_table`; only the ratio is computed here.
    """
    tbl = (table if table is not None
           else hl.confirmation_table(scan, payload=payload, results_root=results_root))
    if not tbl.attrs.get('data_seeds'):
        return pd.DataFrame()
    out = tbl[['method', 'display', 'n_data_seeds', 'val_conf', 'val_conf_dsmean',
               'val_conf_dsspread', 'val_conf_seedspread', 'val_conf_ds0']].copy()
    out['ratio_instance_over_seed'] = (out['val_conf_dsspread']
                                       / out['val_conf_seedspread'].replace(0, np.nan))
    out = out.sort_values('val_conf', kind='mergesort').reset_index(drop=True)
    out.attrs['data_seeds'] = tbl.attrs['data_seeds']
    out.attrs['scan'] = scan
    return out


def plot_instance_spread(scan, ax, table=None, payload=None, results_root=None,
                         methods=None):
    """Per-instance means (one marker per data seed, seed band as the error bar) against
    the pooled confirmation mean, one column per method."""
    ds = hl.data_seed_table(scan, payload=payload, results_root=results_root)
    if ds.empty:
        return []
    order = methods or method_ranking(scan, table=table, payload=payload,
                                      results_root=results_root)
    order = [m for m in order if m in set(ds['method'])]
    seeds = sorted(ds['data_seed'].unique())
    markers = ['o', 's', '^', 'D', 'v']
    for x, m in enumerate(order):
        sub = ds[ds['method'] == m]
        for i, s in enumerate(seeds):
            r = sub[sub['data_seed'] == s]
            if r.empty:
                continue
            ax.errorbar([x + (i - (len(seeds) - 1) / 2) * 0.18], [r['val'].iloc[0]],
                        yerr=[[0], [r['val_std'].iloc[0]]],
                        fmt=markers[i % len(markers)], ms=5, lw=1.1, capsize=2,
                        color=style.method_color(m),
                        mfc='none' if i else style.method_color(m),
                        label=f'data seed {int(s)}' if x == 0 else None)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([style.method_label(m) for m in order], rotation=45, ha='right')
    ax.set_yscale('log')
    ax.set_ylabel('Final validation loss (mean over 5 model seeds)')
    ax.grid(axis='y', ls='--', alpha=0.4)
    ax.legend(fontsize=8, title='one marker = one random problem', title_fontsize=8)
    return order


# ---------------------------------------------------------------------------
# The closing "what this scan shows" numbers
# ---------------------------------------------------------------------------
def equal_budget_table(curves, n=BUDGET_EQUAL_N):
    """Every method's expected best seed-mean val loss at ONE tuning budget.

    ``curves`` is :func:`budget.best_of_n_curve`'s frame (via
    :func:`headline.best_of_n_table`).  A method whose grid has fewer than ``n`` points
    cannot draw ``n`` distinct configurations, so it is read at its own grid size -- which
    is the most generous honest reading: at a budget of eight trials a six-point grid has
    been exhausted.  ``n_used`` records which it was.

    This is the number that answers the budget objection, and reading it is the only way
    to answer it: the distinct-trajectory count says how much of Sven's grid was
    redundant, which if anything strengthens the objection.
    """
    rows = []
    for m, g in curves.groupby('method', sort=False):
        g = g.sort_values('n')
        use = g[g['n'] <= n]
        if not len(use):
            continue
        r = use.iloc[-1]
        rows.append({'method': hl.method_key(m), 'display': hl.display_name(m),
                     'grid_points': int(r['grid_points']), 'n_used': int(r['n']),
                     'expected_best': float(r['expected_best'])})
    out = pd.DataFrame(rows).sort_values('expected_best', kind='mergesort')
    out = out.reset_index(drop=True)
    out.attrs['n'] = n
    out.attrs['metric'] = curves.attrs.get('metric', hl.SELECTION_METRIC)
    out.attrs['diverged_value'] = curves.attrs.get('diverged_value')
    return out


def _equal_budget(curves, n=BUDGET_EQUAL_N):
    """The equal-budget numbers the closing cell quotes, as a flat dict."""
    at_n = equal_budget_table(curves, n)
    out = {'budget_equal_n': n, 'n_methods_at_equal_n': int(len(at_n))}
    if not len(at_n):
        return out
    names = list(at_n['method'])
    if hl.SVEN_LABEL not in names:
        return out
    i = names.index(hl.SVEN_LABEL)
    sven = at_n.iloc[i]
    out['sven_best_of_n'] = float(sven['expected_best'])
    out['sven_budget_rank_at_equal_n'] = i + 1
    out['n_better_than_sven_at_equal_n'] = i
    out['best_at_equal_n'] = at_n.iloc[0]['display']
    out['best_value_at_equal_n'] = float(at_n.iloc[0]['expected_best'])
    # how large a budget Sven needs before its expected best beats the best method's
    # value at the equal budget -- the honest form of "Sven catches up eventually"
    target = float(at_n.iloc[0]['expected_best'])
    sv = curves[curves['method'].map(hl.method_key) == hl.SVEN_LABEL].sort_values('n')
    hit = sv[sv['expected_best'] <= target]
    out['sven_n_to_match_best_at_equal_n'] = (int(hit['n'].iloc[0]) if len(hit)
                                              else None)
    return out



def closing_numbers(scan, scan_obj=None, conf=None, eff=None, ttt=None, budget_tbl=None,
                    div=None, paired_tbl=None, bon=None, df=None, campaign=None,
                    payload=None, results_root=None, verbose=True):
    """Compute (and print) every number the closing markdown cell refers to.

    Nothing here is re-derived: the confirmation, efficiency, time-to-target, budget,
    best-of-n and paired numbers come from :mod:`headline`, the divergence counts from
    :func:`divergence_by_method`.  Returned as a dict so the prose can be checked
    against it -- and so that a claim the prose wants to make ("Sven is ahead of the
    field on the steps axis", "the budget objection does not survive") has to appear here
    as a number first, which is how three wrong claims in the first draft of these
    notebooks were caught.

    ``campaign=False`` skips the campaign-wide on-grid count (~4 s, 22 scan loads).
    """
    conf = conf if conf is not None else hl.confirmation_table(
        scan, payload=payload, results_root=results_root)
    eff = eff if eff is not None else hl.efficiency_table(
        scan, payload=payload, results_root=results_root)
    ttt = ttt if ttt is not None else hl.time_to_target_table(
        scan, payload=payload, results_root=results_root)
    budget_tbl = budget_tbl if budget_tbl is not None else hl.budget_table(
        scan, results_root=results_root)
    paired_tbl = paired_tbl if paired_tbl is not None else hl.paired_vs_sven(
        scan, payload=payload, results_root=results_root)
    # the tuning grid itself: the divergence counts and the (rtol, lr) pattern come off
    # it.  `hl.load` is memoised, so this is free once the notebook has loaded the scan.
    if df is None:
        df = hl.load(scan, '', results_root=results_root)
    if div is None:
        div = divergence_by_method(df)

    ranked = conf.sort_values('val_conf', kind='mergesort', na_position='last')
    sven = conf[conf['method'] == hl.SVEN_LABEL]
    sven = sven.iloc[0] if len(sven) else None
    out = {'scan': scan, 'title': hl.scan_title(scan), 'n_methods': int(len(conf))}
    if sven is not None:
        rank = int(list(ranked['method']).index(hl.SVEN_LABEL)) + 1
        out.update({
            'sven_config': sven['config'], 'sven_rank_val': rank,
            'sven_val': float(sven['val_conf']), 'sven_val_std': float(sven['val_conf_std']),
            'sven_test': float(sven['test_conf']),
            'sven_finished': int(sven['fin_conf']), 'sven_attempted': int(sven['att_conf']),
            'sven_gap_rel': float(sven['gap_val_same_instance_rel'])
            if np.isfinite(sven.get('gap_val_same_instance_rel', np.nan))
            else float(sven['gap_val_rel']),
        })
        if np.isfinite(sven.get('acc_conf', np.nan)):
            out['sven_acc'] = float(sven['acc_conf'])
    best = ranked.iloc[0]
    out['best_method'] = best['display']
    out['best_val'] = float(best['val_conf'])
    if sven is not None and len(ranked) > 1:
        out['sven_over_best'] = float(sven['val_conf'] / best['val_conf'])

    # paired: the methods Sven beats with an interval that excludes zero, and vice versa
    sig = paired_tbl[paired_tbl['significant'].astype(bool)]
    out['n_paired'] = int(len(paired_tbl))
    out['n_sven_better_sig'] = int((sig['mean'] < 0).sum())
    out['n_sven_worse_sig'] = int((sig['mean'] > 0).sum())
    out['n_paired_inconclusive'] = int(len(paired_tbl) - len(sig))

    # efficiency.  The neighbours are here because "the same order as SOAP / Shampoo /
    # HIG and well below L-BFGS" was written by hand into the first draft and is false on
    # three of the four MLP scans (MNIST-CE: Sven 4.0 ms, Shampoo 129, L-BFGS 3.1).
    if len(eff):
        e = eff.set_index('method')
        fastest = eff.sort_values('epoch_s').iloc[0]
        out['fastest_method'] = fastest['display']
        out['fastest_epoch_s'] = float(fastest['epoch_s'])
        if hl.SVEN_LABEL in e.index:
            out['sven_epoch_s'] = float(e.loc[hl.SVEN_LABEL, 'epoch_s'])
            out['sven_ms_per_step'] = float(e.loc[hl.SVEN_LABEL, 'ms_per_step'])
            out['sven_peak_mem_mb'] = float(e.loc[hl.SVEN_LABEL, 'peak_gpu_mem_mb'])
            out['sven_slowdown_vs_fastest'] = out['sven_epoch_s'] / out['fastest_epoch_s']
            ms = eff.dropna(subset=['ms_per_step']).sort_values('ms_per_step')
            names = list(ms['method'])
            if hl.SVEN_LABEL in names:
                i = names.index(hl.SVEN_LABEL)
                out['n_ms_per_step_cheaper_than_sven'] = i
                out['n_ms_per_step_dearer_than_sven'] = len(names) - i - 1
                if i:
                    r = ms.iloc[i - 1]
                    out['ms_per_step_just_below_sven'] = (f"{r['display']} "
                                                          f"{r['ms_per_step']:.3g} ms")
                if i < len(names) - 1:
                    r = ms.iloc[i + 1]
                    out['ms_per_step_just_above_sven'] = (f"{r['display']} "
                                                          f"{r['ms_per_step']:.3g} ms")
                dearest = ms.iloc[-1]
                out['ms_per_step_dearest'] = (f"{dearest['display']} "
                                              f"{dearest['ms_per_step']:.3g} ms")

    # time to target at the median-method target.  `sven_rank_epochs_to_target` is the
    # steps-axis claim as a number: the first draft asserted "on the steps and examples
    # axes Sven is ahead of that field", and Sven is 2nd to 4th on every scan.
    med = ttt[ttt['target'] == hl.TARGET_NAMES[1.0]]
    if len(med):
        out['target_median'] = float(med['target_value'].iloc[0])
        reached = med[med['n_reached'] > 0].sort_values('sync_train_s')
        if len(reached):
            out['ttt_fastest_method'] = reached.iloc[0]['display']
            out['ttt_fastest_s'] = float(reached.iloc[0]['sync_train_s'])
        by_epochs = (med[med['n_reached'] > 0].dropna(subset=['epochs'])
                     .sort_values('epochs', kind='mergesort'))
        out['n_reached_target'] = int(len(by_epochs))
        if len(by_epochs):
            out['ttt_fewest_epochs_method'] = by_epochs.iloc[0]['display']
            out['ttt_fewest_epochs'] = float(by_epochs.iloc[0]['epochs'])
            names = list(by_epochs['method'])
            if hl.SVEN_LABEL in names:
                out['sven_rank_epochs_to_target'] = names.index(hl.SVEN_LABEL) + 1
        srow = med[med['method'] == hl.SVEN_LABEL]
        if len(srow):
            out['sven_ttt_epochs'] = float(srow['epochs'].iloc[0])
            out['sven_ttt_s'] = float(srow['sync_train_s'].iloc[0])
            out['sven_ttt_reached'] = int(srow['n_reached'].iloc[0])
        out['n_methods_never_reached'] = int((med['n_reached'] == 0).sum())

    # tuning budget: the grid size, how much of it is distinct trajectories, AND the
    # equal-budget reading of the best-of-n curves.  The second is the one that answers
    # the budget objection, and it does not answer it in Sven's favour: at n = 8 draws
    # most baselines are ahead of Sven on three of the four MLP scans.
    b = budget_tbl.set_index('method') if 'method' in budget_tbl.columns else budget_tbl
    if hl.SVEN_LABEL in b.index:
        out['sven_grid_points'] = int(b.loc[hl.SVEN_LABEL, 'grid_points'])
        out['sven_distinct'] = int(b.loc[hl.SVEN_LABEL, 'distinct'])
        out['sven_distinct_frac'] = float(b.loc[hl.SVEN_LABEL, 'distinct_frac'])
    if bon is not False:
        curves = bon if bon is not None else budget.best_of_n_curve(
            df, metric=hl.SELECTION_METRIC, diverged='worst')
        out.update(_equal_budget(curves))

    # robustness
    d = div.set_index('method')
    if hl.SVEN_LABEL in d.index:
        out['sven_grid_runs'] = int(d.loc[hl.SVEN_LABEL, 'n_runs'])
        out['sven_grid_diverged'] = int(d.loc[hl.SVEN_LABEL, 'n_diverged'])
        out['sven_grid_recorded'] = int(d.loc[hl.SVEN_LABEL, 'n_recorded'])
        out['sven_grid_frac'] = float(d.loc[hl.SVEN_LABEL, 'frac_diverged'])
    worst = div.iloc[0]
    out['worst_method'] = worst['display']
    out['worst_frac'] = float(worst['frac_diverged'])
    out['n_grid_runs'] = int(div['n_runs'].sum())
    out['n_grid_diverged'] = int(div['n_diverged'].sum())

    # WHERE Sven's divergences sit -- as a verdict computed from this grid's own counts,
    # not as the toy-scan pattern asserted over a table of zeros
    pat = divergence_pattern(df)
    out['sven_div_pattern'] = pat['summary']
    out['sven_div_any'] = bool(pat['any'])
    if pat['any']:
        out['sven_div_rtols_affected'] = [f'{v:g}' for v in pat['rtol_affected']]
        out['sven_div_worst_cell'] = pat['worst_cell']
        out['sven_div_rises_with_lr'] = bool(pat.get('rises_with_lr'))
        out['sven_div_confined_to_low_rtol'] = bool(pat.get('confined_to_low_rtol'))

    # campaign-wide, all 22 tuning grids: quoted from EXPERIMENTS.md section 7 in the
    # first draft, which is a count that moves while the extension round lands
    if campaign is not False:
        camp = (campaign if isinstance(campaign, pd.DataFrame)
                else campaign_divergence(results_root=results_root))
        out['campaign_grids'] = int(camp.attrs['n_scans'])
        out['campaign_sven_runs'] = int(camp.attrs['n_runs'])
        out['campaign_sven_diverged'] = int(camp.attrs['n_diverged'])
        out['campaign_sven_recorded'] = int(camp.attrs['n_recorded'])
        out['campaign_sven_frac'] = float(camp.attrs['frac_diverged'])
        out['campaign_grids_affected'] = int(camp.attrs['n_scans_affected'])

    # data-seed replicates
    spread = instance_spread(scan, table=conf, payload=payload, results_root=results_root)
    if len(spread):
        s = spread[spread['method'] == hl.SVEN_LABEL]
        out['n_data_seeds'] = int(spread['n_data_seeds'].iloc[0])
        out['median_instance_over_seed'] = float(
            spread['ratio_instance_over_seed'].median(skipna=True))
        if len(s):
            out['sven_between_instance'] = float(s['val_conf_dsspread'].iloc[0])
            out['sven_within_seed'] = float(s['val_conf_seedspread'].iloc[0])

    if verbose:
        width = max(len(k) for k in out)
        print(f'--- numbers behind the closing cell: {out["title"]} ({scan}) ---')
        for k, v in out.items():
            print(f'  {k:<{width}}  ' + (f'{v:.6g}' if isinstance(v, float) else str(v)))
    return out
