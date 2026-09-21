"""Paired differences between two configurations, matched by model seed (C-A6).

Two configurations run on the same ``model_seed`` share the initialisation and the
data order, so their difference removes the seed-to-seed variation that dominates
the unpaired spread.  Comparing the two seed MEANS throws that away; the per-seed
difference keeps it, and its spread is the right uncertainty for "does A beat B".

Wording matters here as much as the arithmetic (C-A6): the band drawn everywhere
else in this analysis is one standard deviation over seeds, which is NOT a
confidence interval on the mean -- :data:`SEED_SPREAD_LABEL` is the label to use
for it.  Where an interval on the MEAN difference is wanted this module gives a
Student-t interval and labels it as such (:func:`interval_label`).

Caveat (F26): the legacy scans share one ``loader_seed`` across model seeds, so a
pair differs only in initialisation until C-S2 varies the data order too.  The
pairing is still valid; it just removes less variance than it will later.

    best = analysis_helpers.best_per_method(scan.df)
    sven = best[best.method == 'Sven'].iloc[0]
    tbl  = paired.paired_table(scan.df, best[best.method != 'Sven'], sven)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import analysis_helpers as ah
import style

#: THE label for the seed band / error bar drawn in every other plot.  Not a
#: confidence interval: it is the spread of the seeds themselves.
SEED_SPREAD_LABEL = r'$\pm$ 1 std over seeds'


def interval_label(level=0.95, n=None):
    """Label for the interval on a MEAN paired difference (never "the seed band")."""
    return f'{level:.0%} $t$-interval on the mean' + (f' ($n$={n} paired seeds)' if n else '')


def _config_runs(df, row) -> pd.DataFrame:
    """:func:`analysis_helpers.config_runs`, but empty instead of raising for a
    configuration with no finished run: ``config_table`` joins those in with an outer
    join, so their ``run_ids`` is NaN (not a list) and ``isin`` would raise TypeError.
    Comparing against an all-diverged configuration is a legitimate question with the
    answer "no paired seeds", and these grids are full of them."""
    ids = row.get('run_ids')
    if not isinstance(ids, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
        return df.iloc[:0]
    return ah.config_runs(df, row)


def paired_runs(df, row_a, row_b, metric='final_val_loss', seed_col='model_seed') -> pd.DataFrame:
    """The seeds both configurations finished, as ``[seed, a, b, diff]``.

    ``row_a`` / ``row_b`` are rows of :func:`analysis_helpers.config_table` (or of
    :func:`analysis_helpers.best_per_method`): the runs behind them come from
    :func:`analysis_helpers.config_runs`, i.e. the finished, non-diverged ones.
    A seed with more than one run for the same configuration is averaged.  A
    configuration no seed finished contributes nothing and the frame comes back
    empty (``n = 0``) rather than raising.
    """
    style.assert_selection_metric(metric, 'paired_runs')
    if metric not in df.columns:
        df = ah.add_derived(df)
    take = lambda row: (_config_runs(df, row).groupby(seed_col, dropna=False)[metric]
                        .mean().rename(None))
    a, b = take(row_a), take(row_b)
    out = pd.DataFrame({seed_col: sorted(set(a.index) & set(b.index))})
    for c, s in (('a', a), ('b', b)):        # float even when there is no shared seed
        out[c] = pd.to_numeric(out[seed_col].map(s), errors='coerce').astype(float)
    out = out[np.isfinite(out['a']) & np.isfinite(out['b'])].reset_index(drop=True)
    out['diff'] = out['a'] - out['b']
    out.attrs['n_a_only'] = int(len(set(a.index) - set(b.index)))
    out.attrs['n_b_only'] = int(len(set(b.index) - set(a.index)))
    return out


def paired_difference(df, row_a, row_b, metric='final_val_loss', seed_col='model_seed',
                      level=0.95) -> dict:
    """Mean, std and Student-t interval of ``a - b`` over the shared seeds.

    ``std`` is over the paired differences (ddof=1) and ``sem = std / sqrt(n)``;
    the interval is ``mean +/- t_{1-(1-level)/2, n-1} * sem``, and is NaN for
    ``n < 2``.  ``n_unpaired`` counts the seeds only one side finished -- they are
    excluded, so a lopsided comparison is visible rather than silent.
    """
    p = paired_runs(df, row_a, row_b, metric=metric, seed_col=seed_col)
    d = np.asarray(p['diff'], dtype=float)
    n = d.size
    mean = float(d.mean()) if n else np.nan
    sd = float(d.std(ddof=1)) if n > 1 else np.nan
    sem = sd / np.sqrt(n) if n > 1 else np.nan
    half = float(stats.t.ppf(0.5 + level / 2, n - 1) * sem) if n > 1 else np.nan
    return {
        'metric': metric, 'n': n, 'mean': mean, 'std': sd, 'sem': sem,
        'ci_low': mean - half, 'ci_high': mean + half, 'half_width': half, 'level': level,
        'mean_a': float(p['a'].mean()) if n else np.nan,
        'mean_b': float(p['b'].mean()) if n else np.nan,
        'n_unpaired': p.attrs['n_a_only'] + p.attrs['n_b_only'],
        # Sign convention stated explicitly: a < b means A is better for a loss.
        'a_better': int((d < 0).sum()), 'runs': p,
    }


def paired_table(df, rows, reference, metric='final_val_loss', seed_col='model_seed',
                 level=0.95, label_col='method') -> pd.DataFrame:
    """One paired comparison per row of ``rows`` against the single ``reference`` row.

    The difference is ``reference - other``, so a negative mean means the reference
    configuration has the lower loss.  ``significant`` is simply "the interval
    excludes zero" -- with 3-5 seeds and no multiplicity correction it is a
    readability aid, not a test to quote.
    """
    out = []
    ref_label = reference.get(label_col, 'reference')
    for _, r in (rows.iterrows() if isinstance(rows, pd.DataFrame) else enumerate(rows)):
        s = paired_difference(df, reference, r, metric=metric, seed_col=seed_col, level=level)
        s.pop('runs')
        out.append({label_col: r.get(label_col), 'reference': ref_label, **s})
    tbl = pd.DataFrame(out)
    if len(tbl):
        tbl['significant'] = (tbl['ci_low'] > 0) | (tbl['ci_high'] < 0)
    tbl.attrs['interval'] = interval_label(level)
    tbl.attrs['metric'] = metric
    return tbl


def fmt_difference(s, spec='.3g') -> str:
    """``mean +/- half-width (level t-interval, n=...)`` for one row / summary dict."""
    return (f"{s['mean']:{spec}} ± {s['half_width']:{spec}} "
            f"({s['level']:.0%} t-interval, n={int(s['n'])})")


def plot_paired(tbl, ax, label_col='method', spec='.2g'):
    """Horizontal t-intervals of the paired differences, ordered best-first.

    The zero line is the reference configuration; a bar entirely to its left means
    the interval on the MEAN difference excludes zero in the reference's favour.
    The axis label names the sign convention.
    """
    d = tbl.sort_values('mean').reset_index(drop=True)
    y = np.arange(len(d))
    ax.errorbar(d['mean'], y, xerr=d['half_width'], fmt='o', color='k', capsize=3, lw=1.2,
                label=tbl.attrs.get('interval', interval_label()))
    ax.axvline(0.0, color='crimson', ls='--', lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{v}' for v in d[label_col]], fontsize=9)
    ax.invert_yaxis()
    ref = d['reference'].iloc[0] if len(d) else 'reference'
    ax.set_xlabel(f"paired difference in {style.metric_label(tbl.attrs.get('metric', 'final_val_loss'), seed_mean=False)}"
                  f"  ({ref} $-$ other, matched model seed)")
    ax.grid(axis='x', ls='--', alpha=0.5)
    ax.legend(fontsize=8, frameon=False)
