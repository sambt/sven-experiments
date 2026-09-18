"""Tuning-budget disclosure from the existing scans (C-A4).

The budget objection (FABLE_CRITIQUES.md, Codex Mid 1) is that Sven is tuned over
72-128 grid points while most baselines get four learning rates, so its advantage
could be search effort rather than the method.  Two numbers answer it without a
single new run, and both come out of a scan DataFrame that is already loaded:

**How large is the grid really** -- :func:`trajectory_table`.  A Sven grid point is
not an independent trial: whenever the rtol-rank stays below ``k``, a larger ``k``
truncates nothing extra and the run is the *identical trajectory*, down to the last
bit of the stored final loss.  Counting DISTINCT final validation losses per
``(lr, seed)`` therefore measures the search actually performed.  Verified against
Fable's audit: toy 12 of 18, polynomial 11.3 of 18, MNIST-CE 22.4 of 32, MNIST
label-reg 16 of 32 -- an effective budget of 48 / 45 / 90 / 64 against 4.

**What the grid bought** -- :func:`best_of_n_curve`.  The expected best seed-mean
validation loss when ``n`` configurations are drawn uniformly at random from a
method's grid, exactly (order statistics of a finite population, no resampling
noise).  Reading each method's curve at the same ``n`` compares methods at an
EQUAL tuning budget; where the curves flatten is how much of the grid mattered.

Selection stays the seed-mean final validation loss (:func:`style.assert_selection_metric`);
nothing here may be driven by a test metric.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import analysis_helpers as ah
import style
from style import method_color


# ---------------------------------------------------------------------------
# How many distinct trajectories does a grid contain?
# ---------------------------------------------------------------------------
def _n_distinct(values, sig=None):
    """Distinct values in a metric series: every non-finite entry (a divergence)
    counts as ONE further group, so a method whose grid mostly blows up does not
    look like a wide search.  ``sig`` rounds to that many significant digits
    before comparing (default: exact -- an identical trajectory reproduces the
    stored float bit for bit)."""
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if sig is not None and finite.size:
        with np.errstate(divide='ignore'):
            mag = np.floor(np.log10(np.abs(np.where(finite == 0, 1.0, finite))))
        finite = np.round(finite, decimals=(sig - 1 - mag.astype(int)).max())
    return int(len(np.unique(finite)) + (v.size - finite.size))


def trajectory_table(df, metric='final_val_loss', seed_col='model_seed',
                     within=('lr',), sig=None) -> pd.DataFrame:
    """Grid points vs distinct trajectories, one row per method.

    ``within`` are the knobs the grid is broken down BY (the learning rate, by
    default): the per-group columns are means over the ``(seed, *within)`` groups,
    which is the shape Fable's audit reports ("12 of 18" = 12 distinct trajectories
    among the 18 ``(k, rtol)`` points at one learning rate and one seed).

    Columns

    * ``grid_points`` -- configurations in this method's grid (seeds collapsed);
    * ``grid_per_group`` / ``distinct_per_group`` -- mean over ``(seed, *within)``
      groups of the configurations in the group and of the distinct ``metric``
      values among them;
    * ``distinct`` -- ``distinct_per_group`` x the number of ``within`` values, i.e.
      the effective number of trials the whole grid bought for one seed;
    * ``distinct_frac`` -- ``distinct / grid_points``;
    * ``n_runs`` / ``n_seeds`` / ``n_diverged`` -- what the count is based on.
    """
    style.assert_selection_metric(metric, 'trajectory_table')
    if metric not in df.columns or 'method' not in df.columns:
        df = ah.add_derived(df)
    keys = [c for c in ah.config_columns(df, seed_col) if c in df.columns]
    within = [w for w in within if w in df.columns]
    rows = []
    for method, d in df.groupby('method', dropna=False, sort=False):
        groups = [g for _, g in d.groupby([seed_col, *within], dropna=False)] if within else \
                 [g for _, g in d.groupby(seed_col, dropna=False)]
        n_within = int(d[within].drop_duplicates().shape[0]) if within else 1
        per_group = [_n_distinct(g[metric], sig) for g in groups] or [0]
        distinct_per_group = float(np.mean(per_group))
        grid_points = int(d.drop_duplicates(keys).shape[0]) if keys else len(d)
        rows.append({
            'method': method, 'n_runs': len(d), 'n_seeds': int(d[seed_col].nunique()),
            'n_diverged': int(d['diverged'].sum()) if 'diverged' in d.columns else 0,
            'grid_points': grid_points,
            'grid_per_group': float(np.mean([len(g) for g in groups])) if groups else np.nan,
            'distinct_per_group': distinct_per_group,
            'distinct': distinct_per_group * n_within,
            'distinct_frac': distinct_per_group * n_within / grid_points if grid_points else np.nan,
        })
    out = pd.DataFrame(rows).sort_values('grid_points', ascending=False).reset_index(drop=True)
    out.attrs['within'] = tuple(within)
    return out


# ---------------------------------------------------------------------------
# What did the grid buy?  Best-of-n at an equal budget.
# ---------------------------------------------------------------------------
def best_of_n(values, n, minimize=True, replace=False):
    """Expected best of ``n`` configurations drawn uniformly at random from ``values``.

    Exact, by order statistics of the finite population (no resampling noise).
    Without replacement the ``i``-th best of ``M`` is the winner with probability
    ``C(M-1-i, n-1) / C(M, n)``; with replacement, with probability
    ``((M-i)/M)^n - ((M-i-1)/M)^n``.  NaN for an empty population.
    """
    v = np.sort(np.asarray(values, dtype=float))
    if not minimize:
        v = v[::-1]                       # index 0 is always "best"
    M = v.size
    if M == 0 or n < 1:
        return np.nan
    if replace:
        w = [((M - i) / M) ** n - ((M - i - 1) / M) ** n for i in range(M)]
    else:
        n = min(int(n), M)
        tot = math.comb(M, n)
        w = [math.comb(M - 1 - i, n - 1) / tot for i in range(M)]
    return float(np.dot(v, w))


def best_of_n_curve(df, metric='final_val_loss', minimize=True, n_max=None, replace=False,
                    diverged='worst', seed_col='model_seed') -> pd.DataFrame:
    """Expected best seed-mean ``metric`` vs budget ``n``, for every method.

    The population is one value per CONFIGURATION -- the seed mean from
    :func:`analysis_helpers.config_table`, so the curve compares like with like
    with the selection rule used everywhere else.

    ``diverged`` scores a configuration that has no finished seed, which a random
    search would still spend a trial on:  ``'worst'`` (default) gives it the worst
    seed mean anywhere in the scan, ``'drop'`` removes it from the population (and
    flatters methods that diverge a lot), a float uses that value.

    Returns long rows ``[method, n, expected_best, grid_points, n_scored, n_filled]``
    with ``attrs['diverged_value']``.
    """
    style.assert_selection_metric(metric, 'best_of_n_curve')
    cfg = ah.config_table(df, metric=metric, minimize=minimize, seed_col=seed_col)
    finite = cfg[metric][np.isfinite(cfg[metric])]
    fill = (np.nan if diverged == 'drop' else
            (float(finite.max() if minimize else finite.min()) if diverged == 'worst' else float(diverged)))
    rows = []
    for method, d in cfg.groupby('method', dropna=False, sort=False):
        v = np.asarray(d[metric], dtype=float)
        n_filled = int((~np.isfinite(v)).sum())
        v = v[np.isfinite(v)] if diverged == 'drop' else np.where(np.isfinite(v), v, fill)
        top = int(n_max or v.size)
        for n in range(1, min(top, v.size) + 1 if not replace else top + 1):
            rows.append({'method': method, 'n': n, 'expected_best': best_of_n(v, n, minimize, replace),
                         'grid_points': int(len(d)), 'n_scored': int(v.size), 'n_filled': n_filled})
    out = pd.DataFrame(rows, columns=['method', 'n', 'expected_best', 'grid_points',
                                      'n_scored', 'n_filled'])
    out.attrs['diverged_value'] = fill
    out.attrs['metric'] = metric
    return out


def plot_best_of_n(curves, ax, methods=None, logy=True, **plot_kw):
    """The :func:`best_of_n_curve` frame, one line per method (global method colours)."""
    methods = methods or list(dict.fromkeys(curves['method']))
    for m in methods:
        d = curves[curves.method == m].sort_values('n')
        if d.empty:
            continue
        ax.plot(d['n'], d['expected_best'], '-', lw=2.2 if m == 'Sven' else 1.4,
                color=method_color(m), label=f"{m} ({int(d.grid_points.iloc[0])} pts)", **plot_kw)
    ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel('tuning budget $n$ (configurations drawn at random)')
    ax.set_ylabel(f"expected best {style.metric_label(curves.attrs.get('metric', 'final_val_loss'))}")
    ax.grid(True, which='both', ls='--', alpha=0.4)
