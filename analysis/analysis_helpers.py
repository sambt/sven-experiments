"""Shared helpers for the study analysis notebooks (Gram-backend result format).

Each result row has an ``optimizer`` field ('SVD' == Sven, else the baseline name)
and a nested ``losses`` dict with per-epoch curves ('train','val','val_acc',...)
plus scalars ('total_time','peak_gpu_mem_mb'). These helpers flatten the pieces
the notebooks need and pick the best config per method -- by SEED MEAN, with the
shared conventions of ANALYSIS_FIXES.md (diverged = failed; clipped seed bands).
"""
import numpy as np
import pandas as pd

from scan_analysis import seed_band  # noqa: F401  (re-exported: mean + clipped seed band of a curve)
from style import clipped_yerr, config_eligible, final_value, is_diverged
from style import method_color  # noqa: F401  (re-exported: the global optimizer colours)


def loss_curve(row, which='val'):
    """The per-epoch curve `which` ('train'|'val'|'val_acc'|'train_acc') or None.

    Per-BATCH series ('train_batch', 'val_batch', 'batch_times_*') are no longer
    in `losses`: they live in `{scan}/diag/{run_id}.npz`. Read them with
    `sv_diagnostics.batch_curve(row, ...)` / `style.load_diagnostics(row)`.
    """
    L = row.get('losses')
    if not isinstance(L, dict):
        return None
    return L.get(which)


def _final(row, which):
    """Final value of a curve; NaN if the run diverged (the one rule: style.final_value)."""
    return final_value(loss_curve(row, which))


def _scalar(L, key):
    return L.get(key) if isinstance(L, dict) else np.nan


def add_derived(df):
    """Add flat columns used across the notebooks (final losses/acc, time, mem,
    a tidy `method` label, LM perplexity, and the `diverged` flag)."""
    df = df.copy()
    df['final_val_loss'] = df.apply(lambda r: _final(r, 'val'), axis=1)
    df['final_train_loss'] = df.apply(lambda r: _final(r, 'train'), axis=1)
    df['final_val_acc'] = df.apply(lambda r: _final(r, 'val_acc'), axis=1)
    df['total_time'] = df['losses'].apply(lambda L: _scalar(L, 'total_time'))
    df['peak_gpu_mem_mb'] = df['losses'].apply(lambda L: _scalar(L, 'peak_gpu_mem_mb'))
    df['method'] = df['optimizer'].apply(lambda o: 'Sven' if o == 'SVD' else str(o))
    df['val_ppl'] = np.exp(df['final_val_loss'].clip(upper=20))  # LM only
    # diverged = failed (style.is_diverged, the same definition as scan_analysis.Scan)
    df['diverged'] = df.apply(lambda r: is_diverged(loss_curve(r, 'train'), loss_curve(r, 'val')), axis=1)
    df.loc[df['diverged'], ['final_val_loss', 'final_train_loss', 'final_val_acc', 'val_ppl']] = np.nan
    return df


def valid(df, metric='final_val_loss', max_val=1e6):
    """Runs with a finite `metric` below `max_val`.  For ad-hoc filtering only --
    selection goes through :func:`config_table`, which keeps the diverged runs so it
    can count them."""
    m = pd.to_numeric(df[metric], errors='coerce')
    return df[np.isfinite(m) & (m.abs() < max_val)]


# Columns that are outcomes or run bookkeeping, never part of a configuration.
OUTCOMES = ['final_val_loss', 'final_train_loss', 'final_val_acc', 'total_time',
            'peak_gpu_mem_mb', 'val_ppl']
_NOT_CONFIG = {'model_seed', 'loader_seed', 'run_id', 'diag_file', 'method', 'diverged',
               '_scan', *OUTCOMES}
_SCALAR = (str, bool, int, float, np.integer, np.floating)


def config_columns(df, seed_col='model_seed'):
    """The hyperparameter columns that jointly identify a configuration: every scalar
    column that is not the seed, run bookkeeping or an outcome."""
    skip = _NOT_CONFIG | {seed_col}
    return [c for c in df.columns if c not in skip and df[c].notna().any()
            and df[c].dropna().map(lambda v: isinstance(v, _SCALAR)).all()]


def config_table(df, metric='final_val_loss', minimize=True, seed_col='model_seed'):
    """One row per CONFIGURATION (not per run), ranked best-first.

    Runs are grouped by :func:`config_columns`.  For every outcome ``q`` in
    :data:`OUTCOMES` the row carries the seed mean ``q``, the seed std ``q_std``
    (ddof=1) and the lowest seed ``q_min`` (for :func:`style.clipped_band`).

    Diverged = failed (see ANALYSIS_FIXES.md, A4): a diverged run is not in the means.
    ``n_seeds`` counts the seeds that finished and ``n_diverged`` the ones that did not;
    ``run_ids`` lists the finished runs (see :func:`config_runs`).  Ranking is

    1. ``eligible`` -- more than half of the scan's seeds finished;
    2. fewest diverged seeds (dropping a config's failures from its mean flatters it,
       so a config that blows up on some seeds never beats one that finishes them all);
    3. seed-mean ``metric``; remaining exact ties broken by the config values, so the
       result is the same on every machine.
    """
    df = df if 'diverged' in df.columns else add_derived(df)
    config = config_columns(df, seed_col)
    outcomes = [q for q in OUTCOMES if q in df.columns]
    ok = df[~df['diverged'] & df[metric].notna()]
    g = ok.groupby(config, dropna=False)
    out = g[outcomes].mean()
    out = out.join(g[outcomes].std(ddof=1).add_suffix('_std'))
    out = out.join(g[outcomes].min().add_suffix('_min'))
    out = out.join(g.size().rename('n_seeds'))
    out = out.join(g['run_id'].agg(list).rename('run_ids'))
    out = out.join(df.groupby(config, dropna=False).size().rename('n_runs'), how='outer')
    out = out.reset_index()
    out['n_seeds'] = out['n_seeds'].fillna(0).astype(int)
    out['n_diverged'] = out.pop('n_runs') - out['n_seeds']
    out['n_missing'] = df[seed_col].nunique() - out['n_seeds'] - out['n_diverged']  # no result file
    out['eligible'] = config_eligible(out['n_seeds'], df[seed_col].nunique())
    out['method'] = out['optimizer'].apply(lambda o: 'Sven' if o == 'SVD' else str(o))
    order = ['eligible', 'n_diverged', metric] + config
    ascending = [False, True, minimize] + [True] * len(config)
    return out.sort_values(order, ascending=ascending, kind='mergesort',
                           na_position='last').reset_index(drop=True)


def best_per_method(df, by='final_val_loss', minimize=True, extra_group=None):
    """Best CONFIGURATION per method (optionally within each value of `extra_group`),
    chosen by the seed mean of `by` -- never a single lucky run.

    Returns rows of :func:`config_table`: the columns notebooks already use
    (``method``, ``final_val_loss``, ``final_train_loss``, ``final_val_acc``,
    ``total_time``, ...) are seed MEANS, with ``*_std`` / ``*_min`` / ``n_seeds`` /
    ``n_diverged`` beside them.  A method with no eligible configuration is absent.
    """
    cfg = config_table(df, metric=by, minimize=minimize)
    cfg = cfg[cfg['eligible']]
    keys = ['method'] + list(extra_group or [])
    best = cfg.groupby(keys, dropna=False, sort=False).head(1)   # cfg is ranked best-first
    return best.sort_values(by, ascending=minimize).reset_index(drop=True)


def seed_mean_best(df, by, metric='final_val_loss', minimize=True):
    """Best configuration per value of `by` (e.g. per ``kappa`` / ``param_fraction``),
    all methods pooled.  Same rows and ranking as :func:`best_per_method`."""
    cfg = config_table(df, metric=metric, minimize=minimize)
    cfg = cfg[cfg['eligible']]
    return cfg.groupby(by, dropna=False, sort=False).head(1).sort_values(by).reset_index(drop=True)


def config_runs(df, config_row):
    """The finished seed-runs behind one row of :func:`config_table` /
    :func:`best_per_method` (e.g. to draw that configuration's curves)."""
    return df[df['run_id'].isin(config_row['run_ids'])]


def fmt_pm(row, q, spec='.3e'):
    """``mean +/- std`` of outcome `q` from a config row, for the printed tables."""
    std = row.get(f'{q}_std')
    return f"{row[q]:{spec}} ± {(0.0 if pd.isna(std) else std):{spec}}"


def errorbar_seeds(ax, rows, x, q, **kw):
    """``ax.errorbar`` of the seed mean of outcome `q` vs column `x`, with THE seed
    error bar: +/- 1 std, lower end clipped at the lowest seed (style.clipped_yerr)."""
    rows = rows.sort_values(x)
    kw = {'marker': 'o', 'capsize': 3, **kw}
    return ax.errorbar(rows[x], rows[q], yerr=clipped_yerr(rows[q], rows[f'{q}_std'],
                                                          rows[f'{q}_min']), **kw)


def epochs_to_target(row, target, which='val'):
    """First EPOCH whose `which` loss <= target, else NaN.  The val curve has a
    pre-training entry at index 0, so the index is the number of epochs trained."""
    c = loss_curve(row, which)
    if not c:
        return np.nan
    for i, v in enumerate(c):
        if v is not None and v <= target:
            return i
    return np.nan


steps_to_target = epochs_to_target   # old name; the unit was always epochs


def epochs_to_target_table(df, mult=1.2, which='val', group=('method', 'batch_size')):
    """Critical-batch table: epochs to reach a common target, per `group` cell.

    * target = `mult` x the best SEED-MEAN final loss of any eligible configuration;
    * a configuration counts only if EVERY seed reaches the target (a seed that
      diverged or never got there disqualifies it); its score is the seed-mean
      epochs-to-target (``e2t``, with ``e2t_std`` / ``e2t_min``);
    * per cell, the configuration with the lowest mean ``e2t``.  Cells where no
      configuration qualifies are kept, with ``reached=False`` and NaN ``e2t``, so they
      can be drawn as "not reached" instead of silently vanishing.

    Returns ``(table, target)``.
    """
    df = df if 'diverged' in df.columns else add_derived(df)
    metric = {'val': 'final_val_loss', 'train': 'final_train_loss'}[which]
    cfg = config_table(df, metric=metric)
    target = float(cfg.loc[cfg['eligible'], metric].min()) * mult
    n_expected = df['model_seed'].nunique()
    e2t = df.set_index('run_id').apply(lambda r: epochs_to_target(r, target, which), axis=1)
    rows = []
    for _, c in cfg.iterrows():
        v = e2t.reindex(c['run_ids']).to_numpy(dtype=float) if isinstance(c['run_ids'], list) else np.array([])
        ok = len(v) == n_expected and np.isfinite(v).all()
        rows.append({'reached': ok, 'e2t': v.mean() if ok else np.nan,
                     'e2t_std': v.std(ddof=1) if ok and len(v) > 1 else 0.0 if ok else np.nan,
                     'e2t_min': v.min() if ok else np.nan})
    cfg = pd.concat([cfg, pd.DataFrame(rows, index=cfg.index)], axis=1)
    cfg = cfg.sort_values(['reached', 'e2t'], ascending=[False, True], kind='mergesort')
    table = cfg.groupby(list(group), dropna=False, sort=False).head(1)
    return table.sort_values(list(group)).reset_index(drop=True), target


# Consistent method ordering/colour intent: Sven first (highlight), baselines after.
def method_order(methods):
    ms = list(methods)
    front = [m for m in ms if m == 'Sven']
    return front + sorted(m for m in ms if m != 'Sven')


def n_params_of(df, fallback=None, what=''):
    """Parameter count P from the result records (``n_params``, written by
    ``generic_scan._scan_facts`` since 2026-09-17).  Older records lack it; then
    ``fallback`` is used and a warning printed, so a hard-coded P is at least visible
    (ANALYSIS_FIXES C25; RERUNS_NEEDED 6)."""
    if 'n_params' in df.columns and df['n_params'].notna().any():
        return int(df['n_params'].dropna().iloc[0])
    print(f'  [n_params] {what}: records carry no n_params -- using the hard-coded P={fallback}')
    return fallback

