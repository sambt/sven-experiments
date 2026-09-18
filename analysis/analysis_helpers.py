"""Shared helpers for the study analysis notebooks (Gram-backend result format).

Each result row has an ``optimizer`` field ('SVD' == Sven, else the baseline name)
and a nested ``losses`` dict with per-epoch curves ('train','val','val_acc',...)
plus scalars ('total_time','peak_gpu_mem_mb'). These helpers flatten the pieces
the notebooks need and pick the best config per method -- by SEED MEAN, with the
shared conventions of ANALYSIS_FIXES.md (diverged = failed; clipped seed bands).
"""
import numpy as np
import pandas as pd

import style
from scan_analysis import seed_band  # noqa: F401  (re-exported: mean + clipped seed band of a curve)
from style import (assert_selection_metric, clipped_yerr, config_eligible, final_value,
                   hparam_columns, is_diverged, is_failed, status_of)
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
    a tidy `method` label, LM perplexity, and the `diverged` / `failed` flags).

    Schema 2 (C-R1): a record's ``status`` decides ``diverged`` too, and an
    ``oom`` / ``error`` record -- an incomplete attempt the runner retries -- is
    flagged ``failed``; both are kept in the table (so they can be counted) and
    left out of every mean.  ``final_test_*`` are added when the records carry a
    test curve: OUTCOMES, reported beside a selected configuration, never a
    selection metric (C-E1).
    """
    df = df.copy()
    df['final_val_loss'] = df.apply(lambda r: _final(r, 'val'), axis=1)
    df['final_train_loss'] = df.apply(lambda r: _final(r, 'train'), axis=1)
    df['final_val_acc'] = df.apply(lambda r: _final(r, 'val_acc'), axis=1)
    nan_cols = ['final_val_loss', 'final_train_loss', 'final_val_acc', 'val_ppl']
    for col, which in (('final_test_loss', 'test'), ('final_test_acc', 'test_acc')):
        if df['losses'].apply(lambda L, w=which: _scalar(L, w) is not None).any():
            df[col] = df.apply(lambda r, w=which: _final(r, w), axis=1)
            nan_cols.append(col)
    df['total_time'] = df['losses'].apply(lambda L: _scalar(L, 'total_time'))
    df['peak_gpu_mem_mb'] = df['losses'].apply(lambda L: _scalar(L, 'peak_gpu_mem_mb'))
    df['method'] = df['optimizer'].apply(lambda o: 'Sven' if o == 'SVD' else str(o))
    df['val_ppl'] = np.exp(df['final_val_loss'].clip(upper=20))  # LM only
    # diverged = failed (style.is_diverged, the same definition as scan_analysis.Scan)
    df['diverged'] = df.apply(
        lambda r: is_diverged(loss_curve(r, 'train'), loss_curve(r, 'val'), status_of(r)), axis=1)
    df['failed'] = df.apply(lambda r: is_failed(status_of(r)), axis=1)
    df.loc[df['diverged'] | df['failed'], [c for c in nan_cols if c in df.columns]] = np.nan
    return df


def valid(df, metric='final_val_loss', max_val=1e6):
    """Runs with a finite `metric` below `max_val`.  For ad-hoc filtering only --
    selection goes through :func:`config_table`, which keeps the diverged runs so it
    can count them."""
    m = pd.to_numeric(df[metric], errors='coerce')
    return df[np.isfinite(m) & (m.abs() < max_val)]


# Columns that are outcomes or run bookkeeping, never part of a configuration.
# `final_test_*` are OUTCOMES: present only when the records carry a test curve,
# shown beside the selected configuration, never selected on (C-E1).
OUTCOMES = ['final_val_loss', 'final_train_loss', 'final_val_acc',
            'final_test_loss', 'final_test_acc', 'total_time',
            'peak_gpu_mem_mb', 'val_ppl']
# Back-compat aliases; configuration identity now comes from the allow-list in
# style (see style.HPARAM_COLUMNS for why auto-detection had to go).
_BACKEND = set(style.BACKEND_COLUMNS)
_NOT_CONFIG = set(style.PROVENANCE_COLUMNS) | _BACKEND | {'method', 'diverged', 'failed',
                                                          *OUTCOMES}


def config_columns(df, seed_col='model_seed'):
    """The hyperparameter columns that jointly identify a configuration.

    The ALLOW-LIST :data:`style.HPARAM_COLUMNS`, not auto-detection: schema 2 puts
    ``status``, ``run_hash``, ``git_sha``, ``host``, timestamps, ``n_test``,
    ``steps_per_epoch``, ... on every record, and the old "every scalar column that
    is not on a small denylist" rule would have taken each of them as part of a
    configuration's identity -- one configuration per run, every seed mean a single
    run, every table in this module silently collapsed (C-A2)."""
    return hparam_columns(df, seed_col)


_manifest_memo = {}


def expected_run_ids(df, results_root=None):
    """The run_ids the scan(s) in ``df`` intend to contain, from their manifests
    (C-R2), or an empty set for a legacy scan that has none.  Memoised per
    ``(scan, root)``: :func:`config_table` is called several times per notebook."""
    out = set()
    if '_scan' not in df.columns:
        return out
    root = style.resolve_results_root(results_root)
    for name in df['_scan'].dropna().unique():
        key = (str(name), str(root))
        if key not in _manifest_memo:
            _manifest_memo[key] = style.manifest_run_ids(name, results_root=root)
        out |= _manifest_memo[key]
    return out


def missing_configs(df, expected=None, results_root=None):
    """Configurations the manifest expects that have **no record at all** -- the ones
    :func:`config_table` cannot show, because a configuration with no file has no
    hyperparameter columns to group on.  Identified by
    :func:`style.config_key` (the run_id minus its seed suffix).

    Returns a DataFrame ``[config_key, n_missing]`` (empty without a manifest)."""
    expected = expected_run_ids(df, results_root) if expected is None else expected
    have = {style.config_key(r) for r in df.get('run_id', [])}
    rows = [{'config_key': key, 'n_missing': n}
            for key, n in sorted(style.expected_per_config(expected).items())
            if key not in have]
    return pd.DataFrame(rows, columns=['config_key', 'n_missing'])


def config_table(df, metric='final_val_loss', minimize=True, seed_col='model_seed',
                 expected=None, results_root=None):
    """One row per CONFIGURATION (not per run), ranked best-first.

    Runs are grouped by :func:`config_columns`.  For every outcome ``q`` in
    :data:`OUTCOMES` the row carries the seed mean ``q``, the seed std ``q_std``
    (ddof=1) and the lowest seed ``q_min`` (for :func:`style.clipped_band`).
    ``metric`` may not be a test quantity (:func:`style.assert_selection_metric`).

    Diverged = failed (see ANALYSIS_FIXES.md, A4): a diverged run is not in the means.
    Counts per configuration:

    * ``n_seeds`` / ``finished`` -- seeds that finished;
    * ``n_diverged`` -- seeds that diverged (curve rule or ``status: diverged``);
    * ``n_failed`` -- ``oom`` / ``error`` records: incomplete attempts (C-R1), not
      in the means and NOT counted as divergences;
    * ``n_missing`` -- runs with no result file, and ``attempted`` the number the
      **manifest** expects (C-R2), falling back to the seed count when the scan has
      no manifest.  A configuration with no file *at all* still cannot appear as a
      row; :func:`missing_configs` lists those, also on ``.attrs['missing_configs']``.

    ``run_ids`` lists the finished runs (see :func:`config_runs`).  Ranking is

    1. ``eligible`` -- more than half of the configuration's runs finished;
    2. fewest diverged seeds (dropping a config's failures from its mean flatters it,
       so a config that blows up on some seeds never beats one that finishes them all);
    3. seed-mean ``metric``; remaining exact ties broken by the config values, so the
       result is the same on every machine.
    """
    assert_selection_metric(metric, 'config_table')
    df = df if 'diverged' in df.columns else add_derived(df)
    failed = df['failed'] if 'failed' in df.columns else pd.Series(False, index=df.index)
    config = config_columns(df, seed_col)
    outcomes = [q for q in OUTCOMES if q in df.columns]
    ok = df[~df['diverged'] & ~failed & df[metric].notna()]
    g = ok.groupby(config, dropna=False)
    g_all = df.assign(_failed=failed.astype(int)).groupby(config, dropna=False)
    out = g[outcomes].mean()
    out = out.join(g[outcomes].std(ddof=1).add_suffix('_std'))
    out = out.join(g[outcomes].min().add_suffix('_min'))
    out = out.join(g.size().rename('n_seeds'))
    out = out.join(g['run_id'].agg(list).rename('run_ids'))
    out = out.join(g_all.size().rename('n_runs'), how='outer')
    out = out.join(g_all['_failed'].sum().rename('n_failed'))
    out = out.join(g_all['run_id'].agg(
        lambda s: frozenset(style.config_key(r) for r in s)).rename('_keys'))
    out = out.reset_index()
    out['n_seeds'] = out['n_seeds'].fillna(0).astype(int)
    out['n_failed'] = out['n_failed'].fillna(0).astype(int)
    out['n_diverged'] = out.pop('n_runs') - out['n_seeds'] - out['n_failed']
    # How many runs this configuration was supposed to have: the manifest when the
    # scan has one (so a seed that never wrote a file is visible), else -- as ever --
    # the number of distinct seeds anywhere in the table.  Summed over EVERY
    # config_key in the row (:func:`style.expected_runs`), since a schema-drifted
    # column set can split one configuration over two rows and back.
    expected = expected_run_ids(df, results_root) if expected is None else expected
    per_config, n_seeds_total = style.expected_per_config(expected), df[seed_col].nunique()
    out['attempted'] = [style.expected_runs(ks, per_config, n_seeds_total)
                        for ks in out.pop('_keys')]
    out['finished'] = out['n_seeds']
    out['n_missing'] = (out['attempted'] - out['n_seeds'] - out['n_diverged']
                        - out['n_failed']).clip(lower=0)
    out['eligible'] = config_eligible(out['n_seeds'], out['attempted'])
    out['method'] = out['optimizer'].apply(lambda o: 'Sven' if o == 'SVD' else str(o))
    order = ['eligible', 'n_diverged', metric] + config
    ascending = [False, True, minimize] + [True] * len(config)
    out = out.sort_values(order, ascending=ascending, kind='mergesort',
                          na_position='last').reset_index(drop=True)
    out.attrs['missing_configs'] = missing_configs(df, expected)
    out.attrs['missing_run_ids'] = style.missing_run_ids(df, expected)
    # No notebook reads `.attrs` (and pandas drops it through a merge), while a
    # configuration with no result file has no ROW here -- so say it out loud too,
    # the way `scan_analysis.load_scan` reports the same thing for a whole scan.
    # Unprinted, a wholly missing configuration stays exactly as invisible as
    # before C-A2.
    miss = out.attrs['missing_configs']
    if len(miss):
        print(f'  [config_table] {len(miss)} configuration(s) with no result file at '
              f'all ({int(miss["n_missing"].sum())} runs): '
              f'{list(miss["config_key"][:3])}'
              f'{" ..." if len(miss) > 3 else ""}')
    return out


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
    e2t = df.set_index('run_id').apply(lambda r: epochs_to_target(r, target, which), axis=1)
    rows = []
    for _, c in cfg.iterrows():
        v = e2t.reindex(c['run_ids']).to_numpy(dtype=float) if isinstance(c['run_ids'], list) else np.array([])
        # every run the configuration was supposed to have must reach the target
        ok = len(v) == int(c['attempted']) and np.isfinite(v).all()
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

