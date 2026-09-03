"""Shared helpers for the study analysis notebooks (Gram-backend result format).

Each result row has an ``optimizer`` field ('SVD' == Sven, else the baseline name)
and a nested ``losses`` dict with per-epoch curves ('train','val','val_acc',...)
plus scalars ('total_time','peak_gpu_mem_mb'). These helpers flatten the pieces
the notebooks need and pick the best config per method.
"""
import numpy as np
import pandas as pd


def loss_curve(row, which='val'):
    """The per-epoch curve `which` ('train'|'val'|'val_acc'|'train_acc') or None."""
    L = row.get('losses')
    if not isinstance(L, dict):
        return None
    return L.get(which)


def _final(row, which):
    c = loss_curve(row, which)
    if not c:
        return np.nan
    v = c[-1]
    return float(v) if v is not None else np.nan


def _scalar(L, key):
    return L.get(key) if isinstance(L, dict) else np.nan


def add_derived(df):
    """Add flat columns used across the notebooks (final losses/acc, time, mem,
    a tidy `method` label, and LM perplexity)."""
    df = df.copy()
    df['final_val_loss'] = df.apply(lambda r: _final(r, 'val'), axis=1)
    df['final_train_loss'] = df.apply(lambda r: _final(r, 'train'), axis=1)
    df['final_val_acc'] = df.apply(lambda r: _final(r, 'val_acc'), axis=1)
    df['total_time'] = df['losses'].apply(lambda L: _scalar(L, 'total_time'))
    df['peak_gpu_mem_mb'] = df['losses'].apply(lambda L: _scalar(L, 'peak_gpu_mem_mb'))
    df['method'] = df['optimizer'].apply(lambda o: 'Sven' if o == 'SVD' else str(o))
    df['val_ppl'] = np.exp(df['final_val_loss'].clip(upper=20))  # LM only
    return df


def valid(df, metric='final_val_loss', max_val=1e6):
    """Drop diverged / NaN runs on `metric`."""
    m = pd.to_numeric(df[metric], errors='coerce')
    return df[np.isfinite(m) & (m.abs() < max_val)]


def best_per_method(df, by='final_val_loss', minimize=True, extra_group=None):
    """Best row per method (optionally within each value of `extra_group`),
    selected by `by`. Returns a DataFrame sorted by `by`."""
    d = valid(df, metric=by)
    keys = ['method'] + (extra_group or [])
    if d.empty:
        return d
    idx = (d.groupby(keys)[by].idxmin() if minimize else d.groupby(keys)[by].idxmax())
    return d.loc[idx].sort_values(by, ascending=minimize)


def steps_to_target(row, target, which='val'):
    """First epoch index whose `which` loss <= target, else NaN (for critical-batch
    / convergence-speed analyses)."""
    c = loss_curve(row, which)
    if not c:
        return np.nan
    for i, v in enumerate(c):
        if v is not None and v <= target:
            return i
    return np.nan


# Consistent method ordering/colour intent: Sven first (highlight), baselines after.
def method_order(methods):
    ms = list(methods)
    front = [m for m in ms if m == 'Sven']
    return front + sorted(m for m in ms if m != 'Sven')
