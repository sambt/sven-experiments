#!/usr/bin/env python3
"""
Profile analysis: timing & memory comparison of Sven vs standard optimizers.

Generates plots and LaTeX tables from the JSON files produced by
experiments/optimizer_profile.py.

Usage:
  python analysis/profile_analysis.py profile_results/toy_1d_profile
  python analysis/profile_analysis.py profile_results/cifar10_resnet_ce_profile \\
      --output-dir analysis/plots/profile/cifar10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from style import set_style as _base_set_style

# Line widths used throughout the module. The thin-line value is for dashed
# reference lines / horizontal axhlines; the thick-line value is for data series.
_LW_THIN  = 2.5
_LW_THICK = 3.0


def set_style() -> None:
    _base_set_style()
    plt.rcParams['lines.linewidth'] = _LW_THICK

# ---------------------------------------------------------------------------
# Fixed orderings and colour scheme
# ---------------------------------------------------------------------------

STANDARD_ORDER = [
    'Adam', 'AdamW', 'SGD', 'RMSprop',
    'LBFGS', 'PolyakSGD',
    'HIG', 'JD_UPGrad',
]

BASELINE_LABELS = {
    'JD_UPGrad': 'JD (UPGrad)',
}

# Modes treated as "baseline" methods (one bar / horizontal reference line each)
BASELINE_MODES = ('standard', 'hig', 'jd')

# Set from --dataset; used by _title() to prefix plot titles and table captions
_DATASET_NAME: str | None = None


def _title(t: str) -> str:
    return f"{_DATASET_NAME}: {t}" if _DATASET_NAME else t


def _baseline_label(opt: str) -> str:
    return BASELINE_LABELS.get(opt, opt)


def _baseline_label_tex(opt: str) -> str:
    return _baseline_label(opt).replace('_', r'\_')


def _format_model_size(mb: float) -> str:
    if mb >= 1024.0:
        return f"{mb / 1024.0:.2f} GB"
    if mb < 1.0:
        return f"{mb * 1024.0:.1f} kB"
    return f"{mb:.1f} MB"


def _format_n_params(n: int) -> str:
    return f"{int(n):,} params"


def _model_info_str(n_params: int | None, model_mb: float | None) -> str | None:
    parts = []
    if n_params is not None:
        parts.append(_format_n_params(n_params))
    if model_mb is not None:
        parts.append(_format_model_size(model_mb))
    return ' / '.join(parts) if parts else None

SVD_MODE_ORDER  = ['torch', 'randomized', 'randomized_v2']
SVD_MODE_LABELS = {'torch': 'full SVD', 'randomized': 'rand.', 'randomized_v2': 'rand. v2'}
# Sven palette is intentionally outside the tab10 hue space used by baselines:
# black / deep magenta / dark teal — high-contrast vs Adam blue, AdamW orange,
# SGD green, RMSprop red, LBFGS purple, etc.
SVD_MODE_CMAPS  = {'torch': 'Greys',     'randomized': 'PuRd',    'randomized_v2': 'BuGn'}
SVD_MODE_BASE   = {'torch': '#000000',   'randomized': '#c2185b', 'randomized_v2': '#00695c'}

# One colour per position in STANDARD_ORDER (seaborn "bright" palette)
_BRIGHT = sns.color_palette('bright', n_colors=max(10, len(STANDARD_ORDER)))
STANDARD_COLORS = {opt: _BRIGHT[i % len(_BRIGHT)] for i, opt in enumerate(STANDARD_ORDER)}

# Metric columns aggregated from the raw DataFrame
_METRICS = [
    'time_mean_ms', 'time_std_ms',
    'mem_peak_mb', 'mem_resident_mb', 'mem_transient_mb',
    'mem_baseline_mb', 'mem_overhead_mb',
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_profile_results(results_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for f in sorted(results_dir.glob('*.json')):
        with open(f) as fh:
            data = json.load(fh)
        row: dict[str, Any] = {'run_id': data['run_id']}
        row.update(data['config'])
        m = data['memory']
        row['mem_baseline_mb']  = m['baseline_model_bytes'] / 1024**2
        row['mem_resident_mb']  = m['resident_bytes_mean']  / 1024**2
        row['mem_peak_mb']      = m['peak_bytes_mean']      / 1024**2
        row['mem_peak_max_mb']  = m['peak_bytes_max']       / 1024**2
        row['mem_transient_mb'] = m['transient_delta_bytes_mean']    / 1024**2
        row['mem_overhead_mb']  = m['peak_over_baseline_bytes_mean'] / 1024**2
        t = data['time']
        row['time_mean_ms'] = t['mean_ms']
        row['time_std_ms']  = t['std_ms']
        if 'detailed_phases' in data:
            row['detailed_phases'] = data['detailed_phases']
        records.append(row)

    if not records:
        raise FileNotFoundError(f"No JSON profile results in {results_dir}")
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} profile runs from {results_dir}")
    return df


def _most_common(series: pd.Series):
    return Counter(series.dropna().tolist()).most_common(1)[0][0]


def _resolve_mlp_width(df_at_bs: pd.DataFrame, requested: int | None) -> int | None:
    if 'mlp_width' not in df_at_bs.columns or not df_at_bs['mlp_width'].notna().any():
        return None
    return requested if requested is not None else int(_most_common(df_at_bs['mlp_width']))


# ---------------------------------------------------------------------------
# Aggregation — average over lr/rtol/kappa (don't affect timing/memory)
# ---------------------------------------------------------------------------

def aggregate_standard(df: pd.DataFrame) -> pd.DataFrame:
    std = df[df['mode'].isin(BASELINE_MODES)].copy()
    if std.empty:
        return pd.DataFrame()

    # LBFGS: max_iter does affect timing — keep only the most-common value
    lbfgs_mask = std['optimizer'] == 'LBFGS'
    if lbfgs_mask.any() and 'lbfgs_max_iter' in std.columns:
        mi = _most_common(std.loc[lbfgs_mask, 'lbfgs_max_iter'].dropna())
        std = std[~lbfgs_mask | (std['lbfgs_max_iter'] == mi)].copy()

    agg = std.groupby('optimizer', sort=False)[_METRICS].mean().reset_index()
    n_params = std.groupby('optimizer', sort=False)['n_params'].first().reset_index()
    return agg.merge(n_params, on='optimizer')


def aggregate_svd(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    svd = df[df['mode'] == 'svd'].copy()
    if svd.empty:
        return pd.DataFrame()
    if 'microbatch_size' in svd.columns:
        svd = svd[_is_default_mb(svd['microbatch_size'])]
    if 'param_fraction' in svd.columns:
        svd = svd[_is_default_pf(svd['param_fraction'])]

    agg = svd.groupby(['svd_mode', 'k'], sort=False)[_METRICS].mean().reset_index()
    agg['k_fraction'] = agg['k'] / batch_size
    n_params = svd.groupby(['svd_mode', 'k'], sort=False)['n_params'].first().reset_index()
    agg = agg.merge(n_params, on=['svd_mode', 'k'])

    # Carry through detailed_phases from the first run that has it
    phases_list: list[Any] = []
    for _, row in agg.iterrows():
        match = svd[(svd['svd_mode'] == row['svd_mode']) & (svd['k'] == row['k'])]
        phases = None
        if 'detailed_phases' in match.columns:
            valid = match['detailed_phases'].apply(lambda x: x is not None)
            if valid.any():
                phases = match.loc[valid, 'detailed_phases'].iloc[0]
        phases_list.append(phases)
    agg['detailed_phases'] = phases_list
    return agg


# ---------------------------------------------------------------------------
# Aggregation by model width (for scaling plots)
# ---------------------------------------------------------------------------

def aggregate_svd_by_width(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate SVD runs by (svd_mode, k, n_params), averaging over lr/rtol/kappa.

    Filters to default microbatch_size (1 or None) and param_fraction (1.0 or None).
    """
    svd = df[df['mode'] == 'svd'].copy()
    if svd.empty:
        return pd.DataFrame()
    if 'microbatch_size' in svd.columns:
        svd = svd[_is_default_mb(svd['microbatch_size'])]
    if 'param_fraction' in svd.columns:
        svd = svd[_is_default_pf(svd['param_fraction'])]
    if svd.empty:
        return pd.DataFrame()
    agg = svd.groupby(['svd_mode', 'k', 'n_params'], sort=False)[_METRICS].mean().reset_index()
    bs  = int(_most_common(svd['batch_size']))
    agg['k_fraction'] = agg['k'] / bs
    return agg


def aggregate_standard_by_width(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate standard optimizer runs by (optimizer, n_params), averaging over lr/etc."""
    std = df[df['mode'].isin(BASELINE_MODES)].copy()
    if std.empty:
        return pd.DataFrame()
    lbfgs_mask = std['optimizer'] == 'LBFGS'
    if lbfgs_mask.any() and 'lbfgs_max_iter' in std.columns:
        mi = _most_common(std.loc[lbfgs_mask, 'lbfgs_max_iter'].dropna())
        std = std[~lbfgs_mask | (std['lbfgs_max_iter'] == mi)].copy()
    return std.groupby(['optimizer', 'n_params'], sort=False)[_METRICS].mean().reset_index()


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def order_standard(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return agg
    present = [o for o in STANDARD_ORDER if o in agg['optimizer'].values]
    extra   = sorted(o for o in agg['optimizer'].unique() if o not in STANDARD_ORDER)
    rank    = {o: i for i, o in enumerate(present + extra)}
    return agg.iloc[agg['optimizer'].map(rank).argsort()].reset_index(drop=True)


def order_svd(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return agg
    mode_rank = {m: i for i, m in enumerate(SVD_MODE_ORDER)}
    agg = agg.copy()
    agg['_mr'] = agg['svd_mode'].map(lambda m: mode_rank.get(m, 99))
    return agg.sort_values(['_mr', 'k']).drop(columns='_mr').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _svd_bar_colors(svd_agg: pd.DataFrame) -> list:
    colors = []
    for mode in SVD_MODE_ORDER:
        rows = svd_agg[svd_agg['svd_mode'] == mode]
        if rows.empty:
            continue
        n    = len(rows)
        cmap = plt.get_cmap(SVD_MODE_CMAPS[mode])
        for i in range(n):
            colors.append(cmap(0.35 + 0.50 * i / max(n - 1, 1)))
    return colors


def _svd_line_color(mode: str):
    return SVD_MODE_BASE.get(mode, 'black')


def _std_color(opt: str):
    return STANDARD_COLORS.get(opt, 'grey')


def _svd_label(row: pd.Series) -> str:
    return f"Sven $k$={int(row['k'])} ({SVD_MODE_LABELS.get(row['svd_mode'], row['svd_mode'])})"


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)
    print(f"  Saved {path.stem}.pdf")


# ---------------------------------------------------------------------------
# Plot 1 & 3: Horizontal bar charts (time or memory)
# ---------------------------------------------------------------------------

def _bar_chart(
    std_agg: pd.DataFrame,
    svd_agg: pd.DataFrame,
    std_vals: np.ndarray, std_errs: np.ndarray,
    svd_vals: np.ndarray, svd_errs: np.ndarray,
    xlabel: str,
    title: str,
    out_path: Path,
) -> None:
    set_style()

    n_svd = len(svd_agg)
    n_std = len(std_agg)
    gap   = 1.2   # extra blank space between groups (in bar-unit steps)

    y_svd = np.arange(n_svd, dtype=float)
    y_std = (y_svd[-1] + gap + 1 + np.arange(n_std, dtype=float)) if n_svd else np.arange(n_std, dtype=float)

    figheight = max(4.0, 0.38 * (n_svd + n_std) + 2.0)
    fig, ax = plt.subplots(figsize=(9, figheight))

    if n_svd:
        svd_colors = _svd_bar_colors(svd_agg)
        ax.barh(y_svd, svd_vals, xerr=svd_errs, color=svd_colors,
                capsize=3, error_kw={'linewidth': 1.2}, height=0.72)

    if n_std:
        std_cols = [_std_color(o) for o in std_agg['optimizer']]
        ax.barh(y_std, std_vals, xerr=std_errs, color=std_cols,
                capsize=3, error_kw={'linewidth': 1.2}, height=0.72)

    all_y      = np.concatenate([y_svd, y_std]) if (n_svd and n_std) else (y_svd if n_svd else y_std)
    all_labels = (
        [_svd_label(r) for _, r in svd_agg.iterrows()]
        + [_baseline_label(o) for o in std_agg['optimizer']]
    )

    ax.set_yticks(all_y)
    ax.set_yticklabels(all_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(_title(title), pad=10)

    # Separator between SVD and standard groups
    if n_svd and n_std:
        sep = (y_svd[-1] + y_std[0]) / 2.0
        ax.axhline(sep, color='#999999', linewidth=0.9, linestyle='--')

    fig.tight_layout()
    _save(fig, out_path)


def _bs_suffix(batch_size: int | None) -> str:
    return f" ($B = {int(batch_size)}$)" if batch_size is not None else ''


def plot_time_bars(std_agg: pd.DataFrame, svd_agg: pd.DataFrame, out_path: Path,
                   batch_size: int | None = None) -> None:
    _bar_chart(
        std_agg, svd_agg,
        std_vals=std_agg['time_mean_ms'].values,
        std_errs=std_agg['time_std_ms'].values,
        svd_vals=svd_agg['time_mean_ms'].values,
        svd_errs=svd_agg['time_std_ms'].values,
        xlabel='Step time (ms)',
        title='Per-step wall-clock time' + _bs_suffix(batch_size),
        out_path=out_path,
    )


def plot_memory_bars(std_agg: pd.DataFrame, svd_agg: pd.DataFrame, out_path: Path,
                     batch_size: int | None = None) -> None:
    _bar_chart(
        std_agg, svd_agg,
        std_vals=std_agg['mem_peak_mb'].values,
        std_errs=np.zeros(len(std_agg)),
        svd_vals=svd_agg['mem_peak_mb'].values,
        svd_errs=np.zeros(len(svd_agg)),
        xlabel='Peak GPU memory (MB)',
        title='Peak GPU memory per step' + _bs_suffix(batch_size),
        out_path=out_path,
    )


# ---------------------------------------------------------------------------
# Plot 2 & 4: k-scaling line plots
# ---------------------------------------------------------------------------

def _k_scaling_plot(
    svd_agg: pd.DataFrame,
    std_agg: pd.DataFrame,
    svd_col: str,
    svd_err_col: str | None,
    std_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    legend_loc: str = 'best',
    legend_ncol: int = 1,
    n_params: int | None = None,
    model_mb: float | None = None,
    log_y: bool = False,
) -> None:
    set_style()
    fig, ax = plt.subplots()

    for mode in SVD_MODE_ORDER:
        rows = svd_agg[svd_agg['svd_mode'] == mode].sort_values('k')
        if rows.empty:
            continue
        color = _svd_line_color(mode)
        label = f"Sven ({SVD_MODE_LABELS.get(mode, mode)})"
        xs    = rows['k'].values
        ys    = rows[svd_col].values
        ax.plot(xs, ys, marker='o', color=color, label=label)
        if svd_err_col and svd_err_col in rows.columns:
            errs = rows[svd_err_col].values
            ax.fill_between(xs, ys - errs, ys + errs, alpha=0.18, color=color)

    if not std_agg.empty:
        for _, row in std_agg.iterrows():
            ax.axhline(row[std_col], color=_std_color(row['optimizer']),
                       linestyle='--', linewidth=_LW_THIN, alpha=0.85,
                       label=_baseline_label(row['optimizer']))

    ax.set_xscale('log', base=2)
    ax.set_xlabel('$k$')
    ax.set_ylabel(ylabel)
    ax.set_title(_title(title))
    ax.legend(loc=legend_loc, fontsize=10, ncol=legend_ncol)

    info = _model_info_str(n_params, model_mb)
    if info:
        # Anchor opposite the legend if the legend is upper-left, else upper-right.
        if 'upper' in legend_loc and 'left' in legend_loc:
            x_anchor, ha = 0.98, 'right'
        else:
            x_anchor, ha = 0.02, 'left'
        ax.text(x_anchor, 0.98, info, transform=ax.transAxes,
                ha=ha, va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#bbbbbb', alpha=0.85))

    if log_y:
        ax.set_yscale('log')
        ax.set_ylim(bottom=10)
    else:
        ax.set_ylim(bottom=0)
    # Add headspace above data so an upper-anchored legend doesn't overlap lines.
    if 'upper' in legend_loc:
        ymin, ymax = ax.get_ylim()
        if log_y:
            log_min, log_max = np.log10(ymin), np.log10(ymax)
            ax.set_ylim(ymin, 10 ** (log_min + (log_max - log_min) * 1.25))
        else:
            ax.set_ylim(ymin, ymin + (ymax - ymin) * 1.25)

    fig.tight_layout()
    _save(fig, out_path)


def plot_time_vs_k(svd_agg: pd.DataFrame, std_agg: pd.DataFrame, out_path: Path,
                   batch_size: int | None = None,
                   n_params: int | None = None,
                   model_mb: float | None = None) -> None:
    _k_scaling_plot(
        svd_agg, std_agg,
        svd_col='time_mean_ms', svd_err_col='time_std_ms', std_col='time_mean_ms',
        legend_loc='upper left', legend_ncol=3,
        ylabel='Step time (ms)',
        title='Per-step time vs $k$' + _bs_suffix(batch_size),
        out_path=out_path,
        n_params=n_params, model_mb=model_mb,
    )


def plot_memory_vs_k(svd_agg: pd.DataFrame, std_agg: pd.DataFrame, out_path: Path,
                     batch_size: int | None = None,
                     n_params: int | None = None,
                     model_mb: float | None = None) -> None:
    _k_scaling_plot(
        svd_agg, std_agg,
        svd_col='mem_peak_mb', svd_err_col=None, std_col='mem_peak_mb',
        legend_loc='upper left', legend_ncol=3,
        ylabel='Peak GPU memory (MB)',
        title='Peak GPU memory vs $k$' + _bs_suffix(batch_size),
        out_path=out_path,
        n_params=n_params, model_mb=model_mb,
        log_y=True,
    )


# ---------------------------------------------------------------------------
# Plot 5: Overhead ratios vs k (relative to Adam)
# ---------------------------------------------------------------------------

def _adam_value(std_agg: pd.DataFrame, col: str) -> float | None:
    if std_agg.empty:
        return None
    base = std_agg[std_agg['optimizer'] == 'Adam']
    if base.empty:
        return None
    return float(base.iloc[0][col])


def plot_time_overhead_vs_k(svd_agg: pd.DataFrame, std_agg: pd.DataFrame, out_path: Path,
                            batch_size: int | None = None,
                            n_params: int | None = None,
                            model_mb: float | None = None) -> None:
    adam_t = _adam_value(std_agg, 'time_mean_ms')
    if adam_t is None:
        print(f"  [skip] {out_path.name}: no Adam baseline")
        return
    svd_r = svd_agg.copy()
    svd_r['time_mean_ms'] = svd_r['time_mean_ms'] / adam_t
    if 'time_std_ms' in svd_r.columns:
        svd_r['time_std_ms'] = svd_r['time_std_ms'] / adam_t
    std_r = std_agg.copy()
    std_r['time_mean_ms'] = std_r['time_mean_ms'] / adam_t
    _k_scaling_plot(
        svd_r, std_r,
        svd_col='time_mean_ms', svd_err_col='time_std_ms', std_col='time_mean_ms',
        legend_loc='upper left', legend_ncol=3,
        ylabel='Step time / Adam',
        title='Per-step time overhead vs $k$' + _bs_suffix(batch_size),
        out_path=out_path,
        n_params=n_params, model_mb=model_mb,
    )


def plot_memory_overhead_vs_k(svd_agg: pd.DataFrame, std_agg: pd.DataFrame, out_path: Path,
                              batch_size: int | None = None,
                              n_params: int | None = None,
                              model_mb: float | None = None) -> None:
    adam_m = _adam_value(std_agg, 'mem_peak_mb')
    if adam_m is None:
        print(f"  [skip] {out_path.name}: no Adam baseline")
        return
    svd_r = svd_agg.copy()
    svd_r['mem_peak_mb'] = svd_r['mem_peak_mb'] / adam_m
    std_r = std_agg.copy()
    std_r['mem_peak_mb'] = std_r['mem_peak_mb'] / adam_m
    _k_scaling_plot(
        svd_r, std_r,
        svd_col='mem_peak_mb', svd_err_col=None, std_col='mem_peak_mb',
        legend_loc='upper left', legend_ncol=3,
        ylabel='Peak memory / Adam',
        title='Peak memory overhead vs $k$' + _bs_suffix(batch_size),
        out_path=out_path,
        n_params=n_params, model_mb=model_mb,
        log_y=True,
    )


# ---------------------------------------------------------------------------
# Plot 6: Memory over step phases (only when detailed=true was run)
# ---------------------------------------------------------------------------

def plot_memory_breakdown(svd_agg: pd.DataFrame, out_path: Path) -> None:
    detail_rows = svd_agg[svd_agg['detailed_phases'].apply(lambda x: x is not None)]
    if detail_rows.empty:
        return

    print("  Detailed phase data found — plotting memory breakdown.")
    set_style()

    n    = len(detail_rows)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    phase_labels: list[str] = []

    for idx, (_, row) in enumerate(detail_rows.iterrows()):
        ax     = axes[idx // ncols][idx % ncols]
        phases = row['detailed_phases']
        if not phase_labels:
            phase_labels = [p['phase'].split('. ', 1)[-1] for p in phases]

        current = [p['mean_current_bytes'] / 1024**2 for p in phases]
        peak    = [p['mean_peak_bytes']    / 1024**2 for p in phases]
        xs      = list(range(len(phases)))
        color   = _svd_line_color(row['svd_mode'])

        ax.step(xs, current, where='mid', color=color, linewidth=_LW_THICK, label='Current')
        ax.step(xs, peak,    where='mid', color=color, linewidth=_LW_THIN,
                linestyle='--', alpha=0.65, label='Phase peak')
        ax.set_xticks(xs)
        ax.set_xticklabels([str(i) for i in xs], fontsize=9)
        ax.set_ylabel('Memory (MB)')
        ax.set_title(f"$k$={int(row['k'])} ({SVD_MODE_LABELS.get(row['svd_mode'], row['svd_mode'])})",
                     fontsize=12)
        ax.legend(fontsize=9)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    if phase_labels:
        legend_text = '\n'.join(f'{i}: {l}' for i, l in enumerate(phase_labels))
        fig.text(0.01, 0.01, legend_text, fontsize=7, va='bottom', family='monospace', alpha=0.7)

    fig.suptitle(_title('Memory over SVD step phases'), fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Scaling plots (time / memory vs n_params, one figure per svd_mode)
# ---------------------------------------------------------------------------

def _scaling_panel(
    ax: plt.Axes,
    svd_rows: pd.DataFrame | None,
    std_rows: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    mode: str | None,
) -> None:
    """Populate one axis of a scaling figure."""
    if svd_rows is not None and not svd_rows.empty:
        all_ks = sorted(svd_rows['k'].unique())
        n_ks   = len(all_ks)
        cmap   = plt.get_cmap(SVD_MODE_CMAPS[mode])
        for i, k in enumerate(all_ks):
            rows = svd_rows[svd_rows['k'] == k].sort_values('n_params')
            color = cmap(0.35 + 0.50 * i / max(n_ks - 1, 1))
            ax.plot(rows['n_params'].values, rows[y_col].values,
                    marker='o', color=color, label=f'$k$={k}')

    if not std_rows.empty:
        for opt in STANDARD_ORDER:
            rows = std_rows[std_rows['optimizer'] == opt].sort_values('n_params')
            if rows.empty:
                continue
            ax.plot(rows['n_params'].values, rows[y_col].values,
                    marker='s', linestyle='--', linewidth=_LW_THIN, alpha=0.85,
                    color=_std_color(opt), label=_baseline_label(opt))

    ax.set_xscale('log')
    ax.set_xlabel('Parameters')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9, ncol=2)


def plot_scaling(
    svd_by_width: pd.DataFrame,
    std_by_width: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Generate one scaling figure per svd_mode + one for standard optimizers."""
    multi_width = (
        (not svd_by_width.empty  and svd_by_width['n_params'].nunique()  > 1) or
        (not std_by_width.empty  and std_by_width['n_params'].nunique()  > 1)
    )
    if not multi_width:
        return

    # --- Standard optimizers ---
    if not std_by_width.empty and std_by_width['n_params'].nunique() > 1:
        set_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        _scaling_panel(ax1, None, std_by_width, 'time_mean_ms',
                       'Step time (ms)', 'Step time vs model size', None)
        _scaling_panel(ax2, None, std_by_width, 'mem_peak_mb',
                       'Peak GPU memory (MB)', 'Peak memory vs model size', None)
        fig.suptitle(_title('Standard optimizers — scaling with model size'), fontsize=14)
        fig.tight_layout()
        _save(fig, out_dir / 'scaling_standard')

    # --- One figure per svd_mode ---
    if not svd_by_width.empty and svd_by_width['n_params'].nunique() > 1:
        for mode in SVD_MODE_ORDER:
            svd_mode_rows = svd_by_width[svd_by_width['svd_mode'] == mode]
            if svd_mode_rows.empty or svd_mode_rows['n_params'].nunique() < 2:
                continue
            mode_label = SVD_MODE_LABELS.get(mode, mode)

            set_style()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            _scaling_panel(ax1, svd_mode_rows, std_by_width, 'time_mean_ms',
                           'Step time (ms)', 'Step time vs model size', mode)
            _scaling_panel(ax2, svd_mode_rows, std_by_width, 'mem_peak_mb',
                           'Peak GPU memory (MB)', 'Peak memory vs model size', mode)
            fig.suptitle(_title(f'Sven ({mode_label}) — scaling with model size'), fontsize=14)
            fig.tight_layout()
            _save(fig, out_dir / f'scaling_{mode}')


# ---------------------------------------------------------------------------
# Scans over batch_size / microbatch_size / param_fraction
# ---------------------------------------------------------------------------

def _is_default_mb(s: pd.Series) -> pd.Series:
    return s.isna() | (s == 1)


def _is_default_pf(s: pd.Series) -> pd.Series:
    return s.isna() | (s == 1.0)


def _aggregate_svd(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    svd = df[df['mode'] == 'svd'].copy()
    if svd.empty:
        return pd.DataFrame()
    return svd.groupby(group_cols, sort=False, dropna=False)[_METRICS].mean().reset_index()


def aggregate_baselines_by_bs(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate baseline runs by (optimizer, batch_size), averaging over lr/etc."""
    std = df[df['mode'].isin(BASELINE_MODES)].copy()
    if std.empty:
        return pd.DataFrame()
    lbfgs_mask = std['optimizer'] == 'LBFGS'
    if lbfgs_mask.any() and 'lbfgs_max_iter' in std.columns:
        mi = _most_common(std.loc[lbfgs_mask, 'lbfgs_max_iter'].dropna())
        std = std[~lbfgs_mask | (std['lbfgs_max_iter'] == mi)].copy()
    return std.groupby(['optimizer', 'batch_size'], sort=False)[_METRICS].mean().reset_index()


def _k_cmap_color(svd_mode: str, k_index: int, n_ks: int):
    cmap = plt.get_cmap(SVD_MODE_CMAPS.get(svd_mode, 'viridis'))
    return cmap(0.30 + 0.55 * k_index / max(n_ks - 1, 1))


def _annotate_model_size(ax, model_mb: float) -> None:
    ax.text(0.98, 0.98, f"Model size: {_format_model_size(model_mb)}",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#bbbbbb', alpha=0.85))


# --- Batch size scan -------------------------------------------------------

def plot_batch_size_scan(df_scans: pd.DataFrame, out_dir: Path) -> None:
    """SVD at default mb=1, pf=1.0; standard baselines per bs.

    Produces one subdirectory per svd_mode (out_dir/<svd_mode>/) containing
    time / memory / overhead_adam / overhead_sgd. Sven curves overlaid by k;
    Adam/SGD as dashed reference lines on the time/memory plots; overheads
    normalize by Adam/SGD at the matching batch size.
    """
    if 'microbatch_size' not in df_scans.columns or 'param_fraction' not in df_scans.columns:
        return
    svd_default = df_scans[
        (df_scans['mode'] == 'svd')
        & _is_default_mb(df_scans['microbatch_size'])
        & _is_default_pf(df_scans['param_fraction'])
    ].copy()
    if svd_default.empty or svd_default['batch_size'].nunique() < 2:
        return

    std_agg = aggregate_baselines_by_bs(df_scans)
    modes_present = [m for m in SVD_MODE_ORDER if m in svd_default['svd_mode'].unique()]

    for svd_mode in modes_present:
        svd_mode_rows = svd_default[svd_default['svd_mode'] == svd_mode]
        if svd_mode_rows.empty or svd_mode_rows['batch_size'].nunique() < 2:
            continue
        svd_agg = _aggregate_svd(svd_mode_rows, ['svd_mode', 'k', 'batch_size'])

        ks = sorted(svd_agg['k'].unique())
        model_mb = float(svd_mode_rows['mem_baseline_mb'].iloc[0])
        mode_label = SVD_MODE_LABELS.get(svd_mode, svd_mode)

        def _draw_sven(ax, y_col):
            for i, k in enumerate(ks):
                rows = svd_agg[svd_agg['k'] == k].sort_values('batch_size')
                color = _k_cmap_color(svd_mode, i, len(ks))
                ax.plot(rows['batch_size'], rows[y_col], marker='o',
                        color=color, label=f'$k = {int(k)}$')

        def _draw_baselines(ax, y_col, optimizers):
            for opt in optimizers:
                rows = std_agg[std_agg['optimizer'] == opt].sort_values('batch_size')
                if rows.empty:
                    continue
                ax.plot(rows['batch_size'], rows[y_col], marker='s', linestyle='--',
                        linewidth=_LW_THIN, alpha=0.85,
                        color=_std_color(opt), label=_baseline_label(opt))

        # Time and memory
        for y_col, ylabel, fstem, ttl in (
            ('time_mean_ms', 'Step time (ms)', 'time', 'time'),
            ('mem_peak_mb',  'Peak GPU memory (MB)', 'memory', 'peak memory'),
        ):
            set_style()
            fig, ax = plt.subplots()
            _draw_sven(ax, y_col)
            _draw_baselines(ax, y_col, STANDARD_ORDER)
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            if y_col == 'mem_peak_mb':
                ax.set_ylim(bottom=10)
            ax.set_xlabel('Batch size $B$')
            ax.set_ylabel(ylabel)
            ax.set_title(_title(f'Sven ({mode_label}) — {ttl} vs batch size'))
            ax.legend(loc='upper left', fontsize=8, ncol=3)
            _annotate_model_size(ax, model_mb)
            ymin, ymax = ax.get_ylim()
            log_min, log_max = np.log10(ymin), np.log10(ymax)
            ax.set_ylim(ymin, 10 ** (log_min + (log_max - log_min) * 1.25))
            fig.tight_layout()
            _save(fig, out_dir / svd_mode / fstem)

        # Overhead vs Adam and SGD
        for baseline in ('Adam', 'SGD'):
            base_rows = std_agg[std_agg['optimizer'] == baseline]
            if base_rows.empty:
                print(f"  [skip] bs_scan/{svd_mode}/overhead_{baseline.lower()}: no {baseline}")
                continue
            base_lbl = _baseline_label(baseline)
            merged = svd_agg.merge(
                base_rows[['batch_size', 'time_mean_ms', 'mem_peak_mb']].rename(
                    columns={'time_mean_ms': '_b_t', 'mem_peak_mb': '_b_m'}),
                on='batch_size', how='inner',
            )
            if merged.empty:
                continue
            merged['_t_ratio'] = merged['time_mean_ms'] / merged['_b_t']
            merged['_m_ratio'] = merged['mem_peak_mb']  / merged['_b_m']

            set_style()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            for i, k in enumerate(ks):
                rows = merged[merged['k'] == k].sort_values('batch_size')
                if rows.empty:
                    continue
                color = _k_cmap_color(svd_mode, i, len(ks))
                ax1.plot(rows['batch_size'], rows['_t_ratio'], marker='o',
                         color=color, label=f'$k = {int(k)}$')
                ax2.plot(rows['batch_size'], rows['_m_ratio'], marker='o',
                         color=color, label=f'$k = {int(k)}$')
            for ax in (ax1, ax2):
                ax.axhline(1.0, color='#777', linestyle=':', linewidth=_LW_THIN,
                           label=f'{base_lbl} (ref.)')
                ax.set_xscale('log', base=2)
                ax.set_xlabel('Batch size $B$')
                ax.set_ylim(bottom=0)
                ax.legend(fontsize=8, ncol=2)
                _annotate_model_size(ax, model_mb)
            ax1.set_ylabel(f'Step Time Ratio ({base_lbl})')
            ax1.set_title(_title(f'Sven ({mode_label}) — timing overhead vs {base_lbl}'))
            ax2.set_ylabel(f'Peak Memory Ratio ({base_lbl})')
            ax2.set_title(_title(f'Sven ({mode_label}) — memory overhead vs {base_lbl}'))
            fig.tight_layout()
            _save(fig, out_dir / svd_mode / f'overhead_{baseline.lower()}')


# --- Microbatch / param-fraction scan (with k overlay at fixed batch_size) -

def _plot_inner_scan_for_mode(
    svd_mode_rows: pd.DataFrame,
    std_agg: pd.DataFrame,
    out_dir: Path,
    svd_mode: str,
    batch_size: int,
    *,
    x_col: str,
    x_label: str,
    title_stem: str,
    log_x: bool,
) -> None:
    """One svd_mode figure-set: time / memory / overhead-Adam / overhead-SGD,
    Sven curves overlaid by k at a single batch_size."""
    if svd_mode_rows.empty or svd_mode_rows[x_col].nunique() < 2:
        return

    svd_agg = _aggregate_svd(svd_mode_rows, ['svd_mode', 'k', x_col])
    ks = sorted(svd_agg['k'].unique())
    model_mb = float(svd_mode_rows['mem_baseline_mb'].iloc[0])
    mode_label = SVD_MODE_LABELS.get(svd_mode, svd_mode)
    title_tag = f'Sven ({mode_label}, $B = {int(batch_size)}$)'
    sub_dir   = out_dir / svd_mode

    def _draw_per_k(ax, src, y_col):
        for i, k in enumerate(ks):
            rows = src[src['k'] == k].sort_values(x_col)
            color = _k_cmap_color(svd_mode, i, len(ks))
            ax.plot(rows[x_col], rows[y_col], marker='o',
                    color=color, label=f'$k = {int(k)}$')

    base_at_bs = std_agg[std_agg['batch_size'] == batch_size]

    def _draw_baseline_hlines(ax, y_col):
        for opt in STANDARD_ORDER:
            rows = base_at_bs[base_at_bs['optimizer'] == opt]
            if rows.empty:
                continue
            val = float(rows.iloc[0][y_col])
            ax.axhline(val, color=_std_color(opt), linestyle='--',
                       linewidth=_LW_THIN, alpha=0.85,
                       label=_baseline_label(opt))

    # Time and memory
    for y_col, ylabel, suffix, ttl in (
        ('time_mean_ms', 'Step time (ms)', 'time', 'time'),
        ('mem_peak_mb',  'Peak GPU memory (MB)', 'memory', 'peak memory'),
    ):
        set_style()
        fig, ax = plt.subplots()
        _draw_per_k(ax, svd_agg, y_col)
        _draw_baseline_hlines(ax, y_col)
        if log_x:
            ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        if y_col == 'mem_peak_mb':
            ax.set_ylim(bottom=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title(_title(f'{title_tag} — {ttl} vs {title_stem}'))
        ax.legend(loc='upper left', fontsize=9, ncol=3)
        _annotate_model_size(ax, model_mb)
        ymin, ymax = ax.get_ylim()
        log_min, log_max = np.log10(ymin), np.log10(ymax)
        ax.set_ylim(ymin, 10 ** (log_min + (log_max - log_min) * 1.25))
        fig.tight_layout()
        _save(fig, sub_dir / suffix)

    # Overheads — single baseline value at this batch_size
    base_at_bs = std_agg[std_agg['batch_size'] == batch_size]
    for baseline in ('Adam', 'SGD'):
        base_rows = base_at_bs[base_at_bs['optimizer'] == baseline]
        if base_rows.empty:
            print(f"  [skip] {sub_dir.name}/overhead_{baseline.lower()}: "
                  f"no {baseline} at bs={batch_size}")
            continue
        base_t   = float(base_rows.iloc[0]['time_mean_ms'])
        base_m   = float(base_rows.iloc[0]['mem_peak_mb'])
        base_lbl = _baseline_label(baseline)

        set_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for i, k in enumerate(ks):
            rows = svd_agg[svd_agg['k'] == k].sort_values(x_col)
            if rows.empty:
                continue
            color = _k_cmap_color(svd_mode, i, len(ks))
            ax1.plot(rows[x_col], rows['time_mean_ms'] / base_t, marker='o',
                     color=color, label=f'$k = {int(k)}$')
            ax2.plot(rows[x_col], rows['mem_peak_mb']  / base_m, marker='o',
                     color=color, label=f'$k = {int(k)}$')
        for ax in (ax1, ax2):
            ax.axhline(1.0, color='#777', linestyle=':', linewidth=_LW_THIN,
                       label=f'{base_lbl} (ref.)')
            if log_x:
                ax.set_xscale('log', base=2)
            ax.set_xlabel(x_label)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=9)
            _annotate_model_size(ax, model_mb)
        ax1.set_ylabel(f'Step Time Ratio ({base_lbl})')
        ax1.set_title(_title(f'{title_tag} — timing overhead vs {base_lbl}'))
        ax2.set_ylabel(f'Peak Memory Ratio ({base_lbl})')
        ax2.set_title(_title(f'{title_tag} — memory overhead vs {base_lbl}'))
        fig.tight_layout()
        _save(fig, sub_dir / f'overhead_{baseline.lower()}')


def _run_inner_scan(
    df_scans: pd.DataFrame,
    out_dir: Path,
    *,
    x_col: str,
    x_label: str,
    title_stem: str,
    fix_mb_default: bool,
    fix_pf_default: bool,
    log_x: bool,
) -> None:
    """One figure-set per svd_mode at the most-common batch_size, with k overlaid."""
    if x_col not in df_scans.columns:
        return
    svd = df_scans[df_scans['mode'] == 'svd'].copy()
    if svd.empty:
        return
    if fix_mb_default and 'microbatch_size' in svd.columns:
        svd = svd[_is_default_mb(svd['microbatch_size'])]
    if fix_pf_default and 'param_fraction' in svd.columns:
        svd = svd[_is_default_pf(svd['param_fraction'])]
    if svd.empty:
        return

    bs = int(_most_common(svd['batch_size']))
    svd = svd[svd['batch_size'] == bs]
    if svd.empty:
        return

    std_agg = aggregate_baselines_by_bs(df_scans)
    modes_present = [m for m in SVD_MODE_ORDER if m in svd['svd_mode'].unique()]

    for svd_mode in modes_present:
        rows = svd[svd['svd_mode'] == svd_mode]
        _plot_inner_scan_for_mode(
            rows, std_agg, out_dir, svd_mode, bs,
            x_col=x_col, x_label=x_label,
            title_stem=title_stem, log_x=log_x,
        )


def plot_microbatch_scan(df_scans: pd.DataFrame, out_dir: Path) -> None:
    _run_inner_scan(
        df_scans, out_dir,
        x_col='microbatch_size', x_label='Microbatch size',
        title_stem='microbatch size',
        fix_mb_default=False, fix_pf_default=True, log_x=True,
    )


def plot_param_fraction_scan(df_scans: pd.DataFrame, out_dir: Path) -> None:
    _run_inner_scan(
        df_scans, out_dir,
        x_col='param_fraction', x_label='Parameter fraction',
        title_stem='parameter fraction',
        fix_mb_default=True, fix_pf_default=False, log_x=False,
    )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _pm(mean: float, std: float) -> str:
    return f"${mean:.2f} \\pm {std:.2f}$"


def _mb(val: float) -> str:
    return f"${val:.1f}$"


def make_main_table(
    std_agg: pd.DataFrame,
    svd_agg: pd.DataFrame,
    batch_size: int,
    n_params: int,
) -> str:
    dataset_prefix = f"Dataset: {_DATASET_NAME}. " if _DATASET_NAME else ""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (r"\caption{" + dataset_prefix +
         r"Per-step wall-clock time and GPU memory for Sven and standard "
         r"optimizers. "
         f"Batch size $B = {batch_size}$, {n_params:,} parameters. "
         r"Time is mean $\pm$ std over measured steps; "
         r"peak memory is mean max-allocated during a step.}"),
        r"\label{tab:optimizer_profile_main}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Optimizer & Step time (ms) & Resident (MB) & Peak (MB) & Transient (MB) \\",
        r"\midrule",
    ]

    if not svd_agg.empty:
        lines.append(r"\multicolumn{5}{l}{\emph{Sven (SVD optimizer)}} \\[2pt]")
        for _, row in svd_agg.iterrows():
            label = (r"\quad Sven $k$=" + str(int(row['k'])) +
                     f" ({SVD_MODE_LABELS.get(row['svd_mode'], row['svd_mode'])})")
            lines.append(
                f"{label} & {_pm(row['time_mean_ms'], row['time_std_ms'])} & "
                f"{_mb(row['mem_resident_mb'])} & {_mb(row['mem_peak_mb'])} & "
                f"{_mb(row['mem_transient_mb'])} \\\\"
            )

    if not svd_agg.empty and not std_agg.empty:
        lines.append(r"\midrule")

    if not std_agg.empty:
        lines.append(r"\multicolumn{5}{l}{\emph{Standard optimizers}} \\[2pt]")
        for _, row in std_agg.iterrows():
            lines.append(
                f"\\quad {_baseline_label_tex(row['optimizer'])} & "
                f"{_pm(row['time_mean_ms'], row['time_std_ms'])} & "
                f"{_mb(row['mem_resident_mb'])} & {_mb(row['mem_peak_mb'])} & "
                f"{_mb(row['mem_transient_mb'])} \\\\"
            )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_k_scaling_table(
    svd_agg: pd.DataFrame,
    std_agg: pd.DataFrame,
    batch_size: int,
) -> str:
    adam     = std_agg[std_agg['optimizer'] == 'Adam'] if not std_agg.empty else pd.DataFrame()
    has_adam = not adam.empty
    adam_time = float(adam.iloc[0]['time_mean_ms']) if has_adam else None
    adam_mem  = float(adam.iloc[0]['mem_peak_mb'])  if has_adam else None

    ncols    = 6 if has_adam else 4
    col_spec = 'l' + 'r' * (ncols - 1)
    header   = ['$k$', 'SVD mode', 'Step time (ms)', 'Peak mem (MB)']
    if has_adam:
        header += ['Time / Adam', 'Mem / Adam']

    ratio_note = r" Ratios are relative to Adam." if has_adam else ""
    dataset_prefix = f"Dataset: {_DATASET_NAME}. " if _DATASET_NAME else ""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (r"\caption{" + dataset_prefix +
         r"Sven $k$-scaling: per-step time and peak memory "
         f"(batch size $B = {batch_size}$).{ratio_note}}}"),
        r"\label{tab:sven_k_scaling}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]

    prev_mode = None
    for _, row in svd_agg.iterrows():
        mode = row['svd_mode']
        if mode != prev_mode:
            if prev_mode is not None:
                lines.append(r"\midrule")
            lines.append(
                f"\\multicolumn{{{ncols}}}{{l}}"
                f"{{\\emph{{svd\\_mode = {mode}}}}} \\\\"
            )
            prev_mode = mode

        cells = [
            f"${int(row['k'])}$",
            mode,
            _pm(row['time_mean_ms'], row['time_std_ms']),
            _mb(row['mem_peak_mb']),
        ]
        if has_adam:
            cells.append(f"${row['time_mean_ms'] / adam_time:.2f}\\times$")
            cells.append(f"${row['mem_peak_mb']  / adam_mem:.2f}\\times$")
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def save_table(table: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sep = '─' * 60
    print(f"\n{sep}\nTable → {out_path.name}\n{sep}")
    print(table)
    out_path.write_text(table + "\n")
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_single_config(
    df: pd.DataFrame,
    bs: int,
    mw: int | None,
    out_dir: Path,
) -> None:
    """All single-config plots and tables for one (batch_size, mlp_width).

    "Default" means microbatch_size=1 and param_fraction=1.0 (filtering done
    inside the aggregators).
    """
    df_at_bs = df[df['batch_size'] == bs].copy()
    if mw is not None and 'mlp_width' in df_at_bs.columns:
        df_single = df_at_bs[df_at_bs['mlp_width'] == mw].copy()
    else:
        df_single = df_at_bs
    if df_single.empty:
        print(f"  [skip] no runs at bs={bs}, mlp_width={mw}")
        return

    std_agg = order_standard(aggregate_standard(df_single))
    svd_agg = order_svd(aggregate_svd(df_single, bs))
    both = not std_agg.empty and not svd_agg.empty

    if both:
        plot_time_bars(std_agg, svd_agg,     out_dir / 'time_bar', batch_size=bs)
        plot_memory_bars(std_agg, svd_agg,   out_dir / 'memory_bar', batch_size=bs)

    if not svd_agg.empty:
        np_series = df_single['n_params'].dropna()
        n_params  = int(np_series.iloc[0]) if not np_series.empty else None
        mb_series = df_single['mem_baseline_mb'].dropna() if 'mem_baseline_mb' in df_single.columns else pd.Series(dtype=float)
        model_mb  = float(mb_series.iloc[0]) if not mb_series.empty else None
        plot_time_vs_k(svd_agg, std_agg,   out_dir / 'time_vs_k', batch_size=bs,
                       n_params=n_params, model_mb=model_mb)
        plot_memory_vs_k(svd_agg, std_agg, out_dir / 'memory_vs_k', batch_size=bs,
                         n_params=n_params, model_mb=model_mb)
        if both:
            plot_time_overhead_vs_k(svd_agg, std_agg, out_dir / 'time_overhead_vs_k',
                                    batch_size=bs, n_params=n_params, model_mb=model_mb)
            plot_memory_overhead_vs_k(svd_agg, std_agg, out_dir / 'memory_overhead_vs_k',
                                      batch_size=bs, n_params=n_params, model_mb=model_mb)
        plot_memory_breakdown(svd_agg,     out_dir / 'memory_breakdown')

    # Scaling plots (only generated when multiple mlp_widths are present at this bs)
    svd_by_w = aggregate_svd_by_width(df_at_bs)
    std_by_w = aggregate_standard_by_width(df_at_bs)
    plot_scaling(svd_by_w, std_by_w, out_dir)

    if not std_agg.empty or not svd_agg.empty:
        n_params = int(df_single['n_params'].dropna().iloc[0])
        t1 = make_main_table(std_agg, svd_agg, bs, n_params)
        save_table(t1, out_dir / 'table_main.tex')
    if not svd_agg.empty:
        t2 = make_k_scaling_table(svd_agg, std_agg, bs)
        save_table(t2, out_dir / 'table_k_scaling.tex')


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile analysis: Sven vs baselines")
    ap.add_argument('results_dir', type=Path,
                    help="Path to profile_results/{scan_name}/ directory")
    ap.add_argument('--output-dir', type=Path, default=None,
                    help="Output directory (default: analysis/plots/profile/<scan_name>/)")
    ap.add_argument('--mlp-width', type=int, default=None,
                    help="mlp_width to use for single-width plots and scans "
                         "(default: most common per batch size)")
    ap.add_argument('--dataset', type=str, default=None,
                    help="Dataset name to display in plot titles and table captions")
    args = ap.parse_args()

    global _DATASET_NAME
    _DATASET_NAME = args.dataset

    scan_name  = args.results_dir.name
    output_dir = args.output_dir or (
        Path(__file__).parent / 'plots' / 'profile' / scan_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    df = load_profile_results(args.results_dir)

    # ----- Single-config plots & tables: one subdir per batch size -----
    unique_bss = sorted(int(b) for b in df['batch_size'].dropna().unique())
    print(f"\n=== Single-config plots & tables — {len(unique_bss)} batch size(s) ===")
    for bs in unique_bss:
        df_at_bs = df[df['batch_size'] == bs]
        mw = _resolve_mlp_width(df_at_bs, args.mlp_width)
        sub_dir = output_dir / 'single_config' / f'bs{bs}'
        print(f"\n--- bs={bs}"
              + (f", mlp_width={mw}" if mw is not None else "")
              + f" → {sub_dir.relative_to(output_dir)} ---")
        _run_single_config(df, bs, mw, sub_dir)

    # ----- Scan plots: one mlp_width across all batch sizes -----
    svd_all = df[df['mode'] == 'svd']
    if not svd_all.empty and 'svd_mode' in svd_all.columns:
        scan_mw = _resolve_mlp_width(svd_all, args.mlp_width)
        df_scans = df.copy()
        if scan_mw is not None and 'mlp_width' in df_scans.columns:
            df_scans = df_scans[df_scans['mlp_width'] == scan_mw]
        print(f"\n=== Scan plots"
              + (f" (mlp_width={scan_mw})" if scan_mw is not None else "")
              + " ===")
        plot_batch_size_scan(df_scans,       output_dir / 'bs_scan')
        plot_microbatch_scan(df_scans,       output_dir / 'microbatch_scan')
        plot_param_fraction_scan(df_scans,   output_dir / 'param_fraction_scan')

    print(f"\nDone. All outputs in {output_dir}/")


if __name__ == '__main__':
    main()
