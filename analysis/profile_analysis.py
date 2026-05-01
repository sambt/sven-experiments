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

sys.path.insert(0, str(Path(__file__).parent))
from style import set_style

# ---------------------------------------------------------------------------
# Fixed orderings and colour scheme
# ---------------------------------------------------------------------------

STANDARD_ORDER = [
    'Adam', 'AdamW', 'SGD', 'RMSprop',
    'LBFGS', 'PolyakSGD',
]

SVD_MODE_ORDER  = ['torch', 'randomized', 'randomized_v2']
SVD_MODE_LABELS = {'torch': 'full SVD', 'randomized': 'rand.', 'randomized_v2': 'rand. v2'}
SVD_MODE_CMAPS  = {'torch': 'Greens', 'randomized': 'Oranges', 'randomized_v2': 'Purples'}
SVD_MODE_BASE   = {'torch': '#1a7c3a', 'randomized': '#d95f02', 'randomized_v2': '#7570b3'}

# One colour per position in STANDARD_ORDER (tab10 palette)
_TAB10 = plt.cm.tab10(np.linspace(0, 1, 10))
STANDARD_COLORS = {opt: _TAB10[i % 10] for i, opt in enumerate(STANDARD_ORDER)}

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


def select_defaults(
    df: pd.DataFrame,
    batch_size: int | None,
    model_seed: int | None,
    mlp_width: int | None,
) -> tuple[int, int, int | None, pd.DataFrame, pd.DataFrame]:
    """Return (bs, seed, mw, df_single, df_all_widths).

    df_single  — filtered to one (bs, seed, mlp_width), used for per-config plots.
    df_all_widths — filtered to (bs, seed) only, used for scaling plots.
    mw is None when no mlp_width column is present.
    """
    bs   = batch_size if batch_size is not None else int(_most_common(df['batch_size']))
    seed = model_seed if model_seed is not None else int(_most_common(df['model_seed']))

    df_all = df[(df['batch_size'] == bs) & (df['model_seed'] == seed)].copy()
    if df_all.empty:
        raise ValueError(f"No runs for batch_size={bs}, model_seed={seed}. "
                         f"Available batch sizes: {sorted(df['batch_size'].unique())}")

    has_width = 'mlp_width' in df_all.columns and df_all['mlp_width'].notna().any()
    mw: int | None = None
    if has_width:
        mw = mlp_width if mlp_width is not None else int(_most_common(df_all['mlp_width']))
        df_single = df_all[df_all['mlp_width'] == mw].copy()
    else:
        df_single = df_all

    n_widths = df_all['mlp_width'].nunique() if has_width else 1
    print(f"Using batch_size={bs}, model_seed={seed}"
          + (f", mlp_width={mw}" if mw is not None else "")
          + f"  ({len(df_single)} single-width runs"
          + (f", {n_widths} widths available" if n_widths > 1 else "") + ")")
    return bs, seed, mw, df_single, df_all


# ---------------------------------------------------------------------------
# Aggregation — average over lr/rtol/kappa (don't affect timing/memory)
# ---------------------------------------------------------------------------

def aggregate_standard(df: pd.DataFrame) -> pd.DataFrame:
    std = df[df['mode'] == 'standard'].copy()
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
    """Aggregate SVD runs by (svd_mode, k, n_params), averaging over lr/rtol/kappa."""
    svd = df[df['mode'] == 'svd'].copy()
    if svd.empty:
        return pd.DataFrame()
    agg = svd.groupby(['svd_mode', 'k', 'n_params'], sort=False)[_METRICS].mean().reset_index()
    bs  = int(_most_common(svd['batch_size']))
    agg['k_fraction'] = agg['k'] / bs
    return agg


def aggregate_standard_by_width(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate standard optimizer runs by (optimizer, n_params), averaging over lr/etc."""
    std = df[df['mode'] == 'standard'].copy()
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
    fig.savefig(path.with_suffix('.png'))
    plt.close(fig)
    print(f"  Saved {path.stem}.pdf + .png")


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
    all_labels = [_svd_label(r) for _, r in svd_agg.iterrows()] + list(std_agg['optimizer'])

    ax.set_yticks(all_y)
    ax.set_yticklabels(all_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=10)

    # Separator between SVD and standard groups
    if n_svd and n_std:
        sep = (y_svd[-1] + y_std[0]) / 2.0
        ax.axhline(sep, color='#999999', linewidth=0.9, linestyle='--')

    fig.tight_layout()
    _save(fig, out_path)


def plot_time_bars(std_agg: pd.DataFrame, svd_agg: pd.DataFrame, out_path: Path) -> None:
    _bar_chart(
        std_agg, svd_agg,
        std_vals=std_agg['time_mean_ms'].values,
        std_errs=std_agg['time_std_ms'].values,
        svd_vals=svd_agg['time_mean_ms'].values,
        svd_errs=svd_agg['time_std_ms'].values,
        xlabel='Step time (ms)',
        title='Per-step wall-clock time',
        out_path=out_path,
    )


def plot_memory_bars(std_agg: pd.DataFrame, svd_agg: pd.DataFrame, out_path: Path) -> None:
    _bar_chart(
        std_agg, svd_agg,
        std_vals=std_agg['mem_peak_mb'].values,
        std_errs=np.zeros(len(std_agg)),
        svd_vals=svd_agg['mem_peak_mb'].values,
        svd_errs=np.zeros(len(svd_agg)),
        xlabel='Peak GPU memory (MB)',
        title='Peak GPU memory per step',
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
) -> None:
    set_style()
    fig, ax = plt.subplots()

    for mode in SVD_MODE_ORDER:
        rows = svd_agg[svd_agg['svd_mode'] == mode].sort_values('k_fraction')
        if rows.empty:
            continue
        color = _svd_line_color(mode)
        label = f"Sven ({SVD_MODE_LABELS.get(mode, mode)})"
        xs    = rows['k_fraction'].values
        ys    = rows[svd_col].values
        ax.plot(xs, ys, marker='o', color=color, label=label)
        if svd_err_col and svd_err_col in rows.columns:
            errs = rows[svd_err_col].values
            ax.fill_between(xs, ys - errs, ys + errs, alpha=0.18, color=color)

    if not std_agg.empty:
        for _, row in std_agg.iterrows():
            ax.axhline(row[std_col], color=_std_color(row['optimizer']),
                       linestyle='--', linewidth=1.5, alpha=0.85, label=row['optimizer'])

    ax.set_xlabel('$k / B$')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', fontsize=10)

    fig.tight_layout()
    _save(fig, out_path)


def plot_time_vs_k(svd_agg: pd.DataFrame, std_agg: pd.DataFrame, out_path: Path) -> None:
    _k_scaling_plot(
        svd_agg, std_agg,
        svd_col='time_mean_ms', svd_err_col='time_std_ms', std_col='time_mean_ms',
        ylabel='Step time (ms)',
        title='Per-step time vs $k$',
        out_path=out_path,
    )


def plot_memory_vs_k(svd_agg: pd.DataFrame, std_agg: pd.DataFrame, out_path: Path) -> None:
    _k_scaling_plot(
        svd_agg, std_agg,
        svd_col='mem_peak_mb', svd_err_col=None, std_col='mem_peak_mb',
        ylabel='Peak GPU memory (MB)',
        title='Peak GPU memory vs $k$',
        out_path=out_path,
    )


# ---------------------------------------------------------------------------
# Plot 5: Overhead ratios vs k (relative to Adam)
# ---------------------------------------------------------------------------

def plot_overhead_ratios(
    svd_agg: pd.DataFrame,
    std_agg: pd.DataFrame,
    out_path: Path,
) -> None:
    adam = std_agg[std_agg['optimizer'] == 'Adam']
    if adam.empty:
        print("  [skip] overhead_ratios: no Adam baseline found")
        return

    adam_time = float(adam.iloc[0]['time_mean_ms'])
    adam_mem  = float(adam.iloc[0]['mem_peak_mb'])

    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for mode in SVD_MODE_ORDER:
        rows = svd_agg[svd_agg['svd_mode'] == mode].sort_values('k_fraction')
        if rows.empty:
            continue
        color = _svd_line_color(mode)
        label = f"Sven ({SVD_MODE_LABELS.get(mode, mode)})"
        xs    = rows['k_fraction'].values
        ax1.plot(xs, rows['time_mean_ms'].values / adam_time, marker='o', color=color, label=label)
        ax2.plot(xs, rows['mem_peak_mb'].values  / adam_mem,  marker='o', color=color, label=label)

    for ax in (ax1, ax2):
        ax.axhline(1.0, color='#777777', linestyle=':', linewidth=1.5, label='Adam (ref.)')
        ax.set_xlabel('$k / B$')
        ax.legend(fontsize=10)

    ax1.set_ylabel('Step time / Adam step time')
    ax1.set_title('Timing overhead vs Adam')
    ax2.set_ylabel('Peak memory / Adam peak memory')
    ax2.set_title('Memory overhead vs Adam')

    fig.tight_layout()
    _save(fig, out_path)


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

        ax.step(xs, current, where='mid', color=color, linewidth=2.0, label='Current')
        ax.step(xs, peak,    where='mid', color=color, linewidth=1.5,
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

    fig.suptitle('Memory over SVD step phases', fontsize=14, y=1.01)
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
                    marker='s', linestyle='--', linewidth=1.5, alpha=0.85,
                    color=_std_color(opt), label=opt)

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
        fig.suptitle('Standard optimizers — scaling with model size', fontsize=14)
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
            fig.suptitle(f'Sven ({mode_label}) — scaling with model size', fontsize=14)
            fig.tight_layout()
            _save(fig, out_dir / f'scaling_{mode}')


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
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (r"\caption{Per-step wall-clock time and GPU memory for Sven and standard "
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
                f"\\quad {row['optimizer']} & "
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

    ncols    = 7 if has_adam else 5
    col_spec = 'l' + 'r' * (ncols - 1)
    header   = ['$k$', 'SVD mode', '$k/B$', 'Step time (ms)', 'Peak mem (MB)']
    if has_adam:
        header += ['Time / Adam', 'Mem / Adam']

    ratio_note = r" Ratios are relative to Adam." if has_adam else ""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (r"\caption{Sven $k$-scaling: per-step time and peak memory "
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
            f"${row['k_fraction']:.2f}$",
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

def main() -> None:
    ap = argparse.ArgumentParser(description="Profile analysis: Sven vs baselines")
    ap.add_argument('results_dir', type=Path,
                    help="Path to profile_results/{scan_name}/ directory")
    ap.add_argument('--output-dir', type=Path, default=None,
                    help="Output directory (default: analysis/plots/profile/<scan_name>/)")
    ap.add_argument('--batch-size', type=int, default=None,
                    help="Batch size to focus on (default: most common in results)")
    ap.add_argument('--model-seed', type=int, default=None,
                    help="Model seed to focus on (default: most common in results)")
    ap.add_argument('--mlp-width', type=int, default=None,
                    help="mlp_width to use for single-width plots (default: most common)")
    args = ap.parse_args()

    scan_name  = args.results_dir.name
    output_dir = args.output_dir or (
        Path(__file__).parent / 'plots' / 'profile' / scan_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    df = load_profile_results(args.results_dir)
    bs, seed, mw, df_single, df_all = select_defaults(
        df, args.batch_size, args.model_seed, args.mlp_width
    )
    n_params = int(df_single['n_params'].dropna().iloc[0])

    std_agg = order_standard(aggregate_standard(df_single))
    svd_agg = order_svd(aggregate_svd(df_single, bs))

    if std_agg.empty:
        print("No standard optimizer runs found.")
    if svd_agg.empty:
        print("No SVD runs found.")

    both = not std_agg.empty and not svd_agg.empty

    print("\n--- Plots ---")
    if both:
        plot_time_bars(std_agg, svd_agg,     output_dir / 'time_bar')
        plot_memory_bars(std_agg, svd_agg,   output_dir / 'memory_bar')
        plot_overhead_ratios(svd_agg, std_agg, output_dir / 'overhead_ratios')

    if not svd_agg.empty:
        plot_time_vs_k(svd_agg, std_agg,   output_dir / 'time_vs_k')
        plot_memory_vs_k(svd_agg, std_agg, output_dir / 'memory_vs_k')
        plot_memory_breakdown(svd_agg,     output_dir / 'memory_breakdown')

    # Scaling plots (only generated when multiple mlp_widths are present)
    svd_by_w = aggregate_svd_by_width(df_all)
    std_by_w = aggregate_standard_by_width(df_all)
    plot_scaling(svd_by_w, std_by_w, output_dir)

    print("\n--- Tables ---")
    t1 = make_main_table(std_agg, svd_agg, bs, n_params)
    save_table(t1, output_dir / 'table_main.tex')

    if not svd_agg.empty:
        t2 = make_k_scaling_table(svd_agg, std_agg, bs)
        save_table(t2, output_dir / 'table_k_scaling.tex')

    print(f"\nDone. All outputs in {output_dir}/")


if __name__ == '__main__':
    main()
