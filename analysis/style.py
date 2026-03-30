import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(name, results_root='../experiment_results', selection_fn=None):
    """Load experiment results, supporting both old and new storage formats.

    Old format: a single ``{name}.jsonl`` file with one JSON object per line.
    New format: a directory ``{name}/`` containing one ``.jsonl`` file per run.

    The function tries the directory format first, then falls back to the
    single-file format.  Returns a :class:`pd.DataFrame`.
    """
    root = Path(results_root)
    scan_dir = root / name

    records: list[dict] = []

    # New format: directory of per-run JSONL files
    if scan_dir.is_dir():
        for f in sorted(scan_dir.glob('*.jsonl')):
            if selection_fn is not None and not selection_fn(str(f).split("/")[-1]):
                continue
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        if records:
            print(f"Loaded {len(records)} runs from {scan_dir}/ (directory format)")
            return pd.DataFrame(records)

    # Old format: single JSONL file
    jsonl_path = root / f'{name}.jsonl'
    if jsonl_path.is_file():
        with open(jsonl_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"Loaded {len(records)} runs from {jsonl_path} (single-file format)")
        return pd.DataFrame(records)

    raise FileNotFoundError(
        f"No results found for '{name}': tried {scan_dir}/ and {jsonl_path}"
    )

def load_results_jsonl(name, results_root='../experiment_results'):
    """Load experiment results from a single JSONL file.

    This is the old storage format, where all runs are stored in a single
    ``{name}.jsonl`` file.  Returns a :class:`pd.DataFrame`.
    """
    jsonl_path = Path(results_root) / f'{name}.jsonl'
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"No results found for '{name}': {jsonl_path} does not exist")

    records: list[dict] = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} runs from {jsonl_path}")
    return pd.DataFrame(records)

# Scalar quantity columns added by add_derived_columns — excluded from auto-detected config cols.
_DERIVED_QUANTITY_COLS = {
    'final_val_loss', 'final_train_loss',
    'final_val_acc', 'final_train_acc',
    'total_time', 'avg_batch_time_train',
    'effective_bs',
}


def _stack_arrays(arrays):
    """Stack a list of arrays after truncating to the minimum length. Returns 2-D ndarray."""
    arrs = [np.asarray(a) for a in arrays if a is not None]
    if not arrs:
        return None
    min_len = min(len(a) for a in arrs)
    return np.stack([a[:min_len] for a in arrs], axis=0)  # shape (n_seeds, T)


def _avg_arrays(arrays):
    """Element-wise mean of a list of arrays, truncated to the minimum length."""
    stacked = _stack_arrays(arrays)
    return None if stacked is None else np.mean(stacked, axis=0).tolist()


def _std_arrays(arrays):
    """Element-wise std (ddof=1) of a list of arrays, truncated to the minimum length."""
    stacked = _stack_arrays(arrays)
    if stacked is None:
        return None
    ddof = 1 if stacked.shape[0] > 1 else 0
    return np.std(stacked, axis=0, ddof=ddof).tolist()


def _avg_dicts(dicts):
    """Mean of a list of dicts whose values are arrays or scalars."""
    dicts = [d for d in dicts if isinstance(d, dict)]
    if not dicts:
        return None
    result = {}
    for k in dicts[0]:
        vals = [d[k] for d in dicts if k in d and d[k] is not None]
        if not vals:
            result[k] = None
            continue
        if isinstance(vals[0], (list, np.ndarray)):
            result[k] = _avg_arrays(vals)
        elif isinstance(vals[0], (int, float, np.integer, np.floating)):
            result[k] = float(np.mean(vals))
        else:
            result[k] = vals[0]
    return result


def _std_dicts(dicts):
    """Std (ddof=1) of a list of dicts whose values are arrays or scalars."""
    dicts = [d for d in dicts if isinstance(d, dict)]
    if not dicts:
        return None
    n = len(dicts)
    result = {}
    for k in dicts[0]:
        vals = [d[k] for d in dicts if k in d and d[k] is not None]
        if not vals:
            result[k] = None
            continue
        if isinstance(vals[0], (list, np.ndarray)):
            result[k] = _std_arrays(vals)
        elif isinstance(vals[0], (int, float, np.integer, np.floating)):
            ddof = 1 if n > 1 else 0
            result[k] = float(np.std(vals, ddof=ddof))
        else:
            result[k] = None
    return result


def average_over_seeds(df, seed_col='model_seed', config_cols=None):
    """Average all quantities over model seeds for each distinct configuration.

    Groups the DataFrame by *config_cols* (the hyperparameter axes) and computes
    mean and standard deviation over the different values of *seed_col*.  Works on
    both raw DataFrames and ones processed by ``add_derived_columns``.

    For every averaged quantity column a parallel ``{col}_std`` column is added
    with the same structure (scalar → scalar std, list → list of per-step stds,
    dict → dict of stds).  Use these for ``ax.fill_between`` error bands.

    Supported column value types:

    * **dict** (e.g. ``losses``, ``svd_info``) — each key is averaged
      independently; array-valued keys are averaged element-wise (truncated to
      the shortest seed's length).  A ``{col}_std`` dict mirrors the structure.
    * **list / ndarray** — element-wise mean/std, truncated to shortest length.
    * **scalar numeric** — arithmetic mean / std (ddof=1).
    * **other** (str, None, …) — first non-null value kept as-is; std is None.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame, one row per run.
    seed_col : str
        Column that identifies the random seed to average over.
    config_cols : list[str] or None
        Columns that jointly identify a unique configuration (groupby keys).
        If *None*, auto-detected as all scalar columns except *seed_col* and
        known derived quantity columns.

    Returns
    -------
    pd.DataFrame
        One row per distinct configuration.  *seed_col* is replaced by
        ``n_seeds``.  For each quantity column ``col``, a ``col_std`` column
        is inserted immediately after it.
    """
    df = df.copy()

    if config_cols is None:
        config_cols = []
        for col in df.columns:
            if col == seed_col or col in _DERIVED_QUANTITY_COLS:
                continue
            first = df[col].dropna()
            if first.empty:
                continue
            first = first.iloc[0]
            if isinstance(first, (str, bool, int, float, np.integer, np.floating)):
                config_cols.append(col)

    quantity_cols = [c for c in df.columns if c != seed_col and c not in config_cols]

    rows = []
    for config_vals, group in df.groupby(config_cols, dropna=False, sort=True):
        if not isinstance(config_vals, tuple):
            config_vals = (config_vals,)
        row = dict(zip(config_cols, config_vals))
        row['n_seeds'] = len(group)

        for col in quantity_cols:
            vals = group[col].tolist()
            first_valid = next((v for v in vals if v is not None), None)
            if first_valid is None:
                row[col] = None
                row[f'{col}_std'] = None
            elif isinstance(first_valid, dict):
                row[col] = _avg_dicts(vals)
                row[f'{col}_std'] = _std_dicts(vals)
            elif isinstance(first_valid, (list, np.ndarray)):
                row[col] = _avg_arrays(vals)
                row[f'{col}_std'] = _std_arrays(vals)
            elif isinstance(first_valid, (int, float, np.integer, np.floating)):
                numeric = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
                row[col] = float(np.mean(numeric)) if numeric else float('nan')
                ddof = 1 if len(numeric) > 1 else 0
                row[f'{col}_std'] = float(np.std(numeric, ddof=ddof)) if numeric else float('nan')
            else:
                row[col] = first_valid
                row[f'{col}_std'] = None

        rows.append(row)

    # Column order: config cols, n_seeds, then quantity / quantity_std pairs
    result = pd.DataFrame(rows)
    ordered = list(config_cols) + ['n_seeds']
    for col in quantity_cols:
        ordered.append(col)
        std_col = f'{col}_std'
        if std_col in result.columns:
            ordered.append(std_col)
    return result[[c for c in ordered if c in result.columns]]


def set_style():
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'legend.fontsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2,
        'axes.grid': False,
        'font.family': 'arial',
        'legend.frameon': False,
        'mathtext.fontset': 'cm',
    })
    
lr_labels = {10**k: f"10^{{{k}}}" for k in range(-10,10)}
lr_labels[0.5] = "0.5"
lr_labels[0.0003] = "3 \\times 10^{-4}"
lr_labels[0.003] = "3 \\times 10^{-3}"
lr_labels[0.03] = "3 \\times 10^{-2}"
lr_labels[0.05] = "0.05"
lr_labels[50.0] = "50"