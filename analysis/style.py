import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The bulk of every SVD result file is diagnostics the analysis rarely needs:
# svd_info.svs (~2.4 MB/file, the per-step singular values) plus the per-BATCH
# arrays in losses (another ~0.5 MB). Together ~98% of the bytes. `slim=True`
# drops them so a scan loads (and, cached, reloads) in a fraction of the time;
# only the sv-spectra / batch-wise notebooks need them (pass slim=False there).
_DROP_TOP = ('svd_info',)
_DROP_LOSSES = ('batch_times_train', 'batch_times_val', 'train_batch', 'val_batch')


def _slim_record(r):
    for k in _DROP_TOP:
        r.pop(k, None)
    L = r.get('losses')
    if isinstance(L, dict):
        for k in _DROP_LOSSES:
            L.pop(k, None)
    return r


_CACHE_VERSION = 3  # bump when the slim schema / signature scheme changes


# ---------------------------------------------------------------------------
# Light / heavy result files
# ---------------------------------------------------------------------------
# New-format runs are split by generic_scan._write_run into a light
# {run_id}.jsonl (epoch curves + a per-epoch `svd_summary`) and a heavy
# diag/{run_id}.npz (per-batch arrays, Sven spectra). Old-format files carry
# everything inline. `load_results(slim=False)` re-attaches the heavy arrays to
# each record in the old inline layout, so notebook code that reads
# row['svd_info']['svs'] / ['num_nonzero_svs'] / row['losses']['train_batch']
# works unchanged on both layouts. NOTE: for new-format runs `svs` holds only
# every `svd_summary['spectra_every']`-th step (default 20); `svs_step` gives
# the step index of each saved spectrum.
_DIAG_LOSS_KEYS = ('train_batch', 'val_batch', 'batch_times_train',
                   'batch_times_val', 'train_batch_per_model')


def load_diagnostics(row, name=None, results_root='../experiment_results'):
    """Heavy per-batch diagnostics for one run, as a dict of numpy arrays.

    ``row`` is a record/Series from :func:`load_results`. New-format runs are
    read from ``{results_root}/{name}/diag/{run_id}.npz`` (``name`` defaults to
    ``row['_scan']``, set by load_results); legacy inline runs are re-read from
    their JSONL. Keys (when present): train_batch, val_batch, batch_times_train,
    batch_times_val, num_nonzero_svs, sv_max, sv_min, svs (2-D, NaN-padded),
    svs_step, k_used, variable_k_substep_losses.
    """
    name = name or row.get('_scan')
    if name is None:
        raise ValueError("pass name= (scan directory name) or use a row from load_results")
    scan_dir = Path(results_root) / name
    diag_file = row.get('diag_file')
    if diag_file:
        with np.load(scan_dir / diag_file) as z:
            return {k: z[k] for k in z.files}
    # Legacy inline layout: re-read the full JSONL for this run.
    f = scan_dir / f"{row['run_id']}.jsonl"
    if not f.is_file():
        return {}
    with open(f) as fh:
        rec = json.loads(fh.readline())
    out = {}
    L = rec.get('losses') or {}
    for k in _DIAG_LOSS_KEYS:
        if k in L:
            out[k] = np.asarray(L[k], dtype=np.float32)
    si = rec.get('svd_info') or {}
    if si.get('num_nonzero_svs'):
        out['num_nonzero_svs'] = np.asarray(si['num_nonzero_svs'], dtype=np.int32)
    if si.get('svs'):
        rows = [np.asarray(r, dtype=np.float32) for r in si['svs']]
        w = max(len(r) for r in rows)
        svs = np.full((len(rows), w), np.nan, dtype=np.float32)
        for i, r in enumerate(rows):
            svs[i, :len(r)] = r
        out['svs'], out['svs_step'] = svs, np.arange(len(rows), dtype=np.int32)
    if si.get('k_used'):
        out['k_used'] = np.asarray(si['k_used'], dtype=np.int32)
    return out


def spectra_list(diag):
    """``diag['svs']`` as a list of 1-D arrays with the NaN padding stripped
    (the shape notebook code expects from the old inline ``svd_info['svs']``)."""
    svs = diag.get('svs')
    if svs is None:
        return []
    return [r[~np.isnan(r)] for r in svs]


def _attach_diagnostics(rec, scan_dir):
    """Rebuild the legacy inline layout on a new-format light record."""
    if not rec.get('diag_file'):
        return rec  # legacy record (inline) or no diagnostics written
    with np.load(scan_dir / rec['diag_file']) as z:
        d = {k: z[k] for k in z.files}
    L = rec.setdefault('losses', {})
    for k in _DIAG_LOSS_KEYS:
        if k in d:
            L[k] = d[k].tolist()
    if rec.get('svd_summary') is not None:
        si = {'num_nonzero_svs': d['num_nonzero_svs'].tolist() if 'num_nonzero_svs' in d else [],
              'svs': spectra_list(d),
              'svs_step': d['svs_step'].tolist() if 'svs_step' in d else [],
              'k_used': d['k_used'].tolist() if 'k_used' in d else [],
              'variable_k_substep_losses': (d['variable_k_substep_losses'].tolist()
                                            if 'variable_k_substep_losses' in d else [])}
        for k in ('sv_max', 'sv_min'):
            if k in d:
                si[k] = d[k].tolist()
        rec['svd_info'] = si
    else:
        rec['svd_info'] = None
    return rec


def _dir_signature(files):
    """Content-sensitive cache key over ALL files: a hash of each file's
    (name, size, mtime_ns). Unlike a (count, max-mtime) key, this also detects
    an in-place overwrite of an existing run_id (re-running a config that
    regenerates the same filename) — count and max-mtime can both be unchanged
    there, but that file's size/mtime moves, so the hash changes.
    """
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    h.update(str(_CACHE_VERSION).encode())
    for f in files:  # files must be pre-sorted for a stable hash
        st = f.stat()
        h.update(f'{f.name}\0{st.st_size}\0{st.st_mtime_ns}\0'.encode())
    return (len(files), h.hexdigest())


def load_results(name, results_root='../experiment_results', selection_fn=None,
                 slim=True, use_cache=True):
    """Load experiment results into a DataFrame (new per-run directory format,
    with a fallback to the legacy single ``{name}.jsonl`` file).

    slim (default True): drop the heavy diagnostics (``svd_info`` and the per-batch
        arrays) — ~98% of the bytes, unused by most notebooks. Pass ``slim=False``
        for the sv-spectra / batch-wise analyses that need them: for new-format
        (light + diag/*.npz) runs they are then read from the npz and attached in
        the old inline layout (see :func:`load_diagnostics`). Every record gets a
        ``_scan`` column (the scan name) for :func:`load_diagnostics`.
    use_cache (default True): for a slim, unfiltered load, cache the result to
        ``{results_root}/_cache/{name}.slim.pkl`` and reuse it while the scan dir
        is unchanged. The cache key is content-sensitive (see :func:`_dir_signature`),
        so it invalidates on new files, removed files, AND in-place re-runs of an
        existing run_id. First build reads every file once; reloads are near-instant.
    """
    root = Path(results_root)
    scan_dir = root / name
    cacheable = slim and use_cache and selection_fn is None and scan_dir.is_dir()

    # Glob ONCE and reuse the same file list for the signature and the load, so
    # the cached signature always matches the data actually loaded (no race
    # between a check-glob and a load-glob while a job is still writing files).
    files = sorted(scan_dir.glob('*.jsonl')) if scan_dir.is_dir() else []

    if cacheable:
        cache = root / '_cache' / f'{name}.slim.pkl'
        sig = _dir_signature(files)
        if cache.is_file():
            try:
                blob = pickle.loads(cache.read_bytes())
                if blob.get('sig') == sig:
                    print(f"Loaded {len(blob['df'])} runs from {cache} (slim cache)")
                    return blob['df']
            except Exception:
                pass  # stale/corrupt cache -> rebuild

    records: list[dict] = []

    # New format: directory of per-run JSONL files
    if files:
        for f in files:
            if selection_fn is not None and not selection_fn(f.name):
                continue
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        rec['_scan'] = name
                        records.append(_slim_record(rec) if slim
                                       else _attach_diagnostics(rec, scan_dir))
        if records:
            print(f"Loaded {len(records)} runs from {scan_dir}/ (directory format"
                  f"{', slim' if slim else ''})")
            df = pd.DataFrame(records)
            if cacheable:
                cache.parent.mkdir(parents=True, exist_ok=True)
                # Re-signature from the same `files` list (unchanged since the glob),
                # written atomically so a concurrent reader never sees a partial pickle.
                tmp = cache.with_suffix('.pkl.tmp')
                tmp.write_bytes(pickle.dumps({'sig': sig, 'df': df}))
                tmp.replace(cache)
            return df

    # Old format: single JSONL file
    jsonl_path = root / f'{name}.jsonl'
    if jsonl_path.is_file():
        with open(jsonl_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    records.append(_slim_record(rec) if slim else rec)
        print(f"Loaded {len(records)} runs from {jsonl_path} (single-file format)")
        return pd.DataFrame(records)

    raise FileNotFoundError(
        f"No results found for '{name}': tried {scan_dir}/ and {jsonl_path}"
    )

def load_results_jsonl(name, results_root='../experiment_results', slim=True):
    """Back-compat shim. Results are now stored per-run in a ``{name}/`` directory
    (the fresh Gram-backend runs); the old single ``{name}.jsonl`` file is gone.
    Delegates to :func:`load_results` (which handles both layouts + the slim cache).
    """
    return load_results(name, results_root=results_root, slim=slim)

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
        # C27: a fallback list instead of a bare 'arial' (absent on Linux -> warning spam)
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
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


# ---------------------------------------------------------------------------
# Optimizer colours -- THE convention, for every notebook and helper module
# ---------------------------------------------------------------------------
# One colour per optimizer, keyed by name, so a method looks the same in the scan
# notebooks, the study notebooks and the profiling plots.  Never colour an optimizer
# positionally (f'C{i}', a palette indexed by plot order, matplotlib's default
# cycle): the colour then depends on which methods happen to be present and how
# they are sorted.  Look it up with :func:`method_color` instead.
#
# Sven is black.  Baselines are seaborn "deep" plus three extra hues chosen to stay
# apart from it (olive, navy, dark grey).  To add an optimizer, add a line here.
METHOD_COLORS = {
    'Sven':      '#000000',
    'SGD':       '#55A868',   # green
    'PolyakSGD': '#CCB974',   # khaki
    'RMSprop':   '#C44E52',   # red
    'Adam':      '#4C72B0',   # blue
    'AdamW':     '#64B5CD',   # light blue
    'LBFGS':     '#8C8C8C',   # grey
    'Muon':      '#8172B3',   # purple
    'MuonW':     '#B7A9DB',   # light purple (Muon at its default wd = 0.1, as AdamW is to Adam)
    'SOAP':      '#DD8452',   # orange
    'Shampoo':   '#DA8BC3',   # pink
    'KFAC':      '#937860',   # brown
    'JD':        '#6B8E23',   # olive
    'HIG':       '#1B3A57',   # navy
}
# Other spellings of the same optimizer (result files, profiler, legacy labels).
METHOD_ALIASES = {
    'SVD': 'Sven', 'sven': 'Sven',
    'JD_UPGrad': 'JD', 'JD (UPGrad)': 'JD',
    'LBFGS1': 'LBFGS', 'K-FAC': 'KFAC', 'Polyak': 'PolyakSGD',
}
_UNKNOWN_METHOD_COLOR = '#333333'


def canonical_method(name):
    """The name an optimizer goes by in :data:`METHOD_COLORS` (``'SVD'`` -> ``'Sven'``)."""
    return METHOD_ALIASES.get(name, name)


def method_color(name):
    """The global colour of an optimizer.  Unknown names get a dark grey and a
    one-time warning, so a new optimizer is noticed rather than silently cycled."""
    key = canonical_method(name)
    if key not in METHOD_COLORS:
        if key not in method_color._warned:
            method_color._warned.add(key)
            print(f"[style] no colour registered for optimizer {name!r}; "
                  f"add it to style.METHOD_COLORS")
        return _UNKNOWN_METHOD_COLOR
    return METHOD_COLORS[key]


method_color._warned = set()


# ---------------------------------------------------------------------------
# Two conventions shared by every notebook (see ANALYSIS_FIXES.md, A4 / A5)
# ---------------------------------------------------------------------------
DIVERGED_FACTOR = 10.0


def final_value(curve):
    """The final value of a per-epoch curve -- NaN if it ends non-finite.

    DIVERGED = FAILED.  A diverged run (see :func:`is_diverged`) has no final loss: it
    is left out of its configuration's seed mean and counted in ``n_diverged``; it is
    NOT scored by its last finite value, which would flatter a method that blows up late.
    """
    if curve is None or len(curve) == 0:
        return np.nan
    try:
        v = float(curve[-1])
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def is_diverged(train_curve, val_curve):
    """The one definition of a diverged run: its train or val curve ends non-finite, OR
    its validation loss ends more than ``DIVERGED_FACTOR`` times above the pre-training
    value at ``val[0]`` (a finite blow-up -- the paramfrac scans end some runs at
    1e7..1e15 without ever producing a NaN).  Only the val curve is used for the ratio:
    it starts with the untrained model, whereas ``train[0]`` is already the post-epoch-1
    loss and makes a poor reference."""
    fv, ft = final_value(val_curve), final_value(train_curve)
    if not (np.isfinite(fv) and np.isfinite(ft)):
        return True
    try:
        v0 = float(val_curve[0])
    except (TypeError, ValueError, IndexError):
        return False
    return bool(np.isfinite(v0) and v0 > 0 and fv > DIVERGED_FACTOR * v0)


def config_eligible(n_ok, n_expected):
    """Whether a configuration may be picked as a method's best: more than half of
    the scan's seeds must have finished (not diverged, not missing)."""
    return np.asarray(n_ok) > np.asarray(n_expected) / 2


def clipped_band(mean, std, lo):
    """``(lower, upper)`` of the seed band drawn everywhere: mean +/- 1 std (ddof=1),
    with the lower edge clipped at the lowest seed ``lo`` so it never reaches <= 0
    on a log axis.  The line itself stays the arithmetic seed mean."""
    mean, lo = np.asarray(mean, dtype=float), np.asarray(lo, dtype=float)
    std = np.nan_to_num(np.asarray(std, dtype=float))
    return np.maximum(mean - std, lo), mean + std


def clipped_yerr(mean, std, lo):
    """The same band as an asymmetric ``yerr=[below, above]`` for ``bar`` / ``errorbar``."""
    lower, upper = clipped_band(mean, std, lo)
    mean = np.asarray(mean, dtype=float)
    return np.vstack([mean - lower, upper - mean])


# ---------------------------------------------------------------------------
# Dataset / task names -- one spelling everywhere (ANALYSIS_FIXES C22)
# ---------------------------------------------------------------------------
DATASET_TITLES = {
    'toy_1d':         'Toy 1D',
    'polynomial':     'Random Polynomial',
    'mnist_ce':       'MNIST (CE)',
    'mnist_labelreg': 'MNIST (label reg.)',
    'cifar_ce':       'CIFAR-10 (CE)',
    'cifar_labelreg': 'CIFAR-10 (label reg.)',
    'nanogpt':        'nanoGPT (tiny-shakespeare)',
}
# Axis labels for the final-loss metrics ("seed mean" is appended by the plot helpers).
METRIC_LABELS = {'final_val_loss': 'Final validation loss', 'final_train_loss': 'Final train loss',
                 'final_val_acc': 'Final validation accuracy'}


def metric_label(metric, seed_mean=True):
    base = METRIC_LABELS.get(metric, metric.replace('_', ' '))
    return base + (' (seed mean)' if seed_mean else '')

