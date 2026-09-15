"""Singular-value diagnostics for the split result layout.

A run is written as two files (see ``experiments/experiment_code/generic_scan``):

* ``{scan}/{run_id}.jsonl`` -- light: hparams, per-EPOCH curves, timings, and a
  per-epoch ``svd_summary``.  This is what :func:`style.load_results` reads (and
  caches); loading a whole scan is cheap.
* ``{scan}/diag/{run_id}.npz`` -- heavy: the per-BATCH arrays
  (``train_batch``/``val_batch``/batch times), the per-STEP ``num_nonzero_svs``
  /``sv_max``/``sv_min``, and the spectra ``svs`` (NaN-padded, one row per saved
  step) with their step indices ``svs_step``.

The rule this module follows -- and the one the notebooks should follow -- is:
**load the scan light, pull diagnostics only for the handful of runs you plot.**
Do NOT call ``load_results(..., slim=False)`` on a full scan: that re-inflates
every one of the ~650-920 runs into the legacy inline layout.

Two things changed versus the old inline format, and both bite silently:

1. **Spectra are subsampled.**  ``svs`` holds every ``spectra_every``-th step
   (default 20), not every step.  The old idiom ``svs[ep*bpe:(ep+1)*bpe]`` with
   ``bpe = len(svs)//n_epoch`` therefore mis-bins steps into epochs.  Use
   ``svs_step`` -- :func:`epoch_spectra` does.
2. **Scans are multi-seed** (5 model seeds per config).  The old
   ``assert len(sel) == 1`` no longer holds; select a seed or average over them
   with :func:`seed_mean`.

The per-step SV count recorded is ``num_nonzero_svs`` = ``min(k, #{sigma_i >
rtol * sigma_0})`` -- it already saturates at the ``k`` cap.  So the same array
carries both readings, distinguished by which axis you sweep:

* **rank**: fix ``k = B`` and vary ``rtol`` -- the rtol-rank of the batch
  Jacobian, uncapped in practice (:func:`plot_rank_vs_batch`).
* **used**: fix ``lr``/``rtol`` and vary ``k`` -- how many SVs the optimizer
  actually inverts, and where that saturates against ``k``
  (:func:`plot_used_vs_batch`).
"""
import warnings

import numpy as np

from style import load_diagnostics, lr_labels

RESULTS_ROOT = '../experiment_results'

__all__ = ['RESULTS_ROOT', 'sven_runs', 'select', 'n_epochs', 'n_steps',
           'rank_per_batch', 'rank_per_epoch', 'batch_curve', 'epoch_spectra',
           'seed_mean', 'smooth', 'sci', 'plot_rank_vs_batch',
           'plot_used_vs_batch', 'plot_epoch_spectra', 'epoch_colorbar']


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def sven_runs(df):
    """The Sven (``optimizer == 'SVD'``) rows of a scan DataFrame."""
    return df[df['optimizer'] == 'SVD'].copy()


def select(df, **constraints):
    """Rows matching every ``column=value`` constraint (None constraints ignored).

    Unlike the old notebooks this does NOT assert a unique match: the fresh scans
    carry several model seeds per configuration.  Returns a DataFrame; pass the
    result to :func:`seed_mean`, or add ``model_seed=...`` to pin one run.
    """
    sel = df
    for col, val in constraints.items():
        if val is None:
            continue
        sel = sel[sel[col] == val]
    return sel


def n_epochs(row):
    """Number of training epochs recorded for a run."""
    return len(row['losses']['train'])


def n_steps(row):
    """Number of optimizer steps (train batches), from the light ``svd_summary``.

    Baseline runs carry no ``svd_summary``; for them this falls back to an inline
    ``losses['train_batch']``, which only legacy records have (the per-batch
    series moved into ``diag/*.npz``), and otherwise returns 0.  That is fine
    here: the only caller that needs a step count is :func:`epoch_spectra`, which
    runs on Sven runs.
    """
    summary = row.get('svd_summary')
    if isinstance(summary, dict) and summary.get('n_steps'):
        return int(summary['n_steps'])
    curve = (row.get('losses') or {}).get('train_batch')
    return len(curve) if curve is not None else 0


# ---------------------------------------------------------------------------
# Per-run diagnostics
# ---------------------------------------------------------------------------
def rank_per_batch(row, results_root=RESULTS_ROOT):
    """Per-step ``num_nonzero_svs`` for one run, as a float array, or None.

    Reads the run's ``diag/*.npz``.  One value per train batch; this is the
    quantity plotted by both :func:`plot_rank_vs_batch` and
    :func:`plot_used_vs_batch`.
    """
    diag = load_diagnostics(row, results_root=results_root)
    nnz = diag.get('num_nonzero_svs')
    return None if nnz is None else np.asarray(nnz, dtype=float)


def rank_per_epoch(row):
    """Per-epoch mean ``num_nonzero_svs``, straight from the light record.

    No ``diag/`` read at all -- ``svd_summary['num_nonzero_svs_epoch']`` is
    written alongside the epoch curves.  Use this when per-epoch resolution is
    enough; use :func:`rank_per_batch` for the batch-resolved version.
    """
    summary = row.get('svd_summary')
    if not isinstance(summary, dict):
        return None
    curve = summary.get('num_nonzero_svs_epoch')
    return None if not curve else np.asarray(curve, dtype=float)


def batch_curve(row, which='train_batch', results_root=RESULTS_ROOT):
    """A per-batch series (``train_batch``, ``val_batch``, ``batch_times_*``).

    These moved out of ``losses`` into the npz; this is the replacement for the
    old ``row['losses']['train_batch']``.
    """
    diag = load_diagnostics(row, results_root=results_root)
    arr = diag.get(which)
    return None if arr is None else np.asarray(arr, dtype=float)


def epoch_spectra(row, results_root=RESULTS_ROOT, normalize=True):
    """Per-epoch mean spectrum *shape*, shape ``(n_epoch, width)``.

    Each saved spectrum is a descending ``sigma``; its shape is
    ``sigma_i / sigma_0``.  Shapes are averaged over the saved steps that fall
    inside each epoch, ignoring the NaN padding -- so index ``i`` is averaged
    only over the steps whose rank actually reached ``i`` (the old code's
    ``denoms`` bookkeeping, vectorised).

    Steps are binned by ``svs_step`` against the run's total step count, because
    spectra are saved only every ``spectra_every``-th step.  Rows for epochs with
    no saved spectrum stay NaN.

    Pass ``normalize=False`` for the raw sigma scale instead of the shape.
    """
    diag = load_diagnostics(row, results_root=results_root)
    svs, step = diag.get('svs'), diag.get('svs_step')
    if svs is None or step is None or svs.size == 0:
        return None
    svs = np.asarray(svs, dtype=float)
    n_ep = n_epochs(row)
    total = n_steps(row) or (int(step[-1]) + 1)
    steps_per_epoch = total / n_ep
    epoch_of = np.minimum((np.asarray(step) // steps_per_epoch).astype(int), n_ep - 1)

    shape = svs / svs[:, :1] if normalize else svs
    out = np.full((n_ep, svs.shape[1]), np.nan)
    with warnings.catch_warnings():  # all-NaN columns are expected (ragged ranks)
        warnings.simplefilter('ignore', RuntimeWarning)
        for ep in range(n_ep):
            rows = shape[epoch_of == ep]
            if len(rows):
                out[ep] = np.nanmean(rows, axis=0)
    return out


# ---------------------------------------------------------------------------
# Seeds and smoothing
# ---------------------------------------------------------------------------
def seed_mean(rows, fn, **kwargs):
    """Mean of ``fn(row, **kwargs)`` over the runs in ``rows`` (a DataFrame).

    Curves are truncated to the shortest and averaged with ``nanmean``; returns
    None if no run yields a curve.  Works for both 1-D curves
    (:func:`rank_per_batch`) and 2-D spectra (:func:`epoch_spectra`).
    """
    curves = [np.asarray(c, dtype=float)
              for c in (fn(r, **kwargs) for _, r in rows.iterrows()) if c is not None]
    if not curves:
        return None
    # Axis 0 is time (batch/epoch): truncate to the shortest run.  Any trailing
    # axis is SV rank, which is ragged across seeds (a seed whose rtol-rank ran
    # lower has a narrower spectrum) -- pad those with NaN so index i still means
    # rank i, and let nanmean average over the seeds that reached it.
    n = min(len(c) for c in curves)
    width = tuple(max(c.shape[ax] for c in curves) for ax in range(1, curves[0].ndim))
    padded = []
    for c in curves:
        c = c[:n]
        if width:
            out = np.full((n,) + width, np.nan)
            out[tuple(slice(0, d) for d in c.shape)] = c
            c = out
        padded.append(c)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(np.stack(padded), axis=0)


def smooth(x, window):
    """Centred-ish moving average; returns ``(x_smoothed, indices)``.

    ``indices`` are the original positions of the smoothed points, so the result
    plots against the true batch axis.
    """
    x = np.asarray(x, dtype=float)
    window = int(max(1, min(window, len(x))))
    if window == 1:
        return x, np.arange(len(x))
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='valid'), np.arange(window - 1, len(x))


# ---------------------------------------------------------------------------
# The three plots
# ---------------------------------------------------------------------------
def _x_axis(n, progress):
    """Batch indices, or 0..1 training progress when scans of different lengths
    have to share an axis."""
    return np.linspace(0, 1, n) if progress else np.arange(n)


def sci(value):
    """``value`` as a LaTeX-ready power of ten, e.g. 1e-4 -> ``10^{-4}``."""
    return lr_labels.get(value, f'{value:g}')


def plot_rank_vs_batch(df, ax, k, lr, rtols, colors=None, styles=None,
                       smooth_frac=0.02, normalize_by=None, raw_alpha=0.15,
                       x_progress=False, results_root=RESULTS_ROOT, **plot_kw):
    """Rank (nonzero SVs) vs train batch, one line per ``rtol``, averaged over seeds.

    With ``k`` set to the batch size the count is not clipped by the cap, so it
    reads as the rtol-rank of the batch Jacobian.  ``normalize_by=B`` divides the
    y-axis by the batch size (giving the 0..1 "fraction of B" axis the paper
    plots use).  ``smooth_frac`` is the moving-average window as a fraction of
    the run's total steps; the unsmoothed trace is drawn faintly underneath.

    Returns the list of legend handles (one per rtol).
    """
    from matplotlib.lines import Line2D

    colors = [f'C{i}' for i in range(len(rtols))] if colors is None else list(colors)
    styles = ['-'] * len(rtols) if styles is None else list(styles)
    handles = []
    for i, rtol in enumerate(rtols):
        runs = select(df, k=k, lr=lr, rtol=rtol)
        curve = seed_mean(runs, rank_per_batch, results_root=results_root)
        if curve is None:
            continue
        if normalize_by:
            curve = curve / normalize_by
        window = max(1, int(smooth_frac * len(curve)))
        x_all = _x_axis(len(curve), x_progress)
        ax.plot(x_all, curve, color=colors[i], lw=1, alpha=raw_alpha, zorder=1)
        ys, xs = smooth(curve, window)
        ax.plot(x_all[xs], ys, color=colors[i], linestyle=styles[i], zorder=2, **plot_kw)
        handles.append(Line2D([], [], color=colors[i], linestyle=styles[i],
                              label=f'rtol $= {sci(rtol)}$'))
    ax.set_xlabel('Training progress' if x_progress else 'Train batch')
    ax.set_ylabel('Nonzero SVs' + (' / $B$' if normalize_by else ''))
    return handles


def plot_used_vs_batch(df, ax, lr, rtol, ks, colors=None, smooth_frac=0.02,
                       show_cap=True, raw_alpha=0.15, x_progress=False,
                       normalize_by=None, results_root=RESULTS_ROOT, **plot_kw):
    """Used SVs vs train batch, one line per ``k``, averaged over seeds.

    ``show_cap`` draws each ``k`` as a dotted horizontal line, which is where the
    count saturates once the rtol-rank exceeds the cap.  ``x_progress`` /
    ``normalize_by`` rescale the axes to 0..1 so different scans can be overlaid.

    Returns the list of legend handles (one per k).
    """
    from matplotlib.lines import Line2D

    ks = list(ks)
    colors = [f'C{i}' for i in range(len(ks))] if colors is None else list(colors)
    handles = []
    for i, k in enumerate(ks):
        runs = select(df, k=k, lr=lr, rtol=rtol)
        curve = seed_mean(runs, rank_per_batch, results_root=results_root)
        if curve is None:
            continue
        if normalize_by:
            curve = curve / normalize_by
        window = max(1, int(smooth_frac * len(curve)))
        x_all = _x_axis(len(curve), x_progress)
        ax.plot(x_all, curve, color=colors[i], lw=1, alpha=raw_alpha, zorder=1)
        ys, xs = smooth(curve, window)
        ax.plot(x_all[xs], ys, color=colors[i], zorder=2, **plot_kw)
        if show_cap:
            ax.axhline(k / normalize_by if normalize_by else k, color=colors[i],
                       lw=0.8, ls=':', alpha=0.6, zorder=0)
        handles.append(Line2D([], [], color=colors[i], label=f'$k = {int(k)}$'))
    ax.set_xlabel('Training progress' if x_progress else 'Train batch')
    ax.set_ylabel('SVs used' + (' / $B$' if normalize_by else ''))
    if not normalize_by:
        ax.set_yscale('log')
    return handles


def plot_epoch_spectra(df, ax, k, lr, rtol, cmap='plasma', lo=0.25, hi=1.0,
                       floor=1e-6, x_fraction=True, results_root=RESULTS_ROOT,
                       **plot_kw):
    """Per-epoch spectrum shapes ``sigma_i / sigma_0``, coloured by epoch.

    One line per epoch, averaged over the batches in that epoch and over seeds.
    ``x_fraction`` puts SV rank on a 0..1 axis (so different ``k`` overlay);
    otherwise the raw rank index is used.  ``lo``/``hi`` restrict the colormap
    range so early epochs stay visible against a light background.

    Returns ``(spectra, norm)`` -- the ``(n_epoch, width)`` array and the
    :class:`~matplotlib.colors.Normalize` for the epoch colourbar.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    runs = select(df, k=k, lr=lr, rtol=rtol)
    spectra = seed_mean(runs, epoch_spectra, results_root=results_root)
    if spectra is None:
        return None, None
    cmap = plt.get_cmap(cmap)
    n_ep, width = spectra.shape
    norm = mcolors.Normalize(vmin=1, vmax=n_ep)
    lw = plot_kw.pop('lw', 1.3)
    for ep in range(n_ep):
        shape = spectra[ep]
        valid = ~np.isnan(shape)
        if not valid.any():
            continue
        y = np.clip(shape[valid], floor, 1.0)
        x = (np.arange(width)[valid] / max(width - 1, 1) if x_fraction
             else np.arange(width)[valid])
        ax.plot(x, y, color=cmap(lo + (hi - lo) * norm(ep + 1)), lw=lw, **plot_kw)
    ax.set_yscale('log')
    ax.set_xlabel('SV rank' + (' / $k$' if x_fraction else ''))
    ax.set_ylabel(r'$\sigma_i \ / \ \sigma_0$')
    return spectra, norm


def epoch_colorbar(fig, ax, norm, cmap='plasma', lo=0.25, hi=1.0, label='Epoch',
                   pad=0.02, width=0.017):
    """Colourbar for :func:`plot_epoch_spectra`, matched to its clipped colormap.

    ``plot_epoch_spectra`` only uses the ``lo..hi`` slice of ``cmap`` (so epoch 1
    is not invisibly pale); this rebuilds that slice so the bar shows the colours
    actually drawn.  Call after the figure is laid out.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    base = plt.get_cmap(cmap)
    clipped = mcolors.LinearSegmentedColormap.from_list(
        f'{base.name}_clipped', base(np.linspace(lo, hi, 256)))
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + pad, pos.y0, width, pos.height])
    sm = cm.ScalarMappable(cmap=clipped, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(label)
    lo_ep, hi_ep = int(np.ceil(norm.vmin)), int(np.floor(norm.vmax))
    step = max(1, round((hi_ep - lo_ep) / 4))
    cbar.set_ticks(list(range(lo_ep, hi_ep + 1, step)))  # epochs are integers
    return cbar
