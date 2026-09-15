"""Shared analysis for the per-dataset scan notebooks (toy-1d / polynomial / MNIST).

The four scan notebooks were near-identical copies of the same ~50 cells.  This
module holds that body once; each notebook is now a thin driver that builds a
:class:`Scan` and calls the plot functions it wants.

Written against the split result layout (see :mod:`sv_diagnostics` for the full
description):

* the scan is loaded **light** -- :func:`style.load_results` with the default
  ``slim=True``, which reads only the per-run JSONL (hparams, per-epoch curves,
  timings, ``svd_summary``) and caches the DataFrame.  Loading ~900 runs is then
  near-instant.
* per-**batch** series (``train_batch``, ``val_batch``, batch times) and the SV
  spectra now live in ``diag/{run_id}.npz`` and are pulled **per run, on demand**
  by :func:`sv_diagnostics.batch_curve` / the ``plot_*`` functions here.  The old
  ``row['losses']['train_batch']`` no longer exists.

The other change these notebooks have to absorb: the fresh scans run **5 model
seeds per configuration**, so the old ``assert len(sel) == 1`` is gone.  A
"configuration" here is ``(k, lr, rtol)`` for Sven and ``(optimizer, lr, ...)``
for a baseline; every metric and curve is aggregated over the seeds of a config,
with the seed spread available as an error band.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import sv_diagnostics as sv
from style import load_results, lr_labels

RESULTS_ROOT = '../experiment_results'

# seaborn's "deep" palette, inlined so this module needs only numpy/pandas/
# matplotlib (the notebooks may still import seaborn for their own tweaks).
DEEP = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
        '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']

# Hyperparameters that jointly identify one configuration, per optimizer family.
SVEN_CONFIG = ['k', 'lr', 'rtol']
BASELINE_CONFIG = ['optimizer', 'lr', 'lbfgs_max_iter', 'lbfgs_history_size']


def method_colors(baselines, colors=None, sven='k'):
    """Stable ``method -> colour`` map: Sven black, baselines in ``baselines`` order.

    Every plot looks colours up by method name through this, so a method keeps the
    same colour across the loss curves, the timing bars and the efficiency scatter
    -- which a positional ``f'C{i}'`` does not once there are more than ten
    methods, or once a plot orders them by score rather than by config.
    ``colors`` may be a list (cycled) or an explicit ``{method: colour}`` dict.
    """
    if isinstance(colors, dict):
        return {'Sven': sven, **colors}
    palette = DEEP if colors is None else list(colors)
    return {'Sven': sven,
            **{b: palette[i % len(palette)] for i, b in enumerate(baselines)}}


# ---------------------------------------------------------------------------
# Curve helpers
# ---------------------------------------------------------------------------
def final(curve):
    """Last finite value of an epoch curve, or NaN.

    Not simply ``curve[-1]``: LBFGS runs can go non-finite part-way and the last
    real value is the informative one.
    """
    if curve is None:
        return np.nan
    a = np.asarray(curve, dtype=float)
    ok = np.isfinite(a)
    return float(a[ok][-1]) if ok.any() else np.nan


def curve_of(row, which='val'):
    """A per-epoch curve from the light record (``train``/``val``/``*_acc``/
    ``epoch_times``), as a float array, or None."""
    losses = row.get('losses')
    if not isinstance(losses, dict) or losses.get(which) is None:
        return None
    return np.asarray(losses[which], dtype=float)


def seed_curve(rows, which='val'):
    """``(mean, std)`` of a per-epoch curve over the seeds of one configuration.

    Curves are truncated to the shortest seed.  ``std`` uses ddof=1 when more
    than one seed is present, so it can be used directly for a
    ``fill_between`` band.  Returns ``(None, None)`` if nothing is available.
    """
    curves = [c for c in (curve_of(r, which) for _, r in rows.iterrows()) if c is not None]
    if not curves:
        return None, None
    n = min(len(c) for c in curves)
    stacked = np.stack([c[:n] for c in curves])
    ddof = 1 if len(stacked) > 1 else 0
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=ddof)


def epoch_axis(rows, curve, which='val', versus='epoch'):
    """x-values for a seed-averaged epoch curve.

    ``val``/``val_acc`` are recorded once before training starts, so they are one
    entry longer than ``train`` and must be plotted from epoch 0 -- and, against
    wall time, from t=0 rather than from the end of epoch 1.  Getting this wrong
    silently shifts every validation curve by one epoch (and, on a time axis,
    plots epoch indices as seconds).
    """
    train = curve_of(rows.iloc[0], 'train')
    n_ep = 0 if train is None else len(train)
    pre_training = len(curve) == n_ep + 1
    if versus == 'time':
        times, _ = seed_curve(rows, 'epoch_times')
        if times is None:
            return np.arange(len(curve))
        x = np.cumsum(times)
        if pre_training:
            x = np.concatenate([[0.0], x])
    else:
        x = np.arange(len(curve)) if pre_training else np.arange(1, len(curve) + 1)
    return x[:len(curve)]


def sliding_average(data, window=10):
    """Moving average, ``mode='valid'`` (kept for notebook-level ad-hoc use)."""
    return np.convolve(np.asarray(data, dtype=float), np.ones(window) / window, mode='valid')


def fmt(value):
    """A hyperparameter value as LaTeX-ready maths (``10^{-4}``, ``0.5``, ...)."""
    return lr_labels.get(value, f'{value:g}')


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------
@dataclass
class Scan:
    """One experiment-results directory, loaded light, with the axes it sweeps.

    Attributes
    ----------
    df / sven / baseline : the full, Sven-only and baseline-only run tables.
    B : batch size.  ``k = B`` is the uncapped-rank configuration.
    ks / sven_lrs / rtols : the Sven sweep axes actually present.
    baselines : baseline optimizer names, in the order plots should use.
    """
    name: str
    title: str
    plot_dir: Path
    df: pd.DataFrame
    metric: str = 'final_val_loss'
    results_root: str = RESULTS_ROOT
    sven: pd.DataFrame = field(init=False)
    baseline: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.plot_dir = Path(self.plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        d = self.df
        d['final_val_loss'] = d.apply(lambda r: final(curve_of(r, 'val')), axis=1)
        d['final_train_loss'] = d.apply(lambda r: final(curve_of(r, 'train')), axis=1)
        d['final_val_acc'] = d.apply(lambda r: final(curve_of(r, 'val_acc')), axis=1)
        d['total_time'] = d['losses'].apply(lambda L: L.get('total_time', np.nan))
        d['avg_epoch_time'] = d['losses'].apply(lambda L: L.get('avg_epoch_time', np.nan))
        d['avg_batch_time_train'] = d['losses'].apply(
            lambda L: L.get('avg_batch_time_train', np.nan))
        d['peak_gpu_mem_mb'] = d['losses'].apply(lambda L: L.get('peak_gpu_mem_mb', np.nan))
        self.sven = d[d['optimizer'] == 'SVD'].copy()
        self.baseline = d[d['optimizer'] != 'SVD'].copy()

    # -- axes ------------------------------------------------------------
    @property
    def B(self):
        return int(self.df['batch_size'].iloc[0])

    @property
    def n_epochs(self):
        return len(self.df['losses'].iloc[0]['train'])

    @property
    def ks(self):
        return sorted(int(k) for k in self.sven['k'].dropna().unique())

    @property
    def rtols(self):
        return sorted(float(r) for r in self.sven['rtol'].dropna().unique())

    @property
    def sven_lrs(self):
        return sorted(float(v) for v in self.sven['lr'].dropna().unique())

    @property
    def seeds(self):
        return sorted(int(v) for v in self.df['model_seed'].dropna().unique())

    @property
    def baselines(self):
        return sorted(self.baseline['optimizer'].unique())

    @property
    def has_acc(self):
        return self.df['final_val_acc'].notna().any()

    # -- selection -------------------------------------------------------
    def runs(self, df=None, **constraints):
        """All seed-runs matching ``column=value`` constraints.

        ``None`` means "do not constrain".  A NaN value means "this hyperparameter
        does not apply to this optimizer" (e.g. ``lbfgs_history_size`` on Adam) and
        is matched against NaN -- plain ``== nan`` would silently match nothing,
        which is how the config dicts from :meth:`best_baseline` used to come back
        empty.
        """
        out = self.sven if df is None else df
        for col, val in constraints.items():
            if val is None or col not in out.columns:
                continue
            if isinstance(val, float) and np.isnan(val):
                out = out[out[col].isna()]
            else:
                out = out[out[col] == val]
        return out

    def configs(self, df, keys, metric=None):
        """Per-configuration table: seed mean/std of ``metric``, plus ``n_seeds``.

        This is what replaces picking a single run: a configuration's score is
        the mean over its seeds, so `best_*` never latches onto a lucky seed.
        """
        metric = metric or self.metric
        keys = [k for k in keys if k in df.columns]
        g = df.dropna(subset=[metric]).groupby(keys, dropna=False)[metric]
        out = g.agg(score='mean', score_std='std', n_seeds='size').reset_index()
        return out.sort_values('score')

    def best_sven(self, metric=None, **fixed):
        """The best Sven configuration (seed-mean ``metric``) as a dict of hparams.

        ``fixed`` pins any of ``k``/``lr``/``rtol``; the rest are optimised over.
        """
        sel = self.runs(**fixed)
        cfg = self.configs(sel, SVEN_CONFIG, metric)
        if cfg.empty:
            return None
        return {k: cfg.iloc[0][k] for k in SVEN_CONFIG if k in cfg.columns}

    def best_baseline(self, optimizer, metric=None):
        """The best configuration for one baseline optimizer, as a dict of hparams."""
        sel = self.baseline[self.baseline['optimizer'] == optimizer]
        cfg = self.configs(sel, BASELINE_CONFIG, metric)
        if cfg.empty:
            return None
        return {k: cfg.iloc[0][k] for k in BASELINE_CONFIG if k in cfg.columns}

    def sven_rows(self, **cfg):
        """Every seed-run of one Sven configuration."""
        return self.runs(**cfg)

    def baseline_rows(self, cfg):
        """Every seed-run of one baseline configuration."""
        return self.runs(df=self.baseline, **cfg)

    # -- labels / IO -----------------------------------------------------
    def sven_label(self, cfg, with_rtol=True):
        bits = [rf'\eta={fmt(cfg["lr"])}', rf'k={int(cfg["k"])}']
        if with_rtol:
            bits.append(rf'\mathrm{{rtol}}={fmt(cfg["rtol"])}')
        return 'Sven (' + ', '.join(f'${b}$' for b in bits) + ')'

    def save(self, fig, filename):
        """Save under the scan's plot directory and return the path."""
        path = self.plot_dir / filename
        fig.savefig(path)
        return path


def load_scan(name, title, plot_dir, results_root=RESULTS_ROOT, drop_diverged=True,
              metric='final_val_loss'):
    """Load one scan directory light and wrap it in a :class:`Scan`.

    ``drop_diverged`` removes runs with no finite validation loss at all (LBFGS
    blows up on some configs and would otherwise poison the group means).
    """
    df = load_results(name, results_root=results_root)
    scan = Scan(name=name, title=title, plot_dir=plot_dir, df=df, metric=metric,
                results_root=results_root)
    if drop_diverged:
        bad = scan.df['final_val_loss'].isna()
        if bad.any():
            print(f'  dropping {int(bad.sum())} run(s) with no finite val loss '
                  f'({sorted(scan.df.loc[bad, "optimizer"].unique())})')
            scan.df = scan.df[~bad].copy()
            scan.__post_init__()
    print(f'{scan.name}: {len(scan.df)} runs  |  B={scan.B}, {scan.n_epochs} epochs, '
          f'{len(scan.seeds)} seeds  |  Sven k={scan.ks} rtol={scan.rtols}')
    return scan


# ---------------------------------------------------------------------------
# Loss / convergence
# ---------------------------------------------------------------------------
def plot_best_curves(scan, ax, which='train', versus='epoch', baselines=None,
                     colors=None, band=True, sven_kw=None, **plot_kw):
    """Best Sven vs the best of each baseline, seed-averaged, with a seed band.

    ``which``   -- ``'train'``, ``'val'``, ``'val_acc'``, ...
    ``versus``  -- ``'epoch'`` or ``'time'`` (cumulative epoch wall time).

    Returns the dict of configurations plotted, keyed by label.
    """
    baselines = baselines if baselines is not None else scan.baselines
    cmap = method_colors(baselines, colors)
    chosen = {}

    def draw(rows, label, color, lw, zorder):
        mean, std = seed_curve(rows, which)
        if mean is None:
            return
        x = epoch_axis(rows, mean, which, versus)
        mean, std = mean[:len(x)], (None if std is None else std[:len(x)])
        ax.plot(x, mean, color=color, lw=lw, label=label, zorder=zorder, **plot_kw)
        if band and std is not None and len(rows) > 1:
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2,
                            lw=0, zorder=zorder - 0.5)

    cfg = scan.best_sven()
    if cfg is not None:
        label = scan.sven_label(cfg)
        chosen[label] = cfg
        draw(scan.sven_rows(**cfg), label, cmap['Sven'], (sven_kw or {}).get('lw', 3), 3)

    for opt in baselines:
        cfg = scan.best_baseline(opt)
        if cfg is None:
            continue
        chosen[opt] = cfg
        draw(scan.baseline_rows(cfg), opt, cmap[opt], 2, 2)

    ax.set_xlabel('Wall time (s)' if versus == 'time' else 'Epoch')
    ax.set_ylabel({'train': 'Train loss', 'val': 'Validation loss',
                   'val_acc': 'Validation accuracy'}.get(which, which))
    if which != 'val_acc':
        ax.set_yscale('log')
    return chosen


def plot_k_sweep(scan, ax, lr, rtol, which='train', reference='SGD', cmap='viridis',
                 band=True, **plot_kw):
    """Loss curves across ``k`` at fixed ``lr``/``rtol``, with a baseline reference.

    Seed-averaged, with a seed band.  Returns the ks actually drawn.
    """
    import matplotlib.pyplot as plt

    ks = scan.ks
    colors = plt.get_cmap(cmap)(np.linspace(0, 0.9, len(ks)))
    drawn = []
    for i, k in enumerate(ks):
        rows = scan.sven_rows(k=k, lr=lr, rtol=rtol)
        mean, std = seed_curve(rows, which)
        if mean is None:
            continue
        x = epoch_axis(rows, mean, which)
        mean, std = mean[:len(x)], (None if std is None else std[:len(x)])
        ax.plot(x, mean, color=colors[i], lw=3, label=f'$k={int(k)}$', **plot_kw)
        if band and std is not None and len(rows) > 1:
            ax.fill_between(x, mean - std, mean + std, color=colors[i], alpha=0.2, lw=0)
        drawn.append(k)

    if reference:
        cfg = scan.best_baseline(reference)
        if cfg is not None:
            ref_rows = scan.baseline_rows(cfg)
            mean, _ = seed_curve(ref_rows, which)
            if mean is not None:
                x = epoch_axis(ref_rows, mean, which)
                ax.plot(x, mean[:len(x)], color='k', ls='--', lw=3,
                        label=f'{reference} ($\\eta={fmt(cfg["lr"])}$)')

    ax.set_xlabel('Epoch')
    ax.set_ylabel({'train': 'Train loss', 'val': 'Validation loss'}.get(which, which))
    ax.set_yscale('log')
    return drawn


def plot_batch_loss(scan, ax, configs, which='train_batch', smooth_batches=50,
                    colors=None, labels=None, **plot_kw):
    """Batch-resolved loss for a few configurations, moving-averaged.

    ``configs`` is a list of hparam dicts (Sven configs, or baseline configs
    including ``optimizer``).  The per-batch series lives in ``diag/*.npz``, so
    this reads one npz per configuration -- it uses the first seed of each rather
    than averaging, since batch-level traces are noisy and seed-specific anyway.
    """
    colors = DEEP if colors is None else colors
    for i, cfg in enumerate(configs):
        rows = scan.runs(df=None if 'optimizer' not in cfg else scan.baseline, **cfg)
        if rows.empty:
            continue
        series = sv.batch_curve(rows.iloc[0], which, results_root=scan.results_root)
        if series is None:
            continue
        ys, xs = sv.smooth(series, smooth_batches)
        label = (labels[i] if labels else
                 cfg['optimizer'] if 'optimizer' in cfg else scan.sven_label(cfg))
        ax.plot(xs, ys, color=colors[i % len(colors)], lw=1.5, label=label, **plot_kw)
    ax.set_xlabel('Train batch')
    ax.set_ylabel('Train loss' if which == 'train_batch' else which)
    ax.set_yscale('log')


# ---------------------------------------------------------------------------
# Hyperparameter sensitivity
# ---------------------------------------------------------------------------
def plot_hparam_heatmap(scan, axes, metric=None, cmap='viridis'):
    """log10(metric) over the (k, lr) grid, one panel per rtol, seed-averaged."""
    metric = metric or scan.metric
    cfg = scan.configs(scan.sven, SVEN_CONFIG, metric)
    vmin, vmax = np.log10(cfg['score'].min()), np.log10(cfg['score'].max())
    im = None
    for ax, rtol in zip(np.atleast_1d(axes), scan.rtols):
        pivot = (cfg[cfg['rtol'] == rtol]
                 .pivot_table(values='score', index='k', columns='lr'))
        im = ax.imshow(np.log10(pivot.values), aspect='auto', cmap=cmap,
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{c:g}' for c in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{int(i)}' for i in pivot.index])
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('$k$')
        ax.set_title(f'rtol $= {fmt(rtol)}$')
    for ax in np.atleast_1d(axes)[1:]:
        ax.set_ylabel('')
        ax.set_yticklabels([])
    return im


def plot_sensitivity(scan, ax, x='lr', lines='k', metric=None, **plot_kw):
    """Seed-mean ``metric`` vs one Sven axis, one line per value of another.

    ``x``/``lines`` are any two of ``'k'``, ``'lr'``, ``'rtol'``; the third axis
    is minimised over.  This subsumes the old lr-/k-/rtol-sensitivity cells.
    """
    metric = metric or scan.metric
    cfg = scan.configs(scan.sven, SVEN_CONFIG, metric)
    for i, val in enumerate(sorted(cfg[lines].unique())):
        sub = cfg[cfg[lines] == val].groupby(x)['score'].min().reset_index()
        label = (f'$k={int(val)}$' if lines == 'k' else
                 f'${lines}={fmt(val)}$')
        ax.plot(sub[x], sub['score'], 'o-', color=f'C{i}', label=label, **plot_kw)
    ax.set_xlabel({'lr': 'Learning rate', 'k': '$k$', 'rtol': 'rtol'}[x])
    ax.set_ylabel(metric.replace('_', ' ').title())
    if x in ('lr', 'rtol'):
        ax.set_xscale('log')
    ax.set_yscale('log')


# ---------------------------------------------------------------------------
# Wall time
# ---------------------------------------------------------------------------
def time_excl_first_epoch(row):
    """Total wall time with the first epoch removed (drops compile/warm-up)."""
    et = curve_of(row, 'epoch_times')
    return np.nan if et is None or len(et) < 2 else float(np.sum(et[1:]))


def plot_time_summary(scan, ax, quantity='total_time', baselines=None, **bar_kw):
    """Bar chart of a timing quantity for the best config of each method.

    ``quantity`` is any column on the run table (``total_time``,
    ``avg_batch_time_train``, ``peak_gpu_mem_mb``, ...) or the callable
    :func:`time_excl_first_epoch`.
    """
    baselines = baselines if baselines is not None else scan.baselines
    cmap = method_colors(baselines)
    labels, values = [], []

    def value(rows):
        if callable(quantity):
            return float(np.nanmean([quantity(r) for _, r in rows.iterrows()]))
        return float(np.nanmean(rows[quantity]))

    cfg = scan.best_sven()
    if cfg is not None:
        labels.append('Sven')
        values.append(value(scan.sven_rows(**cfg)))
    for opt in baselines:
        cfg = scan.best_baseline(opt)
        if cfg is None:
            continue
        labels.append(opt)
        values.append(value(scan.baseline_rows(cfg)))

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=[cmap[m] for m in labels], **bar_kw)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(quantity if isinstance(quantity, str) else quantity.__name__)
    return dict(zip(labels, values)), bars


def plot_time_vs_k(scan, ax, quantity='total_time', **plot_kw):
    """Seed-mean wall time vs ``k``, one line per learning rate."""
    for i, lr in enumerate(scan.sven_lrs):
        sub = (scan.runs(lr=lr).groupby('k')[quantity].mean().reset_index())
        ax.plot(sub['k'], sub[quantity], 'o-', color=f'C{i}',
                label=f'$\\eta={fmt(lr)}$', **plot_kw)
    ax.set_xlabel('$k$')
    ax.set_ylabel('Total training time (s)')


def plot_efficiency(scan, ax, metric=None, quantity='total_time', baselines=None):
    """Scatter of ``metric`` vs wall time over every run -- Sven against baselines."""
    metric = metric or scan.metric
    baselines = baselines if baselines is not None else scan.baselines
    cmap = method_colors(baselines)
    ax.scatter(scan.sven[quantity], scan.sven[metric], c=cmap['Sven'], alpha=0.6,
               s=40, label='Sven')
    for opt in baselines:
        sub = scan.baseline[scan.baseline['optimizer'] == opt]
        ax.scatter(sub[quantity], sub[metric], c=cmap[opt], alpha=0.5, s=30,
                   marker='s', label=opt)
    ax.set_xlabel('Total training time (s)')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_yscale('log')
    # Shampoo/KFAC run 1-2 orders of magnitude slower than SGD on some scans,
    # so a linear time axis collapses everything else into one stripe.
    ax.set_xscale('log')


# ---------------------------------------------------------------------------
# Singular values -- the three plots kept from the old SV sections
# ---------------------------------------------------------------------------
def plot_sv_rank(scan, ax, lr, rtols=None, k=None, normalize=True, **kw):
    """Rank (nonzero SVs) vs train batch, one line per rtol, at ``k = B``.

    With the cap at the batch size the count is not clipped, so it reads as the
    rtol-rank of the batch Jacobian.  See :func:`sv_diagnostics.plot_rank_vs_batch`.
    """
    handles = sv.plot_rank_vs_batch(
        scan.sven, ax, k=k if k is not None else scan.B, lr=lr,
        rtols=rtols if rtols is not None else scan.rtols,
        normalize_by=scan.B if normalize else None,
        results_root=scan.results_root, **kw)
    if normalize:
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0],
                      labels=['$0$', '$B/4$', '$B/2$', '$3B/4$', '$B$'])
    ax.set_title(f'{scan.title}: rank ($k = B = {scan.B}$, $\\eta = {fmt(lr)}$)')
    return handles


def plot_sv_used(scan, ax, lr, rtol, ks=None, cmap='viridis', **kw):
    """Used SVs vs train batch, one line per ``k``, showing saturation at the cap.

    See :func:`sv_diagnostics.plot_used_vs_batch`.
    """
    import matplotlib.pyplot as plt

    ks = ks if ks is not None else scan.ks
    colors = plt.get_cmap(cmap)(np.linspace(0, 0.9, len(ks)))
    handles = sv.plot_used_vs_batch(scan.sven, ax, lr=lr, rtol=rtol, ks=ks,
                                    colors=colors, results_root=scan.results_root,
                                    **kw)
    ax.set_title(f'{scan.title}: SVs used '
                 f'($\\eta = {fmt(lr)}$, rtol $= {fmt(rtol)}$)')
    return handles


def plot_sv_spectra(scan, fig, ax, lr, rtol, k=None, floor=1e-4, colorbar=True, **kw):
    """Per-epoch spectrum shapes ``sigma_i / sigma_0``, coloured by epoch.

    Averaged over the batches in each epoch (binned via ``svs_step``, since
    spectra are subsampled) and over seeds.  See
    :func:`sv_diagnostics.plot_epoch_spectra`.
    """
    spectra, norm = sv.plot_epoch_spectra(
        scan.sven, ax, k=k if k is not None else scan.B, lr=lr, rtol=rtol,
        floor=floor, results_root=scan.results_root, **kw)
    if spectra is None:
        print(f'  no spectra for k={k or scan.B}, lr={lr}, rtol={rtol}')
        return None
    ax.set_ylim(floor, 2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0],
                  labels=['$1$', '$k/4$', '$k/2$', '$3k/4$', '$k$'])
    ax.set_title(f'{scan.title}: SV spectra '
                 f'($\\eta = {fmt(lr)}$, rtol $= {fmt(rtol)}$)')
    if colorbar:
        sv.epoch_colorbar(fig, ax, norm)
    return spectra


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summary_table(scan, metric=None, baselines=None):
    """Best configuration per method, as a DataFrame (seed mean +/- std)."""
    metric = metric or scan.metric
    baselines = baselines if baselines is not None else scan.baselines
    rows = []

    def add(method, cfg, runs):
        if runs.empty:
            return
        entry = {
            'method': method,
            'config': ', '.join(f'{k}={v:g}' for k, v in cfg.items()
                                if k != 'optimizer' and pd.notna(v)),
            'n_seeds': len(runs),
            metric: float(np.nanmean(runs[metric])),
            f'{metric}_std': float(np.nanstd(runs[metric], ddof=1)) if len(runs) > 1 else 0.0,
            'final_train_loss': float(np.nanmean(runs['final_train_loss'])),
            'total_time_s': float(np.nanmean(runs['total_time'])),
        }
        if scan.has_acc:
            entry['final_val_acc'] = float(np.nanmean(runs['final_val_acc']))
        rows.append(entry)

    cfg = scan.best_sven(metric)
    if cfg is not None:
        add('Sven', cfg, scan.sven_rows(**cfg))
    for opt in baselines:
        cfg = scan.best_baseline(opt, metric)
        if cfg is not None:
            add(opt, cfg, scan.baseline_rows(cfg))
    return pd.DataFrame(rows).sort_values(metric).reset_index(drop=True)
