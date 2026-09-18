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
from style import (clipped_band, clipped_yerr, config_eligible, final_value, is_diverged,
                   load_results, lr_labels, method_color, metric_label)

RESULTS_ROOT = '../experiment_results'

# seaborn's "deep" palette, inlined so this module needs only numpy/pandas/
# matplotlib (the notebooks may still import seaborn for their own tweaks).
DEEP = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
        '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']

# Hyperparameters that jointly identify one configuration, per optimizer family.
SVEN_CONFIG = ['k', 'lr', 'rtol']
BASELINE_CONFIG = ['optimizer', 'lr', 'weight_decay', 'lbfgs_max_iter', 'lbfgs_history_size',
                   'tau', 'aggregator', 'inner_optimizer']  # AdamW/Muon wd; HIG tau; JD aggregator/inner


def method_colors(baselines, colors=None, sven=None):
    """``method -> colour`` map for ``'Sven'`` plus ``baselines``.

    Colours come from the global, name-keyed convention in :data:`style.METHOD_COLORS`
    (Sven black), so a method has the same colour in every plot of every notebook --
    independent of which baselines are present or of their order.  ``colors`` may be
    an explicit ``{method: colour}`` dict to override individual entries.
    """
    out = {m: method_color(m) for m in ['Sven', *baselines]}
    if sven is not None:
        out['Sven'] = sven
    if isinstance(colors, dict):
        out.update(colors)
    return out


# ---------------------------------------------------------------------------
# Curve helpers
# ---------------------------------------------------------------------------
def final(curve):
    """Final value of an epoch curve; NaN if the run diverged (:func:`style.final_value`).

    DIVERGED = FAILED: a curve that ends non-finite has no final value.  It used to be
    scored by its last finite value, which flattered LBFGS configs that blow up late --
    and disagreed with :mod:`analysis_helpers`, so the same scan ranked differently in
    the scan and the study notebooks.
    """
    return final_value(curve)


def curve_of(row, which='val'):
    """A per-epoch curve from the light record (``train``/``val``/``*_acc``/
    ``epoch_times``), as a float array, or None."""
    losses = row.get('losses')
    if not isinstance(losses, dict) or losses.get(which) is None:
        return None
    return np.asarray(losses[which], dtype=float)


def _seed_stack(rows, which):
    """Seed curves of one configuration, stacked (truncated to the shortest).

    Diverged runs are left out -- of the loss curves AND of the time series, so a
    curve and its time axis are always averaged over the same seeds.
    """
    curves = []
    for _, r in rows.iterrows():
        if r.get('diverged', False):
            continue
        c = curve_of(r, which)
        if c is not None:
            curves.append(c)
    if not curves:
        return None
    n = min(len(c) for c in curves)
    return np.stack([c[:n] for c in curves])


def seed_curve(rows, which='val'):
    """``(mean, std)`` of a per-epoch curve over the (non-diverged) seeds of one
    configuration; ``std`` is ddof=1.  ``(None, None)`` if nothing is available.
    For plotting a band use :func:`seed_band`."""
    stacked = _seed_stack(rows, which)
    if stacked is None:
        return None, None
    ddof = 1 if len(stacked) > 1 else 0
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=ddof)


def seed_band(rows, which='val'):
    """``(mean, lower, upper)`` -- THE seed band of every plot: arithmetic mean, +/- 1 std
    with the lower edge clipped at the lowest seed (:func:`style.clipped_band`), so it
    cannot reach <= 0 on a log axis.  ``(None, None, None)`` if nothing is available."""
    stacked = _seed_stack(rows, which)
    if stacked is None:
        return None, None, None
    mean, std = seed_curve(rows, which)
    lower, upper = clipped_band(mean, std, stacked.min(axis=0))
    return mean, lower, upper


def epoch_axis(rows, curve, which='val', versus='epoch', time_key='epoch_times'):
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
        times, _ = seed_curve(rows, time_key)
        if times is None:
            # No timings (e.g. no standalone run of this config).  Falling back to
            # the epoch index here would plot epochs as seconds; let the caller skip.
            return None
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
        # diverged = failed (style.is_diverged): left out of every seed mean, counted per config
        d['diverged'] = d.apply(lambda r: is_diverged(curve_of(r, 'train'), curve_of(r, 'val')), axis=1)
        d.loc[d['diverged'], ['final_val_loss', 'final_train_loss', 'final_val_acc']] = np.nan
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
    def n_params(self):
        """Parameter count P, if the records carry it (runs made after 2026-09-17 do:
        ``generic_scan._scan_facts``); else None."""
        if 'n_params' not in self.df.columns or self.df['n_params'].isna().all():
            return None
        return int(self.df['n_params'].dropna().iloc[0])

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
        """Per-configuration table: seed mean/std/min/max of ``metric``.

        This is what replaces picking a single run: a configuration's score is the
        mean over its seeds, so `best_*` never latches onto a lucky seed.

        ``n_seeds`` counts the seeds that finished, ``n_diverged`` those that did not
        (diverged = failed: they are not in the mean).  Because dropping a config's
        failures from its mean flatters it, configs are ranked by **fewest diverged
        seeds first, then seed mean**: a config that blows up on 2 of 5 seeds never
        beats one that finishes all 5.  ``eligible`` is False when no more than half of
        the scan's seeds finished; such a config is listed last and is never returned
        by `best_*`.
        """
        metric = metric or self.metric
        keys = [k for k in keys if k in df.columns]
        ok = df[~df['diverged']].dropna(subset=[metric])
        out = (ok.groupby(keys, dropna=False)[metric]
               .agg(score='mean', score_std='std', score_min='min', score_max='max',
                    n_seeds='size'))
        n_all = df.groupby(keys, dropna=False).size().rename('n_runs')
        out = out.join(n_all, how='outer').reset_index()
        out['n_seeds'] = out['n_seeds'].fillna(0).astype(int)
        out['n_diverged'] = out.pop('n_runs') - out['n_seeds']
        out['n_missing'] = len(self.seeds) - out['n_seeds'] - out['n_diverged']  # no result file
        out['eligible'] = config_eligible(out['n_seeds'], len(self.seeds))
        # Exact ties are common, not a corner case: whenever the rtol-rank stays
        # below k, every larger k (and every rtol that cuts nothing extra) is the
        # *same trajectory*.  Break them deterministically -- smallest k, then
        # largest rtol, i.e. the cheapest equivalent config -- so `best_*` names the
        # same config on every machine.  An unstable sort on `score` alone is how
        # the standalone timing runs (selected on the cluster) came to be for a
        # tied twin of the config the notebooks pick, leaving Sven without a bar.
        order = ['eligible', 'n_diverged', 'score'] + [k for k in ('k', 'rtol', 'lr') if k in keys]
        ascending = [c not in ('rtol', 'eligible') for c in order]
        return out.sort_values(order, ascending=ascending, kind='mergesort',
                               na_position='last')

    def best_sven(self, metric=None, **fixed):
        """The best Sven configuration (seed-mean ``metric``) as a dict of hparams.

        ``fixed`` pins any of ``k``/``lr``/``rtol``; the rest are optimised over.
        """
        sel = self.runs(**fixed)
        cfg = self.configs(sel, SVEN_CONFIG, metric)
        cfg = cfg[cfg['eligible']]
        if cfg.empty:
            return None
        return {k: cfg.iloc[0][k] for k in SVEN_CONFIG if k in cfg.columns}

    def best_baseline(self, optimizer, metric=None):
        """The best configuration for one baseline optimizer, as a dict of hparams."""
        sel = self.baseline[self.baseline['optimizer'] == optimizer]
        cfg = self.configs(sel, BASELINE_CONFIG, metric)
        cfg = cfg[cfg['eligible']]
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


def load_scan(name, title, plot_dir, results_root=RESULTS_ROOT, metric='final_val_loss'):
    """Load one scan directory light and wrap it in a :class:`Scan`.

    Diverged runs (non-finite final loss) stay in the table, flagged ``diverged``:
    they are left out of every seed mean / curve and counted per configuration in
    ``n_diverged`` (see :meth:`Scan.configs`).
    """
    df = load_results(name, results_root=results_root)
    scan = Scan(name=name, title=title, plot_dir=plot_dir, df=df, metric=metric,
                results_root=results_root)
    print(f'{scan.name}: {len(scan.df)} runs  |  B={scan.B}, {scan.n_epochs} epochs, '
          f'{len(scan.seeds)} seeds  |  Sven k={scan.ks} rtol={scan.rtols}')
    bad = scan.df[scan.df['diverged']]
    if len(bad):
        print(f'  {len(bad)} diverged run(s), excluded from seed means: '
              f'{bad["optimizer"].value_counts().to_dict()}')
    return scan


# ---------------------------------------------------------------------------
# Loss / convergence
# ---------------------------------------------------------------------------
def has_standalone(scan):
    """Whether :func:`attach_standalone_times` found standalone timing runs."""
    return 'standalone_total_time' in scan.df.columns and scan.df['standalone_total_time'].notna().any()


def resolve_time_key(scan, time_key='auto'):
    """``'auto'`` -> the standalone epoch times when attached, else the scan's own."""
    if time_key == 'auto':
        return 'epoch_times_standalone' if has_standalone(scan) else 'epoch_times'
    return time_key


def time_axis_label(time_key):
    return ('Wall time (s), standalone' if time_key == 'epoch_times_standalone'
            else 'Wall time (s), sharded scan')


def plot_best_curves(scan, ax, which='train', versus='epoch', baselines=None,
                     colors=None, band=True, sven_kw=None, time_key='auto', **plot_kw):
    """Best Sven vs the best of each baseline, seed-averaged, with a seed band.

    ``which``   -- ``'train'``, ``'val'``, ``'val_acc'``, ...
    ``versus``  -- ``'epoch'`` or ``'time'`` (cumulative epoch wall time).
    ``time_key`` -- which per-epoch time series to use for ``versus='time'``:
                 ``'auto'`` (default: standalone if :func:`attach_standalone_times`
                 found any, else the scan's own), ``'epoch_times'`` (the scan's
                 own, shard-inflated) or ``'epoch_times_standalone'``.  The x label
                 says which.

    Returns the dict of configurations plotted, keyed by label.
    """
    baselines = baselines if baselines is not None else scan.baselines
    cmap = method_colors(baselines, colors)
    time_key = resolve_time_key(scan, time_key)
    chosen = {}

    def draw(rows, label, color, lw, zorder):
        mean, lower, upper = seed_band(rows, which)
        if mean is None:
            return
        x = epoch_axis(rows, mean, which, versus, time_key)
        if x is None:
            print(f'  [plot_best_curves] no {time_key!r} for {label} -- not drawn')
            return
        n = len(x)
        ax.plot(x, mean[:n], color=color, lw=lw, label=label, zorder=zorder, **plot_kw)
        if band and len(rows) > 1:
            ax.fill_between(x, lower[:n], upper[:n], color=color, alpha=0.2,
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

    ax.set_xlabel(time_axis_label(time_key) if versus == 'time' else 'Epoch')
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
        mean, lower, upper = seed_band(rows, which)
        if mean is None:
            continue
        x = epoch_axis(rows, mean, which)
        n = len(x)
        ax.plot(x, mean[:n], color=colors[i], lw=3, label=f'$k={int(k)}$', **plot_kw)
        if band and len(rows) > 1:
            ax.fill_between(x, lower[:n], upper[:n], color=colors[i], alpha=0.2, lw=0)
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
    if colors is None:  # by method name, never by position
        colors = [method_color(cfg.get('optimizer', 'Sven')) for cfg in configs]
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
    cfg.loc[~cfg['eligible'], 'score'] = np.nan   # mostly-diverged cells are left blank
    vmin, vmax = np.log10(cfg['score'].min()), np.log10(cfg['score'].max())
    im = None
    for ax, rtol in zip(np.atleast_1d(axes), scan.rtols):
        pivot = (cfg[cfg['rtol'] == rtol]
                 .pivot_table(values='score', index='k', columns='lr', dropna=False))
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


def plot_sensitivity(scan, ax, x='lr', lines='k', metric=None, third='best', band=True,
                     **plot_kw):
    """Seed-mean ``metric`` vs one Sven axis, one line per value of another.

    ``x``/``lines`` are any two of ``'k'``, ``'lr'``, ``'rtol'``.  Every point is
    one configuration: the mean over its seeds, with a +/- 1 std seed band (the
    lower edge is clipped at the best seed, so it survives the log axis).

    ``third`` says what happens to the remaining axis:

    * ``'best'`` (default) -- pinned at its value in the best Sven config, so the
      plot is a plain slice through the optimum;
    * a number -- pinned at that value;
    * ``'min'`` -- the old behaviour: per point, the config with the lowest seed
      mean over the third axis (a best-case envelope, so neighbouring points can
      come from different configs).

    Returns the pinned value (None for ``'min'``).
    """
    metric = metric or scan.metric
    cfg = scan.configs(scan.sven, SVEN_CONFIG, metric)
    cfg = cfg[cfg['eligible']]
    other = next(a for a in SVEN_CONFIG if a not in (x, lines))
    pinned = None
    if third != 'min':
        pinned = scan.best_sven(metric)[other] if third == 'best' else third
        cfg = cfg[cfg[other] == pinned]
    import matplotlib.pyplot as plt

    vals = sorted(cfg[lines].unique())
    # k is coloured viridis (dark -> light with k) everywhere it is a line family
    colors = (plt.get_cmap('viridis')(np.linspace(0, 0.9, len(vals))) if lines == 'k'
              else [f'C{i}' for i in range(len(vals))])
    for i, val in enumerate(vals):
        sub = cfg[cfg[lines] == val]
        sub = sub.loc[sub.groupby(x)['score'].idxmin()].sort_values(x)
        label = (f'$k={int(val)}$' if lines == 'k' else
                 f'${lines}={fmt(val)}$')
        ax.plot(sub[x], sub['score'], 'o-', color=colors[i], label=label, **plot_kw)
        if band and (sub['n_seeds'] > 1).any():
            lower, upper = clipped_band(sub['score'], sub['score_std'], sub['score_min'])
            ax.fill_between(sub[x], lower, upper, color=colors[i], alpha=0.15, lw=0)
    ax.set_xlabel({'lr': 'Learning rate', 'k': '$k$', 'rtol': 'rtol'}[x])
    ax.set_ylabel(metric_label(metric))
    if x in ('lr', 'rtol'):
        ax.set_xscale('log')
    ax.set_yscale('log')
    return pinned


# ---------------------------------------------------------------------------
# Knob sweeps (microbatch size, parameter fraction): Sven-only scans that vary one
# extra column on top of (k, lr, rtol)
# ---------------------------------------------------------------------------
KNOB_LABELS = {'microbatch_size': lambda v: f'$\\mu B = {int(v)}$',
               'param_fraction': lambda v: f'$f = {v:g}$'}
KNOB_AXIS = {'microbatch_size': 'Microbatch size $\\mu B$',
             'param_fraction': 'Parameter fraction $f$'}


def knob_values(scan, knob):
    return sorted(float(v) for v in scan.sven[knob].dropna().unique())


def knob_ref_lr(scan, knob, ref, metric=None):
    """The learning rate the knob sweep is drawn at: the best (seed-mean ``metric``)
    lr at the reference value ``ref`` of the knob (mb = 1 / f = 1, i.e. plain Sven)."""
    cfg = scan.best_sven(metric, **{knob: ref})
    if cfg is None:
        raise ValueError(f'no eligible run at {knob} = {ref}')
    return float(cfg['lr'])


def plot_knob_curves(scan, ax, knob, lr, which='val', values=None, cmap='viridis',
                     band=True, **fixed):
    """Seed-mean loss curves (+ clipped seed band) vs epoch, one line per knob value,
    at a fixed ``lr`` (and any other ``fixed`` hparams).  Returns the values drawn."""
    import matplotlib.pyplot as plt

    values = values if values is not None else knob_values(scan, knob)
    colors = plt.get_cmap(cmap)(np.linspace(0, 0.9, len(values)))
    drawn = []
    for i, v in enumerate(values):
        rows = scan.runs(lr=lr, **{knob: v}, **fixed)
        mean, lower, upper = seed_band(rows, which)
        if mean is None:
            continue
        x = epoch_axis(rows, mean, which)
        n = len(x)
        ax.plot(x, mean[:n], color=colors[i], lw=3, label=KNOB_LABELS[knob](v))
        if band and len(rows) > 1:
            ax.fill_between(x, lower[:n], upper[:n], color=colors[i], alpha=0.2, lw=0)
        drawn.append(v)
    ax.set_xlabel('Epoch')
    ax.set_ylabel({'train': 'Train loss', 'val': 'Validation loss'}.get(which, which))
    ax.set_yscale('log')
    return drawn


def knob_table(scan, knob, metric=None):
    """Per (knob value, lr) configuration table -- :meth:`Scan.configs` keyed by the
    knob and ``lr`` (k and rtol are single-valued in these scans)."""
    return scan.configs(scan.sven, [knob, 'lr'], metric)


def plot_knob_summary(scan, ax, knob, lr, metric=None, quantity=None, best_lr=True):
    """Seed-mean ``metric`` (or another run ``quantity``, e.g. ``total_time``) vs the
    knob, with the clipped seed error bar.  Solid: at the fixed ``lr``.  Dashed
    (``best_lr``): the best lr per knob value, annotated with that lr."""
    metric = metric or scan.metric
    q = quantity or metric
    cfg = knob_table(scan, knob, q)
    at = cfg[(cfg['lr'] == lr) & cfg['eligible']].sort_values(knob)
    lower, upper = clipped_band(at['score'], at['score_std'], at['score_min'])
    ax.errorbar(at[knob], at['score'], yerr=[at['score'] - lower, upper - at['score']],
                fmt='o-', color='k', lw=2.5, capsize=3, label=f'$\\eta = {fmt(lr)}$')
    if best_lr and quantity is None:
        b = cfg[cfg['eligible']].groupby(knob, sort=True).head(1).sort_values(knob)
        lower, upper = clipped_band(b['score'], b['score_std'], b['score_min'])
        ax.errorbar(b[knob], b['score'], yerr=[b['score'] - lower, upper - b['score']],
                    fmt='s--', color='0.5', lw=1.5, capsize=3, label='best $\\eta$ per value')
        for _, r in b.iterrows():
            ax.annotate(f'${fmt(r["lr"])}$', (r[knob], r['score']), fontsize=9,
                        color='0.4', textcoords='offset points', xytext=(4, 4))
    ax.set_xlabel(KNOB_AXIS[knob])
    ax.set_ylabel('Total training time (s), sharded scan (seed mean)' if q == 'total_time'
                  else metric_label(q))
    if knob == 'microbatch_size':
        ax.set_xscale('log', base=2)
    if q != 'total_time':
        ax.set_yscale('log')
    return cfg


# ---------------------------------------------------------------------------
# Wall time
# ---------------------------------------------------------------------------
def time_excl_first_epoch(row):
    """Total wall time with the first epoch removed (drops compile/warm-up)."""
    et = curve_of(row, 'epoch_times')
    return np.nan if et is None or len(et) < 2 else float(np.sum(et[1:]))


def plot_time_summary(scan, ax, quantity='total_time', baselines=None, source='auto', **bar_kw):
    """Bar chart of a timing / memory quantity for the best config of each method,
    seed mean with the clipped seed error bar.

    ``quantity`` is one of :data:`STANDALONE_QUANTITIES`, ``'time_excl_first_epoch'``,
    or any other column on the run table.  ``source``: ``'auto'`` (standalone values
    when :func:`attach_standalone_times` found them, else the scan's own),
    ``'standalone'`` or ``'scan'``.  The y label says which; the scan's own wall
    times are inflated by shard contention.  Returns ``({method: mean}, bars, source)``.
    """
    baselines = baselines if baselines is not None else scan.baselines
    cmap = method_colors(baselines)
    if source == 'auto':
        source = 'standalone' if has_standalone(scan) else 'scan'
    col = f'standalone_{quantity}' if source == 'standalone' else quantity
    if source == 'standalone' and col not in scan.df.columns:
        raise ValueError(f'{col} missing -- call attach_standalone_times(scan) first')

    def values(rows):
        rows = rows[~rows['diverged']]
        if col == 'time_excl_first_epoch':
            v = np.array([time_excl_first_epoch(r) for _, r in rows.iterrows()], dtype=float)
        else:
            v = rows[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return np.nan, 0.0, np.nan
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), v.min()

    labels, stats = [], []
    cfg = scan.best_sven()
    if cfg is not None:
        labels.append('Sven')
        stats.append(values(scan.sven_rows(**cfg)))
    for opt in baselines:
        cfg = scan.best_baseline(opt)
        if cfg is None:
            continue
        labels.append(opt)
        stats.append(values(scan.baseline_rows(cfg)))
    for m, (mean, _, _) in zip(labels, stats):
        if not np.isfinite(mean):
            print(f'  [plot_time_summary] no {source} {quantity} for the best {m} config')

    means = np.array([st[0] for st in stats])
    x = np.arange(len(labels))
    yerr = clipped_yerr(means, [st[1] for st in stats], [st[2] for st in stats])
    bars = ax.bar(x, means, yerr=np.nan_to_num(yerr), capsize=3,
                  color=[cmap[m] for m in labels], **bar_kw)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(quantity.replace('_', ' ') + (' (standalone)' if source == 'standalone'
                                                else ' (sharded scan)'))
    return dict(zip(labels, means)), bars, source


STANDALONE_QUANTITIES = ('total_time', 'avg_epoch_time', 'avg_batch_time_train',
                         'avg_batch_time_val', 'peak_gpu_mem_mb')


def attach_standalone_times(scan, name=None, results_root=None, verbose=True):
    """Join standalone (one-run-per-GPU) timings onto the scan's rows, by ``run_id``.

    The scans run several trainings per GPU (``NPROC`` shards), which inflates
    every wall time by contention.  ``submit_timing_runs.sh`` re-runs the best
    (optimizer, setting) of each method alone on a GPU, with the same seeds, into
    ``experiment_results/{scan}_timing/`` -- same run_ids, honest times.  This
    adds, on ``scan.df`` / ``scan.sven`` / ``scan.baseline``:

    * ``standalone_<q>`` for each ``q`` in :data:`STANDALONE_QUANTITIES`, plus
      ``standalone_time_excl_first_epoch`` (NaN where no standalone run exists);
    * ``losses['epoch_times_standalone']`` on each matched row, so
      :func:`plot_best_curves` can draw loss-vs-time with ``time_key=
      'epoch_times_standalone'``.

    It also checks that the standalone runs reproduced the scan's trajectories
    (same seeds => same init and data order; GPU kernels are not bit-exact) and
    returns a small report dict: matched rows, and the max / median relative
    deviation of the final validation loss.  Returns None if the timing directory
    does not exist yet.
    """
    name = name or f'{scan.name}_timing'
    root = results_root or scan.results_root
    try:
        t = load_results(name, results_root=root)
    except FileNotFoundError:
        if verbose:
            print(f'[standalone] no results yet in {root}/{name}/')
        return None
    t = t.drop_duplicates('run_id').set_index('run_id')
    tl = t['losses']
    cols = {f'standalone_{q}': tl.apply(lambda L, q=q: L.get(q, np.nan)) for q in STANDALONE_QUANTITIES}
    cols['standalone_time_excl_first_epoch'] = t.apply(time_excl_first_epoch, axis=1)
    fin_stand = t.apply(lambda r: final(curve_of(r, 'val')), axis=1)
    et_stand = tl.apply(lambda L: L.get('epoch_times'))

    devs = []
    for df in (scan.df, scan.sven, scan.baseline):
        for c, series in cols.items():
            df[c] = df['run_id'].map(series)
        for i, r in df.iterrows():
            et = et_stand.get(r['run_id'])
            if et is not None and isinstance(r['losses'], dict):
                r['losses']['epoch_times_standalone'] = et  # dict is shared across the three frames
    matched = scan.df['run_id'].isin(t.index)
    for _, r in scan.df[matched].iterrows():
        a, b = final(curve_of(r, 'val')), fin_stand.get(r['run_id'])
        if np.isfinite(a) and np.isfinite(b):
            devs.append(abs(a - b) / max(abs(a), 1e-30))
    report = {'name': name, 'n_standalone': int(len(t)), 'n_matched': int(matched.sum()),
              'max_rel_dev_final_val': float(np.max(devs)) if devs else np.nan,
              'median_rel_dev_final_val': float(np.median(devs)) if devs else np.nan,
              'unmatched_standalone': sorted(set(t.index) - set(scan.df['run_id']))}
    if verbose:
        print(f"[standalone] {report['n_standalone']} runs in {name}/, {report['n_matched']} joined; "
              f"final-val-loss deviation vs scan: max {report['max_rel_dev_final_val']:.2e}, "
              f"median {report['median_rel_dev_final_val']:.2e}")
        if report['unmatched_standalone']:
            print(f"[standalone] WARNING {len(report['unmatched_standalone'])} standalone run_ids not in the scan")
    return report


def plot_time_comparison(scan, ax, quantity='total_time', baselines=None, width=0.38,
                         labels=('sharded scan', 'standalone'), **bar_kw):
    """Grouped bars: the scan's (shard-inflated) timing vs the standalone timing,
    for the best config of each method.  Requires :func:`attach_standalone_times`.

    ``quantity`` is one of :data:`STANDALONE_QUANTITIES` or
    ``'time_excl_first_epoch'``.  Methods without a standalone run get only the
    scan bar.  Returns ``{method: (scan_value, standalone_value)}``.
    """
    baselines = baselines if baselines is not None else scan.baselines
    cmap = method_colors(baselines)
    scan_col = quantity if quantity in scan.df.columns else None
    stand_col = f'standalone_{quantity}'
    if stand_col not in scan.df.columns:
        raise ValueError(f'{stand_col} missing -- call attach_standalone_times(scan) first')

    def value(rows, col):
        if col is None and quantity == 'time_excl_first_epoch':
            return float(np.nanmean([time_excl_first_epoch(r) for _, r in rows.iterrows()]))
        if col not in rows or not rows[col].notna().any():
            return np.nan
        return float(np.nanmean(rows[col]))

    out = {}
    cfg = scan.best_sven()
    if cfg is not None:
        rows = scan.sven_rows(**cfg)
        out['Sven'] = (value(rows, scan_col), value(rows, stand_col))
    for opt in baselines:
        cfg = scan.best_baseline(opt)
        if cfg is None:
            continue
        rows = scan.baseline_rows(cfg)
        out[opt] = (value(rows, scan_col), value(rows, stand_col))

    for m, (_, stand) in out.items():
        if not np.isfinite(stand):
            print(f'  [plot_time_comparison] no standalone run for the best {m} config')
    methods = list(out)
    x = np.arange(len(methods))
    ax.bar(x - width / 2, [out[m][0] for m in methods], width, color=[cmap[m] for m in methods],
           alpha=0.35, hatch='//', label=labels[0], **bar_kw)
    ax.bar(x + width / 2, [out[m][1] for m in methods], width, color=[cmap[m] for m in methods],
           label=labels[1], **bar_kw)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel(quantity.replace('_', ' '))
    ax.legend()
    return out


def plot_time_vs_k(scan, ax, quantity='total_time', rtol='best', lrs=None, **plot_kw):
    """Seed-mean wall time vs ``k`` at ONE rtol (default: the best Sven config's),
    one line per learning rate, with the clipped seed error bar.

    Uses the scan's own (shard-inflated) times: standalone timings exist only for
    the best configuration, not for a k sweep -- see RERUNS_NEEDED.md.
    """
    if rtol == 'best':
        rtol = scan.best_sven()['rtol']
    lrs = lrs if lrs is not None else scan.sven_lrs
    for i, lr in enumerate(lrs):
        cfg = scan.configs(scan.runs(lr=lr, rtol=rtol), ['k'], quantity)
        cfg = cfg[cfg['eligible']].sort_values('k')
        if cfg.empty:
            continue
        ax.errorbar(cfg['k'], cfg['score'],
                    yerr=clipped_yerr(cfg['score'], cfg['score_std'], cfg['score_min']),
                    fmt='o-', capsize=3, color=f'C{i}', label=f'$\\eta={fmt(lr)}$', **plot_kw)
    ax.set_xlabel('$k$')
    ax.set_ylabel('Total training time (s), sharded scan')
    return rtol


def plot_efficiency(scan, ax, metric=None, quantity='total_time', baselines=None):
    """Scatter of ``metric`` vs wall time over every (non-diverged) run -- Sven against
    baselines.  Every run means the scan's own shard-inflated times (standalone
    timings exist only for the best configs); the axis label says so."""
    metric = metric or scan.metric
    baselines = baselines if baselines is not None else scan.baselines
    cmap = method_colors(baselines)
    sven = scan.sven[~scan.sven['diverged']]
    ax.scatter(sven[quantity], sven[metric], c=cmap['Sven'], alpha=0.6, s=40, label='Sven')
    for opt in baselines:
        sub = scan.baseline[(scan.baseline['optimizer'] == opt) & ~scan.baseline['diverged']]
        ax.scatter(sub[quantity], sub[metric], c=cmap[opt], alpha=0.5, s=30,
                   marker='s', label=opt)
    ax.set_xlabel('Total training time (s), sharded scan')
    ax.set_ylabel(metric_label(metric, seed_mean=False))
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

    Drawn for the ``k = B`` runs (unless ``k`` is given) at the given ``lr``/
    ``rtol``, averaged over the batches in each epoch (binned via ``svs_step``,
    since spectra are subsampled) and over seeds.  The x-axis is the SV index as
    a fraction of ``k``; the dotted line is ``rtol``.  Only the SVs that survive
    the rtol cut are ever recorded, so every curve ends at that line by
    construction -- see :func:`sv_diagnostics.plot_epoch_spectra`.
    """
    k = k if k is not None else scan.B
    spectra, norm = sv.plot_epoch_spectra(
        scan.sven, ax, k=k, lr=lr, rtol=rtol,
        floor=floor, results_root=scan.results_root, **kw)
    if spectra is None:
        print(f'  no spectra for k={k}, lr={lr}, rtol={rtol}')
        return None
    ax.set_ylim(floor, 2)
    ax.set_xlim(-0.03, 1.03)
    cap = 'B' if k == scan.B else 'k'
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0],
                  labels=['$1$', f'${cap}/4$', f'${cap}/2$', f'$3{cap}/4$', f'${cap}$'])
    ax.set_xlabel('SV rank')
    k_label = f'k = B = {int(k)}' if cap == 'B' else f'k = {int(k)}'
    ax.set_title(f'{scan.title}: SV spectra (${k_label}$, '
                 f'$\\eta = {fmt(lr)}$, rtol $= {fmt(rtol)}$)')
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
        ok = runs[~runs['diverged']]          # diverged = failed: not in the means
        entry = {
            'method': method,
            # string-valued hparams (JD aggregator / inner_optimizer) have no ':g'
            'config': ', '.join(f'{k}={v}' if isinstance(v, str) else f'{k}={v:g}'
                                for k, v in cfg.items()
                                if k != 'optimizer' and pd.notna(v)
                                and not (k == 'weight_decay' and v == 0)),   # wd=0 is the norm
            'n_seeds': len(ok),
            'n_diverged': len(runs) - len(ok),
            'n_missing': len(scan.seeds) - len(runs),
            metric: float(ok[metric].mean()),
            f'{metric}_std': float(ok[metric].std(ddof=1)) if len(ok) > 1 else 0.0,
            f'{metric}_min': float(ok[metric].min()),   # for the clipped error bar
            # the other loss, so a table selected on val also shows train (and vice versa)
            ('final_train_loss' if metric != 'final_train_loss' else 'final_val_loss'):
                float(ok['final_train_loss' if metric != 'final_train_loss' else 'final_val_loss'].mean()),
            'sharded_time_s': float(ok['total_time'].mean()),   # shard-inflated scan time
        }
        if 'standalone_total_time' in ok and ok['standalone_total_time'].notna().any():
            entry['standalone_time_s'] = float(ok['standalone_total_time'].mean())
        if scan.has_acc:
            entry['final_val_acc'] = float(ok['final_val_acc'].mean())
        rows.append(entry)

    cfg = scan.best_sven(metric)
    if cfg is not None:
        add('Sven', cfg, scan.sven_rows(**cfg))
    for opt in baselines:
        cfg = scan.best_baseline(opt, metric)
        if cfg is not None:
            add(opt, cfg, scan.baseline_rows(cfg))
    return pd.DataFrame(rows).sort_values(metric).reset_index(drop=True)
