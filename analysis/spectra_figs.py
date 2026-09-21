"""Mechanism figures for Sven: what the truncation keeps and what it throws away.

Two measurements answer that question and they are **different objects** -- the
single most important thing to keep straight when reading anything in this module:

* **online, per batch** (``{scan}_diag/diag/*.npz``, logged by the optimizer,
  :data:`ONLINE`).  On each logged step Sven records all ``M = B/microbatch``
  singular values of THAT step's batch Jacobian before the ``k`` / ``rtol`` cut,
  together with ``utr = U^T r`` (C-L1).  The batch changes every step, so the
  spectrum is a different matrix's spectrum every time, and the width is capped
  at ``B``: a "fraction of the residual in the top ``k``" computed here is a
  fraction of *this batch's* residual, in *this batch's* ``B``-dimensional row
  space.
* **offline, on a fixed probe set** (``analysis/ckpt_spectra/``, written by
  ``tools/compute_ckpt_spectra.py``, read with ``ckpt_tools.load_spectra``,
  :data:`PROBE`).  A float64 Jacobian of the SAME rows for every optimizer,
  every seed and every checkpoint, so trajectories are comparable and the rank
  axis runs to ``min(n_rows, P)`` rather than to ``B``.

They disagree, and both are right.  On the polynomial scan the top ``k = 16``
directions hold **99.9 % of the batch residual** but only **58.8 % of the
probe-set residual** (:func:`probe_energy`) -- 32 rows can be fitted by 16
directions in a way 10,000 rows cannot.  Every figure below states which object
it is showing.

**The float32 floor is NOT 1e-7 here.**  Sven takes sigma from a float64 ``eigh``
of ``J J^T``, which squares the condition number, so with float32 parameters the
error on sigma_i is ~``eps * sigma_max^2 / (2 sigma_i)`` and everything below
``sqrt(eps) * sigma_max ~ 3.45e-4 * sigma_max`` is round-off (``sven.opt.sven``
docstring, F19).  The optimizer records that number per logged step as
``sv_noise_floor``; :func:`noise_floor_rel` reads it, and
:data:`style.FLOAT32_NOISE_FLOOR` (1e-7, the floor an ``svdvals(J)`` would give)
is only the fallback for a legacy record that has none.  Phase A measured the
consequence directly: online and offline singular values agree to ~5e-7
*relative* above ``1e-2 * sigma_max`` and not at all below it.

Ownership: WP3 phase B.  Shared helpers (``style``, ``analysis_helpers``,
``scan_analysis``, ``headline``, ``paired``, ``budget``, ``ckpt_tools``) are
frozen; everything new lives here.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import ckpt_tools as ct
import headline
import style
import sv_diagnostics as sv

__all__ = [
    'ONLINE', 'PROBE', 'SVEN_OPTIMIZER', 'PROBE_METHODS', 'savefig',
    'SvenDiag', 'sven_diag', 'diag_arrays', 'seed_stack', 'noise_floor_rel',
    'cumulative_energy', 'energy_in_top', 'rtol_rank', 'used_rank',
    'used_rank_grid', 'mechanism_table', 'plot_spectra_over_training',
    'plot_utr_over_training', 'plot_energy_capture', 'plot_energy_profile',
    'plot_rank_used', 'plot_norms', 'probe_scan_name', 'probe_methods',
    'probe_metrics', 'probe_energy', 'effective_rank', 'plot_probe_spectra',
    'plot_probe_metric', 'plot_probe_energy_profile', 'subsample_rows',
    'progress_colors', 'smooth_steps', 'low4_table', 'low4_verdict',
    'probe_widths',
]

#: the two measurements, named so a figure title / caption cannot be ambiguous
ONLINE = 'online per-batch Gram (float32 model, sqrt-eps floor)'
PROBE = 'offline float64 Jacobian on the fixed probe set'

#: Sven's ``optimizer`` value in a record (``style.canonical_method`` -> 'Sven')
SVEN_OPTIMIZER = 'SVD'

#: the trajectories ``tools/compute_ckpt_spectra.py`` caches, in plot order
PROBE_METHODS = ('Sven', 'Adam', 'MuonW', 'HIG')

#: relative singular value below which an offline float64 spectrum is round-off.
#: The probe Jacobian is formed and decomposed in float64, so this is ~1e-13
#: rather than the online Gram's 3.45e-4 -- the whole reason the probe set can
#: see a tail the online record cannot.
PROBE_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def savefig(fig, plot_dir, name, formats=('pdf', 'png'), **kw):
    """Save one figure under ``plot_dir`` as PDF *and* PNG; return the paths.

    ``scan_analysis.Scan.save`` writes only the PDF and needs a ``Scan``; these
    notebooks compare several scans in one figure, so the directory is passed
    directly and the PNG (which is what a review thread or a draft can show
    inline) is written beside it.  ``bbox_inches='tight'`` comes from
    :func:`style.set_style`'s rcParams.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    name = name[:-4] if name.endswith('.pdf') else name
    paths = []
    for ext in formats:
        path = plot_dir / f'{name}.{ext}'
        fig.savefig(path, **kw)
        paths.append(path)
    return paths


def subsample_rows(n, limit):
    """Indices of at most ``limit`` rows of ``n``, log-spaced, always incl. 0 and n-1.

    A dense schedule logs 1,200-1,700 spectra per run; drawing all of them makes an
    unreadable figure and a 20 MB PDF.  Log spacing keeps the early steps -- where
    the spectrum moves fastest -- instead of thinning them away.
    """
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=int)
    if limit is None or n <= limit:
        return np.arange(n)
    idx = np.unique(np.round(np.geomspace(1, n, int(limit))).astype(int) - 1)
    return np.unique(np.concatenate([[0], idx, [n - 1]]))


def smooth_steps(step, y, window):
    """Centred moving average of ``y`` over a window measured in OPTIMIZER STEPS.

    ``sv_diagnostics.smooth`` averages a fixed number of SAMPLES, which is right on a
    uniformly logged series.  The dense spectra schedule is **not** uniform: it logs
    every step up to 1,000 and then every 20th (measured on all seven diag passes --
    toy logs 1,262 points over 6,240 steps with step differences ``{1: 1000, 20: 261}``).
    A 25-sample window is therefore 25 steps early and 500 steps late, so the apparent
    sharpness of a curve changes along it for no physical reason.  This averages over
    ``+/- window/2`` STEPS instead, so every part of a curve is smoothed by the same
    amount of *training*.

    ``step`` must be ascending (``svs_step`` is).  Returns ``(step, smoothed)``, same
    length as the input, so the caller keeps the true step axis -- which is the other
    half of the same problem: index fraction is not training progress on this schedule.
    """
    step = np.asarray(step, dtype=float)
    y = np.asarray(y, dtype=float)
    if not step.size:
        return step, y
    half = float(window) / 2.0
    lo = np.searchsorted(step, step - half, side='left')
    hi = np.searchsorted(step, step + half, side='right')
    good = np.isfinite(y)
    csum = np.concatenate([[0.0], np.cumsum(np.where(good, y, 0.0))])
    ccnt = np.concatenate([[0.0], np.cumsum(good.astype(float))])
    den = ccnt[hi] - ccnt[lo]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        out = np.where(den > 0, (csum[hi] - csum[lo]) / np.where(den > 0, den, 1.0),
                       np.nan)
    return step, out


def progress_colors(values, cmap='plasma', lo=0.15, hi=1.0):
    """Colours for a set of monotone x values (steps), plus the Normalize for a bar.

    The colour encodes TRAINING PROGRESS on a log-step scale, so the first few
    steps -- which the log schedule samples densely -- do not all collapse into
    one colour.  Returns ``(colors, norm)``; pass ``norm`` to
    :func:`sv_diagnostics.epoch_colorbar` with the same ``cmap``/``lo``/``hi``.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=float)
    norm = mcolors.LogNorm(vmin=max(values.min(), 1.0), vmax=max(values.max(), 2.0))
    base = plt.get_cmap(cmap)
    frac = norm(np.clip(values, norm.vmin, norm.vmax))
    return base(lo + (hi - lo) * np.asarray(frac, dtype=float)), norm


# ---------------------------------------------------------------------------
# The Sven runs of a `<scan>_diag` pass
# ---------------------------------------------------------------------------
@dataclass
class SvenDiag:
    """The five Sven seeds of one ``{scan}_diag`` pass and the configuration they ran.

    ``k`` / ``rtol`` / ``lr`` / ``B`` come from the RECORDS, never from
    ``bench/best_configs.json``: the diag pass was launched from a selection that
    may since have moved (CIFAR-CE, plan decision 5).  ``selection_matches`` says
    whether the configuration on disk is still the selected one, and
    ``selection_note`` is the sentence a notebook should print when it is not --
    that is the whole provisional-CIFAR-CE contract in one attribute.
    """
    scan: str
    dir_name: str
    title: str
    rows: pd.DataFrame
    k: int
    rtol: float
    lr: float
    B: int
    n_epochs: int
    n_steps: int
    seeds: list = field(default_factory=list)
    selection_matches: bool = True
    selection_note: str = ''

    @property
    def label(self):
        """``Sven ($k=16$, rtol$=0.03$, $\\eta=0.5$)`` -- what a panel title says."""
        return (f'Sven ($k={self.k}$, rtol$\\,={self.rtol:g}$, $\\eta={self.lr:g}$)')

    @property
    def truncates(self):
        """Whether ``k`` actually cuts the batch spectrum (``k < B``)."""
        return self.k < self.B


def _one(rows, column):
    """The single value ``column`` takes over ``rows`` (raises if it takes two)."""
    vals = pd.unique(pd.to_numeric(rows[column], errors='coerce').dropna())
    if len(vals) != 1:
        raise ValueError(f'expected one {column} in the diag pass, got {sorted(vals)}')
    return vals[0]


def sven_diag(scan, payload=None, results_root=None, verbose=True):
    """:class:`SvenDiag` for one headline scan, from its ``_diag`` pass.

    The diag pass reran the selected configuration of every method with the dense
    spectra schedule, so its Sven rows ARE the best Sven configuration -- but only
    as of the selection the pass was launched from.  This cross-checks them against
    the selection of record and reports a mismatch instead of silently plotting a
    superseded configuration.
    """
    name = headline.dir_name(scan, 'diag')
    df = headline.load(scan, 'diag', results_root=results_root)
    rows = df[df['optimizer'] == SVEN_OPTIMIZER].sort_values('model_seed')
    if not len(rows):
        raise FileNotFoundError(f'{name} has no Sven runs')
    k, rtol, lr = int(_one(rows, 'k')), float(_one(rows, 'rtol')), float(_one(rows, 'lr'))
    B = int(_one(rows, 'batch_size'))
    ref = rows.iloc[0]
    matches, note = True, ''
    try:
        sel = headline.selection_methods(scan, payload).get(SVEN_OPTIMIZER)
    except KeyError:
        sel = None
    if sel is not None:
        want = {'k': k, 'rtol': rtol, 'lr': lr}
        have = {key: sel.get('hparams', {}).get(key) for key in want}
        matches = all(have[key] is not None and np.isclose(float(have[key]), want[key])
                      for key in want)
        if not matches:
            note = (f'[{name}] the diag pass ran Sven at k={k}, rtol={rtol:g}, lr={lr:g}, '
                    f'but the selection of record now says {headline.config_label(sel)}. '
                    f'The spectra below describe the configuration ON DISK; the diag pass '
                    f'has to be re-run for the new pick before they describe the headline '
                    f'Sven again.')
    out = SvenDiag(scan=scan, dir_name=name, title=headline.scan_title(scan), rows=rows,
                   k=k, rtol=rtol, lr=lr, B=B, n_epochs=sv.n_epochs(ref),
                   n_steps=sv.n_steps(ref),
                   seeds=sorted(int(s) for s in rows['model_seed']),
                   selection_matches=matches, selection_note=note)
    if verbose:
        print(f'{name}: Sven k={k} rtol={rtol:g} lr={lr:g} B={B} | '
              f'{len(rows)} seed(s) {out.seeds} | {out.n_epochs} epochs, {out.n_steps} steps'
              + ('' if matches else '  ** superseded selection **'))
        if note:
            print('  ' + note)
    return out


# ---------------------------------------------------------------------------
# Per-run derived quantities
# ---------------------------------------------------------------------------
#: ``sv_noise_floor / sigma_max`` per logged step -- the Gram round-off floor,
#: READ from the record rather than assumed.  One implementation, in
#: :mod:`sv_diagnostics` (which owns the schema-2 diagnostics loaders); re-exported
#: here so a figure module does not grow a second copy that can drift.
noise_floor_rel = sv.noise_floor_rel


def cumulative_energy(utr):
    """Cumulative fraction of the residual energy in the leading directions.

    ``utr[t, i] = u_i . r`` with ``U`` orthonormal, so ``sum_i utr^2 = ||P_U r||^2``
    and ``out[t, i] = sum_{j<=i} utr[t, j]^2 / sum_j utr[t, j]^2`` is the fraction
    of the residual (as far as the Jacobian can see it) that the top ``i+1``
    directions carry.  Only ``utr**2`` is meaningful: ``eigh`` fixes no sign and
    within a degenerate sigma cluster the sub-basis is arbitrary.

    Online (``M = B`` values, ``B <= n_rows``) the basis is complete in the batch's
    row space and the denominator IS ``||r||^2``.  Offline on a probe set with
    ``n_rows > P`` the basis is thin and a further part of the residual lies
    outside the column space entirely -- :func:`probe_energy` reports that
    separately rather than hiding it in the normalisation.
    """
    e = np.asarray(utr, dtype=float) ** 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        total = np.nansum(e, axis=-1, keepdims=True)
        return np.nancumsum(np.nan_to_num(e), axis=-1) / np.where(total > 0, total, np.nan)


def energy_in_top(utr, k):
    """Fraction of the residual energy in the top ``k`` directions, per step.

    ``k`` larger than the spectrum width means "no cut", which is exactly 1 and is
    returned as such (``k = B`` on four of the seven headline scans).
    """
    cum = cumulative_energy(utr)
    idx = int(np.clip(k, 1, cum.shape[-1])) - 1
    return cum[..., idx]


def rtol_rank(svs, rtol):
    """``#{i : sigma_i >= rtol * sigma_0}`` per step -- the cut rtol alone makes."""
    svs = np.asarray(svs, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nansum(svs >= rtol * svs[..., :1], axis=-1)


def used_rank(svs, k, rtol):
    """``min(k, rtol-rank)`` -- the number of directions actually inverted.

    This is ``num_nonzero_svs`` recomputed from the saved spectrum, which the
    notebooks check against the optimizer's own per-step count: the two are
    independent paths to the same number and agreeing is what says the saved
    spectrum really is the one the step used.
    """
    return np.minimum(int(k), rtol_rank(svs, rtol))


def diag_arrays(row, results_root=None):
    """Everything one Sven diag run offers, raw and derived, on the LOGGED steps.

    Keys: ``step`` (the optimizer's own step index, ``svs_step``), ``svs``, ``utr``,
    ``sv_max``, ``sv_min_kept``, ``sv_noise_floor``, ``update_norm``, ``resid_norm``
    (all per logged step); ``svs_rel`` (``sigma_i / sigma_max``), ``noise_rel``,
    ``cum_energy``, ``rtol_rank``, ``used_rank``, ``frac_top_k``, ``frac_used``;
    and ``nnz_logged`` -- the optimizer's own ``num_nonzero_svs`` sampled at the
    logged steps, for the cross-check :func:`used_rank` documents.

    ``k`` and ``rtol`` come from the row, so one function serves any scan.
    """
    diag = style.load_diagnostics(row, results_root=results_root)
    svs = np.asarray(diag['svs'], dtype=float)
    utr = np.abs(np.asarray(diag['utr'], dtype=float))
    step = np.asarray(diag['svs_step'], dtype=int)
    k, rtol = int(row['k']), float(row['rtol'])
    nnz = diag.get('num_nonzero_svs')
    nnz = (np.asarray(nnz, dtype=float)[np.clip(step, 0, len(nnz) - 1)]
           if nnz is not None else None)
    out = {'step': step, 'svs': svs, 'utr': utr, 'k': k, 'rtol': rtol,
           'width': svs.shape[1], 'nnz_logged': nnz}
    for key in ('sv_max', 'sv_min_kept', 'sv_noise_floor', 'update_norm', 'resid_norm'):
        out[key] = (np.asarray(diag[key], dtype=float) if diag.get(key) is not None
                    else None)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        out['svs_rel'] = svs / svs[:, :1]
    out['noise_rel'] = noise_floor_rel(diag)
    out['cum_energy'] = cumulative_energy(utr)
    out['rtol_rank'] = rtol_rank(svs, rtol)
    out['used_rank'] = np.minimum(k, out['rtol_rank'])
    out['frac_top_k'] = energy_in_top(utr, k)
    cum = out['cum_energy']
    out['frac_used'] = cum[np.arange(len(cum)), np.maximum(out['used_rank'] - 1, 0)]
    return out


def seed_stack(rows, key, results_root=None, arrays=None):
    """``(step, mean, std, n)`` of one :func:`diag_arrays` key over the seeds.

    Curves are truncated to the shortest seed (a run that stopped early keeps its
    partial curve, C-R1) and averaged with ``nanmean``; the std is ``ddof=1`` over
    seeds, i.e. :func:`style.seed_spread_label`'s band.  ``arrays`` lets a caller
    reuse an already-loaded list instead of re-reading the npz files.
    """
    arrays = (arrays if arrays is not None
              else [diag_arrays(r, results_root=results_root) for _, r in rows.iterrows()])
    curves = [np.asarray(a[key], dtype=float) for a in arrays if a.get(key) is not None]
    if not curves:
        return None, None, None, 0
    n = min(len(c) for c in curves)
    stack = np.stack([c[:n] for c in curves])
    step = np.asarray(arrays[0]['step'][:n])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
        std = (np.nanstd(stack, axis=0, ddof=1) if len(stack) > 1
               else np.zeros_like(mean))
    return step, mean, std, len(stack)


# ---------------------------------------------------------------------------
# The k x rtol grid (light records only -- no npz reads)
# ---------------------------------------------------------------------------
GRID_COLUMNS = ['k', 'rtol', 'lr', 'B', 'used', 'used_std', 'used_frac_k',
                'used_frac_B', 'n_seeds', 'n_records', 'n_diverged', 'n_no_curve',
                'counts']


def used_rank_grid(scan, lr=None, results_root=None, when='final'):
    """Mean SVs used per ``(k, rtol)`` of the TUNING scan, seed-averaged.

    Reads ``svd_summary['num_nonzero_svs_epoch']`` from the light records, so a
    whole grid costs one scan load and no ``diag/`` reads at all.  ``when`` is
    ``'final'`` (the last epoch), ``'first'`` or ``'mean'`` (over the run).

    The two readings this table carries are the ones
    :mod:`sv_diagnostics` documents: sweeping ``rtol`` at ``k = B`` gives the
    rtol-rank of the batch Jacobian, sweeping ``k`` at fixed ``rtol`` gives where
    the count saturates against the cap.  ``used / k`` says which of the two cuts
    binds: 1.0 means ``k`` is the binding constraint, below 1.0 means ``rtol`` is.

    **The dropped runs are counted, not skipped.**  Sven diverges on 688 of 7,825
    on-grid runs, concentrated at the bottom of every rtol grid (``EXPERIMENTS.md``
    section 7), so a cell of this grid can rest on one seed while its neighbour rests
    on five -- and this is the figure where that has to be visible.  Per cell:
    ``n_records`` (records present = attempted), ``n_diverged`` (excluded by the
    divergence rule), ``n_no_curve`` (finished but no ``num_nonzero_svs_epoch``),
    ``n_seeds`` (what the mean is actually over) and ``counts`` =
    ``"n_seeds/n_records"``, the binding finished/attempted string.  A cell whose every
    run diverged is kept with ``used = NaN`` and ``n_seeds = 0`` -- distinguishable from
    a cell the grid never contained at all, which is simply absent.
    """
    df = headline.load(scan, '', results_root=results_root)
    rows = df[df['optimizer'] == SVEN_OPTIMIZER]
    if lr is not None:
        rows = rows[np.isclose(pd.to_numeric(rows['lr'], errors='coerce'), float(lr))]
    out, counts = [], {}
    for _, r in rows.iterrows():
        key = (int(r['k']), float(r['rtol']))
        c = counts.setdefault(key, {'k': key[0], 'rtol': key[1], 'lr': float(r['lr']),
                                    'B': int(r['batch_size']), 'n_records': 0,
                                    'n_diverged': 0, 'n_no_curve': 0})
        c['n_records'] += 1
        if r.get('diverged') or r.get('failed'):
            c['n_diverged'] += 1
            continue
        curve = sv.rank_per_epoch(r)
        if curve is None or not len(curve):
            c['n_no_curve'] += 1
            continue
        value = {'final': curve[-1], 'first': curve[0], 'mean': float(np.mean(curve))}[when]
        out.append({'k': key[0], 'rtol': key[1],
                    'model_seed': int(r['model_seed']), 'used': float(value)})
    if not counts:
        return pd.DataFrame(columns=GRID_COLUMNS)
    grid = pd.DataFrame(list(counts.values()))
    if out:
        agg = pd.DataFrame(out).groupby(['k', 'rtol'], as_index=False).agg(
            used=('used', 'mean'), used_std=('used', 'std'),
            n_seeds=('model_seed', 'nunique'))
    else:
        agg = pd.DataFrame(columns=['k', 'rtol', 'used', 'used_std', 'n_seeds'])
    grp = grid.merge(agg, on=['k', 'rtol'], how='left')
    grp['n_seeds'] = grp['n_seeds'].fillna(0).astype(int)
    grp['used_frac_k'] = grp['used'] / grp['k']
    grp['used_frac_B'] = grp['used'] / grp['B']
    grp['counts'] = [headline.fmt_counts(int(s), int(a))
                     for s, a in zip(grp['n_seeds'], grp['n_records'])]
    return grp[GRID_COLUMNS].sort_values(['k', 'rtol']).reset_index(drop=True)


def mechanism_table(diags, results_root=None):
    """One row per scan: what the ``k`` / rtol cut does, seed-averaged (ONLINE).

    Columns: ``B``, ``k``, ``rtol``; ``used_first`` / ``used_final`` (SVs actually
    inverted); ``binds`` (which of the two cuts is active); ``energy_top_k`` and
    ``energy_used`` at the first and last logged step (the fraction of the BATCH
    residual the cut keeps); ``noise_rel`` (the recorded Gram floor); ``n_seeds``.
    """
    rows = []
    for d in diags:
        arrays = [diag_arrays(r, results_root=results_root) for _, r in d.rows.iterrows()]
        def m(key, idx):
            return float(np.nanmean([a[key][idx] for a in arrays]))
        used_first, used_final = m('used_rank', 0), m('used_rank', -1)
        binds = ('k' if d.k < np.nanmean([a['rtol_rank'][-1] for a in arrays])
                 else 'rtol' if used_final < d.k - 0.5 else 'neither ($k=B$)')
        rows.append({
            'scan': d.title, 'B': d.B, 'k': d.k, 'rtol': d.rtol,
            'used_first': used_first, 'used_final': used_final,
            'binds': binds,
            'energy_top_k_first': m('frac_top_k', 0), 'energy_top_k_final': m('frac_top_k', -1),
            'energy_used_first': m('frac_used', 0), 'energy_used_final': m('frac_used', -1),
            'noise_rel': float(np.nanmedian(np.concatenate([a['noise_rel'] for a in arrays]))),
            'n_seeds': len(arrays)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Online figures
# ---------------------------------------------------------------------------
def _floor_lines(ax, d, arrays, show_rtol=True, show_noise=True, show_k=True,
                 width=None):
    """The three reference lines every spectrum panel carries."""
    handles = []
    if show_rtol:
        handles.append(ax.axhline(d.rtol, color='0.35', lw=1.1, ls=':', zorder=0,
                                  label=r'rtol $\sigma_{\max}$'))
    if show_noise:
        floor = float(np.nanmedian(np.concatenate([a['noise_rel'] for a in arrays])))
        handles.append(ax.axhline(floor, color='0.35', lw=1.1, ls='--', zorder=0,
                                  label=rf'Gram float32 floor '
                                        rf'($\sqrt{{\epsilon}}\,\sigma_{{\max}}$'
                                        rf' $= {floor:.1e}$)'))
    if show_k and d.truncates and width:
        handles.append(ax.axvline(d.k - 1, color='#C44E52', lw=1.4, zorder=0,
                                  label=f'$k = {d.k}$'))
    return handles


def plot_spectra_over_training(ax, d, arrays=None, results_root=None, seed=None,
                               limit=45, cmap='plasma', lw=0.9, alpha=0.85,
                               show_rtol=True, show_noise=True, show_k=True):
    """Full-width batch spectra ``sigma_i / sigma_max`` over training (ONLINE).

    One line per logged step, coloured by step (log scale), from ONE seed -- seeds
    are not averaged here because each line is a different random batch and
    averaging them would draw a matrix that no step ever had.  The ``k`` cut, the
    ``rtol * sigma_max`` line and the RECORDED Gram noise floor are drawn on top;
    everything below that floor is round-off, not structure.

    Returns ``(norm, handles)`` for :func:`sv_diagnostics.epoch_colorbar`.
    """
    rows = d.rows if seed is None else d.rows[d.rows['model_seed'] == seed]
    arrays = arrays or [diag_arrays(rows.iloc[0], results_root=results_root)]
    a = arrays[0]
    sel = subsample_rows(len(a['step']), limit)
    colors, norm = progress_colors(np.maximum(a['step'][sel], 1), cmap=cmap)
    x = np.arange(a['width'])
    for j, i in enumerate(sel):
        y = a['svs_rel'][i]
        ax.plot(x[np.isfinite(y)], y[np.isfinite(y)], color=colors[j], lw=lw,
                alpha=alpha, zorder=2)
    handles = _floor_lines(ax, d, arrays, show_rtol, show_noise, show_k, a['width'])
    ax.set_yscale('log')
    ax.set_xlabel('Singular value index $i$')
    ax.set_ylabel(r'$\sigma_i \,/\, \sigma_{\max}$')
    return norm, handles


def plot_utr_over_training(ax, d, arrays=None, results_root=None, seed=None,
                           limit=45, cmap='viridis', lw=0.9, alpha=0.85,
                           normalize=True, show_k=True, floor=1e-9):
    """``|u_i . r|`` profiles over training (ONLINE) -- where the residual sits.

    The companion of :func:`plot_spectra_over_training`: the spectrum says how hard
    a direction is to move, this says how much residual is in it, and only the two
    together say whether the cut discards anything that mattered.  ``normalize``
    divides each step by ``||U^T r||`` so steps with very different residual scales
    are comparable.
    """
    rows = d.rows if seed is None else d.rows[d.rows['model_seed'] == seed]
    arrays = arrays or [diag_arrays(rows.iloc[0], results_root=results_root)]
    a = arrays[0]
    sel = subsample_rows(len(a['step']), limit)
    colors, norm = progress_colors(np.maximum(a['step'][sel], 1), cmap=cmap)
    x = np.arange(a['width'])
    for j, i in enumerate(sel):
        y = a['utr'][i]
        if normalize:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                y = y / np.sqrt(np.nansum(a['utr'][i] ** 2))
        good = np.isfinite(y)
        ax.plot(x[good], np.maximum(y[good], floor), color=colors[j], lw=lw,
                alpha=alpha, zorder=2)
    if show_k and d.truncates:
        ax.axvline(d.k - 1, color='#C44E52', lw=1.4, zorder=0, label=f'$k = {d.k}$')
    ax.set_yscale('log')
    ax.set_xlabel('Singular value index $i$')
    ax.set_ylabel(r'$|u_i^{\top} r| \,/\, \|U^{\top} r\|$' if normalize
                  else r'$|u_i^{\top} r|$')
    return norm


def plot_energy_capture(ax, d, arrays=None, results_root=None, smooth=25,
                        show_used=True, band=True, color=None, label=None):
    """THE mechanism plot: residual energy kept by the cut, over training (ONLINE).

    Two curves, seed-averaged with a ``+/- 1 std`` band:

    * **top ``k``** -- what the truncation rank alone keeps,
      ``sum_{i<k} (u_i.r)^2 / ||U^T r||^2``;
    * **actually used** (``show_used``) -- the same sum over ``min(k, rtol-rank)``
      directions, i.e. after the ``rtol`` cut as well.  Where this sits far below
      the first curve, it is ``rtol``, not ``k``, that is discarding the residual
      (MNIST-CE at rtol = 0.3 is the extreme case).

    Both are fractions of THIS BATCH's residual in THIS BATCH's row space; the
    probe-set answer (:func:`probe_energy`) is a different and usually smaller
    number, and the notebook says so beside every figure.
    """
    arrays = arrays or [diag_arrays(r, results_root=results_root)
                        for _, r in d.rows.iterrows()]
    color = color or style.method_color('Sven')
    out = {}
    for key, ls, lab in (('frac_top_k', '-', label or f'top $k={d.k}$'),
                         ('frac_used', '--', f'used (min($k$, rtol-rank))')):
        if key == 'frac_used' and not show_used:
            continue
        step, mean, std, n = seed_stack(d.rows, key, arrays=arrays)
        if step is None:
            continue
        if smooth and smooth > 1:
            mean, idx = sv.smooth(mean, smooth)
            std = sv.smooth(std, smooth)[0]
            step = step[idx]
        ax.plot(step, mean, color=color, ls=ls, lw=2.0, label=lab, zorder=3)
        if band and n > 1:
            ax.fill_between(step, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                            color=color, alpha=0.22, lw=0, zorder=1)
        out[key] = (step, mean, std, n)
    if band and len(arrays) > 1:
        style.band_legend(ax, color=color)
    ax.set_xlabel('Optimizer step')
    ax.set_ylabel('Residual energy kept')
    return out


def plot_energy_profile(ax, d, arrays=None, results_root=None, fractions=(0.0, 0.5, 1.0),
                        cmap='plasma', show_k=True):
    """Cumulative residual energy vs direction index at a few points in training.

    ``out[i]`` is the fraction of the batch residual in the top ``i+1`` directions,
    seed-averaged, at the logged step nearest each of ``fractions`` of training.
    Reading off the ``k`` line gives the number :func:`plot_energy_capture` plots
    against time; the shape says whether the residual is concentrated in a few
    directions or spread over the whole batch.
    """
    import matplotlib.pyplot as plt

    arrays = arrays or [diag_arrays(r, results_root=results_root)
                        for _, r in d.rows.iterrows()]
    n = min(len(a['step']) for a in arrays)
    base = plt.get_cmap(cmap)
    out = {}
    for j, frac in enumerate(fractions):
        i = int(np.clip(round(frac * (n - 1)), 0, n - 1))
        cum = np.nanmean(np.stack([a['cum_energy'][i] for a in arrays]), axis=0)
        step = int(arrays[0]['step'][i])
        color = base(0.15 + 0.7 * (j / max(len(fractions) - 1, 1)))
        ax.plot(np.arange(1, len(cum) + 1), cum, color=color, lw=2.0,
                label=f'step {step}')
        out[step] = cum
    if show_k and d.truncates:
        ax.axvline(d.k, color='#C44E52', lw=1.4, zorder=0, label=f'$k = {d.k}$')
    ax.set_xlabel('Directions kept (top $i$)')
    ax.set_ylabel('Cumulative residual energy')
    ax.set_ylim(0, 1.02)
    return out


def plot_rank_used(ax, d, arrays=None, results_root=None, smooth=25, band=True,
                   show_cap=True, color=None):
    """SVs actually inverted vs step, against the ``k`` cap and the rtol-rank (ONLINE).

    Three lines: ``min(k, rtol-rank)`` (what the step used), the rtol-rank alone
    (uncapped) and, dotted, the caps ``k`` and ``B``.  The gap between the first
    two is what ``k`` removes; the gap between the second and ``B`` is what
    ``rtol`` removes.
    """
    arrays = arrays or [diag_arrays(r, results_root=results_root)
                        for _, r in d.rows.iterrows()]
    color = color or style.method_color('Sven')
    for key, ls, lab, col in (('used_rank', '-', 'used $= \\min(k,\\,$rtol-rank$)$', color),
                              ('rtol_rank', '-.', 'rtol-rank (uncapped)', '#4C72B0')):
        step, mean, std, n = seed_stack(d.rows, key, arrays=arrays)
        if step is None:
            continue
        if smooth and smooth > 1:
            mean, idx = sv.smooth(mean, smooth)
            std = sv.smooth(std, smooth)[0]
            step = step[idx]
        ax.plot(step, mean, color=col, ls=ls, lw=2.0, label=lab, zorder=3)
        if band and n > 1:
            ax.fill_between(step, mean - std, mean + std, color=col, alpha=0.20,
                            lw=0, zorder=1)
    if show_cap:
        ax.axhline(d.k, color='#C44E52', lw=1.2, ls=':', zorder=0, label=f'$k = {d.k}$')
        if d.B != d.k:
            ax.axhline(d.B, color='0.45', lw=1.2, ls=':', zorder=0, label=f'$B = {d.B}$')
    if band and len(arrays) > 1:
        style.band_legend(ax, color=color)
    ax.set_xlabel('Optimizer step')
    ax.set_ylabel('Singular values used')
    return ax


def plot_norms(ax, d, arrays=None, results_root=None, smooth=25, band=True,
               keys=('update_norm', 'resid_norm')):
    """``||theta_new - theta_old||`` and ``||r||`` vs step, seed-averaged (ONLINE).

    ``update_norm`` is the APPLIED change, ``lr`` included (``sven.opt.sven``
    docstring) -- divide by the run's ``lr`` for the raw solve.  Plotted together
    they say whether Sven's steps shrink because the residual shrank or because the
    solve got smaller relative to it.
    """
    arrays = arrays or [diag_arrays(r, results_root=results_root)
                        for _, r in d.rows.iterrows()]
    colors = {'update_norm': style.method_color('Sven'), 'resid_norm': '#DD8452',
              'sv_min_kept': '#55A868'}
    labels = {'update_norm': r'$\|\Delta\theta\|$ (applied, incl. $\eta$)',
              'resid_norm': r'$\|r\|$', 'sv_min_kept': r'$\sigma_{\min}$ kept'}
    for key in keys:
        step, mean, std, n = seed_stack(d.rows, key, arrays=arrays)
        if step is None:
            continue
        if smooth and smooth > 1:
            mean, idx = sv.smooth(mean, smooth)
            std = sv.smooth(std, smooth)[0]
            step = step[idx]
        col = colors.get(key, '0.3')
        ax.plot(step, mean, color=col, lw=2.0, label=labels.get(key, key), zorder=3)
        if band and n > 1:
            # a log axis cannot take mean - std <= 0; style.clipped_band is the one
            # place that decides how the lower edge is pinned (it is also what the
            # seed-band legend promises)
            lower, upper = style.clipped_band(mean, std, np.maximum(mean * 1e-3, 1e-14))
            ax.fill_between(step, lower, upper, color=col, alpha=0.20, lw=0, zorder=1)
    if band and len(arrays) > 1:
        style.band_legend(ax)
    ax.set_yscale('log')
    ax.set_xlabel('Optimizer step')
    ax.set_ylabel('Norm')
    return ax


# ---------------------------------------------------------------------------
# Probe-set (offline, float64) trajectories
# ---------------------------------------------------------------------------
def probe_scan_name(scan):
    """``toy_1d_scan`` -> ``toy_1d_scan_diag`` (what the cache is keyed by)."""
    return scan if scan.endswith('_diag') else f'{scan}_diag'


def probe_methods(scan, n_probe=-1, epochs_only=False, out_dir=None):
    """The methods cached for one scan, in :data:`PROBE_METHODS` order."""
    try:
        ents = ct.load_spectra(probe_scan_name(scan), n_probe=n_probe,
                               epochs_only=epochs_only, out_dir=out_dir)
    except FileNotFoundError:
        return []
    have = {str(e['method']) for e in ents}
    return [m for m in PROBE_METHODS if m in have] + sorted(have - set(PROBE_METHODS))


def probe_widths(scans, n_probe=-1, epochs_only=False, out_dir=None):
    """Inventory of the cached probe trajectories: rows, parameters, MEASURED width.

    The spectrum width is ``min(n_rows, P)`` in principle, but only one of the two
    bounds is ever active and quoting the other is misleading: on MNIST ``P`` is 27,562
    while the probe set is 512 rows, so the spectrum is 512 wide and resolves 512
    directions, not 27k.  This reports what is in the files -- ``width`` from
    ``svals.shape[1]`` -- together with the filters used, and the count under those
    filters, which is the number every downstream table is actually computed from
    (``load_spectra`` with no filters also returns the extra probe tags: MNIST label-reg
    has 4 ``probe2000_epochs`` files beside the 20 ``probe512`` ones).
    """
    rows = []
    for scan in scans:
        name = probe_scan_name(scan)
        try:
            ents = ct.load_spectra(name, n_probe=n_probe, epochs_only=epochs_only,
                                   out_dir=out_dir)
            allents = ct.load_spectra(name, out_dir=out_dir)
        except FileNotFoundError:
            continue
        if not ents:
            continue
        widths = sorted({int(np.asarray(e['svals']).shape[1]) for e in ents})
        rows.append({
            'scan': scan, 'n_rows': int(ents[0]['n_rows']),
            'n_params': int(ents[0]['n_params']),
            'width': widths[0] if len(widths) == 1 else widths,
            'n_ckpts': int(len(ents[0]['step'])),
            'methods': len({str(e['method']) for e in ents}),
            'seeds': len({int(e['model_seed']) for e in ents}),
            'n_used': len(ents), 'n_cached_any_tag': len(allents)})
    return pd.DataFrame(rows)


def effective_rank(svals, base='entropy'):
    """Effective rank of a spectrum: ``exp(H(p))`` with ``p_i = sigma_i / sum sigma``.

    The participation / entropy rank (Roy & Vetterli): it is ``r`` for a flat
    rank-``r`` spectrum and drops smoothly as the spectrum concentrates, so it can
    be tracked along a trajectory where the numerical rank (a step function at a
    threshold) cannot.  Sensitive to the tail only through its mass, which is what
    makes it usable on a float64 probe spectrum that never reaches exact zero.
    """
    s = np.asarray(svals, dtype=float)
    s = s[np.isfinite(s) & (s > 0)]
    if not s.size:
        return np.nan
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def probe_metrics(scan, n_probe=-1, epochs_only=False, methods=None, floor=PROBE_FLOOR,
                  sigma_b=None, out_dir=None, results_root=None):
    """Tidy per-checkpoint table of the cached probe-set spectra (OFFLINE, float64).

    One row per (method, seed, checkpoint) with ``step``, ``epoch``, ``sigma_max``,
    ``sigma_min`` (above ``floor``), ``cond`` (``sigma_max / sigma_min``, over the
    numerically resolved part only -- the raw ratio on a rank-deficient Jacobian is
    a round-off artefact), ``rank`` (``#{sigma > floor * sigma_max}``),
    ``eff_rank`` (:func:`effective_rank`), ``sigma_B_over_1``
    (``sigma_{sigma_b} / sigma_1``, the damping at Sven's batch cut; ``sigma_b``
    defaults to the run's ``batch_size``), ``probe_loss``, ``param_norm``,
    ``dist_init`` and ``resid_norm``.

    Rows are comparable ACROSS optimizers by construction: the probe set, the row
    definition and the dtype are identical for every trajectory of a scan
    (``ckpt_tools.row_spec_for_scan`` raises if a scan's runs disagree).

    **``cond`` and ``sigma_B_over_1`` are bounded by ``floor``, not only by the data.**
    ``cond`` is taken over the resolved part, so it can never exceed ``1 / floor``
    (= 1e12); where a Jacobian's true ``sigma_min / sigma_max`` is far below the
    threshold -- toy, whose raw ratio is 1.3e-17 with only 14 of 593 directions above
    ``floor * sigma_max`` -- the number that comes out measures the threshold rather
    than the matrix.  ``sigma_b_resolved`` says whether index ``sigma_b`` is inside the
    resolved rank at all (on toy at step 0 it is not: ``sigma_32`` of a numerically
    rank-14 spectrum does not exist), so a figure can mark those points instead of
    drawing round-off as a measurement.  Columns ``floor``, ``sigma_b`` and
    ``sigma_b_resolved`` carry that information with the numbers.
    """
    name = probe_scan_name(scan)
    ents = ct.load_spectra(name, n_probe=n_probe, epochs_only=epochs_only, out_dir=out_dir)
    if methods is not None:
        ents = [e for e in ents if str(e['method']) in set(methods)]
    rows = []
    for e in ents:
        svals, utr = np.asarray(e['svals'], float), np.asarray(e['utr'], float)
        b = sigma_b
        if b is None:
            # the record, not the Run: `load_run` also resolves the Hydra config and
            # would rebuild the dataset, which is minutes of MNIST/CIFAR work for one
            # integer.  The cache deliberately does not duplicate hyperparameters
            # (ckpt_tools.load_spectra), so they are read here and cannot go stale.
            try:
                b = int(ct.load_record(name, str(e['run_id']),
                                       results_root=results_root).get('batch_size') or 0)
            except Exception:                         # a record we cannot resolve
                b = 0
        for i in range(len(e['step'])):
            s = svals[i][np.isfinite(svals[i])]
            keep = s[s > floor * s[0]] if s.size and s[0] > 0 else s
            u = utr[i][np.isfinite(utr[i])]
            rows.append({
                'scan': scan, 'method': str(e['method']),
                'model_seed': int(e['model_seed']), 'step': int(e['step'][i]),
                'epoch': int(e['epoch'][i]), 'n_rows': int(e['n_rows']),
                'n_params': int(e['n_params']),
                'sigma_max': float(s[0]) if s.size else np.nan,
                'sigma_min': float(keep[-1]) if keep.size else np.nan,
                'cond': float(s[0] / keep[-1]) if keep.size else np.nan,
                'rank': int(keep.size),
                'eff_rank': effective_rank(s),
                'sigma_B_over_1': (float(s[min(b, len(s)) - 1] / s[0])
                                   if b and s.size else np.nan),
                'sigma_b': int(min(b, len(s))) if b and s.size else 0,
                'sigma_b_resolved': bool(b and s.size
                                         and min(b, len(s)) <= keep.size),
                'floor': float(floor),
                'probe_loss': float(e['probe_loss'][i]),
                'param_norm': float(e['param_norm'][i]),
                'dist_init': float(e['dist_init'][i]),
                'resid_norm': float(e['rows_norm'][i]),
                'proj_frac': float((u ** 2).sum() / float(e['rows_norm'][i]) ** 2)
                if float(e['rows_norm'][i]) > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def probe_energy(scan, ks, n_probe=-1, epochs_only=False, method='Sven',
                 when=('first', 'last'), out_dir=None):
    """Residual energy inside the top-``k`` PROBE directions (OFFLINE), seed mean.

    The honest counterpart of :func:`plot_energy_capture`.  Two normalisations are
    reported because they answer different questions:

    * ``in_span`` -- ``||P_U r||^2 / ||r||^2``: how much of the residual the
      Jacobian's column space can reach AT ALL.  Below 1 only when the probe set
      has more rows than the model has parameters (toy / polynomial: 10,000 rows,
      ~600 parameters), and then the shortfall is unreachable by ANY optimizer,
      Sven or not.
    * ``top_k`` -- the fraction of the reachable residual in the leading ``k``
      directions, i.e. what Sven's truncation keeps.

    On polynomial (``k = 16``) ``top_k`` is ~0.59 at the end of training against
    0.999 for the same ``k`` measured on a 32-row batch: the probe set is where the
    truncation is visible.
    """
    name = probe_scan_name(scan)
    ents = [e for e in ct.load_spectra(name, method=method, n_probe=n_probe,
                                       epochs_only=epochs_only, out_dir=out_dir)]
    if not ents:
        return pd.DataFrame(columns=['when', 'step', 'k', 'top_k', 'in_span', 'n_seeds'])
    rows = []
    for tag in when:
        idx = 0 if tag == 'first' else -1
        per_seed = {k: [] for k in ks}
        span, steps = [], []
        for e in ents:
            u = np.asarray(e['utr'], float)[idx]
            u = u[np.isfinite(u)]
            total = (u ** 2).sum()
            cum = np.cumsum(u ** 2) / total if total > 0 else np.full(len(u), np.nan)
            for k in ks:
                per_seed[k].append(cum[min(int(k), len(cum)) - 1])
            span.append(total / float(e['rows_norm'][idx]) ** 2)
            steps.append(int(e['step'][idx]))
        for k in ks:
            rows.append({'when': tag, 'step': int(np.median(steps)), 'k': int(k),
                         'top_k': float(np.mean(per_seed[k])),
                         'top_k_std': float(np.std(per_seed[k], ddof=1))
                         if len(ents) > 1 else 0.0,
                         'in_span': float(np.mean(span)), 'n_seeds': len(ents)})
    return pd.DataFrame(rows)


def plot_probe_spectra(ax, scan, method, n_probe=-1, epochs_only=False, seed=None,
                       limit=14, cmap='plasma', normalize=True, lw=1.3,
                       floor=PROBE_FLOOR, out_dir=None, annotate_seed=True):
    """Probe-set spectrum evolution along ONE optimizer's trajectory, ONE seed (OFFLINE).

    One line per cached checkpoint, coloured by step.  Unlike the online figures
    this IS the same matrix's spectrum at every point -- the rows never change --
    so the motion is the model's, not the batch's.  Returns the Normalize for a
    colourbar.

    **One seed, never a seed average.**  Five trajectories are cached per (scan,
    method); this draws the one ``seed`` names, or the lowest-numbered one when
    ``seed`` is None -- and then warns, because a 5-seed cache silently reduced to its
    first element is exactly the figure a reader over-reads.  ``annotate_seed`` writes
    the seed into the panel so the figure itself says which trajectory it is; averaging
    would be wrong here rather than merely coarse, since each seed's probe spectrum is
    conditioned on that seed's initialisation.
    """
    # loaded unfiltered, then filtered here: the panel has to say "1 of FIVE cached",
    # which a pre-filtered load cannot know
    cached = ct.load_spectra(probe_scan_name(scan), method=method, n_probe=n_probe,
                             epochs_only=epochs_only, out_dir=out_dir)
    ents = ([e for e in cached if int(e['model_seed']) == int(seed)] if seed is not None
            else cached)
    if not ents:
        return None
    if seed is None and len(cached) > 1:
        warnings.warn(
            f'plot_probe_spectra({scan!r}, {method!r}): {len(cached)} cached seeds '
            f'{sorted(int(e["model_seed"]) for e in cached)}; drawing only '
            f'{int(ents[0]["model_seed"])}. Pass seed= to choose deliberately.',
            stacklevel=2)
    e = ents[0]
    if annotate_seed:
        ax.annotate(f'seed {int(e["model_seed"])} (1 of {len(cached)} cached)',
                    xy=(0.03, 0.04), xycoords='axes fraction', fontsize=8,
                    color='0.25', ha='left', va='bottom')
    svals = np.asarray(e['svals'], float)
    sel = subsample_rows(len(e['step']), limit)
    colors, norm = progress_colors(np.maximum(np.asarray(e['step'])[sel], 1), cmap=cmap)
    for j, i in enumerate(sel):
        s = svals[i][np.isfinite(svals[i])]
        y = s / s[0] if normalize else s
        ax.plot(np.arange(1, len(y) + 1), np.maximum(y, floor), color=colors[j], lw=lw)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Singular value index $i$')
    ax.set_ylabel(r'$\sigma_i \,/\, \sigma_1$' if normalize else r'$\sigma_i$')
    return norm


def plot_probe_metric(ax, table, metric, methods=None, x='step', band=True,
                      logx=True, logy=False, marker='o', resolution=True,
                      near_decades=6):
    """One :func:`probe_metrics` column vs step, one line per optimizer, seed band.

    Colours and names come from :data:`style.METHOD_COLORS` / :func:`style.method_label`,
    so an optimizer is the same colour here as in every other figure of the campaign.

    ``resolution`` draws the float64 limit the quantity runs into, the offline
    counterpart of :func:`_floor_lines`' recorded Gram floor:

    * ``cond`` cannot exceed ``1 / floor`` by construction (the ratio is taken over the
      resolved part), so the ceiling is drawn and a curve lying on it is reported as
      floor-limited rather than as a condition number;
    * ``sigma_B_over_1`` is meaningless where index ``B`` is outside the resolved rank
      (``sigma_b_resolved`` is False), so those points are drawn hollow and faint and
      ``floor`` itself is drawn as a line.

    Marking them is the whole point: on toy the spectrum is numerically rank-14 at step
    0 out of 593 columns, so both quantities are threshold artefacts there.  The line is
    drawn only where it is relevant -- when the data come within ``near_decades`` of it
    or when a point is unresolved -- so a panel whose spectrum is fully resolved is not
    stretched by ten empty decades to carry a limit it never approaches.
    """
    floor = float(table['floor'].iloc[0]) if 'floor' in table and len(table) \
        else PROBE_FLOOR
    resolved_col = 'sigma_b_resolved' if metric == 'sigma_B_over_1' else None
    table = table[table[metric].notna()]
    methods = methods or [m for m in PROBE_METHODS if m in set(table['method'])]
    n_unresolved = 0
    extremes = []
    for method in methods:
        sub = table[table['method'] == method]
        if not len(sub):
            continue
        grp = sub.groupby(x)[metric].agg(['mean', 'std', 'count']).reset_index()
        xs = grp[x].to_numpy(dtype=float)
        if logx:
            # step + 1, so the untrained point (step 0) gets its own place on a log
            # axis instead of being stacked on top of step 1 -- which drew a spurious
            # vertical jump and a band 5x too wide at the left edge
            xs = xs + 1.0
        color = style.method_color(method)
        ax.plot(xs, grp['mean'], color=color, lw=2.0, marker=marker, ms=3.5,
                label=style.method_label(method), zorder=3)
        extremes.append(grp['mean'].max() if metric == 'cond' else grp['mean'].min())
        if resolved_col is not None and resolved_col in sub:
            # a point is drawn hollow when index B sits outside the resolved rank for
            # ANY seed at that step: then sigma_B does not numerically exist
            ok = sub.groupby(x)[resolved_col].all().reindex(grp[x]).to_numpy()
            bad = ~np.asarray(ok, dtype=bool)
            n_unresolved += int(bad.sum())
            if bad.any():
                ax.plot(xs[bad], grp['mean'].to_numpy()[bad], ls='', marker='o',
                        ms=8, mfc='none', mec=color, mew=1.6, zorder=4)
        if band and grp['count'].max() > 1:
            lo, hi = grp['mean'] - grp['std'].fillna(0), grp['mean'] + grp['std'].fillna(0)
            if logy:
                lo = np.maximum(lo, np.nanmin(grp['mean']) * 1e-3)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0, zorder=1)
    reach = max(extremes) if extremes and metric == 'cond' else (
        min(extremes) if extremes else np.nan)
    if resolution and metric == 'cond':
        near = np.isfinite(reach) and reach > (1.0 / floor) * 10.0 ** -near_decades
        if near:
            ax.axhline(1.0 / floor, color='0.35', lw=1.2, ls='--', zorder=0,
                       label=rf'resolution ceiling $1/{floor:g}$')
    if resolution and metric == 'sigma_B_over_1':
        near = np.isfinite(reach) and reach < floor * 10.0 ** near_decades
        if near or n_unresolved:
            ax.axhline(floor, color='0.35', lw=1.2, ls='--', zorder=0,
                       label=rf'float64 probe floor ({floor:g}$\,\sigma_1$)')
        if n_unresolved:
            ax.plot([], [], ls='', marker='o', ms=8, mfc='none', mec='0.35', mew=1.6,
                    label=r'$\sigma_B$ outside the resolved rank')
    if band:
        style.band_legend(ax)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel('Optimizer step $+\\,1$' if logx else 'Optimizer step')
    ax.set_ylabel(PROBE_METRIC_LABELS.get(metric, metric.replace('_', ' ')))
    return ax


def low4_table(tables, metrics=('dist_init', 'param_norm'), reference='Sven',
               level=0.95, titles=None):
    """Codex Low 4, with the seed spread that decides whether a rank means anything.

    For each scan and metric: the last cached checkpoint's seed mean and ``ddof=1`` std
    per optimizer, ``reference``'s rank among them, and -- the part a rank alone cannot
    give -- the PAIRED per-seed difference between ``reference`` and the best other
    method, with a Student-t interval (``paired.paired_difference``'s convention:
    ``std`` over the differences, ``sem = std / sqrt(n)``, half-width
    ``t_{1-(1-level)/2, n-1} * sem``).  Pairing is by ``model_seed``, which is the same
    initialisation and the same probe rows on both sides.

    ``verdict`` keeps both facts, because either alone misleads: ``smallest`` /
    ``larger`` is the rank of the MEAN, and ``resolved`` says whether the paired
    interval excludes zero.  A rank of 1 that is not resolved is not a win: on MNIST
    label-reg Sven's 12.476 against HIG's 12.518 is a paired -0.042 +/- 0.20 with 2 of 5
    seeds the other way and seed stds of 0.27 / 0.34, six times the gap.  With five
    seeds the interval is wide, so most of these comparisons come out unresolved -- that
    is the honest answer to Codex Low 4 and it is reported as such rather than rounded
    into a ranking.
    """
    items = (tables.items() if hasattr(tables, 'items')
             else [(t['scan'].iloc[0], t) for t in tables])
    rows = []
    for scan, tbl in items:
        title = (titles or {}).get(scan, headline.scan_title(scan)
                                   if scan in headline.HEADLINE_SCANS else scan)
        last = tbl[tbl['step'] == tbl['step'].max()]
        for metric in metrics:
            agg = last.groupby('method')[metric].agg(['mean', 'std', 'count'])
            order = agg['mean'].sort_values()
            if reference not in order.index:
                continue
            rank = int(list(order.index).index(reference)) + 1
            others = [m for m in order.index if m != reference]
            piv = last.pivot_table(index='model_seed', columns='method', values=metric)
            best_other = others[0] if others else None
            d = np.asarray([], dtype=float)
            if best_other is not None and best_other in piv and reference in piv:
                pair = piv[[reference, best_other]].dropna()
                d = (pair[reference] - pair[best_other]).to_numpy(dtype=float)
            n = d.size
            mean = float(d.mean()) if n else np.nan
            sd = float(d.std(ddof=1)) if n > 1 else np.nan
            half = (float(stats.t.ppf(0.5 + level / 2, n - 1) * sd / np.sqrt(n))
                    if n > 1 else np.nan)
            tie = bool(n > 1 and np.isfinite(half) and abs(mean) <= half)
            rows.append({
                'scan': title, 'metric': metric,
                reference: float(order[reference]),
                f'{reference}_std': float(agg.loc[reference, 'std']),
                'rank': rank, 'n_methods': len(order),
                'best_other': best_other,
                'best_other_mean': float(order.iloc[0]) if rank != 1
                else float(order.iloc[1]) if len(order) > 1 else np.nan,
                'paired_mean': mean, 'paired_std': sd, 'paired_half': half,
                'n_paired': n, 'n_ref_lower': int((d < 0).sum()),
                'resolved': bool(n > 1 and np.isfinite(half) and not tie),
                'verdict': ('smallest' if rank == 1 else 'larger')
                           + ('' if (n > 1 and np.isfinite(half) and not tie)
                              else ' (not resolved)'),
                'n_seeds': int(agg.loc[reference, 'count'])})
    return pd.DataFrame(rows)


def low4_verdict(tbl, metric, reference='Sven'):
    """The reading of :func:`low4_table` for one metric: rank AND whether it resolves.

    Two sentences, because one number cannot carry it: how often ``reference``'s mean is
    the smallest, and on how many of those the paired seed interval actually excludes
    zero.  A rank quoted without the second half is the mistake this replaces.
    """
    sub = tbl[tbl['metric'] == metric]
    if not len(sub):
        return f'{metric}: nothing cached'
    small = sub[sub['rank'] == 1]
    big = sub[sub['rank'] != 1]

    def detail(r):
        return (f"{r['scan']} ({r[reference]:.4g} vs {r['best_other']} "
                f"{r['best_other_mean']:.4g}; paired {r['paired_mean']:+.3g} "
                f"± {r['paired_half']:.2g}, {r['n_ref_lower']}/{r['n_paired']} seeds "
                f"lower, {'resolved' if r['resolved'] else 'NOT resolved'})")
    lines = [f'{metric}: {reference} has the smallest mean on {len(small)} of '
             f'{len(sub)} scans, of which the paired 95% interval excludes zero on '
             f'{int(small["resolved"].sum()) if len(small) else 0}.']
    if len(small):
        lines.append('    smallest: ' + '; '.join(detail(r) for _, r in small.iterrows()))
    if len(big):
        lines.append('    larger:   ' + '; '.join(detail(r) for _, r in big.iterrows()))
    return '\n'.join(lines)


def plot_probe_energy_profile(ax, scan, k=None, n_probe=-1, epochs_only=False,
                              method='Sven', out_dir=None, cmap='plasma', limit=8,
                              seed=None):
    """Cumulative residual energy vs directions kept, on the PROBE set, over training.

    The offline counterpart of :func:`plot_energy_profile`, and the figure the
    truncation claim should actually be read off: the x axis runs to
    ``min(n_rows, P)``, not to ``B``, so a ``k`` that keeps ~100 % of a 32-row batch
    residual can be seen keeping much less of the real one.  The ``k`` cut is drawn
    where it falls.
    """
    import matplotlib.pyplot as plt

    ents = ct.load_spectra(probe_scan_name(scan), method=method, model_seed=seed,
                           n_probe=n_probe, epochs_only=epochs_only, out_dir=out_dir)
    if not ents:
        return None
    n_ck = min(len(e['step']) for e in ents)
    sel = subsample_rows(n_ck, limit)
    base = plt.get_cmap(cmap)
    colors, norm = progress_colors(np.asarray(ents[0]['step'])[sel] + 1.0, cmap=cmap)
    for j, i in enumerate(sel):
        cum = np.nanmean(np.stack([cumulative_energy(np.asarray(e['utr'], float)[i])
                                   for e in ents]), axis=0)
        ax.plot(np.arange(1, len(cum) + 1), cum, color=colors[j], lw=1.8,
                label=f'step {int(ents[0]["step"][i])}')
    if k:
        ax.axvline(k, color='#C44E52', lw=1.4, zorder=0, label=f'$k = {int(k)}$')
    ax.set_xscale('log')
    ax.set_xlabel('Directions kept, top $i$')
    ax.set_ylabel('Cumulative probe residual')
    ax.set_ylim(0, 1.02)
    return norm


#: axis labels for :func:`probe_metrics` columns -- one spelling per quantity
PROBE_METRIC_LABELS = {
    'sigma_max': r'$\sigma_{\max}$ (probe Jacobian)',
    'cond': r'$\sigma_{\max} / \sigma_{\min}$ (resolved part)',
    'rank': 'Numerically resolved rank',
    'eff_rank': r'Effective rank $\exp(H(\sigma))$',
    'sigma_B_over_1': r'$\sigma_B \,/\, \sigma_1$',
    'dist_init': r'$\|\theta_t - \theta_0\|$',
    'param_norm': r'$\|\theta_t\|$',
    'probe_loss': 'Probe-set loss',
    'resid_norm': r'$\|r\|$ on the probe set',
    'proj_frac': r'$\|P_U r\|^2 / \|r\|^2$',
}
