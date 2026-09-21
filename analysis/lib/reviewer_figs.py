"""WP4a -- the reviewer-facing MLP studies, in one place.

Five notebooks share this module: ``overparam_analysis`` (R1's crux, dataset-level
over-parameterisation P > N), ``batchsize_analysis`` (R2 Q1), ``kappa_analysis`` (R1's
"kappa = 1 is the theory, kappa = 2 is what runs"), ``microbatch_analysis`` and
``paramfrac_analysis`` (R1's memory ask).  Everything here is a table or an axes-level
plot; no notebook re-derives a seed mean, a divergence count or a rank by hand.

What this module does NOT do
----------------------------
It does not re-implement selection.  A configuration is chosen by
:func:`analysis_helpers.best_per_method`, i.e. by the binding rule of
``CHANGES_NEEDED.md`` section 1 -- **eligible -> fewest diverged -> seed-mean final
VALIDATION loss** -- applied *inside each arm* of the study (each ``n_data``, each batch
size, each knob value), because an arm is a separate experiment and a configuration tuned
at one N has no claim on another.  Test loss and test accuracy are OUTCOMES: they are
reported beside the selected configuration and never ranked on
(:func:`style.assert_selection_metric` enforces this one level down).

It does not re-implement the confirmation / paired / budget / timing tables either --
those are :mod:`headline`, the one implementation, and none of the studies here has a
``_confirm`` or ``_timing`` pass of its own (only the seven headline scans do).  That is
also why every wall-clock number in this module is labelled as the SCAN's own clock:
these grids ran many processes per GPU, so a second here is an upper bound contaminated
by co-tenancy, and the honest budget axes are epochs / steps / examples.

Conventions kept from the rest of the analysis
----------------------------------------------
* Diverged = failed (:func:`style.is_diverged`: recorded ``status``, a non-finite final
  value, or a final value above 10x the first): excluded from every mean and COUNTED.
  Every table carries ``finished`` / ``attempted`` and a printable ``counts`` string.
* Seed band = mean +/- 1 std (ddof=1), lower edge clipped at the lowest seed
  (:func:`style.clipped_yerr`), labelled once per axes by :func:`style.band_legend` --
  both come for free through :func:`analysis_helpers.errorbar_seeds`.
* Display names are :func:`style.method_label` (``LBFGS`` -> "Stochastic L-BFGS",
  ``SGDm`` -> "SGD + momentum"); colours are :data:`style.METHOD_COLORS` (Sven black).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import paired
import style
from analysis_helpers import (add_derived, best_per_method, config_runs, errorbar_seeds,
                              expected_run_ids, loss_curve, method_order)
from scan_analysis import PRE_TRAINING_CURVES
from style import METHOD_COLORS, final_value, method_color, method_label

__all__ = [
    'savefig', 'final_of', 'scalar_of', 'effective_rank', 'curve_epochs',
    'arm_table', 'rank_table', 'divergence_table', 'edge_optima', 'scan_census',
    'reach_epoch', 'reach_table', 'median_method_target', 'fastest_to_target',
    'reach_count_table', 'divergence_vs_reference',
    'knob_table', 'kappa_table', 'matched_step_table', 'eff_step_coverage',
    'paired_seed_diff', 'matched_step_paired', 'neighbour_gaps',
    'monotonicity_table', 'fixed_knob_cost', 'rank_vs_cap',
    'plot_arm', 'plot_knob', 'arm_ticks', 'figure_legend',
    'counts_str', 'fmt_pm',
]

#: :data:`scan_analysis.PRE_TRAINING_CURVES` -- the curves whose index 0 is the UNTRAINED
#: model, so index == epochs trained.  The ``train`` curve has no such entry (one value
#: per finished epoch), a factor of one epoch in every time-to-target number.

#: Sven's line is thicker everywhere (it is the method under test).
SVEN = 'Sven'


# ---------------------------------------------------------------------------
# Figure IO
# ---------------------------------------------------------------------------
def savefig(fig, plot_dir, stem, formats=('pdf', 'png'), **kw):
    """Save ``fig`` as ``{plot_dir}/{stem}.{pdf,png}`` and return the paths.

    The campaign wants both: the PDF goes in the paper, the PNG is what a report or a
    review thread can show.  ``set_style`` already sets ``savefig.bbox='tight'`` and
    ``savefig.dpi=300``, so the two files agree by construction.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in formats:
        path = plot_dir / f'{stem}.{ext}'
        fig.savefig(path, **kw)
        out.append(path)
    return out


# ---------------------------------------------------------------------------
# Small extractors
# ---------------------------------------------------------------------------
def final_of(row, which='val'):
    """Final value of curve ``which``, or NaN -- and NaN for a diverged / failed run.

    :func:`style.final_value` already returns NaN for a non-finite or blown-up curve;
    the status check is what makes a ``status: diverged`` record with a finite curve
    (the ``linalg`` failures of section 7) drop out of the means as well.
    """
    if bool(row.get('diverged', False)) or bool(row.get('failed', False)):
        return np.nan
    return final_value(loss_curve(row, which))


def scalar_of(row, key):
    """A scalar out of the record's ``losses`` dict (``avg_batch_time_train``,
    ``peak_gpu_mem_mb``, ...), or NaN."""
    L = row.get('losses')
    if not isinstance(L, dict):
        return np.nan
    v = L.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def effective_rank(row):
    """Mean number of singular values actually kept per step, from ``svd_summary``.

    ``k`` is a CAP; ``rtol`` cuts below it, so the rank the update really used is
    ``num_nonzero_svs_epoch`` -- the quantity a "k / B fraction" claim is about.  NaN
    for a baseline run or a scan that logged no summary.
    """
    s = row.get('svd_summary')
    if not isinstance(s, dict):
        return np.nan
    v = s.get('num_nonzero_svs_epoch')
    if v is None or not len(v):
        return np.nan
    return float(np.nanmean(np.asarray(v, dtype=float)))


def curve_epochs(row, which):
    """Number of epochs the value at index ``i`` of curve ``which`` corresponds to,
    as the function ``i -> epochs``.

    Decided by LENGTH against the run's own ``train`` curve, not by the key: schema 2
    records partial curves for runs that stopped early, and the pre-training entry is
    the difference between "97 epochs to the target" and "98".
    """
    curve = loss_curve(row, which)
    train = loss_curve(row, 'train')
    if curve is None:
        return None
    n_ep = len(train) if train is not None else None
    if n_ep is not None and len(curve) == n_ep + 1:
        return lambda i: i
    if n_ep is not None and len(curve) == n_ep:
        return lambda i: i + 1
    return (lambda i: i) if which in PRE_TRAINING_CURVES else (lambda i: i + 1)


def counts_str(finished, attempted):
    """``'5/5'`` -- the finished / attempted every table in the campaign shows."""
    try:
        return f'{int(finished)}/{int(attempted)}'
    except (TypeError, ValueError):
        return '--'


def fmt_pm(mean, std, spec='.3g'):
    """``mean ± std`` (the seed band, :func:`style.seed_spread_label`)."""
    if mean is None or not np.isfinite(mean):
        return '--'
    if std is None or not np.isfinite(std):
        return f'{mean:{spec}}'
    return f'{mean:{spec}} ± {std:{spec}}'


# ---------------------------------------------------------------------------
# (a) per-arm selection, with the outcomes beside it
# ---------------------------------------------------------------------------
#: Extra per-run CURVES aggregated over the seeds of the selected configuration.
#: ``final_train_eval`` is the full-training-set loss (schema 2's ``train_eval``): the
#: number that says whether a method INTERPOLATED, which the running ``train`` curve --
#: an average over minibatches visited while the parameters moved -- does not.
EXTRA_CURVES = {'final_train_eval': 'train_eval'}
#: Extra per-run SCALARS out of ``losses``.
EXTRA_SCALARS = {'epoch_s': 'avg_epoch_time', 'step_s': 'avg_batch_time_train',
                 'train_s': 'avg_train_time'}


def _agg(values):
    """``(mean, std(ddof=1), min, n)`` over the finite entries of ``values``."""
    v = np.asarray([x for x in values], dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return np.nan, np.nan, np.nan, 0
    return (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0,
            float(v.min()), int(v.size))


_LABEL_KNOBS = {'lr': 'lr', 'k': 'k', 'rtol': 'rtol', 'kappa': 'kappa',
                'microbatch_size': 'mb', 'param_fraction': 'pf',
                'lbfgs_max_iter': 'mi', 'lbfgs_history_size': 'hs',
                'polyak_max_lr': 'lr_max', 'weight_decay': 'wd'}


def _config_label(row, knobs=('lr', 'k', 'rtol', 'lbfgs_max_iter', 'lbfgs_history_size',
                              'polyak_max_lr')):
    """A short, readable label for the configuration a row selected."""
    bits = []
    for c in knobs:
        v = row.get(c)
        if v is None or pd.isna(v):
            continue
        name = _LABEL_KNOBS.get(c, c)
        bits.append(f'{name}={v:g}' if isinstance(v, (int, float, np.number)) else f'{name}={v}')
    return ', '.join(bits) or '(no free knob)'


def arm_table(df, arm, metric='final_val_loss', extras=True):
    """Best configuration per (method, arm), with its outcomes -- the study workhorse.

    ``arm`` is the study axis (``'n_data'``, ``'batch_size'``, ``'param_fraction'``, ...)
    or a list of them.  Selection is :func:`analysis_helpers.best_per_method` *within*
    each arm, i.e. the full binding rule; ``metric`` may not be a test quantity.

    On top of the ``config_table`` columns (seed mean / ``_std`` / ``_min`` of
    ``final_val_loss``, ``final_train_loss``, ``final_val_acc``, ``final_test_loss``,
    ``final_test_acc``, ``total_time``, ``peak_gpu_mem_mb``) each row gains

    * ``final_train_eval`` (+ ``_std`` / ``_min``) -- the full-train-set loss, for
      "did it interpolate";
    * ``epoch_s`` / ``step_s`` / ``train_s`` -- the run's own recorded epoch, per-step
      and per-epoch training times (the SCAN clock: co-tenancy inflated, see the module
      docstring);
    * ``rank_eff`` -- the singular values actually kept per step (Sven rows only);
    * ``display`` (:func:`style.method_label`), ``config`` (a printable label) and
      ``counts`` (``finished/attempted``).

    The extras are aggregated over the FINISHED seeds of the selected configuration
    (``config_runs``), so they are the same population as the seed means beside them.
    """
    arms = [arm] if isinstance(arm, str) else list(arm)
    best = best_per_method(df, by=metric, extra_group=arms)
    if best.empty:
        return best
    best = best.copy()
    best['display'] = best['method'].map(method_label)
    best['config'] = [_config_label(r) for _, r in best.iterrows()]
    best['counts'] = [counts_str(r['finished'], r['attempted']) for _, r in best.iterrows()]
    if not extras:
        return best.reset_index(drop=True)
    rows = []
    for _, r in best.iterrows():
        runs = config_runs(df, r)
        out = {}
        for col, which in EXTRA_CURVES.items():
            m, s, lo, n = _agg([final_of(rr, which) for _, rr in runs.iterrows()])
            out[col], out[f'{col}_std'], out[f'{col}_min'], out[f'{col}_n'] = m, s, lo, n
        for col, key in EXTRA_SCALARS.items():
            m, s, lo, _ = _agg([scalar_of(rr, key) for _, rr in runs.iterrows()])
            out[col], out[f'{col}_std'], out[f'{col}_min'] = m, s, lo
        m, s, lo, _ = _agg([effective_rank(rr) for _, rr in runs.iterrows()])
        out['rank_eff'], out['rank_eff_std'], out['rank_eff_min'] = m, s, lo
        # the budget axes: steps and examples actually stepped on (drop_last=True)
        spe = pd.to_numeric(runs.get('steps_per_epoch'), errors='coerce')
        bs = pd.to_numeric(runs.get('batch_size'), errors='coerce')
        ne = pd.to_numeric(runs.get('num_epochs'), errors='coerce')
        out['steps_per_epoch'] = float(spe.dropna().iloc[0]) if spe is not None and spe.notna().any() else np.nan
        out['batch'] = float(bs.dropna().iloc[0]) if bs is not None and bs.notna().any() else np.nan
        out['epochs'] = float(ne.dropna().iloc[0]) if ne is not None and ne.notna().any() else np.nan
        out['total_steps'] = out['steps_per_epoch'] * out['epochs']
        out['total_examples'] = out['total_steps'] * out['batch']
        rows.append(out)
    best = pd.concat([best.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    best.attrs['selection_metric'] = metric
    best.attrs['arms'] = arms
    return best


def rank_table(best, arm, value='final_val_loss', higher_is_better=False):
    """Rank of every method inside each arm, on ``value``.

    Ranks are ``method='min'`` (ties share the better rank) over the arm's FINITE
    values only, and ``n_methods`` is the field the rank is out of -- a method with no
    eligible configuration in that arm simply is not in the field, which is a fact about
    the arm, not a missing value to impute.
    """
    arms = [arm] if isinstance(arm, str) else list(arm)
    out = []
    for key, sub in best.groupby(arms, dropna=False, sort=True):
        v = pd.to_numeric(sub[value], errors='coerce')
        rank = v.rank(method='min', ascending=not higher_is_better)
        n = int(np.isfinite(v).sum())
        key = key if isinstance(key, tuple) else (key,)
        for (_, r), rk in zip(sub.iterrows(), rank):
            row = dict(zip(arms, key))
            row.update({'method': r['method'], 'display': r['display'],
                        'value': r[value], 'rank': rk, 'n_methods': n,
                        'counts': r.get('counts'), 'config': r.get('config')})
            out.append(row)
    return pd.DataFrame(out)


def divergence_table(df, arm, methods=None):
    """Divergence rate per (method, arm) over EVERY run in the arm, not just the winner.

    This is the robustness statement: ``n_diverged`` uses the wide analysis rule
    (:func:`style.is_diverged` -- recorded status, non-finite, or > 10x the first value),
    which on Sven is the only rule that sees anything at all (``status: diverged`` is 0
    for Sven outside the param-fraction scans; EXPERIMENTS.md section 7).  ``frac`` is
    over the records present; ``n_failed`` counts ``oom`` / ``error`` records, which are
    incomplete attempts and NOT divergences.
    """
    arms = [arm] if isinstance(arm, str) else list(arm)
    d = df if 'diverged' in df.columns else add_derived(df)
    if methods is not None:
        d = d[d['method'].isin(methods)]
    g = d.groupby(['method'] + arms, dropna=False)
    out = g.agg(n_runs=('run_id', 'size'), n_diverged=('diverged', 'sum'),
                n_failed=('failed', 'sum')).reset_index()
    out['n_diverged'] = out['n_diverged'].astype(int)
    out['n_failed'] = out['n_failed'].astype(int)
    out['frac'] = out['n_diverged'] / out['n_runs'].clip(lower=1)
    out['display'] = out['method'].map(method_label)
    return out.sort_values(['method'] + arms).reset_index(drop=True)


def edge_optima(df, best, arm, knobs=('lr', 'k', 'rtol')):
    """Selected configurations whose knob sits on an EDGE of the grid it was searched on.

    The grid is taken per (method, arm) from the runs actually present, so an optimum is
    "on the edge" only against the values that method really had available there.  A
    single-valued knob is not an edge (there was nothing to choose), and the returned
    ``side`` is ``'low'`` / ``'high'``.  The point of the table: an edge optimum means
    the grid, not the method, may be what is being measured.
    """
    arms = [arm] if isinstance(arm, str) else list(arm)
    rows = []
    for _, r in best.iterrows():
        mask = (df['method'] == r['method'])
        for a in arms:
            mask &= (df[a] == r[a]) if not pd.isna(r[a]) else df[a].isna()
        sub = df[mask]
        for knob in knobs:
            if knob not in sub.columns:
                continue
            vals = np.sort(pd.to_numeric(sub[knob], errors='coerce').dropna().unique())
            v = pd.to_numeric(pd.Series([r.get(knob)]), errors='coerce').iloc[0]
            if len(vals) < 2 or not np.isfinite(v):
                continue
            side = 'low' if np.isclose(v, vals[0]) else ('high' if np.isclose(v, vals[-1]) else None)
            if side is None:
                continue
            row = {a: r[a] for a in arms}
            row.update({'method': r['method'], 'display': r['display'], 'knob': knob,
                        'value': float(v), 'side': side, 'n_values': len(vals),
                        'grid': f'[{vals[0]:g} .. {vals[-1]:g}]'})
            rows.append(row)
    out = pd.DataFrame(rows)
    return out if out.empty else out.sort_values(['knob'] + arms + ['method']).reset_index(drop=True)


def scan_census(df, name=None):
    """One line of provenance for a scan: records, manifest, divergences, GPU types.

    ``n_missing`` is the manifest's run_ids that produced no record at all, so a table
    below it can say ``finished/attempted`` without the reader wondering what attempted
    counted.  ``gpus`` matters because the scans ran mostly on MIG A100-40GB slices --
    several optimizers are not bit-reproducible across GPU types (EXPERIMENTS.md
    section 8), so no claim here is a bit-reproducibility claim.
    """
    d = df if 'diverged' in df.columns else add_derived(df)
    expected = expected_run_ids(d)
    present = set(d['run_id'])
    gpus = d['gpu_name'].value_counts().to_dict() if 'gpu_name' in d.columns else {}
    return {
        'scan': name or (d['_scan'].iloc[0] if '_scan' in d.columns and len(d) else None),
        'n_records': len(d), 'n_expected': len(expected) or len(d),
        'n_missing': len(expected - present),
        'n_diverged': int(d['diverged'].sum()), 'n_failed': int(d['failed'].sum()),
        'n_methods': d['method'].nunique(), 'seeds': sorted(d['model_seed'].dropna().unique().tolist()),
        'gpus': gpus,
    }


# ---------------------------------------------------------------------------
# (b) time to target
# ---------------------------------------------------------------------------
def reach_epoch(row, target, which='val'):
    """Epochs trained before curve ``which`` first reached ``target``, or None.

    None has two meanings the caller must keep apart -- the run diverged (no trajectory
    to ask) or it trained to the end and never got there -- which is why
    :func:`reach_table` counts them separately.
    """
    if bool(row.get('diverged', False)) or bool(row.get('failed', False)):
        return None
    curve = loss_curve(row, which)
    if curve is None:
        return None
    to_epochs = curve_epochs(row, which)
    for i, v in enumerate(curve):
        if v is not None and np.isfinite(v) and float(v) <= target:
            return int(to_epochs(i))
    return None


def _seconds_to(row, epochs, key='train_times'):
    """Cumulative recorded seconds over the first ``epochs`` epochs of a run."""
    L = row.get('losses')
    if not isinstance(L, dict):
        return np.nan
    series = L.get(key)
    if series is None:
        return np.nan
    if epochs <= 0:
        return 0.0
    return float(np.nansum(np.asarray(series[:int(epochs)], dtype=float)))


def reach_table(df, best, arm, targets, which='val'):
    """Epochs / steps / examples / seconds to a target, per (method, arm).

    ``targets`` is either a number (one target everywhere) or a mapping ``arm value ->
    target``; ``which`` is the curve the target is on -- ``'train_eval'`` for the paper's
    "wall-time to train loss < 1e-3" claim (the full-train-set loss, not the running
    minibatch average), ``'val'`` for the selection metric.

    The row is the SELECTED configuration of :func:`arm_table`, so this is
    "how fast is the configuration you would actually ship", and

    * ``n_reached`` / ``n_never_reached`` / ``n_diverged`` are reported separately and
      never summed into one "failed" number;
    * ``sync_s`` is the run's own synchronised TRAINING time to that epoch
      (``train_times``, evaluation excluded) -- the least contaminated clock these scans
      have, and still an upper bound (many processes shared each GPU);
    * ``epoch_s_cum`` adds evaluation back in (``epoch_times``), for scale.
    """
    arms = [arm] if isinstance(arm, str) else list(arm)
    rows = []
    for _, r in best.iterrows():
        key = r[arms[0]] if len(arms) == 1 else tuple(r[a] for a in arms)
        target = float(targets[key]) if isinstance(targets, dict) else float(targets)
        # `config_runs` keeps only the FINISHED runs; the diverged seeds of the same
        # configuration have to be counted too, so rebuild the configuration's runs from
        # the full arm by matching the hyperparameter VALUES
        mask = (df['method'] == r['method'])
        for a in arms:
            mask &= (df[a] == r[a]) if not pd.isna(r[a]) else df[a].isna()
        for c in ('lr', 'k', 'rtol', 'lbfgs_max_iter', 'lbfgs_history_size',
                  'weight_decay', 'microbatch_size', 'param_fraction', 'kappa'):
            if c not in df.columns or c not in r.index:
                continue
            v = r[c]
            mask &= df[c].isna() if pd.isna(v) else (df[c] == v)
        all_runs = df[mask]
        ep, st, ex, sy, ec = [], [], [], [], []
        n_never = n_div = 0
        for _, rr in all_runs.iterrows():
            e = reach_epoch(rr, target, which)
            if e is None:
                if bool(rr.get('diverged', False)) or bool(rr.get('failed', False)):
                    n_div += 1
                else:
                    n_never += 1
                continue
            spe = float(rr['steps_per_epoch']) if pd.notna(rr.get('steps_per_epoch')) else np.nan
            bs = float(rr['batch_size']) if pd.notna(rr.get('batch_size')) else np.nan
            ep.append(e)
            st.append(e * spe)
            ex.append(e * spe * bs)
            sy.append(_seconds_to(rr, e, 'train_times'))
            ec.append(_seconds_to(rr, e, 'epoch_times'))
        row = {a: r[a] for a in arms}
        row.update({'method': r['method'], 'display': r['display'],
                    'config': r.get('config'), 'target': target, 'which': which,
                    'n_runs': len(all_runs), 'n_reached': len(ep),
                    'n_never_reached': n_never, 'n_diverged': n_div,
                    'all_reached': bool(len(ep) and len(ep) == len(all_runs))})
        for col, vals in (('epochs', ep), ('steps', st), ('examples', ex),
                          ('sync_s', sy), ('epoch_s_cum', ec)):
            m, s, lo, _ = _agg(vals)
            row[col], row[f'{col}_std'], row[f'{col}_min'] = m, s, lo
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(arms + ['epochs'], kind='mergesort',
                              na_position='last').reset_index(drop=True)
    out.attrs['which'] = which
    return out


def median_method_target(best, column='final_val_loss', mult=1.0, arm=None):
    """Per-arm target = ``mult`` x the MEDIAN over methods of ``column``.

    The median and not the best, for the reason :func:`headline.target_reference` gives:
    a target set by the winner is a target only the winner reaches, and one set by the
    loser is met at epoch 1 by everybody.  Per arm, because the arms of these studies are
    different problems (a different N, a different batch size) whose losses live on
    different scales.  ``arm=None`` returns one number for the whole table.
    """
    v = pd.to_numeric(best[column], errors='coerce')
    if arm is None:
        v = v[np.isfinite(v)]
        return float(np.median(v)) * mult if len(v) else np.nan
    arms = [arm] if isinstance(arm, str) else list(arm)
    out = {}
    for key, sub in best.groupby(arms, dropna=False, sort=True):
        vv = pd.to_numeric(sub[column], errors='coerce')
        vv = vv[np.isfinite(vv)]
        key = key[0] if isinstance(key, tuple) and len(key) == 1 else key
        out[key] = float(np.median(vv)) * mult if len(vv) else np.nan
    return out


def fastest_to_target(df, arm, targets, which='train_eval', methods=None,
                      require_all_seeds=True):
    """The legacy question: per (method, arm), the configuration that reaches the target
    FASTEST -- not the one the validation loss selects.

    The rebuttal's "Sven is consistently 2nd-fastest for P/N > 1" was computed this way
    (best hyperparameter config per optimizer, ranked by wall-time to train loss < 1e-3),
    so reproducing it needs the same question asked of the fresh runs.  It is NOT the
    binding rule and must never be presented as a headline number: choosing a
    configuration by how fast it hits the target is selection on the very quantity
    reported.  ``require_all_seeds`` keeps only configurations where EVERY run present
    reached the target, so a config that gets there on 1 lucky seed of 5 cannot win.
    """
    arms = [arm] if isinstance(arm, str) else list(arm)
    d = df if 'diverged' in df.columns else add_derived(df)
    if methods is not None:
        d = d[d['method'].isin(methods)]
    cfg_cols = [c for c in ('lr', 'k', 'rtol', 'weight_decay', 'lbfgs_max_iter',
                            'lbfgs_history_size', 'kappa', 'microbatch_size',
                            'param_fraction') if c in d.columns]
    rows = []
    for key, sub in d.groupby(['method'] + arms + cfg_cols, dropna=False, sort=False):
        method, arm_vals = key[0], key[1:1 + len(arms)]
        target = float(targets[arm_vals[0] if len(arms) == 1 else arm_vals]) \
            if isinstance(targets, dict) else float(targets)
        ep, sy = [], []
        n_ok = 0
        for _, rr in sub.iterrows():
            e = reach_epoch(rr, target, which)
            if e is None:
                continue
            n_ok += 1
            ep.append(e)
            sy.append(_seconds_to(rr, e, 'train_times'))
        if not n_ok or (require_all_seeds and n_ok != len(sub)):
            continue
        row = dict(zip(arms, arm_vals))
        row.update({'method': method, 'display': method_label(method),
                    'target': target, 'n_runs': len(sub), 'n_reached': n_ok,
                    'epochs': float(np.mean(ep)), 'sync_s': float(np.nanmean(sy)),
                    'config': ', '.join(f'{c}={v:g}' for c, v in
                                        zip(cfg_cols, key[1 + len(arms):])
                                        if isinstance(v, (int, float, np.number))
                                        and np.isfinite(v))})
        rows.append(row)
    cols = arms + ['method', 'display', 'target', 'n_runs', 'n_reached', 'epochs',
                   'sync_s', 'config']
    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out
    out = out.sort_values(['method'] + arms + ['sync_s'], kind='mergesort')
    out = out.groupby(['method'] + arms, dropna=False, sort=False).head(1)
    return out.sort_values(arms + ['sync_s']).reset_index(drop=True)


def reach_count_table(df, arm, targets, which='train_eval', methods=None,
                      require_all_seeds=True):
    """How MANY methods reach the target in each arm, and out of how big a field.

    The companion of :func:`fastest_to_target`: that function answers "who, and how
    fast", this one answers "how many of the field", which is the claim a sentence like
    "nothing reaches 1e-3 on this task" makes.  Computing it beats asserting it -- the two
    prose blocks of a notebook cannot then disagree with each other.  ``n_field`` is the
    methods PRESENT in the arm (not the eligible ones: a method whose every run diverged
    did not reach the target either).
    """
    arms = [arm] if isinstance(arm, str) else list(arm)
    f = fastest_to_target(df, arms, targets, which=which, methods=methods,
                          require_all_seeds=require_all_seeds)
    d = df if methods is None else df[df['method'].isin(methods)]
    rows = []
    for key, sub in d.groupby(arms, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        hit = f
        for a, v in zip(arms, key):
            hit = hit[hit[a] == v] if (not f.empty and not pd.isna(v)) else hit
        row = dict(zip(arms, key))
        row.update({'n_field': int(sub['method'].nunique()),
                    'n_reached': 0 if f.empty else int(len(hit)),
                    'methods': [] if f.empty else sorted(hit['display'].tolist())})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs['which'] = which
    return out


def divergence_vs_reference(df, reference=SVEN, methods=None):
    """Wide-rule divergence rate per method over a WHOLE scan, ranked, with the reference
    method's place in that ranking and its grid size relative to the others.

    A pooled "Sven diverges less than X" sentence is only honest with two extra facts on
    the page: it is a per-SCAN statement (pooling scans of different difficulty hides the
    sign), and these grids are not the same size -- Sven's spans the ``rtol`` axis the
    baselines do not have, and that axis deliberately includes values known to break
    (EXPERIMENTS.md section 7).  Both come back as columns: ``n_runs`` and
    ``grid_ratio_vs_reference``.
    """
    d = df if 'diverged' in df.columns else add_derived(df)
    if methods is not None:
        d = d[d['method'].isin(methods)]
    t = (d.groupby('method')['diverged'].agg(n_diverged='sum', n_runs='size')
         .reset_index())
    t['n_diverged'] = t['n_diverged'].astype(int)
    t['frac'] = t['n_diverged'] / t['n_runs'].clip(lower=1)
    t['display'] = t['method'].map(method_label)
    t = t.sort_values('frac', ascending=False).reset_index(drop=True)
    t['rank_worst_first'] = t['frac'].rank(method='min', ascending=False).astype(int)
    ref = t[t['method'] == reference]
    if len(ref):
        t['grid_ratio_vs_reference'] = t['n_runs'] / float(ref['n_runs'].iloc[0])
        t['worse_than_reference'] = t['frac'] > float(ref['frac'].iloc[0])
        t.attrs['reference'] = reference
        t.attrs['reference_frac'] = float(ref['frac'].iloc[0])
        t.attrs['reference_rank'] = int(ref['rank_worst_first'].iloc[0])
        t.attrs['above_reference'] = sorted(
            t.loc[t['worse_than_reference'] & (t['method'] != reference), 'display'])
        t.attrs['below_reference'] = sorted(
            t.loc[~t['worse_than_reference'] & (t['method'] != reference), 'display'])
    return t


# ---------------------------------------------------------------------------
# (b2) is the gap bigger than the seeds?  and is a curve really monotone?
# ---------------------------------------------------------------------------
def paired_seed_diff(sub_a, sub_b, metric='final_val_loss', seed_col='model_seed',
                     level=0.95):
    """Paired-by-seed difference ``a - b`` between two sub-frames of RUNS.

    The same statistic :func:`paired.paired_difference` computes for two rows of a
    configuration table, but taken from two arbitrary run selections -- which is what a
    comparison between two cells of the SAME method needs (two ``kappa`` at a matched
    effective step, two micro-batch sizes): there is no "other method" row to pass.
    Pairing is what makes it worth doing at all: the seeds of these scans fix the
    initialisation and the data order, so the same seed on both sides removes exactly the
    variance the seed band shows.

    Diverged / failed runs are dropped first (their final value is NaN), a seed with
    several runs is averaged, and only the seeds BOTH sides finished are paired --
    ``n_a_only`` / ``n_b_only`` say how many were dropped for that reason.  ``t`` and
    ``p`` are the two-sided one-sample t statistic on the differences; with 5 seeds and
    no multiplicity correction they are a readability aid, not a test to quote.
    """
    style.assert_selection_metric(metric, 'paired_seed_diff')

    def take(sub):
        s = sub if 'diverged' in sub.columns else add_derived(sub)
        ok = s[~s['diverged'].astype(bool) & ~s['failed'].astype(bool)]
        v = pd.to_numeric(ok[metric], errors='coerce') if metric in ok.columns else None
        if v is None:
            v = pd.Series([final_of(r, 'val') for _, r in ok.iterrows()], index=ok.index)
        g = pd.DataFrame({seed_col: ok[seed_col], 'v': v})
        g = g[np.isfinite(g['v'])]
        return g.groupby(seed_col, dropna=False)['v'].mean()

    a, b = take(sub_a), take(sub_b)
    shared = sorted(set(a.index) & set(b.index))
    d = np.asarray([a[s] - b[s] for s in shared], dtype=float)
    n = d.size
    mean = float(d.mean()) if n else np.nan
    sd = float(d.std(ddof=1)) if n > 1 else np.nan
    sem = sd / np.sqrt(n) if n > 1 else np.nan
    t = mean / sem if (n > 1 and sem) else np.nan
    p = float(2 * stats.t.sf(abs(t), n - 1)) if np.isfinite(t) else np.nan
    half = float(stats.t.ppf(0.5 + level / 2, n - 1) * sem) if n > 1 else np.nan
    return {'n': n, 'mean': mean, 'std': sd, 'sem': sem, 't': t, 'p': p,
            'ci_low': mean - half, 'ci_high': mean + half, 'level': level,
            'mean_a': float(np.mean([a[s] for s in shared])) if n else np.nan,
            'mean_b': float(np.mean([b[s] for s in shared])) if n else np.nan,
            'significant': bool(np.isfinite(half) and abs(mean) > abs(half)),
            'n_a_only': int(len(set(a.index) - set(b.index))),
            'n_b_only': int(len(set(b.index) - set(a.index)))}


def neighbour_gaps(df, best, arm, reference=SVEN, metric='final_val_loss', n_rivals=2,
                   level=0.95):
    """Per arm: the reference method's rank, and the PAIRED gap to its nearest rivals.

    A rank is a point estimate over seed means; with 5 seeds a rank-1 placement can sit
    entirely inside the seed spread.  This table puts the two side by side -- the rank,
    and :func:`paired.paired_difference` between the reference's selected configuration
    and each of the ``n_rivals`` configurations nearest to it in the arm's ordering (the
    ones the rank is actually decided against).  ``mean`` is ``reference - rival``, so a
    negative mean means the reference is better; ``significant`` is "the t-interval
    excludes 0".
    """
    arms = [arm] if isinstance(arm, str) else list(arm)
    style.assert_selection_metric(metric, 'neighbour_gaps')
    rows = []
    for key, sub in best.groupby(arms, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        s = sub.sort_values(metric).reset_index(drop=True)
        where = s.index[s['method'] == reference].tolist()
        if not where:
            continue
        i = where[0]
        neigh = [j for j in (i - 1, i + 1, i - 2, i + 2)
                 if 0 <= j < len(s) and j != i][:n_rivals]
        for j in sorted(neigh):
            st = paired.paired_difference(df, s.loc[i], s.loc[j], metric=metric,
                                          level=level)
            row = dict(zip(arms, key))
            row.update({
                'reference': reference, 'rank': i + 1, 'n_field': len(s),
                'rival': s.loc[j, 'method'], 'rival_display': s.loc[j, 'display'],
                'rival_rank': j + 1, 'ref_value': s.loc[i, metric],
                'rival_value': s.loc[j, metric], 'n': st['n'], 'mean': st['mean'],
                'std': st['std'], 'sem': st['sem'],
                't': st['mean'] / st['sem'] if st['sem'] else np.nan,
                'ci_low': st['ci_low'], 'ci_high': st['ci_high'],
                'significant': bool(np.isfinite(st['ci_low'])
                                    and (st['ci_low'] > 0 or st['ci_high'] < 0))})
            rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs['metric'] = metric
    out.attrs['interval'] = paired.interval_label(level)
    return out


def monotonicity_table(best, x, q, group='method', ascending=True):
    """Is ``q`` monotone in ``x`` for each ``group``, and where does its optimum sit?

    "Every method improves monotonically along this axis" is a claim about 13 curves and
    is cheap to check, so it is checked rather than asserted.  ``x`` is sorted ascending
    (``ascending=False`` reverses it, e.g. to read a batch-size axis as a steps axis);
    ``falling`` means ``q`` never rises as ``x`` grows, ``rising`` the converse, and
    ``monotone`` is either of them.  Only the finite values of a group are used, and
    ``n`` says how many points that left.
    """
    rows = []
    for g, sub in best.groupby(group, dropna=False, sort=True):
        s = sub.copy()
        s['_x'] = pd.to_numeric(s[x], errors='coerce')
        s['_q'] = pd.to_numeric(s[q], errors='coerce')
        s = s[np.isfinite(s['_x']) & np.isfinite(s['_q'])].sort_values(
            '_x', ascending=ascending)
        v = s['_q'].to_numpy(dtype=float)
        if not v.size:
            continue
        d = np.diff(v)
        rows.append({group: g,
                     'display': (s['display'].iloc[0] if 'display' in s.columns
                                 else method_label(str(g))),
                     'n': int(v.size),
                     'falling': bool(np.all(d <= 0)), 'rising': bool(np.all(d >= 0)),
                     'monotone': bool(np.all(d <= 0) or np.all(d >= 0)),
                     'first': float(v[0]), 'last': float(v[-1]),
                     'best': float(v.min()), 'worst': float(v.max()),
                     'x_first': float(s['_x'].iloc[0]), 'x_last': float(s['_x'].iloc[-1]),
                     'x_at_best': float(s['_x'].iloc[int(np.argmin(v))]),
                     'range_ratio': float(v.max() / v.min()) if v.min() > 0 else np.nan})
    out = pd.DataFrame(rows)
    out.attrs['x'] = x
    out.attrs['q'] = q
    return out


def fixed_knob_cost(df, arm, knob, key='avg_batch_time_train', method=SVEN, scale=1.0):
    """A ``losses`` scalar per (``knob`` value, ``arm`` value) for ONE method.

    The confound-free version of a cost-vs-arm curve.  Reading a cost off the per-arm
    SELECTED configuration mixes two effects, because the selection can jump to a
    different knob value between arms -- and on Sven the ``rtol`` value alone changes the
    step time by about a factor of two.  Holding the knob fixed and letting only the arm
    move separates them.  ``value`` is the mean over the cell's FINISHED runs (diverged /
    failed ones carry no meaningful timing and are excluded, and counted in
    ``n_diverged``); ``n_runs`` is the whole cell, ``n`` the runs the mean used, and
    ``scale`` multiplies the value (``1e3`` for ms).
    """
    d = df if 'diverged' in df.columns else add_derived(df)
    d = d[d['method'] == method]
    arms = [arm] if isinstance(arm, str) else list(arm)
    rows = []
    for gkey, sub in d.groupby([knob] + arms, dropna=False, sort=True):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        ok = sub[~sub['diverged'].astype(bool) & ~sub['failed'].astype(bool)]
        m, s, lo, n = _agg([scalar_of(r, key) for _, r in ok.iterrows()])
        row = dict(zip([knob] + arms, gkey))
        row.update({'value': m * scale, 'value_std': s * scale, 'value_min': lo * scale,
                    'n': n, 'n_runs': len(sub), 'n_diverged': int(len(sub) - len(ok))})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs['key'] = key
    out.attrs['method'] = method
    return out


def rank_vs_cap(tbl, knob, cap, rank_col='rank_eff'):
    """The rank the update used against the rank the KNOB alone would allow.

    ``cap`` is the arithmetic cap -- ``B / microbatch_size``, ``k``, ``B`` -- and
    ``rank_eff`` is what the run recorded.  The two agree only where the knob is the
    binding constraint: ``rtol`` cuts independently, so the honest statement is
    ``rank = min(rtol-limited rank, cap)``.  ``matches`` is the per-row test at 1% and
    ``attrs['all_match']`` the claim "falls exactly as advertised".
    """
    t = tbl.copy()
    t['cap'] = pd.to_numeric(t[knob], errors='coerce').map(
        cap if callable(cap) else (lambda v: cap / v))
    t['rank_used'] = pd.to_numeric(t[rank_col], errors='coerce')
    t['ratio'] = t['rank_used'] / t['cap']
    t['matches'] = np.isclose(t['rank_used'], t['cap'], rtol=0.01)
    t['cap_binds'] = t['rank_used'] <= t['cap'] * 1.01
    out = t[[knob, 'cap', 'rank_used', 'ratio', 'matches', 'cap_binds']].copy()
    out.attrs['all_match'] = bool(out['matches'].all())
    out.attrs['n_match'] = int(out['matches'].sum())
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# (c) the Sven-only knob studies: micro-batch and parameter fraction
# ---------------------------------------------------------------------------
def knob_table(df, knob, lr=None, metric='final_val_loss'):
    """Seed statistics per (knob value, lr) for a Sven-only ablation scan.

    Columns: the seed mean / std / min of ``final_val_loss``, ``final_test_loss``,
    ``final_val_acc``, ``final_test_acc`` and ``final_train_eval``; ``step_s`` (measured
    seconds per optimizer step, ``avg_batch_time_train``), ``epoch_s``, ``peak_mem_mb``,
    ``rank_eff`` (singular values actually kept); ``actual_param_fraction`` -- the
    fraction the mask really hit, which is NOT the requested one (elementwise masking of
    a 593-parameter MLP cannot hit 0.1 exactly: it lands on 0.09949); and the counts
    ``finished`` / ``n_diverged`` / ``n_runs``.

    ``lr`` restricts to one learning rate (the notebooks pin the reference-value optimum
    so a figure's lines differ only in the knob); ``lr=None`` keeps them all.
    """
    d = df if 'diverged' in df.columns else add_derived(df)
    if lr is not None:
        d = d[d['lr'] == lr]
    rows = []
    for key, sub in d.groupby([knob, 'lr'], dropna=False, sort=True):
        ok = sub[~sub['diverged'] & ~sub['failed']]
        row = {knob: key[0], 'lr': key[1], 'n_runs': len(sub),
               'finished': len(ok), 'n_diverged': int(sub['diverged'].sum()),
               'n_failed': int(sub['failed'].sum())}
        row['counts'] = counts_str(len(ok), len(sub))
        for col, which in (('final_val_loss', 'val'), ('final_test_loss', 'test'),
                           ('final_val_acc', 'val_acc'), ('final_test_acc', 'test_acc'),
                           ('final_train_eval', 'train_eval')):
            m, s, lo, n = _agg([final_of(rr, which) for _, rr in ok.iterrows()])
            row[col], row[f'{col}_std'], row[f'{col}_min'], row[f'{col}_n'] = m, s, lo, n
        for col, key2 in (('step_s', 'avg_batch_time_train'), ('epoch_s', 'avg_epoch_time'),
                          ('peak_mem_mb', 'peak_gpu_mem_mb'), ('total_time', 'total_time')):
            m, s, lo, _ = _agg([scalar_of(rr, key2) for _, rr in ok.iterrows()])
            row[col], row[f'{col}_std'], row[f'{col}_min'] = m, s, lo
        m, s, lo, _ = _agg([effective_rank(rr) for _, rr in ok.iterrows()])
        row['rank_eff'], row['rank_eff_std'], row['rank_eff_min'] = m, s, lo
        if 'actual_param_fraction' in sub.columns:
            m, s, lo, _ = _agg(pd.to_numeric(sub['actual_param_fraction'], errors='coerce'))
            row['actual_param_fraction'] = m
            row['actual_param_fraction_std'] = s
            row['actual_param_fraction_min'] = lo
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs['knob'] = knob
    out.attrs['metric'] = metric
    return out


# ---------------------------------------------------------------------------
# (d) kappa: is it more than a learning-rate rescaling?
# ---------------------------------------------------------------------------
def kappa_table(df):
    """Seed statistics per (k, kappa, lr) with the EFFECTIVE step ``2 lr / kappa``.

    In the untruncated, full-row-rank solve the kappa update is exactly ``2/kappa``
    times the kappa = 2 update, so a kappa sweep at fixed lr is a learning-rate sweep in
    disguise; the scan's lr list is chosen so that three effective steps (0.25, 0.5, 1.0)
    are realised by all three kappas (EXPERIMENTS.md section 3.4).  ``matched`` marks the
    rows that sit on one of those shared effective steps.
    """
    d = df if 'diverged' in df.columns else add_derived(df)
    d = d.copy()
    d['eff_step'] = 2.0 * pd.to_numeric(d['lr'], errors='coerce') / pd.to_numeric(d['kappa'], errors='coerce')
    rows = []
    for (k, kappa, lr), sub in d.groupby(['k', 'kappa', 'lr'], dropna=False, sort=True):
        ok = sub[~sub['diverged'] & ~sub['failed']]
        row = {'k': k, 'kappa': kappa, 'lr': lr, 'eff_step': 2.0 * lr / kappa,
               'n_runs': len(sub), 'finished': len(ok),
               'n_diverged': int(sub['diverged'].sum()),
               'counts': counts_str(len(ok), len(sub))}
        for col, which in (('final_val_loss', 'val'), ('final_test_loss', 'test'),
                           ('final_val_acc', 'val_acc'), ('final_test_acc', 'test_acc'),
                           ('final_train_eval', 'train_eval')):
            m, s, lo, n = _agg([final_of(rr, which) for _, rr in ok.iterrows()])
            row[col], row[f'{col}_std'], row[f'{col}_min'], row[f'{col}_n'] = m, s, lo, n
        m, s, lo, _ = _agg([effective_rank(rr) for _, rr in ok.iterrows()])
        row['rank_eff'], row['rank_eff_std'], row['rank_eff_min'] = m, s, lo
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(['k', 'kappa', 'lr']).reset_index(drop=True)
    shared = _shared_eff_steps(out)
    out['matched'] = [any(np.isclose(e, s) for s in shared) for e in out['eff_step']]
    out.attrs['shared_eff_steps'] = shared
    return out


def _shared_eff_steps(tbl, tol=1e-9):
    """Effective steps realised by EVERY kappa present (the matched cells)."""
    per_kappa = [set(np.round(sub['eff_step'].to_numpy(dtype=float) / tol) * tol)
                 for _, sub in tbl.groupby('kappa')]
    if not per_kappa:
        return []
    shared = set.intersection(*per_kappa)
    return sorted(float(v) for v in shared)


def matched_step_table(tbl, column='final_val_loss'):
    """At each MATCHED effective step, the value per kappa and how far they spread.

    ``rel_spread`` is ``(max - min) / min`` over the kappas at that (k, effective step).
    If kappa were nothing but a learning-rate rescaling, this is 0 up to float noise --
    and at ``k = B`` (no truncation) it very nearly is, while at ``k < B`` it is not,
    which is the whole point of running the sweep at two k.
    """
    shared = tbl.attrs.get('shared_eff_steps') or _shared_eff_steps(tbl)
    rows = []
    for k, sub_k in tbl.groupby('k', dropna=False, sort=True):
        for e in shared:
            cell = sub_k[np.isclose(sub_k['eff_step'], e)]
            if cell.empty:
                continue
            row = {'k': k, 'eff_step': e}
            vals = []
            for _, r in cell.sort_values('kappa').iterrows():
                row[f'kappa={int(r["kappa"])}'] = r[column]
                row[f'n(kappa={int(r["kappa"])})'] = r['counts']
                if np.isfinite(r[column]):
                    vals.append(float(r[column]))
            if len(vals) > 1 and min(vals) > 0:
                row['rel_spread'] = (max(vals) - min(vals)) / min(vals)
            else:
                row['rel_spread'] = np.nan
            row['n_kappa'] = len(vals)
            rows.append(row)
    return pd.DataFrame(rows)


def eff_step_coverage(df):
    """Which effective steps each ``kappa`` was actually RUN at.

    The lr list is shared, so the effective step ``2 lr / kappa`` is not: the largest
    effective step in the grid falls like ``1/kappa``.  A "largest effective step that
    still trains" table therefore compares unequal ranges, and a ceiling that equals
    ``max_eff_step`` here is a GRID edge, not a stability limit.  ``n_above_matched``
    counts the runs a kappa has beyond the last matched step -- 0 means nothing can be
    concluded about it there.
    """
    d = df.copy()
    d['eff_step'] = 2.0 * pd.to_numeric(d['lr'], errors='coerce') \
        / pd.to_numeric(d['kappa'], errors='coerce')
    tbl = kappa_table(df)
    shared = tbl.attrs.get('shared_eff_steps') or _shared_eff_steps(tbl)
    top = max(shared) if shared else np.nan
    rows = []
    for kap, sub in d.groupby('kappa', dropna=False, sort=True):
        e = np.sort(sub['eff_step'].dropna().unique())
        rows.append({'kappa': kap, 'n_eff_steps': len(e),
                     'min_eff_step': float(e[0]) if len(e) else np.nan,
                     'max_eff_step': float(e[-1]) if len(e) else np.nan,
                     'eff_steps': [round(float(v), 4) for v in e],
                     'n_above_matched': int((sub['eff_step'] > top * (1 + 1e-9)).sum())
                     if np.isfinite(top) else 0})
    out = pd.DataFrame(rows)
    out.attrs['matched_eff_steps'] = shared
    out.attrs['top_matched'] = top
    return out


def matched_step_paired(df, column='final_val_loss', pairs=((1, 2), (2, 3), (1, 3))):
    """At each matched effective step, the PAIRED-by-seed difference between the kappas.

    The relative spread of :func:`matched_step_table` is a spread of seed MEANS; whether
    it is more than seed noise is a different question, and the answer differs cell by
    cell.  This is that question, asked with the same 5 model seeds on both sides
    (:func:`paired_seed_diff`): ``mean`` is ``kappa_a - kappa_b``, ``significant`` means
    the t-interval excludes 0.
    """
    d = df if 'diverged' in df.columns else add_derived(df)
    d = d.copy()
    d['eff_step'] = 2.0 * pd.to_numeric(d['lr'], errors='coerce') \
        / pd.to_numeric(d['kappa'], errors='coerce')
    tbl = kappa_table(df)
    shared = tbl.attrs.get('shared_eff_steps') or _shared_eff_steps(tbl)
    rows = []
    for k, sub_k in d.groupby('k', dropna=False, sort=True):
        for e in shared:
            cell = sub_k[np.isclose(sub_k['eff_step'], e)]
            for a, b in pairs:
                sa, sb = cell[cell['kappa'] == a], cell[cell['kappa'] == b]
                if sa.empty or sb.empty:
                    continue
                st = paired_seed_diff(sa, sb, metric=column)
                rows.append({'k': k, 'eff_step': e, 'pair': f'kappa{a}-kappa{b}',
                             'n': st['n'], 'mean': st['mean'], 'std': st['std'],
                             'sem': st['sem'], 't': st['t'], 'p': st['p'],
                             'significant': st['significant'],
                             f'kappa={a}': st['mean_a'], f'kappa={b}': st['mean_b']})
    out = pd.DataFrame(rows)
    out.attrs['matched_eff_steps'] = shared
    out.attrs['column'] = column
    return out


# ---------------------------------------------------------------------------
# (e) plotting
# ---------------------------------------------------------------------------
def _lw(method):
    return 2.6 if method == SVEN else 1.6


def plot_arm(ax, best, x, q, methods=None, logx=False, logy=True, label_col='method',
             **kw):
    """One errorbar line per method of ``q`` against arm column ``x``.

    Goes through :func:`analysis_helpers.errorbar_seeds`, so the error bars are THE seed
    band (clipped at the lowest seed) and the axes gets the one legend entry naming it;
    display names and colours come from :mod:`style`.
    """
    methods = method_order(best['method'].unique()) if methods is None else methods
    for m in methods:
        sub = best[best['method'] == m]
        if sub.empty or not np.isfinite(pd.to_numeric(sub[q], errors='coerce')).any():
            continue
        errorbar_seeds(ax, sub, x, q, color=method_color(m), lw=_lw(m),
                       label=(m if label_col == 'method' else sub[label_col].iloc[0]),
                       zorder=3 if m == SVEN else 1, **kw)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    return ax


def arm_ticks(ax, values, fmt='{:g}', axis='x', min_log_sep=None):
    """Tick a log axis at the ARM values only, with no minor ticks.

    A log axis with four points inside one decade gets matplotlib's decade minor labels
    (``3x10^0`` next to ``4x10^0``), which collide.  The arms are a short, known list, so
    they are the ticks.  ``min_log_sep`` blanks the LABEL (never the tick) of an arm
    closer than that many decades to the last labelled one -- two arms a factor 1.25 apart
    cannot both be written out at this font size.
    """
    import matplotlib.ticker as mticker
    vals = sorted(float(v) for v in values)
    labels, last = [], None
    for v in vals:
        lg = np.log10(v) if v > 0 else None
        if (min_log_sep and last is not None and lg is not None
                and abs(lg - last) < min_log_sep):
            labels.append('')
            continue
        labels.append(fmt.format(v))
        last = lg
    a = ax.xaxis if axis == 'x' else ax.yaxis
    a.set_major_locator(mticker.FixedLocator(vals))
    a.set_major_formatter(mticker.FixedFormatter(labels))
    a.set_minor_locator(mticker.NullLocator())
    return ax


def figure_legend(fig, axes, ncol=7, fontsize=11, bottom=0.10, **kw):
    """ONE legend for a multi-panel figure, outside the axes, deduplicated.

    A legend inside a panel sits on the data -- and with ``legend.frameon = False`` the
    data shows THROUGH it, so a stray marker reads as an unlabelled legend entry.  A
    saved figure also travels without the notebook's prose, so the panels have to be
    readable on their own.  Handles are collected in order over ``axes``, deduplicated by
    label (the seed-band entry of :func:`style.band_legend` included, once), and empty /
    ``_``-prefixed labels dropped.  ``bottom`` is the fraction of the figure reserved for
    the legend by the ``tight_layout`` this function runs.
    """
    seen, handles, labels = set(), [], []
    for ax in np.atleast_1d(np.asarray(axes, dtype=object)).ravel():
        hs, ls = ax.get_legend_handles_labels()
        for h, l in zip(hs, ls):
            if not l or l.startswith('_') or l in seen:
                continue
            seen.add(l)
            handles.append(h)
            labels.append(l)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    leg = fig.legend(handles, labels, loc='lower center', ncol=ncol, fontsize=fontsize,
                     bbox_to_anchor=(0.5, 0.0), **kw)
    return leg


def plot_knob(ax, tbl, x, q, label=None, color=None, **kw):
    """A single knob curve (one Sven ablation) with the seed band."""
    kw = {'marker': 'o', 'capsize': 3, 'color': color or METHOD_COLORS.get(SVEN, 'k'),
          **kw}
    if label is not None:
        kw['label'] = label
    sub = tbl.sort_values(x)
    style.band_legend(ax)
    return ax.errorbar(sub[x], sub[q],
                       yerr=style.clipped_yerr(sub[q], sub[f'{q}_std'], sub[f'{q}_min']),
                       **kw)
