"""Figures and tables for the LARGE models: CIFAR-10 / ResNet18, nanoGPT, GPT-2 (WP4b).

Three notebooks use this module -- ``cifar_analysis``, ``nanogpt_analysis`` and
``gpt2_analysis`` -- and it exists so that none of them re-derives a number that
:mod:`headline` already owns.  The division of labour is strict:

* **:mod:`headline` owns every headline number.**  Confirmation / tuning tables, paired
  differences, selection optimism, budget curves, time-to-target, the efficiency table and
  ``freshness_report``.  Everything here that reports such a quantity is a *view* on the
  table :mod:`headline` returns (:func:`cost_view`, :func:`optimisation_view`), never a
  second computation of it.
* **This module owns the curves, the landscapes and the two scans no selection file
  covers**: ``rebuttal_fig5_cifar_paramfrac_scan`` (Fig 5, R1 Q2) and
  ``exp_gpt2_small_comparison`` (one seed, one epoch, step-based evaluations).

Conventions inherited, not re-invented: colours :data:`style.METHOD_COLORS`, display names
:func:`style.method_label` (``LBFGS`` -> "Stochastic L-BFGS", ``SGDm`` -> "SGD + momentum"),
seed band = mean +/- 1 std (ddof=1) with the lower edge clipped
(:func:`style.clipped_band`) and named once per axes by :func:`style.band_legend`,
diverged = failed and left out of every mean but counted, and ``finished / attempted`` on
every table.  ``bench/best_configs.json``'s own ``label`` field is pre-C-B7 and is never
printed.

Two facts about the large scans that shape the API:

1. **The curves must come from the same configuration as the tables.**  A notebook that
   called ``scan_analysis.plot_best_curves`` would re-select from the scan's own grid; it
   agrees with the selection of record today (85/85), but "agrees today" is not the same
   as "is the same object".  :func:`selected_runs_by_method` takes the selection file and
   returns the runs of the selected configuration in whichever pass is asked for, so a
   curve and the table beside it cannot drift apart.
2. **Time axes are per pass, and per SEED SET.**  ``{scan}_timing`` is the only pass whose
   wall clock is honest (one run per GPU, logging off).  ``losses.train_times`` -- the
   SYNCHRONISED per-batch training time of an epoch (C-T1) -- is recorded in every pass but
   was measured under co-tenancy in the scan and confirmation passes, so a time axis has to
   come from the timing pass.  That pass re-runs the TUNING seeds (base+0..4) while the
   headline tables are the confirmation seeds (base+100..104), so a time-axis panel and the
   table beside it describe the same configuration on different initialisations:
   :func:`seed_note` puts the seed set on the figure so the two are not matched up.
3. **Fig 5 holds TWO Sven configurations**, and which one a reader looks at decides what
   the figure means, so the split is computed rather than written down
   (:func:`paramfrac_groups`): the configuration the matching headline scan SELECTED
   (:func:`headline_sven_config`, ``k=128, lr=0.5, rtol=1e-3``, run 2026-09-20) is the main
   result, and the legacy set point the scan's config still carries (``k=64, lr=1``, the
   pre-Gram ``SET POINT, STILL TENTATIVE`` of EXPERIMENTS.md section 3.3) is the comparison
   beside it.  Within each configuration nothing but ``param_fraction`` moves
   (:func:`paramfrac_config`'s ``varies``), and where the divergences sit
   (:func:`divergence_lrs`) is what the legacy panel is read against: its single ``lr`` is
   the top of the headline grid, the cell all 5 of that scan's Sven divergences sit in.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import analysis_helpers as ah
import headline as hl
import scan_analysis as sa
import style

#: Fig 5 (R1 Q2): the 3-seed parameter-fraction scan.  The 1-seed
#: ``cifar10_resnet_paramFrac_scan_labelReg`` was CUT from the campaign
#: (EXPERIMENTS.md 1.2) and is not in the fresh results root.  It holds TWO Sven
#: configurations -- see :func:`paramfrac_groups`.
FIG5_SCAN = 'rebuttal_fig5_cifar_paramfrac_scan'

#: The GPT-2 comparison: 29 runs, ONE model seed, one epoch of 13,125 steps, B = 16 and
#: k = B = 16.  NOTE: four legacy GPT-2 records exist in the legacy root and are
#: CONTAMINATED (v1 token files, overlapping val/test); nothing here may read them, which
#: is why this constant names the fresh directory and no function takes a results root
#: pointing at the legacy one.
GPT2_SCAN = 'exp_gpt2_small_comparison'

#: The two CIFAR headline scans, in report order.
CIFAR_SCANS = ('cifar10_resnet_scan_labelRegression', 'cifar10_resnet_ce_scan')

#: Per-epoch curves, with what a y axis calls them and whether larger is better.
CURVE_LABELS = {
    'train': 'Train loss (batch mean)',
    'train_eval': 'Train loss (eval mode, fixed subset)',
    'val': 'Validation loss',
    'test': 'Test loss',
    'train_acc': 'Train accuracy',
    'train_eval_acc': 'Train accuracy (eval mode, fixed subset)',
    'val_acc': 'Validation accuracy',
    'test_acc': 'Test accuracy',
}

#: Curves whose entry 0 is the UNTRAINED model (:data:`scan_analysis.PRE_TRAINING_CURVES`
#: plus the accuracy twins CIFAR records).
PRE_TRAINING_CURVES = tuple(sorted(set(sa.PRE_TRAINING_CURVES)
                                   | {'test_acc', 'train_eval_acc'}))

#: Accuracy twins that :mod:`scan_analysis` knows about under another name.  Routing
#: ``train_eval_acc`` through ``train_eval`` is what makes :func:`curve_axis` correct for
#: a TRUNCATED curve as well: ``scan_analysis._has_pre_training`` decides by length first
#: and falls back on the KEY, and the key it is given has to be one it recognises.
#: ``test_acc`` / ``val_acc`` are already in ``scan_analysis.PRE_TRAINING_CURVES``.
PRE_TRAINING_ALIAS = {'train_eval_acc': 'train_eval'}


def is_accuracy(key):
    """Whether ``key`` is an accuracy curve -- larger is better, no log axis."""
    return str(key).endswith('_acc')


def curve_label(key):
    return CURVE_LABELS.get(key, str(key).replace('_', ' '))


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def savefig(fig, name, plot_dir):
    """Save one figure under ``plot_dir`` as BOTH ``.pdf`` and ``.png``, return the paths.

    The paper wants vector, a review thread wants something a browser shows.  ``name`` may
    carry an extension or not; the directory is created.  ``style.set_style`` has already
    set ``savefig.bbox = 'tight'`` and ``savefig.dpi = 300``, so nothing is passed here --
    a per-call ``bbox_inches`` would silently differ from every other figure in the repo.
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(str(name)).stem
    out = []
    for ext in ('.pdf', '.png'):
        path = plot_dir / f'{stem}{ext}'
        fig.savefig(path)
        out.append(path)
    return out


# ---------------------------------------------------------------------------
# The selected configuration, in any pass
# ---------------------------------------------------------------------------
def selected_runs_by_method(scan, kind='confirm', payload=None, results_root=None,
                            methods=None):
    """``{method key: the runs of its SELECTED configuration in pass ``kind``}``.

    The selection is ``bench/best_configs.json`` via :func:`headline.selected_runs`, so
    the curves a notebook draws are the curves of the configuration whose numbers the
    tables report -- including after ``tools/select_best.py`` is re-run, which is what
    makes a CIFAR-CE cell re-executable while the ``rtol`` extension lands.

    Rows are returned *including* diverged / failed runs, because every consumer here
    (:func:`curve_band`, :func:`run_counts`) has to count them; nothing averages them in.
    Sven first, then the baselines alphabetically (:func:`headline.method_order`).
    """
    sels = hl.selection_methods(scan, payload)
    df = hl.load(scan, kind, results_root=results_root)
    out = {}
    for method in hl.method_order(sels):
        key = hl.method_key(method)
        if methods is not None and key not in methods:
            continue
        rows, _ = hl.selected_runs(df, sels[method])
        out[key] = rows
    return out


def run_counts(rows):
    """``(finished, n_runs, n_diverged, n_failed)`` of a set of runs."""
    n = int(len(rows))
    div = int(rows['diverged'].sum()) if 'diverged' in rows.columns else 0
    fail = int(rows['failed'].sum()) if 'failed' in rows.columns else 0
    return n - div - fail, n, div, fail


def seed_note(runs_by_method, seed_col='model_seed'):
    """"model seeds 4000-4004 (5)" for a set of runs -- WHICH seeds a figure drew.

    A figure of the ``{scan}_timing`` pass and a table of ``{scan}_confirm`` describe the
    SAME configuration on DIFFERENT seeds: timing re-runs the tuning seeds (base+0..4),
    confirmation uses base+100..104.  On CIFAR-CE that is the difference between a
    Sven seed-mean final validation loss of 1.3995 and the table's 1.4239, so a time-axis
    panel that says only "(5 seeds)" invites the reader to match it to the wrong number.
    """
    seeds = sorted({int(s) for rows in _iter_run_frames(runs_by_method)
                    for s in pd.to_numeric(rows.get(seed_col), errors='coerce').dropna()})
    if not seeds:
        return 'model seeds unrecorded'
    contiguous = len(seeds) > 1 and seeds[-1] - seeds[0] == len(seeds) - 1
    body = (f'{seeds[0]}-{seeds[-1]}' if contiguous
            else ', '.join(str(s) for s in seeds))
    return f'model seed{"s" if len(seeds) > 1 else ""} {body} ({len(seeds)})'


def _iter_run_frames(runs):
    """``runs`` as an iterable of frames, whether it is a dict of them or one frame."""
    if isinstance(runs, pd.DataFrame):
        return [runs]
    if isinstance(runs, dict):
        return list(runs.values())
    return list(runs)


def grid_divergence_counts(grid):
    """Sven's divergence count on a grid, over the runs that HAVE A RECORD.

    ``scan_analysis.Scan.configs`` reports ``attempted`` from the manifest, so it includes
    runs that have not started -- on ``cifar10_resnet_ce_scan`` while the ``rtol``
    extension lands, 195 attempted against 182 records.  Dividing the divergence count by
    ``attempted`` therefore counts runs that do not exist as clean ones and moves as the
    jobs land; EXPERIMENTS.md section 7 quotes that row on its on-grid RECORDS for the
    same reason.  Returns ``{recorded, diverged, failed, missing, attempted, rate}`` with
    ``rate`` over ``recorded``.
    """
    col = lambda name: int(pd.to_numeric(grid.get(name), errors='coerce').fillna(0).sum())
    fin, div, fail = col('finished'), col('n_diverged'), col('n_failed')
    recorded = fin + div + fail
    return {'recorded': recorded, 'diverged': div, 'failed': fail,
            'missing': col('n_missing'), 'attempted': col('attempted'),
            'rate': (div / recorded) if recorded else np.nan}


def distinct_trajectory_check(scan=None, method='Sven', curve='val', results_root=None,
                              decimals=12, df=None):
    """Do any two configurations of ``method`` share a trajectory, seed by seed?

    :func:`headline.budget_table` answers the tuning-budget question by counting DISTINCT
    ``final_val_loss`` values within each ``(seed, lr)`` group and scaling by the number of
    learning rates (``budget.trajectory_table``).  On a RAGGED grid that scaling is
    diluted: while the CIFAR-CE ``rtol`` extension exists for 3 of 5 seeds and 3 of 5
    learning rates, ``distinct_frac`` reads below 1 even though no two configurations
    actually coincide.  This is the direct check -- the full recorded ``losses[curve]``
    of every configuration of every seed, compared bit for bit -- so a notebook can say
    which of the two it means.

    Returns ``{n_runs, n_configs, n_seeds, n_collisions, collisions}``; ``collisions``
    lists the colliding hyperparameter tuples per seed.
    """
    if df is None:
        df = ah.add_derived(style.load_results(scan, results_root=results_root))
    rows = df[df['method'] == method]
    keys = [c for c in ah.config_columns(df, 'model_seed') if c in rows.columns]
    collisions = []
    n_curves = 0
    for seed, sub in rows.groupby('model_seed', dropna=False):
        seen = {}
        for _, r in sub.iterrows():
            L = r.get('losses')
            v = (L or {}).get(curve) if isinstance(L, dict) else None
            if v is None:
                continue
            sig = tuple(np.round(np.asarray(v, dtype=float), decimals))
            cfg = tuple((c, r.get(c)) for c in keys)
            n_curves += 1
            if sig in seen:
                collisions.append({'model_seed': seed, 'a': seen[sig], 'b': cfg})
            else:
                seen[sig] = cfg
    return {'n_runs': int(len(rows)), 'n_curves': n_curves,
            'n_configs': int(rows.drop_duplicates(keys).shape[0]) if keys else len(rows),
            'n_seeds': int(rows['model_seed'].nunique()),
            'n_collisions': len(collisions), 'collisions': collisions}


def config_label_of(scan, method, payload=None):
    """The selected configuration of one method, spelled as :func:`headline.config_label`."""
    sels = hl.selection_methods(scan, payload)
    for name, sel in sels.items():
        if hl.method_key(name) == hl.method_key(method):
            return hl.config_label(sel)
    raise KeyError(f'{scan}: no selection for {method!r}')


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------
def curve_band(rows, key):
    """``(mean, lower, upper, n_seeds)`` of a per-epoch curve over a configuration's seeds.

    :func:`scan_analysis.seed_band` does the arithmetic -- mean +/- 1 std (ddof=1) with the
    lower edge clipped at the lowest seed so a log axis survives it -- and drops the
    diverged / failed runs, which is why ``n_seeds`` is reported beside the band: a band
    over 3 of 5 seeds is not the same statement as a band over 5.
    ``(None, None, None, 0)`` when the curve is absent from every usable run.
    """
    mean, lower, upper = sa.seed_band(rows, key)
    if mean is None:
        return None, None, None, 0
    stack = sa._seed_stack(rows, key)
    return mean, lower, upper, int(len(stack))


def curve_axis(rows, curve, key, versus='epoch', time_key='train_times'):
    """x values for a seed-averaged curve (:func:`scan_analysis.epoch_axis`).

    ``versus`` is ``'epoch'``, ``'step'``, ``'examples'`` or ``'time'``.  The pre-training
    entry is handled ENTIRELY by ``epoch_axis``: a validation / test / ``train_eval``
    curve is one entry longer than ``train`` because index 0 is the untrained model, and
    drawing it from 1 shifts every curve by an epoch.  ``epoch_axis`` detects that by
    LENGTH (``len(curve) == n_epochs + 1``) and only falls back on the key when the curve
    is truncated, so all this function has to do is hand it a key it recognises --
    :data:`PRE_TRAINING_ALIAS` maps ``train_eval_acc`` onto its loss sibling.

    Prepending the entry here as well -- which is what an earlier version did for the keys
    of :data:`PRE_TRAINING_CURVES` that ``scan_analysis`` does not list -- double-counts it
    on a TIME axis: ``curve_axis(..., 'train_eval_acc', 'time')`` on the CIFAR-CE timing
    rows came out ``[0, 0, 62.9, 125.0, ...]`` and ended at 1182 s where ``val`` on the
    same runs ends at 1244 s, i.e. the whole curve one epoch late and its last point 62 s
    early.
    """
    which = PRE_TRAINING_ALIAS.get(key, key)
    x = sa.epoch_axis(rows, curve, which if which in sa.PRE_TRAINING_CURVES else 'train',
                      versus, time_key)
    if x is None:
        return None
    return np.asarray(x)[:len(curve)]


def plot_curves(ax, runs_by_method, key='val', versus='epoch', time_key='train_times',
                band=True, labels=None, lw_sven=2.8, lw=1.7, **plot_kw):
    """Seed-mean curves of one quantity, one line per method, with the seed band.

    Sven is black and thicker (:data:`style.METHOD_COLORS`); the legend shows the display
    names, and :func:`style.band_legend` adds ONE entry saying what the shading is.  A
    method whose axis cannot be built (no timings) is skipped with a note rather than
    drawn in the wrong unit.  Returns ``{method: n_seeds}`` for what was drawn.
    """
    drawn = {}
    banded = False
    for method, rows in runs_by_method.items():
        mean, lower, upper, n = curve_band(rows, key)
        if mean is None:
            continue
        x = curve_axis(rows, mean, key, versus, time_key)
        if x is None:
            print(f'  [plot_curves] no {versus!r} axis for {method} -- not drawn')
            continue
        m = len(x)
        label = (labels or {}).get(method, style.method_label(method))
        color = style.method_color(method)
        ax.plot(x, mean[:m], color=color, lw=lw_sven if method == 'Sven' else lw,
                label=f'{label} ({n} seeds)' if n else label,
                zorder=3 if method == 'Sven' else 2, **plot_kw)
        if band and n > 1:
            ax.fill_between(x, lower[:m], upper[:m], color=color, alpha=0.18, lw=0,
                            zorder=(3 if method == 'Sven' else 2) - 0.5)
            banded = True
        drawn[method] = n
    if banded:
        style.band_legend(ax)
    ax.set_xlabel(sa.axis_label(versus, time_key))
    ax.set_ylabel(curve_label(key))
    if not is_accuracy(key):
        ax.set_yscale('log')
    return drawn


# ---------------------------------------------------------------------------
# Optimisation vs generalisation
# ---------------------------------------------------------------------------
def optimisation_view(tbl, spec='.4g'):
    """The optimisation-vs-generalisation view of a :func:`headline.confirmation_table`.

    Three numbers of the SAME runs, side by side, and the pair of CIFAR scans is exactly
    why they have to be:

    * ``train (eval mode)`` -- ``losses.train_eval``, the training loss of the trained
      model in EVAL mode on a fixed subset.  Low = the optimiser did its job.
    * ``val`` / ``test`` -- the same model on data it did not fit.
    * ``val - train`` -- the generalisation gap of that run, as a difference and relative
      to the training loss.

    A method with a LOW train_eval and a high val is a generalisation story (CIFAR
    label-regression: Sven fits the train subset and generalises worse); a method whose
    train_eval is high has not optimised at all (CIFAR cross-entropy), and the two must
    not be read off the validation loss alone.  Every column comes from the confirmation
    table; nothing is recomputed here.
    """
    out = []
    for _, r in tbl.iterrows():
        tr = float(r.get('treval_conf', np.nan))
        va = float(r.get('val_conf', np.nan))
        row = {
            'method': r.get('display', r['method']),
            'config': r['config'],
            'train (eval mode)': hl.fmt_pm(tr, r.get('treval_conf_std'), spec),
            'val (confirm)': hl.fmt_pm(va, r.get('val_conf_std'), spec),
            'test (confirm)': hl.fmt_pm(r.get('test_conf'), r.get('test_conf_std'), spec),
            'val - train': ('--' if not np.isfinite(va - tr) else f'{va - tr:+{spec}}'),
            'val / train': ('--' if not (np.isfinite(va) and np.isfinite(tr) and tr)
                            else f'{va / tr:.2f}'),
            'finished/attempted': hl.fmt_counts(r['fin_conf'], r['att_conf']),
        }
        if np.isfinite(pd.to_numeric(r.get('acc_conf'), errors='coerce')):
            row['test acc'] = hl.fmt_pm(r.get('acc_conf'), r.get('acc_conf_std'), '.3g')
        out.append(row)
    view = pd.DataFrame(out)
    view.attrs.update(tbl.attrs)
    return view


def generalisation_gaps(tbl):
    """The numeric gaps behind :func:`optimisation_view`, for a plot or a sentence.

    Columns ``method``, ``train_eval``, ``val``, ``test``, ``gap`` (= val - train_eval),
    ``gap_rel`` and ``acc``.  Sorted by ``val`` -- the confirmation ranking.
    """
    out = pd.DataFrame({
        'method': tbl['method'],
        'display': tbl.get('display', tbl['method']),
        'train_eval': pd.to_numeric(tbl.get('treval_conf'), errors='coerce'),
        'val': pd.to_numeric(tbl['val_conf'], errors='coerce'),
        'test': pd.to_numeric(tbl.get('test_conf'), errors='coerce'),
        'acc': pd.to_numeric(tbl.get('acc_conf'), errors='coerce'),
    })
    out['gap'] = out['val'] - out['train_eval']
    out['gap_rel'] = out['gap'] / out['train_eval'].replace(0.0, np.nan)
    return out.sort_values('val').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
def _drop_tiny_std(mean, std, rel=1e-9):
    """``None`` for a spread that is float round-off, else ``std``.

    ``peak_gpu_mem_mb`` is bit-identical across the seeds of a configuration, so its
    ddof=1 std is ~6e-14 rather than 0 and a table would print ``22965 +/- 6.3553e-14``.
    The same round-off is what :func:`style.clipped_band` exists to survive.
    """
    m, s = pd.to_numeric(mean, errors='coerce'), pd.to_numeric(std, errors='coerce')
    if s is None or not np.isfinite(s):
        return None
    if np.isfinite(m) and m and abs(s / m) < rel:
        return 0.0
    return float(s)


def cost_view(eff, spec='.4g'):
    """The cost table a reader sees, from :func:`headline.efficiency_table`.

    Seconds per epoch and peak GPU memory come from the STANDALONE ``{scan}_timing`` pass
    -- the only pass that had the GPU to itself -- and ``vs fastest`` is the ratio to the
    cheapest method in the same column, which is the number the cost objection is about.
    ``fin/att`` is the timing pass's own ``finished / attempted``: a cost row backed by 3
    of 5 runs says so.
    """
    eff = eff.copy()
    epoch = pd.to_numeric(eff['epoch_s'], errors='coerce')
    mem = pd.to_numeric(eff['peak_gpu_mem_mb'], errors='coerce')
    fastest = np.nanmin(epoch) if np.isfinite(epoch).any() else np.nan
    leanest = np.nanmin(mem) if np.isfinite(mem).any() else np.nan
    out = []
    for i, r in eff.iterrows():
        out.append({
            'method': r.get('display', r['method']),
            'config': r['config'],
            's/epoch': hl.fmt_pm(r['epoch_s'], _drop_tiny_std(r['epoch_s'],
                                                              r.get('epoch_s_std')), spec),
            'x fastest': ('--' if not np.isfinite(epoch[i] / fastest)
                          else f'{epoch[i] / fastest:.2f}x'),
            'ms/step': hl.fmt_pm(r['ms_per_step'], None, '.4g'),
            'peak mem (MB)': hl.fmt_pm(
                r['peak_gpu_mem_mb'],
                _drop_tiny_std(r['peak_gpu_mem_mb'], r.get('peak_gpu_mem_mb_std')), '.5g'),
            'x leanest': ('--' if not np.isfinite(mem[i] / leanest)
                          else f'{mem[i] / leanest:.2f}x'),
            'total wall (s)': hl.fmt_pm(r['wall_s'], None, '.5g'),
            'fin/att': hl.fmt_counts(r.get('fin_timing', 0), max(r.get('att_timing', 0), 0)),
            'GPU': r.get('gpu_timing'),
        })
    view = pd.DataFrame(out)
    view.attrs.update(eff.attrs)
    return view


def plot_cost_bars(axes, eff, log=True):
    """Two bars per method: seconds per epoch and peak GPU memory, standalone pass.

    Error bars are the seed spread of the timing runs, clipped by
    :func:`style.clipped_yerr` (a memory value is bit-identical across seeds, and the
    unclipped band then goes negative by one float bit -- WP1's fix).
    """
    ax_t, ax_m = axes
    eff = eff.sort_values('epoch_s', na_position='last')
    labels = [style.method_label(m) for m in eff['method']]
    colors = [style.method_color(m) for m in eff['method']]
    x = np.arange(len(eff))
    for ax, col, std_col, ylabel in (
            (ax_t, 'epoch_s', 'epoch_s_std', 'Seconds per epoch (standalone)'),
            (ax_m, 'peak_gpu_mem_mb', 'peak_gpu_mem_mb_std', 'Peak GPU memory (MB)')):
        v = pd.to_numeric(eff[col], errors='coerce').to_numpy(dtype=float)
        s = pd.to_numeric(eff.get(std_col), errors='coerce').to_numpy(dtype=float) \
            if std_col in eff else np.zeros_like(v)
        yerr = style.clipped_yerr(v, s, v - np.nan_to_num(s))
        ax.bar(x, v, color=colors, yerr=yerr, capsize=2.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale('log')
    style.band_legend(ax_t)
    ax_t.legend(fontsize=8)
    return eff


# ---------------------------------------------------------------------------
# The Sven hyperparameter landscape (k, lr, rtol) -- ragged grids included
# ---------------------------------------------------------------------------
def sven_grid(scan, results_root=None, metric='final_val_loss'):
    """Every Sven configuration of a scan: seed mean, spread, divergences, eligibility.

    :meth:`scan_analysis.Scan.configs` does the grouping and the binding ordering
    (eligible -> fewest diverged -> seed mean), so this is the scan's own view of its grid
    and it can be compared with the selection file rather than replacing it.

    The CIFAR-CE grid is RAGGED on purpose: decision 5's ``rtol`` extension adds 0.03 /
    0.1 / 0.3 at ``k = 128`` and three of the five learning rates only.  A cell that was
    never run is absent from this table (and blank in :func:`plot_sven_landscape`) --
    which is the honest picture and must not be filled in by an interpolation.
    """
    scan_obj = scan if isinstance(scan, sa.Scan) else load_scan_obj(scan, results_root)
    cfg = scan_obj.configs(scan_obj.sven, sa.SVEN_CONFIG, metric)
    cfg = cfg.rename(columns={'score': 'val', 'score_std': 'val_std',
                              'score_min': 'val_min', 'score_max': 'val_max'})
    keep = ['k', 'lr', 'rtol', 'val', 'val_std', 'val_min', 'finished', 'attempted',
            'n_diverged', 'n_failed', 'n_missing', 'eligible']
    return cfg[[c for c in keep if c in cfg.columns]].reset_index(drop=True)


def load_scan_obj(name, results_root=None, title=None, plot_dir=None):
    """A :class:`scan_analysis.Scan` for one directory, with the campaign's title."""
    title = title or hl.scan_title(name)
    plot_dir = plot_dir or Path('plots_v2') / 'scratch'
    return sa.load_scan(name, title, plot_dir, results_root=results_root)


def mark_selected(grid, scan, payload=None):
    """Add a boolean ``selected`` column: which grid row is the selection of record.

    Matching is on the hyperparameter VALUES in ``bench/best_configs.json`` (``k``, ``lr``,
    ``rtol``), so the mark moves by itself when ``tools/select_best.py`` is re-run after
    the ``rtol`` extension -- no cell in a notebook names a configuration.
    """
    sels = hl.selection_methods(scan, payload)
    sel = next((s for m, s in sels.items() if hl.method_key(m) == hl.SVEN_LABEL), None)
    grid = grid.copy()
    if sel is None:
        grid['selected'] = False
        return grid
    hp = sel.get('hparams') or {}
    mask = pd.Series(True, index=grid.index)
    for name in ('k', 'lr', 'rtol'):
        if name in grid.columns and hp.get(name) is not None:
            mask &= np.isclose(pd.to_numeric(grid[name], errors='coerce'),
                               float(hp[name]), rtol=1e-9, atol=0.0)
    grid['selected'] = mask
    return grid


def landscape_axes(grid, x='lr', y='rtol'):
    """The sorted values of two grid axes -- pass them to every panel of a multi-panel
    landscape so the panels share their axes and can be read against each other."""
    return (sorted(pd.to_numeric(grid[x], errors='coerce').dropna().unique()),
            sorted(pd.to_numeric(grid[y], errors='coerce').dropna().unique()))


def landscape_scale(grid, value='val', log_value=True, eligible_only=False):
    """``(vmin, vmax)`` for a shared colour scale over a whole grid.

    A multi-panel landscape MUST pass this: ``pcolormesh`` autoscales per axes, so two
    panels drawn without it are coloured on two different scales and the one colour bar
    beside them describes only the last panel drawn -- a figure that looks like a
    comparison and is not one.
    """
    v = pd.to_numeric(grid[value], errors='coerce')
    if eligible_only and 'eligible' in grid.columns:
        v = v[grid['eligible'].astype(bool)]
    v = v[np.isfinite(v)]
    if not len(v):
        return None, None
    lo, hi = float(v.min()), float(v.max())
    return (np.log10(lo), np.log10(hi)) if log_value else (lo, hi)


def plot_sven_landscape(ax, grid, x='lr', y='rtol', value='val', log_value=True,
                        cmap='viridis', annotate=True, missing_text='--',
                        xs=None, ys=None, vmin=None, vmax=None):
    """Seed-mean ``value`` over two Sven axes as a heat map, blank where nothing ran.

    Drawn with ``pcolormesh`` on CATEGORICAL axes (the grid values are not evenly spaced
    in either linear or log units), and a cell no run covers is left blank and marked
    ``--``, so the extension's ragged coverage reads as coverage rather than as a loss of
    0.  Two further states are marked, and they are NOT the same state:

    * a cell whose seeds are INCOMPLETE (``finished < attempted``) carries its own
      ``finished/attempted`` under the number.  This is the case that actually occurs and
      the one a reader is most likely to misread: while the CIFAR-CE ``rtol`` extension
      lands, its nine cells each hold 3 or 4 of 5 seeds, they are all *eligible* (more
      than half landed), and the best of them undercuts the red-ringed selection of
      record -- drawn unmarked, a 3-seed mean would be compared with a 5-seed one.
    * a cell that is present but NOT ELIGIBLE -- fewer than half its seeds produced a
      result, because they have not landed or because they diverged (CIFAR
      label-regression at lr 1.0, rtol 1e-4) -- is dashed and marked ``*``.  It is drawn
      rather than hidden, so the landscape does not silently change shape when the jobs
      finish, but it never entered the selection.

    The selected configuration, when ``grid`` carries a ``selected`` column, is ringed in
    red.

    Returns ``(im, mat, ineligible, partial)``: the mesh, the value matrix and the two
    boolean matrices above.
    """
    auto_x, auto_y = landscape_axes(grid, x, y)
    xs = list(auto_x if xs is None else xs)
    ys = list(auto_y if ys is None else ys)
    mat = np.full((len(ys), len(xs)), np.nan)
    inelig = np.zeros((len(ys), len(xs)), dtype=bool)
    partial = np.zeros((len(ys), len(xs)), dtype=bool)
    counts = {}
    for _, r in grid.iterrows():
        if not np.isfinite(r[value]) or float(r[y]) not in ys or float(r[x]) not in xs:
            continue
        i, j = ys.index(float(r[y])), xs.index(float(r[x]))
        v = float(r[value])
        if np.isnan(mat[i, j]) or v < mat[i, j]:
            mat[i, j] = v
            inelig[i, j] = not bool(r.get('eligible', True))
            fin = pd.to_numeric(r.get('finished'), errors='coerce')
            att = pd.to_numeric(r.get('attempted'), errors='coerce')
            partial[i, j] = bool(np.isfinite(fin) and np.isfinite(att) and fin < att)
            counts[(i, j)] = (fin, att)
    shown = np.log10(mat) if log_value else mat
    im = ax.pcolormesh(np.arange(len(xs) + 1), np.arange(len(ys) + 1), shown,
                       cmap=cmap, shading='flat', vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(xs)) + 0.5)
    ax.set_xticklabels([f'{v:g}' for v in xs], rotation=45)
    ax.set_yticks(np.arange(len(ys)) + 0.5)
    ax.set_yticklabels([f'{v:g}' for v in ys])
    ax.set_xlabel({'lr': 'Learning rate', 'k': '$k$', 'rtol': 'rtol'}.get(x, x))
    ax.set_ylabel({'lr': 'Learning rate', 'k': '$k$', 'rtol': 'rtol'}.get(y, y))
    lo = vmin if vmin is not None else (np.nanmin(shown) if np.isfinite(shown).any() else 0.0)
    hi = vmax if vmax is not None else (np.nanmax(shown) if np.isfinite(shown).any() else 1.0)
    mid = 0.5 * (lo + hi)          # the colour scale's midpoint, so the annotation
    #                                contrast follows the SHARED scale, not this panel
    if annotate:
        for i in range(len(ys)):
            for j in range(len(xs)):
                v = mat[i, j]
                if not np.isfinite(v):
                    text = missing_text
                else:
                    text = f'{v:.3g}' + ('*' if inelig[i, j] else '')
                    if partial[i, j]:
                        fin, att = counts.get((i, j), (np.nan, np.nan))
                        text += f'\n{fin:.0f}/{att:.0f}'
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', fontsize=7,
                        linespacing=1.15,
                        color='0.4' if not np.isfinite(v) else
                        ('white' if shown[i, j] < mid else 'black'))
    for i, j in zip(*np.nonzero(inelig)):
        ax.add_patch(plt_rect(j, i, edgecolor='0.85', lw=1.2, linestyle='--'))
    if 'selected' in grid.columns:
        for _, r in grid[grid['selected'].astype(bool)].iterrows():
            if float(r[x]) in xs and float(r[y]) in ys:
                ax.add_patch(plt_rect(xs.index(float(r[x])), ys.index(float(r[y]))))
    return im, mat, inelig, partial


def plt_rect(j, i, **kw):
    """The ring drawn round the selected cell (kept separate so it is testable)."""
    from matplotlib.patches import Rectangle
    kw.setdefault('fill', False)
    kw.setdefault('edgecolor', 'red')
    kw.setdefault('lw', 2.0)
    return Rectangle((j, i), 1, 1, **kw)


def sven_rank_used(scan, kind='diag', results_root=None, payload=None):
    """Per epoch, the rank Sven actually inverted, from the diag pass's ``svd_summary``.

    ``num_nonzero_svs_epoch`` is the number of singular values that survived the ``rtol``
    cut, averaged over the steps of an epoch, and ``sv_max_epoch`` /
    ``sv_min_all_epoch`` bracket the spectrum.  Reading it answers "does the ``k``
    truncation bind, or is ``rtol`` the binding constraint?" -- on both CIFAR scans the
    rank sits at ``k``, which is also why the budget table finds every Sven grid point a
    distinct trajectory there.

    Returns a tidy frame ``run_id, model_seed, epoch, rank_used, k, sv_max, sv_min_all``;
    empty when the pass has no Sven record with a summary.
    """
    df = hl.load(scan, kind, results_root=results_root)
    sels = hl.selection_methods(scan, payload)
    sel = next((s for m, s in sels.items() if hl.method_key(m) == hl.SVEN_LABEL), None)
    rows_out = []
    rows = hl.selected_runs(df, sel)[0] if sel is not None else df[df['method'] == 'Sven']
    for _, r in rows.iterrows():
        summary = r.get('svd_summary')
        if not isinstance(summary, dict):
            continue
        rank = summary.get('num_nonzero_svs_epoch') or []
        smax = summary.get('sv_max_epoch') or []
        smin = summary.get('sv_min_all_epoch') or []
        for e, val in enumerate(rank, start=1):
            rows_out.append({'run_id': r['run_id'], 'model_seed': r.get('model_seed'),
                             'epoch': e, 'rank_used': float(val), 'k': r.get('k'),
                             'sv_max': float(smax[e - 1]) if e <= len(smax) else np.nan,
                             'sv_min_all': float(smin[e - 1]) if e <= len(smin) else np.nan})
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# Fig 5: the parameter-fraction scan
# ---------------------------------------------------------------------------
#: The parameter-fraction sweep resamples its mask EVERY STEP (``mask_mode:
#: elementwise``, a fresh Bernoulli draw per optimizer step), so a fraction ``f`` is not
#: "train this subnetwork" -- every parameter is updated eventually.  Stated on the figure
#: because the reviewer's question (R1 Q2) is about updating a subset of parameters, and a
#: fixed mask would be a different experiment.
MASK_NOTE = ('masks are resampled every step (elementwise Bernoulli), so $f$ is the '
             'fraction updated per step, not a fixed subnetwork')

#: The OTHER thing Fig 5 has to say about itself (EXPERIMENTS.md section 3.3): every one of
#: its 15 runs shares ONE Sven configuration, it was not re-tuned per fraction, and that
#: configuration is the pre-Gram ``SET POINT, STILL TENTATIVE`` rather than the optimum the
#: BN-fixed headline scan selected.  Both of the figure's blow-ups sit at that single
#: learning rate, which is also where all 5 divergences of the headline label-regression
#: scan sit -- so the low-``f`` collapse bounds *this configuration at this learning rate*,
#: not "masking at fraction f".
FIXED_CONFIG_NOTE = ('one fixed Sven configuration for every fraction (not re-tuned per '
                     '$f$); see EXPERIMENTS.md section 3.3')

#: The headline scan whose Sven selection Fig 5's set point is supposed to follow: the
#: figure fixes ONE ``(k, lr, rtol)`` at every fraction, and the only defensible choice of
#: that point is the one the matching headline scan selected.  Same objective (label
#: regression), same model, same BatchNorm policy.
FIG5_HEADLINE_SCAN = 'cifar10_resnet_scan_labelRegression'

#: What makes a Fig-5 configuration, i.e. what :func:`paramfrac_groups` splits the scan on.
FIG5_CONFIG_KEYS = ('k', 'lr', 'rtol')

#: Loss keys as a caption spells them -- the Fig-5 scan is label regression and sits in a
#: notebook that also reports cross-entropy, so its objective has to be on the figure.
LOSS_TITLES = {'label_regression': 'label regression (one-hot MSE)',
               'mse': 'MSE', 'ce': 'cross-entropy',
               'cross_entropy': 'cross-entropy'}


def loss_title(loss):
    return LOSS_TITLES.get(str(loss), str(loss).replace('_', ' '))


def paramfrac_config(df=None, results_root=None, scan=FIG5_SCAN,
                     keys=('k', 'lr', 'rtol', 'kappa', 'loss', 'batch_size',
                           'num_epochs', 'gram_capture', 'bn_mode')):
    """The ONE configuration Fig 5's scan ran, with the seeds and the fractions it covers.

    Every run of ``rebuttal_fig5_cifar_paramfrac_scan`` shares ``k``, ``lr``, ``rtol``,
    ``kappa`` and the objective -- only ``param_fraction`` varies -- so the figure can and
    must name it.  ``varies`` lists any key that turned out NOT to be constant, so a future
    re-launch that does re-tune per fraction makes the caller's assertion fail rather than
    printing one value of several.

    Returns a dict of the constant values plus ``label`` (``k=64, lr=1, rtol=0.001``),
    ``loss_label``, ``n_seeds``, ``seeds``, ``fractions``, ``n_runs`` and ``varies``.
    """
    if df is None:
        df = ah.add_derived(style.load_results(scan, results_root=results_root))
    out, varies = {}, []
    for key in keys:
        if key not in df.columns:
            continue
        vals = df[key].dropna().unique()
        out[key] = vals[0] if len(vals) else None
        if len(vals) > 1:
            varies.append(key)
    def _fmt(value):
        v = pd.to_numeric(value, errors='coerce')
        return f'{v:g}' if np.isfinite(v) else str(value)

    hp = [f'{k}={_fmt(out[k])}' for k in ('k', 'lr', 'rtol') if out.get(k) is not None]
    out.update({
        'label': ', '.join(hp),
        'loss_label': loss_title(out.get('loss')),
        'seeds': sorted(int(s) for s in df['model_seed'].dropna().unique()),
        'n_seeds': int(df['model_seed'].nunique()),
        'fractions': sorted(float(f) for f in df['param_fraction'].dropna().unique()),
        'n_runs': int(len(df)),
        'varies': varies,
        'mask_note': MASK_NOTE,
        'fixed_config_note': FIXED_CONFIG_NOTE,
    })
    return out


def headline_sven_config(scan, payload=None):
    """The Sven configuration the headline ``scan`` SELECTED, as ``{hparams..., label}``.

    Fig 5 has to be read against this: it fixed ``k``, ``lr`` and ``rtol`` at a set point
    that predates the BatchNorm-fixed headline scan, so the two differ, and the difference
    is the confound EXPERIMENTS.md section 3.3 flags.  Read from
    ``bench/best_configs.json`` so the comparison moves by itself if the selection does.
    """
    sels = hl.selection_methods(scan, payload)
    sel = next((s for m, s in sels.items() if hl.method_key(m) == hl.SVEN_LABEL), None)
    if sel is None:
        return {'label': '(no Sven selection)', 'hparams': {}}
    hp = dict(sel.get('hparams') or {})
    out = {k: hp.get(k) for k in ('k', 'lr', 'rtol', 'kappa')}
    out['hparams'] = hp
    out['label'] = hl.config_label(sel)
    return out


def paramfrac_groups(df=None, results_root=None, scan=FIG5_SCAN,
                     headline_scan=FIG5_HEADLINE_SCAN, payload=None,
                     keys=FIG5_CONFIG_KEYS):
    """Fig 5 split by Sven CONFIGURATION, the SELECTED one first.

    The scan holds two configurations, and which one a reader looks at decides what the
    figure means:

    * the configuration ``headline_scan`` selected (``k``, ``lr``, ``rtol`` from
      ``bench/best_configs.json``) -- the main result, added 2026-09-20;
    * the legacy set point the scan's config still carries (the pre-Gram
      ``SET POINT, STILL TENTATIVE``, never re-derived from the BatchNorm-fixed headline
      scan before the first launch) -- kept as the comparison, because its 15 runs are a
      real measurement of masking at a learning rate that is also the divergence-prone top
      of the headline grid.

    Splitting is what keeps :func:`paramfrac_config`'s ``varies`` assertion meaningful: on
    the whole frame ``k`` and ``lr`` now vary *between* configurations while still being
    fixed *within* each one, which is exactly the property the figure claims.

    Returns a list of dicts, selected first, each with ``df`` (the subframe), ``table``
    (:func:`paramfrac_table` of it), ``cfg`` (:func:`paramfrac_config` of it), ``key``
    (the ``keys`` tuple), ``is_selected``, ``role`` (``'selected'`` /
    ``'not selected'``), ``role_label`` for a legend, and ``diff`` -- the per-key
    comparison against the selection, empty for the selected configuration.  The
    selection itself is on every entry as ``headline`` (:func:`headline_sven_config`).
    """
    if df is None:
        df = ah.add_derived(style.load_results(scan, results_root=results_root))
    hsven = headline_sven_config(headline_scan, payload)

    def _num(value):
        return pd.to_numeric(value, errors='coerce')

    keys = [k for k in keys if k in df.columns]
    assert keys, f'{scan}: none of {FIG5_CONFIG_KEYS} is a column of these records'
    out = []
    for key, sub in df.groupby(list(keys), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        cfg = paramfrac_config(sub)
        diff = [f'{k}: {_num(cfg.get(k)):g} vs selected {_num(hsven[k]):g}'
                for k in keys
                if hsven.get(k) is not None
                and not np.isclose(_num(cfg.get(k)), _num(hsven[k]))]
        is_sel = not diff and all(hsven.get(k) is not None for k in keys)
        out.append({
            'key': key, 'df': sub, 'cfg': cfg, 'table': paramfrac_table(sub, scan=scan),
            'is_selected': is_sel, 'diff': diff, 'headline': hsven,
            'headline_scan': headline_scan,
            'role': 'selected' if is_sel else 'not selected',
            'role_label': (f"selected ({cfg['label']})" if is_sel
                           else f"not selected ({cfg['label']})"),
        })
    out.sort(key=lambda e: (not e['is_selected'], e['key']))
    return out


def divergence_lrs(scan, method='Sven', results_root=None, by=('lr', 'k', 'rtol')):
    """Which hyperparameter cells ``method``'s divergences on ``scan`` sit in.

    Fig 5's two blow-ups are at its single ``lr``; so are all 5 divergences of the CIFAR
    label-regression headline scan.  Quoting that beside Fig 5 is what keeps the low-``f``
    collapse from being read as a statement about ``f`` alone.  Returns
    ``(n_diverged, n_runs, frame)`` with one row per cell that contains a divergence.
    """
    df = ah.add_derived(style.load_results(scan, results_root=results_root))
    rows = df[df['method'] == method]
    bad = rows[rows['diverged'].astype(bool)] if 'diverged' in rows.columns else rows.iloc[:0]
    by = [c for c in by if c in rows.columns]
    if not len(bad) or not by:
        return int(len(bad)), int(len(rows)), pd.DataFrame(columns=[*by, 'n'])
    tbl = (bad.groupby(by, dropna=False).size().rename('n').reset_index()
           .sort_values('n', ascending=False).reset_index(drop=True))
    return int(len(bad)), int(len(rows)), tbl


def paramfrac_table(df=None, results_root=None, scan=FIG5_SCAN):
    """Fig 5's table: per parameter fraction, quality and cost with the seed spread.

    One row per ``param_fraction`` actually run.  ``actual`` is the recorded
    ``actual_param_fraction`` -- the fraction the mask really covered, which is what the x
    axis must use, not the requested value.  Quality columns are the seed mean +/- 1 std
    over the runs that FINISHED; ``n_diverged`` counts the rest (a diverged run has no
    final loss and is not scored by its last finite value).  Cost columns
    (``peak_mem_mb``, ``ms_per_step``) come from the same runs: this scan has no separate
    timing pass, so they carry the scan's co-tenancy and are comparable across fractions
    but not against a standalone number.
    """
    if df is None:
        df = ah.add_derived(style.load_results(scan, results_root=results_root))
    rows = []
    for frac, sub in df.groupby('param_fraction', dropna=False):
        ok = sub[~sub['diverged'] & ~sub['failed']]
        entry = {'param_fraction': float(frac),
                 'actual': float(pd.to_numeric(sub.get('actual_param_fraction'),
                                               errors='coerce').dropna().mean())
                 if 'actual_param_fraction' in sub else float(frac),
                 'mask_mode': next((str(v) for v in sub.get('mask_mode', pd.Series(dtype=object))
                                    .dropna().unique()), 'none'),
                 'finished': int(len(ok)), 'attempted': int(len(sub)),
                 'n_diverged': int(sub['diverged'].sum()),
                 'n_failed': int(sub['failed'].sum())}
        for col, name in (('final_val_loss', 'val'), ('final_test_loss', 'test'),
                          ('final_val_acc', 'val_acc'), ('final_test_acc', 'test_acc')):
            v = pd.to_numeric(ok.get(col), errors='coerce') if col in ok else pd.Series(dtype=float)
            v = v[np.isfinite(v)]
            entry[name] = float(v.mean()) if len(v) else np.nan
            entry[f'{name}_std'] = float(v.std(ddof=1)) if len(v) > 1 else (
                0.0 if len(v) == 1 else np.nan)
            entry[f'{name}_min'] = float(v.min()) if len(v) else np.nan
            entry[f'{name}_max'] = float(v.max()) if len(v) else np.nan
        eff = hl.run_efficiency(sub)          # cost is measured for diverged runs too
        for col, name in (('peak_gpu_mem_mb', 'peak_mem_mb'), ('ms_per_step', 'ms_per_step'),
                          ('epoch_s', 'epoch_s')):
            v = pd.to_numeric(eff[col], errors='coerce')
            v = v[np.isfinite(v)]
            entry[name] = float(v.mean()) if len(v) else np.nan
            entry[f'{name}_std'] = float(v.std(ddof=1)) if len(v) > 1 else (
                0.0 if len(v) == 1 else np.nan)
        rows.append(entry)
    out = pd.DataFrame(rows).sort_values('actual').reset_index(drop=True)
    out.attrs['scan'] = scan
    out.attrs['mask_note'] = MASK_NOTE
    if len(out):
        ref = out[np.isclose(out['actual'], 1.0)]
        if len(ref):
            for col in ('peak_mem_mb', 'ms_per_step'):
                out[f'{col}_vs_full'] = out[col] / float(ref.iloc[0][col])
    return out


def paramfrac_view(tbl, spec='.4g'):
    """Fig 5's table as a reader sees it: quality, cost, and what did not finish.

    ``x full`` columns are the cost relative to ``f = 1`` (no mask at all), which is the
    comparison the reviewer's question turns on: a per-step time ABOVE 1 means masking
    made the step more expensive, not less.
    """
    out = []
    for _, r in tbl.iterrows():
        row = {
            'f (requested)': f"{r['param_fraction']:g}",
            'f (actual)': f"{r['actual']:g}",
            'mask': r['mask_mode'],
            'val': hl.fmt_pm(r['val'], _drop_tiny_std(r['val'], r['val_std']), spec),
            'test': hl.fmt_pm(r['test'], _drop_tiny_std(r['test'], r['test_std']), spec),
            'val acc': hl.fmt_pm(r['val_acc'], _drop_tiny_std(r['val_acc'],
                                                              r['val_acc_std']), '.3g'),
            'test acc': hl.fmt_pm(r['test_acc'], _drop_tiny_std(r['test_acc'],
                                                                r['test_acc_std']), '.3g'),
            'peak mem (MB)': hl.fmt_pm(r['peak_mem_mb'],
                                       _drop_tiny_std(r['peak_mem_mb'],
                                                      r['peak_mem_mb_std']), '.5g'),
            'ms/step': hl.fmt_pm(r['ms_per_step'],
                                 _drop_tiny_std(r['ms_per_step'],
                                                r['ms_per_step_std']), '.4g'),
            'finished/attempted': hl.fmt_counts(r['finished'], r['attempted']),
            'diverged': int(r['n_diverged']),
        }
        for col, name in (('peak_mem_mb_vs_full', 'mem x full'),
                          ('ms_per_step_vs_full', 'ms/step x full')):
            v = pd.to_numeric(r.get(col), errors='coerce')
            row[name] = '--' if not np.isfinite(v) else f'{v:.2f}x'
        out.append(row)
    view = pd.DataFrame(out)
    view.attrs.update(tbl.attrs)
    return view


def plot_paramfrac_quality(axes, tbl, keys=(('val', 'test'), ('val_acc', 'test_acc')),
                           annotate_counts=True, ls='-', marker='o', label_suffix='',
                           hollow_label=True):
    """Fig 5, quality half: loss and accuracy vs the ACTUAL parameter fraction.

    Error bars are the seed band (:func:`style.clipped_yerr`).  A fraction where some
    seeds diverged is drawn from the survivors, and ``annotate_counts`` puts its
    ``finished/attempted`` on the point: Fig 5's two smallest fractions rest on 2 surviving
    seeds of 3, which EXPERIMENTS.md section 3.3 requires any seed band on this figure to
    show, and an ordinary error bar cannot -- a 2-seed spread is drawn exactly like a
    3-seed one.  The counts are also returned for the table below the figure.

    ``ls`` / ``marker`` / ``label_suffix`` exist so the SAME axes can carry the scan's two
    Sven configurations (:func:`paramfrac_groups`) without a second pair of colours: val
    and test keep ``C0`` / ``C1``, and the configuration is the line style.  Call the
    second one with ``hollow_label=False`` / ``annotate_counts=False`` so the
    "some seeds diverged" entry and the seed counts are not written twice.
    """
    x = pd.to_numeric(tbl['actual'], errors='coerce').to_numpy(dtype=float)
    fin = pd.to_numeric(tbl['finished'], errors='coerce').to_numpy(dtype=float)
    att = pd.to_numeric(tbl['attempted'], errors='coerce').to_numpy(dtype=float)
    short = np.isfinite(fin) & np.isfinite(att) & (fin < att)
    for ax, group in zip(np.atleast_1d(axes), keys):
        for i, key in enumerate(group):
            v = pd.to_numeric(tbl[key], errors='coerce').to_numpy(dtype=float)
            s = pd.to_numeric(tbl[f'{key}_std'], errors='coerce').to_numpy(dtype=float)
            lo = pd.to_numeric(tbl[f'{key}_min'], errors='coerce').to_numpy(dtype=float)
            if not np.isfinite(v).any():
                continue
            ax.errorbar(x, v, yerr=style.clipped_yerr(v, s, lo), marker=marker, ls=ls,
                        color=f'C{i}', capsize=3, label=curve_label(key) + label_suffix)
            # the reduced-n points, drawn hollow on top of their own marker
            if short.any():
                ax.plot(x[short], v[short], marker, ms=9, mfc='none', mew=1.6,
                        color=f'C{i}', zorder=4,
                        label=('open marker: some seeds diverged'
                               if i == 0 and hollow_label else None))
            if annotate_counts and i == 0:
                for xi, vi, f, a in zip(x[short], v[short], fin[short], att[short]):
                    if np.isfinite(vi):
                        ax.annotate(f'{f:.0f}/{a:.0f} seeds', (xi, vi), fontsize=7,
                                    xytext=(6, 7), textcoords='offset points',
                                    color='0.25')
        ax.set_xscale('log')
        ax.set_xlabel('Actual parameter fraction $f$')
        ax.set_ylabel('Accuracy' if group[0].endswith('_acc') else 'Final loss')
        if not group[0].endswith('_acc'):
            ax.set_yscale('log')
        style.band_legend(ax)
        ax.legend(fontsize=8)
    return tbl[['actual', 'finished', 'attempted', 'n_diverged']]


#: Why the ``profile_results_v2`` step times may not share an axis with the records.
PROFILE_V2_STEP_NOTE = ('profile v2 called torch.cuda.empty_cache() every step, which\n'
                        'penalises the unmasked full-capture path most: its step times\n'
                        'are NOT comparable with the records and are omitted here')


def plot_paramfrac_cost(axes, tbl, profiles=None, profile_step_times=False,
                        profile_note=PROFILE_V2_STEP_NOTE, ls='-', marker='o',
                        record_label='records (scan runs)'):
    """Fig 5, cost half: peak memory and time per step vs the actual fraction.

    The RECORDS are the measurement of record (the campaign code, ``empty_cache`` off).
    ``profiles`` optionally overlays the ``param_fraction`` study of
    ``profile_results_v*`` -- pass ``profile_helpers.load_profiles()`` filtered to
    ``arch == 'cifar_resnet18'`` and ``study == 'param_fraction'``.

    ``profile_step_times`` gates the overlay on the STEP-TIME panel, and it defaults to
    ``False`` for a reason: on memory the two agree to 0.06% (profile 11829 / 12121 /
    12993 / 14447 / 22978 MB against records 11816 / 12107 / 12979 / 14433 / 22965), but
    on step time they say OPPOSITE things.  v2 measured masked 530-626 ms and unmasked
    846 ms -- masking looks 1.4x *faster* -- because it called
    ``torch.cuda.empty_cache()`` every step, which costs the unmasked full-capture path
    most; the records give masked 342-389 ms against 178 ms unmasked, i.e. masking 1.9-2.2x
    SLOWER, which is Fig 5's conclusion.  Drawn on one axis with only a legend entry to
    separate them, the dashed v2 curve reverses that conclusion for anyone who reads the
    figure without the surrounding text -- and the figure is the object that goes into the
    rebuttal.  So while the root is v2 the step-time overlay is replaced by
    ``profile_note`` printed ON the axes; pass ``profile_step_times=True`` once
    ``profile_helpers.results_root()`` is v3, which was measured with ``empty_cache`` off
    and is comparable.
    """
    ax_m, ax_t = np.atleast_1d(axes)
    x = pd.to_numeric(tbl['actual'], errors='coerce').to_numpy(dtype=float)
    for ax, col, ylabel in ((ax_m, 'peak_mem_mb', 'Peak GPU memory (MB)'),
                            (ax_t, 'ms_per_step', 'Time per step (ms)')):
        v = pd.to_numeric(tbl[col], errors='coerce').to_numpy(dtype=float)
        s = pd.to_numeric(tbl[f'{col}_std'], errors='coerce').to_numpy(dtype=float)
        ax.errorbar(x, v, yerr=style.clipped_yerr(v, s, v - np.nan_to_num(s)),
                    marker=marker, ls=ls, color=style.method_color('Sven'), capsize=3,
                    label=record_label)
        ax.set_xscale('log')
        ax.set_xlabel('Actual parameter fraction $f$')
        ax.set_ylabel(ylabel)
        style.band_legend(ax)
    have_profiles = profiles is not None and len(profiles)
    if have_profiles:
        panels = [(ax_m, 'peak_mb')] + ([(ax_t, 'step_ms')] if profile_step_times else [])
        for ax, col in panels:
            for mode, sub in profiles.groupby('mask_mode'):
                sub = sub.sort_values('pf')
                ax.plot(sub['pf'], pd.to_numeric(sub[col], errors='coerce'), 's--',
                        ms=4, lw=1.2, alpha=0.8, label=f'profile, mask={mode}')
    if have_profiles and not profile_step_times and profile_note:
        ax_t.text(0.03, 0.03, profile_note, transform=ax_t.transAxes, fontsize=7,
                  va='bottom', ha='left', color='0.3', linespacing=1.25,
                  bbox=dict(boxstyle='round,pad=0.35', fc='0.96', ec='0.8', lw=0.6))
    for ax in (ax_m, ax_t):
        ax.legend(fontsize=7)
    return tbl


# ---------------------------------------------------------------------------
# GPT-2: one seed, one epoch, step-based evaluations
# ---------------------------------------------------------------------------
#: The GPT-2 runs evaluate every ``eval_every_steps`` optimizer steps rather than per
#: epoch: ``losses.val_step`` / ``test_step`` are the curves and ``losses.eval_step_idx``
#: the step each entry was taken at.  The per-EPOCH ``val`` / ``test`` curves exist too
#: but have two entries (before and after the single epoch), so a convergence plot must
#: use the step curves.
GPT2_STEP_CURVES = ('val_step', 'test_step')


def add_run_cost(df):
    """Add the per-run cost columns of :func:`headline.run_efficiency` to a frame.

    ``wall_s`` / ``wall_h`` (``losses.total_time``), ``ms_per_step`` (the SYNCHRONISED
    training time ``losses.train_times`` over the steps actually taken, C-T1),
    ``sync_train_s``, ``steps``, ``examples`` and ``peak_gpu_mem_mb``.  One function, so a
    step time means the same thing in a GPT-2 table as in :func:`headline.efficiency_table`
    -- and so a notebook cannot quietly divide a wall clock by an epoch count instead.
    """
    eff = hl.run_efficiency(df).set_index('run_id')
    out = df.copy()
    for col in ('wall_s', 'ms_per_step', 'peak_gpu_mem_mb', 'steps', 'examples',
                'sync_train_s', 'epoch_s'):
        out[col] = out['run_id'].map(eff[col])
    out['wall_h'] = out['wall_s'] / 3600.0
    return out


def gpt2_frame(results_root=None, scan=GPT2_SCAN):
    """The GPT-2 comparison as a frame, with the per-run derived columns.

    ONE model seed (6000), so there is no seed band anywhere in this notebook and no
    confirmation pass: every number is a single run and is reported as one.  The frame
    carries ``method`` (``SVD`` -> ``Sven``), the final val / test losses, wall time, peak
    memory and ``ms_per_step`` (:func:`add_run_cost`).
    """
    df = add_run_cost(ah.add_derived(style.load_results(scan, results_root=results_root)))
    df['n_seeds'] = 1
    return df


def gpt2_best_per_method(df, metric='final_val_loss'):
    """The best learning rate per method, by final VALIDATION loss.

    ``style.assert_selection_metric`` refuses a test metric here as everywhere else: with
    one seed and one epoch the temptation to quote the best test number is exactly the
    error the campaign's selection rule exists to prevent.  Returns one row per method,
    Sven first, with ``n_configs`` = how many learning rates were tried.
    """
    style.assert_selection_metric(metric, 'large_figs.gpt2_best_per_method')
    ok = df[~df['diverged'] & ~df['failed']]
    rows = []
    for method in ah.method_order(df['method'].unique()):
        sub = ok[ok['method'] == method]
        allsub = df[df['method'] == method]
        if not len(sub):
            continue
        best = sub.loc[pd.to_numeric(sub[metric], errors='coerce').idxmin()]
        rows.append({'method': method, 'display': style.method_label(method),
                     'lr': best['lr'], 'k': best.get('k'),
                     'val': best[metric], 'test': best.get('final_test_loss'),
                     'val_ppl': float(np.exp(min(best[metric], 20))),
                     'wall_h': best['wall_h'], 'ms_per_step': best['ms_per_step'],
                     'peak_gpu_mem_mb': best['peak_gpu_mem_mb'],
                     'run_id': best['run_id'],
                     'n_configs': int(len(allsub)),
                     'n_diverged': int(allsub['diverged'].sum()),
                     'finished': int(len(sub)), 'attempted': int(len(allsub))})
    out = pd.DataFrame(rows)
    out.attrs['metric'] = metric
    out.attrs['n_seeds'] = 1
    return out


def gpt2_lr_table(df, metric='final_val_loss'):
    """Every (method, learning rate) of the GPT-2 comparison: the lr-sensitivity table.

    One run per cell, so this is the whole evidence about how sharply each method depends
    on its learning rate -- with 1 seed a difference smaller than the seed spread of the
    other scans cannot be called a difference, which is why the notebook's framing says
    so rather than ranking the middle of the field.
    """
    style.assert_selection_metric(metric, 'large_figs.gpt2_lr_table')
    out = df[['method', 'lr', 'k', metric, 'final_test_loss', 'wall_h',
              'peak_gpu_mem_mb', 'ms_per_step', 'diverged', 'failed', 'status']].copy()
    out['display'] = [style.method_label(m) for m in out['method']]
    out = out.rename(columns={metric: 'val', 'final_test_loss': 'test'})
    order = {m: i for i, m in enumerate(ah.method_order(out['method'].unique()))}
    out['_o'] = out['method'].map(order)
    return out.sort_values(['_o', 'lr']).drop(columns='_o').reset_index(drop=True)


def gpt2_step_curve(row, key='val_step'):
    """``(steps, values)`` of one GPT-2 run's step-based evaluation curve."""
    L = row.get('losses')
    if not isinstance(L, dict) or L.get(key) is None:
        return None, None
    idx = L.get('eval_step_idx')
    v = np.asarray(L[key], dtype=float)
    x = np.asarray(idx, dtype=float) if idx is not None else np.arange(1, len(v) + 1)
    n = min(len(x), len(v))
    return x[:n], v[:n]


def plot_gpt2_curves(ax, rows, key='val_step', label_fn=None, color_fn=None,
                     lw_sven=2.6, lw=1.6, **plot_kw):
    """One line per run of ``rows`` -- the step-based evaluation curve.

    No band: one seed.  ``label_fn`` / ``color_fn`` default to the method's display name
    and colour, which is what the "best lr per method" panel wants; the "all five Sven
    lrs" panel passes its own.
    """
    drawn = []
    for _, r in rows.iterrows():
        x, v = gpt2_step_curve(r, key)
        if x is None:
            continue
        method = r['method']
        ax.plot(x, v, lw=lw_sven if method == 'Sven' else lw,
                color=(color_fn(r) if color_fn else style.method_color(method)),
                label=(label_fn(r) if label_fn
                       else f"{style.method_label(method)} (lr={r['lr']:g})"),
                zorder=3 if method == 'Sven' else 2, **plot_kw)
        drawn.append(r['run_id'])
    ax.set_xlabel('Optimizer step')
    ax.set_ylabel('Validation loss' if key == 'val_step' else 'Test loss')
    return drawn


def gpt2_summary_view(best, spec='.4g'):
    """The GPT-2 final table a reader sees: quality, cost and what it cost to get there."""
    if not len(best):
        return pd.DataFrame()
    ref = pd.to_numeric(best['wall_h'], errors='coerce')
    fastest = np.nanmin(ref)
    out = []
    for i, r in best.sort_values('val').iterrows():
        out.append({
            'method': r['display'],
            'best lr': f"{r['lr']:g}",
            'val loss': format(float(r['val']), spec),
            'val ppl': format(float(r['val_ppl']), '.4g'),
            'test loss': ('--' if not np.isfinite(pd.to_numeric(r['test'], errors='coerce'))
                          else format(float(r['test']), spec)),
            'wall (h)': format(float(r['wall_h']), '.3g'),
            'x fastest': ('--' if not np.isfinite(r['wall_h'] / fastest)
                          else f"{r['wall_h'] / fastest:.2f}x"),
            'ms/step': format(float(r['ms_per_step']), '.4g'),
            'peak mem (MB)': format(float(r['peak_gpu_mem_mb']), '.5g'),
            'lrs tried (fin/att)': hl.fmt_counts(r['finished'], r['attempted']),
        })
    view = pd.DataFrame(out)
    view.attrs['n_seeds'] = 1
    return view
