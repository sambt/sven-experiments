"""The paper's headline tables for the seven scans (ANALYSIS_PLAN.md section 2).

Everything here is driven by ONE input, ``bench/best_configs.json`` -- the selection of
record written by :mod:`tools.select_best` under the binding rule (eligible -> fewest
diverged seeds -> lowest seed-mean final VALIDATION loss).  This module never re-selects:
it takes each method's selected configuration and reports what that configuration did, so
a table here can never disagree with the timing / diagnostics / confirmation passes, which
were generated from the same file (``tools/gen_phase5_plan.py``).

Four result directories per scan, all read-only:

* ``{scan}/``          -- the tuning grid the selection was made on (5 model seeds);
* ``{scan}_confirm/``  -- the selected configurations on 5 FRESH model seeds (base+100..104),
  and on 3 DATA seeds for toy / polynomial (``data_seed`` joins ``result_id_fields`` in the
  ``_confirm`` config, so a replicate has its own run_id);
* ``{scan}_timing/``   -- the selected configurations alone on a GPU, logging off: the only
  place wall times may be quoted from;
* ``{scan}_diag/``     -- not used here (WP3's spectra).

**Headline numbers are the CONFIRMATION seeds** (decision 1 of the plan), with the tuning
seeds beside them; their difference is the selection optimism and is reported as
``gap_val``.  Diverged = failed (:func:`style.is_diverged`): never in a mean, always
counted, and every table carries ``finished / attempted`` (the manifest's expectation,
:func:`style.expected_runs`).  Seed bands are mean +/- 1 std (ddof=1),
:data:`paired.SEED_SPREAD_LABEL`.

**Selection never reads a test metric.**  Every ranking metric passes through
:func:`style.assert_selection_metric`, and :func:`assert_no_test_selection` checks the
selection file itself plus the two chokepoints, as an executable assertion rather than a
comment.  ``final_test_loss`` / ``final_test_acc`` appear only as OUTCOMES of the
configuration validation already chose.

Time-to-target uses targets that are **pre-declared from validation only**, once per scan
(:func:`targets_for`): the median method's confirmation-seed mean final validation loss,
and 2x / 0.5x of it.  Runs that never reach a target are counted, never dropped silently.

    import headline as hl
    hl.assert_no_test_selection()
    hl.confirmation_table('mnist_scan_ce')
    hl.paired_vs_sven('mnist_scan_ce')
    hl.efficiency_table('mnist_scan_ce')
    hl.ranking_summary()
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import analysis_helpers as ah
import budget
import paired
import style

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

#: the selection of record -- `tools/select_best.py`'s output, schema 2
SELECTION_PATH = REPO / 'bench' / 'best_configs.json'

#: where the exported markdown / LaTeX tables go
TABLES_DIR = HERE / 'tables'

#: the seven headline scans, in report order (`tools.select_best.HEADLINE_SCANS`)
HEADLINE_SCANS = (
    'toy_1d_scan',
    'polynomial_scan',
    'mnist_scan_labelRegression',
    'mnist_scan_ce',
    'cifar10_resnet_scan_labelRegression',
    'cifar10_resnet_ce_scan',
    'exp_nanogpt_speedrun',
)

#: scan -> the key it goes by in :data:`style.DATASET_TITLES` (one spelling everywhere)
TITLE_KEYS = {
    'toy_1d_scan': 'toy_1d',
    'polynomial_scan': 'polynomial',
    'mnist_scan_labelRegression': 'mnist_labelreg',
    'mnist_scan_ce': 'mnist_ce',
    'cifar10_resnet_scan_labelRegression': 'cifar_labelreg',
    'cifar10_resnet_ce_scan': 'cifar_ce',
    'exp_nanogpt_speedrun': 'nanogpt',
}

#: Sven's `optimizer` value in the records, and the label it is reported under
SVEN_METHOD = 'SVD'
SVEN_LABEL = 'Sven'

#: the metric every selection and every ranking in this module uses
SELECTION_METRIC = 'final_val_loss'

#: outcome columns reported beside a selected configuration (never selected on, C-E1)
OUTCOME_METRICS = ('final_test_loss', 'final_test_acc')

#: metrics a LARGER value of which is the better outcome.  Everything else this module
#: reports is a loss, and the sign of a paired difference means the opposite thing for
#: the two -- :mod:`paired` documents its convention for a loss only ("a < b means A is
#: better"), so a table reporting a difference in an ACCURACY must flip the count.
_HIGHER_IS_BETTER_RE = re.compile(r'(^|_)(acc|accuracy|auc)(_|$)')


def higher_is_better(metric):
    """Whether a LARGER value of ``metric`` is the better outcome (an accuracy).

    The direction is derived from the metric's NAME rather than passed in by the caller,
    so no table can be built that forgets it: :func:`paired_outcome_vs_sven` counts wins
    and orders its rows through this, and a mis-signed accuracy row reads as exactly the
    opposite of what happened (Sven's CIFAR-CE accuracy is 0.53 against Muon's 0.78; the
    paired mean is -0.25 on 5 of 5 seeds, which is five LOSSES, not five wins).
    """
    return bool(_HIGHER_IS_BETTER_RE.search(str(metric)))

#: `bench/best_configs.json` names some hyperparameters as the GRID names them
#: (`optim_name`, `max_iter`, ...), the records as the run records name them.  Matching a
#: selection to its runs goes through this map; a name missing from it would silently
#: match nothing, so :func:`selected_runs` reports every hparam it could not use.
HPARAM_ALIAS = {
    'optim_name': 'optimizer',
    'max_iter': 'lbfgs_max_iter',
    'history_size': 'lbfgs_history_size',
    'line_search_fn': 'lbfgs_line_search_fn',
    'f_star': 'polyak_f_star',
    'max_lr': 'polyak_max_lr',
    'eps': 'polyak_eps',
}

#: time-to-target multipliers applied to :func:`target_reference` (the median method's
#: confirmation-seed mean final validation loss).  Pre-declared: fixed here, once, for
#: every scan, so no target can be chosen after seeing which method reaches it.
TARGET_MULTIPLIERS = (2.0, 1.0, 0.5)
TARGET_NAMES = {2.0: 'easy (2x median)', 1.0: 'median method', 0.5: 'hard (0.5x median)'}

#: where `bench/submit_timing_phase5.sh` puts the timing jobs' logs, which carry the
#: `bench/calibrate_step.py` lines a step-time claim has to be checked against
TIMING_LOG_DIR = Path('/n/holystore01/LABS/iaifi_lab/Users/sambt/sv3_campaign_scratch/'
                      'logs/phase5')


def scan_title(scan):
    """The scan's one spelling (:data:`style.DATASET_TITLES`)."""
    return style.DATASET_TITLES.get(TITLE_KEYS.get(scan, scan), scan)


def method_key(method):
    """The KEY an optimizer goes by everywhere in this analysis: ``'SVD'`` -> ``'Sven'``
    (:func:`style.canonical_method`).

    The key is what a ``method`` column, a groupby and :data:`style.METHOD_COLORS` use;
    :func:`display_name` is what a reader sees.
    """
    return style.canonical_method(method)


def display_name(method):
    """The name a table column or a legend SHOWS (:func:`style.method_label`, C-B7):
    ``LBFGS`` is ``torch.optim.LBFGS`` on minibatches, so it reads "Stochastic L-BFGS",
    and ``SGDm`` reads "SGD + momentum".

    Falls back to the key where the registry has no display names yet, so this module
    never depends on the order in which the analysis modules are updated.
    """
    label = getattr(style, 'method_label', None)
    return label(method) if callable(label) else method_key(method)


# ---------------------------------------------------------------------------
# The selection of record
# ---------------------------------------------------------------------------
_selection_memo = {}


def load_selection(path=None):
    """``bench/best_configs.json`` as a dict, memoised per path.

    Refuses anything that is not a schema-2 selection file: the pre-campaign file is a
    list of ``[method, cfg, ...]`` rows with no verified overrides, and reading it would
    silently report a different configuration than the passes ran.
    """
    path = Path(path or SELECTION_PATH)
    key = str(path)
    if key not in _selection_memo:
        with open(path) as fh:
            payload = json.load(fh)
        schema = payload.get('schema') if isinstance(payload, dict) else None
        if schema != 2:
            raise ValueError(f'{path} is not a schema-2 selection file '
                             f'(schema={schema!r}, {type(payload).__name__}); '
                             f'run tools/select_best.py')
        _selection_memo[key] = payload
    return _selection_memo[key]


def selection_methods(scan, payload=None):
    """``{method: selection}`` for one scan, from the selection of record."""
    payload = payload or load_selection()
    try:
        return dict(payload['scans'][scan]['methods'])
    except KeyError:
        raise KeyError(f'{scan!r} is not in the selection file '
                       f'({sorted(payload.get("scans", {}))})') from None


def method_order(methods):
    """Sven first, then the baselines alphabetically (:func:`analysis_helpers.method_order`
    on the record-level names)."""
    ms = list(methods)
    front = [m for m in ms if method_key(m) == SVEN_LABEL]
    return front + sorted(m for m in ms if method_key(m) != SVEN_LABEL)


def config_label(sel):
    """A short, human-readable statement of what was selected.

    The authoritative statement is ``sel['overrides']`` (verified to expand to exactly the
    selected run_ids); this is the same content without the hydra noise.
    """
    skip = {'optim_name', 'svd_mode', 'line_search_fn'}
    bits = []
    for name, value in (sel.get('hparams') or {}).items():
        if name in skip or value is None:
            continue
        if name == 'weight_decay' and not value:
            continue                                   # wd = 0 is the norm
        if name == 'kappa' and float(value) == 2.0:
            continue                                   # kappa = 2 is the default
        bits.append(f'{name}={value:g}' if isinstance(value, (int, float))
                    and not isinstance(value, bool) else f'{name}={value}')
    return ', '.join(bits) or '(no swept hyperparameter)'


def record_hparams(sel):
    """The selected configuration as ``{record column: value}`` (see :data:`HPARAM_ALIAS`).

    ``optimizer`` and ``batch_size`` are added from the selection's own fields, so a
    family whose grid names carry no optimizer (svd / jd / hig / polyak / lbfgs) is still
    matched on the optimizer it is.
    """
    out = {}
    for name, value in (sel.get('hparams') or {}).items():
        out[HPARAM_ALIAS.get(name, name)] = value
    out.setdefault('optimizer', sel['method'])
    if sel.get('batch_size') is not None:
        out.setdefault('batch_size', sel['batch_size'])
    return out


def assert_no_test_selection(payload=None, scans=None):
    """``selection uses validation only`` -- as an executable assertion (C-E1).

    Three things are checked, none of them by inspection:

    1. the selection file scores nothing but ``seed_mean_final_val`` and carries no test
       key anywhere in a selection entry (``tools/select_best.assert_no_test_metric``'s
       check, re-run here on the file the tables are actually built from);
    2. every metric this module ranks on passes :func:`style.assert_selection_metric`;
    3. the two chokepoints -- :meth:`analysis_helpers.config_table` and
       :func:`budget.best_of_n_curve` -- still REFUSE a test metric, so a future edit that
       passed one in would raise rather than produce a table.

    Returns the number of (scan, method) selections checked.
    """
    payload = payload or load_selection()
    scans = HEADLINE_SCANS if scans is None else scans
    test_keys = ('test', 'test_acc', 'test_step', 'final_test_loss', 'final_test_acc')
    n = 0
    for scan in scans:
        for method, sel in selection_methods(scan, payload).items():
            n += 1
            bad = [k for k in test_keys if k in sel]
            if bad:
                raise AssertionError(
                    f'{scan}/{method}: the selection carries test key(s) {bad}; '
                    f'selection uses the seed-mean final VALIDATION loss only')
            if 'seed_mean_final_val' not in sel:
                raise AssertionError(f'{scan}/{method}: no seed_mean_final_val -- what '
                                     f'was this selected on?')
    style.assert_selection_metric(SELECTION_METRIC, 'headline.assert_no_test_selection')
    for metric in OUTCOME_METRICS:
        if not style.is_test_metric(metric):
            raise AssertionError(f'{metric!r} is reported as a test outcome but '
                                 f'style.is_test_metric does not recognise it')
    probe = pd.DataFrame({'run_id': ['a_mseed1_lseed1'], 'optimizer': ['SVD'],
                          'model_seed': [1], 'lr': [0.1],
                          'losses': [{'train': [1.0], 'val': [1.0, 0.5],
                                      'test': [1.0, 0.5]}]})
    for name, call in (('analysis_helpers.config_table',
                        lambda: ah.config_table(probe, metric='final_test_loss')),
                       ('budget.best_of_n_curve',
                        lambda: budget.best_of_n_curve(probe, metric='final_test_loss'))):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f'{name} accepted a test metric as a selector')
    return n


# ---------------------------------------------------------------------------
# Loading (read-only; the slim cache under {root}/_cache is the one allowed write)
# ---------------------------------------------------------------------------
_frame_memo = {}


def dir_name(scan, kind=''):
    """The results directory of one pass: ``''`` / ``'confirm'`` / ``'timing'`` / ``'diag'``."""
    return scan if not kind else f'{scan}_{kind}'


def load(scan, kind='', results_root=None, cache_dir=None, verbose=False):
    """One pass of one scan as a DataFrame with the derived columns, memoised.

    ``analysis_helpers.add_derived`` is applied once (final losses, ``final_test_*`` when
    the records carry a test curve, ``method``, ``diverged`` / ``failed``), so every table
    below shares one definition of "diverged" and one of "the final loss".
    """
    root = str(style.resolve_results_root(results_root))
    name = dir_name(scan, kind)
    key = (name, root)
    if key not in _frame_memo:
        df = style.load_results(name, results_root=results_root, cache_dir=cache_dir)
        _frame_memo[key] = ah.add_derived(df)
        if verbose:
            print(f'[headline] {name}: {len(df)} run(s)')
    return _frame_memo[key]


def clear_cache():
    """Forget the memoised frames and the selection (used by the tests, which point
    ``$SV3_RESULTS_ROOT`` at a fresh tmp_path per test)."""
    _frame_memo.clear()
    _selection_memo.clear()


def has_pass(scan, kind, results_root=None):
    """Whether ``{scan}_{kind}/`` exists at all (a pass that has not run yet)."""
    root = Path(style.resolve_results_root(results_root))
    return (root / dir_name(scan, kind)).is_dir()


# ---------------------------------------------------------------------------
# A selected configuration -> its runs, in any pass
# ---------------------------------------------------------------------------
def _matches(series, value):
    """Element-wise "this column equals this selected value", NaN-aware.

    A selected ``None`` (Sven's ``microbatch_size`` / ``param_fraction``, PolyakSGD's
    ``lr``) means the column must be ABSENT for the run, not "do not care": matching it
    against NaN is what keeps a microbatched twin of the configuration out of the table.
    """
    if value is None:
        return series.isna()
    if isinstance(value, bool):
        return series == value
    if isinstance(value, (int, float, np.integer, np.floating)):
        num = pd.to_numeric(series, errors='coerce')
        return pd.Series(np.isclose(num.to_numpy(dtype=float), float(value),
                                    rtol=1e-12, atol=0.0, equal_nan=False),
                         index=series.index)
    return series.astype('object') == value


def selected_runs(df, sel, seeds=None):
    """The runs of one selected configuration, plus a small report.

    Matching is on the hyperparameter VALUES (:func:`record_hparams`), not on the run_id
    string: the ``_confirm`` configs insert ``data_seed`` into the run_id, so a selected
    ``config_key`` from the tuning scan is not a substring of its own confirmation runs.

    ``seeds`` optionally restricts to a set of ``model_seed`` values.  Returns
    ``(rows, report)`` where ``report`` carries the columns used, the hyperparameters that
    could not be checked (a column the pass does not record), and the ``config_key``s the
    match landed on -- which is what proves the table describes one configuration.
    """
    hp = record_hparams(sel)
    mask = pd.Series(True, index=df.index)
    used, unusable = [], []
    for col, value in sorted(hp.items()):
        if col not in df.columns:
            if value is not None:
                unusable.append(f'{col}={value!r} (column absent)')
            continue
        mask &= _matches(df[col], value)
        used.append(col)
    rows = df[mask]
    if seeds is not None and 'model_seed' in rows.columns:
        rows = rows[pd.to_numeric(rows['model_seed'], errors='coerce').isin(list(seeds))]
    keys = sorted({style.config_key(r) for r in rows.get('run_id', [])})
    return rows, {'n': len(rows), 'columns_used': used, 'unusable': unusable,
                  'config_keys': keys}


def method_runs(scan, kind='', payload=None, results_root=None):
    """``{method: (rows, report)}`` for every selected method of ``scan`` in one pass."""
    df = load(scan, kind, results_root=results_root)
    out = {}
    for method, sel in selection_methods(scan, payload).items():
        out[method] = selected_runs(df, sel)
    return out


def data_seeds(scan, kind='confirm', results_root=None):
    """The ``data_seed`` values a pass contains (``[]`` where the scan has none).

    Toy and polynomial get 3 data-seed replicates (F27); every other scan has no
    ``data_seed`` at all, and the column is then absent or all-NaN.
    """
    df = load(scan, kind, results_root=results_root)
    if 'data_seed' not in df.columns:
        return []
    vals = pd.to_numeric(df['data_seed'], errors='coerce').dropna().unique()
    return sorted(int(v) for v in vals)


def _attempted(rows, scan, kind, results_root=None):
    """How many runs the pass's MANIFEST expects of the matched configuration (C-R2).

    Falls back to the number of matched records for a pass with no manifest, which is what
    every table did before manifests existed.
    """
    expected = style.manifest_run_ids(dir_name(scan, kind), results_root=results_root)
    keys = {style.config_key(r) for r in rows.get('run_id', [])}
    if not expected or not keys:
        return int(len(rows))
    n_seeds = int(rows['model_seed'].nunique()) if 'model_seed' in rows.columns else len(rows)
    return int(style.expected_runs(keys, style.expected_per_config(expected), n_seeds))


def _stats(rows, column):
    """``(mean, std, n)`` over the runs that produced a value, ddof=1 (0 for n=1).

    Diverged and ``oom`` / ``error`` runs have NaN in every outcome column
    (``analysis_helpers.add_derived``), so they drop out here and are counted separately.
    """
    if column not in rows.columns:
        return np.nan, np.nan, 0
    v = pd.to_numeric(rows[column], errors='coerce').to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size)


def _curve_stats(rows, key):
    """``(mean, std, n)`` of the FINAL value of a per-epoch curve, over the usable runs.

    For the curves ``analysis_helpers.add_derived`` does not flatten -- ``train_eval``,
    the training loss in EVAL mode on a fixed subset, which is what separates "Sven fits
    the training data and generalises worse" from "Sven does not optimise at all" on the
    two CIFAR scans.  Diverged / failed runs are skipped, exactly as in :func:`_stats`.
    """
    vals = []
    for _, r in rows.iterrows():
        if r.get('diverged', False) or r.get('failed', False):
            continue
        L = r.get('losses')
        curve = L.get(key) if isinstance(L, dict) else None
        v = style.final_value(curve)
        if np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return np.nan, np.nan, 0
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, a.size


def _counts(rows, scan, kind, results_root=None):
    """``finished / attempted`` plus the divergence / failure / missing counts."""
    div = int(rows['diverged'].sum()) if 'diverged' in rows.columns else 0
    fail = int(rows['failed'].sum()) if 'failed' in rows.columns else 0
    finished = int(len(rows)) - div - fail
    attempted = _attempted(rows, scan, kind, results_root=results_root)
    return {'finished': finished, 'attempted': attempted, 'n_diverged': div,
            'n_failed': fail, 'n_missing': max(attempted - len(rows), 0)}


# ---------------------------------------------------------------------------
# (a) the confirmation table
# ---------------------------------------------------------------------------
def confirmation_table(scan, payload=None, results_root=None):
    """The paper's per-scan table: the selected configuration of every method, on the
    CONFIRMATION seeds, with the tuning seeds beside it.

    One row per method.  Columns (``*_conf`` = the 5 fresh model seeds of
    ``{scan}_confirm``, ``*_tune`` = the tuning seeds the selection was made on):

    * ``val_conf`` / ``val_conf_std`` -- final validation loss, mean +/- 1 std over seeds;
    * ``test_conf`` / ``acc_conf`` (+ ``_std``) -- final test loss / accuracy: OUTCOMES of
      the configuration the validation loss chose, never a selector;
    * ``treval_conf`` (+ ``_std``) -- the final TRAIN loss in eval mode on a fixed subset
      (``losses.train_eval``), which separates "fits the training data and generalises
      worse" from "does not optimise at all" -- the two different CIFAR stories;
    * ``fin_conf`` / ``att_conf`` / ``div_conf`` -- finished / attempted / diverged, and
      ``elig_conf`` -- whether MORE THAN HALF of the confirmation runs finished
      (:func:`style.config_eligible`).  A configuration can be eligible on the seeds it
      was tuned on and not on the fresh ones (polynomial L-BFGS: 3/5 and then 2/15), and
      its confirmation mean is then a mean over the survivors, not the method's loss;
    * the same five for the tuning seeds, and
    * ``gap_val = val_conf - val_tune`` with ``gap_val_rel`` -- the selection optimism
      (positive = the tuning seeds flattered the configuration);
    * ``val_conf_dsmean`` / ``val_conf_dsspread`` / ``val_conf_seedspread`` /
      ``val_conf_ds0`` / ``n_data_seeds`` -- for toy and polynomial, the mean over the 3
      data-seed replicates, the spread BETWEEN the instances (std over the per-data-seed
      means, ddof=1), the within-instance seed spread averaged over instances, and the
      replicate the tuning scan itself used.  The between-instance spread answers how much
      the answer depends on which random problem was drawn (F27) and is NOT the seed band;
      ``val_conf_std`` pools both and is therefore the wider number.  NaN elsewhere.
    * ``gap_val_same_instance`` (+ ``_rel``) -- the gap computed on ``val_conf_ds0``, i.e.
      fresh model seeds on the problem the tuning ran on.  On every scan without data
      seeds it equals ``gap_val``; on toy / polynomial ``gap_val`` also carries two unseen
      data sets, and the difference is not small -- Sven's toy gap is +77% pooled and
      -63% same-instance, so the pooled number is mostly a statement about data seeds
      1001 / 1002 being harder problems, not about the selection.  **The
      selection-optimism number is the same-instance one**
      (:func:`selection_optimism_table`).

    Sorted by ``val_conf``, so the row order IS the ranking on the confirmation seeds.
    """
    style.assert_selection_metric(SELECTION_METRIC, 'headline.confirmation_table')
    sels = selection_methods(scan, payload)
    tune = load(scan, '', results_root=results_root)
    have_confirm = has_pass(scan, 'confirm', results_root)
    conf = load(scan, 'confirm', results_root=results_root) if have_confirm else None
    seeds = data_seeds(scan, 'confirm', results_root=results_root) if have_confirm else []

    rows = []
    for method in method_order(sels):
        sel = sels[method]
        t_rows, t_rep = selected_runs(tune, sel)
        entry = {'method': method_key(method), 'display': display_name(method),
                 'config': config_label(sel),
                 'overrides': sel['overrides'], 'verified': bool(sel.get('verified'))}
        # --- tuning seeds: the runs the selection itself scored -------------
        t_counts = _counts(t_rows, scan, '', results_root=results_root)
        v, s, _ = _stats(t_rows, SELECTION_METRIC)
        entry.update({'val_tune': v, 'val_tune_std': s,
                      'fin_tune': t_counts['finished'], 'att_tune': t_counts['attempted'],
                      'div_tune': t_counts['n_diverged'],
                      'fail_tune': t_counts['n_failed'],
                      'miss_tune': t_counts['n_missing'],
                      'sel_seed_mean_val': sel['seed_mean_final_val']})
        for col, name in zip(OUTCOME_METRICS, ('test_tune', 'acc_tune')):
            v, s, _ = _stats(t_rows, col)
            entry[name], entry[f'{name}_std'] = v, s
        entry['treval_tune'], entry['treval_tune_std'], _ = _curve_stats(t_rows,
                                                                        'train_eval')
        # --- confirmation seeds: the numbers to report ----------------------
        if have_confirm:
            c_rows, c_rep = selected_runs(conf, sel)
            c_counts = _counts(c_rows, scan, 'confirm', results_root=results_root)
            v, s, _ = _stats(c_rows, SELECTION_METRIC)
            entry.update({'val_conf': v, 'val_conf_std': s,
                          'fin_conf': c_counts['finished'],
                          'att_conf': c_counts['attempted'],
                          'div_conf': c_counts['n_diverged'],
                          'fail_conf': c_counts['n_failed'],
                          'miss_conf': c_counts['n_missing'],
                          'config_keys_conf': c_rep['config_keys']})
            for col, name in zip(OUTCOME_METRICS, ('test_conf', 'acc_conf')):
                v, s, _ = _stats(c_rows, col)
                entry[name], entry[f'{name}_std'] = v, s
            entry['treval_conf'], entry['treval_conf_std'], _ = _curve_stats(
                c_rows, 'train_eval')
            # the data-seed replicates (toy / polynomial): three DIFFERENT problems, so
            # the spread between their means is variation of the target, not of the
            # optimiser's luck -- a different quantity from the seed band, and the two
            # must not be added into one number.  `val_conf_ds0` is the replicate the
            # TUNING scan itself used (gen_phase5_plan numbers them base .. base+2), which
            # is the only one whose gap against the tuning seeds is a clean statement
            # about the seeds rather than about the problem.
            if len(seeds) > 1 and 'data_seed' in c_rows.columns:
                per_ds, within = [], []
                for ds in seeds:
                    sub = c_rows[_matches(c_rows['data_seed'], ds)]
                    m, s, n = _stats(sub, SELECTION_METRIC)
                    if n:
                        per_ds.append(m)
                        if np.isfinite(s):
                            within.append(s)
                    if ds == min(seeds):
                        entry['val_conf_ds0'] = m
                entry['n_data_seeds'] = len(per_ds)
                entry['val_conf_dsmean'] = float(np.mean(per_ds)) if per_ds else np.nan
                entry['val_conf_dsspread'] = (float(np.std(per_ds, ddof=1))
                                              if len(per_ds) > 1 else np.nan)
                entry['val_conf_seedspread'] = float(np.mean(within)) if within else np.nan
            else:
                entry['n_data_seeds'] = len(seeds)
                entry['val_conf_dsmean'] = np.nan
                entry['val_conf_dsspread'] = np.nan
                entry['val_conf_seedspread'] = entry['val_conf_std']
                entry['val_conf_ds0'] = entry['val_conf']
        else:
            for key in ('val_conf', 'val_conf_std', 'test_conf', 'test_conf_std',
                        'acc_conf', 'acc_conf_std', 'val_conf_dsmean',
                        'val_conf_dsspread', 'val_conf_seedspread', 'val_conf_ds0',
                        'treval_conf', 'treval_conf_std'):
                entry[key] = np.nan
            entry.update({'fin_conf': 0, 'att_conf': 0, 'div_conf': 0, 'fail_conf': 0,
                          'miss_conf': 0, 'n_data_seeds': 0, 'config_keys_conf': []})
        entry['gap_val'] = entry['val_conf'] - entry['val_tune']
        entry['gap_val_rel'] = (entry['gap_val'] / entry['val_tune']
                                if np.isfinite(entry['val_tune']) and entry['val_tune']
                                else np.nan)
        # the gap on the SAME problem instance: for toy / polynomial the headline gap also
        # carries two data sets the tuning scan never saw
        entry['gap_val_same_instance'] = entry['val_conf_ds0'] - entry['val_tune']
        entry['gap_val_same_instance_rel'] = (
            entry['gap_val_same_instance'] / entry['val_tune']
            if np.isfinite(entry['val_tune']) and entry['val_tune'] else np.nan)
        # A configuration can be eligible on the seeds it was tuned on and NOT eligible on
        # the confirmation seeds -- polynomial L-BFGS diverges on 2 of 5 tuning seeds and
        # on 13 of 15 confirmation runs.  Its confirmation mean is then a mean over the
        # two survivors and must not be read as "the loss of this method".
        entry['elig_tune'] = bool(style.config_eligible(entry['fin_tune'],
                                                        max(entry['att_tune'], 1)))
        entry['elig_conf'] = bool(style.config_eligible(entry['fin_conf'],
                                                        max(entry['att_conf'], 1)))
        rows.append(entry)
    out = pd.DataFrame(rows)
    out = out.sort_values('val_conf', na_position='last').reset_index(drop=True)
    out.attrs['scan'] = scan
    out.attrs['title'] = scan_title(scan)
    out.attrs['data_seeds'] = seeds
    out.attrs['seed_band'] = paired.SEED_SPREAD_LABEL
    return out


def selection_optimism_table(tables=None, scans=None, payload=None, results_root=None):
    """The cross-scan selection-optimism table: ``(confirmation - tuning) / tuning``.

    Two columns, and the distinction is the whole point.

    * ``*_same`` -- the gap on the problem instance the tuning scan itself ran on
      (``gap_val_same_instance_rel``).  Fresh model seeds, same data, same everything
      else: this IS selection optimism, and it is what the paper should quote.
    * ``*_pooled`` -- the gap against all confirmation runs.  On the five scans without
      data seeds it is the identical number; on toy and polynomial it compares 15 runs on
      3 data seeds against 5 tuning runs on 1, so most of it is problem-instance variation
      and for toy it has the opposite sign.  Recomputed: polynomial's median gap is
      +22.6% pooled with 15/15 methods worse, and -1.8% same-instance with 7/15 worse.

    ``n_data_seeds`` says where the two can differ at all.  ``tables`` optionally reuses
    already-computed confirmation tables (``{scan: table}``).
    """
    scans = HEADLINE_SCANS if scans is None else scans
    rows = []
    for scan in scans:
        tbl = (tables or {}).get(scan)
        if tbl is None:
            tbl = confirmation_table(scan, payload=payload, results_root=results_root)
        entry = {'scan': scan_title(scan), 'n_methods': len(tbl),
                 'n_data_seeds': int(pd.to_numeric(tbl['n_data_seeds'],
                                                   errors='coerce').max() or 0)}
        sven = tbl[tbl['method'] == SVEN_LABEL]
        for suffix, col in (('same', 'gap_val_same_instance_rel'),
                            ('pooled', 'gap_val_rel')):
            g = pd.to_numeric(tbl.get(col), errors='coerce').dropna()
            entry[f'median_{suffix}_%'] = 100 * float(np.median(g)) if len(g) else np.nan
            entry[f'worst_{suffix}_%'] = 100 * float(g.max()) if len(g) else np.nan
            entry[f'sven_{suffix}_%'] = (100 * float(sven.iloc[0][col])
                                         if len(sven) and np.isfinite(sven.iloc[0][col])
                                         else np.nan)
            entry[f'n_worse_{suffix}'] = int((g > 0).sum())
        rows.append(entry)
    out = pd.DataFrame(rows)
    out.attrs['same'] = ('gap on the tuning data instance = selection optimism '
                         '(fresh model seeds only)')
    out.attrs['pooled'] = ('gap over all confirmation runs; on toy / polynomial this also '
                           'carries 2 problem instances the tuning never saw')
    return out


def data_seed_table(scan, payload=None, results_root=None):
    """Per (method, data seed) confirmation numbers for toy / polynomial (F27).

    The between-instance question the seed band cannot answer: each data seed is a
    DIFFERENT random problem, so the spread over the three ``val`` columns is variation of
    the target, not of the optimiser's luck.  Empty for a scan without ``data_seed``.
    """
    seeds = data_seeds(scan, 'confirm', results_root=results_root) \
        if has_pass(scan, 'confirm', results_root) else []
    if len(seeds) < 2:
        return pd.DataFrame(columns=['method', 'data_seed', 'val', 'val_std', 'finished'])
    sels = selection_methods(scan, payload)
    conf = load(scan, 'confirm', results_root=results_root)
    rows = []
    for method in method_order(sels):
        c_rows, _ = selected_runs(conf, sels[method])
        for ds in seeds:
            sub = c_rows[_matches(c_rows['data_seed'], ds)]
            v, s, n = _stats(sub, SELECTION_METRIC)
            t, _, _ = _stats(sub, 'final_test_loss')
            div = int(sub['diverged'].sum()) if 'diverged' in sub.columns else 0
            rows.append({'method': method_key(method), 'data_seed': ds, 'val': v,
                         'val_std': s, 'test': t, 'finished': n, 'attempted': len(sub),
                         'n_diverged': div})
    out = pd.DataFrame(rows)
    out.attrs['scan'] = scan
    return out


def check_selection_reproduces(scan, payload=None, results_root=None, tol=1e-9):
    """Recompute each selection's score from the raw records and diff it.

    ``bench/best_configs.json`` was written by a torch-free tool that reads the JSONL
    directly; this recomputes the same seed-mean final validation loss through the
    analysis layer (:func:`analysis_helpers.add_derived`, :func:`style.is_diverged`) over
    the run_ids the selection NAMES.  A disagreement means the two definitions of
    "diverged" or "the final value" have drifted apart, which would make every table below
    describe something the passes did not run.
    """
    sels = selection_methods(scan, payload)
    tune = load(scan, '', results_root=results_root)
    by_id = tune.set_index('run_id')
    rows = []
    for method in method_order(sels):
        sel = sels[method]
        ids = [r for r in sel['run_ids'] if r in by_id.index]
        sub = by_id.loc[ids]
        mine, _, n = _stats(sub, SELECTION_METRIC)
        theirs = sel['seed_mean_final_val']
        rows.append({'method': method_key(method), 'n_run_ids': len(sel['run_ids']),
                     'n_records': len(ids), 'n_scored': n,
                     'select_best': theirs, 'recomputed': mine,
                     'abs_diff': abs(mine - theirs) if np.isfinite(mine) else np.nan,
                     'rel_diff': (abs(mine - theirs) / abs(theirs)
                                  if np.isfinite(mine) and theirs else np.nan),
                     'n_ok_select_best': sel['n_ok'], 'n_expected': sel['n_expected']})
    out = pd.DataFrame(rows)
    out.attrs['scan'] = scan
    out.attrs['max_rel_diff'] = float(np.nanmax(out['rel_diff'])) if len(out) else np.nan
    out.attrs['ok'] = bool(np.all(out['rel_diff'].fillna(0.0) <= tol))
    return out


#: ``tools/reconcile.CLAIM_TIMEOUT_S`` / ``experiments/experiment_code/claims.py``: a
#: ``claims/{run_id}.claim`` whose mtime is younger than this means a live worker has it
CLAIM_TIMEOUT_S = 600.0


def freshness_report(scans=None, kinds=('', 'confirm', 'timing'), payload=None,
                     results_root=None, now=None):
    """Is the selection of record still the selection of what is on disk?

    Every table in this module is built from ``bench/best_configs.json``, which was
    written at one instant against the records that existed then.  A scan that is still
    running does not make a table wrong -- it makes it PROVISIONAL, which is a different
    thing and has to be said out loud.  Per scan and pass:

    * ``n_records`` / ``newest_record`` -- the ``*.jsonl`` files and the newest mtime;
    * ``n_after_selection`` -- records written AFTER the selection's ``generated_at``.  On
      the TUNING pass those are grid points the selection never scored, so the selection
      itself may be out of date.  On ``_confirm`` / ``_timing`` they are expected: those
      passes were generated FROM the selection and are necessarily newer, which is why
      ``selection_input`` marks which pass the count means something for;
    * ``n_claims_live`` -- ``claims/{run_id}.claim`` files younger than
      :data:`CLAIM_TIMEOUT_S` with no record yet: runs in flight right now (the same
      contract ``tools/reconcile.py`` calls ``claimed-live``);
    * ``provisional`` -- a live claim in any pass, or a post-selection record in the
      tuning grid.

    At the time of writing this fires on ``cifar10_resnet_ce_scan``: plan decision 5's
    ``rtol`` extension (``mode=svd k_values=[128] lrs=[0.05,0.1,0.5] rtol=[0.03,0.1,0.3]``,
    45 runs) is landing, and decision 5 says ``tools/select_best.py`` must be re-run
    afterwards -- so Sven's selected CIFAR-CE configuration, its rank and its efficiency
    row can all still move.  Note that a scan's ``n_missing`` in the selection file is NOT
    this signal: on both CIFAR scans that number is the 100 parked JD + HIG runs.
    """
    payload = payload or load_selection()
    scans = HEADLINE_SCANS if scans is None else scans
    root = Path(style.resolve_results_root(results_root))
    gen = str(payload.get('generated_at', ''))
    gen_ts = pd.to_datetime(gen, errors='coerce', utc=True)
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    rows = []
    for scan in scans:
        for kind in kinds:
            d = root / dir_name(scan, kind)
            if not d.is_dir():
                continue
            recs = sorted(d.glob('*.jsonl'))
            mtimes = pd.to_datetime([p.stat().st_mtime for p in recs], unit='s', utc=True)
            n_after = (int((mtimes > gen_ts).sum())
                       if len(mtimes) and pd.notna(gen_ts) else 0)
            have = {p.name[:-len('.jsonl')] for p in recs}
            live = 0
            for c in (d / 'claims').glob('*.claim*'):
                rid = c.name.split('.claim')[0]
                if rid in have:
                    continue
                age = (now - pd.to_datetime(c.stat().st_mtime, unit='s', utc=True))
                if age.total_seconds() < CLAIM_TIMEOUT_S:
                    live += 1
            rows.append({
                'scan': scan, 'pass': kind or 'scan',
                'selection_input': kind == '', 'n_records': len(recs),
                'newest_record': (mtimes.max().tz_convert(None).isoformat(sep=' ',
                                                                         timespec='seconds')
                                  if len(mtimes) else None),
                'n_after_selection': n_after, 'n_claims_live': live,
                # a _confirm / _timing record is newer BY CONSTRUCTION (the pass was
                # generated from the selection), so only the tuning grid's count can say
                # the selection has moved
                'provisional': bool(live or (kind == '' and n_after))})
    out = pd.DataFrame(rows)
    out.attrs['selection_generated_at'] = gen
    out.attrs['provisional_scans'] = sorted(
        {r['scan'] for r in rows if r['provisional']})
    out.attrs['n_provisional'] = len(out.attrs['provisional_scans'])
    return out


# ---------------------------------------------------------------------------
# (b) paired differences against Sven
# ---------------------------------------------------------------------------
PAIR_COL = '_pair_key'


def _with_pair_key(df, scan, kind='confirm', results_root=None):
    """A copy of ``df`` carrying :data:`PAIR_COL`, the thing a pair is matched ON.

    Normally the model seed (same initialisation and, through ``derive_loader_seed``, the
    same data order).  Where the pass has data-seed replicates the pair is
    ``(model_seed, data_seed)``: two runs on the same model seed but different data seeds
    are different PROBLEMS, and differencing them would compare a method on one target
    against a method on another.

    The column lives on a copy, never on the memoised frame: it varies and is not an
    allow-listed hyperparameter, so :func:`style.hparam_columns` would (correctly) warn
    that it is being averaged over.
    """
    out = df.copy()
    seeds = data_seeds(scan, kind, results_root=results_root)
    if len(seeds) > 1 and 'data_seed' in out.columns:
        out[PAIR_COL] = (out['model_seed'].astype('Int64').astype(str) + '@'
                         + out['data_seed'].astype('Int64').astype(str))
    else:
        out[PAIR_COL] = out['model_seed'].astype('Int64').astype(str)
    return out


def _pseudo_rows(scan, df, sels, metric=SELECTION_METRIC, results_root=None):
    """``{method: {'method': label, 'run_ids': [...]}}`` -- the rows :mod:`paired` wants.

    :func:`paired.paired_runs` takes rows of a config table and reads only ``run_ids`` and
    the label column, so the selected configuration can be handed to it directly.  Going
    through :func:`analysis_helpers.config_table` instead would RE-GROUP the confirmation
    runs and, for toy / polynomial, split each method into three data-seed configurations.

    ``run_ids`` holds the USABLE runs only, which is what
    :func:`analysis_helpers.config_table` puts there and what :mod:`paired` documents:
    a diverged run left in the list would make its seed look paired-and-NaN instead of
    unpaired, and ``n_unpaired`` -- the whole point of reporting a lopsided comparison --
    would come back 0.
    """
    out = {}
    for method, sel in sels.items():
        rows, _ = selected_runs(df, sel)
        if len(rows):
            keep = ~rows['diverged'] & ~rows['failed']
            if metric in rows.columns:
                keep &= np.isfinite(pd.to_numeric(rows[metric], errors='coerce'))
            rows = rows[keep]
        out[method] = {'method': method_key(method),
                       'run_ids': list(rows['run_id']) if 'run_id' in rows else []}
    return out


def paired_vs_sven(scan, metric=SELECTION_METRIC, payload=None, level=0.95,
                   kind='confirm', results_root=None):
    """Per method, the paired difference ``Sven - method`` on the confirmation seeds.

    :func:`paired.paired_table` does the arithmetic (C-A6): the difference is taken WITHIN
    a pair -- same model seed, hence same initialisation and data order -- which removes
    the seed-to-seed variation that dominates the unpaired spread, and the reported
    interval is a Student-t interval on the MEAN difference, not the seed band.  A
    negative mean therefore means Sven has the lower loss on the same problem.

    ``n`` is the number of pairs actually formed and ``n_unpaired`` the seeds only one side
    finished (a diverged run has no value, so it cannot be paired and is not silently
    imputed).  For toy / polynomial a pair is ``(model seed, data seed)``: 15 pairs, not 5.
    """
    style.assert_selection_metric(metric, 'headline.paired_vs_sven')
    sels = selection_methods(scan, payload)
    if SVEN_METHOD not in sels:
        raise KeyError(f'{scan}: no Sven selection to compare against')
    df = _with_pair_key(load(scan, kind, results_root=results_root), scan, kind,
                        results_root=results_root)
    rows = _pseudo_rows(scan, df, sels, metric=metric, results_root=results_root)
    ref = rows[SVEN_METHOD]
    others = [rows[m] for m in method_order(sels) if m != SVEN_METHOD]
    tbl = paired.paired_table(df, others, ref, metric=metric, seed_col=PAIR_COL,
                              level=level)
    tbl = tbl.sort_values('mean').reset_index(drop=True)
    if len(tbl):
        tbl.insert(1, 'display', [display_name(m) for m in tbl['method']])
        # `paired.a_better` is "the reference won this pair" under paired.py's LOSS
        # convention (d < 0).  Here the reference IS Sven and the metric IS a loss, so the
        # two names agree -- but the column is spelled the way the outcome tables spell it
        # (`sven_better`), because a reader comparing the two must not have to know which
        # side "a" was.
        tbl['sven_better'] = tbl['a_better']
    tbl.attrs['scan'] = scan
    tbl.attrs['pass'] = kind
    tbl.attrs['pair_on'] = ('model_seed x data_seed'
                            if len(data_seeds(scan, kind, results_root=results_root)) > 1
                            else 'model_seed')
    tbl.attrs['sign'] = (f'mean = Sven - other in {metric} (a loss): NEGATIVE means Sven '
                         f'is better; sven_better counts the pairs Sven won')
    tbl.attrs['higher_is_better'] = False
    return tbl


def paired_outcome_vs_sven(scan, metric='final_test_loss', payload=None, level=0.95,
                           kind='confirm', results_root=None):
    """The same paired difference on an OUTCOME (test loss / accuracy).

    :func:`paired.paired_difference` deliberately refuses a test metric -- it is the
    module every SELECTION goes through, and the guard is what makes "selection uses
    validation only" mechanical.  Reporting a paired difference in a test outcome for the
    configuration validation already chose is a different act, so the arithmetic is
    repeated here (and only here), on a metric that may not select anything: the ddof and
    the t-interval are :mod:`paired`'s.

    The SIGN is not.  ``mean`` is always ``Sven - other`` in the metric's own units, but
    :mod:`paired`'s "a is better when the difference is negative" holds for a LOSS only,
    so for an accuracy (:func:`higher_is_better`) the count of pairs Sven won is
    ``(d > 0)`` and the rows are ordered the other way round.  The column is therefore
    named ``sven_better`` rather than ``a_better``, and ``.attrs['sign']`` states the
    convention in the table itself -- reading an accuracy row under the loss convention
    turns "Sven lost all five seeds" into "Sven won all five".
    """
    if not style.is_test_metric(metric):
        raise ValueError(f'{metric!r} is not a test outcome -- use paired_vs_sven, which '
                         f'goes through analysis/paired.py')
    sels = selection_methods(scan, payload)
    df = _with_pair_key(load(scan, kind, results_root=results_root), scan, kind,
                        results_root=results_root)
    rows = _pseudo_rows(scan, df, sels, metric=metric, results_root=results_root)
    if metric not in df.columns:
        return pd.DataFrame(columns=['method', 'reference', 'metric', 'n', 'mean'])

    def per_pair(row):
        sub = df[df['run_id'].isin(row['run_ids'])]
        return sub.groupby(PAIR_COL, dropna=False)[metric].mean()

    higher = higher_is_better(metric)
    ref = per_pair(rows[SVEN_METHOD])
    out = []
    for method in method_order(sels):
        if method == SVEN_METHOD:
            continue
        other = per_pair(rows[method])
        shared = sorted(set(ref.index) & set(other.index))
        d = np.asarray([ref[s] - other[s] for s in shared], dtype=float)
        d = d[np.isfinite(d)]
        n = d.size
        sd = float(d.std(ddof=1)) if n > 1 else np.nan
        sem = sd / np.sqrt(n) if n > 1 else np.nan
        half = float(stats.t.ppf(0.5 + level / 2, n - 1) * sem) if n > 1 else np.nan
        mean = float(d.mean()) if n else np.nan
        out.append({'method': method_key(method), 'reference': SVEN_LABEL,
                    'metric': metric, 'n': n, 'mean': mean, 'std': sd, 'sem': sem,
                    'ci_low': mean - half, 'ci_high': mean + half, 'half_width': half,
                    'level': level,
                    'n_unpaired': len(set(ref.index) ^ set(other.index)),
                    # direction-aware: for an accuracy the pairs Sven WON are d > 0
                    'sven_better': int((d > 0).sum() if higher else (d < 0).sum())})
    tbl = pd.DataFrame(out)
    if len(tbl):
        tbl['significant'] = (tbl['ci_low'] > 0) | (tbl['ci_high'] < 0)
        tbl.insert(1, 'display', [display_name(m) for m in tbl['method']])
        # best-for-Sven first in BOTH kinds of table, so an accuracy table read next to a
        # loss table does not silently reverse its row order
        tbl = tbl.sort_values('mean', ascending=not higher).reset_index(drop=True)
    tbl.attrs['interval'] = paired.interval_label(level)
    tbl.attrs['metric'] = metric
    tbl.attrs['scan'] = scan
    tbl.attrs['higher_is_better'] = higher
    tbl.attrs['sign'] = (
        f'mean = Sven - other in {metric}; '
        + ('POSITIVE means Sven is better (larger is better)' if higher
           else 'NEGATIVE means Sven is better (a loss)')
        + '; sven_better counts the pairs Sven won')
    return tbl


# ---------------------------------------------------------------------------
# (c) tuning-budget disclosure
# ---------------------------------------------------------------------------
BUDGET_N = (1, 2, 4, 8, 16, 32, 64, 128)


def budget_table(scan, results_root=None, sig=None):
    """:func:`budget.trajectory_table` on the scan's own grid: how many of a method's grid
    points are DISTINCT trajectories.

    The budget objection is that Sven is tuned over 72-128 points against four learning
    rates; the answer is that a Sven point whose rtol-rank stays below ``k`` is the
    identical trajectory, so the search actually performed is smaller than the grid.
    """
    df = load(scan, '', results_root=results_root)
    tbl = budget.trajectory_table(df, metric=SELECTION_METRIC, sig=sig)
    tbl.insert(0, 'label', [display_name(m) for m in tbl['method']])
    tbl.attrs['scan'] = scan
    return tbl


def best_of_n_table(scan, results_root=None, n_values=BUDGET_N, diverged='worst'):
    """``(curves, at_n)``: :func:`budget.best_of_n_curve` plus a methods x n pivot.

    Reading every method's column at the same ``n`` compares them at an EQUAL tuning
    budget -- the comparison the grid sizes otherwise prevent.  A configuration no seed
    finished is scored at the worst seed mean in the scan (``diverged='worst'``), because
    a random search would still have spent a trial on it.
    """
    df = load(scan, '', results_root=results_root)
    curves = budget.best_of_n_curve(df, metric=SELECTION_METRIC, diverged=diverged)
    keep = curves[curves['n'].isin(list(n_values))]
    at_n = keep.pivot_table(index='method', columns='n', values='expected_best')
    grid = curves.groupby('method')['grid_points'].max()
    at_n.insert(0, 'grid_points', grid)
    at_n.index = [display_name(m) for m in at_n.index]
    at_n = at_n.sort_values('grid_points', ascending=False)
    at_n.attrs['scan'] = scan
    at_n.attrs['diverged_value'] = curves.attrs.get('diverged_value')
    curves.attrs['scan'] = scan
    return curves, at_n


# ---------------------------------------------------------------------------
# (d) efficiency: time, memory, steps, examples, time-to-target
# ---------------------------------------------------------------------------
def _sum_curve(row, key):
    L = row.get('losses')
    if not isinstance(L, dict) or L.get(key) is None:
        return np.nan
    v = np.asarray(L[key], dtype=float)
    return float(np.nansum(v)) if v.size else np.nan


def _scalar(row, key):
    L = row.get('losses')
    if not isinstance(L, dict):
        return np.nan
    v = L.get(key)
    return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else np.nan


def _modal_gpu(rows):
    """``(the most common gpu_name, its share)`` over a set of runs.

    A time ratio between two passes is only a statement about the optimizer if both passes
    ran on the same device, and on the MLP scans they did not (see
    :func:`efficiency_table`), so the device is carried in the table rather than assumed.
    """
    if 'gpu_name' not in rows.columns or not len(rows):
        return None, np.nan
    v = rows['gpu_name'].dropna()
    if v.empty:
        return None, np.nan
    counts = v.value_counts()
    return str(counts.index[0]), float(counts.iloc[0] / len(v))


def run_efficiency(rows):
    """Per-run efficiency quantities for one configuration's runs, as a DataFrame.

    * ``wall_s`` -- ``losses.total_time``: the run's own wall clock.  Honest ONLY in the
      timing pass, where the run had the GPU to itself; a scan run shared its GPU with up
      to NPROC-1 other runs of the same job (``campaign/CONTRACTS.md``: 12 concurrent MLP
      runs on an A100) and, on the MLP scans, ran on a smaller GPU as well -- see
      :func:`efficiency_table`, which reports the GPU of each pass beside the ratio.
    * ``epoch_s`` / ``train_epoch_s`` -- ``avg_epoch_time`` / ``avg_train_time``.
    * ``sync_train_s`` -- the sum of ``train_times``, i.e. SYNCHRONISED per-batch training
      time (C-T1), which excludes evaluation -- a large and identical overhead for every
      method that does not belong on a convergence-vs-time axis.
    * ``ms_per_step`` -- ``sync_train_s`` / steps.  A launch-bound MLP step is a CPU
      measurement wearing a GPU costume (see :func:`calibration_report`).
    * ``peak_gpu_mem_mb`` -- ``losses.peak_gpu_mem_mb``.
    * ``steps`` / ``examples`` -- ``num_epochs x steps_per_epoch`` and that times the batch
      size: the two axes that are NOT time (F21).
    """
    out = []
    for _, r in rows.iterrows():
        spe = r.get('steps_per_epoch')
        ne = r.get('num_epochs')
        bs = r.get('batch_size')
        spe = float(spe) if pd.notna(spe) else np.nan
        ne = float(ne) if pd.notna(ne) else np.nan
        steps = spe * ne
        sync = _sum_curve(r, 'train_times')
        out.append({
            'run_id': r.get('run_id'), 'model_seed': r.get('model_seed'),
            'diverged': bool(r.get('diverged', False)),
            'failed': bool(r.get('failed', False)),
            'wall_s': _scalar(r, 'total_time'),
            'epoch_s': _scalar(r, 'avg_epoch_time'),
            'train_epoch_s': _scalar(r, 'avg_train_time'),
            'eval_epoch_s': _scalar(r, 'avg_eval_time'),
            'sync_train_s': sync,
            'ms_per_step': 1e3 * sync / steps if steps and np.isfinite(steps) else np.nan,
            'peak_gpu_mem_mb': _scalar(r, 'peak_gpu_mem_mb'),
            'n_epochs': ne, 'steps_per_epoch': spe, 'steps': steps,
            'examples': steps * float(bs) if pd.notna(bs) else np.nan,
            'final_val_loss': r.get('final_val_loss'),
        })
    return pd.DataFrame(out)


def efficiency_table(scan, payload=None, results_root=None):
    """Wall time, synchronised training time, step time and peak memory per method.

    Times come from ``{scan}_timing`` -- the standalone pass, one selected configuration
    at a time with ``svd_info: none`` and ``checkpoints: none`` -- because the scan's own
    wall times are not comparable: a scan run shared its GPU with other runs of the same
    job and, on the MLP scans, ran on a different GPU.  The scan's time is kept beside
    them (``scan_wall_s``) so the difference is visible rather than assumed, and
    ``fin_timing / att_timing`` says how much of the pass is actually there.

    ``scan_inflation = scan_wall_s / wall_s`` is therefore **not** a clean "what sharding
    cost".  The records say what it mixes, which is why ``gpu_scan`` and ``gpu_timing``
    (the modal ``gpu_name`` of each pass, with ``gpu_scan_frac`` for how modal it is) sit
    in the table:

    * toy / polynomial / MNIST: the scans ran overwhelmingly on an ``A100-SXM4-40GB MIG
      3g.20gb`` slice (1596/1770, 1507/1630, 1328/1360, 1490/1610) while every timing run
      ran on a full ``A100-SXM4-80GB``, so the 2.6-3.1x is co-tenancy PLUS a GPU-capability
      change and neither part can be read off alone;
    * both CIFAR scans and nanoGPT ran on the full A100-80GB throughout, so there
      ``scan_inflation`` is the co-tenancy cost by itself.

    (``n_shards`` is 1 on every record of every pass: concurrency came from NPROC
    processes per GPU under the claim queue, not from static grid sharding, so a
    ``n_shards`` of 1 says nothing about how many runs shared the device.)

    Before quoting an MLP step time, read :func:`calibration_report`: the GPU probe
    measured the same configuration at 1.13 ms on a quiet node and 4.58 ms on a loaded
    one, so a step time is only as good as the machine it was measured on.
    """
    sels = selection_methods(scan, payload)
    tune = load(scan, '', results_root=results_root)
    have_timing = has_pass(scan, 'timing', results_root)
    timing = load(scan, 'timing', results_root=results_root) if have_timing else None
    rows = []
    for method in method_order(sels):
        sel = sels[method]
        entry = {'method': method_key(method), 'display': display_name(method),
                 'config': config_label(sel)}
        t_rows, _ = selected_runs(tune, sel)
        scan_eff = run_efficiency(t_rows[~t_rows['diverged'] & ~t_rows['failed']])
        entry['scan_wall_s'] = float(scan_eff['wall_s'].mean()) if len(scan_eff) else np.nan
        entry['gpu_scan'], entry['gpu_scan_frac'] = _modal_gpu(t_rows)
        if have_timing:
            s_rows, _ = selected_runs(timing, sel)
            entry['gpu_timing'], entry['gpu_timing_frac'] = _modal_gpu(s_rows)
            counts = _counts(s_rows, scan, 'timing', results_root=results_root)
            eff = run_efficiency(s_rows[~s_rows['diverged'] & ~s_rows['failed']])
            for col in ('wall_s', 'epoch_s', 'train_epoch_s', 'sync_train_s',
                        'ms_per_step', 'peak_gpu_mem_mb', 'steps', 'examples'):
                v = pd.to_numeric(eff[col], errors='coerce') if len(eff) else pd.Series(dtype=float)
                v = v[np.isfinite(v)]
                entry[col] = float(v.mean()) if len(v) else np.nan
                if col in ('wall_s', 'epoch_s', 'ms_per_step', 'peak_gpu_mem_mb'):
                    entry[f'{col}_std'] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            entry.update({'fin_timing': counts['finished'],
                          'att_timing': counts['attempted'],
                          'div_timing': counts['n_diverged']})
        else:
            for col in ('wall_s', 'epoch_s', 'train_epoch_s', 'sync_train_s',
                        'ms_per_step', 'peak_gpu_mem_mb', 'steps', 'examples'):
                entry[col] = np.nan
            entry.update({'fin_timing': 0, 'att_timing': 0, 'div_timing': 0,
                          'gpu_timing': None, 'gpu_timing_frac': np.nan})
        rows.append(entry)
    out = pd.DataFrame(rows)
    if len(out) and 'wall_s' in out:
        sven = out.loc[out['method'] == SVEN_LABEL, 'wall_s']
        ref = float(sven.iloc[0]) if len(sven) and np.isfinite(sven.iloc[0]) else np.nan
        out['wall_vs_sven'] = out['wall_s'] / ref if np.isfinite(ref) else np.nan
        out['scan_inflation'] = out['scan_wall_s'] / out['wall_s']
    out = out.sort_values('wall_s', na_position='last').reset_index(drop=True)
    out.attrs['scan'] = scan
    out.attrs['times_from'] = dir_name(scan, 'timing')
    gs = set(out['gpu_scan'].dropna()) if 'gpu_scan' in out else set()
    gt = set(out['gpu_timing'].dropna()) if 'gpu_timing' in out else set()
    out.attrs['same_gpu'] = bool(gs and gt and gs == gt)
    out.attrs['inflation_means'] = (
        'co-tenancy on the same GPU' if out.attrs['same_gpu'] else
        'co-tenancy PLUS a different GPU between the two passes '
        f'(scan {sorted(gs)} vs timing {sorted(gt)})')
    return out


def timing_join_report(scan, payload=None, results_root=None, tol=1e-3):
    """Did the standalone timing runs reproduce the scan's trajectories?

    Same run_ids, same seeds, so the same initialisation and data order -- but GPU kernels
    are not bit-exact across GPU types, and the passes partly ran on A100-80GB while the
    scans ran on MIG slices.  Per method: the max / median relative deviation of the final
    validation loss between the scan run and its standalone twin.  Muon (bf16
    Newton-Schulz), L-BFGS, SOAP / Shampoo / HIG and every CIFAR run are expected to
    deviate; the plan's section 0 expects Sven and the plain first-order MLP runs not to
    -- an expectation this table CONTRADICTS on the two MNIST scans (see the notebook).

    This is **not** a restatement of ``bench/check_timing_join.py``, which compares the
    raw recorded finals.  ``max_rel_dev`` / ``median_rel_dev`` use the DERIVED finals, so
    a run :func:`analysis_helpers.add_derived` NaNs out (diverged, oom, error) has no
    value to compare and cannot enter them.  Three columns keep that from hiding anything:

    * ``n_diverged_only_in_timing`` -- the scan twin finished and the standalone run did
      not.  On ``mnist_scan_labelRegression`` SOAP has one: the scan run ends at 0.901 and
      its standalone twin at 184.8 (a 204x deviation), which the deviation columns cannot
      show because 184.8 is a divergence;
    * ``n_diverged_only_in_scan`` -- the reverse;
    * ``max_rel_dev_raw`` / ``worst_run_raw`` -- the deviation over the RAW final values of
      the validation curve, divergence and all, plus the run_id that produced it.  This is
      ``check_timing_join``'s quantity, so the two agree, and it is the column to quote
      when asking "did the standalone pass reproduce the scan".
    """
    cols = ['method', 'n_joined', 'max_rel_dev', 'n_diverged_only_in_timing',
            'max_rel_dev_raw']
    if not has_pass(scan, 'timing', results_root):
        return pd.DataFrame(columns=cols)
    sels = selection_methods(scan, payload)
    tune = load(scan, '', results_root=results_root).set_index('run_id')
    timing = load(scan, 'timing', results_root=results_root)

    def _raw_final(row):
        L = row.get('losses')
        curve = L.get('val') if isinstance(L, dict) else None
        return style.final_value(curve)

    rows = []
    for method in method_order(sels):
        s_rows, _ = selected_runs(timing, sels[method])
        devs, raw_devs, raw_ids = [], [], []
        only_timing = only_scan = 0
        for _, r in s_rows.iterrows():
            rid = r['run_id']
            if rid not in tune.index:
                continue
            scan_row = tune.loc[rid]
            a = scan_row[SELECTION_METRIC]
            b = r[SELECTION_METRIC]
            if np.isfinite(a) and np.isfinite(b):
                devs.append(abs(a - b) / max(abs(a), 1e-30))
            elif np.isfinite(a):
                only_timing += 1          # the scan run is usable, its twin is not
            elif np.isfinite(b):
                only_scan += 1
            ra, rb = _raw_final(scan_row), _raw_final(r)
            if np.isfinite(ra) and np.isfinite(rb):
                raw_devs.append(abs(ra - rb) / max(abs(ra), 1e-30))
                raw_ids.append(rid)
        worst = int(np.argmax(raw_devs)) if raw_devs else None
        rows.append({'method': method_key(method), 'display': display_name(method),
                     'n_timing': len(s_rows),
                     'n_joined': len(devs),
                     'max_rel_dev': float(np.max(devs)) if devs else np.nan,
                     'median_rel_dev': float(np.median(devs)) if devs else np.nan,
                     'bit_reproduced': bool(devs and np.max(devs) <= tol
                                            and not only_timing and not only_scan),
                     'n_diverged_only_in_timing': only_timing,
                     'n_diverged_only_in_scan': only_scan,
                     'max_rel_dev_raw': (float(np.max(raw_devs)) if raw_devs else np.nan),
                     'worst_run_raw': raw_ids[worst] if worst is not None else None})
    out = pd.DataFrame(rows)
    out.attrs['scan'] = scan
    out.attrs['tol'] = tol
    out.attrs['n_diverged_only'] = int(out['n_diverged_only_in_timing'].sum()
                                       + out['n_diverged_only_in_scan'].sum())
    return out


CALIB_RE = re.compile(r'\[calib\]\s*(\{.*\})\s*$')


def calibration_report(log_dir=None, scans=None):
    """The ``bench/calibrate_step.py`` lines the timing jobs wrote, start against end.

    A fixed Adam-MLP step time, measured before and after each timing pass on the same
    machine.  Host CPU load, not the GPU, sets a launch-bound MLP step time (1.13 ms on a
    quiet node vs 4.58 ms at 4/4 GPUs in ``campaign/stage0_reports/gpu.probe.md``), so a
    start/end drift is the evidence for or against contamination of everything
    :func:`efficiency_table` reports for that scan.  ``drift`` is
    ``end/start - 1`` on the MEDIAN, which a single preemption cannot move.

    Empty (with a note on ``.attrs``) when the log directory is not readable from here.
    """
    log_dir = Path(log_dir or TIMING_LOG_DIR)
    scans = HEADLINE_SCANS if scans is None else scans
    cols = ['scan', 'job', 'host', 'gpu', 'start_ms', 'end_ms', 'drift',
            'start_loadavg', 'end_loadavg', 'contaminated']
    if not log_dir.is_dir():
        out = pd.DataFrame(columns=cols)
        out.attrs['note'] = f'{log_dir} is not readable -- no calibration evidence'
        return out
    rows = []
    for path in sorted(log_dir.glob('timing-*.out')):
        entries = {}
        for line in path.read_text(errors='replace').splitlines():
            m = CALIB_RE.search(line)
            if not m:
                continue
            try:
                rec = json.loads(m.group(1))
            except ValueError:
                continue
            tag = str(rec.get('tag', ''))
            which = tag.split(':', 1)[0]
            entries[which] = rec
        if not entries:
            continue
        a, b = entries.get('start'), entries.get('end')
        one = a or b
        scan = str(one.get('tag', '')).split(':', 1)[-1]
        start_ms = a.get('median_ms') if a else np.nan
        end_ms = b.get('median_ms') if b else np.nan
        drift = (end_ms / start_ms - 1.0
                 if all(isinstance(v, (int, float)) for v in (start_ms, end_ms))
                 and start_ms else np.nan)
        rows.append({'scan': scan, 'job': one.get('slurm_job_id'),
                     'host': one.get('host'), 'gpu': one.get('gpu'),
                     'start_ms': start_ms, 'end_ms': end_ms, 'drift': drift,
                     'start_loadavg': (a or {}).get('loadavg'),
                     'end_loadavg': (b or {}).get('loadavg'),
                     'contaminated': bool(np.isfinite(drift) and abs(drift) > 0.10)})
    out = pd.DataFrame(rows, columns=cols)
    if len(out):
        out = out.sort_values('scan').reset_index(drop=True)
        out.attrs['max_abs_drift'] = float(np.nanmax(np.abs(out['drift'])))
        out.attrs['n_contaminated'] = int(out['contaminated'].sum())
    out.attrs['log_dir'] = str(log_dir)
    out.attrs['threshold'] = 0.10
    out.attrs['missing_scans'] = [s for s in scans if s not in set(out.get('scan', []))]
    return out


# --- time to target --------------------------------------------------------
def target_reference(scan, payload=None, results_root=None):
    """The reference value the scan's targets are multiples of.

    The MEDIAN over methods of the confirmation-seed mean final VALIDATION loss.  The
    median and not the best: a target set by the winner is a target only the winner
    reaches, and a target set by the loser is reached at epoch 1 by everybody.  Validation
    only, so declaring a target cannot leak the test split.
    """
    tbl = confirmation_table(scan, payload=payload, results_root=results_root)
    v = pd.to_numeric(tbl['val_conf'], errors='coerce')
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else np.nan


def targets_for(scan, payload=None, results_root=None):
    """``{label: value}`` -- the pre-declared targets of one scan (:data:`TARGET_MULTIPLIERS`)."""
    ref = target_reference(scan, payload=payload, results_root=results_root)
    return {TARGET_NAMES[m]: ref * m for m in TARGET_MULTIPLIERS}


def epochs_to_target(row, target, which='val'):
    """The first epoch index whose ``which`` loss is at or below ``target``, else None.

    Index 0 of the validation curve is the UNTRAINED model, so the index is the number of
    epochs trained -- and 0 is a legitimate answer for a target the initialisation already
    meets.  A diverged run never reaches anything (its curve is not scored).
    """
    if row.get('diverged', False) or row.get('failed', False):
        return None
    L = row.get('losses')
    curve = L.get(which) if isinstance(L, dict) else None
    if curve is None:
        return None
    for i, v in enumerate(curve):
        if v is not None and np.isfinite(v) and float(v) <= target:
            return i
    return None


def time_to_target_table(scan, payload=None, results_root=None, kind='confirm'):
    """Epochs / steps / examples / seconds to reach each pre-declared target.

    The epochs come from the CONFIRMATION runs (the numbers we report); the seconds per
    epoch come from the standalone ``{scan}_timing`` pass of the SAME configuration, since
    that is the only pass whose wall clock is honest -- the confirmation runs were sharded.
    ``sync_train_s`` instead uses each confirmation run's own synchronised training time
    (``train_times``), which needs no cross-pass assumption but excludes evaluation.

    Runs that produced no answer are COUNTED, not dropped, and for two DIFFERENT reasons
    that must not be added up into one number:

    * ``n_never_reached`` -- the run trained to the end and its curve never got to the
      target.  That is a statement about the method;
    * ``n_diverged`` -- the run diverged or failed, so it has no trajectory to ask (a
      diverged curve is not scored anywhere in this module).  That is a statement about
      the run.  Polynomial L-BFGS reads ``n_runs 15, n_reached 0`` at every target, but 13
      of those 15 diverged and only 2 ever ran to completion; ``elig_conf`` is carried
      into the row for the same reason (its confirmation mean is a mean over 2 survivors).

    ``n_never = n_never_reached + n_diverged`` is kept so ``n_runs = n_reached + n_never``
    still holds, and ``all_reached`` says whether a mean over the survivors is conditional.

    **Per-instance caveat (toy / polynomial).**  A target is ONE pooled value
    (:func:`target_reference`) applied to runs on three different random problems, so an
    ``epochs`` mean can be conditional on which instance a run was drawn on -- polynomial's
    hard target 0.1086 is not reachable on instance 2001 at all.  Compare the methods at a
    target, not the targets with each other.
    """
    sels = selection_methods(scan, payload)
    targets = targets_for(scan, payload=payload, results_root=results_root)
    df = load(scan, kind, results_root=results_root)
    eff = efficiency_table(scan, payload=payload, results_root=results_root)
    sec_per_epoch = dict(zip(eff['method'], eff['epoch_s']))
    conf = confirmation_table(scan, payload=payload, results_root=results_root)
    eligible = dict(zip(conf['method'], conf['elig_conf'])) if len(conf) else {}
    rows = []
    for label, target in targets.items():
        for method in method_order(sels):
            c_rows, _ = selected_runs(df, sels[method])
            recs = []
            n_never_reached = n_diverged = 0
            for _, r in c_rows.iterrows():
                e = epochs_to_target(r, target)
                if e is None:
                    # the two reasons are different facts: a run that trained to the end
                    # and never got there vs a run with no trajectory to ask
                    if r.get('diverged', False) or r.get('failed', False):
                        n_diverged += 1
                    else:
                        n_never_reached += 1
                    continue
                spe = float(r['steps_per_epoch']) if pd.notna(r.get('steps_per_epoch')) else np.nan
                bs = float(r['batch_size']) if pd.notna(r.get('batch_size')) else np.nan
                L = r.get('losses') or {}
                tt = L.get('train_times')
                sync = (float(np.nansum(np.asarray(tt[:e], dtype=float)))
                        if tt is not None and e > 0 else (0.0 if tt is not None else np.nan))
                recs.append({'epochs': e, 'steps': e * spe, 'examples': e * spe * bs,
                             'sync_train_s': sync})
            n = len(recs)
            per_epoch = sec_per_epoch.get(method_key(method), np.nan)
            entry = {'target': label, 'target_value': target,
                     'method': method_key(method), 'display': display_name(method),
                     'n_runs': len(c_rows),
                     'n_reached': n,
                     'n_never_reached': n_never_reached, 'n_diverged': n_diverged,
                     'n_never': n_never_reached + n_diverged,
                     'elig_conf': bool(eligible.get(method_key(method), True)),
                     'all_reached': bool(n and n == len(c_rows))}
            for col in ('epochs', 'steps', 'examples', 'sync_train_s'):
                v = np.asarray([d[col] for d in recs], dtype=float) if n else np.array([])
                v = v[np.isfinite(v)]
                entry[col] = float(v.mean()) if v.size else np.nan
                entry[f'{col}_std'] = float(v.std(ddof=1)) if v.size > 1 else (
                    0.0 if v.size == 1 else np.nan)
            entry['standalone_s'] = (entry['epochs'] * per_epoch
                                     if np.isfinite(entry['epochs']) else np.nan)
            rows.append(entry)
    out = pd.DataFrame(rows)
    out.attrs['scan'] = scan
    out.attrs['targets'] = targets
    out.attrs['pass'] = kind
    out.attrs['seconds_per_epoch_from'] = dir_name(scan, 'timing')
    return out


# ---------------------------------------------------------------------------
# (e) the cross-scan ranking summary
# ---------------------------------------------------------------------------
def ranking_summary(scans=None, payload=None, results_root=None):
    """One table, scans x methods: the rank of every method on the confirmation seeds.

    Ranked on ``val_conf`` (the validation loss, the quantity everything was selected on)
    and on the test OUTCOMES -- test loss ascending, test accuracy descending -- so the
    two questions "did it optimise" and "did it generalise" are answered side by side and
    can disagree, which on CIFAR they do.  ``n_methods`` is the field a rank is out of, so
    "9th" is readable; a method with no finite confirmation value gets no rank.
    """
    scans = HEADLINE_SCANS if scans is None else scans
    rows = []
    for scan in scans:
        tbl = confirmation_table(scan, payload=payload, results_root=results_root)
        rank_val = pd.to_numeric(tbl['val_conf'], errors='coerce').rank(method='min')
        rank_test = pd.to_numeric(tbl.get('test_conf'), errors='coerce').rank(method='min')
        rank_acc = pd.to_numeric(tbl.get('acc_conf'), errors='coerce').rank(
            method='min', ascending=False)
        n_methods = int(np.isfinite(pd.to_numeric(tbl['val_conf'],
                                                  errors='coerce')).sum())
        for i, r in tbl.iterrows():
            rows.append({
                'scan': scan, 'title': scan_title(scan), 'method': r['method'],
                'display': r.get('display', r['method']), 'n_methods': n_methods,
                'val_conf': r['val_conf'], 'rank_val': rank_val.get(i),
                'test_conf': r.get('test_conf'), 'rank_test': rank_test.get(i),
                'acc_conf': r.get('acc_conf'), 'rank_acc': rank_acc.get(i),
                'finished': r['fin_conf'], 'attempted': r['att_conf'],
                'elig_conf': r.get('elig_conf', True),
            })
    return pd.DataFrame(rows)


def rank_matrix(summary=None, value='rank_val', scans=None, **kw):
    """The ranking summary as a scans x methods matrix (``rank_val`` / ``rank_test`` /
    ``rank_acc`` / ``val_conf`` / ...), Sven's column first."""
    scans = HEADLINE_SCANS if scans is None else scans
    summary = ranking_summary(scans=scans, **kw) if summary is None else summary
    col = 'display' if 'display' in summary.columns else 'method'
    mat = summary.pivot_table(index='title', columns=col, values=value, sort=False)
    order = [SVEN_LABEL] + sorted(c for c in mat.columns if c != SVEN_LABEL)
    mat = mat[[c for c in order if c in mat.columns]]
    titles = [scan_title(s) for s in scans if scan_title(s) in set(mat.index)]
    return mat.reindex(titles)


# ---------------------------------------------------------------------------
# Formatting / export
# ---------------------------------------------------------------------------
def fmt_pm(mean, std, spec='.3g'):
    """``mean +/- std`` (the seed band, :data:`paired.SEED_SPREAD_LABEL`)."""
    if mean is None or not np.isfinite(mean):
        return '--'
    if std is None or not np.isfinite(std):
        return f'{mean:{spec}}'
    return f'{mean:{spec}} ± {std:{spec}}'


def fmt_counts(finished, attempted):
    return f'{int(finished)}/{int(attempted)}'


def _cell(value, floatfmt):
    if value is None:
        return ''
    if isinstance(value, float) and not np.isfinite(value):
        return ''
    if isinstance(value, (float, np.floating)):
        return format(float(value), floatfmt)
    if isinstance(value, (bool, np.bool_)):
        return 'True' if value else 'False'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value).replace('|', r'\|').replace('\n', ' ')


def to_markdown(df, index=False, floatfmt='.4g'):
    """A markdown table, with or without the optional ``tabulate`` package.

    ``DataFrame.to_markdown`` needs ``tabulate``, which this environment does not have and
    cannot install (the venv has no ``pip``), so the fallback renders the same
    GitHub-flavoured shape: header, alignment row, one row per record, floats at
    ``floatfmt``, non-finite values blank.  Every export and every printed table goes
    through here, so a table reads the same whether or not ``tabulate`` is present.
    """
    try:
        return df.to_markdown(index=index, floatfmt=floatfmt)
    except ImportError:
        pass
    frame = df.reset_index() if index else df
    cols = [str(c) for c in frame.columns]
    lines = ['| ' + ' | '.join(cols) + ' |',
             '|' + '|'.join('---' for _ in cols) + '|']
    for _, row in frame.iterrows():
        lines.append('| ' + ' | '.join(_cell(v, floatfmt) for v in row.tolist()) + ' |')
    return '\n'.join(lines)


def confirmation_view(tbl, spec='.4g'):
    """The confirmation table as it goes into the paper: one column per quantity,
    ``mean +/- 1 std over seeds``, and ``finished/attempted`` on every row.

    On a scan WITH data-seed replicates (toy / polynomial) the single ``gap`` column is
    replaced by two named ones -- the gap on the tuning instance (the selection optimism)
    and the pooled gap, which also carries the two problem instances the tuning never saw
    -- because the two differ in magnitude and, for toy, in sign.
    """
    has_acc = 'acc_conf' in tbl and pd.to_numeric(tbl['acc_conf'],
                                                  errors='coerce').notna().any()
    has_ds = 'val_conf_dsspread' in tbl and pd.to_numeric(
        tbl['val_conf_dsspread'], errors='coerce').notna().any()
    has_treval = 'treval_conf' in tbl and pd.to_numeric(tbl['treval_conf'],
                                                        errors='coerce').notna().any()
    out = []
    for _, r in tbl.iterrows():
        row = {
            'method': r.get('display', r['method']), 'config': r['config'],
            'val (confirm)': fmt_pm(r['val_conf'], r['val_conf_std'], spec),
            'test (confirm)': fmt_pm(r.get('test_conf'), r.get('test_conf_std'), spec),
        }
        if has_acc:
            row['test acc'] = fmt_pm(r.get('acc_conf'), r.get('acc_conf_std'), '.3g')
        if has_treval:
            row['train (eval mode)'] = fmt_pm(r.get('treval_conf'),
                                              r.get('treval_conf_std'), spec)
        row['finished/attempted'] = fmt_counts(r['fin_conf'], r['att_conf'])
        if not r.get('elig_conf', True):
            # more than half the confirmation runs failed: the mean beside it is a mean
            # over the survivors, which is not this method's loss
            row['finished/attempted'] += ' (not eligible)'
        row['val (tuning)'] = fmt_pm(r['val_tune'], r['val_tune_std'], spec)
        row['tuning fin/att'] = fmt_counts(r['fin_tune'], r['att_tune'])

        def _pct(value):
            return '--' if not np.isfinite(value) else f'{value:+.1%}'

        if has_ds:
            # On toy / polynomial the pooled gap is mostly problem-instance variation, so
            # neither column may be left unlabelled: the SELECTION OPTIMISM is the gap on
            # the instance the tuning ran on, and the pooled one is named for what it also
            # contains.
            row['val, mean +/- between-instance'] = fmt_pm(r.get('val_conf_dsmean'),
                                                           r.get('val_conf_dsspread'), spec)
            row['val, within-instance seed std'] = fmt_pm(r.get('val_conf_seedspread'),
                                                          None, spec)
            row['val, tuning instance'] = fmt_pm(r.get('val_conf_ds0'), None, spec)
            row['gap, tuning instance'] = (
                '--' if not np.isfinite(r.get('gap_val_same_instance', np.nan))
                else f"{r['gap_val_same_instance']:+{spec}}")
            row['gap rel, tuning instance (optimism)'] = _pct(
                r.get('gap_val_same_instance_rel', np.nan))
            row['gap rel, pooled (+2 unseen instances)'] = _pct(r['gap_val_rel'])
        else:
            row['gap (confirm-tune)'] = ('--' if not np.isfinite(r['gap_val'])
                                         else f"{r['gap_val']:+{spec}}")
            row['gap rel'] = _pct(r['gap_val_rel'])
        out.append(row)
    view = pd.DataFrame(out)
    view.attrs.update(tbl.attrs)
    return view


def write_table(df, name, subdir=None, index=False, float_format='%.4g',
                tables_dir=None):
    """Export one table to ``analysis/tables/`` as markdown and LaTeX.

    Returns the paths written.  The markdown is what the report quotes; the LaTeX is what
    the paper includes (``booktabs``, so it needs ``\\usepackage{booktabs}``).
    """
    root = Path(tables_dir or TABLES_DIR)
    if subdir:
        root = root / subdir
    root.mkdir(parents=True, exist_ok=True)
    md, tex = root / f'{name}.md', root / f'{name}.tex'
    md.write_text(to_markdown(df, index=index) + '\n')
    tex.write_text(df.to_latex(index=index, float_format=float_format,
                               escape=True, longtable=False))
    return md, tex


def print_table(df, title=None, index=False):
    """A table on stdout the way the report quotes it."""
    if title:
        print(f'\n### {title}')
    print(to_markdown(df, index=index))
    return df
