"""CPU tests for the offline ("Phase 0") analysis: C-T2, C-A1, C-A4, C-A6 + notebook execution.

Everything here is either synthetic (built in a tmp dir) or reads a handful of the
small profile JSONs under ``profile_results_v2/``; nothing writes to a results root
and nothing needs a GPU or torch.

The one REAL acceptance number is C-T2's: SOAP on ``profile_mnist`` must read
6.14 ms amortised where the old spike-filtered mean read 5.56 ms, and
``gram_hooks`` -- which has no 10-step refresh -- must not move (tolerance 0.05 ms).
via ``SV3_CHECK_REAL_SCANS=1`` because it reads ~33 MB of npz off Lustre; run it
through ``campaign/run_cpu_tests.sh``.

``experiments/optimizer_profile.py`` imports torch at module scope, so its copy of
``cycle_mean`` / ``summarize`` cannot be imported here.  The tests EXTRACT those two
functions from the source with ``ast`` and exercise them directly, which is what
keeps the duplicate in ``analysis/lib/profile_helpers.py`` honest.
"""

import ast
import importlib.util
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'analysis' / 'lib'))   # the notebooks' own sys.path.insert(0, '.')
sys.path.insert(0, str(REPO / 'analysis'))

import analysis_helpers as ah    # noqa: E402
import budget                    # noqa: E402
import paired                    # noqa: E402
import profile_helpers as ph     # noqa: E402
import style                     # noqa: E402

PROFILE_ROOT = REPO / 'profile_results_v2'
PROFILER_SRC = REPO / 'experiments' / 'optimizer_profile.py'


# ===========================================================================
# C-T2  --  amortised step time
# ===========================================================================
def _from_profiler(*names):
    """``names`` extracted from ``experiments/optimizer_profile.py`` and executed in
    a namespace holding only numpy + that module's simple constants.  Lets the test
    call the profiler's OWN statistics without importing torch."""
    tree = ast.parse(PROFILER_SRC.read_text())
    ns = {'np': np}
    for node in tree.body:       # module-level `NAME = <literal>` constants (CYCLE)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant):
            ns[node.targets[0].id] = node.value.value
    want = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {f.name for f in want} == set(names), f'{PROFILER_SRC.name} is missing {names}'
    exec(compile(ast.Module(body=want, type_ignores=[]), str(PROFILER_SRC), 'exec'), ns)
    return [ns[n] for n in names]


def test_cycle_mean_is_the_last_80pc_in_whole_cycles():
    # 50 steps: 1 ms, with every 10th step costing 11 ms (a refresh), plus a
    # start-up ramp in the first 10 steps that must not be counted.
    a = np.ones(50)
    a[::10] = 11.0
    a[:10] = 100.0
    a[0] = 100.0
    assert ph.cycle_mean(a) == pytest.approx((36 * 1.0 + 4 * 11.0) / 40)
    # The whole-cycle truncation makes the answer independent of the window phase:
    # any 40 consecutive steps of a period-10 series contain exactly 4 refreshes.
    b = np.ones(55)
    b[3::10] = 11.0
    assert ph.cycle_mean(b) == pytest.approx((36 * 1.0 + 4 * 11.0) / 40)
    # Shorter than one cycle: the tail mean (the last 80%), not an error.
    assert ph.cycle_mean([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(3.5)
    assert np.isnan(ph.cycle_mean([]))
    assert np.isnan(ph.cycle_mean(None))


def test_profiler_and_analysis_cycle_mean_agree():
    """The duplicate in the runner and the one in the analysis are the same function."""
    profiler_cycle_mean, summarize = _from_profiler('cycle_mean', 'summarize')
    rng = np.random.default_rng(0)
    for n in (1, 4, 9, 10, 11, 37, 50, 123):
        v = rng.lognormal(size=n)
        v[::10] *= 20
        assert profiler_cycle_mean(v) == pytest.approx(ph.cycle_mean(v), rel=0, abs=0)
    # ... and `summarize` exports it beside the steady mean it replaces (not instead).
    s = summarize(list(v))
    assert s['cycle_mean'] == pytest.approx(ph.cycle_mean(v))
    assert 'steady_mean' in s and s['cycle'] == 10


def test_profiler_records_nonfinite_status_after_the_measured_steps():
    """The parameter check must run AFTER the timings are recorded, so a blown-up
    configuration still reports its step time -- with ``status: nonfinite``."""
    src = PROFILER_SRC.read_text()
    assert 'result["status"] = "nonfinite"' in src
    assert src.index('result["memory"]') < src.index('result["status"] = "nonfinite"')
    assert src.index('result["status"] = "nonfinite"') < src.index('except _Infeasible')
    assert 'nonfinite' in ph.STATUS_NOTE


@pytest.mark.skipif(not (PROFILE_ROOT / 'profile_mnist').is_dir(),
                    reason='profile_results_v2/profile_mnist not present')
@pytest.mark.parametrize('method, cycle_ms, steady_ms, tol', [
    ('SOAP', 6.14, 5.56, 0.005),        # precondition_frequency=10: +10.5%
    ('gram_hooks', 10.40, 10.40, 0.05),  # no periodic refresh: unchanged
])
def test_real_profile_reads_amortised_step_time(method, cycle_ms, steady_ms, tol):
    """C-T2's acceptance test, on the stored profiles (no rerun)."""
    f = PROFILE_ROOT / 'profile_mnist' / f'methods__{method}__B64__kf1.json'
    row = ph._row(json.loads(f.read_text()))
    assert row['step_ms'] == pytest.approx(cycle_ms, abs=tol)
    assert row['step_ms_steady'] == pytest.approx(steady_ms, abs=tol)
    # Recomputed from the stored raw list, not from a `time` summary that predates C-T2.
    raw = json.loads(f.read_text())['raw']['step_ms']
    assert row['step_ms'] == pytest.approx(ph.cycle_mean(raw))


def _profile_record(run_id, step_ms, status='ok', stored_cycle=None, method='Adam',
                    study='methods', n_params=10, peak_bytes=1_000_000, params=None,
                    raw_extra=None):
    t = {'step_ms': {}} if not len(step_ms) else {
        'step_ms': {'steady_mean': float(np.mean(step_ms)), 'median': float(np.median(step_ms)),
                    'mean': float(np.mean(step_ms)), 'p10': 1.0, 'p90': 2.0, 'n': len(step_ms),
                    'trimmed_mean': float(np.mean(step_ms))}}
    if stored_cycle is not None:
        t['step_ms']['cycle_mean'] = stored_cycle
    return {'run_id': run_id, 'arch': 'toy_1d', 'config_name': 'c', 'study': study,
            'method': method, 'status': status, 'n_params': n_params,
            'params': {'batch_size': 8, **(params or {})},
            'time': t, 'raw': {'step_ms': list(map(float, step_ms)), **(raw_extra or {})},
            'memory': {'peak_alloc_bytes_max': peak_bytes}}


def test_row_prefers_a_stored_cycle_mean_and_falls_back_to_the_raw_list():
    steps = list(np.r_[np.full(10, 100.0), np.full(40, 1.0)])
    assert ph._row(_profile_record('a', steps))['step_ms'] == pytest.approx(1.0)
    assert ph._row(_profile_record('a', steps, stored_cycle=7.0))['step_ms'] == pytest.approx(7.0)
    r = _profile_record('a', steps)
    r.pop('raw')                     # neither stored nor recomputable -> the old column
    assert ph._row(r)['step_ms'] == pytest.approx(r['time']['step_ms']['steady_mean'])


def test_profile_cache_key_carries_a_version(tmp_path, monkeypatch):
    """Without a version in the key, every cached frame would keep serving the
    pre-C-T2 ``step_ms``: the files did not change, only what is derived from them."""
    # the cache no longer lives inside the results root (profile_helpers._cache_path), so
    # without this the tmp roots of this test would drop pickles in the SHARED
    # experiment_results/_cache/.  Same redirect as tests/test_profile_helpers.py.
    monkeypatch.setenv('SV3_PROFILE_CACHE_DIR', str(tmp_path / '_cache'))
    d = tmp_path / 'cfg'
    d.mkdir()
    (d / 'r.json').write_text(json.dumps(_profile_record('r', [1.0] * 50)))
    first = ph.load_profiles(tmp_path, use_cache=True)
    assert 'sentinel' not in first.columns

    real_row = ph._row
    monkeypatch.setattr(ph, '_row', lambda r: {**real_row(r), 'sentinel': 1})
    assert 'sentinel' not in ph.load_profiles(tmp_path, use_cache=True).columns   # cache hit
    monkeypatch.setattr(ph, '_CACHE_VERSION', ph._CACHE_VERSION + 1)
    assert 'sentinel' in ph.load_profiles(tmp_path, use_cache=True).columns       # invalidated


def test_nonfinite_status_is_shown_not_crashed_on():
    df = ph.add_relative(pd.DataFrame([
        ph._row(_profile_record('ok', [2.0] * 50)),
        ph._row(_profile_record('bad', [2.0] * 50, status='nonfinite', method='SOAP')),
    ]))
    tbl = ph.method_table(df, 'toy_1d')
    assert set(tbl['Note']) == {'', 'non-finite'}
    assert list(ph.status_report(df)['status']) == ['nonfinite']
    fig, ax = plt.subplots()
    ph.plot_method_bars(df, 'toy_1d', ax)      # used to KeyError on an unknown status
    ph.plot_heatmap(df, ax)
    plt.close(fig)


def test_a_nonfinite_row_is_used_as_a_timing_and_only_flagged():
    """``nonfinite`` means the parameters blew up AFTER the steps were timed, so the
    timings are real.  Dropping such a row from the ``status == 'ok'`` filters would
    blank out ``rel_time`` for every method whenever the REFERENCE diverged."""
    df = ph.add_relative(pd.DataFrame([
        ph._row(_profile_record('adam', [2.0] * 50, method='Adam', status='nonfinite')),
        ph._row(_profile_record('sgd', [2.0] * 50, method='SGD', status='nonfinite')),
        ph._row(_profile_record('sven', [8.0] * 50, method='gram_hooks', peak_bytes=4_000_000)),
    ]))
    r = df[df.method == 'gram_hooks'].iloc[0]
    assert r['rel_time'] == pytest.approx(4.0)     # 8 ms / the non-finite Adam's 2 ms
    assert r['rel_mem'] == pytest.approx(4.0)      # 4 MB / the non-finite SGD's 1 MB
    tbl = ph.method_table(df, 'toy_1d').set_index('Method')
    assert tbl.loc['Adam', 'x Adam'] == pytest.approx(1.0) and tbl.loc['Adam', 'Note'] == 'non-finite'
    # The heatmap prints the measured number with a `*` instead of hiding it.
    fig, ax = plt.subplots()
    ph.plot_heatmap(df, ax, value='rel_time')
    assert '1.0*' in [t.get_text() for t in ax.texts]
    plt.close(fig)


def _sweep_rows(status_by_b, **kw):
    return [ph._row(_profile_record(f'b{b}', [] if s in ('oom', 'error') else [2.0 * b] * 50,
                                    status=s, method='gram_hooks', study='batch_size',
                                    params={'batch_size': b}, **kw))
            for b, s in status_by_b.items()]


def test_a_nonfinite_point_stays_on_its_sweep_line_and_phase_bars():
    raw = {'capture_ms': [1.0] * 50, 'solve_ms': [0.5] * 50}
    df = pd.DataFrame(_sweep_rows({8: 'ok', 16: 'nonfinite', 32: 'oom'}, raw_extra=raw))
    fig, ax = plt.subplots()
    ph.plot_sweep(df, 'toy_1d', 'batch_size', 'B', ax)
    lines = {l.get_label(): l for l in ax.get_lines()}
    assert len(lines[ph.label('gram_hooks')].get_xdata()) == 2        # 8 and 16, both timed
    assert len(lines['non-finite parameters'].get_xdata()) == 1       # ... 16 ringed, not a wall
    assert [k for k in lines if 'oom' in k]                           # only the OOM is a wall
    plt.close(fig)
    # Phase bars: the capture/solve split of a diverged variant is still drawn, with a `*`.
    d = pd.DataFrame(_sweep_rows({16: 'nonfinite'}, raw_extra=raw))
    fig, ax = plt.subplots()
    ph.plot_phase_bars(d, ax, study='batch_size')
    assert [t.get_text() for t in ax.get_xticklabels()] == ['(Gram, hooks)*']
    plt.close(fig)


def test_scaling_table_and_steadiness_count_a_nonfinite_row_as_measured():
    df = pd.DataFrame([
        ph._row(_profile_record('w1', [1.0] * 50, method='gram_hooks', study='width', n_params=10)),
        ph._row(_profile_record('w2', [4.0] * 50, method='gram_hooks', study='width', n_params=100,
                                status='nonfinite')),
        ph._row(_profile_record('w3', [], method='gram_hooks', study='width', n_params=1000,
                                status='oom')),
    ])
    t = ph.scaling_table(df, 'toy_1d').iloc[0]
    assert t['largest timed P'] == '100' and t['growth'] == pytest.approx(4.0)
    assert t['first failure'] == 'oom @ P=1,000' and t['non-finite'] == 1
    fig, ax = plt.subplots()
    ph.plot_steadiness(df, ax)
    assert 'of 2 configurations' in ax.get_title()      # the OOM row has no timings
    plt.close(fig)


# ===========================================================================
# Synthetic scan frames for C-A4 / C-A6
# ===========================================================================
def _losses(final, n_epochs=3):
    """A record's curves: ``val`` starts at the untrained value and ends at ``final``."""
    return {'train': [10 * final, 2 * final, final], 'val': [20 * final, 3 * final, final],
            'epoch_times': [1.0] * n_epochs, 'total_time': 3.0}


def _sven(k, lr, rtol, seed, final):
    return {'run_id': f'svd_bs8_k{k}_lr{lr:g}_rtol{rtol:g}_mseed{seed}_lseed1000',
            'optimizer': 'SVD', 'loss': 'mse', 'batch_size': 8, 'k': k, 'lr': lr, 'rtol': rtol,
            'model_seed': seed, 'loader_seed': 1000, 'losses': _losses(final)}


def _base(opt, lr, seed, final):
    return {'run_id': f'std_bs8_lr{lr:g}_optim{opt}_mseed{seed}_lseed1000', 'optimizer': opt,
            'loss': 'mse', 'batch_size': 8, 'lr': lr, 'weight_decay': 0.0,
            'model_seed': seed, 'loader_seed': 1000, 'losses': _losses(final)}


SEEDS = (1000, 1001, 1002)
KS, LRS = (4, 8, 16), (0.01, 0.1)


@pytest.fixture
def scan_df():
    """Sven 3 k x 2 lr, where k = 8 and k = 16 truncate nothing extra and are the
    SAME trajectory (identical stored loss), plus one baseline at 2 learning rates."""
    rows = []
    for si, seed in enumerate(SEEDS):
        for lr in LRS:
            for k in KS:
                rows.append(_sven(k, lr, 1e-4, seed, lr * (1.0 if k >= 8 else 1.5) + 0.001 * si))
            rows.append(_base('Adam', lr, seed, 2 * lr + 0.001 * si))
    return ah.add_derived(pd.DataFrame(rows))


# ===========================================================================
# C-A4  --  tuning budget
# ===========================================================================
def test_trajectory_table_counts_distinct_trajectories(scan_df):
    t = budget.trajectory_table(scan_df).set_index('method')
    assert t.loc['Sven', 'grid_points'] == 6            # 3 k x 2 lr
    assert t.loc['Sven', 'grid_per_group'] == pytest.approx(3.0)       # per (lr, seed)
    assert t.loc['Sven', 'distinct_per_group'] == pytest.approx(2.0)   # k=8 == k=16
    assert t.loc['Sven', 'distinct'] == pytest.approx(4.0)             # x 2 learning rates
    assert t.loc['Sven', 'distinct_frac'] == pytest.approx(4 / 6)
    assert t.loc['Adam', 'grid_points'] == 2
    assert t.loc['Adam', 'distinct'] == pytest.approx(2.0)             # 1 config per (lr, seed)
    assert t.attrs['within'] == ('lr',)


def test_trajectory_table_counts_a_divergence_as_one_group(scan_df):
    df = scan_df.copy()
    df.loc[df['run_id'].str.contains('k4_lr0.1'), 'final_val_loss'] = np.nan
    t = budget.trajectory_table(df).set_index('method')
    # lr=0.1 still has 2 distinct trajectories (one of them "it diverged"), lr=0.01 has 2.
    assert t.loc['Sven', 'distinct_per_group'] == pytest.approx(2.0)


def test_best_of_n_is_the_exact_order_statistic():
    v = [1.0, 2.0, 3.0, 4.0]
    assert budget.best_of_n(v, 1) == pytest.approx(2.5)
    assert budget.best_of_n(v, 2) == pytest.approx(10 / 6)
    assert budget.best_of_n(v, 4) == pytest.approx(1.0)
    assert budget.best_of_n(v, 9) == pytest.approx(1.0)          # capped at the grid size
    for n in (1, 2, 3, 4):                                       # vs brute force
        exact = np.mean([min(c) for c in itertools.combinations(v, n)])
        assert budget.best_of_n(v, n) == pytest.approx(exact)
    assert budget.best_of_n(v, 1, minimize=False) == pytest.approx(2.5)
    assert budget.best_of_n(v, 3, minimize=False) == pytest.approx(
        np.mean([max(c) for c in itertools.combinations(v, 3)]))
    # With replacement: E[min] = sum v_i ((M-i)^n - (M-i-1)^n) / M^n
    assert budget.best_of_n(v, 2, replace=True) == pytest.approx(
        sum(min(a, b) for a in v for b in v) / 16)
    assert np.isnan(budget.best_of_n([], 1))


def test_best_of_n_curve_per_method(scan_df):
    c = budget.best_of_n_curve(scan_df)
    sven = c[c.method == 'Sven'].sort_values('n')
    assert list(sven['n']) == [1, 2, 3, 4, 5, 6] and int(sven.grid_points.iloc[0]) == 6
    assert sven['expected_best'].is_monotonic_decreasing
    # n = the whole grid is the best configuration's seed mean.
    cfg = ah.config_table(scan_df)
    best = cfg[cfg.method == 'Sven']['final_val_loss'].min()
    assert sven['expected_best'].iloc[-1] == pytest.approx(best)
    assert sven['expected_best'].iloc[0] == pytest.approx(
        cfg[cfg.method == 'Sven']['final_val_loss'].mean())
    fig, ax = plt.subplots()
    budget.plot_best_of_n(c, ax)
    assert 'budget' in ax.get_xlabel()
    plt.close(fig)


def test_best_of_n_scores_a_diverged_configuration(scan_df):
    df = scan_df.copy()
    df.loc[df['run_id'].str.contains('k4_lr0.1'), 'diverged'] = True
    df.loc[df['diverged'], 'final_val_loss'] = np.nan
    worst = budget.best_of_n_curve(df)
    assert worst.attrs['diverged_value'] == pytest.approx(
        ah.config_table(df)['final_val_loss'].max())
    assert int(worst[worst.method == 'Sven']['n_filled'].iloc[0]) == 1
    dropped = budget.best_of_n_curve(df, diverged='drop')
    assert int(dropped[dropped.method == 'Sven']['n_scored'].iloc[0]) == 5
    # Charging the divergence the worst score can only make a budget look worse.
    assert (worst[worst.method == 'Sven']['expected_best'].values[:5] >=
            dropped[dropped.method == 'Sven']['expected_best'].values).all()


def test_budget_refuses_a_test_metric(scan_df):
    df = scan_df.copy()
    df['final_test_loss'] = df['final_val_loss']
    for fn in (budget.trajectory_table, budget.best_of_n_curve):
        with pytest.raises(ValueError, match='test split'):
            fn(df, metric='final_test_loss')


# ===========================================================================
# C-A6  --  paired differences
# ===========================================================================
@pytest.fixture
def pair_df():
    """Two configurations over 5 shared seeds with differences -1 .. -5, plus a
    sixth seed that only one of them finished."""
    rows = []
    for i, seed in enumerate(range(1000, 1005)):
        rows.append(_sven(8, 0.01, 1e-4, seed, 1.0 + i))
        rows.append(_base('Adam', 0.001, seed, 2.0 + 2 * i))
    rows.append(_sven(8, 0.01, 1e-4, 1005, 6.0))
    return ah.add_derived(pd.DataFrame(rows))


def _two_configs(df):
    cfg = ah.config_table(df)
    return cfg[cfg.method == 'Sven'].iloc[0], cfg[cfg.method == 'Adam'].iloc[0]


def test_paired_difference_mean_std_and_t_interval(pair_df):
    a, b = _two_configs(pair_df)
    p = paired.paired_runs(pair_df, a, b)
    assert list(p['model_seed']) == list(range(1000, 1005))      # 1005 has no partner
    assert list(p['diff']) == pytest.approx([-1.0, -2.0, -3.0, -4.0, -5.0])

    s = paired.paired_difference(pair_df, a, b, level=0.95)
    assert s['n'] == 5 and s['n_unpaired'] == 1 and s['a_better'] == 5
    assert s['mean'] == pytest.approx(-3.0)
    assert s['std'] == pytest.approx(math.sqrt(2.5))
    assert s['sem'] == pytest.approx(math.sqrt(2.5 / 5))
    assert s['half_width'] == pytest.approx(2.7764451 * math.sqrt(0.5), rel=1e-6)
    assert s['ci_low'] == pytest.approx(-3.0 - s['half_width'])
    assert s['ci_high'] == pytest.approx(-3.0 + s['half_width'])
    # The paired interval is TIGHTER than the unpaired spread it replaces: that is
    # the whole point of matching on the model seed.
    assert s['half_width'] < pair_df['final_val_loss'].std()


def test_paired_difference_is_nan_for_a_single_seed(pair_df):
    df = pair_df[pair_df['model_seed'].isin([1000])]
    a, b = _two_configs(df)
    s = paired.paired_difference(df, a, b)
    assert s['n'] == 1 and s['mean'] == pytest.approx(-1.0)
    assert np.isnan(s['std']) and np.isnan(s['half_width'])


def test_paired_table_and_labels(pair_df):
    a, b = _two_configs(pair_df)
    tbl = paired.paired_table(pair_df, pd.DataFrame([b]), a)
    assert list(tbl['method']) == ['Adam'] and list(tbl['reference']) == ['Sven']
    assert bool(tbl['significant'].iloc[0])           # the interval excludes zero
    assert 'n=5' in paired.fmt_difference(tbl.iloc[0])
    # C-A6's wording: the seed band is a spread, never a confidence interval.
    assert 'std over seeds' in paired.SEED_SPREAD_LABEL
    assert 'confidence' not in paired.SEED_SPREAD_LABEL.lower()
    assert 'confidence' not in paired.interval_label().lower()
    assert 't$-interval' in paired.interval_label() or 't-interval' in paired.interval_label()
    fig, ax = plt.subplots()
    paired.plot_paired(tbl, ax)
    assert 'matched model seed' in ax.get_xlabel()
    plt.close(fig)


def test_paired_difference_against_an_all_diverged_configuration(pair_df):
    """``config_table`` outer-joins a configuration whose every seed diverged, so its
    ``run_ids`` is NaN.  Walking the whole table ("Sven's best vs every Adam config")
    must report n=0 for it, not raise ``TypeError: ... you passed a float``."""
    rows = [pair_df, pd.DataFrame([_base('Adam', 10.0, s, np.nan) for s in range(1000, 1005)])]
    df = ah.add_derived(pd.concat(rows, ignore_index=True))
    cfg = ah.config_table(df)
    diverged = cfg[cfg.lr == 10.0].iloc[0]
    assert not isinstance(diverged['run_ids'], list) and diverged['n_diverged'] == 5
    sven = cfg[cfg.method == 'Sven'].iloc[0]

    s = paired.paired_difference(df, sven, diverged)
    # n_unpaired = the 6 seeds only the Sven side finished; the diverged side has none.
    assert s['n'] == 0 and np.isnan(s['mean']) and np.isnan(s['std']) and s['n_unpaired'] == 6
    tbl = paired.paired_table(df, cfg[cfg.method == 'Adam'], sven)
    assert len(tbl) == 2 and int(tbl['n'].min()) == 0     # the whole table, no exception
    assert list(tbl['significant']) == [True, False]      # NaN excludes nothing
    # ... and the reference itself may be the diverged one.
    assert paired.paired_difference(df, diverged, sven)['n'] == 0


def test_paired_refuses_a_test_metric(pair_df):
    df = pair_df.copy()
    df['final_test_loss'] = df['final_val_loss']
    a, b = _two_configs(df)
    with pytest.raises(ValueError, match='test split'):
        paired.paired_difference(df, a, b, metric='final_test_loss')


# ===========================================================================
# Notebook execution (F25 / scout section 4 (v))
# ===========================================================================
def test_make_plots_uses_the_venv_jupyter():
    src = (REPO / 'make_plots.sh').read_text()
    assert '.venv/bin/jupyter' in src
    # No bare `jupyter nbconvert`: the one on PATH cannot import jupyter_core.
    assert not any(line.strip().startswith('jupyter ') for line in src.splitlines())


def test_nbconvert_is_a_recorded_dev_dependency_and_installed():
    pyproject = (REPO / 'pyproject.toml').read_text()
    assert 'nbconvert' in pyproject.split('[dependency-groups]', 1)[1].split('[tool.uv', 1)[0]
    assert importlib.util.find_spec('nbconvert') is not None, \
        "uv pip install --python .venv/bin/python nbconvert"
    assert (REPO / '.venv' / 'bin' / 'jupyter').is_file()


# ===========================================================================
# The analysis/ layout (2026-09-21 reorganisation)
# ===========================================================================
NB_GROUPS = ('headline', 'spectra', 'mlp_studies', 'large_models', 'profiling',
             'paper')


def _notebooks():
    return sorted((REPO / 'analysis' / 'notebooks').glob('*/*.ipynb'))


def test_every_notebook_lives_in_a_group_directory():
    """No notebook is left at the top of analysis/, and every one is in a known group."""
    assert not list((REPO / 'analysis').glob('*.ipynb'))
    nbs = _notebooks()
    assert len(nbs) == 25, [p.name for p in nbs]      # 20 analysis + 5 paper-figure
    assert {p.parent.name for p in nbs} <= set(NB_GROUPS)
    names = [p.stem for p in nbs]
    assert len(set(names)) == len(names), 'a notebook name must resolve to ONE path'


def test_every_notebook_bootstraps_to_the_analysis_directory():
    """The first cell must chdir to analysis/ and put analysis/lib on sys.path: every
    path in the notebooks (`../experiment_results`, `plots_v2/`, `tables/`) and in the
    helpers is relative to analysis/, not to the notebook's group directory."""
    for path in _notebooks():
        first = next(''.join(c['source']) for c in json.loads(path.read_text())['cells']
                     if c['cell_type'] == 'code')
        assert 'os.chdir(_A)' in first, path.name
        assert "sys.path.insert(0, os.path.join(_A, 'lib'))" in first, path.name
        assert "sys.path.insert(0, '.')" not in first, path.name


def test_the_helper_modules_are_in_analysis_lib():
    lib = REPO / 'analysis' / 'lib'
    for module in ('style', 'scan_analysis', 'analysis_helpers', 'headline', 'headline_figs',
                   'reviewer_figs', 'large_figs', 'spectra_figs', 'sv_diagnostics',
                   'ckpt_tools', 'paired', 'budget', 'profile_helpers'):
        assert (lib / f'{module}.py').is_file(), module
    assert not list((REPO / 'analysis').glob('*.py'))       # none left at the top


def test_helper_anchors_point_at_the_analysis_directory():
    """The modules that derive paths from __file__ moved one level deeper, so their
    anchors must climb one level further (headline.TABLES_DIR = analysis/tables, ...)."""
    import headline as hl
    import profile_helpers as ph
    analysis = REPO / 'analysis'
    assert hl.TABLES_DIR == analysis / 'tables'
    assert hl.SELECTION_PATH == REPO / 'bench' / 'best_configs.json'
    assert ph.ROOT_V3 == REPO / 'profile_results_v3'


def test_make_plots_resolves_a_notebook_by_name_and_a_group_by_directory():
    src = (REPO / 'make_plots.sh').read_text()
    assert 'analysis/notebooks' in src
    # the script names notebooks WITHOUT a group or a suffix and looks them up; the four
    # lists must therefore stay in step with what is on disk, in both directions.
    known = {p.stem for p in _notebooks()
             if p.parent.name != 'paper'}          # the paper figures run as a group
    listed = {w for w in re.findall(r'[A-Za-z0-9_]+', src) if w in known}
    assert listed == known, (known - listed, listed - known)
    assert 'find "$NBDIR" -name "${1%.ipynb}.ipynb"' in src   # by name
    assert 'if [ -d "$NBDIR/$1" ]; then' in src               # by group directory


def test_the_paper_figure_notebooks_are_generated_from_the_registry():
    """`analysis/notebooks/paper/` is one notebook per figure group, and every figure the
    paper carries must have a cell in one of them -- which is what the generator
    (`tools/build_fig_notebooks.py`) guarantees by reading the registry.
    Regenerate after adding a figure; this test is what catches forgetting to."""
    sys.path.insert(0, str(REPO / 'analysis'))
    from paper_assets import figspec as F
    reg = F.registry()
    cells = {}
    for path in (REPO / 'analysis' / 'notebooks' / 'paper').glob('fig_*.ipynb'):
        cells[path.stem.replace('fig_', '')] = json.loads(path.read_text())
    assert set(cells) == {s.group for s in reg.values()}, sorted(cells)
    for name, spec in reg.items():
        src = ''.join(''.join(c['source']) for c in cells[spec.group]['cells'])
        assert f"pf.make('{name}')" in src, f'{name} has no cell in fig_{spec.group}'


def test_the_paper_notebooks_never_write_the_manuscript_when_executed():
    """Executing one of them must be safe: the save()/pin()/rebuild() lines are there for
    the person editing, but commented, so a top-to-bottom run draws and writes nothing."""
    for path in (REPO / 'analysis' / 'notebooks' / 'paper').glob('fig_*.ipynb'):
        nb = json.loads(path.read_text())
        for cell in nb['cells']:
            if cell['cell_type'] != 'code':
                continue
            for line in cell['source']:
                bare = line.strip()
                if bare.startswith('#'):
                    continue
                assert '.save(' not in bare, (path.name, bare)
                assert 'pf.pin(' not in bare, (path.name, bare)
                assert 'pf.rebuild(' not in bare, (path.name, bare)
                assert 'pf.unpin(' not in bare, (path.name, bare)
