"""CPU tests for `analysis/lib/headline_figs.py` -- the WP2 phase-B notebook helpers.

The synthetic campaign is the one `tests/test_headline.py` builds (a tuning scan, its
``_confirm`` and ``_timing`` passes and a hand-written schema-2 selection file, read
through ``$SV3_RESULTS_ROOT``), imported rather than copied so the two test files cannot
drift apart.  Nothing here reads ``experiment_results/``.

The properties under test are the three things this module owns and `headline.py` does
not:

* the **time axis** really is the standalone pass's synchronised per-epoch training time,
  cumulated, and a pre-training curve starts at t = 0 rather than at the end of epoch 1
  (getting that wrong shifts every validation curve by an epoch);
* the **divergence table** reports the narrow (``status``) and the wide
  (:func:`style.is_diverged`) counts as different numbers, since quoting only the first is
  how "Sven never diverges" gets written down;
* the **grid-edge** verdict distinguishes ``k = B`` -- a natural maximum -- from a real
  edge of a swept range.

Plus the plumbing that would silently produce a wrong figure: the curve helper draws every
method it is given, skips (rather than mis-draws) one whose axis cannot be built, and the
legend relabeller turns a raw optimizer key into its display name.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'analysis' / 'lib'))
sys.path.insert(0, str(REPO / 'analysis'))
sys.path.insert(0, str(REPO / 'tests'))

import headline as hl                    # noqa: E402
import headline_figs as hf               # noqa: E402
import style                             # noqa: E402
from test_headline import (SCAN, N_EPOCHS, STEPS_PER_EPOCH, BATCH_SIZE,  # noqa: E402,F401
                           CONF_SEEDS, DATA_SEEDS, campaign)             # noqa: F401


# ---------------------------------------------------------------------------
# curves and the three axes
# ---------------------------------------------------------------------------
def test_confirm_runs_are_the_selected_configuration_only(campaign):
    """The fixture puts a MICROBATCHED twin of Sven's configuration in the confirmation
    pass, much better than the real one; matching on the selected ``microbatch_size=None``
    is what keeps it out."""
    runs = hf.confirm_runs(SCAN)
    assert set(runs) == {'Sven', 'Adam'}
    for m, rows in runs.items():
        assert len(rows) == len(CONF_SEEDS) * len(DATA_SEEDS), m
        assert not any('_mb' in r for r in rows['run_id']), f'{m} matched a decoy'
        # one configuration per data seed (the data seed is part of the run_id), no more
        keys = {style.config_key(r) for r in rows['run_id']}
        assert len(keys) == len(DATA_SEEDS), f'{m} matched {len(keys)} configurations'


def test_time_axis_is_the_standalone_synchronised_training_time(campaign):
    """The x values are the CUMULATIVE per-epoch ``train_times`` of the timing pass --
    not the wall clock, and not the scan's own (sharded) times."""
    times = hf.standalone_epoch_times(SCAN)
    assert set(times) == {'Sven', 'Adam'}
    # the fixture's timing pass records train_times of 1.6 s (Sven) and 0.8 s (Adam) per
    # epoch, against epoch_times of 2.0 / 1.0: a time axis built from the WALL clock would
    # be 25% longer, and one built from the scan's own times longer still
    assert np.allclose(times['Sven'], 1.6)
    assert np.allclose(times['Adam'], 0.8)
    assert len(times['Sven']) == N_EPOCHS

    rows = hf.confirm_runs(SCAN)['Sven']
    import scan_analysis as sa
    mean, _, _ = sa.seed_band(rows, 'val')
    x = hf.curve_axis(rows, mean, which='val', versus='time', epoch_times=times['Sven'])
    # `val` carries the untrained model at index 0, so the axis starts at 0 and the last
    # point is the whole pass: 4 epochs x 1.6 s
    assert len(x) == len(mean) == N_EPOCHS + 1
    assert x[0] == 0.0
    assert np.isclose(x[-1], N_EPOCHS * 1.6)


def test_a_train_curve_has_no_pre_training_point_and_starts_at_the_first_epoch(campaign):
    rows = hf.confirm_runs(SCAN)['Sven']
    import scan_analysis as sa
    mean, _, _ = sa.seed_band(rows, 'train')
    x = hf.curve_axis(rows, mean, which='train', versus='time',
                      epoch_times=np.full(N_EPOCHS, 0.5))
    assert len(x) == N_EPOCHS
    assert np.isclose(x[0], 0.5)          # the END of epoch 1, not 0


def test_step_and_example_axes_use_the_recorded_steps_per_epoch(campaign):
    rows = hf.confirm_runs(SCAN)['Sven']
    import scan_analysis as sa
    mean, _, _ = sa.seed_band(rows, 'val')
    steps = hf.curve_axis(rows, mean, which='val', versus='step')
    examples = hf.curve_axis(rows, mean, which='val', versus='examples')
    assert steps[0] == 0 and steps[-1] == N_EPOCHS * STEPS_PER_EPOCH
    assert np.allclose(examples, steps * BATCH_SIZE)


def test_a_curve_with_no_time_axis_is_skipped_not_drawn_in_the_wrong_unit(campaign):
    """No standalone times for a method -> its curve is left out, not plotted against
    epoch indices pretending to be seconds."""
    runs = hf.confirm_runs(SCAN)
    fig, ax = plt.subplots()
    drawn = hf.plot_curves(SCAN, ax, which='val', versus='time', runs=runs,
                           times={'Sven': np.full(N_EPOCHS, 0.5)}, quiet=True)
    plt.close(fig)
    assert drawn == ['Sven']


def test_the_log_time_axis_masks_the_pre_training_point_instead_of_clipping_it(campaign):
    """Every validation curve starts with the UNTRAINED model at x = 0, and matplotlib's
    default for a log scale is ``nonpositive='clip'``, which maps 0 to log10 = -1000
    rather than dropping it: the segment from the pre-training point to epoch 1 then
    enters the axes at the left-hand edge and is drawn as a FLAT LINE at the epoch-1 loss,
    i.e. as data at times the method was never measured at.  Masking is what keeps the
    first drawn point at the first measurement."""
    runs = hf.confirm_runs(SCAN)
    times = {m: np.full(N_EPOCHS, 0.5) for m in runs}
    fig, ax = plt.subplots()
    hf.plot_curves(SCAN, ax, which='val', versus='time', runs=runs, times=times,
                   quiet=True)
    t = ax.get_xaxis().get_transform().transform([0.0])[0]
    # the x data still carries the pre-training sample (curve_axis is unchanged) ...
    line = [ln for ln in ax.lines if ln.get_label() == 'Sven'][0]
    assert line.get_xdata()[0] == 0.0
    plt.close(fig)
    # ... and the axis does NOT place it at a finite coordinate near the left edge
    assert not np.isfinite(t), f'x = 0 transformed to {t}: clipped, not masked'


def test_plot_curves_draws_every_method_and_puts_sven_last(campaign):
    runs = hf.confirm_runs(SCAN)
    fig, ax = plt.subplots()
    drawn = hf.plot_curves(SCAN, ax, which='val', versus='step', runs=runs, quiet=True)
    labels = [ln.get_label() for ln in ax.lines]
    plt.close(fig)
    assert set(drawn) == {'Sven', 'Adam'}
    assert drawn[-1] == 'Sven', 'Sven must be drawn last so it is on top'
    assert 'Sven' in labels and 'Adam' in labels


def test_curve_figure_is_two_rows_by_three_axes_and_names_its_methods(campaign):
    fig, axes, info = hf.curve_figure(SCAN, which='val', top_n=1)
    plt.close(fig)
    assert axes.shape == (2, 3)
    assert info['axes'] == list(hf.CURVE_AXES)
    assert 'Sven' in info['top'] and set(info['all']) == {'Sven', 'Adam'}


def test_method_ranking_is_by_confirmation_validation_loss(campaign):
    # the fixture's Sven confirmation mean (0.25) beats Adam's (0.5)
    assert hf.method_ranking(SCAN) == ['Sven', 'Adam']


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------
def test_divergence_table_reports_the_narrow_and_the_wide_count_separately(campaign):
    """The fixture's confirmation Adam loses one run to ``status: diverged`` and one to
    the 10x blow-up rule, which the runner recorded as ``ok``.  The wide count sees both,
    the narrow one only the first; a table that reported one number could not say that --
    and on the real `toy_1d_scan` the difference for Sven is 194 against 0."""
    df = hl.load(SCAN, 'confirm')
    div = hf.divergence_by_method(df).set_index('method')
    assert div.loc['Adam', 'n_recorded'] == 1
    assert div.loc['Adam', 'n_diverged'] == 2, 'the analysis rule must be the wider one'
    assert div.loc['Adam', 'n_recorded'] < div.loc['Adam', 'n_diverged']
    assert 0 < div.loc['Adam', 'frac_diverged'] <= 1
    assert div.loc['Adam', 'n_configs_all_bad'] <= div.loc['Adam', 'n_configs_any_bad']
    # and the counts are of runs, not of configurations
    assert div['n_runs'].sum() == len(df)
    # the tuning grid's `oom` record is a FAILED attempt, counted apart from a divergence
    tune = hf.divergence_by_method(hl.load(SCAN, '')).set_index('method')
    assert tune.loc['Adam', 'n_failed'] == 1
    assert tune.loc['Adam', 'n_diverged'] == 0


def test_divergence_table_works_on_a_frame_that_only_has_an_optimizer_column(campaign):
    """``scan_analysis.load_scan`` gives no ``method`` column, and its record spellings
    are the raw ones (``SVD``); the table must still be keyed by the analysis name."""
    df = hl.load(SCAN, '').drop(columns=['method'])
    div = hf.divergence_by_method(df)
    assert 'Sven' in set(div['method'])
    assert 'SVD' not in set(div['method'])


def test_sven_divergence_grid_leaves_absent_grid_points_blank(campaign):
    df = hl.load(SCAN, '')
    frac, counts = hf.sven_divergence_grid(df, index='rtol', columns='lr')
    assert frac is not None
    assert ((frac.fillna(0) >= 0) & (frac.fillna(0) <= 1)).all().all()
    # a cell with no run at all is NaN, never 0.0 ("nothing diverged here")
    missing = counts.isna()
    assert bool((frac[missing].isna()).all().all()) if missing.any().any() else True


def test_plot_divergence_grid_annotates_every_cell(campaign):
    df = hl.load(SCAN, '')
    frac, counts = hf.sven_divergence_grid(df, index='rtol', columns='lr')
    fig, ax = plt.subplots()
    hf.plot_divergence_grid(frac, ax, counts)
    n_cells = frac.shape[0] * frac.shape[1]
    texts = [t for t in ax.texts]
    plt.close(fig)
    assert len(texts) == n_cells


# ---------------------------------------------------------------------------
# the Sven landscape
# ---------------------------------------------------------------------------
def test_k_equals_B_is_a_natural_maximum_not_a_grid_edge(campaign):
    """``k = B`` means "no truncation at all", so calling it a grid edge would ask for a
    sweep that cannot exist."""
    import scan_analysis as sa
    scan_obj = sa.load_scan(SCAN, 'fixture', Path(hl.HERE) / '_unused_plots')
    chosen, _ = hf.selected_sven(SCAN)
    edges = hf.grid_edges(scan_obj, {**chosen, 'k': float(scan_obj.B)})
    k_row = edges[edges['axis'] == 'k'].iloc[0]
    assert 'natural maximum' in k_row['position']
    assert 'k' not in edges.attrs['on_edge']


def test_a_selected_value_at_the_end_of_a_swept_range_is_called_an_edge(campaign):
    import scan_analysis as sa
    scan_obj = sa.load_scan(SCAN, 'fixture', Path(hl.HERE) / '_unused_plots')
    chosen, _ = hf.selected_sven(SCAN)
    edges = hf.grid_edges(scan_obj, {**chosen, 'lr': max(scan_obj.sven_lrs)})
    lr_row = edges[edges['axis'] == 'lr'].iloc[0]
    assert lr_row['position'] == 'HIGH EDGE of the grid'
    assert 'lr' in edges.attrs['on_edge']


def test_plot_selected_marker_rings_the_right_cell():
    piv = pd.DataFrame(np.arange(6.0).reshape(3, 2), index=[1.0, 2.0, 4.0],
                       columns=[0.1, 0.5])
    fig, ax = plt.subplots()
    ax.imshow(piv.values)
    patch = hf.plot_selected_marker(ax, piv, {'k': 2.0, 'lr': 0.5})
    xy = patch.get_xy()
    plt.close(fig)
    assert np.allclose(xy, (0.5, 0.5))     # column index 1, row index 1, minus half a cell


def test_plot_selected_marker_is_silent_when_the_config_is_off_the_pivot():
    piv = pd.DataFrame(np.zeros((2, 2)), index=[1.0, 2.0], columns=[0.1, 0.5])
    fig, ax = plt.subplots()
    assert hf.plot_selected_marker(ax, piv, {'k': 99.0, 'lr': 0.5}) is None
    plt.close(fig)


# ---------------------------------------------------------------------------
# instance spread, cross-scan matrices, labels
# ---------------------------------------------------------------------------
def test_instance_spread_is_the_ratio_of_the_two_headline_spreads(campaign):
    conf = hl.confirmation_table(SCAN)
    spread = hf.instance_spread(SCAN, table=conf)
    assert list(spread.attrs['data_seeds']) == list(DATA_SEEDS)
    row = spread[spread['method'] == 'Sven'].iloc[0]
    src = conf[conf['method'] == 'Sven'].iloc[0]
    assert np.isclose(row['val_conf_dsspread'], src['val_conf_dsspread'])
    assert np.isclose(row['ratio_instance_over_seed'],
                      src['val_conf_dsspread'] / src['val_conf_seedspread'])


def test_instance_spread_is_empty_without_data_seeds(campaign, monkeypatch):
    conf = hl.confirmation_table(SCAN)
    conf.attrs['data_seeds'] = []
    assert hf.instance_spread(SCAN, table=conf).empty


def test_scan_matrix_puts_sven_first_and_leaves_a_missing_method_blank(campaign):
    a = pd.DataFrame({'display': ['Sven', 'Adam'], 'v': [1.0, 2.0]})
    b = pd.DataFrame({'display': ['Adam'], 'v': [3.0]})
    m = hf.scan_matrix({'toy_1d_scan': a, 'polynomial_scan': b}, 'v')
    assert list(m.index) == ['Sven', 'Adam']
    assert np.isnan(m.loc['Sven', hl.scan_title('polynomial_scan')])
    assert m.loc['Adam', hl.scan_title('polynomial_scan')] == 3.0


def test_relabel_methods_turns_raw_keys_into_display_names():
    fig, ax = plt.subplots()
    ax.plot([0, 1], label='LBFGS (45 pts)')
    ax.plot([0, 1], label='JD_UPGrad (6 pts)')
    ax.plot([0, 1], label='Sven (180 pts)')
    ax.legend()
    leg = hf.relabel_methods(ax)
    labels = [t.get_text() for t in leg.get_texts()]
    plt.close(fig)
    assert labels == ['Stochastic L-BFGS (45 pts)', 'JD (UPGrad) (6 pts)',
                      'Sven (180 pts)']


def test_save_writes_a_pdf_and_a_png(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    paths = hf.save(fig, tmp_path / 'figs', 'demo')
    plt.close(fig)
    assert [p.name for p in paths] == ['demo.pdf', 'demo.png']
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


# ---------------------------------------------------------------------------
# the pass-level checks a wrong table would otherwise slip past
# ---------------------------------------------------------------------------
def test_pass_gpus_flags_a_device_mixed_pass(capsys):
    """Every rank and paired difference comes out of ONE pass, and is only a clean
    comparison between methods if that pass ran on one device type -- which the two MNIST
    confirmation passes did not (Sven on a full A100-80GB against MuonW and HIG on MIG
    slices).  Silence on that is what the check exists to prevent."""
    mixed = {'Sven': pd.DataFrame({'gpu_name': ['A100-80GB'] * 5}),
             'MuonW': pd.DataFrame({'gpu_name': ['A100-40GB MIG'] * 5})}
    g = hf.pass_gpus('toy_1d_scan', runs=mixed)
    assert g.attrs['homogeneous'] is False
    assert g.attrs['gpus'] == {'A100-80GB': 5, 'A100-40GB MIG': 5}
    hf.gpu_homogeneity_line('toy_1d_scan', gpus=g)
    out = capsys.readouterr().out
    assert 'NOT homogeneous' in out and 'Sven' in out and 'MuonW' in out

    same = {'Sven': pd.DataFrame({'gpu_name': ['A100-80GB'] * 5}),
            'Adam': pd.DataFrame({'gpu_name': ['A100-80GB'] * 5})}
    g2 = hf.pass_gpus('toy_1d_scan', runs=same)
    assert g2.attrs['homogeneous'] is True
    hf.gpu_homogeneity_line('toy_1d_scan', gpus=g2)
    assert 'one device type' in capsys.readouterr().out


def test_pass_gpus_is_silent_about_a_pass_that_never_ran(campaign):
    g = hf.pass_gpus(SCAN, 'diag')
    assert g.empty and g.attrs['homogeneous'] is None


def test_missing_confirmation_shouts_about_a_zero_row_match(campaign, capsys):
    """A selection that moved after the ``_confirm`` pass ran matches 0 records; the
    confirmation table reports that as ``0/0`` with a NaN mean, which ``ranking_summary``
    sorts LAST -- i.e. the method would appear at the bottom of the rank matrix with no
    error at all.  ANALYSIS_PLAN decision 5's re-run order exists for exactly this."""
    tbl = hl.confirmation_table(SCAN)
    assert hf.missing_confirmation(tbl, quiet=True) == []
    fake = tbl.copy()
    fake.loc[fake['method'] == 'Adam', ['fin_conf', 'att_conf']] = 0
    fake.attrs['scan'] = SCAN
    assert hf.missing_confirmation(fake) == ['Adam']
    out = capsys.readouterr().out
    assert 'NO CONFIRMATION RUNS' in out and 'must not be ranked' in out


def test_timing_join_view_names_the_tolerance_it_tested(campaign):
    """``bit_reproduced`` is ``max_rel_dev <= 1e-3``, and nothing in the campaign is
    bit-reproducible across GPU types; the notebook column has to say which it is."""
    tj = hl.timing_join_report(SCAN)
    view = hf.timing_join_view(tj)
    assert 'bit_reproduced' not in view.columns
    assert 'reproduced_within_0.001' in view.columns
    assert view.attrs['tol'] == tj.attrs['tol']
    # and the values are untouched
    assert list(view['reproduced_within_0.001']) == list(tj['bit_reproduced'])


def test_campaign_divergence_counts_only_the_tuning_grids(campaign):
    """The campaign-wide Sven count is recomputed rather than quoted from
    ``EXPERIMENTS.md``, because the CIFAR-CE extension is still moving its denominator.
    ``_confirm`` / ``_timing`` re-run one already-selected configuration and are not a
    search, so they must not be in it."""
    assert hf.on_grid_scans() == [SCAN]           # not SCAN_confirm / SCAN_timing
    camp = hf.campaign_divergence()
    assert list(camp['scan']) == [SCAN]
    # 3 seeds x (k=4, k=8) Sven records in the tuning grid, none diverged
    assert camp.attrs['n_runs'] == 6
    assert camp.attrs['n_diverged'] == 0
    assert camp.attrs['n_scans'] == 1 and camp.attrs['n_scans_affected'] == 0
    div = hf.divergence_by_method(hl.load(SCAN, '')).set_index('method')
    assert camp.loc[camp['scan'] == SCAN, 'n_runs'].iloc[0] == div.loc['Sven', 'n_runs']


def test_divergence_pattern_says_none_when_there_is_none(campaign):
    """Templated prose asserting "at the bottom of the rtol grid, worse as lr grows" over
    a table of zeros is what this replaces: Sven has 0 of 800 on ``mnist_scan_ce``."""
    pat = hf.divergence_pattern(hl.load(SCAN, ''))
    assert pat['any'] is False
    assert 'NO divergence anywhere' in pat['summary']
    assert pat['n_diverged'] == 0 and pat['n_runs'] == 6


def test_divergence_pattern_describes_a_real_pattern(campaign):
    """Adam's confirmation pass loses one run per data seed, so the pattern helper must
    report the cells and not claim a clean grid."""
    pat = hf.divergence_pattern(hl.load(SCAN, 'confirm'), method='Adam',
                                index='data_seed', columns='lr')
    assert pat['any'] is True
    # one loss per data seed: `status: diverged` on the first, the 10x rule on the second,
    # and the WIDE rule is what sees both
    assert pat['n_diverged'] == len(DATA_SEEDS)
    assert 'diverge' in pat['summary']
    assert len(pat['cells']) == len(DATA_SEEDS)
    assert set(pat['data_seed_affected']) == set(float(s) for s in DATA_SEEDS)


def test_equal_budget_reads_a_small_grid_at_its_own_size(campaign):
    """At n = 8 a six-point grid has been exhausted, so it is read at n = 6 -- the most
    generous honest reading.  Getting this wrong (dropping the method) is what let the
    first draft claim the budget objection was answered."""
    import budget
    curves = budget.best_of_n_curve(hl.load(SCAN, ''), metric=hl.SELECTION_METRIC)
    eq = hf.equal_budget_table(curves, n=8)
    assert set(eq['method']) == {'Sven', 'Adam'}
    assert eq.attrs['n'] == 8
    # Sven has 6 grid points (3 seeds x 2 k values -> 2 configurations? no: 2 configs),
    # so n_used is its grid size, never more than it has
    for r in eq.itertuples():
        assert r.n_used == min(8, r.grid_points)
    # best first
    assert list(eq['expected_best']) == sorted(eq['expected_best'])


# ---------------------------------------------------------------------------
# the closing numbers
# ---------------------------------------------------------------------------
def test_closing_numbers_quote_headline_not_its_own_arithmetic(campaign):
    """Every value the closing cell prints must be the one `headline.py` computed."""
    conf = hl.confirmation_table(SCAN)
    nums = hf.closing_numbers(SCAN, conf=conf, verbose=False)
    sven = conf[conf['method'] == 'Sven'].iloc[0]
    assert nums['sven_rank_val'] == 1
    assert nums['n_methods'] == len(conf)
    assert np.isclose(nums['sven_val'], sven['val_conf'])
    assert np.isclose(nums['sven_val_std'], sven['val_conf_std'])
    assert nums['sven_config'] == sven['config']
    assert nums['best_method'] == sven['display']
    # the counts are the ones divergence_by_method reports
    div = hf.divergence_by_method(hl.load(SCAN, '')).set_index('method')
    assert nums['sven_grid_diverged'] == div.loc['Sven', 'n_diverged']
    assert nums['sven_grid_recorded'] == div.loc['Sven', 'n_recorded']
    # the three claims the first draft of the closing cells asserted without a number
    assert 'sven_rank_epochs_to_target' in nums and 'n_reached_target' in nums
    assert nums['budget_equal_n'] == hf.BUDGET_EQUAL_N
    assert 'sven_budget_rank_at_equal_n' in nums
    assert 'n_better_than_sven_at_equal_n' in nums
    assert 'sven_div_pattern' in nums and 'campaign_sven_runs' in nums


def test_closing_numbers_epoch_rank_matches_the_time_to_target_table(campaign):
    """`sven_rank_epochs_to_target` is Sven's place among the methods that REACHED the
    pre-declared median target, ordered by epochs -- the steps-axis claim as a number."""
    ttt = hl.time_to_target_table(SCAN)
    nums = hf.closing_numbers(SCAN, ttt=ttt, verbose=False)
    med = ttt[ttt['target'] == hl.TARGET_NAMES[1.0]]
    reached = (med[med['n_reached'] > 0].dropna(subset=['epochs'])
               .sort_values('epochs', kind='mergesort'))
    assert nums['n_reached_target'] == len(reached)
    assert (nums['sven_rank_epochs_to_target']
            == list(reached['method']).index('Sven') + 1)
    assert nums['ttt_fewest_epochs_method'] == reached.iloc[0]['display']


def test_closing_numbers_survive_a_scan_with_no_timing_pass(campaign):
    """A pass that has not run must not take the closing cell down with it: the timing
    rows are simply absent from the dict the prose reads."""
    nums = hf.closing_numbers(SCAN, conf=hl.confirmation_table(SCAN),
                              eff=pd.DataFrame(), ttt=pd.DataFrame(columns=['target']),
                              verbose=False)
    assert 'sven_epoch_s' not in nums and 'fastest_method' not in nums
    assert nums['sven_rank_val'] == 1


def test_closing_numbers_print_every_value_they_return(campaign, capsys):
    """The closing markdown cell may only refer to names the code cell PRINTED."""
    nums = hf.closing_numbers(SCAN, verbose=True)
    printed = capsys.readouterr().out
    for key in nums:
        assert f'  {key} ' in printed or f'  {key}  ' in printed, f'{key} not printed'
