"""CPU tests for `analysis/large_figs.py` -- the CIFAR / nanoGPT / GPT-2 figures (WP4b).

Two kinds of fixture, no dependence on `experiment_results/`:

* hand-built DataFrames in the record schema, for the functions that take a frame
  (:func:`large_figs.paramfrac_table`, the GPT-2 helpers, the views);
* a tiny synthetic scan written into a tmp root and read through ``$SV3_RESULTS_ROOT``,
  with its own schema-2 selection file, for the functions that go through
  :mod:`headline` (:func:`large_figs.selected_runs_by_method`, :func:`sven_grid`,
  :func:`mark_selected`).

Every expected number is written down by hand, which is the point: these pin the
ARITHMETIC (which runs enter a mean, what the seed band is, which direction a cost ratio
points) rather than the plumbing.  The properties under test:

* diverged = failed: a diverged run is counted, kept out of every mean, and its COST is
  still measured (Fig 5's low fractions diverge and still have a step time);
* the actual parameter fraction, not the requested one, is what the x axis reports, and
  the cost-vs-full ratios point the way the honest note claims (masking made the step
  SLOWER);
* a curve whose entry 0 is the untrained model is drawn from epoch 0, including the
  accuracy twins ``test_acc`` / ``train_eval_acc`` that ``scan_analysis`` does not list;
* GPT-2 selection is by VALIDATION loss and refuses a test metric;
* the landscape keeps a not-yet-eligible cell, marks it, and rings the selected one;
* :func:`large_figs.savefig` writes both a PDF and a PNG.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'analysis'))   # the notebooks' own sys.path.insert(0, '.')

import analysis_helpers as ah            # noqa: E402
import headline as hl                    # noqa: E402
import large_figs as lf                  # noqa: E402
import style                             # noqa: E402

N_EPOCHS = 3
BATCH_SIZE = 8
STEPS_PER_EPOCH = 10


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------
def losses(final, *, acc=None, mem=100.0, step_s=0.5, train_eval=None, val=None):
    """A ``losses`` dict: ``train`` per epoch, ``val`` / ``test`` one longer (entry 0 is
    the untrained model), and the timing series the efficiency helpers read."""
    curve = list(val) if val is not None else [10.0, 2.0, 1.0, final][-(N_EPOCHS + 1):]
    out = {
        'train': [0.9 * v for v in curve[1:]],
        'val': curve,
        'test': [1.1 * v for v in curve],
        'epoch_times': [step_s * STEPS_PER_EPOCH] * N_EPOCHS,
        'train_times': [step_s * STEPS_PER_EPOCH] * N_EPOCHS,
        'total_time': step_s * STEPS_PER_EPOCH * N_EPOCHS,
        'avg_epoch_time': step_s * STEPS_PER_EPOCH,
        'avg_train_time': step_s * STEPS_PER_EPOCH,
        'peak_gpu_mem_mb': mem,
    }
    if acc is not None:
        out['val_acc'] = [0.1] + [acc] * N_EPOCHS
        out['test_acc'] = [0.1] + [acc] * N_EPOCHS
        out['train_eval_acc'] = [0.1] + [min(1.0, acc + 0.05)] * N_EPOCHS
    if train_eval is not None:
        out['train_eval'] = [10.0] + [train_eval] * N_EPOCHS
    return out


def record(run_id, optimizer, seed, **extra):
    rec = {'run_id': run_id, 'optimizer': optimizer, 'model_seed': seed,
           'loader_seed': 1000, 'batch_size': BATCH_SIZE, 'num_epochs': N_EPOCHS,
           'steps_per_epoch': STEPS_PER_EPOCH, 'status': 'ok', 'schema_version': 2,
           'k': None, 'rtol': None, 'lr': 0.1, 'weight_decay': 0.0,
           'svd_summary': None, 'diag_file': None, 'loss': 'mse'}
    rec.update(extra)
    return rec


def frame(records):
    return ah.add_derived(pd.DataFrame(records))


# ---------------------------------------------------------------------------
# Fig 5 fixture: 3 seeds x 3 fractions, one diverged run at the smallest fraction
# ---------------------------------------------------------------------------
@pytest.fixture
def paramfrac_frame():
    """f = 1.0 (no mask, cheap step, big memory), f = 0.5 and f = 0.1 (masked, SLOWER
    step, smaller memory); one seed at f = 0.1 blows up under the 10x rule.

    Hand-computable: at f = 0.5 the three finals are 1.0 / 2.0 / 3.0 -> mean 2.0, std 1.0.
    """
    recs = []
    for seed, bump in zip((4000, 4001, 4002), (0.0, 1.0, 2.0)):
        recs.append(record(f'svd_pf1.0_mseed{seed}', 'SVD', seed, k=8, rtol=1e-3,
                           param_fraction=1.0, actual_param_fraction=1.0, mask_mode=None,
                           losses=losses(0.5 + bump, acc=0.74, mem=20000.0, step_s=0.2,
                                         train_eval=0.4)))
        recs.append(record(f'svd_pf0.5_mseed{seed}', 'SVD', seed, k=8, rtol=1e-3,
                           param_fraction=0.5, actual_param_fraction=0.48,
                           mask_mode='elementwise',
                           losses=losses(1.0 + bump, acc=0.70, mem=14000.0, step_s=0.4,
                                         train_eval=0.8)))
        # the smallest fraction: one seed ends 100x above val[0] = the 10x rule
        blown = seed == 4002
        recs.append(record(f'svd_pf0.1_mseed{seed}', 'SVD', seed, k=8, rtol=1e-3,
                           param_fraction=0.1, actual_param_fraction=0.09,
                           mask_mode='elementwise',
                           losses=losses(6.0 + bump, acc=0.12, mem=12000.0, step_s=0.4,
                                         train_eval=6.0,
                                         val=[10.0, 9.0, 9.5, 1000.0] if blown else None)))
    return frame(recs)


def test_paramfrac_table_reports_the_actual_fraction_and_the_seed_spread(paramfrac_frame):
    tbl = lf.paramfrac_table(paramfrac_frame)
    assert list(tbl['param_fraction']) == [0.1, 0.5, 1.0]        # sorted by ACTUAL
    assert list(tbl['actual']) == pytest.approx([0.09, 0.48, 1.0])
    half = tbl[tbl['param_fraction'] == 0.5].iloc[0]
    assert half['val'] == pytest.approx(2.0)                      # 1, 2, 3
    assert half['val_std'] == pytest.approx(1.0)                  # ddof = 1
    assert half['val_min'] == pytest.approx(1.0)
    assert half['finished'] == 3 and half['attempted'] == 3


def test_paramfrac_table_counts_a_diverged_run_and_still_measures_its_cost(paramfrac_frame):
    tbl = lf.paramfrac_table(paramfrac_frame)
    low = tbl[tbl['param_fraction'] == 0.1].iloc[0]
    assert low['n_diverged'] == 1
    assert (low['finished'], low['attempted']) == (2, 3)
    # the two survivors end at 6.0 and 7.0; the blown run is NOT scored by its last value
    assert low['val'] == pytest.approx(6.5)
    # ... but its step time and memory are real measurements and stay in the cost columns
    assert low['ms_per_step'] == pytest.approx(400.0)
    assert low['peak_mem_mb'] == pytest.approx(12000.0)


def test_paramfrac_cost_ratios_say_masking_made_the_step_slower(paramfrac_frame):
    """The honest note of Fig 5: an element mask does not make full-Jacobian capture
    cheaper.  Memory falls; the step time RISES, so the ratio is above 1."""
    tbl = lf.paramfrac_table(paramfrac_frame).set_index('param_fraction')
    assert tbl.loc[1.0, 'ms_per_step_vs_full'] == pytest.approx(1.0)
    assert tbl.loc[0.5, 'ms_per_step_vs_full'] == pytest.approx(2.0)
    assert tbl.loc[0.1, 'ms_per_step_vs_full'] == pytest.approx(2.0)
    assert tbl.loc[0.1, 'peak_mem_mb_vs_full'] == pytest.approx(0.6)


def test_paramfrac_view_is_readable_and_keeps_the_counts(paramfrac_frame):
    view = lf.paramfrac_view(lf.paramfrac_table(paramfrac_frame))
    row = view[view['f (requested)'] == '0.1'].iloc[0]
    assert row['finished/attempted'] == '2/3'
    assert row['diverged'] == 1
    assert row['ms/step x full'] == '2.00x'
    assert view.attrs['mask_note'] == lf.MASK_NOTE


def test_drop_tiny_std_kills_a_round_off_spread():
    """`peak_gpu_mem_mb` is bit-identical across seeds, so its ddof=1 std is ~6e-14 and a
    table would print `22965 +/- 6.3553e-14`."""
    assert lf._drop_tiny_std(22964.58496, 6.3553e-14) == 0.0
    assert lf._drop_tiny_std(22964.58496, 12.0) == pytest.approx(12.0)
    assert lf._drop_tiny_std(np.nan, np.nan) is None


def test_paramfrac_config_is_one_configuration_and_names_the_objective(paramfrac_frame):
    """Fig 5's 15 runs share ONE (k, lr, rtol, objective); the notebook has to print it,
    because the section sits beside a cross-entropy table and the loss is not otherwise
    on the figure."""
    cfg = lf.paramfrac_config(paramfrac_frame)
    assert cfg['varies'] == []                     # nothing but param_fraction moves
    assert cfg['k'] == 8 and cfg['lr'] == pytest.approx(0.1)
    assert cfg['label'] == 'k=8, lr=0.1, rtol=0.001'
    assert cfg['n_seeds'] == 3 and cfg['seeds'] == [4000, 4001, 4002]
    assert cfg['fractions'] == [0.1, 0.5, 1.0]
    assert cfg['loss_label'] == 'MSE'
    assert cfg['fixed_config_note'] == lf.FIXED_CONFIG_NOTE


def test_paramfrac_config_reports_a_knob_that_was_re_tuned(paramfrac_frame):
    """If a future re-launch DOES re-tune per fraction, `varies` says so rather than
    printing one learning rate of several as though it were the configuration."""
    df = paramfrac_frame.copy()
    df.loc[df['param_fraction'] == 0.1, 'lr'] = 0.5
    assert 'lr' in lf.paramfrac_config(df)['varies']


#: A selection payload for the label-regression headline scan: Sven at k = 16, lr = 0.5.
HEAD_SCAN = 'wp4b_headline_fixture'
HEAD_PAYLOAD = {'schema': 2, 'scans': {HEAD_SCAN: {'methods': {'SVD': {
    'method': 'SVD', 'overrides': 'mode=svd k_values=[16] lrs=[0.5] rtol=[0.001]',
    'hparams': {'k': 16, 'lr': 0.5, 'rtol': 1e-3, 'kappa': 2}}}}}}


@pytest.fixture
def two_config_frame(paramfrac_frame):
    """The real shape of Fig 5 after the re-run: the same fractions and seeds measured at
    TWO Sven configurations -- the legacy set point (k = 8, lr = 0.1, the fixture's) and
    the one :data:`HEAD_PAYLOAD` selects (k = 16, lr = 0.5), which reaches half the loss.
    """
    sel = paramfrac_frame.copy()
    sel['k'] = 16
    sel['lr'] = 0.5
    sel['run_id'] = sel['run_id'] + '_k16_lr0.5'
    for i in sel.index:                        # the selected config is twice as good
        cur = dict(sel.at[i, 'losses'])
        cur['val'] = [v / 2 for v in cur['val']]
        cur['test'] = [v / 2 for v in cur['test']]
        sel.at[i, 'losses'] = cur
    return ah.add_derived(pd.concat([paramfrac_frame, sel], ignore_index=True))


def test_paramfrac_groups_puts_the_selected_configuration_first(two_config_frame):
    """The figure's main result is the configuration the headline scan SELECTED, and which
    one that is has to be computed from `bench/best_configs.json` rather than written into
    the notebook -- so the figure re-points itself if the selection moves again."""
    groups = lf.paramfrac_groups(two_config_frame, headline_scan=HEAD_SCAN,
                                 payload=HEAD_PAYLOAD)
    assert len(groups) == 2
    first, second = groups
    assert first['is_selected'] and not second['is_selected']
    assert first['cfg']['label'] == 'k=16, lr=0.5, rtol=0.001'
    assert first['diff'] == []
    # ... and the one that is NOT selected says exactly how it differs
    assert second['diff'] == ['k: 8 vs selected 16', 'lr: 0.1 vs selected 0.5']
    assert second['role'] == 'not selected'
    assert first['headline']['label'] == 'k=16, lr=0.5, rtol=0.001'


def test_paramfrac_groups_keeps_each_configuration_internally_fixed(two_config_frame):
    """Splitting is what keeps `paramfrac_config`'s assertion meaningful: on the whole
    frame `k` and `lr` vary, and the notebook's `assert not varies` would fire on a scan
    that is in fact two clean configurations."""
    assert set(lf.paramfrac_config(two_config_frame)['varies']) == {'k', 'lr'}
    for g in lf.paramfrac_groups(two_config_frame, headline_scan=HEAD_SCAN,
                                 payload=HEAD_PAYLOAD):
        assert g['cfg']['varies'] == []
        assert g['cfg']['fractions'] == [0.1, 0.5, 1.0]
        assert g['cfg']['n_runs'] == 9


def test_paramfrac_groups_table_is_the_group_not_the_whole_scan(two_config_frame):
    """Each group's table must be built from ITS OWN runs: a mean over both
    configurations would be a number that describes no configuration at all."""
    groups = lf.paramfrac_groups(two_config_frame, headline_scan=HEAD_SCAN,
                                 payload=HEAD_PAYLOAD)
    sel, legacy = (g['table'].set_index('param_fraction') for g in groups)
    assert legacy.loc[0.5, 'val'] == pytest.approx(2.0)        # 1, 2, 3
    assert sel.loc[0.5, 'val'] == pytest.approx(1.0)           # halved
    assert (legacy.loc[0.5, 'finished'], legacy.loc[0.5, 'attempted']) == (3, 3)
    # the diverged seed at f = 0.1 is diverged in both copies and counted in both
    assert legacy.loc[0.1, 'n_diverged'] == 1 and sel.loc[0.1, 'n_diverged'] == 1


def test_paramfrac_quality_can_overlay_two_configurations(two_config_frame):
    """Both configurations on one pair of axes: val / test keep their colours and the
    configuration is the LINE STYLE, so the figure needs no second colour cycle -- and the
    'some seeds diverged' entry is not written twice."""
    groups = lf.paramfrac_groups(two_config_frame, headline_scan=HEAD_SCAN,
                                 payload=HEAD_PAYLOAD)
    _fig, axes = plt.subplots(1, 2)
    lf.plot_paramfrac_quality(axes, groups[0]['table'], label_suffix=' (selected)')
    lf.plot_paramfrac_quality(axes, groups[1]['table'], ls='--', marker='s',
                              label_suffix=' (legacy set point)', hollow_label=False,
                              annotate_counts=False)
    labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert sum(lab == 'open marker: some seeds diverged' for lab in labels) == 1
    assert 'Validation loss (selected)' in labels
    assert 'Validation loss (legacy set point)' in labels
    # the two configurations are told apart by the line style of their errorbar, not by
    # a colour: ErrorbarContainer.lines[0] is the data line
    styles = {c.lines[0].get_linestyle() for c in axes[0].containers}
    assert styles == {'-', '--'}
    plt.close('all')


def test_paramfrac_quality_marks_the_points_with_a_diverged_seed(paramfrac_frame):
    """EXPERIMENTS.md 3.3: a Fig-5 point at pf <= 0.1 rests on 2 surviving seeds of 3, and
    any seed band on that figure has to show `finished / attempted`.  An ordinary error
    bar cannot -- a 2-seed spread is drawn exactly like a 3-seed one."""
    tbl = lf.paramfrac_table(paramfrac_frame)
    fig, axes = plt.subplots(1, 2)
    lf.plot_paramfrac_quality(axes, tbl)
    notes = [t.get_text() for t in axes[0].texts]
    assert '2/3 seeds' in notes                    # the f = 0.1 point, one seed blown up
    assert len(notes) == 1                         # ... and only that point
    assert any('open marker' in lab
               for lab in axes[0].get_legend_handles_labels()[1])
    plt.close(fig)


def test_paramfrac_cost_keeps_the_v2_profile_off_the_step_time_axis(paramfrac_frame):
    """profile v2 and the records disagree about the SIGN of the masking cost (v2 called
    empty_cache every step, which costs the unmasked path most).  On memory they agree to
    0.06%, so that overlay stays; on step time the figure alone would reverse Fig 5's
    conclusion, so it is replaced by an on-axes note until the root is v3."""
    tbl = lf.paramfrac_table(paramfrac_frame)
    prof = pd.DataFrame({'pf': [0.1, 0.5, 1.0], 'mask_mode': ['elementwise'] * 2 + ['none'],
                         'step_ms': [530.0, 626.0, 845.7],
                         'peak_mb': [12000.0, 14000.0, 20000.0]})
    fig, axes = plt.subplots(1, 2)
    lf.plot_paramfrac_cost(axes, tbl, profiles=prof)
    assert sum('profile' in lab for lab in axes[0].get_legend_handles_labels()[1]) == 2
    assert sum('profile' in lab for lab in axes[1].get_legend_handles_labels()[1]) == 0
    assert any('not comparable' in t.get_text().lower() for t in axes[1].texts)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2)         # v3: comparable, so the overlay comes back
    lf.plot_paramfrac_cost(axes, tbl, profiles=prof, profile_step_times=True)
    assert sum('profile' in lab for lab in axes[1].get_legend_handles_labels()[1]) == 2
    assert not axes[1].texts
    plt.close(fig)


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------
def test_curve_band_is_the_seed_band_over_the_usable_runs(paramfrac_frame):
    rows = paramfrac_frame[paramfrac_frame['param_fraction'] == 0.5]
    mean, lower, upper, n = lf.curve_band(rows, 'val')
    assert n == 3
    assert mean[-1] == pytest.approx(2.0)
    assert upper[-1] == pytest.approx(3.0)          # mean + 1 std
    assert lower[-1] == pytest.approx(1.0)          # clipped at the lowest seed
    assert mean[0] == pytest.approx(10.0)           # entry 0 = the untrained model


def test_curve_band_drops_the_diverged_seed(paramfrac_frame):
    rows = paramfrac_frame[paramfrac_frame['param_fraction'] == 0.1]
    _, _, _, n = lf.curve_band(rows, 'val')
    assert n == 2


@pytest.mark.parametrize('key', ['val', 'test', 'test_acc', 'train_eval'])
def test_pre_training_curves_are_drawn_from_epoch_zero(paramfrac_frame, key):
    """A curve one entry longer than `train` starts at epoch 0.  `test_acc` and
    `train_eval_acc` are NOT in scan_analysis.PRE_TRAINING_CURVES, which is why
    `curve_axis` handles them itself -- drawing them from 1 would shift every accuracy
    curve by an epoch."""
    rows = paramfrac_frame[paramfrac_frame['param_fraction'] == 0.5]
    mean, _, _, _ = lf.curve_band(rows, key)
    x = lf.curve_axis(rows, mean, key, 'epoch')
    assert len(x) == len(mean) == N_EPOCHS + 1
    assert x[0] == 0 and x[-1] == N_EPOCHS


def test_train_curve_is_drawn_from_epoch_one(paramfrac_frame):
    rows = paramfrac_frame[paramfrac_frame['param_fraction'] == 0.5]
    mean, _, _, _ = lf.curve_band(rows, 'train')
    x = lf.curve_axis(rows, mean, 'train', 'epoch')
    assert list(x) == [1, 2, 3]


def test_curve_axis_on_synchronised_time_starts_at_zero(paramfrac_frame):
    rows = paramfrac_frame[paramfrac_frame['param_fraction'] == 0.5]
    mean, _, _, _ = lf.curve_band(rows, 'val')
    x = lf.curve_axis(rows, mean, 'val', 'time', 'train_times')
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(3 * 0.4 * STEPS_PER_EPOCH)


@pytest.mark.parametrize('key', ['test_acc', 'train_eval_acc'])
def test_accuracy_twins_share_the_time_axis_of_their_loss_sibling(paramfrac_frame, key):
    """`train_eval_acc` is the one key in `large_figs.PRE_TRAINING_CURVES` that
    `scan_analysis` does not list, and an extra prepend here double-counted the
    pre-training entry on a TIME axis: the curve came out one epoch late and its last
    point 62 s early on the real CIFAR-CE timing rows.  `epoch_axis` already detects the
    entry by length, so both twins must land on exactly the `val` axis."""
    rows = paramfrac_frame[paramfrac_frame['param_fraction'] == 0.5]
    ref, _, _, _ = lf.curve_band(rows, 'val')
    cur, _, _, _ = lf.curve_band(rows, key)
    assert len(cur) == len(ref)
    for versus in ('epoch', 'time'):
        want = lf.curve_axis(rows, ref, 'val', versus, 'train_times')
        got = lf.curve_axis(rows, cur, key, versus, 'train_times')
        assert np.allclose(got, want), versus


def test_seed_note_names_which_seeds_a_panel_drew(paramfrac_frame):
    """A timing-pass figure and a confirmation table describe the same configuration on
    different seeds (base+0..4 vs base+100..104), so "(5 seeds)" is not enough."""
    rows = paramfrac_frame[paramfrac_frame['param_fraction'] == 0.5]
    assert lf.seed_note({'Sven': rows}) == 'model seeds 4000-4002 (3)'
    assert lf.seed_note(rows.iloc[:1]) == 'model seed 4000 (1)'
    assert lf.seed_note({'Sven': rows.iloc[[0, 2]]}) == 'model seeds 4000, 4002 (2)'


def test_grid_divergence_counts_uses_the_records_not_the_manifest():
    """`Scan.configs` counts `attempted` from the manifest, so a grid extension that has
    not started yet would be counted as clean runs: 0 of 195 rather than 0 of 182, a
    denominator that moves as the jobs land.  EXPERIMENTS.md 7 quotes the CIFAR-CE row on
    its records for the same reason."""
    grid = ragged_grid()
    got = lf.grid_divergence_counts(grid)
    assert (got['recorded'], got['attempted'], got['missing']) == (9, 15, 6)
    assert got['diverged'] == 0 and got['rate'] == pytest.approx(0.0)
    grid.loc[0, ['finished', 'n_diverged']] = [3, 2]
    got = lf.grid_divergence_counts(grid)
    assert (got['recorded'], got['diverged']) == (9, 2)
    assert got['rate'] == pytest.approx(2 / 9)


def test_distinct_trajectory_check_finds_a_real_collision(paramfrac_frame):
    """`budget.trajectory_table`'s `distinct_frac` counts distinct finals within each
    (seed, lr) group and scales by the number of learning rates, which a RAGGED grid
    dilutes; this is the direct question -- do two configurations of one seed actually
    produce the same curve?"""
    df = paramfrac_frame.copy()
    df['method'] = 'Sven'
    got = lf.distinct_trajectory_check(df=df)
    assert got['n_seeds'] == 3 and got['n_curves'] == 9
    assert got['n_collisions'] == 0
    # give one seed two configurations with the same trajectory
    twin = df[df['param_fraction'] == 0.5].iloc[[0]].copy()
    twin['run_id'] = 'svd_twin'
    twin['rtol'] = 0.3
    got = lf.distinct_trajectory_check(df=pd.concat([df, twin], ignore_index=True))
    assert got['n_collisions'] == 1
    assert got['collisions'][0]['model_seed'] == 4000


def test_plot_curves_labels_the_band_once_and_uses_display_names(paramfrac_frame):
    fig, ax = plt.subplots()
    runs = {'Sven': paramfrac_frame[paramfrac_frame['param_fraction'] == 1.0]}
    drawn = lf.plot_curves(ax, runs, 'val')
    assert drawn == {'Sven': 3}
    labels = ax.get_legend_handles_labels()[1]
    assert sum(lab == style.seed_spread_label() for lab in labels) == 1
    assert any(lab.startswith('Sven') for lab in labels)
    assert ax.get_yscale() == 'log'
    plt.close(fig)


def test_plot_curves_does_not_log_an_accuracy_axis(paramfrac_frame):
    fig, ax = plt.subplots()
    lf.plot_curves(ax, {'Sven': paramfrac_frame[paramfrac_frame['param_fraction'] == 1.0]},
                   'test_acc')
    assert ax.get_yscale() == 'linear'
    plt.close(fig)


def test_savefig_writes_a_pdf_and_a_png(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = lf.savefig(fig, 'thing.pdf', tmp_path / 'sub')
    plt.close(fig)
    assert [p.name for p in paths] == ['thing.pdf', 'thing.png']
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


# ---------------------------------------------------------------------------
# GPT-2 (one seed, one epoch, step-based evaluations)
# ---------------------------------------------------------------------------
@pytest.fixture
def gpt2_df():
    """Two baselines at two learning rates each and one Sven, one seed throughout.

    Adam's better VALIDATION loss is at lr = 1e-4 while its better TEST loss is at
    lr = 1e-3, so a helper that selected on test would pick a different row.
    """
    recs = []
    for lr, val, test in ((1e-4, 3.0, 4.0), (1e-3, 4.0, 2.0)):
        L = losses(val)
        L['test'] = [10.0, 9.0, 8.0, test]
        L['val_step'] = [9.0, 6.0, val]
        L['test_step'] = [9.5, 6.5, test]
        L['eval_step_idx'] = [500, 1000, 1500]
        recs.append(record(f'std_lr{lr:g}_optimAdamW', 'AdamW', 6000, lr=lr, losses=L))
    L = losses(5.0)
    L['test'] = [10.0, 9.0, 8.0, 5.5]
    L['val_step'] = [9.0, 7.0, 5.0]
    L['test_step'] = [9.5, 7.5, 5.5]
    L['eval_step_idx'] = [500, 1000, 1500]
    recs.append(record('svd_k16_lr0.1', 'SVD', 6000, k=16, rtol=1e-3, lr=0.1, losses=L))
    return lf.add_run_cost(frame(recs))


def test_gpt2_best_per_method_selects_on_validation(gpt2_df):
    best = lf.gpt2_best_per_method(gpt2_df).set_index('method')
    assert best.loc['AdamW', 'lr'] == pytest.approx(1e-4)      # val 3.0 < 4.0
    assert best.loc['AdamW', 'val'] == pytest.approx(3.0)
    assert best.loc['AdamW', 'test'] == pytest.approx(4.0)     # the OUTCOME, not the pick
    assert best.loc['Sven', 'val'] == pytest.approx(5.0)
    assert best.index[0] == 'Sven'                              # Sven first
    assert best.loc['AdamW', 'attempted'] == 2


def test_gpt2_helpers_refuse_a_test_metric(gpt2_df):
    for call in (lf.gpt2_best_per_method, lf.gpt2_lr_table):
        with pytest.raises(ValueError):
            call(gpt2_df, metric='final_test_loss')


def test_gpt2_lr_table_covers_every_run(gpt2_df):
    tbl = lf.gpt2_lr_table(gpt2_df)
    assert len(tbl) == len(gpt2_df)
    assert tbl.iloc[0]['method'] == 'Sven'
    assert set(tbl.columns) >= {'val', 'test', 'wall_h', 'ms_per_step', 'display'}


def test_gpt2_step_curve_uses_the_recorded_step_index(gpt2_df):
    row = gpt2_df[gpt2_df['method'] == 'Sven'].iloc[0]
    x, v = lf.gpt2_step_curve(row, 'val_step')
    assert list(x) == [500, 1000, 1500]
    assert list(v) == [9.0, 7.0, 5.0]
    assert lf.gpt2_step_curve(row, 'nope')[0] is None


def test_gpt2_summary_view_ranks_by_validation_and_prices_the_run(gpt2_df):
    view = lf.gpt2_summary_view(lf.gpt2_best_per_method(gpt2_df))
    assert list(view['method'])[0] == 'AdamW'          # 3.0 < 5.0
    assert view.iloc[0]['x fastest'] == '1.00x'
    assert view.attrs['n_seeds'] == 1
    assert 'val ppl' in view.columns


# ---------------------------------------------------------------------------
# A synthetic scan + selection file, for the headline-driven helpers
# ---------------------------------------------------------------------------
SCAN = 'wp4b_fixture_scan'


def write_scan(root, name, records, also_expected=()):
    """The real layout: ``{root}/{name}/{run_id}.jsonl`` + a manifest.

    ``also_expected`` are run_ids the manifest INTENDS but that have produced no record
    yet -- the shape of a grid extension while its jobs run, and the only way a
    configuration can be 1 of 3 rather than 1 of 1."""
    d = Path(root) / name
    (d / 'manifest').mkdir(parents=True, exist_ok=True)
    for rec in records:
        (d / f"{rec['run_id']}.jsonl").write_text(json.dumps(rec) + '\n')
    (d / 'manifest' / 'job0.json').write_text(
        json.dumps({'run_ids': [r['run_id'] for r in records] + list(also_expected)}))


def sven_rec(seed, final, *, k=8, lr=0.1, rtol=1e-3, suffix='', **loss_kw):
    rid = f'svd_bs8_k{k}_lr{lr:g}_rtol{rtol:g}_mseed{seed}_lseed1000{suffix}'
    return record(rid, 'SVD', seed, k=k, lr=lr, rtol=rtol, svd_mode='torch',
                  microbatch_size=None, param_fraction=None, kappa=2,
                  losses=losses(final, **loss_kw))


def adam_rec(seed, final, *, lr=0.01, suffix='', **loss_kw):
    rid = f'std_bs8_lr{lr:g}_optimAdam_mseed{seed}_lseed1000{suffix}'
    return record(rid, 'Adam', seed, lr=lr, weight_decay=0.0,
                  losses=losses(final, **loss_kw))


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """A tuning grid (two Sven cells, one of them half-finished) + a ``_confirm`` pass +
    a schema-2 selection file, read through ``$SV3_RESULTS_ROOT``."""
    tune = []
    for seed in (1000, 1001, 1002):
        tune += [sven_rec(seed, 1.0, rtol=1e-3, train_eval=0.4, acc=0.7),
                 adam_rec(seed, 2.0, train_eval=1.5, acc=0.6)]
    # a second Sven cell with only 1 of 3 seeds: present, NOT eligible (the rtol
    # extension's shape while it lands)
    tune.append(sven_rec(1000, 0.5, rtol=0.3, train_eval=0.2, acc=0.8))
    pending = [sven_rec(s, 0.5, rtol=0.3)['run_id'] for s in (1001, 1002)]
    write_scan(tmp_path, SCAN, tune, also_expected=pending)
    conf = []
    for seed in (1100, 1101, 1102):
        conf += [sven_rec(seed, 1.2, train_eval=0.5, acc=0.68),
                 adam_rec(seed, 2.2, train_eval=1.6, acc=0.58)]
    write_scan(tmp_path, f'{SCAN}_confirm', conf)
    sel_path = tmp_path / 'best_configs.json'
    sel_path.write_text(json.dumps({
        'schema': 2, 'generated_at': '2026-09-20T03:58:14-0400',
        'scans': {SCAN: {'methods': {
            'SVD': {'method': 'SVD', 'batch_size': 8, 'overrides': 'mode=svd',
                    'hparams': {'k': 8, 'lr': 0.1, 'rtol': 1e-3, 'svd_mode': 'torch',
                                'microbatch_size': None, 'param_fraction': None,
                                'kappa': 2},
                    'run_ids': [r['run_id'] for r in tune if r['optimizer'] == 'SVD'
                                and r['rtol'] == 1e-3],
                    'seed_mean_final_val': 1.0, 'n_ok': 3, 'n_expected': 3,
                    'verified': True},
            'Adam': {'method': 'Adam', 'batch_size': 8, 'overrides': 'mode=std',
                     'hparams': {'optim_name': 'Adam', 'lr': 0.01, 'weight_decay': 0.0},
                     'run_ids': [r['run_id'] for r in tune if r['optimizer'] == 'Adam'],
                     'seed_mean_final_val': 2.0, 'n_ok': 3, 'n_expected': 3,
                     'verified': True}}}}}))
    monkeypatch.setenv('SV3_RESULTS_ROOT', str(tmp_path))
    monkeypatch.setattr(hl, 'SELECTION_PATH', sel_path)
    hl.clear_cache()
    yield tmp_path
    hl.clear_cache()


def test_selected_runs_by_method_follows_the_selection_file(campaign):
    runs = lf.selected_runs_by_method(SCAN, 'confirm')
    assert list(runs) == ['Sven', 'Adam']                 # Sven first
    assert len(runs['Sven']) == 3
    assert set(runs['Sven']['model_seed']) == {1100, 1101, 1102}
    # the rtol = 0.3 cell is a DIFFERENT configuration and must not leak in
    assert set(runs['Sven']['rtol']) == {1e-3}


def test_selected_runs_match_the_confirmation_tables_numbers(campaign):
    """The curves and the table describe the same runs: the last point of the seed-mean
    val curve is the confirmation table's `val_conf`."""
    runs = lf.selected_runs_by_method(SCAN, 'confirm')
    mean, _, _, n = lf.curve_band(runs['Sven'], 'val')
    tbl = hl.confirmation_table(SCAN)
    sven = tbl[tbl['method'] == 'Sven'].iloc[0]
    assert mean[-1] == pytest.approx(sven['val_conf'])
    assert n == sven['fin_conf']


def test_run_counts_and_config_label(campaign):
    runs = lf.selected_runs_by_method(SCAN, 'confirm')
    assert lf.run_counts(runs['Adam']) == (3, 3, 0, 0)
    assert lf.config_label_of(SCAN, 'Sven') == 'k=8, lr=0.1, rtol=0.001'
    with pytest.raises(KeyError):
        lf.config_label_of(SCAN, 'Muon')


def test_sven_grid_keeps_the_half_finished_cell_and_marks_it_ineligible(campaign):
    grid = lf.mark_selected(lf.sven_grid(SCAN), SCAN)
    assert len(grid) == 2
    by_rtol = grid.set_index('rtol')
    assert bool(by_rtol.loc[1e-3, 'eligible']) is True
    assert bool(by_rtol.loc[0.3, 'eligible']) is False      # 1 of 3 seeds
    assert (by_rtol.loc[0.3, 'finished'], by_rtol.loc[0.3, 'attempted']) == (1, 3)
    # the selection of record is the eligible cell, even though the other scored lower
    assert list(grid[grid['selected']]['rtol']) == [1e-3]


def test_landscape_draws_the_provisional_cell_and_rings_the_selected_one(campaign):
    grid = lf.mark_selected(lf.sven_grid(SCAN), SCAN)
    fig, ax = plt.subplots()
    im, mat, inelig, partial = lf.plot_sven_landscape(ax, grid, x='lr', y='rtol')
    assert mat.shape == (2, 1)
    assert np.isfinite(mat).all()                 # the partial cell is NOT hidden
    assert inelig.sum() == 1                      # ... it is marked ineligible
    assert partial.sum() == 1                     # ... and its seed count is shown
    texts = [t.get_text() for t in ax.texts]
    assert any('*' in t for t in texts)           # the ineligible annotation
    assert any('1/3' in t for t in texts)         # the seed count of that cell
    assert len(ax.patches) == 2                   # the dashed ring + the red one
    plt.close(fig)


def ragged_grid():
    """A CIFAR-CE-shaped grid: a complete 5-seed cell (the selection of record), an
    extension cell with 3 of 5 seeds that is STILL ELIGIBLE and scores BETTER, and a cell
    so thin it is not eligible at all."""
    return pd.DataFrame([
        {'k': 128, 'lr': 0.1, 'rtol': 0.01, 'val': 1.40, 'finished': 5, 'attempted': 5,
         'n_diverged': 0, 'n_failed': 0, 'n_missing': 0, 'eligible': True,
         'selected': True},
        {'k': 128, 'lr': 0.1, 'rtol': 0.10, 'val': 1.36, 'finished': 3, 'attempted': 5,
         'n_diverged': 0, 'n_failed': 0, 'n_missing': 2, 'eligible': True,
         'selected': False},
        {'k': 128, 'lr': 0.5, 'rtol': 0.10, 'val': 1.55, 'finished': 1, 'attempted': 5,
         'n_diverged': 0, 'n_failed': 0, 'n_missing': 4, 'eligible': False,
         'selected': False},
    ])


def test_landscape_marks_a_partial_cell_that_is_still_eligible():
    """The case that actually occurs and the one a reader will misread: a 3-of-5 cell of
    the `rtol` extension, eligible (more than half its seeds landed) and scoring BELOW the
    red-ringed selection of record.  Marking provisional-ness from `eligible` alone drew it
    exactly like the complete 5-seed cell beside it."""
    fig, ax = plt.subplots()
    im, mat, inelig, partial = lf.plot_sven_landscape(ax, ragged_grid(), x='lr', y='rtol')
    assert inelig.sum() == 1                       # only the 1-of-5 cell
    assert partial.sum() == 2                      # BOTH incomplete cells
    texts = {t.get_text() for t in ax.texts}
    complete = next(t for t in texts if t.startswith('1.4'))
    three_of_five = next(t for t in texts if t.startswith('1.36'))
    one_of_five = next(t for t in texts if t.startswith('1.55'))
    assert complete == '1.4'                       # no count, no star: 5 of 5
    assert '3/5' in three_of_five and '*' not in three_of_five
    assert '*' in one_of_five and '1/5' in one_of_five
    assert len(ax.patches) == 2                    # one dashed ring + the red one
    plt.close(fig)


def test_optimisation_view_shows_the_generalisation_gap(campaign):
    tbl = hl.confirmation_table(SCAN)
    view = lf.optimisation_view(tbl)
    sven = view[view['method'] == 'Sven'].iloc[0]
    assert sven['train (eval mode)'].startswith('0.5')
    assert sven['val - train'] == '+0.7'           # 1.2 - 0.5
    assert sven['finished/attempted'] == '3/3'
    gaps = lf.generalisation_gaps(tbl)
    assert gaps.iloc[0]['method'] == 'Sven'        # sorted by val
    assert gaps.iloc[0]['gap'] == pytest.approx(0.7)
    assert gaps.iloc[0]['gap_rel'] == pytest.approx(1.4)


def test_cost_view_ratios_are_against_the_cheapest_method(campaign):
    """No timing pass in the fixture, so every time is NaN and the view must still
    render -- a cost table that raises on a missing pass is a table nobody can print
    while a pass is still queued."""
    eff = hl.efficiency_table(SCAN)
    view = lf.cost_view(eff)
    assert len(view) == 2
    assert set(view['s/epoch']) == {'--'}
    assert set(view['fin/att']) == {'0/0'}


def test_landscape_axes_and_shared_scale(campaign):
    """A multi-panel landscape needs both, and for the same reason: `pcolormesh`
    autoscales per axes, so panels drawn independently are coloured on different scales
    under one colour bar, and panels with different y values cannot be read against each
    other at all."""
    grid = lf.sven_grid(SCAN)
    xs, ys = lf.landscape_axes(grid, 'lr', 'rtol')
    assert xs == [0.1]
    assert ys == [1e-3, 0.3]
    vmin, vmax = lf.landscape_scale(grid, 'val')
    assert vmin == pytest.approx(np.log10(0.5))     # the ineligible cell is the lowest
    assert vmax == pytest.approx(np.log10(1.0))
    vmin_e, _ = lf.landscape_scale(grid, 'val', eligible_only=True)
    assert vmin_e == pytest.approx(np.log10(1.0))


def test_landscape_honours_a_forced_frame(campaign):
    """Cells outside the forced axes are dropped, cells inside but absent stay blank."""
    grid = lf.sven_grid(SCAN)
    fig, ax = plt.subplots()
    im, mat, _, _ = lf.plot_sven_landscape(ax, grid[grid['rtol'] == 1e-3], x='lr', y='rtol',
                                           xs=[0.1, 0.5], ys=[1e-3, 0.3],
                                           vmin=-1.0, vmax=1.0)
    assert mat.shape == (2, 2)
    assert np.isfinite(mat).sum() == 1               # only (rtol 1e-3, lr 0.1) ran
    assert im.get_clim() == (-1.0, 1.0)
    plt.close(fig)
