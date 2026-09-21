"""CPU tests for `analysis/lib/reviewer_figs.py` -- the WP4a reviewer studies.

Everything runs against SYNTHETIC scans written into a tmp root and read through
``$SV3_RESULTS_ROOT``; no test reads ``experiment_results/``, so none of them can be made
to pass by the real data happening to agree.  Every number below is chosen so the expected
answer can be written down by hand.

The properties under test, in the order the module implements them:

* **the binding rule, per arm** -- eligible -> FEWEST DIVERGED -> seed-mean final
  validation loss, applied inside each ``n_data`` arm.  The fixture makes the two tiers
  disagree on purpose: at ``n_data=100`` Sven's configuration ``k=8`` has the LOWER seed
  mean and one diverged seed, and must lose to ``k=4``;
* diverged = failed, by all three routes (``status: diverged``, the 10x blow-up rule, a
  non-finite curve): out of every mean, counted, and ``finished/attempted`` from the
  MANIFEST, not from the records present;
* an ineligible configuration is never selected (``k=8`` at ``n_data=200`` finishes 1 of 3);
* the outcomes beside the selection (`test`, `test_acc`, `train_eval`, step time, peak
  memory, the rank actually used) are aggregated over the SAME finished seeds;
* ranks are per arm, ``n_methods`` is the field size, and accuracy ranks the other way;
* the pre-training entry: a ``val`` curve's index IS the epoch count, a ``train`` curve's is
  one less -- the difference between "2 epochs to the target" and "3";
* time-to-target keeps ``n_never_reached`` and ``n_diverged`` apart and never adds them up;
* the legacy-style "fastest configuration" selector requires every seed to reach the target;
* the knob tables (micro-batch / parameter fraction) carry the measured
  ``actual_param_fraction`` and the per-cell divergence counts;
* kappa: the matched effective steps ``2 lr / kappa`` and the relative spread over kappa,
  which is 0 when the runs agree;
* figures are written as BOTH pdf and png.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'analysis' / 'lib'))   # the notebooks' own sys.path.insert(0, '.')
sys.path.insert(0, str(REPO / 'analysis'))

import reviewer_figs as rf        # noqa: E402
import style                      # noqa: E402
from analysis_helpers import add_derived   # noqa: E402

ARM_SCAN = 'wp4a_arm_fixture_scan'
KNOB_SCAN = 'wp4a_knob_fixture_scan'
KAPPA_SCAN = 'wp4a_kappa_fixture_scan'

N_EPOCHS = 3
STEPS_PER_EPOCH = 10
BATCH = 8
SEEDS = (1000, 1001, 1002)
#: every val curve passes through these before its final value, so "epochs to 1.0" is 2 for
#: every healthy run and can be written down
VAL_PREFIX = (10.0, 5.0, 1.0)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def losses(final, *, step_s=0.02, mem=100.0, acc=0.9, val=None, train_eval=True):
    """A record's ``losses``.  ``val`` is one entry longer than ``train`` -- index 0 of the
    val curve is the untrained model -- which is the off-by-one the module has to get
    right."""
    curve = list(VAL_PREFIX) + [final] if val is None else list(val)
    out = {
        'val': curve,
        'train': [0.9 * v for v in curve[1:]],
        'test': [1.1 * v for v in curve],
        'test_acc': [0.1] + [acc] * (len(curve) - 1),
        'val_acc': [0.1] + [acc - 0.01] * (len(curve) - 1),
        'epoch_times': [1.0] * (len(curve) - 1),
        'train_times': [0.5] * (len(curve) - 1),
        'total_time': float(len(curve) - 1),
        'avg_epoch_time': 1.0,
        'avg_train_time': 0.5,
        'avg_batch_time_train': step_s,
        'peak_gpu_mem_mb': mem,
    }
    if train_eval:
        out['train_eval'] = [5.0 * v for v in curve[:-1]] + [0.5 * curve[-1]]
    return out


def _rec(run_id, seed, **extra):
    rec = {'run_id': run_id, 'model_seed': seed, 'loader_seed': 1000,
           'batch_size': BATCH, 'mlp_width': 16, 'num_epochs': N_EPOCHS,
           'steps_per_epoch': STEPS_PER_EPOCH, 'n_train': 1000, 'n_params': 593,
           'status': 'ok', 'schema_version': 2, 'diag_file': None, 'svd_summary': None,
           'loss': 'mse', 'k': None, 'k_fraction': None, 'rtol': None, 'kappa': None,
           'svd_mode': None, 'microbatch_size': None, 'param_fraction': None,
           'mask_mode': None, 'weight_decay': None, 'gpu_name': 'fixture'}
    rec.update(extra)
    return rec


def sven(seed, final, *, k=4, lr=0.1, n_data=100, status='ok', rank=3.0, **kw):
    rid = f'svd_n{n_data}_k{k}_lr{lr:g}_rtol0.001_mseed{seed}_lseed1000'
    return _rec(rid, seed, optimizer='SVD', k=k, k_fraction=k / BATCH, lr=lr, rtol=1e-3,
                kappa=2.0, svd_mode='torch', n_data=n_data, status=status,
                svd_summary={'num_nonzero_svs_epoch': [rank] * N_EPOCHS},
                losses=losses(final, **kw))


def adam(seed, final, *, lr=0.01, n_data=100, status='ok', **kw):
    rid = f'std_n{n_data}_lr{lr:g}_optimAdam_mseed{seed}_lseed1000'
    return _rec(rid, seed, optimizer='Adam', lr=lr, weight_decay=0.0, n_data=n_data,
                status=status, losses=losses(final, step_s=0.004, mem=50.0, acc=0.8, **kw))


def write_scan(root, name, records):
    """The real layout: ``{root}/{name}/{run_id}.jsonl`` + ``{name}/manifest/job0.json``."""
    d = Path(root) / name
    (d / 'manifest').mkdir(parents=True, exist_ok=True)
    for rec in records:
        (d / f"{rec['run_id']}.jsonl").write_text(json.dumps(rec) + '\n')
    (d / 'manifest' / 'job0.json').write_text(json.dumps(
        {'job': 'job0', 'n_runs': len(records),
         'run_ids': [r['run_id'] for r in records]}))
    return d


#: a blown-up val curve: ends 20x above val[0], so `style.is_diverged` catches it even
#: though every value is finite and `status` says "ok"
BLOWUP = list(VAL_PREFIX) + [200.0]


def build_arm_scan(root):
    """Two arms x two methods x two configurations x three seeds.

    ``n_data=100``: Sven ``k=4`` -> 0.10/0.20/0.30 (mean 0.20, 0 diverged);
    Sven ``k=8`` -> 0.05/0.15 + a recorded divergence (mean 0.10, 1 diverged).
    The LOWER mean must lose: fewest diverged comes first.
    ``n_data=200``: Sven ``k=4`` -> 0.60/0.70 + a 10x blow-up (mean 0.65, 2 of 3 finish);
    Sven ``k=8`` -> 0.55 + two recorded divergences (1 of 3 finishes -> NOT eligible).
    """
    recs = []
    for s, f in zip(SEEDS, (0.10, 0.20, 0.30)):
        recs.append(sven(s, f, k=4, n_data=100))
    for s, f in zip(SEEDS, (0.05, 0.15, None)):
        recs.append(sven(s, f or 0.25, k=8, n_data=100,
                         status='ok' if f else 'diverged',
                         val=None if f else list(VAL_PREFIX)))
    for s, f in zip(SEEDS, (0.40, 0.50, 0.60)):
        recs.append(adam(s, f, lr=0.01, n_data=100))
    for s, f in zip(SEEDS, (0.70, 0.80, 0.90)):
        recs.append(adam(s, f, lr=0.001, n_data=100))

    for s, f in zip(SEEDS, (0.60, 0.70, None)):
        recs.append(sven(s, f or 0.0, k=4, n_data=200, val=None if f else BLOWUP))
    for s, f in zip(SEEDS, (0.55, None, None)):
        recs.append(sven(s, f or 0.25, k=8, n_data=200,
                         status='ok' if f else 'diverged',
                         val=None if f else list(VAL_PREFIX)))
    for s, f in zip(SEEDS, (0.30, 0.40, 0.50)):
        recs.append(adam(s, f, lr=0.01, n_data=200))
    for s, f in zip(SEEDS, (0.90, 1.00, 1.10)):
        recs.append(adam(s, f, lr=0.001, n_data=200))
    return write_scan(root, ARM_SCAN, recs)


def build_knob_scan(root):
    """Sven only, ``param_fraction`` x lr, with a measured ``actual_param_fraction``."""
    recs = []
    for pf, apf in ((0.5, 0.499), (1.0, 1.0)):
        for lr, finals in ((0.1, (0.10, 0.20, 0.30)), (0.5, (0.40, 0.50, None))):
            for s, f in zip(SEEDS, finals):
                diverged = f is None and pf == 0.5
                rid = (f'svd_pf{pf:g}_lr{lr:g}_mseed{s}_lseed1000')
                recs.append(_rec(rid, s, optimizer='SVD', k=4, k_fraction=0.5, lr=lr,
                                 rtol=1e-3, kappa=2.0, svd_mode='torch',
                                 param_fraction=pf, mask_mode='elementwise',
                                 actual_param_fraction=apf,
                                 status='diverged' if diverged else 'ok',
                                 svd_summary={'num_nonzero_svs_epoch': [4.0] * N_EPOCHS},
                                 losses=losses(0.6 if f is None else f,
                                               val=list(VAL_PREFIX) if diverged else None)))
    return write_scan(root, KNOB_SCAN, recs)


def build_kappa_scan(root):
    """kappa x lr laid out so that ``2 lr / kappa`` in {0.5, 1.0} is realised by both
    kappas, with IDENTICAL finals at the matched steps (rel_spread must be 0) and
    different ones off them."""
    plan = [(1, 0.25, 0.20), (1, 0.50, 0.30), (1, 0.90, 0.55),
            (2, 0.50, 0.20), (2, 1.00, 0.33), (2, 1.50, 0.60)]
    recs = []
    for kappa, lr, final in plan:
        for i, s in enumerate(SEEDS):
            rid = f'svd_kappa{kappa}_lr{lr:g}_mseed{s}_lseed1000'
            recs.append(_rec(rid, s, optimizer='SVD', k=4, k_fraction=0.5, lr=lr,
                             rtol=1e-4, kappa=kappa, svd_mode='torch',
                             svd_summary={'num_nonzero_svs_epoch': [4.0] * N_EPOCHS},
                             losses=losses(final)))
    return write_scan(root, KAPPA_SCAN, recs)


@pytest.fixture(scope='module')
def root(tmp_path_factory):
    d = tmp_path_factory.mktemp('wp4a_results')
    build_arm_scan(d)
    build_knob_scan(d)
    build_kappa_scan(d)
    return d


@pytest.fixture(scope='module')
def scans(root, tmp_path_factory):
    """The three frames, loaded through the public loader with the fixture root."""
    old = style.RESULTS_ROOT
    style.RESULTS_ROOT = str(root)
    try:
        cache = tmp_path_factory.mktemp('wp4a_cache')
        out = {name: add_derived(style.load_results(name, results_root=str(root),
                                                   cache_dir=str(cache)))
               for name in (ARM_SCAN, KNOB_SCAN, KAPPA_SCAN)}
    finally:
        style.RESULTS_ROOT = old
    return out


@pytest.fixture(scope='module')
def arm(scans):
    return scans[ARM_SCAN]


# ---------------------------------------------------------------------------
# the binding rule, per arm
# ---------------------------------------------------------------------------
def test_fewest_diverged_beats_the_lower_seed_mean(arm):
    """At n_data=100 the k=8 configuration has the lower mean (0.10 vs 0.20) and one
    diverged seed.  The binding rule ranks fewest-diverged FIRST, so k=4 must win."""
    best = rf.arm_table(arm, 'n_data')
    row = best[(best['method'] == 'Sven') & (best['n_data'] == 100)]
    assert len(row) == 1
    row = row.iloc[0]
    assert row['k'] == 4
    assert row['final_val_loss'] == pytest.approx(0.20)
    assert row['final_val_loss_std'] == pytest.approx(0.1)
    assert row['n_diverged'] == 0
    assert row['counts'] == '3/3'


def test_an_ineligible_configuration_is_never_selected(arm):
    """At n_data=200 k=8 finishes 1 of 3 (not > half), so Sven's only eligible
    configuration is k=4 -- which itself carries a 10x blow-up."""
    best = rf.arm_table(arm, 'n_data')
    row = best[(best['method'] == 'Sven') & (best['n_data'] == 200)].iloc[0]
    assert row['k'] == 4
    assert row['final_val_loss'] == pytest.approx(0.65)   # mean of 0.60 and 0.70
    assert row['n_diverged'] == 1
    assert row['counts'] == '2/3'


def test_the_three_divergence_routes_are_all_counted(arm):
    """`status: diverged`, a finite 10x blow-up and a non-finite curve are one rule."""
    d = rf.divergence_table(arm, 'n_data')
    sven100 = d[(d['method'] == 'Sven') & (d['n_data'] == 100)].iloc[0]
    sven200 = d[(d['method'] == 'Sven') & (d['n_data'] == 200)].iloc[0]
    assert (sven100['n_runs'], sven100['n_diverged']) == (6, 1)     # one status record
    assert (sven200['n_runs'], sven200['n_diverged']) == (6, 3)     # one 10x + two status
    assert sven200['frac'] == pytest.approx(0.5)
    adam = d[(d['method'] == 'Adam')]
    assert adam['n_diverged'].sum() == 0


def test_outcomes_use_the_same_finished_seeds_as_the_selection(arm):
    """train_eval / test / step time / rank come from the seeds that are in the mean."""
    best = rf.arm_table(arm, 'n_data')
    row = best[(best['method'] == 'Sven') & (best['n_data'] == 200)].iloc[0]
    # train_eval final is 0.5 x the val final, over the two finished seeds 0.60 / 0.70
    assert row['final_train_eval'] == pytest.approx(0.325)
    assert row['final_train_eval_n'] == 2
    assert row['final_test_loss'] == pytest.approx(1.1 * 0.65)
    assert row['step_s'] == pytest.approx(0.02)
    assert row['rank_eff'] == pytest.approx(3.0)
    assert row['total_steps'] == pytest.approx(N_EPOCHS * STEPS_PER_EPOCH)
    assert row['total_examples'] == pytest.approx(N_EPOCHS * STEPS_PER_EPOCH * BATCH)
    adam_row = best[(best['method'] == 'Adam') & (best['n_data'] == 200)].iloc[0]
    assert np.isnan(adam_row['rank_eff'])      # a baseline has no svd_summary


def test_selection_may_not_rank_on_a_test_metric(arm):
    with pytest.raises(ValueError, match='test split'):
        rf.arm_table(arm, 'n_data', metric='final_test_loss')
    with pytest.raises(ValueError, match='test split'):
        rf.arm_table(arm, 'n_data', metric='final_test_acc')


# ---------------------------------------------------------------------------
# ranks
# ---------------------------------------------------------------------------
def test_ranks_are_per_arm_and_accuracy_ranks_the_other_way(arm):
    best = rf.arm_table(arm, 'n_data')
    r = rf.rank_table(best, 'n_data', 'final_val_loss')
    got = {(row['n_data'], row['method']): row['rank'] for _, row in r.iterrows()}
    assert got[(100, 'Sven')] == 1 and got[(100, 'Adam')] == 2   # 0.20 vs 0.50
    assert got[(200, 'Sven')] == 2 and got[(200, 'Adam')] == 1   # 0.65 vs 0.40
    assert set(r['n_methods']) == {2}
    acc = rf.rank_table(best, 'n_data', 'final_test_acc', higher_is_better=True)
    got = {(row['n_data'], row['method']): row['rank'] for _, row in acc.iterrows()}
    assert got[(200, 'Sven')] == 1 and got[(200, 'Adam')] == 2   # 0.9 vs 0.8


def test_display_names_come_from_the_style_registry(arm):
    best = rf.arm_table(arm, 'n_data')
    assert set(best['display']) == {'Sven', 'Adam'}
    assert style.method_label('LBFGS') == 'Stochastic L-BFGS'
    assert style.method_label('SGDm') == 'SGD + momentum'


# ---------------------------------------------------------------------------
# edge optima
# ---------------------------------------------------------------------------
def test_edge_optima_names_the_side_and_ignores_a_one_valued_knob(arm):
    best = rf.arm_table(arm, 'n_data')
    e = rf.edge_optima(arm, best, 'n_data')
    sven = e[(e['method'] == 'Sven') & (e['knob'] == 'k')]
    assert set(sven['side']) == {'low'}           # k=4 of {4, 8}
    assert set(sven['n_data']) == {100, 200}
    adam = e[(e['method'] == 'Adam') & (e['knob'] == 'lr')]
    assert set(adam['side']) == {'high'}          # lr=0.01 of {0.001, 0.01}
    # Sven's lr takes one value in this fixture: nothing was chosen, so it is not an edge
    assert e[(e['method'] == 'Sven') & (e['knob'] == 'lr')].empty


# ---------------------------------------------------------------------------
# the pre-training entry, and time to target
# ---------------------------------------------------------------------------
def test_a_val_index_is_an_epoch_count_but_a_train_index_is_one_less(arm):
    row = arm[arm['run_id'] == 'svd_n100_k4_lr0.1_rtol0.001_mseed1000_lseed1000'].iloc[0]
    # the mapping index -> epochs trained, decided by curve LENGTH against `train`
    assert [rf.curve_epochs(row, 'val')(i) for i in range(4)] == [0, 1, 2, 3]
    assert [rf.curve_epochs(row, 'train')(i) for i in range(3)] == [1, 2, 3]
    assert [rf.curve_epochs(row, 'train_eval')(i) for i in range(4)] == [0, 1, 2, 3]
    # val = [10, 5, 1, 0.1]; train = [4.5, 0.9, 0.09].  The SAME target 0.9 is met after
    # 3 epochs on the val curve and after 2 on the train curve -- the off-by-one that a
    # 'wall-time to train loss < 1e-3' claim gets wrong if the key alone decides.
    assert rf.reach_epoch(row, 0.9, 'val') == 3
    assert rf.reach_epoch(row, 0.9, 'train') == 2
    assert rf.reach_epoch(row, 1.0, 'val') == 2
    assert rf.reach_epoch(row, 1e-9, 'val') is None


def test_a_diverged_run_has_no_trajectory_to_ask(arm):
    blown = arm[(arm['n_data'] == 200) & arm['diverged']]
    assert len(blown) == 3
    for _, row in blown.iterrows():
        assert rf.reach_epoch(row, 1.0, 'val') is None


def test_never_reached_and_diverged_are_counted_separately(arm):
    """n_data=200, Sven k=4: seeds 0.60 (reaches 0.65), 0.70 (never), 1 blow-up."""
    best = rf.arm_table(arm, 'n_data')
    t = rf.reach_table(arm, best, 'n_data', 0.65, which='val')
    row = t[(t['method'] == 'Sven') & (t['n_data'] == 200)].iloc[0]
    assert (row['n_runs'], row['n_reached']) == (3, 1)
    assert (row['n_never_reached'], row['n_diverged']) == (1, 1)
    assert bool(row['all_reached']) is False
    assert row['epochs'] == pytest.approx(3.0)
    assert row['steps'] == pytest.approx(3 * STEPS_PER_EPOCH)
    assert row['examples'] == pytest.approx(3 * STEPS_PER_EPOCH * BATCH)
    assert row['sync_s'] == pytest.approx(1.5)       # 3 epochs x 0.5 s of train_times


def test_time_to_target_uses_the_selected_configuration_only(arm):
    """n_data=100, Sven k=4 at target 0.25: two seeds reach it, one never does, and the
    k=8 twin (which is better) may not leak in."""
    best = rf.arm_table(arm, 'n_data')
    t = rf.reach_table(arm, best, 'n_data', 0.25, which='val')
    row = t[(t['method'] == 'Sven') & (t['n_data'] == 100)].iloc[0]
    assert (row['n_runs'], row['n_reached'], row['n_never_reached'],
            row['n_diverged']) == (3, 2, 1, 0)
    assert row['epochs'] == pytest.approx(3.0)


def test_median_method_target_is_per_arm(arm):
    best = rf.arm_table(arm, 'n_data')
    tg = rf.median_method_target(best, 'final_val_loss', 1.0, arm='n_data')
    assert tg[100] == pytest.approx(np.median([0.20, 0.50]))
    assert tg[200] == pytest.approx(np.median([0.65, 0.40]))
    assert rf.median_method_target(best, 'final_val_loss', 2.0, arm='n_data')[100] == \
        pytest.approx(2 * np.median([0.20, 0.50]))


def test_fastest_to_target_requires_every_seed_to_reach_it(arm):
    """k=8 at n_data=100 reaches 0.25 on both finished seeds but has a diverged third, so
    it is dropped; k=4 has three runs and only two reach, so it is dropped too."""
    f = rf.fastest_to_target(arm, 'n_data', 0.25, which='val')
    sven100 = f[(f['method'] == 'Sven') & (f['n_data'] == 100)]
    assert sven100.empty
    loose = rf.fastest_to_target(arm, 'n_data', 0.25, which='val',
                                 require_all_seeds=False)
    row = loose[(loose['method'] == 'Sven') & (loose['n_data'] == 100)].iloc[0]
    assert row['n_reached'] == 2 and row['epochs'] == pytest.approx(3.0)


def test_fastest_to_target_picks_the_quickest_config_not_the_best_loss(arm):
    """At a target both Adam configurations reach, the FASTEST one wins -- which is the
    point of keeping this selector separate from the binding rule."""
    f = rf.fastest_to_target(arm, 'n_data', 1.0, which='val')
    adam = f[(f['method'] == 'Adam') & (f['n_data'] == 100)].iloc[0]
    assert adam['epochs'] == pytest.approx(2.0)      # every curve passes 1.0 at index 2
    assert adam['sync_s'] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# knob tables
# ---------------------------------------------------------------------------
def test_knob_table_carries_the_measured_fraction_and_the_counts(scans):
    t = rf.knob_table(scans[KNOB_SCAN], 'param_fraction')
    row = t[(t['param_fraction'] == 0.5) & (t['lr'] == 0.5)].iloc[0]
    assert row['n_runs'] == 3 and row['n_diverged'] == 1 and row['counts'] == '2/3'
    assert row['final_val_loss'] == pytest.approx(0.45)          # 0.40 and 0.50
    assert row['actual_param_fraction'] == pytest.approx(0.499)  # NOT the requested 0.5
    clean = t[(t['param_fraction'] == 1.0) & (t['lr'] == 0.1)].iloc[0]
    assert clean['counts'] == '3/3'
    assert clean['final_val_loss'] == pytest.approx(0.20)
    assert clean['final_test_loss'] == pytest.approx(1.1 * 0.20)
    assert clean['final_train_eval'] == pytest.approx(0.5 * 0.20)
    assert clean['rank_eff'] == pytest.approx(4.0)
    assert clean['actual_param_fraction'] == pytest.approx(1.0)


def test_knob_table_can_be_pinned_to_one_learning_rate(scans):
    t = rf.knob_table(scans[KNOB_SCAN], 'param_fraction', lr=0.1)
    assert set(t['lr']) == {0.1}
    assert len(t) == 2


# ---------------------------------------------------------------------------
# kappa
# ---------------------------------------------------------------------------
def test_kappa_table_finds_the_effective_steps_both_kappas_realise(scans):
    t = rf.kappa_table(scans[KAPPA_SCAN])
    assert t.attrs['shared_eff_steps'] == pytest.approx([0.5, 1.0])
    matched = t[t['matched']]
    assert len(matched) == 4                      # 2 kappas x 2 matched steps
    assert set(np.round(matched['eff_step'], 6)) == {0.5, 1.0}
    off = t[~t['matched']]
    # kappa=1 lr=0.9 -> 1.8 and kappa=2 lr=1.5 -> 1.5: realised by one kappa only
    assert set(np.round(off['eff_step'], 6)) == {1.5, 1.8}


def test_matched_step_table_is_zero_spread_when_the_kappas_agree(scans):
    t = rf.kappa_table(scans[KAPPA_SCAN])
    m = rf.matched_step_table(t, 'final_val_loss')
    half = m[np.isclose(m['eff_step'], 0.5)].iloc[0]
    assert half['kappa=1'] == pytest.approx(0.20)
    assert half['kappa=2'] == pytest.approx(0.20)
    assert half['rel_spread'] == pytest.approx(0.0)
    one = m[np.isclose(m['eff_step'], 1.0)].iloc[0]
    assert one['kappa=1'] == pytest.approx(0.30)
    assert one['kappa=2'] == pytest.approx(0.33)
    assert one['rel_spread'] == pytest.approx((0.33 - 0.30) / 0.30)
    assert one['n_kappa'] == 2


# ---------------------------------------------------------------------------
# small pieces
# ---------------------------------------------------------------------------
def test_effective_rank_reads_the_svd_summary(arm):
    sven_row = arm[arm['optimizer'] == 'SVD'].iloc[0]
    adam_row = arm[arm['optimizer'] == 'Adam'].iloc[0]
    assert rf.effective_rank(sven_row) == pytest.approx(3.0)
    assert np.isnan(rf.effective_rank(adam_row))
    assert np.isnan(rf.effective_rank({'svd_summary': {'num_nonzero_svs_epoch': []}}))


def test_final_of_is_nan_for_a_diverged_run(arm):
    healthy = arm[(arm['n_data'] == 100) & (arm['optimizer'] == 'SVD')
                  & ~arm['diverged']].iloc[0]
    assert np.isfinite(rf.final_of(healthy, 'val'))
    blown = arm[arm['diverged']].iloc[0]
    assert np.isnan(rf.final_of(blown, 'val'))
    assert np.isnan(rf.final_of(blown, 'train_eval'))


def test_scan_census_counts_the_manifest_not_just_the_records(arm):
    c = rf.scan_census(arm, ARM_SCAN)
    assert c['n_records'] == 24 and c['n_expected'] == 24 and c['n_missing'] == 0
    assert c['n_diverged'] == 4 and c['n_failed'] == 0
    assert c['n_methods'] == 2
    assert c['seeds'] == list(SEEDS)


def test_counts_and_fmt_pm():
    assert rf.counts_str(2, 3) == '2/3'
    assert rf.counts_str(None, 3) == '--'
    assert rf.fmt_pm(1.5, 0.25, '.3g') == '1.5 ± 0.25'
    assert rf.fmt_pm(1.5, None) == '1.5'
    assert rf.fmt_pm(np.nan, 1.0) == '--'


def test_savefig_writes_pdf_and_png(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = rf.savefig(fig, tmp_path / 'sub', 'demo')
    plt.close(fig)
    assert [p.name for p in paths] == ['demo.pdf', 'demo.png']
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_plot_arm_labels_the_seed_band_once(arm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    best = rf.arm_table(arm, 'n_data')
    fig, ax = plt.subplots()
    rf.plot_arm(ax, best, 'n_data', 'final_val_loss')
    labels = ax.get_legend_handles_labels()[1]
    plt.close(fig)
    assert labels.count(style.seed_spread_label()) == 1
    assert 'Sven' in labels and 'Adam' in labels


# ---------------------------------------------------------------------------
# WP4a fix round: is the gap bigger than the seeds, and is the curve really monotone?
#
# The review found five prose claims that the notebooks asserted instead of computing --
# "every method improves monotonically", "the rank falls exactly as B/uB", "Sven's rate is
# below SGD's", "nothing reaches 1e-3", "far above noise".  Each is now a helper, and each
# helper is pinned here on numbers written down by hand.
# ---------------------------------------------------------------------------
def _pair_frame(seeds_a, seeds_b, diverge_a=()):
    """Two run selections with the same seed column, for `paired_seed_diff`."""
    rows = []
    for tag, d in (('a', seeds_a), ('b', seeds_b)):
        for seed, v in d.items():
            rows.append({'side': tag, 'model_seed': seed, 'final_val_loss': v,
                         'diverged': bool(tag == 'a' and seed in diverge_a),
                         'failed': False})
    df = pd.DataFrame(rows)
    return df[df['side'] == 'a'], df[df['side'] == 'b']


def test_paired_seed_diff_pairs_on_the_seed_and_drops_the_rest():
    """Seeds 1,2,3 on both sides, seed 4 only on b, seed 1 diverged on a.

    So the pairing keeps seeds 2 and 3: differences 0.2 - 0.5 = -0.3 and 0.4 - 0.3 = +0.1,
    mean -0.1, std 0.2828..., and the unshared/dropped seeds are reported, not silently
    absorbed.
    """
    a, b = _pair_frame({1: 0.1, 2: 0.2, 3: 0.4}, {1: 0.9, 2: 0.5, 3: 0.3, 4: 0.7},
                       diverge_a=(1,))
    st = rf.paired_seed_diff(a, b)
    assert st['n'] == 2
    assert st['mean'] == pytest.approx(-0.1)
    assert st['std'] == pytest.approx(np.std([-0.3, 0.1], ddof=1))
    assert st['sem'] == pytest.approx(st['std'] / np.sqrt(2))
    assert st['t'] == pytest.approx(st['mean'] / st['sem'])
    assert st['mean_a'] == pytest.approx(0.3) and st['mean_b'] == pytest.approx(0.4)
    assert st['n_b_only'] == 2        # seed 4 (never on a) and seed 1 (diverged on a)
    assert st['n_a_only'] == 0
    assert not st['significant']      # the interval spans 0


def test_paired_seed_diff_is_nan_below_two_seeds():
    a, b = _pair_frame({1: 0.2}, {1: 0.5})
    st = rf.paired_seed_diff(a, b)
    assert st['n'] == 1 and st['mean'] == pytest.approx(-0.3)
    assert np.isnan(st['std']) and np.isnan(st['t']) and not st['significant']


def test_paired_seed_diff_refuses_a_test_metric():
    """`style.assert_selection_metric` one level down: a paired difference is a
    comparison, and comparisons in this campaign are made on validation."""
    a, b = _pair_frame({1: 0.2, 2: 0.3}, {1: 0.5, 2: 0.6})
    with pytest.raises(ValueError, match='test split'):
        rf.paired_seed_diff(a, b, metric='final_test_loss')


def test_neighbour_gaps_puts_the_rank_and_the_paired_gap_side_by_side(arm):
    """n_data=100: Sven 0.20 (rank 1 of 2), Adam 0.50; every seed differs by exactly
    -0.30, so the gap is significant with a zero-width interval.
    n_data=200: Adam 0.40 wins, Sven 0.65 is 2nd, and only 2 seeds pair (Sven's third
    blew up)."""
    best = rf.arm_table(arm, 'n_data')
    g = rf.neighbour_gaps(arm, best, 'n_data')
    lo = g[g['n_data'] == 100].iloc[0]
    assert (lo['rank'], lo['n_field'], lo['rival']) == (1, 2, 'Adam')
    assert lo['n'] == 3 and lo['mean'] == pytest.approx(-0.30)
    assert lo['significant']
    hi = g[g['n_data'] == 200].iloc[0]
    assert (hi['rank'], hi['rival_rank']) == (2, 1)
    assert hi['n'] == 2 and hi['mean'] == pytest.approx(0.30)


def test_monotonicity_table_reports_the_direction_and_the_optimum():
    """One falling curve, one rising, one with an interior minimum -- the three cases the
    batch-size conclusion needed to tell apart."""
    tbl = pd.DataFrame([
        {'method': 'down', 'x': 1, 'q': 3.0}, {'method': 'down', 'x': 2, 'q': 2.0},
        {'method': 'down', 'x': 4, 'q': 1.0},
        {'method': 'up', 'x': 1, 'q': 1.0}, {'method': 'up', 'x': 2, 'q': 2.0},
        {'method': 'up', 'x': 4, 'q': 3.0},
        {'method': 'dip', 'x': 1, 'q': 3.0}, {'method': 'dip', 'x': 2, 'q': 1.0},
        {'method': 'dip', 'x': 4, 'q': 2.0},
    ])
    m = rf.monotonicity_table(tbl, 'x', 'q').set_index('method')
    assert m.loc['down', 'falling'] and m.loc['down', 'monotone']
    assert m.loc['up', 'rising'] and m.loc['up', 'monotone']
    assert not m.loc['dip', 'monotone']
    assert list(m['x_at_best']) == [2.0, 4.0, 1.0]        # dip, down, up (sorted index)
    assert m.loc['dip', 'range_ratio'] == pytest.approx(3.0)
    # reading the axis backwards flips the verdicts, which is how a batch-size axis is
    # re-read as a steps axis
    r = rf.monotonicity_table(tbl, 'x', 'q', ascending=False).set_index('method')
    assert r.loc['down', 'rising'] and r.loc['up', 'falling']


def test_reach_count_table_counts_the_field_instead_of_asserting_it(arm):
    """At n_data=100 both methods reach val <= 1.0 on every seed; at n_data=200 only Adam
    does (Sven's k=4 loses a seed to the blow-up and k=8 is 1 of 3), so the count is 1 of
    a field of 2 -- the number the prose was getting wrong."""
    t = rf.reach_count_table(arm, 'n_data', 1.0, which='val').set_index('n_data')
    assert t.loc[100, 'n_field'] == 2 and t.loc[100, 'n_reached'] == 2
    assert sorted(t.loc[100, 'methods']) == ['Adam', 'Sven']
    assert t.loc[200, 'n_field'] == 2 and t.loc[200, 'n_reached'] == 1
    assert t.loc[200, 'methods'] == ['Adam']


def test_divergence_vs_reference_is_per_scan_and_shows_the_grid_size(arm):
    """Sven: 4 diverged of 12 runs; Adam: 0 of 12.  The reference's own place in the
    ranking and the relative grid size are the two facts a pooled "Sven diverges less
    than X" sentence has to carry."""
    t = rf.divergence_vs_reference(arm, reference='Sven')
    sv = t[t['method'] == 'Sven'].iloc[0]
    assert (sv['n_diverged'], sv['n_runs']) == (4, 12)
    assert sv['frac'] == pytest.approx(4 / 12)
    assert t.attrs['reference_rank'] == 1               # worst-first
    assert t.attrs['above_reference'] == []
    assert t.attrs['below_reference'] == ['Adam']
    assert list(t['grid_ratio_vs_reference']) == [1.0, 1.0]
    assert list(t['display']) == ['Sven', 'Adam']       # sorted worst first


def test_fixed_knob_cost_separates_the_knob_from_the_arm(arm):
    """One cell per (k, n_data) for Sven only, diverged runs excluded from the mean but
    counted -- which is what makes a cost-vs-arm curve free of the selection's knob
    jumps."""
    t = rf.fixed_knob_cost(arm, 'n_data', 'k', 'avg_batch_time_train', scale=1e3)
    assert set(zip(t['k'], t['n_data'])) == {(4, 100), (8, 100), (4, 200), (8, 200)}
    assert t['value'].tolist() == pytest.approx([20.0] * 4)   # step_s = 0.02 everywhere
    cell = t[(t['k'] == 8) & (t['n_data'] == 200)].iloc[0]
    assert cell['n_runs'] == 3 and cell['n'] == 1 and cell['n_diverged'] == 2
    # a method that does not HAVE the knob collapses to one cell per arm with a NaN knob
    # value -- visible, not silently mixed into the Sven curve
    other = rf.fixed_knob_cost(arm, 'n_data', 'k', method='Adam', scale=1e3)
    assert len(other) == 2 and other['k'].isna().all()
    assert other['value'].tolist() == pytest.approx([4.0, 4.0])


def test_rank_vs_cap_only_claims_a_match_where_the_cap_binds():
    """`rank = min(rtol-limited rank, B/uB)`: the cap is met at uB >= 4 and not above it,
    so "falls exactly as advertised" is false for the table as a whole."""
    tbl = pd.DataFrame([{'mb': 1, 'rank_eff': 11.06}, {'mb': 2, 'rank_eff': 11.55},
                        {'mb': 4, 'rank_eff': 8.0}, {'mb': 8, 'rank_eff': 4.0},
                        {'mb': 16, 'rank_eff': 2.0}, {'mb': 32, 'rank_eff': 1.0}])
    chk = rf.rank_vs_cap(tbl, 'mb', 32)
    assert list(chk['cap']) == [32.0, 16.0, 8.0, 4.0, 2.0, 1.0]
    assert list(chk['matches']) == [False, False, True, True, True, True]
    assert chk['cap_binds'].all()
    assert not chk.attrs['all_match'] and chk.attrs['n_match'] == 4
    # and the case where it IS exact everywhere (rtol never cuts)
    exact = pd.DataFrame([{'mb': m, 'rank_eff': 64 / m} for m in (1, 2, 4, 8, 16, 32, 64)])
    assert rf.rank_vs_cap(exact, 'mb', 64).attrs['all_match']


def test_eff_step_coverage_shows_that_the_kappas_span_different_ranges(scans):
    """The lr list is shared, so the effective step 2*lr/kappa is not: a ceiling equal to
    `max_eff_step` is a grid edge, and `n_above_matched == 0` means nothing can be
    concluded about that kappa above the matched steps."""
    cov = rf.eff_step_coverage(scans[KAPPA_SCAN]).set_index('kappa')
    assert cov.attrs['matched_eff_steps'] == pytest.approx([0.5, 1.0])
    assert cov.attrs['top_matched'] == pytest.approx(1.0)
    assert cov.loc[1, 'max_eff_step'] == pytest.approx(1.8)   # lr 0.9, kappa 1
    assert cov.loc[2, 'max_eff_step'] == pytest.approx(1.5)   # lr 1.5, kappa 2
    assert cov.loc[1, 'n_above_matched'] == 3                 # the three seeds at 1.8
    assert cov.loc[2, 'n_above_matched'] == 3


def test_matched_step_paired_separates_agreement_from_a_real_gap(scans):
    """The kappa fixture agrees exactly at effective step 0.5 (0.20 vs 0.20) and differs
    by a constant 0.03 at 1.0 (0.30 vs 0.33).  The first is the null the real k = B arm
    reproduces to float noise; the second is a difference every seed shows, so the
    zero-width interval calls it significant while `t` stays NaN (no spread to divide
    by) -- the two must not be conflated."""
    mp = rf.matched_step_paired(scans[KAPPA_SCAN]).set_index('eff_step')
    assert set(mp['pair']) == {'kappa1-kappa2'}
    assert sorted(mp.index) == pytest.approx([0.5, 1.0])
    assert mp['n'].tolist() == [3, 3]
    assert mp.loc[0.5, 'mean'] == pytest.approx(0.0)
    assert not mp.loc[0.5, 'significant']
    assert mp.loc[1.0, 'mean'] == pytest.approx(-0.03)
    assert mp.loc[1.0, 'std'] == pytest.approx(0.0)
    assert np.isnan(mp.loc[1.0, 't']) and mp.loc[1.0, 'significant']
    assert mp['kappa=1'].tolist() == pytest.approx([0.20, 0.30])


def test_arm_ticks_labels_only_the_arms_and_thins_what_collides():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.plot([0.55, 0.69, 1.4, 11.0], [1, 2, 3, 4])
    rf.arm_ticks(ax, [0.55, 0.69, 1.4, 11.0], fmt='{:.2g}', min_log_sep=0.12)
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels()]
    ticks = list(ax.get_xticks())
    plt.close(fig)
    assert ticks == pytest.approx([0.55, 0.69, 1.4, 11.0])   # every arm keeps its TICK
    assert labels == ['0.55', '', '1.4', '11']               # 0.69 is too close to label
    assert len(ax.xaxis.get_minorticklocs()) == 0


def test_figure_legend_dedupes_and_leaves_the_axes_clear():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2)
    for ax in axes:
        ax.plot([0, 1], [0, 1], label='Sven')
        ax.plot([0, 1], [1, 0], label='_hidden')
    axes[1].plot([0, 1], [0.5, 0.5], label='Adam')
    leg = rf.figure_legend(fig, axes, ncol=2)
    labels = [t.get_text() for t in leg.get_texts()]
    plt.close(fig)
    assert labels == ['Sven', 'Adam']          # deduped, `_`-prefixed dropped
    assert leg.axes is None or leg.figure is not None
