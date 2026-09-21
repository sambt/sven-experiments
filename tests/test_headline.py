"""CPU tests for `analysis/headline.py` -- the paper's headline tables (WP2).

Everything runs against a SYNTHETIC campaign built in a tmp root and read through
``$SV3_RESULTS_ROOT``: a tuning scan, its ``_confirm`` pass (5 fresh seeds, here 3, on 2
data seeds) and its ``_timing`` pass, plus a hand-written schema-2 selection file.  No
test reads ``experiment_results/``, so none of them can be made to pass by the real data
happening to agree.

Every number the fixture produces is chosen so the expected answer can be written down by
hand -- means, stds, finished/attempted, the pairs, the epochs to target -- which is the
point: these tests pin the ARITHMETIC of the confirmation table, not its plumbing.

The properties under test, in the order the plan lists them:

* the confirmation table's means / stds / gaps, and the data-seed decomposition (the
  between-instance spread is NOT the seed band);
* diverged = failed: the three ways a run fails (``status: diverged``, the 10x rule, an
  ``oom`` record) are counted, kept out of every mean, and distinguished from each other;
* a selected configuration is matched by hyperparameter VALUES, so a neighbouring grid
  point (a different ``k``, a microbatched twin) can never enter its row;
* pairing is by ``(model seed, data seed)`` and an unpaired seed is reported, not imputed;
* time-to-target: pre-declared targets, the epoch index of a pre-training curve, and the
  runs that never reach it;
* selection never reads a test metric -- checked as an assertion that FAILS when a test
  key is put into the selection file.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'analysis'))   # the notebooks' own sys.path.insert(0, '.')

import headline as hl            # noqa: E402
import style                     # noqa: E402

SCAN = 'toy_fixture_scan'
N_EPOCHS = 4
STEPS_PER_EPOCH = 10
BATCH_SIZE = 8
#: every val curve descends through the same values before its final one, so the epoch at
#: which a target is met is the same for every run and can be written down
VAL_PREFIX = (10.0, 1.0, 0.5, 0.2)

TUNE_SEEDS = (1000, 1001, 1002)
CONF_SEEDS = (1100, 1101, 1102)
DATA_SEEDS = (1000, 1001)


# ---------------------------------------------------------------------------
# The synthetic campaign
# ---------------------------------------------------------------------------
def losses(final, *, epoch_time=1.0, train_time=0.5, mem=100.0, val=None,
           test_acc=None, train_eval=None):
    """A record's ``losses``: ``train`` per epoch, ``val`` one longer (index 0 is the
    UNTRAINED model), ``test`` beside it, and the timing series the efficiency table
    reads.  ``val`` overrides the curve for the divergence fixtures."""
    curve = list(VAL_PREFIX) + [final] if val is None else list(val)
    out = {
        'train': [0.9 * v for v in curve[1:]],
        'val': curve,
        'test': [1.1 * v for v in curve],
        'epoch_times': [epoch_time] * N_EPOCHS,
        'train_times': [train_time] * N_EPOCHS,
        'total_time': epoch_time * N_EPOCHS,
        'avg_epoch_time': epoch_time,
        'avg_train_time': train_time,
        'peak_gpu_mem_mb': mem,
    }
    if test_acc is not None:
        out['test_acc'] = [0.1] + [test_acc] * (len(curve) - 1)
    if train_eval is not None:
        out['train_eval'] = [5.0] + [train_eval] * (len(curve) - 1)
    return out


def _common(run_id, seed, data_seed, **extra):
    rec = {'run_id': run_id, 'batch_size': BATCH_SIZE, 'model_seed': seed,
           'loader_seed': 1000, 'model_width': None, 'mlp_width': 16,
           'num_epochs': N_EPOCHS, 'steps_per_epoch': STEPS_PER_EPOCH,
           'status': 'ok', 'schema_version': 2, 'diag_file': None,
           'svd_summary': None, 'loss': 'mse'}
    if data_seed is not None:
        rec['data_seed'] = data_seed
    rec.update(extra)
    return rec


def sven(seed, final, *, k=4, lr=0.1, rtol=1e-3, data_seed=None, microbatch=None,
         status='ok', **loss_kw):
    ds = '' if data_seed is None else f'_data_seed{data_seed}'
    mb = '' if microbatch is None else f'_mb{microbatch}'
    run_id = f'svd_bs8_mlp_width16{ds}_k{k}_lr{lr:g}_rtol{rtol:g}{mb}_mseed{seed}_lseed1000'
    return _common(run_id, seed, data_seed, optimizer='SVD', k=k, k_fraction=k / 8,
                   lr=lr, rtol=rtol, kappa=2.0, svd_mode='torch', status=status,
                   microbatch_size=microbatch, param_fraction=None, mask_mode=None,
                   losses=losses(final, **loss_kw))


def adam(seed, final, *, lr=0.01, data_seed=None, status='ok', **loss_kw):
    ds = '' if data_seed is None else f'_data_seed{data_seed}'
    run_id = f'std_bs8_mlp_width16{ds}_lr{lr:g}_optimAdam_mseed{seed}_lseed1000'
    return _common(run_id, seed, data_seed, optimizer='Adam', lr=lr, weight_decay=0.0,
                   k=None, rtol=None, status=status,
                   losses=losses(final, **loss_kw))


def write_scan(root, name, records, manifest=True):
    """The real layout: ``{root}/{name}/{run_id}.jsonl`` + ``{name}/manifest/job0.json``."""
    d = Path(root) / name
    (d / 'manifest').mkdir(parents=True, exist_ok=True)
    for rec in records:
        (d / f"{rec['run_id']}.jsonl").write_text(json.dumps(rec) + '\n')
    if manifest:
        ids = [r['run_id'] for r in records]
        (d / 'manifest' / 'job0.json').write_text(
            json.dumps({'job': 'job0', 'n_runs': len(ids), 'run_ids': ids}))
    return d


#: Sven's tuning finals -> mean 0.2, std(ddof=1) 0.1
TUNE_SVEN = (0.10, 0.20, 0.30)
#: Adam's tuning finals; the third seed is an `oom` record (FAILED, not diverged), so the
#: mean is over two seeds: 0.45
TUNE_ADAM = (0.40, 0.50, None)
#: the confirmation finals, per data seed: Sven's two instances have means 0.21 / 0.29
CONF_SVEN = {1000: (0.11, 0.21, 0.31), 1001: (0.19, 0.29, 0.39)}
#: Adam loses one run per instance: `status: diverged` on the first, the 10x rule on the
#: second, so 4 of 6 finish and the mean is 0.5
CONF_ADAM = {1000: (0.42, 0.52, 'status'), 1001: (0.48, 0.58, 'blowup')}


def build(root):
    """Write the three passes and return the selection payload that names them."""
    tune = []
    for seed, final in zip(TUNE_SEEDS, TUNE_SVEN):
        tune.append(sven(seed, final, train_eval=0.5 * final, test_acc=0.9))
        # a k = 8 twin with the IDENTICAL trajectory: a larger k that truncates nothing
        # extra is the same run, which is what the budget disclosure measures
        tune.append(sven(seed, final, k=8, train_eval=0.5 * final, test_acc=0.9))
    for seed, final in zip(TUNE_SEEDS, TUNE_ADAM):
        if final is None:
            tune.append(adam(seed, 0.6, status='oom', train_eval=0.3, test_acc=0.8))
        else:
            tune.append(adam(seed, final, train_eval=0.5 * final, test_acc=0.8))
    write_scan(root, SCAN, tune)

    conf = []
    for ds in DATA_SEEDS:
        for seed, final in zip(CONF_SEEDS, CONF_SVEN[ds]):
            conf.append(sven(seed, final, data_seed=ds, train_eval=0.5 * final,
                             test_acc=0.9))
        # a decoy: the same k/lr/rtol but MICROBATCHED, and much better.  Matching on the
        # selected `microbatch_size=None` must keep it out of Sven's row.
        conf.append(sven(CONF_SEEDS[0], 0.001, data_seed=ds, microbatch=4,
                         train_eval=0.001, test_acc=0.99))
        for seed, final in zip(CONF_SEEDS, CONF_ADAM[ds]):
            if final == 'status':
                conf.append(adam(seed, 0.5, data_seed=ds, status='diverged',
                                 val=[10.0, 1.0, 0.5], train_eval=0.3, test_acc=0.8))
            elif final == 'blowup':
                conf.append(adam(seed, 200.0, data_seed=ds, train_eval=0.3,
                                 test_acc=0.1))
            else:
                conf.append(adam(seed, final, data_seed=ds, train_eval=0.5 * final,
                                 test_acc=0.8))
    write_scan(root, f'{SCAN}_confirm', conf)

    # the timing pass: the scan's own run_ids, alone on a GPU -- 2 s an epoch for Sven,
    # 1 s for Adam, so a time-to-target second is a number the test can predict
    timing = []
    for seed, final in zip(TUNE_SEEDS, TUNE_SVEN):
        timing.append(sven(seed, final, epoch_time=2.0, train_time=1.6, mem=500.0))
    for i, (seed, final) in enumerate(zip(TUNE_SEEDS, TUNE_ADAM)):
        # the first Adam standalone run ends 10% off its scan twin: the passes partly ran
        # on a different GPU type and several optimizers are not bit-reproducible there
        final = 0.6 if final is None else final * (1.1 if i == 0 else 1.0)
        timing.append(adam(seed, final, epoch_time=1.0, train_time=0.8, mem=120.0))
    write_scan(root, f'{SCAN}_timing', timing)

    sven_ids = sorted(r['run_id'] for r in tune
                      if r['optimizer'] == 'SVD' and r['k'] == 4)
    adam_ids = sorted(r['run_id'] for r in tune if r['optimizer'] == 'Adam')
    payload = {
        'schema': 2, 'generated_at': '2026-09-20T00:00:00+0000',
        'generated_by': 'tests/test_headline.py', 'results_root': str(root),
        'rule_name': 'full', 'rule': 'eligible -> fewest diverged -> seed-mean final VAL',
        'scans': {SCAN: {'groups': ['mode=all'], 'scan_dir': str(Path(root) / SCAN),
                         'n_expected': len(tune), 'n_records': len(tune),
                         'n_missing': 0, 'warnings': [], 'errors': [],
                         'reconcile_disagreements': [], 'methods': {
            'SVD': {'method': 'SVD', 'label': 'Sven', 'family': 'svd',
                    'config_key': 'svd_bs8_mlp_width16_k4_lr0.1_rtol0.001',
                    'overrides': 'mode=svd k_values=[4] lrs=[0.1] rtol=[0.001]',
                    'verified': True, 'verify_error': None,
                    'seed_mean_final_val': float(np.mean(TUNE_SVEN)),
                    'n_ok': 3, 'n_expected': 3, 'n_diverged': 0,
                    'batch_size': BATCH_SIZE,
                    'hparams': {'k': 4, 'lr': 0.1, 'rtol': 0.001, 'svd_mode': 'torch',
                                'microbatch_size': None, 'param_fraction': None,
                                'kappa': 2.0},
                    'model_seeds': list(TUNE_SEEDS), 'run_ids': sven_ids},
            'Adam': {'method': 'Adam', 'label': 'Adam', 'family': 'standard',
                     'config_key': 'std_bs8_mlp_width16_lr0.01_optimAdam',
                     'overrides': 'mode=standard optimizers_standard=[Adam] lrs_standard=[0.01]',
                     'verified': True, 'verify_error': None,
                     'seed_mean_final_val': 0.45,
                     'n_ok': 2, 'n_expected': 3, 'n_diverged': 0,
                     'batch_size': BATCH_SIZE,
                     'hparams': {'optim_name': 'Adam', 'lr': 0.01, 'weight_decay': 0.0},
                     'model_seeds': list(TUNE_SEEDS), 'run_ids': adam_ids},
        }}},
    }
    path = Path(root) / 'best_configs.json'
    path.write_text(json.dumps(payload))
    return payload, path


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """``(root, selection path)`` with ``$SV3_RESULTS_ROOT`` pointing at the fixture."""
    root = tmp_path / 'experiment_results'
    root.mkdir()
    monkeypatch.setenv(style.RESULTS_ROOT_ENV, str(root))
    _, path = build(root)
    monkeypatch.setattr(hl, 'SELECTION_PATH', path)
    monkeypatch.setattr(hl, 'HEADLINE_SCANS', (SCAN,))
    hl.clear_cache()
    yield root, path
    hl.clear_cache()


def row_of(tbl, method):
    sub = tbl[tbl['method'] == method]
    assert len(sub) == 1, f'{method}: {len(sub)} rows'
    return sub.iloc[0]


# ===========================================================================
# 1. The confirmation table's arithmetic
# ===========================================================================
def test_confirmation_table_means_and_stds(campaign):
    """The seed mean, the +/- 1 std (ddof=1) band and the selection-optimism gap, all
    written down by hand from the fixture's finals."""
    tbl = hl.confirmation_table(SCAN)
    sven = row_of(tbl, 'Sven')

    pooled = [v for ds in DATA_SEEDS for v in CONF_SVEN[ds]]
    assert sven['val_conf'] == pytest.approx(np.mean(pooled))          # 0.25
    assert sven['val_conf_std'] == pytest.approx(np.std(pooled, ddof=1))
    assert sven['val_tune'] == pytest.approx(0.2)
    assert sven['val_tune_std'] == pytest.approx(0.1)
    # the gap IS confirmation minus tuning, and it is positive here: the tuning seeds
    # flattered the configuration
    assert sven['gap_val'] == pytest.approx(0.05)
    assert sven['gap_val_rel'] == pytest.approx(0.25)
    # ... and on the instance the tuning ran on it is much smaller: 0.21 - 0.2
    assert sven['val_conf_ds0'] == pytest.approx(0.21)
    assert sven['gap_val_same_instance'] == pytest.approx(0.01, abs=1e-12)

    # test outcomes are the 1.1x curve of the fixture -- reported, never selected on
    assert sven['test_conf'] == pytest.approx(1.1 * np.mean(pooled))
    assert sven['acc_conf'] == pytest.approx(0.9)
    assert sven['treval_conf'] == pytest.approx(0.5 * np.mean(pooled))

    # rows are ranked by the confirmation validation loss: Sven (0.25) before Adam (0.5)
    assert list(tbl['method']) == ['Sven', 'Adam']


def test_data_seed_spread_is_not_the_seed_band(campaign):
    """The three quantities the fixture separates on purpose: the pooled std mixes the
    seeds and the instances, the between-instance spread is the std of the per-instance
    MEANS, and the within-instance seed spread is the mean of the per-instance stds."""
    sven = row_of(hl.confirmation_table(SCAN), 'Sven')
    per_ds = [np.mean(CONF_SVEN[ds]) for ds in DATA_SEEDS]            # 0.21, 0.29
    within = [np.std(CONF_SVEN[ds], ddof=1) for ds in DATA_SEEDS]     # 0.1, 0.1

    assert sven['n_data_seeds'] == 2
    assert sven['val_conf_dsmean'] == pytest.approx(np.mean(per_ds))
    assert sven['val_conf_dsspread'] == pytest.approx(np.std(per_ds, ddof=1))
    assert sven['val_conf_seedspread'] == pytest.approx(np.mean(within))
    # the between-instance number comes from the two instance MEANS -- three numbers that
    # cannot be substituted for one another, which is why they get three columns
    assert sven['val_conf_dsspread'] == pytest.approx(abs(0.29 - 0.21) / np.sqrt(2))
    assert sven['val_conf_dsspread'] != pytest.approx(sven['val_conf_std'])
    assert sven['val_conf_seedspread'] != pytest.approx(sven['val_conf_std'])

    ds_tbl = hl.data_seed_table(SCAN)
    assert len(ds_tbl) == 4                                            # 2 methods x 2 seeds
    sven_rows = ds_tbl[ds_tbl['method'] == 'Sven'].sort_values('data_seed')
    assert list(sven_rows['val']) == pytest.approx(per_ds)
    assert list(sven_rows['finished']) == [3, 3]


def test_a_scan_without_data_seeds_reports_no_instance_spread(campaign, tmp_path,
                                                              monkeypatch):
    """Where a scan has one data set, the between-instance columns are NaN and the
    same-instance gap collapses onto the plain gap -- no silent 1-sample 'spread'."""
    root = tmp_path / 'plain'
    root.mkdir()
    monkeypatch.setenv(style.RESULTS_ROOT_ENV, str(root))
    hl.clear_cache()
    write_scan(root, SCAN, [sven(s, f) for s, f in zip(TUNE_SEEDS, TUNE_SVEN)])
    write_scan(root, f'{SCAN}_confirm',
               [sven(s, f) for s, f in zip(CONF_SEEDS, CONF_SVEN[1000])])
    payload = json.loads(Path(campaign[1]).read_text())
    del payload['scans'][SCAN]['methods']['Adam']
    p = root / 'sel.json'
    p.write_text(json.dumps(payload))
    monkeypatch.setattr(hl, 'SELECTION_PATH', p)

    sven_row = row_of(hl.confirmation_table(SCAN), 'Sven')
    assert sven_row['n_data_seeds'] == 0
    assert np.isnan(sven_row['val_conf_dsspread'])
    assert sven_row['gap_val_same_instance'] == pytest.approx(sven_row['gap_val'])
    assert hl.data_seed_table(SCAN).empty


# ===========================================================================
# 2. Diverged = failed
# ===========================================================================
def test_diverged_and_failed_runs_are_counted_and_never_averaged(campaign):
    """The three failure modes of the fixture, each in its own counter.

    ``status: diverged`` (a finite partial curve), the 10x rule (a finite blow-up with
    ``status: ok``) and an ``oom`` record.  None of them may reach a mean, all of them
    must reach ``finished / attempted``, and an ``oom`` -- an incomplete attempt the
    runner retries -- must NOT be counted as a divergence.
    """
    adam_row = row_of(hl.confirmation_table(SCAN), 'Adam')
    survivors = [v for ds in DATA_SEEDS for v in CONF_ADAM[ds]
                 if isinstance(v, float)]
    assert adam_row['val_conf'] == pytest.approx(np.mean(survivors))   # 0.5, not 33.6
    assert adam_row['fin_conf'] == 4 and adam_row['att_conf'] == 6
    assert adam_row['div_conf'] == 2 and adam_row['fail_conf'] == 0
    assert bool(adam_row['elig_conf']) is True                         # 4 > 6/2

    # the tuning side: an `oom` record is failed, not diverged
    assert adam_row['val_tune'] == pytest.approx(0.45)
    assert adam_row['fin_tune'] == 2 and adam_row['att_tune'] == 3
    assert adam_row['div_tune'] == 0 and adam_row['fail_tune'] == 1

    # the blow-up really is a blow-up, i.e. the fixture exercises the 10x rule and not
    # some other branch: same finite train curve, same status, only the val end differs
    train = [0.9, 0.5, 0.3, 0.2]
    assert style.is_diverged(train, list(VAL_PREFIX) + [200.0], 'ok') is True
    assert style.is_diverged(train, list(VAL_PREFIX) + [0.42], 'ok') is False


def test_a_configuration_that_fails_most_confirmation_runs_is_flagged(campaign,
                                                                      tmp_path,
                                                                      monkeypatch):
    """Eligibility is re-asked on the confirmation seeds: a configuration that survived
    its tuning seeds and then blows up on most fresh ones is marked, because the mean
    beside it is a mean over the survivors (polynomial L-BFGS: 3/5, then 2/15)."""
    root = tmp_path / 'fragile'
    root.mkdir()
    monkeypatch.setenv(style.RESULTS_ROOT_ENV, str(root))
    hl.clear_cache()
    write_scan(root, SCAN, [sven(s, f) for s, f in zip(TUNE_SEEDS, TUNE_SVEN)])
    write_scan(root, f'{SCAN}_confirm',
               [sven(CONF_SEEDS[0], 0.11),
                sven(CONF_SEEDS[1], 200.0),
                sven(CONF_SEEDS[2], 0.5, status='diverged')])
    payload = json.loads(Path(campaign[1]).read_text())
    del payload['scans'][SCAN]['methods']['Adam']
    p = root / 'sel.json'
    p.write_text(json.dumps(payload))
    monkeypatch.setattr(hl, 'SELECTION_PATH', p)

    tbl = hl.confirmation_table(SCAN)
    sven_row = row_of(tbl, 'Sven')
    assert sven_row['fin_conf'] == 1 and sven_row['att_conf'] == 3
    assert sven_row['div_conf'] == 2
    assert bool(sven_row['elig_conf']) is False
    assert bool(sven_row['elig_tune']) is True
    assert '(not eligible)' in hl.confirmation_view(tbl).iloc[0]['finished/attempted']


# ===========================================================================
# 3. Matching a selected configuration
# ===========================================================================
def test_only_the_selected_configuration_enters_its_row(campaign):
    """Matching is on hyperparameter VALUES, and a selected ``None`` means the column must
    be absent: the microbatched decoy (better by two orders of magnitude) and the k = 8
    twin are both excluded, and the row lands on exactly one ``config_key``."""
    conf = hl.load(SCAN, 'confirm')
    sel = hl.selection_methods(SCAN)['SVD']
    rows, report = hl.selected_runs(conf, sel)

    assert len(rows) == 6                                  # 3 seeds x 2 data seeds
    assert report['unusable'] == []
    assert not any('_mb4_' in r for r in rows['run_id'])
    assert not any('_k8_' in r for r in rows['run_id'])
    assert 'microbatch_size' in report['columns_used']
    # one configuration per data seed, and nothing else
    assert len(report['config_keys']) == len(DATA_SEEDS)
    assert all('_k4_' in key for key in report['config_keys'])

    # the k = 8 twin is in the TUNING grid and must not be in Sven's tuning row either
    tune_rows, _ = hl.selected_runs(hl.load(SCAN), sel)
    assert len(tune_rows) == 3
    assert set(tune_rows['run_id']) == set(sel['run_ids'])


def test_selection_scores_reproduce_from_the_records(campaign):
    """``bench/best_configs.json`` is written by a torch-free tool reading the JSONL;
    recomputing its score through the analysis layer must give the same number, or the
    two definitions of 'diverged' / 'the final value' have drifted apart."""
    chk = hl.check_selection_reproduces(SCAN)
    assert chk.attrs['ok'], chk
    assert float(np.nanmax(chk['abs_diff'])) < 1e-12
    assert list(chk['n_records']) == [3, 3]


def test_an_alias_gap_is_reported_not_silently_ignored(campaign):
    """A grid name the record layer spells differently is matched through
    :data:`headline.HPARAM_ALIAS`; one that reaches no column at all is REPORTED, because
    a silently skipped hyperparameter is how a table comes to describe a neighbour."""
    sel = dict(hl.selection_methods(SCAN)['Adam'])
    sel['hparams'] = dict(sel['hparams'], not_a_column=7)
    rows, report = hl.selected_runs(hl.load(SCAN), sel)
    assert len(rows) == 3
    assert report['unusable'] == ['not_a_column=7 (column absent)']
    # the alias itself: `optim_name` in the grid is `optimizer` on a record
    assert hl.record_hparams(sel)['optimizer'] == 'Adam'
    assert 'optim_name' not in hl.record_hparams(sel)


# ===========================================================================
# 4. Pairing
# ===========================================================================
def test_pairing_is_by_model_seed_and_data_seed(campaign):
    """A pair is one problem and one initialisation, so the differences are the fixture's
    per-(seed, instance) differences -- and the two runs Adam lost leave Sven with two
    unpaired seeds, which are reported rather than imputed."""
    tbl = hl.paired_vs_sven(SCAN)
    assert list(tbl['method']) == ['Adam']
    r = tbl.iloc[0]

    diffs = [s - a for ds in DATA_SEEDS
             for s, a in zip(CONF_SVEN[ds], CONF_ADAM[ds]) if isinstance(a, float)]
    assert r['n'] == len(diffs) == 4
    assert r['mean'] == pytest.approx(np.mean(diffs))      # -0.30: Sven is better
    assert r['std'] == pytest.approx(np.std(diffs, ddof=1))
    assert r['mean'] < 0 and bool(r['significant'])
    assert r['n_unpaired'] == 2                            # the diverged + the blow-up
    assert r['a_better'] == 4
    assert tbl.attrs['pair_on'] == 'model_seed x data_seed'

    # the paired mean is NOT the difference of the two reported means here: the pairs
    # exclude the instances/seeds one side lost
    conf = hl.confirmation_table(SCAN)
    unpaired_gap = row_of(conf, 'Sven')['val_conf'] - row_of(conf, 'Adam')['val_conf']
    assert unpaired_gap != pytest.approx(r['mean'])


def test_the_win_count_follows_the_METRIC_direction_not_paired_pys_loss_convention(campaign):
    """An accuracy is better when it is LARGER, so a paired accuracy table may not reuse
    :mod:`paired`'s loss convention (``a_better = (d < 0)``).

    The fixture makes the two directions disagree on purpose: Sven has both the lower loss
    and the higher accuracy, so ``sven_better`` must be 4 in BOTH tables -- a count taken
    with the wrong sign comes back 0 on exactly one of them.  (On the real
    ``cifar10_resnet_ce_scan`` the loss convention turned "Sven's accuracy lost all five
    seeds to Muon, 0.53 vs 0.78" into ``a_better = 5``.)
    """
    loss = hl.paired_outcome_vs_sven(SCAN, metric='final_test_loss').iloc[0]
    acc = hl.paired_outcome_vs_sven(SCAN, metric='final_test_acc').iloc[0]

    assert hl.higher_is_better('final_test_acc') is True
    assert hl.higher_is_better('final_test_loss') is False
    assert hl.higher_is_better('final_val_loss') is False

    assert loss['mean'] < 0 and loss['sven_better'] == 4      # lower loss on all 4 pairs
    assert acc['mean'] == pytest.approx(0.1) and acc['sven_better'] == 4
    accs = hl.paired_outcome_vs_sven(SCAN, metric='final_test_acc')
    assert accs.attrs['higher_is_better'] is True
    assert 'POSITIVE means Sven is better' in accs.attrs['sign']
    assert 'NEGATIVE means Sven is better' in hl.paired_outcome_vs_sven(
        SCAN, metric='final_test_loss').attrs['sign']
    # the val-loss table spells the same count the same way, so the two read together
    assert hl.paired_vs_sven(SCAN).iloc[0]['sven_better'] == 4


def test_a_metric_on_which_sven_loses_counts_zero_wins(campaign, tmp_path, monkeypatch):
    """The other direction of the same bug: Sven keeps the lower loss but now has the
    WORSE accuracy, so ``sven_better`` must be 0 on the accuracy table and 3 on the loss
    table -- an unsigned count could not tell the two apart."""
    root = tmp_path / 'loses_acc'
    root.mkdir()
    monkeypatch.setenv(style.RESULTS_ROOT_ENV, str(root))
    hl.clear_cache()
    write_scan(root, SCAN, [sven(s, f, test_acc=0.70) for s, f in zip(TUNE_SEEDS, TUNE_SVEN)]
               + [adam(s, f, test_acc=0.85) for s, f in zip(TUNE_SEEDS, (0.4, 0.5, 0.6))])
    write_scan(root, f'{SCAN}_confirm',
               [sven(s, f, test_acc=0.70) for s, f in zip(CONF_SEEDS, CONF_SVEN[1000])]
               + [adam(s, f, test_acc=0.85) for s, f in zip(CONF_SEEDS, (0.42, 0.52, 0.62))])
    payload = json.loads(Path(campaign[1]).read_text())
    p = root / 'sel.json'
    p.write_text(json.dumps(payload))
    monkeypatch.setattr(hl, 'SELECTION_PATH', p)

    acc = hl.paired_outcome_vs_sven(SCAN, metric='final_test_acc').iloc[0]
    assert acc['mean'] == pytest.approx(0.70 - 0.85)
    assert acc['n'] == 3 and acc['sven_better'] == 0
    loss = hl.paired_outcome_vs_sven(SCAN, metric='final_test_loss').iloc[0]
    assert loss['mean'] < 0 and loss['sven_better'] == 3


def test_paired_differences_refuse_a_test_metric_and_outcomes_have_their_own_call(campaign):
    """:mod:`paired` is the module every SELECTION goes through, so it must refuse a test
    metric; reporting a paired difference in a test OUTCOME is a separate call whose
    arithmetic still matches."""
    with pytest.raises(ValueError, match='test split'):
        hl.paired_vs_sven(SCAN, metric='final_test_loss')
    with pytest.raises(ValueError, match='not a test outcome'):
        hl.paired_outcome_vs_sven(SCAN, metric='final_val_loss')

    out = hl.paired_outcome_vs_sven(SCAN, metric='final_test_loss').iloc[0]
    val = hl.paired_vs_sven(SCAN).iloc[0]
    # the fixture's test curve is 1.1x the val curve, so every paired difference is too
    assert out['n'] == val['n']
    assert out['mean'] == pytest.approx(1.1 * val['mean'])


# ===========================================================================
# 5. Efficiency and time-to-target
# ===========================================================================
def test_efficiency_comes_from_the_timing_pass(campaign):
    """Wall times come from ``{scan}_timing`` (alone on a GPU), not from the scan, and the
    step time is the SYNCHRONISED training time over the steps."""
    eff = hl.efficiency_table(SCAN)
    sven = row_of(eff, 'Sven')
    assert sven['wall_s'] == pytest.approx(2.0 * N_EPOCHS)             # timing pass
    assert sven['scan_wall_s'] == pytest.approx(1.0 * N_EPOCHS)        # the scan's own
    assert sven['epoch_s'] == pytest.approx(2.0)
    assert sven['sync_train_s'] == pytest.approx(1.6 * N_EPOCHS)
    assert sven['ms_per_step'] == pytest.approx(
        1e3 * 1.6 * N_EPOCHS / (N_EPOCHS * STEPS_PER_EPOCH))
    assert sven['peak_gpu_mem_mb'] == pytest.approx(500.0)
    assert sven['steps'] == pytest.approx(N_EPOCHS * STEPS_PER_EPOCH)
    assert sven['examples'] == pytest.approx(N_EPOCHS * STEPS_PER_EPOCH * BATCH_SIZE)
    assert sven['fin_timing'] == 3 and sven['att_timing'] == 3
    assert row_of(eff, 'Adam')['wall_vs_sven'] == pytest.approx(0.5)


def test_targets_are_pre_declared_from_validation_only(campaign):
    """The reference is the MEDIAN method's confirmation validation loss and the targets
    are the fixed multiples of it -- no test quantity anywhere near them."""
    ref = hl.target_reference(SCAN)
    assert ref == pytest.approx(np.median([0.25, 0.5]))                # 0.375
    targets = hl.targets_for(SCAN)
    assert sorted(targets.values()) == pytest.approx(sorted(
        [0.5 * ref, ref, 2 * ref]))
    assert set(targets) == set(hl.TARGET_NAMES.values())


def test_time_to_target_counts_the_runs_that_never_arrive(campaign):
    """The epoch index of a curve whose index 0 is the untrained model, the step / example
    / second conversions, and -- the point of the table -- ``n_never``."""
    ttt = hl.time_to_target_table(SCAN)
    ref = hl.target_reference(SCAN)

    easy = ttt[(ttt['target'] == hl.TARGET_NAMES[2.0]) & (ttt['method'] == 'Sven')].iloc[0]
    # VAL_PREFIX = (10, 1, 0.5, 0.2): 0.5 <= 0.75 at index 2, i.e. after 2 epochs
    assert 2 * ref == pytest.approx(0.75)
    assert easy['epochs'] == pytest.approx(2.0) and easy['n_never'] == 0
    assert easy['steps'] == pytest.approx(2 * STEPS_PER_EPOCH)
    assert easy['examples'] == pytest.approx(2 * STEPS_PER_EPOCH * BATCH_SIZE)
    assert easy['sync_train_s'] == pytest.approx(2 * 0.5)     # the confirm runs' own
    assert easy['standalone_s'] == pytest.approx(2 * 2.0)     # 2 epochs x the timing pass

    mid = ttt[(ttt['target'] == hl.TARGET_NAMES[1.0]) & (ttt['method'] == 'Sven')].iloc[0]
    assert mid['epochs'] == pytest.approx(3.0)                # 0.2 <= 0.375
    assert bool(mid['all_reached']) is True

    hard = ttt[(ttt['target'] == hl.TARGET_NAMES[0.5]) & (ttt['method'] == 'Sven')].iloc[0]
    # only the 0.11 run ends at or below 0.1875
    assert hard['n_reached'] == 1 and hard['n_never'] == 5
    assert hard['epochs'] == pytest.approx(float(N_EPOCHS))
    assert bool(hard['all_reached']) is False

    # a diverged run never reaches a target, even though its recorded curve dips below it
    diverged = [r for _, r in hl.load(SCAN, 'confirm').iterrows()
                if r['status'] == 'diverged']
    assert diverged and all(v is not None for v in [diverged[0]['losses']['val'][2]])
    assert diverged[0]['losses']['val'][2] <= 0.75
    assert hl.epochs_to_target(diverged[0], 0.75) is None

    adam_easy = ttt[(ttt['target'] == hl.TARGET_NAMES[2.0])
                    & (ttt['method'] == 'Adam')].iloc[0]
    assert adam_easy['n_runs'] == 6 and adam_easy['n_reached'] == 4
    assert adam_easy['n_never'] == 2


def test_never_reached_and_diverged_are_different_columns(campaign):
    """"n_reached 0" says nothing until you know whether the runs got there and failed to,
    or never ran: polynomial L-BFGS reads ``n_runs 15, n_reached 0`` at every target, and
    13 of those 15 diverged.

    In the fixture Adam loses exactly two confirmation runs -- one ``status: diverged``,
    one 10x blow-up -- and Sven loses none, so the split is a count that can be written
    down: Adam's misses are all divergences, Sven's are all real failures to arrive.
    """
    ttt = hl.time_to_target_table(SCAN)
    for _, r in ttt.iterrows():
        # the invariant the two columns must preserve
        assert r['n_never'] == r['n_never_reached'] + r['n_diverged']
        assert r['n_runs'] == r['n_reached'] + r['n_never']

    adam = ttt[(ttt['target'] == hl.TARGET_NAMES[2.0]) & (ttt['method'] == 'Adam')].iloc[0]
    assert adam['n_never'] == 2
    assert adam['n_diverged'] == 2 and adam['n_never_reached'] == 0
    assert bool(adam['elig_conf']) is True                 # 4 of 6 finished

    hard = ttt[(ttt['target'] == hl.TARGET_NAMES[0.5]) & (ttt['method'] == 'Sven')].iloc[0]
    assert hard['n_never'] == 5
    assert hard['n_diverged'] == 0 and hard['n_never_reached'] == 5
    # ... and where the divergences ARE the story, elig_conf travels with the row
    assert set(ttt['elig_conf']) == {True}


def test_selection_optimism_separates_the_instance_from_the_seeds(campaign):
    """The cross-scan optimism table must quote the SAME-INSTANCE gap.

    The fixture's Sven is +25% pooled (0.25 vs 0.2, three of those six runs on a data seed
    the tuning never saw) and +5% on the tuning instance (0.21 vs 0.2).  Quoting the
    pooled number as "selection optimism" is quoting the second problem instance; on the
    real toy scan the two even have opposite signs (+77% vs -63%).
    """
    tbl = hl.selection_optimism_table(scans=(SCAN,))
    r = tbl.iloc[0]
    assert r['n_data_seeds'] == 2
    assert r['sven_pooled_%'] == pytest.approx(25.0)
    assert r['sven_same_%'] == pytest.approx(5.0, abs=1e-9)
    # Adam: 0.5 pooled and 0.47 same-instance against a tuning 0.45
    adam_same = 100 * (np.mean([0.42, 0.52]) - 0.45) / 0.45
    assert r['median_same_%'] == pytest.approx(np.median([5.0, adam_same]))
    assert r['n_worse_same'] == 2 and r['n_worse_pooled'] == 2
    assert 'selection optimism' in tbl.attrs['same']

    # where a scan has no data seeds the two columns are the identical number
    reduced = hl.confirmation_table(SCAN).copy()
    reduced['gap_val_same_instance_rel'] = reduced['gap_val_rel']
    same = hl.selection_optimism_table(tables={SCAN: reduced}).iloc[0]
    assert same['median_same_%'] == pytest.approx(same['median_pooled_%'])


def _age_records(root, before='2026-09-19T00:00:00Z'):
    """Backdate every record of the fixture so they precede the selection file's
    ``generated_at`` -- the fixture writes them at test time, which is by construction
    after it."""
    import os
    ts = pd.Timestamp(before).timestamp()
    for p in Path(root).glob('*/*.jsonl'):
        os.utime(p, (ts, ts))


def test_freshness_report_flags_a_scan_with_runs_in_flight(campaign):
    """Every table is built from a selection file written at one instant.  A record newer
    than that instant, or a live claim with no record, makes the scan PROVISIONAL -- which
    is what ``cifar10_resnet_ce_scan`` is while decision 5's rtol extension lands."""
    import os
    root = campaign[0]
    _age_records(root)
    fresh = hl.freshness_report(scans=(SCAN,))
    assert set(fresh['pass']) == {'scan', 'confirm', 'timing'}
    assert list(fresh['selection_input']) == [True, False, False]
    assert not fresh['provisional'].any()
    assert fresh.attrs['provisional_scans'] == []

    # a _confirm record newer than the selection is EXPECTED -- that pass was generated
    # from the selection -- so it must not raise the flag on its own
    later = pd.Timestamp('2026-09-21T00:00:00Z').timestamp()
    os.utime(next((root / f'{SCAN}_confirm').glob('*.jsonl')), (later, later))
    after = hl.freshness_report(scans=(SCAN,))
    assert after[after['pass'] == 'confirm'].iloc[0]['n_after_selection'] == 1
    assert not after['provisional'].any()

    # a live claim with no record: a worker has it right now
    claims = root / SCAN / 'claims'
    claims.mkdir()
    (claims / 'svd_bs8_mlp_width16_k4_lr0.1_rtol0.001_mseed9999_lseed1000.claim').touch()
    live = hl.freshness_report(scans=(SCAN,))
    row = live[live['pass'] == 'scan'].iloc[0]
    assert row['n_claims_live'] == 1 and bool(row['provisional'])
    assert live.attrs['provisional_scans'] == [SCAN]
    assert live.attrs['n_provisional'] == 1

    # a claim whose record has landed is not in flight, and a stale claim is not live
    import os
    landed = (root / SCAN /
              'svd_bs8_mlp_width16_k4_lr0.1_rtol0.001_mseed9999_lseed1000.jsonl')
    landed.write_text('{}\n')
    _age_records(root)
    assert hl.freshness_report(scans=(SCAN,)).iloc[0]['n_claims_live'] == 0
    (claims / 'other.claim').touch()
    old = os.path.getmtime(claims / 'other.claim') - 10 * hl.CLAIM_TIMEOUT_S
    os.utime(claims / 'other.claim', (old, old))
    assert hl.freshness_report(scans=(SCAN,)).iloc[0]['n_claims_live'] == 0


def test_freshness_report_counts_records_written_after_the_selection(campaign):
    """The other half: a record whose mtime is newer than the selection's ``generated_at``
    is a run the selection never saw, so the tables describe a grid that has moved."""
    import os
    root = campaign[0]
    _age_records(root)
    assert hl.freshness_report(scans=(SCAN,))['n_after_selection'].sum() == 0
    rec = next((root / SCAN).glob('*.jsonl'))
    future = pd.Timestamp('2026-09-21T00:00:00Z').timestamp()
    os.utime(rec, (future, future))
    row = hl.freshness_report(scans=(SCAN,)).iloc[0]
    assert row['pass'] == 'scan'
    assert row['n_after_selection'] == 1 and bool(row['provisional'])
    assert hl.freshness_report(scans=(SCAN,)).attrs['provisional_scans'] == [SCAN]


def test_timing_join_report_flags_a_trajectory_that_did_not_reproduce(campaign):
    """The standalone runs carry the scan's run_ids, so their final validation losses
    must match the scan's where the optimizer is reproducible across GPUs."""
    rep = hl.timing_join_report(SCAN)
    assert list(rep['method']) == ['Sven', 'Adam']          # Sven first everywhere
    assert list(rep['n_timing']) == [3, 3]
    # Adam's `oom` tuning run has no final value to compare against, so only 2 of its 3
    # standalone runs can be joined -- counted, not quietly treated as agreement
    assert list(rep['n_joined']) == [3, 2]

    sven, adam_ = rep.iloc[0], rep.iloc[1]
    assert sven['max_rel_dev'] == pytest.approx(0.0) and bool(sven['bit_reproduced'])
    assert adam_['max_rel_dev'] == pytest.approx(0.1)      # the perturbed standalone run
    assert not bool(adam_['bit_reproduced'])
    # the oom scan run has no derived value; its standalone twin does -- counted in its
    # own column, not silently absorbed into "agreement"
    assert adam_['n_diverged_only_in_scan'] == 1
    assert list(rep['n_diverged_only_in_timing']) == [0, 0]


def test_timing_join_surfaces_a_standalone_run_that_diverged(campaign, tmp_path,
                                                             monkeypatch):
    """A deviation the derived columns CANNOT show.

    ``add_derived`` NaNs a diverged run, so a standalone twin that blew up has no value to
    difference and drops out of ``max_rel_dev`` -- which is how the real
    ``mnist_scan_labelRegression`` SOAP run (scan 0.901, standalone 184.8, a 204x
    deviation, flagged by ``bench/check_timing_join.py``) was reported as
    ``max_rel_dev 0.21, n_joined 4``.  It has to appear as a count AND in the raw column.
    """
    root = tmp_path / 'blown_timing'
    root.mkdir()
    monkeypatch.setenv(style.RESULTS_ROOT_ENV, str(root))
    hl.clear_cache()
    write_scan(root, SCAN, [sven(s, f) for s, f in zip(TUNE_SEEDS, TUNE_SVEN)])
    # the standalone twin of the FIRST seed diverges (10x rule: val[0] is 10.0); the
    # others reproduce their scan twins exactly
    write_scan(root, f'{SCAN}_timing',
               [sven(TUNE_SEEDS[0], 500.0)]
               + [sven(s, f) for s, f in zip(TUNE_SEEDS[1:], TUNE_SVEN[1:])])
    payload = json.loads(Path(campaign[1]).read_text())
    del payload['scans'][SCAN]['methods']['Adam']
    p = root / 'sel.json'
    p.write_text(json.dumps(payload))
    monkeypatch.setattr(hl, 'SELECTION_PATH', p)

    rep = hl.timing_join_report(SCAN)
    r = rep.iloc[0]
    assert r['n_timing'] == 3 and r['n_joined'] == 2
    assert r['max_rel_dev'] == pytest.approx(0.0)          # the two that joined agree ...
    assert r['n_diverged_only_in_timing'] == 1             # ... and the third is counted
    assert not bool(r['bit_reproduced'])                   # not "reproduced", either
    # the raw column carries the size of it: |0.1 - 500| / 0.1
    assert r['max_rel_dev_raw'] == pytest.approx((500.0 - 0.10) / 0.10)
    assert f'mseed{TUNE_SEEDS[0]}' in r['worst_run_raw']
    assert rep.attrs['n_diverged_only'] == 1


# ===========================================================================
# 6. Calibration lines, budget, ranking, export
# ===========================================================================
def test_calibration_report_reads_start_and_end_lines(tmp_path):
    """A timing job's log carries a fixed Adam-MLP step time before and after the pass;
    the drift between them is the evidence for or against host-load contamination."""
    d = tmp_path / 'logs'
    d.mkdir()
    def line(tag, ms, load):
        return ('[calib] ' + json.dumps(
            {'tag': tag, 'host': 'h1', 'slurm_job_id': '42', 'gpu': 'A100',
             'median_ms': ms, 'loadavg': load}))
    (d / 'timing-quiet-42.out').write_text(
        f"noise\n{line('start:quiet', 0.60, [2.0, 2.0, 2.0])}\nmore\n"
        f"{line('end:quiet', 0.61, [2.1, 2.0, 2.0])}\n")
    (d / 'timing-loaded-43.out').write_text(
        f"{line('start:loaded', 1.13, [1.0, 1.0, 1.0])}\n"
        f"{line('end:loaded', 4.58, [30.0, 28.0, 25.0])}\n")

    rep = hl.calibration_report(log_dir=d, scans=('quiet', 'loaded'))
    assert list(rep['scan']) == ['loaded', 'quiet']
    loaded = rep[rep['scan'] == 'loaded'].iloc[0]
    assert loaded['drift'] == pytest.approx(4.58 / 1.13 - 1)
    assert bool(loaded['contaminated']) is True
    quiet = rep[rep['scan'] == 'quiet'].iloc[0]
    assert abs(quiet['drift']) < 0.02 and bool(quiet['contaminated']) is False
    assert rep.attrs['missing_scans'] == []
    assert hl.calibration_report(log_dir=tmp_path / 'nope').empty


def test_budget_counts_distinct_trajectories_not_grid_points(campaign):
    """The k = 8 twin of every Sven point is the IDENTICAL trajectory, so the grid bought
    half the trials its size suggests -- the budget objection's actual answer."""
    tbl = hl.budget_table(SCAN)
    sven = row_of(tbl, 'Sven')
    assert sven['grid_points'] == 2
    assert sven['grid_per_group'] == pytest.approx(2.0)
    assert sven['distinct_per_group'] == pytest.approx(1.0)
    assert sven['distinct_frac'] == pytest.approx(0.5)

    curves, at_n = hl.best_of_n_table(SCAN, n_values=(1, 2))
    assert at_n.loc['Sven', 'grid_points'] == 2
    # both points score 0.2, so any budget finds 0.2
    assert at_n.loc['Sven', 1] == pytest.approx(0.2)
    assert at_n.loc['Sven', 2] == pytest.approx(0.2)


def test_ranking_summary_ranks_val_and_test_separately(campaign):
    summary = hl.ranking_summary()
    assert set(summary['scan']) == {SCAN}
    assert dict(zip(summary['method'], summary['rank_val'])) == {'Sven': 1.0, 'Adam': 2.0}
    # accuracy ranks DESCENDING (0.9 beats 0.8), the losses ascending
    assert dict(zip(summary['method'], summary['rank_acc'])) == {'Sven': 1.0, 'Adam': 2.0}
    assert list(summary['n_methods']) == [2, 2]
    mat = hl.rank_matrix(summary, 'rank_val')
    assert list(mat.columns)[0] == 'Sven'


def test_no_selection_step_reads_a_test_metric(campaign):
    """The assertion the notebook runs, and proof that it is not vacuous: putting a test
    key into the selection file makes it fail."""
    assert hl.assert_no_test_selection() == 2

    payload = json.loads(Path(campaign[1]).read_text())
    payload['scans'][SCAN]['methods']['SVD']['test_acc'] = 0.99
    with pytest.raises(AssertionError, match='test key'):
        hl.assert_no_test_selection(payload=payload)

    payload = json.loads(Path(campaign[1]).read_text())
    del payload['scans'][SCAN]['methods']['Adam']['seed_mean_final_val']
    with pytest.raises(AssertionError, match='selected on'):
        hl.assert_no_test_selection(payload=payload)


def test_markdown_and_latex_export(campaign, tmp_path, monkeypatch):
    """Both exports, and the markdown renderer that works without ``tabulate`` (which
    this environment has no ``pip`` to install), pinned by forcing the fallback."""
    import pandas as pd

    monkeypatch.setattr(pd.DataFrame, 'to_markdown',
                        lambda *a, **k: (_ for _ in ()).throw(ImportError('tabulate')))
    tbl = hl.confirmation_view(hl.confirmation_table(SCAN))
    md = hl.to_markdown(tbl)
    assert md.splitlines()[0].startswith('| method |')
    assert re.match(r'^\|(-{3}\|)+$', md.splitlines()[1])
    assert len(md.splitlines()) == len(tbl) + 2
    assert 'Sven' in md and '±' in md

    md_path, tex_path = hl.write_table(tbl, 'fixture_confirmation',
                                       tables_dir=tmp_path / 'tables')
    assert md_path.read_text() == md + '\n'
    assert r'\begin{tabular}' in tex_path.read_text()
    # a non-finite value renders blank, not 'nan'
    assert 'nan' not in hl.to_markdown(hl.confirmation_table(SCAN))


def test_a_schema_1_selection_file_is_refused(tmp_path):
    p = tmp_path / 'old.json'
    p.write_text(json.dumps([['SVD', 'cfg', 0.1]]))
    with pytest.raises(ValueError, match='schema-2'):
        hl.load_selection(p)
