"""CPU tests for `analysis/legacy_diff.py` -- the legacy-vs-fresh account (WP5).

Two SYNTHETIC results roots in a tmp directory (a "legacy" one with no manifest and no
test columns, a "fresh" one with both and a ``_confirm`` pass), a hand-written repair
parquet and a hand-written selection file.  No test reads `experiment_results/` or
`experiment_results_legacy_2026-09-18/`, so none of them can pass because the real data
happens to agree.

What is pinned here is the part of the diff a reader has to trust:

* the legacy metric is the EXAMPLE-WEIGHTED repair joined on ``run_id`` -- not the number
  the legacy run recorded -- and a scan with no repair parquet says so instead of quietly
  falling back;
* the binding rule is applied identically to both roots, including its fewest-diverged
  tier, which can prefer a config with a WORSE mean;
* a rank is a rank in a named field: a method the legacy campaign never ran gets no rank
  and does not shift anyone else's, and the like-for-like comparison is the common field;
* the cause vocabulary is closed, and every cause is assigned from a fact (the dataset,
  the method's absence from the legacy root, the grid sizes), never from prose;
* the two testable causes are actually tested: AdamW-as-Adam duplication (F9) and the
  invisibility of legacy failures (F6).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'analysis'))   # the notebooks' own sys.path.insert(0, '.')

import headline as hl             # noqa: E402
import legacy_diff as ld          # noqa: E402
import repair_legacy as RL        # noqa: E402
import scan_analysis as sa        # noqa: E402
import style                      # noqa: E402

N_EPOCHS = 4
BATCH_SIZE = 8
SEEDS = (1000, 1001, 1002, 1003, 1004)
CONF_SEEDS = (1100, 1101, 1102, 1103, 1104)


# ---------------------------------------------------------------------------
# The two synthetic roots
# ---------------------------------------------------------------------------
def losses(final, *, val=None, test=True):
    curve = [10.0, 1.0, 0.5, 0.2, final] if val is None else list(val)
    out = {'train': [0.9 * v for v in curve[1:]], 'val': curve,
           'epoch_times': [1.0] * N_EPOCHS, 'total_time': 4.0}
    if test:
        out['test'] = [1.1 * v for v in curve]
        out['test_acc'] = [0.1] + [0.9] * (len(curve) - 1)
    return out


def record(run_id, optimizer, seed, final, *, status='ok', test=True, val=None, **extra):
    rec = {'run_id': run_id, 'optimizer': optimizer, 'model_seed': seed,
           'loader_seed': 1000, 'batch_size': BATCH_SIZE, 'mlp_width': 16,
           'num_epochs': N_EPOCHS, 'status': status, 'schema_version': 2,
           'diag_file': None, 'svd_summary': None, 'loss': 'mse',
           'losses': losses(final, val=val, test=test)}
    rec.update(extra)
    return rec


def sven(seed, final, *, k=4, lr=0.1, rtol=1e-3, tag='', **kw):
    rid = f'svd_bs8_mlp_width16_k{k}_lr{lr:g}_rtol{rtol:g}{tag}_mseed{seed}_lseed1000'
    return record(rid, 'SVD', seed, final, k=k, lr=lr, rtol=rtol, kappa=2.0,
                  svd_mode='torch', **kw)


def plain(optimizer, seed, final, *, lr=0.01, wd=0.0, **kw):
    rid = (f'std_bs8_mlp_width16_lr{lr:g}_optim{optimizer}'
           + (f'_wd{wd:g}' if optimizer == 'AdamW' else '')
           + f'_mseed{seed}_lseed1000')
    return record(rid, optimizer, seed, final, lr=lr, weight_decay=wd, k=None, rtol=None,
                  **kw)


def write_scan(root, name, records, manifest=True):
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    for rec in records:
        (d / f"{rec['run_id']}.jsonl").write_text(json.dumps(rec) + '\n')
    if manifest:
        (d / 'manifest').mkdir(exist_ok=True)
        (d / 'manifest' / 'job0.json').write_text(json.dumps(
            {'job': 'job0', 'n_runs': len(records),
             'run_ids': [r['run_id'] for r in records]}))
    return d


SCAN = 'mnist_fixture_scan'           # 'mnist' in the name -> the `split` cause


def build_roots(tmp_path):
    """A legacy root, a fresh root with a ``_confirm`` pass, and the repair parquet.

    The numbers are chosen so every expected answer can be written down:

    * legacy Sven mean 0.20, Adam 0.10, SGD 0.30  -> Adam 1st, Sven 2nd, SGD 3rd;
    * fresh Sven mean 0.05, Adam 0.10, SGD 0.30, MuonW 0.01 (new) -> Sven 1st in the
      common field, 2nd in the full field;
    * the repair multiplies every legacy validation loss by 1.10, so a test can tell the
      repaired metric from the recorded one.
    """
    legacy, fresh = tmp_path / 'legacy', tmp_path / 'fresh'
    leg_records = []
    for s in SEEDS:
        leg_records.append(sven(s, 0.20 / 1.10, test=False))
        leg_records.append(plain('Adam', s, 0.10 / 1.10, test=False))
        leg_records.append(plain('SGD', s, 0.30 / 1.10, test=False))
        # AdamW at wd = 0 IS Adam under another name (F9): identical final loss
        leg_records.append(plain('AdamW', s, 0.10 / 1.10, wd=0.0, test=False))
    # a legacy Sven config that is BETTER on the seeds that finished but blew up on two of
    # them: the fewest-diverged tier must keep it from winning
    for s in SEEDS[:3]:
        leg_records.append(sven(s, 0.05 / 1.10, k=8, tag='_alt', test=False))
    for s in SEEDS[3:]:
        leg_records.append(sven(s, 0.05 / 1.10, k=8, tag='_alt', test=False,
                                status='diverged', val=[10.0, 1.0]))
    write_scan(legacy, SCAN, leg_records, manifest=False)

    fresh_records = []
    for s in SEEDS:
        fresh_records.append(sven(s, 0.05))
        fresh_records.append(plain('Adam', s, 0.10))
        fresh_records.append(plain('SGD', s, 0.30))
        fresh_records.append(plain('MuonW', s, 0.01))       # a baseline legacy never ran
        fresh_records.append(plain('AdamW', s, 0.12, wd=0.01))   # now decays: not Adam
        # a second Sven grid point: the fresh grid is larger -> the `grid` cause
        fresh_records.append(sven(s, 0.07, k=8))
    write_scan(fresh, SCAN, fresh_records)
    conf = []
    for i, s in enumerate(CONF_SEEDS):
        conf.append(sven(s, 0.06))
        conf.append(plain('Adam', s, 0.11))
        conf.append(plain('MuonW', s, 0.02))
        conf.append(plain('AdamW', s, 0.13, wd=0.01))
        # SGD blows up on 4 of the 5 CONFIRMATION seeds although it finished all 5 tuning
        # seeds: its confirmation mean (0.02, the one survivor) is a mean over the lucky
        # run, so the configuration is not eligible there and the rule must rank it on its
        # TUNING loss (0.30) instead -- while the table still prints 0.02.
        conf.append(plain('SGD', s, 0.02) if i == 0 else
                    plain('SGD', s, 0.31, status='diverged', val=[10.0, 1.0]))
    write_scan(fresh, f'{SCAN}_confirm', conf)

    # the repair parquet: 1.10x every legacy value (and NaN for the diverged runs)
    rows = []
    for rec in leg_records:
        ok = rec['status'] == 'ok'
        rows.append({'run_id': rec['run_id'],
                     'final_val_loss': rec['losses']['val'][-1],
                     'final_val_loss_ew': rec['losses']['val'][-1] * 1.10 if ok else np.nan,
                     'rel_corr_final': 0.10 if ok else np.nan})
    repair_dir = tmp_path / 'repair'
    repair_dir.mkdir()
    pd.DataFrame(rows).to_parquet(repair_dir / f'{SCAN}.parquet', index=False)
    return legacy, fresh, repair_dir


def selection_payload(fresh_root):
    """A schema-2 selection file naming the fresh winners (what :mod:`headline` reads)."""
    def entry(method, val, hparams):
        return {'method': method, 'label': method, 'family': 'std',
                'config_key': 'x', 'overrides': 'x', 'verified': True,
                'verify_error': None, 'seed_mean_final_val': val, 'n_ok': 5,
                'n_expected': 5, 'n_diverged': 0, 'n_eligible_configs': 1,
                'tied_with': [], 'edges': {}, 'batch_size': BATCH_SIZE,
                'hash8': {}, 'hparams': hparams}
    return {'schema': 2, 'generated_at': '2026-09-20T00:00:00-0400',
            'generated_by': 'tests/test_legacy_diff.py',
            'results_root': str(fresh_root), 'config_dir': 'x', 'rule_name': 'full',
            'rule': 'eligible -> fewest diverged -> seed mean of final VALIDATION loss',
            'default_groups': ['mode=all'],
            'scans': {SCAN: {'groups': ['mode=all'], 'scan_dir': str(fresh_root / SCAN),
                             'n_expected': 30, 'n_records': 30, 'n_missing': 0,
                             'warnings': [], 'errors': [], 'reconcile_disagreements': [],
                             'methods': {
                                 'SVD': entry('SVD', 0.05, {'k': 4, 'lr': 0.1,
                                                            'rtol': 1e-3}),
                                 'Adam': entry('Adam', 0.10, {'lr': 0.01}),
                                 'SGD': entry('SGD', 0.30, {'lr': 0.01}),
                                 'MuonW': entry('MuonW', 0.01, {'lr': 0.01}),
                                 'AdamW': entry('AdamW', 0.12, {'lr': 0.01,
                                                                'weight_decay': 0.01}),
                             }}}}


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """Both roots wired into the modules under test, and the selection payload."""
    legacy, fresh, repair_dir = build_roots(tmp_path)
    monkeypatch.setenv(style.RESULTS_ROOT_ENV, str(fresh))
    monkeypatch.setattr(style, 'RESULTS_ROOT', str(fresh))
    monkeypatch.setattr(sa, 'RESULTS_ROOT', str(fresh))
    monkeypatch.setattr(ld, 'LEGACY_ROOT', str(legacy))
    # the fixture scan takes the headline path: its fresh numbers come from the selection
    # file and the `_confirm` pass, exactly as the seven real headline scans do
    monkeypatch.setattr(ld, 'HEADLINE_SCANS', (SCAN,))
    monkeypatch.setattr(ld, 'LEGACY_CACHE', str(tmp_path / 'cache'))
    monkeypatch.setattr(ld, 'PLOT_DIR', tmp_path / 'plots')
    monkeypatch.setattr(RL, 'OUT_DIR', repair_dir)
    hl.clear_cache()
    yield {'legacy': legacy, 'fresh': fresh, 'repair': repair_dir,
           'payload': selection_payload(fresh)}
    hl.clear_cache()


# ---------------------------------------------------------------------------
# The legacy metric
# ---------------------------------------------------------------------------
def test_legacy_metric_is_the_example_weighted_repair(campaign):
    scobj, cov = ld.load_side(SCAN, legacy=True)
    assert cov['repaired'] and cov['coverage'] == 1.0
    row = scobj.df[scobj.df['optimizer'] == 'Adam'].iloc[0]
    # the RECORDED number is 0.10/1.10; the repair is what the tables must use
    assert row['final_val_loss'] == pytest.approx(0.10 / 1.10)
    assert row[ld.METRIC] == pytest.approx(0.10)
    # and the diverged runs carry no repaired value at all
    diverged = scobj.df[scobj.df['diverged']]
    assert len(diverged) == 2 and diverged[ld.METRIC].isna().all()


def test_a_scan_without_a_repair_parquet_says_so(campaign, tmp_path, monkeypatch):
    monkeypatch.setattr(RL, 'OUT_DIR', tmp_path / 'empty')
    scobj, cov = ld.load_side(SCAN, legacy=True)
    assert cov['repaired'] is False
    assert 'RECORDED' in cov['source']
    # the fallback is the recorded, batch-averaged number -- usable, but a caveat
    row = scobj.df[scobj.df['optimizer'] == 'Adam'].iloc[0]
    assert row[ld.METRIC] == pytest.approx(0.10 / 1.10)


def test_the_fresh_side_uses_the_recorded_example_weighted_loss(campaign):
    scobj, cov = ld.load_side(SCAN, legacy=False)
    assert cov['repaired'] is False and 'example-weighted' in cov['source']
    assert (scobj.df[ld.METRIC].dropna() == scobj.df['final_val_loss'].dropna()).all()


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------
def test_the_rule_prefers_fewest_diverged_over_a_better_mean(campaign):
    scobj, _ = ld.load_side(SCAN, legacy=True)
    best = ld.best_per_method(scobj, SCAN).set_index('method')
    # the k=8 twin has the better mean (0.05) but blew up on 2 of 5 seeds
    assert best.loc['Sven', 'val'] == pytest.approx(0.20)
    assert best.loc['Sven', 'n_diverged'] == 0
    assert best.loc['Sven', 'finished'] == 5 and best.loc['Sven', 'attempted'] == 5


def test_legacy_ranks_come_out_in_the_expected_order(campaign):
    scobj, _ = ld.load_side(SCAN, legacy=True)
    best = ld.best_per_method(scobj, SCAN).set_index('method')
    # Adam and AdamW are the same run (0.10) and tie at rank 1; Sven 0.20, SGD 0.30
    assert best.loc['Adam', 'rank'] == 1 and best.loc['AdamW', 'rank'] == 1
    assert best.loc['Sven', 'rank'] == 3 and best.loc['SGD', 'rank'] == 4


def test_selection_agreement_reproduces_the_selection_of_record(campaign):
    agree = ld.selection_agreement(scans=[SCAN], payload=campaign['payload'])
    assert len(agree) == 5 and agree['agrees'].all()


def test_selection_agreement_flags_a_disagreement(campaign):
    payload = campaign['payload']
    payload['scans'][SCAN]['methods']['SVD']['seed_mean_final_val'] = 0.123
    agree = ld.selection_agreement(scans=[SCAN], payload=payload)
    bad = agree[~agree['agrees']]
    assert list(bad['method']) == ['Sven']


# ---------------------------------------------------------------------------
# Ranks and the field they are in
# ---------------------------------------------------------------------------
def test_ranks_are_taken_in_the_common_field_and_in_the_full_one(campaign):
    tbl = ld.diff_table(SCAN, payload=campaign['payload'])
    row = tbl.set_index('method').loc['Sven']
    # legacy field: Adam, AdamW, SGD, Sven -> Sven 3rd of 4 common methods
    assert row['n_common'] == 4 and row['rank_leg_common'] == 3
    # fresh: Sven 0.06 beats Adam 0.11, AdamW 0.13 and SGD 0.31 -> 1st of the 4 common
    assert row['rank_fresh_common'] == 1 and row['d_rank'] == -2
    # but MuonW (0.02) is new, so in the FULL field Sven is 2nd of 5
    assert row['n_fresh_methods'] == 5 and row['rank_conf'] == 2


def test_a_method_only_the_fresh_campaign_ran_has_no_rank_change(campaign):
    tbl = ld.diff_table(SCAN, payload=campaign['payload']).set_index('method')
    assert not tbl.loc['MuonW', 'in_legacy']
    assert np.isnan(tbl.loc['MuonW', 'rank_leg_common'])
    assert np.isnan(tbl.loc['MuonW', 'd_rank'])
    assert 'newbase' in tbl.loc['MuonW', 'causes']


def test_grouped_ranks_are_taken_within_the_axis(campaign, tmp_path, monkeypatch):
    # one scan, two batch sizes, and Sven wins only at the larger one
    name = 'polynomial_fixture_batch'
    monkeypatch.setitem(ld.GROUPED_SCANS, name, 'batch_size')
    for root, sven_final in ((campaign['legacy'], (0.4, 0.4)),
                             (campaign['fresh'], (0.4, 0.05))):
        recs = []
        for i, B in enumerate((8, 16)):
            for s in SEEDS:
                r = sven(s, sven_final[i], tag=f'_B{B}')
                r['batch_size'] = B
                recs.append(r)
                a = plain('Adam', s, 0.10, lr=0.01 + i)
                a['batch_size'] = B
                recs.append(a)
        write_scan(root, name, recs, manifest=root == campaign['fresh'])
    tbl = ld.grouped_rank_table(name).set_index('batch_size')
    assert list(tbl.index) == [8, 16]
    assert tbl.loc[8, 'rank_leg'] == 2 and tbl.loc[8, 'rank_fresh'] == 2
    assert tbl.loc[16, 'rank_leg'] == 2 and tbl.loc[16, 'rank_fresh'] == 1
    assert tbl.loc[16, 'd_rank'] == -1


def test_ranks_only_scans_never_print_a_legacy_loss(campaign, tmp_path, monkeypatch):
    assert ld.ranks_only('polynomial_scan') and not ld.ranks_only('mnist_scan_ce')
    tbl = ld.diff_table(SCAN, payload=campaign['payload'])
    tbl.attrs['ranks_only'] = True
    view = ld.diff_view(tbl)
    shown = set(view[view['legacy config'] != '--']['legacy val'])
    assert shown == {'n/a (different target)'}


# ---------------------------------------------------------------------------
# The cause vocabulary
# ---------------------------------------------------------------------------
def test_every_cause_is_in_the_closed_vocabulary(campaign):
    tbl = ld.diff_table(SCAN, payload=campaign['payload'])
    for codes in tbl['causes']:
        for c in codes.split(','):
            assert c in ld.CAUSES, c


def test_causes_follow_the_dataset_and_the_facts():
    mnist = ld.causes_for('mnist_scan_ce', 'Adam')
    assert 'split' in mnist and 'bn' not in mnist and 'target' not in mnist
    cifar = ld.causes_for('cifar10_resnet_ce_scan', 'SVD')
    assert 'bn' in cifar and 'split' in cifar
    poly = ld.causes_for('polynomial_scan', 'Adam')
    assert 'target' in poly and 'fixedval' in poly
    assert 'muon' in ld.causes_for('toy_1d_scan', 'MuonW')
    assert 'adamw_wd' in ld.causes_for('toy_1d_scan', 'AdamW')
    assert 'newbase' in ld.causes_for('toy_1d_scan', 'SGDm')
    assert 'grid' in ld.causes_for('toy_1d_scan', 'Adam', grid_grew=True)
    assert 'grid' not in ld.causes_for('toy_1d_scan', 'Adam', grid_grew=False)
    assert 'newbase' in ld.causes_for('toy_1d_scan', 'SOAP', in_legacy=False)


def test_grid_growth_is_read_off_the_records(campaign):
    tbl = ld.diff_table(SCAN, payload=campaign['payload']).set_index('method')
    # the fresh Sven grid has two configurations, the legacy one had two as well (the
    # blown-up `_alt` twin counts, because its records exist)
    assert tbl.loc['Sven', 'n_configs_fresh'] == 2
    assert tbl.loc['Sven', 'n_configs_leg'] == 2
    assert 'grid' not in tbl.loc['Sven', 'causes']
    assert tbl.loc['Adam', 'n_configs_leg'] == 1


# ---------------------------------------------------------------------------
# The two causes that can be tested from the data
# ---------------------------------------------------------------------------
def test_adamw_duplicate_check_finds_the_duplication(campaign):
    leg, _ = ld.load_side(SCAN, legacy=True)
    got = ld.adamw_duplicate_check(leg, wd=0.0)
    assert got['n_pairs'] == 5 and got['n_identical'] == 5
    assert got['max_rel_diff'] == pytest.approx(0.0, abs=1e-12)
    fresh, _ = ld.load_side(SCAN, legacy=False)
    got = ld.adamw_duplicate_check(fresh, wd=0.01)
    assert got['n_pairs'] == 5 and got['n_identical'] == 0


def test_failure_visibility_separates_records_from_failures(campaign):
    leg, _ = ld.load_side(SCAN, legacy=True)
    fresh, _ = ld.load_side(SCAN, legacy=False)
    vis = ld.failure_visibility(leg, fresh, methods=('Sven',)).iloc[0]
    assert vis['records_leg'] == 10 and vis['diverged_leg'] == 2
    assert vis['usable_leg'] == 8
    assert vis['records_fresh'] == 10 and vis['diverged_fresh'] == 0


# ---------------------------------------------------------------------------
# The rendered views
# ---------------------------------------------------------------------------
def test_diff_view_shows_counts_and_both_ranks(campaign):
    view = ld.diff_view(ld.diff_table(SCAN, payload=campaign['payload']))
    row = view.set_index('method').loc[style.method_label('SVD')]
    assert row['legacy fin/att'] == '5/5' and row['fresh fin/att'] == '5/5'
    assert row['legacy rank'] == '3/4' and row['fresh rank'] == '1/4'
    assert row['fresh rank (all methods)'] == '2/5'
    assert row['rank change'] == '-2'


def test_sven_rank_summary_collects_scans_and_axis_points(campaign):
    tbl = ld.diff_table(SCAN, payload=campaign['payload'])
    summary = ld.sven_rank_summary({SCAN: tbl})
    assert len(summary) == 1
    r = summary.iloc[0]
    assert r['rank_leg'] == 3 and r['rank_fresh'] == 1 and r['d_rank'] == -2
    assert r['n_common'] == 4


def test_display_names_come_from_the_style_registry(campaign):
    view = ld.diff_view(ld.diff_table(SCAN, payload=campaign['payload']))
    assert style.method_label('LBFGS') == 'Stochastic L-BFGS'
    assert set(view['method']) >= {'Sven', 'Adam', 'AdamW', 'SGD', 'MuonW'}


# ---------------------------------------------------------------------------
# What a fresh rank is computed ON
# ---------------------------------------------------------------------------
def test_a_rank_ranked_on_the_tuning_seeds_is_recorded_as_such(campaign):
    """SGD finished 5/5 tuning seeds and 1/5 confirmation seeds.

    A mean over the one survivor (0.02) would make SGD the best method on the scan, so the
    rule ranks it on its tuning loss (0.30) -- and the row must say that the rank and the
    printed confirmation mean are two different numbers.
    """
    tbl = ld.diff_table(SCAN, payload=campaign['payload']).set_index('method')
    sgd = tbl.loc['SGD']
    assert sgd['fin_conf'] == 1 and sgd['att_conf'] == 5 and not sgd['elig_conf']
    assert sgd['val_conf'] == pytest.approx(0.02)          # printed
    assert sgd['val_fresh_rankon'] == pytest.approx(0.30)  # ranked on
    assert sgd['fresh_basis'] == 'tuning'
    # and every eligible row is ranked on its confirmation mean, Sven included
    assert tbl.loc['Sven', 'fresh_basis'] == 'confirm'
    assert tbl.loc['Sven', 'val_fresh_rankon'] == pytest.approx(0.06)
    # the fallback must not silently make the method look good: ranked on 0.30 it is last
    assert sgd['rank_fresh_common'] == 4


def test_the_view_marks_a_tuning_seed_rank_and_gives_its_basis(campaign):
    view = ld.diff_view(ld.diff_table(SCAN, payload=campaign['payload'])).set_index('method')
    sgd = view.loc[style.method_label('SGD')]
    assert sgd['fresh rank'].endswith(ld.FALLBACK_MARK)
    assert sgd['fresh rank (all methods)'].endswith(ld.FALLBACK_MARK)
    assert sgd['fresh rank basis'].startswith('tuning') and '1/5' in sgd['fresh rank basis']
    sven = view.loc[style.method_label('SVD')]
    assert not sven['fresh rank'].endswith(ld.FALLBACK_MARK)
    assert sven['fresh rank basis'] == 'confirm'


def test_fallback_rows_names_the_substitution(campaign):
    fb = ld.fallback_rows(ld.diff_table(SCAN, payload=campaign['payload']))
    assert list(fb['method']) == ['SGD']
    r = fb.iloc[0]
    assert r['val_conf'] == pytest.approx(0.02) and r['val_tune'] == pytest.approx(0.30)
    assert r['val_ranked_on'] == pytest.approx(0.30)
    assert r['fin_conf'] == 1 and r['att_conf'] == 5


# ---------------------------------------------------------------------------
# Aggregating ranks across scans
# ---------------------------------------------------------------------------
def test_rank_change_tally_never_pools_the_ranks_only_points():
    """A polynomial rank is a rank on another function, so it may not join the mean.

    Three same-target points that sum to 0 and two ranks-only points that sum to -10: a
    pooled mean would read -2.0 places and would be entirely the target change.
    """
    summary = pd.DataFrame([
        {'axis': '', 'd_rank': -1.0, 'ranks_only': False},
        {'axis': '', 'd_rank': +1.0, 'ranks_only': False},
        {'axis': 'n_data', 'd_rank': 0.0, 'ranks_only': False},
        {'axis': '', 'd_rank': -6.0, 'ranks_only': True},
        {'axis': 'batch_size', 'd_rank': -4.0, 'ranks_only': True},
        {'axis': '', 'd_rank': np.nan, 'ranks_only': False},   # unranked: dropped
    ])
    t = ld.rank_change_tally(summary).set_index('group')
    same = t.loc['same target']
    assert same['n'] == 3 and same['sum_d_rank'] == 0.0 and same['mean_d_rank'] == 0.0
    assert (same['better'], same['same'], same['worse']) == (1, 1, 1)
    assert (same['headline'], same['swept']) == (2, 1)
    ro = t.loc['ranks only (different target)']
    assert ro['n'] == 2 and ro['sum_d_rank'] == -10.0 and ro['mean_d_rank'] == -5.0
    # `all` exists but the two groups are what a sentence may quote
    assert t.loc['all', 'n'] == 5 and t.loc['all', 'mean_d_rank'] == pytest.approx(-2.0)


def test_rank_change_tally_reads_the_flag_the_summary_writes(campaign):
    summary = ld.sven_rank_summary({SCAN: ld.diff_table(SCAN,
                                                        payload=campaign['payload'])})
    t = ld.rank_change_tally(summary).set_index('group')
    # the fixture scan is MNIST-like: same target, so the ranks-only group is empty
    assert t.loc['same target', 'n'] == 1 and t.loc['ranks only (different target)', 'n'] == 0
    assert t.loc['same target', 'mean_d_rank'] == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# The Sven-only ablations read both grids
# ---------------------------------------------------------------------------
def test_sven_only_table_attributes_the_grid_cause_from_both_grids(campaign, monkeypatch):
    """The legacy side searched one configuration, the fresh side three.

    A level change measured across two different searches is not a measurement of the
    robustness fixes, so the row must carry the grid sizes, the `grid` cause, and the fact
    that the fresh winner is not even on the legacy grid.
    """
    name = 'mnist_fixture_kappa'
    monkeypatch.setitem(ld.SVEN_ONLY_SCANS, name, 'kappa')
    # legacy: the default kappa = 2 only, which `_config_str` does not print
    write_scan(campaign['legacy'], name,
               [sven(s, 0.20 / 1.10, test=False) for s in SEEDS], manifest=False)
    fre = []
    for kappa, final in ((1.0, 0.15), (2.0, 0.30), (3.0, 0.10)):
        for s in SEEDS:
            r = sven(s, final, tag=f'_kappa{kappa:g}')
            r['kappa'] = kappa
            fre.append(r)
    write_scan(campaign['fresh'], name, fre)
    # no repair parquet for this scan -> the recorded number, and the row says so
    row = ld.sven_only_table(scans=[name]).iloc[0]
    assert row['n_configs_leg'] == 1 and row['n_configs_fresh'] == 3
    assert row['grid_grew'] and 'grid' in row['causes']
    assert bool(row['fresh_config_in_legacy_grid']) is False
    assert 'kappa=3' in row['config_fresh'] and 'kappa' not in row['config_leg']
    assert not row['repaired']


def test_sven_only_table_does_not_invent_a_grid_cause(campaign, monkeypatch):
    name = 'mnist_fixture_microbatch'
    monkeypatch.setitem(ld.SVEN_ONLY_SCANS, name, 'microbatch_size')
    for root, final in ((campaign['legacy'], 0.20 / 1.10), (campaign['fresh'], 0.22)):
        recs = []
        for mb in (1, 2):
            for s in SEEDS:
                r = sven(s, final, tag=f'_mb{mb}')
                r['microbatch_size'] = mb
                recs.append(r)
        write_scan(root, name, recs, manifest=root == campaign['fresh'])
    row = ld.sven_only_table(scans=[name]).iloc[0]
    assert row['n_configs_leg'] == 2 and row['n_configs_fresh'] == 2
    assert not row['grid_grew'] and 'grid' not in row['causes']
    assert bool(row['fresh_config_in_legacy_grid']) is True


def test_grid_config_strings_render_both_roots_the_same_way(campaign):
    leg, _ = ld.load_side(SCAN, legacy=True)
    fre, _ = ld.load_side(SCAN, legacy=False)
    cl, cf = ld.grid_config_strings(leg, SCAN), ld.grid_config_strings(fre, SCAN)
    assert len(cl['Sven']) == 2 and len(cf['Sven']) == 2
    # the k=4 winner exists on both sides and renders identically, so the two sets meet
    assert cl['Sven'] & cf['Sven']
    # wd = 0 is not printed, so legacy AdamW renders exactly like Adam; the fresh one decays
    assert cl['AdamW'] == cl['Adam']
    assert cf['AdamW'] != cf['Adam'] and 'weight_decay=0.01' in list(cf['AdamW'])[0]


# ---------------------------------------------------------------------------
# What is outside the intersection of the two roots
# ---------------------------------------------------------------------------
def test_scan_inventory_reports_what_only_one_root_has(campaign):
    """`scans_in_both_roots` is an intersection: the complement must not vanish."""
    write_scan(campaign['legacy'], 'mnist_fixture_cut_scan',
               [plain('Adam', s, 0.5, test=False) for s in SEEDS], manifest=False)
    inv = ld.scan_inventory().set_index('scan')
    assert inv.loc[SCAN, 'where'] == 'both'
    assert inv.loc['mnist_fixture_cut_scan', 'where'] == 'legacy only'
    assert inv.loc['mnist_fixture_cut_scan', 'n_records_leg'] == len(SEEDS)
    assert inv.loc['mnist_fixture_cut_scan', 'n_records_fresh'] == 0
    # the `_confirm` pass is fresh-only and is recognised as a pass, not a lost scan
    assert inv.loc[f'{SCAN}_confirm', 'where'] == 'fresh only'
    assert inv.loc[f'{SCAN}_confirm', 'disposition'] == f'confirm pass of {SCAN}'
    assert ld.scan_inventory().attrs['n_legacy_only'] == 1


def test_every_real_legacy_only_scan_has_a_recorded_disposition():
    """The dispositions are quoted from `EXPERIMENTS.md` section 10, not invented here."""
    assert len(ld.LEGACY_ONLY_DISPOSITION) == 7
    for scan, text in ld.LEGACY_ONLY_DISPOSITION.items():
        assert text.startswith(('CUT', 'PARKED')), scan
        assert len(text) > 60, scan


# ---------------------------------------------------------------------------
# The split cause is per dataset
# ---------------------------------------------------------------------------
def test_the_split_cause_is_per_dataset():
    """MNIST/CIFAR had an official test set to select on; the character corpus had none."""
    assert 'split' in ld.causes_for('mnist_scan_ce', 'Adam')
    assert 'split' in ld.causes_for('cifar10_resnet_ce_scan', 'SVD')
    ng = ld.causes_for('exp_nanogpt_speedrun', 'SVD')
    assert 'testsplit_carved' in ng and 'split' not in ng
    # the quarantined GPT-2 scan is never diffed, so it gets no split-family cause at all
    gpt2 = ld.causes_for('exp_gpt2_small_comparison', 'Adam')
    assert 'split' not in gpt2 and 'testsplit_carved' not in gpt2
    assert 'exp_gpt2_small_comparison' in ld.EXCLUDED
    # and the `split` text no longer claims a Shakespeare split it does not describe
    assert 'Shakespeare' not in ld.CAUSES['split']
    assert ld.CAUSES['testsplit_carved'].startswith('a test split was carved out')


def test_nanogpt_split_check_tells_the_three_splits_apart(campaign, monkeypatch):
    """Legacy val must come out as the FRESH TEST split, not as the fresh validation set."""
    name = 'exp_nanogpt_fixture'
    # legacy: 90/10 by position, no test set; its val curve is the LAST tenth of the text
    legacy_val = [4.368765, 2.0, 1.5]
    fresh_val = [4.362548, 2.1, 1.6]            # a new tenth, carved out of legacy train
    leg = record('std_bs64_lr0.0001_optimAdamW_mseed5000_lseed1000', 'AdamW', 5000, 1.5,
                 test=False, val=legacy_val, lr=1e-4, weight_decay=0.01,
                 n_train=7842, n_val=871)
    write_scan(campaign['legacy'], name, [leg], manifest=False)
    fre = record('std_bs64_lr0.0001_optimAdamW_mseed5000_lseed1000', 'AdamW', 5000, 1.6,
                 test=False, val=fresh_val, lr=1e-4, weight_decay=0.01,
                 n_train=6971, n_val=871, n_test=871)
    fre['losses']['test'] = list(legacy_val)     # the legacy val blocks, now the test split
    write_scan(campaign['fresh'], name, [fre])

    got = ld.nanogpt_split_check(scan=name)
    assert got['legacy_val_is_fresh_test']
    assert got['rel_legval_vs_freshtest'] < 1e-6
    assert got['rel_legval_vs_freshval'] > 1e-4
    assert got['n_val_unchanged'] and got['train_shrank_by_test']
    assert 'ARE the fresh test split' in got['verdict']


def test_nanogpt_split_check_would_notice_a_shared_validation_set(campaign, monkeypatch):
    """The negative case: if the two roots really shared their validation set, say so."""
    name = 'exp_nanogpt_fixture_shared'
    shared = [4.368765, 2.0, 1.5]
    for root, n_train in ((campaign['legacy'], 7842), (campaign['fresh'], 6971)):
        rec = record('std_bs64_lr0.0001_optimAdamW_mseed5000_lseed1000', 'AdamW', 5000, 1.5,
                     test=False, val=shared, lr=1e-4, weight_decay=0.01,
                     n_train=n_train, n_val=871)
        write_scan(root, name, [rec], manifest=root == campaign['fresh'])
    got = ld.nanogpt_split_check(scan=name)
    assert not got['legacy_val_is_fresh_test']
    assert 'share their validation set' in got['verdict']


def test_savefig_writes_pdf_and_png(campaign, tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    summary = ld.sven_rank_summary({SCAN: ld.diff_table(SCAN,
                                                        payload=campaign['payload'])})
    ld.plot_rank_slope(summary, ax, label_col='title')
    paths = ld.savefig(fig, 'fixture', plot_dir=tmp_path / 'figs')
    plt.close(fig)
    assert [p.suffix for p in paths] == ['.pdf', '.png']
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)
