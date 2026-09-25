"""CPU tests for the analysis core (C-A2, C-A3, C-A5 axes, 4.1 results root, F20).

Everything here runs against SYNTHETIC result directories built in a tmp root and
read through ``$SV3_RESULTS_ROOT``, so the tests are fast (no Lustre, no real
scan) and cannot touch ``experiment_results/``.  The properties under test are the
ones that break silently:

* a configuration's identity comes from an ALLOW-LIST, so the provenance columns
  of schema 2 (``status``, ``run_hash``, ``git_sha``, ``host``, timestamps,
  ``n_test``, ``steps_per_epoch``, ...) can never fragment the grouping;
* no selector accepts a metric derived from the test split;
* ``status`` / manifest handling, inert on legacy records;
* partial curves do not mis-size any axis;
* ``sv_min`` and ``sv_min_kept`` never merge (F20);
* full-width spectra are normalised by the spectrum width and marked with the k
  cut, rtol and the float32 noise floor, while legacy truncated records plot as
  they always did.

The check against the REAL scans (same grouping as before the allow-list) is
``test_real_scan_grouping_unchanged``, opt-in via ``SV3_CHECK_REAL_SCANS=1``
because a cold load of those directories takes 15-25 s on Lustre; run it through
``campaign/run_cpu_tests.sh``.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')          # no plotting windows, no DISPLAY
import matplotlib.pyplot as plt   # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'analysis' / 'lib'))   # the notebooks' own sys.path.insert(0, '.')
sys.path.insert(0, str(REPO / 'analysis'))

import analysis_helpers as ah    # noqa: E402
import scan_analysis as sa       # noqa: E402
import style                     # noqa: E402
import sv_diagnostics as sv      # noqa: E402

N_EPOCHS = 4
SEEDS = (1000, 1001, 1002)


# ---------------------------------------------------------------------------
# Synthetic scans
# ---------------------------------------------------------------------------
def losses(n_epochs=N_EPOCHS, level=1.0, acc=True, test=False, times=False):
    """A record's ``losses`` dict: ``train`` per epoch, ``val`` one longer (index
    0 = untrained), exactly as the loops write them."""
    out = {'train': [level * (1 + i) for i in range(n_epochs)],
           'val': [10.0 * level] + [level * (1 + i) for i in range(n_epochs)],
           'epoch_times': [1.0] * n_epochs,
           'total_time': float(n_epochs)}
    if acc:
        out['val_acc'] = [0.1] + [0.5] * n_epochs
    if test:
        out['test'] = [10.0 * level] + [level * (2 + i) for i in range(n_epochs)]
        out['test_acc'] = [0.1] + [0.4] * n_epochs
    if times:
        out['train_times'] = [0.5] * n_epochs
    return out


def sven_record(k, lr, rtol, seed, **extra):
    run_id = f'svd_bs8_k{k}_lr{lr:g}_rtol{rtol:g}_mseed{seed}_lseed1000'
    rec = {'run_id': run_id, 'optimizer': 'SVD', 'loss': 'mse', 'batch_size': 8,
           'k': k, 'k_fraction': k / 8, 'lr': lr, 'rtol': rtol, 'kappa': 2,
           'mlp_width': 16, 'model_seed': seed, 'loader_seed': 1000,
           'svd_mode': 'gram', 'use_gram': True, 'variable_k': False,
           'microbatch_size': None, 'param_fraction': None, 'mask_mode': None,
           'gram_capture': 'full', 'diag_file': None, 'svd_summary': None,
           'losses': losses(level=lr * 100)}
    rec.update(extra)
    return rec


def baseline_record(optimizer, lr, seed, **extra):
    run_id = f'std_bs8_lr{lr:g}_optim{optimizer}_mseed{seed}_lseed1000'
    rec = {'run_id': run_id, 'optimizer': optimizer, 'loss': 'mse', 'batch_size': 8,
           'k': None, 'k_fraction': None, 'lr': lr, 'rtol': None,
           'weight_decay': 0.0, 'mlp_width': 16, 'model_seed': seed,
           'loader_seed': 1000, 'diag_file': None, 'svd_summary': None,
           'losses': losses(level=lr * 200)}
    rec.update(extra)
    return rec


#: Everything the CAMPAIGN runner writes onto every record that is an OUTCOME
#: (``generic_scan.summarize_curves`` + the test / train-eval finals, C-E5) or a
#: recorded FACT (the evaluation protocol, the checkpoint bookkeeping, the Muon
#: grouping rule, C-R4).  Every value varies per run here, which is exactly what
#: makes an unregistered one fragment the grouping or warn.
#:
#: The list is the one found empirically on 2026-09-20 by loading all 43
#: directories of ``experiment_results/`` and collecting the "[style] column ...
#: AVERAGED OVER" warnings; :func:`test_schema2_columns_are_registered` pins it.
def schema2_outcomes(i):
    return {'val_final': 1.0 * i, 'val_best': 0.5 * i, 'val_best_index': i % 4,
            'val_last3_mean': 0.9 * i, 'test': 1.1 * i, 'test_acc': 0.01 * i,
            'train_eval_final': 0.8 * i,
            'effective_loader_seed': 1000 + i, 'checkpoint_policy': 'final' if i % 2 else 'log',
            'ckpt_init_file': f'ckpt/init_mseed{i}.pt', 'ckpt_error': None if i % 2 else 'quota',
            'train_eval_size': 10_000, 'eval_every_steps': None if i % 2 else 50,
            'muon_variant': f'v{i % 2}', 'muon_rule': 'hidden2d+convflat:match_rms_adamw:v1'}


#: What schema 2 adds to every record and what a run is allowed to differ in
#: without becoming a different configuration (C-R3 / C-R4).
def provenance(seed, i):
    return {'status': 'ok', 'schema_version': 2, 'run_hash': f'{i:064x}',
            'git_sha': 'a' * 40 if i % 2 else 'b' * 40, 'git_dirty': bool(i % 2),
            'sven_git_sha': 'c' * 40, 'host': f'gpunode{i}', 'slurm_job_id': str(i),
            'n_shards': 4, 'shard_id': i % 4, 'gpu_name': 'A100' if i % 2 else 'V100',
            'start_time': f'2026-09-18T0{i % 10}:00:00+00:00', 'start_unix': 1.0 * i,
            'end_time': f'2026-09-18T0{i % 10}:30:00+00:00', 'end_unix': 2.0 * i,
            'wall_time_s': 1.0 * i, 'n_params': 321, 'n_train': 800, 'n_val': 200,
            'n_test': 200, 'steps_per_epoch': 100, 'torch_version': '2.9.1',
            **schema2_outcomes(i)}


def write_scan(root, name, records, manifest=None, diag=None):
    """Write records as ``{root}/{name}/{run_id}.jsonl`` (the real layout).

    ``manifest``: run_ids for ``{name}/manifest/job0.json`` (C-R2).
    ``diag``: ``{run_id: {array: ...}}`` written as ``{name}/diag/{run_id}.npz``.
    """
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    for rec in records:
        (d / f"{rec['run_id']}.jsonl").write_text(json.dumps(rec) + '\n')
    if manifest is not None:
        (d / 'manifest').mkdir(exist_ok=True)
        (d / 'manifest' / 'job0.json').write_text(
            json.dumps({'job': 'job0', 'n_runs': len(manifest), 'run_ids': list(manifest)}))
    for run_id, arrays in (diag or {}).items():
        (d / 'diag').mkdir(exist_ok=True)
        np.savez_compressed(d / 'diag' / f'{run_id}.npz', **arrays)
    return d


def grid_records(with_provenance=False, seeds=SEEDS):
    """Two Sven configs x seeds + two baselines x seeds = 4 configurations."""
    out, i = [], 0
    for seed in seeds:
        for k, lr, rtol in ((8, 0.01, 1e-4), (4, 0.01, 1e-4)):
            out.append(sven_record(k, lr, rtol, seed))
        for opt, lr in (('Adam', 0.001), ('SGD', 0.1)):
            out.append(baseline_record(opt, lr, seed))
        for rec in out[-4:]:
            i += 1
            if with_provenance:
                rec.update(provenance(seed, i))
    return out


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A tmp results root, installed as ``$SV3_RESULTS_ROOT``."""
    r = tmp_path / 'experiment_results'
    r.mkdir()
    monkeypatch.setenv(style.RESULTS_ROOT_ENV, str(r))
    return r


def load(root, name, **kw):
    return style.load_results(name, **kw)


def scan_of(root, name, tmp_path, **kw):
    return sa.load_scan(name, name, tmp_path / 'plots' / name, **kw)


def groups(df, cols=None):
    """The PARTITION of the runs into configurations, as a set of run_id sets.

    Compared instead of the group keys, which legitimately differ between two
    frames that record different columns; what must not change is which runs
    share a configuration."""
    cols = ah.config_columns(df) if cols is None else cols
    return {frozenset(g['run_id']) for _, g in df.groupby(cols, dropna=False)}


# ---------------------------------------------------------------------------
# 1. Configuration identity: an allow-list ignores provenance
# ---------------------------------------------------------------------------
def test_allow_list_grouping_ignores_provenance(root):
    """The provenance columns of schema 2 -- every one of which varies per RUN --
    must not enter a configuration's identity: grouping is byte-identical with and
    without them.  Auto-detection made every run its own configuration."""
    write_scan(root, 'plain', grid_records(with_provenance=False))
    write_scan(root, 'prov', grid_records(with_provenance=True))
    plain, prov = load(root, 'plain'), load(root, 'prov')

    assert set(prov.columns) - set(plain.columns), 'the fixture must add columns'
    assert groups(plain) == groups(prov)
    assert len(groups(prov)) == 4
    assert {len(g) for g in groups(prov)} == {len(SEEDS)}

    for col in ('status', 'run_hash', 'git_sha', 'git_dirty', 'host', 'slurm_job_id',
                'n_shards', 'shard_id', 'gpu_name', 'start_time', 'end_unix',
                'wall_time_s', 'n_params', 'n_val', 'n_test', 'steps_per_epoch'):
        assert col not in ah.config_columns(prov), col
    # ... while the genuine knobs are all there
    assert set(ah.config_columns(prov)) >= {'optimizer', 'lr', 'k', 'rtol', 'batch_size',
                                            'kappa', 'mlp_width', 'n_train'}

    a, b = ah.config_table(plain), ah.config_table(prov)
    assert len(a) == len(b) == 4
    assert list(a['n_seeds']) == list(b['n_seeds']) == [len(SEEDS)] * 4
    assert a['final_val_loss'].tolist() == pytest.approx(b['final_val_loss'].tolist())


def test_average_over_seeds_ignores_provenance(root):
    """style.average_over_seeds shares the flaw and the fix (scout section 4a)."""
    write_scan(root, 'plain', grid_records(with_provenance=False))
    write_scan(root, 'prov', grid_records(with_provenance=True))
    a = style.average_over_seeds(load(root, 'plain'))
    b = style.average_over_seeds(load(root, 'prov'))
    assert len(a) == len(b) == 4
    assert list(a['n_seeds']) == list(b['n_seeds']) == [len(SEEDS)] * 4


def test_average_over_seeds_tolerates_a_partly_string_column(root):
    """A column that is a STRING for some optimizers and absent for the rest --
    `gram_capture`, `muon_variant` / `muon_rule` on every real scan -- arrives as
    NaN where it does not apply.  Reading the NaN as the column's first value sent
    it down the numeric branch, where np.mean met the strings and raised
    ``the resolved dtypes are not compatible with add.reduce`` (seen on
    mnist_scan_ce / mnist_scan_labelRegression / rebuttal_overparam_mnist_scan)."""
    recs = grid_records()
    for r in recs:
        if r['optimizer'] == 'SVD':
            r['gram_capture'] = 'chunked'
            r['muon_variant'] = None
        else:
            r['gram_capture'] = None     # -> NaN in the frame, and sorted first
            r['muon_variant'] = 'hidden2d+adamw:match_rms_adamw:v1'
    write_scan(root, 'mixed', recs)
    out = style.average_over_seeds(ah.add_derived(load(root, 'mixed')))
    assert len(out) == 4
    assert set(out['gram_capture'].dropna()) == {'chunked'}          # kept as-is
    assert set(out['muon_variant'].dropna()) == {'hidden2d+adamw:match_rms_adamw:v1'}
    assert out['final_val_loss'].notna().all()                       # numbers still averaged


def test_schema2_columns_are_registered(root, capsys):
    """WP1.1: every column the campaign runner adds is registered as an OUTCOME or
    as PROVENANCE, so a fresh scan produces NO "[style] ... AVERAGED OVER" warning
    and none of them can become part of a configuration's identity.

    The list is what `experiment_results/` actually contains (all 43 directories,
    loaded once through `style.load_results` -> `analysis_helpers.config_table`
    and `scan_analysis.Scan.configs` on 2026-09-20).
    """
    outcomes = ('val_final', 'val_best', 'val_best_index', 'val_last3_mean',
                'test', 'test_acc', 'train_eval_final')
    prov = ('effective_loader_seed', 'checkpoint_policy', 'ckpt_init_file', 'ckpt_error',
            'train_eval_size', 'eval_every_steps', 'svd_spectra_schedule',
            'muon_variant', 'muon_rule')
    for col in outcomes:
        assert col in style.OUTCOME_COLUMNS, col
        assert col not in style.HPARAM_COLUMNS, col
    for col in prov:
        assert col in style.PROVENANCE_COLUMNS, col
        assert col not in style.HPARAM_COLUMNS, col
    # the runner's own `test` / `test_acc` are test-split quantities, so the
    # selection guard covers them too (C-E1)
    assert style.is_test_metric('test') and style.is_test_metric('test_acc')
    assert not any(style.is_test_metric(c) for c in
                   ('val_final', 'val_best', 'val_last3_mean', 'train_eval_final'))

    # and on a scan that carries them all, nothing is warned about and the
    # grouping is the same as without them
    write_scan(root, 'plain2', grid_records(with_provenance=False))
    write_scan(root, 'schema2', grid_records(with_provenance=True))
    for col in (*outcomes, *prov):
        style._unlisted_warned.discard(col)
    d = load(root, 'schema2')
    assert {*outcomes, *(c for c in prov if c != 'svd_spectra_schedule')} <= set(d.columns)
    cols = ah.config_columns(d)
    out = capsys.readouterr().out
    assert 'AVERAGED OVER' not in out, out
    for col in (*outcomes, *prov):
        assert col not in cols, col
    assert groups(d) == groups(load(root, 'plain2'))
    assert len(ah.config_table(d)) == 4


def test_unlisted_varying_column_is_reported(root, capsys):
    """The allow-list's one failure mode -- a new knob nobody listed -- is a printed
    warning, not a silent average."""
    recs = grid_records()
    for i, rec in enumerate(recs):
        rec['brand_new_knob'] = 0.5 if i % 2 else 1.5
    write_scan(root, 'newknob', recs)
    style._unlisted_warned.discard('brand_new_knob')
    ah.config_columns(load(root, 'newknob'))
    out = capsys.readouterr().out
    assert 'brand_new_knob' in out and 'HPARAM_COLUMNS' in out


# ---------------------------------------------------------------------------
# 2. No selector may rank on a test metric (C-E1 / C-A2)
# ---------------------------------------------------------------------------
def test_selector_assertion_fires_for_test_metrics(root, tmp_path):
    write_scan(root, 'withtest', [r | {'losses': losses(level=r['lr'] * 100, test=True)}
                                  for r in grid_records()])
    df = load(root, 'withtest')
    scan = scan_of(root, 'withtest', tmp_path)

    for metric in ('final_test_loss', 'final_test_acc', 'test', 'test_acc'):
        with pytest.raises(ValueError, match='test split'):
            ah.config_table(df, metric=metric)
        with pytest.raises(ValueError, match='test split'):
            scan.configs(scan.sven, sa.SVEN_CONFIG, metric)
        with pytest.raises(ValueError, match='test split'):
            scan.best_sven(metric)
        with pytest.raises(ValueError, match='test split'):
            sa.summary_table(scan, metric=metric)
    with pytest.raises(ValueError, match='test split'):
        ah.best_per_method(df, by='final_test_loss')

    # `quantity=` reaches configs() as the metric -- the reason the check lives
    # inside the chokepoints and not in the notebooks (scout section 5 (i)).
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match='test split'):
        sa.plot_time_vs_k(scan, ax, quantity='final_test_loss')
    with pytest.raises(ValueError, match='test split'):
        sa.knob_table(scan, 'k', 'final_test_loss')
    plt.close(fig)

    # validation metrics are of course fine, and the test outcome comes along
    assert scan.has_test
    tab = ah.config_table(df)
    assert 'final_test_loss' in tab.columns and tab['final_test_loss'].notna().all()
    summary = sa.summary_table(scan)
    assert 'final_test_loss' in summary.columns and 'final_test_acc' in summary.columns


# ---------------------------------------------------------------------------
# 3. status / manifest handling (C-R1, C-R2), inert on legacy records
# ---------------------------------------------------------------------------
def test_status_diverged_counts_and_excludes(root, tmp_path):
    """A `status: diverged` run has a FINITE partial curve, so only the status can
    reveal it.  It must leave the means and be counted in n_diverged."""
    recs = grid_records(seeds=SEEDS)
    victim = next(r for r in recs if r['optimizer'] == 'SGD' and r['model_seed'] == 1002)
    victim.update({'status': 'diverged', 'diverged_at_step': 7,
                   'error': 'RuntimeError: empty spectrum',
                   'losses': losses(n_epochs=2, level=0.2)})
    write_scan(root, 'divscan', recs)
    df = load(root, 'divscan')

    row = df[df['run_id'] == victim['run_id']].iloc[0]
    assert np.isfinite(row['losses']['val'][-1]), 'the fixture must stay finite'
    assert style.is_diverged(row['losses']['train'], row['losses']['val'],
                             style.status_of(row))
    assert not style.is_diverged(row['losses']['train'], row['losses']['val'])

    tab = ah.config_table(df)
    sgd = tab[tab['method'] == 'SGD'].iloc[0]
    assert (sgd['n_seeds'], sgd['n_diverged'], sgd['n_failed']) == (2, 1, 0)
    assert sgd['final_val_loss'] == pytest.approx(0.1 * 200 * N_EPOCHS)   # 1002 not in it

    scan = scan_of(root, 'divscan', tmp_path)
    cfg = scan.configs(scan.baseline[scan.baseline['optimizer'] == 'SGD'],
                       sa.BASELINE_CONFIG)
    assert (int(cfg.iloc[0]['n_seeds']), int(cfg.iloc[0]['n_diverged'])) == (2, 1)


def test_status_oom_error_excluded_and_counted(root, tmp_path):
    """`oom` / `error` are incomplete ATTEMPTS (retried by the runner): out of every
    mean, counted in n_failed, and never confused with a divergence."""
    recs = grid_records()
    for status, seed in (('oom', 1001), ('error', 1002)):
        r = next(r for r in recs if r['optimizer'] == 'Adam' and r['model_seed'] == seed)
        r.update({'status': status, 'error': f'{status} at step 3',
                  'losses': losses(n_epochs=1, level=1e-9)})   # a tiny "lucky" loss
    write_scan(root, 'oomscan', recs)
    df = load(root, 'oomscan')

    tab = ah.config_table(df)
    adam = tab[tab['method'] == 'Adam'].iloc[0]
    assert (adam['n_seeds'], adam['n_failed'], adam['n_diverged']) == (1, 2, 0)
    # the 1e-9 partial curves must not be in the mean (they would win the scan)
    assert adam['final_val_loss'] == pytest.approx(0.001 * 200 * N_EPOCHS)
    assert not adam['eligible'], 'only 1 of 3 runs finished'

    scan = scan_of(root, 'oomscan', tmp_path)
    assert int(scan.df['failed'].sum()) == 2
    summary = sa.summary_table(scan)
    assert 'Adam' not in set(summary['method']), 'ineligible: no eligible config'
    mean, _, _ = sa.seed_band(scan.baseline[scan.baseline['optimizer'] == 'Adam'], 'val')
    assert len(mean) == N_EPOCHS + 1, 'a 1-epoch failed run must not truncate the band'


def test_manifest_makes_missing_runs_visible(root, tmp_path, capsys):
    """n_missing comes from the manifest union, so a seed with no result file -- and
    a whole configuration with none -- stop being invisible (C-A2)."""
    recs = grid_records()
    expected = [r['run_id'] for r in recs]
    # one seed of a Sven config, and a whole extra config, are never written
    dropped = next(r for r in recs if r['optimizer'] == 'SVD' and r['k'] == 4
                   and r['model_seed'] == 1002)
    recs.remove(dropped)
    ghost = [f'svd_bs8_k2_lr0.01_rtol0.0001_mseed{s}_lseed1000' for s in SEEDS]
    write_scan(root, 'manifestscan', recs, manifest=expected + ghost)
    df = load(root, 'manifestscan')

    assert style.manifest_run_ids('manifestscan') == set(expected) | set(ghost)
    assert style.missing_run_ids(df, style.manifest_run_ids('manifestscan')) == sorted(
        [dropped['run_id']] + ghost)

    tab = ah.config_table(df)
    k4 = tab[(tab['optimizer'] == 'SVD') & (tab['k'] == 4)].iloc[0]
    assert (k4['attempted'], k4['n_seeds'], k4['n_missing']) == (3, 2, 1)
    k8 = tab[(tab['optimizer'] == 'SVD') & (tab['k'] == 8)].iloc[0]
    assert (k8['attempted'], k8['n_missing']) == (3, 0)
    # the configuration with NO file at all cannot be a row; it is listed instead --
    # and PRINTED, since no notebook reads `.attrs` (nor survives a merge), so
    # without the line it stays as invisible as it was before C-A2
    miss = tab.attrs['missing_configs']
    assert list(miss['config_key']) == ['svd_bs8_k2_lr0.01_rtol0.0001']
    assert list(miss['n_missing']) == [3]
    printed = capsys.readouterr().out
    assert '[config_table] 1 configuration(s) with no result file' in printed
    assert 'svd_bs8_k2_lr0.01_rtol0.0001' in printed and '(3 runs)' in printed

    scan = scan_of(root, 'manifestscan', tmp_path)
    assert len(scan.expected_run_ids) == len(expected) + 3
    assert len(scan.missing_run_ids) == 4
    cfg = scan.configs(scan.sven, sa.SVEN_CONFIG)
    assert dict(zip(cfg['k'], cfg['n_missing'])) == {8: 0, 4: 1}
    assert list(sa.summary_table(scan)['attempted']) == [3] * 3


def n_train_records(n_trains=(100, 1000), seeds=(1000, 1001, 1002, 1003, 1004),
                    failed_after=None, drop=()):
    """One k/lr/rtol Sven configuration at several ``n_train`` -- so ONE row of
    ``Scan.configs(df, SVEN_CONFIG)`` (k/lr/rtol only) covers several run
    configurations, as the finetune / kappa / microbatch scans do."""
    recs, expected, i = [], [], 0
    for n in n_trains:
        for seed in seeds:
            run_id = f'svd_bs8_N{n}_k8_lr0.01_rtol0.0001_mseed{seed}_lseed1000'
            expected.append(run_id)
            i += 1
            if run_id in drop:
                continue
            r = sven_record(8, 0.01, 1e-4, seed)
            r['run_id'], r['n_train'] = run_id, n
            if failed_after is not None and i > failed_after:
                r.update({'status': 'error', 'error': 'oom at step 3',
                          'losses': losses(n_epochs=1, level=1e-9)})
            recs.append(r)
    return recs, expected


def test_coarse_group_keeps_the_seed_rule_and_sums_the_manifest(root, tmp_path):
    """The expectation of a table ROW is the sum over EVERY configuration in it.

    Reading the first row's config_key (or falling back to the row's own size) both
    breaks C-A2: the first hides the missing runs of every other configuration in
    the row, the second silently replaces the binding eligibility rule -- more than
    half the SEEDS finished (section 1) -- with 'more than half of all runs here'.
    """
    recs, _ = n_train_records(failed_after=4)          # 4 of 10 runs finished
    write_scan(root, 'coarse', recs)                   # no manifest
    scan = scan_of(root, 'coarse', tmp_path)
    cfg = scan.configs(scan.sven, sa.SVEN_CONFIG)
    assert len(cfg) == 1 and len(scan.seeds) == 5
    row = cfg.iloc[0]
    assert (row['n_seeds'], row['n_failed']) == (4, 6)
    assert row['attempted'] == 5, 'no manifest: the scan seed count, not the row size'
    assert row['n_missing'] == 0, 'never negative, and unknowable without a manifest'
    assert row['eligible'] and scan.best_sven() is not None    # 4 of 5 seeds > half

    # with a manifest the row's expectation is 5 + 5, so the two absent runs show
    recs, expected = n_train_records(
        drop=[f'svd_bs8_N{n}_k8_lr0.01_rtol0.0001_mseed1004_lseed1000'
              for n in (100, 1000)])
    write_scan(root, 'coarse_manifest', recs, manifest=expected)
    scan = scan_of(root, 'coarse_manifest', tmp_path)
    row = scan.configs(scan.sven, sa.SVEN_CONFIG).iloc[0]
    assert (row['n_seeds'], row['attempted'], row['n_missing']) == (8, 10, 2)
    assert row['eligible']
    summary = sa.summary_table(scan)
    assert list(summary[summary['method'] == 'Sven'][['attempted', 'n_missing']]
                .itertuples(index=False, name=None)) == [(10, 2)]


def test_legacy_records_still_load(root, tmp_path):
    """No status, no manifest, no test curve, no n_train: everything must behave
    exactly as before -- divergence from the curves, n_missing from the seed count."""
    recs = grid_records(with_provenance=False)
    nan_run = next(r for r in recs if r['optimizer'] == 'SGD' and r['model_seed'] == 1000)
    nan_run['losses']['val'][-1] = float('nan')
    recs.remove(next(r for r in recs if r['optimizer'] == 'Adam'
                     and r['model_seed'] == 1002))
    write_scan(root, 'legacy', recs)
    df = load(root, 'legacy')

    assert 'status' not in df.columns and not style.manifest_run_ids('legacy')
    assert df['losses'].apply(lambda L: 'test' in L).sum() == 0
    tab = ah.config_table(df)
    sgd = tab[tab['method'] == 'SGD'].iloc[0]
    assert (sgd['n_seeds'], sgd['n_diverged'], sgd['n_failed']) == (2, 1, 0)
    adam = tab[tab['method'] == 'Adam'].iloc[0]
    assert (adam['attempted'], adam['n_seeds'], adam['n_missing']) == (3, 2, 1)
    assert 'final_test_loss' not in tab.columns

    scan = scan_of(root, 'legacy', tmp_path)
    assert not scan.has_test and scan.n_epochs == N_EPOCHS
    assert sa.resolve_time_key(scan) == 'epoch_times'      # no train_times recorded
    # both Sven configs score the same here, so the documented tie-break applies:
    # smallest k, largest rtol -- the cheapest of the tied twins
    best = scan.best_sven()
    assert (int(best['k']), float(best['lr']), float(best['rtol'])) == (4, 0.01, 1e-4)
    cfg = scan.configs(scan.baseline, sa.BASELINE_CONFIG)
    assert dict(zip(cfg['optimizer'], cfg['n_missing'])) == {'Adam': 1, 'SGD': 0}
    # an empty selection (a k / lr combination the scan does not contain) must give
    # an empty table, not raise -- plot_time_vs_k walks combinations that may be absent
    assert scan.configs(scan.runs(lr=99.0), ['k']).empty


# ---------------------------------------------------------------------------
# 4. Partial curves must not mis-size an axis
# ---------------------------------------------------------------------------
def test_partial_curve_robustness(root, tmp_path):
    """`Scan.n_epochs` and `epoch_axis` used to read ``rows.iloc[0]``; a record that
    stopped early (C-R1) then shrank every epoch axis in the scan."""
    recs = grid_records()
    recs[0]['losses'] = losses(n_epochs=1, level=1.0)          # first row: 1 epoch
    recs[0]['status'] = 'diverged'
    write_scan(root, 'partial', recs)
    scan = scan_of(root, 'partial', tmp_path)
    assert scan.n_epochs == N_EPOCHS

    rows = scan.sven_rows(k=8, lr=0.01, rtol=1e-4)
    assert len(rows) == len(SEEDS)
    mean, _, _ = sa.seed_band(rows, 'val')
    assert len(mean) == N_EPOCHS + 1        # the 1-epoch run is diverged -> excluded
    x = sa.epoch_axis(rows, mean, 'val')
    assert list(x) == list(range(N_EPOCHS + 1))
    train, _ = sa.seed_curve(rows, 'train')
    assert list(sa.epoch_axis(rows, train, 'train')) == list(range(1, N_EPOCHS + 1))

    # a truncated val curve is still recognised as starting at epoch 0, by key
    short = np.asarray(mean[:2])
    assert list(sa.epoch_axis(rows, short, 'val')) == [0, 1]
    assert list(sa.epoch_axis(rows, short, 'train')) == [1, 2]

    # sv_diagnostics.n_epochs tolerates a record with no train curve at all
    assert sv.n_epochs({'losses': {'train': [1.0, 2.0]}}) == 2
    assert sv.n_epochs({'losses': {}}) == 0
    assert sv.n_epochs({}) == 0


# ---------------------------------------------------------------------------
# 5. One results root, and a read-only one still loads (4.1)
# ---------------------------------------------------------------------------
def test_results_root_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv(style.RESULTS_ROOT_ENV, raising=False)
    assert style.resolve_results_root() == style.DEFAULT_RESULTS_ROOT == '../experiment_results'
    monkeypatch.setenv(style.RESULTS_ROOT_ENV, str(tmp_path))
    assert style.resolve_results_root() == str(tmp_path)
    assert style.resolve_results_root('/explicit') == '/explicit'

    write_scan(tmp_path, 'envscan', grid_records())
    df = style.load_results('envscan')            # no results_root= anywhere
    assert len(df) == len(SEEDS) * 4
    scan = sa.load_scan('envscan', 'env', tmp_path / 'plots')
    assert scan.results_root == str(tmp_path)


def test_read_only_root_still_loads(root, capsys):
    """A genuinely read-only legacy root used to raise in the unguarded cache write
    after the load had already succeeded."""
    write_scan(root, 'ro', grid_records())
    mode = root.stat().st_mode
    os.chmod(root, 0o555)
    try:
        style._cache_warned.clear()
        df = style.load_results('ro')
        assert len(df) == len(SEEDS) * 4
        assert 'could not write the slim cache' in capsys.readouterr().out
        assert not (root / '_cache').exists()
    finally:
        os.chmod(root, mode)


def test_cache_dir_redirects_the_cache(root, tmp_path):
    write_scan(root, 'cached', grid_records())
    elsewhere = tmp_path / 'cache'
    style.load_results('cached', cache_dir=elsewhere)
    assert (elsewhere / 'cached.slim.pkl').is_file()
    assert not (root / '_cache').exists()
    again = style.load_results('cached', cache_dir=elsewhere)   # served from there
    assert len(again) == len(SEEDS) * 4


# ---------------------------------------------------------------------------
# 6. F20: sv_min and sv_min_kept never merge
# ---------------------------------------------------------------------------
def test_sv_min_kept_is_separate_from_sv_min(root):
    run_id = 'svd_bs8_k4_lr0.01_rtol0.0001_mseed1000_lseed1000'
    rec = sven_record(4, 0.01, 1e-4, 1000,
                      diag_file=f'diag/{run_id}.npz',
                      svd_summary={'n_steps': 8, 'spectra_every': 2})
    arrays = {'svs': np.array([[1.0, 0.5, 0.25, 1e-9]] * 4, dtype=np.float32),
              'svs_step': np.array([0, 2, 4, 6], dtype=np.int32),
              'sv_max': np.ones(8, dtype=np.float32),
              'sv_min': np.full(8, 1e-9, dtype=np.float32),        # sigma_B (noise)
              'sv_min_kept': np.full(8, 0.25, dtype=np.float32)}   # smallest inverted
    write_scan(root, 'f20', [rec], diag={run_id: arrays})

    diag = style.load_diagnostics(rec, name='f20')
    assert diag['sv_min'][0] == pytest.approx(1e-9)
    assert diag['sv_min_kept'][0] == pytest.approx(0.25)
    assert style.SV_MIN_KEYS == ('sv_min', 'sv_min_kept')

    fat = style.load_results('f20', slim=False).iloc[0]
    assert fat['svd_info']['sv_min'][0] == pytest.approx(1e-9)
    assert fat['svd_info']['sv_min_kept'][0] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 7. Spectrum plots: full width vs legacy truncated (C-A3)
# ---------------------------------------------------------------------------
def spectra_scan(root, name, width, k=4, B=8, with_utr=True):
    """One Sven config whose stored spectra are ``width`` wide."""
    recs, diag = [], {}
    for seed in SEEDS[:2]:
        rec = sven_record(k, 0.01, 1e-4, seed)
        rec['batch_size'] = B
        run_id = rec['run_id']
        rec['diag_file'] = f'diag/{run_id}.npz'
        rec['svd_summary'] = {'n_steps': 8, 'spectra_every': 2}
        sigma = np.geomspace(1.0, 1e-9, width).astype(np.float32)
        arrays = {'svs': np.tile(sigma, (4, 1)),
                  'svs_step': np.array([0, 2, 4, 6], dtype=np.int32)}
        if with_utr:
            arrays['utr'] = np.tile(np.geomspace(1.0, 1e-6, width).astype(np.float32),
                                    (4, 1))
        recs.append(rec)
        diag[run_id] = arrays
    write_scan(root, name, recs, diag=diag)
    return style.load_results(name)


def labels_of(ax):
    return [ln.get_label() for ln in ax.lines]


def legend_labels(ax):
    """What a reader sees in ``ax.legend()`` -- lines, bars, error bars AND the
    shaded-band proxy patches, which are not in ``ax.lines``."""
    return ax.get_legend_handles_labels()[1]


def test_full_width_spectrum_plot_marks_k_rtol_and_floor(root):
    """Full spectra (C-L1): x normalised by the WIDTH (so the axis really ends at
    1), a vertical line at the k cut, horizontals at rtol and the float32 floor."""
    df = spectra_scan(root, 'fullspec', width=8, k=4, B=8)
    fig, ax = plt.subplots()
    spectra, norm = sv.plot_epoch_spectra(df, ax, k=4, lr=0.01, rtol=1e-4, floor=1e-12)
    assert spectra.shape == (N_EPOCHS, 8)
    labs = labels_of(ax)
    assert any('rtol' in ln for ln in labs) and any('float32' in ln for ln in labs)
    assert '$k = 4$' in labs

    kline = next(ln for ln in ax.lines if ln.get_label() == '$k = 4$')
    assert kline.get_xdata()[0] == pytest.approx(3 / 7)      # index k-1 of width-1
    floor = next(ln for ln in ax.lines if 'float32' in ln.get_label())
    assert floor.get_ydata()[0] == pytest.approx(style.FLOAT32_NOISE_FLOOR) == 1e-7
    rtol = next(ln for ln in ax.lines if 'rtol' in ln.get_label())
    assert rtol.get_ydata()[0] == pytest.approx(1e-4)

    data = [ln for ln in ax.lines if ln.get_label().startswith('_')]
    assert len(data) == N_EPOCHS
    assert max(ln.get_xdata().max() for ln in data) == pytest.approx(1.0)
    assert ax.get_xlabel().endswith('width')
    plt.close(fig)


def test_full_width_spectrum_default_floor_shows_the_tail(root, tmp_path):
    """What the notebooks actually call: ``plot_sv_spectra`` with no ``floor``.  The
    old fixed 1e-4 clipped a full spectrum (which runs down to round-off) and
    suppressed the float32 line with it, so the default has to follow the record."""
    spectra_scan(root, 'fullspec_default', width=8, k=4, B=8)
    scan = scan_of(root, 'fullspec_default', tmp_path)
    fig, ax = plt.subplots()
    spectra = sa.plot_sv_spectra(scan, fig, ax, lr=0.01, rtol=1e-4, k=4,
                                 colorbar=False)
    labs = labels_of(ax)
    assert any('float32' in ln for ln in labs), 'the C-A3 noise-floor line'
    assert any('rtol' in ln for ln in labs) and '$k = 4$' in labs
    assert ax.get_ylim()[0] < 1e-9, 'the 1e-9 tail must be inside the axes'
    data = [ln for ln in ax.lines if ln.get_label().startswith('_')]
    assert min(ln.get_ydata().min() for ln in data) == pytest.approx(1e-9, rel=1e-3)
    assert [t.get_text() for t in ax.get_xticklabels()][1] == '$B/4$'   # width == B
    plt.close(fig)

    # a full spectrum narrower than B (micro-batched Gram: M = B / microbatch) is
    # labelled in units of its own width, not B
    spectra_scan(root, 'microspec', width=4, k=2, B=8)
    micro = scan_of(root, 'microspec', tmp_path)
    fig, ax = plt.subplots()
    sa.plot_sv_spectra(micro, fig, ax, lr=0.01, rtol=1e-4, k=2, colorbar=False)
    assert [t.get_text() for t in ax.get_xticklabels()][1] == '$W/4$'
    plt.close(fig)

    # an explicit floor still wins (comparisons.ipynb passes floor=1e-4)
    fig, ax = plt.subplots()
    sa.plot_sv_spectra(scan, fig, ax, lr=0.01, rtol=1e-4, k=4, floor=1e-4,
                       colorbar=False)
    assert ax.get_ylim() == (1e-4, 2)
    plt.close(fig)


def test_legacy_truncated_spectrum_plot_unchanged(root, capsys):
    """A legacy record stored only the SVs above rtol: the axis stays normalised by
    k (so a partial spectrum is not stretched over the whole axis), the truncation
    warning still fires, and no k / floor line is drawn."""
    df = spectra_scan(root, 'truncspec', width=3, k=8, B=8, with_utr=False)
    fig, ax = plt.subplots()
    spectra, _ = sv.plot_epoch_spectra(df, ax, k=8, lr=0.01, rtol=1e-4, floor=1e-12)
    assert spectra.shape == (N_EPOCHS, 3)
    assert 'truncated at rtol' in capsys.readouterr().out
    labs = labels_of(ax)
    assert '$k = 8$' not in labs and not any('float32' in ln for ln in labs)
    assert any('rtol' in ln for ln in labs)
    data = [ln for ln in ax.lines if ln.get_label().startswith('_')]
    assert max(ln.get_xdata().max() for ln in data) == pytest.approx(2 / 7)
    assert ax.get_xlabel().endswith('$k$')
    plt.close(fig)


def test_legacy_truncated_rank_ticks_are_in_units_of_k(root, tmp_path):
    """The tick CAPTION must name the denominator the axis really used.  A truncated
    record is normalised by ``k``, so labelling its ticks in units of the stored
    width (30 of k=64, say) reads x=0.25 as W/4 when it is k/4 -- every legacy
    spectrum figure in the four scan notebooks."""
    spectra_scan(root, 'trunc_ticks', width=3, k=4, B=8)      # k != B, width < k
    scan = scan_of(root, 'trunc_ticks', tmp_path)
    fig, ax = plt.subplots()
    spectra = sa.plot_sv_spectra(scan, fig, ax, lr=0.01, rtol=1e-4, k=4,
                                 colorbar=False)
    assert spectra.shape == (N_EPOCHS, 3)
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        '$1$', '$k/4$', '$k/2$', '$3k/4$', '$k$']
    data = [ln for ln in ax.lines if ln.get_label().startswith('_')]
    assert max(ln.get_xdata().max() for ln in data) == pytest.approx(2 / 3)  # /(k-1)
    assert ax.get_ylim() == (1e-4, 2), 'legacy y range unchanged'
    plt.close(fig)

    # k = B, the usual legacy case: the old 'B' caption, and it is still right
    spectra_scan(root, 'trunc_ticks_kB', width=3, k=8, B=8)
    scan = scan_of(root, 'trunc_ticks_kB', tmp_path)
    fig, ax = plt.subplots()
    sa.plot_sv_spectra(scan, fig, ax, lr=0.01, rtol=1e-4, k=8, colorbar=False)
    assert [t.get_text() for t in ax.get_xticklabels()][1] == '$B/4$'
    plt.close(fig)


def test_utr_plot(root, tmp_path):
    """|u_i . r| vs index with the k cut -- and nothing at all on legacy records."""
    df = spectra_scan(root, 'utrspec', width=8, k=4, B=8)
    fig, ax = plt.subplots()
    utr, norm = sv.plot_epoch_utr(df, ax, k=4, lr=0.01, rtol=1e-4)
    assert utr.shape == (N_EPOCHS, 8)
    assert utr[0][0] > utr[0][-1]                       # descending overlap
    assert np.nansum(utr[0]) == pytest.approx(np.nansum(np.abs(
        np.geomspace(1.0, 1e-6, 8)) / np.linalg.norm(np.geomspace(1.0, 1e-6, 8))), rel=1e-5)
    kline = next(ln for ln in ax.lines if ln.get_label() == '$k = 4$')
    assert kline.get_xdata()[0] == pytest.approx(3 / 7)
    assert '|u_i' in ax.get_ylabel()
    plt.close(fig)

    no_utr = spectra_scan(root, 'noutr', width=8, k=4, B=8, with_utr=False)
    fig, ax = plt.subplots()
    assert sv.plot_epoch_utr(no_utr, ax, k=4, lr=0.01, rtol=1e-4) == (None, None)
    plt.close(fig)

    scan = scan_of(root, 'utrspec', tmp_path)
    fig, ax = plt.subplots()
    assert sa.plot_sv_utr(scan, fig, ax, lr=0.01, rtol=1e-4, k=4).shape == (N_EPOCHS, 8)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Epoch / step / examples / synchronised-time axes (C-A5)
# ---------------------------------------------------------------------------
def test_step_and_example_axes(root, tmp_path):
    recs = grid_records()
    for r in recs:
        # consistent with batch_size 8 (drop_last: 25 steps x 8 = 200 examples)
        r['steps_per_epoch'] = 25
        r['n_train'] = 200
        r['losses'] = losses(level=r['lr'] * 100, times=True)
    write_scan(root, 'axes', recs)
    scan = scan_of(root, 'axes', tmp_path)
    rows = scan.sven_rows(k=8, lr=0.01, rtol=1e-4)

    assert sa.steps_per_epoch(rows) == 25
    assert sa.examples_per_epoch(rows) == 200
    mean, _ = sa.seed_curve(rows, 'train')
    assert list(sa.epoch_axis(rows, mean, 'train', 'step')) == [25, 50, 75, 100]
    assert list(sa.epoch_axis(rows, mean, 'train', 'examples')) == [200, 400, 600, 800]
    val, _ = sa.seed_curve(rows, 'val')
    assert list(sa.epoch_axis(rows, val, 'val', 'step')) == [0, 25, 50, 75, 100]

    # train_times (C-T1) is the time key when it is recorded
    assert sa.resolve_time_key(scan) == 'train_times'
    assert 'synchronised' in sa.time_axis_label('train_times')
    x = sa.epoch_axis(rows, val, 'val', 'time', 'train_times')
    assert list(x) == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert sa.axis_label('step') == 'Optimizer steps'
    assert sa.axis_label('examples') == 'Examples processed'

    fig, ax = plt.subplots()
    chosen = sa.plot_best_curves(scan, ax, which='train', versus='step')
    assert chosen and ax.get_xlabel() == 'Optimizer steps'
    plt.close(fig)


def test_step_axis_falls_back_to_n_train_and_refuses_to_guess(root, tmp_path):
    recs = grid_records()
    for r in recs:                       # no steps_per_epoch: derive from n_train
        r['n_train'] = 800
    write_scan(root, 'axes_fallback', recs)
    scan = scan_of(root, 'axes_fallback', tmp_path)
    rows = scan.sven_rows(k=8, lr=0.01, rtol=1e-4)
    assert sa.steps_per_epoch(rows) == 100          # 800 // batch_size 8
    assert sa.examples_per_epoch(rows) == 800

    # drop_last=True (C-S3): the examples axis counts what was STEPPED on, so an
    # n_train that is not a multiple of the batch size does not inflate it
    recs = grid_records()
    for r in recs:
        r['n_train'] = 205                          # 25 full batches of 8, 5 dropped
    write_scan(root, 'axes_droplast', recs)
    dl = scan_of(root, 'axes_droplast', tmp_path).sven_rows(k=8, lr=0.01, rtol=1e-4)
    assert sa.steps_per_epoch(dl) == 25
    assert sa.examples_per_epoch(dl) == 200         # not 205
    # ... and the small-N case, where n_train would be 56% too high
    for r in recs:
        r['n_train'], r['batch_size'] = 100, 64
        r['run_id'] = r['run_id'].replace('bs8', 'bs64')
    write_scan(root, 'axes_smalln', recs)
    sn = scan_of(root, 'axes_smalln', tmp_path).sven_rows(k=8, lr=0.01, rtol=1e-4)
    assert (sa.steps_per_epoch(sn), sa.examples_per_epoch(sn)) == (1, 64)

    write_scan(root, 'axes_none', grid_records())   # neither recorded
    bare = scan_of(root, 'axes_none', tmp_path)
    brows = bare.sven_rows(k=8, lr=0.01, rtol=1e-4)
    mean, _ = sa.seed_curve(brows, 'train')
    assert sa.steps_per_epoch(brows) is None
    assert sa.epoch_axis(brows, mean, 'train', 'step') is None     # never guessed
    assert list(sa.epoch_axis(brows, mean, 'train')) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 9. Nothing the notebooks call may have broken (they cannot be edited here)
# ---------------------------------------------------------------------------
def test_notebook_surface_smoke(root, tmp_path):
    """Every helper the scan / study notebooks call, once, on a schema-2 scan with
    a manifest, a diverged run, an oom run, test curves and train_times."""
    recs = grid_records(with_provenance=True)
    for r in recs:
        r['losses'] = losses(level=r['lr'] * 100, test=True, times=True)
        r['microbatch_size'] = 8
    recs[-1].update({'status': 'oom', 'losses': losses(n_epochs=1, level=1e-9)})
    recs[-2].update({'status': 'diverged', 'losses': losses(n_epochs=2, level=1e5)})
    write_scan(root, 'smoke', recs, manifest=[r['run_id'] for r in recs])
    # a standalone timing scan for the same run_ids (attach_standalone_times)
    write_scan(root, 'smoke_timing', [dict(r) for r in recs[:4]])

    scan = scan_of(root, 'smoke', tmp_path)
    report = sa.attach_standalone_times(scan)
    assert report['n_matched'] == 4 and sa.has_standalone(scan)
    assert sa.resolve_time_key(scan) == 'epoch_times_standalone'

    def fresh():
        fig, ax = plt.subplots()
        plt.close(fig)
        return fig, ax

    for versus in ('epoch', 'step', 'examples', 'time'):
        assert sa.plot_best_curves(scan, fresh()[1], which='val', versus=versus)
    assert sa.plot_k_sweep(scan, fresh()[1], lr=0.01, rtol=1e-4)
    assert sa.plot_sensitivity(scan, fresh()[1], x='lr', lines='k') is not None
    assert sa.plot_hparam_heatmap(scan, fresh()[1]) is not None
    assert sa.plot_time_summary(scan, fresh()[1])[2] == 'standalone'
    assert sa.plot_time_comparison(scan, fresh()[1])
    assert sa.plot_time_vs_k(scan, fresh()[1]) == 1e-4
    sa.plot_efficiency(scan, fresh()[1])
    assert sa.knob_ref_lr(scan, 'microbatch_size', 8) == 0.01
    assert sa.plot_knob_summary(scan, fresh()[1], 'microbatch_size', lr=0.01) is not None
    assert sa.plot_knob_curves(scan, fresh()[1], 'microbatch_size', lr=0.01) == [8.0]

    summary = sa.summary_table(scan).set_index('method')
    assert set(summary.index) == {'Sven', 'Adam', 'SGD'}   # each lost only 1 of 3
    assert list(summary['attempted']) == [3, 3, 3]
    assert (int(summary.loc['Adam', 'n_diverged']), int(summary.loc['Adam', 'n_failed']),
            int(summary.loc['Adam', 'finished'])) == (1, 0, 2)
    assert (int(summary.loc['SGD', 'n_diverged']), int(summary.loc['SGD', 'n_failed']),
            int(summary.loc['SGD', 'finished'])) == (0, 1, 2)
    assert {'n_missing', 'final_test_loss', 'final_val_acc'} <= set(summary.columns)

    df = ah.add_derived(load(root, 'smoke'))
    assert ah.best_per_method(df)['method'].is_unique
    assert len(ah.seed_mean_best(df, 'k')) == 3       # k = 4, 8 and NaN (baselines)
    assert not ah.config_runs(df, ah.best_per_method(df).iloc[0]).empty
    assert ah.fmt_pm(ah.best_per_method(df).iloc[0], 'final_val_loss')
    table, target = ah.epochs_to_target_table(df, group=('method',))
    assert target > 0 and len(table) == len(set(df['method']))
    fig, ax = plt.subplots()
    ah.errorbar_seeds(ax, ah.best_per_method(df), 'lr', 'final_val_loss')
    plt.close(fig)
    assert ah.n_params_of(df, fallback=1) == 321

    sven = sv.sven_runs(df)
    assert len(sv.select(sven, k=8)) == len(SEEDS)
    assert sv.rank_per_epoch(sven.iloc[0]) is None        # no svd_summary here
    assert sv.smooth(np.arange(10.0), 3)[0].shape == (8,)


# ---------------------------------------------------------------------------
# 10. The real scans: the allow-list must not change the grouping (run via
#     campaign/run_cpu_tests.sh; a cold load of these is 15-25 s on Lustre)
# ---------------------------------------------------------------------------
LEGACY_NOT_CONFIG = {'model_seed', 'loader_seed', 'run_id', 'diag_file', 'method',
                     'diverged', '_scan', 'final_val_loss', 'final_train_loss',
                     'final_val_acc', 'total_time', 'peak_gpu_mem_mb', 'val_ppl',
                     'gram_capture', 'gram_chunk_numel', 'losses', 'svd_summary'}


def legacy_config_columns(df):
    """The auto-detection rule as it was before the allow-list -- the oracle."""
    scalar = (str, bool, int, float, np.integer, np.floating)
    return [c for c in df.columns if c not in LEGACY_NOT_CONFIG and df[c].notna().any()
            and df[c].dropna().map(lambda v: isinstance(v, scalar)).all()]


@pytest.mark.skipif(os.environ.get('SV3_CHECK_REAL_SCANS') != '1',
                    reason='set SV3_CHECK_REAL_SCANS=1 (slow: Lustre)')
@pytest.mark.parametrize('name', ['toy_1d_scan', 'mnist_scan_ce'])
def test_real_scan_grouping_unchanged(name):
    """On the real results the allow-list must partition the runs EXACTLY as
    auto-detection did (same number of groups, same sizes) -- and as the run_ids
    themselves say (:func:`style.config_key`), which is what the manifest counts
    against.

    Checked over all 33 directories of the legacy root when this landed: the
    partition is unchanged everywhere.  The run_id partition agrees on every
    headline scan; it is COARSER on five legacy directories (critbatch, nanogpt
    speedrun, two microbatch/paramfrac scans) whose jsonl files hold two record
    generations with different columns for the same configuration -- F16's schema
    drift, and the run_id reading is the right one there.
    """
    root = os.environ.get('SV3_LEGACY_RESULTS_ROOT',
                          str(REPO / 'experiment_results_legacy_2026-09-18'))
    df = style.load_results(name, results_root=root,
                            cache_dir=os.environ.get('SV3_CACHE_DIR'))
    old = sorted(len(g) for _, g in df.groupby(legacy_config_columns(df), dropna=False))
    new = sorted(len(g) for _, g in df.groupby(ah.config_columns(df), dropna=False))
    by_id = sorted(style.expected_per_config(df['run_id']).values())
    print(f'{name}: {len(df)} runs, {len(old)} groups; sizes {sorted(set(old))}')
    assert new == old, f'{name}: grouping changed ({len(new)} vs {len(old)} groups)'
    assert new == by_id, (f'{name}: column grouping ({len(new)}) disagrees with the '
                          f'run_ids ({len(by_id)}) -- two record generations in one '
                          f'configuration?')
    assert len(ah.config_table(df)) == len(old)


@pytest.mark.skipif(os.environ.get('SV3_CHECK_REAL_SCANS') != '1',
                    reason='set SV3_CHECK_REAL_SCANS=1 (slow: Lustre)')
@pytest.mark.parametrize('name', ['toy_1d_scan', 'mnist_scan_ce',
                                  'cifar10_resnet_ce_scan', 'mnist_scan_ce_confirm'])
def test_real_fresh_scan_grouping_matches_run_ids(name, capsys):
    """The same check on the CAMPAIGN root, where auto-detection is no longer a
    usable oracle -- that is the whole point of the allow-list.

    Schema 2 puts `run_hash`, `effective_loader_seed`, `val_final`, `test`, ... on
    every record, so the pre-C-A2 rule makes EVERY RUN its own configuration
    (mnist_scan_ce: 1610 runs -> 1610 "configurations" instead of 322).  The
    invariant that still holds, and the one the manifest counts against, is the
    run_id partition (:func:`style.config_key`).  This also asserts that loading a
    fresh scan prints no "[style] ... AVERAGED OVER" warning (WP1.1).
    """
    root = os.environ.get('SV3_RESULTS_ROOT_FRESH', str(REPO / 'experiment_results'))
    df = style.load_results(name, results_root=root,
                            cache_dir=os.environ.get('SV3_CACHE_DIR'))
    capsys.readouterr()                      # drop the loader's own chatter
    cols = ah.config_columns(df)
    assert 'AVERAGED OVER' not in capsys.readouterr().out
    new = sorted(len(g) for _, g in df.groupby(cols, dropna=False))
    by_id = sorted(style.expected_per_config(df['run_id']).values())
    fragmented = df.groupby(legacy_config_columns(df), dropna=False).ngroups
    print(f'{name}: {len(df)} runs, {len(new)} configurations by columns, '
          f'{len(by_id)} by run_id, {fragmented} under auto-detection')
    assert new == by_id, (f'{name}: column grouping ({len(new)}) disagrees with the '
                          f'run_ids ({len(by_id)})')
    assert fragmented == len(df), (f'{name}: auto-detection no longer fragments '
                                   f'({fragmented} of {len(df)}) -- has a schema-2 '
                                   f'column been dropped from the records?')


# ---------------------------------------------------------------------------
# 11. The method registry: a colour and a display name for every optimizer in
#     the campaign records (WP1.2; C-B2 SGDm, C-B7 stochastic L-BFGS, C-A6)
# ---------------------------------------------------------------------------
#: Every `optimizer` value in the fresh results (all 43 directories of
#: `experiment_results/`, 2026-09-20), as `style.canonical_method` spells it.
CAMPAIGN_METHODS = ('Sven', 'SGD', 'SGDm', 'PolyakSGD', 'RMSprop', 'Adam', 'AdamW',
                    'LBFGS', 'Muon', 'MuonW', 'SOAP', 'Shampoo', 'KFAC', 'JD', 'HIG')


def test_every_campaign_method_has_a_colour(capsys):
    """No method may fall back to the unknown-optimizer grey: that is how a new
    baseline (SGDm) ends up the same colour as another one."""
    for m in CAMPAIGN_METHODS:
        style.method_color._warned.discard(m)
    colours = {m: style.method_color(m) for m in CAMPAIGN_METHODS}
    assert 'no colour registered' not in capsys.readouterr().out
    assert colours['Sven'] == '#000000'                      # Sven is black, always
    assert len(set(colours.values())) == len(colours)        # no two methods share one
    # the record spellings reach the same entries
    assert style.method_color('SVD') == colours['Sven']
    assert style.method_color('JD_UPGrad') == colours['JD']


def test_method_labels(root, tmp_path):
    """`LBFGS` is minibatch L-BFGS and must say so wherever a reader sees it
    (C-B7); the machine key stays the record's own spelling."""
    assert style.method_label('LBFGS') == 'Stochastic L-BFGS'
    assert style.method_label('LBFGS1') == 'Stochastic L-BFGS'      # alias
    assert style.method_label('SGDm') == 'SGD + momentum'
    assert style.method_label('SVD') == 'Sven'
    assert style.method_label('Adam') == 'Adam'                     # unregistered: unchanged
    assert style.method_label('BrandNewOptimizer') == 'BrandNewOptimizer'
    assert ah.method_label is style.method_label                    # one implementation

    recs = [r for seed in SEEDS for r in
            (sven_record(8, 0.01, 1e-4, seed), baseline_record('LBFGS', 0.1, seed),
             baseline_record('SGDm', 0.1, seed))]
    write_scan(root, 'labels', recs)
    scan = scan_of(root, 'labels', tmp_path)

    t = sa.summary_table(scan).set_index('method')
    assert list(t.columns)[0] == 'label'                            # right after `method`
    assert t.loc['LBFGS', 'label'] == 'Stochastic L-BFGS'
    assert t.loc['SGDm', 'label'] == 'SGD + momentum'
    assert t.loc['Sven', 'label'] == 'Sven'

    # plots: legend entries and tick labels carry the display name, the returned
    # dicts keep the machine key (so notebook code indexing by 'LBFGS' still works)
    fig, ax = plt.subplots()
    chosen = sa.plot_best_curves(scan, ax, which='val')
    assert {'LBFGS', 'SGDm'} <= set(chosen)
    legend = {t.get_text() for t in ax.legend().get_texts()}
    assert 'Stochastic L-BFGS' in legend and 'SGD + momentum' in legend
    plt.close(fig)

    fig, ax = plt.subplots()
    means, _, _ = sa.plot_time_summary(scan, ax, quantity='total_time')
    assert {'LBFGS', 'SGDm'} <= set(means)
    assert 'Stochastic L-BFGS' in {t.get_text() for t in ax.get_xticklabels()}
    plt.close(fig)


def test_seed_spread_label_is_the_one_band_legend(root, tmp_path):
    """C-A6: every seed band says the same thing, from one constant -- and it is on
    the FIGURE, not only in a module.  Pinning the string alone let every band ship
    unlabelled: `scan_analysis`'s four `fill_between` calls and the clipped error
    bars carried no legend entry at all, so no figure in the paper said what its
    shaded band was."""
    import paired
    assert paired.SEED_SPREAD_LABEL == r'$\pm$ 1 std over seeds'
    assert style.seed_spread_label() == paired.SEED_SPREAD_LABEL     # one source
    assert style.seed_spread_label(plain=True) == '± 1 std over seeds'
    assert ah.seed_spread_label is style.seed_spread_label
    LABEL = paired.SEED_SPREAD_LABEL

    # `band_legend` adds exactly one entry, however many times it is called, and
    # its proxy has no data, so it cannot move an axis (log axes included)
    fig, ax = plt.subplots()
    ax.plot([3, 4], [10, 20])
    fig.canvas.draw()
    before = (ax.get_xlim(), ax.get_ylim())
    assert style.band_legend(ax) is not None
    assert style.band_legend(ax) is None
    fig.canvas.draw()
    assert (ax.get_xlim(), ax.get_ylim()) == before
    assert legend_labels(ax).count(LABEL) == 1
    plt.close(fig)

    recs = [r for seed in SEEDS for r in
            (sven_record(8, 0.01, 1e-4, seed), sven_record(4, 0.01, 1e-4, seed),
             baseline_record('Adam', 0.001, seed), baseline_record('SGD', 0.1, seed))]
    write_scan(root, 'bands', recs)
    scan = scan_of(root, 'bands', tmp_path)

    # every helper that draws seed uncertainty names it
    for draw in (lambda ax: sa.plot_best_curves(scan, ax, which='val'),
                 lambda ax: sa.plot_k_sweep(scan, ax, lr=0.01, rtol=1e-4, which='val'),
                 lambda ax: sa.plot_sensitivity(scan, ax, x='k', lines='lr'),
                 lambda ax: sa.plot_time_summary(scan, ax, quantity='total_time'),
                 lambda ax: sa.plot_time_vs_k(scan, ax, quantity='total_time')):
        fig, ax = plt.subplots()
        draw(ax)
        assert legend_labels(ax).count(LABEL) == 1, draw
        plt.close(fig)

    # ... and so does the notebooks' own error-bar helper (one entry for N methods)
    tab = ah.best_per_method(ah.add_derived(load(root, 'bands')), by='final_val_loss',
                             extra_group=['batch_size'])
    fig, ax = plt.subplots()
    for m in tab['method'].unique():
        ah.errorbar_seeds(ax, tab[tab.method == m], 'batch_size', 'final_val_loss', label=m)
    assert legend_labels(ax).count(LABEL) == 1
    plt.close(fig)

    # a band drawn with band=False is not labelled (nothing to name)
    fig, ax = plt.subplots()
    sa.plot_best_curves(scan, ax, which='val', band=False)
    assert LABEL not in legend_labels(ax)
    plt.close(fig)


def test_errorbar_seeds_shows_display_names(root):
    """C-B7 reaches the notebook legends too: every notebook passes `label=m`
    straight from the `method` column, so the mapping belongs in the helper -- that
    is what kept "LBFGS" in the legends of overparam / batchsize / critbatch /
    finetune while the tables beside them said "Stochastic L-BFGS"."""
    recs = [r for seed in SEEDS for r in
            (sven_record(8, 0.01, 1e-4, seed), baseline_record('LBFGS', 0.1, seed),
             baseline_record('SGDm', 0.1, seed))]
    write_scan(root, 'labelled_bars', recs)
    tab = ah.best_per_method(ah.add_derived(load(root, 'labelled_bars')),
                             by='final_val_loss', extra_group=['batch_size'])
    fig, ax = plt.subplots()
    for m in tab['method'].unique():
        ah.errorbar_seeds(ax, tab[tab.method == m], 'batch_size', 'final_val_loss', label=m)
    shown = legend_labels(ax)
    assert 'Stochastic L-BFGS' in shown and 'SGD + momentum' in shown and 'Sven' in shown
    assert 'LBFGS' not in shown and 'SGDm' not in shown
    plt.close(fig)

    # a label that is not a method key passes through untouched
    fig, ax = plt.subplots()
    ah.errorbar_seeds(ax, tab[tab.method == 'Sven'], 'batch_size', 'final_val_loss',
                      label='$N=100$')
    assert '$N=100$' in legend_labels(ax)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 12. A frame remembers WHICH root it was loaded from (foreign-root loads)
# ---------------------------------------------------------------------------
def test_expected_run_ids_follow_the_frames_own_root(tmp_path, monkeypatch):
    """A frame loaded from a foreign root must not read the DEFAULT root's manifest.

    `critbatch_analysis` / `finetune_analysis` load the legacy root explicitly, and
    the legacy-vs-fresh diff loads scans that exist in BOTH roots; the helpers they
    hand the frame to take no root argument.  Before `_results_root`, a legacy
    `mnist_scan_ce` frame (1040 runs) read the FRESH manifest: 1610 expected
    run_ids and 122 "missing" configurations that were fresh-campaign grid points
    the legacy scan never intended."""
    fresh, legacy = tmp_path / 'fresh', tmp_path / 'legacy'
    for d in (fresh, legacy):
        d.mkdir()
    monkeypatch.setenv(style.RESULTS_ROOT_ENV, str(fresh))

    # the same scan NAME in both roots: the legacy copy has 2 seeds, the fresh one 3
    # plus a whole extra configuration
    legacy_recs = [sven_record(8, 0.01, 1e-4, s) for s in SEEDS[:2]]
    fresh_recs = [sven_record(k, 0.01, 1e-4, s) for s in SEEDS for k in (8, 4)]
    write_scan(legacy, 'both', legacy_recs, manifest=[r['run_id'] for r in legacy_recs])
    write_scan(fresh, 'both', fresh_recs, manifest=[r['run_id'] for r in fresh_recs])

    df_leg = style.load_results('both', results_root=str(legacy))
    df_fresh = style.load_results('both')
    assert style.frame_results_root(df_leg) == str(legacy)
    assert style.frame_results_root(df_fresh) == str(fresh)
    assert '_results_root' in style.PROVENANCE_COLUMNS      # never a config column
    assert '_results_root' not in ah.config_columns(df_leg)

    assert ah.expected_run_ids(df_leg) == {r['run_id'] for r in legacy_recs}
    assert ah.expected_run_ids(df_fresh) == {r['run_id'] for r in fresh_recs}
    assert ah.missing_configs(df_leg).empty          # was: the fresh grid's k=4 config
    # an explicit root still wins over the frame's own
    assert ah.expected_run_ids(df_leg, results_root=str(fresh)) == \
        {r['run_id'] for r in fresh_recs}

    row = ah.config_table(df_leg).iloc[0]
    assert (row['n_seeds'], row['attempted'], row['n_missing']) == (2, 2, 0)

    # a frame concatenated from two roots: each scan keeps its own root
    write_scan(legacy, 'legacy_only', legacy_recs, manifest=[r['run_id'] for r in legacy_recs])
    mixed = pd.concat([style.load_results('legacy_only', results_root=str(legacy)),
                       df_fresh], ignore_index=True)
    assert ah.expected_run_ids(mixed) == \
        {r['run_id'] for r in legacy_recs} | {r['run_id'] for r in fresh_recs}

    # a hand-built frame with no `_results_root` behaves exactly as before
    plain = df_fresh.drop(columns=['_results_root'])
    assert ah.expected_run_ids(plain) == {r['run_id'] for r in fresh_recs}


def test_diagnostics_come_from_the_rows_own_root(tmp_path, monkeypatch):
    """Same rule for the heavy npz: a row loaded from the legacy root reads the
    LEGACY diag file, not the fresh scan's file of the same run_id."""
    fresh, legacy = tmp_path / 'fresh', tmp_path / 'legacy'
    for d in (fresh, legacy):
        d.mkdir()
    monkeypatch.setenv(style.RESULTS_ROOT_ENV, str(fresh))
    recs = [sven_record(8, 0.01, 1e-4, SEEDS[0])]
    run_id = recs[0]['run_id']
    recs[0]['diag_file'] = f'diag/{run_id}.npz'
    for d, value in ((fresh, 1.0), (legacy, 2.0)):
        write_scan(d, 'both', recs, diag={run_id: {'train_batch': np.full(4, value)}})

    row = style.load_results('both', results_root=str(legacy)).iloc[0]
    assert style.load_diagnostics(row)['train_batch'].tolist() == [2.0] * 4
    assert style.load_diagnostics(row, results_root=str(fresh))['train_batch'].tolist() \
        == [1.0] * 4                                   # an explicit root still wins
    assert style.load_diagnostics(style.load_results('both').iloc[0])[
        'train_batch'].tolist() == [1.0] * 4


def test_clipped_band_never_crosses_the_mean():
    """The seed band's lower edge is clipped at the lowest seed, which is <= the mean
    in exact arithmetic but NOT in floats when every seed recorded the same value:
    `peak_gpu_mem_mb` is bit-identical across seeds, so `v.mean()` can land a few
    1e-15 BELOW `v.min()`.  The raw clip then produced a NEGATIVE yerr and
    `ax.bar` raised "'yerr' must not contain negative values", which is how the
    memory panel of mnist_analysis / mnist_analysis_labelRegression died."""
    v = np.array([31.476224] * 5)
    mean, std, lo = v.mean(), v.std(ddof=1), v.min()
    assert lo > mean          # the round-off that caused it (else this test is moot)
    lower, upper = style.clipped_band(mean, std, lo)
    assert lower <= mean <= upper
    yerr = style.clipped_yerr(mean, std, lo)
    assert (yerr >= 0).all()

    # the normal case is untouched: band = mean +/- std, floored at the lowest seed
    lower, upper = style.clipped_band(10.0, 4.0, 8.0)
    assert (float(lower), float(upper)) == (8.0, 14.0)
    lower, upper = style.clipped_band(10.0, 1.0, 2.0)
    assert (float(lower), float(upper)) == (9.0, 11.0)
    # a NaN std (a single seed) stays a zero-width band, not a NaN one
    lower, upper = style.clipped_band(10.0, np.nan, 10.0)
    assert (float(lower), float(upper)) == (10.0, 10.0)
