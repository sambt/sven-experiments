"""CPU tests for the WP3 phase-B mechanism helpers (`analysis/lib/spectra_figs.py`)
and for the one shared-helper fix they rest on (`sv_diagnostics` noise floor).

Everything runs against SYNTHETIC diag passes and a SYNTHETIC probe-spectra cache
built in a tmp root, so no test here touches `experiment_results/` or
`analysis/ckpt_spectra/` and none of them needs torch.

The properties under test are the ones that would be wrong silently:

* the residual-energy fraction really is `sum_{i<k} (u_i.r)^2 / ||U^T r||^2`, and
  `min(k, rtol-rank)` really is the optimizer's own `num_nonzero_svs` -- those two
  numbers ARE the mechanism claim;
* the k / rtol cut is read from the RECORDS, and a diag pass that no longer matches
  the selection of record says so (the provisional-CIFAR-CE contract);
* the online float32 floor is the RECORDED `sqrt(eps) sigma_max` (~3.45e-4), not
  `style.FLOAT32_NOISE_FLOOR` (1e-7) -- three and a half decades of "tail" hang on
  that, and the old `plot_epoch_spectra` drew the wrong one;
* the probe-set energy is normalised by the projected residual and reports
  `||P_U r||^2 / ||r||^2` separately, so a residual that no optimizer can reach is
  never charged to Sven's truncation;
* the dense spectra schedule is NOT uniform (every step to 1,000, then every 20th),
  so smoothing and the x axis are in step units, not sample units -- an
  index-fraction axis put the whole last 80 % of toy training into x > 0.8;
* a `(k, rtol)` cell counts the runs it dropped, because Sven's divergences sit at
  the bottom of every rtol grid and a one-seed cell must not look like a five-seed
  one;
* a probe figure names the single seed it draws, and a quantity that has run into
  the float64 threshold (toy's `cond`, `sigma_B/sigma_1`) is marked rather than
  quoted;
* a rank of 1 whose paired seed interval covers zero is reported as a TIE.
"""

import json
import sys
import warnings
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

import headline                  # noqa: E402
import spectra_figs as sf        # noqa: E402
import style                     # noqa: E402
import sv_diagnostics as sv      # noqa: E402

SCAN = 'synth_scan'
SEEDS = (1000, 1001, 1002)
B, K, RTOL, LR = 8, 4, 0.1, 0.5
N_EPOCH, N_STEP, N_LOG = 4, 40, 10


# ---------------------------------------------------------------------------
# A synthetic `{scan}_diag` pass
# ---------------------------------------------------------------------------
def spectra(n_log=N_LOG, width=B, seed=0):
    """A descending spectrum per logged step plus a matching ``utr``.

    Geometric decay with a per-seed ratio, so the rtol-rank is a known function of
    the step and differs between seeds (which is what the seed band has to survive).
    """
    rng = np.random.default_rng(seed)
    ratio = 0.3 + 0.05 * (seed % 5)
    svs = np.stack([10.0 * ratio ** np.arange(width) * (1 + 0.01 * t)
                    for t in range(n_log)])
    utr = np.abs(rng.normal(size=(n_log, width))) + 0.1
    return svs, utr


def diag_arrays_npz(seed):
    svs, utr = spectra(seed=seed)
    step = np.linspace(0, N_STEP - 1, N_LOG).astype(np.int32)
    used = np.minimum(K, (svs >= RTOL * svs[:, :1]).sum(axis=1))
    # the optimizer writes num_nonzero_svs on EVERY step; only the logged ones have
    # a spectrum, so the per-step array is filled by nearest logged value
    nnz = np.zeros(N_STEP, dtype=np.int32)
    for i, s in enumerate(step):
        nnz[s:] = used[i]
    return {
        'svs': svs.astype(np.float32), 'utr': utr.astype(np.float32),
        'svs_step': step,
        'sv_max': svs[:, 0].astype(np.float32),
        'sv_min_kept': svs[np.arange(N_LOG), used - 1].astype(np.float32),
        # the Gram floor the optimizer records: sqrt(eps_float32) * sigma_max
        'sv_noise_floor': (np.sqrt(np.finfo(np.float32).eps)
                           * svs[:, 0]).astype(np.float32),
        'update_norm': np.linspace(1.0, 0.1, N_LOG).astype(np.float32),
        'resid_norm': np.linspace(9.0, 0.5, N_LOG).astype(np.float32),
        'num_nonzero_svs': nnz,
        'train_batch': np.linspace(1.0, 0.1, N_STEP).astype(np.float32),
    }


def record(optimizer, seed, k=K, rtol=RTOL, lr=LR, run_id=None, diag=True):
    run_id = run_id or (f'svd_bs{B}_k{k}_lr{lr:g}_rtol{rtol:g}_mseed{seed}_lseed1000'
                        if optimizer == 'SVD'
                        else f'std_bs{B}_lr{lr:g}_optim{optimizer}_mseed{seed}_lseed1000')
    rec = {'run_id': run_id, 'optimizer': optimizer, 'loss': 'mse', 'batch_size': B,
           'k': k if optimizer == 'SVD' else None,
           'lr': lr, 'rtol': rtol if optimizer == 'SVD' else None,
           'model_seed': seed, 'loader_seed': 1000, 'status': 'ok',
           'schema_version': 2, 'mlp_width': 16,
           'losses': {'train': [1.0 / (i + 1) for i in range(N_EPOCH)],
                      'val': [2.0] + [1.0 / (i + 1) for i in range(N_EPOCH)],
                      'epoch_times': [1.0] * N_EPOCH},
           'svd_summary': ({'n_steps': N_STEP, 'mode': 'gram',
                            'num_nonzero_svs_epoch': [float(k)] * N_EPOCH}
                           if optimizer == 'SVD' else None),
           'diag_file': f'diag/{run_id}.npz' if diag else None}
    return rec


def write_pass(root, name, records, diag=None):
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    for rec in records:
        (d / f"{rec['run_id']}.jsonl").write_text(json.dumps(rec) + '\n')
    for run_id, arrays in (diag or {}).items():
        (d / 'diag').mkdir(exist_ok=True)
        np.savez_compressed(d / 'diag' / f'{run_id}.npz', **arrays)
    return d


def payload(k=K, rtol=RTOL, lr=LR):
    """A schema-2 selection payload naming one Sven configuration."""
    return {'schema': 2, 'generated_at': '2026-09-20T00:00:00-0400',
            'scans': {SCAN: {'methods': {'SVD': {
                'method': 'SVD', 'family': 'svd', 'batch_size': B,
                'hparams': {'k': k, 'lr': lr, 'rtol': rtol}}}}}}


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A tmp results root holding ``{SCAN}`` and ``{SCAN}_diag``."""
    r = tmp_path / 'experiment_results'
    r.mkdir()
    monkeypatch.setenv(style.RESULTS_ROOT_ENV, str(r))
    headline.clear_cache()
    recs = [record('SVD', s) for s in SEEDS] + [record('Adam', s) for s in SEEDS]
    write_pass(r, f'{SCAN}_diag', recs,
               diag={record('SVD', s)['run_id']: diag_arrays_npz(s) for s in SEEDS})
    # the tuning grid: three k x two rtol, seed-averaged used-rank from the light record
    grid = []
    for k in (2, 4, 8):
        for rtol in (0.01, 0.1):
            for s in SEEDS:
                rec = record('SVD', s, k=k, rtol=rtol, diag=False)
                rec['svd_summary']['num_nonzero_svs_epoch'] = [
                    float(min(k, 6 if rtol == 0.01 else 3))] * N_EPOCH
                grid.append(rec)
    write_pass(r, SCAN, grid)
    yield r
    headline.clear_cache()


# ---------------------------------------------------------------------------
# 1. Pure numerics: the energy fraction and the two cuts
# ---------------------------------------------------------------------------
def test_cumulative_energy_is_the_projected_residual_fraction():
    """``out[t, i]`` is ``sum_{j<=i} utr^2 / sum_j utr^2`` -- monotone, ending at 1."""
    utr = np.array([[3.0, 4.0, 0.0], [1.0, 1.0, 1.0]])
    cum = sf.cumulative_energy(utr)
    assert np.allclose(cum[0], [9 / 25, 1.0, 1.0])
    assert np.allclose(cum[1], [1 / 3, 2 / 3, 1.0])
    assert np.all(np.diff(cum, axis=-1) >= -1e-15)
    assert np.allclose(cum[:, -1], 1.0)


def test_energy_in_top_saturates_when_k_reaches_the_width():
    """``k >= width`` is "no cut", which is exactly 1.0 -- the case of four of the
    seven headline scans (``k = B``), where the truncation discards nothing."""
    utr = np.abs(np.random.default_rng(0).normal(size=(5, 8)))
    assert np.allclose(sf.energy_in_top(utr, 8), 1.0)
    assert np.allclose(sf.energy_in_top(utr, 99), 1.0)
    assert np.all(sf.energy_in_top(utr, 3) < 1.0)


def test_rtol_and_k_cuts_are_the_two_separate_constraints():
    """``rtol_rank`` counts ``sigma_i >= rtol sigma_0``; ``used_rank`` then caps at k."""
    svs = np.array([[1.0, 0.5, 0.2, 0.05, 0.001]])
    assert sf.rtol_rank(svs, 0.1)[0] == 3            # 1, .5, .2
    assert sf.rtol_rank(svs, 0.01)[0] == 4
    assert sf.used_rank(svs, k=2, rtol=0.01)[0] == 2  # k binds
    assert sf.used_rank(svs, k=9, rtol=0.1)[0] == 3   # rtol binds


def test_effective_rank_of_a_flat_spectrum_is_its_rank():
    """exp(H(p)) is r for a flat rank-r spectrum and falls as it concentrates."""
    assert sf.effective_rank(np.ones(7)) == pytest.approx(7.0)
    assert sf.effective_rank(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert sf.effective_rank(np.array([1.0, 0.1, 0.01])) < 2.0


def test_subsample_rows_keeps_the_ends():
    idx = sf.subsample_rows(1262, 45)
    assert idx[0] == 0 and idx[-1] == 1261
    assert len(idx) <= 47 and np.all(np.diff(idx) > 0)
    assert np.array_equal(sf.subsample_rows(5, 45), np.arange(5))


def test_smooth_steps_averages_in_step_units_not_sample_units():
    """The real schedule: every step to 1,000, then every 20th.  A 25-SAMPLE window is
    25 steps in the dense part and 500 in the sparse part; this must be 60 steps in
    both, which is what keeps the apparent sharpness of a curve constant."""
    step = np.concatenate([np.arange(0, 1001), np.arange(1020, 6240, 20)])
    assert dict(zip(*np.unique(np.diff(step), return_counts=True))) == {1: 1000, 20: 261}
    y = np.zeros_like(step, dtype=float)
    y[step == 500] = 1.0                       # a spike in the DENSE region
    y[step == 4000] = 1.0                      # and one in the SPARSE region
    xs, out = sf.smooth_steps(step, y, 60)
    assert np.array_equal(xs, step.astype(float))
    # the dense spike is averaged over 61 samples, the sparse one over 3-4: both are
    # 60 steps wide, so both keep the same total mass
    dense = out[np.abs(step - 500) <= 30]
    sparse = out[np.abs(step - 4000) <= 30]
    assert dense.max() == pytest.approx(1 / 61, rel=0.02)
    assert sparse.max() == pytest.approx(1 / 4, rel=0.4)
    assert len(dense) == 61 and len(sparse) <= 4
    # a sample-window average would instead smear the sparse spike over 500 steps
    assert sparse.max() > 10 * dense.max()


def test_smooth_steps_ignores_nans_and_survives_an_empty_curve():
    x, out = sf.smooth_steps([0, 1, 2, 3], [1.0, np.nan, 3.0, 4.0], 2)
    assert np.allclose(out, [1.0, 2.0, 3.5, 3.5])
    assert sf.smooth_steps([], [], 10)[0].size == 0


# ---------------------------------------------------------------------------
# 2. The recorded noise floor (the shared-helper fix)
# ---------------------------------------------------------------------------
def test_noise_floor_is_the_recorded_sqrt_eps_not_1e_7(root):
    """The Gram route's floor is ``sqrt(eps) sigma_max ~ 3.45e-4 sigma_max``, three
    and a half decades above ``style.FLOAT32_NOISE_FLOOR``.  It is READ from the
    record, so a run at another dtype would report its own."""
    d = sf.sven_diag(SCAN, payload=payload(), verbose=False)
    rel = sv.noise_floor(d.rows)
    assert rel == pytest.approx(float(np.sqrt(np.finfo(np.float32).eps)), rel=1e-4)
    assert rel > 100 * style.FLOAT32_NOISE_FLOOR


def test_noise_floor_falls_back_for_a_record_without_one():
    """A legacy record carries no ``sv_noise_floor``: the per-step helper falls back
    to the documented 1e-7 and the per-run one says ``None`` (so a caller keeps the
    old line rather than inventing a floor)."""
    legacy = {'svs_step': np.arange(3)}
    assert np.allclose(sv.noise_floor_rel(legacy), style.FLOAT32_NOISE_FLOOR)
    assert np.allclose(sv.noise_floor_rel(legacy, default=1e-3), 1e-3)
    assert sv.noise_floor(pd.DataFrame(columns=['run_id'])) is None


def test_spectrum_plot_draws_the_recorded_floor(root):
    """Regression: ``plot_epoch_spectra`` used to draw 1e-7 and call it "the float32
    floor" on a Gram spectrum, which hides 3.5 decades of round-off as structure."""
    df = headline.load(SCAN, 'diag')
    rows = df[df['optimizer'] == 'SVD']
    fig, ax = plt.subplots()
    spectra_, _ = sv.plot_epoch_spectra(rows, ax, k=K, lr=LR, rtol=RTOL)
    assert spectra_ is not None
    floors = [ln.get_ydata()[0] for ln in ax.get_lines()
              if ln.get_label().startswith('Gram float32 floor')]
    assert len(floors) == 1
    assert floors[0] == pytest.approx(float(np.sqrt(np.finfo(np.float32).eps)), rel=1e-3)
    assert not any('10^{-7}' in ln.get_label() for ln in ax.get_lines())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. The diag pass: configuration from the records, selection cross-check
# ---------------------------------------------------------------------------
def test_sven_diag_reads_the_configuration_from_the_records(root):
    d = sf.sven_diag(SCAN, payload=payload(), verbose=False)
    assert (d.k, d.rtol, d.lr, d.B) == (K, RTOL, LR, B)
    assert d.seeds == list(SEEDS)
    assert len(d.rows) == len(SEEDS)
    assert set(d.rows['optimizer']) == {'SVD'}
    assert d.truncates is True and d.selection_matches and not d.selection_note


def test_a_superseded_selection_is_reported_not_silently_plotted(root):
    """The provisional-CIFAR-CE contract: if ``tools/select_best.py`` now picks a
    different Sven, the diag pass on disk is the OLD one and the notebook must say
    so instead of labelling it "the best Sven"."""
    d = sf.sven_diag(SCAN, payload=payload(k=8, rtol=0.3), verbose=False)
    assert d.k == K and d.rtol == RTOL           # still what is on disk
    assert d.selection_matches is False
    assert 'selection of record' in d.selection_note
    assert 're-run' in d.selection_note.lower()


def test_a_re_run_diag_pass_takes_the_selected_configuration_of_two(root):
    """A diag directory can hold TWO Sven configurations, and then it must plot the
    selected one.

    Real case: the CIFAR-CE `rtol` extension moved Sven's pick from
    `k=128 lr=0.1 rtol=0.01` to `k=128 lr=0.5 rtol=0.3`, the pass was re-run, and the
    old pick's five runs keep their own run_ids on disk.  Before the fix `_one` raised
    `expected one rtol in the diag pass, got [0.01, 0.3]` and the notebook died; taking
    both would have averaged two configurations into one spectrum, which is worse.
    """
    superseded = [record('SVD', s, rtol=0.01, lr=0.1) for s in SEEDS]
    write_pass(root, f'{SCAN}_diag', superseded,
               diag={r['run_id']: diag_arrays_npz(s)
                     for r, s in zip(superseded, SEEDS)})
    headline.clear_cache()
    d = sf.sven_diag(SCAN, payload=payload(), verbose=False)
    assert (d.k, d.rtol, d.lr) == (K, RTOL, LR)          # the SELECTED one
    assert len(d.rows) == len(SEEDS)                      # not 6
    assert set(pd.to_numeric(d.rows['rtol'])) == {RTOL}
    assert d.selection_matches and not d.selection_note
    # ... and if the selection names a configuration NEITHER copy ran, one configuration
    # is still what gets plotted -- averaging two into one spectrum is the failure this
    # guards -- and the report fires
    d2 = sf.sven_diag(SCAN, payload=payload(k=8, rtol=0.3), verbose=False)
    assert len(d2.rows) == len(SEEDS)
    assert sf._n_configs(d2.rows) == 1
    assert d2.selection_matches is False
    assert 'selection of record' in d2.selection_note


def test_diag_arrays_reproduce_the_optimizers_own_nonzero_count(root):
    """``min(k, rtol-rank)`` recomputed from the saved spectrum equals the
    optimizer's per-step ``num_nonzero_svs``: two independent paths to the number
    that every mechanism claim rests on."""
    d = sf.sven_diag(SCAN, payload=payload(), verbose=False)
    for _, row in d.rows.iterrows():
        a = sf.diag_arrays(row)
        assert np.array_equal(a['used_rank'], a['nnz_logged'].astype(int))
        assert a['frac_top_k'].shape == a['step'].shape
        assert np.all(a['frac_used'] <= a['frac_top_k'] + 1e-12)
        assert np.all((a['frac_top_k'] > 0) & (a['frac_top_k'] <= 1.0))
        assert np.allclose(a['svs_rel'][:, 0], 1.0)


def test_diag_arrays_energy_matches_an_explicit_sum(root):
    """The headline fraction, recomputed by hand from the npz."""
    d = sf.sven_diag(SCAN, payload=payload(), verbose=False)
    row = d.rows.iloc[0]
    a = sf.diag_arrays(row)
    raw = np.load(Path(style.resolve_results_root()) / f'{SCAN}_diag' / 'diag'
                  / f"{row['run_id']}.npz")
    utr = np.abs(raw['utr'].astype(float))
    want = (utr[:, :K] ** 2).sum(axis=1) / (utr ** 2).sum(axis=1)
    assert np.allclose(a['frac_top_k'], want)


def test_seed_stack_averages_over_seeds_with_a_ddof1_band(root):
    d = sf.sven_diag(SCAN, payload=payload(), verbose=False)
    arrays = [sf.diag_arrays(r) for _, r in d.rows.iterrows()]
    step, mean, std, n = sf.seed_stack(d.rows, 'frac_top_k', arrays=arrays)
    assert n == len(SEEDS) and len(step) == N_LOG
    stack = np.stack([a['frac_top_k'] for a in arrays])
    assert np.allclose(mean, stack.mean(axis=0))
    assert np.allclose(std, stack.std(axis=0, ddof=1))


def test_seed_stack_truncates_to_the_shortest_run(root):
    """A run that stopped early (C-R1) keeps its partial curve and must not
    lengthen or NaN-poison the seed mean."""
    d = sf.sven_diag(SCAN, payload=payload(), verbose=False)
    arrays = [sf.diag_arrays(r) for _, r in d.rows.iterrows()]
    arrays[1] = dict(arrays[1])
    arrays[1]['frac_top_k'] = arrays[1]['frac_top_k'][:4]
    step, mean, std, n = sf.seed_stack(d.rows, 'frac_top_k', arrays=arrays)
    assert len(mean) == 4 and len(step) == 4 and n == len(SEEDS)


def test_mechanism_table_names_the_binding_cut(root):
    d = sf.sven_diag(SCAN, payload=payload(), verbose=False)
    tbl = sf.mechanism_table([d])
    assert len(tbl) == 1
    row = tbl.iloc[0]
    assert row['k'] == K and row['B'] == B and row['n_seeds'] == len(SEEDS)
    # every seed's spectrum decays past rtol before it reaches k = 4, so the rtol
    # cut is the binding one and `used` sits below the cap
    assert row['binds'] == 'rtol'
    # decay ratios 0.30 / 0.35 / 0.40 put 2 / 3 / 3 values above rtol = 0.1
    assert row['used_final'] == pytest.approx(8 / 3)
    assert 0 < row['energy_used_final'] <= row['energy_top_k_final'] <= 1.0
    assert row['noise_rel'] == pytest.approx(
        float(np.sqrt(np.finfo(np.float32).eps)), rel=1e-3)


def test_used_rank_grid_reads_the_light_records_only(root, monkeypatch):
    """The (k, rtol) grid comes from ``svd_summary['num_nonzero_svs_epoch']``; a
    ``diag/`` read on a whole tuning scan would be thousands of npz files."""
    monkeypatch.setattr(style, 'load_diagnostics',
                        lambda *a, **kw: pytest.fail('the grid must not read diag/'))
    g = sf.used_rank_grid(SCAN, lr=LR)
    assert set(g['k']) == {2, 4, 8} and set(g['rtol']) == {0.01, 0.1}
    assert set(g['n_seeds']) == {len(SEEDS)}
    tight = g[(g['k'] == 8) & (g['rtol'] == 0.1)].iloc[0]
    assert tight['used'] == pytest.approx(3.0)     # rtol binds
    assert tight['used_frac_k'] == pytest.approx(3 / 8)
    capped = g[(g['k'] == 2) & (g['rtol'] == 0.01)].iloc[0]
    assert capped['used_frac_k'] == pytest.approx(1.0)   # k binds
    assert set(g['counts']) == {f'{len(SEEDS)}/{len(SEEDS)}'}
    assert not g['n_diverged'].any()


@pytest.fixture
def root_diverged(tmp_path, monkeypatch):
    """A tuning grid whose tightest rtol cell lost 2 of 3 seeds to divergence, plus
    one cell that lost all three -- Sven's real failure pattern (688 of 7,825 on-grid
    runs, at the bottom of every rtol grid, EXPERIMENTS.md section 7)."""
    r = tmp_path / 'experiment_results'
    r.mkdir()
    monkeypatch.setenv(style.RESULTS_ROOT_ENV, str(r))
    headline.clear_cache()
    grid = []
    for rtol, n_div in ((1e-6, 3), (1e-5, 2), (0.1, 0)):
        for i, s in enumerate(SEEDS):
            rec = record('SVD', s, k=8, rtol=rtol, diag=False)
            rec['svd_summary']['num_nonzero_svs_epoch'] = [6.0] * N_EPOCH
            if i < n_div:
                rec['status'] = 'diverged'
            grid.append(rec)
    write_pass(r, SCAN, grid)
    yield r
    headline.clear_cache()


def test_used_rank_grid_counts_the_runs_it_dropped(root_diverged):
    """The bug this closes: a cell resting on ONE seed looked exactly like a cell
    resting on five, in the one figure where robustness has to be honest."""
    g = sf.used_rank_grid(SCAN, lr=LR).set_index('rtol')
    assert list(g['n_records']) == [3, 3, 3]              # attempted, incl. diverged
    assert g.loc[1e-6, 'n_seeds'] == 0 and g.loc[1e-6, 'n_diverged'] == 3
    assert not np.isfinite(g.loc[1e-6, 'used'])           # kept as a row, value NaN
    assert g.loc[1e-6, 'counts'] == '0/3'
    assert g.loc[1e-5, 'n_seeds'] == 1 and g.loc[1e-5, 'counts'] == '1/3'
    assert g.loc[0.1, 'n_seeds'] == len(SEEDS) and g.loc[0.1, 'counts'] == '3/3'
    # the value itself is still the mean over the surviving seeds only
    assert g.loc[1e-5, 'used'] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# 4. The probe-set cache
# ---------------------------------------------------------------------------
def write_probe(out_dir, scan, method, seed, n_rows=20, n_params=6, n_ckpt=4,
                rank=None, dist_init=None):
    """One cached trajectory in ``ckpt_tools.spectra_path``'s layout.

    ``rank`` zeroes the tail, giving a numerically rank-deficient spectrum (toy's
    situation: 14 resolved of 593); ``dist_init`` overrides the per-checkpoint
    ``||theta_t - theta_0||`` so a tie between two methods can be built exactly.
    """
    rng = np.random.default_rng(abs(hash((method, seed))) % 2 ** 32)
    width = min(n_rows, n_params)
    svals = np.stack([10.0 * 0.5 ** np.arange(width) * (1 + 0.1 * t)
                      for t in range(n_ckpt)])
    if rank is not None:
        svals[:, int(rank):] = 0.0
    utr = np.abs(rng.normal(size=(n_ckpt, width))) + 0.5
    # rows_norm > ||P_U r|| whenever the probe set has more rows than parameters
    rows_norm = np.sqrt((utr ** 2).sum(axis=1) + (2.0 if n_rows > n_params else 0.0))
    d = Path(out_dir) / scan
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / f'{method}_mseed{seed}_probe{"full" if n_rows > n_params else n_rows}.npz',
        step=np.array([0, 1, 2, 4][:n_ckpt]), epoch=np.zeros(n_ckpt, dtype=np.int64),
        svals=svals, utr=utr, probe_loss=np.linspace(1.0, 0.1, n_ckpt),
        rows_norm=rows_norm,
        param_norm=np.linspace(4.0, 9.0, n_ckpt),
        dist_init=(np.asarray(dist_init, dtype=float) if dist_init is not None
                   else np.linspace(0.0, 5.0 + seed % 3, n_ckpt)),
        probe_indices=np.arange(n_rows), n_rows=np.int64(n_rows),
        n_params=np.int64(n_params), run_id=np.str_(f'{method}_{seed}'),
        scan=np.str_(scan), method=np.str_(method), model_seed=np.int64(seed),
        verify_rel_error=np.float64(1e-8))


@pytest.fixture
def probe_dir(tmp_path):
    out = tmp_path / 'ckpt_spectra'
    for method in ('Sven', 'Adam', 'MuonW'):
        for seed in SEEDS:
            write_probe(out, f'{SCAN}_diag', method, seed)
    return out


def test_probe_methods_are_listed_in_plot_order(probe_dir):
    assert sf.probe_methods(SCAN, out_dir=probe_dir) == ['Sven', 'Adam', 'MuonW']
    assert sf.probe_methods('nothing_here', out_dir=probe_dir) == []


def test_probe_widths_reports_the_measured_width_not_min_rows_p(probe_dir):
    """The intro table used to promise ``min(n_rows, P)`` up to 27,562 on MNIST, where
    the probe set is 512 rows and the spectrum is 512 wide.  Only one of the two
    bounds is ever active, so the MEASURED width is the only honest number."""
    w = sf.probe_widths([SCAN], out_dir=probe_dir).iloc[0]
    assert w['n_rows'] == 20 and w['n_params'] == 6
    assert w['width'] == 6                       # min(20, 6): P binds here
    assert w['n_used'] == 3 * len(SEEDS) and w['seeds'] == len(SEEDS)
    assert sf.probe_widths(['nothing_here'], out_dir=probe_dir).empty


def test_probe_metrics_columns_and_values(probe_dir):
    tbl = sf.probe_metrics(SCAN, out_dir=probe_dir, sigma_b=3)
    assert len(tbl) == 3 * len(SEEDS) * 4
    assert set(tbl['method']) == {'Sven', 'Adam', 'MuonW'}
    row = tbl.iloc[0]
    assert row['sigma_max'] == pytest.approx(10.0 * (1 + 0.1 * row['step'] // 1)
                                             if row['step'] < 3 else row['sigma_max'])
    assert row['rank'] == 6                                   # width, all resolved
    assert row['sigma_B_over_1'] == pytest.approx(0.25)       # sigma_3 / sigma_1
    assert 0 < row['proj_frac'] < 1.0    # more rows than parameters: a part is unreachable
    assert row['eff_rank'] == pytest.approx(sf.effective_rank(
        10.0 * 0.5 ** np.arange(6)), rel=1e-6)


def test_probe_energy_separates_the_span_from_the_truncation(probe_dir):
    """``in_span`` is what NO optimizer can reach; ``top_k`` is what Sven's cut keeps.
    Charging the first to the second is the mistake this split exists to prevent."""
    tbl = sf.probe_energy(SCAN, ks=[1, 3, 6], out_dir=probe_dir)
    assert set(tbl['when']) == {'first', 'last'}
    last = tbl[tbl['when'] == 'last'].set_index('k')
    assert last.loc[6, 'top_k'] == pytest.approx(1.0)       # k = width: no cut
    assert last.loc[1, 'top_k'] < last.loc[3, 'top_k'] < 1.0
    assert 0 < last.loc[6, 'in_span'] < 1.0
    assert set(tbl['n_seeds']) == {len(SEEDS)}


# ---------------------------------------------------------------------------
# 5. The plots run headless and put something on the axes
# ---------------------------------------------------------------------------
def test_online_plots_draw(root):
    d = sf.sven_diag(SCAN, payload=payload(), verbose=False)
    arrays = [sf.diag_arrays(r) for _, r in d.rows.iterrows()]
    for fn in (sf.plot_spectra_over_training, sf.plot_utr_over_training):
        fig, ax = plt.subplots()
        fn(ax, d, arrays=arrays[:1], limit=5)
        assert ax.get_lines() and ax.get_yscale() == 'log'
        plt.close(fig)
    for fn in (sf.plot_energy_capture, sf.plot_energy_profile, sf.plot_rank_used,
               sf.plot_norms):
        fig, ax = plt.subplots()
        fn(ax, d, arrays=arrays)
        assert ax.get_lines()
        plt.close(fig)


def test_seed_band_is_named_in_every_band_legend(root):
    """C-A6: a shaded band always carries ``style.seed_spread_label()``."""
    d = sf.sven_diag(SCAN, payload=payload(), verbose=False)
    arrays = [sf.diag_arrays(r) for _, r in d.rows.iterrows()]
    for fn in (sf.plot_energy_capture, sf.plot_rank_used, sf.plot_norms):
        fig, ax = plt.subplots()
        fn(ax, d, arrays=arrays)
        assert style.seed_spread_label() in ax.get_legend_handles_labels()[1], fn.__name__
        plt.close(fig)


def test_probe_plots_draw_with_registered_colours(probe_dir):
    tbl = sf.probe_metrics(SCAN, out_dir=probe_dir, sigma_b=3)
    fig, ax = plt.subplots()
    sf.plot_probe_metric(ax, tbl, 'dist_init', methods=['Sven', 'Adam'])
    labels = ax.get_legend_handles_labels()[1]
    assert style.method_label('Sven') in labels and style.method_label('Adam') in labels
    colors = {ln.get_label(): ln.get_color() for ln in ax.get_lines()}
    assert colors[style.method_label('Sven')] == style.method_color('Sven')
    plt.close(fig)

    fig, ax = plt.subplots()
    norm = sf.plot_probe_spectra(ax, SCAN, 'Sven', out_dir=probe_dir, limit=3,
                                 seed=SEEDS[1])
    assert norm is not None and ax.get_lines()
    plt.close(fig)


def test_probe_spectra_names_the_one_seed_it_draws(probe_dir):
    """Five trajectories are cached per (scan, method) and this plots ONE of them.
    Unlabelled, a reader takes it for a 5-seed result; so the panel is annotated and a
    caller who does not choose gets a warning."""
    fig, ax = plt.subplots()
    with pytest.warns(UserWarning, match='cached seeds'):
        sf.plot_probe_spectra(ax, SCAN, 'Sven', out_dir=probe_dir, limit=3)
    assert any(f'seed {min(SEEDS)}' in t.get_text() for t in ax.texts)
    plt.close(fig)

    fig, ax = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter('error')                 # an explicit seed must not warn
        sf.plot_probe_spectra(ax, SCAN, 'Sven', out_dir=probe_dir, limit=3,
                              seed=SEEDS[2])
    assert any(f'seed {SEEDS[2]}' in t.get_text() for t in ax.texts)
    plt.close(fig)


def test_probe_metrics_flags_an_unresolved_sigma_b(tmp_path):
    """``sigma_B/sigma_1`` at an index outside the numerically resolved rank is
    round-off, not damping (toy: ``sigma_32`` of a rank-14 spectrum reads 9.5e-17).
    ``cond`` over the resolved part is likewise capped at ``1/floor``."""
    out = tmp_path / 'ckpt_spectra'
    for seed in SEEDS:
        write_probe(out, f'{SCAN}_diag', 'Sven', seed, rank=2)
    tbl = sf.probe_metrics(SCAN, out_dir=out, sigma_b=4)
    assert set(tbl['rank']) == {2}
    assert not tbl['sigma_b_resolved'].any() and set(tbl['sigma_b']) == {4}
    assert (tbl['cond'] <= 1.0 / sf.PROBE_FLOOR).all()
    assert (tbl['floor'] == sf.PROBE_FLOOR).all()
    # inside the resolved rank the flag flips, same files
    inside = sf.probe_metrics(SCAN, out_dir=out, sigma_b=2)
    assert inside['sigma_b_resolved'].all()


def test_plot_probe_metric_marks_the_float64_resolution_limit(tmp_path):
    out = tmp_path / 'ckpt_spectra'
    for seed in SEEDS:
        write_probe(out, f'{SCAN}_diag', 'Sven', seed, rank=2)
    tbl = sf.probe_metrics(SCAN, out_dir=out, sigma_b=4)

    # this fixture's cond is 2, thirteen decades below the ceiling: the line is NOT
    # drawn, so a fully resolved panel is not stretched by ten empty decades
    fig, ax = plt.subplots()
    sf.plot_probe_metric(ax, tbl, 'cond', logy=True)
    assert not [ln for ln in ax.get_lines()
                if 'resolution ceiling' in str(ln.get_label())]
    plt.close(fig)

    fig, ax = plt.subplots()
    sf.plot_probe_metric(ax, tbl, 'cond', logy=True, near_decades=13)
    ceilings = [ln.get_ydata()[0] for ln in ax.get_lines()
                if 'resolution ceiling' in str(ln.get_label())]
    assert ceilings and ceilings[0] == pytest.approx(1.0 / sf.PROBE_FLOOR)
    plt.close(fig)

    fig, ax = plt.subplots()
    sf.plot_probe_metric(ax, tbl, 'sigma_B_over_1', logy=True)
    labels = [str(ln.get_label()) for ln in ax.get_lines()]
    assert any('float64 probe floor' in s for s in labels)
    assert any('outside the resolved rank' in s for s in labels)
    # the unresolved points are drawn hollow
    assert any(ln.get_markerfacecolor() == 'none' and len(ln.get_xdata())
               for ln in ax.get_lines())
    plt.close(fig)


def test_low4_table_calls_a_statistical_tie_a_tie(tmp_path):
    """The bug this closes: ``Sven is smallest on 2 of 4 scans`` counted MNIST
    label-reg, where Sven 12.476 vs HIG 12.518 is a paired -0.042 +/- 0.16 with 2 of 5
    seeds the other way.  A rank of 1 is only a win if the paired interval excludes 0."""
    out = tmp_path / 'ckpt_spectra'
    # a tie: Sven's mean is lower but the per-seed differences straddle zero
    tie = {1000: (10.0, 10.4), 1001: (10.6, 10.2), 1002: (10.0, 10.5)}
    for seed, (sven, other) in tie.items():
        write_probe(out, f'{SCAN}_diag', 'Sven', seed, dist_init=[0, 0, 0, sven])
        write_probe(out, f'{SCAN}_diag', 'HIG', seed, dist_init=[0, 0, 0, other])
    tbl = sf.probe_metrics(SCAN, out_dir=out, sigma_b=3)
    low4 = sf.low4_table({SCAN: tbl}, metrics=('dist_init',), titles={SCAN: 'Synthetic'})
    row = low4.iloc[0]
    assert row['rank'] == 1 and row['best_other'] == 'HIG'
    assert row['n_paired'] == 3 and row['n_ref_lower'] == 2
    assert row['paired_mean'] == pytest.approx((10.2 - 10.3666667), abs=1e-6)
    assert not row['resolved'] and row['verdict'] == 'smallest (not resolved)'
    text = sf.low4_verdict(low4, 'dist_init')
    assert 'smallest mean on 1 of 1 scans' in text
    assert 'excludes zero on 0' in text
    assert 'NOT resolved' in text and '2/3 seeds lower' in text


def test_low4_table_reports_a_real_win_as_a_win(tmp_path):
    out = tmp_path / 'ckpt_spectra'
    for seed in SEEDS:
        write_probe(out, f'{SCAN}_diag', 'Sven', seed, dist_init=[0, 0, 0, 7.5])
        write_probe(out, f'{SCAN}_diag', 'HIG', seed, dist_init=[0, 0, 0, 8.1])
    tbl = sf.probe_metrics(SCAN, out_dir=out, sigma_b=3)
    low4 = sf.low4_table({SCAN: tbl}, metrics=('dist_init',), titles={SCAN: 'Synthetic'})
    row = low4.iloc[0]
    assert row['rank'] == 1 and row['resolved'] and row['verdict'] == 'smallest'
    assert row['paired_mean'] == pytest.approx(-0.6)
    text = sf.low4_verdict(low4, 'dist_init')
    assert 'smallest mean on 1 of 1 scans' in text and 'excludes zero on 1' in text


def test_savefig_writes_pdf_and_png(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = sf.savefig(fig, tmp_path / 'plots_v2' / 'spectra_analysis', 'demo')
    plt.close(fig)
    assert [p.suffix for p in paths] == ['.pdf', '.png']
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)
