"""`analysis/profile_helpers.py`: which results root is read, where the cache goes, and what
the v2-vs-v3 comparison is allowed to claim.

The first two were found by review of the 2026-09-20 re-profile work package; the third came
with the finished v3 pass (job 47396284), whose numbers only mean something next to v2's if
the difference between the two passes can be read off the records:

1. **The completeness gate.** `results_root()` used to prefer `profile_results_v3` the
   moment it held ONE json.  The re-profile job writes 720 files over up to 6 h, so any
   notebook run while the pass was in flight -- including `./make_plots.sh`, whose default
   set contains the four profile notebooks -- would have silently reported a partial root,
   with `add_relative()` dividing by an Adam / SGD reference that had not landed yet.
   A partial v3 must now be ANNOUNCED and skipped, and `load_profiles` must print the
   count.
2. **The cache location.** The derived-frame pickle used to live at
   `<root>/_profile_cache.pkl`, inside a results root that ANALYSIS_CONTRACTS.md declares
   read-only, and `compare_profiles(old=ROOT_V2, ...)` therefore rewrote
   `profile_results_v2/` on every call.  It now lives outside every results root.
3. **What the comparison may claim.** `PROVENANCE_COLS` must reach the frame as recorded and
   never as a default -- v2 wrote down none of them, and a `False` invented for it would make
   the frozen before-table claim it was measured correctly.  `sven_change_table` must pair the
   step-time change with the memory change (the fix should move one and not the other), and
   `baseline_control` must carry, for any baseline that moved, the two things that say it was
   not the machine: the per-step series' steadiness and the run's status.

Everything here is synthetic (a tmp dir per test) and needs neither torch, a GPU nor the
real profile results.  `$SV3_PROFILE_CACHE_DIR` is redirected into the tmp dir throughout,
so the suite never writes to the shared `experiment_results/_cache/` either.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'analysis'))   # the notebooks' own sys.path.insert(0, '.')

import profile_helpers as ph     # noqa: E402


@pytest.fixture(autouse=True)
def _cache_in_tmp(tmp_path, monkeypatch):
    """No test here may write to the real cache directory."""
    monkeypatch.setenv('SV3_PROFILE_CACHE_DIR', str(tmp_path / '_cache'))


def _record(run_id, method='Adam', study='methods'):
    return {'run_id': run_id, 'arch': 'toy_1d', 'config_name': 'c', 'study': study,
            'method': method, 'status': 'ok', 'n_params': 10,
            'params': {'batch_size': 8},
            'time': {'step_ms': {'cycle_mean': 1.0, 'steady_mean': 1.0, 'median': 1.0,
                                 'mean': 1.0, 'p10': 1.0, 'p90': 1.0, 'n': 50}},
            'raw': {'step_ms': [1.0] * 50},
            'memory': {'peak_alloc_bytes_max': 1_000_000}}


def _write(root, counts):
    """``counts`` = {config_name: n} json files under ``root``."""
    for cfg, n in counts.items():
        d = Path(root) / cfg
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f'r{i}.json').write_text(json.dumps(_record(f'{cfg}-{i}')))
    return Path(root)


# ---------------------------------------------------------------------------
# the expectation itself
# ---------------------------------------------------------------------------
def test_n_expected_sums_to_one_full_pass():
    assert ph.N_CONFIGS == 720 == sum(ph.N_EXPECTED.values())
    assert set(ph.N_EXPECTED) == {
        'profile_toy_1d', 'profile_polynomial', 'profile_mnist', 'profile_mnist_width',
        'profile_nanogpt', 'profile_nanogpt_width', 'profile_cifar'}


@pytest.mark.skipif(not ph.ROOT_V2.is_dir(), reason='profile_results_v2 not present')
def test_the_frozen_v2_root_is_a_complete_pass():
    """The yardstick is calibrated against the one root known to be finished."""
    done, n, exp = ph.profile_status(ph.ROOT_V2)
    assert (done, n, exp) == (True, 720, 720)


# ---------------------------------------------------------------------------
# profile_status / results_root
# ---------------------------------------------------------------------------
def test_a_partial_root_is_incomplete_and_a_full_one_is_complete(tmp_path):
    part = _write(tmp_path / 'part', {c: max(1, k // 2) for c, k in ph.N_EXPECTED.items()})
    assert ph.profile_status(part)[0] is False
    assert ph.is_complete(part) is False
    full = _write(tmp_path / 'full', dict(ph.N_EXPECTED))
    assert ph.profile_status(full) == (True, 720, 720)


def test_one_short_config_is_enough_to_be_incomplete(tmp_path):
    counts = dict(ph.N_EXPECTED)
    counts['profile_cifar'] -= 1                     # 719 of 720
    root = _write(tmp_path / 'r', counts)
    done, n, exp = ph.profile_status(root)
    assert (done, n, exp) == (False, 719, 720)


def test_the_sentinel_certifies_a_root_the_count_rule_would_reject(tmp_path):
    """`bench/profile_serial.sbatch` writes it after a clean full pass; the count rule is
    the fallback for a pass whose deploy snapshot predates the sentinel."""
    root = _write(tmp_path / 'r', {'profile_cifar': 3})
    assert ph.is_complete(root) is False
    (root / ph.COMPLETE_SENTINEL).write_text('job 1\n')
    assert ph.is_complete(root) is True


def test_results_root_stays_on_v2_while_v3_is_partial(tmp_path, monkeypatch, capsys):
    v2 = _write(tmp_path / 'v2', dict(ph.N_EXPECTED))
    v3 = _write(tmp_path / 'v3', {'profile_toy_1d': 4})
    monkeypatch.delenv('SV3_PROFILE_ROOT', raising=False)
    monkeypatch.setattr(ph, 'ROOT_V2', v2)
    monkeypatch.setattr(ph, 'ROOT_V3', v3)
    assert ph.results_root() == v2
    out = capsys.readouterr().out
    assert 'PARTIAL (4/720 configurations)' in out and 'v3' in out          # announced, not silent


def test_results_root_switches_to_v3_when_the_pass_is_complete(tmp_path, monkeypatch):
    v2 = _write(tmp_path / 'v2', dict(ph.N_EXPECTED))
    v3 = _write(tmp_path / 'v3', dict(ph.N_EXPECTED))
    monkeypatch.delenv('SV3_PROFILE_ROOT', raising=False)
    monkeypatch.setattr(ph, 'ROOT_V2', v2)
    monkeypatch.setattr(ph, 'ROOT_V3', v3)
    assert ph.results_root(verbose=False) == v3


def test_an_explicit_request_reads_a_partial_root_but_says_so(tmp_path, monkeypatch, capsys):
    """`$SV3_PROFILE_ROOT` / `prefer=` are deliberate: they win, with the count printed."""
    v2 = _write(tmp_path / 'v2', dict(ph.N_EXPECTED))
    v3 = _write(tmp_path / 'v3', {'profile_toy_1d': 4})
    monkeypatch.setattr(ph, 'ROOT_V2', v2)
    monkeypatch.setattr(ph, 'ROOT_V3', v3)
    monkeypatch.delenv('SV3_PROFILE_ROOT', raising=False)
    assert ph.results_root(prefer=v3) == v3
    assert 'PARTIAL 4/720' in capsys.readouterr().out
    monkeypatch.setenv('SV3_PROFILE_ROOT', str(v3))
    assert ph.results_root() == v3
    assert 'PARTIAL 4/720' in capsys.readouterr().out


def test_load_profiles_prints_the_shortfall(tmp_path, capsys):
    root = _write(tmp_path / 'r', {'profile_toy_1d': 4})
    df = ph.load_profiles(root)
    assert len(df) == 4
    assert 'PARTIAL, 4/720' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the cache never lands in a results root
# ---------------------------------------------------------------------------
def test_the_cache_is_written_outside_the_results_root(tmp_path):
    root = _write(tmp_path / 'profile_results_vX', {'profile_toy_1d': 3})
    before = sorted(p.name for p in root.iterdir())
    ph.load_profiles(root, use_cache=True)
    assert sorted(p.name for p in root.iterdir()) == before      # nothing added to the root
    cache = ph._cache_path(root)
    assert cache.is_file() and root not in cache.parents
    assert cache.name == 'profile_profile_results_vX.pkl'


def test_the_cache_still_hits_and_still_invalidates(tmp_path, monkeypatch):
    root = _write(tmp_path / 'r', {'profile_toy_1d': 3})
    assert 'sentinel' not in ph.load_profiles(root).columns
    real_row = ph._row
    monkeypatch.setattr(ph, '_row', lambda r: {**real_row(r), 'sentinel': 1})
    assert 'sentinel' not in ph.load_profiles(root).columns       # cache hit
    monkeypatch.setattr(ph, '_CACHE_VERSION', ph._CACHE_VERSION + 1)
    assert 'sentinel' in ph.load_profiles(root).columns           # version invalidates


def test_two_roots_with_the_SAME_BASENAME_do_not_share_a_cached_frame(tmp_path):
    """`_cache_path` names the file after the root's basename, so two roots called e.g.
    `profile_results_v3` in different places land on the same file.  The root's absolute
    path is therefore part of the cache signature: the second root reparses instead of
    being served the first one's frame."""
    a = _write(tmp_path / 'one' / 'profile_results_v3', {'profile_toy_1d': 3})
    b = _write(tmp_path / 'two' / 'profile_results_v3', {'profile_toy_1d': 3})
    assert ph._cache_path(a) == ph._cache_path(b)                 # same file, on purpose
    assert set(ph.load_profiles(a).run_id) == {f'profile_toy_1d-{i}' for i in range(3)}
    for cfg in (b / 'profile_toy_1d').iterdir():                  # distinguishable rows
        r = json.loads(cfg.read_text()); r['run_id'] = 'B-' + r['run_id']
        cfg.write_text(json.dumps(r))
    assert all(x.startswith('B-') for x in ph.load_profiles(b).run_id)
    assert not any(x.startswith('B-') for x in ph.load_profiles(a).run_id)


def test_an_unwritable_cache_dir_does_not_break_loading(tmp_path, monkeypatch, capsys):
    root = _write(tmp_path / 'r', {'profile_toy_1d': 3})
    monkeypatch.setenv('SV3_PROFILE_CACHE_DIR', str(tmp_path / 'r' / 'profile_toy_1d'
                                                    / 'r0.json' / 'nope'))
    assert len(ph.load_profiles(root)) == 3                       # reparsed, no exception
    assert 'not written' in capsys.readouterr().out


def test_compare_profiles_writes_nothing_into_either_root(tmp_path, monkeypatch):
    """The concrete regression: `compare_profiles(old=ROOT_V2)` rewrote v2's cache.

    Also pins that the roots are resolved at CALL time: as a default argument `old` would
    have stayed bound to the real `profile_results_v2` no matter what the caller set."""
    old = _write(tmp_path / 'v2', dict(ph.N_EXPECTED))
    new = _write(tmp_path / 'v3', dict(ph.N_EXPECTED))
    monkeypatch.setattr(ph, 'ROOT_V2', old)
    monkeypatch.setattr(ph, 'ROOT_V3', new)
    snap = {r: sorted(p.name for p in r.iterdir()) for r in (old, new)}
    cmp = ph.compare_profiles()
    assert len(cmp) == 720 and (cmp.step_ms_ratio == 1.0).all()
    for r, names in snap.items():
        assert sorted(p.name for p in r.iterdir()) == names


# ---------------------------------------------------------------------------
# provenance: the two settings v2 got wrong, read off the records
# ---------------------------------------------------------------------------
def _provenanced(run_id, **env):
    """A record written the way `experiments/optimizer_profile.py` writes one now."""
    r = _record(run_id)
    r['env'] = {'gpu': 'NVIDIA A100-SXM4-80GB', 'gpu_total_bytes': 85_118_156_800,
                'torch': '2.9.1+cu128', 'host': 'h1', 'slurm_job_id': '1',
                'alloc_conf': 'expandable_segments:True', 'sven_empty_cache': False,
                'bn_mode': 'frozen', **env}
    r['provenance'] = {'git_sha': 'a' * 40, 'git_dirty': True, 'sven_git_sha': 'b' * 40,
                       'host': 'h1', 'slurm_job_id': '1', 'torch_version': '2.9.1+cu128'}
    return r


def test_the_four_settings_a_cost_claim_needs_are_columns(tmp_path):
    """`empty_cache` False and the allocator have to be READABLE from the frame: the whole
    reason v3 exists is that v2's numbers could not be told apart from a correct pass by
    looking at the records (`empty_cache` was not in them)."""
    root = tmp_path / 'v3' / 'profile_toy_1d'
    root.mkdir(parents=True)
    (root / 'a.json').write_text(json.dumps(_provenanced('a')))
    df = ph.load_profiles(root.parent, use_cache=False)
    row = df.iloc[0]
    # False, and not None / 'False' / missing: the notebook prints this as the proof
    assert row.empty_cache is not None and bool(row.empty_cache) is False
    assert row.alloc_conf == 'expandable_segments:True'
    assert row.git_sha == 'a' * 40 and row.sven_git_sha == 'b' * 40
    assert row.gpu == 'NVIDIA A100-SXM4-80GB' and row.bn_mode == 'frozen'
    assert row.slurm_job_id == '1' and row.torch == '2.9.1+cu128'
    for c in ph.PROVENANCE_COLS:
        assert c in row.index


def test_a_record_without_provenance_reports_none_not_a_default(tmp_path):
    """v2's records carry no `alloc_conf` / `sven_empty_cache`.  Defaulting either to
    False would have made the frozen before-table claim it was measured correctly."""
    root = tmp_path / 'v2' / 'profile_toy_1d'
    root.mkdir(parents=True)
    (root / 'a.json').write_text(json.dumps(_record('a')))       # no env, no provenance
    row = ph.load_profiles(root.parent, use_cache=False).iloc[0]
    assert row.empty_cache is None and row.alloc_conf is None
    assert row.git_sha is None and row.gpu is None


def test_provenance_report_counts_one_row_per_distinct_pass(tmp_path):
    root = tmp_path / 'r' / 'profile_toy_1d'
    root.mkdir(parents=True)
    for i in range(3):                                   # one pass, three configurations
        (root / f'a{i}.json').write_text(json.dumps(_provenanced(f'a{i}')))
    (root / 'b.json').write_text(json.dumps(_provenanced('b', bn_mode='batch')))
    rep = ph.provenance_report(ph.load_profiles(root.parent, use_cache=False))
    assert list(rep.columns) == list(ph.PROVENANCE_COLS) + ['n']
    assert rep.n.tolist() == [3, 1]                      # biggest group first
    assert set(rep.bn_mode) == {'frozen', 'batch'}
    assert (rep.empty_cache == False).all()              # noqa: E712  (not a truth test)


def test_provenance_report_says_not_recorded_rather_than_blank(tmp_path):
    """A blank cell reads as 'False' to a hurried reader; v2 recorded nothing at all."""
    root = tmp_path / 'r' / 'profile_toy_1d'
    root.mkdir(parents=True)
    (root / 'a.json').write_text(json.dumps(_record('a')))
    rep = ph.provenance_report(ph.load_profiles(root.parent, use_cache=False))
    assert len(rep) == 1 and rep.n.iloc[0] == 1
    assert rep.empty_cache.iloc[0] == ph.NOT_RECORDED
    assert rep.alloc_conf.iloc[0] == ph.NOT_RECORDED
    assert rep.git_sha.iloc[0] == ph.NOT_RECORDED


def test_provenance_report_of_an_empty_frame_keeps_its_columns():
    import pandas as pd
    rep = ph.provenance_report(pd.DataFrame())
    assert list(rep.columns) == list(ph.PROVENANCE_COLS) + ['n'] and rep.empty


# ---------------------------------------------------------------------------
# the two tables the v2-vs-v3 cell shows: Sven's change, and the baseline control
# ---------------------------------------------------------------------------
def _cmp_record(run_id, method, arch='toy_1d', step=10.0, peak=1e6, p10=None, p90=None,
                status='ok'):
    p10 = step if p10 is None else p10
    p90 = step if p90 is None else p90
    return {'run_id': run_id, 'arch': arch, 'config_name': f'profile_{arch}',
            'study': 'methods', 'method': method, 'status': status, 'n_params': 10,
            'params': {'batch_size': 8},
            'time': {'step_ms': {'cycle_mean': step, 'steady_mean': step, 'median': step,
                                 'mean': step, 'p10': p10, 'p90': p90, 'n': 50}},
            'raw': {'step_ms': [step] * 50},
            'memory': {'peak_alloc_bytes_max': peak}}


def _two_roots(tmp_path, rows):
    """``rows`` = [(run_id, method, arch, step_v2, step_v3, peak_v2, peak_v3, kw2, kw3)]."""
    old, new = tmp_path / 'v2', tmp_path / 'v3'
    for root, i in ((old, 0), (new, 1)):
        for rid, method, arch, steps, peaks, kws in rows:
            d = root / f'profile_{arch}'
            d.mkdir(parents=True, exist_ok=True)
            (d / f'{rid}.json').write_text(json.dumps(
                _cmp_record(rid, method, arch, steps[i], peaks[i], **kws[i])))
    return old, new


def test_sven_change_table_pairs_step_time_with_memory(tmp_path, monkeypatch):
    """The re-profile's claim is a PAIR: step time falls, memory does not move (the fix
    changes when the allocator hands blocks back, not how many are live).  One table."""
    rows = [('a', 'gram_full', 'cifar_resnet18', (845.0, 190.0), (2.3e10, 2.3e10), ({}, {})),
            ('b', 'gram_hooks', 'mnist', (10.4, 10.3), (2.1e7, 2.1e7), ({}, {})),
            ('c', 'Adam', 'mnist', (4.0, 4.0), (1e7, 1e7), ({}, {}))]
    old, new = _two_roots(tmp_path, rows)
    cmp = ph.compare_profiles(old=old, new=new)
    tbl = ph.sven_change_table(cmp)
    assert len(tbl) == 2                                  # the baseline is not in it
    assert tbl.arch.tolist() == ['mnist', 'cifar_resnet18']          # ARCH_ORDER
    cif = tbl[tbl.arch == 'cifar_resnet18'].iloc[0]
    assert cif['step v3/v2'] == 0.225 and cif['mem v3/v2'] == 1.0
    assert cif['step ms v2'] == 845.0 and cif['step ms v3'] == 190.0


def test_baseline_control_flags_the_architecture_whose_baselines_moved(tmp_path):
    rows = [('a', 'Adam', 'mnist', (4.0, 4.02), (1e7, 1e7), ({}, {})),
            ('b', 'SGD', 'mnist', (3.9, 3.88), (1e7, 1e7), ({}, {})),
            ('c', 'Adam', 'toy_1d', (4.0, 4.0), (1e7, 1e7), ({}, {})),
            ('d', 'LBFGS3', 'toy_1d', (112.7, 15.0), (1e7, 1e7), ({}, {})),
            ('e', 'gram_full', 'toy_1d', (15.0, 8.0), (1e7, 1e7), ({}, {}))]
    old, new = _two_roots(tmp_path, rows)
    ctl = ph.baseline_control(ph.compare_profiles(old=old, new=new), tol=0.10)
    assert ctl.arch.tolist() == ['toy_1d', 'mnist']
    mn, toy = ctl[ctl.arch == 'mnist'].iloc[0], ctl[ctl.arch == 'toy_1d'].iloc[0]
    assert mn['n baselines'] == 2 and bool(mn['within 10%']) is True  # Sven excluded
    assert bool(toy['within 10%']) is False and toy['worst'] == ph.label('LBFGS3')
    assert toy['worst v3/v2'] == round(15.0 / 112.7, 3)


def test_baseline_control_carries_the_steadiness_and_status_of_the_worst_entry(tmp_path):
    """Why a baseline moved: an L-BFGS series that switched regime mid-measurement is
    bimodal (v2 toy LBFGS3: 15 steps at 14 ms then 35 at 126 ms, p90/p10 = 9.1), and a
    `nonfinite` entry diverged -- neither says the machine changed."""
    rows = [('a', 'Adam', 'mnist', (4.0, 4.0), (1e7, 1e7), ({}, {})),
            ('b', 'LBFGS1', 'mnist', (13.0, 19.0), (1e7, 1e7),
             ({'p10': 12.0, 'p90': 14.0}, {'p10': 9.5, 'p90': 127.0,
                                           'status': 'nonfinite'}))]
    old, new = _two_roots(tmp_path, rows)
    ctl = ph.baseline_control(ph.compare_profiles(old=old, new=new)).iloc[0]
    assert ctl['worst'] == ph.label('LBFGS1')
    assert ctl['steady v2'] == round(14.0 / 12.0, 2)
    assert ctl['steady v3'] == round(127.0 / 9.5, 2)      # the bimodal one
    assert (ctl['status v2'], ctl['status v3']) == ('ok', 'nonfinite')


def test_baseline_control_skips_a_configuration_that_failed_in_either_root(tmp_path):
    """An `oom` / `error` has no step time, so it cannot be a control point."""
    rows = [('a', 'Adam', 'mnist', (4.0, 4.0), (1e7, 1e7), ({}, {})),
            ('b', 'HIG', 'mnist', (70.0, 70.0), (1e7, 1e7), ({}, {'status': 'oom'}))]
    old, new = _two_roots(tmp_path, rows)
    ctl = ph.baseline_control(ph.compare_profiles(old=old, new=new)).iloc[0]
    assert ctl['n baselines'] == 1 and bool(ctl['within 10%']) is True


def test_both_tables_survive_an_empty_comparison():
    import pandas as pd
    for f in (ph.sven_change_table, ph.baseline_control):
        assert f(pd.DataFrame()).empty
