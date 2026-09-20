"""`analysis/profile_helpers.py`: which results root is read, and where the cache goes.

Two things this pins, both found by review of the 2026-09-20 re-profile work package:

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
