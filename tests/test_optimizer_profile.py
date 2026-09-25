"""`experiments/optimizer_profile.py`: the settings v2 got wrong, pinned.

`profile_results_v2` (2026-09-17) got TWO settings wrong: Sven's per-step
`torch.cuda.empty_cache()` -- up to 4.5x slower for full-capture Sven on CIFAR, 841 vs
187 ms/step -- and the missing `expandable_segments` allocator.  These tests are what stop
`profile_results_v3` from repeating either.

The third test group, `bn_mode`, guards a DIFFERENT thing: it is not a v2 defect (all 33
v2 CIFAR Gram records carry `meta.freeze_norm_stats: False`, i.e. v2 already used the
batch statistics the CIFAR scans use, because the pre-C-E2 config still set
`gram_freeze_norm_stats: false`).  C-E2 removed that key, after which the profiler's own
default would have started freezing CIFAR's statistics -- a regression v3 would have
shipped had `bn_mode_of` not been added.

Last comes the plumbing that decides WHERE a run writes, which matters because the real
job runs from a deploy snapshot (cwd = the frozen export).

No GPU: every function under test is pure. The module imports torch but touches
`torch.cuda` only inside the profiling functions, so importing it on a CPU node is fine
(~7 s).
"""

import ast
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'experiments' / 'optimizer_profile.py'

op = pytest.importorskip('experiments.optimizer_profile',
                         reason='needs torch + the sven package')


# ---------------------------------------------------------------------------
# empty_cache: False, and never inside a measured step
# ---------------------------------------------------------------------------
def test_sven_is_profiled_with_empty_cache_off():
    """The optimizer's default is already False; the profiler states it anyway, so a
    change of that default cannot silently put the 4.5x penalty back into the tables."""
    assert op.EMPTY_CACHE is False
    src = SRC.read_text()
    built = {}
    for call in (n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)):
        name = getattr(call.func, 'id', None) or getattr(call.func, 'attr', None)
        if name in ('Sven', 'SvenGram'):
            built.setdefault(name, []).append(
                {kw.arg: ast.unparse(kw.value) for kw in call.keywords})
    assert set(built) == {'Sven', 'SvenGram'}, f'constructed: {sorted(built)}'
    for name, calls in built.items():
        for kwargs in calls:
            assert kwargs.get('empty_cache') == 'EMPTY_CACHE', (name, kwargs)
    assert 'empty_cache=True' not in src


def test_empty_cache_is_called_only_between_configurations():
    """`torch.cuda.empty_cache()` inside the timed region is exactly what made v2 wrong.

    The measured region is the body of the two step loops in `profile_one`; the calls that
    remain are its own set-up and teardown, which run between configurations.
    """
    tree = ast.parse(SRC.read_text())
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert 'profile_one' in fns

    def calls_empty_cache(node):
        return [c for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and c.func.attr == 'empty_cache']

    # every call in the module, and the line of every `for` loop that measures steps
    everywhere = calls_empty_cache(tree)
    assert everywhere, 'no empty_cache() at all -- has the teardown been dropped?'
    loops = [n for n in ast.walk(fns['profile_one']) if isinstance(n, ast.For)]
    assert len(loops) >= 2, 'expected the warm-up and the measurement loops'
    inside = [c for loop in loops for c in calls_empty_cache(loop)]
    assert not inside, ('empty_cache() is called inside a profiled loop at line(s) '
                        f'{[c.lineno for c in inside]}')
    # ... and the builder never passes an optimizer that would call it per step either
    assert 'EMPTY_CACHE = False' in SRC.read_text()


# ---------------------------------------------------------------------------
# Where the results go
# ---------------------------------------------------------------------------
def test_the_default_root_is_v3_and_v2_is_never_written():
    assert op.DEFAULT_OUTPUT_ROOT == 'profile_results_v3'
    assert 'profile_results_v2' not in op.output_root({}, base='/w', env={})
    for cfg in sorted((REPO / 'experiments' / 'configs').glob('profile_*.yaml')):
        text = cfg.read_text()
        assert 'output_dir: profile_results_v3' in text, cfg.name
        assert 'output_dir: profile_results_v2' not in text, cfg.name


@pytest.mark.parametrize('env,prof,want', [
    ({}, {}, '/w/profile_results_v3'),                              # default
    ({}, {'output_dir': 'somewhere'}, '/w/somewhere'),              # config
    ({}, {'output_dir': '/abs/root'}, '/abs/root'),                 # absolute config
    ({'SV3_PROFILE_ROOT': '/tmp/smoke'}, {'output_dir': 'somewhere'}, '/tmp/smoke'),
])
def test_output_root_precedence_and_base(env, prof, want):
    """env > config > default, and a RELATIVE root resolves against the base the caller
    passes (hydra's original cwd), not against wherever the process has been chdir'ed."""
    assert op.output_root(prof, base='/w', env=env) == want


def test_a_relative_root_does_not_follow_the_process_into_the_snapshot(tmp_path):
    """The failure this guards: the real job runs with cwd = the deploy snapshot, so a
    cwd-relative root would write 34 MB of results into a frozen export."""
    snap = tmp_path / 'deploy' / 'abc123_def456'
    snap.mkdir(parents=True)
    here = os.getcwd()
    try:
        os.chdir(snap)
        assert op.output_root({}, base='/somewhere/sven-experiments', env={}) == \
            '/somewhere/sven-experiments/profile_results_v3'
        # and with no base at all it is at least absolute, never a bare relative path
        assert os.path.isabs(op.output_root({}, env={}))
    finally:
        os.chdir(here)


# ---------------------------------------------------------------------------
# bn_mode: the profile inherits its scan's norm policy (C-E2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('rcfg,want', [
    ({'bn_mode': 'batch', 'use_gram': True}, 'batch'),              # CIFAR, C-E2
    ({'bn_mode': 'frozen', 'use_gram': True}, 'frozen'),
    ({'use_gram': True}, 'frozen'),                                 # MLPs: the old default
    ({'gram_freeze_norm_stats': False}, 'batch'),                   # deprecated alias
    ({'gram_freeze_norm_stats': True}, 'frozen'),
])
def test_bn_mode_follows_the_scan_config(rcfg, want):
    assert op.bn_mode_of(rcfg) == want


def test_the_cifar_profile_inherits_batch_statistics_and_the_mlps_do_not():
    """The bug this fixes: with `gram_freeze_norm_stats` gone from the configs (C-E2
    replaced it with `bn_mode`), the profiler's own default froze the statistics, so the
    CIFAR step time it reported was not the step time of the runs it is compared against.
    """
    reconcile = _reconcile()
    with reconcile.ConfigLoader(None) as loader:
        cifar = loader.compose('profile_cifar', '')
        mnist = loader.compose('profile_mnist', '')
    assert op.bn_mode_of(cifar) == 'batch'
    assert op.bn_mode_of(mnist) == 'frozen'
    # hooks capture requires frozen statistics, which is why CIFAR never profiles it
    assert 'gram_hooks' not in list(cifar['sven_variants'])


def _reconcile():
    import importlib.util
    import sys
    path = REPO / 'tools' / 'reconcile.py'
    spec = importlib.util.spec_from_file_location('_sv3_reconcile_profile_test', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def test_alloc_conf_reads_both_spellings():
    assert op.alloc_conf(env={'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}) \
        == 'expandable_segments:True'
    assert op.alloc_conf(env={'PYTORCH_ALLOC_CONF': 'x:1'}) == 'x:1'
    assert op.alloc_conf(env={}) is None


def test_the_sbatch_exports_expandable_segments_and_asks_for_a_whole_node():
    sb = (REPO / 'bench' / 'profile_serial.sbatch').read_text()
    assert 'PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}' in sb
    assert '#SBATCH --exclusive' in sb
    assert '#SBATCH --time=06:00:00' in sb
    # runs the snapshot, exactly as tools/worker_pool.sh does
    assert 'export PYTHONPATH="$CODE:$CODE/sven"' in sb
    assert '.deploy_complete' in sb
    # and sends the results to an ABSOLUTE root in the repo: cwd is a scratch workdir and
    # the code is a frozen export, so a relative root would land next to neither
    assert 'export SV3_PROFILE_ROOT=${SV3_PROFILE_ROOT:-$REPO/profile_results_v3}' in sb


def test_every_result_carries_both_repos_and_the_two_settings():
    """The fields a v2-vs-v3 table needs in order to say what changed."""
    src = SRC.read_text()
    assert 'provenance.collect(REPO_ROOT, SVEN_ROOT)' in src
    assert 'provenance=prov' in src
    for field in ('"alloc_conf"', '"sven_empty_cache"', '"bn_mode"', '"output_root"'):
        assert field in src, field
