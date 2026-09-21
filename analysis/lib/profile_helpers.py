"""Loading + plotting for the optimizer memory / step-time profile.

Input: the per-configuration JSON files written by ``experiments/optimizer_profile.py``
under ``<results root>/<config>/``.  :func:`load_profiles` flattens them into ONE tidy
DataFrame (a row per configuration); every plot function here takes that frame, so the
notebooks are thin drivers.

Which results root (:func:`results_root`)
    ``$SV3_PROFILE_ROOT``, else :data:`ROOT_V3` once it is COMPLETE (:func:`profile_status`),
    else :data:`ROOT_V2`.  The completeness gate matters because the re-profile job writes
    :data:`N_CONFIGS` files over up to 6 h: on a half-written root :func:`add_relative`
    would normalise against an Adam / SGD reference that has not landed yet and report
    wrong or NaN ``rel_time`` / ``rel_mem`` instead of failing.  A partial v3 is always
    PRINTED -- never silently used, never silently ignored.

    What v2 (2026-09-17) got wrong, and v3 does not, is TWO settings: Sven's per-step
    ``torch.cuda.empty_cache()`` (full-capture Sven up to 4.5x slower on CIFAR, 841 vs
    187 ms/step, and the sole source of its step-time variance) and the missing
    ``expandable_segments`` allocator.  The BatchNorm policy is NOT one of the differences:
    all 33 v2 CIFAR Gram records carry ``meta.freeze_norm_stats: False``, so v2 already
    profiled CIFAR with the batch statistics its scans use (the pre-C-E2 config still set
    ``gram_freeze_norm_stats: false``).  What v3's ``bn_mode`` work prevents is a FUTURE
    regression: C-E2 removed that key from the configs, after which the profiler's own
    default would have started freezing CIFAR's statistics.
    :func:`compare_profiles` joins the two roots on ``run_id`` so the change is reported
    rather than quietly applied.  Both measurement roots are READ-ONLY
    (ANALYSIS_CONTRACTS.md), so the derived-frame cache lives OUTSIDE them
    (:func:`_cache_path`).

Conventions
    step_ms      AMORTISED mean step time: the plain mean over the last 80% of the measured
                 steps, truncated to a whole number of 10-step cycles (C-T2).  Methods with
                 a periodic refresh -- SOAP's ``precondition_frequency=10``, Sven's
                 re-factorisations -- pay for it on one step in ten, and a spike filter
                 would simply delete that cost (SOAP: 5.56 ms filtered vs 6.14 ms amortised).
    step_ms_steady  the old steady-state mean (>3 MAD spikes dropped), kept for reference
    peak_mb      max over measured steps of ``torch.cuda.max_memory_allocated``
    overhead_mb  peak_mb minus the model's own parameter memory
    rel_time     step_ms / the reference first-order method (Adam, else AdamW) at the SAME
                 architecture, batch size and width;  rel_mem = peak_mb / SGD (SGD only)
    status       ok | oom | infeasible | error | nonfinite  -- non-ok rows are kept and
                 drawn as markers.  ``nonfinite`` (the parameters blew up during the
                 profiled steps) still TIMED the steps, so its timings and memory are
                 used like an ok row's (:data:`TIMED`) and only flagged in the labels;
                 oom / infeasible / error have no numbers at all.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import METHOD_COLORS, method_color

#: repo root, so a root resolves the same from a notebook (cwd = analysis/) and a test
_REPO = Path(__file__).resolve().parent.parent.parent   # repo root (module in analysis/lib/)
#: 2026-09-17, measured with the per-step `empty_cache()` and without
#: `expandable_segments`: FROZEN as the before-table (read-only, cache kept outside it).
ROOT_V2 = _REPO / 'profile_results_v2'
#: the re-measurement (ANALYSIS_PLAN.md section 7.4): `empty_cache` off,
#: expandable_segments on, the scans' own `bn_mode`.
ROOT_V3 = _REPO / 'profile_results_v3'
MB = 1e6

#: Configurations per profile config in a FULL pass: exactly what `profile_results_v2`
#: holds and what `bench/profile_serial.sbatch`'s default `PROFILE_CONFIGS` writes.  This
#: is the completeness yardstick (:func:`profile_status`) -- not a grid definition.
N_EXPECTED = {
    'profile_toy_1d': 129, 'profile_polynomial': 129, 'profile_mnist': 143,
    'profile_mnist_width': 91, 'profile_nanogpt': 92, 'profile_nanogpt_width': 55,
    'profile_cifar': 81,
}
#: 720: the size of one full pass.
N_CONFIGS = sum(N_EXPECTED.values())
#: File `bench/profile_serial.sbatch` writes at the end of a clean full pass.  The count
#: rule below is the fallback for a pass whose deploy snapshot predates the sentinel.
COMPLETE_SENTINEL = '_profile_complete'

SVEN = ['gram_hooks', 'gram_full', 'gram_chunked', 'classic']
LABELS = {
    'gram_hooks': 'Sven (Gram, hooks)', 'gram_full': 'Sven (Gram, full $J$)',
    'gram_chunked': 'Sven (Gram, chunked)', 'classic': r'Sven (classic, rand. SVD)',
    'JD': 'JD (UPGrad)', 'LBFGS1': 'LBFGS (1 it.)', 'LBFGS3': 'LBFGS (3 it.)',
}
# Baselines take the global optimizer colours (style.METHOD_COLORS).  The Sven variants
# only ever appear side by side here: the default backend (Gram, hooks) is Sven's black,
# the others magenta / teal / brown, outside the baseline hues.  LBFGS (3 it.) is a
# darker shade of the LBFGS grey.
COLORS = {
    'gram_hooks': method_color('Sven'), 'gram_full': '#C2185B', 'gram_chunked': '#00796B', 'classic': '#A0522D',
    **{m: c for m, c in METHOD_COLORS.items() if m != 'Sven'},
    'LBFGS1': method_color('LBFGS'), 'LBFGS3': '#4D4D4D',
}
MARKERS = {'gram_hooks': 'o', 'gram_full': 's', 'gram_chunked': 'D', 'classic': '^'}
#: Short note per non-ok status (used by the tables and the failure markers).
STATUS_NOTE = {'oom': 'OOM', 'infeasible': 'infeasible', 'error': 'n/a', 'nonfinite': 'non-finite'}
#: Statuses whose timings and memory ARE measurements.  ``nonfinite`` is recorded
#: AFTER the measured steps (``optimizer_profile.py``), so the step cost is valid even
#: though the parameters diverged: every function that CONSUMES a value filters on this
#: set (a non-finite REFERENCE method would otherwise blank out the whole ``x Adam``
#: column), and the tables / markers keep flagging the status.
TIMED = ('ok', 'nonfinite')
#: SOAP's ``precondition_frequency``: the period of the most expensive refresh profiled.
CYCLE = 10
ARCH_TITLES = {'toy_1d': 'Toy 1D MLP', 'polynomial': 'Polynomial MLP', 'mnist': 'MNIST MLP',
               'mnist_width': 'MNIST MLP (width sweep)', 'cifar_resnet18': 'CIFAR-10 ResNet18',
               'nanogpt': 'nanoGPT', 'nanogpt_width': 'nanoGPT (width sweep)'}
ARCH_ORDER = ['toy_1d', 'polynomial', 'mnist', 'nanogpt', 'cifar_resnet18']


def label(m): return LABELS.get(m, m)
def color(m): return COLORS[m] if m in COLORS else method_color(m)
def is_sven(m): return m in SVEN


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def cycle_mean(values, cycle=CYCLE):
    """Mean over the last 80% of a per-step series, truncated to whole ``cycle``-step
    cycles -- the amortised cost of a step (C-T2).

    The first 20% are start-up; the remainder is truncated to a multiple of ``cycle``
    so that a refresh with period ``cycle`` is counted exactly the right number of
    times, whatever the window's phase.  NaN for an empty series.

    Duplicated in ``experiments/optimizer_profile.summarize`` (which cannot be imported
    here: it imports torch); ``tests/test_analysis_offline.py`` checks the two agree.
    """
    a = np.asarray(values if values is not None else [], dtype=float)
    if a.size == 0:
        return np.nan
    tail = a[int(0.2 * a.size):]
    n = (tail.size // cycle) * cycle
    return float(tail[:n].mean() if n else tail.mean())


def _timing(t: dict, raw: dict, key: str):
    """The cycle mean of the timing series ``key``: the value stored by a fresh profile,
    else RECOMPUTED from the stored raw list, so legacy profiles are repaired without
    rerunning them (C-T2).  Falls back to ``steady_mean`` when neither exists."""
    d = t.get(key) or {}
    if d.get('cycle_mean') is not None:
        return d['cycle_mean']
    v = cycle_mean(raw.get(key))
    if v == v:
        return v
    # NaN, not None, for a series this method does not have (the Sven-only phase
    # splits): a column of Nones is an object column, and `.round()` raises on it.
    sm = d.get('steady_mean')
    return sm if sm is not None else np.nan


def _row(r: dict) -> dict:
    p, meta = r.get('params', {}), r.get('meta') or {}
    t, mem = r.get('time') or {}, r.get('memory') or {}
    raw = r.get('raw') or {}
    st = t.get('step_ms') or {}
    env, prov = r.get('env') or {}, r.get('provenance') or {}
    g = lambda d, k: (d.get(k) if d else None)
    out = {
        'run_id': r['run_id'], 'arch': r.get('arch'), 'config_name': r.get('config_name'), 'study': r['study'],
        'method': r['method'], 'family': 'sven' if is_sven(r['method']) else 'baseline', 'status': r['status'],
        'error': r.get('error'), 'n_params': r.get('n_params'), 'B': p.get('batch_size'),
        'rows': meta.get('rows'), 'k': meta.get('k'), 'k_fraction': p.get('k_fraction'),
        'pf': p.get('param_fraction', 1.0), 'mask_mode': p.get('mask_mode') or 'none',
        'mb': p.get('microbatch_size', 1), 'chunk_fraction': p.get('chunk_fraction'),
        'chunk_numel': meta.get('chunk_numel'), 'n_groups': meta.get('n_groups'), 'width': p.get('width'),
        'step_ms': _timing(t, raw, 'step_ms'), 'step_ms_steady': st.get('steady_mean'),
        'step_ms_median': st.get('median'), 'step_ms_trimmed': st.get('trimmed_mean'),
        'step_ms_mean': st.get('mean'), 'step_ms_p10': st.get('p10'), 'step_ms_p90': st.get('p90'), 'n_steps': st.get('n'),
        'wall_ms': _timing(t, raw, 'wall_ms'), 'capture_ms': _timing(t, raw, 'capture_ms'),
        'solve_ms': _timing(t, raw, 'solve_ms'),
        'model_mb': (r.get('baseline_model_bytes') or 0) / MB,
        'peak_mb': (mem.get('peak_alloc_bytes_max') or np.nan) / MB,
        'peak_reserved_mb': (mem.get('peak_reserved_bytes_max') or np.nan) / MB,
        'resident_mb': (mem.get('resident_bytes_mean') or np.nan) / MB,
        'peak_capture_mb': (mem.get('peak_capture_bytes_max') or np.nan) / MB,
        'analytic_jac_mb': (meta.get('analytic_jacobian_bytes') or np.nan) / MB,
        'analytic_gram_mb': (meta.get('analytic_gram_bytes') or np.nan) / MB,
        'oom_peak_mb': (r.get('peak_alloc_bytes_at_failure') or np.nan) / MB,
        'gpu': env.get('gpu'), 'gpu_total_mb': (env.get('gpu_total_bytes') or np.nan) / MB,
        # PROVENANCE, one column per field a cost claim has to be able to name (below).
        # `None` where the pass did not record it -- v2 recorded none of these but the
        # GPU, which is itself the finding, so nothing here may default to a value.
        'alloc_conf': env.get('alloc_conf'), 'empty_cache': env.get('sven_empty_cache'),
        'bn_mode': env.get('bn_mode'), 'git_sha': prov.get('git_sha'),
        'git_dirty': prov.get('git_dirty'), 'sven_git_sha': prov.get('sven_git_sha'),
        'torch': env.get('torch') or prov.get('torch_version'),
        'host': env.get('host') or prov.get('host'),
        'slurm_job_id': env.get('slurm_job_id') or prov.get('slurm_job_id'),
        '_path': r.get('_path'),
    }
    out['overhead_mb'] = out['peak_mb'] - out['model_mb']
    out['steadiness'] = (out['step_ms_p90'] / out['step_ms_p10']) if out['step_ms_p10'] else np.nan
    return out


#: The provenance a step-time / memory claim has to be able to name: which code measured
#: it, on which GPU, under which allocator, and with Sven's per-step ``empty_cache`` on or
#: off.  :func:`provenance_report` groups a frame by exactly these.  ``profile_results_v2``
#: carries none of them except ``gpu`` (its ``env`` block is gpu / torch / cuda / host /
#: slurm_job_id only), so they read as ``None`` there -- the absence is the reason v2 could
#: not be told apart from a correct pass by reading the records, only by reading the code.
PROVENANCE_COLS = ('gpu', 'alloc_conf', 'empty_cache', 'bn_mode', 'git_sha', 'sven_git_sha',
                   'torch', 'host', 'slurm_job_id')
#: What :func:`provenance_report` prints for a field the pass never recorded.
NOT_RECORDED = '(not recorded)'

#: Bump when :func:`_row` changes what it derives from the same JSON files -- the file
#: listing alone cannot see a change of summary statistic (C-T2 moved ``step_ms`` from
#: ``steady_mean`` to the cycle mean, and every cached frame would have stayed stale;
#: version 3 added :data:`PROVENANCE_COLS`).
_CACHE_VERSION = 3


def has_profiles(root) -> bool:
    """Whether ``root`` holds at least one profile JSON (an empty or absent root is not
    a results root -- the v3 directory exists as soon as a job starts writing)."""
    root = Path(root)
    return root.is_dir() and any(root.glob('*/*.json'))


def profile_count(root) -> int:
    """Profile JSONs under ``root`` (0 for an absent root)."""
    root = Path(root)
    return sum(1 for _ in root.glob('*/*.json')) if root.is_dir() else 0


def profile_status(root):
    """``(complete, n_found, n_expected)`` -- is ``root`` a FINISHED pass?

    Complete iff :data:`COMPLETE_SENTINEL` is there (written by
    ``bench/profile_serial.sbatch`` after a clean full pass) **or** every directory of
    :data:`N_EXPECTED` holds at least its full count.  The count rule is what covers a
    pass whose deploy snapshot predates the sentinel, e.g. the 2026-09-20 job 47396284.

    The profiler writes a record for a failed configuration too (``oom`` / ``error`` are
    720 of 720 in v2), so a short count means the pass is unfinished -- resubmit
    ``profile_serial.sbatch`` (it skips what exists) or, to read it anyway, pass the root
    explicitly / set ``$SV3_PROFILE_ROOT``.
    """
    root = Path(root)
    n = profile_count(root)
    if (root / COMPLETE_SENTINEL).exists():
        return True, n, N_CONFIGS
    done = all(sum(1 for _ in (root / c).glob('*.json')) >= k for c, k in N_EXPECTED.items())
    return done, n, N_CONFIGS


def is_complete(root) -> bool:
    """:func:`profile_status`'s first element."""
    return profile_status(root)[0]


def results_root(prefer=None, verbose=True):
    """The profile results root to read: ``$SV3_PROFILE_ROOT`` > ``prefer`` > v3 (only
    once COMPLETE) > v2.

    v3 takes over from v2 on its own, but only when the whole pass has landed: a root the
    re-profile job is still filling is announced as PARTIAL and skipped, because
    :func:`add_relative` would otherwise divide by an Adam / SGD reference that is not
    there yet and report wrong or NaN relative columns rather than an error.
    ``$SV3_PROFILE_ROOT`` and ``prefer`` are explicit requests, so they win even when
    partial -- with the count printed.  Pass ``prefer=ph.ROOT_V2`` in a cell that must
    stay on the old numbers.
    """
    env = os.environ.get('SV3_PROFILE_ROOT')
    for cand, why, gated in ((env, '$SV3_PROFILE_ROOT', False), (prefer, 'requested', False),
                             (ROOT_V3, 'v3', True), (ROOT_V2, 'v2', False)):
        if not cand or not has_profiles(cand):
            continue
        cand = Path(cand)
        done, n, exp = profile_status(cand)
        if gated and not done:
            if verbose:
                print(f'profile results: {cand} is PARTIAL ({n}/{exp} configurations) -- '
                      f'staying on {ROOT_V2.name}. Pass prefer=ph.ROOT_V3 or set '
                      f'$SV3_PROFILE_ROOT to read it anyway.')
            continue
        if verbose:
            print(f'profile results: {cand}  ({why}'
                  + ('' if done else f'; PARTIAL {n}/{exp}') + ')')
        return cand
    # nothing has results yet: name v3 anyway, so the error says what is missing
    return Path(env or prefer or ROOT_V3)


def _cache_path(root) -> Path:
    """Where :func:`load_profiles` caches the derived frame -- NOT inside the results root.

    The measurement roots are read-only (ANALYSIS_CONTRACTS.md: "the loader cache under
    ``experiment_results/_cache/`` is the one allowed write"), and a cache inside them
    meant every ``compare_profiles`` call rewrote ``profile_results_v2/``.
    ``$SV3_PROFILE_CACHE_DIR`` overrides the directory (the tests point it at a tmp dir).
    """
    d = Path(os.environ.get('SV3_PROFILE_CACHE_DIR')
             or (_REPO / 'experiment_results' / '_cache'))
    return d / f'profile_{Path(root).name}.pkl'


def load_profiles(root=None, configs=None, use_cache=True) -> pd.DataFrame:
    """Every profile JSON under ``root`` as one tidy frame (cached on the file listing).

    ``root=None`` asks :func:`results_root`.  The printed line carries
    ``<n>/<N_CONFIGS>`` so a partial root can never be read as a finished one.
    """
    root = results_root() if root is None else Path(root)
    files = sorted(f for f in root.glob('*/*.json') if configs is None or f.parent.name in configs)
    sig = (_CACHE_VERSION, str(root.resolve()) if root.exists() else str(root),
           len(files), max((f.stat().st_mtime_ns for f in files), default=0),
           tuple(configs or ()))
    cache = _cache_path(root)
    if use_cache and cache.is_file():
        try:
            blob = pickle.loads(cache.read_bytes())
            if blob['sig'] == sig:
                return blob['df']
        except Exception:
            pass
    rows = []
    for f in files:
        r = json.loads(f.read_text()); r['_path'] = str(f)
        rows.append(_row(r))
    df = add_relative(pd.DataFrame(rows)) if rows else pd.DataFrame()
    if use_cache and rows:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps({'sig': sig, 'df': df}))
        except Exception as exc:    # an unwritable cache dir is fine: reparse next time
            print(f'({cache} not written: {type(exc).__name__}: {exc})')
    done, n, exp = profile_status(root)
    short = '' if (configs or done) else f'  -- PARTIAL, {n}/{exp} configurations written'
    print(f'{len(df)} configurations from {root}/ '
          f'({df["status"].value_counts().to_dict() if len(df) else {}}){short}')
    return df


def provenance_report(df: pd.DataFrame, cols=PROVENANCE_COLS) -> pd.DataFrame:
    """One row per distinct :data:`PROVENANCE_COLS` combination in ``df``, with its count.

    What it is for: a cost claim in the paper ("Sven's CIFAR step is 190 ms") is only worth
    the provenance of the pass behind it, and the two settings v2 got wrong are exactly
    two of these columns.  A one-row report means every configuration in the frame was
    measured by the same code on the same GPU under the same allocator -- which is what
    makes a v2-vs-v3 ratio a measurement of the fix rather than of the machine.

    ``bn_mode`` legitimately differs by architecture (the Gram family's default is
    ``frozen``; CIFAR's scan config sets ``bn_mode: batch``, which the profile inherits),
    so a v3 pass reports TWO rows and their counts say how the 720 split.  A field the
    pass did not record reads :data:`NOT_RECORDED` rather than blank -- "v2 never wrote
    this down" and "v2 wrote down False" must not look the same in a table.  v2's norm
    policy is therefore NOT readable from this report: it lives in the per-record
    ``meta.freeze_norm_stats`` (``False`` for all 33 of its CIFAR Gram records, i.e. batch
    statistics, the same policy v3's ``bn_mode: batch`` gives CIFAR).
    """
    cols = list(cols)
    if df.empty:
        return pd.DataFrame(columns=cols + ['n'])
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=['n'])
    keys = [df[c].astype(object).where(df[c].notna(), NOT_RECORDED).rename(c) for c in cols]
    out = df.groupby(keys, dropna=False).size().reset_index(name='n')
    out.columns = cols + ['n']
    return out.sort_values('n', ascending=False).reset_index(drop=True)


#: Columns :func:`compare_profiles` reports side by side, with the suffix each root gets.
#: ``steadiness`` (p90 / p10 of the per-step series) is carried so that a value which
#: moved between the roots can be read together with whether either measurement was a
#: single-regime one at all -- :func:`baseline_control` needs exactly that.
COMPARE_VALUES = ('step_ms', 'capture_ms', 'solve_ms', 'peak_mb', 'rel_time', 'steadiness')


def compare_profiles(old=None, new=None, values=COMPARE_VALUES, use_cache=True):
    """v2-vs-v3: one row per configuration present in EITHER root, joined on ``run_id``.

    ``old`` / ``new`` default to :data:`ROOT_V2` / :data:`ROOT_V3` AT CALL TIME (a default
    argument would have frozen them at import, so a test or a notebook that redirects the
    roots would have been ignored).

    Why this is a report and not a silent swap: v2 measured Sven with the per-step
    ``empty_cache()`` (up to 4.5x slower for full capture on CIFAR) and without
    ``expandable_segments``.  Those TWO settings are the whole difference -- the BatchNorm
    policy is not: v2's CIFAR Gram records all carry ``meta.freeze_norm_stats: False``, the
    same batch statistics v3 uses (v3's ``bn_mode`` plumbing keeps it that way now that
    C-E2 has removed the config key; see the module docstring).  Every number that moves is
    therefore expected to move DOWN for the Sven variants and to stay put for the baselines
    -- and a baseline that moved by more than the noise is a sign that the two passes ran
    on different hardware or a busy node, which is what ``gpu_old`` / ``gpu_new`` and the
    ``[calib]`` lines in the job logs are for.

    Returns an empty frame (no exception) while ``new`` has no results yet, so a notebook
    cell written today runs today and reports as soon as the re-profile lands.
    ``<value>_ratio`` is new / old.
    """
    old = ROOT_V2 if old is None else old
    new = ROOT_V3 if new is None else new
    if not has_profiles(old) or not has_profiles(new):
        print(f'v2-vs-v3 comparison: nothing to compare yet '
              f'(old={"ok" if has_profiles(old) else "missing"} {Path(old).name}, '
              f'new={"ok" if has_profiles(new) else "missing"} {Path(new).name})')
        return pd.DataFrame()
    for label, root in (('old', old), ('new', new)):
        done, n, exp = profile_status(root)
        if not done:
            print(f'v2-vs-v3 comparison: WARNING -- the {label} root {Path(root).name} is '
                  f'PARTIAL ({n}/{exp}); missing configurations show as NaN and its '
                  f'`rel_time` is normalised against whatever reference has landed.')
    keys = ['arch', 'config_name', 'study', 'method', 'run_id']
    cols = keys + [v for v in values] + ['status', 'gpu', 'n_steps']
    a = load_profiles(old, use_cache=use_cache)[cols]
    b = load_profiles(new, use_cache=use_cache)[cols]
    out = a.merge(b, on=keys, how='outer', suffixes=('_old', '_new'))
    for v in values:
        out[f'{v}_ratio'] = out[f'{v}_new'] / out[f'{v}_old']
    return out.sort_values(keys).reset_index(drop=True)


def compare_table(cmp: pd.DataFrame, study='methods', value='step_ms', n=None):
    """The readable form of :func:`compare_profiles`: old, new and the ratio of one value,
    biggest change first, for one study (default the per-architecture set points)."""
    if cmp.empty:
        return cmp
    d = cmp[cmp.study == study].copy() if study else cmp.copy()
    d['Method'] = d.method.map(label)
    out = pd.DataFrame({
        'arch': d.arch, 'Method': d.Method,
        f'{value} v2': d[f'{value}_old'].round(3), f'{value} v3': d[f'{value}_new'].round(3),
        'v3/v2': d[f'{value}_ratio'].round(3),
        'status v2': d.status_old, 'status v3': d.status_new,
    })
    out = out.reindex(out['v3/v2'].sub(1).abs().sort_values(ascending=False).index)
    return (out if n is None else out.head(n)).reset_index(drop=True)


def sven_change_table(cmp: pd.DataFrame, study='methods') -> pd.DataFrame:
    """The headline of the re-profile: per architecture and Sven variant, step time AND
    peak memory v2 -> v3 with both ratios, in :data:`ARCH_ORDER`.

    One table rather than two :func:`compare_table` calls, because the claim being made is
    a pair: the ``empty_cache`` fix buys step time and must NOT have moved memory (it
    changes when the allocator returns blocks, not how many are live).
    """
    if cmp.empty:
        return cmp
    d = cmp[cmp.study == study] if study else cmp
    d = d[d.method.map(is_sven)].copy()
    d['Method'] = d.method.map(label)
    out = pd.DataFrame({
        'arch': d.arch, 'Method': d.Method,
        'step ms v2': d.step_ms_old.round(1), 'step ms v3': d.step_ms_new.round(1),
        'step v3/v2': d.step_ms_ratio.round(3),
        'peak MB v2': d.peak_mb_old.round(0), 'peak MB v3': d.peak_mb_new.round(0),
        'mem v3/v2': d.peak_mb_ratio.round(3),
        'status v2': d.status_old, 'status v3': d.status_new,
    })
    order = {a: i for i, a in enumerate(ARCH_ORDER)}
    out = out.reindex(out.arch.map(lambda a: order.get(a, len(order)))
                      .sort_values(kind='stable').index)
    return out.reset_index(drop=True)


def baseline_control(cmp: pd.DataFrame, value='step_ms', study='methods', tol=0.10):
    """The control on the re-profile: by how much did the BASELINES move, per architecture?

    Only Sven called ``empty_cache``, so the FIRST fix cannot move a baseline at all and a
    baseline ratio far from 1 is not a result.  Three things can produce one, and the table
    is built to tell them apart:

    * the two passes did not measure the same machine -- different GPU model, a co-tenant,
      or host-CPU load on the cheap MLP steps (``campaign/stage0_reports/gpu.probe.md``
      measured the same MNIST-Adam step at 1.13 ms on a quiet node and 4.58 ms on a busy
      one).  Read the two ``[calib]`` lines in each profile job's log against each other;
    * the measurement was never single-regime.  ``steady v2`` / ``steady v3`` are the
      p90 / p10 of the worst entry's per-step series in each root: L-BFGS's strong-Wolfe
      line search costs one function evaluation or a dozen depending on where it is on the
      loss surface, and its series can switch regime mid-measurement (toy LBFGS 3-it. in
      v2: 15 steps at 14 ms, then 35 at 126 ms, steadiness 9.1).  A ratio whose ``steady``
      is far from 1 says the *optimizer's own* cost is bimodal, not that the pass changed;
    * the SECOND fix, ``expandable_segments``, is NOT Sven-only -- it is an allocator
      setting the whole process runs under, so an allocation-heavy baseline may legitimately
      move a little.  Only the first fix is Sven-specific.

    Only configurations whose status is in :data:`TIMED` in BOTH roots are counted (an
    ``oom``/``error`` has no step time to compare), and ``within`` is the per-architecture
    verdict at ``tol``.  Returns one row per architecture in :data:`ARCH_ORDER`.
    """
    if cmp.empty:
        return cmp
    d = cmp[cmp.study == study] if study else cmp
    d = d[~d.method.map(is_sven)
          & d.status_old.isin(TIMED) & d.status_new.isin(TIMED)]
    r = d[f'{value}_ratio']
    d = d[np.isfinite(r)]
    rows = []
    for arch, g in d.groupby('arch'):
        rr = g[f'{value}_ratio']
        worst = g.loc[rr.sub(1).abs().idxmax()]
        row = {'arch': arch, 'n baselines': len(g),
               'median v3/v2': round(float(rr.median()), 3),
               'min': round(float(rr.min()), 3), 'max': round(float(rr.max()), 3),
               'worst': label(worst.method),
               'worst v3/v2': round(float(worst[f'{value}_ratio']), 3)}
        for suf, tag in (('_old', 'v2'), ('_new', 'v3')):
            if f'steadiness{suf}' in g.columns:
                row[f'steady {tag}'] = round(float(worst[f'steadiness{suf}']), 2)
            # a `nonfinite` entry was still measured, but its optimizer diverged: its step
            # cost says nothing about the machine, so the status travels with the row
            row[f'status {tag}'] = worst[f'status{suf}']
        row[f'within {tol:.0%}'] = bool(rr.sub(1).abs().max() <= tol)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {a: i for i, a in enumerate(ARCH_ORDER)}
    out = out.reindex(out.arch.map(lambda a: order.get(a, len(order)))
                      .sort_values(kind='stable').index)
    return out.reset_index(drop=True)


def add_relative(df: pd.DataFrame) -> pd.DataFrame:
    """rel_time (vs Adam, else AdamW) and rel_mem (vs SGD) at matching arch / B / width."""
    df = df.copy()
    key = ['arch', 'B', 'width']
    ok = df[df.status.isin(TIMED) & (df.pf == 1.0) & (df.mb == 1)]
    def ref(methods, col):
        out = {}
        for m in methods:
            for k_, v in ok[ok.method == m].groupby(key, dropna=False)[col].mean().items():
                out.setdefault(k_, v)
        return out
    # rel_mem is "/ SGD" and only SGD: a silent fallback to Adam made the label a lie
    rt, rm = ref(['Adam', 'AdamW'], 'step_ms'), ref(['SGD'], 'peak_mb')
    keys = list(zip(df.arch, df.B, df.width))
    norm = lambda k_: tuple(None if (isinstance(x, float) and np.isnan(x)) else x for x in k_)
    rt = {norm(k_): v for k_, v in rt.items()}; rm = {norm(k_): v for k_, v in rm.items()}
    df['rel_time'] = [r / rt.get(norm(k_), np.nan) if r == r else np.nan for r, k_ in zip(df.step_ms, keys)]
    df['rel_mem'] = [r / rm.get(norm(k_), np.nan) if r == r else np.nan for r, k_ in zip(df.peak_mb, keys)]
    return df


def raw_steps(row, which='step_ms'):
    """The raw per-step series of one configuration (read back from its JSON)."""
    return np.asarray(json.loads(Path(row['_path']).read_text())['raw'].get(which, []), dtype=float)


def legend_below(fig, ax=None, ncol=4, fontsize=9):
    """One shared legend under the figure (keeps it off the data and the failure markers).

    Collects the handles of EVERY axes in the figure (``ax`` is accepted for backwards
    compatibility and ignored), so a method or failure marker that only appears in a
    later panel still gets an entry.
    """
    seen = {}
    for a in fig.axes:
        h, l = a.get_legend_handles_labels()
        for label, handle in zip(l, h):
            seen.setdefault(label, handle)
    fig.legend(seen.values(), seen.keys(), loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=ncol,
               fontsize=fontsize, frameon=False)


def save(fig, name, plot_dir):
    plot_dir = Path(plot_dir); plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / name, bbox_inches='tight')
    return plot_dir / name


def _fmt(v, unit=''):
    if v != v: return '--'
    return (f'{v:.3g}' if v < 100 else f'{v:,.0f}') + unit


# ---------------------------------------------------------------------------
# Study 1: method comparison
# ---------------------------------------------------------------------------
def method_table(df, arch, study='methods') -> pd.DataFrame:
    """One row per method at the set point: time, memory, ratios, status."""
    d = df[(df.arch == arch) & (df.study == study)].copy()
    d['Method'] = d.method.map(label)
    note = d.status.apply(lambda s: '' if s == 'ok' else STATUS_NOTE.get(s, s))
    out = pd.DataFrame({
        'Method': d.Method, 'Step (ms)': d.step_ms.round(2), 'x Adam': d.rel_time.round(2),
        'Peak mem (MB)': d.peak_mb.round(1), 'x SGD': d.rel_mem.round(2),
        'Capture (ms)': d.capture_ms.round(2), 'Solve+apply (ms)': d.solve_ms.round(2),
        'p90/p10': d.steadiness.round(2), 'Note': note})
    out['_s'] = d.family.map({'sven': 0, 'baseline': 1}).values
    return out.sort_values(['_s', 'Step (ms)'], na_position='last').drop(columns='_s').reset_index(drop=True)


def plot_method_bars(df, arch, ax, value='step_ms', study='methods', log=True):
    """Horizontal bars, one per method, sorted; non-ok configurations are annotated.

    A ``nonfinite`` row keeps its bar and its number (they were measured) and carries
    the crimson note as well."""
    d = df[(df.arch == arch) & (df.study == study)].copy()
    d = d.sort_values(value, ascending=True, na_position='first')
    y = np.arange(len(d))
    vals = d[value].fillna(0).values
    ax.barh(y, vals, color=[color(m) for m in d.method], edgecolor='k', lw=0.4)
    ax.set_yticks(y); ax.set_yticklabels([label(m) for m in d.method], fontsize=9)
    for yi, (_, r) in zip(y, d.iterrows()):
        if r.status in TIMED and r[value] == r[value]:
            ax.text(r[value], yi, ' ' + _fmt(r[value]), va='center', fontsize=8)
        if r.status != 'ok':
            ax.text(ax.get_xlim()[0] if not log else max(vals[vals > 0].min() if (vals > 0).any() else 1, 1e-3),
                    yi, ' ' + STATUS_NOTE.get(r.status, r.status), va='center', fontsize=8,
                    color='crimson')
    if log: ax.set_xscale('log')
    ax.set_xlabel({'step_ms': 'Amortised step time (ms)', 'peak_mb': 'Peak GPU memory (MB)',
                   'overhead_mb': 'Peak memory above model (MB)', 'rel_time': r'Step time / Adam',
                   'rel_mem': 'Peak memory / SGD'}.get(value, value))
    ax.set_title(ARCH_TITLES.get(arch, arch)); ax.grid(axis='x', ls='--', alpha=0.5)


def plot_heatmap(df, ax, value='rel_time', study='methods', archs=None, methods=None, fmt='{:.1f}'):
    """methods x architectures grid of a relative cost; OOM / infeasible / n/a cells are
    labelled, and a measured cell whose parameters went non-finite carries a ``*``."""
    d = df[df.study == study]
    archs = archs or [a for a in ARCH_ORDER if a in set(d.arch)]
    methods = methods or (SVEN + [m for m in COLORS if m not in SVEN])
    methods = [m for m in methods if m in set(d.method)]
    M = np.full((len(methods), len(archs)), np.nan); txt = [['' for _ in archs] for _ in methods]
    for i, m in enumerate(methods):
        for j, a in enumerate(archs):
            r = d[(d.method == m) & (d.arch == a)]
            if r.empty: txt[i][j] = '--'; continue
            r = r.iloc[0]
            if r.status in TIMED and r[value] == r[value]:
                M[i, j] = r[value]; txt[i][j] = fmt.format(r[value]) + ('' if r.status == 'ok' else '*')
            else: txt[i][j] = {'infeasible': 'inf.'}.get(r.status) or STATUS_NOTE.get(r.status, r.status)
    im = ax.imshow(np.log10(M), cmap='viridis_r', aspect='auto')
    for i in range(len(methods)):
        for j in range(len(archs)):
            ax.text(j, i, txt[i][j], ha='center', va='center', fontsize=8,
                    color='crimson' if np.isnan(M[i, j]) else 'w' if np.log10(M[i, j]) > np.nanmean(np.log10(M)) else 'k')
    ax.set_xticks(range(len(archs))); ax.set_xticklabels([ARCH_TITLES.get(a, a) for a in archs], rotation=25, ha='right')
    ax.set_yticks(range(len(methods))); ax.set_yticklabels([label(m) for m in methods], fontsize=9)
    return im


# ---------------------------------------------------------------------------
# Sweeps (batch size, chunk fraction, param fraction, microbatch, k, width)
# ---------------------------------------------------------------------------
def plot_sweep(df, arch, study, x, ax, value='step_ms', methods=None, style_by=None, logx=True, logy=True,
               analytic=None, ideal=None):
    """``value`` vs ``x`` for one study, a line per method (and per ``style_by`` value, e.g. mask_mode).

    Points with no measurement (oom / infeasible / error) are drawn as red crosses at the
    top of the axis so OOM walls are visible; a ``nonfinite`` point was timed, so it stays
    on its line and gets a crimson ring instead of being turned into a wall.
    ``analytic``: a column to overlay as a dotted line (e.g. 'analytic_jac_mb').
    ``ideal``: 'linear' draws the proportional-to-x reference through each method's x=max point.
    """
    d = df[(df.arch == arch) & (df.study == study)]
    methods = methods or [m for m in (SVEN + list(COLORS)) if m in set(d.method)]
    methods = list(dict.fromkeys(methods))
    ls_cycle = ['-', '--', ':', '-.']
    n_bad_rows = 0
    for m in methods:
        dm = d[d.method == m]
        groups = [(None, dm)] if style_by is None else list(dm.groupby(style_by, dropna=False))
        for gi, (gval, g) in enumerate(groups):
            if style_by is not None and gval == 'none':
                continue
            if style_by is not None:       # the unmasked pf=1 point belongs to every mask_mode line
                g = pd.concat([g, dm[dm[style_by] == 'none']])
            g = g.sort_values(x)
            ok = g[g.status.isin(TIMED)]
            lab = label(m) + (f' [{gval}]' if style_by is not None else '')
            ax.plot(ok[x], ok[value], marker=MARKERS.get(m, 'o'), ms=5, lw=2.2 if is_sven(m) else 1.4,
                    ls=ls_cycle[gi % 4] if style_by is not None else '-', color=color(m), label=lab)
            nf = ok[ok.status == 'nonfinite']
            if len(nf):    # timed, but the parameters diverged: same point, flagged
                ax.plot(nf[x], nf[value], MARKERS.get(m, 'o'), ms=9, mfc='none', color=color(m),
                        mec='crimson', mew=1.2, ls='none', label='non-finite parameters')
            bad = g[~g.status.isin(TIMED)]
            if len(bad):   # failures: method-coloured X with a red edge, stacked above the data
                top = (np.nanmax(d[value]) if d[value].notna().any() else 1) * (1.6 * 1.35 ** n_bad_rows)
                n_bad_rows += 1
                ax.plot(bad[x], [top] * len(bad), 'X', color=color(m), mec='crimson', mew=1.2, ms=9,
                        label=f'{label(m)}: ' + '/'.join(sorted(set(bad.status))))
            if analytic and ok[analytic].notna().any():
                ax.plot(ok[x], ok[analytic], ls=':', lw=1, color=color(m))
            if ideal == 'linear' and len(ok) > 1:
                x0, y0 = ok[x].iloc[-1], ok[value].iloc[-1]
                ax.plot(ok[x], y0 * ok[x] / x0, ls=':', lw=1, color=color(m), alpha=0.6)
    if logx: ax.set_xscale('log', base=2)
    if logy: ax.set_yscale('log')
    ax.set_xlabel({'B': 'Batch size $B$', 'chunk_fraction': 'Chunk size / $P$', 'pf': 'Parameter fraction',
                   'mb': 'Micro-batch size', 'k_fraction': '$k$ / rows', 'n_params': 'Parameters $P$'}.get(x, x))
    ax.set_ylabel({'step_ms': 'Step time (ms)', 'peak_mb': 'Peak GPU memory (MB)', 'overhead_mb': 'Peak memory above model (MB)',
                   'capture_ms': 'Capture time (ms)', 'solve_ms': 'Solve + apply time (ms)'}.get(value, value))
    ax.set_title(ARCH_TITLES.get(arch, arch)); ax.grid(True, which='both', ls='--', alpha=0.4)


def plot_pareto(df, arch, ax):
    """Chunk-fraction trade-off: peak memory vs step time, annotated with the chunk fraction."""
    d = df[(df.arch == arch) & (df.study == 'chunk_fraction') & df.status.isin(TIMED)].sort_values('chunk_fraction')
    ax.plot(d.peak_mb, d.step_ms, '-D', color=color('gram_chunked'), lw=2, label=label('gram_chunked'))
    for _, r in d.iterrows():
        ax.annotate(f"$f$={r.chunk_fraction:g}\n({int(r.n_groups)} grp)", (r.peak_mb, r.step_ms), fontsize=8,
                    textcoords='offset points', xytext=(6, 4))
    ref = df[(df.arch == arch) & (df.study == 'methods') & df.status.isin(TIMED)]
    for m in ['gram_hooks', 'gram_full', 'classic', 'Adam']:
        r = ref[ref.method == m]
        if len(r): ax.plot(r.peak_mb, r.step_ms, MARKERS.get(m, '*'), ms=10, color=color(m), label=label(m), mec='k')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Peak GPU memory (MB)'); ax.set_ylabel('Step time (ms)')
    ax.set_title(ARCH_TITLES.get(arch, arch) + ': chunk-size trade-off'); ax.grid(True, which='both', ls='--', alpha=0.4)


def plot_phase_bars(df, ax, archs=None, study='methods'):
    """Stacked capture vs solve+apply time for each Sven variant, grouped by architecture.

    A ``nonfinite`` variant keeps its bar (the phases were timed) with a ``*`` on its tick."""
    d = df[(df.study == study) & (df.family == 'sven') & df.status.isin(TIMED)]
    archs = archs or [a for a in ARCH_ORDER if a in set(d.arch)]
    xs, labs = [], []; x = 0
    for a in archs:
        for m in SVEN:
            r = d[(d.arch == a) & (d.method == m)]
            if r.empty: continue
            r = r.iloc[0]
            ax.bar(x, r.capture_ms, color=color(m), edgecolor='k', lw=0.4)
            ax.bar(x, r.solve_ms, bottom=r.capture_ms, color=color(m), alpha=0.35, hatch='//', edgecolor='k', lw=0.4)
            xs.append(x); labs.append(label(m).replace('Sven ', '') + ('' if r.status == 'ok' else '*')); x += 1
        x += 0.8
    ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=60, ha='right', fontsize=8)
    ax.set_yscale('log'); ax.set_ylabel('Time per step (ms)')
    ax.bar(0, 0, color='grey', label='capture (loss + Jacobian/Gram)'); ax.bar(0, 0, color='grey', alpha=0.35, hatch='//', label='solve + apply')
    ax.legend(fontsize=9)


def fit_exponent(x, y):
    """Slope of log y vs log x over the upper half of the range (the asymptotic regime)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[m], y[m]
    if len(x) < 3: return np.nan
    o = np.argsort(x); x, y = x[o][len(x) // 2 - 1:], y[o][len(y) // 2 - 1:]
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def scaling_table(df, arch, value='step_ms') -> pd.DataFrame:
    """Per method: value at the smallest / largest TIMED model (:data:`TIMED`, so a run
    whose parameters diverged still counts -- the column ``non-finite`` says how many),
    the growth factor, the fitted exponent in P and the first configuration with no
    measurement at all."""
    d = df[(df.arch == arch) & (df.study == 'width')]
    rows = []
    for m, g in d.groupby('method'):
        ok = g[g.status.isin(TIMED)].sort_values('n_params')
        bad = g[~g.status.isin(TIMED)].sort_values('n_params')
        rows.append({'Method': label(m), 'smallest P': _fmt(ok[value].iloc[0]) if len(ok) else '--',
                     'largest timed P': f"{int(ok.n_params.iloc[-1]):,}" if len(ok) else '--',
                     'at largest': _fmt(ok[value].iloc[-1]) if len(ok) else '--',
                     'growth': round(ok[value].iloc[-1] / ok[value].iloc[0], 1) if len(ok) > 1 else np.nan,
                     'exponent': round(fit_exponent(ok.n_params, ok[value]), 2),
                     'first failure': (f"{bad.status.iloc[0]} @ P={int(bad.n_params.iloc[0]):,}" if len(bad) and bad.n_params.notna().any()
                                       else (bad.status.iloc[0] if len(bad) else '')),
                     # Timed but diverged: not a failure of the measurement, still worth seeing.
                     'non-finite': int((ok.status == 'nonfinite').sum()) or ''})
    return pd.DataFrame(rows).sort_values('exponent').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------
def plot_steadiness(df, ax, threshold=1.3):
    """p90/p10 of the per-step times for every TIMED configuration; flags the unsteady ones."""
    d = df[df.status.isin(TIMED)].sort_values('steadiness')
    ax.plot(np.arange(len(d)), d.steadiness, '.', ms=3, color='k')
    ax.axhline(threshold, color='crimson', ls='--', lw=1)
    ax.set_yscale('log'); ax.set_xlabel('configuration (sorted)'); ax.set_ylabel('step time p90 / p10')
    bad = d[d.steadiness > threshold]
    ax.set_title(f'{len(bad)} of {len(d)} configurations above {threshold}')
    return bad[['arch', 'study', 'method', 'B', 'step_ms', 'step_ms_p10', 'step_ms_p90',
                'steadiness', 'status']]


def plot_raw_steps(row, ax):
    """Raw per-step times of one configuration with the summary statistics overlaid."""
    s = raw_steps(row)
    ax.plot(s, '.-', color='k', lw=0.8)
    for v, c, l in [(row.step_ms, 'crimson', 'cycle mean'), (row.step_ms_steady, 'C1', 'steady mean'),
                    (row.step_ms_median, 'C0', 'median'), (row.step_ms_mean, 'C2', 'plain mean')]:
        if v is None or v != v:
            continue
        ax.axhline(v, color=c, ls='--', lw=1, label=f'{l} {v:.2f} ms')
    ax.set_xlabel('measured step'); ax.set_ylabel('ms'); ax.legend(fontsize=8)
    ax.set_title(f"{row.arch} / {label(row.method)}")


def status_report(df) -> pd.DataFrame:
    """Every configuration that did not complete, with the reason."""
    d = df[df.status != 'ok']
    return d[['arch', 'study', 'method', 'B', 'pf', 'mask_mode', 'mb', 'width', 'status', 'error']].reset_index(drop=True)
