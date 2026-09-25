import json
import os
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ONE results root for the whole analysis layer (EXPERIMENTS.md 4.1)
# ---------------------------------------------------------------------------
# `experiment_results` is renamed to `experiment_results_legacy_2026-09-18/` for
# the campaign and a fresh, empty root takes its place, so every module has to
# agree on which root it reads -- and the legacy root has to stay switchable
# from one place (it used to be hard-coded four times: here twice,
# `scan_analysis.RESULTS_ROOT`, `sv_diagnostics.RESULTS_ROOT`).  `scan_analysis`
# and `sv_diagnostics` import :data:`RESULTS_ROOT` from here; every loader takes
# ``results_root=None`` and resolves it at CALL time, so setting the env var in
# a notebook cell (or a test) takes effect without re-importing anything.
DEFAULT_RESULTS_ROOT = '../experiment_results'
RESULTS_ROOT_ENV = 'SV3_RESULTS_ROOT'


def resolve_results_root(root=None):
    """The results root to read: an explicit ``root``, else ``$SV3_RESULTS_ROOT``,
    else :data:`DEFAULT_RESULTS_ROOT` (``'../experiment_results'``, relative to
    ``analysis/`` as every notebook runs)."""
    if root is not None:
        return root
    return os.environ.get(RESULTS_ROOT_ENV) or DEFAULT_RESULTS_ROOT


#: The root as resolved at import time -- for ``from style import RESULTS_ROOT``
#: and for printing.  Loaders re-resolve per call; prefer
#: :func:`resolve_results_root` over this constant in new code.
RESULTS_ROOT = resolve_results_root()

# The bulk of every SVD result file is diagnostics the analysis rarely needs:
# svd_info.svs (~2.4 MB/file, the per-step singular values) plus the per-BATCH
# arrays in losses (another ~0.5 MB). Together ~98% of the bytes. `slim=True`
# drops them so a scan loads (and, cached, reloads) in a fraction of the time;
# only the sv-spectra / batch-wise notebooks need them (pass slim=False there).
_DROP_TOP = ('svd_info',)
_DROP_LOSSES = ('batch_times_train', 'batch_times_val', 'train_batch', 'val_batch')


def _slim_record(r):
    for k in _DROP_TOP:
        r.pop(k, None)
    L = r.get('losses')
    if isinstance(L, dict):
        for k in _DROP_LOSSES:
            L.pop(k, None)
    return r


_CACHE_VERSION = 3  # bump when the slim schema / signature scheme changes


# ---------------------------------------------------------------------------
# Light / heavy result files
# ---------------------------------------------------------------------------
# New-format runs are split by generic_scan._write_run into a light
# {run_id}.jsonl (epoch curves + a per-epoch `svd_summary`) and a heavy
# diag/{run_id}.npz (per-batch arrays, Sven spectra). Old-format files carry
# everything inline. `load_results(slim=False)` re-attaches the heavy arrays to
# each record in the old inline layout, so notebook code that reads
# row['svd_info']['svs'] / ['num_nonzero_svs'] / row['losses']['train_batch']
# works unchanged on both layouts. NOTE: for new-format runs `svs` holds only
# every `svd_summary['spectra_every']`-th step (default 20); `svs_step` gives
# the step index of each saved spectrum.
_DIAG_LOSS_KEYS = ('train_batch', 'val_batch', 'batch_times_train',
                   'batch_times_val', 'train_batch_per_model')

# F20: `sv_min` and `sv_min_kept` are DIFFERENT quantities and must never share a
# column name.  Old records' `sv_min` is the smallest SV the optimizer *kept*
# (only the survivors of the rtol cut were recorded); new records log the full
# spectrum, so their `sv_min` is sigma_B -- numerical noise -- and the smallest
# inverted value is recorded separately as `sv_min_kept` (C-L1).  Both are passed
# through unchanged; a plot that wants "the smallest inverted SV" must ask for
# `sv_min_kept` and treat its absence as "legacy record, use `sv_min`" explicitly.
SV_MIN_KEYS = ('sv_min', 'sv_min_kept')
_DIAG_SV_SCALARS = ('sv_max', *SV_MIN_KEYS)


def load_diagnostics(row, name=None, results_root=None):
    """Heavy per-batch diagnostics for one run, as a dict of numpy arrays.

    ``row`` is a record/Series from :func:`load_results`. New-format runs are
    read from ``{results_root}/{name}/diag/{run_id}.npz`` (``name`` defaults to
    ``row['_scan']`` and the root to ``row['_results_root']``, both set by
    load_results, so a row from a FOREIGN root reads that root's diagnostics);
    legacy inline runs are re-read from
    their JSONL. Keys (when present): train_batch, val_batch, batch_times_train,
    batch_times_val, num_nonzero_svs, sv_max, sv_min, sv_min_kept (new records
    only -- see :data:`SV_MIN_KEYS`), svs (2-D, NaN-padded), svs_step, utr (2-D,
    NaN-padded, new records only), k_used, variable_k_substep_losses.
    """
    name = name or row.get('_scan')
    if name is None:
        raise ValueError("pass name= (scan directory name) or use a row from load_results")
    if results_root is None:
        stamped = row.get('_results_root')          # the root this row came from
        if isinstance(stamped, str) and stamped:
            results_root = stamped
    scan_dir = Path(resolve_results_root(results_root)) / name
    diag_file = row.get('diag_file')
    if diag_file:
        with np.load(scan_dir / diag_file) as z:
            return {k: z[k] for k in z.files}
    # Legacy inline layout: re-read the full JSONL for this run.
    f = scan_dir / f"{row['run_id']}.jsonl"
    if not f.is_file():
        return {}
    with open(f) as fh:
        rec = json.loads(fh.readline())
    out = {}
    L = rec.get('losses') or {}
    for k in _DIAG_LOSS_KEYS:
        if k in L:
            out[k] = np.asarray(L[k], dtype=np.float32)
    si = rec.get('svd_info') or {}
    if si.get('num_nonzero_svs'):
        out['num_nonzero_svs'] = np.asarray(si['num_nonzero_svs'], dtype=np.int32)
    if si.get('svs'):
        rows = [np.asarray(r, dtype=np.float32) for r in si['svs']]
        w = max(len(r) for r in rows)
        svs = np.full((len(rows), w), np.nan, dtype=np.float32)
        for i, r in enumerate(rows):
            svs[i, :len(r)] = r
        out['svs'], out['svs_step'] = svs, np.arange(len(rows), dtype=np.int32)
    if si.get('k_used'):
        out['k_used'] = np.asarray(si['k_used'], dtype=np.int32)
    return out


def spectra_list(diag):
    """``diag['svs']`` as a list of 1-D arrays with the NaN padding stripped
    (the shape notebook code expects from the old inline ``svd_info['svs']``)."""
    svs = diag.get('svs')
    if svs is None:
        return []
    return [r[~np.isnan(r)] for r in svs]


def _attach_diagnostics(rec, scan_dir):
    """Rebuild the legacy inline layout on a new-format light record."""
    if not rec.get('diag_file'):
        return rec  # legacy record (inline) or no diagnostics written
    with np.load(scan_dir / rec['diag_file']) as z:
        d = {k: z[k] for k in z.files}
    L = rec.setdefault('losses', {})
    for k in _DIAG_LOSS_KEYS:
        if k in d:
            L[k] = d[k].tolist()
    if rec.get('svd_summary') is not None:
        si = {'num_nonzero_svs': d['num_nonzero_svs'].tolist() if 'num_nonzero_svs' in d else [],
              'svs': spectra_list(d),
              'svs_step': d['svs_step'].tolist() if 'svs_step' in d else [],
              'k_used': d['k_used'].tolist() if 'k_used' in d else [],
              'variable_k_substep_losses': (d['variable_k_substep_losses'].tolist()
                                            if 'variable_k_substep_losses' in d else [])}
        if 'utr' in d:       # C-L1: U^T r, one row per saved step (NaN-padded)
            si['utr'] = spectra_list({'svs': d['utr']})
        for k in _DIAG_SV_SCALARS:   # sv_min / sv_min_kept stay separate (F20)
            if k in d:
                si[k] = d[k].tolist()
        rec['svd_info'] = si
    else:
        rec['svd_info'] = None
    return rec


def _dir_signature(files):
    """Content-sensitive cache key over ALL files: a hash of each file's
    (name, size, mtime_ns). Unlike a (count, max-mtime) key, this also detects
    an in-place overwrite of an existing run_id (re-running a config that
    regenerates the same filename) — count and max-mtime can both be unchanged
    there, but that file's size/mtime moves, so the hash changes.
    """
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    h.update(str(_CACHE_VERSION).encode())
    for f in files:  # files must be pre-sorted for a stable hash
        st = f.stat()
        h.update(f'{f.name}\0{st.st_size}\0{st.st_mtime_ns}\0'.encode())
    return (len(files), h.hexdigest())


_cache_warned = set()


def _write_cache(cache, payload):
    """Write the slim cache, tolerating a READ-ONLY results root.

    The legacy root becomes read-only for the campaign (EXPERIMENTS.md 4.1), and
    an unguarded ``mkdir`` / write there raises in the middle of a load that had
    already succeeded.  A failure is reported once per path and then ignored --
    the caller still gets its DataFrame, just no cache (pass ``cache_dir=`` to
    put the pickles on a writable disk instead).
    """
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Written atomically so a concurrent reader never sees a partial pickle.
        tmp = cache.with_suffix('.pkl.tmp')
        tmp.write_bytes(pickle.dumps(payload))
        tmp.replace(cache)
    except OSError as exc:
        if str(cache) not in _cache_warned:
            _cache_warned.add(str(cache))
            print(f'[style] could not write the slim cache {cache} ({exc.strerror}); '
                  f'loading uncached -- pass cache_dir= for a writable location')


def _stamp_root(df, root):
    """Record on every row WHICH results root the frame was loaded from.

    A notebook may load a scan from a foreign root (``results_root=LEGACY_ROOT`` in
    ``finetune_analysis``, a scan
    that exists in both roots), and the frame then travels through helpers that take
    no root argument.  Without this column they fall back to the PROCESS default and
    read the wrong scan directory: ``analysis_helpers.expected_run_ids`` on a legacy
    ``mnist_scan_ce`` frame read the FRESH manifest and reported 1610 expected runs
    and 122 "missing" configurations for a 1040-run legacy scan.

    Provenance, never a configuration column (see :data:`PROVENANCE_COLUMNS`), and
    stamped on the way OUT of the cache too, so the pickles stay compatible in both
    directions.  Read it back with :func:`frame_results_root`.
    """
    df['_results_root'] = str(root)
    return df


def frame_results_root(df, scan=None):
    """The results root ``df`` was loaded from (:func:`_stamp_root`), or None.

    ``scan``: restrict to the rows of that ``_scan`` -- a frame concatenated from two
    roots has one root per scan.  None when the frame carries no ``_results_root``
    (a hand-built or pre-existing frame) or when the rows disagree, so the caller
    falls back to :func:`resolve_results_root` exactly as before.
    """
    if getattr(df, 'columns', None) is None or '_results_root' not in df.columns:
        return None
    sub = df if scan is None or '_scan' not in df.columns else df[df['_scan'] == scan]
    roots = [str(v) for v in pd.Series(sub['_results_root']).dropna().unique()]
    return roots[0] if len(roots) == 1 else None


def load_results(name, results_root=None, selection_fn=None,
                 slim=True, use_cache=True, cache_dir=None):
    """Load experiment results into a DataFrame (new per-run directory format,
    with a fallback to the legacy single ``{name}.jsonl`` file).

    results_root: the scan directory's parent; ``None`` resolves it through
        :func:`resolve_results_root` (``$SV3_RESULTS_ROOT``, else
        ``'../experiment_results'``).
    cache_dir: where the slim cache pickles go.  Default ``{results_root}/_cache``;
        point it elsewhere when the results root is read-only (the write is
        guarded either way, see :func:`_write_cache`).
    slim (default True): drop the heavy diagnostics (``svd_info`` and the per-batch
        arrays) — ~98% of the bytes, unused by most notebooks. Pass ``slim=False``
        for the sv-spectra / batch-wise analyses that need them: for new-format
        (light + diag/*.npz) runs they are then read from the npz and attached in
        the old inline layout (see :func:`load_diagnostics`). Every record gets a
        ``_scan`` column (the scan name) for :func:`load_diagnostics`, and every
        frame a ``_results_root`` column (the root it was loaded FROM) for the
        helpers that take no root argument -- :func:`_stamp_root`.
    use_cache (default True): for a slim, unfiltered load, cache the result to
        ``{results_root}/_cache/{name}.slim.pkl`` and reuse it while the scan dir
        is unchanged. The cache key is content-sensitive (see :func:`_dir_signature`),
        so it invalidates on new files, removed files, AND in-place re-runs of an
        existing run_id. First build reads every file once; reloads are near-instant.
    """
    root = Path(resolve_results_root(results_root))
    scan_dir = root / name
    cacheable = slim and use_cache and selection_fn is None and scan_dir.is_dir()

    # Glob ONCE and reuse the same file list for the signature and the load, so
    # the cached signature always matches the data actually loaded (no race
    # between a check-glob and a load-glob while a job is still writing files).
    files = sorted(scan_dir.glob('*.jsonl')) if scan_dir.is_dir() else []

    if cacheable:
        cache = Path(cache_dir or (root / '_cache')) / f'{name}.slim.pkl'
        sig = _dir_signature(files)
        if cache.is_file():
            try:
                blob = pickle.loads(cache.read_bytes())
                if blob.get('sig') == sig:
                    print(f"Loaded {len(blob['df'])} runs from {cache} (slim cache)")
                    return _stamp_root(blob['df'], root)
            except Exception:
                pass  # stale/corrupt cache -> rebuild

    records: list[dict] = []

    # New format: directory of per-run JSONL files
    if files:
        for f in files:
            if selection_fn is not None and not selection_fn(f.name):
                continue
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        rec['_scan'] = name
                        records.append(_slim_record(rec) if slim
                                       else _attach_diagnostics(rec, scan_dir))
        if records:
            print(f"Loaded {len(records)} runs from {scan_dir}/ (directory format"
                  f"{', slim' if slim else ''})")
            df = pd.DataFrame(records)
            if cacheable:
                # Re-signature from the same `files` list (unchanged since the glob).
                _write_cache(cache, {'sig': sig, 'df': df})
            return _stamp_root(df, root)

    # Old format: single JSONL file
    jsonl_path = root / f'{name}.jsonl'
    if jsonl_path.is_file():
        with open(jsonl_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    records.append(_slim_record(rec) if slim else rec)
        print(f"Loaded {len(records)} runs from {jsonl_path} (single-file format)")
        return _stamp_root(pd.DataFrame(records), root)

    raise FileNotFoundError(
        f"No results found for '{name}': tried {scan_dir}/ and {jsonl_path}"
    )

def load_results_jsonl(name, results_root=None, slim=True):
    """Back-compat shim. Results are now stored per-run in a ``{name}/`` directory
    (the fresh Gram-backend runs); the old single ``{name}.jsonl`` file is gone.
    Delegates to :func:`load_results` (which handles both layouts + the slim cache).
    """
    return load_results(name, results_root=results_root, slim=slim)


# ---------------------------------------------------------------------------
# The manifest: what the scan INTENDS to contain (C-R2 / C-A2)
# ---------------------------------------------------------------------------
# Each job writes the run_ids it is responsible for to `{scan}/manifest/{job}.json`
# (`experiments/experiment_code/claims.py`); the union over jobs is the intended
# grid.  Without it a configuration whose runs never produced a file is simply
# absent from every table -- `n_missing` could only ever count the seeds MISSING
# FROM A CONFIG THAT HAS AT LEAST ONE FILE.  Reading it here (rather than
# importing `claims.read_manifest_union`) keeps the analysis free of
# `experiments.experiment_code`, whose package __init__ imports torch.
MANIFEST_DIRNAME = 'manifest'


def manifest_run_ids(name, results_root=None):
    """The union of ``{name}/manifest/*.json`` ``run_ids`` -- the scan's intended
    grid -- as a set.  Empty when the scan has no manifest (every legacy scan)."""
    out = set()
    d = Path(resolve_results_root(results_root)) / name / MANIFEST_DIRNAME
    if not d.is_dir():
        return out
    for f in sorted(d.glob('*.json')):
        if f.name.startswith('.tmp-'):
            continue
        try:
            payload = json.loads(f.read_text())
        except (OSError, ValueError):
            print(f'[style] unreadable manifest {f}')
            continue
        out.update(payload.get('run_ids') or ())
    return out


_SEED_SUFFIX_RE = re.compile(r'_mseed-?\d+(_lseed-?\d+)?')


def config_key(run_id):
    """A run_id with its seed suffix (``_mseed1000_lseed1000``) removed.

    Two runs of the SAME configuration differ only in that suffix, so this is a
    configuration's identity as a *string* -- available for a run that has no
    record at all, which the hyperparameter columns are not.  Verified on
    ``toy_1d_scan``: 734 files -> 148 keys, the same 148 configurations
    :func:`analysis_helpers.config_table` finds by grouping on columns.
    """
    return _SEED_SUFFIX_RE.sub('', str(run_id))


def expected_per_config(run_ids):
    """``{config_key: number of runs the manifest expects}`` for that config."""
    out = {}
    for rid in run_ids:
        key = config_key(rid)
        out[key] = out.get(key, 0) + 1
    return out


def expected_runs(config_keys, per_config, seed_count):
    """How many runs ONE ROW of a config table was supposed to have (C-A2).

    ``config_keys`` must be EVERY :func:`config_key` in that row's group, not just
    the first one: a selector may group more coarsely than a run configuration
    (`scan_analysis.SVEN_CONFIG` is only ``k``/``lr``/``rtol``, so one row can span
    several ``n_train`` / ``kappa`` / ``microbatch_size`` values, as
    ``exp_finetune_cifar_smallN`` and the kappa / microbatch scans do).  Reading one
    key then under-counts the row and hides the missing runs of every other
    configuration in it -- exactly the invisibility C-A2 removes.

    Without a manifest (``per_config`` empty) there is nothing to read, so the
    binding rule stands (section 1 of EXPERIMENTS.md): a configuration is
    eligible when more than half of the scan's SEEDS finished, i.e. ``seed_count``
    is the expectation.  ``seed_count`` is also the per-key fallback for a key a
    partial manifest does not list.
    """
    if not per_config:
        return int(seed_count)
    return int(sum(per_config.get(key, seed_count) for key in config_keys))


def missing_run_ids(df, expected):
    """The run_ids in ``expected`` (a manifest union) that have no record in ``df``."""
    if not expected:
        return []
    have = set(df['run_id']) if 'run_id' in df.columns else set()
    return sorted(set(expected) - have)


# ---------------------------------------------------------------------------
# Run status (C-R1, schema 2) -- inert on legacy records, which have no `status`
# ---------------------------------------------------------------------------
STATUS_OK = 'ok'
STATUS_DIVERGED = 'diverged'
#: An `oom` / `error` run is an *incomplete attempt*: the runner retries it
#: (C-R1).  It is excluded from every mean and counted separately -- not folded
#: into `n_diverged`, which is a statement about the optimizer, not the cluster.
FAILED_STATUSES = ('oom', 'error')


def status_of(row):
    """A record's ``status``, defaulting to ``'ok'`` for legacy records (no such
    field) and for a NaN left by pandas when only some records carry one."""
    s = row.get('status') if hasattr(row, 'get') else None
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return STATUS_OK
    return str(s)


def is_failed(status):
    """Whether ``status`` marks an incomplete attempt (:data:`FAILED_STATUSES`)."""
    return str(status) in FAILED_STATUSES


# ---------------------------------------------------------------------------
# Configuration identity: an ALLOW-LIST of hyperparameters (C-A2)
# ---------------------------------------------------------------------------
# This USED to be auto-detected ("every scalar column that is not on a small
# denylist"), here and in `analysis_helpers.config_columns`.  Schema 2 adds
# `status`, `run_hash`, `git_sha`, `host`, timestamps, `n_test`,
# `steps_per_epoch`, ... to every record, and auto-detection would take each of
# them as part of a configuration's identity: every run would become its own
# configuration, every seed mean would be a single run, and every table would
# silently collapse (scout report, section 4a).  An allow-list cannot fail that
# way: a column nobody listed is ignored, and if it varies it is reported by
# :func:`_warn_unlisted_columns` rather than fragmenting the grouping.
#
# To add a hyperparameter: put it here.  Everything a RUN records about ITSELF
# (where it ran, when, which code, how big the data was) belongs in
# :data:`PROVENANCE_COLUMNS`.
HPARAM_COLUMNS = (
    # what is being optimised, and how
    'optimizer', 'loss', 'batch_size', 'lr', 'weight_decay',
    # Sven
    'k', 'k_fraction', 'rtol', 'kappa', 'svd_mode', 'decomposition', 'use_gram',
    'variable_k', 'signed_residual', 'microbatch_size', 'param_fraction',
    'mask_mode', 'gram_freeze_norm_stats',
    # baselines
    'lbfgs_max_iter', 'lbfgs_history_size', 'lbfgs_line_search_fn',
    'polyak_f_star', 'polyak_max_lr', 'polyak_eps',
    'aggregator', 'inner_optimizer', 'tau', 'rmsProp', 'alpha_rmsProp',
    # model / data / schedule axes that studies sweep
    'mlp_width', 'n_train', 'n_data', 'num_epochs', 'split_seed', 'data_seed',
    # schema 2
    'bn_mode',
)

#: Backend / memory-layout choices that compute the SAME update (chunked vs full
#: Gram capture and the chunk size).  Not part of a configuration's identity: the
#: CIFAR label-reg scan captured some seeds of a config 'chunked' and others
#: 'full', and keying on them split every such config into 3+2 / 4+1 seed
#: fragments that were then reported as "missing seeds" and mostly ineligible.
BACKEND_COLUMNS = ('gram_capture', 'gram_chunk_numel')

#: Bookkeeping, provenance (C-R3) and recorded facts (C-R4).  Never a
#: configuration; never warned about.  `n_train` is deliberately NOT here -- the
#: overparam studies sweep it -- but `n_val` / `n_test` / `n_params` /
#: `steps_per_epoch` are consequences of the config, not knobs.
#:
#: The schema-2 block at the end is what the campaign runner records ABOUT a run
#: (found empirically by loading every fresh scan and collecting the
#: "AVERAGED OVER" warnings, 2026-09-20):
#:
#: * ``effective_loader_seed`` -- the data-order seed actually used (derived from
#:   ``loader_seed`` / ``data_seed``), not a knob;
#: * ``checkpoint_policy`` / ``ckpt_init_file`` / ``ckpt_error`` -- which
#:   checkpoints the run was asked to write, where its shared per-seed initial
#:   state lives, and why a write failed (``checkpoints`` / ``ckpt_file`` were
#:   already here);
#: * ``train_eval_size`` / ``eval_every_steps`` -- the EVALUATION protocol (like
#:   ``eval_batch_size``): they change what is measured, not what is optimised;
#: * ``svd_spectra_schedule`` / ``svd_summary`` -- diagnostic logging cadence and
#:   its per-epoch summary (siblings of ``svd_info``);
#: * ``muon_variant`` / ``muon_rule`` -- the Muon parameter-grouping rule the run
#:   actually built (C-B5): a recorded fact, while the CHOICE is ``optimizer``
#:   (``Muon`` vs ``MuonW``).
PROVENANCE_COLUMNS = (
    'run_id', 'model_seed', 'loader_seed', '_scan', '_results_root',
    'diag_file', 'ckpt_file',
    'status', 'error', 'diverged_at_step', 'run_hash', 'schema_version',
    'git_sha', 'git_dirty', 'git_source', 'sven_git_sha', 'sven_git_dirty',
    'sven_git_source', 'host', 'slurm_job_id', 'n_shards', 'shard_id',
    'python_version', 'torch_version', 'cuda_version', 'gpu_name',
    'collected_at', 'start_time', 'start_unix', 'end_time', 'end_unix',
    'wall_time_s', 'n_params', 'n_val', 'n_test', 'steps_per_epoch',
    'actual_param_fraction', 'eval_batch_size', 'checkpoints', 'svd_info',
    # schema 2 (campaign, 2026-09)
    'effective_loader_seed', 'checkpoint_policy', 'ckpt_init_file', 'ckpt_error',
    'train_eval_size', 'eval_every_steps', 'svd_spectra_schedule', 'svd_summary',
    'muon_variant', 'muon_rule',
)

#: Scalar quantity columns added by the derived-column helpers -- outcomes, never
#: configuration.  `final_test_*` are outcomes shown BESIDE a selected config
#: (C-E1); no selector may rank on them (:func:`assert_selection_metric`).
#:
#: The second block is the schema-2 outcome summary the RUNNER writes onto every
#: record (``generic_scan.summarize_curves`` + the three ``_final`` lines): these
#: are the same quantities the derived columns recompute from the curves, so they
#: are outcomes for exactly the same reason.  ``test`` / ``test_acc`` are matched
#: by :func:`is_test_metric`, so no selector can rank on them either.  The third
#: block is the standalone-timing join (:func:`scan_analysis.attach_standalone_times`).
OUTCOME_COLUMNS = (
    'final_val_loss', 'final_train_loss', 'final_val_acc', 'final_train_acc',
    'final_test_loss', 'final_test_acc', 'val_ppl',
    'total_time', 'avg_epoch_time', 'avg_batch_time_train', 'avg_batch_time_val',
    'peak_gpu_mem_mb', 'effective_bs', 'diverged', 'failed', 'method',
    # schema 2: the runner's own outcome summary (C-E5 / C-E1)
    'val_final', 'val_best', 'val_best_index', 'val_last3_mean',
    'test', 'test_acc', 'train_eval_final',
    # the standalone timing join
    'time_excl_first_epoch', 'standalone_total_time', 'standalone_avg_epoch_time',
    'standalone_avg_batch_time_train', 'standalone_avg_batch_time_val',
    'standalone_peak_gpu_mem_mb', 'standalone_time_excl_first_epoch',
)
# Back-compat alias (was the auto-detection denylist).
_DERIVED_QUANTITY_COLS = set(OUTCOME_COLUMNS)

_HPARAM_SET = set(HPARAM_COLUMNS)
_NEVER_CONFIG = set(PROVENANCE_COLUMNS) | set(BACKEND_COLUMNS) | set(OUTCOME_COLUMNS)
_SCALAR = (str, bool, int, float, np.integer, np.floating)
_unlisted_warned = set()


def _warn_unlisted_columns(df, skip):
    """One-time warning for a VARYING scalar column that is neither an allow-listed
    hyperparameter nor known bookkeeping -- i.e. a new knob whose runs would be
    averaged together.  The allow-list's one failure mode, made visible."""
    for col in df.columns:
        if col in _HPARAM_SET or col in _NEVER_CONFIG or col in skip:
            continue
        if col in _unlisted_warned or col.endswith(('_std', '_min')):
            continue
        vals = df[col].dropna()
        if vals.empty or not isinstance(vals.iloc[0], _SCALAR):
            continue
        if vals.nunique() > 1:
            _unlisted_warned.add(col)
            print(f"[style] column {col!r} varies but is not in style.HPARAM_COLUMNS: "
                  f"its values are being AVERAGED OVER.  Add it there if it is a "
                  f"hyperparameter, or to PROVENANCE_COLUMNS if it is not.")


def hparam_columns(df, seed_col='model_seed', extra_skip=(), warn=True):
    """The columns that jointly identify a configuration: the allow-listed
    hyperparameters (:data:`HPARAM_COLUMNS`) this DataFrame actually carries with
    a non-null value, in column order.  See the note above the allow-list."""
    skip = {seed_col, *extra_skip}
    if warn:
        _warn_unlisted_columns(df, skip)
    out = []
    for col in df.columns:
        if col not in _HPARAM_SET or col in skip:
            continue
        vals = df[col].dropna()
        if vals.empty or not isinstance(vals.iloc[0], _SCALAR):
            continue
        out.append(col)
    return out


# ---------------------------------------------------------------------------
# The selection metric may never come from the test split (C-E1 / C-A2)
# ---------------------------------------------------------------------------
# Selection uses VALIDATION only; test numbers are reported for the configuration
# validation already chose.  The chokepoints are `Scan.configs(metric)` and
# `analysis_helpers.config_table(metric)` -- both also reached through the
# `quantity=` argument of `plot_knob_summary` / `plot_time_vs_k`, which is why
# the check lives inside them and not in the notebooks.
_TEST_METRIC_RE = re.compile(r'(^|_)test(_|$)')


def is_test_metric(metric):
    """Whether ``metric`` names a quantity derived from the test split."""
    return bool(_TEST_METRIC_RE.search(str(metric)))


def assert_selection_metric(metric, where=''):
    """Raise unless ``metric`` may be used to RANK configurations.

    Returns ``metric`` so it can wrap an assignment."""
    if is_test_metric(metric):
        raise ValueError(
            f"{where or 'selection'}: {metric!r} is derived from the test split. "
            f"Selection uses the seed-mean final VALIDATION loss "
            f"(EXPERIMENTS.md section 5); test outcomes are reported beside the "
            f"configuration validation chose, never used to choose it.")
    return metric


def _stack_arrays(arrays):
    """Stack a list of arrays after truncating to the minimum length. Returns 2-D ndarray."""
    arrs = [np.asarray(a) for a in arrays if a is not None]
    if not arrs:
        return None
    min_len = min(len(a) for a in arrs)
    return np.stack([a[:min_len] for a in arrs], axis=0)  # shape (n_seeds, T)


def _avg_arrays(arrays):
    """Element-wise mean of a list of arrays, truncated to the minimum length."""
    stacked = _stack_arrays(arrays)
    return None if stacked is None else np.mean(stacked, axis=0).tolist()


def _std_arrays(arrays):
    """Element-wise std (ddof=1) of a list of arrays, truncated to the minimum length."""
    stacked = _stack_arrays(arrays)
    if stacked is None:
        return None
    ddof = 1 if stacked.shape[0] > 1 else 0
    return np.std(stacked, axis=0, ddof=ddof).tolist()


def _avg_dicts(dicts):
    """Mean of a list of dicts whose values are arrays or scalars."""
    dicts = [d for d in dicts if isinstance(d, dict)]
    if not dicts:
        return None
    result = {}
    for k in dicts[0]:
        vals = [d[k] for d in dicts if k in d and d[k] is not None]
        if not vals:
            result[k] = None
            continue
        if isinstance(vals[0], (list, np.ndarray)):
            result[k] = _avg_arrays(vals)
        elif isinstance(vals[0], (int, float, np.integer, np.floating)):
            result[k] = float(np.mean(vals))
        else:
            result[k] = vals[0]
    return result


def _std_dicts(dicts):
    """Std (ddof=1) of a list of dicts whose values are arrays or scalars."""
    dicts = [d for d in dicts if isinstance(d, dict)]
    if not dicts:
        return None
    n = len(dicts)
    result = {}
    for k in dicts[0]:
        vals = [d[k] for d in dicts if k in d and d[k] is not None]
        if not vals:
            result[k] = None
            continue
        if isinstance(vals[0], (list, np.ndarray)):
            result[k] = _std_arrays(vals)
        elif isinstance(vals[0], (int, float, np.integer, np.floating)):
            ddof = 1 if n > 1 else 0
            result[k] = float(np.std(vals, ddof=ddof))
        else:
            result[k] = None
    return result


def average_over_seeds(df, seed_col='model_seed', config_cols=None):
    """Average all quantities over model seeds for each distinct configuration.

    Groups the DataFrame by *config_cols* (the hyperparameter axes) and computes
    mean and standard deviation over the different values of *seed_col*.  Works on
    both raw DataFrames and ones processed by ``add_derived_columns``.

    For every averaged quantity column a parallel ``{col}_std`` column is added
    with the same structure (scalar → scalar std, list → list of per-step stds,
    dict → dict of stds).  Use these for ``ax.fill_between`` error bands.

    Supported column value types:

    * **dict** (e.g. ``losses``, ``svd_info``) — each key is averaged
      independently; array-valued keys are averaged element-wise (truncated to
      the shortest seed's length).  A ``{col}_std`` dict mirrors the structure.
    * **list / ndarray** — element-wise mean/std, truncated to shortest length.
    * **scalar numeric** — arithmetic mean / std (ddof=1).
    * **other** (str, None, …) — first non-null value kept as-is; std is None.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame, one row per run.
    seed_col : str
        Column that identifies the random seed to average over.
    config_cols : list[str] or None
        Columns that jointly identify a unique configuration (groupby keys).
        If *None*, :func:`hparam_columns` -- the allow-list, NOT auto-detection:
        the provenance columns of schema 2 would otherwise make every run its
        own "configuration" (C-A2).

    Returns
    -------
    pd.DataFrame
        One row per distinct configuration.  *seed_col* is replaced by
        ``n_seeds``.  For each quantity column ``col``, a ``col_std`` column
        is inserted immediately after it.
    """
    df = df.copy()

    if config_cols is None:
        config_cols = hparam_columns(df, seed_col)

    quantity_cols = [c for c in df.columns if c != seed_col and c not in config_cols]

    rows = []
    for config_vals, group in df.groupby(config_cols, dropna=False, sort=True):
        if not isinstance(config_vals, tuple):
            config_vals = (config_vals,)
        row = dict(zip(config_cols, config_vals))
        row['n_seeds'] = len(group)

        for col in quantity_cols:
            vals = group[col].tolist()
            # NaN is not a value: a column that is a string for SOME optimizers and
            # absent for the rest (`gram_capture`, `muon_variant` / `muon_rule`)
            # comes back as NaN where it does not apply, and pandas puts those rows
            # first.  Taking the NaN as `first_valid` sent such a column down the
            # numeric branch, where `np.mean` then met the strings and raised
            # "the resolved dtypes are not compatible with add.reduce".
            first_valid = next((v for v in vals if v is not None
                                and not (isinstance(v, float) and np.isnan(v))), None)
            if first_valid is None:
                row[col] = None
                row[f'{col}_std'] = None
            elif isinstance(first_valid, dict):
                row[col] = _avg_dicts(vals)
                row[f'{col}_std'] = _std_dicts(vals)
            elif isinstance(first_valid, (list, np.ndarray)):
                row[col] = _avg_arrays(vals)
                row[f'{col}_std'] = _std_arrays(vals)
            elif isinstance(first_valid, (int, float, np.integer, np.floating)):
                # only the actual numbers: a mixed column (see above) must not drag
                # a string into np.mean
                numeric = [v for v in vals
                           if isinstance(v, (int, float, np.integer, np.floating))
                           and not (isinstance(v, float) and np.isnan(v))]
                row[col] = float(np.mean(numeric)) if numeric else float('nan')
                ddof = 1 if len(numeric) > 1 else 0
                row[f'{col}_std'] = float(np.std(numeric, ddof=ddof)) if numeric else float('nan')
            else:
                row[col] = first_valid
                row[f'{col}_std'] = None

        rows.append(row)

    # Column order: config cols, n_seeds, then quantity / quantity_std pairs
    result = pd.DataFrame(rows)
    ordered = list(config_cols) + ['n_seeds']
    for col in quantity_cols:
        ordered.append(col)
        std_col = f'{col}_std'
        if std_col in result.columns:
            ordered.append(std_col)
    return result[[c for c in ordered if c in result.columns]]


def set_style():
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'legend.fontsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2,
        'axes.grid': False,
        # C27: a fallback list instead of a bare 'arial' (absent on Linux -> warning spam)
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
        'legend.frameon': False,
        'mathtext.fontset': 'cm',
    })
    
lr_labels = {10**k: f"10^{{{k}}}" for k in range(-10,10)}
lr_labels[0.5] = "0.5"
lr_labels[0.0003] = "3 \\times 10^{-4}"
lr_labels[0.003] = "3 \\times 10^{-3}"
lr_labels[0.03] = "3 \\times 10^{-2}"
lr_labels[0.05] = "0.05"
lr_labels[50.0] = "50"


# ---------------------------------------------------------------------------
# Optimizer colours -- THE convention, for every notebook and helper module
# ---------------------------------------------------------------------------
# One colour per optimizer, keyed by name, so a method looks the same in the scan
# notebooks, the study notebooks and the profiling plots.  Never colour an optimizer
# positionally (f'C{i}', a palette indexed by plot order, matplotlib's default
# cycle): the colour then depends on which methods happen to be present and how
# they are sorted.  Look it up with :func:`method_color` instead.
#
# Sven is black.  Baselines are seaborn "deep" plus three extra hues chosen to stay
# apart from it (olive, navy, dark grey).  To add an optimizer, add a line here.
METHOD_COLORS = {
    'Sven':      '#000000',
    'SGD':       '#55A868',   # green
    'SGDm':      '#8FC99C',   # light green (SGD at momentum 0.9, C-B2 -- as AdamW is to Adam)
    'PolyakSGD': '#CCB974',   # khaki
    'RMSprop':   '#C44E52',   # red
    'Adam':      '#4C72B0',   # blue
    'AdamW':     '#64B5CD',   # light blue
    'LBFGS':     '#8C8C8C',   # grey
    'Muon':      '#8172B3',   # purple
    'MuonW':     '#B7A9DB',   # light purple (Muon at its default wd = 0.1, as AdamW is to Adam)
    'SOAP':      '#DD8452',   # orange
    'Shampoo':   '#DA8BC3',   # pink
    'KFAC':      '#937860',   # brown
    'JD':        '#6B8E23',   # olive
    'HIG':       '#1B3A57',   # navy
}
# Other spellings of the same optimizer (result files, profiler, legacy labels).
METHOD_ALIASES = {
    'SVD': 'Sven', 'sven': 'Sven',
    'JD_UPGrad': 'JD', 'JD (UPGrad)': 'JD',
    'LBFGS1': 'LBFGS', 'K-FAC': 'KFAC', 'Polyak': 'PolyakSGD',
}
_UNKNOWN_METHOD_COLOR = '#333333'


def canonical_method(name):
    """The name an optimizer goes by in :data:`METHOD_COLORS` (``'SVD'`` -> ``'Sven'``)."""
    return METHOD_ALIASES.get(name, name)


def method_color(name):
    """The global colour of an optimizer.  Unknown names get a dark grey and a
    one-time warning, so a new optimizer is noticed rather than silently cycled."""
    key = canonical_method(name)
    if key not in METHOD_COLORS:
        if key not in method_color._warned:
            method_color._warned.add(key)
            print(f"[style] no colour registered for optimizer {name!r}; "
                  f"add it to style.METHOD_COLORS")
        return _UNKNOWN_METHOD_COLOR
    return METHOD_COLORS[key]


method_color._warned = set()


# ---------------------------------------------------------------------------
# Optimizer DISPLAY names -- what a legend / table column shows (C-B7, C-B2)
# ---------------------------------------------------------------------------
# The record's `optimizer` is the machine key ('LBFGS', 'SGDm', 'JD_UPGrad') and
# stays the key everywhere -- in `METHOD_COLORS`, in a `method` column, in a
# groupby.  What a READER sees is this map, so the name in a figure says what was
# actually run:
#
# * `LBFGS` is `torch.optim.LBFGS` on MINIBATCHES, not full-batch L-BFGS -- the
#   single most misleading label in the old plots (C-B7 / F33);
# * `SGDm` is `torch.optim.SGD(momentum=0.9)`, `SGD` is momentum 0 (C-B2);
# * `AdamW` / `MuonW` are Adam / Muon at their own default weight decay.
#
# Look a name up with :func:`method_label`; a method with no entry displays as
# itself, so adding an optimizer needs no change here.
METHOD_LABELS = {
    'LBFGS':     'Stochastic L-BFGS',
    'SGDm':      'SGD + momentum',
    'JD':        'JD (UPGrad)',
    'PolyakSGD': 'Polyak SGD',
}


def method_label(name):
    """The display name of an optimizer (:data:`METHOD_LABELS`), for legends,
    tick labels and table columns.  Aliases are canonicalised first, so
    ``'SVD'``/``'LBFGS1'``/``'JD_UPGrad'`` give the same answer as their
    canonical spelling; an unregistered name displays unchanged."""
    key = canonical_method(name)
    return METHOD_LABELS.get(key, key)


# ---------------------------------------------------------------------------
# Two conventions shared by every notebook (see EXPERIMENTS.md section 12, A4 / A5)
# ---------------------------------------------------------------------------
DIVERGED_FACTOR = 10.0


def final_value(curve):
    """The final value of a per-epoch curve -- NaN if it ends non-finite.

    DIVERGED = FAILED.  A diverged run (see :func:`is_diverged`) has no final loss: it
    is left out of its configuration's seed mean and counted in ``n_diverged``; it is
    NOT scored by its last finite value, which would flatter a method that blows up late.
    """
    if curve is None or len(curve) == 0:
        return np.nan
    try:
        v = float(curve[-1])
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def is_diverged(train_curve, val_curve, status=None):
    """The one definition of a diverged run: it is RECORDED as diverged
    (``status == 'diverged'``, C-R1), or its train or val curve ends non-finite, OR
    its validation loss ends more than ``DIVERGED_FACTOR`` times above the pre-training
    value at ``val[0]`` (a finite blow-up -- the paramfrac scans end some runs at
    1e7..1e15 without ever producing a NaN).  Only the val curve is used for the ratio:
    it starts with the untrained model, whereas ``train[0]`` is already the post-epoch-1
    loss and makes a poor reference.

    ``status`` is the schema-2 field (None / absent on every legacy record, where the
    curve rules alone decide, exactly as before).  A recorded ``diverged`` run stops
    early with a *finite* partial curve, so the curves alone would call it healthy."""
    if status is not None and str(status) == STATUS_DIVERGED:
        return True
    fv, ft = final_value(val_curve), final_value(train_curve)
    if not (np.isfinite(fv) and np.isfinite(ft)):
        return True
    try:
        v0 = float(val_curve[0])
    except (TypeError, ValueError, IndexError):
        return False
    return bool(np.isfinite(v0) and v0 > 0 and fv > DIVERGED_FACTOR * v0)


def config_eligible(n_ok, n_expected):
    """Whether a configuration may be picked as a method's best: more than half of
    the scan's seeds must have finished (not diverged, not missing)."""
    return np.asarray(n_ok) > np.asarray(n_expected) / 2


def clipped_band(mean, std, lo):
    """``(lower, upper)`` of the seed band drawn everywhere: mean +/- 1 std (ddof=1),
    with the lower edge clipped at the lowest seed ``lo`` so it never reaches <= 0
    on a log axis.  The line itself stays the arithmetic seed mean.

    The edges are then pinned to the mean's own side of it: ``lo <= mean`` holds in
    exact arithmetic, but not in floats when every seed recorded the SAME value --
    ``peak_gpu_mem_mb`` is bit-identical across the seeds of a config, so
    ``v.mean()`` lands ~4e-15 BELOW ``v.min()`` and the raw clip put the lower edge
    above the mean.  :func:`clipped_yerr` then handed matplotlib a yerr of -3.6e-15
    and ``ax.bar`` refused it ("'yerr' must not contain negative values") -- one
    round-off bit killing the whole memory panel.
    """
    mean, lo = np.asarray(mean, dtype=float), np.asarray(lo, dtype=float)
    std = np.nan_to_num(np.asarray(std, dtype=float))
    return np.minimum(np.maximum(mean - std, lo), mean), np.maximum(mean + std, mean)


def clipped_yerr(mean, std, lo):
    """The same band as an asymmetric ``yerr=[below, above]`` for ``bar`` / ``errorbar``."""
    lower, upper = clipped_band(mean, std, lo)
    mean = np.asarray(mean, dtype=float)
    return np.vstack([mean - lower, upper - mean])


def seed_spread_label(plain=False):
    """THE words for the band / error bar :func:`clipped_band` draws (C-A6).

    One spelling, from one constant: :data:`paired.SEED_SPREAD_LABEL`
    (``'$\\pm$ 1 std over seeds'``).  It is the spread of the SEEDS, not a confidence
    interval on their mean -- ``paired.interval_label`` is the label for that.
    ``plain=True`` gives the same words without mathtext, for a printed table header
    (``'± 1 std over seeds'``); a figure legend wants the default.

    Imported lazily: :mod:`paired` imports this module, so a top-level import here
    would be a cycle (``scan_analysis`` -> ``paired`` -> ``analysis_helpers`` ->
    ``scan_analysis``).  The constant lives in :mod:`paired` because that is where
    the seed-band-vs-interval distinction is documented.
    """
    from paired import SEED_SPREAD_LABEL
    return SEED_SPREAD_LABEL.replace(r'$\pm$', '±') if plain else SEED_SPREAD_LABEL


def band_legend(ax, color='0.45', alpha=0.25):
    """Give ``ax`` ONE legend entry naming the seed band (C-A6), and return it.

    Every helper that draws seed uncertainty calls this, so a figure legend says
    what its shaded band / error bar means instead of leaving the reader to guess.
    The entry is a neutral grey proxy patch with no data (it cannot move the axis
    limits) and appears only if something calls ``ax.legend()``; calling this twice
    on the same axes adds nothing the second time, so a loop over methods is safe.
    """
    label = seed_spread_label()
    if label in ax.get_legend_handles_labels()[1]:
        return None
    return ax.fill_between([], [], color=color, alpha=alpha, lw=0, label=label)


# ---------------------------------------------------------------------------
# Dataset / task names -- one spelling everywhere (EXPERIMENTS.md C22)
# ---------------------------------------------------------------------------
DATASET_TITLES = {
    'toy_1d':         'Toy 1D',
    'polynomial':     'Random Polynomial',
    'mnist_ce':       'MNIST (CE)',
    'mnist_labelreg': 'MNIST (label reg.)',
    'cifar_ce':       'CIFAR-10 (CE)',
    'cifar_labelreg': 'CIFAR-10 (label reg.)',
    'nanogpt':        'nanoGPT (tiny-shakespeare)',
}
# Axis labels for the final-loss metrics ("seed mean" is appended by the plot helpers).
METRIC_LABELS = {'final_val_loss': 'Final validation loss', 'final_train_loss': 'Final train loss',
                 'final_val_acc': 'Final validation accuracy',
                 # reported for the selected config only (C-E1); never a selector
                 'final_test_loss': 'Final test loss',
                 'final_test_acc': 'Final test accuracy'}

#: Relative size of the float32 round-off floor of a spectrum: singular values
#: below ~1e-7 * sigma_max are numerical noise, not structure (F19).  Drawn as a
#: horizontal line on every full-width spectrum plot so the tail is not read as
#: a measurement.
FLOAT32_NOISE_FLOOR = 1e-7


def metric_label(metric, seed_mean=True):
    base = METRIC_LABELS.get(metric, metric.replace('_', ' '))
    return base + (' (seed mean)' if seed_mean else '')

