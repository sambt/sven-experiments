import copy
import glob
import json
import os
import time
import uuid
from collections import Counter, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from hydra.utils import instantiate
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from . import claims, provenance
from .checkpointing import Checkpointer, save_init_state
from .experiment_utils import (
    DivergedError, train_loop_svd, train_loop_standard, train_loop_hig, train_loop_jd,
    set_seed, build_standard_optimizer, summarize_curves,
)
# the loops' own tail (totals, averages, the C-E5 summary). `execute` calls it on
# the failure path so an `oom` / `error` / non-DivergedError record carries the
# same curve keys as a finished one (see the `except` block).
from .experiment_utils import _finish_losses
from .grid import (
    LOSS_KEYS, SCHEMA_VERSION, SIGNED_RESIDUAL_LOSS_KEYS, SVD_INFO_MODES,
    build_id_string, expand_grid, inject_dataset_facts, listify, mode_flags,
    model_generation, resolve_scan_settings, resolve_svd_settings, shard,
)
from .grid import run_hash as compute_run_hash
from .optim_factory import get_muon_variant
from .sampler import EpochPermutationSampler, seed_for_run, set_loader_epoch
from experiments.datasets import subsample_indices
from experiments.nn.norm_utils import norm_stat_modules
from sven.opt import Sven, SvenGram
from sven.nn import SvenWrapper, GramSvenWrapper
from experiments.optimizers.hig import HIGWrapper, HIGOptimizer

try:  # Jacobian Descent baseline (optional dependency)
    from torchjd.aggregation import UPGrad, Mean, Sum
    _JD_AGGREGATORS = {"UPGrad": UPGrad, "Mean": Mean, "Sum": Sum}
    _HAS_TORCHJD = True
except ImportError:  # pragma: no cover
    _JD_AGGREGATORS = {}
    _HAS_TORCHJD = False


# ---------------------------------------------------------------------------
# Loss function registries
# ---------------------------------------------------------------------------

def _one_hot_like(pred, y):
    """One-hot targets (B, C) in ``pred``'s dtype/device; broadcasts against
    (B, C) or multi-model (M, B, C) predictions."""
    return F.one_hot(y.to(torch.long), num_classes=pred.shape[-1]).to(pred)


def _label_regression(pred, y):
    """Per-sample squared error between the RAW network outputs and the one-hot
    label, L_i = ||f(x_i) - y_i||^2. This is exactly the paper's Sec. 4
    definition (no softmax) -- the standard "square loss for classification"
    (Hui & Belkin, ICLR 2021) -- and the definition behind every
    label-regression result on disk. Accuracy is argmax over ``pred``, which
    is invariant under softmax, so no probabilities are needed anywhere."""
    return (pred - _one_hot_like(pred, y)).pow(2).sum(dim=-1)


def _brier(pred, y):
    """Per-sample squared error between softmax PROBABILITIES and the one-hot
    label, L_i = ||softmax(f(x_i)) - y_i||^2 -- the multiclass Brier score
    (a.k.a. label regression on softmax outputs).
    A different objective from ``label_regression``: bounded in [0, 2] and
    non-convex in the logits, with gradients that vanish once the softmax
    saturates -- for confidently-WRONG samples too (the classic softmax+MSE
    plateau), where a Gauss-Newton step is badly linearised and overshoots.
    Two Sven-specific consequences: (i) in float32 the loss hits exactly 0
    once the correct-class margin exceeds ~50, so ``kappa < 2`` (residual
    ``loss**(kappa/2)``, infinite slope at 0) NaNs out -- keep ``kappa = 2``;
    (ii) saturated samples contribute ~0 rows to the Gram matrix, so the
    ``rtol``/``lr`` that were best for ``label_regression`` do not transfer
    (expect to need a smaller ``lr``). Kept as a separate registry key, not a
    redefinition: results are not comparable across the two keys, the run_id
    carries a ``_loss{key}`` suffix for every non-legacy key, and every result
    row records its ``loss`` key."""
    return (F.softmax(pred, dim=-1) - _one_hot_like(pred, y)).pow(2).sum(dim=-1)


# Signed scalar residual r per sample for losses of the form loss = r**2 (scalar
# outputs only). When available and `signed_residual: true` (default), Sven's
# Jacobian rows are sign(r)|r|^kappa instead of loss^(kappa/2) = |r|^kappa for
# every kappa. The rows and their Jacobian both pick up the same per-row sign
# D = diag(sign r), and D cancels in the pseudo-inverse (M' = DM has the same
# singular values, M'^+ = M^+ D, so M'^+ R' = M^+ D D R = M^+ R): the SAME update
# up to rounding, but with a finite gradient at r = 0, which removes the
# kappa < 2 NaN of the fractional power. Multi-output losses (label_regression,
# brier, ce) have no scalar signed residual and stay on the loss path.
SVD_RESIDUAL_FNS = {
    "mse": lambda pred, y: pred - y,
}

# SVD loss must return per-sample losses (reduction='none')
SVD_LOSS_FNS = {
    "ce": lambda pred, y: F.cross_entropy(pred, y, reduction='none'),
    "mse": lambda pred, y: ((pred - y) ** 2).sum(dim=-1),
    "label_regression": _label_regression,
    "brier": _brier,
    # language modeling: logits (B, T, V), targets (B, T) -> per-sample mean CE (B,)
    "lm_ce": lambda pred, y: F.cross_entropy(
        pred.reshape(-1, pred.shape[-1]), y.reshape(-1), reduction='none'
    ).reshape(y.shape[0], -1).mean(dim=1),
}

# Standard loss returns a scalar
STANDARD_LOSS_FNS = {
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "label_regression": lambda pred, y: _label_regression(pred, y).mean(),
    "brier": lambda pred, y: _brier(pred, y).mean(),
    "lm_ce": lambda pred, y: F.cross_entropy(pred.reshape(-1, pred.shape[-1]), y.reshape(-1)),
}


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _to_json_serializable(obj):
    """Recursively convert numpy/torch types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    return obj


def _write_result(jsonl_path, result):
    """Write a single result dict as a JSON line to its own file."""
    serializable = _to_json_serializable(result)
    with open(jsonl_path, 'w') as f:
        f.write(json.dumps(serializable) + '\n')


# ---------------------------------------------------------------------------
# Light / heavy result split
# ---------------------------------------------------------------------------
# A run is written as TWO files:
#   {scan}/{run_id}.jsonl      "light": hparams, per-EPOCH curves, timings and a
#                              per-epoch SVD summary -- a few KB, all the loss-
#                              curve notebooks need. Written last of the run's
#                              files; the dedup marker itself is the zero-byte
#                              {scan}/done/{run_id}.{hash8}.{status} that
#                              `run_grid` writes after it (C-R3).
#   {scan}/diag/{run_id}.npz   "heavy": per-BATCH arrays (batch losses/times) and
#                              the Sven spectra, compressed float32. Loaded on
#                              demand by analysis.style.load_diagnostics /
#                              load_results(slim=False).
# Previously the spectra alone were ~85% of a 7-25 MB JSON file per Sven run.
# `val_batch` / `batch_times_val` are gone: one example-weighted `evaluate()` call
# replaced the timed per-batch validation loop (C-E1), so there is no per-batch
# validation series any more -- its per-epoch successor is `eval_times`, which is
# small and stays in the jsonl.
_DIAG_LOSS_KEYS = ('train_batch', 'batch_times_train', 'train_batch_per_model')

#: Sven diagnostics the optimizer only records on scheduled steps (C-L1/C-L2):
#: per-logged-step scalars, indexed by `svs_step`, never by position.
_SCHEDULED_SCALAR_KEYS = ('update_norm', 'resid_norm', 'sv_min_kept', 'sv_noise_floor')
# SVD_INFO_MODES now lives in grid.py (torch-free) and is imported above; grid.py
# also mirrors the loss-registry keys so it can validate `loss` without torch.
assert set(LOSS_KEYS) == set(SVD_LOSS_FNS) == set(STANDARD_LOSS_FNS)
assert set(SIGNED_RESIDUAL_LOSS_KEYS) == set(SVD_RESIDUAL_FNS)


def _pad_ragged(rows, dtype=np.float32):
    """Stack variable-length 1-D rows into a NaN-padded 2-D array."""
    rows = [np.asarray(r, dtype=dtype).reshape(-1) for r in rows]
    width = max((len(r) for r in rows), default=0)
    out = np.full((len(rows), width), np.nan, dtype=dtype)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def _epoch_mean(per_batch, n_epochs):
    """Per-epoch means of a per-batch series (tolerates a non-divisible tail)."""
    a = np.asarray(per_batch, dtype=np.float64)
    if n_epochs <= 0 or a.size == 0:
        return []
    return [float(c.mean()) if c.size else float('nan')
            for c in np.array_split(a, n_epochs)]


def _epoch_mean_at_steps(values, steps, n_epochs, steps_per_epoch):
    """Per-epoch means of a SPARSE per-step series, from its step indices (C-L2).

    The scheduled Sven diagnostics exist only on the steps the schedule logged,
    so they cannot be chopped into ``n_epochs`` equal pieces the way a dense
    per-batch series can (:func:`_epoch_mean`): the step index says which epoch
    a value belongs to. An epoch with no logged step gets ``nan`` rather than
    borrowing a neighbour's value.
    """
    a = np.asarray(values, dtype=np.float64)
    s = np.asarray(steps, dtype=np.int64)
    if n_epochs <= 0 or a.size == 0 or not steps_per_epoch:
        return []
    epoch_of = np.minimum(s // int(steps_per_epoch), n_epochs - 1)
    out = []
    for e in range(n_epochs):
        chunk = a[epoch_of == e]
        out.append(float(np.nanmean(chunk)) if chunk.size else float('nan'))
    return out


def _split_diagnostics(result, svd_info_mode="full", spectra_schedule=None):
    """Split a result dict into (light, diag).

    ``light`` is ``result`` minus the per-batch arrays and ``svd_info``, plus a
    small per-epoch ``svd_summary``. ``diag`` maps array names to numpy arrays:

    * the per-batch loss / time series and ``num_nonzero_svs``, which the
      optimizer records on **every** step (C-L1) and which therefore keep their
      dense per-step indexing;
    * the **scheduled** Sven diagnostics -- the spectra ``svs`` (NaN-padded, one
      row per logged step) and ``utr`` for ``full``, plus ``update_norm``,
      ``resid_norm``, ``sv_min_kept`` and ``sv_noise_floor`` -- all indexed by
      ``svs_step``, which is ``svd_info['step']`` verbatim: what the optimizer
      logged, at the steps it logged it, with no positional subsampling (C-L2).
    * ``sv_max`` and ``sv_min_all`` (= sigma_1 and sigma_M of the full logged
      spectrum) follow ``svs_step`` as well. The legacy ``sv_min`` column is
      deliberately NOT written: in old records it was the smallest *kept*
      singular value, which is now ``sv_min_kept`` (F20), and reusing the name
      would silently mix the two meanings.

    ``spectra_schedule`` is recorded in ``svd_summary`` and used for nothing
    else here -- the optimizer decides what to log, this function only stores
    what it finds.
    """
    assert svd_info_mode in SVD_INFO_MODES, svd_info_mode
    light = dict(result)
    losses = dict(light.get('losses') or {})
    diag = {}
    for key in _DIAG_LOSS_KEYS:
        if key in losses:
            v = losses.pop(key)
            diag[key] = (_pad_ragged(v) if key == 'train_batch_per_model'
                         else np.asarray(v, dtype=np.float32))
    light['losses'] = losses
    n_epochs = len(losses.get('train') or [])
    steps_per_epoch = light.get('steps_per_epoch')

    si = light.pop('svd_info', None)
    summary = None
    if isinstance(si, dict):
        nnz = np.asarray(si.get('num_nonzero_svs') or [], dtype=np.int32)
        svs = list(si.get('svs') or [])
        steps = np.asarray(si.get('step') or [], dtype=np.int32)
        summary = {
            'mode': svd_info_mode,
            'n_steps': int(len(nnz)),
            'num_nonzero_svs_epoch': _epoch_mean(nnz, n_epochs),
            'schedule': dict(spectra_schedule) if spectra_schedule else None,
            'spectra_saved': int(len(steps)),
        }
        if len(nnz):
            diag['num_nonzero_svs'] = nnz
        if len(steps) and svd_info_mode != 'none':
            diag['svs_step'] = steps
            for key in _SCHEDULED_SCALAR_KEYS:
                v = si.get(key)
                if v is not None and len(v):
                    diag[key] = np.asarray(v, dtype=np.float32)
            if len(svs):
                sv_max = np.array([np.max(x) if np.size(x) else np.nan for x in svs],
                                  np.float32)
                sv_min_all = np.array([np.min(x) if np.size(x) else np.nan for x in svs],
                                      np.float32)
                diag['sv_max'], diag['sv_min_all'] = sv_max, sv_min_all
                summary['sv_max_epoch'] = _epoch_mean_at_steps(
                    sv_max, steps, n_epochs, steps_per_epoch)
                summary['sv_min_all_epoch'] = _epoch_mean_at_steps(
                    sv_min_all, steps, n_epochs, steps_per_epoch)
                if svd_info_mode == 'full':
                    diag['svs'] = _pad_ragged(svs)
                    if si.get('utr') is not None and len(si['utr']):
                        diag['utr'] = _pad_ragged(si['utr'])
        if si.get('k_used'):
            diag['k_used'] = np.asarray(si['k_used'], dtype=np.int32)
        if si.get('variable_k_substep_losses'):
            diag['variable_k_substep_losses'] = _pad_ragged(
                [[float(t) for t in row] for row in si['variable_k_substep_losses']])
    light['svd_summary'] = summary
    return light, diag


def _write_run(scan_dir, run_id, result, svd_info_mode="full", spectra_schedule=None,
               common=None):
    """Write the heavy diagnostics (npz) first, then the light JSONL.

    ``common`` holds per-scan facts every record should carry (``n_params``,
    ``n_train``, ``n_val``, ``n_test``, ``split_seed``; see :func:`_scan_facts`) --
    the analysis needs them for P/N and steps-per-epoch and used to hard-code them.
    The run's own checkpoint file, if any, is written before this is called, and
    the done marker after it: ckpt -> npz -> jsonl -> ``done/`` (C-L3/C-R3), so a
    run counts as finished only once every one of its files exists.
    """
    if common:
        for k_, v in common.items():
            result.setdefault(k_, v)
    light, diag = _split_diagnostics(result, svd_info_mode, spectra_schedule)
    light['diag_file'] = None
    if diag:
        diag_dir = os.path.join(scan_dir, 'diag')
        os.makedirs(diag_dir, exist_ok=True)
        np.savez_compressed(os.path.join(diag_dir, run_id + '.npz'), **diag)
        light['diag_file'] = os.path.join('diag', run_id + '.npz')
    _write_result(os.path.join(scan_dir, run_id + '.jsonl'), light)


def _scan_facts(model, dataset):
    """Facts the analysis otherwise has to hard-code (C-R4).

    The parameter count P, the three split sizes N (C-E1) and the dataset's
    ``split_seed`` -- the value that decided which examples are held out, and
    which the record must carry because it is independent of the model and
    loader seeds (``None`` for the positionally-split text corpora).
    ``steps_per_epoch`` is per-run (it depends on the batch size) and is recorded
    by :func:`execute` from ``len(train_loader)``; with ``drop_last=True`` for
    every family (C-S3) it is ``n_train // batch_size``.
    """
    facts = {'n_params': int(sum(p.numel() for p in model.parameters()))}
    for key, attr in (('n_train', 'train_dataset'), ('n_val', 'val_dataset'),
                      ('n_test', 'test_dataset')):
        ds = getattr(dataset, attr, None)
        try:
            facts[key] = int(len(ds)) if ds is not None else None
        except TypeError:
            facts[key] = None
    facts['split_seed'] = getattr(dataset, 'split_seed', None)
    return facts


# ---------------------------------------------------------------------------
# Scan logic: expand_grid (grid.py) -> shard -> dedup -> execute
# ---------------------------------------------------------------------------
# `grid.expand_grid` enumerates the whole grid up front as RunSpecs (see grid.py:
# torch-free, byte-identical run_ids and order to the six inline product() loops
# this replaced), `grid.shard` takes this worker's slice and `execute` runs one
# spec. Per-family code therefore exists exactly once.

_FAMILY_BANNER = {
    "svd": "Running SVD optimizer scan",
    "standard": "Running standard optimizer scan",
    "lbfgs": "Running standard optimizer scan",
    "polyak": "Running standard optimizer scan",
    "jd": "Running Jacobian Descent scan",
    "hig": "Running Half-Inverse Gradients scan",
}

# The exception message each family printed before the refactor ("standard" is
# the optimizer's own name).
_FAMILY_ERROR = {
    "svd": "Training failed",
    "lbfgs": "LBFGS run failed",
    "polyak": "PolyakSGD run failed",
    "jd": "JD run failed",
    "hig": "HIG run failed",
}

# Families whose model is moved to the device by the runner; the svd and hig
# wrappers do it themselves.
_TO_DEVICE_FAMILIES = ("standard", "lbfgs", "polyak", "jd")

# Models above this parameter count rewrite their checkpoint file at every epoch
# end (C-L3), so a 12 h timeout loses one epoch instead of the whole trajectory.
# Below it a run is short enough that one write at the end is enough, and the
# whole-file rewrite is pure overhead. See `_rewrite_each_epoch`: the policy
# matters as much as the size.
_REWRITE_CKPT_ABOVE_PARAMS = 1e6


def _rewrite_each_epoch(policy, num_epochs, n_params):
    """Whether the checkpointer writes its file at every epoch end (C-L3).

    Two independent reasons, and the parameter count alone is not one of them.
    An **accumulating** policy (``epochs``, ``log``) holds every state it took in
    host memory until ``flush()``, so a multi-epoch run killed by the 12 h limit
    loses all of them -- nanoGPT is 826k parameters x 51 states (~170 MB per run,
    x NPROC) and sits far below any parameter threshold, which is exactly the run
    C-L3's "long runs rewrite at each epoch end" was written for. A **large**
    model is slow enough that one epoch is worth protecting even under ``final``,
    which keeps a single slot. A single-epoch run has nothing to protect: its
    ``flush()`` at the end is the same write.
    """
    if policy in (None, "none") or not num_epochs or int(num_epochs) <= 1:
        return False
    return policy in ("epochs", "log") or n_params > _REWRITE_CKPT_ABOVE_PARAMS

#: this repo and the nested, separately-versioned `sven` repo, for provenance
#: (C-R3). Resolved from this file, so a run out of a deploy snapshot reports the
#: snapshot's `DEPLOY_INFO.json` rather than whatever checkout the cwd sits in.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SVEN_ROOT = os.path.join(REPO_ROOT, "sven")

#: how often the walk refreshes its done index (one `os.listdir`), so a worker
#: sees the progress of the other workers on the same scan instead of attempting
#: a claim on every grid point they already finished. The *correctness* of dedup
#: never rests on the index: `claims.is_done_now` re-checks the disk under the
#: claim for every run that is about to execute.
_INDEX_REFRESH_RUNS = 200

#: longer messages are truncated in the record's `error` field (a torch OOM
#: message carries a whole allocator dump).
_ERROR_MESSAGE_CHARS = 2000

#: one zero-byte file per FAILED attempt: `{run_id}.{hash8}.{status}.{token}`.
#: `done/` cannot count attempts (one file per hash and status), and `oom` /
#: `error` are retried, so without this a deterministically failing grid point --
#: a bad `_target_`, `batch_size > n_train`, a missing token file, a Muon raise --
#: is re-attempted by every worker of every job for the whole campaign.
ATTEMPTS_DIRNAME = "attempts"
#: failed attempts at ONE hash after which the run is left alone (reported as
#: `poisoned`, and still visible as its `oom` / `error` record + markers). Three,
#: so a genuinely transient OOM (a co-tenant's peak) still gets retries.
_MAX_FAILED_ATTEMPTS = 3


def _failed_attempts(scan_dir, run_id, hash8):
    """How many times THIS generation of the run has already failed.

    One glob per run that is about to execute (never per grid point), like
    :func:`claims.is_done_now` and :func:`_stale_hashes_now`.
    """
    pattern = os.path.join(scan_dir, ATTEMPTS_DIRNAME,
                           glob.escape(f"{run_id}.{hash8}") + ".*")
    return len(glob.glob(pattern))


def _record_attempt(scan_dir, run_id, hash8, status):
    """Record one failed attempt (see :data:`ATTEMPTS_DIRNAME`).

    The token keeps concurrent workers from writing the same name, so the file
    count is the attempt count.
    """
    directory = os.path.join(scan_dir, ATTEMPTS_DIRNAME)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory,
                        f"{run_id}.{hash8}.{status}.{uuid.uuid4().hex[:8]}")
    with open(path, "wb"):
        pass
    return path


# ---------------------------------------------------------------------------
# Failure classification (C-R1): every failure is a record with a status
# ---------------------------------------------------------------------------
# `torch.cuda.OutOfMemoryError` and `torch.OutOfMemoryError` are the same class on
# current torch and both subclass RuntimeError; collected defensively so the
# classifier cannot be broken by a rename.
_OOM_ERRORS = tuple({e for e in (getattr(torch, "OutOfMemoryError", None),
                                 getattr(torch.cuda, "OutOfMemoryError", None))
                     if isinstance(e, type)})
_LINALG_ERRORS = tuple({e for e in (getattr(torch.linalg, "LinAlgError", None),)
                        if isinstance(e, type)})
#: an allocator failure that surfaces as a plain RuntimeError (older torch, and
#: some cuDNN / cuBLAS paths) rather than as an OutOfMemoryError.
_OOM_MESSAGE_TOKENS = ("out of memory", "cuda oom")
#: messages that mean "the numbers blew up", not "the code is wrong": Sven's
#: empty-spectrum guard (`sven/sven/opt/sven.py`, "the Gram matrix is zero or
#: non-finite -- run diverged") and the LAPACK failures a non-finite matrix
#: produces inside an eigendecomposition / SVD.
_DIVERGED_MESSAGE_TOKENS = ("no singular value above rtol", "failed to converge",
                            "ill-conditioned")


def _classify_failure(exc):
    """``(status, diverged_at_step)`` for an exception out of a training loop.

    The order is the contract (C-R1) and matters: :class:`DivergedError`
    subclasses ``RuntimeError`` and both OOM and the linalg errors carry
    messages, so a message-based test placed first would mislabel them.

    ``oom`` and ``error`` are retried by the dedup rules while ``ok`` and
    ``diverged`` are final, so the distinction is not cosmetic: calling a
    genuine divergence an ``error`` would make every worker re-run it forever,
    and calling an OOM a divergence would silently accept a missing result as a
    scientific finding.
    """
    if isinstance(exc, DivergedError):          # ALWAYS first (a RuntimeError)
        return "diverged", exc.step
    if _OOM_ERRORS and isinstance(exc, _OOM_ERRORS):
        return "oom", None
    message = str(exc).lower()
    if any(token in message for token in _OOM_MESSAGE_TOKENS):
        return "oom", None
    if _LINALG_ERRORS and isinstance(exc, _LINALG_ERRORS):
        return "diverged", None
    if any(token in message for token in _DIVERGED_MESSAGE_TOKENS):
        return "diverged", None
    return "error", None


class _EpochLoader:
    """A training DataLoader whose sampler is advanced one epoch per iteration.

    :class:`~experiments.experiment_code.sampler.EpochPermutationSampler` makes
    the data order a pure function of ``(loader_seed, epoch)`` (C-S2), which
    requires ``set_epoch(e)`` once per epoch; the four training loops iterate
    ``for xb, yb in train_loader`` and do not call it. One ``__iter__`` call is
    exactly one epoch in every loop, so advancing here needs no loop change and
    cannot drift out of step with the epoch counter. ``len()`` and attribute
    access pass through, so the loops' ``len(train_loader)`` (steps per epoch)
    and everything else still see the DataLoader.
    """

    def __init__(self, loader, first_epoch=0):
        self.loader = loader
        self.epoch = int(first_epoch) - 1

    def __iter__(self):
        self.epoch += 1
        set_loader_epoch(self.loader, self.epoch)
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):          # `loader` / `epoch` are instance attrs
        return getattr(self.loader, name)


#: what :meth:`_ScanContext.loaders` hands to :func:`execute`.
_RunLoaders = namedtuple("_RunLoaders", "train val test train_eval sampler")


class _ScanContext:
    """Everything :func:`execute` needs that does not vary across grid points.

    Also owns the per-seed initial state: one base model per model seed, whose
    ``state_dict`` every run of that seed starts from (the legacy per-seed
    preamble). Only the current seed is kept, which is what the legacy loop held
    too -- ``expand_grid`` is seed-major, so nothing older is ever needed.
    """

    def __init__(self, cfg, rcfg, dataset, scan_dir, settings, svd_settings):
        self.cfg = cfg
        self.rcfg = rcfg
        self.device = rcfg["device"]
        self.dataset = dataset
        self.scan_dir = scan_dir
        self.svd = svd_settings
        self.settings = settings
        self.num_epochs = rcfg["num_epochs"]
        self.loss_key = settings["loss_key"]
        self.track_acc = settings["track_acc"]
        self.is_lm = settings["is_lm"]
        self.track_param_norm = settings["track_param_norm"]
        self.svd_info_mode = settings["svd_info_mode"]
        self.spectra_schedule = settings["svd_spectra_schedule"]
        self.eval_batch_size = settings["eval_batch_size"]
        self.train_eval_size = settings["train_eval_size"]
        self.eval_every_steps = settings["eval_every_steps"]
        self.empty_cache = settings["empty_cache"]
        self.stop_on_nonfinite = settings["stop_on_nonfinite"]
        # C-E1: ALL FOUR loops now get the per-sample loss. `evaluate` accepts
        # either form, but only per-sample losses make the online train curve
        # exactly example-weighted (F10) and give every family the same definition
        # of "the loss" -- STANDARD_LOSS_FNS["mse"] is nn.MSELoss, a mean over
        # output dimensions too, so a multi-output regression run would report a
        # different quantity from the Sven run it is compared against. The scalar
        # registry stays reachable (optimizer_profile.py imports it directly) and
        # the loops reduce with .mean(), which is the identical gradient.
        self.loss_fn_svd = SVD_LOSS_FNS[self.loss_key]            # per-sample (all loops)
        self.loss_fn_standard = STANDARD_LOSS_FNS[self.loss_key]  # scalar (unused here)
        self.residual_fn_svd = (SVD_RESIDUAL_FNS[self.loss_key]
                                if settings["signed_residual"] else None)
        self._seed = None
        self._init_state = None
        self._common = None
        self._eval_loaders = None         # val / test / train_eval: seed-independent

    # -- per-family settings ------------------------------------------------

    def bn_mode(self, family):
        """The norm-statistics policy of one family (C-E2).

        The svd family's is resolved together with the Gram settings (its legacy
        default is ``frozen``); everything else follows the config, defaulting to
        batch statistics. ``grid.bn_mode_suffix`` puts the same value in the
        run_id, and ``spec.record_extra['bn_mode']`` in the record.
        """
        if family == "svd":
            return self.svd["bn_mode"]
        return self.settings["bn_mode"] or "batch"

    def checkpoint_policy(self, family):
        """``checkpoints``, or ``checkpoints_svd`` for the svd family (C-L3)."""
        if family == "svd" and self.settings["checkpoints_svd"] is not None:
            return self.settings["checkpoints_svd"]
        return self.settings["checkpoints"]

    def log_schedule(self, family):
        """``step -> bool``: whether Sven logs its full spectrum on this step.

        ``svd_spectra_schedule: {dense_first: D, every: E}`` (C-L1/C-L2) keeps
        every one of the first ``D`` steps -- where the spectrum moves fastest --
        and then every ``E``-th step. Only the svd family has the flag; handing a
        schedule to a torch optimizer would just grow an unused attribute.
        """
        if family != "svd" or self.svd_info_mode == "none":
            return None
        dense_first = self.spectra_schedule["dense_first"]
        every = self.spectra_schedule["every"]
        return lambda step: step < dense_first or step % every == 0

    # -- per-seed state -----------------------------------------------------

    def init_state_name(self, model_seed):
        """File name of the shared initial state: seed AND model generation (C-L3).

        ``ckpt/init_mseed{seed}.{model_digest}.pt``. The digest is what makes the
        name unique per *model*: one scan directory holds several models (an
        ``result_id_fields: [mlp_width, n_data]`` scan, or any generation change),
        ``save_init_state`` skips a file that exists, and nothing retires this
        one -- so a seed-only name would let the first job's initialisation stand
        in for every other model's, with no record saying so.
        """
        return f"init_mseed{model_seed}.{model_generation(self.rcfg)}.pt"

    def seed_state(self, model_seed):
        """``(init_state, common)`` for a model seed, built once per seed."""
        if self._seed != model_seed:
            set_seed(model_seed)
            base_model = instantiate(self.cfg.model)
            self._seed = model_seed
            self._init_state = copy.deepcopy(base_model.state_dict())
            # n_params / n_train / n_val / n_test / split_seed, on every record
            self._common = _scan_facts(base_model, self.dataset)
            del base_model
        return self._init_state, self._common

    # -- loaders ------------------------------------------------------------

    def _train_eval_dataset(self):
        """The fixed ``min(n_train, train_eval_size)`` training subset (C-E3).

        Drawn with the dataset's ``split_seed``, so it depends on neither the
        model seed, the loader seed nor the optimizer: every run of the scan
        measures its end-of-epoch training loss on exactly the same examples, at
        a fixed parameter vector, which is the quantity the convergence claim is
        about. ``subsample_indices`` returns a prefix of one permutation, so the
        subset is also nested in ``n_train``.
        """
        train_dataset = self.dataset.train_dataset
        n_train = len(train_dataset)
        n_eval = min(int(n_train), int(self.train_eval_size))
        if n_eval <= 0:
            return None
        if n_eval >= n_train:
            return train_dataset
        split_seed = getattr(self.dataset, 'split_seed', None)
        idx = subsample_indices(n_train, n_eval, 0 if split_seed is None else int(split_seed))
        return Subset(train_dataset, idx.tolist())

    def _eval_loader(self, dataset):
        """A sequential evaluation loader at ``eval_batch_size`` (never the train one)."""
        if dataset is None:
            return None
        return DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False)

    def loaders(self, spec):
        """The four loaders of one run plus the training sampler (C-E1/C-S2/C-S3).

        The training loader gets the deterministic
        :class:`EpochPermutationSampler` as its ``batch_sampler`` -- seeded with
        ``derive_loader_seed(loader_seed, model_seed)``, identical across
        optimizers at one model seed, ``drop_last=True`` for every family -- and
        is wrapped so its epoch is advanced once per pass (:class:`_EpochLoader`).
        Validation, test and ``train_eval`` are sequential, share the three
        datasets across every run of the scan and are built once.
        """
        sampler = EpochPermutationSampler.for_run(
            len(self.dataset.train_dataset), spec.loader_seed, spec.model_seed,
            spec.batch_size, drop_last=True,
        )
        train_loader = _EpochLoader(
            DataLoader(self.dataset.train_dataset, batch_sampler=sampler))
        if self._eval_loaders is None:
            self._eval_loaders = (
                self._eval_loader(getattr(self.dataset, 'val_dataset', None)),
                self._eval_loader(getattr(self.dataset, 'test_dataset', None)),
                self._eval_loader(self._train_eval_dataset()),
            )
        val_loader, test_loader, train_eval_loader = self._eval_loaders
        return _RunLoaders(train_loader, val_loader, test_loader, train_eval_loader,
                           sampler)


def _describe(spec, ctx):
    """The one-line "what is running now" message of the legacy blocks."""
    hp, bs = spec.hparams, spec.batch_size
    if spec.family == "svd":
        msg = (f"SVD: bs={bs}, k={hp['k']}, lr={hp['lr']}, rtol={hp['rtol']}, "
               f"svd_mode={hp['svd_mode']}")
        if hp["microbatch_size"] is not None:
            msg += f", mb={hp['microbatch_size']}"
        if hp["param_fraction"] is not None:
            msg += f", pf={hp['param_fraction']}"
        if hp["kappa"] != 2.0:
            msg += f", kappa={hp['kappa']}"
        if ctx.svd["variable_k"]:
            msg += ", variable_k=True"
        return msg
    if spec.family == "standard":
        wd_str = f", wd={hp['weight_decay']}" if hp["weight_decay"] != 0.0 else ""
        return f"Standard: bs={bs}, lr={hp['lr']}, optim={hp['optim_name']}{wd_str}"
    if spec.family == "lbfgs":
        return (f"LBFGS: bs={bs}, lr={hp['lr']}, max_iter={hp['max_iter']}, "
                f"history_size={hp['history_size']}, line_search={hp['line_search_fn']}")
    if spec.family == "polyak":
        return (f"PolyakSGD: bs={bs}, f_star={hp['f_star']}, "
                f"max_lr={hp['max_lr']}, eps={hp['eps']}")
    if spec.family == "jd":
        return (f"JD: bs={bs}, lr={hp['lr']}, aggregator={hp['aggregator']}, "
                f"inner={hp['inner_optimizer']}")
    return f"HIG: bs={bs}, lr={hp['lr']}, tau={hp['tau']}"


def _final(losses, key):
    """The last point of a curve, or ``None`` when the curve is absent/empty."""
    curve = losses.get(key)
    if not curve:
        return None
    value = curve[-1]
    return None if value is None else float(value)


def _run_facts(spec, ctx, run_loaders, checkpoint_policy):
    """Per-run facts every schema-2 record carries.

    The scan-level ones (``n_params``, ``n_train``, ``n_val``, ``n_test``,
    ``split_seed``) come from :func:`_scan_facts` via :func:`_write_run`; these
    are the ones that vary per grid point: how many steps an epoch had (C-R4),
    what evaluation the numbers were measured with (C-E1/C-E3), which data order
    the run actually saw (C-S2: the run_id keeps the scan's *base* loader seed,
    so the effective one has to be recorded), and which logging policies were in
    force (C-L2/C-L3).

    ``run_loaders`` is ``None`` when a run failed before its loaders existed
    (C-R1 still writes that record); the two facts that come from them are then
    ``None`` rather than a guess.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "num_epochs": ctx.num_epochs,
        "steps_per_epoch": (None if run_loaders is None
                            else int(len(run_loaders.train))),
        "eval_batch_size": ctx.eval_batch_size,
        "train_eval_size": ctx.train_eval_size,
        "eval_every_steps": ctx.eval_every_steps,
        "effective_loader_seed": (None if run_loaders is None
                                  else int(run_loaders.sampler.loader_seed)),
        "checkpoint_policy": checkpoint_policy,
        "svd_spectra_schedule": (dict(ctx.spectra_schedule) if spec.family == "svd"
                                 else None),
    }


def execute(spec, ctx, run_hash=None, prov=None):
    """Run one grid point and write its record -- whatever happens (C-R1).

    The single copy of what used to be six near-identical blocks: build the model
    from the seed's initial state, re-seed for this run (C-S1), build the
    optimizer (and wrapper) for the family, build the four loaders (C-E1), call
    the family's training loop, assemble the record and write it.

    **Every failure is a result** (C-R1): an exception is classified by
    :func:`_classify_failure` into ``diverged`` / ``oom`` / ``error``, and the
    record is written from *outside* the ``try`` with the status, the exception's
    type and message, the partial curves (``losses`` is owned here and handed to
    the loop, so they survive; a loop that raised before touching it leaves them
    on ``err.losses``), whatever ``svd_info`` the optimizer managed to collect,
    and the run's flushed checkpoints. A grid point that produced no record at
    all therefore means "timed out or was killed", which is the one failure mode
    that cannot write its own record -- and which the ``started`` marker covers.

    ``run_hash``/``prov`` are the run's identity and the job's provenance, passed
    in by :func:`run_grid` because they are computed once per grid point / per
    job; both are recomputed here when absent, so a direct call still produces a
    complete schema-2 record.

    Returns the status string, which the caller writes as the done marker.
    """
    rcfg, cfg, device = ctx.rcfg, ctx.cfg, ctx.device
    hp = spec.hparams
    init_state, common = ctx.seed_state(spec.model_seed)
    bn_mode = ctx.bn_mode(spec.family)
    checkpoint_policy = ctx.checkpoint_policy(spec.family)

    print(f"\n{_describe(spec, ctx)}")
    # Pre-bound before the `try` so the failure path can read what exists: the
    # optimizer holds `svd_info`, the wrapper `mean_actual_param_fraction`, the
    # loaders `steps_per_epoch`, and the checkpointer has to be flushed.
    optimizer = None
    train_model = None
    checkpointer = None
    run_loaders = None
    ckpt_init_file = None
    losses = {}                # C-R1: caller-owned, so partial curves survive
    status, error, diverged_at_step = "ok", None, None
    timing = provenance.start_stamp()
    t0 = time.perf_counter()   # the loops' clock, for the failure path's totals
    try:
        model = instantiate(cfg.model)
        model.load_state_dict(init_state)
        if spec.family in _TO_DEVICE_FAMILIES:
            model = model.to(device)

        # C-E2 guards, before anything trains. A model with running statistics
        # must not be trained under a policy that is implicit or that its family
        # does not actually enforce: that is F2/F3, the defect C-E2 exists to
        # remove, and it produces plausible-looking numbers measured under a
        # different normalisation from the methods they are compared against.
        # Both raise, i.e. become a loud `error` record.
        if norm_stat_modules(model):
            if ctx.settings["bn_mode"] is None:
                raise ValueError(
                    f"{spec.run_id}: this model has norm layers with running "
                    "statistics, but the config names neither `bn_mode` nor the "
                    "deprecated `gram_freeze_norm_stats`, so every family would "
                    "fall back to its own legacy default (frozen for svd+gram, "
                    "batch for everything else) and Sven would be compared "
                    "against baselines under a different normalisation; set "
                    "`bn_mode: batch|frozen` explicitly")
            if spec.family == "hig" and bn_mode == "batch":
                raise ValueError(
                    f"{spec.run_id}: HIG cannot run with `bn_mode: batch` on a "
                    "model with running statistics. HIGWrapper takes no norm "
                    "policy and runs TWO train-mode forwards per step "
                    "(experiments/optimizers/hig.py:121-141), so its running "
                    "means advance at twice every other optimizer's rate while "
                    "`num_batches_tracked` stays 0 -- an inconsistent state_dict "
                    "and an eval-mode comparison against a different "
                    "normalisation. Use `bn_mode: frozen`, or give HIGWrapper the "
                    "SvenWrapper treatment (both forwards under "
                    "`no_norm_stat_updates`, one explicit `torch.no_grad()` "
                    "train-mode forward per step, `bn_mode` accepted and passed "
                    "in here)")

        # C-S1: every run gets its own RNG stream, seeded here -- after the
        # initial state is loaded (the model is identical for every run of the
        # seed) and BEFORE the wrapper's parameter mask and the optimizer's state
        # are drawn, so a masked or randomized-SVD run no longer depends on its
        # position in the process.
        set_seed(seed_for_run(spec.model_seed, spec.run_id))

        run_loaders = ctx.loaders(spec)
        checkpointer = Checkpointer(
            os.path.join(ctx.scan_dir, "ckpt", spec.run_id + ".pt"),
            checkpoint_policy, len(run_loaders.train), ctx.num_epochs,
            rewrite_each_epoch=_rewrite_each_epoch(checkpoint_policy, ctx.num_epochs,
                                                   common['n_params']),
        )
        if checkpoint_policy == "final":
            # One slot per run, so the shared initialisation is stored once per
            # model seed AND model generation instead of once per run (C-L3);
            # skipped if it exists. The path goes on the record, so an offline
            # tool never has to guess which init belongs to which run.
            ckpt_init_file = os.path.join("ckpt", ctx.init_state_name(spec.model_seed))
            save_init_state(os.path.join(ctx.scan_dir, ckpt_init_file), init_state)
        # The loop interface every family shares (C-E1/C-E4/C-L3/C-R1/C-E2).
        loop_kwargs = dict(
            losses=losses, test_loader=run_loaders.test,
            train_eval_loader=run_loaders.train_eval,
            eval_every_steps=ctx.eval_every_steps, checkpointer=checkpointer,
            log_schedule=ctx.log_schedule(spec.family),
            stop_on_nonfinite=ctx.stop_on_nonfinite, bn_mode=bn_mode,
        )

        if spec.family == "svd":
            sv = ctx.svd
            mb = hp["microbatch_size"] if hp["microbatch_size"] is not None else 1
            pf = hp["param_fraction"] if hp["param_fraction"] is not None else 1.0
            # signed residual rows: scalar regression, microbatch 1 only (any kappa)
            use_residual = spec.record_extra["signed_residual"]
            if sv["use_gram"]:
                # Gram trick: exact same update via B x B G = J J^T (no B x P Jacobian).
                # svd_mode is irrelevant (eigendecomposition of G replaces the SVD of J).
                train_model = GramSvenWrapper(
                    model, ctx.loss_fn_svd, device,
                    kappa=hp["kappa"],
                    microbatch_size=mb, param_fraction=pf,
                    mask_mode=(sv["mask_mode"] if pf < 1.0 else None),
                    capture=sv["gram_capture"],
                    bn_mode=bn_mode,
                    chunk_numel=sv["gram_chunk_numel"],
                    residual_fn=(ctx.residual_fn_svd if use_residual else None),
                )
                optimizer = SvenGram(train_model, lr=hp["lr"], k=hp["k"], rtol=hp["rtol"],
                                     track_svd_info=(ctx.svd_info_mode != "none"),
                                     empty_cache=ctx.empty_cache)
            else:
                train_model = SvenWrapper(
                    model, ctx.loss_fn_svd, device, kappa=hp["kappa"],
                    microbatch_size=mb, param_fraction=pf,
                    mask_mode=(sv["mask_mode"] if pf < 1.0 else None),
                    bn_mode=bn_mode,
                    residual_fn=(ctx.residual_fn_svd if use_residual else None),
                )
                optimizer = Sven(
                    train_model, lr=hp["lr"], k=hp["k"], rtol=hp["rtol"],
                    track_svd_info=(ctx.svd_info_mode != "none"), svd_mode=hp["svd_mode"],
                    variable_k=sv["variable_k"], empty_cache=ctx.empty_cache,
                )
            train_model, losses, optimizer = train_loop_svd(
                train_model, optimizer, ctx.loss_fn_svd,
                run_loaders.train, run_loaders.val,
                ctx.num_epochs, device, track_acc=ctx.track_acc,
                track_param_norm=ctx.track_param_norm, is_lm=ctx.is_lm, **loop_kwargs,
            )

        elif spec.family == "standard":
            optimizer = build_standard_optimizer(model, hp["optim_name"], hp["lr"],
                                                 weight_decay=hp["weight_decay"])
            model, losses = train_loop_standard(
                model, optimizer, ctx.loss_fn_svd,
                run_loaders.train, run_loaders.val,
                ctx.num_epochs, device, track_acc=ctx.track_acc,
                track_param_norm=ctx.track_param_norm, is_lm=ctx.is_lm, **loop_kwargs,
            )

        elif spec.family == "lbfgs":
            lbfgs_kwargs = {
                "max_iter": hp["max_iter"],
                "history_size": hp["history_size"],
                "line_search_fn": (hp["line_search_fn"]
                                   if hp["line_search_fn"] != "none" else None),
            }
            optimizer = build_standard_optimizer(model, "LBFGS", hp["lr"], **lbfgs_kwargs)
            model, losses = train_loop_standard(
                model, optimizer, ctx.loss_fn_svd,
                run_loaders.train, run_loaders.val,
                ctx.num_epochs, device, track_acc=ctx.track_acc,
                is_lm=ctx.is_lm, **loop_kwargs,
            )

        elif spec.family == "polyak":
            polyak_kwargs = {"f_star": hp["f_star"], "max_lr": hp["max_lr"],
                             "eps": hp["eps"]}
            optimizer = build_standard_optimizer(model, "PolyakSGD", lr=None,
                                                 **polyak_kwargs)
            model, losses = train_loop_standard(
                model, optimizer, ctx.loss_fn_svd,
                run_loaders.train, run_loaders.val,
                ctx.num_epochs, device, track_acc=ctx.track_acc,
                is_lm=ctx.is_lm, **loop_kwargs,
            )

        elif spec.family == "jd":
            aggregator = _JD_AGGREGATORS[hp["aggregator"]]()
            optimizer = build_standard_optimizer(model, hp["inner_optimizer"], hp["lr"])
            model, losses = train_loop_jd(
                model, optimizer, aggregator, ctx.loss_fn_svd,
                run_loaders.train, run_loaders.val, ctx.num_epochs, device,
                track_acc=ctx.track_acc, track_param_norm=ctx.track_param_norm,
                is_lm=ctx.is_lm, **loop_kwargs,
            )

        elif spec.family == "hig":
            train_model = HIGWrapper(model, ctx.loss_fn_svd, device)
            optimizer = HIGOptimizer(train_model, lr=hp["lr"], tau=hp["tau"])
            train_model, losses = train_loop_hig(
                train_model, optimizer, ctx.loss_fn_svd,
                run_loaders.train, run_loaders.val, ctx.num_epochs, device,
                track_acc=ctx.track_acc, track_param_norm=ctx.track_param_norm,
                is_lm=ctx.is_lm, **loop_kwargs,
            )

        else:
            # A family added to grid.FAMILIES must get a branch here; without this
            # it would silently run under the wrong wrapper and write a record.
            raise AssertionError(f"execute: unhandled family {spec.family!r}")

    except Exception as e:
        # "standard" is the only family whose message uses the optimizer's own name
        # (the .get keeps the unhandled-family raise above from turning into a KeyError).
        label = _FAMILY_ERROR.get(spec.family) or f"{hp.get('optim_name', spec.family)} run failed"
        status, diverged_at_step = _classify_failure(e)
        error = {"type": type(e).__name__,
                 "message": str(e)[:_ERROR_MESSAGE_CHARS]}
        if not losses:
            # A loop that raised before it filled the caller's dict attaches the
            # curves it did finish to the exception (`DivergedError.losses`).
            losses = dict(getattr(e, "losses", None) or {})
        if losses.get('total_time') is None:
            # ONE curve schema for all four statuses. Only `DivergedError` runs
            # the loop's own tail (`_partial_record_on_divergence`), so any other
            # mid-training failure -- a LinAlgError, Sven's empty-spectrum guard,
            # a HIG/KFAC solve, an injected error -- would otherwise leave a
            # record without `total_time` / `avg_train_time` / `avg_eval_time` /
            # the C-E5 summary, and `status` does not distinguish the two cases.
            _finish_losses(losses, t0)
        print(f"  [{status}] {label}: {e}")
        if torch.cuda.is_available():
            # Every family, svd included: Sven is the OOM candidate (CIFAR at
            # `gram_capture: full`, NPROC=1, a B x P Jacobian), so keeping its
            # allocator cache after a failure is what makes the rest of this
            # worker's walk fail too. Independent of the `compiler.reset()`
            # below, which drops compiled code and not allocator blocks.
            torch.cuda.empty_cache()
    finally:
        if checkpointer is not None:
            # A failed run keeps whatever it collected: the last state before a
            # blow-up is the diagnostic. No-op after a successful flush.
            checkpointer.flush()
        if spec.family == "svd":
            torch.compiler.reset()

    # The record is assembled and written OUTSIDE the try: success and failure
    # produce the same shape, differing only in `status` / `error` / how much of
    # the curves is there (C-R1). Write order (C-L3): checkpoint (flushed above)
    # -> npz -> jsonl, and the done marker after all three (written by the caller).
    result = {"run_id": spec.run_id, **spec.record_extra, "losses": losses}
    if spec.family == "svd":
        result["svd_info"] = getattr(optimizer, "svd_info", {})
        # 1.0 for an unmasked run; the mean over steps of the fraction of
        # parameters the mask actually admitted (C-R4).
        result["actual_param_fraction"] = getattr(
            train_model, "mean_actual_param_fraction", None)
    if spec.family == "standard":
        # Which weights Muon got and how its lr was adjusted; None otherwise.
        result["muon_variant"] = get_muon_variant(optimizer)
    result.update(_run_facts(spec, ctx, run_loaders, checkpoint_policy))
    # C-R1 / C-R3: the outcome of the run and the identity of the code that made it.
    result["status"] = status
    result["error"] = error
    result["diverged_at_step"] = diverged_at_step
    result["run_hash"] = run_hash or compute_run_hash(spec, rcfg)
    result.update(prov or {})
    result.update(provenance.end_stamp(timing))
    # Outcome columns (C-E5 / C-E1): the selection metric is still the last
    # epoch's val, these sit beside it. No selection function may use `test`.
    result.update(summarize_curves(losses))
    result["test"] = _final(losses, 'test')
    result["test_acc"] = _final(losses, 'test_acc')
    result["train_eval_final"] = _final(losses, 'train_eval')
    result["ckpt_error"] = None if checkpointer is None else checkpointer.last_error
    # The run's own checkpoint, as a path relative to the scan dir (like
    # `diag_file`), or None when the policy wrote nothing / the write failed.
    result["ckpt_file"] = (
        os.path.join("ckpt", spec.run_id + ".pt")
        if (checkpointer is not None
            and os.path.exists(checkpointer.path)) else None)
    # The shared per-seed/per-model initial state (`final` policy only), so C-L4
    # can find step 0 of this run without reconstructing the name.
    result["ckpt_init_file"] = ckpt_init_file
    for f in rcfg.get("result_id_fields", []):
        result[f] = rcfg[f]

    _write_run(ctx.scan_dir, spec.run_id, result, ctx.svd_info_mode,
               ctx.spectra_schedule, common)
    return status


def results_root():
    """The results root: ``$SV3_RESULTS_ROOT``, else ``experiment_results``.

    EXPERIMENTS.md section 12 "Results root". Reading it here rather than from the config is
    what lets a deploy snapshot, a worker or a test write somewhere else without
    editing 22 config files -- and what keeps every test out of the real
    ``experiment_results/`` symlink.
    """
    return os.environ.get("SV3_RESULTS_ROOT") or "experiment_results"


def _manifest_job_name(rcfg):
    """A name for this process's slice of the intended grid (C-R2).

    ``claims.write_manifest`` overwrites one file per name, so the name must
    identify the **work item** -- the mode / optimizer / seed subset this process
    enumerates -- and not the job: naming it after ``$SLURM_JOB_ID`` would
    accumulate one manifest per resubmission and the union (= "what the scan is
    supposed to contain") would keep run_ids from grids nobody intends any more.
    With the work item as the name, re-running it replaces its own manifest, and
    a grid that shrank is reported by ``write_manifest``'s dropped-run_ids warning.

    The name carries everything an override can change *in the run_ids
    themselves* -- the mode, the optimizer subset, ``result_id_fields`` (``n_data``
    is an item axis in ``campaign/plan_campaign.yaml``: six jobs of one scan, six
    disjoint sets of run_ids), the loss key and the seeds. Two work items that
    differ only in a hyperparameter *list* (e.g. one job per half of the lr grid)
    would share a name and the second would replace the first's manifest -- give
    such a split its own ``result_id_fields`` value, or read the warning.
    """
    parts = [str(rcfg.get("mode", "both"))]
    id_str = build_id_string(rcfg).lstrip("_")          # e.g. "mlp_width16_n_data600"
    if id_str:
        parts.append(id_str)
    optimizers = rcfg.get("optimizers_standard")
    if optimizers is not None:
        parts.append("-".join(str(o) for o in listify(optimizers)))
    if rcfg.get("loss") is not None:
        parts.append(str(rcfg["loss"]))
    seeds = rcfg.get("model_seeds")
    if seeds is not None:
        parts.append("mseed" + "-".join(str(s) for s in listify(seeds)))
    return ".".join(parts)


def _save_resolved_config(scan_dir, cfg, job_name):
    """The resolved config of this job, once, in ``{scan}/configs/`` (C-R3).

    Written after the ``cfg.model`` mutation, so the file shows the model the runs
    actually instantiate -- which is also the config a torch-free consumer must
    read to reproduce this scan's ``hash8`` (see
    :func:`grid.inject_dataset_facts`). Via a temp file + ``os.replace`` because
    every worker of every job serving this work item writes the same name and a
    reader must never see half a file. The temp name carries a random token, not
    just the pid: identically configured SLURM nodes hand out the same pids, and
    two writers sharing a temp name interleave their bytes in it and then promote
    the result (the same hazard ``checkpointing._atomic_save`` documents).
    """
    directory = os.path.join(scan_dir, "configs")
    os.makedirs(directory, exist_ok=True)
    # the manifest's own file name, so `configs/X.yaml` is the config that produced
    # `manifest/X[.shard*].json` (and the job name is sanitised exactly once).
    stem = os.path.basename(claims.manifest_path(scan_dir, job_name))[:-len(".json")]
    path = os.path.join(directory, stem + ".yaml")
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    try:
        with open(tmp, "w") as fh:
            fh.write(OmegaConf.to_yaml(cfg))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _stale_hashes_now(scan_dir, run_id, hash8):
    """The run's done markers on disk that belong to ANOTHER generation (C-R3).

    ``claims.stale_hashes`` answers the same question from the process-start
    index, which is a snapshot: a generation another worker finished (or retired)
    after that listdir is invisible to it, and missing one is not harmless --
    the jsonl / npz / ckpt paths carry no hash, so the old generation's files
    would be silently **overwritten** instead of moved to ``_stale/``. One glob
    per run that is about to execute (never per grid point) closes that window,
    the same trade ``claims.is_done_now`` makes.
    """
    pattern = os.path.join(scan_dir, claims.DONE_DIRNAME, glob.escape(run_id) + ".*")
    found = set()
    for path in glob.glob(pattern):
        parts = os.path.basename(path).rsplit(".", 2)   # run_ids contain '.' (lr0.01)
        if (len(parts) == 3 and parts[0] == run_id
                and parts[2] in claims.STATUSES and parts[1] != hash8):
            found.add(parts[1])
    return sorted(found)


def scan(cfg):
    """
    Unified hyperparameter scan supporting both SVD and standard optimizers.

    Hydra entry point: it resolves the scan name and the results root and hands
    everything else to :func:`run_grid`, which is the whole run lifecycle and is
    what the tests drive (with an explicit ``scan_dir``, so no test can resolve
    onto the real results root).

    The config should contain:
      - mode: "svd", "standard", "both" (= svd + standard, default), "jd", "hig"
        or "all" (svd + standard + jd + hig). The jd/hig scans run only when
        the config carries their grids (lrs_jd/aggregators_jd, lrs_hig/tau_hig).
      - loss: key into SVD_LOSS_FNS / STANDARD_LOSS_FNS ("ce", "mse",
        "label_regression", "brier", "lm_ce")
      - signed_residual: bool (default true) -- scalar-output regression only;
        Sven rows sign(r)|r|^kappa instead of loss^(kappa/2), any kappa (same
        update, no NaN at r = 0). Ignored for other losses and microbatch_size > 1.
      - svd_info: none | summary | full (default full);
        svd_spectra_schedule: {dense_first: 200, every: 20} -- which steps Sven
        logs its full spectrum on (C-L1/C-L2; the older svd_spectra_every is
        still honoured). Each run writes {run_id}.jsonl (light) +
        diag/{run_id}.npz (per-batch arrays, spectra).
      - eval_batch_size: int (default 2048) -- validation / test / train_eval,
        never the training batch size; train_eval_size: int (default 10000) --
        the fixed training subset the end-of-epoch train_eval loss uses (C-E1/E3).
      - bn_mode: batch | frozen (C-E2; gram_freeze_norm_stats is the deprecated
        alias). Defaults per family to what the config did before the campaign.
      - checkpoints: none | final | epochs | log (default final) and
        checkpoints_svd: the svd family's override (C-L3).
      - empty_cache: bool (default false) -- per-step torch.cuda.empty_cache()
        inside Sven (C-T3); stop_on_nonfinite: bool (default true) -- end a
        diverged run at the non-finite batch with status `diverged` (C-R1).
      - eval_every_steps: int -- extra validation/test at step multiples (C-E4).
      - scheduler: claims (default) | static -- see run_grid.
      - result_id_fields: list of config keys to include in output filenames
      - model_seeds: list of model seeds to sweep over
      - All hparams consumed by grid.process_hparam_config()

    Results go to ``{$SV3_RESULTS_ROOT or experiment_results}/{scan_name}/``.
    """
    # Derive scan name from the Hydra config name (e.g. "mnist_scan")
    scan_name = HydraConfig.get().job.config_name
    return run_grid(cfg, os.path.join(results_root(), scan_name))


def run_grid(cfg, scan_dir):
    """Expand the grid, then claim / dedup / execute / mark every run in it.

    The run lifecycle (C-R1, C-R2, C-R3 and the "Dedup" / "Scheduling" decisions
    in EXPERIMENTS.md section 12), around :func:`execute`. The grid itself lives in
    :mod:`grid`: ``expand_grid(rcfg)`` returns one :class:`~grid.RunSpec` per run,
    seed-major and in the legacy enumeration order.

    Per job, once: the resolved config into ``{scan}/configs/``, this process's
    slice of the intended grid into ``{scan}/manifest/`` (the union over workers
    is what ``tools/reconcile.py`` counts against, C-R2), the provenance of both
    repos (C-R3) and ONE ``os.listdir`` of ``done/`` as the dedup index.

    Per run, in the order ``claims.py``'s docstring prescribes -- cheap index
    skip, claim, fresh on-disk re-check, retire older generations, execute, done
    marker LAST:

    * **skip** when a done marker with *this* run's ``hash8`` and status ``ok`` or
      ``diverged`` exists; ``oom`` and ``error`` are retried (CONTRACTS "Dedup")
      until :data:`_MAX_FAILED_ATTEMPTS` attempts of that hash have failed, after
      which the run is reported as ``poisoned`` and left alone.
    * **claim** ``{scan}/claims/{run_id}.claim`` with ``O_CREAT|O_EXCL`` and hold
      it under a heartbeat, so many processes and many jobs can serve the same
      scan concurrently and a run whose worker died is taken over after 10 min.
    * **retire** a generation with a different ``hash8`` (the config or the code
      changed) into ``{scan}/_stale/{old_hash8}/`` -- moved, never deleted -- and
      run.
    * ``{scan}/started/{run_id}.started`` marks the attempt; a started marker
      with no record is the timeout / hard-kill case (C-R1).

    A run held by another worker is retried once at the end of the walk (one
    mop-up pass), and any that is still claimed then is reported on an
    ``[incomplete]`` line: the exit code stays 0 (CONTRACTS "Runner CLI"), but the
    pass says so rather than looking like a finished scan.

    ``scheduler: claims`` (the default) walks the **whole** grid, so a resubmitted
    job mops up whatever is left instead of re-running its own static slice;
    ``scheduler: static`` restores ``specs[shard_id::n_shards]`` (the legacy
    modulo counter, see :func:`grid.shard`) and is the fallback. Claims are taken
    under both: they cost one file create and they are what makes the
    move-to-stale safe. Returns a :class:`~collections.Counter` of outcomes, and
    exits cleanly (nothing left to claim is success, not failure).
    """
    rcfg = OmegaConf.to_container(cfg, resolve=True)

    # Which families to enumerate, and the scan-level settings derived from the
    # config (loss key + run_id suffix, accuracy/LM flags, diagnostics level,
    # signed residuals). Both validate the config and raise as before.
    flags = mode_flags(rcfg)
    settings = resolve_scan_settings(rcfg)
    svd_settings = resolve_svd_settings(rcfg) if flags["svd"] else None
    os.makedirs(scan_dir, exist_ok=True)

    # Static work-sharding for intra-GPU parallelism: launch N processes with
    # n_shards=N and shard_id=0..N-1; each runs a disjoint 1/N slice of the runs.
    # Only meaningful with `scheduler=static` (CONTRACTS "Runner CLI"): under
    # `claims` a sharded worker would walk only its own slice and could not mop up
    # another shard's leftovers, which is the whole point of the claim queue.
    scheduler = settings["scheduler"]
    n_shards = int(rcfg.get("n_shards", 1))
    shard_id = int(rcfg.get("shard_id", 0))
    if scheduler != "static" and n_shards != 1:
        raise ValueError(
            f"+n_shards/+shard_id require scheduler=static (got scheduler={scheduler!r}, "
            f"n_shards={n_shards}); with the claim queue every worker walks the full grid")

    # Dataset (shared across seeds — same data, different model inits)
    dataset = instantiate(cfg.dataset)

    # Language-model datasets carry vocab_size/block_size the model must match;
    # inject them into the model config so the config need not hardcode the vocab.
    if hasattr(dataset, "vocab_size"):
        # One named function (grid.inject_dataset_facts) so tools/reconcile.py can
        # apply the SAME mutation without importing torch -- its hash8 has to match
        # this one or every LM run reports as a stale generation.
        injected = inject_dataset_facts(
            rcfg["model"], vocab_size=int(dataset.vocab_size),
            block_size=(int(dataset.block_size)
                        if hasattr(dataset, "block_size") else None))
        OmegaConf.set_struct(cfg.model, False)
        for key in ("vocab_size", "block_size"):
            if key in injected:
                cfg.model[key] = injected[key]
        # keep rcfg in step with the mutation: grid.run_hash must see the model
        # config the run actually instantiates, not the un-injected one (C-R3).
        rcfg["model"] = injected

    specs = expand_grid(rcfg, has_torchjd=_HAS_TORCHJD,
                        jd_aggregators=tuple(_JD_AGGREGATORS))
    mine = shard(specs, n_shards, shard_id) if scheduler == "static" else list(specs)

    # Per job, once (C-R2/C-R3). The manifest declares the FULL grid of this
    # process's mode/optimizer/seed subset -- under `claims` every worker of the
    # item is responsible for all of it, and under `static` the per-shard files
    # keep the workers of one SLURM job from overwriting each other's slice.
    job_name = _manifest_job_name(rcfg)
    claims.write_manifest(scan_dir, job_name, [s.run_id for s in specs],
                          shard_id=shard_id if scheduler == "static" else None)
    _save_resolved_config(scan_dir, cfg, job_name)
    prov = provenance.collect(REPO_ROOT, SVEN_ROOT, n_shards=n_shards,
                              shard_id=shard_id)

    print(f"\nGrid: {len(specs)} runs; this worker ({scheduler}"
          f"{f', shard {shard_id + 1}/{n_shards}' if scheduler == 'static' else ''}): "
          f"{len(mine)}")
    print(f"Eval: batch {settings['eval_batch_size']}, train_eval "
          f"{settings['train_eval_size']}; checkpoints {settings['checkpoints']}"
          f"{'' if settings['checkpoints_svd'] is None else '/svd ' + settings['checkpoints_svd']}"
          f"; bn_mode {settings['bn_mode'] or 'per-family default'}"
          f"; spectra {settings['svd_spectra_schedule']}")
    print(f"Code: sv3 {str(prov.get('git_sha'))[:8]}"
          f"{'-dirty' if prov.get('git_dirty') else ''}, sven "
          f"{str(prov.get('sven_git_sha'))[:8]}"
          f"{'-dirty' if prov.get('sven_git_dirty') else ''}"
          f" ({prov.get('git_source')}); manifest {job_name}")

    ctx = _ScanContext(cfg, rcfg, dataset, scan_dir, settings, svd_settings)
    index = claims.done_index(scan_dir)
    counts = Counter()

    def attempt(spec):
        """One grid point, in the order ``claims.py`` prescribes.

        Returns the outcome key: ``skipped``, ``claimed_elsewhere``, ``poisoned``
        or the run's status. ``retired`` is counted on the way through.
        """
        nonlocal index
        run_hash = compute_run_hash(spec, rcfg)
        h8 = run_hash[:8]
        if claims.is_done(index, spec.run_id, h8):
            print(f"  [skip] {spec.run_id}")
            return "skipped"
        claim = claims.try_claim(scan_dir, spec.run_id)
        if claim is None:            # another live worker has it, or it is poisoned
            return "claimed_elsewhere"
        try:
            if claims.is_done_now(scan_dir, spec.run_id, h8):
                print(f"  [skip] {spec.run_id} (finished since this worker started)")
                return "skipped"
            # `oom` / `error` are retried by design, but not forever: a
            # deterministic failure would otherwise be re-executed by every
            # worker of every job for the rest of the campaign.
            failed = _failed_attempts(scan_dir, spec.run_id, h8)
            if failed >= _MAX_FAILED_ATTEMPTS:
                print(f"  [poisoned] {spec.run_id}: {failed} failed attempts at "
                      f"hash {h8}; not retried (see {ATTEMPTS_DIRNAME}/)")
                return "poisoned"
            for old in _stale_hashes_now(scan_dir, spec.run_id, h8):
                moved = claims.move_to_stale(scan_dir, spec.run_id, old)
                if moved:
                    print(f"  [stale] {spec.run_id}: {len(moved)} file(s) -> "
                          f"_stale/{old}/")
                    counts["retired"] += 1
            if claim.took_over:
                print(f"  [takeover] {spec.run_id}: generation {claim.gen}")
            with claims.Heartbeat(claim):
                claims.mark_started(scan_dir, spec.run_id, info=claim.info)
                status = execute(spec, ctx, run_hash=run_hash, prov=prov)
                if status not in claims.DONE_STATUSES:
                    _record_attempt(scan_dir, spec.run_id, h8, status)
                # LAST write, after ckpt + npz + jsonl: only now is the run done.
                claims.mark_done(scan_dir, spec.run_id, h8, status)
            return status
        finally:
            claims.clear_started(scan_dir, spec.run_id)
            claim.release()

    seed, banner, deferred = None, None, []
    for walked, spec in enumerate(mine):
        if spec.model_seed != seed:
            seed = spec.model_seed
            banner = None
            print(f"\n{'#'*80}")
            print(f"# Model seed: {seed}")
            print(f"{'#'*80}")
        if _FAMILY_BANNER[spec.family] != banner:
            banner = _FAMILY_BANNER[spec.family]
            print(f"\n{'='*80}")
            print(banner)
            print(f"{'='*80}")
        # One listdir every N grid points, so a long walk notices what the other
        # workers finished meanwhile and stops attempting claims on it.
        if walked and walked % _INDEX_REFRESH_RUNS == 0:
            index = claims.done_index(scan_dir)

        outcome = attempt(spec)
        if outcome == "claimed_elsewhere":
            deferred.append(spec)      # second pass, below
        else:
            counts[outcome] += 1

    # ONE mop-up pass over the runs a sibling held. The walk never returns to a
    # claimed run, so without this a worker that was hard-killed (timeout,
    # SIGKILL, node failure) has its in-flight runs skipped by every job that was
    # already walking -- and the pass still exits 0 with work left, which is
    # indistinguishable from "the scan is finished". A second look with a fresh
    # index picks up whatever the siblings finished or abandoned; the claim is
    # what keeps this from duplicating live work.
    if deferred:
        print(f"\n[mop-up] {len(deferred)} run(s) were claimed elsewhere; "
              f"second pass")
        index = claims.done_index(scan_dir)
        still = []
        for spec in deferred:
            outcome = attempt(spec)
            if outcome == "claimed_elsewhere":
                still.append(spec.run_id)
            else:
                counts[outcome] += 1
        if still:
            counts["claimed_elsewhere"] = len(still)

    print(f"\nScan complete. Results in {scan_dir}/")
    print("  " + (", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
                  or "nothing left to claim"))
    if counts["claimed_elsewhere"]:
        print(f"  [incomplete] {counts['claimed_elsewhere']} run(s) still claimed by "
              f"another live worker: this pass did NOT cover the whole grid "
              f"(resubmit after the siblings finish; tools/reconcile.py counts them)")
    if counts["poisoned"]:
        print(f"  [poisoned] {counts['poisoned']} run(s) have failed "
              f"{_MAX_FAILED_ATTEMPTS} times at the current hash and are no longer "
              f"retried -- fix the cause and delete their {ATTEMPTS_DIRNAME}/ files")
    return counts
