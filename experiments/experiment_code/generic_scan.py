import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from .experiment_utils import (
    train_loop_svd, train_loop_standard, train_loop_hig, train_loop_jd, set_seed,
    build_standard_optimizer,
)
from .grid import (
    LOSS_KEYS, SIGNED_RESIDUAL_LOSS_KEYS, SVD_INFO_MODES,
    expand_grid, mode_flags, resolve_scan_settings, resolve_svd_settings, shard,
)
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
#                              curve notebooks need (also the dedup marker, so it
#                              is written last).
#   {scan}/diag/{run_id}.npz   "heavy": per-BATCH arrays (batch losses/times) and
#                              the Sven spectra, compressed float32. Loaded on
#                              demand by analysis.style.load_diagnostics /
#                              load_results(slim=False).
# Previously the spectra alone were ~85% of a 7-25 MB JSON file per Sven run.
_DIAG_LOSS_KEYS = ('train_batch', 'val_batch', 'batch_times_train',
                   'batch_times_val', 'train_batch_per_model')
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


def _split_diagnostics(result, svd_info_mode="full", svd_spectra_every=20):
    """Split a result dict into (light, diag).

    ``light`` is ``result`` minus the per-batch arrays and ``svd_info``, plus a
    small per-epoch ``svd_summary``. ``diag`` maps array names to numpy arrays:
    the per-batch loss/time series, ``num_nonzero_svs`` (per step), ``sv_max`` /
    ``sv_min`` (per step, ``summary`` and ``full``) and, for ``full``, the
    spectra ``svs`` (NaN-padded, one row per saved step) with their step
    indices ``svs_step`` -- every ``svd_spectra_every``-th step, starting at 0.
    """
    assert svd_info_mode in SVD_INFO_MODES, svd_info_mode
    every = max(1, int(svd_spectra_every))
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

    si = light.pop('svd_info', None)
    summary = None
    if isinstance(si, dict):
        nnz = np.asarray(si.get('num_nonzero_svs') or [], dtype=np.int32)
        svs = si.get('svs') or []
        summary = {
            'mode': svd_info_mode,
            'n_steps': int(len(nnz)),
            'num_nonzero_svs_epoch': _epoch_mean(nnz, n_epochs),
            'spectra_every': every if svd_info_mode == 'full' else None,
            'spectra_saved': 0,
        }
        if len(nnz):
            diag['num_nonzero_svs'] = nnz
        if len(svs) and svd_info_mode != 'none':
            sv_max = np.array([np.max(x) if np.size(x) else np.nan for x in svs], np.float32)
            sv_min = np.array([np.min(x) if np.size(x) else np.nan for x in svs], np.float32)
            diag['sv_max'], diag['sv_min'] = sv_max, sv_min
            summary['sv_max_epoch'] = _epoch_mean(sv_max, n_epochs)
            summary['sv_min_epoch'] = _epoch_mean(sv_min, n_epochs)
            if svd_info_mode == 'full':
                idx = np.arange(0, len(svs), every)
                diag['svs'] = _pad_ragged([svs[i] for i in idx])
                diag['svs_step'] = idx.astype(np.int32)
                summary['spectra_saved'] = int(len(idx))
        if si.get('k_used'):
            diag['k_used'] = np.asarray(si['k_used'], dtype=np.int32)
        if si.get('variable_k_substep_losses'):
            diag['variable_k_substep_losses'] = _pad_ragged(
                [[float(t) for t in row] for row in si['variable_k_substep_losses']])
    light['svd_summary'] = summary
    return light, diag


def _write_run(scan_dir, run_id, result, svd_info_mode="full", svd_spectra_every=20,
               common=None):
    """Write the heavy diagnostics (npz) first, then the light JSONL (the dedup
    marker), so a run is only ever counted as done once both files exist.

    ``common`` holds per-scan facts every record should carry (``n_params``,
    ``n_train``, ``n_val``; see :func:`_scan_facts`) -- the analysis needs them for
    P/N and steps-per-epoch and used to hard-code them.
    """
    if common:
        for k_, v in common.items():
            result.setdefault(k_, v)
    light, diag = _split_diagnostics(result, svd_info_mode, svd_spectra_every)
    light['diag_file'] = None
    if diag:
        diag_dir = os.path.join(scan_dir, 'diag')
        os.makedirs(diag_dir, exist_ok=True)
        np.savez_compressed(os.path.join(diag_dir, run_id + '.npz'), **diag)
        light['diag_file'] = os.path.join('diag', run_id + '.npz')
    _write_result(os.path.join(scan_dir, run_id + '.jsonl'), light)


def _scan_facts(model, dataset):
    """Facts the analysis otherwise has to hard-code: the parameter count P and the
    train / val set sizes N (steps per epoch = ceil(n_train / batch_size))."""
    facts = {'n_params': int(sum(p.numel() for p in model.parameters()))}
    for key, attr in (('n_train', 'train_dataset'), ('n_val', 'val_dataset')):
        ds = getattr(dataset, attr, None)
        try:
            facts[key] = int(len(ds)) if ds is not None else None
        except TypeError:
            facts[key] = None
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
        self.loss_key = settings["loss_key"]
        self.track_acc = settings["track_acc"]
        self.is_lm = settings["is_lm"]
        self.track_param_norm = settings["track_param_norm"]
        self.svd_info_mode = settings["svd_info_mode"]
        self.svd_spectra_every = settings["svd_spectra_every"]
        self.loss_fn_svd = SVD_LOSS_FNS[self.loss_key]            # per-sample (svd/jd/hig)
        self.loss_fn_standard = STANDARD_LOSS_FNS[self.loss_key]  # scalar
        self.residual_fn_svd = (SVD_RESIDUAL_FNS[self.loss_key]
                                if settings["signed_residual"] else None)
        self._seed = None
        self._init_state = None
        self._common = None

    def seed_state(self, model_seed):
        """``(init_state, common)`` for a model seed, built once per seed."""
        if self._seed != model_seed:
            set_seed(model_seed)
            base_model = instantiate(self.cfg.model)
            self._seed = model_seed
            self._init_state = copy.deepcopy(base_model.state_dict())
            # n_params / n_train / n_val, on every record
            self._common = _scan_facts(base_model, self.dataset)
            del base_model
        return self._init_state, self._common

    def loaders(self, spec, drop_last=False):
        """Train / val loaders for one run (val batch size = train batch size)."""
        train_loader = DataLoader(
            self.dataset.train_dataset, batch_size=spec.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(spec.loader_seed),
            drop_last=drop_last,
        )
        val_loader = DataLoader(self.dataset.val_dataset, batch_size=spec.batch_size,
                                shuffle=False)
        return train_loader, val_loader


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


def execute(spec, ctx):
    """Run one grid point and write its record.

    The single copy of what used to be six near-identical blocks: build the model
    from the seed's initial state, build the optimizer (and wrapper) for the
    family, build the loaders, call the family's training loop, assemble the
    record and write it. Exceptions are printed and swallowed, as before (C-R1
    turns them into records; not this change).
    """
    rcfg, cfg, device = ctx.rcfg, ctx.cfg, ctx.device
    hp = spec.hparams
    init_state, common = ctx.seed_state(spec.model_seed)

    print(f"\n{_describe(spec, ctx)}")
    optimizer = None
    try:
        model = instantiate(cfg.model)
        model.load_state_dict(init_state)
        if spec.family in _TO_DEVICE_FAMILIES:
            model = model.to(device)

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
                    freeze_norm_stats=sv["gram_freeze_norm_stats"],
                    chunk_numel=sv["gram_chunk_numel"],
                    residual_fn=(ctx.residual_fn_svd if use_residual else None),
                )
                optimizer = SvenGram(train_model, lr=hp["lr"], k=hp["k"], rtol=hp["rtol"],
                                     track_svd_info=(ctx.svd_info_mode != "none"))
            else:
                train_model = SvenWrapper(
                    model, ctx.loss_fn_svd, device, kappa=hp["kappa"],
                    microbatch_size=mb, param_fraction=pf,
                    mask_mode=(sv["mask_mode"] if pf < 1.0 else None),
                    residual_fn=(ctx.residual_fn_svd if use_residual else None),
                )
                optimizer = Sven(
                    train_model, lr=hp["lr"], k=hp["k"], rtol=hp["rtol"],
                    track_svd_info=(ctx.svd_info_mode != "none"), svd_mode=hp["svd_mode"],
                    variable_k=sv["variable_k"],
                )
            train_loader, val_loader = ctx.loaders(
                spec, drop_last=(hp["microbatch_size"] is not None))
            train_model, losses, optimizer = train_loop_svd(
                train_model, optimizer, ctx.loss_fn_svd,
                train_loader, val_loader,
                rcfg["num_epochs"], device, track_acc=ctx.track_acc,
                track_param_norm=ctx.track_param_norm, is_lm=ctx.is_lm,
            )

        elif spec.family == "standard":
            optimizer = build_standard_optimizer(model, hp["optim_name"], hp["lr"],
                                                 weight_decay=hp["weight_decay"])
            train_loader, val_loader = ctx.loaders(spec)
            model, losses = train_loop_standard(
                model, optimizer, ctx.loss_fn_standard,
                train_loader, val_loader,
                rcfg["num_epochs"], device, track_acc=ctx.track_acc,
                track_param_norm=ctx.track_param_norm, is_lm=ctx.is_lm,
            )

        elif spec.family == "lbfgs":
            lbfgs_kwargs = {
                "max_iter": hp["max_iter"],
                "history_size": hp["history_size"],
                "line_search_fn": (hp["line_search_fn"]
                                   if hp["line_search_fn"] != "none" else None),
            }
            optimizer = build_standard_optimizer(model, "LBFGS", hp["lr"], **lbfgs_kwargs)
            train_loader, val_loader = ctx.loaders(spec)
            model, losses = train_loop_standard(
                model, optimizer, ctx.loss_fn_standard,
                train_loader, val_loader,
                rcfg["num_epochs"], device, track_acc=ctx.track_acc,
                is_lm=ctx.is_lm,
            )

        elif spec.family == "polyak":
            polyak_kwargs = {"f_star": hp["f_star"], "max_lr": hp["max_lr"],
                             "eps": hp["eps"]}
            optimizer = build_standard_optimizer(model, "PolyakSGD", lr=None,
                                                 **polyak_kwargs)
            train_loader, val_loader = ctx.loaders(spec)
            model, losses = train_loop_standard(
                model, optimizer, ctx.loss_fn_standard,
                train_loader, val_loader,
                rcfg["num_epochs"], device, track_acc=ctx.track_acc,
                is_lm=ctx.is_lm,
            )

        elif spec.family == "jd":
            aggregator = _JD_AGGREGATORS[hp["aggregator"]]()
            optimizer = build_standard_optimizer(model, hp["inner_optimizer"], hp["lr"])
            train_loader, val_loader = ctx.loaders(spec)
            model, losses = train_loop_jd(
                model, optimizer, aggregator, ctx.loss_fn_svd,
                train_loader, val_loader, rcfg["num_epochs"], device,
                track_acc=ctx.track_acc, track_param_norm=ctx.track_param_norm,
            )

        elif spec.family == "hig":
            train_model = HIGWrapper(model, ctx.loss_fn_svd, device)
            optimizer = HIGOptimizer(train_model, lr=hp["lr"], tau=hp["tau"])
            train_loader, val_loader = ctx.loaders(spec)
            train_model, losses = train_loop_hig(
                train_model, optimizer, ctx.loss_fn_svd,
                train_loader, val_loader, rcfg["num_epochs"], device,
                track_acc=ctx.track_acc, track_param_norm=ctx.track_param_norm,
            )

        else:
            # A family added to grid.FAMILIES must get a branch here; without this
            # it would silently run under the wrong wrapper and write a record.
            raise AssertionError(f"execute: unhandled family {spec.family!r}")

        result = {"run_id": spec.run_id, **spec.record_extra, "losses": losses}
        if spec.family == "svd":
            result["svd_info"] = getattr(optimizer, "svd_info", {})
        for f in rcfg.get("result_id_fields", []):
            result[f] = rcfg[f]

        _write_run(ctx.scan_dir, spec.run_id, result, ctx.svd_info_mode,
                   ctx.svd_spectra_every, common)

    except Exception as e:
        # "standard" is the only family whose message uses the optimizer's own name
        # (the .get keeps the unhandled-family raise above from turning into a KeyError).
        label = _FAMILY_ERROR.get(spec.family) or f"{hp.get('optim_name', spec.family)} run failed"
        print(f"  [error] {label}: {e}")
        if spec.family != "svd" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        if spec.family == "svd":
            torch.compiler.reset()


def scan(cfg):
    """
    Unified hyperparameter scan supporting both SVD and standard optimizers.

    The grid itself lives in :mod:`grid`: ``expand_grid(rcfg)`` returns one
    ``RunSpec`` per run (family, run_id, seeds, batch size, hyperparameters and
    the record scaffold), seed-major and in the legacy enumeration order. This
    function builds the shared context, takes its shard of the specs, skips the
    ones already on disk and calls :func:`execute` on the rest.

    Results are stored as JSONL (one file per run) inside
    ``experiment_results/{scan_name}/``; ``{run_id}.jsonl`` is also the dedup
    marker.

    The config should contain:
      - mode: "svd", "standard", "both" (= svd + standard, default), "jd", "hig"
        or "all" (svd + standard + jd + hig). The jd/hig scans run only when
        the config carries their grids (lrs_jd/aggregators_jd, lrs_hig/tau_hig).
      - loss: key into SVD_LOSS_FNS / STANDARD_LOSS_FNS ("ce", "mse",
        "label_regression", "brier", "lm_ce")
      - signed_residual: bool (default true) -- scalar-output regression only;
        Sven rows sign(r)|r|^kappa instead of loss^(kappa/2), any kappa (same
        update, no NaN at r = 0). Ignored for other losses and microbatch_size > 1.
      - svd_info: none | summary | full (default full); svd_spectra_every: int
        (default 20) -- see _split_diagnostics. Each run writes {run_id}.jsonl
        (light) + diag/{run_id}.npz (per-batch arrays, spectra).
      - result_id_fields: list of config keys to include in output filenames
      - model_seeds: list of model seeds to sweep over
      - All hparams consumed by grid.process_hparam_config()
    """
    rcfg = OmegaConf.to_container(cfg, resolve=True)

    # Which families to enumerate, and the scan-level settings derived from the
    # config (loss key + run_id suffix, accuracy/LM flags, diagnostics level,
    # signed residuals). Both validate the config and raise as before.
    flags = mode_flags(rcfg)
    settings = resolve_scan_settings(rcfg)
    svd_settings = resolve_svd_settings(rcfg) if flags["svd"] else None

    # Derive scan name from the Hydra config name (e.g. "mnist_scan")
    scan_name = HydraConfig.get().job.config_name

    # Output directory: one JSONL file per run inside {output_dir}/{scan_name}/
    output_dir = "experiment_results"
    scan_dir = os.path.join(output_dir, scan_name)
    os.makedirs(scan_dir, exist_ok=True)

    # Optional work-sharding for intra-GPU parallelism: launch N processes with
    # n_shards=N and shard_id=0..N-1; each runs a disjoint 1/N slice of the runs.
    # The slice is taken over the full grid, before dedup, so shard assignment is
    # stable across resumes; disjoint shards write disjoint run_ids, so it is
    # race-free. (Identical to the legacy modulo counter, see grid.shard.)
    n_shards = int(rcfg.get("n_shards", 1))
    shard_id = int(rcfg.get("shard_id", 0))

    # Dataset (shared across seeds — same data, different model inits)
    dataset = instantiate(cfg.dataset)

    # Language-model datasets carry vocab_size/block_size the model must match;
    # inject them into the model config so the config need not hardcode the vocab.
    if hasattr(dataset, "vocab_size"):
        OmegaConf.set_struct(cfg.model, False)
        cfg.model.vocab_size = int(dataset.vocab_size)
        if hasattr(dataset, "block_size") and "block_size" in cfg.model:
            cfg.model.block_size = int(dataset.block_size)
        # keep rcfg in step with the mutation: grid.run_hash must see the model
        # config the run actually instantiates, not the un-injected one (C-R3).
        rcfg["model"] = OmegaConf.to_container(cfg.model, resolve=True)

    specs = expand_grid(rcfg, has_torchjd=_HAS_TORCHJD,
                        jd_aggregators=tuple(_JD_AGGREGATORS))
    mine = shard(specs, n_shards, shard_id)
    print(f"\nGrid: {len(specs)} runs; this shard ({shard_id + 1}/{n_shards}): {len(mine)}")

    ctx = _ScanContext(cfg, rcfg, dataset, scan_dir, settings, svd_settings)

    seed, banner = None, None
    for spec in mine:
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
        if os.path.exists(os.path.join(scan_dir, spec.run_id + ".jsonl")):
            print(f"  [skip] {spec.run_id}")
            continue
        execute(spec, ctx)

    print(f"\nScan complete. Results in {scan_dir}/")
