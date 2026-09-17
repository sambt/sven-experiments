import copy
import json
import os
from itertools import product

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
    process_hparam_config, build_standard_optimizer,
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
SVD_INFO_MODES = ('none', 'summary', 'full')


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


def _write_run(scan_dir, run_id, result, svd_info_mode="full", svd_spectra_every=20):
    """Write the heavy diagnostics (npz) first, then the light JSONL (the dedup
    marker), so a run is only ever counted as done once both files exist."""
    light, diag = _split_diagnostics(result, svd_info_mode, svd_spectra_every)
    light['diag_file'] = None
    if diag:
        diag_dir = os.path.join(scan_dir, 'diag')
        os.makedirs(diag_dir, exist_ok=True)
        np.savez_compressed(os.path.join(diag_dir, run_id + '.npz'), **diag)
        light['diag_file'] = os.path.join('diag', run_id + '.npz')
    _write_result(os.path.join(scan_dir, run_id + '.jsonl'), light)


# ---------------------------------------------------------------------------
# Scan logic
# ---------------------------------------------------------------------------

def _build_id_string(cfg):
    """Build a model-identifier string from config-specified fields."""
    fields = cfg.get("result_id_fields", [])
    if not fields:
        return ""
    return "_" + "_".join(f"{f}{cfg[f]}" for f in fields)


def scan(cfg):
    """
    Unified hyperparameter scan supporting both SVD and standard optimizers.

    Results are stored as JSONL (one JSON object per line, one file per scan).
    Each result row includes a 'run_id' string for deduplication — if a run_id
    already exists in the file, that run is skipped.

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
      - seeds: list of model seeds to sweep over (optional; falls back to model_seed)
      - All hparams consumed by process_hparam_config()
    """
    rcfg = OmegaConf.to_container(cfg, resolve=True)
    device = rcfg["device"]

    mode = rcfg.get("mode", "both")
    _VALID_MODES = ("svd", "standard", "both", "jd", "hig", "all")
    assert mode in _VALID_MODES, f"Unknown mode: {mode}. Choose from {_VALID_MODES}"
    run_svd = mode in ("svd", "both", "all")
    run_standard = mode in ("standard", "both", "all")
    run_jd = mode in ("jd", "all") and ("lrs_jd" in rcfg or "aggregators_jd" in rcfg)
    run_hig = mode in ("hig", "all") and ("lrs_hig" in rcfg or "tau_hig" in rcfg)

    loss_key = rcfg.get("loss", "ce")
    if loss_key not in SVD_LOSS_FNS or loss_key not in STANDARD_LOSS_FNS:
        raise KeyError(f"Unknown loss key {loss_key!r}; known: {sorted(SVD_LOSS_FNS)}")
    # The legacy keys keep their historical (unsuffixed) run_ids so dedup against
    # the existing result files is unaffected. Any other key (e.g.
    # brier) is encoded in the run_id, so a `loss=` override
    # on an existing config name can never dedup against -- or be mistaken for --
    # that config's original-loss results. Every result row also records `loss`.
    _LEGACY_LOSS_KEYS = ("ce", "mse", "label_regression", "lm_ce")
    loss_suffix = "" if loss_key in _LEGACY_LOSS_KEYS else f"_loss{loss_key}"
    # Only track accuracy for classification (CE, label regression, Brier).
    track_acc = loss_key in ("ce", "brier") or ("label_regression" in loss_key)
    is_lm = loss_key == "lm_ce"  # language modeling: 3D logits, no classification accuracy
    track_param_norm = rcfg.get("track_param_norm", False)

    # How much Sven SVD diagnostics to keep (see _split_diagnostics):
    #   none    -- nothing recorded (track_svd_info=False);
    #   summary -- per-step rank + largest/smallest kept singular value only;
    #   full    -- also the spectrum itself, every `svd_spectra_every`-th step.
    # Default full/20: ~1/20th of the spectra, which is all the notebooks sample.
    # Set svd_spectra_every: 1 to keep every step (heavy: k floats per step).
    svd_info_mode = rcfg.get("svd_info", "full")
    if svd_info_mode not in SVD_INFO_MODES:
        raise ValueError(f"svd_info must be one of {SVD_INFO_MODES}, got {svd_info_mode!r}")
    svd_spectra_every = int(rcfg.get("svd_spectra_every", 20))

    # Signed-residual rows for scalar-output regression (see SVD_RESIDUAL_FNS).
    # Not encoded in the run_id: the update is identical to the loss path
    # wherever the latter is finite; the flag is recorded in the result row.
    signed_residual = bool(rcfg.get("signed_residual", True)) and loss_key in SVD_RESIDUAL_FNS
    residual_fn_svd = SVD_RESIDUAL_FNS[loss_key] if signed_residual else None

    # Derive scan name from the Hydra config name (e.g. "mnist_scan")
    scan_name = HydraConfig.get().job.config_name

    # Output directory: one JSONL file per run inside {output_dir}/{scan_name}/
    output_dir = "experiment_results"
    scan_dir = os.path.join(output_dir, scan_name)
    os.makedirs(scan_dir, exist_ok=True)

    # Parse hparam grid
    hparams = process_hparam_config(rcfg)
    id_str = _build_id_string(rcfg)

    # Seed list: use 'seeds' if provided, otherwise single 'model_seed'
    seeds = rcfg.get("model_seeds")
    loader_seed = rcfg["loader_seed"]

    # Optional work-sharding for intra-GPU parallelism: launch N processes with
    # n_shards=N and shard_id=0..N-1; each runs a disjoint 1/N slice of the runs.
    # The counter advances before the dedup check so shard assignment is stable
    # across resumes; disjoint shards write disjoint run_ids, so it is race-free.
    n_shards = int(rcfg.get("n_shards", 1))
    shard_id = int(rcfg.get("shard_id", 0))
    _run_idx = [0]
    def _shard_skip():
        take = (_run_idx[0] % n_shards) == shard_id
        _run_idx[0] += 1
        return not take

    # Dataset (shared across seeds — same data, different model inits)
    dataset = instantiate(cfg.dataset)

    # Language-model datasets carry vocab_size/block_size the model must match;
    # inject them into the model config so the config need not hardcode the vocab.
    if hasattr(dataset, "vocab_size"):
        OmegaConf.set_struct(cfg.model, False)
        cfg.model.vocab_size = int(dataset.vocab_size)
        if hasattr(dataset, "block_size") and "block_size" in cfg.model:
            cfg.model.block_size = int(dataset.block_size)

    for model_seed in seeds:
        print(f"\n{'#'*80}")
        print(f"# Model seed: {model_seed}")
        print(f"{'#'*80}")

        seed_str = f"_mseed{model_seed}_lseed{loader_seed}"

        # Initialize model with this seed
        set_seed(model_seed)
        base_model = instantiate(cfg.model)
        init_state = copy.deepcopy(base_model.state_dict())
        del base_model

        # --------------------------------------------------------------
        # SVD optimizer scan
        # --------------------------------------------------------------
        if run_svd:
            print(f"\n{'='*80}")
            print("Running SVD optimizer scan")
            print(f"{'='*80}")

            k_scan_values = hparams.get('k_fractions', hparams.get('k_values'))
            use_k_values = 'k_values' in hparams

            # Under the Gram backend the SVD of J is replaced by an exact eigh of
            # the B x B Gram matrix, so `svd_mode` selects nothing. Collapse it to
            # the single canonical token "torch" so (a) a list of modes cannot
            # multiply the grid into duplicate runs and (b) the run_id never
            # advertises a randomized SVD that was not used.
            if rcfg.get("use_gram", False) and list(hparams['svd_mode']) != ['torch']:
                print(f"  [note] use_gram: ignoring svd_mode={hparams['svd_mode']} "
                      "(exact Gram eigendecomposition); run_ids use 'torch'")
                hparams['svd_mode'] = ['torch']

            svd_grid = product(
                hparams['batch_size'],
                k_scan_values,
                hparams['lrs'],
                hparams['rtol'],
                hparams['svd_mode'],
                hparams['microbatch_sizes'],
                hparams['param_fractions'],
                hparams['kappas'],
            )

            loss_fn_svd = SVD_LOSS_FNS[loss_key]
            variable_k = rcfg.get("variable_k", False)
            # Gram-trick backend: same exact update, ~400x faster / far less memory
            # (eigendecomposes B x B G = J J^T instead of materializing the B x P Jacobian).
            # Incompatible with variable_k.
            use_gram = rcfg.get("use_gram", False)
            if use_gram and variable_k:
                raise ValueError("use_gram is incompatible with variable_k")
            # Gram capture backend: "hooks" (fast, per-sample-decoupled layers only),
            # "chunked" (exact for any architecture; one jacrev per parameter group of
            # <= gram_chunk_numel elements) or "full" (exact; ONE jacrev over all
            # parameters = the full (B, P) Jacobian materialised once, B*P*4 bytes).
            # Default "hooks" for the MLP suite; CIFAR/ResNet uses "full".
            gram_capture = rcfg.get("gram_capture", "hooks")
            # BatchNorm handling under Gram. True (default): norm layers run in eval
            # mode (running stats) during every wrapper pass -- required by hooks
            # capture, but the running stats are then never updated from their init,
            # so a BN net is effectively un-normalised. False: batch statistics, as in
            # the classic jacrev path / the paper; needs capture="chunked".
            gram_freeze_norm_stats = bool(rcfg.get("gram_freeze_norm_stats", True))
            # Chunked capture: parameters are split into groups of <= gram_chunk_numel
            # elements, one jacrev per group per step. A cap above the parameter count
            # gives ONE group = the full (B, P) Jacobian materialised once (B*P*4 bytes)
            # and contracted into the Gram -- fewest passes, most memory.
            gram_chunk_numel = int(rcfg.get("gram_chunk_numel", 2 ** 22))
            if use_gram and not gram_freeze_norm_stats and gram_capture not in ("chunked", "full"):
                raise ValueError("gram_freeze_norm_stats=false requires gram_capture: chunked or full")
            # Parameter-fraction mask structure (param_fraction < 1 only):
            # "elementwise" (default, matches the paper: random individual weights),
            # "rows" (whole output neurons/channels — coarse, and cannot split
            # BatchNorm so it needs capture="chunked"), or "tensor". Elementwise
            # runs on the fast hooks path for Linear/Conv2d/_NormBase.
            svd_mask_mode = rcfg.get("mask_mode", "elementwise")

            for batch_size, k_item, lr, rtol, svd_mode, microbatch_size, param_fraction, kappa in svd_grid:
                k = max(1, int(k_item * batch_size)) if not use_k_values else k_item

                # Build run_id for deduplication
                run_id = (
                    f"svd_bs{batch_size}{id_str}"
                    f"_k{k}_lr{lr}_rtol{rtol}_svd{svd_mode}{seed_str}"
                )
                if microbatch_size is not None:
                    run_id += f"_mb{microbatch_size}"
                if param_fraction is not None:
                    run_id += f"_pf{param_fraction}"
                    if param_fraction < 1.0:
                        run_id += f"_{svd_mask_mode}"  # elementwise vs rows -> distinct runs
                if variable_k:
                    run_id += "_variablek"
                if use_gram:
                    run_id += "_gram"
                    if not gram_freeze_norm_stats:
                        run_id += "_bnbatch"  # batch-statistics BatchNorm (distinct from frozen-stats runs)
                if kappa != 2.0:
                    run_id += f"_kappa{kappa}"
                run_id += loss_suffix

                if _shard_skip():
                    continue
                if os.path.exists(os.path.join(scan_dir, run_id + ".jsonl")):
                    print(f"  [skip] {run_id}")
                    continue

                print(f"\nSVD: bs={batch_size}, k={k}, lr={lr}, rtol={rtol}, svd_mode={svd_mode}", end="")
                if microbatch_size is not None:
                    print(f", mb={microbatch_size}", end="")
                if param_fraction is not None:
                    print(f", pf={param_fraction}", end="")
                if kappa != 2.0:
                    print(f", kappa={kappa}", end="")
                if variable_k:
                    print(f", variable_k=True", end="")
                print()

                try:
                    model = instantiate(cfg.model)
                    model.load_state_dict(init_state)

                    mb = microbatch_size if microbatch_size is not None else 1
                    pf = param_fraction if param_fraction is not None else 1.0
                    use_residual = signed_residual and mb == 1  # any kappa
                    if use_gram:
                        # Gram trick: exact same update via B x B G = J J^T (no B x P Jacobian).
                        # svd_mode is irrelevant (eigendecomposition of G replaces the SVD of J).
                        train_model = GramSvenWrapper(
                            model, loss_fn_svd, device,
                            kappa=kappa,
                            microbatch_size=mb, param_fraction=pf,
                            mask_mode=(svd_mask_mode if pf < 1.0 else None),
                            capture=gram_capture,
                            freeze_norm_stats=gram_freeze_norm_stats,
                            chunk_numel=gram_chunk_numel,
                            residual_fn=(residual_fn_svd if use_residual else None),
                        )
                        optimizer = SvenGram(train_model, lr=lr, k=k, rtol=rtol, track_svd_info=(svd_info_mode != "none"))
                    else:
                        train_model = SvenWrapper(
                            model, loss_fn_svd, device, kappa=kappa,
                            microbatch_size=mb, param_fraction=pf,
                            mask_mode=(svd_mask_mode if pf < 1.0 else None),
                            residual_fn=(residual_fn_svd if use_residual else None),
                        )
                        optimizer = Sven(
                            train_model, lr=lr, k=k, rtol=rtol,
                            track_svd_info=(svd_info_mode != "none"), svd_mode=svd_mode,
                            variable_k=variable_k,
                        )

                    train_loader = DataLoader(
                        dataset.train_dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(loader_seed),
                        drop_last=(microbatch_size is not None),
                    )
                    val_loader = DataLoader(dataset.val_dataset, batch_size=batch_size, shuffle=False)

                    train_model, losses, optimizer = train_loop_svd(
                        train_model, optimizer, loss_fn_svd,
                        train_loader, val_loader,
                        rcfg["num_epochs"], device, track_acc=track_acc,
                        track_param_norm=track_param_norm, is_lm=is_lm,
                    )

                    result = {
                        "run_id": run_id,
                        "optimizer": "SVD",
                        "loss": loss_key,
                        "batch_size": batch_size,
                        "k_fraction": k / batch_size,
                        "k": k,
                        "lr": lr,
                        "rtol": rtol,
                        "model_seed": model_seed,
                        "loader_seed": loader_seed,
                        "svd_mode": svd_mode,
                        # exact eigh of G (Gram) vs the SVD algorithm named by svd_mode
                        "decomposition": "gram_eigh" if use_gram else f"svd_{svd_mode}",
                        "microbatch_size": microbatch_size,
                        "param_fraction": param_fraction,
                        "variable_k": variable_k,
                        "use_gram": use_gram,
                        "gram_capture": gram_capture if use_gram else None,
                        "gram_freeze_norm_stats": gram_freeze_norm_stats if use_gram else None,
                        "gram_chunk_numel": gram_chunk_numel if (use_gram and gram_capture == "chunked") else None,
                        "mask_mode": (svd_mask_mode if param_fraction is not None and param_fraction < 1.0 else None),
                        "kappa": kappa,
                        "signed_residual": bool(use_residual),
                        "losses": losses,
                        "svd_info": getattr(optimizer, "svd_info", {})
                    }
                    for f in rcfg.get("result_id_fields", []):
                        result[f] = rcfg[f]

                    _write_run(scan_dir, run_id, result, svd_info_mode, svd_spectra_every)

                except Exception as e:
                    print(f"  [error] Training failed: {e}")

                torch.compiler.reset()

        # --------------------------------------------------------------
        # Standard optimizer scan
        # --------------------------------------------------------------
        if run_standard:
            print(f"\n{'='*80}")
            print("Running standard optimizer scan")
            print(f"{'='*80}")

            # Build the grid — LBFGS and PolyakSGD get their own dedicated scan blocks
            has_lbfgs = "LBFGS" in hparams['optimizers_standard']
            has_polyak = "PolyakSGD" in hparams['optimizers_standard']
            non_lbfgs_optimizers = [o for o in hparams['optimizers_standard'] if o not in ("LBFGS", "PolyakSGD")]

            loss_fn_standard = STANDARD_LOSS_FNS[loss_key]

            # --- Non-LBFGS optimizers (original grid) ---
            if non_lbfgs_optimizers:
                standard_grid = product(
                    hparams['batch_size'],
                    hparams['lrs_standard'],
                    non_lbfgs_optimizers,
                    hparams['weight_decays'],
                )

                for batch_size, lr, optim_name, weight_decay in standard_grid:
                    # Non-zero weight decay is only meaningful for AdamW and Muon
                    if optim_name not in ("AdamW", "Muon") and weight_decay != 0.0:
                        continue

                    run_id = f"std_bs{batch_size}{id_str}_lr{lr}_optim{optim_name}"
                    if weight_decay != 0.0:
                        run_id += f"_wd{weight_decay}"
                    run_id += seed_str + loss_suffix

                    if _shard_skip():
                        continue
                    if os.path.exists(os.path.join(scan_dir, run_id + ".jsonl")):
                        print(f"  [skip] {run_id}")
                        continue

                    wd_str = f", wd={weight_decay}" if weight_decay != 0.0 else ""
                    print(f"\nStandard: bs={batch_size}, lr={lr}, optim={optim_name}{wd_str}")

                    try:
                        model = instantiate(cfg.model)
                        model.load_state_dict(init_state)
                        model = model.to(device)

                        optimizer = build_standard_optimizer(model, optim_name, lr,
                                                             weight_decay=weight_decay)

                        train_loader = DataLoader(
                            dataset.train_dataset, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(loader_seed),
                        )
                        val_loader = DataLoader(dataset.val_dataset, batch_size=batch_size, shuffle=False)

                        model, losses = train_loop_standard(
                            model, optimizer, loss_fn_standard,
                            train_loader, val_loader,
                            rcfg["num_epochs"], device, track_acc=track_acc,
                            track_param_norm=track_param_norm, is_lm=is_lm,
                        )

                        result = {
                            "run_id": run_id,
                            "optimizer": optim_name,
                            "loss": loss_key,
                            "batch_size": batch_size,
                            "k_fraction": None,
                            "k": None,
                            "lr": lr,
                            "rtol": None,
                            "weight_decay": weight_decay,
                            "model_seed": model_seed,
                            "loader_seed": loader_seed,
                            "svd_mode": None,
                            "svd_info": None,
                            "losses": losses,
                        }
                        for f in rcfg.get("result_id_fields", []):
                            result[f] = rcfg[f]

                        _write_run(scan_dir, run_id, result, svd_info_mode, svd_spectra_every)
                    except Exception as e:
                        print(f"  [error] {optim_name} run failed: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            # --- LBFGS optimizer (separate grid with LBFGS-specific params) ---
            if has_lbfgs:
                lbfgs_grid = product(
                    hparams['batch_size'],
                    hparams['lrs_lbfgs'],
                    hparams['lbfgs_max_iter'],
                    hparams['lbfgs_history_size'],
                    hparams['lbfgs_line_search_fn'],
                )

                for batch_size, lr, max_iter, history_size, line_search_fn in lbfgs_grid:
                    run_id = (
                        f"std_bs{batch_size}{id_str}_lr{lr}_optimLBFGS"
                        f"_mi{max_iter}_hs{history_size}_ls{line_search_fn}{seed_str}"
                        f"{loss_suffix}"
                    )

                    if _shard_skip():
                        continue
                    if os.path.exists(os.path.join(scan_dir, run_id + ".jsonl")):
                        print(f"  [skip] {run_id}")
                        continue

                    print(f"\nLBFGS: bs={batch_size}, lr={lr}, max_iter={max_iter}, "
                          f"history_size={history_size}, line_search={line_search_fn}")

                    try:
                        model = instantiate(cfg.model)
                        model.load_state_dict(init_state)
                        model = model.to(device)

                        lbfgs_kwargs = {
                            "max_iter": max_iter,
                            "history_size": history_size,
                            "line_search_fn": line_search_fn if line_search_fn != "none" else None,
                        }
                        optimizer = build_standard_optimizer(model, "LBFGS", lr, **lbfgs_kwargs)

                        train_loader = DataLoader(
                            dataset.train_dataset, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(loader_seed),
                        )
                        val_loader = DataLoader(dataset.val_dataset, batch_size=batch_size, shuffle=False)

                        model, losses = train_loop_standard(
                            model, optimizer, loss_fn_standard,
                            train_loader, val_loader,
                            rcfg["num_epochs"], device, track_acc=track_acc,
                            is_lm=is_lm,
                        )

                        result = {
                            "run_id": run_id,
                            "optimizer": "LBFGS",
                            "loss": loss_key,
                            "batch_size": batch_size,
                            "k_fraction": None,
                            "k": None,
                            "lr": lr,
                            "rtol": None,
                            "model_seed": model_seed,
                            "loader_seed": loader_seed,
                            "svd_mode": None,
                            "svd_info": None,
                            "lbfgs_max_iter": max_iter,
                            "lbfgs_history_size": history_size,
                            "lbfgs_line_search_fn": line_search_fn,
                            "losses": losses,
                        }
                        for f in rcfg.get("result_id_fields", []):
                            result[f] = rcfg[f]

                        _write_run(scan_dir, run_id, result, svd_info_mode, svd_spectra_every)
                    except Exception as e:
                        print(f"  [error] LBFGS run failed: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            # --- PolyakSGD optimizer (no LR sweep) ---
            if has_polyak:
                polyak_grid = product(
                    hparams['batch_size'],
                    hparams['polyak_f_star'],
                    hparams['polyak_max_lr'],
                    hparams['polyak_eps'],
                )

                for batch_size, f_star, max_lr, eps in polyak_grid:
                    run_id = (
                        f"std_bs{batch_size}{id_str}_optimPolyakSGD"
                        f"_fstar{f_star}_maxlr{max_lr}_eps{eps}{seed_str}"
                        f"{loss_suffix}"
                    )

                    if _shard_skip():
                        continue
                    if os.path.exists(os.path.join(scan_dir, run_id + ".jsonl")):
                        print(f"  [skip] {run_id}")
                        continue

                    print(f"\nPolyakSGD: bs={batch_size}, f_star={f_star}, max_lr={max_lr}, eps={eps}")

                    try:
                        model = instantiate(cfg.model)
                        model.load_state_dict(init_state)
                        model = model.to(device)

                        polyak_kwargs = {"f_star": f_star, "max_lr": max_lr, "eps": eps}
                        optimizer = build_standard_optimizer(model, "PolyakSGD", lr=None, **polyak_kwargs)

                        train_loader = DataLoader(
                            dataset.train_dataset, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(loader_seed),
                        )
                        val_loader = DataLoader(dataset.val_dataset, batch_size=batch_size, shuffle=False)

                        model, losses = train_loop_standard(
                            model, optimizer, loss_fn_standard,
                            train_loader, val_loader,
                            rcfg["num_epochs"], device, track_acc=track_acc,
                            is_lm=is_lm,
                        )

                        result = {
                            "run_id": run_id,
                            "optimizer": "PolyakSGD",
                            "loss": loss_key,
                            "batch_size": batch_size,
                            "k_fraction": None,
                            "k": None,
                            "lr": None,
                            "rtol": None,
                            "model_seed": model_seed,
                            "loader_seed": loader_seed,
                            "svd_mode": None,
                            "svd_info": None,
                            "polyak_f_star": f_star,
                            "polyak_max_lr": max_lr,
                            "polyak_eps": eps,
                            "losses": losses,
                        }
                        for f in rcfg.get("result_id_fields", []):
                            result[f] = rcfg[f]

                        _write_run(scan_dir, run_id, result, svd_info_mode, svd_spectra_every)
                    except Exception as e:
                        print(f"  [error] PolyakSGD run failed: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        # --------------------------------------------------------------
        # Jacobian Descent (torchjd) scan
        # --------------------------------------------------------------
        if run_jd:
            if not _HAS_TORCHJD:
                print("  [skip] torchjd not installed -- skipping JD scan")
            else:
                print(f"\n{'='*80}")
                print("Running Jacobian Descent scan")
                print(f"{'='*80}")
                loss_fn_jd = SVD_LOSS_FNS[loss_key]  # per-sample losses
                jd_grid = product(
                    hparams['batch_size'], hparams['lrs_jd'],
                    hparams['aggregators_jd'], hparams['inner_optimizers_jd'],
                )
                for batch_size, lr, aggregator_name, inner_optim_name in jd_grid:
                    if aggregator_name not in _JD_AGGREGATORS:
                        print(f"  [skip] Unknown JD aggregator: {aggregator_name}")
                        continue
                    run_id = (
                        f"jd_bs{batch_size}{id_str}"
                        f"_lr{lr}_agg{aggregator_name}_inner{inner_optim_name}{seed_str}{loss_suffix}"
                    )
                    if _shard_skip():
                        continue
                    if os.path.exists(os.path.join(scan_dir, run_id + ".jsonl")):
                        print(f"  [skip] {run_id}")
                        continue
                    print(f"\nJD: bs={batch_size}, lr={lr}, aggregator={aggregator_name}, inner={inner_optim_name}")
                    try:
                        model = instantiate(cfg.model)
                        model.load_state_dict(init_state)
                        model = model.to(device)
                        aggregator = _JD_AGGREGATORS[aggregator_name]()
                        inner_optimizer = build_standard_optimizer(model, inner_optim_name, lr)
                        train_loader = DataLoader(
                            dataset.train_dataset, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(loader_seed),
                        )
                        val_loader = DataLoader(dataset.val_dataset, batch_size=batch_size, shuffle=False)
                        model, losses = train_loop_jd(
                            model, inner_optimizer, aggregator, loss_fn_jd,
                            train_loader, val_loader, rcfg["num_epochs"], device,
                            track_acc=track_acc, track_param_norm=track_param_norm,
                        )
                        result = {
                            "run_id": run_id,
                            "optimizer": f"JD_{aggregator_name}",
                            "loss": loss_key,
                            "batch_size": batch_size,
                            "lr": lr,
                            "aggregator": aggregator_name,
                            "inner_optimizer": inner_optim_name,
                            "model_seed": model_seed,
                            "loader_seed": loader_seed,
                            "svd_info": None,
                            "losses": losses,
                        }
                        for f in rcfg.get("result_id_fields", []):
                            result[f] = rcfg[f]
                        _write_run(scan_dir, run_id, result, svd_info_mode, svd_spectra_every)
                    except Exception as e:
                        print(f"  [error] JD run failed: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        # --------------------------------------------------------------
        # Half-Inverse Gradients (HIG) scan
        # --------------------------------------------------------------
        if run_hig:
            print(f"\n{'='*80}")
            print("Running Half-Inverse Gradients scan")
            print(f"{'='*80}")
            loss_fn_hig = SVD_LOSS_FNS[loss_key]  # per-sample losses
            hig_grid = product(hparams['batch_size'], hparams['lrs_hig'], hparams['tau_hig'])
            for batch_size, lr, tau in hig_grid:
                run_id = f"hig_bs{batch_size}{id_str}_lr{lr}_tau{tau}{seed_str}{loss_suffix}"
                if _shard_skip():
                    continue
                if os.path.exists(os.path.join(scan_dir, run_id + ".jsonl")):
                    print(f"  [skip] {run_id}")
                    continue
                print(f"\nHIG: bs={batch_size}, lr={lr}, tau={tau}")
                try:
                    model = instantiate(cfg.model)
                    model.load_state_dict(init_state)
                    train_model = HIGWrapper(model, loss_fn_hig, device)
                    optimizer = HIGOptimizer(train_model, lr=lr, tau=tau)
                    train_loader = DataLoader(
                        dataset.train_dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(loader_seed),
                    )
                    val_loader = DataLoader(dataset.val_dataset, batch_size=batch_size, shuffle=False)
                    train_model, losses = train_loop_hig(
                        train_model, optimizer, loss_fn_hig,
                        train_loader, val_loader, rcfg["num_epochs"], device,
                        track_acc=track_acc, track_param_norm=track_param_norm,
                    )
                    result = {
                        "run_id": run_id,
                        "optimizer": "HIG",
                        "loss": loss_key,
                        "batch_size": batch_size,
                        "lr": lr,
                        "tau": tau,
                        "model_seed": model_seed,
                        "loader_seed": loader_seed,
                        "svd_info": None,
                        "losses": losses,
                    }
                    for f in rcfg.get("result_id_fields", []):
                        result[f] = rcfg[f]
                    _write_run(scan_dir, run_id, result, svd_info_mode, svd_spectra_every)
                except Exception as e:
                    print(f"  [error] HIG run failed: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    print(f"\nScan complete. Results in {scan_dir}/")
