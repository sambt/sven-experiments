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
    train_loop_svd, train_loop_standard, set_seed,
    process_hparam_config, build_standard_optimizer,
)
from sv3.sven import Sven
from sv3.nn import SvenWrapper


# ---------------------------------------------------------------------------
# Loss function registries
# ---------------------------------------------------------------------------
# SVD loss must return per-sample losses (reduction='none')
SVD_LOSS_FNS = {
    "ce": lambda pred, y: F.cross_entropy(pred, y, reduction='none'),
    "mse": lambda pred, y: ((pred - y) ** 2).sum(dim=-1),
    "label_regression": lambda pred, y: (pred - F.one_hot(y.to(torch.long),num_classes=pred.shape[-1]).to(pred)).pow(2).sum(dim=1)
}

# Standard loss returns a scalar
STANDARD_LOSS_FNS = {
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "label_regression": lambda pred, y: (pred - F.one_hot(y.to(torch.long),num_classes=pred.shape[-1]).to(pred)).pow(2).sum(dim=1).mean()
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


def _load_existing_run_ids(jsonl_path):
    """Load set of run_id strings from an existing JSONL file."""
    run_ids = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if 'run_id' in row:
                        run_ids.add(row['run_id'])
    return run_ids


def _append_result(jsonl_path, result):
    """Append a single result dict as a JSON line."""
    serializable = _to_json_serializable(result)
    with open(jsonl_path, 'a') as f:
        f.write(json.dumps(serializable) + '\n')


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
      - mode: "svd", "standard", or "both" (default: "both")
      - loss: key into SVD_LOSS_FNS / STANDARD_LOSS_FNS (e.g. "ce", "mse")
      - result_id_fields: list of config keys to include in output filenames
      - seeds: list of model seeds to sweep over (optional; falls back to model_seed)
      - All hparams consumed by process_hparam_config()
    """
    rcfg = OmegaConf.to_container(cfg, resolve=True)
    device = rcfg["device"]

    mode = rcfg.get("mode", "both")
    assert mode in ("svd", "standard", "both"), f"Unknown mode: {mode}"

    loss_key = rcfg.get("loss", "ce")
    track_acc = loss_key == "ce" or ("label_regression" in loss_key) # only track accuracy for classification
    track_param_norm = rcfg.get("track_param_norm", False)

    # Derive scan name from the Hydra config name (e.g. "mnist_scan")
    scan_name = HydraConfig.get().job.config_name

    # Output path: single JSONL file per scan
    output_dir = "experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = f"{output_dir}/{scan_name}.jsonl"

    # Load existing run IDs for deduplication
    existing_run_ids = _load_existing_run_ids(jsonl_path)
    if existing_run_ids:
        print(f"Found {len(existing_run_ids)} existing runs in {jsonl_path}")

    # Parse hparam grid
    hparams = process_hparam_config(rcfg)
    id_str = _build_id_string(rcfg)

    # Seed list: use 'seeds' if provided, otherwise single 'model_seed'
    seeds = rcfg.get("seeds", [rcfg["model_seed"]])
    loader_seed = rcfg["loader_seed"]

    # Dataset (shared across seeds — same data, different model inits)
    dataset = instantiate(cfg.dataset)

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
        if mode in ("svd", "both"):
            print(f"\n{'='*80}")
            print("Running SVD optimizer scan")
            print(f"{'='*80}")

            k_scan_values = hparams.get('k_fractions', hparams.get('k_values'))
            use_k_values = 'k_values' in hparams

            svd_grid = product(
                hparams['batch_size'],
                k_scan_values,
                hparams['lrs'],
                hparams['rtol'],
                hparams['svd_mode'],
                hparams['microbatch_sizes'],
                hparams['param_fractions'],
            )

            loss_fn_svd = SVD_LOSS_FNS[loss_key]
            use_rmsprop = rcfg.get("use_rmsprop", False)
            alpha_rmsprop = rcfg.get("alpha_rmsProp", 0.99)
            variable_k = rcfg.get("variable_k", False)

            for batch_size, k_item, lr, rtol, svd_mode, microbatch_size, param_fraction in svd_grid:
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
                if use_rmsprop:
                    run_id += f"_RMSpropAlpha{alpha_rmsprop}"
                if variable_k:
                    run_id += "_variablek"

                if run_id in existing_run_ids:
                    print(f"  [skip] {run_id}")
                    continue

                print(f"\nSVD: bs={batch_size}, k={k}, lr={lr}, rtol={rtol}, svd_mode={svd_mode}", end="")
                if microbatch_size is not None:
                    print(f", mb={microbatch_size}", end="")
                if param_fraction is not None:
                    print(f", pf={param_fraction}", end="")
                if use_rmsprop:
                    print(f", rmsprop_alpha={alpha_rmsprop}", end="")
                if variable_k:
                    print(f", variable_k=True", end="")
                print()

                try:
                    model = instantiate(cfg.model)
                    model.load_state_dict(init_state)

                    mb = microbatch_size if microbatch_size is not None else 1
                    pf = param_fraction if param_fraction is not None else 1.0
                    train_model = SvenWrapper(model, loss_fn_svd, device, microbatch_size=mb, param_fraction=pf)
                    optimizer = Sven(
                        train_model, lr=lr, k=k, rtol=rtol,
                        track_svd_info=True, svd_mode=svd_mode,
                        use_rmsprop=use_rmsprop, alpha_rmsprop=alpha_rmsprop,
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
                        track_param_norm=track_param_norm,
                    )

                    result = {
                        "run_id": run_id,
                        "optimizer": "SVD",
                        "batch_size": batch_size,
                        "k_fraction": k / batch_size,
                        "k": k,
                        "lr": lr,
                        "rtol": rtol,
                        "model_seed": model_seed,
                        "loader_seed": loader_seed,
                        "svd_mode": svd_mode,
                        "rmsProp": use_rmsprop,
                        "alpha_rmsProp": alpha_rmsprop,
                        "microbatch_size": microbatch_size,
                        "param_fraction": param_fraction,
                        "variable_k": variable_k,
                        "losses": losses,
                        "svd_info": getattr(optimizer, "svd_info", {})
                    }
                    for f in rcfg.get("result_id_fields", []):
                        result[f] = rcfg[f]

                    _append_result(jsonl_path, result)
                    existing_run_ids.add(run_id)

                except Exception as e:
                    print(f"  [error] Training failed: {e}")

                torch.compiler.reset()

        # --------------------------------------------------------------
        # Standard optimizer scan
        # --------------------------------------------------------------
        if mode in ("standard", "both"):
            print(f"\n{'='*80}")
            print("Running standard optimizer scan")
            print(f"{'='*80}")

            # Build the grid — for LBFGS we also sweep over its specific params
            has_lbfgs = "LBFGS" in hparams['optimizers_standard']
            non_lbfgs_optimizers = [o for o in hparams['optimizers_standard'] if o != "LBFGS"]

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
                    # Non-zero weight decay is only meaningful for AdamW
                    if optim_name != "AdamW" and weight_decay != 0.0:
                        continue

                    run_id = f"std_bs{batch_size}{id_str}_lr{lr}_optim{optim_name}"
                    if weight_decay != 0.0:
                        run_id += f"_wd{weight_decay}"
                    run_id += seed_str

                    if run_id in existing_run_ids:
                        print(f"  [skip] {run_id}")
                        continue

                    wd_str = f", wd={weight_decay}" if weight_decay != 0.0 else ""
                    print(f"\nStandard: bs={batch_size}, lr={lr}, optim={optim_name}{wd_str}")

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
                        track_param_norm=track_param_norm,
                    )

                    result = {
                        "run_id": run_id,
                        "optimizer": optim_name,
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

                    _append_result(jsonl_path, result)
                    existing_run_ids.add(run_id)

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
                    )

                    if run_id in existing_run_ids:
                        print(f"  [skip] {run_id}")
                        continue

                    print(f"\nLBFGS: bs={batch_size}, lr={lr}, max_iter={max_iter}, "
                          f"history_size={history_size}, line_search={line_search_fn}")

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
                    )

                    result = {
                        "run_id": run_id,
                        "optimizer": "LBFGS",
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

                    _append_result(jsonl_path, result)
                    existing_run_ids.add(run_id)

    print(f"\nScan complete. Results in {jsonl_path}")
