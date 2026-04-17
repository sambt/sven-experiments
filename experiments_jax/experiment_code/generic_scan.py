"""Unified hyperparameter scan for JAX Sven + optax baselines.

Mirrors ``experiments.experiment_code.generic_scan.scan``: same YAML shape,
same JSONL output layout, same run_id deduplication. Each `(hparam-tuple,
seed)` combination becomes its own JSONL file inside
``experiment_results/<scan_name>/``.
"""

from __future__ import annotations

import json
import os
from itertools import product
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sven.jax import Sven, SvenWrapper

from .experiment_utils import (
    process_hparam_config,
    set_seed,
    train_loop_standard,
    train_loop_svd,
    translate_svd_mode,
)


# ---------------------------------------------------------------------------
# Loss registries
# ---------------------------------------------------------------------------

def _ce_per_sample(pred, y):
    return optax.softmax_cross_entropy_with_integer_labels(pred, y.astype(jnp.int32))


def _mse_per_sample(pred, y):
    return jnp.sum((pred - y) ** 2, axis=-1)


def _label_regression_per_sample(pred, y):
    onehot = jax.nn.one_hot(y.astype(jnp.int32), pred.shape[-1])
    return jnp.sum((pred - onehot) ** 2, axis=-1)


# Per-sample losses for SvenWrapper; shape (B,)
SVD_LOSS_FNS = {
    "ce": _ce_per_sample,
    "mse": _mse_per_sample,
    "label_regression": _label_regression_per_sample,
}

# Scalar losses for optax baselines
STANDARD_LOSS_FNS = {
    "ce": lambda pred, y: _ce_per_sample(pred, y).mean(),
    "mse": lambda pred, y: _mse_per_sample(pred, y).mean(),
    "label_regression": lambda pred, y: _label_regression_per_sample(pred, y).mean(),
}


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _to_json_serializable(obj):
    if isinstance(obj, (jax.Array, jnp.ndarray)):
        return np.asarray(obj).tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    return obj


def _write_result(jsonl_path: str, result: dict[str, Any]) -> None:
    with open(jsonl_path, "w") as f:
        f.write(json.dumps(_to_json_serializable(result)) + "\n")


def _build_id_string(cfg) -> str:
    fields = cfg.get("result_id_fields", [])
    if not fields:
        return ""
    return "_" + "_".join(f"{f}{cfg[f]}" for f in fields)


# ---------------------------------------------------------------------------
# Model / dataset helpers
# ---------------------------------------------------------------------------

def _init_model(model, rng_key, dummy_x):
    """Return ``(apply_fn, params)``.

    Accepts Flax Linen modules. The params pytree is whatever ``model.init``
    returns; Sven's ``ravel_pytree`` handles arbitrary structure. Block-level
    masking walks leaves in pytree-leaf order.
    """
    variables = model.init(rng_key, dummy_x)
    # Flax returns {'params': ...}; flatten that to a cleaner pytree.
    params = variables
    apply_fn = model.apply
    return apply_fn, params


def _dummy_input(dataset):
    x, _ = dataset.train
    return x[:1]


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def scan(cfg):
    rcfg = OmegaConf.to_container(cfg, resolve=True)

    mode = rcfg.get("mode", "both")
    assert mode in ("svd", "standard", "both"), f"Unknown mode: {mode}"

    loss_key = rcfg.get("loss", "ce")
    track_acc = loss_key == "ce" or ("label_regression" in loss_key)
    track_param_norm = rcfg.get("track_param_norm", False)

    scan_name = HydraConfig.get().job.config_name
    output_dir = "experiment_results_jax"
    scan_dir = os.path.join(output_dir, scan_name)
    os.makedirs(scan_dir, exist_ok=True)

    hparams = process_hparam_config(rcfg)
    id_str = _build_id_string(rcfg)

    seeds = rcfg.get("model_seeds")
    loader_seed = rcfg["loader_seed"]

    # Instantiate the dataset once — shared across seeds.
    dataset = instantiate(cfg.dataset)
    dummy_x = _dummy_input(dataset)

    for model_seed in seeds:
        print(f"\n{'#'*80}\n# Model seed: {model_seed}\n{'#'*80}")
        seed_str = f"_mseed{model_seed}_lseed{loader_seed}"

        set_seed(model_seed)
        model = instantiate(cfg.model)
        apply_fn_all, init_params = _init_model(model, jax.random.PRNGKey(int(model_seed)), dummy_x)

        # ------------------------------------------------------------------
        # SVD optimizer scan
        # ------------------------------------------------------------------
        if mode in ("svd", "both"):
            print(f"\n{'='*80}\nRunning SVD optimizer scan\n{'='*80}")

            k_scan_values = hparams.get("k_fractions", hparams.get("k_values"))
            use_k_values = "k_values" in hparams

            svd_grid = product(
                hparams["batch_size"],
                k_scan_values,
                hparams["lrs"],
                hparams["rtol"],
                hparams["svd_mode"],
                hparams["microbatch_sizes"],
                hparams["param_fractions"],
            )

            loss_fn_svd = SVD_LOSS_FNS[loss_key]
            use_rmsprop = rcfg.get("use_rmsprop", False)
            alpha_rmsprop = rcfg.get("alpha_rmsProp", 0.99)
            variable_k = rcfg.get("variable_k", False)
            if variable_k:
                print("  [warn] variable_k is not supported in the JAX port (ignored).")

            for batch_size, k_item, lr, rtol, svd_mode, microbatch_size, param_fraction in svd_grid:
                k = max(1, int(k_item * batch_size)) if not use_k_values else k_item

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

                out_path = os.path.join(scan_dir, run_id + ".jsonl")
                if os.path.exists(out_path):
                    print(f"  [skip] {run_id}")
                    continue

                print(f"\nSVD: bs={batch_size}, k={k}, lr={lr}, rtol={rtol}, "
                      f"svd_mode={svd_mode}", end="")
                if microbatch_size is not None:
                    print(f", mb={microbatch_size}", end="")
                if param_fraction is not None:
                    print(f", pf={param_fraction}", end="")
                if use_rmsprop:
                    print(f", rmsprop_alpha={alpha_rmsprop}", end="")
                print()

                try:
                    mb = microbatch_size if microbatch_size is not None else 1
                    pf = param_fraction if param_fraction is not None else 1.0

                    wrapped = SvenWrapper(
                        apply_fn_all, init_params, loss_fn_svd,
                        param_fraction=pf,
                        microbatch_size=mb,
                    )
                    jax_mode = translate_svd_mode(svd_mode)
                    optimizer = Sven(
                        wrapped,
                        lr=lr, k=k, rtol=rtol,
                        svd_mode=jax_mode,
                        use_rmsprop=use_rmsprop,
                        alpha_rmsprop=alpha_rmsprop,
                        track_svd_info=True,
                        seed=int(model_seed) ^ int(loader_seed),
                    )

                    wrapped, losses, optimizer = train_loop_svd(
                        wrapped, optimizer, loss_fn_svd,
                        dataset.train, dataset.val,
                        batch_size, rcfg["num_epochs"], loader_seed,
                        track_acc=track_acc, track_param_norm=track_param_norm,
                        microbatch_size=microbatch_size,
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
                        "variable_k": False,
                        "losses": losses,
                        "svd_info": getattr(optimizer, "svd_info", {}),
                    }
                    for f in rcfg.get("result_id_fields", []):
                        result[f] = rcfg[f]

                    _write_result(out_path, result)

                except Exception as e:
                    print(f"  [error] Training failed: {e!r}")

        # ------------------------------------------------------------------
        # Standard optimizer scan
        # ------------------------------------------------------------------
        if mode in ("standard", "both"):
            print(f"\n{'='*80}\nRunning standard optimizer scan\n{'='*80}")

            from experiments_jax.optimizers import build_standard_optimizer

            loss_fn_scalar = STANDARD_LOSS_FNS[loss_key]
            per_sample_fn = SVD_LOSS_FNS[loss_key]

            # Filter out optimizers we don't support (LBFGS / PolyakSGD / Muon)
            all_opts = hparams["optimizers_standard"]
            skipped = [o for o in all_opts if o in {"LBFGS", "PolyakSGD", "Muon"}]
            supported_opts = [o for o in all_opts if o not in {"LBFGS", "PolyakSGD", "Muon"}]
            if skipped:
                print(f"  [info] Skipping unsupported optimizers in JAX port: {skipped}")

            standard_grid = product(
                hparams["batch_size"],
                hparams["lrs_standard"],
                supported_opts,
                hparams["weight_decays"],
            )

            for batch_size, lr, optim_name, weight_decay in standard_grid:
                if optim_name not in ("AdamW",) and weight_decay != 0.0:
                    continue

                run_id = f"std_bs{batch_size}{id_str}_lr{lr}_optim{optim_name}"
                if weight_decay != 0.0:
                    run_id += f"_wd{weight_decay}"
                run_id += seed_str

                out_path = os.path.join(scan_dir, run_id + ".jsonl")
                if os.path.exists(out_path):
                    print(f"  [skip] {run_id}")
                    continue

                wd_str = f", wd={weight_decay}" if weight_decay != 0.0 else ""
                print(f"\nStandard: bs={batch_size}, lr={lr}, optim={optim_name}{wd_str}")

                try:
                    optimizer = build_standard_optimizer(
                        optim_name, lr=lr, weight_decay=weight_decay
                    )
                    params = jax.tree_util.tree_map(lambda x: x, init_params)

                    params, losses = train_loop_standard(
                        apply_fn_all, params, optimizer,
                        loss_fn_scalar, per_sample_fn,
                        dataset.train, dataset.val,
                        batch_size, rcfg["num_epochs"], loader_seed,
                        track_acc=track_acc, track_param_norm=track_param_norm,
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

                    _write_result(out_path, result)

                except Exception as e:
                    print(f"  [error] Training failed: {e!r}")

    print(f"\nScan complete. Results in {scan_dir}/")
