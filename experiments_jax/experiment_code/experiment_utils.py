"""Shared utilities for JAX Sven experiments.

Mirrors ``experiments/experiment_code/experiment_utils.py`` but everything is
JAX / Flax / optax native. Datasets live fully on-device; batches are drawn
with a simple host-side permutation.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from sven.jax import Sven, SvenWrapper


# ---------------------------------------------------------------------------
# RNG / misc
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def listify(settings):
    if isinstance(settings, (list, tuple)):
        return settings
    return [settings]


# ---------------------------------------------------------------------------
# Hyperparameter config processing — same shape as the PyTorch version so
# existing YAML files translate one-for-one.
# ---------------------------------------------------------------------------

def process_hparam_config(cfg) -> dict[str, Iterable]:
    output: dict[str, Any] = {}
    output["batch_size"] = listify(cfg.get("batch_size", 32))

    if "k_fractions" in cfg and "k_values" in cfg:
        raise ValueError("Specify either k_values or k_fractions, not both.")
    if "k_values" in cfg:
        output["k_values"] = listify(cfg["k_values"])
    elif "k_fractions" in cfg:
        output["k_fractions"] = listify(cfg["k_fractions"])
    else:
        output["k_fractions"] = [0.1, 0.25, 0.5, 0.75, 1.0]

    output["lrs"] = listify(cfg.get("lrs", [0.01, 0.1, 0.5, 1.0]))
    output["rtol"] = listify(cfg.get("rtol", 1e-3))
    # JAX port ships ``randomized`` and ``full`` only. Map legacy names on read.
    output["svd_mode"] = listify(cfg.get("svd_mode", "randomized"))

    output["lrs_standard"] = listify(cfg.get("lrs_standard", [1e-4, 1e-3, 1e-2, 1e-1]))
    output["optimizers_standard"] = listify(
        cfg.get("optimizers_standard", ["Adam", "AdamW", "SGD", "RMSprop"])
    )

    output["microbatch_sizes"] = listify(cfg.get("microbatch_sizes", [None]))
    output["param_fractions"] = listify(cfg.get("param_fractions", [None]))
    output["weight_decays"] = listify(cfg.get("weight_decays", [0.0]))

    # LBFGS / PolyakSGD: parsed for compat, but skipped in the scan loop.
    output["lrs_lbfgs"] = listify(cfg.get("lrs_lbfgs", output["lrs_standard"]))
    output["lbfgs_max_iter"] = listify(cfg.get("lbfgs_max_iter", 20))
    output["lbfgs_history_size"] = listify(cfg.get("lbfgs_history_size", 100))
    output["lbfgs_line_search_fn"] = listify(cfg.get("lbfgs_line_search_fn", "strong_wolfe"))
    output["polyak_f_star"] = listify(cfg.get("polyak_f_star", 0.0))
    output["polyak_max_lr"] = listify(cfg.get("polyak_max_lr", 1.0))
    output["polyak_eps"] = listify(cfg.get("polyak_eps", 1e-8))

    return output


# ---------------------------------------------------------------------------
# SVD mode translation: the PyTorch backend name -> JAX backend name.
# ---------------------------------------------------------------------------

_SVD_MODE_MAP = {
    "torch": "full",
    "full": "full",
    "randomized": "randomized",
    "randomized_v2": "randomized",
    "scipy": "full",
    "lobpcg": "full",
}


def translate_svd_mode(mode: str) -> str:
    return _SVD_MODE_MAP.get(mode, mode)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def iter_batches(
    x: jnp.ndarray,
    y: jnp.ndarray,
    batch_size: int,
    rng_np: np.random.Generator | None = None,
    shuffle: bool = True,
    drop_last: bool = False,
):
    """Iterate mini-batches over ``(x, y)``.

    We permute indices with ``numpy`` (fast, no GPU sync) and gather on-device.
    """
    n = x.shape[0]
    if shuffle:
        perm = rng_np.permutation(n)
    else:
        perm = np.arange(n)
    end = n - (n % batch_size) if drop_last else n
    perm = jnp.asarray(perm)
    for i in range(0, end, batch_size):
        idx = perm[i : i + batch_size]
        if idx.shape[0] < batch_size and drop_last:
            break
        yield x[idx], y[idx]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_acc(ypred: jnp.ndarray, yb: jnp.ndarray) -> float:
    if ypred.ndim == 3:
        preds = jnp.argmax(ypred, axis=2)  # (M, B)
        return float((preds == yb[None]).astype(jnp.float32).mean())
    preds = jnp.argmax(ypred, axis=-1)
    return float((preds == yb).astype(jnp.float32).mean())


def _compute_per_model_acc(ypred: jnp.ndarray, yb: jnp.ndarray) -> list[float]:
    preds = jnp.argmax(ypred, axis=2)
    return (preds == yb[None]).astype(jnp.float32).mean(axis=1).tolist()


def _param_norm(flat_params: jnp.ndarray) -> float:
    return float(jnp.linalg.norm(flat_params))


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_loop_svd(
    wrapped: SvenWrapper,
    optimizer: Sven,
    loss_fn: Callable,
    train_data: tuple[jnp.ndarray, jnp.ndarray],
    val_data: tuple[jnp.ndarray, jnp.ndarray],
    batch_size: int,
    num_epochs: int,
    loader_seed: int,
    track_acc: bool = False,
    track_param_norm: bool = False,
    microbatch_size: int | None = None,
) -> tuple[SvenWrapper, dict[str, Any], Sven]:
    losses: dict[str, Any] = defaultdict(list)
    is_multi: bool | None = None
    num_models: int | None = None

    xtr, ytr = train_data
    xv, yv = val_data

    # Untrained validation loss
    val_batches_init = list(iter_batches(xv, yv, batch_size, shuffle=False))
    for xb, yb in val_batches_init:
        ypred = wrapped.evaluate(xb)
        per_sample_loss = loss_fn(ypred, yb)
        losses["val_init"].append(float(per_sample_loss.mean()))
        if track_acc:
            losses["val_init_acc"].append(_compute_acc(ypred, yb))
        if is_multi is None:
            is_multi = ypred.ndim == 3
            if is_multi:
                num_models = int(ypred.shape[0])
                losses["num_models"] = num_models
    losses["val"].append(float(np.mean(losses["val_init"])))
    del losses["val_init"]
    if track_acc:
        losses["val_acc"].append(float(np.mean(losses["val_init_acc"])))
        del losses["val_init_acc"]

    total_start = time.perf_counter()

    # Each epoch gets its own numpy RNG derived from loader_seed + epoch.
    rng_master = np.random.default_rng(loader_seed)

    # Every step needs a fresh JAX key only when param_fraction < 1.
    key = jax.random.PRNGKey(loader_seed ^ 0xA5A5)

    n_train_batches = (xtr.shape[0] // batch_size) if microbatch_size else (xtr.shape[0] // batch_size + (0 if xtr.shape[0] % batch_size == 0 else 1))
    pbar = tqdm(total=max(1, xtr.shape[0] // batch_size), leave=True)
    for epoch in range(num_epochs):
        pbar.reset()
        pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_start = time.perf_counter()
        epoch_losses: dict[str, list] = defaultdict(list)
        epoch_pm: dict[str, list] = defaultdict(list)

        epoch_rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        drop = microbatch_size is not None and microbatch_size > 1
        for xb, yb in iter_batches(xtr, ytr, batch_size, epoch_rng, shuffle=True, drop_last=drop):
            batch_start = time.perf_counter()
            key, sub = jax.random.split(key)
            batch_losses, ypred = wrapped.loss_and_grad((xb, yb), key=sub)
            optimizer.step()
            # Force completion so timing is honest.
            jax.block_until_ready(wrapped.flat_params)
            batch_end = time.perf_counter()
            losses["batch_times_train"].append(batch_end - batch_start)
            pbar.update(1)
            epoch_losses["train"].append(float(batch_losses.mean()))
            if is_multi:
                pm_losses = batch_losses.reshape(num_models, -1).mean(axis=1).tolist()
                epoch_pm["train"].append(pm_losses)
                losses["train_batch_per_model"].append(pm_losses)
            if track_acc:
                epoch_losses["train_acc"].append(_compute_acc(ypred, yb))
                if is_multi:
                    epoch_pm["train_acc"].append(_compute_per_model_acc(ypred, yb))

        for xb, yb in iter_batches(xv, yv, batch_size, shuffle=False):
            batch_start = time.perf_counter()
            ypred = wrapped.evaluate(xb)
            per_sample_loss = loss_fn(ypred, yb)
            loss = float(per_sample_loss.mean())
            batch_end = time.perf_counter()
            losses["batch_times_val"].append(batch_end - batch_start)
            epoch_losses["val"].append(loss)
            if is_multi:
                pm_losses = per_sample_loss.reshape(num_models, -1).mean(axis=1).tolist()
                epoch_pm["val"].append(pm_losses)
            if track_acc:
                epoch_losses["val_acc"].append(_compute_acc(ypred, yb))
                if is_multi:
                    epoch_pm["val_acc"].append(_compute_per_model_acc(ypred, yb))

        epoch_end = time.perf_counter()
        losses["epoch_times"].append(epoch_end - epoch_start)
        losses["train_batch"].extend(epoch_losses["train"])
        losses["val_batch"].extend(epoch_losses["val"])
        for k_name, v in epoch_losses.items():
            losses[k_name].append(float(np.mean(v)))
        for k_name, v in epoch_pm.items():
            losses[f"{k_name}_per_model"].append(np.mean(v, axis=0).tolist())
        if track_param_norm:
            losses["param_norm"].append(_param_norm(wrapped.flat_params))

    pbar.close()
    total_end = time.perf_counter()
    out = dict(losses)
    out["total_time"] = total_end - total_start
    out["avg_epoch_time"] = float(np.mean(out["epoch_times"]))
    out["avg_batch_time_train"] = float(np.mean(out["batch_times_train"]))
    out["avg_batch_time_val"] = float(np.mean(out["batch_times_val"]))

    return wrapped, out, optimizer


def train_loop_standard(
    apply_fn: Callable,
    params: Any,
    optimizer: optax.GradientTransformation,
    loss_fn_scalar: Callable,
    per_sample_loss_fn: Callable,  # used only when is_multi to break out per-model losses
    train_data: tuple[jnp.ndarray, jnp.ndarray],
    val_data: tuple[jnp.ndarray, jnp.ndarray],
    batch_size: int,
    num_epochs: int,
    loader_seed: int,
    track_acc: bool = False,
    track_param_norm: bool = False,
) -> tuple[Any, dict[str, Any]]:
    losses: dict[str, Any] = defaultdict(list)
    is_multi: bool | None = None
    num_models: int | None = None

    opt_state = optimizer.init(params)

    @jax.jit
    def step_fn(params, opt_state, xb, yb):
        def compute(p):
            pred = apply_fn(p, xb)
            return loss_fn_scalar(pred, yb), pred
        (loss, pred), grads = jax.value_and_grad(compute, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, pred

    @jax.jit
    def eval_fn(params, xb):
        return apply_fn(params, xb)

    xtr, ytr = train_data
    xv, yv = val_data

    # Untrained validation loss
    for xb, yb in iter_batches(xv, yv, batch_size, shuffle=False):
        ypred = eval_fn(params, xb)
        losses["val_init"].append(float(loss_fn_scalar(ypred, yb)))
        if track_acc:
            losses["val_init_acc"].append(_compute_acc(ypred, yb))
        if is_multi is None:
            is_multi = ypred.ndim == 3
            if is_multi:
                num_models = int(ypred.shape[0])
                losses["num_models"] = num_models
    losses["val"].append(float(np.mean(losses["val_init"])))
    del losses["val_init"]
    if track_acc:
        losses["val_acc"].append(float(np.mean(losses["val_init_acc"])))
        del losses["val_init_acc"]

    total_start = time.perf_counter()
    rng_master = np.random.default_rng(loader_seed)

    pbar = tqdm(total=max(1, xtr.shape[0] // batch_size), leave=True)
    for epoch in range(num_epochs):
        pbar.reset()
        pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_start = time.perf_counter()
        epoch_losses: dict[str, list] = defaultdict(list)
        epoch_pm: dict[str, list] = defaultdict(list)

        epoch_rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        for xb, yb in iter_batches(xtr, ytr, batch_size, epoch_rng):
            batch_start = time.perf_counter()
            params, opt_state, loss, ypred = step_fn(params, opt_state, xb, yb)
            jax.block_until_ready(loss)
            batch_end = time.perf_counter()
            losses["batch_times_train"].append(batch_end - batch_start)
            pbar.update(1)
            epoch_losses["train"].append(float(loss))
            if is_multi:
                per_sample = per_sample_loss_fn(ypred, yb)
                pm_losses = per_sample.reshape(num_models, -1).mean(axis=1).tolist()
                epoch_pm["train"].append(pm_losses)
                losses["train_batch_per_model"].append(pm_losses)
            if track_acc:
                epoch_losses["train_acc"].append(_compute_acc(ypred, yb))
                if is_multi:
                    epoch_pm["train_acc"].append(_compute_per_model_acc(ypred, yb))

        for xb, yb in iter_batches(xv, yv, batch_size, shuffle=False):
            batch_start = time.perf_counter()
            ypred = eval_fn(params, xb)
            loss = float(loss_fn_scalar(ypred, yb))
            batch_end = time.perf_counter()
            losses["batch_times_val"].append(batch_end - batch_start)
            epoch_losses["val"].append(loss)
            if is_multi:
                per_sample = per_sample_loss_fn(ypred, yb)
                pm_losses = per_sample.reshape(num_models, -1).mean(axis=1).tolist()
                epoch_pm["val"].append(pm_losses)
            if track_acc:
                epoch_losses["val_acc"].append(_compute_acc(ypred, yb))
                if is_multi:
                    epoch_pm["val_acc"].append(_compute_per_model_acc(ypred, yb))

        epoch_end = time.perf_counter()
        losses["epoch_times"].append(epoch_end - epoch_start)
        losses["train_batch"].extend(epoch_losses["train"])
        losses["val_batch"].extend(epoch_losses["val"])
        for k_name, v in epoch_losses.items():
            losses[k_name].append(float(np.mean(v)))
        for k_name, v in epoch_pm.items():
            losses[f"{k_name}_per_model"].append(np.mean(v, axis=0).tolist())
        if track_param_norm:
            flat = jnp.concatenate([p.reshape(-1) for p in jax.tree_util.tree_leaves(params)])
            losses["param_norm"].append(float(jnp.linalg.norm(flat)))

    pbar.close()
    total_end = time.perf_counter()
    out = dict(losses)
    out["total_time"] = total_end - total_start
    out["avg_epoch_time"] = float(np.mean(out["epoch_times"]))
    out["avg_batch_time_train"] = float(np.mean(out["batch_times_train"]))
    out["avg_batch_time_val"] = float(np.mean(out["batch_times_val"]))

    return params, out
