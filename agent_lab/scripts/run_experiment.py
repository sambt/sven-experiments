#!/usr/bin/env python3
"""
Standalone experiment runner for agent_lab.

This provides a simpler alternative to the Hydra-based experiment framework
for quick, targeted experiments. Agents should modify or copy this script.

Usage:
    python agent_lab/scripts/run_experiment.py
"""

import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sv3.nn import MLP, SvenWrapper
from sv3.sven import Sven
from experiments.datasets import MNISTDataset
from experiments.experiment_code.experiment_utils import (
    set_seed, train_loop_svd, train_loop_standard, build_standard_optimizer,
)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
RESULTS_DIR = REPO_ROOT / "agent_lab" / "results"
PLOTS_DIR = REPO_ROOT / "agent_lab" / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Loss functions (from experiments/experiment_code/generic_scan.py)
# ---------------------------------------------------------------------------
# SVD losses must return per-sample losses (no reduction)
SVD_LOSS_FNS = {
    "ce": lambda pred, y: F.cross_entropy(pred, y, reduction='none'),
    "mse": lambda pred, y: ((pred - y) ** 2).sum(dim=-1),
    "label_regression": lambda pred, y: (
        pred - F.one_hot(y.to(torch.long), num_classes=pred.shape[-1]).to(pred)
    ).pow(2).sum(dim=1),
}

# Standard losses return a scalar
STANDARD_LOSS_FNS = {
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "label_regression": lambda pred, y: (
        pred - F.one_hot(y.to(torch.long), num_classes=pred.shape[-1]).to(pred)
    ).pow(2).sum(dim=1).mean(),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def to_serializable(obj):
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
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def save_result(result: dict, name: str) -> Path:
    """Save a result dict as a JSONL file."""
    serializable = to_serializable(result)
    path = RESULTS_DIR / f"{name}.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps(serializable) + "\n")
    print(f"Saved: {path}")
    return path


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_mlp(input_dim=784, output_dim=10, width=32, depth=3, activation=nn.GELU):
    """Create an MLP using sv3's MLP class."""
    hidden_dims = [width] * depth
    return MLP(input_dim, hidden_dims, output_dim, activation=activation)


def get_mnist(digits=None):
    """Load MNIST using the project's MNISTDataset class."""
    return MNISTDataset(digits=digits)


# ---------------------------------------------------------------------------
# Single-run helpers
# ---------------------------------------------------------------------------

def run_sven(
    model, dataset, loss_key="ce", device="cpu",
    lr=1.0, k=32, rtol=1e-3, batch_size=64,
    num_epochs=10, seed=42, loader_seed=51159,
    use_rmsprop=False, alpha_rmsprop=0.99,
    track_svd_info=True, svd_mode="torch",
    microbatch_size=1, param_fraction=1.0,
    variable_k=False,
):
    """Run a single Sven training run. Returns (result_dict, model, optimizer)."""
    loss_fn = SVD_LOSS_FNS[loss_key]
    track_acc = loss_key in ("ce", "label_regression")

    train_model = SvenWrapper(
        model, loss_fn, device,
        microbatch_size=microbatch_size,
        param_fraction=param_fraction,
    )
    optimizer = Sven(
        train_model, lr=lr, k=k, rtol=rtol,
        track_svd_info=track_svd_info, svd_mode=svd_mode,
        use_rmsprop=use_rmsprop, alpha_rmsprop=alpha_rmsprop,
        variable_k=variable_k,
    )

    train_loader = DataLoader(
        dataset.train_dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(loader_seed),
        drop_last=(microbatch_size > 1),
    )
    val_loader = DataLoader(dataset.val_dataset, batch_size=batch_size, shuffle=False)

    train_model, losses, optimizer = train_loop_svd(
        train_model, optimizer, loss_fn,
        train_loader, val_loader,
        num_epochs, device, track_acc=track_acc,
    )

    result = {
        "optimizer": "Sven",
        "loss_fn": loss_key,
        "batch_size": batch_size,
        "k": k,
        "lr": lr,
        "rtol": rtol,
        "svd_mode": svd_mode,
        "use_rmsprop": use_rmsprop,
        "alpha_rmsprop": alpha_rmsprop if use_rmsprop else None,
        "microbatch_size": microbatch_size,
        "param_fraction": param_fraction,
        "variable_k": variable_k,
        "model_seed": seed,
        "loader_seed": loader_seed,
        "num_epochs": num_epochs,
        "losses": losses,
        "svd_info": getattr(optimizer, "svd_info", {}),
    }
    return result, train_model, optimizer


def run_baseline(
    model, dataset, loss_key="ce", device="cpu",
    optim_name="Adam", lr=1e-3, batch_size=64,
    num_epochs=10, seed=42, loader_seed=51159,
    **optim_kwargs,
):
    """Run a single baseline optimizer training run. Returns (result_dict, model)."""
    loss_fn = STANDARD_LOSS_FNS[loss_key]
    track_acc = loss_key in ("ce", "label_regression")

    model = model.to(device)
    optimizer = build_standard_optimizer(model, optim_name, lr, **optim_kwargs)

    train_loader = DataLoader(
        dataset.train_dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(loader_seed),
    )
    val_loader = DataLoader(dataset.val_dataset, batch_size=batch_size, shuffle=False)

    model, losses = train_loop_standard(
        model, optimizer, loss_fn,
        train_loader, val_loader,
        num_epochs, device, track_acc=track_acc,
    )

    result = {
        "optimizer": optim_name,
        "loss_fn": loss_key,
        "batch_size": batch_size,
        "lr": lr,
        "model_seed": seed,
        "loader_seed": loader_seed,
        "num_epochs": num_epochs,
        "losses": losses,
        **{k: v for k, v in optim_kwargs.items()},
    }
    return result, model


# ---------------------------------------------------------------------------
# Example experiment
# ---------------------------------------------------------------------------

def main():
    """
    Example: compare Sven (CE) vs Sven (label regression) vs Adam (CE) on MNIST.
    Agents should modify this or write new scripts for specific experiments.
    """
    device = get_device()
    seed = 42
    width = 32
    batch_size = 64
    num_epochs = 10
    k = 32
    lr_sven = 1.0
    lr_adam = 1e-3

    print(f"Device: {device}")
    print(f"MLP width: {width}, batch_size: {batch_size}, epochs: {num_epochs}")

    dataset = get_mnist()

    # --- Sven with cross-entropy ---
    set_seed(seed)
    model_ce = make_mlp(width=width)
    init_state = copy.deepcopy(model_ce.state_dict())

    result_sven_ce, _, _ = run_sven(
        model_ce, dataset, loss_key="ce", device=device,
        lr=lr_sven, k=k, num_epochs=num_epochs, seed=seed,
    )
    result_sven_ce["run_id"] = "sven_ce"
    save_result(result_sven_ce, f"sven_ce_{timestamp()}")

    # --- Sven with label regression ---
    set_seed(seed)
    model_lr = make_mlp(width=width)
    model_lr.load_state_dict(init_state)

    result_sven_lr, _, _ = run_sven(
        model_lr, dataset, loss_key="label_regression", device=device,
        lr=lr_sven, k=k, num_epochs=num_epochs, seed=seed,
    )
    result_sven_lr["run_id"] = "sven_label_regression"
    save_result(result_sven_lr, f"sven_label_regression_{timestamp()}")

    # --- Adam baseline with cross-entropy ---
    set_seed(seed)
    model_adam = make_mlp(width=width)
    model_adam.load_state_dict(init_state)

    result_adam, _ = run_baseline(
        model_adam, dataset, loss_key="ce", device=device,
        optim_name="Adam", lr=lr_adam, num_epochs=num_epochs, seed=seed,
    )
    result_adam["run_id"] = "adam_ce"
    save_result(result_adam, f"adam_ce_{timestamp()}")

    print("\nDone! Results saved to agent_lab/results/")


if __name__ == "__main__":
    main()
