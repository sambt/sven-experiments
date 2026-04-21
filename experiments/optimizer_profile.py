#!/usr/bin/env python3
"""
GPU memory + step-time profiler for SV3 optimizers (SVD and standard).

For a single config, reports:
  • Baseline GPU memory  — model only, before optimizer construction.
  • Resident GPU memory  — model + optimizer state + persistent tensors,
                            measured at the start of each profiled step.
  • Peak GPU memory      — max_memory_allocated during a single step.
                            (activations + grads + Jacobian + SVD factors
                            + optimizer scratch, whichever applies).
  • Wall-clock step time — CUDA-event timed, end-to-end (host->device
                            transfer + forward + backward/jac + step).

Usage:
  python -m experiments.optimizer_profile --config-name cifar10_resnet_ce_test

Profile-specific config keys (nest under `profile:` in the YAML):
  mode            : "svd" (default) or "standard"
  optimizer_name  : standard-mode optimizer, default "Adam"
  lr              : LR override (falls back to lrs[0] from config)
  num_steps       : measured steps (default 5)
  warmup_steps    : warm-up steps before measurement (default 2)
  detailed        : SVD-only fine-grained phase breakdown (default False)
  output_json     : JSON artifact path, default "optimizer_profile.json"
                    (relative paths land in the Hydra run directory)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from experiments.experiment_code.experiment_utils import (
    build_standard_optimizer, set_seed,
)
from sven.nn import SvenWrapper
from sven.opt import Sven
from sven.opt.pinv import pinv


# ---------------------------------------------------------------------------
# Loss registries (mirror generic_scan.py)
# ---------------------------------------------------------------------------

SVD_LOSS_FNS = {
    "ce": lambda pred, y: F.cross_entropy(pred, y, reduction="none"),
    "mse": lambda pred, y: ((pred - y) ** 2).sum(dim=-1),
    "label_regression": lambda pred, y: (
        pred - F.one_hot(y.to(torch.long), num_classes=pred.shape[-1]).to(pred)
    ).pow(2).sum(dim=1),
}

STANDARD_LOSS_FNS = {
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "label_regression": lambda pred, y: (
        pred - F.one_hot(y.to(torch.long), num_classes=pred.shape[-1]).to(pred)
    ).pow(2).sum(dim=1).mean(),
}


def _first(v):
    return v[0] if isinstance(v, (list, tuple)) else v


def _mb(n_bytes: float) -> str:
    return f"{n_bytes / 1024**2:.1f} MB"


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def _svd_step(train_model, optimizer, batch):
    train_model.loss_and_grad(batch)
    optimizer.step(batch)


def _standard_step(model, optimizer, loss_fn, batch):
    xb, yb = batch
    optimizer.zero_grad(set_to_none=True)
    pred = model(xb)
    loss = loss_fn(pred, yb)
    loss.backward()
    optimizer.step()


# ---------------------------------------------------------------------------
# Optional SVD-only phase breakdown
# ---------------------------------------------------------------------------

class _PhaseTracker:
    def __init__(self, device):
        self.device = device
        self.records: list[tuple[str, int, int]] = []

    def reset(self):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.records.clear()

    def checkpoint(self, label: str):
        torch.cuda.synchronize(self.device)
        cur = torch.cuda.memory_allocated(self.device)
        peak = torch.cuda.max_memory_allocated(self.device)
        self.records.append((label, cur, peak))
        torch.cuda.reset_peak_memory_stats(self.device)


def _detailed_svd_step(train_model, optimizer, batch, tracker: _PhaseTracker):
    xb, yb = batch
    tracker.checkpoint("0. Baseline (model + state on GPU)")
    tracker.checkpoint("1. Batch on GPU")
    train_model.loss_and_grad((xb, yb))
    tracker.checkpoint("2. After loss_and_grad (Jacobian built)")
    jacobian = train_model.grads
    VhT, S_inv, U_T = pinv(
        jacobian, k=optimizer.k, rtol=optimizer.rtol, mode=optimizer.svd_mode
    )
    tracker.checkpoint("3. After pinv (SVD computed)")
    del jacobian
    torch.cuda.empty_cache()
    tracker.checkpoint("4. After del Jacobian + empty_cache")
    residuals = train_model.residuals
    optimizer._update_params(U_T, S_inv, VhT, residuals)
    tracker.checkpoint("5. After parameter update")
    del VhT, S_inv, U_T
    torch.cuda.empty_cache()
    tracker.checkpoint("6. After cleanup")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="configs", version_base=None)
def main(cfg: DictConfig) -> None:
    rcfg = OmegaConf.to_container(cfg, resolve=True)
    device = rcfg.get("device", "cuda")
    if not torch.cuda.is_available():
        print("CUDA not available — cannot profile GPU memory/time.")
        return

    profile_cfg = rcfg.get("profile", {}) or {}
    mode = profile_cfg.get("mode", "svd")
    assert mode in ("svd", "standard"), f"Unknown mode: {mode}"
    num_steps = int(profile_cfg.get("num_steps", 5))
    warmup_steps = int(profile_cfg.get("warmup_steps", 2))
    detailed = bool(profile_cfg.get("detailed", False))
    output_json = profile_cfg.get("output_json", "optimizer_profile.json")

    loss_key = rcfg.get("loss", "ce")
    batch_size = _first(rcfg.get("batch_size", 64))
    model_seed = _first(rcfg.get("model_seeds", [42]))
    loader_seed = rcfg.get("loader_seed", 0)
    lr = profile_cfg.get("lr", _first(rcfg.get("lrs", [0.01])))

    # SVD-only hparams
    if "k_values" in rcfg:
        k = _first(rcfg["k_values"])
    elif "k_fractions" in rcfg:
        k = max(1, int(_first(rcfg["k_fractions"]) * batch_size))
    else:
        k = max(1, batch_size // 4)
    rtol = _first(rcfg.get("rtol", [1e-3]))
    svd_mode = _first(rcfg.get("svd_mode", ["randomized"]))

    # Standard-only hparams
    optimizer_name = profile_cfg.get("optimizer_name", "Adam")

    set_seed(model_seed)
    dataset = instantiate(cfg.dataset)
    train_loader = DataLoader(
        dataset.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(loader_seed),
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Baseline: model only, before optimizer/wrapper construction
    # ------------------------------------------------------------------
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    pre_model_bytes = torch.cuda.memory_allocated(device)

    model = instantiate(cfg.model).to(device)

    torch.cuda.synchronize(device)
    baseline_model_bytes = torch.cuda.memory_allocated(device) - pre_model_bytes

    n_params = sum(p.numel() for p in model.parameters())

    # ------------------------------------------------------------------
    # Build optimizer / wrapper + pick step fn
    # ------------------------------------------------------------------
    if mode == "svd":
        loss_fn = SVD_LOSS_FNS[loss_key]
        train_model = SvenWrapper(model, loss_fn, device)
        optimizer = Sven(train_model, lr=lr, k=k, rtol=rtol, svd_mode=svd_mode)

        def step_fn(batch):
            _svd_step(train_model, optimizer, batch)
    else:
        loss_fn = STANDARD_LOSS_FNS[loss_key]
        optimizer = build_standard_optimizer(model, optimizer_name, lr)
        train_model = None  # only used for SVD detailed breakdown

        def step_fn(batch):
            _standard_step(model, optimizer, loss_fn, batch)

    # ------------------------------------------------------------------
    # Data iterator w/ auto-restart
    # ------------------------------------------------------------------
    data_iter = iter(train_loader)

    def _next_batch():
        nonlocal data_iter
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            xb, yb = next(data_iter)
        return xb.to(device), yb.to(device)

    # ------------------------------------------------------------------
    # Warm-up (allocates optimizer state, caches CUDA workspaces, etc.)
    # ------------------------------------------------------------------
    step_ctx = torch.no_grad() if mode == "svd" else torch.enable_grad()
    with step_ctx:
        for _ in range(warmup_steps):
            step_fn(_next_batch())

    # ------------------------------------------------------------------
    # Measurement loop
    # ------------------------------------------------------------------
    resident_bytes: list[int] = []
    peak_bytes: list[int] = []
    step_times_ms: list[float] = []

    with step_ctx:
        for _ in range(num_steps):
            batch = _next_batch()

            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            resident = torch.cuda.memory_allocated(device)

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            step_fn(batch)
            end_evt.record()
            torch.cuda.synchronize(device)

            step_ms = start_evt.elapsed_time(end_evt)
            peak = torch.cuda.max_memory_allocated(device)

            resident_bytes.append(int(resident))
            peak_bytes.append(int(peak))
            step_times_ms.append(float(step_ms))

    # ------------------------------------------------------------------
    # Optional SVD-only phase breakdown
    # ------------------------------------------------------------------
    phase_rows: list[dict] | None = None
    if detailed and mode == "svd":
        tracker = _PhaseTracker(device)
        records_per_step: list[list[tuple[str, int, int]]] = []
        with torch.no_grad():
            for _ in range(num_steps):
                batch = _next_batch()
                tracker.reset()
                _detailed_svd_step(train_model, optimizer, batch, tracker)
                records_per_step.append(list(tracker.records))
        n_phases = len(records_per_step[0])
        phase_rows = []
        for i in range(n_phases):
            label = records_per_step[0][i][0]
            mean_cur = float(np.mean([r[i][1] for r in records_per_step]))
            mean_peak = float(np.mean([r[i][2] for r in records_per_step]))
            phase_rows.append({
                "phase": label,
                "mean_current_bytes": mean_cur,
                "mean_peak_bytes": mean_peak,
            })

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    resident_mean = float(np.mean(resident_bytes))
    peak_mean = float(np.mean(peak_bytes))
    peak_max = float(np.max(peak_bytes))
    transient_delta = peak_mean - resident_mean
    peak_over_baseline = peak_mean - float(baseline_model_bytes)

    times = np.asarray(step_times_ms)

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    width = 60
    print(f"\n{'='*width}\nOptimizer Profile\n{'='*width}")
    print(f"  mode             : {mode}")
    print(f"  device           : {device}")
    print(f"  batch_size       : {batch_size}")
    print(f"  n_params         : {n_params:,}")
    print(f"  loss             : {loss_key}")
    if mode == "svd":
        print(f"  k                : {k}  (k/batch = {k/batch_size:.2f})")
        print(f"  rtol             : {rtol}")
        print(f"  svd_mode         : {svd_mode}")
    else:
        print(f"  optimizer        : {optimizer_name}")
        print(f"  lr               : {lr}")
    print(f"  warmup / measured: {warmup_steps} / {num_steps}")
    print(f"{'='*width}\n")

    print("Memory")
    print(f"  Baseline (model on GPU)       : {_mb(baseline_model_bytes)}")
    print(f"  Resident at step start (mean) : {_mb(resident_mean)}")
    print(f"  Peak during step (mean)       : {_mb(peak_mean)}")
    print(f"  Peak during step (max)        : {_mb(peak_max)}")
    print(f"  Transient overhead (mean)     : {_mb(transient_delta)}")
    print(f"  Peak over baseline            : {_mb(peak_over_baseline)}")

    print("\nTime per step (ms)")
    print(f"  Mean  : {times.mean():.3f}")
    print(f"  Std   : {times.std(ddof=0):.3f}")
    print(f"  Min   : {times.min():.3f}")
    print(f"  Max   : {times.max():.3f}")

    if phase_rows is not None:
        print("\nSVD phase breakdown (mean over measurement steps)")
        w = 44
        print(f"{'Phase':<{w}} {'Current':>14} {'Peak':>14}")
        print("-" * (w + 30))
        for row in phase_rows:
            print(
                f"{row['phase']:<{w}} "
                f"{_mb(row['mean_current_bytes']):>14} "
                f"{_mb(row['mean_peak_bytes']):>14}"
            )

    # ------------------------------------------------------------------
    # JSON artifact
    # ------------------------------------------------------------------
    artifact: dict[str, Any] = {
        "config": {
            "mode": mode,
            "device": str(device),
            "batch_size": batch_size,
            "loss": loss_key,
            "n_params": int(n_params),
            "num_steps": num_steps,
            "warmup_steps": warmup_steps,
            "model_seed": model_seed,
            "loader_seed": loader_seed,
        },
        "memory": {
            "baseline_model_bytes": int(baseline_model_bytes),
            "resident_bytes_per_step": resident_bytes,
            "peak_bytes_per_step": peak_bytes,
            "resident_bytes_mean": resident_mean,
            "peak_bytes_mean": peak_mean,
            "peak_bytes_max": peak_max,
            "transient_delta_bytes_mean": transient_delta,
            "peak_over_baseline_bytes_mean": peak_over_baseline,
        },
        "time": {
            "step_times_ms": step_times_ms,
            "mean_ms": float(times.mean()),
            "std_ms": float(times.std(ddof=0)),
            "min_ms": float(times.min()),
            "max_ms": float(times.max()),
        },
    }
    if mode == "svd":
        artifact["config"].update(
            {"k": int(k), "rtol": float(rtol), "svd_mode": svd_mode}
        )
    else:
        artifact["config"].update(
            {"optimizer_name": optimizer_name, "lr": float(lr)}
        )
    if phase_rows is not None:
        artifact["detailed_phases"] = phase_rows

    out_path = Path(output_json)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nWrote JSON artifact → {out_path}")


if __name__ == "__main__":
    main()
