#!/usr/bin/env python3
"""
GPU memory + step-time profiler for SV3 optimizers, scanned over a full
hyperparameter grid (mirrors the scan logic of generic_scan.py).

For each optimizer configuration in the grid, reports:
  • Baseline GPU memory  — model only, before optimizer construction.
  • Resident GPU memory  — model + optimizer state + persistent tensors,
                            measured at the start of each profiled step.
  • Peak GPU memory      — max_memory_allocated during a single step.
  • Wall-clock step time — CUDA-event timed, end-to-end.

Each configuration is saved as a uniquely-named JSON file under
  {output_dir}/{config_name}/{run_id}.json

Existing files are skipped (resumable scan).

Usage:
  python -m experiments.optimizer_profile --config-name toy_1d_scan

Profile-specific config keys (nest under `profile:` in the YAML):
  num_steps    : measured steps (default 5)
  warmup_steps : warm-up steps before measurement (default 2)
  detailed     : SVD-only fine-grained phase breakdown (default False)
  output_dir   : root output directory (default "profile_results")
"""

from __future__ import annotations

import copy
import json
import os
from itertools import product
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from experiments.experiment_code.experiment_utils import (
    build_standard_optimizer, process_hparam_config, set_seed,
    _is_closure_optimizer, listify,
)
from sven.nn import SvenWrapper
from sven.opt import Sven
from sven.opt.pinv import pinv
from experiments.optimizers.hig import HIGWrapper, HIGOptimizer

try:
    from torchjd.aggregation import UPGrad, Mean, Sum
    from torchjd.autojac import backward as jd_backward, jac_to_grad
    _JD_AGGREGATORS = {"UPGrad": UPGrad, "Mean": Mean, "Sum": Sum}
    _HAS_TORCHJD = True
except ImportError:
    _JD_AGGREGATORS = {}
    _HAS_TORCHJD = False


# ---------------------------------------------------------------------------
# Loss function registries
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


def _mb(n_bytes: float) -> str:
    return f"{n_bytes / 1024**2:.1f} MB"


def _build_id_string(cfg: dict) -> str:
    fields = cfg.get("result_id_fields", [])
    if not fields:
        return ""
    return "_" + "_".join(f"{f}{cfg[f]}" for f in fields)


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def _svd_step(train_model, optimizer, batch):
    train_model.loss_and_grad(batch)
    optimizer.step(batch)


def _standard_step(model, optimizer, loss_fn, batch):
    xb, yb = batch
    if _is_closure_optimizer(optimizer):
        def closure():
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            return loss
        optimizer.step(closure)
    else:
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()


def _jd_step(model, inner_optimizer, aggregator, per_sample_loss_fn, batch):
    xb, yb = batch
    inner_optimizer.zero_grad()
    pred = model(xb)
    losses = per_sample_loss_fn(pred, yb)  # (B,)
    jd_backward(losses)
    jac_to_grad(model.parameters(), aggregator)
    inner_optimizer.step()


def _hig_step(train_model, optimizer, batch):
    train_model.output_and_loss_grad(batch)
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
# Core profiling routine (single optimizer configuration)
# ---------------------------------------------------------------------------

def _profile_single(
    init_state: dict,
    dataset,
    cfg: DictConfig,
    device: str,
    mode: str,
    loss_key: str,
    batch_size: int,
    loader_seed: int,
    num_steps: int,
    warmup_steps: int,
    detailed: bool,
    # SVD params
    k: int | None = None,
    lr: float | None = None,
    rtol: float | None = None,
    svd_mode: str | None = None,
    kappa: float = 2.0,
    microbatch_size: int | None = None,
    param_fraction: float | None = None,
    # Standard params
    optim_name: str | None = None,
    std_lr: float | None = None,
    weight_decay: float = 0.0,
    lbfgs_kwargs: dict | None = None,
    polyak_kwargs: dict | None = None,
    # JD params
    jd_aggregator_name: str | None = None,
    jd_inner_optim: str | None = None,
    jd_lr: float | None = None,
    # HIG params
    hig_lr: float | None = None,
    hig_tau: float = 1e-4,
) -> dict[str, Any]:
    """Profile a single optimizer configuration. Returns memory/time stats."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    pre_model_bytes = torch.cuda.memory_allocated(device)

    model = instantiate(cfg.model)
    model.load_state_dict(init_state)
    model = model.to(device)

    torch.cuda.synchronize(device)
    baseline_model_bytes = torch.cuda.memory_allocated(device) - pre_model_bytes
    n_params = sum(p.numel() for p in model.parameters())

    # Build optimizer / wrapper + step fn
    if mode == "svd":
        loss_fn = SVD_LOSS_FNS[loss_key]
        mb = microbatch_size if microbatch_size is not None else 1
        pf = param_fraction if param_fraction is not None else 1.0
        train_model = SvenWrapper(model, loss_fn, device, microbatch_size=mb, param_fraction=pf, kappa=kappa)
        optimizer = Sven(train_model, lr=lr, k=k, rtol=rtol, svd_mode=svd_mode)

        def step_fn(batch):
            _svd_step(train_model, optimizer, batch)

    elif mode == "jd":
        per_sample_loss_fn = SVD_LOSS_FNS[loss_key]
        aggregator = _JD_AGGREGATORS[jd_aggregator_name]()
        inner_optimizer = build_standard_optimizer(model, jd_inner_optim, jd_lr)
        train_model = None

        def step_fn(batch):
            _jd_step(model, inner_optimizer, aggregator, per_sample_loss_fn, batch)

    elif mode == "hig":
        loss_fn = SVD_LOSS_FNS[loss_key]
        train_model = HIGWrapper(model, loss_fn, device)
        optimizer = HIGOptimizer(train_model, lr=hig_lr, tau=hig_tau)

        def step_fn(batch):
            _hig_step(train_model, optimizer, batch)

    else:  # "standard"
        loss_fn = STANDARD_LOSS_FNS[loss_key]
        if polyak_kwargs is not None:
            optimizer = build_standard_optimizer(model, "PolyakSGD", lr=None, **polyak_kwargs)
        elif lbfgs_kwargs is not None:
            optimizer = build_standard_optimizer(model, "LBFGS", std_lr, **lbfgs_kwargs)
        else:
            optimizer = build_standard_optimizer(model, optim_name, std_lr, weight_decay=weight_decay)
        train_model = None

        def step_fn(batch):
            _standard_step(model, optimizer, loss_fn, batch)

    # Data loader + auto-restart iterator
    train_loader = DataLoader(
        dataset.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(loader_seed),
        drop_last=(microbatch_size is not None),
    )
    data_iter = iter(train_loader)

    def _next_batch():
        nonlocal data_iter
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            xb, yb = next(data_iter)
        return xb.to(device), yb.to(device)

    def _ctx():
        # JD needs autograd; SVD and HIG manage their own gradient context internally
        if mode in ("standard", "jd"):
            return torch.enable_grad()
        return torch.no_grad()

    # Warm-up (allocates optimizer state, caches CUDA workspaces, etc.)
    with _ctx():
        for _ in range(warmup_steps):
            step_fn(_next_batch())

    # Measurement loop
    resident_bytes: list[int] = []
    peak_bytes: list[int] = []
    step_times_ms: list[float] = []

    with _ctx():
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

    # Optional SVD phase breakdown
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

    # Aggregate
    resident_mean = float(np.mean(resident_bytes))
    peak_mean = float(np.mean(peak_bytes))
    peak_max = float(np.max(peak_bytes))
    times = np.asarray(step_times_ms)

    result: dict[str, Any] = {
        "n_params": int(n_params),
        "memory": {
            "baseline_model_bytes": int(baseline_model_bytes),
            "resident_bytes_per_step": resident_bytes,
            "peak_bytes_per_step": peak_bytes,
            "resident_bytes_mean": resident_mean,
            "peak_bytes_mean": peak_mean,
            "peak_bytes_max": peak_max,
            "transient_delta_bytes_mean": peak_mean - resident_mean,
            "peak_over_baseline_bytes_mean": peak_mean - float(baseline_model_bytes),
        },
        "time": {
            "step_times_ms": step_times_ms,
            "mean_ms": float(times.mean()),
            "std_ms": float(times.std(ddof=0)),
            "min_ms": float(times.min()),
            "max_ms": float(times.max()),
        },
    }
    if phase_rows is not None:
        result["detailed_phases"] = phase_rows

    torch.cuda.empty_cache()
    torch.compiler.reset()
    return result


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_summary(run_id: str, result: dict, batch_size: int, num_steps: int, warmup_steps: int) -> None:
    width = 60
    m = result["memory"]
    t = result["time"]
    print(f"\n{'='*width}")
    print(f"  {run_id}")
    print(f"  n_params={result['n_params']:,}  bs={batch_size}  warmup/measured={warmup_steps}/{num_steps}")
    print(f"  Baseline      : {_mb(m['baseline_model_bytes'])}")
    print(f"  Resident mean : {_mb(m['resident_bytes_mean'])}")
    print(f"  Peak mean     : {_mb(m['peak_bytes_mean'])}  (max {_mb(m['peak_bytes_max'])})")
    print(f"  Transient     : {_mb(m['transient_delta_bytes_mean'])}")
    print(f"  Step time     : {t['mean_ms']:.3f} ± {t['std_ms']:.3f} ms")
    if "detailed_phases" in result:
        w = 44
        print(f"\n  {'Phase':<{w}} {'Current':>14} {'Peak':>14}")
        print(f"  {'-'*(w+30)}")
        for row in result["detailed_phases"]:
            print(f"  {row['phase']:<{w}} {_mb(row['mean_current_bytes']):>14} {_mb(row['mean_peak_bytes']):>14}")


# ---------------------------------------------------------------------------
# Main — scan over all optimizer configurations
# ---------------------------------------------------------------------------

@hydra.main(config_path="configs", version_base=None)
def main(cfg: DictConfig) -> None:
    rcfg = OmegaConf.to_container(cfg, resolve=True)
    device = rcfg.get("device", "cuda")
    if not torch.cuda.is_available():
        print("CUDA not available — cannot profile GPU memory/time.")
        return

    profile_cfg = rcfg.get("profile", {}) or {}
    num_steps = int(profile_cfg.get("num_steps", 5))
    warmup_steps = int(profile_cfg.get("warmup_steps", 2))
    detailed = bool(profile_cfg.get("detailed", False))
    output_dir = profile_cfg.get("output_dir", "profile_results")

    mode = rcfg.get("mode", "both")
    _VALID_MODES = ("svd", "standard", "both", "jd", "hig", "all")
    assert mode in _VALID_MODES, f"Unknown mode: {mode}. Choose from {_VALID_MODES}"

    run_svd      = mode in ("svd", "both", "all")
    run_standard = mode in ("standard", "both", "all")
    run_jd       = mode in ("jd", "all") and ('lrs_jd' in rcfg or 'aggregators_jd' in rcfg)
    run_hig      = mode in ("hig", "all") and ('lrs_hig' in rcfg or 'tau_hig' in rcfg)
    loss_key = rcfg.get("loss", "ce")
    loader_seed = rcfg.get("loader_seed", 0)

    scan_name = HydraConfig.get().job.config_name
    scan_dir = os.path.join(output_dir, scan_name)
    os.makedirs(scan_dir, exist_ok=True)

    hparams = process_hparam_config(rcfg)
    seeds = hparams["model_seeds"]

    mlp_widths = listify(rcfg.get('mlp_widths', rcfg.get('mlp_width', 32)))
    dataset = instantiate(cfg.dataset)

    for mlp_width in mlp_widths:
        if len(mlp_widths) > 1:
            print(f"\n{'*'*80}\n* mlp_width = {mlp_width}\n{'*'*80}")
        with open_dict(cfg):
            cfg.mlp_width = mlp_width
        rcfg['mlp_width'] = mlp_width
        id_str = _build_id_string(rcfg)

        for model_seed in seeds:
            print(f"\n{'#'*80}\n# Model seed: {model_seed}\n{'#'*80}")
            seed_str = f"_mseed{model_seed}_lseed{loader_seed}"

            set_seed(model_seed)
            base_model = instantiate(cfg.model)
            init_state = copy.deepcopy(base_model.state_dict())
            del base_model

            # ------------------------------------------------------------------
            # SVD optimizer scan
            # ------------------------------------------------------------------
            if run_svd:
                print(f"\n{'='*80}\nProfiling SVD optimizer\n{'='*80}")

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
                    hparams["kappas"],
                )

                for batch_size, k_item, lr, rtol, svd_mode, microbatch_size, param_fraction, kappa in svd_grid:
                    k = max(1, int(k_item * batch_size)) if not use_k_values else k_item

                    run_id = (
                        f"profile_svd_bs{batch_size}{id_str}"
                        f"_k{k}_lr{lr}_rtol{rtol}_svd{svd_mode}{seed_str}"
                    )
                    if microbatch_size is not None:
                        run_id += f"_mb{microbatch_size}"
                    if param_fraction is not None:
                        run_id += f"_pf{param_fraction}"
                    if kappa != 2.0:
                        run_id += f"_kappa{kappa}"

                    out_path = os.path.join(scan_dir, run_id + ".json")
                    if os.path.exists(out_path):
                        print(f"  [skip] {run_id}")
                        continue

                    print(f"\nSVD: bs={batch_size}, k={k}, lr={lr}, rtol={rtol}, svd_mode={svd_mode}", end="")
                    if microbatch_size is not None:
                        print(f", mb={microbatch_size}", end="")
                    if param_fraction is not None:
                        print(f", pf={param_fraction}", end="")
                    if kappa != 2.0:
                        print(f", kappa={kappa}", end="")
                    print()

                    try:
                        result = _profile_single(
                            init_state, dataset, cfg, device, "svd", loss_key,
                            batch_size, loader_seed, num_steps, warmup_steps, detailed,
                            k=k, lr=lr, rtol=rtol, svd_mode=svd_mode,
                            kappa=kappa, microbatch_size=microbatch_size, param_fraction=param_fraction,
                        )

                        artifact: dict[str, Any] = {
                            "run_id": run_id,
                            "config": {
                                "mode": "svd",
                                "optimizer": "SVD",
                                "device": str(device),
                                "batch_size": batch_size,
                                "loss": loss_key,
                                "n_params": result["n_params"],
                                "model_seed": model_seed,
                                "loader_seed": loader_seed,
                                "num_steps": num_steps,
                                "warmup_steps": warmup_steps,
                                "k": int(k),
                                "k_fraction": k / batch_size,
                                "lr": float(lr),
                                "rtol": float(rtol),
                                "svd_mode": svd_mode,
                                "kappa": float(kappa),
                                "microbatch_size": microbatch_size,
                                "param_fraction": param_fraction,
                            },
                            "memory": result["memory"],
                            "time": result["time"],
                        }
                        if "detailed_phases" in result:
                            artifact["detailed_phases"] = result["detailed_phases"]
                        for f in rcfg.get("result_id_fields", []):
                            artifact["config"][f] = rcfg[f]

                        with open(out_path, "w") as fh:
                            json.dump(artifact, fh, indent=2)
                        _print_summary(run_id, result, batch_size, num_steps, warmup_steps)
                        print(f"  → {out_path}")

                    except Exception as e:
                        print(f"  [error] {run_id}: {e}")

            # ------------------------------------------------------------------
            # Standard optimizer scan
            # ------------------------------------------------------------------
            if run_standard:
                print(f"\n{'='*80}\nProfiling standard optimizers\n{'='*80}")

                has_lbfgs = "LBFGS" in hparams["optimizers_standard"]
                has_polyak = "PolyakSGD" in hparams["optimizers_standard"]
                non_special = [o for o in hparams["optimizers_standard"] if o not in ("LBFGS", "PolyakSGD")]

                # --- Non-LBFGS / non-PolyakSGD optimizers ---
                if non_special:
                    std_grid = product(
                        hparams["batch_size"],
                        hparams["lrs_standard"],
                        non_special,
                        hparams["weight_decays"],
                    )
                    for batch_size, lr, optim_name, weight_decay in std_grid:
                        if optim_name not in ("AdamW", "Muon") and weight_decay != 0.0:
                            continue

                        run_id = f"profile_std_bs{batch_size}{id_str}_lr{lr}_optim{optim_name}"
                        if weight_decay != 0.0:
                            run_id += f"_wd{weight_decay}"
                        run_id += seed_str

                        out_path = os.path.join(scan_dir, run_id + ".json")
                        if os.path.exists(out_path):
                            print(f"  [skip] {run_id}")
                            continue

                        wd_str = f", wd={weight_decay}" if weight_decay != 0.0 else ""
                        print(f"\nStandard: bs={batch_size}, lr={lr}, optim={optim_name}{wd_str}")

                        try:
                            result = _profile_single(
                                init_state, dataset, cfg, device, "standard", loss_key,
                                batch_size, loader_seed, num_steps, warmup_steps, detailed,
                                optim_name=optim_name, std_lr=lr, weight_decay=weight_decay,
                            )

                            artifact = {
                                "run_id": run_id,
                                "config": {
                                    "mode": "standard",
                                    "optimizer": optim_name,
                                    "device": str(device),
                                    "batch_size": batch_size,
                                    "loss": loss_key,
                                    "n_params": result["n_params"],
                                    "model_seed": model_seed,
                                    "loader_seed": loader_seed,
                                    "num_steps": num_steps,
                                    "warmup_steps": warmup_steps,
                                    "lr": float(lr),
                                    "weight_decay": float(weight_decay),
                                },
                                "memory": result["memory"],
                                "time": result["time"],
                            }
                            for f in rcfg.get("result_id_fields", []):
                                artifact["config"][f] = rcfg[f]

                            with open(out_path, "w") as fh:
                                json.dump(artifact, fh, indent=2)
                            _print_summary(run_id, result, batch_size, num_steps, warmup_steps)
                            print(f"  → {out_path}")

                        except Exception as e:
                            print(f"  [error] {run_id}: {e}")

                # --- LBFGS ---
                if has_lbfgs:
                    lbfgs_grid = product(
                        hparams["batch_size"],
                        hparams["lrs_lbfgs"],
                        hparams["lbfgs_max_iter"],
                        hparams["lbfgs_history_size"],
                        hparams["lbfgs_line_search_fn"],
                    )
                    for batch_size, lr, max_iter, history_size, line_search_fn in lbfgs_grid:
                        run_id = (
                            f"profile_std_bs{batch_size}{id_str}_lr{lr}_optimLBFGS"
                            f"_mi{max_iter}_hs{history_size}_ls{line_search_fn}{seed_str}"
                        )

                        out_path = os.path.join(scan_dir, run_id + ".json")
                        if os.path.exists(out_path):
                            print(f"  [skip] {run_id}")
                            continue

                        print(f"\nLBFGS: bs={batch_size}, lr={lr}, max_iter={max_iter}, "
                              f"history_size={history_size}, line_search={line_search_fn}")

                        lbfgs_kw = {
                            "max_iter": max_iter,
                            "history_size": history_size,
                            "line_search_fn": line_search_fn if line_search_fn != "none" else None,
                        }
                        try:
                            result = _profile_single(
                                init_state, dataset, cfg, device, "standard", loss_key,
                                batch_size, loader_seed, num_steps, warmup_steps, detailed,
                                optim_name="LBFGS", std_lr=lr, lbfgs_kwargs=lbfgs_kw,
                            )

                            artifact = {
                                "run_id": run_id,
                                "config": {
                                    "mode": "standard",
                                    "optimizer": "LBFGS",
                                    "device": str(device),
                                    "batch_size": batch_size,
                                    "loss": loss_key,
                                    "n_params": result["n_params"],
                                    "model_seed": model_seed,
                                    "loader_seed": loader_seed,
                                    "num_steps": num_steps,
                                    "warmup_steps": warmup_steps,
                                    "lr": float(lr),
                                    "lbfgs_max_iter": max_iter,
                                    "lbfgs_history_size": history_size,
                                    "lbfgs_line_search_fn": line_search_fn,
                                },
                                "memory": result["memory"],
                                "time": result["time"],
                            }
                            for f in rcfg.get("result_id_fields", []):
                                artifact["config"][f] = rcfg[f]

                            with open(out_path, "w") as fh:
                                json.dump(artifact, fh, indent=2)
                            _print_summary(run_id, result, batch_size, num_steps, warmup_steps)
                            print(f"  → {out_path}")

                        except Exception as e:
                            print(f"  [error] {run_id}: {e}")

                # --- PolyakSGD ---
                if has_polyak:
                    polyak_grid = product(
                        hparams["batch_size"],
                        hparams["polyak_f_star"],
                        hparams["polyak_max_lr"],
                        hparams["polyak_eps"],
                    )
                    for batch_size, f_star, max_lr, eps in polyak_grid:
                        run_id = (
                            f"profile_std_bs{batch_size}{id_str}_optimPolyakSGD"
                            f"_fstar{f_star}_maxlr{max_lr}_eps{eps}{seed_str}"
                        )

                        out_path = os.path.join(scan_dir, run_id + ".json")
                        if os.path.exists(out_path):
                            print(f"  [skip] {run_id}")
                            continue

                        print(f"\nPolyakSGD: bs={batch_size}, f_star={f_star}, max_lr={max_lr}, eps={eps}")

                        polyak_kw = {"f_star": f_star, "max_lr": max_lr, "eps": eps}
                        try:
                            result = _profile_single(
                                init_state, dataset, cfg, device, "standard", loss_key,
                                batch_size, loader_seed, num_steps, warmup_steps, detailed,
                                polyak_kwargs=polyak_kw,
                            )

                            artifact = {
                                "run_id": run_id,
                                "config": {
                                    "mode": "standard",
                                    "optimizer": "PolyakSGD",
                                    "device": str(device),
                                    "batch_size": batch_size,
                                    "loss": loss_key,
                                    "n_params": result["n_params"],
                                    "model_seed": model_seed,
                                    "loader_seed": loader_seed,
                                    "num_steps": num_steps,
                                    "warmup_steps": warmup_steps,
                                    "polyak_f_star": f_star,
                                    "polyak_max_lr": max_lr,
                                    "polyak_eps": eps,
                                },
                                "memory": result["memory"],
                                "time": result["time"],
                            }
                            for f in rcfg.get("result_id_fields", []):
                                artifact["config"][f] = rcfg[f]

                            with open(out_path, "w") as fh:
                                json.dump(artifact, fh, indent=2)
                            _print_summary(run_id, result, batch_size, num_steps, warmup_steps)
                            print(f"  → {out_path}")

                        except Exception as e:
                            print(f"  [error] {run_id}: {e}")

            # ------------------------------------------------------------------
            # Jacobian Descent (torchjd) profiling
            # ------------------------------------------------------------------
            if run_jd:
                if not _HAS_TORCHJD:
                    print("  [skip] torchjd not installed — skipping JD profiling")
                else:
                    print(f"\n{'='*80}\nProfiling Jacobian Descent\n{'='*80}")

                    jd_grid = product(
                        hparams["batch_size"],
                        hparams["lrs_jd"],
                        hparams["aggregators_jd"],
                        hparams["inner_optimizers_jd"],
                    )

                    for batch_size, lr, aggregator_name, inner_optim_name in jd_grid:
                        if aggregator_name not in _JD_AGGREGATORS:
                            print(f"  [skip] Unknown aggregator: {aggregator_name}")
                            continue

                        run_id = (
                            f"profile_jd_bs{batch_size}{id_str}"
                            f"_lr{lr}_agg{aggregator_name}_inner{inner_optim_name}{seed_str}"
                        )
                        out_path = os.path.join(scan_dir, run_id + ".json")
                        if os.path.exists(out_path):
                            print(f"  [skip] {run_id}")
                            continue

                        print(f"\nJD: bs={batch_size}, lr={lr}, aggregator={aggregator_name}, inner={inner_optim_name}")

                        try:
                            result = _profile_single(
                                init_state, dataset, cfg, device, "jd", loss_key,
                                batch_size, loader_seed, num_steps, warmup_steps, detailed,
                                jd_aggregator_name=aggregator_name,
                                jd_inner_optim=inner_optim_name,
                                jd_lr=lr,
                            )

                            artifact = {
                                "run_id": run_id,
                                "config": {
                                    "mode": "jd",
                                    "optimizer": f"JD_{aggregator_name}",
                                    "device": str(device),
                                    "batch_size": batch_size,
                                    "loss": loss_key,
                                    "n_params": result["n_params"],
                                    "model_seed": model_seed,
                                    "loader_seed": loader_seed,
                                    "num_steps": num_steps,
                                    "warmup_steps": warmup_steps,
                                    "lr": float(lr),
                                    "aggregator": aggregator_name,
                                    "inner_optimizer": inner_optim_name,
                                },
                                "memory": result["memory"],
                                "time": result["time"],
                            }
                            for f in rcfg.get("result_id_fields", []):
                                artifact["config"][f] = rcfg[f]

                            with open(out_path, "w") as fh:
                                json.dump(artifact, fh, indent=2)
                            _print_summary(run_id, result, batch_size, num_steps, warmup_steps)
                            print(f"  → {out_path}")

                        except Exception as e:
                            print(f"  [error] {run_id}: {e}")

            # ------------------------------------------------------------------
            # Half-Inverse Gradients profiling
            # ------------------------------------------------------------------
            if run_hig:
                print(f"\n{'='*80}\nProfiling Half-Inverse Gradients\n{'='*80}")

                hig_grid = product(
                    hparams["batch_size"],
                    hparams["lrs_hig"],
                    hparams["tau_hig"],
                )

                for batch_size, lr, tau in hig_grid:
                    run_id = (
                        f"profile_hig_bs{batch_size}{id_str}"
                        f"_lr{lr}_tau{tau}{seed_str}"
                    )
                    out_path = os.path.join(scan_dir, run_id + ".json")
                    if os.path.exists(out_path):
                        print(f"  [skip] {run_id}")
                        continue

                    print(f"\nHIG: bs={batch_size}, lr={lr}, tau={tau}")

                    try:
                        result = _profile_single(
                            init_state, dataset, cfg, device, "hig", loss_key,
                            batch_size, loader_seed, num_steps, warmup_steps, detailed,
                            hig_lr=lr, hig_tau=tau,
                        )

                        artifact = {
                            "run_id": run_id,
                            "config": {
                                "mode": "hig",
                                "optimizer": "HIG",
                                "device": str(device),
                                "batch_size": batch_size,
                                "loss": loss_key,
                                "n_params": result["n_params"],
                                "model_seed": model_seed,
                                "loader_seed": loader_seed,
                                "num_steps": num_steps,
                                "warmup_steps": warmup_steps,
                                "lr": float(lr),
                                "tau": float(tau),
                            },
                            "memory": result["memory"],
                            "time": result["time"],
                        }
                        for f in rcfg.get("result_id_fields", []):
                            artifact["config"][f] = rcfg[f]

                        with open(out_path, "w") as fh:
                            json.dump(artifact, fh, indent=2)
                        _print_summary(run_id, result, batch_size, num_steps, warmup_steps)
                        print(f"  → {out_path}")

                    except Exception as e:
                        print(f"  [error] {run_id}: {e}")

    print(f"\nProfile scan complete. Results in {scan_dir}/")


if __name__ == "__main__":
    main()
