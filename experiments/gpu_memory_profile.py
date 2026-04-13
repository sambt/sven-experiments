#!/usr/bin/env python3
"""
GPU memory profiler for the Sven optimizer.

Measures peak GPU memory at each phase of a single Sven training step:
  0. Baseline (model on GPU, no batch)
  1. Batch loaded to GPU
  2. Jacobian computation  (loss_and_grad / jacrev)  ← usually the bottleneck
  3. SVD computation       (pinv)
  4. After deleting Jacobian + empty_cache
  5. Parameter update
  6. Full cleanup

Usage (from the repo root, same way as run_experiment.py):
  python -m experiments.gpu_memory_profile --config-name cifar10_resnet_ce_test

Config overrides (pass as extra args on the command line):
  profile.num_steps=3      # measurement steps to average over (default 3)
  profile.warmup_steps=1   # warm-up steps before measurement   (default 1)

All other config keys (batch_size, k_values / k_fractions, lrs, rtol,
svd_mode, model_seeds, loader_seed, …) are read from the Hydra config as
usual. When a value is a list only the first element is used.
"""

from __future__ import annotations

import hydra
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from experiments.experiment_code.experiment_utils import set_seed
from sven.nn import SvenWrapper
from sven.opt import Sven
from sven.opt.pinv import pinv


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _mb(n_bytes: float) -> str:
    return f"{n_bytes / 1024**2:.1f} MB"


class PhaseTracker:
    """Records (current, phase-peak) memory after each phase.

    Call ``checkpoint(label)`` immediately *after* the phase finishes.
    Each call records the peak since the previous checkpoint, then
    resets the peak counter so the next phase gets its own high-water mark.
    """

    def __init__(self, device: str | torch.device):
        self.device = device
        self.records: list[tuple[str, int, int]] = []

    def reset(self) -> None:
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.records.clear()

    def checkpoint(self, label: str) -> None:
        torch.cuda.synchronize(self.device)
        cur  = torch.cuda.memory_allocated(self.device)
        peak = torch.cuda.max_memory_allocated(self.device)
        self.records.append((label, cur, peak))
        # Reset so the next phase starts fresh
        torch.cuda.reset_peak_memory_stats(self.device)

    def print_table(self) -> None:
        w = 44
        print(f"\n{'Phase':<{w}} {'Current':>12} {'Peak (this phase)':>20}")
        print("-" * (w + 34))
        for label, cur, peak in self.records:
            print(f"{label:<{w}} {_mb(cur):>12} {_mb(peak):>20}")
        print()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _first(v):
    """Return v[0] if v is a list, else v."""
    return v[0] if isinstance(v, (list, tuple)) else v


SVD_LOSS_FNS = {
    "ce": lambda pred, y: F.cross_entropy(pred, y, reduction="none"),
    "mse": lambda pred, y: ((pred - y) ** 2).sum(dim=-1),
    "label_regression": lambda pred, y: (
        pred - F.one_hot(y.to(torch.long), num_classes=pred.shape[-1]).to(pred)
    ).pow(2).sum(dim=1),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="configs", version_base=None)
def main(cfg: DictConfig) -> None:
    rcfg = OmegaConf.to_container(cfg, resolve=True)
    device = rcfg.get("device", "cuda")

    if not torch.cuda.is_available():
        print("CUDA not available — cannot profile GPU memory.")
        return

    # ------------------------------------------------------------------
    # Profile settings
    # ------------------------------------------------------------------
    profile_cfg  = rcfg.get("profile", {})
    num_steps    = int(profile_cfg.get("num_steps",    3))
    warmup_steps = int(profile_cfg.get("warmup_steps", 1))

    # ------------------------------------------------------------------
    # Resolve scalar hyperparams (take first element when config is a list)
    # ------------------------------------------------------------------
    loss_key  = rcfg.get("loss", "ce")
    batch_size = _first(rcfg.get("batch_size", 64))

    if "k_values" in rcfg:
        k = _first(rcfg["k_values"])
    elif "k_fractions" in rcfg:
        k = max(1, int(_first(rcfg["k_fractions"]) * batch_size))
    else:
        k = max(1, batch_size // 4)

    lr       = _first(rcfg.get("lrs",      [0.01]))
    rtol     = _first(rcfg.get("rtol",     [1e-3]))
    svd_mode = _first(rcfg.get("svd_mode", ["randomized"]))

    model_seed  = _first(rcfg.get("model_seeds", [42]))
    loader_seed = rcfg.get("loader_seed", 0)

    loss_fn = SVD_LOSS_FNS[loss_key]

    # ------------------------------------------------------------------
    # Build model, data, optimizer
    # ------------------------------------------------------------------
    set_seed(model_seed)
    dataset     = instantiate(cfg.dataset)
    model       = instantiate(cfg.model)
    train_model = SvenWrapper(model, loss_fn, device)
    optimizer   = Sven(train_model, lr=lr, k=k, rtol=rtol, svd_mode=svd_mode)

    train_loader = DataLoader(
        dataset.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(loader_seed),
        drop_last=True,
    )

    n_params = train_model.n_params
    jacobian_bytes = batch_size * n_params * 4  # float32

    print(f"\n{'='*60}")
    print(f"GPU Memory Profile — Sven Optimizer")
    print(f"{'='*60}")
    print(f"  device      : {device}")
    print(f"  batch_size  : {batch_size}")
    print(f"  k           : {k}  (k/batch = {k/batch_size:.2f})")
    print(f"  rtol        : {rtol}")
    print(f"  svd_mode    : {svd_mode}")
    print(f"  loss        : {loss_key}")
    print(f"  n_params    : {n_params:,}")
    print(f"  Jacobian    : ({batch_size} × {n_params})  ≈ {_mb(jacobian_bytes)} (fp32)")
    print(f"  warmup steps: {warmup_steps}   measurement steps: {num_steps}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Profiling loop
    # ------------------------------------------------------------------
    data_iter   = iter(train_loader)
    tracker     = PhaseTracker(device)
    all_records : list[list[tuple[str, int, int]]] = []

    with torch.no_grad():
        for step_idx in range(warmup_steps + num_steps):
            try:
                xb, yb = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                xb, yb = next(data_iter)

            measuring = step_idx >= warmup_steps

            if measuring:
                tracker.reset()

            # ---- Phase 0: baseline (model on GPU, no batch yet) ----------
            if measuring:
                tracker.checkpoint("0. Baseline (model on GPU)")

            # ---- Phase 1: batch to GPU -----------------------------------
            xb, yb = xb.to(device), yb.to(device)
            if measuring:
                tracker.checkpoint("1. Batch loaded to GPU")

            # ---- Phase 2: Jacobian computation (loss_and_grad) -----------
            _batch_losses, _preds = train_model.loss_and_grad((xb, yb))
            if measuring:
                tracker.checkpoint("2. After loss_and_grad (Jacobian built)")

            # ---- Phase 3: SVD (pinv) -------------------------------------
            jacobian = train_model.grads   # (B, P) — still allocated
            VhT, S_inv, U_T = pinv(
                jacobian, k=optimizer.k, rtol=optimizer.rtol, mode=optimizer.svd_mode
            )
            if measuring:
                tracker.checkpoint("3. After pinv (SVD computed)")

            # ---- Phase 4: delete Jacobian + empty cache ------------------
            del jacobian
            torch.cuda.empty_cache()
            if measuring:
                tracker.checkpoint("4. After del Jacobian + empty_cache")

            # ---- Phase 5: parameter update -------------------------------
            residuals = train_model.residuals
            optimizer._update_params(U_T, S_inv, VhT, residuals)
            if measuring:
                tracker.checkpoint("5. After parameter update")

            # ---- Phase 6: full cleanup -----------------------------------
            del VhT, S_inv, U_T
            del train_model.residuals, train_model.grads, train_model.losses
            torch.cuda.empty_cache()
            if measuring:
                tracker.checkpoint("6. After full cleanup")
                step_num = step_idx - warmup_steps + 1
                print(f"--- Step {step_num}/{num_steps} ---")
                tracker.print_table()
                all_records.append(list(tracker.records))

    # ------------------------------------------------------------------
    # Aggregate across steps
    # ------------------------------------------------------------------
    if len(all_records) > 1:
        import numpy as np

        n_phases = len(all_records[0])
        w = 44
        print("=" * (w + 34))
        print(f"Mean across {num_steps} steps")
        print("=" * (w + 34))
        print(f"{'Phase':<{w}} {'Mean current':>14} {'Mean peak':>14}")
        print("-" * (w + 30))
        for i in range(n_phases):
            label     = all_records[0][i][0]
            mean_cur  = float(np.mean([r[i][1] for r in all_records]))
            mean_peak = float(np.mean([r[i][2] for r in all_records]))
            print(f"{label:<{w}} {_mb(mean_cur):>14} {_mb(mean_peak):>14}")
        print()

        # Highlight the phase with the highest peak
        max_phase_idx  = int(np.argmax([
            np.mean([r[i][2] for r in all_records]) for i in range(n_phases)
        ]))
        max_phase_name = all_records[0][max_phase_idx][0]
        max_peak_mb    = float(np.mean([r[max_phase_idx][2] for r in all_records])) / 1024**2
        print(f"Peak memory phase : {max_phase_name}")
        print(f"Peak memory usage : {max_peak_mb:.1f} MB\n")


if __name__ == "__main__":
    main()
