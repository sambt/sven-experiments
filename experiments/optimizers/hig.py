"""
Half-Inverse Gradients (HIG) optimizer for Sven comparison experiments.

Reference: Schnell et al., "Half-Inverse Gradients for Physical Deep Learning",
ICLR 2022. https://arxiv.org/abs/2203.10131

The HIG update is a generalisation of the GD <-> Gauss-Newton family:
    Δθ(η, κ) = -η · J^κ · g
where J = U Λ V^T (SVD of the stacked output Jacobian) and J^κ := V Λ^κ U^T.

  κ =  1  → standard gradient descent
  κ = -1  → Gauss-Newton (full pseudoinverse)
  κ = -½  → HIG (used here)

Unlike SvenWrapper (which Jacobian-differentiates the *loss*), HIGWrapper
differentiates the *model output* f(x;θ), then multiplies by ∂L/∂ŷ separately.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.nn.utils import parameters_to_vector


class HIGWrapper:
    """Functional wrapper around a PyTorch model for HIG output-Jacobian computation.

    Computes two quantities per batch and stores them for HIGOptimizer.step():
      - output_jac : (B·|y|, |θ|)  stacked output Jacobians  ∂f(xᵢ;θ)/∂θ
      - loss_grad  : (B·|y|,)      stacked loss gradients     ∂Lᵢ/∂ŷᵢ

    Args:
        model:   The PyTorch model to wrap.
        loss_fn: Per-sample loss function ``(y_pred, *args) -> (B,)``.
        device:  Device for model and parameter tensors.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[..., torch.Tensor],
        device: torch.device | str,
    ) -> None:
        self.model: nn.Module = model.to(device)
        self.device: torch.device = torch.device(device) if isinstance(device, str) else device
        self.loss_fn = loss_fn

        self.param_shapes: list[tuple[str, torch.Size, int]] = [
            (name, p.shape, p.numel()) for name, p in model.named_parameters()
        ]
        self.params: torch.Tensor = self._tie_parameters_to_flat()
        self.n_params: int = self.params.shape[0]

        # Populated by output_and_loss_grad(), consumed by HIGOptimizer.step()
        self.output_jac: torch.Tensor = torch.empty(0, device=self.device)
        self.loss_grad: torch.Tensor = torch.empty(0, device=self.device)
        self.losses: torch.Tensor = torch.empty(0, device=self.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _func_call(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Functional forward pass: reconstructs param dict from flat vector."""
        param_dict: dict[str, torch.Tensor] = {}
        start = 0
        for name, shape, size in self.param_shapes:
            param_dict[name] = params[start : start + size].view(shape)
            start += size
        for name, buf in self.model.named_buffers():
            param_dict[name] = buf
        return functional_call(self.model, param_dict, x)

    def _forward(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass used by jacrev: returns (B, out_dim) outputs."""
        out = self._func_call(params, x)
        # Ensure 2-D output even for scalar-per-sample models
        return out if out.dim() > 1 else out.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without gradient tracking."""
        return self._func_call(self.params, x)

    def output_and_loss_grad(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute output Jacobian and loss gradient; store for HIGOptimizer.

        Algorithm:
          1. J = ∂f(x;θ)/∂θ  via jacrev → shape (B, |y|, |θ|)
          2. g = ∂L_i/∂ŷ_i   via autograd on detached predictions → shape (B, |y|)

        Args:
            batch: ``(x, y, ...)`` tensors.

        Returns:
            ``(per_sample_losses, predictions)`` — both detached.
        """
        x, *args = batch

        # 1. Output Jacobian: torch.func.jacrev creates its own grad context
        #    and works correctly inside an outer no_grad block.
        J = torch.func.jacrev(self._forward, argnums=0)(self.params, x)
        # J: (B, out_dim, n_params)

        # 2. Model prediction (detached — used for loss gradient + returned)
        y_pred = self._forward(self.params, x).detach()  # (B, out_dim)

        # 3. Loss gradient ∂L_i/∂ŷ_i via standard autograd on the output.
        #    torch.enable_grad() overrides any outer no_grad context.
        with torch.enable_grad():
            y_for_grad = y_pred.detach().requires_grad_(True)
            losses = self.loss_fn(y_for_grad, *args)  # (B,)
            losses.sum().backward()
            g = y_for_grad.grad.detach()  # (B, out_dim)

        B, out_dim, n_params = J.shape
        self.output_jac = J.reshape(B * out_dim, n_params).detach()
        self.loss_grad = g.reshape(-1).detach()
        self.losses = losses.detach()

        return self.losses, y_pred

    # ------------------------------------------------------------------
    # Parameter management (mirrors SvenWrapper)
    # ------------------------------------------------------------------

    def _tie_parameters_to_flat(self) -> torch.Tensor:
        """Flatten all model parameters and rebind them as views into the flat vector."""
        flat = parameters_to_vector(self.model.parameters()).detach()

        start = 0
        for name, p in self.model.named_parameters():
            n = p.numel()
            view = flat[start : start + n].view_as(p)

            mod: nn.Module = self.model
            *prefix, leaf = name.split(".")
            for part in prefix:
                mod = getattr(mod, part)
            mod._parameters[leaf] = nn.Parameter(view, requires_grad=False)
            start += n

        return flat


class HIGOptimizer:
    """Half-Inverse Gradients optimizer: κ = -½ SVD of the output Jacobian.

    Update: Δθ = -η · V · Σ^{-½} · U^T · g
    where J_stacked = U Σ V^T and singular values below tau * S_max are zeroed.

    Must be paired with HIGWrapper: call wrapper.output_and_loss_grad(batch)
    before each optimizer.step().

    Args:
        model: HIGWrapper instance (provides output_jac and loss_grad).
        lr:    Learning rate η.
        tau:   Relative truncation threshold — singular values below tau * S_max
               are zeroed (matches rcond convention in paper's TF implementation).
    """

    def __init__(self, model: HIGWrapper, lr: float, tau: float = 1e-4) -> None:
        self.model = model
        self.lr = lr
        self.tau = tau

    @torch.no_grad()
    def step(self) -> None:
        J = self.model.output_jac  # (B·|y|, |θ|)
        g = self.model.loss_grad   # (B·|y|,)

        # SVD: J = U Σ Vʰ  (economy / thin SVD)
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)

        # Apply κ = -½ with relative truncation: zero out singular values < tau * S_max
        # (matches the rcond convention used in the paper's TF implementation)
        S_neg_half = torch.where(S > self.tau * S.max(), S.pow(-0.5), torch.zeros_like(S))

        # Δθ = Vʰᵀ · (S^{-½} · (Uᵀ · g))
        delta = Vh.T @ (S_neg_half * (U.T @ g))  # (|θ|,)

        self.model.params.add_(delta, alpha=-self.lr)
