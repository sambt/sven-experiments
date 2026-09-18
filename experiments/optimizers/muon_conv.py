"""Muon for convolutional (>2-D) weights -- vendored from ``torch.optim.Muon`` (C-B5, F13).

``torch.optim.Muon`` (torch 2.9.1, ``torch/optim/_muon.py``) *raises* for every
parameter with ``p.ndim != 2``, so conv kernels cannot be handed to it at all and
"Muon on a ResNet" silently degenerates into AdamW plus one ``fc`` layer (F13).
This module vendors the algorithm -- the same way SOAP is vendored -- with the one
change the conv case needs:

    a weight of shape ``(out, in, *kernel)`` is flattened to the matrix
    ``(out, in * prod(kernel))``, orthogonalised there, and reshaped back.

That is the standard Muon treatment of conv kernels (Keller Jordan's reference
implementation does exactly this): each output channel's filter is one row of the
matrix, so the orthogonalisation acts on the map from the *whole* receptive field
to the output channels.  Two consequences are deliberate:

* the learning-rate adjustment (``adjust_lr_fn``) uses the **flattened** shape.
  ``torch``'s ``_adjust_lr`` reads ``param.shape[:2]``, which for a 4-D kernel
  would be ``(out, in)`` and would ignore the kernel extent; the update lives in
  the flattened space, so its RMS is set by the flattened dimensions.
* the momentum buffer has the parameter's own shape (as in ``torch``); only the
  Newton-Schulz iteration sees the 2-D view.

Everything else -- momentum/Nesterov update, the quintic Newton-Schulz iteration
in bfloat16, decoupled weight decay at the *unadjusted* lr, and the order of the
two in-place param updates -- is copied verbatim from ``torch/optim/_muon.py`` so
that on a 2-D parameter ``MuonConv`` reproduces ``torch.optim.Muon`` exactly
(``tests/test_optim_factory.py`` asserts this over several steps).
"""

import math
from typing import Optional

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.optimizer import _to_scalar   # as torch/optim/_muon.py imports it

__all__ = ["MuonConv"]

# Constants from Keller Jordan's Muon post, as in torch/optim/_muon.py.
EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


def _zeropower_via_newtonschulz(
    grad: Tensor, ns_coefficients: tuple[float, float, float], ns_steps: int, eps: float
) -> Tensor:
    """Quintic Newton-Schulz orthogonalisation.  Verbatim from ``torch/optim/_muon.py``
    (bfloat16 arithmetic included -- changing the dtype would change the result)."""
    if ns_steps >= 100:
        raise ValueError(
            "Number of steps must be less than 100 for computational efficiency"
        )
    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    # Ensure spectral norm is at most 1
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    # Perform the NS iterations
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c
        )
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


def _adjust_lr(lr: float, adjust_lr_fn: Optional[str], matrix_shape) -> float:
    """Muon's learning-rate adjustment.  Verbatim from ``torch/optim/_muon.py``, except
    that the caller passes the shape of the **flattened** matrix (see module docstring)."""
    A, B = matrix_shape[:2]

    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


class MuonConv(Optimizer):
    """Muon accepting any parameter with ``ndim >= 2``; >2-D weights are flattened
    to ``(shape[0], -1)`` for the orthogonalisation.

    Same defaults and same keyword arguments as ``torch.optim.Muon``, so the two are
    interchangeable for 2-D parameters.  1-D parameters (biases, norm weights) are
    rejected here exactly as ``torch.optim.Muon`` rejects them -- they belong in the
    AdamW half of the optimizer (see ``optim_factory.muon_param_groups``).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: Optional[str] = None,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in ["original", "match_rms_adamw"]:
            raise ValueError(
                f"Adjust learning rate function {adjust_lr_fn} is not supported"
            )

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim < 2:
                    raise ValueError(
                        "MuonConv only supports parameters with ndim >= 2 whereas we "
                        f"found a parameter with size: {p.size()}"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = _to_scalar(group["lr"])     # a 1-element tensor lr -> 0-dim, as in torch
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if torch.is_complex(p):
                    raise RuntimeError("MuonConv does not support complex parameters")
                if p.grad.is_sparse:
                    raise RuntimeError("MuonConv does not support sparse gradients")

                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        grad, memory_format=torch.preserve_format
                    )
                buf = state["momentum_buffer"]

                buf.lerp_(grad, 1 - momentum)
                update = grad.lerp(buf, momentum) if group["nesterov"] else buf

                # (out, in, *kernel) -> (out, in * prod(kernel)); a no-op for 2-D.
                matrix = update.reshape(update.shape[0], -1)
                ortho = _zeropower_via_newtonschulz(
                    matrix, group["ns_coefficients"], group["ns_steps"], group["eps"]
                )
                adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], matrix.shape)

                p.mul_(1 - lr * weight_decay)
                p.add_(ortho.reshape(p.shape), alpha=-adjusted_lr)

        return loss
