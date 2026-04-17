"""Optax-based baselines for side-by-side comparison with Sven.

Notes
-----
- The PyTorch ``baselines.py`` adds Lion, ScheduleFreeAdamW, ScheduleFreeSGD.
  optax ships ``optax.lion`` and ``optax.contrib.schedule_free_adamw`` so we
  surface those by name when available.
- LBFGS and PolyakSGD have no clean stateless optax equivalent (LBFGS needs
  closures with a line search; PolyakSGD needs the current loss value). They
  are intentionally **not** supported in the JAX framework; the corresponding
  entries in hyperparameter sweeps are simply skipped with a warning.
"""

from __future__ import annotations

from typing import Any

import optax


_UNSUPPORTED_OPTIMIZERS = {"LBFGS", "PolyakSGD", "Muon"}


def _maybe(name: str):
    """Return the optax builder for ``name`` or ``None`` if it's not shipped."""
    # optax.contrib lives behind an import in newer versions
    if name == "ScheduleFreeAdamW":
        try:
            from optax.contrib import schedule_free_adamw  # type: ignore
            return schedule_free_adamw
        except Exception:
            return None
    if name == "ScheduleFreeSGD":
        try:
            from optax.contrib import schedule_free_sgd  # type: ignore
            return schedule_free_sgd
        except Exception:
            return None
    return None


def build_standard_optimizer(
    optim_name: str,
    lr: float,
    weight_decay: float = 0.0,
    **kwargs: Any,
) -> optax.GradientTransformation:
    """Construct a standard optax optimizer by name."""
    name = optim_name
    if name in _UNSUPPORTED_OPTIMIZERS:
        raise NotImplementedError(
            f"{name} is not supported in the JAX experiment framework. "
            f"It is skipped automatically in scans."
        )
    if name == "Adam":
        return optax.adam(learning_rate=lr)
    if name == "AdamW":
        return optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    if name == "SGD":
        return optax.sgd(learning_rate=lr)
    if name == "RMSprop":
        return optax.rmsprop(learning_rate=lr)
    if name == "Lion":
        if weight_decay:
            return optax.lion(learning_rate=lr, weight_decay=weight_decay)
        return optax.lion(learning_rate=lr)
    builder = _maybe(name)
    if builder is not None:
        return builder(learning_rate=lr)
    raise ValueError(f"Unknown / unsupported optimizer: {name}")


SUPPORTED_STANDARD_OPTIMIZERS = {
    "Adam", "AdamW", "SGD", "RMSprop", "Lion",
    "ScheduleFreeAdamW", "ScheduleFreeSGD",
}
