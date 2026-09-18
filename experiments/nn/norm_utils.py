"""Norm-layer statistics policy for the training loops (C-E2).

The module-level counterpart of ``SvenWrapper.no_norm_stat_updates`` /
``_frozen_norm_stats`` / ``_eval_mode`` (``sven/sven/nn/sven_wrapper.py``), for
the plain ``nn.Module`` that the standard / LBFGS / HIG / Jacobian-descent loops
train directly.  Both repos must apply the same policy or the CIFAR comparison
is between two different algorithms (F2/F3).

Two policies, named by ``bn_mode`` (CONTRACTS.md):

``batch``
    Train with batch statistics and update the running statistics from the
    *training* batch **exactly once per optimizer step**.  Every repeated
    forward inside one step (LBFGS closure calls after the first, a line
    search) runs under :func:`no_norm_stat_updates`, which suppresses the write
    *without* changing the normalisation -- unlike ``.eval()``, which would
    switch to the running statistics and hence change the update itself.
``frozen``
    Norm layers stay in eval mode for training and evaluation alike
    (:func:`freeze_norm_layers`), so the (pretrained) running statistics are
    used throughout: the fine-tune study, open decision O2.

Evaluation always runs in eval mode (:func:`eval_mode`) and writes no buffer.
"""

from contextlib import contextmanager
from typing import Iterator

import torch.nn as nn
from torch.nn.modules.batchnorm import _NormBase

__all__ = [
    "norm_stat_modules",
    "no_norm_stat_updates",
    "freeze_norm_layers",
    "eval_mode",
]


def norm_stat_modules(module: nn.Module) -> list[_NormBase]:
    """Every norm submodule of *module* that owns running statistics.

    ``_NormBase`` covers BatchNorm1d/2d/3d and InstanceNorm, including the
    ``torch.func``-compatible replacement in ``experiments/nn/batchnorm.py``.
    LayerNorm / RMSNorm keep no statistics and normalise identically in both
    modes, so they are deliberately not listed (nanoGPT is unaffected).
    """
    return [
        mod
        for mod in module.modules()
        if isinstance(mod, _NormBase) and mod.running_mean is not None
    ]


@contextmanager
def no_norm_stat_updates(module: nn.Module) -> Iterator[None]:
    """Suppress running-statistic writes inside, leaving the normalisation alone.

    Sets ``track_running_stats = False`` on every norm module that owns running
    statistics: a train-mode forward then still normalises with **batch**
    statistics (``F.batch_norm`` receives ``None`` buffers, so nothing is
    written and ``num_batches_tracked`` does not advance either) and an
    eval-mode one still normalises with the running statistics.  Each module's
    own previous flag *and* training mode are restored exactly -- never a
    blanket ``.train()``, which would wake up layers the caller froze.
    """
    saved = [
        (mod, mod.track_running_stats, mod.training) for mod in norm_stat_modules(module)
    ]
    for mod, _, _ in saved:
        mod.track_running_stats = False
    try:
        yield
    finally:
        for mod, track, training in saved:
            mod.track_running_stats = track
            mod.training = training


def freeze_norm_layers(module: nn.Module) -> list[_NormBase]:
    """Put every norm layer in eval mode (``bn_mode="frozen"``).

    Called after each ``model.train()``, which would otherwise wake the norm
    layers up again.  Returns the layers it froze (empty for a norm-free net).
    """
    frozen = [mod for mod in module.modules() if isinstance(mod, _NormBase)]
    for mod in frozen:
        mod.training = False
    return frozen


@contextmanager
def eval_mode(module: nn.Module) -> Iterator[None]:
    """Run the block with the whole module in eval mode, restoring every flag.

    Restores each submodule's own previous ``training`` flag, so a caller that
    had deliberately frozen part of the network gets it back unchanged.
    """
    saved = [(mod, mod.training) for mod in module.modules() if mod.training]
    for mod, _ in saved:
        mod.training = False
    try:
        yield
    finally:
        for mod, training in saved:
            mod.training = training
