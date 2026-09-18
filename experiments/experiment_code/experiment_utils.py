import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any
import os
import pandas as pd
import fcntl
import tempfile
import shutil
import time
import random
from sven.opt import PolyakSGD
from experiments.nn.norm_utils import eval_mode, freeze_norm_layers, no_norm_stat_updates

# The standard-optimizer factory moved to optim_factory.py (C-B2 / C-B5 / C-B6);
# the names are re-exported unchanged, so every call site that imports them from
# here -- including experiments/optimizer_profile.py -- keeps working.
from .optim_factory import (  # noqa: F401  (re-export)
    _CUSTOM_OPTIMIZERS, _DEFAULT_WEIGHT_DECAY, _KFACOptimizer, _CombinedOptimizer,
    _REMOVED_OPTIMIZERS, build_standard_optimizer, get_muon_variant,
    resolve_weight_decay,
)

def set_seed(seed: int, deterministic: bool = False):
    """
    Set random seeds for reproducible experiments.

    Args:
        seed: The random seed to use for all random number generators.
        deterministic: If True, enables CUDA deterministic algorithms for full
            reproducibility. This may impact performance. Default is False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

def listify(settings):
    if type(settings) is list or type(settings) is tuple:
        return settings
    else:
        return [settings]
    
def process_hparam_config(cfg) -> dict[str,Iterable]:
    output = {}
    if "batch_size" not in cfg:
        output['batch_size'] = listify(32)
        print("No batch size specified; defaulting to", output['batch_size'])
    else:
        output['batch_size'] = listify(cfg["batch_size"])

    if "k_fractions" not in cfg and "k_values" not in cfg:
        output['k_fractions'] = [0.1, 0.25, 0.5, 0.75, 1.0]
        print("No k_values or k_fractions specified; defaulting to fractions = ", output['k_fractions'])
    else:
        assert ("k_values" in cfg) ^ ("k_fractions" in cfg), "Specify either k_values or k_fractions, not both."
        if "k_fractions" in cfg:
            output['k_fractions'] = listify(cfg["k_fractions"])
        else:
            output['k_values'] = listify(cfg["k_values"])

    if "lrs" not in cfg:
        output['lrs'] = [0.01, 0.1, 0.5, 1.0]
        print("No learning rates specified; defaulting to", output['lrs'])
    else:
        output['lrs'] = listify(cfg["lrs"])

    if "rtol" not in cfg:
        output['rtol'] = listify(1e-3)
        print("No rtol specified; defaulting to", output['rtol'])
    else:
        output['rtol'] = listify(cfg["rtol"])

    if "svd_mode" not in cfg:
        output['svd_mode'] = listify('randomized')
        print("No SVD mode specified; defaulting to 'randomized'")
    else:
        output['svd_mode'] = listify(cfg["svd_mode"])

    if "lrs_standard" not in cfg:
        output['lrs_standard'] = [1e-4,1e-3,1e-2,1e-1]
        print("No learning rates for standard optimizers specified; defaulting to", output['lrs_standard'])
    else:
        output['lrs_standard'] = listify(cfg["lrs_standard"])

    if "optimizers_standard" not in cfg:
        output['optimizers_standard'] = ['Adam','AdamW','SGD','RMSprop','Muon']
        print("No standard optimizers specified; defaulting to", output['optimizers_standard'])
    else:
        output['optimizers_standard'] = listify(cfg["optimizers_standard"])

    if "microbatch_sizes" in cfg:
        output['microbatch_sizes'] = listify(cfg["microbatch_sizes"])
    else:
        output['microbatch_sizes'] = [None]

    if "param_fractions" in cfg:
        output['param_fractions'] = listify(cfg["param_fractions"])
    else:
        output['param_fractions'] = [None]

    # kappa sweep for the residual decomposition (Sven only). Default [2.0] so
    # non-kappa configs are unaffected (kappa==2.0 leaves run_id unchanged).
    output['kappas'] = listify(cfg.get("kappa", 2.0))

    # LBFGS-specific hyperparameters (only used when "LBFGS" is in optimizers_standard)
    # Separate LR list for LBFGS since it typically needs much larger LRs than Adam/SGD
    output['lrs_lbfgs'] = listify(cfg.get("lrs_lbfgs", output['lrs_standard']))
    output['lbfgs_max_iter'] = listify(cfg.get("lbfgs_max_iter", 20))
    output['lbfgs_history_size'] = listify(cfg.get("lbfgs_history_size", 100))
    output['lbfgs_line_search_fn'] = listify(cfg.get("lbfgs_line_search_fn", "strong_wolfe"))


    # Weight decay sweep.  Default [None] = "the optimizer's own default" (see
    # resolve_weight_decay): AdamW then runs with its PyTorch default 0.01 -- weight decay
    # ON, which is the point of AdamW -- instead of the explicit 0.0 the old default
    # forced on it (which made AdamW bit-identical to Adam in every headline scan).
    # Non-zero values are only applied to AdamW / Muon in the scan loop.
    output['weight_decays'] = listify(cfg.get("weight_decays", [None]))

    # PolyakSGD-specific hyperparameters (only used when "PolyakSGD" is in optimizers_standard)
    # No LR sweep — the step size is computed automatically from the loss
    output['polyak_f_star'] = listify(cfg.get("polyak_f_star", 0.0))
    output['polyak_max_lr'] = listify(cfg.get("polyak_max_lr", 1.0))
    output['polyak_eps'] = listify(cfg.get("polyak_eps", 1e-8))

    # Jacobian Descent (torchjd) hyperparameters (mode "jd"/"all")
    output['lrs_jd'] = listify(cfg.get("lrs_jd", output['lrs_standard']))
    output['aggregators_jd'] = listify(cfg.get("aggregators_jd", ["UPGrad"]))
    output['inner_optimizers_jd'] = listify(cfg.get("inner_optimizers_jd", ["Adam"]))

    # Half-Inverse Gradients hyperparameters (mode "hig"/"all")
    output['lrs_hig'] = listify(cfg.get("lrs_hig", output['lrs']))
    output['tau_hig'] = listify(cfg.get("tau_hig", [1e-4]))

    return output

def _compute_per_model_acc(ypred, yb):
    """Compute per-model accuracy for (M, B, C) predictions. Returns list of M floats."""
    preds = torch.argmax(ypred, dim=2)  # (M, B)
    return (preds == yb.unsqueeze(0)).float().mean(dim=1).tolist()

def _is_closure_optimizer(optimizer):
    """Check if an optimizer requires a closure (e.g. LBFGS)."""
    return isinstance(optimizer, torch.optim.LBFGS) or isinstance(optimizer, PolyakSGD)


def _is_schedule_free_optimizer(optimizer):
    """Check if an optimizer uses the schedule-free interface (train/eval modes)."""
    return getattr(optimizer, 'schedule_free', False)


# ---------------------------------------------------------------------------
# Training-loop infrastructure, shared by all four loops
#
#   C-R1  caller-owned `losses` dict + DivergedError early stop
#   C-E1  one example-weighted `evaluate()` for val / test / train_eval
#   C-E2  norm-statistics policy (`bn_mode`), via experiments.nn.norm_utils
#   C-E3  LBFGS records its FIRST closure call; `train_eval` curve
#   C-E4  optional `eval_every_steps`
#   C-E5  `summarize_curves()`
#   C-T1  synchronised batch timers, per-epoch `train_times`
#   C-L3  checkpointer hooks
# ---------------------------------------------------------------------------


class DivergedError(RuntimeError):
    """A training batch loss went non-finite, so the run stops here (C-R1).

    ``step`` is the optimizer step whose loss was non-finite (0-based, counted
    over the whole run): the runner records it as ``diverged_at_step`` and
    writes a ``status: diverged`` record with the partial curves it owns.

    Beware when classifying failures: this subclasses ``RuntimeError``, so an
    ``isinstance``/message-based classifier must test ``DivergedError`` FIRST.
    ``.losses`` is the loop's (caller-owned) curve dict, already passed through
    :func:`_finish_losses`, so a caller that forgot to pass ``losses=`` still
    gets the partial curves.
    """

    def __init__(self, step: int, value: float | None = None) -> None:
        self.step = int(step)
        self.value = None if value is None else float(value)
        self.losses: dict[str, Any] | None = None
        msg = f"non-finite training loss at optimizer step {self.step}"
        if value is not None:
            msg += f" (loss={value})"
        super().__init__(msg)


def _cuda_active(device) -> bool:
    """True iff the batch timers must synchronise the device (C-T1)."""
    return torch.cuda.is_available() and torch.device(device).type == "cuda"


def _sync(cuda: bool) -> None:
    """``torch.cuda.synchronize()`` when CUDA is in use, otherwise nothing."""
    if cuda:
        torch.cuda.synchronize()


def _ap(losses, key, value):
    """Append to a curve of the CALLER-owned ``losses`` dict (C-R1).

    The dict is filled in place, so everything collected before an exception
    survives it -- which is what lets a failed run still be a record.
    """
    losses.setdefault(key, []).append(value)


def _safe_mean(values):
    """Mean of a possibly empty or absent curve, without numpy's empty warning."""
    return float(np.mean(values)) if values else float('nan')


def _scalar_loss(loss):
    """A scalar to call ``.backward()`` on, for a per-sample or mean-reduced loss.

    The standard loop accepts either: ``STANDARD_LOSS_FNS`` (mean-reduced, what
    the runner passes today) or ``SVD_LOSS_FNS`` (per-sample, which makes the
    online train loss exactly example-weighted even with ``drop_last=False``).
    """
    return loss if loss.dim() == 0 else loss.mean()


def _correct_count(ypred, yb):
    """``(#correct, #predictions)`` for (B, C) or multi-model (M, B, C) logits.

    Counts, not a mean of batch means: with a ragged last batch the two differ
    (F10), which is why the old ``_compute_acc`` batch-mean helper is gone.
    ``ypred`` are raw network outputs (logits) and no softmax is applied:
    softmax preserves the ordering within each row (exp is strictly increasing
    and the per-row normaliser is shared), so argmax -- and hence accuracy --
    is the same with or without it for every loss (CE, label_regression,
    brier), and raw argmax is the numerically safer choice (float32 softmax can
    collapse near-tied logits).
    """
    if ypred.dim() == 3:
        preds = torch.argmax(ypred, dim=2)  # (M, B)
        return int((preds == yb.unsqueeze(0)).sum().item()), preds.numel()
    preds = torch.argmax(ypred, dim=1)
    return int((preds == yb).sum().item()), preds.numel()


def _loss_sum_weight(batch_loss, yb, is_lm=False):
    """Example-weighted contribution of one batch: ``(weighted_sum, weight)``.

    ``batch_loss`` is either per-sample -- ``(B,)``, or ``(M, B)`` for the
    multi-model nets -- or a 0-dim mean over the batch.  For a mean-reduced
    loss ``batch_mean * n_b`` summed over batches and divided by ``N`` is
    exactly the mean over examples, which is why both forms are accepted.
    ``is_lm`` switches to token weighting: ``SVD_LOSS_FNS["lm_ce"]`` returns a
    per-sequence value that is itself a mean over that sequence's ``T``
    targets, and ``STANDARD_LOSS_FNS["lm_ce"]`` a mean over all ``B*T``.
    """
    n_rows = yb.shape[0]
    per_row = (yb.numel() // n_rows) if is_lm else 1
    if batch_loss.dim() == 0:
        weight = n_rows * per_row
        return float(batch_loss.item()) * weight, weight
    return (
        float(batch_loss.detach().double().sum().item()) * per_row,
        batch_loss.numel() * per_row,
    )


def _batch_loss_stats(batch_loss, yb, is_lm=False):
    """``(batch_mean, weighted_sum, weight)``; ``batch_mean = sum / weight``."""
    total, weight = _loss_sum_weight(batch_loss, yb, is_lm)
    return total / weight, total, weight


def _forward_module(forward_fn):
    """The ``nn.Module`` behind an evaluation callable, or ``None``.

    Defence in depth for :func:`evaluate`: the svd / hig loops evaluate through
    a *bound method* (``SvenWrapper.evaluate``, ``HIGWrapper.evaluate``), so
    without this the eval-mode / no-buffer-write guarantee would rest entirely
    on the wrapper -- a repo this track does not own, being changed in
    parallel. Resolving the wrapper's ``.model`` here makes the guarantee hold
    even if a wrapper's ``evaluate`` forgets eval mode (F2), and is idempotent
    with the wrapper's own eval-mode context when it does not.
    """
    if isinstance(forward_fn, nn.Module):
        return forward_fn
    owner = getattr(forward_fn, "__self__", None)  # bound method -> its object
    if isinstance(owner, nn.Module):
        return owner
    inner = getattr(owner, "model", None) if owner is not None else None
    return inner if isinstance(inner, nn.Module) else None


@contextmanager
def _eval_context(module):
    """Eval mode with no running-stat writes, or nothing when there is no module.

    ``module is None`` means no module could be found behind the forward
    callable (:func:`_forward_module`), so it owns its own mode.
    """
    if module is None:
        yield
        return
    with eval_mode(module), no_norm_stat_updates(module):
        yield


@torch.no_grad()
def evaluate(forward_fn, per_sample_loss_fn, loader, device, *, track_acc=False, is_lm=False):
    """Example-weighted evaluation of one split in eval mode (C-E1).

    The single evaluation path of all four training loops, used for validation,
    test and ``train_eval``, before training and after every epoch.

    ``forward_fn`` is either an ``nn.Module`` or a bound method of one (or of a
    wrapper holding one as ``.model``, e.g. the Sven / HIG wrappers'
    ``evaluate``): whatever :func:`_forward_module` finds is put in eval mode
    here, with every submodule's previous flag restored afterwards.
    ``per_sample_loss_fn`` may return per-sample losses or a 0-dim batch mean
    (:func:`_loss_sum_weight`).  The reported loss is the mean over *examples*
    (over *tokens* when ``is_lm``) and the accuracy is #correct / #predictions,
    neither of them a mean of batch means (F10).  Nothing is mutated: eval mode
    plus :func:`~experiments.nn.norm_utils.no_norm_stat_updates` on top, so no
    norm buffer is written even if a caller left a layer in train mode (C-E2).

    Returns ``{"loss": float, "acc": float | None, "n": int}``, ``n`` being the
    number of examples seen (rows, not tokens and not model copies).
    """
    module = _forward_module(forward_fn)
    loss_total = 0.0
    weight = 0
    correct = 0
    n_pred = 0
    n_examples = 0
    with _eval_context(module):
        for batch in loader:
            xb, yb = batch[0].to(device), batch[1].to(device)
            ypred = forward_fn(xb)
            total, w = _loss_sum_weight(per_sample_loss_fn(ypred, yb), yb, is_lm)
            loss_total += total
            weight += w
            n_examples += yb.shape[0]
            if track_acc and not is_lm:
                c, n = _correct_count(ypred, yb)
                correct += c
                n_pred += n
    return {
        "loss": loss_total / weight if weight else float('nan'),
        "acc": (correct / n_pred) if (track_acc and n_pred) else None,
        "n": n_examples,
    }


_EVAL_SPLITS = ("val", "test", "train_eval")


def _record_evals(losses, forward_fn, loss_fn, loaders, device, *, track_acc, is_lm, suffix=""):
    """Evaluate every split with a loader and append to ``{split}{suffix}`` curves.

    ``suffix=""`` gives the per-epoch curves ``val`` / ``val_acc`` / ``test`` /
    ``test_acc`` / ``train_eval`` (index 0 = untrained); ``suffix="_step"`` the
    C-E4 step-based ones.
    """
    for split in _EVAL_SPLITS:
        loader = loaders.get(split)
        if loader is None:
            continue
        out = evaluate(forward_fn, loss_fn, loader, device, track_acc=track_acc, is_lm=is_lm)
        _ap(losses, f"{split}{suffix}", out["loss"])
        if out["acc"] is not None:
            _ap(losses, f"{split}{suffix}_acc", out["acc"])


def _ckpt_module(model):
    """The ``nn.Module`` to checkpoint (C-L3).

    The Sven and HIG wrappers are not modules; they hold theirs as ``.model``,
    whose parameters are *views* into the wrapper's flat vector, so its
    ``state_dict()`` always reflects the optimizer's updates.
    """
    if isinstance(model, nn.Module):
        return model
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        return inner
    raise TypeError(f"cannot find the nn.Module to checkpoint on {type(model).__name__}")


def _param_norm(model):
    """L2 norm of all parameters; the wrappers' flat vector when there is one."""
    params = getattr(model, "params", None)
    if isinstance(params, torch.Tensor):
        return params.norm().item()
    with torch.no_grad():
        return torch.cat([p.detach().flatten() for p in model.parameters()]).norm().item()


def _apply_log_schedule(optimizer, log_schedule, step):
    """C-L2/C-L1: tell a Sven optimizer whether to log this step.

    Set unconditionally (harmless on a torch optimizer, which just grows the
    attribute) and deliberately NOT guarded by ``hasattr``: a silent no-op here
    means every Sven step logs the full pre-cut spectrum, which is the npz
    blow-up C-L1/C-L2 exist to prevent. If the optimizer ever turns
    ``log_this_step`` into a read-only property, the ``AttributeError`` is the
    point.
    """
    if log_schedule is not None:
        optimizer.log_this_step = bool(log_schedule(step))


def _apply_bn_mode(module, bn_mode):
    """Re-apply the norm policy after a ``model.train()`` (C-E2)."""
    if bn_mode not in ("batch", "frozen"):
        raise ValueError(f"bn_mode must be 'batch' or 'frozen', got {bn_mode!r}")
    if bn_mode == "frozen":
        freeze_norm_layers(module)


class _TrainEpoch:
    """Example-weighted accumulators for one training epoch (F10, C-T1, C-E4)."""

    __slots__ = ("loss_total", "weight", "correct", "n_pred", "train_time", "eval_time")

    def __init__(self):
        self.loss_total = 0.0
        self.weight = 0
        self.correct = 0
        self.n_pred = 0
        self.train_time = 0.0   # sum of synchronised batch times (C-T1)
        self.eval_time = 0.0    # mid-epoch evaluation, excluded from both timers

    @property
    def loss(self):
        return self.loss_total / self.weight if self.weight else float('nan')

    @property
    def acc(self):
        return (self.correct / self.n_pred) if self.n_pred else None


def _pre_step_check(losses, acc, batch_loss, ypred, yb, *, step, batch_start_time, cuda,
                    track_acc=False, is_lm=False, stop_on_nonfinite=True):
    """Divergence check BEFORE the update; returns the batch-loss stats (C-R1).

    Wherever the pre-update loss is in hand before the optimizer runs -- every
    loop except the closure path, whose closure is called *inside*
    ``optimizer.step()`` -- the check happens here, so a non-finite loss never
    reaches the update. Otherwise a NaN Gram / output Jacobian / curvature
    estimate surfaces as an opaque ``LinAlgError`` out of Sven's ``eigh``,
    HIG's solve or K-FAC's inverse instead of ``DivergedError(step)``, and the
    Sven / HIG / K-FAC families lose ``diverged_at_step`` (F6).

    The stats are handed on to :func:`_record_train_batch`, so the device sync
    behind ``.item()`` happens once per batch either way. On divergence the
    batch is recorded (with its time measured up to here -- it never finished)
    before the error is raised, so the partial curve shows where it stopped.
    """
    stats = _batch_loss_stats(batch_loss, yb, is_lm)
    if stop_on_nonfinite and not np.isfinite(stats[0]):
        _sync(cuda)
        # records the non-finite batch and then raises DivergedError(step)
        _record_train_batch(losses, acc, batch_loss, ypred, yb,
                            batch_time=time.perf_counter() - batch_start_time, step=step,
                            track_acc=track_acc, is_lm=is_lm, stop_on_nonfinite=True,
                            stats=stats)
    return stats


def _record_train_batch(losses, acc, batch_loss, ypred, yb, *, batch_time, step,
                        track_acc=False, is_lm=False, stop_on_nonfinite=True, stats=None):
    """Online-train bookkeeping for one batch; returns its mean loss.

    The batch loss is the **pre-update** loss in every loop (for LBFGS that is
    its first closure call, C-E3), accumulated example-weighted (F10).  A
    non-finite value ends the run with :class:`DivergedError` instead of
    burning the remaining epochs on NaN (C-R1); it is recorded first, so the
    partial curve shows where it happened.  ``stats`` are the stats
    :func:`_pre_step_check` already computed, if it ran (only the closure path
    leaves them ``None``, and only there is this check post-update: the
    non-finite loss it sees is then the first closure call of the step AFTER
    the one whose line search left the parameters non-finite).
    """
    mean, total, weight = _batch_loss_stats(batch_loss, yb, is_lm) if stats is None else stats
    _ap(losses, 'batch_times_train', batch_time)
    _ap(losses, 'train_batch', mean)
    acc.loss_total += total
    acc.weight += weight
    acc.train_time += batch_time
    if track_acc and not is_lm and ypred is not None:
        c, n = _correct_count(ypred, yb)
        acc.correct += c
        acc.n_pred += n
    if stop_on_nonfinite and not np.isfinite(mean):
        raise DivergedError(step, mean)
    return mean


def _step_evals(losses, acc, forward_fn, loss_fn, loaders, device, *, step, eval_every_steps,
                track_acc, is_lm, optimizer=None, is_sf=False):
    """Optional mid-epoch evaluation at optimizer-step multiples (C-E4).

    Called with ``step`` = the number of completed optimizer steps, so
    ``eval_step_idx`` is an honest optimizer-step count even for LBFGS with
    ``strong_wolfe`` (whose inner iterations are not fixed).  Schedule-free
    optimizers are toggled y -> x -> y around the evaluation and model modes are
    handled by :func:`evaluate`; the time spent is booked to ``acc.eval_time``
    and therefore excluded from ``train_times`` and ``epoch_times``.
    """
    if not eval_every_steps or step % eval_every_steps:
        return
    eval_start = time.perf_counter()
    if is_sf:
        optimizer.eval()
    _record_evals(losses, forward_fn, loss_fn, loaders, device,
                  track_acc=track_acc, is_lm=is_lm, suffix="_step")
    _ap(losses, 'eval_step_idx', step)
    if is_sf:
        optimizer.train()
    acc.eval_time += time.perf_counter() - eval_start


def _record_epoch(losses, acc, forward_fn, loss_fn, loaders, device, *, epoch_start_time,
                  track_acc, is_lm, optimizer=None, is_sf=False):
    """End-of-epoch evaluation and epoch aggregates, for every loop.

    ``train_times`` is the sum of the synchronised batch times (C-T1) and so
    contains no evaluation; ``epoch_times`` keeps its old meaning (the whole
    epoch including the end-of-epoch evaluation of every split it was given)
    minus any C-E4 mid-epoch evaluation, which belongs to neither and is
    recorded separately in ``eval_step_times`` (absent when there was none).
    ``eval_times`` is the end-of-epoch evaluation alone, so
    ``epoch_times[i] - eval_times[i]`` is the training-plus-loader time.
    """
    eval_start = time.perf_counter()
    if is_sf:
        optimizer.eval()  # switch schedule-free params from y -> x
    _record_evals(losses, forward_fn, loss_fn, loaders, device, track_acc=track_acc, is_lm=is_lm)
    if is_sf:
        optimizer.train()  # switch schedule-free params back from x -> y
    _ap(losses, 'eval_times', time.perf_counter() - eval_start)
    _ap(losses, 'train', acc.loss)
    if acc.acc is not None:
        _ap(losses, 'train_acc', acc.acc)
    _ap(losses, 'train_times', acc.train_time)
    if acc.eval_time:
        _ap(losses, 'eval_step_times', acc.eval_time)  # C-E4 mid-epoch evaluation
    _ap(losses, 'epoch_times', time.perf_counter() - epoch_start_time - acc.eval_time)


def summarize_curves(losses):
    """Final / best-epoch / mean-of-last-three validation loss (C-E5).

    The selection metric does not change (still the last epoch); these are
    recorded beside it so open decision D28 can be settled without reruns and
    so the overfitting gap (F32) is visible.  Non-finite entries never win the
    argmin.

    ``val_best_index`` is an INDEX into the ``val`` curve, not an epoch number:
    index 0 is the untrained model, so the best epoch is ``val_best_index - 1``
    and ``val_best_index == 0`` means nothing beat the untrained model.
    ``val_last3_mean`` is ``None`` for a curve shorter than 4 points (fewer
    than three *trained* epochs), rather than a mean that silently includes the
    untrained point.
    """
    curve = [float(v) for v in losses.get('val', [])]
    if not curve:
        return {}
    arr = np.asarray(curve, dtype=float)
    masked = np.where(np.isfinite(arr), arr, np.inf)
    best = int(np.argmin(masked)) if bool(np.isfinite(masked).any()) else 0
    return {
        'val_final': curve[-1],
        'val_best': curve[best],
        'val_best_index': best,
        'val_last3_mean': float(np.mean(arr[-3:])) if len(curve) >= 4 else None,
    }


@contextmanager
def _partial_record_on_divergence(losses, total_start_time):
    """Complete the caller's curve dict even when the run diverges (C-R1).

    :func:`_finish_losses` (totals, averages, the C-E5 summary) normally runs
    on the way out of a loop; on :class:`DivergedError` it runs here instead,
    so the ``status: diverged`` record the runner writes carries the same
    scalars as a finished one and a diverged grid point is never
    indistinguishable from one that never started.  The dict is also attached
    to the exception, so a caller that forgot to pass ``losses=`` can still
    recover the partial curves from ``err.losses``.
    """
    try:
        yield
    except DivergedError as err:
        err.losses = _finish_losses(losses, total_start_time)
        raise


def _finish_losses(losses, total_start_time):
    """Shared tail of the training loops: totals, averages, peak memory, C-E5.

    Written INTO the caller's dict (C-R1) and returned, so the loops' old
    ``return model, _finish_losses(...)`` still hands back the same object the
    caller passed in.
    """
    losses['total_time'] = time.perf_counter() - total_start_time
    losses['avg_epoch_time'] = _safe_mean(losses.get('epoch_times'))
    losses['avg_train_time'] = _safe_mean(losses.get('train_times'))
    losses['avg_batch_time_train'] = _safe_mean(losses.get('batch_times_train'))
    losses['avg_eval_time'] = _safe_mean(losses.get('eval_times'))
    losses.update(summarize_curves(losses))
    if torch.cuda.is_available():
        losses['peak_gpu_mem_mb'] = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.empty_cache()
    return losses


def _closure_step(optimizer, model, loss_fn, xb, yb):
    """One step of a closure optimizer (LBFGS, PolyakSGD); the FIRST call's loss.

    C-E3: the first closure evaluation is the pre-update loss at the parameters
    every other method reports, so that -- not the last line-search probe -- is
    the recorded train loss.  C-E2: only that first forward may advance the
    running statistics, so every later call in the same step runs under
    ``no_norm_stat_updates`` (which keeps batch-statistic normalisation, hence
    the same search direction, and writes no buffer).
    """
    state = {'loss': None, 'ypred': None, 'calls': 0}

    def closure():
        optimizer.zero_grad()
        first = state['calls'] == 0
        state['calls'] += 1
        if first:
            ypred = model(xb)
            loss = loss_fn(ypred, yb)
        else:
            with no_norm_stat_updates(model):
                ypred = model(xb)
                loss = loss_fn(ypred, yb)
        _scalar_loss(loss).backward()
        if first:
            state['loss'] = loss.detach()
            state['ypred'] = ypred.detach()
        return _scalar_loss(loss)

    optimizer.step(closure)
    if state['loss'] is None:  # pragma: no cover - would mean the optimizer never evaluated
        raise RuntimeError(f"{type(optimizer).__name__}.step() never called its closure")
    return state['loss'], state['ypred']


def train_loop_standard(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device,
                        track_acc=False, track_param_norm=False, is_lm=False, *,
                        losses=None, test_loader=None, train_eval_loader=None,
                        eval_every_steps=None, checkpointer=None, log_schedule=None,
                        stop_on_nonfinite=True, bn_mode="batch") -> tuple[Any, dict[str, Any]]:
    """First-order / closure optimizers (Adam, SGD, Muon, LBFGS, PolyakSGD, ...).

    ``loss_fn`` may be mean-reduced (``STANDARD_LOSS_FNS``) or per-sample
    (``SVD_LOSS_FNS``); see :func:`_loss_sum_weight`.  The keyword-only
    arguments are all optional and default to the pre-campaign behaviour except
    where the spec changes it (evaluation is now example-weighted, in eval
    mode, and includes the untrained model).
    """
    losses = {} if losses is None else losses
    cuda = _cuda_active(device)
    uses_closure = _is_closure_optimizer(optimizer)
    is_sf = _is_schedule_free_optimizer(optimizer)
    module = _ckpt_module(model)
    eval_loaders = {'val': val_loader, 'test': test_loader, 'train_eval': train_eval_loader}
    step_loaders = {'val': val_loader, 'test': test_loader}
    is_multi = None  # detected on the first forward pass
    num_models = 0
    _apply_bn_mode(module, bn_mode)

    print("Using device {}".format(device))

    # Index 0 of every evaluation curve: the untrained model, in EVAL mode (C-E2).
    if is_sf:
        optimizer.eval()
    _record_evals(losses, model, loss_fn, eval_loaders, device, track_acc=track_acc, is_lm=is_lm)
    if is_sf:
        optimizer.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    total_start_time = time.perf_counter()
    step = 0

    with _partial_record_on_divergence(losses, total_start_time):
        for epoch in tqdm(range(num_epochs)):
            epoch_start_time = time.perf_counter()
            acc = _TrainEpoch()
            epoch_pm = defaultdict(list)  # per-model metrics for this epoch
            model.train()
            _apply_bn_mode(module, bn_mode)
            for xb, yb in train_loader:
                if checkpointer is not None:
                    checkpointer.maybe_save(step, epoch, module)
                _apply_log_schedule(optimizer, log_schedule, step)
                _sync(cuda)
                batch_start_time = time.perf_counter()
                xb, yb = xb.to(device), yb.to(device)

                if uses_closure:
                    # C-E3: the first closure evaluation, not the last one (F11).
                    # The loss only exists after .step(), so the divergence check
                    # is the post-update one in _record_train_batch.
                    loss, ypred = _closure_step(optimizer, model, loss_fn, xb, yb)
                    stats = None
                else:
                    optimizer.zero_grad()
                    ypred = model(xb)
                    loss = loss_fn(ypred, yb)
                    # Nothing non-finite reaches the update (K-FAC's inverse, F6).
                    stats = _pre_step_check(losses, acc, loss, ypred, yb, step=step,
                                            batch_start_time=batch_start_time, cuda=cuda,
                                            track_acc=track_acc, is_lm=is_lm,
                                            stop_on_nonfinite=stop_on_nonfinite)
                    _scalar_loss(loss).backward()
                    optimizer.step()

                _sync(cuda)
                batch_time = time.perf_counter() - batch_start_time
                if is_multi is None:
                    # LM logits are (B, T, V) -- 3D too, but NOT multi-model; is_lm disambiguates.
                    is_multi = (ypred.dim() == 3) and not is_lm
                    if is_multi:
                        num_models = ypred.shape[0]
                        losses['num_models'] = num_models
                if is_multi:
                    with torch.no_grad():
                        pm_losses = [
                            float(_scalar_loss(loss_fn(ypred[i], yb)).item())
                            for i in range(num_models)
                        ]
                    epoch_pm['train'].append(pm_losses)
                    _ap(losses, 'train_batch_per_model', pm_losses)
                    if track_acc:
                        epoch_pm['train_acc'].append(_compute_per_model_acc(ypred, yb))
                _record_train_batch(losses, acc, loss, ypred, yb, batch_time=batch_time,
                                    step=step, track_acc=track_acc, is_lm=is_lm,
                                    stop_on_nonfinite=stop_on_nonfinite, stats=stats)
                step += 1
                _step_evals(losses, acc, model, loss_fn, step_loaders, device, step=step,
                            eval_every_steps=eval_every_steps, track_acc=track_acc, is_lm=is_lm,
                            optimizer=optimizer, is_sf=is_sf)

            _record_epoch(losses, acc, model, loss_fn, eval_loaders, device,
                          epoch_start_time=epoch_start_time, track_acc=track_acc, is_lm=is_lm,
                          optimizer=optimizer, is_sf=is_sf)
            for k, v in epoch_pm.items():
                _ap(losses, f'{k}_per_model', np.mean(v, axis=0).tolist())
            if track_param_norm:
                _ap(losses, 'param_norm', _param_norm(model))
            if checkpointer is not None:
                checkpointer.epoch_end(epoch, step, module)

    return model, _finish_losses(losses, total_start_time)


def train_loop_svd(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device,
                   track_acc=False, track_param_norm=False, is_lm=False, *,
                   losses=None, test_loader=None, train_eval_loader=None,
                   eval_every_steps=None, checkpointer=None, log_schedule=None,
                   stop_on_nonfinite=True, bn_mode="batch") -> tuple[Any, dict[str, Any], Any]:
    """Sven (``SvenWrapper``/``GramSvenWrapper`` + ``Sven``/``SvenGram``).

    The wrapper owns the norm-statistics policy inside a step (one train-mode
    forward updates the running statistics, every capture / jvp / line-search
    pass is suppressed), so this loop only has to freeze the norm layers when
    either it or the wrapper is in ``frozen`` mode, and to evaluate through
    ``model.evaluate`` (eval mode, side-effect-free).
    """
    losses = {} if losses is None else losses
    cuda = _cuda_active(device)
    module = _ckpt_module(model)
    eval_loaders = {'val': val_loader, 'test': test_loader, 'train_eval': train_eval_loader}
    step_loaders = {'val': val_loader, 'test': test_loader}
    is_multi = None  # detected on the first forward pass
    num_models = 0
    # The wrapper's own bn_mode also freezes: the runner configures it there,
    # and this loop must not leave the norm layers awake behind its back.
    _apply_bn_mode(module, bn_mode)
    if getattr(model, "bn_mode", None) == "frozen":
        freeze_norm_layers(module)

    _record_evals(losses, model.evaluate, loss_fn, eval_loaders, device,
                  track_acc=track_acc, is_lm=is_lm)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    total_start_time = time.perf_counter()
    step = 0

    # Ensure all computations are done without gradients
    with torch.no_grad(), _partial_record_on_divergence(losses, total_start_time):
        for epoch in tqdm(range(num_epochs)):
            epoch_start_time = time.perf_counter()
            acc = _TrainEpoch()
            epoch_pm = defaultdict(list)  # per-model metrics for this epoch
            for xb, yb in train_loader:
                if checkpointer is not None:
                    checkpointer.maybe_save(step, epoch, module)
                _apply_log_schedule(optimizer, log_schedule, step)
                _sync(cuda)
                batch_start_time = time.perf_counter()
                xb, yb = xb.to(device), yb.to(device)
                batch = (xb, yb)
                batch_losses, ypred = model.loss_and_grad(batch)
                # Nothing non-finite reaches the update: a NaN Gram would
                # otherwise die inside Sven's eigh/SVD, not as DivergedError (F6).
                stats = _pre_step_check(losses, acc, batch_losses, ypred, yb, step=step,
                                        batch_start_time=batch_start_time, cuda=cuda,
                                        track_acc=track_acc, is_lm=is_lm,
                                        stop_on_nonfinite=stop_on_nonfinite)
                optimizer.step(batch)
                _sync(cuda)
                batch_time = time.perf_counter() - batch_start_time
                if is_multi is None:
                    is_multi = (ypred.dim() == 3) and not is_lm
                    if is_multi:
                        num_models = ypred.shape[0]
                        losses['num_models'] = num_models
                if is_multi:
                    pm_losses = batch_losses.reshape(num_models, -1).mean(dim=1).tolist()
                    epoch_pm['train'].append(pm_losses)
                    _ap(losses, 'train_batch_per_model', pm_losses)
                    if track_acc:
                        epoch_pm['train_acc'].append(_compute_per_model_acc(ypred, yb))
                _record_train_batch(losses, acc, batch_losses, ypred, yb, batch_time=batch_time,
                                    step=step, track_acc=track_acc, is_lm=is_lm,
                                    stop_on_nonfinite=stop_on_nonfinite, stats=stats)
                step += 1
                _step_evals(losses, acc, model.evaluate, loss_fn, step_loaders, device, step=step,
                            eval_every_steps=eval_every_steps, track_acc=track_acc, is_lm=is_lm)

            _record_epoch(losses, acc, model.evaluate, loss_fn, eval_loaders, device,
                          epoch_start_time=epoch_start_time, track_acc=track_acc, is_lm=is_lm)
            for k_name, v in epoch_pm.items():
                _ap(losses, f'{k_name}_per_model', np.mean(v, axis=0).tolist())
            if track_param_norm:
                _ap(losses, 'param_norm', _param_norm(model))
            if checkpointer is not None:
                checkpointer.epoch_end(epoch, step, module)

    return model, _finish_losses(losses, total_start_time), optimizer


def train_loop_hig(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device,
                   track_acc=False, track_param_norm=False, is_lm=False, *,
                   losses=None, test_loader=None, train_eval_loader=None,
                   eval_every_steps=None, checkpointer=None, log_schedule=None,
                   stop_on_nonfinite=True, bn_mode="batch") -> tuple[Any, dict[str, Any]]:
    """Half-Inverse Gradients (``HIGWrapper`` + ``HIGOptimizer``).

    Mirrors :func:`train_loop_svd`: ``model.output_and_loss_grad(batch)``
    (output Jacobian + loss gradient, no backward) then ``optimizer.step()``.
    ``loss_fn`` is the per-sample loss.
    """
    losses = {} if losses is None else losses
    cuda = _cuda_active(device)
    module = _ckpt_module(model)
    eval_loaders = {'val': val_loader, 'test': test_loader, 'train_eval': train_eval_loader}
    step_loaders = {'val': val_loader, 'test': test_loader}
    _apply_bn_mode(module, bn_mode)

    _record_evals(losses, model.evaluate, loss_fn, eval_loaders, device,
                  track_acc=track_acc, is_lm=is_lm)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    total_start_time = time.perf_counter()
    step = 0

    with _partial_record_on_divergence(losses, total_start_time):
        for epoch in tqdm(range(num_epochs)):
            epoch_start_time = time.perf_counter()
            acc = _TrainEpoch()
            for xb, yb in train_loader:
                if checkpointer is not None:
                    checkpointer.maybe_save(step, epoch, module)
                _apply_log_schedule(optimizer, log_schedule, step)
                _sync(cuda)
                batch_start_time = time.perf_counter()
                xb, yb = xb.to(device), yb.to(device)
                batch_losses, ypred = model.output_and_loss_grad((xb, yb))
                # Nothing non-finite reaches the update: HIG's half-inverse
                # would otherwise die inside its SVD, not as DivergedError (F6).
                stats = _pre_step_check(losses, acc, batch_losses, ypred, yb, step=step,
                                        batch_start_time=batch_start_time, cuda=cuda,
                                        track_acc=track_acc, is_lm=is_lm,
                                        stop_on_nonfinite=stop_on_nonfinite)
                optimizer.step()
                _sync(cuda)
                batch_time = time.perf_counter() - batch_start_time
                _record_train_batch(losses, acc, batch_losses, ypred, yb, batch_time=batch_time,
                                    step=step, track_acc=track_acc, is_lm=is_lm,
                                    stop_on_nonfinite=stop_on_nonfinite, stats=stats)
                step += 1
                _step_evals(losses, acc, model.evaluate, loss_fn, step_loaders, device, step=step,
                            eval_every_steps=eval_every_steps, track_acc=track_acc, is_lm=is_lm)

            _record_epoch(losses, acc, model.evaluate, loss_fn, eval_loaders, device,
                          epoch_start_time=epoch_start_time, track_acc=track_acc, is_lm=is_lm)
            if track_param_norm:
                _ap(losses, 'param_norm', _param_norm(model))
            if checkpointer is not None:
                checkpointer.epoch_end(epoch, step, module)

    return model, _finish_losses(losses, total_start_time)


def train_loop_jd(model, inner_optimizer, aggregator, per_sample_loss_fn, train_loader, val_loader,
                  num_epochs, device, track_acc=False, track_param_norm=False, *,
                  losses=None, test_loader=None, train_eval_loader=None,
                  eval_every_steps=None, checkpointer=None, log_schedule=None,
                  stop_on_nonfinite=True, bn_mode="batch", is_lm=False) -> tuple[Any, dict[str, Any]]:
    """Jacobian Descent (torchjd).

    ``torchjd.autojac.backward`` + ``jac_to_grad`` replace ``loss.backward()``:
    the per-sample loss Jacobian is aggregated (e.g. UPGrad) into ``.grad`` and
    ``inner_optimizer`` (Adam, SGD, ...) applies the update.
    """
    try:
        from torchjd.autojac import backward as jd_backward, jac_to_grad
    except ImportError as e:
        raise ImportError("torchjd is required for train_loop_jd (uv add torchjd)") from e
    losses = {} if losses is None else losses
    cuda = _cuda_active(device)
    module = _ckpt_module(model)
    eval_loaders = {'val': val_loader, 'test': test_loader, 'train_eval': train_eval_loader}
    step_loaders = {'val': val_loader, 'test': test_loader}
    _apply_bn_mode(module, bn_mode)

    _record_evals(losses, model, per_sample_loss_fn, eval_loaders, device,
                  track_acc=track_acc, is_lm=is_lm)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    total_start_time = time.perf_counter()
    step = 0

    with _partial_record_on_divergence(losses, total_start_time):
        for epoch in tqdm(range(num_epochs)):
            epoch_start_time = time.perf_counter()
            acc = _TrainEpoch()
            model.train()
            _apply_bn_mode(module, bn_mode)
            for xb, yb in train_loader:
                if checkpointer is not None:
                    checkpointer.maybe_save(step, epoch, module)
                _apply_log_schedule(inner_optimizer, log_schedule, step)
                _sync(cuda)
                batch_start_time = time.perf_counter()
                xb, yb = xb.to(device), yb.to(device)
                inner_optimizer.zero_grad()
                ypred = model(xb)
                sample_losses = per_sample_loss_fn(ypred, yb)  # (B,)
                # Nothing non-finite reaches the aggregation / update (F6).
                stats = _pre_step_check(losses, acc, sample_losses, ypred, yb, step=step,
                                        batch_start_time=batch_start_time, cuda=cuda,
                                        track_acc=track_acc, is_lm=is_lm,
                                        stop_on_nonfinite=stop_on_nonfinite)
                jd_backward(sample_losses)
                jac_to_grad(list(model.parameters()), aggregator)
                inner_optimizer.step()
                _sync(cuda)
                batch_time = time.perf_counter() - batch_start_time
                _record_train_batch(losses, acc, sample_losses.detach(), ypred.detach(), yb,
                                    batch_time=batch_time, step=step, track_acc=track_acc,
                                    is_lm=is_lm, stop_on_nonfinite=stop_on_nonfinite,
                                    stats=stats)
                step += 1
                _step_evals(losses, acc, model, per_sample_loss_fn, step_loaders, device,
                            step=step, eval_every_steps=eval_every_steps, track_acc=track_acc,
                            is_lm=is_lm)

            _record_epoch(losses, acc, model, per_sample_loss_fn, eval_loaders, device,
                          epoch_start_time=epoch_start_time, track_acc=track_acc, is_lm=is_lm)
            if track_param_norm:
                _ap(losses, 'param_norm', _param_norm(model))
            if checkpointer is not None:
                checkpointer.epoch_end(epoch, step, module)

    return model, _finish_losses(losses, total_start_time)
