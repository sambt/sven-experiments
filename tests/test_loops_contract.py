"""CPU-only tests for the "Training loops" interface of ``EXPERIMENTS.md section 12``.

Covers, for all four loops: caller-owned ``losses`` and the ``DivergedError``
early stop (C-R1), the extra evaluation splits (C-E1), LBFGS's first-closure
train loss (C-E3), step-based evaluation (C-E4), ``summarize_curves`` (C-E5),
synchronised timers and ``train_times`` (C-T1), checkpointer and ``log_schedule``
hooks (C-L2/C-L3), and that the pre-campaign positional call sites still work.

The svd loop is exercised on an MLP only: its BatchNorm behaviour lives in the
Sven wrapper, which another track is changing in parallel, so the BN + Sven
combination is an integration test, not a unit test here.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import experiments.experiment_code.experiment_utils as eu
from experiments.experiment_code.experiment_utils import (
    DivergedError,
    evaluate,
    summarize_curves,
    train_loop_hig,
    train_loop_jd,
    train_loop_standard,
    train_loop_svd,
)
from experiments.optimizers.hig import HIGOptimizer, HIGWrapper
from sven.nn import SvenWrapper
from sven.opt import Sven


def _mlp(in_dim=3, out_dim=2, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(in_dim, 8), nn.GELU(), nn.Linear(8, out_dim)).double()


def _data(n=8, in_dim=3, out_dim=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return TensorDataset(
        torch.randn(n, in_dim, generator=g, dtype=torch.float64),
        torch.randn(n, out_dim, generator=g, dtype=torch.float64),
    )


def _loader(ds, batch_size=4, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=drop_last)


def _per_sample(pred, y):
    return ((pred - y) ** 2).sum(dim=-1)


def _mean(pred, y):
    return _per_sample(pred, y).mean()


class _RecordingCheckpointer:
    """Records the C-L3 hook calls instead of writing files."""

    def __init__(self):
        self.saves = []
        self.epoch_ends = []

    def maybe_save(self, step, epoch, module):
        self.saves.append((step, epoch, module))

    def epoch_end(self, epoch, step, module):
        self.epoch_ends.append((epoch, step, module))


class _CountingOptimizer:
    """Wraps a real optimizer and counts the updates it actually performed."""

    def __init__(self, inner):
        self.inner = inner
        self.updates = 0

    @property
    def param_groups(self):
        return self.inner.param_groups

    def zero_grad(self, *args, **kwargs):
        return self.inner.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.updates += 1
        return self.inner.step(*args, **kwargs)


class _NanWrapper:
    """A Sven/HIG-shaped wrapper whose per-sample loss goes non-finite at a step.

    Stands in for the real wrappers (which another track is changing) so the
    divergence-ordering test is deterministic and costs no linear algebra.
    """

    def __init__(self, model, loss_fn, nan_at=None):
        self.model = model
        self.loss_fn = loss_fn
        self.nan_at = nan_at
        self.calls = 0
        self.last_nonfinite = False

    def _losses(self, batch):
        xb, yb = batch[0], batch[1]
        with torch.no_grad():
            ypred = self.model(xb)
            out = self.loss_fn(ypred, yb)
        if self.calls == self.nan_at:
            out = out * float('inf')
        self.calls += 1
        self.last_nonfinite = not bool(torch.isfinite(out).all())
        return out, ypred

    loss_and_grad = _losses           # the svd loop's entry point
    output_and_loss_grad = _losses    # the hig loop's

    def evaluate(self, x):
        with torch.no_grad():
            return self.model(x)


class _LinalgOptimizer:
    """Refuses a non-finite loss the way a real linear-algebra step does.

    Sven's ``eigh`` / HIG's SVD raise an opaque ``LinAlgError`` when handed a
    NaN Gram or output Jacobian, which is exactly what must not happen: the run
    has to be classified as ``DivergedError(step)`` instead (F6).
    """

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.updates = 0

    def step(self, *args):
        if self.wrapper.last_nonfinite:
            raise RuntimeError("linalg failure: the update consumed a non-finite loss")
        self.updates += 1


# ---------------------------------------------------------------------------
# Backwards compatibility: the call sites in generic_scan.py today
# ---------------------------------------------------------------------------

def test_old_style_positional_call_still_works():
    model = _mlp()
    loader = _loader(_data())
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)

    returned, losses = train_loop_standard(
        model, opt, _mean, loader, loader, 2, "cpu", False, False, False
    )

    assert returned is model
    assert len(losses['val']) == 3          # index 0 = untrained
    assert len(losses['train']) == 2
    assert 'test' not in losses and 'train_eval' not in losses
    assert 'val_final' in losses and 'total_time' in losses


def test_losses_dict_is_the_caller_s_object():
    model = _mlp()
    loader = _loader(_data())
    losses = {}
    _, returned = train_loop_standard(model, torch.optim.SGD(model.parameters(), lr=1e-2),
                                      _mean, loader, loader, 1, "cpu", losses=losses)
    assert returned is losses


def test_extra_splits_are_recorded():
    model = _mlp()
    ds = _data()
    loader = _loader(ds)
    losses = {}
    train_loop_standard(model, torch.optim.SGD(model.parameters(), lr=1e-2), _mean,
                        loader, loader, 2, "cpu", losses=losses,
                        test_loader=_loader(_data(seed=2)),
                        train_eval_loader=_loader(ds))
    for key in ('val', 'test', 'train_eval'):
        assert len(losses[key]) == 3, key


def test_recorded_curves_come_from_evaluate():
    """One evaluation path: the last val / test entries are exactly what
    :func:`evaluate` returns for the trained parameters."""
    model = _mlp()
    loader = _loader(_data())
    test_loader = _loader(_data(seed=3))
    losses = {}
    train_loop_standard(model, torch.optim.SGD(model.parameters(), lr=1e-2), _mean,
                        loader, loader, 1, "cpu", losses=losses, test_loader=test_loader)

    assert losses['val'][-1] == pytest.approx(
        evaluate(model, _mean, loader, "cpu")["loss"], abs=1e-12)
    assert losses['test'][-1] == pytest.approx(
        evaluate(model, _mean, test_loader, "cpu")["loss"], abs=1e-12)


# ---------------------------------------------------------------------------
# C-R1: early stop with partial curves
# ---------------------------------------------------------------------------

def test_absurd_lr_raises_diverged_error_and_keeps_partial_curves():
    model = _mlp()
    loader = _loader(_data(n=40), batch_size=4)
    opt = torch.optim.SGD(model.parameters(), lr=1e200)
    losses = {}

    with pytest.raises(DivergedError) as excinfo:
        train_loop_standard(model, opt, _mean, loader, loader, 3, "cpu", losses=losses)

    assert excinfo.value.step >= 1
    assert len(losses['val']) == 1               # the untrained point survived
    assert np.isfinite(losses['val'][0])
    assert losses['train_batch']                 # partial online curve survived
    assert not np.isfinite(losses['train_batch'][-1])
    assert len(losses['train_batch']) == excinfo.value.step + 1
    assert len(losses['batch_times_train']) == len(losses['train_batch'])
    assert 'train' not in losses                               # no epoch completed
    # the diverged record carries the same scalars as a finished one, and the
    # curves are reachable from the exception even without a caller-owned dict
    assert excinfo.value.losses is losses
    for key in ('total_time', 'avg_train_time', 'val_final', 'val_best'):
        assert key in losses, key


def test_diverged_curves_survive_without_a_caller_owned_dict():
    model = _mlp()
    loader = _loader(_data(n=40), batch_size=4)
    opt = torch.optim.SGD(model.parameters(), lr=1e200)

    with pytest.raises(DivergedError) as excinfo:
        train_loop_standard(model, opt, _mean, loader, loader, 3, "cpu")

    assert excinfo.value.losses is not None
    assert not np.isfinite(excinfo.value.losses['train_batch'][-1])
    assert 'total_time' in excinfo.value.losses


@pytest.mark.parametrize("loop", ["standard", "jd"])
def test_the_update_never_sees_a_nonfinite_loss(loop):
    """C-R1/F6: the check is BEFORE the update, so no step is taken on the
    diverged batch (and Sven / HIG / K-FAC never get a NaN to factorise)."""
    model = _mlp()
    loader = _loader(_data(n=40), batch_size=4)
    opt = _CountingOptimizer(torch.optim.SGD(model.parameters(), lr=1e200))
    losses = {}

    with pytest.raises(DivergedError) as excinfo:
        if loop == "standard":
            train_loop_standard(model, opt, _mean, loader, loader, 2, "cpu", losses=losses)
        else:
            from torchjd.aggregation import UPGrad
            train_loop_jd(model, opt, UPGrad(), _per_sample, loader, loader, 2, "cpu",
                          losses=losses)

    assert excinfo.value.step >= 1
    assert opt.updates == excinfo.value.step     # the diverged step did not update


@pytest.mark.parametrize("loop", ["svd", "hig"])
def test_wrapper_loops_classify_divergence_before_the_linalg(loop):
    model = _mlp()
    wrapper = _NanWrapper(model, _per_sample, nan_at=1)
    opt = _LinalgOptimizer(wrapper)
    loader = _loader(_data(n=8), batch_size=4)   # 2 steps per epoch
    losses = {}
    run = train_loop_svd if loop == "svd" else train_loop_hig

    with pytest.raises(DivergedError) as excinfo:
        run(wrapper, opt, _per_sample, loader, loader, 2, "cpu", losses=losses)

    assert excinfo.value.step == 1
    assert opt.updates == 1                      # step 1 never reached the update
    assert not np.isfinite(losses['train_batch'][-1])
    assert 'val_final' in losses


def test_closure_divergence_is_reported_at_the_first_nonfinite_batch_loss():
    """The closure path checks AFTER the step -- the closure only runs inside
    ``optimizer.step()`` -- so the reported step is the first one whose
    pre-update (first-closure) loss is non-finite, i.e. one after the line
    search that blew the parameters up.  Read ``diverged_at_step`` that way for
    the LBFGS / PolyakSGD families."""
    model = _mlp()
    loader = _loader(_data(n=40), batch_size=4)
    opt = torch.optim.LBFGS(model.parameters(), lr=1e30, max_iter=3, history_size=5)
    losses = {}

    with pytest.raises(DivergedError) as excinfo:
        train_loop_standard(model, opt, _mean, loader, loader, 2, "cpu", losses=losses)

    step = excinfo.value.step
    assert step >= 1
    assert len(losses['train_batch']) == step + 1
    assert not np.isfinite(losses['train_batch'][-1])
    assert all(np.isfinite(v) for v in losses['train_batch'][:-1])


def test_stop_on_nonfinite_can_be_switched_off():
    model = _mlp()
    loader = _loader(_data(n=8), batch_size=4)
    opt = torch.optim.SGD(model.parameters(), lr=1e200)
    losses = {}
    train_loop_standard(model, opt, _mean, loader, loader, 1, "cpu",
                        losses=losses, stop_on_nonfinite=False)
    assert len(losses['train']) == 1
    assert not np.isfinite(losses['train'][0])


# ---------------------------------------------------------------------------
# C-E3 / F11: LBFGS records its FIRST closure evaluation
# ---------------------------------------------------------------------------

def test_lbfgs_train_loss_is_the_pre_update_loss():
    model = _mlp()
    ds = _data(n=8)
    loader = _loader(ds, batch_size=8)
    xb, yb = ds.tensors
    with torch.no_grad():
        pre_update = _mean(model(xb), yb).item()

    losses = {}
    opt = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=3, history_size=5,
                            line_search_fn="strong_wolfe")
    train_loop_standard(model, opt, _mean, loader, loader, 1, "cpu", losses=losses)

    assert losses['train_batch'][0] == pytest.approx(pre_update, abs=1e-12)
    assert losses['train'][0] == pytest.approx(pre_update, abs=1e-12)
    with torch.no_grad():  # the last closure call would have recorded a lower value
        assert _mean(model(xb), yb).item() < pre_update


# ---------------------------------------------------------------------------
# F10: the online train loss is example-weighted
# ---------------------------------------------------------------------------

def test_online_train_loss_is_example_weighted():
    model = _mlp()
    loader = _loader(_data(n=10), batch_size=4)  # batches of 4, 4, 2
    losses = {}
    train_loop_standard(model, torch.optim.SGD(model.parameters(), lr=1e-3), _mean,
                        loader, loader, 1, "cpu", losses=losses)

    means = losses['train_batch']
    assert len(means) == 3
    weighted = (4 * means[0] + 4 * means[1] + 2 * means[2]) / 10
    batch_averaged = float(np.mean(means))
    assert losses['train'][0] == pytest.approx(weighted, abs=1e-12)
    assert abs(batch_averaged - weighted) > 1e-6


# ---------------------------------------------------------------------------
# C-T1: synchronised timers
# ---------------------------------------------------------------------------

def test_train_times_are_bounded_by_epoch_times():
    model = _mlp()
    loader = _loader(_data(n=8), batch_size=4)
    losses = {}
    train_loop_standard(model, torch.optim.SGD(model.parameters(), lr=1e-3), _mean,
                        loader, loader, 2, "cpu", losses=losses,
                        test_loader=loader, train_eval_loader=loader)

    assert len(losses['train_times']) == len(losses['epoch_times']) == 2
    assert all(t <= e for t, e in zip(losses['train_times'], losses['epoch_times']))
    assert losses['train_times'][0] == pytest.approx(sum(losses['batch_times_train'][:2]))
    assert losses['avg_train_time'] == pytest.approx(float(np.mean(losses['train_times'])))
    assert 'eval_step_times' not in losses      # no mid-epoch evaluation happened


def _run_two_batches(loop, epochs=2):
    """One epoch-pair of the given loop on CPU tensors, for the sync counter."""
    model = _mlp()
    loader = _loader(_data(n=8), batch_size=4)  # 2 batches per epoch
    if loop == "standard":
        train_loop_standard(model, torch.optim.SGD(model.parameters(), lr=1e-3), _mean,
                            loader, loader, epochs, "cpu", losses={})
    else:
        wrapper = _NanWrapper(model, _per_sample)
        train_loop_svd(wrapper, _LinalgOptimizer(wrapper), _per_sample, loader, loader,
                       epochs, "cpu", losses={})


@pytest.mark.parametrize("loop", ["standard", "svd"])
def test_batch_timers_synchronise_the_device_twice_per_batch(loop, monkeypatch):
    """C-T1: exactly one ``torch.cuda.synchronize()`` immediately before and one
    immediately after every batch, and none at all when CUDA is not in use.

    The tensors stay on the CPU; only the loops' device detection is faked, so
    a dropped sync (or one moved outside the timed region) is visible here.
    """
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: calls.append(1))

    monkeypatch.setattr(eu, "_cuda_active", lambda device: True)
    _run_two_batches(loop)
    assert len(calls) == 2 * 2 * 2          # 2 epochs x 2 batches x (start, end)

    calls.clear()
    monkeypatch.setattr(eu, "_cuda_active", lambda device: False)
    _run_two_batches(loop)
    assert calls == []


def test_cuda_active_needs_cuda_to_be_both_available_and_the_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert eu._cuda_active("cuda") is True
    assert eu._cuda_active(torch.device("cuda", 0)) is True
    assert eu._cuda_active("cpu") is False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert eu._cuda_active("cuda") is False


# ---------------------------------------------------------------------------
# C-L2: the per-step logging flag
# ---------------------------------------------------------------------------

def test_log_schedule_is_set_even_on_an_optimizer_that_never_had_the_flag():
    """No ``hasattr`` guard: a silent no-op would log the full pre-cut spectrum
    on every Sven step, which is the npz blow-up C-L1/C-L2 exist to prevent."""
    model = _mlp()
    loader = _loader(_data(n=8), batch_size=4)  # steps 0, 1
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    assert not hasattr(opt, "log_this_step")

    train_loop_standard(model, opt, _mean, loader, loader, 1, "cpu", losses={},
                        log_schedule=lambda step: step == 0)

    assert opt.log_this_step is False           # the last step was step 1


def test_log_schedule_failure_is_loud_not_silent():
    class _ReadOnlyFlag(torch.optim.SGD):
        @property
        def log_this_step(self):
            return True

    model = _mlp()
    loader = _loader(_data(n=8), batch_size=4)
    opt = _ReadOnlyFlag(model.parameters(), lr=1e-3)
    with pytest.raises(AttributeError):
        train_loop_standard(model, opt, _mean, loader, loader, 1, "cpu", losses={},
                            log_schedule=lambda step: True)


# ---------------------------------------------------------------------------
# C-E4: step-based evaluation
# ---------------------------------------------------------------------------

def test_eval_every_steps_records_indices_and_is_excluded_from_the_timers():
    model = _mlp()
    loader = _loader(_data(n=8), batch_size=2)  # 4 steps per epoch
    losses = {}
    train_loop_standard(model, torch.optim.SGD(model.parameters(), lr=1e-3), _mean,
                        loader, loader, 1, "cpu", losses=losses,
                        test_loader=loader, eval_every_steps=2)

    assert losses['eval_step_idx'] == [2, 4]
    assert len(losses['val_step']) == 2 and len(losses['test_step']) == 2
    assert 'train_eval_step' not in losses      # train_eval is an epoch-end quantity
    assert losses['train_times'][0] <= losses['epoch_times'][0]
    # the mid-epoch evaluation time is excluded from both timers but recorded
    assert len(losses['eval_step_times']) == 1 and losses['eval_step_times'][0] > 0


# ---------------------------------------------------------------------------
# C-L3: checkpointer hooks
# ---------------------------------------------------------------------------

def test_checkpointer_hooks_fire_before_every_update_and_at_epoch_ends():
    model = _mlp()
    loader = _loader(_data(n=8), batch_size=4)  # 2 steps per epoch
    ckpt = _RecordingCheckpointer()
    train_loop_standard(model, torch.optim.SGD(model.parameters(), lr=1e-3), _mean,
                        loader, loader, 2, "cpu", losses={}, checkpointer=ckpt)

    assert [step for step, _, _ in ckpt.saves] == [0, 1, 2, 3]
    assert [epoch for _, epoch, _ in ckpt.saves] == [0, 0, 1, 1]
    assert [(epoch, step) for epoch, step, _ in ckpt.epoch_ends] == [(0, 2), (1, 4)]
    assert all(module is model for _, _, module in ckpt.saves)


# ---------------------------------------------------------------------------
# C-E5
# ---------------------------------------------------------------------------

def test_summarize_curves():
    out = summarize_curves({'val': [1.0, 0.5, 0.2, 0.9, 0.8]})
    assert out['val_final'] == 0.8
    assert out['val_best'] == 0.2
    # an INDEX into the curve, whose entry 0 is the untrained model: the best
    # epoch is val_best_index - 1 (here epoch 1 of 0, 1, 2, 3)
    assert out['val_best_index'] == 2
    assert 'val_best_epoch' not in out
    assert out['val_last3_mean'] == pytest.approx((0.2 + 0.9 + 0.8) / 3)
    assert summarize_curves({}) == {}
    assert summarize_curves({'val': []}) == {}
    nan_curve = summarize_curves({'val': [1.0, float('nan'), float('inf')]})
    assert nan_curve['val_best'] == 1.0 and nan_curve['val_best_index'] == 0
    # fewer than three trained epochs: no mean-of-last-three, rather than one
    # that silently includes the untrained point
    assert nan_curve['val_last3_mean'] is None
    assert summarize_curves({'val': [1.0]})['val_last3_mean'] is None
    assert summarize_curves({'val': [1.0, 0.5, 0.4, 0.3]})['val_last3_mean'] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# The other three loops
# ---------------------------------------------------------------------------

def test_svd_loop_on_an_mlp():
    model = _mlp()
    wrapper = SvenWrapper(model, _per_sample, "cpu")
    opt = Sven(wrapper, lr=0.05, k=4, rtol=1e-3, track_svd_info=True)
    loader = _loader(_data(n=8), batch_size=4, drop_last=True)  # 2 steps per epoch
    ckpt = _RecordingCheckpointer()
    losses = {}

    returned, filled, returned_opt = train_loop_svd(
        wrapper, opt, _per_sample, loader, loader, 2, "cpu", losses=losses,
        test_loader=loader, train_eval_loader=loader, checkpointer=ckpt,
        log_schedule=lambda step: step == 0,
    )

    assert returned is wrapper and filled is losses and returned_opt is opt
    for key in ('val', 'test', 'train_eval'):
        assert len(losses[key]) == 3, key
    assert len(losses['train']) == 2
    assert all(t <= e for t, e in zip(losses['train_times'], losses['epoch_times']))
    # C-L3: the module handed to the checkpointer is the underlying nn.Module
    assert all(module is model for _, _, module in ckpt.saves)
    assert [step for step, _, _ in ckpt.saves] == [0, 1, 2, 3]
    # C-L2: the loop drives the optimizer's per-step logging flag
    assert opt.log_this_step is False


def test_svd_loop_old_style_positional_call():
    model = _mlp()
    wrapper = SvenWrapper(model, _per_sample, "cpu")
    opt = Sven(wrapper, lr=0.05, k=4, rtol=1e-3)
    loader = _loader(_data(n=8), batch_size=4, drop_last=True)
    returned, losses, returned_opt = train_loop_svd(
        wrapper, opt, _per_sample, loader, loader, 1, "cpu", False, True, False
    )
    assert returned is wrapper and returned_opt is opt
    assert len(losses['val']) == 2 and len(losses['param_norm']) == 1


def test_hig_loop_on_an_mlp():
    model = _mlp()
    wrapper = HIGWrapper(model, _per_sample, "cpu")
    opt = HIGOptimizer(wrapper, lr=1e-3, tau=1e-4)
    loader = _loader(_data(n=8), batch_size=4)
    ckpt = _RecordingCheckpointer()
    losses = {}

    returned, filled = train_loop_hig(wrapper, opt, _per_sample, loader, loader, 2, "cpu",
                                      track_param_norm=True, losses=losses,
                                      test_loader=loader, checkpointer=ckpt)

    assert returned is wrapper and filled is losses
    assert len(losses['val']) == 3 and len(losses['test']) == 3 and len(losses['train']) == 2
    assert all(module is model for _, _, module in ckpt.saves)
    # old-style positional call
    _, plain = train_loop_hig(wrapper, opt, _per_sample, loader, loader, 1, "cpu", False, False)
    assert len(plain['val']) == 2


def test_jd_loop_on_an_mlp():
    from torchjd.aggregation import UPGrad

    model = _mlp()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loader = _loader(_data(n=8), batch_size=4)
    ckpt = _RecordingCheckpointer()
    losses = {}

    returned, filled = train_loop_jd(model, opt, UPGrad(), _per_sample, loader, loader,
                                     2, "cpu", losses=losses, test_loader=loader,
                                     checkpointer=ckpt)

    assert returned is model and filled is losses
    assert len(losses['val']) == 3 and len(losses['test']) == 3 and len(losses['train']) == 2
    assert all(module is model for _, _, module in ckpt.saves)
    # old-style positional call
    _, plain = train_loop_jd(model, opt, UPGrad(), _per_sample, loader, loader, 1, "cpu",
                             False, False)
    assert len(plain['val']) == 2
