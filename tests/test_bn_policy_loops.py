"""CPU-only tests for the BatchNorm policy of the training loops (C-E2, F2/F3).

The probe from the appendix of ``EXPERIMENTS.md``, inverted: on
``SmallResNet`` the running statistics must be advanced by the *training* batch
exactly once per optimizer step, evaluation must leave every buffer
bit-identical, and ``bn_mode="frozen"`` must never touch them at all.  float64
throughout, so the exponential moving average can be asserted exactly.
"""
import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.experiment_code.experiment_utils import evaluate, train_loop_standard
from experiments.nn.nets import SmallResNet
from experiments.nn.norm_utils import (
    eval_mode,
    freeze_norm_layers,
    no_norm_stat_updates,
    norm_stat_modules,
)

BATCH = 8


def _resnet(seed=0):
    torch.manual_seed(seed)
    return SmallResNet(num_classes=4, width=8, num_blocks=1).double()


def _image_data(n=BATCH, seed=1, n_classes=4):
    g = torch.Generator().manual_seed(seed)
    return TensorDataset(
        torch.randn(n, 3, 8, 8, generator=g, dtype=torch.float64),
        torch.randint(0, n_classes, (n,), generator=g),
    )


def _buffers(model):
    return {name: buf.clone() for name, buf in model.named_buffers()}


def _ce(pred, y):
    return F.cross_entropy(pred, y)


def _per_sample_ce(pred, y):
    return F.cross_entropy(pred, y, reduction='none')


def _expected_running_mean(model, xb):
    """``(1 - m) * old + m * batch_mean`` for ``bn1``, applied exactly once."""
    bn = model.bn1
    with torch.no_grad():
        pre_norm = model.conv1(xb)
    return (1 - bn.momentum) * bn.running_mean + bn.momentum * pre_norm.mean(dim=(0, 2, 3))


def _warm_running_stats(model, xb):
    """One train-mode forward, so the running statistics are not still at their init."""
    with torch.no_grad():
        model.train()
        model(xb)


# ---------------------------------------------------------------------------
# One optimizer step = one running-statistics update
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build_optimizer", [
    pytest.param(lambda m: torch.optim.SGD(m.parameters(), lr=1e-3), id="sgd"),
    pytest.param(lambda m: torch.optim.LBFGS(m.parameters(), lr=1e-3, max_iter=3,
                                             history_size=5, line_search_fn="strong_wolfe"),
                 id="lbfgs_max_iter3"),
])
def test_one_step_updates_running_mean_exactly_once(build_optimizer):
    reference = _resnet()
    ds = _image_data()
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    expected = _expected_running_mean(reference, ds.tensors[0])

    model = copy.deepcopy(reference)
    train_loop_standard(model, build_optimizer(model), _ce, loader, loader, 1, "cpu", losses={})

    assert torch.allclose(model.bn1.running_mean, expected, rtol=1e-10, atol=0)
    assert norm_stat_modules(model)
    for mod in norm_stat_modules(model):
        assert int(mod.num_batches_tracked.item()) == 1


def test_evaluation_leaves_every_buffer_bit_identical():
    model = _resnet()
    ds = _image_data(n=2 * BATCH)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    train_loop_standard(model, torch.optim.SGD(model.parameters(), lr=1e-3),
                        _ce, loader, loader, 1, "cpu", losses={})
    before = _buffers(model)

    for _ in range(2):
        evaluate(model, _per_sample_ce, loader, "cpu", track_acc=True)
    after = _buffers(model)

    for name in before:
        assert torch.equal(before[name], after[name]), name


@pytest.mark.parametrize("build_optimizer", [
    pytest.param(lambda m: torch.optim.SGD(m.parameters(), lr=1e-3), id="sgd"),
    pytest.param(lambda m: torch.optim.LBFGS(m.parameters(), lr=1e-3, max_iter=3,
                                             history_size=5, line_search_fn="strong_wolfe"),
                 id="lbfgs_max_iter3"),
])
def test_frozen_mode_never_changes_a_buffer(build_optimizer):
    model = _resnet()
    ds = _image_data(n=2 * BATCH)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    _warm_running_stats(model, ds.tensors[0])  # stand-in for pretrained statistics
    before = _buffers(model)

    losses = {}
    train_loop_standard(model, build_optimizer(model), _ce, loader, loader, 2, "cpu",
                        losses=losses, bn_mode="frozen")

    for name in before:
        assert torch.equal(before[name], model.state_dict()[name]), name
    assert all(not mod.training for mod in norm_stat_modules(model))
    assert len(losses['val']) == 3


def test_pretraining_validation_runs_in_eval_mode():
    """Index 0 of the val curve is the eval-mode value, not the train-mode one
    the standard loop used to record (F2)."""
    model = _resnet()
    ds = _image_data(n=2 * BATCH)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    _warm_running_stats(model, ds.tensors[0])

    losses = {}
    train_loop_standard(model, torch.optim.SGD(model.parameters(), lr=1e-3),
                        _ce, loader, loader, 0, "cpu", losses=losses)

    eval_value = evaluate(model, _per_sample_ce, loader, "cpu")["loss"]
    with torch.no_grad():  # what the loop recorded before C-E2: batch statistics
        model.train()
        batch_stat_value = float(
            torch.stack([_per_sample_ce(model(xb), yb).mean() for xb, yb in loader]).mean()
        )
    assert losses['val'] == [pytest.approx(eval_value, abs=1e-12)]
    assert abs(eval_value - batch_stat_value) > 1e-6


def test_unknown_bn_mode_is_rejected():
    model = _resnet()
    loader = DataLoader(_image_data(), batch_size=BATCH, shuffle=False)
    with pytest.raises(ValueError, match="bn_mode"):
        train_loop_standard(model, torch.optim.SGD(model.parameters(), lr=1e-3),
                            _ce, loader, loader, 1, "cpu", bn_mode="eval")


# ---------------------------------------------------------------------------
# The primitives (EXPERIMENTS.md section 12: track_running_stats=False, NOT .eval())
# ---------------------------------------------------------------------------

def test_no_norm_stat_updates_keeps_batch_statistics_and_writes_nothing():
    base = _resnet()
    ds = _image_data()
    xb = ds.tensors[0]
    _warm_running_stats(base, xb)

    plain, suppressed, evaluated = (copy.deepcopy(base) for _ in range(3))
    before = _buffers(base)
    with torch.no_grad():
        plain.train()
        out_plain = plain(xb)
        suppressed.train()
        with no_norm_stat_updates(suppressed):
            out_suppressed = suppressed(xb)
        with eval_mode(evaluated):
            out_eval = evaluated(xb)

    # same normalisation as an ordinary train-mode forward ...
    assert torch.equal(out_plain, out_suppressed)
    # ... unlike .eval(), which is why the context is not implemented that way ...
    assert not torch.allclose(out_eval, out_suppressed)
    # ... and nothing was written, while the plain forward did write.
    for name, buf in suppressed.named_buffers():
        assert torch.equal(before[name], buf), name
    assert not torch.equal(before['bn1.running_mean'], plain.bn1.running_mean)


def test_freeze_norm_layers_survives_model_train():
    model = _resnet()
    model.train()
    frozen = freeze_norm_layers(model)
    assert frozen
    assert model.training is True
    assert all(not mod.training for mod in frozen)
    # and the restore contract of the contexts: no blanket .train()
    with no_norm_stat_updates(model):
        pass
    assert all(not mod.training for mod in frozen)
