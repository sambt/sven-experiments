"""CPU-only tests for the single evaluation function (C-E1, C-E2, F10).

Everything that asserts exactness runs in float64.  No downloads: every test
builds its own tensors.
"""
import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.experiment_code.experiment_utils import evaluate
from experiments.nn.nets import SmallResNet

N_VAL = 10_000
EVAL_BATCH = 128


def _per_sample_mse(pred, y):
    return ((pred - y) ** 2).sum(dim=-1)


def _mean_mse(pred, y):
    return _per_sample_mse(pred, y).mean()


def _per_sample_ce(pred, y):
    return F.cross_entropy(pred, y, reduction='none')


def _mlp(in_dim=4, out_dim=3, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(in_dim, 16), nn.GELU(), nn.Linear(16, out_dim)).double()


def _regression_data(n=N_VAL, in_dim=4, out_dim=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return TensorDataset(
        torch.randn(n, in_dim, generator=g, dtype=torch.float64),
        torch.randn(n, out_dim, generator=g, dtype=torch.float64),
    )


def _classification_data(n=N_VAL, in_dim=4, n_classes=5, seed=2):
    g = torch.Generator().manual_seed(seed)
    return TensorDataset(
        torch.randn(n, in_dim, generator=g, dtype=torch.float64),
        torch.randint(0, n_classes, (n,), generator=g),
    )


def _image_data(n=16, seed=3, n_classes=4):
    g = torch.Generator().manual_seed(seed)
    return TensorDataset(
        torch.randn(n, 3, 8, 8, generator=g, dtype=torch.float64),
        torch.randint(0, n_classes, (n,), generator=g),
    )


def _buffers(model):
    return {name: buf.clone() for name, buf in model.named_buffers()}


# ---------------------------------------------------------------------------
# C-E1 acceptance test: 10,000 validation examples, eval batch 128
# ---------------------------------------------------------------------------

def test_evaluate_equals_mean_of_per_example_losses():
    """The recorded value is the mean of the per-example losses to 1e-6."""
    model = _mlp()
    ds = _regression_data()
    loader = DataLoader(ds, batch_size=EVAL_BATCH, shuffle=False)
    with torch.no_grad():
        reference = _per_sample_mse(model(ds.tensors[0]), ds.tensors[1]).mean().item()

    out = evaluate(model, _per_sample_mse, loader, "cpu")

    assert out["n"] == N_VAL
    assert out["acc"] is None
    assert abs(out["loss"] - reference) < 1e-6
    # 10_000 = 78 * 128 + 16: the last batch is ragged, which is exactly the
    # case a mean of batch means gets wrong (F10).
    assert len(loader) == 79


def test_evaluate_accepts_a_mean_reduced_loss_too():
    """`batch_mean * n_b` summed over batches is the same example-weighted mean,
    which is what lets the standard loop keep its scalar loss_fn."""
    model = _mlp()
    loader = DataLoader(_regression_data(), batch_size=EVAL_BATCH, shuffle=False)
    per_sample = evaluate(model, _per_sample_mse, loader, "cpu")["loss"]
    mean_reduced = evaluate(model, _mean_mse, loader, "cpu")["loss"]
    assert mean_reduced == pytest.approx(per_sample, abs=1e-9)


def test_ragged_last_batch_is_example_weighted_not_batch_averaged():
    """Constructed case where a mean of batch means is wrong by 20x (F10)."""
    n, batch = 260, 128
    x = torch.zeros(n, 1, dtype=torch.float64)
    y = torch.zeros(n, 1, dtype=torch.float64)
    y[-4:] = 10.0  # the 4 examples of the ragged last batch carry all the loss
    loader = DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False)

    out = evaluate(nn.Identity(), _per_sample_mse, loader, "cpu")

    assert out["loss"] == pytest.approx(4 * 100.0 / n, abs=1e-12)
    assert out["n"] == n
    batch_means = [0.0, 0.0, 100.0]  # 128, 128, 4
    assert abs(float(np.mean(batch_means)) - out["loss"]) > 1.0


def test_evaluate_accuracy_is_correct_counts():
    model = _mlp(out_dim=5)
    ds = _classification_data()
    loader = DataLoader(ds, batch_size=EVAL_BATCH, shuffle=False)
    with torch.no_grad():
        preds = model(ds.tensors[0]).argmax(dim=1)
    reference = (preds == ds.tensors[1]).double().mean().item()

    out = evaluate(model, _per_sample_ce, loader, "cpu", track_acc=True)

    assert out["acc"] == pytest.approx(reference, abs=1e-12)


def test_evaluate_is_token_weighted_for_lm():
    """`lm_ce` is a per-sequence mean over T targets; the split value is the
    mean over tokens, and agrees with the flattened token mean."""
    n, block, vocab = 10, 5, 7
    g = torch.Generator().manual_seed(4)
    tokens = torch.randint(0, vocab, (n, block), generator=g)
    targets = torch.randint(0, vocab, (n, block), generator=g)
    torch.manual_seed(0)
    model = nn.Embedding(vocab, vocab).double()
    loader = DataLoader(TensorDataset(tokens, targets), batch_size=4, shuffle=False)

    def lm_ce(pred, y):  # SVD_LOSS_FNS["lm_ce"]
        return F.cross_entropy(
            pred.reshape(-1, pred.shape[-1]), y.reshape(-1), reduction='none'
        ).reshape(y.shape[0], -1).mean(dim=1)

    def lm_ce_mean(pred, y):  # STANDARD_LOSS_FNS["lm_ce"]
        return F.cross_entropy(pred.reshape(-1, pred.shape[-1]), y.reshape(-1))

    with torch.no_grad():
        reference = lm_ce_mean(model(tokens), targets).item()
    out = evaluate(model, lm_ce, loader, "cpu", is_lm=True)
    out_mean = evaluate(model, lm_ce_mean, loader, "cpu", is_lm=True)

    assert out["loss"] == pytest.approx(reference, abs=1e-9)
    assert out_mean["loss"] == pytest.approx(reference, abs=1e-9)
    assert out["n"] == n


# ---------------------------------------------------------------------------
# C-E2: evaluation mutates nothing and restores the mode
# ---------------------------------------------------------------------------

def test_evaluating_twice_changes_no_buffer():
    torch.manual_seed(0)
    model = SmallResNet(num_classes=4, width=8, num_blocks=1).double()
    ds = _image_data()
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    with torch.no_grad():  # give the running statistics a non-trivial value
        model.train()
        model(ds.tensors[0])
    before = _buffers(model)

    first = evaluate(model, _per_sample_ce, loader, "cpu", track_acc=True)
    second = evaluate(model, _per_sample_ce, loader, "cpu", track_acc=True)
    after = _buffers(model)

    assert before  # the net really does have running statistics
    for name in before:
        assert torch.equal(before[name], after[name]), name
    assert first["loss"] == second["loss"]
    assert first["acc"] == second["acc"]


def test_evaluate_restores_every_previous_mode():
    torch.manual_seed(0)
    model = SmallResNet(num_classes=4, width=8, num_blocks=1).double()
    loader = DataLoader(_image_data(), batch_size=8, shuffle=False)
    model.train()
    model.bn1.training = False  # a deliberately frozen layer must stay frozen

    evaluate(model, _per_sample_ce, loader, "cpu")

    assert model.training is True
    assert model.layer1[0].bn1.training is True
    assert model.bn1.training is False


class _SloppyWrapper:
    """A Sven/HIG-shaped wrapper whose ``evaluate`` forgets eval mode."""

    def __init__(self, model):
        self.model = model

    def evaluate(self, x):
        return self.model(x)  # no eval mode, no write suppression


def test_evaluate_guards_a_wrapper_that_forgets_eval_mode():
    """The svd / hig loops evaluate through such a bound method, so evaluate()
    must enforce the policy itself instead of trusting the wrapper: otherwise a
    single regression in the (separately owned) sven repo brings F2 back with
    every loops test still green."""
    torch.manual_seed(0)
    model = SmallResNet(num_classes=4, width=8, num_blocks=1).double()
    ds = _image_data()
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    with torch.no_grad():  # non-trivial running statistics
        model.train()
        model(ds.tensors[0])
    before = _buffers(model)

    out = evaluate(_SloppyWrapper(model).evaluate, _per_sample_ce, loader, "cpu")

    after = _buffers(model)
    for name in before:
        assert torch.equal(before[name], after[name]), name
    assert out["loss"] == pytest.approx(
        evaluate(model, _per_sample_ce, loader, "cpu")["loss"], abs=1e-12)
    assert model.training is True  # the previous mode is restored


def test_evaluate_uses_running_statistics_not_batch_statistics():
    """Eval mode, so a fixed example's prediction does not depend on its batch
    companions -- the F2 defect, inverted."""
    torch.manual_seed(0)
    model = SmallResNet(num_classes=4, width=8, num_blocks=1).double()
    ds = _image_data(n=16)
    with torch.no_grad():
        model.train()
        model(ds.tensors[0])
    x, y = ds.tensors

    one_at_a_time = evaluate(model, _per_sample_ce, DataLoader(ds, batch_size=1), "cpu")
    whole_batch = evaluate(model, _per_sample_ce, DataLoader(ds, batch_size=16), "cpu")
    with torch.no_grad():
        model.train()
        batch_stat_loss = _per_sample_ce(model(x), y).mean().item()

    assert one_at_a_time["loss"] == pytest.approx(whole_batch["loss"], abs=1e-9)
    assert abs(whole_batch["loss"] - batch_stat_loss) > 1e-6
