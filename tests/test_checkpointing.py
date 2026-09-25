"""CPU tests for ``experiments/experiment_code/checkpointing.py`` (C-L3 / F18).

The acceptance tests of the spec:

* reload the final checkpoint of a tiny trained MLP and reproduce its recorded
  loss to 1e-6 (with BatchNorm, so the *buffers* have to be in the file for the
  eval-mode loss to come back);
* the ``log`` policy saves exactly steps {0, 1, 2, 4, 8, ...} plus every epoch
  end, including the case where an epoch end is itself a power of two (and
  ``epochs`` saves step 0 plus the epoch ends);
* ``flush()`` called from an exception handler keeps whatever was collected,
  and a swallowed write failure is still visible in ``last_error``;
* the state is CPU / float32 / cloned, with non-float buffers left alone;
* the temp name of the atomic write is unique per writer, not per pid.

Plus the offline helpers (``load_checkpoint``, ``load_state_at``,
``save_init_state``) and the atomic single-file layout.

``_train`` reproduces the hook order that ``EXPERIMENTS.md section 12`` fixes for the four
training loops -- ``maybe_save(step, epoch, module)`` before the update of
``step`` (so step 0 is the initial model) and ``epoch_end(epoch, step, module)``
after the epoch -- without importing ``experiment_utils`` (torch + sven + hydra,
and owned by another track); the loops' side of the contract is tested in
``tests/test_loops_contract.py``.
"""

import glob
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
CKPT_PATH = REPO / "experiments" / "experiment_code" / "checkpointing.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ckpt_mod = _load(CKPT_PATH, "sv3_checkpointing_under_test")
Checkpointer = ckpt_mod.Checkpointer


# ---------------------------------------------------------------------------
# A tiny trainable model, with a BatchNorm so that buffers matter
# ---------------------------------------------------------------------------

def _mlp(seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(3, 8), nn.BatchNorm1d(8), nn.GELU(), nn.Linear(8, 2),
    ).to(dtype)


def _data(n=16, dtype=torch.float32, seed=1):
    g = torch.Generator().manual_seed(seed)
    return TensorDataset(torch.randn(n, 3, generator=g, dtype=dtype),
                         torch.randn(n, 2, generator=g, dtype=dtype))


def _eval_loss(model, dataset):
    """Eval-mode MSE: uses the BatchNorm *running* statistics."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        loss = float(nn.functional.mse_loss(model(dataset.tensors[0]), dataset.tensors[1]))
    model.train(was_training)
    return loss


def _train(model, checkpointer, loader, num_epochs, lr=0.05, snapshots=None, raise_at=None):
    """The hook order of the four training loops (EXPERIMENTS.md section 12)."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    step = 0
    for epoch in range(num_epochs):
        model.train()
        for xb, yb in loader:
            checkpointer.maybe_save(step, epoch, model)
            if snapshots is not None:
                snapshots[step] = ckpt_mod.cpu_fp32_state(model)
            if raise_at is not None and step == raise_at:
                raise RuntimeError("simulated blow-up")
            optimizer.zero_grad()
            nn.functional.mse_loss(model(xb), yb).backward()
            optimizer.step()
            step += 1
        checkpointer.epoch_end(epoch, step, model)
        if snapshots is not None:
            snapshots[step] = ckpt_mod.cpu_fp32_state(model)
    return step


def _same_state(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------------------
# Acceptance test 1: the final checkpoint reproduces the recorded loss
# ---------------------------------------------------------------------------

def test_final_checkpoint_reloads_and_reproduces_the_recorded_loss(tmp_path):
    dataset, val = _data(), _data(n=12, seed=2)
    model = _mlp()
    path = str(tmp_path / "ckpt" / "run.pt")
    checkpointer = Checkpointer(path, "final", steps_per_epoch=4, num_epochs=3)
    _train(model, checkpointer, DataLoader(dataset, batch_size=4), 3)

    recorded_val = _eval_loss(model, val)
    recorded_train = _eval_loss(model, dataset)
    assert checkpointer.flush() == path
    assert checkpointer.steps == [12] and checkpointer.epochs == [2]   # end of the run

    reloaded = _mlp(seed=7)                       # different init, then overwritten
    assert abs(_eval_loss(reloaded, val) - recorded_val) > 1e-3
    state = ckpt_mod.load_state_at(path)
    assert reloaded.load_state_dict(state, strict=True) is not None
    assert abs(_eval_loss(reloaded, val) - recorded_val) < 1e-6
    assert abs(_eval_loss(reloaded, dataset) - recorded_train) < 1e-6

    # the running statistics are what the eval-mode loss above depends on
    assert not torch.allclose(model[1].running_mean, torch.zeros(8))
    assert torch.equal(reloaded[1].running_mean, model[1].running_mean)
    assert torch.equal(reloaded[1].num_batches_tracked, model[1].num_batches_tracked)


# ---------------------------------------------------------------------------
# Acceptance test 2: the log policy's step set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "steps_per_epoch, num_epochs, exp_steps, exp_epochs",
    [
        # 12 steps: powers of two {0,1,2,4,8} + epoch ends {3,6,9,12}
        (3, 4, [0, 1, 2, 3, 4, 6, 8, 9, 12], [0, 0, 0, 0, 1, 1, 2, 2, 3]),
        # 6 steps: the epoch ends {2,4,6} are themselves powers of two -> dedup
        (2, 3, [0, 1, 2, 4, 6], [0, 0, 0, 1, 2]),
    ],
)
def test_log_policy_saves_powers_of_two_plus_epoch_ends(
        tmp_path, steps_per_epoch, num_epochs, exp_steps, exp_epochs):
    n = 4 * steps_per_epoch
    loader = DataLoader(_data(n=n), batch_size=4)
    path = str(tmp_path / "run.pt")
    checkpointer = Checkpointer(path, "log", steps_per_epoch, num_epochs)
    snapshots = {}
    total = _train(_mlp(), checkpointer, loader, num_epochs, snapshots=snapshots)

    assert total == steps_per_epoch * num_epochs
    assert checkpointer.steps == exp_steps
    assert checkpointer.epochs == exp_epochs
    checkpointer.flush()
    on_disk = ckpt_mod.load_checkpoint(path)
    assert on_disk["step"] == exp_steps and on_disk["epoch"] == exp_epochs
    assert len(on_disk["state"]) == len(exp_steps)
    assert on_disk["steps_per_epoch"] == steps_per_epoch and on_disk["policy"] == "log"
    # every saved state is the model as it stood at that step
    for i, step in enumerate(exp_steps):
        assert _same_state(on_disk["state"][i], snapshots[step]), step
    # `epochs` gets step 0 (the init, needed by the one `epochs` scan) and the
    # epoch ends, and nothing else
    other = Checkpointer(str(tmp_path / "e.pt"), "epochs", steps_per_epoch, num_epochs)
    snaps = {}
    _train(_mlp(), other, loader, num_epochs, snapshots=snaps)
    assert other.steps == [0] + [steps_per_epoch * (e + 1) for e in range(num_epochs)]
    assert other.epochs == [0] + list(range(num_epochs))
    assert _same_state(other.states[0], snaps[0])


# ---------------------------------------------------------------------------
# Acceptance test 3: flush from an exception handler
# ---------------------------------------------------------------------------

def test_flush_after_an_exception_keeps_the_collected_states(tmp_path):
    loader = DataLoader(_data(n=12), batch_size=4)     # 3 steps per epoch
    path = str(tmp_path / "run.pt")
    checkpointer = Checkpointer(path, "log", 3, 10)
    snapshots = {}
    with pytest.raises(RuntimeError, match="simulated blow-up"):
        _train(_mlp(), checkpointer, loader, 10, snapshots=snapshots, raise_at=5)
    assert not os.path.exists(path)                    # nothing written yet

    assert checkpointer.flush() == path                # the runner's except block
    on_disk = ckpt_mod.load_checkpoint(path)
    assert on_disk["step"] == [0, 1, 2, 3, 4]          # {0,1,2,4} + epoch end 3
    assert _same_state(on_disk["state"][-1], snapshots[4])
    assert checkpointer.flush() == path                # idempotent, nothing new

    # Nothing collected -> no file; a failing write is reported, not raised, so
    # it cannot replace the exception already on its way up.
    empty = Checkpointer(str(tmp_path / "empty.pt"), "log", 3, 10)
    assert empty.flush() is None
    assert not os.path.exists(tmp_path / "empty.pt")
    blocked = Checkpointer(str(tmp_path / "run.pt" / "nested.pt"), "final", 3, 1)
    blocked.epoch_end(0, 3, _mlp())
    assert blocked.flush() is None
    # ... but the swallowed failure stays visible, so a full quota cannot leave
    # an `ok` run with no checkpoint and no trace anywhere (the runner records
    # `last_error` as `ckpt_error`).
    assert blocked.last_error is not None and "Error" in blocked.last_error
    assert checkpointer.last_error is None                  # the write that worked
    blocked.path = str(tmp_path / "unblocked.pt")           # cleared by a good write
    assert blocked.flush() == blocked.path and blocked.last_error is None


# ---------------------------------------------------------------------------
# Acceptance test 4: what a state contains
# ---------------------------------------------------------------------------

def test_state_is_a_cpu_float32_clone_including_buffers(tmp_path):
    model = _mlp(dtype=torch.float64)
    loader = DataLoader(_data(dtype=torch.float64), batch_size=4)
    checkpointer = Checkpointer(str(tmp_path / "run.pt"), "final", 4, 1)
    _train(model, checkpointer, loader, 1)
    state = checkpointer.states[0]

    assert set(state) == set(model.state_dict())       # weights AND buffers
    assert "1.running_mean" in state and "1.running_var" in state
    for key, value in state.items():
        assert value.device.type == "cpu"
        if key == "1.num_batches_tracked":
            assert value.dtype == torch.int64          # non-float: kept as it is
            assert int(value) == 4
        else:
            assert value.dtype == torch.float32        # fp64 model, fp32 storage
        assert value.data_ptr() != model.state_dict()[key].data_ptr()

    # a clone, so training on does not rewrite the checkpoint in memory
    before = state["0.weight"].clone()
    _train(model, Checkpointer(str(tmp_path / "x.pt"), "none", 4, 1), loader, 1)
    assert torch.equal(state["0.weight"], before)
    assert not torch.allclose(state["0.weight"].double(), model[0].weight.detach())


# ---------------------------------------------------------------------------
# Policies, layout, offline helpers
# ---------------------------------------------------------------------------

def test_policy_none_writes_nothing_and_unknown_policies_raise(tmp_path):
    path = str(tmp_path / "run.pt")
    checkpointer = Checkpointer(path, "none", 4, 2)
    _train(_mlp(), checkpointer, DataLoader(_data(), batch_size=4), 2)
    assert checkpointer.states == [] and checkpointer.flush() is None
    assert not os.path.exists(path)
    assert Checkpointer(path, None, 4, 2).policy == "none"
    with pytest.raises(ValueError, match="unknown checkpoint policy"):
        Checkpointer(path, "every", 4, 2)


def test_rewrite_each_epoch_writes_one_atomic_file_per_run(tmp_path):
    path = str(tmp_path / "ckpt" / "run.pt")
    loader = DataLoader(_data(), batch_size=4)
    checkpointer = Checkpointer(path, "log", 4, 3, rewrite_each_epoch=True)
    model = _mlp()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    seen = []
    step = 0
    for epoch in range(3):
        for xb, yb in loader:
            checkpointer.maybe_save(step, epoch, model)
            optimizer.zero_grad()
            nn.functional.mse_loss(model(xb), yb).backward()
            optimizer.step()
            step += 1
        checkpointer.epoch_end(epoch, step, model)
        seen.append(ckpt_mod.load_checkpoint(path)["step"])   # already on disk
    assert seen == [[0, 1, 2, 4], [0, 1, 2, 4, 8], [0, 1, 2, 4, 8, 12]]
    assert os.listdir(tmp_path / "ckpt") == ["run.pt"]        # one file, no .tmp
    assert glob.glob(str(tmp_path / "ckpt" / "*.tmp*")) == []


def test_atomic_temp_name_is_unique_per_writer(tmp_path, monkeypatch):
    """Two writers of one path must not share a temp file.

    The per-seed init file (and a checkpoint written after a stale-claim
    takeover) can have two writers on different nodes, and identically
    configured SLURM nodes hand out the same pids, so a temp name qualified by
    pid alone lets two ``torch.save`` streams interleave into one file whose
    mangled bytes ``os.replace`` then promotes.
    """
    path = str(tmp_path / "ckpt" / "init_mseed0.pt")
    seen = []
    real_save = torch.save
    monkeypatch.setattr(ckpt_mod.torch, "save",
                        lambda payload, target: (seen.append(target), real_save(payload, target))[1])

    for value in (1.0, 2.0):
        payload = {"step": [0], "epoch": [0], "state": [{"w": torch.tensor([value])}]}
        assert ckpt_mod._atomic_save(payload, path) == path

    assert len(set(seen)) == 2                               # not one shared temp file
    for target in seen:
        assert target.startswith(f"{path}.tmp.{os.getpid()}.")
        assert target != f"{path}.tmp.{os.getpid()}"          # the pid-only name
    # the last writer's file is intact, and no temp file is left behind
    assert float(ckpt_mod.load_state_at(path)["w"]) == 2.0
    assert sorted(os.listdir(tmp_path / "ckpt")) == ["init_mseed0.pt"]


def test_load_state_at_by_step_and_by_epoch(tmp_path):
    path = str(tmp_path / "run.pt")
    checkpointer = Checkpointer(path, "log", 3, 4)
    snapshots = {}
    _train(_mlp(), checkpointer, DataLoader(_data(n=12), batch_size=4), 4,
           snapshots=snapshots)
    checkpointer.flush()

    assert _same_state(ckpt_mod.load_state_at(path), snapshots[12])
    assert _same_state(ckpt_mod.load_state_at(path, step=4), snapshots[4])
    # end of epoch e, not the first state recorded inside it
    for epoch, end_step in enumerate([3, 6, 9, 12]):
        assert _same_state(ckpt_mod.load_state_at(path, epoch=epoch), snapshots[end_step])
    loaded = ckpt_mod.load_checkpoint(path)                   # read the file once
    assert _same_state(ckpt_mod.load_state_at(loaded, step=8), snapshots[8])
    with pytest.raises(ValueError, match="no checkpoint at step 5"):
        ckpt_mod.load_state_at(path, step=5)
    with pytest.raises(ValueError, match="no checkpoint for epoch 9"):
        ckpt_mod.load_state_at(path, epoch=9)
    with pytest.raises(ValueError, match="not both"):
        ckpt_mod.load_state_at(path, step=4, epoch=0)


def test_save_init_state_is_shared_per_seed(tmp_path):
    model = _mlp(dtype=torch.float64)
    path = str(tmp_path / "ckpt" / "init_mseed0.pt")
    assert ckpt_mod.save_init_state(path, model.state_dict()) == path

    state = ckpt_mod.load_state_at(path)
    assert set(state) == set(model.state_dict())
    assert state["0.weight"].dtype == torch.float32
    assert torch.allclose(state["0.weight"].double(), model[0].weight.detach())
    assert ckpt_mod.load_checkpoint(path)["step"] == [0]

    # the next run of the same seed must not rewrite it, unless asked
    mtime = os.path.getmtime(path)
    assert ckpt_mod.save_init_state(path, _mlp(seed=7).state_dict()) == path
    assert os.path.getmtime(path) == mtime
    assert torch.allclose(ckpt_mod.load_state_at(path)["0.weight"].double(),
                          model[0].weight.detach())
    ckpt_mod.save_init_state(path, _mlp(seed=7).state_dict(), overwrite=True)
    assert not torch.allclose(ckpt_mod.load_state_at(path)["0.weight"].double(),
                              model[0].weight.detach())
