"""CPU tests for the runner half of a campaign run: how a run is trained and what
it records (``generic_scan.execute`` / ``_ScanContext``).

Everything here drives the real ``execute()`` on a tiny in-memory dataset and a
21-parameter MLP, in **float64** (the exactness assertions are bit-level), with
``scan_dir`` inside ``tmp_path``. ``scan()`` itself is not called, so no Hydra
job, no config file and -- decisively -- no path that can resolve to the real
``experiment_results/``: ``_ScanContext`` is handed the temp directory directly.

What is pinned here, by change id:

* **C-E1/C-E3** four loaders: val / test / train_eval at ``eval_batch_size``, the
  train_eval subset fixed by the dataset's ``split_seed`` and shared by every
  optimizer; recorded ``val`` is the example-weighted mean; ``test`` / ``test_acc``
  are on the record and finite.
* **C-S1** a masked Sven run is bit-identical alone and after another run in the
  same process.
* **C-S2/C-S3** Adam and Sven see the same batches at one model seed, different
  batches across model seeds, with the short tail dropped; the effective loader
  seed is recorded.
* **C-E2** ``bn_mode`` reaches the record, the run_id and the wrappers; on a real
  BatchNorm net every family that may train one advances the running statistics
  **exactly once per optimizer step** under ``batch`` and touches no buffer under
  ``frozen``; HIG (whose wrapper takes no policy) and an unset ``bn_mode`` are
  refused rather than silently measured under a different normalisation.
* **C-L1/C-L2** the scheduled spectra land in the npz with width B and their own
  ``svs_step``; ``sv_min`` is not written; ``num_nonzero_svs`` stays dense.
* **C-L3** the checkpoint file reloads and reproduces the recorded final val/test.
* **C-R4/C-E5** schema-2 record fields.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from experiments.experiment_code import grid
from experiments.experiment_code.checkpointing import POLICIES, load_checkpoint, load_state_at
from experiments.experiment_code.generic_scan import (
    SVD_LOSS_FNS, _EpochLoader, _ScanContext, _rewrite_each_epoch, _split_diagnostics,
    execute,
)
from experiments.experiment_code.optim_factory import MUON_RULE_TOKEN
from experiments.experiment_code.sampler import derive_loader_seed, set_loader_epoch
from hydra.utils import instantiate

REPO = Path(__file__).resolve().parents[1]

N_TRAIN, N_VAL, N_TEST, DIM = 64, 40, 24, 3
SPLIT_SEED = 1234


@pytest.fixture(autouse=True)
def float64():
    """float64 everywhere: this file asserts bit-identity and 1e-6 reloads."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


class TinyData:
    """The dataset contract the runner uses: three splits plus ``split_seed``.

    Deliberately not one of ``experiments.datasets``' classes: those own their own
    tests, and a scan-sized dataset would make this file slow. ``n_train`` is a
    multiple of the batch sizes used here except where a test wants the dropped
    tail (C-S3).
    """

    def __init__(self, n_train=N_TRAIN, classify=False, seed=0):
        g = torch.Generator().manual_seed(seed)

        def split(n, offset):
            x = torch.randn(n, DIM, generator=g)
            if classify:
                return TensorDataset(x, (torch.arange(n) + offset) % 2)
            return TensorDataset(x, torch.randn(n, 1, generator=g))

        self.train_dataset = split(n_train, 0)
        self.val_dataset = split(N_VAL, 1)
        self.test_dataset = split(N_TEST, 2)
        self.n_train, self.n_val, self.n_test = n_train, N_VAL, N_TEST
        self.split_seed = SPLIT_SEED


MODEL_CFG = {
    "_target_": "experiments.nn.MLP",
    "input_dim": DIM,
    "hidden_dims": [4],
    "output_dim": 1,
    "activation": "gelu",
}

BASE_CFG = {
    "device": "cpu",
    "num_epochs": 2,
    "batch_size": 8,
    "loss": "mse",
    "loader_seed": 7,
    "model_seeds": [11],
    "data_seed": 0,
    "use_gram": True,
    "k_values": [8],
    "lrs": [0.05],
    "rtol": [1e-3],
    "svd_mode": ["torch"],
    "lrs_standard": [1e-2],
    "optimizers_standard": ["Adam"],
    "eval_batch_size": 16,
    "train_eval_size": 32,
    "checkpoints": "final",
    "svd_spectra_schedule": {"dense_first": 3, "every": 4},
    "model": MODEL_CFG,
}


def build(tmp_path, dataset=None, classify=False, **overrides):
    """``(ctx, specs_by_family, scan_dir)`` for one config, without Hydra."""
    cfg = OmegaConf.create({**BASE_CFG, **overrides})
    rcfg = OmegaConf.to_container(cfg, resolve=True)
    settings = grid.resolve_scan_settings(rcfg)
    svd_settings = grid.resolve_svd_settings(rcfg)
    scan_dir = Path(tmp_path) / "scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    ctx = _ScanContext(cfg, rcfg, dataset or TinyData(classify=classify),
                       str(scan_dir), settings, svd_settings)
    specs = grid.expand_grid(rcfg, mode=overrides.get("mode", "both"), verbose=False)
    by_family = {}
    for spec in specs:
        by_family.setdefault(spec.family, []).append(spec)
    return ctx, by_family, scan_dir


def record_of(scan_dir, spec, status="ok"):
    """The run's jsonl, asserted to have finished with ``status`` (C-R1).

    `execute` turns every exception into a record, so the file exists either way
    and the status is what says whether the run trained; without this check a
    training assertion would quietly pass on a `status: error` record with empty
    curves. `tests/test_runner_lifecycle.py` owns the failure paths."""
    path = Path(scan_dir) / f"{spec.run_id}.jsonl"
    assert path.exists(), f"no record for {spec.run_id}: execute() wrote nothing"
    with open(path) as f:
        record = json.load(f)
    if status is not None:
        assert record.get("status") == status, (
            f"{spec.run_id}: status {record.get('status')!r}, "
            f"error {record.get('error')}")
    return record


def diag_of(scan_dir, spec):
    return np.load(Path(scan_dir) / "diag" / f"{spec.run_id}.npz")


def run_one(tmp_path, family, **overrides):
    """Execute the first spec of ``family`` and return ``(record, spec, ctx, dir)``."""
    ctx, by_family, scan_dir = build(tmp_path, **overrides)
    spec = by_family[family][0]
    execute(spec, ctx)
    return record_of(scan_dir, spec), spec, ctx, scan_dir


def per_example_loss(model, dataset, loss_key="mse"):
    """The reference the recorded curves are compared against: one eval-mode
    forward over the whole split, mean of the per-example losses."""
    x = torch.stack([dataset[i][0] for i in range(len(dataset))])
    y = torch.stack([torch.as_tensor(dataset[i][1]) for i in range(len(dataset))])
    model.eval()
    with torch.no_grad():
        return float(SVD_LOSS_FNS[loss_key](model(x), y).mean())


# ---------------------------------------------------------------------------
# C-E1 / C-E3: splits, evaluation and the recorded curves
# ---------------------------------------------------------------------------

def test_recorded_val_is_the_example_weighted_mean(tmp_path):
    """The recorded val / test / train_eval curves are means over EXAMPLES, not
    means of batch means (F10), measured in eval mode at the final parameters.

    40 validation examples at eval batch 16 gives a ragged tail (16, 16, 8), which
    is exactly the case where the two differ.
    """
    record, spec, ctx, scan_dir = run_one(tmp_path, "standard")
    losses = record["losses"]
    assert len(losses["val"]) == 3 == len(losses["test"]) == len(losses["train_eval"])

    model = instantiate(ctx.cfg.model)
    model.load_state_dict(load_state_at(Path(scan_dir) / "ckpt" / f"{spec.run_id}.pt"))
    for curve, dataset in (("val", ctx.dataset.val_dataset),
                           ("test", ctx.dataset.test_dataset)):
        reference = per_example_loss(model, dataset)
        assert losses[curve][-1] == pytest.approx(reference, rel=1e-6), curve
    # a mean of batch means would be off: the 8-example tail is up-weighted
    x = torch.stack([ctx.dataset.val_dataset[i][0] for i in range(N_VAL)])
    y = torch.stack([ctx.dataset.val_dataset[i][1] for i in range(N_VAL)])
    model.eval()
    with torch.no_grad():
        per_sample = SVD_LOSS_FNS["mse"](model(x), y)
    batch_means = [float(per_sample[i:i + 16].mean()) for i in (0, 16, 32)]
    assert float(np.mean(batch_means)) != pytest.approx(losses["val"][-1], abs=1e-12)


def test_test_columns_are_on_the_record_and_finite(tmp_path):
    """C-E1/C-E5 outcome columns: `test`, `test_acc` and the val summaries sit on
    the record next to the selection metric (which is still the last val)."""
    record, _, _, _ = run_one(tmp_path, "standard", loss="ce", classify=True,
                              model={**MODEL_CFG, "output_dim": 2})
    for key in ("test", "test_acc", "val_final", "val_best", "val_best_index",
                "train_eval_final"):
        assert key in record, key
    assert np.isfinite(record["test"]) and np.isfinite(record["test_acc"])
    assert 0.0 <= record["test_acc"] <= 1.0
    assert record["val_final"] == record["losses"]["val"][-1]
    assert record["test"] == record["losses"]["test"][-1]
    assert record["losses"]["test_acc"] and record["losses"]["val_acc"]


def test_evaluation_loaders_never_use_the_training_batch_size(tmp_path):
    """C-E1: val / test / train_eval run at `eval_batch_size`; only the training
    loader uses `batch_size` (and the sampler owns its batching)."""
    ctx, by_family, _ = build(tmp_path)
    run_loaders = ctx.loaders(by_family["svd"][0])
    assert run_loaders.train.batch_size is None          # batch_sampler owns it
    assert len(run_loaders.train) == N_TRAIN // BASE_CFG["batch_size"]
    for loader in (run_loaders.val, run_loaders.test, run_loaders.train_eval):
        assert loader.batch_size == BASE_CFG["eval_batch_size"]
        # sequential: no epoch-permutation sampler to advance (they are also shared
        # across every run of the scan, so advancing one would leak between runs)
        assert set_loader_epoch(loader, 1) is False


def test_train_eval_is_one_fixed_split_seeded_subset_for_every_optimizer(tmp_path):
    """C-E3: `min(n_train, train_eval_size)` training examples, drawn with the
    dataset's split_seed -- so it is identical for every optimizer, every model
    seed and every batch size, and the curve is comparable across them."""
    ctx, by_family, _ = build(tmp_path, train_eval_size=16, model_seeds=[11, 12])
    subsets = {}
    for family in ("svd", "standard"):
        for spec in by_family[family][:1]:
            loader = ctx.loaders(spec).train_eval
            assert len(loader.dataset) == 16
            subsets[(family, spec.model_seed)] = torch.stack(
                [loader.dataset[i][0] for i in range(16)])
    reference = next(iter(subsets.values()))
    for key, tensor in subsets.items():
        assert torch.equal(tensor, reference), key
    # a train_eval_size at or above n_train is the whole training set
    ctx2, by2, _ = build(tmp_path, train_eval_size=10_000)
    assert len(ctx2.loaders(by2["svd"][0]).train_eval.dataset) == N_TRAIN


# ---------------------------------------------------------------------------
# C-S1 / C-S2 / C-S3: seeding and data order
# ---------------------------------------------------------------------------

def test_a_masked_sven_run_is_position_independent(tmp_path):
    """C-S1 acceptance test. A `param_fraction=0.5` run draws its parameter mask
    in the wrapper constructor, so before C-S1 its result depended on how many
    other runs had drawn from the global RNG first. Same process, two orders.
    """
    common = dict(param_fractions=[0.5], mask_mode="elementwise", mode="svd",
                  lrs=[0.05, 0.1])
    alone_ctx, alone_specs, alone_dir = build(tmp_path / "alone", **common)
    masked = alone_specs["svd"][1]                     # the lr=0.1 point
    execute(masked, alone_ctx)

    after_ctx, after_specs, after_dir = build(tmp_path / "after", **common)
    execute(after_specs["svd"][0], after_ctx)          # a different configuration
    execute(after_specs["svd"][1], after_ctx)

    a, b = record_of(alone_dir, masked), record_of(after_dir, masked)
    assert a["losses"]["val"] == b["losses"]["val"]           # bit-identical
    assert a["losses"]["train"] == b["losses"]["train"]
    assert a["losses"]["test"] == b["losses"]["test"]
    da, db = diag_of(alone_dir, masked), diag_of(after_dir, masked)
    assert np.array_equal(da["train_batch"], db["train_batch"])
    assert np.array_equal(da["svs"], db["svs"])
    assert a["actual_param_fraction"] == b["actual_param_fraction"]
    assert 0.25 <= a["actual_param_fraction"] <= 0.75


def test_every_optimizer_of_a_model_seed_sees_the_same_batches(tmp_path):
    """C-S2: the data order is a pure function of (base loader seed, model seed,
    epoch), so a seed band carries data-order variance while Adam and Sven stay
    paired. C-S3: the short tail batch is dropped for every family."""
    ctx, by_family, _ = build(tmp_path, model_seeds=[11, 12])
    orders = {}
    for family in ("svd", "standard"):
        for spec in by_family[family]:
            sampler = ctx.loaders(spec).sampler
            # the epoch is set explicitly here: this is about the seed-derived
            # order, not about who advances it (that is the _EpochLoader test)
            orders[(family, spec.model_seed)] = [
                [list(batch) for batch in sampler.set_epoch(epoch)] for epoch in (0, 1)]
    for seed in (11, 12):
        assert orders[("svd", seed)] == orders[("standard", seed)]
    assert orders[("svd", 11)] != orders[("svd", 12)]
    # two different epochs, not the same permutation twice
    epoch0, epoch1 = orders[("svd", 11)]
    assert epoch0 != epoch1

    # C-S3: 60 examples at batch 8 = 7 full batches, the last 4 are dropped
    ctx_odd, by_odd, _ = build(tmp_path, dataset=TinyData(n_train=60))
    for family in ("svd", "standard"):
        loader = ctx_odd.loaders(by_odd[family][0]).train
        assert len(loader) == 7
        assert all(len(b) == 8 for b in iter(loader.batch_sampler))


def test_the_effective_loader_seed_is_recorded_and_the_run_id_keeps_the_base(tmp_path):
    """C-S2: the run_id keeps the scan's BASE loader seed (so names are stable),
    which makes the derived seed unrecoverable from the name -- hence the record."""
    record, spec, _, _ = run_one(tmp_path, "standard")
    assert f"_lseed{BASE_CFG['loader_seed']}" in spec.run_id
    assert record["loader_seed"] == BASE_CFG["loader_seed"]
    expected = derive_loader_seed(BASE_CFG["loader_seed"], spec.model_seed)
    assert record["effective_loader_seed"] == expected != BASE_CFG["loader_seed"]


def test_the_epoch_loader_advances_one_epoch_per_pass(tmp_path):
    """The loops iterate the loader once per epoch and never call `set_epoch`, so
    the wrapper has to drive it; without this every epoch replays epoch 0."""
    ctx, by_family, _ = build(tmp_path)
    loader = ctx.loaders(by_family["svd"][0]).train
    assert isinstance(loader, _EpochLoader)
    seen = []
    for expected_epoch in range(3):
        # the examples in the order this pass delivered them
        seen.append(torch.cat([xb for xb, _ in loader]))
        assert loader.batch_sampler.epoch == expected_epoch
        assert len(seen[-1]) == N_TRAIN
    for i, j in ((0, 1), (1, 2), (0, 2)):
        assert not torch.equal(seen[i], seen[j]), (i, j)


# ---------------------------------------------------------------------------
# C-E2: the BatchNorm policy reaches the record, the name and the wrappers
# ---------------------------------------------------------------------------

def test_bn_mode_reaches_the_record_and_the_run_id(tmp_path):
    ctx, by_family, _ = build(tmp_path)
    assert ctx.bn_mode("svd") == "frozen"        # use_gram's legacy default
    assert ctx.bn_mode("standard") == "batch"
    record, spec, _, _ = run_one(tmp_path / "frozen", "standard", bn_mode="frozen")
    assert record["bn_mode"] == "frozen" and spec.run_id.endswith("_bnfrozen")
    ctx_frozen, _, _ = build(tmp_path / "f2", bn_mode="frozen")
    assert ctx_frozen.bn_mode("svd") == ctx_frozen.bn_mode("hig") == "frozen"
    # the deprecated alias still selects the policy, and the wrapper gets it
    ctx_alias, by_alias, _ = build(tmp_path / "alias", gram_freeze_norm_stats=False,
                                   gram_capture="full")
    assert ctx_alias.bn_mode("svd") == "batch"
    assert by_alias["svd"][0].run_id.endswith("_gram_bnbatch")


# --- the BatchNorm contract itself, on a net that has running statistics -------
# `tests/test_bn_policy_loops.py` covers SGD and LBFGS through
# `train_loop_standard`; the families that go through a wrapper (svd with both
# Gram captures, jd, hig) were left to integration, and that is where the defect
# was: `HIGWrapper` takes no norm policy at all.

RESNET_CFG = {"_target_": "experiments.nn.nets.SmallResNet",
              "num_classes": 2, "width": 2, "num_blocks": 1}

#: 8 examples at batch 4 for 2 epochs = 4 optimizer steps, so "once per step"
#: (4) is distinguishable from "twice per step" (8) and from "never" (0).
BN_STEPS = 4
BN_CFG = dict(
    model=RESNET_CFG, loss="ce", batch_size=4, num_epochs=2, mode="all",
    k_values=[4], lrs=[1e-3], lrs_standard=[1e-3],
    optimizers_standard=["SGD", "LBFGS"], lbfgs_max_iter=[3],
    lbfgs_history_size=[10], lbfgs_line_search_fn=["none"],
    lrs_jd=[1e-3], aggregators_jd=["UPGrad"], lrs_hig=[1e-3], tau_hig=[1e-4],
    eval_batch_size=8, train_eval_size=8, use_gram=True, gram_capture="full",
    checkpoints="final",
)


class TinyImages:
    """(3, 8, 8) images and two classes: the smallest input a SmallResNet takes."""

    def __init__(self, n_train=8):
        g = torch.Generator().manual_seed(0)

        def split(n):
            return TensorDataset(torch.randn(n, 3, 8, 8, generator=g),
                                 torch.arange(n) % 2)

        self.train_dataset = split(n_train)
        self.val_dataset, self.test_dataset = split(8), split(8)
        self.split_seed = SPLIT_SEED


def final_state(scan_dir, spec):
    """The run's final checkpointed ``state_dict`` (weights AND buffers)."""
    return load_state_at(Path(scan_dir) / "ckpt" / f"{spec.run_id}.pt")


def batches_tracked(state):
    return {int(v) for k, v in state.items() if k.endswith("num_batches_tracked")}


def running_mean_max(state):
    return max(float(v.abs().max()) for k, v in state.items()
               if k.endswith("running_mean"))


def test_every_family_updates_the_running_statistics_once_per_step(tmp_path):
    """C-E2 acceptance test, for the families that train through a wrapper.

    ``bn_mode: batch`` means "train with batch statistics and update the running
    statistics from the TRAINING batch exactly once per optimizer step"
    (CONTRACTS). Every method must therefore leave ``num_batches_tracked == 4``
    after 4 steps: a family that forwards twice per step without suppressing the
    write moves its running statistics at twice the rate of the methods it is
    compared against and is then evaluated under a different normalisation
    (F2/F3) -- and its checkpointed state is internally inconsistent, which
    breaks the offline re-evaluation C-L4 is for.
    """
    seen = {}
    for capture in ("full", "chunked"):     # the two Gram captures `batch` allows
        ctx, by_family, scan_dir = build(
            tmp_path / f"gram_{capture}", dataset=TinyImages(), bn_mode="batch",
            **{**BN_CFG, "gram_capture": capture})
        spec = by_family["svd"][0]
        execute(spec, ctx)
        record_of(scan_dir, spec)
        seen[f"svd_gram_{capture}"] = final_state(scan_dir, spec)

    ctx, by_family, scan_dir = build(tmp_path / "rest", dataset=TinyImages(),
                                     bn_mode="batch", **BN_CFG)
    for family in ("standard", "lbfgs", "jd"):     # SGD, LBFGS(max_iter=3), JD
        spec = by_family[family][0]
        execute(spec, ctx)
        record_of(scan_dir, spec)
        seen[family] = final_state(scan_dir, spec)

    assert set(seen) == {"svd_gram_full", "svd_gram_chunked", "standard", "lbfgs", "jd"}
    for name, state in seen.items():
        assert batches_tracked(state) == {BN_STEPS}, (name, batches_tracked(state))
        # ... and they really were updated, so "once" is not "never"
        assert running_mean_max(state) > 0.0, name


def test_hig_is_refused_on_a_batch_statistics_norm_model(tmp_path):
    """`HIGWrapper` takes no `bn_mode` and runs two unsuppressed train-mode
    forwards per step (`experiments/optimizers/hig.py`), so on a norm-statistics
    model under `bn_mode: batch` its running means advance at twice every other
    optimizer's rate while `num_batches_tracked` stays 0. Until the wrapper is
    fixed the run is refused -- a loud `error` record instead of a
    plausible-looking number. `frozen` is safe (nothing is written at all).
    """
    ctx, by_family, scan_dir = build(tmp_path, dataset=TinyImages(),
                                     bn_mode="batch", **BN_CFG)
    spec = by_family["hig"][0]
    assert execute(spec, ctx) == "error"
    record = record_of(scan_dir, spec, status="error")
    assert "bn_mode: batch" in record["error"]["message"]
    assert record["ckpt_file"] is None          # it never trained a step


def test_frozen_mode_never_changes_a_buffer(tmp_path):
    """`bn_mode: frozen` (the fine-tune study, O2): norm layers stay in eval mode
    for training and evaluation alike, so no buffer moves for ANY family -- HIG
    included -- while the weights do."""
    ctx, by_family, scan_dir = build(tmp_path, dataset=TinyImages(),
                                     bn_mode="frozen", **BN_CFG)
    for family in ("svd", "standard", "lbfgs", "jd", "hig"):
        spec = by_family[family][0]
        execute(spec, ctx)
        record_of(scan_dir, spec)
        init, _ = ctx.seed_state(spec.model_seed)
        state = final_state(scan_dir, spec)
        moved = 0
        for key, value in state.items():
            reference = init[key].detach().cpu().to(torch.float64)
            value = value.to(torch.float64)
            if key.endswith(("running_mean", "running_var", "num_batches_tracked")):
                assert torch.equal(value, reference), (family, key)
            elif not torch.equal(value, reference):
                moved += 1
        assert moved, f"{family}: no parameter changed, so the test proves nothing"


def test_a_norm_stat_model_needs_an_explicit_bn_mode(tmp_path):
    """C-E2's goal is one BatchNorm policy for EVERY optimizer. A config that
    names neither `bn_mode` nor `gram_freeze_norm_stats` falls back to each
    family's own legacy default (frozen for svd+gram, batch for the rest), which
    on a norm-statistics model means Sven is trained with frozen statistics and
    its baselines with batch statistics -- with no token in either run_id, since
    each family matches its own default. That combination is refused; a norm-free
    model (MLP, nanoGPT: frozen and batch are the same computation) is untouched.
    """
    ctx, by_family, scan_dir = build(tmp_path, dataset=TinyImages(), **BN_CFG)
    assert ctx.settings["bn_mode"] is None
    assert ctx.bn_mode("svd") == "frozen" and ctx.bn_mode("standard") == "batch"
    spec = by_family["standard"][0]
    assert execute(spec, ctx) == "error"
    record = record_of(scan_dir, spec, status="error")
    assert "bn_mode" in record["error"]["message"]
    record, _, _, _ = run_one(tmp_path / "mlp", "standard")     # no norm layers
    assert record["status"] == "ok" and record["bn_mode"] == "batch"


# ---------------------------------------------------------------------------
# C-L1 / C-L2: scheduled spectra
# ---------------------------------------------------------------------------

def test_spectra_are_stored_with_their_own_step_index(tmp_path):
    """C-L2: `svd_spectra_schedule: {dense_first: 3, every: 4}` over 16 steps logs
    steps 0,1,2 then every 4th; every scheduled array is indexed by `svs_step`
    (never by position), the spectra are full width B, and `num_nonzero_svs` stays
    per step. `sv_min` must NOT exist: in old records it meant the smallest KEPT
    singular value, which is now `sv_min_kept` (F20)."""
    record, spec, ctx, scan_dir = run_one(tmp_path, "svd")
    diag = diag_of(scan_dir, spec)
    steps_per_epoch = N_TRAIN // BASE_CFG["batch_size"]
    n_steps = steps_per_epoch * BASE_CFG["num_epochs"]
    assert record["steps_per_epoch"] == steps_per_epoch
    expected = sorted({0, 1, 2} | set(range(0, n_steps, 4)))
    assert list(diag["svs_step"]) == expected
    assert diag["svs"].shape == (len(expected), BASE_CFG["batch_size"])
    assert diag["utr"].shape == diag["svs"].shape
    assert np.isfinite(diag["svs"]).all()
    for key in ("update_norm", "resid_norm", "sv_min_kept", "sv_noise_floor",
                "sv_max", "sv_min_all"):
        assert diag[key].shape == (len(expected),), key
    assert "sv_min" not in diag.files
    assert diag["num_nonzero_svs"].shape == (n_steps,)
    summary = record["svd_summary"]
    assert summary["schedule"] == {"dense_first": 3, "every": 4}
    assert summary["spectra_saved"] == len(expected) and summary["n_steps"] == n_steps
    assert len(summary["sv_max_epoch"]) == BASE_CFG["num_epochs"]
    assert record["svd_spectra_schedule"] == {"dense_first": 3, "every": 4}
    # the spectrum is the FULL one, before the k / rtol cut (C-L1)
    assert (diag["svs"][:, 0] >= diag["sv_min_kept"]).all()
    assert (diag["sv_min_all"] <= diag["sv_min_kept"] + 1e-12).all()


def test_split_diagnostics_indexes_sparse_series_by_step(tmp_path):
    """The epoch means of a sparse series come from its step indices: an epoch with
    no logged step is nan, not a neighbour's value (positional chopping would give
    every epoch a value and silently misattribute them)."""
    result = {
        "steps_per_epoch": 4,
        "losses": {"train": [0.0, 0.0, 0.0], "train_batch": [1.0] * 12},
        "svd_info": {
            "step": [0, 1, 9],
            "svs": [np.array([3.0, 1.0]), np.array([4.0, 2.0]), np.array([5.0, 1.0])],
            "utr": [np.array([1.0, 1.0])] * 3,
            "num_nonzero_svs": [2] * 12,
            "update_norm": [0.1, 0.2, 0.3],
            "resid_norm": [1.0, 1.0, 1.0],
            "sv_min_kept": [1.0, 2.0, 1.0],
            "sv_noise_floor": [1e-8] * 3,
        },
    }
    light, diag = _split_diagnostics(result, "full", {"dense_first": 2, "every": 4})
    assert list(diag["svs_step"]) == [0, 1, 9]
    summary = light["svd_summary"]
    assert summary["sv_max_epoch"][0] == pytest.approx(3.5)   # steps 0 and 1
    assert np.isnan(summary["sv_max_epoch"][1])               # nothing logged
    assert summary["sv_max_epoch"][2] == pytest.approx(5.0)   # step 9
    assert "train_batch" in diag and "train_batch" not in light["losses"]


# ---------------------------------------------------------------------------
# C-L3: checkpoints
# ---------------------------------------------------------------------------

def test_the_final_checkpoint_reproduces_the_recorded_val_and_test(tmp_path):
    """C-L3 acceptance test: reload the final checkpoint of a toy run; its
    validation and test losses reproduce the recorded final values to 1e-6.
    Buffers travel too, and the shared initial state is stored once per seed."""
    record, spec, ctx, scan_dir = run_one(tmp_path, "svd")
    path = Path(scan_dir) / "ckpt" / f"{spec.run_id}.pt"
    assert path.exists() and record["checkpoint_policy"] == "final"
    ckpt = load_checkpoint(path)
    assert ckpt["steps_per_epoch"] == record["steps_per_epoch"]
    assert ckpt["epoch"][-1] == BASE_CFG["num_epochs"] - 1

    model = instantiate(ctx.cfg.model)
    model.load_state_dict(load_state_at(ckpt))
    for curve, dataset, recorded in (
            ("val", ctx.dataset.val_dataset, record["val_final"]),
            ("test", ctx.dataset.test_dataset, record["test"])):
        reference = per_example_loss(model, dataset)
        assert reference == pytest.approx(recorded, rel=1e-6), curve
    init = Path(scan_dir) / record["ckpt_init_file"]
    assert init.exists(), "the `final` policy stores the shared init once per seed"
    assert init.name.startswith(f"init_mseed{spec.model_seed}.")
    assert record["ckpt_error"] is None


def test_the_log_policy_keeps_the_power_of_two_ladder(tmp_path):
    """`checkpoints_svd` overrides `checkpoints` for the svd family only (C-L3)."""
    ctx, by_family, scan_dir = build(tmp_path, checkpoints="none",
                                     checkpoints_svd="log")
    svd, standard = by_family["svd"][0], by_family["standard"][0]
    execute(svd, ctx)
    execute(standard, ctx)
    assert record_of(scan_dir, svd)["checkpoint_policy"] == "log"
    assert record_of(scan_dir, standard)["checkpoint_policy"] == "none"
    ckpt = load_checkpoint(Path(scan_dir) / "ckpt" / f"{svd.run_id}.pt")
    assert list(ckpt["step"])[:5] == [0, 1, 2, 4, 8]
    assert not (Path(scan_dir) / "ckpt" / f"{standard.run_id}.pt").exists()
    # neither policy is `final`, so no shared per-seed init file is written
    assert not list((Path(scan_dir) / "ckpt").glob("init_mseed*.pt"))


def test_the_shared_init_checkpoint_is_named_per_model_generation(tmp_path):
    """One scan directory holds several MODELS -- `result_id_fields: [mlp_width,
    n_data]` (`rebuttal_overparam_*`) is six jobs with six model configs in one
    directory, and a config edit replaces the model of a whole generation.
    `save_init_state` skips a file that exists and nothing retires it, so a
    seed-only name would keep only the FIRST model's initialisation and silently
    let it stand in for every other run's step 0. The name carries the model
    digest, and the record carries the path.
    """
    init_files, sizes, scan_dir = set(), [], None
    for hidden in ([4], [8]):
        ctx, by_family, scan_dir = build(tmp_path, model={**MODEL_CFG,
                                                          "hidden_dims": hidden})
        spec = by_family["standard"][0]
        execute(spec, ctx)
        record = record_of(scan_dir, spec)
        init_files.add(record["ckpt_init_file"])
        state = load_state_at(Path(scan_dir) / record["ckpt_init_file"])
        sizes.append(sum(v.numel() for v in state.values()))
    assert len(init_files) == 2, init_files
    assert sizes == [21, 41]                     # each generation's OWN init
    assert len(list((Path(scan_dir) / "ckpt").glob("init_mseed11.*.pt"))) == 2


def test_the_epoch_rewrite_policy_follows_the_policy_not_only_the_size(tmp_path):
    """C-L3: an ACCUMULATING policy has to flush per epoch whatever the model
    size. nanoGPT is 826,368 params (below the 1e6 threshold) with `checkpoints:
    epochs` over 50 epochs: the checkpointer would hold 51 CPU-fp32 states
    (~170 MB per run, x NPROC) and write nothing until the end, so a 12 h timeout
    or a preemption loses the whole trajectory -- the exact case the C-L3 design
    names. A single-epoch run has nothing to protect.
    """
    assert _rewrite_each_epoch("epochs", 50, 826_368) is True
    assert _rewrite_each_epoch("log", 50, 21) is True
    assert _rewrite_each_epoch("final", 20, 11_200_000) is True      # CIFAR ResNet18
    assert _rewrite_each_epoch("final", 20, 21) is False             # one write is enough
    assert _rewrite_each_epoch("epochs", 1, 10 ** 9) is False
    assert _rewrite_each_epoch("none", 50, 10 ** 9) is False


def test_the_write_order_is_checkpoint_npz_jsonl(tmp_path):
    """The jsonl is the dedup marker, so it must be the last of the three files."""
    record, spec, _, scan_dir = run_one(tmp_path, "svd")
    mtimes = {
        name: os.path.getmtime(path) for name, path in (
            ("ckpt", Path(scan_dir) / "ckpt" / f"{spec.run_id}.pt"),
            ("npz", Path(scan_dir) / "diag" / f"{spec.run_id}.npz"),
            ("jsonl", Path(scan_dir) / f"{spec.run_id}.jsonl"))}
    assert mtimes["ckpt"] <= mtimes["npz"] <= mtimes["jsonl"]


# ---------------------------------------------------------------------------
# C-R4 / schema 2: what every record carries
# ---------------------------------------------------------------------------

def test_schema_two_record_fields(tmp_path):
    ctx, by_family, scan_dir = build(tmp_path, mode="all", lrs_hig=[0.05],
                                     tau_hig=[1e-4], lrs_jd=[1e-3])
    for family in ("svd", "standard", "hig"):
        spec = by_family[family][0]
        execute(spec, ctx)
        record = record_of(scan_dir, spec)
        assert record["schema_version"] == grid.SCHEMA_VERSION == 2
        for key, value in (("n_train", N_TRAIN), ("n_val", N_VAL), ("n_test", N_TEST),
                           ("split_seed", SPLIT_SEED),
                           ("eval_batch_size", BASE_CFG["eval_batch_size"]),
                           ("train_eval_size", BASE_CFG["train_eval_size"]),
                           ("steps_per_epoch", N_TRAIN // BASE_CFG["batch_size"]),
                           ("num_epochs", BASE_CFG["num_epochs"])):
            assert record[key] == value, (family, key)
        assert record["n_params"] == 21 and record["bn_mode"] in grid.BN_MODES
        assert record["checkpoint_policy"] in POLICIES
        assert record["svd_spectra_schedule"] == (
            {"dense_first": 3, "every": 4} if family == "svd" else None)
        assert (record.get("actual_param_fraction") == 1.0) == (family == "svd")


def test_muon_records_its_variant_and_rule(tmp_path):
    """C-B5: the variant is code-determined and recorded; the rule token is an
    input and comes from the spec, so it is in the run hash as well."""
    record, spec, ctx, _ = run_one(tmp_path, "standard", optimizers_standard=["Muon"],
                                   model={**MODEL_CFG, "hidden_dims": [4, 4]})
    assert record["muon_variant"] and "match_rms_adamw" in record["muon_variant"]
    assert record["muon_rule"] == MUON_RULE_TOKEN == grid.MUON_RULE_TOKEN
    assert grid.run_hash(spec, ctx.rcfg)            # hashable with the new fields


def test_grid_copies_of_runner_side_constants_have_not_drifted():
    """grid.py must stay torch-free, so it carries literal copies of two values
    that live in torch-importing modules; a drift silently changes the hash or
    accepts an invalid policy."""
    assert grid.MUON_RULE_TOKEN == MUON_RULE_TOKEN
    assert grid.CHECKPOINT_POLICIES == POLICIES


def test_importing_the_package_does_not_import_torch():
    """`experiments.experiment_code.__init__` exports `scan` lazily, so a launcher
    or tools/reconcile.py can `import experiments.experiment_code.grid` without
    paying for torch + hydra + sven (1-2 min cold on the cluster)."""
    code = (
        "import sys, time;"
        "t=time.perf_counter();"
        "import experiments.experiment_code.grid as g;"
        "dt=time.perf_counter()-t;"
        "assert not {'torch','hydra','sven','omegaconf'} & set(sys.modules), "
        "sorted(m for m in sys.modules if m in {'torch','hydra','sven','omegaconf'});"
        "assert g.SCHEMA_VERSION == 2;"
        "import experiments.experiment_code as p;"
        "assert 'scan' in dir(p) and 'torch' not in sys.modules;"
        "assert callable(p.scan) and 'torch' in sys.modules;"
        "print('ok', round(dt, 3))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True,
                         text=True, env={**os.environ, "PYTHONPATH": str(REPO)})
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.startswith("ok")
