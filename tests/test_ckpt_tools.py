"""CPU tests for ``analysis/lib/ckpt_tools.py`` (C-L4).

The acceptance tests of plan section 3.2:

* the tool's spectrum equals ``torch.linalg.svdvals`` of the **explicitly**
  differentiated Jacobian (one ``autograd.grad`` per row), in float64, for both
  row definitions -- the loss path ``loss**(kappa/2)`` and the signed-residual
  path ``sign(r)|r|**kappa`` -- and the rows and the Jacobian themselves agree
  with ``SvenWrapper._rows`` / ``SvenWrapper._batch_gradient``, so "the same rows
  Sven uses" is asserted against the optimizer's own code rather than restated;
* a checkpoint written by :class:`~experiments.experiment_code.checkpointing.Checkpointer`
  reloads through :func:`ckpt_tools.load_run` and reproduces the recorded
  validation loss (``verify_checkpoint``);
* the reconstructed batch of a step equals what
  :class:`~experiments.experiment_code.sampler.EpochPermutationSampler` yields;
* the probe set is identical across calls, across optimizers and across seeds of
  one scan, and nested in its size.

``_synthetic_scan`` builds a miniature schema-2 scan directory (resolved config,
records, checkpoints) from the *real* dataset and model classes, so the config
discovery, the dataset rebuild and the checkpoint envelope are all exercised;
the toy dataset is pure torch, so nothing is downloaded.
"""

import copy
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
for path in (str(REPO), str(REPO / "analysis"), str(REPO / "analysis" / "lib")):
    if path not in sys.path:
        sys.path.insert(0, path)

import ckpt_tools as ct                                                   # noqa: E402
from experiments.experiment_code.checkpointing import Checkpointer        # noqa: E402
from experiments.experiment_code.experiment_utils import evaluate         # noqa: E402
from experiments.experiment_code.sampler import (                         # noqa: E402
    EpochPermutationSampler, derive_loader_seed,
)
from experiments.nn import MLP                                            # noqa: E402
from sven.nn import SvenWrapper                                           # noqa: E402

POOL = 64
N_VAL = 32
BATCH = 8
EPOCHS = 2
SPLIT_SEED = 7


def _model(dtype=torch.float64, width=4, out_dim=1):
    torch.manual_seed(0)
    return MLP(input_dim=1, hidden_dims=[width, width], output_dim=out_dim,
               activation="gelu").to(dtype)


def _data(n=16, out_dim=1, dtype=torch.float64):
    g = torch.Generator().manual_seed(3)
    x = torch.randn(n, 1, generator=g, dtype=dtype)
    y = torch.randn(n, out_dim, generator=g, dtype=dtype)
    return x, y


def _explicit_jacobian(model, x, y, spec):
    """``d rows / d theta`` one row at a time, with plain autograd."""
    names = [name for name, _ in model.named_parameters()]
    params = {name: p.detach().clone().requires_grad_(True)
              for name, p in model.named_parameters()}
    buffers = {name: b.detach() for name, b in model.named_buffers()}
    rows = []
    for i in range(x.shape[0]):
        pred = torch.func.functional_call(model, ({**params}, buffers), (x[i:i + 1],))
        value = spec.rows(pred, y[i:i + 1])
        grads = torch.autograd.grad(value.sum(), [params[n] for n in names])
        rows.append(torch.cat([g.reshape(-1) for g in grads]))
    return torch.stack(rows)


# ---------------------------------------------------------------------------
# The rows are Sven's rows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kappa", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("signed", [False, True])
def test_row_spec_matches_sven_wrapper_rows(kappa, signed):
    """``RowSpec.rows`` is ``SvenWrapper._rows`` for every kappa and both paths."""
    from experiments.experiment_code.generic_scan import SVD_LOSS_FNS, SVD_RESIDUAL_FNS
    model = _model()
    x, y = _data()
    spec = ct.RowSpec(loss_key="mse", kappa=kappa, signed_residual=signed)
    wrapper = SvenWrapper(copy.deepcopy(model), SVD_LOSS_FNS["mse"], "cpu", kappa=kappa,
                          residual_fn=SVD_RESIDUAL_FNS["mse"] if signed else None)
    with torch.no_grad():
        pred = model(x)
        mine = spec.rows(pred, y)
        theirs, _ = wrapper._rows(pred, y)
    assert mine.shape == theirs.shape == (x.shape[0],)
    assert torch.allclose(mine, theirs, rtol=0, atol=1e-14)


def test_jacobian_rows_matches_sven_wrapper_jacobian():
    """``jacobian_rows`` equals ``SvenWrapper._batch_gradient`` -- same rows, same
    flattened parameter order."""
    from experiments.experiment_code.generic_scan import SVD_LOSS_FNS, SVD_RESIDUAL_FNS
    model = _model()
    x, y = _data()
    spec = ct.RowSpec(loss_key="mse", kappa=2.0, signed_residual=True)
    wrapper = SvenWrapper(copy.deepcopy(model), SVD_LOSS_FNS["mse"], "cpu", kappa=2.0,
                          residual_fn=SVD_RESIDUAL_FNS["mse"])
    theirs, _, _ = wrapper._batch_gradient((x, y))
    mine = ct.jacobian_rows(model, x, y, spec, chunk_size=5)
    assert mine.shape == theirs.shape
    assert mine.dtype == torch.float64
    assert torch.allclose(mine, theirs, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# The spectrum is the spectrum of the explicit Jacobian
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec_kwargs", [
    dict(loss_key="mse", kappa=2.0, signed_residual=True),
    dict(loss_key="mse", kappa=1.0, signed_residual=True),
    dict(loss_key="mse", kappa=2.0, signed_residual=False),
])
def test_spectrum_equals_svdvals_of_explicit_jacobian(spec_kwargs):
    model = _model()
    x, y = _data(n=20)
    spec = ct.RowSpec(**spec_kwargs)
    reference = _explicit_jacobian(model, x, y, spec)
    expected = torch.linalg.svdvals(reference)
    result = ct.spectrum(model, x, y, spec, chunk_size=7)
    got = torch.as_tensor(result["svals"])
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, rtol=1e-10, atol=1e-14 * float(expected[0]))
    assert result["n_rows"] == x.shape[0]
    assert result["n_params"] == sum(p.numel() for p in model.parameters())


def test_spectrum_label_regression_multi_output():
    """The loss path also works for the multi-output label-regression rows."""
    model = _model(out_dim=3)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(12, 1, generator=g, dtype=torch.float64)
    y = torch.randint(0, 3, (12,), generator=g)
    spec = ct.RowSpec(loss_key="label_regression", kappa=2.0, signed_residual=False)
    expected = torch.linalg.svdvals(_explicit_jacobian(model, x, y, spec))
    got = torch.as_tensor(ct.spectrum(model, x, y, spec, with_utr=False)["svals"])
    assert torch.allclose(got, expected, rtol=1e-10, atol=1e-14 * float(expected[0]))


def test_spectrum_utr_is_residual_in_left_singular_basis():
    """``utr = U^T r`` -- the same quantity C-L1 logs online, on the probe set."""
    model = _model()
    x, y = _data(n=20)
    spec = ct.RowSpec(loss_key="mse", kappa=2.0, signed_residual=True)
    u, svals, _ = torch.linalg.svd(_explicit_jacobian(model, x, y, spec),
                                   full_matrices=False)
    with torch.no_grad():
        rows = spec.rows(model(x), y)
    result = ct.spectrum(model, x, y, spec)
    utr = torch.as_tensor(result["utr"])
    # |u_i . r| is only well defined where sigma_i is resolved: the tail of this
    # Jacobian is numerically zero (which is the point of the diagnostic), and
    # LAPACK's orthonormal completion of that null space is arbitrary.
    gap = (svals[:-1] / svals[1:]) > 1.0 + 1e-6
    resolved = (svals > 1e-6 * svals[0])
    resolved[1:] &= gap
    resolved[:-1] &= gap
    assert int(resolved.sum()) >= 5      # the mask must not be empty
    assert torch.allclose(utr[resolved].abs(), (u.transpose(0, 1) @ rows)[resolved].abs(),
                          rtol=1e-6, atol=1e-10)
    # U is orthogonal here (N = 20 <= P), so the projections carry all of r
    assert utr.norm().item() == pytest.approx(rows.norm().item(), rel=1e-10)


def test_jacobian_rows_refuses_an_oversized_jacobian():
    model = _model()
    x, y = _data(n=8)
    spec = ct.RowSpec(loss_key="mse", kappa=2.0, signed_residual=True)
    with pytest.raises(MemoryError, match="lower n_probe"):
        ct.jacobian_rows(model, x, y, spec, max_bytes=16)


# ---------------------------------------------------------------------------
# A miniature schema-2 scan: config + records + checkpoints
# ---------------------------------------------------------------------------

def _write_config(scan_dir, name, mode, optimizer, width=4, loss="mse",
                  data_seed=SPLIT_SEED, result_id_fields=("mlp_width",)):
    cfg = OmegaConf.create({
        "dataset": {
            "_target_": "experiments.datasets.Toy1DRegressionDataset",
            "n_train": POOL, "n_val": N_VAL, "n_test": N_VAL,
            "pool_size": POOL, "seed": "${data_seed}",
        },
        "model": {
            "_target_": "experiments.nn.MLP", "input_dim": 1,
            "hidden_dims": ["${mlp_width}", "${mlp_width}"], "output_dim": 1,
            "activation": "gelu",
        },
        "mode": mode, "loss": loss, "mlp_width": width,
        "result_id_fields": list(result_id_fields), "data_seed": data_seed,
        "num_epochs": EPOCHS, "batch_size": BATCH, "loader_seed": 11,
        "model_seeds": [0, 1], "optimizers_standard": [optimizer],
        "checkpoints": "log",
    })
    path = scan_dir / "configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(cfg))
    return cfg


def _train_and_record(scan_dir, cfg, run_id, optimizer_name, model_seed, loader_seed=11,
                      extra=None):
    """A two-epoch run with the loops' checkpoint hook order, written as a record.

    Not a training loop under test -- what matters is that the states land in the
    file the way ``Checkpointer`` is called by ``experiment_utils`` (``maybe_save``
    before the update of ``step``, ``epoch_end`` after the epoch) and that the
    ``val`` curve has the untrained model at index 0 (C-E1).
    """
    from hydra.utils import instantiate
    from torch.utils.data import DataLoader

    from experiments.experiment_code.generic_scan import SVD_LOSS_FNS

    dataset = instantiate(cfg.dataset)
    torch.manual_seed(model_seed)
    model = instantiate(cfg.model)
    loss_fn = SVD_LOSS_FNS[str(cfg.loss)]
    sampler = EpochPermutationSampler.for_run(len(dataset.train_dataset), loader_seed,
                                              model_seed, BATCH)
    train_loader = DataLoader(dataset.train_dataset, batch_sampler=sampler)
    val_loader = DataLoader(dataset.val_dataset, batch_size=16, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    checkpointer = Checkpointer(str(scan_dir / "ckpt" / f"{run_id}.pt"), "log",
                                len(train_loader), EPOCHS)

    val = [evaluate(model, loss_fn, val_loader, "cpu")["loss"]]
    step = 0
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        for xb, yb in train_loader:
            checkpointer.maybe_save(step, epoch, model)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).mean().backward()
            optimizer.step()
            step += 1
        val.append(evaluate(model, loss_fn, val_loader, "cpu")["loss"])
        checkpointer.epoch_end(epoch, step, model)
    assert checkpointer.flush()

    record = {
        "run_id": run_id, "optimizer": optimizer_name, "loss": str(cfg.loss),
        "batch_size": BATCH, "model_seed": model_seed, "loader_seed": loader_seed,
        "effective_loader_seed": derive_loader_seed(loader_seed, model_seed),
        "kappa": None, "signed_residual": None, "microbatch_size": None,
        "schema_version": 2, "num_epochs": EPOCHS, "steps_per_epoch": len(train_loader),
        "eval_batch_size": 16, "checkpoint_policy": "log", "status": "ok",
        "n_train": len(dataset.train_dataset), "n_val": N_VAL, "n_test": N_VAL,
        "n_params": sum(p.numel() for p in model.parameters()),
        "split_seed": dataset.split_seed, "mlp_width": int(cfg.mlp_width),
        "ckpt_file": f"ckpt/{run_id}.pt", "ckpt_init_file": None, "ckpt_error": None,
        "losses": {"val": val},
    }
    record.update(extra or {})
    (scan_dir / f"{run_id}.jsonl").write_text(json.dumps(record) + "\n")
    return record


@pytest.fixture(scope="module")
def synthetic_scan(tmp_path_factory):
    """``(results_root, scan_name)`` of a miniature scan: two baselines + one Sven.

    Two resolved configs, one per ``mode``, as a real scan directory has: the
    ``svd_`` run must be matched to the ``svd`` one and the ``std_`` runs to the
    ``standard`` one (``resolved_config``'s prefix rule).
    """
    root = tmp_path_factory.mktemp("results")
    scan = "tiny_scan_diag"
    scan_dir = root / scan
    cfg_std = _write_config(scan_dir, "standard.mlp_width4.SGD.mse.mseed0-1",
                            "standard", "SGD")
    _write_config(scan_dir, "svd.mlp_width4.SGD.mse.mseed0-1", "svd", "SGD")
    _train_and_record(scan_dir, cfg_std, "std_bs8_mlp_width4_lr0.05_optimSGD_mseed0_lseed11",
                      "SGD", 0)
    _train_and_record(scan_dir, cfg_std, "std_bs8_mlp_width4_lr0.05_optimSGD_mseed1_lseed11",
                      "SGD", 1)
    _train_and_record(scan_dir, cfg_std, "svd_bs8_mlp_width4_k8_lr0.05_svdtorch_mseed0_lseed11",
                      "SVD", 0)
    return root, scan


def test_find_runs_and_method_names(synthetic_scan):
    root, scan = synthetic_scan
    runs = ct.find_runs(scan, root)
    assert len(runs) == 3
    assert ct.find_runs(scan, root, prefix="svd_") == [
        "svd_bs8_mlp_width4_k8_lr0.05_svdtorch_mseed0_lseed11"]
    run = ct.load_run(scan, runs[0], root)
    assert run.method == "SGD"
    assert ct.load_run(scan, "svd_bs8_mlp_width4_k8_lr0.05_svdtorch_mseed0_lseed11",
                       root).method == "Sven"


def test_verify_checkpoint_reproduces_the_recorded_loss(synthetic_scan):
    """The acceptance test: reload and reproduce the recorded val loss."""
    root, scan = synthetic_scan
    for run_id in ct.find_runs(scan, root):
        run = ct.load_run(scan, run_id, root)
        assert run.steps[0] == 0 and run.epochs[-1] == EPOCHS - 1
        for epoch in range(EPOCHS):
            detail = ct.verify_checkpoint(run, epoch=epoch, detail=True)
            assert detail["n"] == N_VAL
            assert detail["rel_error"] < 1e-6, detail
        # the plain call returns the relative error of the last epoch
        assert ct.verify_checkpoint(run) < 1e-6


def test_verify_checkpoint_in_float64_also_agrees(synthetic_scan):
    root, scan = synthetic_scan
    run = ct.load_run(scan, ct.find_runs(scan, root)[0], root)
    assert ct.verify_checkpoint(run, dtype=torch.float64) < 1e-5


def test_verify_checkpoint_by_step_uses_that_steps_epoch(synthetic_scan):
    """``step=`` must be compared against the value recorded FOR that step.

    With the epoch label left at ``None`` the curve is indexed at ``[-1]``, so a
    perfectly reconstructed mid-trajectory checkpoint reports the final loss as
    its reference (measured on the real toy Sven run: recorded 1.64e-07 vs
    recomputed 1.21, rel_error 7.4e+06, silently).
    """
    root, scan = synthetic_scan
    run = ct.load_run(scan, ct.find_runs(scan, root)[0], root)
    ends = dict(run.epoch_checkpoints())                       # step -> epoch
    assert len(ends) == EPOCHS
    for step, epoch in ends.items():
        detail = ct.verify_checkpoint(run, step=step, detail=True)
        assert detail["epoch"] == epoch and detail["step"] == step
        assert detail["recorded"] == pytest.approx(run.recorded("val", epoch=epoch))
        assert detail["rel_error"] < 1e-6, detail
        assert detail["recomputed"] == pytest.approx(
            ct.verify_checkpoint(run, epoch=epoch, detail=True)["recomputed"])
    # step 0 is the untrained model: curve index 0 (C-E1), reported as epoch -1
    initial = ct.verify_checkpoint(run, step=0, detail=True)
    assert initial["epoch"] == -1
    assert initial["recorded"] == pytest.approx(run.record["losses"]["val"][0])
    assert initial["rel_error"] < 1e-6, initial
    # a step INSIDE an epoch has no recorded value at all
    mid = [s for s in run.steps if s not in ends and s != 0]
    assert mid, run.steps
    with pytest.raises(ValueError, match="not an epoch end"):
        ct.verify_checkpoint(run, step=mid[0])


def test_verify_checkpoint_rejects_both_selectors(synthetic_scan):
    root, scan = synthetic_scan
    run = ct.load_run(scan, ct.find_runs(scan, root)[0], root)
    with pytest.raises(ValueError, match="step or epoch"):
        ct.verify_checkpoint(run, step=0, epoch=0)


def test_a_partial_curve_is_diagnosed_not_an_index_error(synthetic_scan):
    """A diverged run keeps its checkpoints but only a partial curve (C-R1).

    `polynomial_scan_diag` really contains two such runs (an L-BFGS run that
    diverged at step 159 of epoch 0: checkpoints up to step 128, a `val` curve of
    length 1), so this is the shape the tool meets in the results, not a
    hypothetical.
    """
    root, scan = synthetic_scan
    run = ct.load_run(scan, ct.find_runs(scan, root)[0], root)
    run.record = dict(run.record, status="diverged", diverged_at_step=3,
                      losses={"val": [run.record["losses"]["val"][0]]})
    assert run.verifiable_epochs() == []
    with pytest.raises(IndexError, match="partial curve"):
        run.recorded("val", epoch=0)
    with pytest.raises(ValueError, match="no epoch has both a checkpoint"):
        ct.verify_checkpoint(run)
    # ... and with one epoch recorded, the default picks that epoch, not a later
    # checkpointed one
    full = ct.load_run(scan, ct.find_runs(scan, root)[0], root)
    run.record = dict(run.record, losses={"val": full.record["losses"]["val"][:2]})
    assert run.verifiable_epochs() == [0]
    assert ct.verify_checkpoint(run, detail=True)["epoch"] == 0


def test_reconstructed_batch_equals_the_sampler(synthetic_scan):
    """``batch_of_step`` is what the training loader yielded at that step."""
    root, scan = synthetic_scan
    run = ct.load_run(scan, ct.find_runs(scan, root)[0], root)
    sampler = EpochPermutationSampler.for_run(run.n_train, run.record["loader_seed"],
                                              run.model_seed, run.batch_size)
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        for index, expected in enumerate(sampler):
            step = epoch * run.steps_per_epoch + index
            indices, x, y = ct.batch_of_step(run, step)
            assert indices.tolist() == expected
            assert x.shape == (run.batch_size, 1) and y.shape == (run.batch_size, 1)
    # the reconstructed x/y really are those rows of the training set
    indices, x, y = ct.batch_of_step(run, 1)
    pool_x = torch.stack([run.dataset.train_dataset[i][0] for i in indices.tolist()])
    assert torch.allclose(x, pool_x.to(x.dtype))


def test_batch_of_step_checks_the_effective_seed(synthetic_scan):
    root, scan = synthetic_scan
    run = ct.load_run(scan, ct.find_runs(scan, root)[0], root)
    run.record = dict(run.record, effective_loader_seed=12345)
    with pytest.raises(ValueError, match="effective_loader_seed"):
        ct.batch_of_step(run, 0)


def test_probe_set_is_fixed_across_calls_runs_and_seeds(synthetic_scan):
    root, scan = synthetic_scan
    runs = [ct.load_run(scan, r, root) for r in ct.find_runs(scan, root)]
    reference = ct.probe_indices(runs[0], 16)
    assert torch.equal(reference, ct.probe_indices(runs[0], 16))       # same call twice
    for run in runs[1:]:                                                # other optimizers/seeds
        assert torch.equal(reference, ct.probe_indices(run, 16))
    # nested in the size, and the whole pool in natural order when it is asked for
    assert torch.equal(reference[:8], ct.probe_indices(runs[0], 8))
    assert torch.equal(ct.probe_indices(runs[0], None), torch.arange(POOL))
    with pytest.warns(RuntimeWarning, match="exceeds the training pool"):
        assert torch.equal(ct.probe_indices(runs[0], POOL + 5), torch.arange(POOL))
    indices, x, y = ct.probe_set(runs[0], 16)
    assert torch.equal(indices, reference)
    assert x.shape == (16, 1) and x.dtype == torch.float64


def test_row_spec_from_record_falls_back_to_the_scan_default(synthetic_scan):
    root, scan = synthetic_scan
    # the baselines' records carry no kappa / signed_residual, and `mse` has a
    # scalar residual, so they must get the same rows as the scan's Sven run
    baseline = ct.load_run(scan, "std_bs8_mlp_width4_lr0.05_optimSGD_mseed0_lseed11", root)
    sven = ct.load_run(scan, "svd_bs8_mlp_width4_k8_lr0.05_svdtorch_mseed0_lseed11", root)
    assert baseline.row_spec == sven.row_spec
    assert baseline.row_spec == ct.RowSpec("mse", 2.0, True, 1)
    assert ct.row_spec_for_scan(scan, root) == sven.row_spec
    assert ct.RowSpec.from_record({"loss": "label_regression"}).signed_residual is False


def test_checkpoint_spectra_walks_the_trajectory(synthetic_scan):
    root, scan = synthetic_scan
    run = ct.load_run(scan, ct.find_runs(scan, root)[0], root)
    arrays = ct.checkpoint_spectra(run, n_probe=16)
    assert arrays["step"].tolist() == run.steps
    assert arrays["svals"].shape == (len(run.steps), 16)
    assert arrays["utr"].shape == (len(run.steps), 16)
    assert arrays["dist_init"][0] == pytest.approx(0.0)
    assert (arrays["dist_init"][1:] > 0).all()
    assert (arrays["param_norm"] > 0).all()
    assert arrays["probe_indices"].tolist() == ct.probe_indices(run, 16).tolist()
    assert int(arrays["n_params"]) == run.record["n_params"]
    assert json.loads(str(arrays["row_spec"]))["loss_key"] == "mse"
    # the spectrum at a checkpoint is the spectrum of that checkpoint's model
    model = run.model_at(step=run.steps[3])
    _, x, y = ct.probe_set(run, 16)
    direct = ct.spectrum(model, x, y, run.row_spec, with_utr=False)["svals"]
    assert torch.allclose(torch.as_tensor(arrays["svals"][3]), torch.as_tensor(direct),
                          rtol=1e-12, atol=0)
    # epochs_only keeps the END of each epoch, plus step 0 (the initialisation,
    # which epoch 0's end would otherwise hide)
    per_epoch = ct.checkpoint_spectra(run, n_probe=8, epochs_only=True, with_utr=False)
    assert run.epoch_checkpoints() == [(run.steps_per_epoch * (e + 1), e)
                                       for e in range(EPOCHS)]
    assert per_epoch["step"].tolist() == [0] + [s for s, _ in run.epoch_checkpoints()]
    assert per_epoch["epoch"].tolist() == [0] + list(range(EPOCHS))
    assert per_epoch["dist_init"][0] == pytest.approx(0.0)
    assert "utr" not in per_epoch


def test_distance_from_init_and_param_vector(synthetic_scan):
    root, scan = synthetic_scan
    run = ct.load_run(scan, ct.find_runs(scan, root)[0], root)
    assert ct.distance_from_init(run, step=0) == pytest.approx(0.0)
    assert ct.distance_from_init(run) > 0
    theta = ct.param_vector(run.model_at(step=0))
    assert theta.shape == (run.record["n_params"],)
    assert torch.allclose(theta, ct.param_vector(run.state_at(step=0)))


def test_resolved_config_requires_an_unambiguous_model(synthetic_scan, tmp_path):
    """Two configs that disagree on the model must RAISE, not be picked between."""
    root, scan = synthetic_scan
    record = ct.load_record(scan, ct.find_runs(scan, root)[0], root)
    other = tmp_path / "results"
    (other / scan).mkdir(parents=True)
    _write_config(other / scan, "standard.mlp_width4.SGD.mse.mseed0-1", "standard", "SGD")
    _write_config(other / scan, "standard.mlp_width4.Adam.mse.mseed0-1", "standard",
                  "Adam", width=8)
    # both configs claim mlp_width as a result_id_field, so the record's
    # mlp_width=4 filters the width-8 one out again -- force the clash
    clash = OmegaConf.load(other / scan / "configs" /
                           "standard.mlp_width4.Adam.mse.mseed0-1.yaml")
    clash.mlp_width = 4
    clash.model.hidden_dims = [16, 16]
    OmegaConf.save(clash, other / scan / "configs" /
                   "standard.mlp_width4.Adam.mse.mseed0-1.yaml")
    with pytest.raises(ValueError, match="disagree on `model`"):
        ct.resolved_config(scan, record, other)


def test_resolved_config_sees_an_interpolated_dataset_difference(synthetic_scan, tmp_path):
    """Two configs that differ only in ``${data_seed}`` must still clash.

    The campaign parameterises the dataset by interpolation (``seed: ${data_seed}``),
    so the unresolved subtrees are byte-identical: a check that compares the raw
    YAML is blind to the one field that changes every example.
    """
    root, scan = synthetic_scan
    record = ct.load_record(scan, ct.find_runs(scan, root)[0], root)
    other = tmp_path / "results"
    (other / scan).mkdir(parents=True)
    _write_config(other / scan, "standard.mlp_width4_data_seed7.SGD.mse.mseed0-1",
                  "standard", "SGD", data_seed=7)
    _write_config(other / scan, "standard.mlp_width4_data_seed8.SGD.mse.mseed0-1",
                  "standard", "SGD", data_seed=8)
    a, b = sorted((other / scan / "configs").glob("*.yaml"))
    assert OmegaConf.to_yaml(OmegaConf.load(a).dataset) == \
        OmegaConf.to_yaml(OmegaConf.load(b).dataset)          # the blind spot itself
    with pytest.raises(ValueError, match="disagree on `dataset`"):
        ct.resolved_config(scan, record, other)


@pytest.fixture(scope="module")
def two_data_seed_scan(tmp_path_factory):
    """A scan with two DATA seeds, as ``*_confirm`` has (3) -- the F27 passes."""
    root = tmp_path_factory.mktemp("two_seed_results")
    scan = "two_seed_scan"
    scan_dir = root / scan
    fields = ("mlp_width", "data_seed")
    ids = {}
    for data_seed in (7, 8):
        cfg = _write_config(scan_dir,
                            f"standard.mlp_width4_data_seed{data_seed}.SGD.mse.mseed0-1",
                            "standard", "SGD", data_seed=data_seed, result_id_fields=fields)
        run_id = f"std_bs8_mlp_width4_data_seed{data_seed}_lr0.05_optimSGD_mseed0_lseed11"
        _train_and_record(scan_dir, cfg, run_id, "SGD", 0, extra={"data_seed": data_seed})
        ids[data_seed] = run_id
    return root, scan, ids


def test_each_data_seed_gets_its_own_dataset(two_data_seed_scan):
    """The dataset cache must be keyed on the RESOLVED subtree.

    Keyed on the raw YAML, every data seed of a scan shares one key, so the first
    dataset built in the process is handed to every other run: same sizes, every
    example wrong, and nothing in the reconstruction notices.
    """
    root, scan, ids = two_data_seed_scan
    runs = {seed: ct.load_run(scan, run_id, root) for seed, run_id in ids.items()}
    first = {seed: run.dataset.train_dataset[0][0] for seed, run in runs.items()}
    assert runs[7].dataset is not runs[8].dataset
    assert (runs[7].dataset.split_seed, runs[8].dataset.split_seed) == (7, 8)
    assert runs[7].record["split_seed"] == 7 and runs[8].record["split_seed"] == 8
    assert not torch.equal(first[7], first[8])
    # ... and the reconstruction of each is still the one that was trained
    for run in runs.values():
        assert ct.verify_checkpoint(run) < 1e-6


def test_dataset_guard_catches_a_data_seed_mismatch(two_data_seed_scan):
    """``n_train`` alone cannot see a data-seed swap -- ``split_seed`` can."""
    root, scan, ids = two_data_seed_scan
    run = ct.load_run(scan, ids[7], root)
    run.record = dict(run.record, split_seed=8)
    with pytest.raises(ValueError, match="split_seed"):
        _ = run.dataset


def test_row_spec_for_scan_refuses_a_kappa_sweep(tmp_path):
    """``mnist_kappaScan_labelRegression`` holds kappa 1, 2 and 3 in one scan.

    Picking whichever svd_ run sorts first would give the baselines a different
    row definition from the Sven run they are compared against.
    """
    root = tmp_path / "results"
    scan_dir = root / "kappa_scan"
    scan_dir.mkdir(parents=True)
    for kappa in (1.0, 2.0, 3.0):
        run_id = f"svd_k32_kappa{kappa:g}_mseed0"
        (scan_dir / f"{run_id}.jsonl").write_text(json.dumps({
            "run_id": run_id, "optimizer": "SVD", "loss": "label_regression",
            "kappa": kappa, "model_seed": 0}) + "\n")
    with pytest.raises(ValueError, match="different Sven row definitions"):
        ct.row_spec_for_scan("kappa_scan", root)
    chosen = ct.row_spec_for_scan("kappa_scan", root, run_id="svd_k32_kappa1_mseed0")
    assert chosen.kappa == 1.0


def test_checkpoint_spectra_uses_the_scans_row_spec_not_the_runs(synthetic_scan, tmp_path):
    """A baseline record carries no kappa, so its own ``RowSpec`` is the default;
    the spectra must follow the scan's Sven record instead."""
    root, scan = synthetic_scan
    other = tmp_path / "results"
    shutil.copytree(root / scan, other / scan)
    svd_id = "svd_bs8_mlp_width4_k8_lr0.05_svdtorch_mseed0_lseed11"
    path = other / scan / f"{svd_id}.jsonl"
    record = json.loads(path.read_text().splitlines()[0])
    record["kappa"] = 1.0
    path.write_text(json.dumps(record) + "\n")

    baseline = ct.load_run(scan, "std_bs8_mlp_width4_lr0.05_optimSGD_mseed0_lseed11", other)
    assert baseline.row_spec.kappa == 2.0                      # what the record alone says
    assert ct.row_spec_for_scan(scan, other).kappa == 1.0      # what the scan says
    arrays = ct.checkpoint_spectra(baseline, n_probe=8, epochs_only=True, with_utr=False)
    assert json.loads(str(arrays["row_spec"]))["kappa"] == 1.0
    # ... and an explicit spec still wins over both
    explicit = ct.checkpoint_spectra(baseline, n_probe=8, epochs_only=True, with_utr=False,
                                     spec=ct.RowSpec("mse", 2.0, True, 1))
    assert json.loads(str(explicit["row_spec"]))["kappa"] == 2.0
    assert not np.allclose(arrays["svals"], explicit["svals"])


def test_resolved_config_without_configs_directory_raises(tmp_path):
    (tmp_path / "empty_scan").mkdir()
    with pytest.raises(FileNotFoundError, match="resolved config"):
        ct.resolved_config("empty_scan", {"loss": "mse", "model_seed": 0}, tmp_path)


def test_spectra_cache_round_trip(synthetic_scan, tmp_path):
    """``spectra_path`` / ``load_spectra`` are the writer's and the reader's one
    shared view of the npz cache (what the figure notebooks call)."""
    root, scan = synthetic_scan
    run = ct.load_run(scan, "std_bs8_mlp_width4_lr0.05_optimSGD_mseed0_lseed11", root)
    arrays = ct.checkpoint_spectra(run, n_probe=8, epochs_only=True, with_utr=False)
    destination = ct.spectra_path(scan, run.method, run.model_seed, 8, True,
                                  out_dir=tmp_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    assert destination.name == "SGD_mseed0_probe8_epochs.npz"

    loaded = ct.load_spectra(scan, out_dir=tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["epochs_only"] is True
    assert str(loaded[0]["method"]) == "SGD"
    assert np.array_equal(loaded[0]["svals"], arrays["svals"])
    assert ct.load_spectra(scan, method="SGD", model_seed=0, n_probe=8,
                           epochs_only=True, out_dir=tmp_path)
    # every filter must be able to exclude
    assert ct.load_spectra(scan, method="Sven", out_dir=tmp_path) == []
    assert ct.load_spectra(scan, model_seed=1, out_dir=tmp_path) == []
    assert ct.load_spectra(scan, n_probe=None, out_dir=tmp_path) == []
    assert ct.load_spectra(scan, epochs_only=False, out_dir=tmp_path) == []
    with pytest.raises(FileNotFoundError, match="compute_ckpt_spectra"):
        ct.load_spectra("no_such_scan", out_dir=tmp_path)


def _tool_module():
    """``tools/compute_ckpt_spectra.py`` imported by path (``tools/`` is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "compute_ckpt_spectra", REPO / "tools" / "compute_ckpt_spectra.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_records_a_failed_verification_instead_of_dying(synthetic_scan, tmp_path):
    """A run that blew up inside epoch 0 has checkpoints but no end-of-epoch value.

    Its spectra are still computable; the CLI must store ``verify_rel_error =
    nan`` plus the reason and carry on, not raise AFTER the trajectory has been
    computed and lose that work and every run queued behind it (the diverged
    L-BFGS / KFAC runs a ``--methods all`` pass sweeps up).
    """
    tool = _tool_module()
    root, scan = synthetic_scan
    other = tmp_path / "results"
    shutil.copytree(root / scan, other / scan)
    run_id = ct.find_runs(scan, root)[0]
    path = other / scan / f"{run_id}.jsonl"
    record = json.loads(path.read_text().splitlines()[0])
    record["status"] = "diverged"
    record["losses"] = {"val": record["losses"]["val"][:1]}    # a partial curve (C-R1)
    path.write_text(json.dumps(record) + "\n")

    args = types.SimpleNamespace(probe=8, epochs_only=True, no_utr=True, chunk=8,
                                 verbose=False, force=False, dry_run=False, verify=True,
                                 out_dir=str(tmp_path / "spectra"))
    row = {"run_id": run_id, "record": record, "method": "SGD",
           "model_seed": record["model_seed"], "status": "diverged", "has_ckpt": True}
    summary = tool.compute_one(scan, row, args, other)
    assert np.isnan(summary["verify_rel_error"])
    assert summary["n_ckpt"] > 0
    with np.load(summary["path"], allow_pickle=False) as handle:
        assert "no epoch has both a checkpoint" in str(handle["verify_error"])
        assert np.isnan(handle["verify_rel_error"])
        assert handle["svals"].shape[0] == summary["n_ckpt"]


def test_cli_refuses_two_configs_of_one_method_and_seed(synthetic_scan, tmp_path, capsys):
    """The cache path is (method, seed, probe).

    A diag pass holds one config per method, but pointed at a full scan (or at
    ``mnist_kappaScan_labelRegression``, where one seed has several Sven configs)
    every one of them would write the same npz, last one wins, and phase B would
    plot whichever finished last.
    """
    tool = _tool_module()
    root, scan = synthetic_scan
    other = tmp_path / "results"
    shutil.copytree(root / scan, other / scan)
    run_id = "std_bs8_mlp_width4_lr0.05_optimSGD_mseed0_lseed11"
    twin = run_id.replace("lr0.05", "lr0.1")
    record = json.loads((other / scan / f"{run_id}.jsonl").read_text().splitlines()[0])
    record["run_id"] = twin
    (other / scan / f"{twin}.jsonl").write_text(json.dumps(record) + "\n")

    code = tool.main([scan, "--methods", "SGD", "--seeds", "0", "--probe", "8", "--dry-run",
                      "--results-root", str(other), "--out-dir", str(tmp_path / "spectra")])
    out = capsys.readouterr().out
    assert code == 1
    assert "would overwrite one cache file" in out
    assert out.count("[dry-run] would write") == 1


def test_run_without_a_checkpoint_says_so(synthetic_scan):
    root, scan = synthetic_scan
    run = ct.load_run(scan, ct.find_runs(scan, root)[0], root)
    run.record = dict(run.record, ckpt_file=None, checkpoint_policy="none")
    with pytest.raises(FileNotFoundError, match="no ckpt_file"):
        _ = run.checkpoint
