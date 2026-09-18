"""CPU tests for the RUN LIFECYCLE around ``execute``: what happens to a run
before it starts and after it ends (``generic_scan.run_grid``).

Everything here drives the real ``run_grid(cfg, scan_dir)`` on a 32-example toy
regression dataset and a 13-parameter MLP, with ``scan_dir`` inside ``tmp_path``.
``scan()`` -- the Hydra entry point, the one function that resolves
``$SV3_RESULTS_ROOT`` / ``experiment_results`` -- is never called, so no test here
can reach the real results root even if it is misconfigured.

What is pinned here, by change id:

* **C-R1** every failure is a record: the status taxonomy
  (``ok``/``diverged``/``oom``/``error``), the classification order, the partial
  curves (with the same scalar keys as a finished run, whatever the status), the
  ``started`` marker's lifetime, and that ``oom``/``error`` are retried while
  ``ok``/``diverged`` are final -- until three attempts at one hash have failed,
  after which the run is ``poisoned`` and left alone.
* **C-R2** each worker's manifest, and the union over workers = the intended grid.
* **C-R3** ``run_hash`` on every record, the hash8 in the done marker, the
  ``_stale/{hash8}/`` move when the config changes, the provenance dict and the
  resolved config saved once per job.
* **Scheduling** the claim queue: two workers execute each run exactly once, a
  live claim is respected, a stale claim is taken over, a run that was held while
  the walk passed it is picked up by the mop-up pass, and ``scheduler=static``
  still slices the grid (but may not be combined with the queue).

The training half of a run (loaders, curves, checkpoints, spectra) is pinned by
``tests/test_runner_training.py``; this file only ever needs runs that *finish*,
so its models and grids are the smallest ones that still exercise all six files
a run writes (jsonl, npz, ckpt, claim, started, done).
"""

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from experiments.experiment_code import claims, generic_scan, grid
from experiments.experiment_code.experiment_utils import DivergedError
from experiments.experiment_code.generic_scan import _classify_failure, run_grid

N_TRAIN, N_VAL, N_TEST, BATCH = 32, 16, 16, 8

#: a toy scan whose three families (Sven, SGD, HIG) all run at `lrs: [1e3]`, i.e.
#: the C-R1 acceptance test's config. `mode: all` drops the JD family because the
#: config carries no `lrs_jd` / `aggregators_jd` (grid.mode_flags).
DIVERGING_LR = 1e3
BASE_CFG = {
    "device": "cpu",
    "num_epochs": 3,
    "batch_size": BATCH,
    "loss": "mse",
    "mode": "all",
    "loader_seed": 3,
    "model_seeds": [5],
    "data_seed": 0,
    "use_gram": True,
    "k_values": [BATCH],
    "rtol": [1e-3],
    "svd_mode": ["torch"],
    "lrs": [DIVERGING_LR],
    "lrs_standard": [DIVERGING_LR],
    "optimizers_standard": ["SGD"],
    "lrs_hig": [DIVERGING_LR],
    "tau_hig": [1e-4],
    "eval_batch_size": 16,
    "train_eval_size": 16,
    "checkpoints": "final",
    "svd_spectra_schedule": {"dense_first": 2, "every": 2},
    "dataset": {
        "_target_": "experiments.datasets.Toy1DRegressionDataset",
        "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
        "pool_size": N_TRAIN, "seed": 0,
    },
    "model": {
        "_target_": "experiments.nn.MLP", "input_dim": 1, "hidden_dims": [4],
        "output_dim": 1, "activation": "gelu",
    },
}

#: learning rates every family converges at, for the tests that need runs to finish
GOOD = {"lrs": [0.05], "lrs_standard": [0.01], "lrs_hig": [0.05]}

#: the scalars `experiment_utils._finish_losses` puts on EVERY curve dict, whatever
#: the status (C-R1): a run that failed before it recorded a single curve still
#: carries them (as nan), so every status has one schema.
_FAILED_CURVE_KEYS = {"total_time", "avg_epoch_time", "avg_train_time",
                      "avg_batch_time_train", "avg_eval_time"}


def cfg_of(**overrides):
    return OmegaConf.create({**BASE_CFG, **overrides})


def run(scan_dir, **overrides):
    """One worker's pass over the grid; returns run_grid's outcome Counter."""
    return run_grid(cfg_of(**overrides), str(scan_dir))


def run_ids_of(**overrides):
    """The grid a config describes, without running anything."""
    rcfg = OmegaConf.to_container(cfg_of(**overrides), resolve=True)
    return [s.run_id for s in grid.expand_grid(rcfg, verbose=False)]


def records(scan_dir):
    """``{run_id: record}`` from the scan's jsonl files (run_ids contain dots, so
    the suffix is stripped by length, not by ``Path.stem``)."""
    out = {}
    for path in sorted(Path(scan_dir).glob("*.jsonl")):
        with open(path) as fh:
            out[path.name[:-len(".jsonl")]] = json.load(fh)
    return out


def done_markers(scan_dir):
    """``{(run_id, hash8, status)}`` from the done directory."""
    found = set()
    for name in os.listdir(os.path.join(str(scan_dir), "done")):
        found.add(tuple(name.rsplit(".", 2)))
    return found


def artefacts(scan_dir, run_id, root=None):
    """The three files a finished run owns (the done marker is checked separately),
    under ``root`` -- the scan dir, or a ``_stale/{hash8}/`` generation."""
    base = Path(root or scan_dir)
    return {
        "jsonl": base / f"{run_id}.jsonl",
        "npz": base / "diag" / f"{run_id}.npz",
        "ckpt": base / "ckpt" / f"{run_id}.pt",
    }


# ---------------------------------------------------------------------------
# C-R1: every failure is a record
# ---------------------------------------------------------------------------

def test_a_diverging_toy_scan_writes_three_diverged_records(tmp_path):
    """C-R1 acceptance test: `lrs: [1e3]` for Sven, HIG and SGD gives three
    records with `status: diverged`, an error, a step where it happened and the
    partial curves collected up to there -- not three missing files, which is
    what the pre-campaign runner left behind (it printed and moved on).
    """
    counts = run(tmp_path / "scan", num_epochs=6)
    scan_dir = tmp_path / "scan"
    recs = records(scan_dir)
    assert len(recs) == 3 and counts["diverged"] == 3
    assert {r["optimizer"] for r in recs.values()} == {"SVD", "SGD", "HIG"}

    for run_id, record in recs.items():
        assert record["status"] == "diverged", run_id
        assert record["error"]["type"] and record["error"]["message"], run_id
        # partial curves: the untrained validation point is always there, and the
        # per-batch train series (in the npz) stops at the batch that blew up
        assert record["losses"]["val"], run_id
        with np.load(artefacts(scan_dir, run_id)["npz"]) as npz:
            n_batches = len(npz["train_batch"])
        assert n_batches >= 1, run_id
        # DivergedError knows its step; a LinAlgError / Sven empty-spectrum guard
        # does not, and records None rather than a guess
        if record["error"]["type"] == "DivergedError":
            assert "non-finite training loss at optimizer step" in record["error"]["message"]
            assert record["diverged_at_step"] == n_batches - 1, run_id
        if record["optimizer"] == "SVD":
            # C-R1: whatever the optimizer collected before it died is kept -- the
            # spectrum on the way into a divergence is the diagnostic
            assert record["svd_summary"]["n_steps"] >= 1
            with np.load(artefacts(scan_dir, run_id)["npz"]) as npz:
                assert len(npz["svs_step"]) >= 1 and npz["svs"].shape[1] == BATCH

    # `diverged` counts as done (CONTRACTS "Dedup"), so a second pass runs nothing
    assert {s for _, _, s in done_markers(scan_dir)} == {"diverged"}
    assert run(scan_dir, num_epochs=6)["skipped"] == 3


def test_the_failure_classification_order(tmp_path):
    """The taxonomy C-R1 prescribes, and the ORDER it has to be tested in:
    DivergedError subclasses RuntimeError and every OOM / linalg error carries a
    message, so a message-based test placed first would mislabel them.

    The distinction is not cosmetic: `oom` and `error` are retried while `ok` and
    `diverged` are final, so a misclassified divergence is re-run by every worker
    forever and a misclassified OOM is accepted as a scientific result.
    """
    oom = torch.cuda.OutOfMemoryError(
        "CUDA out of memory. Tried to allocate 20.00 GiB (GPU 0; 39.39 GiB total)")
    cases = [
        (DivergedError(7, float("nan")), ("diverged", 7)),
        (oom, ("oom", None)),
        (RuntimeError("CUDA out of memory"), ("oom", None)),          # older torch
        (torch.linalg.LinAlgError("linalg.eigh: something went wrong"),
         ("diverged", None)),
        (RuntimeError("SvenGram: no singular value above rtol * sigma_max "
                      "(sigma_max=nan); the Gram matrix is zero or non-finite"),
         ("diverged", None)),
        (RuntimeError("linalg.eigh: (Batch element 0): The algorithm failed to "
                      "converge"), ("diverged", None)),
        (RuntimeError("torch.linalg.solve: The matrix is ill-conditioned"),
         ("diverged", None)),
        (ValueError("boom"), ("error", None)),
        (KeyError("optim_name"), ("error", None)),
    ]
    for exc, expected in cases:
        assert _classify_failure(exc) == expected, type(exc).__name__
    # DivergedError is a RuntimeError, so the isinstance chain must not reorder
    assert isinstance(DivergedError(0), RuntimeError)
    assert isinstance(oom, RuntimeError)


@pytest.mark.parametrize("exc,status", [
    (ValueError("injected"), "error"),
    (torch.cuda.OutOfMemoryError("CUDA out of memory (injected)"), "oom"),
])
def test_an_injected_failure_is_recorded_and_retried(tmp_path, monkeypatch, exc, status):
    """An `error` / `oom` record is written (C-R1) and, unlike `ok`/`diverged`,
    does NOT count as done: the next pass re-executes the run and finishes it.

    A failure this early leaves no curves at all, which is exactly why the record
    has to exist -- it is the only difference between "attempted and failed" and
    "never started".
    """
    scan_dir = tmp_path / "scan"
    real = generic_scan.train_loop_standard

    def boom(*a, **kw):
        raise exc

    monkeypatch.setattr(generic_scan, "train_loop_standard", boom)
    counts = run(scan_dir, mode="standard", **GOOD)
    assert counts[status] == 1 and len(records(scan_dir)) == 1
    run_id, record = next(iter(records(scan_dir).items()))
    assert record["status"] == status
    assert record["error"] == {"type": type(exc).__name__, "message": str(exc)}
    # One curve schema for every status (C-R1): nothing trained, so the five
    # timing scalars are nan and there is no curve -- but the keys are there, so
    # a notebook reading `avg_train_time` does not raise on a failed run.
    assert set(record["losses"]) == _FAILED_CURVE_KEYS
    assert np.isnan(record["losses"]["avg_train_time"])
    assert record["diverged_at_step"] is None
    # the record is complete even though nothing trained
    assert record["steps_per_epoch"] == N_TRAIN // BATCH
    assert record["n_train"] == N_TRAIN and record["n_test"] == N_TEST
    assert len(record["run_hash"]) == 64 and record["schema_version"] == 2
    h8 = record["run_hash"][:8]
    assert (run_id, h8, status) in done_markers(scan_dir)

    monkeypatch.setattr(generic_scan, "train_loop_standard", real)
    assert run(scan_dir, mode="standard", **GOOD)["ok"] == 1
    assert records(scan_dir)[run_id]["status"] == "ok"
    assert (run_id, h8, "ok") in done_markers(scan_dir)


def test_a_deterministic_failure_stops_being_retried_after_three_attempts(tmp_path):
    """`oom` and `error` are retried by design, but the done marker cannot count
    attempts (one file per hash and status), and under `scheduler: claims` every
    worker of every job walks the whole grid -- so a DETERMINISTIC failure (a bad
    `_target_`, `batch_size > n_train`, a missing token file, a Muon raise) would
    be re-executed for the rest of the campaign, rewriting its record every time.
    Three failed attempts at one hash are enough; the fourth pass reports
    `poisoned` and leaves it alone. The record and the marker stay, so reconcile
    still counts it as a failure.
    """
    scan_dir = tmp_path / "scan"
    real = generic_scan.train_loop_standard

    def boom(*a, **kw):
        raise ValueError("injected, always")

    generic_scan.train_loop_standard = boom
    try:
        for attempt in range(1, 4):
            counts = run(scan_dir, mode="standard", **GOOD)
            assert counts["error"] == 1 and not counts["poisoned"], attempt
        counts = run(scan_dir, mode="standard", **GOOD)
        assert counts["poisoned"] == 1 and not counts["error"]
    finally:
        generic_scan.train_loop_standard = real

    run_id = next(iter(records(scan_dir)))
    h8 = records(scan_dir)[run_id]["run_hash"][:8]
    attempts = os.listdir(os.path.join(scan_dir, generic_scan.ATTEMPTS_DIRNAME))
    assert len(attempts) == 3 and all(n.startswith(f"{run_id}.{h8}.error.")
                                      for n in attempts)
    assert (run_id, h8, "error") in done_markers(scan_dir)
    # a successful attempt writes no marker, and clearing them re-enables the run
    for name in attempts:
        os.unlink(os.path.join(scan_dir, generic_scan.ATTEMPTS_DIRNAME, name))
    assert run(scan_dir, mode="standard", **GOOD)["ok"] == 1
    assert os.listdir(os.path.join(scan_dir, generic_scan.ATTEMPTS_DIRNAME)) == []


def test_a_mid_training_failure_keeps_the_curve_keys_of_a_finished_run(tmp_path):
    """C-R1: ok / diverged / oom / error share ONE curve shape.

    Only `DivergedError` runs the loops' own tail (`_partial_record_on_divergence`),
    so a mid-training failure of any other kind -- a `LinAlgError`, Sven's
    empty-spectrum guard, a HIG or K-FAC solve -- used to leave a record whose
    `losses` had the curves but none of `total_time` / `avg_train_time` /
    `avg_eval_time` / the C-E5 summary. `status` does not separate the two cases
    (both are `diverged`), so a notebook could only find out by catching KeyError.
    """
    ok_dir, bad_dir = tmp_path / "ok", tmp_path / "bad"
    assert run(ok_dir, mode="standard", num_epochs=2, **GOOD)["ok"] == 1
    finished = next(iter(records(ok_dir).values()))["losses"]

    real = generic_scan.train_loop_standard

    def die_mid_training(*a, **kw):
        curves = kw["losses"]                       # the caller-owned dict (C-R1)
        curves.setdefault("val", []).extend([1.0, 0.5])
        curves.setdefault("train", []).append(0.7)
        curves.setdefault("epoch_times", []).append(0.01)
        raise torch.linalg.LinAlgError("linalg.eigh: failed to converge")

    generic_scan.train_loop_standard = die_mid_training
    try:
        assert run(bad_dir, mode="standard", num_epochs=2, **GOOD)["diverged"] == 1
    finally:
        generic_scan.train_loop_standard = real

    record = next(iter(records(bad_dir).values()))
    assert record["status"] == "diverged" and record["diverged_at_step"] is None
    assert _FAILED_CURVE_KEYS <= set(record["losses"])
    assert _FAILED_CURVE_KEYS <= set(finished)
    # the partial curves are kept, and the C-E5 summary is computed from them
    assert record["losses"]["val"] == [1.0, 0.5]
    assert record["val_final"] == 0.5 and record["val_best"] == 0.5
    assert record["losses"]["total_time"] > 0


def test_a_failure_before_the_loaders_exist_is_still_a_record(tmp_path):
    """The objects the failure path reads are pre-bound to None before the `try`,
    so a run that dies before its loaders / optimizer exist still writes a record
    -- with the facts that come from them as None rather than as a guess."""
    def raising(self, spec):
        raise RuntimeError("no loaders for you")

    scan_dir = tmp_path / "scan"
    original = generic_scan._ScanContext.loaders
    generic_scan._ScanContext.loaders = raising
    try:
        assert run(scan_dir, mode="standard", **GOOD)["error"] == 1
    finally:
        generic_scan._ScanContext.loaders = original
    record = next(iter(records(scan_dir).values()))
    assert record["status"] == "error" and record["error"]["type"] == "RuntimeError"
    assert record["steps_per_epoch"] is None
    assert record["effective_loader_seed"] is None
    assert record["ckpt_error"] is None and record["n_params"] == 13
    assert record["ckpt_file"] is None                 # nothing was ever collected


def test_the_started_marker_lives_exactly_as_long_as_the_attempt(tmp_path):
    """`{scan}/started/{run_id}.started` exists while the run runs and is gone
    afterwards, whatever the status (C-R1): a marker left behind with no record is
    the one failure mode that cannot write its own record -- a timeout or a kill.
    The claim is held for the same window and released with it.
    """
    scan_dir = tmp_path / "scan"
    real = generic_scan.train_loop_standard
    seen = {}

    def spy(*a, **kw):
        seen["started"] = claims.started_ids(scan_dir)
        seen["claims"] = os.listdir(os.path.join(scan_dir, "claims"))
        return real(*a, **kw)

    generic_scan.train_loop_standard = spy
    try:
        run(scan_dir, mode="standard", **GOOD)
    finally:
        generic_scan.train_loop_standard = real

    run_id = next(iter(records(scan_dir)))
    assert seen["started"] == {run_id}
    assert seen["claims"] == [run_id + claims.CLAIM_SUFFIX]
    assert claims.started_ids(scan_dir) == set()
    assert os.listdir(os.path.join(scan_dir, "claims")) == []


# ---------------------------------------------------------------------------
# C-R3: identity, dedup and stale generations
# ---------------------------------------------------------------------------

def test_rerunning_an_unchanged_scan_executes_nothing(tmp_path):
    """The dedup rule: a done marker with THIS run's hash8 and status ok is a
    skip, and nothing on disk is touched again (not even rewritten)."""
    scan_dir = tmp_path / "scan"
    assert run(scan_dir, **GOOD)["ok"] == 3
    before = {p: p.stat().st_mtime_ns for p in sorted(Path(scan_dir).rglob("*"))
              if p.is_file()}
    time.sleep(0.01)
    counts = run(scan_dir, **GOOD)
    assert counts["skipped"] == 3 and not counts["ok"] and not counts["error"]
    after = {p: p.stat().st_mtime_ns for p in sorted(Path(scan_dir).rglob("*"))
             if p.is_file()}
    # the manifest and the saved config are rewritten every job; the runs are not
    rewritten = {p.name for p in before if before[p] != after.get(p)}
    assert not {n for n in rewritten if n.endswith((".jsonl", ".npz", ".pt"))}
    assert set(before) == set(after)


def test_changing_num_epochs_reruns_everything_and_retires_the_old_files(tmp_path):
    """C-R3 acceptance test: change `num_epochs`, run again -- every run
    re-executes and every one of the old files (jsonl, npz, ckpt and the old
    generation's done marker) is MOVED under `_stale/{old_hash8}/`, keeping its
    relative layout so the analysis can still read it. Nothing is deleted.
    """
    scan_dir = tmp_path / "scan"
    assert run(scan_dir, num_epochs=2, **GOOD)["ok"] == 3
    first = records(scan_dir)
    old_hashes = {rid: r["run_hash"][:8] for rid, r in first.items()}
    assert all(len(r["losses"]["val"]) == 3 for r in first.values())   # 2 epochs + init

    counts = run(scan_dir, num_epochs=3, **GOOD)
    assert counts["ok"] == 3 and counts["retired"] == 3 and not counts["skipped"]

    second = records(scan_dir)
    assert set(second) == set(first)                      # same run_ids...
    for run_id, record in second.items():
        old8 = old_hashes[run_id]
        assert record["run_hash"][:8] != old8             # ... different identity
        assert record["num_epochs"] == 3
        assert len(record["losses"]["val"]) == 4
        assert record["ckpt_file"] == f"ckpt/{run_id}.pt"
        stale = Path(scan_dir) / "_stale" / old8
        for kind, path in artefacts(scan_dir, run_id, root=stale).items():
            assert path.exists(), (run_id, kind)
        assert (stale / "done" / f"{run_id}.{old8}.ok").exists()
        with open(stale / f"{run_id}.jsonl") as fh:       # the OLD record, intact
            assert json.load(fh)["num_epochs"] == 2
        assert (run_id, record["run_hash"][:8], "ok") in done_markers(scan_dir)
        assert (run_id, old8, "ok") not in done_markers(scan_dir)
    # and now the new generation is the one that gets skipped
    assert run(scan_dir, num_epochs=3, **GOOD)["skipped"] == 3


def test_a_stale_generation_written_after_the_index_is_still_retired(tmp_path):
    """The done index is one listdir at process start, so a generation that
    appears later is invisible to it. Acting on the snapshot would leave the old
    jsonl/npz/ckpt in place to be OVERWRITTEN (their paths carry no hash), which
    is why the retire check is re-done from disk for every run that executes.
    """
    scan_dir = tmp_path / "scan"
    run(scan_dir, num_epochs=2, **GOOD, mode="standard")
    run_id = next(iter(records(scan_dir)))
    old8 = records(scan_dir)[run_id]["run_hash"][:8]

    # a worker whose index predates that result: same effect as building the index
    # before the marker existed
    index_before = claims.done_index(scan_dir)
    assert claims.stale_hashes(index_before, run_id, "ffffffff") == [old8]
    assert generic_scan._stale_hashes_now(scan_dir, run_id, "ffffffff") == [old8]
    empty = {}
    assert claims.stale_hashes(empty, run_id, "ffffffff") == []      # the blind spot

    counts = run(scan_dir, num_epochs=3, **GOOD, mode="standard")
    assert counts["retired"] == 1
    assert (Path(scan_dir) / "_stale" / old8 / f"{run_id}.jsonl").exists()
    with open(Path(scan_dir) / f"{run_id}.jsonl") as fh:
        assert json.load(fh)["num_epochs"] == 3


def test_the_manifest_the_config_and_the_provenance_are_written_per_job(tmp_path):
    """C-R2: every worker declares the run_ids it is responsible for, and the
    UNION over the workers is the scan's intended grid -- which is what
    `tools/reconcile.py` counts "expected / ok / diverged / never started"
    against. C-R3: the resolved config of each job is saved once, and every record
    carries both repos' provenance and its own timings.
    """
    scan_dir = tmp_path / "scan"
    run(scan_dir, mode="svd", **GOOD)
    run(scan_dir, mode="standard", **GOOD)
    manifests = sorted(p.name for p in (Path(scan_dir) / "manifest").glob("*.json"))
    assert len(manifests) == 2, manifests            # one per work item, not per run
    assert claims.read_manifest_union(scan_dir) == set(run_ids_of(mode="both", **GOOD))
    assert set(records(scan_dir)) == claims.read_manifest_union(scan_dir)

    configs = sorted((Path(scan_dir) / "configs").glob("*.yaml"))
    assert len(configs) == 2
    saved = OmegaConf.load(configs[0])
    assert saved.num_epochs == BASE_CFG["num_epochs"] and "dataset" in saved

    record = records(scan_dir)[run_ids_of(mode="svd", **GOOD)[0]]
    for key in ("git_sha", "git_dirty", "git_source", "sven_git_sha",
                "sven_git_dirty", "host", "slurm_job_id", "torch_version",
                "cuda_version", "gpu_name", "python_version", "n_shards",
                "shard_id", "start_time", "end_time", "wall_time_s", "run_hash"):
        assert key in record, key
    assert record["torch_version"] == torch.__version__
    assert record["wall_time_s"] >= 0 and len(record["run_hash"]) == 64
    assert record["n_shards"] == 1 and record["shard_id"] == 0


def test_the_saved_config_reproduces_the_records_run_hash(tmp_path):
    """C-R3: `{scan}/configs/{job}.yaml` is the POST-mutation config, so a
    torch-free consumer (`tools/reconcile.py`) can recompute `hash8` from it and
    get the value the runner used.

    This is the only way to get it right for an LM scan: `run_grid` injects the
    dataset's `vocab_size` / `block_size` into `cfg.model` before hashing
    (`grid.inject_dataset_facts`, because `model/nanogpt.yaml` omits the vocab),
    so composing the live config and hashing it unchanged gives a different hash8
    for every nanoGPT run -- and a finished scan then reports as "work remains".
    """
    scan_dir = tmp_path / "scan"
    run(scan_dir, mode="standard", **GOOD)
    saved = next((scan_dir / "configs").glob("*.yaml"))
    rcfg = OmegaConf.to_container(OmegaConf.load(saved), resolve=True)
    specs = {s.run_id: s for s in grid.expand_grid(rcfg, verbose=False)}
    assert records(scan_dir)
    for run_id, record in records(scan_dir).items():
        assert grid.hash8(specs[run_id], rcfg) == record["run_hash"][:8], run_id


def test_work_items_that_produce_different_run_ids_get_different_manifests(tmp_path):
    """`n_data` is an item axis in `campaign/plan_campaign.yaml`: six jobs of one
    scan, six disjoint sets of run_ids (it is in `result_id_fields`). If those jobs
    shared a manifest name the last one would replace the others and the union --
    i.e. what reconcile calls "expected" -- would be a strict subset of the grid.
    """
    scan_dir = tmp_path / "scan"
    shared = dict(GOOD, mode="standard", num_epochs=2, result_id_fields=["n_data"])
    for n_data in (16, 32):
        run(scan_dir, n_data=n_data, dataset={**BASE_CFG["dataset"], "n_train": n_data},
            **shared)
    assert len(list((Path(scan_dir) / "manifest").glob("*.json"))) == 2
    assert len(records(scan_dir)) == 2
    assert claims.read_manifest_union(scan_dir) == set(records(scan_dir))
    assert all("n_data" in rid for rid in records(scan_dir))


# ---------------------------------------------------------------------------
# Scheduling: the claim queue and the static fallback
# ---------------------------------------------------------------------------

def _worker(scan_dir, overrides, out_path, barrier):
    """One forked worker: walk the whole grid, report which runs IT executed.

    ``execute`` is replaced by a stub that writes the record a real run would and
    returns ``ok``, for two reasons. The property under test is the lifecycle
    *around* ``execute`` -- claim, write, mark, release -- and every other test in
    this file already runs the real thing; and calling into torch from a process
    forked out of a torch-loaded pytest session is the classic libgomp
    after-fork deadlock (observed: both children hung past a 240 s join in a
    full-suite run, where earlier test files had already warmed torch's thread
    pool). The stub keeps the child's torch use to what ``run_grid`` itself does.
    """
    executed = []

    def stub(spec, ctx, run_hash=None, prov=None):
        executed.append(spec.run_id)
        time.sleep(0.02)                  # a window for the other worker to collide
        generic_scan._write_run(
            ctx.scan_dir, spec.run_id,
            {"run_id": spec.run_id, **spec.record_extra, "status": "ok",
             "run_hash": run_hash, "losses": {"train": [1.0], "train_batch": [1.0]}},
            ctx.svd_info_mode, ctx.spectra_schedule)
        return "ok"

    generic_scan.execute = stub
    barrier.wait()                        # both workers enter the scan together
    try:
        run_grid(cfg_of(**overrides), str(scan_dir))
    finally:
        with open(out_path, "w") as fh:
            json.dump(executed, fh)


def test_two_concurrent_workers_execute_each_run_exactly_once(tmp_path):
    """The claim queue (CONTRACTS "Scheduling"): both workers walk the FULL grid
    -- so a resubmitted job mops up whatever is left instead of re-running its own
    static slice -- and the `O_CREAT|O_EXCL` claim is what keeps them from doing
    the same run twice. Duplicate work here would also mean two workers writing
    the same jsonl/npz at the same time.
    """
    scan_dir = tmp_path / "scan"
    overrides = {"mode": "standard", "num_epochs": 2,
                 "lrs_standard": [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3]}
    expected = run_ids_of(**overrides)
    assert len(expected) == 8

    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(2)
    procs, outs = [], []
    for i in range(2):
        out = tmp_path / f"executed_{i}.json"
        proc = ctx.Process(target=_worker, args=(scan_dir, overrides, str(out), barrier))
        proc.start()
        procs.append(proc)
        outs.append(out)
    for proc in procs:
        proc.join(120)
        if proc.exitcode is None:         # never leave a hung child behind
            proc.terminate()
            proc.join(10)
            raise AssertionError(f"worker {proc.pid} did not finish within 120 s")
        assert proc.exitcode == 0, proc.exitcode

    executed = [json.load(open(p)) for p in outs]
    flat = [r for worker in executed for r in worker]
    assert sorted(flat) == sorted(expected), "a run was executed twice or not at all"
    assert min(len(w) for w in executed) >= 1, f"one worker did nothing: {executed}"
    assert {s for _, _, s in done_markers(scan_dir)} == {"ok"}
    assert len(records(scan_dir)) == 8
    assert claims.started_ids(scan_dir) == set()
    assert os.listdir(os.path.join(scan_dir, "claims")) == []


def test_a_live_claim_is_respected_and_a_stale_one_is_taken_over(tmp_path):
    """A claim younger than `CLAIM_TIMEOUT_S` means a live worker: skip the run
    and leave it to them. An older one means its worker died, and the run is taken
    over by creating the NEXT generation of the claim -- the dead worker's file
    stays behind as the record that somebody died on it.
    """
    scan_dir = tmp_path / "scan"
    os.makedirs(scan_dir, exist_ok=True)
    live, stale = run_ids_of(mode="standard", lrs_standard=[1e-3, 2e-3])
    held = claims.try_claim(scan_dir, live)               # a live worker's claim
    dead = claims.try_claim(scan_dir, stale)
    os.utime(dead.path, (time.time() - 10 * claims.CLAIM_TIMEOUT_S,) * 2)

    counts = run(scan_dir, mode="standard", lrs_standard=[1e-3, 2e-3], num_epochs=2)
    assert counts["ok"] == 1 and counts["claimed_elsewhere"] == 1
    assert set(records(scan_dir)) == {stale}              # the live one was left alone
    assert os.path.exists(held.path), "a live claim must not be removed"
    assert os.path.exists(dead.path), "the dead worker's claim is the evidence"
    assert not os.path.exists(dead.path + ".1"), "the takeover generation is released"
    # the taken-over run is finished; the live one is still unclaimed work
    assert (stale, records(scan_dir)[stale]["run_hash"][:8], "ok") in done_markers(scan_dir)
    assert claims.release(held) and run(scan_dir, mode="standard",
                                       lrs_standard=[1e-3, 2e-3],
                                       num_epochs=2)["ok"] == 1


def test_a_run_held_during_the_walk_is_picked_up_by_the_mop_up_pass(tmp_path):
    """The walk never returns to a run another worker held, so without a second
    pass a worker that was hard-killed (timeout, SIGKILL, node failure) has its
    in-flight runs skipped by every job that was already walking -- and the pass
    still exits 0, which is indistinguishable from a finished scan. One mop-up
    pass with a fresh index picks up whatever the siblings finished or abandoned;
    the claim is what keeps it from duplicating live work.
    """
    scan_dir = tmp_path / "scan"
    first, second = run_ids_of(mode="standard", lrs_standard=[1e-3, 2e-3])
    real_try_claim, held = claims.try_claim, {"n": 0}

    def busy_once(dir_, run_id, *a, **kw):
        """`second` looks claimed while the walk passes it, free afterwards."""
        if run_id == second and held["n"] == 0:
            held["n"] += 1
            return None
        return real_try_claim(dir_, run_id, *a, **kw)

    claims.try_claim = busy_once
    try:
        counts = run(scan_dir, mode="standard", lrs_standard=[1e-3, 2e-3],
                     num_epochs=2)
    finally:
        claims.try_claim = real_try_claim

    assert held["n"] == 1
    assert counts["ok"] == 2 and not counts["claimed_elsewhere"]
    assert set(records(scan_dir)) == {first, second}
    assert claims.started_ids(scan_dir) == set()


def test_static_sharding_still_slices_the_grid_and_is_rejected_under_the_queue(tmp_path):
    """`scheduler=static` keeps `specs[shard_id::n_shards]` as the fallback, with
    one manifest per shard (workers of one SLURM job share `$SLURM_JOB_ID`, so a
    single name would let the last writer replace the others' slice). Combining
    the slicing with the claim queue is refused: a sharded worker walks only its
    own slice and could not mop up another shard's leftovers, which is the whole
    point of the queue.
    """
    scan_dir = tmp_path / "scan"
    with pytest.raises(ValueError, match="scheduler=static"):
        run(scan_dir, n_shards=2, shard_id=0, **GOOD)

    shared = dict(GOOD, scheduler="static", num_epochs=2, mode="standard",
                  lrs_standard=[1e-3, 2e-3, 3e-3])
    assert run(scan_dir, n_shards=2, shard_id=0, **shared)["ok"] == 2
    assert run(scan_dir, n_shards=2, shard_id=1, **shared)["ok"] == 1
    assert set(records(scan_dir)) == set(run_ids_of(**shared))
    names = sorted(p.name for p in (Path(scan_dir) / "manifest").glob("*.json"))
    assert len(names) == 2 and all(".shard" in n for n in names)
    assert claims.read_manifest_union(scan_dir) == set(run_ids_of(**shared))
    record = next(iter(records(scan_dir).values()))
    assert record["n_shards"] == 2 and record["shard_id"] in (0, 1)
