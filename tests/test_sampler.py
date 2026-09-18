"""CPU tests for ``experiments/experiment_code/sampler.py`` (C-S1, C-S2, C-S3).

The acceptance tests of the spec, in order:

* the batch reconstructed offline for a ``(epoch, step)`` equals what the
  ``DataLoader`` actually yielded (C-S2, needed by the checkpoint tools);
* permutations differ across epochs and across loader seeds, and are identical
  across processes -- checked in a subprocess, not just in-process;
* ``seed_for_run`` is stable across interpreter runs with ``PYTHONHASHSEED``
  varied (C-S1: ``zlib.crc32``, not the salted builtin ``hash``);
* ``drop_last=True`` is the default for every optimizer (C-S3);
* ``derive_loader_seed`` depends on the model seed and nothing else.

Plus the two ways this module can be used wrongly without any error: the record
keeps the *base* loader seed while the sampler runs on the effective one
(``batch_indices_for_run`` / ``EpochPermutationSampler.for_run``), and the
forgotten-``set_epoch`` warning must not fire on the deliberate second pass of
a per-epoch evaluation.

``sampler.py`` is loaded straight from its path: the
``experiments.experiment_code`` package ``__init__`` imports ``generic_scan``
(hydra, sven, torchvision), which is both slow and owned by another track, and
this module is deliberately self-contained.
"""

import importlib.util
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
SAMPLER_PATH = REPO / "experiments" / "experiment_code" / "sampler.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sampler = _load(SAMPLER_PATH, "sv3_sampler_under_test")

# (loader_seed, epoch) pairs and (model_seed, run_id) pairs probed in the child
# process as well as here.
PERM_CASES = [(7, 0), (7, 1), (7, 5), (9, 0)]
RUN_CASES = [(0, "svd_bs64_k16_lr0.1_mseed0_lseed1234"),
             (3, "standard_bs64_lr0.001_optimadam_mseed3_lseed1234"),
             (12345, "")]
PERM_N = 17


# ---------------------------------------------------------------------------
# One child process per PYTHONHASHSEED, reporting the same derivations
# ---------------------------------------------------------------------------

_CHILD = f"""
import importlib.util, json, os, sys
spec = importlib.util.spec_from_file_location("child_sampler", {str(SAMPLER_PATH)!r})
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
print(json.dumps({{
    "hashseed": os.environ.get("PYTHONHASHSEED"),
    "hash_randomization": bool(sys.flags.hash_randomization),
    "perms": {{f"{{ls}}:{{e}}": mod.epoch_permutation({PERM_N}, ls, e).tolist()
               for ls, e in {PERM_CASES!r}}},
    "seed_for_run": {{f"{{ms}}:{{rid}}": mod.seed_for_run(ms, rid)
                      for ms, rid in {RUN_CASES!r}}},
    "loader_seeds": {{str(ms): mod.derive_loader_seed(1234, ms) for ms in (0, 1, 2, 3)}},
}}))
"""


@pytest.fixture(scope="module")
def children():
    """The child payload for two different ``PYTHONHASHSEED`` settings."""
    out = []
    for hashseed in ("0", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        proc = subprocess.run([sys.executable, "-c", _CHILD], env=env, cwd=str(REPO),
                              capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stderr[-2000:]
        out.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert out[0]["hashseed"] == "0" and out[1]["hashseed"] == "4242"
    return out


# ---------------------------------------------------------------------------
# C-S2: the DataLoader order is the reconstructable pure function
# ---------------------------------------------------------------------------

def test_offline_batch_reconstruction_matches_the_dataloader():
    n, batch_size, num_epochs = 20, 6, 3          # 3 full batches per epoch, 2 dropped
    dataset = TensorDataset(torch.arange(n, dtype=torch.float64).unsqueeze(1))
    batch_sampler = sampler.EpochPermutationSampler(n, loader_seed=7, batch_size=batch_size)
    loader = DataLoader(dataset, batch_sampler=batch_sampler)
    assert len(loader) == 3

    seen = []
    for epoch in range(num_epochs):
        sampler.set_loader_epoch(loader, epoch)
        for index, (xb,) in enumerate(loader):
            step = epoch * len(batch_sampler) + index
            yielded = xb.squeeze(1).to(torch.int64).tolist()
            offline = sampler.batch_indices(n, 7, batch_size, step).tolist()
            assert yielded == offline, f"step {step}"
            assert batch_sampler.batch_indices(step).tolist() == yielded
            seen.append(yielded)

    # every full batch of an epoch is disjoint and drawn from the epoch permutation
    for epoch in range(num_epochs):
        flat = [i for batch in seen[epoch * 3:(epoch + 1) * 3] for i in batch]
        assert len(set(flat)) == 18
        assert set(flat) <= set(range(n))


def test_batch_indices_for_run_derives_the_effective_seed():
    """The base-vs-effective seed trap: both calls succeed, only one is right.

    The run_id and the record keep the scan's *base* ``loader_seed``, while the
    sampler runs on ``derive_loader_seed(base, model_seed)``.  Passing the base
    to :func:`batch_indices` returns a different, perfectly plausible batch with
    no error, so an offline spectrum would silently answer another question.
    """
    n, base, model_seed, batch_size = 20, 1234, 3, 6
    effective = sampler.derive_loader_seed(base, model_seed)
    batch_sampler = sampler.EpochPermutationSampler.for_run(n, base, model_seed, batch_size)
    assert batch_sampler.loader_seed == effective != base
    # what the runner records next to the base seed it keeps in the run_id
    assert batch_sampler.base_loader_seed == base and batch_sampler.model_seed == model_seed
    assert sampler.EpochPermutationSampler(n, base, batch_size).base_loader_seed is None

    dataset = TensorDataset(torch.arange(n, dtype=torch.float64).unsqueeze(1))
    loader = DataLoader(dataset, batch_sampler=batch_sampler)
    for epoch in range(2):
        sampler.set_loader_epoch(loader, epoch)
        for index, (xb,) in enumerate(loader):
            step = epoch * len(batch_sampler) + index
            yielded = xb.squeeze(1).to(torch.int64).tolist()
            assert sampler.batch_indices_for_run(
                n, base, model_seed, batch_size, step).tolist() == yielded
            assert sampler.batch_indices(n, effective, batch_size, step).tolist() == yielded

    # the mistake the wrapper exists to prevent: the base seed gives a valid,
    # plausible, wrong batch rather than an error
    for step in range(3):
        wrong = sampler.batch_indices(n, base, batch_size, step).tolist()
        right = sampler.batch_indices_for_run(n, base, model_seed, batch_size, step).tolist()
        assert len(wrong) == batch_size and set(wrong) <= set(range(n))
        assert wrong != right
    assert (sampler.batch_indices_for_run(n, base, 3, batch_size, 0).tolist()
            != sampler.batch_indices_for_run(n, base, 4, batch_size, 0).tolist())


def test_drop_last_default_drops_the_tail_and_false_keeps_it():
    n, batch_size = 20, 6
    default = sampler.EpochPermutationSampler(n, 7, batch_size)
    assert default.drop_last is True and len(default) == 3          # C-S3
    assert all(len(b) == batch_size for b in default)

    keep = sampler.EpochPermutationSampler(n, 7, batch_size, drop_last=False)
    assert len(keep) == 4
    batches = list(keep)
    assert [len(b) for b in batches] == [6, 6, 6, 2]
    assert sorted(i for b in batches for i in b) == list(range(n))  # a full epoch
    with pytest.raises(ValueError):
        sampler.EpochPermutationSampler(4, 7, 8)                    # no full batch


# ---------------------------------------------------------------------------
# Purity: across epochs, across loader seeds, across processes
# ---------------------------------------------------------------------------

def test_permutations_differ_across_epochs_and_loader_seeds():
    n = PERM_N
    per_epoch = [sampler.epoch_permutation(n, 7, e).tolist() for e in range(4)]
    assert len({tuple(p) for p in per_epoch}) == 4
    per_seed = [sampler.epoch_permutation(n, ls, 0).tolist() for ls in (7, 8, 9, 1234)]
    assert len({tuple(p) for p in per_seed}) == 4
    for perm in per_epoch + per_seed:
        assert sorted(perm) == list(range(n))          # still a permutation
    # pure: recomputing gives the same order, and the global RNG cannot change it
    torch.manual_seed(999)
    assert sampler.epoch_permutation(n, 7, 1).tolist() == per_epoch[1]
    _ = torch.rand(5)
    assert sampler.epoch_permutation(n, 7, 1).tolist() == per_epoch[1]


def test_permutations_are_identical_across_processes(children):
    mine = {f"{ls}:{e}": sampler.epoch_permutation(PERM_N, ls, e).tolist()
            for ls, e in PERM_CASES}
    assert children[0]["perms"] == mine
    assert children[1]["perms"] == mine


def test_warns_only_when_set_epoch_is_never_called(recwarn):
    """The guard covers a forgotten ``set_epoch``, not a deliberate second pass.

    Re-iterating one epoch is legitimate -- a per-epoch evaluation of the
    training set goes through a loader at the same epoch -- so warning on it
    would fire in every epoch of every run and drown the real failure, which is
    every epoch silently replaying epoch 0's order.
    """
    forgotten = sampler.EpochPermutationSampler(20, 7, 6)
    first = list(forgotten)
    assert not recwarn.list                            # first pass is fine
    with pytest.warns(RuntimeWarning, match="set_epoch"):
        assert list(forgotten) == first                # pure, just not advanced

    driven = sampler.EpochPermutationSampler(20, 7, 6)
    driven.set_epoch(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # any warning fails the test
        assert list(driven) == first
        assert list(driven) == first                   # a second pass at epoch 0
        driven.set_epoch(1)
        second_epoch = list(driven)
        assert list(driven) == second_epoch
    assert second_epoch != first


# ---------------------------------------------------------------------------
# C-S1: run seeds that do not depend on the interpreter or on position
# ---------------------------------------------------------------------------

def test_seed_for_run_is_the_contract_formula():
    import zlib
    for model_seed, run_id in RUN_CASES:
        assert (sampler.seed_for_run(model_seed, run_id)
                == model_seed ^ zlib.crc32(run_id.encode()))
    # distinct runs of one seed get distinct streams
    ids = [f"svd_bs64_k{k}_mseed0_lseed1234" for k in (1, 2, 4, 8, 16, 32, 64)]
    assert len({sampler.seed_for_run(0, i) for i in ids}) == len(ids)


def test_seed_for_run_is_stable_across_interpreter_runs(children):
    mine = {f"{ms}:{rid}": sampler.seed_for_run(ms, rid) for ms, rid in RUN_CASES}
    assert children[0]["seed_for_run"] == mine
    assert children[1]["seed_for_run"] == mine
    # the point of crc32: PYTHONHASHSEED actually differed between the children
    assert children[0]["hash_randomization"] != children[1]["hash_randomization"]
    assert {sampler.seed_for_run(0, i) for i in ("a", "b")} != {hash("a"), hash("b")}


def test_derive_loader_seed_varies_with_model_seed_only(children):
    seeds = {ms: sampler.derive_loader_seed(1234, ms) for ms in range(5)}
    assert len(set(seeds.values())) == 5               # F26: data order moves with the band
    assert {str(ms): seeds[ms] for ms in range(4)} == children[0]["loader_seeds"]
    assert {str(ms): seeds[ms] for ms in range(4)} == children[1]["loader_seeds"]
    # identical for every optimizer of one model seed: it takes no other argument
    assert sampler.derive_loader_seed(1234, 3) == sampler.derive_loader_seed(1234, 3)
    assert sampler.derive_loader_seed(1234, 3) != sampler.derive_loader_seed(1235, 3)


def test_set_loader_epoch_is_a_no_op_without_a_sampler():
    dataset = TensorDataset(torch.zeros(4, 1))
    plain = DataLoader(dataset, batch_size=2, shuffle=False)
    assert sampler.set_loader_epoch(plain, 3) is False
    with_sampler = DataLoader(dataset, batch_sampler=sampler.EpochPermutationSampler(4, 1, 2))
    assert sampler.set_loader_epoch(with_sampler, 3) is True
    assert with_sampler.batch_sampler.epoch == 3
