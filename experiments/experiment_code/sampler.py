"""Deterministic data order and position-independent run seeding (C-S1 - C-S3).

Three defects, one module:

* **F15 / C-S1.** The RNG was seeded once per model seed, so parameter masks,
  the randomized SVD, model construction and the loader's iterator all drew from
  one global stream: a masked run's result depended on its *position* in the
  process.  :func:`seed_for_run` gives every run its own stream,
  ``model_seed XOR crc32(run_id)`` -- ``zlib.crc32``, never Python's
  ``hash()``, which is salted per interpreter.
* **F26 / C-S2.** ``shuffle=True`` with one ``loader_seed`` for every model seed
  meant seed bands contained initialisation variance only, and the batch of a
  given step was unrecoverable afterwards.  :class:`EpochPermutationSampler`
  makes the data order a pure function of ``(loader_seed, epoch)``, so
  :func:`batch_indices` reconstructs the batch of any step offline with no
  stored state -- which is what the checkpoint tools need (C-L4) -- and
  :func:`derive_loader_seed` mixes the model seed into it so a seed band
  contains data-order variance too, while staying **identical across
  optimizers** for a given model seed, keeping the comparison paired.
* **F33 / C-S3.** ``drop_last`` was true only for Sven-microbatch runs.  The
  sampler drops the short tail batch by default for *every* optimizer, so all
  methods take the same number of steps on the same batches.

Every seed derivation here is a pure function of integers and strings: stable
across interpreter runs, machines and ``PYTHONHASHSEED``.  Used as
``DataLoader(dataset, batch_sampler=sampler)`` -- the sampler owns the batching,
so ``batch_size``, ``shuffle``, ``sampler`` and ``drop_last`` must not also be
passed to the ``DataLoader``.  The training loop calls :meth:`set_epoch` (or
:func:`set_loader_epoch`) once per epoch; without it every epoch would replay
the same permutation, which a second pass warns about.  Only the *training*
loader uses this sampler: the ``train_eval`` loader must be a separate
sequential ``DataLoader(train_dataset, shuffle=False)``, or the curve would
measure the ``drop_last``-truncated training set in permuted order.
"""

from __future__ import annotations

import warnings
import zlib

import torch
from torch.utils.data import Sampler

__all__ = [
    "EpochPermutationSampler",
    "batch_indices",
    "batch_indices_for_run",
    "derive_loader_seed",
    "epoch_permutation",
    "mix",
    "seed_for_run",
    "set_loader_epoch",
    "steps_per_epoch",
]


# ---------------------------------------------------------------------------
# Pure seed derivations
# ---------------------------------------------------------------------------

def mix(loader_seed: int, epoch: int) -> int:
    """Generator seed for one epoch's permutation.

    A CRC of the decimal pair: cheap, stable across interpreter runs (unlike
    ``hash``) and uncorrelated between neighbouring epochs, so epoch 0 and
    epoch 1 are unrelated orders rather than shifted ones.
    """
    return zlib.crc32(f"{int(loader_seed)}:{int(epoch)}".encode())


def derive_loader_seed(base_loader_seed: int, model_seed: int) -> int:
    """The run's ``loader_seed``, mixed from the scan's base seed and the model seed.

    Depends on the model seed but on nothing else -- not the optimizer, the
    hyperparameters or the run_id -- so a seed band varies the data order (F26)
    while all optimizers of one model seed see identical batches (C-S2).
    """
    return int(base_loader_seed) ^ zlib.crc32(f"mseed{int(model_seed)}".encode())


def seed_for_run(model_seed: int, run_id: str) -> int:
    """``model_seed XOR crc32(run_id)`` (C-S1): the per-run global seed.

    Set immediately before a run is built (after the model exists and the
    initial state is loaded, before the wrapper constructs its parameter mask),
    so the run no longer depends on its position in the process.
    """
    return int(model_seed) ^ zlib.crc32(run_id.encode())


# ---------------------------------------------------------------------------
# Data order
# ---------------------------------------------------------------------------

def steps_per_epoch(n: int, batch_size: int, drop_last: bool = True) -> int:
    """Batches in one epoch over ``n`` examples."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return int(n) // int(batch_size) if drop_last else -(-int(n) // int(batch_size))


def epoch_permutation(n: int, loader_seed: int, epoch: int) -> torch.Tensor:
    """The example order of epoch ``epoch``, as a pure function of the seed."""
    generator = torch.Generator().manual_seed(mix(loader_seed, epoch))
    return torch.randperm(int(n), generator=generator)


def batch_indices(n: int, loader_seed: int, batch_size: int, step: int,
                  drop_last: bool = True) -> torch.Tensor:
    """Dataset indices of the batch consumed by optimizer step ``step``.

    ``step`` is the global step counter of the training loops (0-based, running
    across epochs), which is also what the checkpoints are labelled with, so a
    checkpoint and its batch can be paired offline.

    ``loader_seed`` is the run's **effective** seed, i.e. the output of
    :func:`derive_loader_seed` -- the run_id and the record keep the scan's
    *base* seed, and passing that instead returns a different, perfectly
    plausible batch with no error at all.  Offline callers that start from a
    record should use :func:`batch_indices_for_run`.
    """
    per_epoch = steps_per_epoch(n, batch_size, drop_last)
    if per_epoch == 0:
        raise ValueError(f"no full batch of {batch_size} in {n} examples with drop_last=True")
    epoch, index = divmod(int(step), per_epoch)
    perm = epoch_permutation(n, loader_seed, epoch)
    return perm[index * int(batch_size):(index + 1) * int(batch_size)]


def batch_indices_for_run(n: int, base_loader_seed: int, model_seed: int, batch_size: int,
                          step: int, drop_last: bool = True) -> torch.Tensor:
    """:func:`batch_indices` from what a *record* holds (C-L4's entry point).

    The record and the run_id carry the scan's base ``loader_seed`` and the
    ``model_seed``; this derives the effective seed itself, so the base-vs-
    effective mistake cannot be made silently.
    """
    return batch_indices(n, derive_loader_seed(base_loader_seed, model_seed),
                         batch_size, step, drop_last)


class EpochPermutationSampler(Sampler[list[int]]):
    """Batch sampler whose epoch-``e`` order is ``randperm(n, seed=mix(loader_seed, e))``.

    Args:
        n: number of training examples.
        loader_seed: the run's loader seed (see :func:`derive_loader_seed`).
        batch_size: examples per batch.
        drop_last: drop the short tail batch (default, C-S3).

    Yields index lists, so it is the ``batch_sampler`` of a ``DataLoader``.
    """

    def __init__(self, n: int, loader_seed: int, batch_size: int, drop_last: bool = True) -> None:
        if int(n) <= 0:
            raise ValueError(f"n must be positive, got {n}")
        self.n = int(n)
        self.loader_seed = int(loader_seed)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        if steps_per_epoch(self.n, self.batch_size, self.drop_last) == 0:
            raise ValueError(f"no full batch of {batch_size} in {n} examples with drop_last=True")
        self.epoch = 0
        self._last_iterated: int | None = None
        self._epoch_set = False
        # Set by `for_run`, so the runner can record the effective seed.
        self.base_loader_seed: int | None = None
        self.model_seed: int | None = None

    @classmethod
    def for_run(cls, n: int, base_loader_seed: int, model_seed: int, batch_size: int,
                drop_last: bool = True) -> "EpochPermutationSampler":
        """Build the sampler of one run from the *base* seed and the model seed.

        The constructor takes the effective seed (C-S2), which is the one thing
        the run_id does not contain; this derives it, and keeps both inputs so
        the runner can record ``effective_loader_seed`` next to them.
        """
        sampler = cls(n, derive_loader_seed(base_loader_seed, model_seed), batch_size, drop_last)
        sampler.base_loader_seed = int(base_loader_seed)
        sampler.model_seed = int(model_seed)
        return sampler

    def set_epoch(self, epoch: int) -> "EpochPermutationSampler":
        """Select the epoch whose permutation the next iteration yields."""
        self.epoch = int(epoch)
        self._epoch_set = True
        return self

    def __len__(self) -> int:
        return steps_per_epoch(self.n, self.batch_size, self.drop_last)

    def __iter__(self):
        # Only the failure this guard exists for: a second pass with set_epoch
        # never called anywhere, i.e. every epoch silently replaying epoch 0.
        # Re-iterating one epoch on purpose is legitimate and must stay silent,
        # or the warning would fire in every run of every scan.
        if not self._epoch_set and self.epoch == 0 and self._last_iterated == 0:
            warnings.warn(
                "EpochPermutationSampler iterated twice with set_epoch() never called: the "
                "data order is a pure function of the epoch, so every epoch is replaying "
                "epoch 0.  Call set_epoch(epoch) (or sampler.set_loader_epoch(loader, "
                "epoch)) once per training epoch.",
                RuntimeWarning, stacklevel=2,
            )
        self._last_iterated = self.epoch
        perm = epoch_permutation(self.n, self.loader_seed, self.epoch)
        for index in range(len(self)):
            yield perm[index * self.batch_size:(index + 1) * self.batch_size].tolist()

    def batch_indices(self, step: int) -> torch.Tensor:
        """This sampler's :func:`batch_indices`, for the global step ``step``."""
        return batch_indices(self.n, self.loader_seed, self.batch_size, step, self.drop_last)

    def __repr__(self) -> str:
        return (f"EpochPermutationSampler(n={self.n}, loader_seed={self.loader_seed}, "
                f"batch_size={self.batch_size}, drop_last={self.drop_last}, epoch={self.epoch})")


def set_loader_epoch(loader, epoch: int) -> bool:
    """Call ``set_epoch(epoch)`` on *loader*'s sampler if it has one.

    One line for the training loops, which must not care whether they were
    handed an :class:`EpochPermutationSampler` (training) or a plain sequential
    loader (evaluation, or a legacy call site).  Returns whether anything
    happened.
    """
    for candidate in (getattr(loader, "batch_sampler", None), getattr(loader, "sampler", None)):
        if hasattr(candidate, "set_epoch"):
            candidate.set_epoch(epoch)
            return True
    return False
