"""Per-run checkpointing for the training loops (C-L3).

Until now no ``state_dict`` was ever written (F18): every trained model was
discarded when the run returned, so a held-out re-evaluation, a BatchNorm-correct
re-evaluation, distance-from-initialisation or an offline Jacobian spectrum along
*any* optimizer's trajectory (C-L4) all required retraining.  This module is the
one place that writes them.

Design, from ``EXPERIMENTS.md section 12`` and the storage table in ``EXPERIMENTS.md``:

* **One file per run**, ``{scan}/ckpt/{run_id}.pt`` =
  ``{"step": [...], "epoch": [...], "state": [...]}``.  Per-checkpoint files
  would put ~10^5 small files per scan on Lustre.
* **Weights *and* buffers**, cloned to CPU in float32 (BatchNorm running stats
  are what make a correct re-evaluation possible; non-float buffers such as
  ``num_batches_tracked`` keep their dtype).  fp32 halves the ResNet footprint
  and is why the reload tolerance is 1e-6 rather than exact.
* Four policies, ``checkpoints: none | final | epochs | log``, where ``log`` =
  steps {0, 1, 2, 4, 8, ...} plus every epoch end and ``epochs`` = step 0 plus
  every epoch end.  Step 0 is the initial model; under ``final`` alone (one slot,
  overwritten) it is instead saved once per *seed* by :func:`save_init_state`
  (``ckpt/init_mseed{seed}.pt``), since it is shared by every run of that seed.
* Nothing is written until :meth:`Checkpointer.flush`, except that long runs
  (ResNet, GPT) pass ``rewrite_each_epoch=True`` and rewrite the whole file at
  each epoch end, so a 12 h timeout loses at most one epoch.  ``flush()`` is
  also safe to call from an exception handler: a diverged or crashed run keeps
  whatever it collected (the last state before a blow-up is the diagnostic), and
  a failing write never masks the original exception.
* Writes are atomic (``tmp`` + :func:`os.replace`), so a killed job never leaves
  a half-written checkpoint behind, and the runner's "jsonl last = dedup marker"
  invariant holds as long as the flush happens before the npz and the jsonl.

The training loops call :meth:`Checkpointer.maybe_save` before the update of
``step`` (so ``step`` labels the state *before* that step) and
:meth:`Checkpointer.epoch_end` after each epoch.  Offline consumers use
:func:`load_checkpoint` / :func:`load_state_at`; the batch of a recorded step is
reconstructed with ``sampler.batch_indices`` (C-S2) from the ``steps_per_epoch``
this module stores alongside the states.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Mapping

import torch
import torch.nn as nn

__all__ = [
    "POLICIES",
    "Checkpointer",
    "cpu_fp32_state",
    "load_checkpoint",
    "load_state_at",
    "save_init_state",
]

#: values of the ``checkpoints`` config key.
POLICIES = ("none", "final", "epochs", "log")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cpu_fp32_state(module_or_state: nn.Module | Mapping[str, Any]) -> dict[str, Any]:
    """``state_dict`` of *module_or_state*, cloned to CPU, floats in float32.

    Accepts a module or an already-extracted ``state_dict``.  Every tensor is
    copied (``copy=True``), never aliased: the module keeps training in place
    after the checkpoint is taken.  Non-float tensors (``num_batches_tracked``)
    keep their dtype, non-tensor entries are passed through.
    """
    items = (module_or_state.state_dict() if hasattr(module_or_state, "state_dict")
             else module_or_state).items()
    out: dict[str, Any] = {}
    for key, value in items:
        if torch.is_tensor(value):
            value = value.detach()
            if value.is_floating_point():
                out[key] = value.to("cpu", torch.float32, copy=True)
            else:
                out[key] = value.to("cpu", copy=True)
        else:
            out[key] = value
    return out


def _is_log_step(step: int) -> bool:
    """Steps {0, 1, 2, 4, 8, ...}: zero and the powers of two."""
    return step == 0 or (step > 0 and step & (step - 1) == 0)


def _atomic_save(payload: dict[str, Any], path: str) -> str:
    """``torch.save`` to a temporary file in the same directory, then rename.

    The temporary name carries a random token, not just the pid: the per-seed
    init file (:func:`save_init_state`) and a checkpoint written after a stale
    claim takeover can have two writers on different nodes, and identically
    configured SLURM nodes hand out the same pids, so a pid-only temp name lets
    two ``torch.save`` streams interleave in one file and ``os.replace``
    promote the mangled bytes.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)          # atomic within one filesystem
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


# ---------------------------------------------------------------------------
# Checkpointer
# ---------------------------------------------------------------------------

class Checkpointer:
    """Collects ``state_dict``s during one run and writes them as one file.

    Args:
        path: the run's checkpoint file, ``{scan}/ckpt/{run_id}.pt``.
        policy: one of :data:`POLICIES` (``None`` is accepted as ``"none"``).
        steps_per_epoch: batches per epoch (``len(train_loader)``).
        num_epochs: epochs the run is configured for.
        rewrite_each_epoch: rewrite the file at every epoch end instead of only
            at :meth:`flush` -- for runs long enough to be killed by a time
            limit.  A whole-file rewrite, so it is off by default.

    ``steps_per_epoch`` and ``num_epochs`` do not decide *which* steps are
    saved; they are recorded in the file so that offline tools can map a step to
    its epoch and reconstruct its batch (C-L4) with no access to the config.

    The collected checkpoints are readable as the parallel lists :attr:`steps`,
    :attr:`epochs` and :attr:`states`.
    """

    def __init__(self, path: str, policy: str | None, steps_per_epoch: int | None,
                 num_epochs: int | None, rewrite_each_epoch: bool = False) -> None:
        policy = "none" if policy is None else str(policy)
        if policy not in POLICIES:
            raise ValueError(f"unknown checkpoint policy {policy!r}, expected one of {POLICIES}")
        self.path = str(path)
        self.policy = policy
        self.steps_per_epoch = None if steps_per_epoch is None else int(steps_per_epoch)
        self.num_epochs = None if num_epochs is None else int(num_epochs)
        self.rewrite_each_epoch = bool(rewrite_each_epoch)

        self.steps: list[int] = []
        self.epochs: list[int] = []
        self.states: list[dict[str, Any]] = []
        self._index: dict[int, int] = {}   # step -> position, for dedup
        self._dirty = False                # something collected but not written
        #: ``repr`` of the last write failure, or ``None``.  `flush` must not
        #: raise (it is called from exception handlers), so this is how a full
        #: quota becomes visible: the runner puts it in the record as
        #: `ckpt_error` instead of marking a checkpoint-less run `ok`.
        self.last_error: str | None = None

    # -- collection -------------------------------------------------------

    def maybe_save(self, step: int, epoch: int, module: nn.Module) -> bool:
        """Save the state *before* optimizer step ``step``, if the policy says so.

        Called at the top of every batch, so ``step == 0`` is the initial model.
        ``log`` saves every power of two; ``epochs`` saves step 0 only -- its
        initial state is not shared per seed the way ``final``'s is
        (:func:`save_init_state`), and without it the one ``epochs`` scan
        (nanoGPT) would have no init anywhere: no distance from initialisation
        and no step-0 spectrum, for one extra state per run.  Returns whether a
        state was taken.
        """
        step, epoch = int(step), int(epoch)
        if self.policy == "log":
            if not _is_log_step(step):
                return False
        elif not (self.policy == "epochs" and step == 0):
            return False
        return self._record(step, epoch, module)

    # The first batch of epoch e+1 sees the step that `epoch_end(e, step)` has
    # just saved (and under `log` it may be a power of two), so `_record`
    # deduplicates by step and only the epoch end may relabel the epoch.

    def epoch_end(self, epoch: int, step: int, module: nn.Module) -> bool:
        """Save the state after epoch ``epoch``, which has just finished at ``step``.

        ``final`` keeps a single slot and overwrites it, so a run that dies in
        epoch 7 of 20 still leaves the end of epoch 6 on disk; ``epochs`` and
        ``log`` append.  Writes the file here when ``rewrite_each_epoch``.
        """
        saved = False
        if self.policy == "final":
            saved = self._record(int(step), int(epoch), module, single=True)
        elif self.policy in ("epochs", "log"):
            saved = self._record(int(step), int(epoch), module, relabel=True)
        if self.rewrite_each_epoch:
            self.flush()
        return saved

    def _record(self, step: int, epoch: int, module: nn.Module,
                single: bool = False, relabel: bool = False) -> bool:
        if self.policy == "none":
            return False
        if single:
            # `final`: one slot, overwritten, so a run killed in epoch 7 of 20
            # still leaves the end of epoch 6 on disk.
            self.steps, self.epochs, self.states, self._index = [], [], [], {}
        elif step in self._index:
            if relabel:
                # An epoch end owns its step's epoch label, so
                # `load_state_at(epoch=e)` means "end of epoch e".
                self.epochs[self._index[step]] = epoch
                self._dirty = True
            return False
        self._index[step] = len(self.steps)
        self.steps.append(step)
        self.epochs.append(epoch)
        self.states.append(cpu_fp32_state(module))
        self._dirty = True
        return True

    # -- writing ----------------------------------------------------------

    def flush(self) -> str | None:
        """Write the collected states to :attr:`path`; return the path or ``None``.

        Safe to call more than once (a clean re-flush is a no-op), with nothing
        collected (no file is created), and from an ``except`` block: a failing
        write is reported and swallowed rather than replacing the exception that
        is already on its way up.  Because it is swallowed on the success path
        too, the failure is also left in :attr:`last_error`; the runner records
        that as ``ckpt_error`` so a quota-failed write is not a silently
        checkpoint-less ``ok`` run.
        """
        if self.policy == "none" or not self.states:
            return None
        if not self._dirty:
            return self.path
        payload = {
            "step": list(self.steps),
            "epoch": list(self.epochs),
            "state": list(self.states),
            # provenance for the offline tools; not part of the C-L3 contract
            "policy": self.policy,
            "steps_per_epoch": self.steps_per_epoch,
            "num_epochs": self.num_epochs,
        }
        try:
            _atomic_save(payload, self.path)
        except Exception as exc:                       # noqa: BLE001 -- see docstring
            self.last_error = repr(exc)
            print(f"  [warn] checkpoint flush failed for {self.path}: {exc}")
            return None
        self.last_error = None
        self._dirty = False
        return self.path

    def __len__(self) -> int:
        return len(self.states)

    def __repr__(self) -> str:
        return (f"Checkpointer(path={self.path!r}, policy={self.policy!r}, "
                f"collected={len(self.states)})")


# ---------------------------------------------------------------------------
# Offline access
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, map_location: Any = "cpu") -> dict[str, Any]:
    """Load one run's checkpoint file as written by :meth:`Checkpointer.flush`."""
    return torch.load(path, map_location=map_location, weights_only=True)


def load_state_at(path: str | dict[str, Any], step: int | None = None,
                  epoch: int | None = None) -> dict[str, Any]:
    """One ``state_dict`` out of a checkpoint file.

    With neither argument: the last state (the final model under every policy).
    With ``step``: the state before that optimizer step, which must have been
    saved.  With ``epoch``: the state at the *end* of that epoch, i.e. the last
    state recorded with that epoch label.  *path* may also be an already-loaded
    checkpoint, so a caller walking a trajectory reads the file once.
    """
    if step is not None and epoch is not None:
        raise ValueError("pass step or epoch, not both")
    ckpt = load_checkpoint(path) if isinstance(path, (str, os.PathLike)) else path
    steps, epochs, states = ckpt["step"], ckpt["epoch"], ckpt["state"]
    if step is not None:
        matches = [i for i, s in enumerate(steps) if s == int(step)]
        if not matches:
            raise ValueError(f"no checkpoint at step {step}; saved steps: {list(steps)}")
        return states[matches[-1]]
    if epoch is not None:
        matches = [i for i, e in enumerate(epochs) if e == int(epoch)]
        if not matches:
            raise ValueError(f"no checkpoint for epoch {epoch}; saved epochs: {list(epochs)}")
        return states[matches[-1]]
    return states[-1]


def save_init_state(path: str, state_dict: Mapping[str, Any], overwrite: bool = False) -> str:
    """Save the shared per-seed initial state (``ckpt/init_mseed{seed}.pt``).

    Under the ``final`` policy every run of a model seed starts from the same
    initialisation, so it is stored once for the seed instead of once per run.
    Written in the same envelope as a run checkpoint (step 0, epoch 0), so
    :func:`load_checkpoint` / :func:`load_state_at` read it unchanged.  Existing
    files are left alone unless ``overwrite``: the first run of the seed writes
    it, the other few hundred skip a whole-file rewrite.

    The existence check is a TOCTOU -- at scan start every process for the seed
    sees no file -- which is harmless only because the writers are racing to
    write the *same* bytes (one seed, one initialisation) into per-writer
    temporary files, so whichever ``os.replace`` lands last still leaves the
    seed's init state.  That holds only with the unique temp name in
    :func:`_atomic_save`.
    """
    if not overwrite and os.path.exists(path):
        return path
    payload = {
        "step": [0],
        "epoch": [0],
        "state": [cpu_fp32_state(state_dict)],
        "policy": "init",
        "steps_per_epoch": None,
        "num_epochs": None,
    }
    return _atomic_save(payload, path)
