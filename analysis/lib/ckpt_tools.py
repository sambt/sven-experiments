"""Offline checkpoint tools: reload a trained run and take its Jacobian (C-L4).

The per-step spectra the optimizer logs are of a *different random batch* on
every logged step and are capped at B values by construction, so comparing them
across training mixes batch noise with genuine evolution and they exist only
along Sven's trajectory (``EXPERIMENTS.md``, "Why per-step batch spectra are
not enough" / F19).  The campaign's ``checkpoints: log`` runs make the honest
measurement possible offline:

* :func:`load_run` rebuilds the dataset and the model of a recorded run exactly
  as ``generic_scan.execute`` did -- from the record plus the resolved Hydra
  config the job saved in ``{scan}/configs/`` -- and loads the state of any
  checkpointed step or epoch;
* :func:`verify_checkpoint` recomputes the run's recorded validation loss from
  that state (eval mode, example-weighted, the runner's own
  :func:`~experiments.experiment_code.experiment_utils.evaluate`) and returns
  the relative error.  It is part of the API, not a test: a spectrum measured
  from a mis-reconstructed model is worse than no spectrum;
* :func:`probe_set` is a FIXED subset of the training pool, drawn with the
  dataset's ``split_seed`` exactly the way the runner draws its ``train_eval``
  subset (C-E3), so it is identical for every run, every seed and every
  optimizer of a scan -- which is what makes Adam's spectra comparable with
  Sven's;
* :func:`jacobian_rows` / :func:`spectrum` take the float64 Jacobian of the
  **same rows Sven differentiates** (:class:`RowSpec`: ``loss**(kappa/2)`` or
  the signed-residual form ``sign(r)|r|**kappa``), so the singular values are of
  the matrix the optimizer actually inverts.  A trajectory
  (:func:`checkpoint_spectra`) takes that definition from the SCAN's Sven
  records (:func:`row_spec_for_scan`), not from each run's own record, because a
  baseline record carries no ``kappa`` at all;
* :func:`batch_of_step` reconstructs the training batch of any step from the
  seeds in the record (C-S2), so a logged per-step spectrum can be reproduced
  and compared against its probe-set counterpart.

Everything runs on the CPU in float64 at the campaign's sizes (toy P = 593,
polynomial P = 673, MNIST P = 27,562): the full 10,000-row pool Jacobian of the
MLPs is under 60 MB and its SVD takes seconds.

``verify_checkpoint`` defaults to float32 -- the dtype the runs trained and
evaluated in, and the dtype the checkpoints are stored in (C-L3) -- so its
residual measures the reconstruction, not a dtype change.  The Jacobians default
to float64, where the spectrum's tail is signal rather than the 1e-7 round-off
floor of a float32 Gram (F19, ``style.FLOAT32_NOISE_FLOOR``).

Sizing note: the Jacobian is built row-chunked but materialised in full (N x P),
because ``J J^T`` cannot be accumulated over row chunks.  :data:`MAX_JAC_BYTES`
refuses a combination that would not fit instead of swapping the node.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent.parent   # analysis/ (this module lives in analysis/lib/)
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    # `import experiments...` has to work whether this module was imported from
    # the repo root (tools/, pytest), from `analysis/` (a notebook's
    # `sys.path.insert(0, '.')`) or by path.  Only the ROOT is added: adding
    # `analysis/` too would put its module names ahead of everything else's for
    # every importer of this module.
    sys.path.insert(0, str(_REPO_ROOT))

from hydra.utils import instantiate                                   # noqa: E402
from omegaconf import OmegaConf                                       # noqa: E402
from torch.utils.data import DataLoader, Subset                       # noqa: E402

# The runner's OWN definitions, imported and never re-derived here: the loss
# registries and the signed-residual rule (generic_scan), the checkpoint
# envelope (checkpointing), the data order (sampler), the example-weighted
# eval-mode evaluation (experiment_utils) and the fixed-subset draw (datasets).
from experiments.experiment_code.generic_scan import (                # noqa: E402
    SVD_LOSS_FNS, SVD_RESIDUAL_FNS,
)
from experiments.experiment_code.grid import SIGNED_RESIDUAL_LOSS_KEYS  # noqa: E402
from experiments.experiment_code.checkpointing import (                # noqa: E402
    load_checkpoint, load_state_at,
)
from experiments.experiment_code.experiment_utils import evaluate      # noqa: E402
from experiments.experiment_code.sampler import (                      # noqa: E402
    batch_indices_for_run, derive_loader_seed, steps_per_epoch,
)
from experiments.datasets import subsample_indices                     # noqa: E402

__all__ = [
    "MAX_JAC_BYTES",
    "SPECTRA_DIR",
    "RowSpec",
    "Run",
    "batch_of_step",
    "checkpoint_spectra",
    "distance_from_init",
    "find_runs",
    "jacobian_rows",
    "load_record",
    "load_run",
    "load_spectra",
    "param_vector",
    "probe_set",
    "probe_indices",
    "resolve_results_root",
    "resolved_config",
    "row_spec_for_scan",
    "spectra_path",
    "spectrum",
    "verify_checkpoint",
]

#: Refuse an (N x P) float64 Jacobian bigger than this (3 GB): the CPU test
#: partition gives 12 GB and the SVD needs room for U and a copy.
MAX_JAC_BYTES = 3 * 1024 ** 3

#: Where ``tools/compute_ckpt_spectra.py`` caches its npz files -- git-ignored
#: (``analysis/.gitignore``), one subdirectory per diag pass.  The path scheme
#: lives here, next to :func:`load_spectra`, so the writer and the figure
#: notebooks cannot drift apart.
SPECTRA_DIR = _REPO_ROOT / "analysis" / "ckpt_spectra"

#: ``run_id`` prefix -> the ``mode`` its job ran under, for picking the right
#: resolved config out of ``{scan}/configs/`` (the lbfgs and polyak families are
#: enumerated by ``mode: standard`` jobs and write ``std_`` run_ids).
_PREFIX_MODE = {"svd_": "svd", "std_": "standard", "jd_": "jd", "hig_": "hig"}
_MODE_FAMILIES = {
    "svd": {"svd"}, "standard": {"standard"}, "jd": {"jd"}, "hig": {"hig"},
    "both": {"svd", "standard"}, "all": {"svd", "standard", "jd", "hig"},
}


def _cast(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``tensor`` in ``dtype`` if it is a float tensor, unchanged otherwise.

    Neither inputs nor targets are always floats: the LM scans feed Long token
    indices into an embedding and the classification scans carry Long class
    labels, and casting either to float64 is an immediate ``RuntimeError``
    inside the model.
    """
    return tensor.to(dtype) if tensor.is_floating_point() else tensor


# ---------------------------------------------------------------------------
# Results root and record loading
# ---------------------------------------------------------------------------

def resolve_results_root(root: str | os.PathLike | None = None) -> Path:
    """The results root as an ABSOLUTE path.

    ``analysis.style.resolve_results_root`` defaults to ``'../experiment_results'``,
    which is correct for a notebook running in ``analysis/`` and wrong for
    ``tools/compute_ckpt_spectra.py`` running from the repo root.  The precedence
    is the same (explicit argument, then ``$SV3_RESULTS_ROOT``, then the default),
    but a relative value is resolved against the repo root when it does not
    exist relative to the cwd, so every caller lands on the same directory.
    """
    if root is None:
        root = os.environ.get("SV3_RESULTS_ROOT") or "experiment_results"
    path = Path(root)
    if not path.is_absolute() and not path.is_dir():
        candidate = (_REPO_ROOT / path).resolve()
        if candidate.is_dir():
            return candidate
        # `../experiment_results` written for analysis/: strip one level up.
        stripped = Path(*[p for p in path.parts if p != ".."])
        candidate = (_REPO_ROOT / stripped).resolve()
        if candidate.is_dir():
            return candidate
    return path.resolve()


def _scan_dir(scan: str, results_root=None) -> Path:
    return resolve_results_root(results_root) / scan


def find_runs(scan: str, results_root=None, prefix: str | None = None) -> list[str]:
    """Every ``run_id`` with a record in ``{scan}/``, sorted; optionally filtered
    by ``run_id`` prefix (``'svd_'``, ``'std_'``, ...)."""
    directory = _scan_dir(scan, results_root)
    if not directory.is_dir():
        raise FileNotFoundError(f"no scan directory {directory}")
    names = sorted(p.stem for p in directory.glob("*.jsonl"))
    return [n for n in names if prefix is None or n.startswith(prefix)]


def load_record(scan: str, run_id: str, results_root=None) -> dict:
    """One run's record (the first JSONL line; later lines are dedup rewrites)."""
    path = _scan_dir(scan, results_root) / f"{run_id}.jsonl"
    with open(path) as fh:
        return json.loads(fh.readline())


# ---------------------------------------------------------------------------
# The resolved Hydra config of a run's job
# ---------------------------------------------------------------------------

_config_cache: dict[tuple[str, float], Any] = {}
_dataset_cache: dict[str, Any] = {}
_row_spec_cache: dict[str, "RowSpec"] = {}


def _load_config_file(path: Path):
    key = (str(path), path.stat().st_mtime)
    if key not in _config_cache:
        _config_cache[key] = OmegaConf.load(path)
    return _config_cache[key]


def _resolved_subtree(cfg, key: str):
    """``cfg[key]`` as a plain container with every interpolation RESOLVED.

    The campaign parameterises the dataset and the model BY interpolation
    (``seed: ${data_seed}``, ``n_train: ${n_data}``,
    ``hidden_dims: [${mlp_width}, ...]``), so the *unresolved* subtree is
    byte-identical for every data seed, pool size and width of a scan: anything
    that compares or caches on it (``OmegaConf.to_yaml(cfg.dataset)``) is blind
    to exactly the fields the scan varies.  Resolution needs the node's parent,
    which is why the subtree is taken off ``cfg`` rather than loaded on its own.
    """
    node = cfg.get(key)
    if node is None:
        return None
    return OmegaConf.to_container(node, resolve=True)


def resolved_config(scan: str, record: dict, results_root=None):
    """The resolved config of the job that produced ``record`` (C-R3).

    One job writes one ``{scan}/configs/{mode}.{id_string}.{optimizers}.{loss}.
    {mseeds}.yaml``, so a scan directory holds a dozen of them and the one that
    belongs to a run has to be identified rather than guessed.  Candidates are
    filtered by the facts the record carries -- the loss key, the
    ``result_id_fields`` values (``mlp_width``, ``n_data``: a scan can hold
    several models) and the model seed -- and then by the ``mode`` implied by the
    run_id prefix.

    The dataset and model subtrees of every surviving candidate must agree --
    compared RESOLVED (:func:`_resolved_subtree`), because the campaign's
    configs differ only in the interpolated values (``${data_seed}``,
    ``${n_data}``, ``${mlp_width}``) and an unresolved comparison sees none of
    them.  A disagreement RAISES rather than picking one, because that is
    exactly the case where the reconstruction would silently build a different
    model, or a different dataset, from the one the run trained on.
    """
    directory = _scan_dir(scan, results_root) / "configs"
    files = sorted(directory.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(
            f"{directory} holds no resolved config; C-L4 needs the config the job "
            "saved (only schema-2 scans have it)")
    loss = record.get("loss", "ce")
    seed = record.get("model_seed")
    candidates = []
    for path in files:
        cfg = _load_config_file(path)
        if str(cfg.get("loss", "ce")) != str(loss):
            continue
        fields = list(cfg.get("result_id_fields") or [])
        if any(record.get(f) is not None and cfg.get(f) != record.get(f) for f in fields):
            continue
        seeds = cfg.get("model_seeds")
        if seeds is not None and seed is not None and int(seed) not in [int(s) for s in seeds]:
            continue
        candidates.append((path, cfg))
    if not candidates:
        raise LookupError(
            f"no config in {directory} matches run {record.get('run_id')!r} "
            f"(loss={loss}, model_seed={seed})")
    prefix = next((p for p in _PREFIX_MODE if str(record.get("run_id", "")).startswith(p)), None)
    family = _PREFIX_MODE.get(prefix or "", None)
    if family is not None:
        preferred = [(p, c) for p, c in candidates
                     if family in _MODE_FAMILIES.get(str(c.get("mode", "both")), set())]
        if preferred:
            candidates = preferred
    reference = candidates[0]
    for path, cfg in candidates[1:]:
        for key in ("dataset", "model"):
            if _resolved_subtree(cfg, key) != _resolved_subtree(reference[1], key):
                raise ValueError(
                    f"configs {reference[0].name} and {path.name} of scan {scan!r} "
                    f"disagree on `{key}`; cannot decide which model run "
                    f"{record.get('run_id')!r} used")
    return reference[1]


def _dataset_of(cfg):
    """``instantiate(cfg.dataset)``, cached by the RESOLVED dataset subtree.

    MNIST is read from Lustre and every run of a scan shares one dataset object,
    exactly as ``_ScanContext`` does (one ``instantiate`` per scan, not per run).
    The cache key must be the resolved subtree (:func:`_resolved_subtree`): the
    saved configs carry ``seed: ${data_seed}`` / ``n_train: ${n_data}``, so an
    unresolved key is the same string for every data seed and every pool size of
    a scan and the first dataset built in the process would be handed to every
    other run -- silently, with the wrong data.
    """
    key = json.dumps(_resolved_subtree(cfg, "dataset"), sort_keys=True, default=str)
    if key not in _dataset_cache:
        _dataset_cache[key] = instantiate(cfg.dataset)
    return _dataset_cache[key]


# ---------------------------------------------------------------------------
# Jacobian rows -- the same definition Sven differentiates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RowSpec:
    """The rows whose Jacobian Sven inverts (``SvenWrapper._rows``).

    Loss path: ``rows = group(loss) ** (kappa / 2)``.  Residual path (scalar
    regression with ``signed_residual``): ``rows = sign(r) * |r| ** kappa`` with
    ``r = pred - y``, which is the same update (a per-row sign cancels in the
    pseudo-inverse) with a finite gradient at ``r = 0``.

    A baseline's record carries no ``kappa`` / ``signed_residual``, so
    :meth:`from_record` falls back to the scan defaults (``kappa = 2``, signed
    residuals wherever the loss key has a scalar residual) -- which is what the
    Sven run of the same scan used.  Use :func:`row_spec_for_scan` to take the
    definition off the scan's Sven record instead of trusting that fallback.
    """

    loss_key: str
    kappa: float = 2.0
    signed_residual: bool = False
    microbatch_size: int = 1

    @classmethod
    def from_record(cls, record: dict) -> "RowSpec":
        loss_key = str(record.get("loss", "ce"))
        kappa = record.get("kappa")
        signed = record.get("signed_residual")
        microbatch = record.get("microbatch_size")
        return cls(
            loss_key=loss_key,
            kappa=2.0 if kappa is None else float(kappa),
            signed_residual=(loss_key in SIGNED_RESIDUAL_LOSS_KEYS if signed is None
                             else bool(signed)),
            microbatch_size=1 if microbatch is None else int(microbatch),
        )

    @property
    def loss_fn(self) -> Callable:
        """The per-sample loss of this scan (``SVD_LOSS_FNS``, reduction none)."""
        return SVD_LOSS_FNS[self.loss_key]

    def rows(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """The Jacobian rows for ``pred`` -- a copy of ``SvenWrapper._rows``."""
        loss = self.loss_fn(pred, y)
        if not self.signed_residual:
            if self.microbatch_size > 1:
                loss = loss.view(-1, self.microbatch_size).mean(dim=1)
            return loss.pow(self.kappa / 2.0)
        if self.microbatch_size > 1:
            raise ValueError("signed residual rows require microbatch_size == 1 "
                             "(SvenWrapper raises the same way)")
        residual_fn = SVD_RESIDUAL_FNS.get(self.loss_key)
        if residual_fn is None:
            raise KeyError(f"no signed residual for loss {self.loss_key!r}; "
                           f"known: {sorted(SVD_RESIDUAL_FNS)}")
        r = residual_fn(pred, y).reshape(-1)
        if self.kappa == 1.0:
            return r
        return torch.sign(r) * r.abs().pow(self.kappa)

    def describe(self) -> str:
        form = ("sign(r)|r|^kappa" if self.signed_residual else "loss^(kappa/2)")
        return (f"{form}, loss={self.loss_key}, kappa={self.kappa:g}, "
                f"microbatch={self.microbatch_size}")


def row_spec_for_scan(scan: str, results_root=None, run_id: str | None = None) -> RowSpec:
    """The scan's row definition, read off its Sven (``svd_``) records.

    The baselines' spectra have to be taken of the SAME rows or they are not
    comparable, and only the svd records carry ``kappa`` / ``signed_residual``.

    EVERY svd record of the scan is read, not the first one: a scan may sweep
    the row definition itself (``mnist_kappaScan_labelRegression`` holds
    kappa 1, 2 and 3), and picking whichever run_id sorts first would hand the
    baselines a different definition from the Sven run they are compared with.
    A disagreement raises; name the run with ``run_id=`` (or pass an explicit
    :class:`RowSpec`) to choose one deliberately.
    """
    if run_id is not None:
        return RowSpec.from_record(load_record(scan, run_id, results_root))
    key = str(_scan_dir(scan, results_root))
    if key in _row_spec_cache:
        return _row_spec_cache[key]
    svd_runs = find_runs(scan, results_root, prefix="svd_")
    if not svd_runs:
        raise LookupError(f"scan {scan!r} has no svd_ run to read the row "
                          "definition from; pass run_id= or build a RowSpec")
    specs: dict[RowSpec, str] = {}
    for candidate in svd_runs:
        specs.setdefault(RowSpec.from_record(load_record(scan, candidate, results_root)),
                         candidate)
    if len(specs) > 1:
        listing = "; ".join(f"{spec.describe()} (e.g. {rid})" for spec, rid in specs.items())
        raise ValueError(
            f"scan {scan!r} has {len(specs)} different Sven row definitions -- {listing}. "
            "The baselines would silently get one of them: pass run_id= (or an explicit "
            "RowSpec) to say which rows the spectra are of.")
    spec = next(iter(specs))
    _row_spec_cache[key] = spec
    return spec


# ---------------------------------------------------------------------------
# One reloadable run
# ---------------------------------------------------------------------------

class _EvalShim:
    """A callable with a ``.model`` attribute, for the runner's :func:`evaluate`.

    ``experiment_utils._forward_module`` puts the module behind the forward
    callable into eval mode and suppresses norm-stat writes; it finds it either
    on an ``nn.Module`` or as ``.model`` of a bound method's owner (the Sven /
    HIG wrappers).  Passing ``shim.forward`` therefore keeps the runner's
    eval-mode guarantee while letting the inputs be cast to the model's dtype.
    """

    def __init__(self, model: nn.Module, dtype: torch.dtype):
        self.model = model
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(_cast(x, self.dtype))


@dataclass
class Run:
    """One recorded run, reloadable at any checkpointed step.

    Built by :func:`load_run`.  Holds the record, the resolved config of its job
    and (lazily) the dataset, so walking a trajectory re-reads nothing.
    """

    scan: str
    run_id: str
    record: dict
    scan_dir: Path
    cfg: Any
    _ckpt: dict | None = field(default=None, repr=False)
    _dataset: Any = field(default=None, repr=False)

    # -- facts ----------------------------------------------------------

    @property
    def method(self) -> str:
        """The optimizer's analysis name (``'SVD'`` -> ``'Sven'``, as
        ``analysis_helpers.add_derived`` spells it)."""
        optimizer = str(self.record.get("optimizer"))
        return "Sven" if optimizer == "SVD" else optimizer

    @property
    def model_seed(self) -> int:
        return int(self.record["model_seed"])

    @property
    def n_train(self) -> int:
        return int(self.record["n_train"])

    @property
    def batch_size(self) -> int:
        return int(self.record["batch_size"])

    @property
    def steps_per_epoch(self) -> int:
        recorded = self.record.get("steps_per_epoch")
        if recorded is not None:
            return int(recorded)
        return steps_per_epoch(self.n_train, self.batch_size, True)

    @property
    def row_spec(self) -> RowSpec:
        return RowSpec.from_record(self.record)

    @property
    def dataset(self):
        """The run's dataset, rebuilt from its config and CHECKED against the record.

        Both checks matter and neither implies the other: the size catches a
        config from a different arm of an ``n_data`` sweep, the ``split_seed``
        catches a config (or a cache entry) from a different DATA seed, which
        changes every example while leaving the sizes alone.
        """
        if self._dataset is None:
            dataset = _dataset_of(self.cfg)
            n_train = len(dataset.train_dataset)
            if n_train != self.n_train:
                raise ValueError(
                    f"{self.run_id}: rebuilt dataset has {n_train} training examples "
                    f"but the record says {self.n_train}; the config in "
                    f"{self.scan_dir / 'configs'} does not describe this run")
            recorded_split = self.record.get("split_seed")
            built_split = getattr(dataset, "split_seed", None)
            if recorded_split is not None and built_split is not None \
                    and int(built_split) != int(recorded_split):
                raise ValueError(
                    f"{self.run_id}: rebuilt dataset has split_seed={int(built_split)} "
                    f"but the record says split_seed={int(recorded_split)}; the config "
                    f"in {self.scan_dir / 'configs'} describes a different data seed "
                    "(every example would be different while the sizes match)")
            self._dataset = dataset
        return self._dataset

    # -- checkpoints ----------------------------------------------------

    @property
    def checkpoint(self) -> dict:
        """The run's checkpoint file, loaded once (weights AND buffers, fp32)."""
        if self._ckpt is None:
            ckpt_file = self.record.get("ckpt_file")
            if not ckpt_file:
                raise FileNotFoundError(
                    f"{self.run_id}: no ckpt_file on the record "
                    f"(checkpoint_policy={self.record.get('checkpoint_policy')!r}); "
                    "only a scan run with `checkpoints: log|epochs|final` can be reloaded")
            self._ckpt = load_checkpoint(self.scan_dir / ckpt_file)
        return self._ckpt

    @property
    def steps(self) -> list[int]:
        return [int(s) for s in self.checkpoint["step"]]

    @property
    def epochs(self) -> list[int]:
        return [int(e) for e in self.checkpoint["epoch"]]

    def epoch_checkpoints(self) -> list[tuple[int, int]]:
        """``(step, epoch)`` of the END of every epoch, in order.

        Under ``log`` an epoch end shares its step with a power of two and the
        epoch end owns the label (``Checkpointer._record(relabel=True)``), so
        "the last entry with label e" is the end of epoch e.
        """
        out = {}
        for step, epoch in zip(self.steps, self.epochs):
            out[epoch] = step
        return [(out[e], e) for e in sorted(out)]

    def state_at(self, step: int | None = None, epoch: int | None = None) -> dict:
        """One ``state_dict`` from the checkpoint file (see
        :func:`~experiments.experiment_code.checkpointing.load_state_at`)."""
        return load_state_at(self.checkpoint, step=step, epoch=epoch)

    def model_at(self, step: int | None = None, epoch: int | None = None,
                 dtype: torch.dtype = torch.float64) -> nn.Module:
        """The run's model with the state of that step/epoch loaded, in eval mode.

        Built by ``instantiate(cfg.model)`` -- the same call
        ``generic_scan.execute`` makes -- so architecture drift cannot creep in.
        Eval mode matters for the ResNet scans: a probe row must not depend on
        its companions, which is the point of a fixed probe set, so the
        normalisation uses the checkpointed running statistics.
        """
        model = instantiate(self.cfg.model)
        model.load_state_dict(self.state_at(step=step, epoch=epoch))
        return model.to(dtype).eval()

    # -- recorded curves ------------------------------------------------

    def recorded(self, which: str = "val", epoch: int | None = None) -> float:
        """The recorded ``which`` loss at the END of ``epoch``.

        Index 0 of every evaluation curve is the untrained model (C-E1), so the
        end of epoch ``e`` is index ``e + 1``.  ``epoch=None`` gives the last
        point and ``epoch=-1`` the untrained one.
        """
        curve = (self.record.get("losses") or {}).get(which)
        if not curve:
            raise KeyError(f"{self.run_id}: no {which!r} curve on the record")
        index = len(curve) - 1 if epoch is None else int(epoch) + 1
        if not 0 <= index < len(curve):
            # A `diverged` / `oom` / `error` record keeps the PARTIAL curves
            # (C-R1) while its checkpointer keeps every state it collected, so a
            # run that blew up mid-epoch-0 has checkpoints for steps it has no
            # end-of-epoch loss for.  An IndexError here would look like a bug
            # in the tool rather than a property of the run.
            raise IndexError(
                f"{self.run_id}: the {which!r} curve has {len(curve)} point(s) "
                f"(index 0 = untrained), so there is no end-of-epoch-{epoch} value. "
                f"status={self.record.get('status')!r}, diverged_at_step="
                f"{self.record.get('diverged_at_step')!r}: a partial curve (C-R1). "
                "Verify an epoch this run actually finished.")
        return float(curve[index])

    def verifiable_epochs(self, which: str = "val") -> list[int]:
        """Epochs that have BOTH a checkpoint and a recorded ``which`` value.

        The intersection is what :func:`verify_checkpoint` can check: a partial
        curve (C-R1) is shorter than the checkpoint list, and a policy other than
        ``log`` / ``epochs`` saves fewer states than there are epochs.
        """
        curve = (self.record.get("losses") or {}).get(which) or []
        return [epoch for _, epoch in self.epoch_checkpoints() if epoch + 1 < len(curve)]

    # -- data -----------------------------------------------------------

    def loader(self, split: str = "val") -> DataLoader:
        """A sequential evaluation loader at the run's ``eval_batch_size`` (C-E1)."""
        dataset = getattr(self.dataset, f"{split}_dataset", None)
        if dataset is None:
            raise KeyError(f"dataset has no {split}_dataset")
        batch_size = int(self.record.get("eval_batch_size") or 2048)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_run(scan: str, run_id: str, results_root=None, record: dict | None = None) -> Run:
    """A :class:`Run` for ``{scan}/{run_id}``: record + resolved config, nothing built yet."""
    scan_dir = _scan_dir(scan, results_root)
    record = load_record(scan, run_id, results_root) if record is None else record
    cfg = resolved_config(scan, record, results_root)
    return Run(scan=scan, run_id=run_id, record=record, scan_dir=scan_dir, cfg=cfg)


# ---------------------------------------------------------------------------
# Verification: the reloaded state must reproduce the recorded loss
# ---------------------------------------------------------------------------

def verify_checkpoint(run: Run, epoch: int | None = None, step: int | None = None,
                      which: str = "val", dtype: torch.dtype = torch.float32,
                      detail: bool = False) -> float | dict:
    """Relative error between the recorded ``which`` loss and the recomputed one.

    Reloads the state at the end of ``epoch`` -- by default the last epoch that
    has both a checkpoint and a recorded value (:meth:`Run.verifiable_epochs`),
    which is the last epoch for a finished run and the last COMPLETED epoch for a
    diverged one -- and re-evaluates the split with the runner's own
    :func:`~experiments.experiment_code.experiment_utils.evaluate` -- eval mode,
    example-weighted, no buffer written.  float32 by default: that is what the
    runs trained and evaluated in and what the checkpoints store, so the residual
    measures the reconstruction and not a dtype change.

    ``step=`` is accepted only for a step the recorded curve has a value FOR --
    an epoch-end step, or step 0 (the untrained model, curve index 0, reported
    as ``epoch = -1``).  Every other saved step sits inside an epoch and the
    record holds no loss for it; comparing it against the end-of-run value
    (which is what indexing the curve with ``epoch=None`` does) would report a
    relative error of ~1e+6 on a perfectly reconstructed checkpoint.

    ``detail=True`` returns the numbers instead of just the error, including
    ``abs_error`` and ``loss_scale`` (the untrained loss, curve index 0).  Read
    them together: a run whose final loss is 1e-9 sits at the float32 round-off
    floor, so an absolute agreement of 1e-11 shows up as a *relative* error of
    1e-2 and says nothing bad about the reconstruction.
    """
    if step is not None and epoch is not None:
        raise ValueError("pass step or epoch, not both")
    if step is not None:
        ends = {s: e for s, e in run.epoch_checkpoints()}
        if int(step) in ends:
            epoch = ends[int(step)]
        elif int(step) == 0:
            epoch = -1                       # the untrained model: curve index 0 (C-E1)
        else:
            raise ValueError(
                f"{run.run_id}: step {step} is not an epoch end, so the record holds no "
                f"{which!r} value for it (evaluation is per epoch, C-E1). Verifiable "
                f"steps: 0 and {sorted(ends)}; the spectra themselves can be taken at "
                "any checkpointed step with checkpoint_spectra(steps=...).")
    if step is None and epoch is None:
        verifiable = run.verifiable_epochs(which)
        if not verifiable:
            raise ValueError(
                f"{run.run_id}: no epoch has both a checkpoint and a recorded "
                f"{which!r} value (status={run.record.get('status')!r}, "
                f"{len((run.record.get('losses') or {}).get(which) or [])} curve point(s), "
                f"checkpointed epochs {[e for _, e in run.epoch_checkpoints()]}). "
                "A run that blew up inside epoch 0 can only be inspected by step.")
        epoch = verifiable[-1]
    # `epoch` is now always set (it labels the recorded value); the STATE is
    # loaded by whichever selector the caller gave, and `load_state_at` refuses
    # to be given both.
    model = (run.model_at(step=step, dtype=dtype) if step is not None
             else run.model_at(epoch=epoch, dtype=dtype))
    shim = _EvalShim(model, dtype)
    spec = run.row_spec
    # token-weighted for the LM scans, exactly as the runner evaluated them (C-E1)
    out = evaluate(shim.forward, spec.loss_fn, run.loader(which), "cpu",
                   is_lm=(spec.loss_key == "lm_ce"))
    recorded = run.recorded(which, epoch=epoch)
    recomputed = float(out["loss"])
    absolute = abs(recomputed - recorded)
    rel = absolute / (abs(recorded) if recorded else 1.0)
    if not detail:
        return rel
    curve = (run.record.get("losses") or {}).get(which) or [None]
    scale = abs(float(curve[0])) if curve[0] is not None else float("nan")
    return {"run_id": run.run_id, "method": run.method, "model_seed": run.model_seed,
            "which": which, "epoch": epoch, "step": step, "recorded": recorded,
            "recomputed": recomputed, "rel_error": rel, "abs_error": absolute,
            "loss_scale": scale, "rel_error_scale": absolute / scale if scale else float("nan"),
            "n": int(out["n"]), "dtype": str(dtype)}


# ---------------------------------------------------------------------------
# The fixed probe set
# ---------------------------------------------------------------------------

def probe_indices(run: Run, n_probe: int | None = None) -> torch.Tensor:
    """Indices into the training pool of the scan's FIXED probe set.

    ``subsample_indices(n_train, n_probe, split_seed)`` -- the same draw the
    runner uses for its ``train_eval`` subset (C-E3): a prefix of one
    permutation keyed by the dataset's ``split_seed``, so the set depends on
    neither the model seed nor the loader seed nor the optimizer, is identical
    for every run of the scan, and is nested in ``n_probe``.  ``n_probe=None``
    (or >= the pool) is the whole training pool, in its natural order.

    Scans whose ``result_id_fields`` include ``n_data`` hold several ``n_train``
    values (the overparam sweeps); the probe set is then per ``n_train``, and
    nested across them, which is the best available comparison.
    """
    split_seed = run.record.get("split_seed")
    n_train = run.n_train
    if n_probe is not None and int(n_probe) > n_train:
        warnings.warn(f"n_probe={n_probe} exceeds the training pool of {n_train}; "
                      "using the whole pool", RuntimeWarning, stacklevel=2)
        n_probe = None
    return subsample_indices(n_train, n_probe, 0 if split_seed is None else int(split_seed))


def _stack(dataset, indices: torch.Tensor, dtype: torch.dtype
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(x, y)`` of ``indices``, stacked, floats cast to ``dtype`` (:func:`_cast`)."""
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(subset, batch_size=min(512, max(1, len(subset))), shuffle=False)
    xs, ys = [], []
    for batch in loader:
        xs.append(batch[0])
        ys.append(batch[1])
    return _cast(torch.cat(xs), dtype), _cast(torch.cat(ys), dtype)


def probe_set(run: Run, n_probe: int | None = None, dtype: torch.dtype = torch.float64
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(indices, x, y)`` of the scan's fixed probe set (:func:`probe_indices`)."""
    indices = probe_indices(run, n_probe)
    x, y = _stack(run.dataset.train_dataset, indices, dtype)
    return indices, x, y


def batch_of_step(run: Run, step: int, dtype: torch.dtype = torch.float64
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(indices, x, y)`` of the training batch consumed by optimizer ``step``.

    ``sampler.batch_indices_for_run`` from the record's BASE ``loader_seed`` and
    ``model_seed`` (C-S2); the derived effective seed is checked against the
    recorded ``effective_loader_seed``, so the base-vs-effective mistake -- which
    returns a different, perfectly plausible batch -- cannot pass silently.
    """
    base = int(run.record["loader_seed"])
    effective = derive_loader_seed(base, run.model_seed)
    recorded = run.record.get("effective_loader_seed")
    if recorded is not None and int(recorded) != effective:
        raise ValueError(
            f"{run.run_id}: derive_loader_seed({base}, {run.model_seed}) = {effective} "
            f"but the record says effective_loader_seed={recorded}")
    indices = batch_indices_for_run(run.n_train, base, run.model_seed,
                                    run.batch_size, int(step), True)
    x, y = _stack(run.dataset.train_dataset, indices, dtype)
    return indices, x, y


# ---------------------------------------------------------------------------
# Jacobians and spectra
# ---------------------------------------------------------------------------

def _param_names(model: nn.Module) -> list[str]:
    return [name for name, _ in model.named_parameters()]


def param_vector(model_or_state, names: Sequence[str] | None = None,
                 dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """The flat parameter vector of a module or a ``state_dict``, in ``names`` order.

    Buffers are left out (BatchNorm running statistics are not parameters and
    ``num_batches_tracked`` is not even a float), so the norm and the distance
    from initialisation are over the trained weights only.
    """
    if isinstance(model_or_state, nn.Module):
        state = dict(model_or_state.named_parameters())
        names = list(state) if names is None else names
    else:
        state = dict(model_or_state)
        if names is None:
            names = [k for k, v in state.items()
                     if torch.is_tensor(v) and v.is_floating_point()]
    return torch.cat([state[n].detach().reshape(-1).to(dtype) for n in names])


def distance_from_init(run: Run, step: int | None = None, epoch: int | None = None,
                       names: Sequence[str] | None = None) -> float:
    """``||theta(step) - theta(0)||_2`` (Codex Low 4: parameter-space movement).

    Step 0 is the initialisation; it is in the run's own file under the ``log``
    and ``epochs`` policies, and under ``final`` it is the per-seed
    ``ckpt_init_file`` instead, which this reads when the run's file has no
    step 0.
    """
    if names is None:
        names = _param_names(instantiate(run.cfg.model))
    if 0 not in run.steps:
        init_file = run.record.get("ckpt_init_file")
        if not init_file:
            raise FileNotFoundError(
                f"{run.run_id}: no step-0 checkpoint and no ckpt_init_file on the record")
        init_state = load_state_at(run.scan_dir / init_file, step=0)
    else:
        init_state = run.state_at(step=0)
    theta0 = param_vector(init_state, names)
    theta = param_vector(run.state_at(step=step, epoch=epoch), names)
    return float((theta - theta0).norm())


def jacobian_rows(model: nn.Module, x: torch.Tensor, y: torch.Tensor, spec: RowSpec,
                  chunk_size: int = 64, dtype: torch.dtype = torch.float64,
                  max_bytes: int = MAX_JAC_BYTES) -> torch.Tensor:
    """The (N x P) Jacobian ``d rows / d theta`` of :class:`RowSpec`'s rows.

    ``torch.func.jacrev`` over a dict of the model's parameters, row-chunked (one
    ``vmap`` per chunk) and concatenated in ``named_parameters`` order.  The full
    matrix is materialised because ``J J^T`` cannot be accumulated over row
    chunks; ``max_bytes`` refuses a combination that would not fit rather than
    swapping the node.

    ``model`` is used as it is handed over: :meth:`Run.model_at` returns it in
    eval mode (running statistics, no dependence on the probe's companions).
    """
    model = model.to(dtype)
    params = {name: value.detach() for name, value in model.named_parameters()}
    buffers = {name: value.detach() for name, value in model.named_buffers()}
    names = list(params)
    n_params = sum(v.numel() for v in params.values())
    n_rows = int(x.shape[0]) // max(1, spec.microbatch_size)
    need = n_rows * n_params * torch.finfo(dtype).bits // 8
    if need > max_bytes:
        raise MemoryError(
            f"a {n_rows} x {n_params} {dtype} Jacobian needs {need / 1024 ** 3:.1f} GB "
            f"(limit {max_bytes / 1024 ** 3:.1f} GB); lower n_probe or raise max_bytes")

    def rows_of(p, xb, yb):
        pred = torch.func.functional_call(model, ({**p}, buffers), (xb,))
        return spec.rows(pred, yb)

    jac_fn = torch.func.jacrev(rows_of, argnums=0)
    step = max(1, int(chunk_size)) * max(1, spec.microbatch_size)
    blocks = []
    for start in range(0, int(x.shape[0]), step):
        xb = _cast(x[start:start + step], dtype)
        yb = _cast(y[start:start + step], dtype)
        jac = jac_fn(params, xb, yb)
        rows = jac[names[0]].shape[0]
        blocks.append(torch.cat([jac[n].reshape(rows, -1) for n in names], dim=1))
    return torch.cat(blocks, dim=0)


def spectrum(model: nn.Module, x: torch.Tensor, y: torch.Tensor, spec: RowSpec,
             with_utr: bool = True, chunk_size: int = 64,
             dtype: torch.dtype = torch.float64, max_bytes: int = MAX_JAC_BYTES) -> dict:
    """Singular values of the row Jacobian on ``(x, y)``, plus the residual's
    projections onto the left singular vectors.

    Returns ``svals`` (descending), ``utr`` = ``U^T r`` (the quantity the online
    ``utr`` diagnostic logs, but on a FIXED probe set and with the full spectrum
    available), ``rows`` (the residual vector ``r``), ``loss`` (the
    example-weighted mean of the per-sample losses on ``(x, y)``), ``n_rows`` and
    ``n_params``.  ``with_utr=False`` skips the singular vectors, which is the
    cheap path for a long trajectory.
    """
    jacobian = jacobian_rows(model, x, y, spec, chunk_size=chunk_size, dtype=dtype,
                             max_bytes=max_bytes)
    with torch.no_grad():
        pred = model(_cast(x, dtype))
        yy = _cast(y, dtype)
        rows = spec.rows(pred, yy)
        loss = float(spec.loss_fn(pred, yy).mean())
    if with_utr:
        u, svals, _ = torch.linalg.svd(jacobian, full_matrices=False)
        utr = (u.transpose(0, 1) @ rows.reshape(-1, 1)).reshape(-1)
    else:
        svals = torch.linalg.svdvals(jacobian)
        utr = None
    return {
        "svals": svals.cpu().numpy(),
        "utr": None if utr is None else utr.cpu().numpy(),
        "rows": rows.detach().cpu().numpy(),
        "loss": loss,
        "n_rows": int(jacobian.shape[0]),
        "n_params": int(jacobian.shape[1]),
    }


def checkpoint_spectra(run: Run, n_probe: int | None = None, steps: Iterable[int] | None = None,
                       epochs_only: bool = False, with_utr: bool = True,
                       chunk_size: int = 64, dtype: torch.dtype = torch.float64,
                       max_bytes: int = MAX_JAC_BYTES, verbose: bool = False,
                       spec: RowSpec | None = None) -> dict:
    """Probe-set spectra along one run's trajectory, as a dict of numpy arrays.

    Walks the run's checkpoints (all of them, the epoch ends only, or an explicit
    ``steps`` list), and at each one records the singular values of the row
    Jacobian on the scan's fixed probe set, the residual projections, the
    probe-set loss, the parameter norm and the distance from initialisation.

    ``spec`` defaults to the SCAN's row definition (:func:`row_spec_for_scan`,
    read off its Sven records) and not to this run's own
    :attr:`Run.row_spec`: a baseline record carries no ``kappa``, so per-run
    specs would silently compare a kappa-2 baseline curve with a kappa-1 Sven
    one on a scan that sweeps kappa.  A scan with no svd_ run at all falls back
    to the run's own spec; an AMBIGUOUS one raises and has to be given a
    ``spec``.

    Keys: ``step``, ``epoch`` (1-D, one entry per checkpoint), ``svals`` and
    ``utr`` (2-D, checkpoint x singular value), ``probe_loss``, ``rows_norm``,
    ``param_norm``, ``dist_init`` (1-D) and ``probe_indices``; plus the scalars
    ``n_rows``, ``n_params`` and the metadata :func:`checkpoint_spectra` needs to
    be self-describing (``run_id``, ``method``, ``model_seed``, ``row_spec``).
    """
    indices, x, y = probe_set(run, n_probe, dtype=dtype)
    if spec is None:
        try:
            spec = row_spec_for_scan(run.scan, run.scan_dir.parent)
        except LookupError:
            spec = run.row_spec
    template = instantiate(run.cfg.model)
    names = _param_names(template)
    n_params = sum(p.numel() for p in template.parameters())
    del template
    if steps is not None:
        pairs = [(int(s), run.epochs[run.steps.index(int(s))]) for s in steps]
    elif epochs_only:
        pairs = run.epoch_checkpoints()
        if 0 in run.steps and (not pairs or pairs[0][0] != 0):
            # step 0 carries the epoch-0 label too, and `epoch_checkpoints` keeps
            # the END of the epoch, so without this the initialisation -- the
            # reference for `dist_init` and the only pre-training spectrum -- would
            # be dropped from every `epochs_only` trajectory.
            pairs = [(0, 0)] + pairs
    else:
        pairs = list(zip(run.steps, run.epochs))
    theta0 = param_vector(run.state_at(step=0), names) if 0 in run.steps else None

    out: dict[str, list] = {k: [] for k in
                            ("step", "epoch", "svals", "utr", "probe_loss",
                             "rows_norm", "param_norm", "dist_init")}
    for step, epoch in pairs:
        model = run.model_at(step=step, dtype=dtype)
        result = spectrum(model, x, y, spec, with_utr=with_utr, chunk_size=chunk_size,
                          dtype=dtype, max_bytes=max_bytes)
        theta = param_vector(model, names)
        out["step"].append(int(step))
        out["epoch"].append(int(epoch))
        out["svals"].append(result["svals"])
        out["utr"].append(np.zeros(0) if result["utr"] is None else result["utr"])
        out["probe_loss"].append(result["loss"])
        out["rows_norm"].append(float(np.linalg.norm(result["rows"])))
        out["param_norm"].append(float(theta.norm()))
        out["dist_init"].append(float("nan") if theta0 is None
                                else float((theta - theta0).norm()))
        if verbose:
            svals = result["svals"]
            print(f"  step {step:7d} epoch {epoch:3d}  sigma_max={svals[0]:.4e}  "
                  f"sigma_min/sigma_max={svals[-1] / svals[0]:.3e}  "
                  f"probe_loss={result['loss']:.6e}")

    arrays = {
        "step": np.asarray(out["step"], dtype=np.int64),
        "epoch": np.asarray(out["epoch"], dtype=np.int64),
        "svals": np.asarray(out["svals"], dtype=np.float64),
        "probe_loss": np.asarray(out["probe_loss"], dtype=np.float64),
        "rows_norm": np.asarray(out["rows_norm"], dtype=np.float64),
        "param_norm": np.asarray(out["param_norm"], dtype=np.float64),
        "dist_init": np.asarray(out["dist_init"], dtype=np.float64),
        "probe_indices": indices.cpu().numpy().astype(np.int64),
        "n_rows": np.asarray(int(x.shape[0]) // max(1, spec.microbatch_size)),
        "n_params": np.asarray(int(n_params)),
        "val_curve": np.asarray((run.record.get("losses") or {}).get("val") or [],
                                dtype=np.float64),
        "run_id": np.asarray(run.run_id),
        "scan": np.asarray(run.scan),
        "method": np.asarray(run.method),
        "model_seed": np.asarray(run.model_seed),
        "row_spec": np.asarray(json.dumps(spec.__dict__)),
    }
    if with_utr:
        arrays["utr"] = np.asarray(out["utr"], dtype=np.float64)
    return arrays


# ---------------------------------------------------------------------------
# The npz cache: one path scheme, one reader
# ---------------------------------------------------------------------------

def spectra_path(scan: str, method: str, model_seed, n_probe: int | None,
                 epochs_only: bool = False, out_dir=None) -> Path:
    """Where :func:`checkpoint_spectra`'s output for one run is cached.

    ``<out_dir>/<scan>/<method>_mseed<seed>_probe<N|full>[_epochs].npz``, with
    ``out_dir`` defaulting to :data:`SPECTRA_DIR`.  ``tools/compute_ckpt_spectra.py``
    writes through this and the figure notebooks read through
    :func:`load_spectra`, so the two cannot drift apart.
    """
    root = SPECTRA_DIR if out_dir is None else Path(out_dir)
    probe = "full" if n_probe is None else str(int(n_probe))
    suffix = "_epochs" if epochs_only else ""
    return root / scan / f"{method}_mseed{model_seed}_probe{probe}{suffix}.npz"


def load_spectra(scan: str, method: str | None = None, model_seed=None,
                 n_probe: int | None = -1, epochs_only: bool | None = None,
                 out_dir=None) -> list[dict]:
    """Every cached trajectory of ``scan`` matching the filters, newest name first.

    Each element is a plain dict of the npz's arrays and scalars (``svals``,
    ``utr``, ``step``, ``epoch``, ``probe_loss``, ``dist_init``, ``param_norm``,
    ``probe_indices``, ``val_curve``, ``verify_rel_error``, ``method``,
    ``model_seed``, ...), so a notebook never touches the file layout.  A
    trajectory whose recorded loss could not be checked at all (a run that
    diverged inside epoch 0 keeps its checkpoints but no end-of-epoch value,
    C-R1) carries ``verify_rel_error = nan`` and a ``verify_error`` string
    saying why, rather than being missing from the cache.
    ``n_probe=-1`` (the default) means "any size"; pass ``None`` for the full
    pool or an integer for that size.

    The run's hyperparameters (``k``, ``rtol``, ``lr``, ``batch_size``, ...) are
    deliberately NOT duplicated in the cache -- a figure that needs the k cut or
    the rtol line reads them from the record with
    ``load_run(scan, str(entry['run_id'])).record``, which cannot go stale.

    Trajectories computed with different ``--probe`` / ``--epochs-only`` settings
    have different lengths, so filter to one setting before stacking them.
    """
    root = (SPECTRA_DIR if out_dir is None else Path(out_dir)) / scan
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} does not exist; run tools/compute_ckpt_spectra.py {scan} first")
    probe = None if n_probe == -1 else ("full" if n_probe is None else str(int(n_probe)))
    out = []
    for path in sorted(root.glob("*.npz")):
        stem = path.stem
        name, _, tail = stem.partition("_mseed")
        seed, _, probe_tag = tail.partition("_probe")
        if method is not None and name != method:
            continue
        if model_seed is not None and str(model_seed) != seed:
            continue
        per_epoch = probe_tag.endswith("_epochs")
        if epochs_only is not None and per_epoch != bool(epochs_only):
            continue
        if probe is not None and probe_tag.removesuffix("_epochs") != probe:
            continue
        with np.load(path, allow_pickle=False) as handle:
            record = {key: handle[key] for key in handle.files}
        record["path"] = str(path)
        record["epochs_only"] = per_epoch
        out.append(record)
    return out
