"""Pure grid enumeration for :func:`generic_scan.scan`.

This module is the single place that answers "which runs does this config
describe, in which order, under which names". It is deliberately **torch-free,
hydra-free and sven-free** (stdlib only) so the grid of a scan can be
enumerated, counted, sharded, hashed and diffed in milliseconds -- by tests, by
``tools/reconcile.py`` and by a launcher -- without paying for a CUDA import.

It replaces the six copy-pasted ``itertools.product`` blocks that used to live
inline in ``generic_scan.scan``: every family (``svd``, ``standard``, ``lbfgs``,
``polyak``, ``jd``, ``hig``) yields :class:`RunSpec` objects whose ``run_id`` is
byte-identical to the legacy one and whose order is identical to the legacy
enumeration (seed-major, then family in the fixed order above, then the
family's product order). The legacy modulo-counter sharding is therefore
exactly ``specs[shard_id::n_shards]`` (see :func:`shard`).

Helpers copied from ``experiment_utils.py`` (which another track owns) are
marked as copies; the originals stay in place.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from dataclasses import dataclass, field
from itertools import product
from typing import Any

# ---------------------------------------------------------------------------
# Constants shared with generic_scan.py
# ---------------------------------------------------------------------------

#: bumped whenever a record's meaning changes (C-R3); part of every run_hash.
SCHEMA_VERSION = 2

#: keys of generic_scan.SVD_LOSS_FNS / STANDARD_LOSS_FNS (asserted there).
LOSS_KEYS = ("ce", "mse", "label_regression", "brier", "lm_ce")
#: losses whose run_ids predate the `_loss{key}` suffix and keep their old names.
LEGACY_LOSS_KEYS = ("ce", "mse", "label_regression", "lm_ce")
#: keys of generic_scan.SVD_RESIDUAL_FNS (asserted there).
SIGNED_RESIDUAL_LOSS_KEYS = ("mse",)
SVD_INFO_MODES = ("none", "summary", "full")
VALID_MODES = ("svd", "standard", "both", "jd", "hig", "all")
#: keys of generic_scan._JD_AGGREGATORS; unknown names are dropped from the grid.
JD_AGGREGATOR_NAMES = ("UPGrad", "Mean", "Sum")
#: the order families are enumerated in inside one model seed.
FAMILIES = ("svd", "standard", "lbfgs", "polyak", "jd", "hig")
#: optimizers whose weight decay is swept (everything else is forced to 0).
_WD_OPTIMIZERS = ("AdamW", "Muon", "MuonW")
#: values of the `bn_mode` config key (C-E2).
BN_MODES = ("batch", "frozen")
#: copy of ``checkpointing.POLICIES`` (that module imports torch); values of the
#: `checkpoints` / `checkpoints_svd` config keys (C-L3).
CHECKPOINT_POLICIES = ("none", "final", "epochs", "log")
#: values of the `scheduler` config key (CONTRACTS "Scheduling"): `claims` = every
#: worker walks the whole grid and runs what it can claim (the default, so a
#: resubmitted job mops up whatever is left); `static` = the legacy
#: `specs[shard_id::n_shards]` slicing, kept as a fallback.
SCHEDULERS = ("claims", "static")
#: default `svd_spectra_schedule` (C-L2): log every step below `dense_first`, then
#: every `every`-th step.
DEFAULT_SPECTRA_SCHEDULE = {"dense_first": 200, "every": 20}
#: copy of ``optim_factory.MUON_RULE_TOKEN`` (importing it would pull torch in):
#: the model-independent Muon construction rule, recorded and hashed so a run made
#: under the old rule does not dedup against one made under the new one (C-B5).
MUON_RULE_TOKEN = "hidden2d+convflat:match_rms_adamw:v1"
#: optimizer names built by the Muon branch of the factory.
MUON_OPTIMIZERS = ("Muon", "MuonW")


# ---------------------------------------------------------------------------
# Pure helpers COPIED from experiment_utils.py (owned by another track: the
# originals are left in place and still used by every other call site).
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHT_DECAY = {"AdamW": 0.01, "MuonW": 0.1}


def listify(settings):
    """Copy of ``experiment_utils.listify``."""
    if type(settings) is list or type(settings) is tuple:
        return settings
    else:
        return [settings]


def resolve_weight_decay(optim_name, weight_decay):
    """Copy of ``experiment_utils.resolve_weight_decay``.

    The weight decay a run actually uses: ``None`` means the optimizer's own
    default (AdamW: 0.01, MuonW: 0.1; everything else: 0.0)."""
    if weight_decay is None:
        return _DEFAULT_WEIGHT_DECAY.get(optim_name, 0.0)
    return float(weight_decay)


def process_hparam_config(cfg) -> dict[str, Any]:
    """Copy of ``experiment_utils.process_hparam_config`` (pure; prints the same
    "no X specified; defaulting to" notes)."""
    output = {}
    if "batch_size" not in cfg:
        output['batch_size'] = listify(32)
        print("No batch size specified; defaulting to", output['batch_size'])
    else:
        output['batch_size'] = listify(cfg["batch_size"])

    if "k_fractions" not in cfg and "k_values" not in cfg:
        output['k_fractions'] = [0.1, 0.25, 0.5, 0.75, 1.0]
        print("No k_values or k_fractions specified; defaulting to fractions = ", output['k_fractions'])
    else:
        assert ("k_values" in cfg) ^ ("k_fractions" in cfg), "Specify either k_values or k_fractions, not both."
        if "k_fractions" in cfg:
            output['k_fractions'] = listify(cfg["k_fractions"])
        else:
            output['k_values'] = listify(cfg["k_values"])

    if "lrs" not in cfg:
        output['lrs'] = [0.01, 0.1, 0.5, 1.0]
        print("No learning rates specified; defaulting to", output['lrs'])
    else:
        output['lrs'] = listify(cfg["lrs"])

    if "rtol" not in cfg:
        output['rtol'] = listify(1e-3)
        print("No rtol specified; defaulting to", output['rtol'])
    else:
        output['rtol'] = listify(cfg["rtol"])

    if "svd_mode" not in cfg:
        output['svd_mode'] = listify('randomized')
        print("No SVD mode specified; defaulting to 'randomized'")
    else:
        output['svd_mode'] = listify(cfg["svd_mode"])

    if "lrs_standard" not in cfg:
        output['lrs_standard'] = [1e-4, 1e-3, 1e-2, 1e-1]
        print("No learning rates for standard optimizers specified; defaulting to", output['lrs_standard'])
    else:
        output['lrs_standard'] = listify(cfg["lrs_standard"])

    if "optimizers_standard" not in cfg:
        output['optimizers_standard'] = ['Adam', 'AdamW', 'SGD', 'RMSprop', 'Muon']
        print("No standard optimizers specified; defaulting to", output['optimizers_standard'])
    else:
        output['optimizers_standard'] = listify(cfg["optimizers_standard"])

    if "microbatch_sizes" in cfg:
        output['microbatch_sizes'] = listify(cfg["microbatch_sizes"])
    else:
        output['microbatch_sizes'] = [None]

    if "param_fractions" in cfg:
        output['param_fractions'] = listify(cfg["param_fractions"])
    else:
        output['param_fractions'] = [None]

    output['kappas'] = listify(cfg.get("kappa", 2.0))

    output['lrs_lbfgs'] = listify(cfg.get("lrs_lbfgs", output['lrs_standard']))
    output['lbfgs_max_iter'] = listify(cfg.get("lbfgs_max_iter", 20))
    output['lbfgs_history_size'] = listify(cfg.get("lbfgs_history_size", 100))
    output['lbfgs_line_search_fn'] = listify(cfg.get("lbfgs_line_search_fn", "strong_wolfe"))

    output['weight_decays'] = listify(cfg.get("weight_decays", [None]))

    output['polyak_f_star'] = listify(cfg.get("polyak_f_star", 0.0))
    output['polyak_max_lr'] = listify(cfg.get("polyak_max_lr", 1.0))
    output['polyak_eps'] = listify(cfg.get("polyak_eps", 1e-8))

    output['lrs_jd'] = listify(cfg.get("lrs_jd", output['lrs_standard']))
    output['aggregators_jd'] = listify(cfg.get("aggregators_jd", ["UPGrad"]))
    output['inner_optimizers_jd'] = listify(cfg.get("inner_optimizers_jd", ["Adam"]))

    output['lrs_hig'] = listify(cfg.get("lrs_hig", output['lrs']))
    output['tau_hig'] = listify(cfg.get("tau_hig", [1e-4]))

    return output


# ---------------------------------------------------------------------------
# RunSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    """One grid point: everything needed to name, hash, dedup and execute a run.

    ``hparams`` is the family's product tuple, named (what ``execute`` needs to
    build the optimizer). ``record_extra`` is the run's record scaffold: every
    key the legacy ``result`` dict carried between ``run_id`` and ``losses``, in
    the legacy order, so ``{"run_id": ..., **record_extra, "losses": ...}``
    reproduces it exactly.

    Frozen but **not hashable**: two of the fields are dicts. Key on ``run_id``
    (unique within a grid, and the dedup key on disk) or on
    :func:`run_hash` -- ``__hash__`` is disabled so ``set(specs)`` fails with
    ``unhashable type: 'RunSpec'`` instead of a confusing ``'dict'``. ``==``
    works as usual (all fields compared).
    """
    family: str
    run_id: str
    model_seed: int
    loader_seed: int
    batch_size: int
    hparams: dict = field(default_factory=dict)
    record_extra: dict = field(default_factory=dict)

    __hash__ = None  # see the docstring: key on run_id, not on the spec

    def __post_init__(self):
        assert self.family in FAMILIES, self.family


# ---------------------------------------------------------------------------
# Scan-level settings (pure resolution of the config; mirrors scan()'s preamble)
# ---------------------------------------------------------------------------

def build_id_string(cfg):
    """Build a model-identifier string from config-specified fields.

    Moved verbatim from ``generic_scan._build_id_string``."""
    fields = cfg.get("result_id_fields", [])
    if not fields:
        return ""
    return "_" + "_".join(f"{f}{cfg[f]}" for f in fields)


def mode_flags(rcfg, mode=None):
    """Which families a ``mode`` enumerates (``generic_scan.py:303-309``)."""
    mode = rcfg.get("mode", "both") if mode is None else mode
    assert mode in VALID_MODES, f"Unknown mode: {mode}. Choose from {VALID_MODES}"
    return {
        "svd": mode in ("svd", "both", "all"),
        "standard": mode in ("standard", "both", "all"),
        "jd": mode in ("jd", "all") and ("lrs_jd" in rcfg or "aggregators_jd" in rcfg),
        "hig": mode in ("hig", "all") and ("lrs_hig" in rcfg or "tau_hig" in rcfg),
    }


def default_bn_mode(family, use_gram=False):
    """The norm-statistics policy a config that names none has always had (C-E2).

    Only the Gram backend ever suppressed the running statistics: its
    ``gram_freeze_norm_stats`` defaulted to ``True``, i.e. frozen (and its
    ``hooks`` capture *requires* frozen).  Every other family trained in
    ``.train()`` mode with batch statistics.  Keeping that as the default means
    ``bn_mode`` changes no run_id and no behaviour until a config asks for it.
    """
    return "frozen" if (family == "svd" and use_gram) else "batch"


def resolve_bn_mode(rcfg):
    """The scan-level ``bn_mode``, or ``None`` when the config names none.

    ``gram_freeze_norm_stats`` is the deprecated alias (``true`` = ``frozen``);
    ``bn_mode`` wins if both are given.  ``None`` means "use
    :func:`default_bn_mode` per family", which is what keeps every existing
    config byte-identical.
    """
    mode = rcfg.get("bn_mode")
    if mode is not None:
        mode = str(mode)
        if mode not in BN_MODES:
            raise ValueError(f"bn_mode must be one of {BN_MODES}, got {mode!r}")
        return mode
    if rcfg.get("gram_freeze_norm_stats") is not None:
        return "frozen" if bool(rcfg["gram_freeze_norm_stats"]) else "batch"
    return None


def bn_mode_suffix(bn_mode, family, use_gram=False):
    """The run_id token for a norm policy that differs from the legacy default.

    ``_bnbatch`` (batch statistics under the Gram backend) is unchanged from
    before the campaign; ``_bnfrozen`` is new and marks a run that freezes the
    norm layers where the family would normally train with batch statistics
    (the fine-tune study, O2).  A config that only restates the default gets no
    token, so run_ids move exactly when the computation does.
    """
    if bn_mode == default_bn_mode(family, use_gram):
        return ""
    return "_bnbatch" if bn_mode == "batch" else "_bnfrozen"


def resolve_spectra_schedule(rcfg):
    """``{"dense_first": int, "every": int}`` from the config (C-L2).

    ``svd_spectra_schedule: {dense_first: D, every: E}`` logs every step below
    ``D`` and then every ``E``-th step.  The superseded ``svd_spectra_every: E``
    is still honoured (as ``{dense_first: 0, every: E}``) so no existing config
    or override breaks.
    """
    schedule = rcfg.get("svd_spectra_schedule")
    if schedule is None:
        if rcfg.get("svd_spectra_every") is not None:
            return {"dense_first": 0, "every": max(1, int(rcfg["svd_spectra_every"]))}
        return dict(DEFAULT_SPECTRA_SCHEDULE)
    unknown = set(schedule) - set(DEFAULT_SPECTRA_SCHEDULE)
    if unknown:
        raise ValueError(f"svd_spectra_schedule: unknown key(s) {sorted(unknown)}; "
                         f"expected {sorted(DEFAULT_SPECTRA_SCHEDULE)}")
    return {
        "dense_first": max(0, int(schedule.get("dense_first",
                                               DEFAULT_SPECTRA_SCHEDULE["dense_first"]))),
        "every": max(1, int(schedule.get("every", DEFAULT_SPECTRA_SCHEDULE["every"]))),
    }


def _checkpoint_policy(rcfg, key, default):
    policy = rcfg.get(key, default)
    policy = "none" if policy is None else str(policy)
    if policy not in CHECKPOINT_POLICIES:
        raise ValueError(f"{key} must be one of {CHECKPOINT_POLICIES}, got {policy!r}")
    return policy


def resolve_scan_settings(rcfg):
    """Scan-level values derived from the config (``generic_scan.py:311-341``)."""
    loss_key = rcfg.get("loss", "ce")
    if loss_key not in LOSS_KEYS:
        raise KeyError(f"Unknown loss key {loss_key!r}; known: {sorted(LOSS_KEYS)}")
    svd_info_mode = rcfg.get("svd_info", "full")
    if svd_info_mode not in SVD_INFO_MODES:
        raise ValueError(f"svd_info must be one of {SVD_INFO_MODES}, got {svd_info_mode!r}")
    eval_batch_size = int(rcfg.get("eval_batch_size", 2048))
    if eval_batch_size < 1:
        raise ValueError(f"eval_batch_size must be positive, got {eval_batch_size}")
    train_eval_size = int(rcfg.get("train_eval_size", 10_000))
    if train_eval_size < 1:
        raise ValueError(f"train_eval_size must be positive, got {train_eval_size}")
    # `checkpoints_svd` overrides `checkpoints` for the svd family only (a Sven run
    # is worth `log` where its baselines are worth `final`); null = no override.
    checkpoints = _checkpoint_policy(rcfg, "checkpoints", "final")
    checkpoints_svd = (None if rcfg.get("checkpoints_svd") is None
                       else _checkpoint_policy(rcfg, "checkpoints_svd", None))
    scheduler = str(rcfg.get("scheduler", "claims"))
    if scheduler not in SCHEDULERS:
        raise ValueError(f"scheduler must be one of {SCHEDULERS}, got {scheduler!r}")
    return {
        "loss_key": loss_key,
        # Non-legacy loss keys are encoded in the run_id so a `loss=` override can
        # never dedup against that config's original-loss results.
        "loss_suffix": "" if loss_key in LEGACY_LOSS_KEYS else f"_loss{loss_key}",
        "track_acc": loss_key in ("ce", "brier") or ("label_regression" in loss_key),
        "is_lm": loss_key == "lm_ce",
        "track_param_norm": rcfg.get("track_param_norm", False),
        "svd_info_mode": svd_info_mode,
        "svd_spectra_schedule": resolve_spectra_schedule(rcfg),
        # Signed-residual Sven rows; scalar-output regression only. Not in the run_id
        # (the update is identical wherever the loss path is finite).
        "signed_residual": (bool(rcfg.get("signed_residual", True))
                            and loss_key in SIGNED_RESIDUAL_LOSS_KEYS),
        # C-E1: validation / test / train_eval never use the training batch size.
        "eval_batch_size": eval_batch_size,
        "train_eval_size": train_eval_size,
        "eval_every_steps": (None if rcfg.get("eval_every_steps") is None
                             else int(rcfg["eval_every_steps"])),
        # C-L3 checkpoint policies; C-T3 allocator; C-R1 early stop; C-E2 norm policy
        # (None = per-family default, see resolve_bn_mode / default_bn_mode).
        "checkpoints": checkpoints,
        "checkpoints_svd": checkpoints_svd,
        "empty_cache": bool(rcfg.get("empty_cache", False)),
        # C-R1: a non-finite training batch loss ends the run with `diverged`
        # instead of burning the remaining epochs on NaN. Default true per
        # EXPERIMENTS.md section 12 "Config keys"; `false` reproduces the legacy behaviour.
        "stop_on_nonfinite": bool(rcfg.get("stop_on_nonfinite", True)),
        "bn_mode": resolve_bn_mode(rcfg),
        "scheduler": scheduler,
    }


def resolve_svd_settings(rcfg):
    """Scan-level Sven/Gram settings, validated (``generic_scan.py:429-460``)."""
    variable_k = rcfg.get("variable_k", False)
    use_gram = rcfg.get("use_gram", False)
    if use_gram and variable_k:
        raise ValueError("use_gram is incompatible with variable_k")
    gram_capture = rcfg.get("gram_capture", "hooks")
    gram_chunk_numel = int(rcfg.get("gram_chunk_numel", 2 ** 22))
    # C-E2: `bn_mode` supersedes `gram_freeze_norm_stats`, which stays as the
    # deprecated alias; the record keeps both (the alias as the boolean it was).
    bn_mode = resolve_bn_mode(rcfg) or default_bn_mode("svd", use_gram)
    gram_freeze_norm_stats = bn_mode == "frozen"
    if use_gram and bn_mode == "batch" and gram_capture not in ("chunked", "full"):
        raise ValueError("bn_mode: batch (= gram_freeze_norm_stats: false) requires "
                         "gram_capture: chunked or full")
    return {
        "variable_k": variable_k,
        "use_gram": use_gram,
        "gram_capture": gram_capture,
        "bn_mode": bn_mode,
        "gram_freeze_norm_stats": gram_freeze_norm_stats,
        "gram_chunk_numel": gram_chunk_numel,
        "mask_mode": rcfg.get("mask_mode", "elementwise"),
    }


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

def _quiet(verbose):
    """Swallow stdout unless ``verbose``.

    The copied ``process_hparam_config`` prints its "no X specified; defaulting
    to" notes unconditionally (it is a byte-identical copy, so it cannot grow a
    flag); a torch-free consumer enumerating dozens of scans must be able to
    silence them.
    """
    return contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())


def expand_grid(rcfg, *, mode=None, has_torchjd=True,
                jd_aggregators=JD_AGGREGATOR_NAMES, verbose=True) -> list[RunSpec]:
    """Every run a config describes, in the legacy enumeration order.

    ``rcfg`` is a plain container (``OmegaConf.to_container(cfg, resolve=True)``).
    ``mode`` defaults to ``rcfg["mode"]``. ``has_torchjd`` / ``jd_aggregators``
    mirror the optional torchjd dependency: a missing torchjd drops the whole JD
    family from the enumeration, exactly as the legacy loop did.
    ``verbose=False`` makes the call print nothing at all.
    """
    flags = mode_flags(rcfg, mode)
    st = resolve_scan_settings(rcfg)
    loss_key, loss_suffix = st["loss_key"], st["loss_suffix"]
    signed_residual = st["signed_residual"]
    # C-E2: one norm policy for every optimizer of the scan; None = each family's
    # own legacy default (so a config that names neither key is unchanged).
    bn_cfg = st["bn_mode"]

    def bn_of(family, use_gram=False):
        return bn_cfg if bn_cfg is not None else default_bn_mode(family, use_gram)

    with _quiet(verbose):
        hparams = process_hparam_config(rcfg)
    id_str = build_id_string(rcfg)
    seeds = rcfg.get("model_seeds")
    loader_seed = rcfg["loader_seed"]

    if flags["svd"]:
        sv = resolve_svd_settings(rcfg)
        k_scan_values = hparams.get('k_fractions', hparams.get('k_values'))
        use_k_values = 'k_values' in hparams
        # Under the Gram backend the SVD of J is replaced by an exact eigh of the
        # B x B Gram matrix, so `svd_mode` selects nothing. Collapse it to "torch"
        # so a list of modes cannot multiply the grid into duplicate runs and the
        # run_id never advertises a randomized SVD that was not used.
        if sv["use_gram"] and list(hparams['svd_mode']) != ['torch']:
            if verbose:
                print(f"  [note] use_gram: ignoring svd_mode={hparams['svd_mode']} "
                      "(exact Gram eigendecomposition); run_ids use 'torch'")
            hparams['svd_mode'] = ['torch']

    if flags["standard"]:
        # LBFGS and PolyakSGD have their own grids (different hyperparameters).
        has_lbfgs = "LBFGS" in hparams['optimizers_standard']
        has_polyak = "PolyakSGD" in hparams['optimizers_standard']
        non_lbfgs_optimizers = [o for o in hparams['optimizers_standard']
                                if o not in ("LBFGS", "PolyakSGD")]

    if flags["jd"] and not has_torchjd and verbose:
        print("  [skip] torchjd not installed -- skipping JD scan")

    specs: list[RunSpec] = []
    for model_seed in seeds:
        seed_str = f"_mseed{model_seed}_lseed{loader_seed}"

        # ---------------- Sven / SVD ----------------
        if flags["svd"]:
            svd_grid = product(
                hparams['batch_size'],
                k_scan_values,
                hparams['lrs'],
                hparams['rtol'],
                hparams['svd_mode'],
                hparams['microbatch_sizes'],
                hparams['param_fractions'],
                hparams['kappas'],
            )
            for (batch_size, k_item, lr, rtol, svd_mode, microbatch_size,
                 param_fraction, kappa) in svd_grid:
                k = max(1, int(k_item * batch_size)) if not use_k_values else k_item
                run_id = (
                    f"svd_bs{batch_size}{id_str}"
                    f"_k{k}_lr{lr}_rtol{rtol}_svd{svd_mode}{seed_str}"
                )
                if microbatch_size is not None:
                    run_id += f"_mb{microbatch_size}"
                if param_fraction is not None:
                    run_id += f"_pf{param_fraction}"
                    if param_fraction < 1.0:
                        run_id += f"_{sv['mask_mode']}"  # elementwise vs rows -> distinct runs
                if sv["variable_k"]:
                    run_id += "_variablek"
                if sv["use_gram"]:
                    run_id += "_gram"
                # "_bnbatch" exactly where it always was; "_bnfrozen" only when a
                # config freezes a family that trains with batch statistics (C-E2).
                run_id += bn_mode_suffix(sv["bn_mode"], "svd", sv["use_gram"])
                if kappa != 2.0:
                    run_id += f"_kappa{kappa}"
                run_id += loss_suffix

                mb = microbatch_size if microbatch_size is not None else 1
                use_residual = signed_residual and mb == 1  # any kappa
                masked = param_fraction is not None and param_fraction < 1.0
                specs.append(RunSpec(
                    family="svd", run_id=run_id, model_seed=model_seed,
                    loader_seed=loader_seed, batch_size=batch_size,
                    hparams={
                        "k": k, "lr": lr, "rtol": rtol, "svd_mode": svd_mode,
                        "microbatch_size": microbatch_size,
                        "param_fraction": param_fraction, "kappa": kappa,
                    },
                    record_extra={
                        "optimizer": "SVD",
                        "loss": loss_key,
                        "batch_size": batch_size,
                        "k_fraction": k / batch_size,
                        "k": k,
                        "lr": lr,
                        "rtol": rtol,
                        "model_seed": model_seed,
                        "loader_seed": loader_seed,
                        "svd_mode": svd_mode,
                        # exact eigh of G (Gram) vs the SVD algorithm named by svd_mode
                        "decomposition": ("gram_eigh" if sv["use_gram"]
                                          else f"svd_{svd_mode}"),
                        "microbatch_size": microbatch_size,
                        "param_fraction": param_fraction,
                        "variable_k": sv["variable_k"],
                        "use_gram": sv["use_gram"],
                        "gram_capture": sv["gram_capture"] if sv["use_gram"] else None,
                        "gram_freeze_norm_stats": (sv["gram_freeze_norm_stats"]
                                                   if sv["use_gram"] else None),
                        "gram_chunk_numel": (sv["gram_chunk_numel"] if (
                            sv["use_gram"] and sv["gram_capture"] == "chunked") else None),
                        "mask_mode": sv["mask_mode"] if masked else None,
                        "kappa": kappa,
                        "signed_residual": bool(use_residual),
                        "bn_mode": sv["bn_mode"],
                    },
                ))

        # ---------------- standard optimizers ----------------
        if flags["standard"]:
            if non_lbfgs_optimizers:
                standard_grid = product(
                    hparams['batch_size'],
                    hparams['lrs_standard'],
                    non_lbfgs_optimizers,
                    hparams['weight_decays'],
                )
                for batch_size, lr, optim_name, weight_decay in standard_grid:
                    # None -> the optimizer's own default (AdamW 0.01, MuonW 0.1)
                    weight_decay = resolve_weight_decay(optim_name, weight_decay)
                    # Non-zero weight decay is only meaningful for AdamW and Muon / MuonW
                    if optim_name not in _WD_OPTIMIZERS and weight_decay != 0.0:
                        continue

                    run_id = f"std_bs{batch_size}{id_str}_lr{lr}_optim{optim_name}"
                    # AdamW / MuonW always carry their wd in the run_id, so the new
                    # default-wd runs do not collide with the old wd=0 ones
                    if weight_decay != 0.0 or optim_name in ("AdamW", "MuonW"):
                        run_id += f"_wd{weight_decay}"
                    run_id += seed_str + bn_mode_suffix(bn_of("standard"), "standard")
                    run_id += loss_suffix

                    record_extra = {
                        "optimizer": optim_name,
                        "loss": loss_key,
                        "batch_size": batch_size,
                        "k_fraction": None,
                        "k": None,
                        "lr": lr,
                        "rtol": None,
                        "weight_decay": weight_decay,
                        "model_seed": model_seed,
                        "loader_seed": loader_seed,
                        "svd_mode": None,
                        "svd_info": None,
                        "bn_mode": bn_of("standard"),
                    }
                    if optim_name in MUON_OPTIMIZERS:
                        # C-B5: which weights go to Muon and how its lr is adjusted
                        # is a code-level rule, so it must enter the run hash --
                        # otherwise a new-rule run dedups against an old-rule one.
                        record_extra["muon_rule"] = MUON_RULE_TOKEN
                    specs.append(RunSpec(
                        family="standard", run_id=run_id, model_seed=model_seed,
                        loader_seed=loader_seed, batch_size=batch_size,
                        hparams={"optim_name": optim_name, "lr": lr,
                                 "weight_decay": weight_decay},
                        record_extra=record_extra,
                    ))

            if has_lbfgs:
                lbfgs_grid = product(
                    hparams['batch_size'],
                    hparams['lrs_lbfgs'],
                    hparams['lbfgs_max_iter'],
                    hparams['lbfgs_history_size'],
                    hparams['lbfgs_line_search_fn'],
                )
                for batch_size, lr, max_iter, history_size, line_search_fn in lbfgs_grid:
                    run_id = (
                        f"std_bs{batch_size}{id_str}_lr{lr}_optimLBFGS"
                        f"_mi{max_iter}_hs{history_size}_ls{line_search_fn}{seed_str}"
                        f"{bn_mode_suffix(bn_of('lbfgs'), 'lbfgs')}{loss_suffix}"
                    )
                    specs.append(RunSpec(
                        family="lbfgs", run_id=run_id, model_seed=model_seed,
                        loader_seed=loader_seed, batch_size=batch_size,
                        hparams={"lr": lr, "max_iter": max_iter,
                                 "history_size": history_size,
                                 "line_search_fn": line_search_fn},
                        record_extra={
                            "optimizer": "LBFGS",
                            "loss": loss_key,
                            "batch_size": batch_size,
                            "k_fraction": None,
                            "k": None,
                            "lr": lr,
                            "rtol": None,
                            "model_seed": model_seed,
                            "loader_seed": loader_seed,
                            "svd_mode": None,
                            "svd_info": None,
                            "lbfgs_max_iter": max_iter,
                            "lbfgs_history_size": history_size,
                            "lbfgs_line_search_fn": line_search_fn,
                            "bn_mode": bn_of("lbfgs"),
                        },
                    ))

            if has_polyak:
                polyak_grid = product(
                    hparams['batch_size'],
                    hparams['polyak_f_star'],
                    hparams['polyak_max_lr'],
                    hparams['polyak_eps'],
                )
                for batch_size, f_star, max_lr, eps in polyak_grid:
                    run_id = (
                        f"std_bs{batch_size}{id_str}_optimPolyakSGD"
                        f"_fstar{f_star}_maxlr{max_lr}_eps{eps}{seed_str}"
                        f"{bn_mode_suffix(bn_of('polyak'), 'polyak')}{loss_suffix}"
                    )
                    specs.append(RunSpec(
                        family="polyak", run_id=run_id, model_seed=model_seed,
                        loader_seed=loader_seed, batch_size=batch_size,
                        hparams={"f_star": f_star, "max_lr": max_lr, "eps": eps},
                        record_extra={
                            "optimizer": "PolyakSGD",
                            "loss": loss_key,
                            "batch_size": batch_size,
                            "k_fraction": None,
                            "k": None,
                            "lr": None,
                            "rtol": None,
                            "model_seed": model_seed,
                            "loader_seed": loader_seed,
                            "svd_mode": None,
                            "svd_info": None,
                            "polyak_f_star": f_star,
                            "polyak_max_lr": max_lr,
                            "polyak_eps": eps,
                            "bn_mode": bn_of("polyak"),
                        },
                    ))

        # ---------------- Jacobian Descent ----------------
        if flags["jd"] and has_torchjd:
            jd_grid = product(
                hparams['batch_size'], hparams['lrs_jd'],
                hparams['aggregators_jd'], hparams['inner_optimizers_jd'],
            )
            for batch_size, lr, aggregator_name, inner_optim_name in jd_grid:
                if aggregator_name not in jd_aggregators:
                    if verbose:
                        print(f"  [skip] Unknown JD aggregator: {aggregator_name}")
                    continue
                run_id = (
                    f"jd_bs{batch_size}{id_str}"
                    f"_lr{lr}_agg{aggregator_name}_inner{inner_optim_name}{seed_str}"
                    f"{bn_mode_suffix(bn_of('jd'), 'jd')}{loss_suffix}"
                )
                specs.append(RunSpec(
                    family="jd", run_id=run_id, model_seed=model_seed,
                    loader_seed=loader_seed, batch_size=batch_size,
                    hparams={"lr": lr, "aggregator": aggregator_name,
                             "inner_optimizer": inner_optim_name},
                    record_extra={
                        "optimizer": f"JD_{aggregator_name}",
                        "loss": loss_key,
                        "batch_size": batch_size,
                        "lr": lr,
                        "aggregator": aggregator_name,
                        "inner_optimizer": inner_optim_name,
                        "model_seed": model_seed,
                        "loader_seed": loader_seed,
                        "svd_info": None,
                        "bn_mode": bn_of("jd"),
                    },
                ))

        # ---------------- Half-Inverse Gradients ----------------
        if flags["hig"]:
            hig_grid = product(hparams['batch_size'], hparams['lrs_hig'], hparams['tau_hig'])
            for batch_size, lr, tau in hig_grid:
                run_id = (f"hig_bs{batch_size}{id_str}_lr{lr}_tau{tau}{seed_str}"
                          f"{bn_mode_suffix(bn_of('hig'), 'hig')}{loss_suffix}")
                specs.append(RunSpec(
                    family="hig", run_id=run_id, model_seed=model_seed,
                    loader_seed=loader_seed, batch_size=batch_size,
                    hparams={"lr": lr, "tau": tau},
                    record_extra={
                        "optimizer": "HIG",
                        "loss": loss_key,
                        "batch_size": batch_size,
                        "lr": lr,
                        "tau": tau,
                        "model_seed": model_seed,
                        "loader_seed": loader_seed,
                        "svd_info": None,
                        "bn_mode": bn_of("hig"),
                    },
                ))

    return specs


def inject_dataset_facts(model_cfg, *, vocab_size=None, block_size=None):
    """The model config a language-model run actually instantiates (C-R3).

    ``experiments/configs/model/nanogpt.yaml`` deliberately omits ``vocab_size``:
    the dataset owns it.  :func:`generic_scan.run_grid` therefore injects the
    dataset's ``vocab_size`` (and ``block_size``, when the model config has that
    key) into ``cfg.model`` *before* anything is instantiated or hashed -- so the
    mutation is part of the run's identity, and any torch-free consumer that
    recomputes ``hash8`` (``tools/reconcile.py``) has to apply the **same**
    mutation or read the post-mutation config the runner saved in
    ``{scan}/configs/{job}.yaml``.  Composing the live config and hashing it
    unchanged gives a different hash8 for every LM run, which reports a finished
    nanoGPT scan as "work remains".

    Returns a new dict; the argument is never mutated.
    """
    model = dict(model_cfg or {})
    if vocab_size is not None:
        model["vocab_size"] = int(vocab_size)
    if block_size is not None and "block_size" in model:
        model["block_size"] = int(block_size)
    return model


def shard(specs, n_shards=1, shard_id=0):
    """The slice of ``specs`` one worker is responsible for.

    Identical to the legacy modulo counter (``generic_scan.py:366-369``), which
    incremented once per enumerated grid point in exactly this order.
    """
    n_shards, shard_id = int(n_shards), int(shard_id)
    assert n_shards >= 1 and 0 <= shard_id < n_shards, (n_shards, shard_id)
    return list(specs)[shard_id::n_shards]


# ---------------------------------------------------------------------------
# Run identity (C-R3)
# ---------------------------------------------------------------------------

#: record keys that are names or derived quantities, not inputs -- excluded from
#: the hash so renaming a field or adding an echo does not invalidate results.
_HASH_EXCLUDED_RECORD_KEYS = frozenset({
    "optimizer", "loss", "batch_size", "model_seed", "loader_seed",
    "k_fraction", "decomposition", "svd_info", "svd_mode",
})
#: evaluation settings (C-E1 / C-E3 / C-E4). They change what a record *says*, not
#: how the model trains, but a record made under a different `eval_batch_size` or a
#: different `train_eval` subset is a different record, so they are hashed. Taken
#: from the RESOLVED settings, so omitting a key and spelling out its default are
#: the same run (otherwise a config tidy-up would re-execute the whole campaign).
_HASH_EVAL_KEYS = ("eval_batch_size", "eval_every_steps", "train_eval_size")


def run_hash(spec, rcfg, *, schema_version=SCHEMA_VERSION):
    """sha256 hex of everything that defines what a run computes (C-R3).

    ``hash8`` = the first 8 characters, which is what the done marker and
    ``_stale/{hash8}/`` carry, so this function alone decides when a result on
    disk is still the result of the current code and config. The final rule:

    **Hashed** -- inputs to the computation or to the meaning of its numbers:
    ``schema_version``; the family, its ``hparams`` and the batch size; the
    model, loader and data seeds; the resolved ``dataset`` and ``model`` configs
    (which for an LM dataset must already carry the ``vocab_size`` /
    ``block_size`` that ``scan()`` injects into ``cfg.model`` -- hash *after* that
    mutation or a nanoGPT run hashes a model it never built); the loss key;
    ``num_epochs``; every input field of ``record_extra`` (kappa, microbatch
    size, param fraction + mask mode, the Gram settings, ``signed_residual``,
    ``bn_mode`` (C-E2) and Muon's construction rule token (C-B5)); and the
    resolved evaluation settings :data:`_HASH_EVAL_KEYS` (C-E1/C-E3/C-E4).

    **Not hashed** -- what is *logged or scheduled*, not computed:

    * the checkpoint policy (``checkpoints`` / ``checkpoints_svd``) and the
      spectra schedule (``svd_spectra_schedule`` / ``svd_spectra_every``): both
      decide which states and which spectra are written, never the trajectory,
      so re-running a scan with a denser ladder must not invalidate the runs it
      already has (C-L2/C-L3);
    * ``empty_cache`` (an allocator knob, C-T3), ``svd_info`` /
      ``track_param_norm`` (how much is recorded), and ``stop_on_nonfinite``
      (it only truncates a run that has already gone non-finite);
    * ``scheduler``, ``n_shards``, ``shard_id``: which worker runs it;
    * names and derived echoes -- :data:`_HASH_EXCLUDED_RECORD_KEYS`, e.g.
      ``k_fraction`` (derived from ``k`` and the batch size) and ``decomposition``.

    The ``dataset`` and ``model`` containers are hashed **as written**, key order
    apart (``sort_keys`` canonicalises that recursively).  Unlike the evaluation
    settings they cannot be resolved here -- their defaults live in the classes
    ``_target_`` names -- so spelling out a value that was already the default
    (``n_test: 10000``) or renaming a key *is* a new generation, and ``run_grid``
    then retires every finished run of that scan into ``_stale/`` and re-executes
    it.  A scan's ``dataset`` / ``model`` yaml is therefore frozen once its first
    job is submitted; check a pending edit with ``tools/reconcile.py`` before
    relaunching.
    """
    settings = {k: v for k, v in spec.record_extra.items()
                if k not in _HASH_EXCLUDED_RECORD_KEYS}
    resolved = resolve_scan_settings(rcfg)
    payload = {
        "schema_version": int(schema_version),
        "family": spec.family,
        "batch_size": spec.batch_size,
        "model_seed": spec.model_seed,
        "loader_seed": spec.loader_seed,
        "data_seed": rcfg.get("data_seed"),
        "hparams": spec.hparams,
        "settings": settings,
        "dataset": rcfg.get("dataset"),
        "model": rcfg.get("model"),
        "loss": rcfg.get("loss", "ce"),
        "num_epochs": rcfg.get("num_epochs"),
        "eval": {k: resolved[k] for k in _HASH_EVAL_KEYS},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash8(spec_or_hash, rcfg=None):
    """First 8 hex characters of :func:`run_hash` (the filename-safe short form)."""
    if isinstance(spec_or_hash, str):
        return spec_or_hash[:8]
    return run_hash(spec_or_hash, rcfg)[:8]


def config_digest(fragment, n=8):
    """Short stable digest of a config fragment (:func:`run_hash`'s canonical form)."""
    canonical = json.dumps(fragment, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:n]


def model_generation(rcfg, n=8):
    """Digest of the resolved ``model`` config: which model an init state belongs to.

    One scan directory can hold several *models*: ``result_id_fields:
    [mlp_width, n_data]`` (``rebuttal_overparam_*``) puts six jobs with six model
    configs in one directory, and a config edit replaces the model of a whole
    generation.  The shared per-seed initial state (``ckpt/init_mseed{seed}.*``)
    is therefore named by the model seed **and** this digest: without it the
    first job's initialisation would be the only one stored (``save_init_state``
    skips an existing file), silently standing in for every other model's.
    """
    return config_digest(rcfg.get("model"), n)
