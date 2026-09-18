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


def resolve_scan_settings(rcfg):
    """Scan-level values derived from the config (``generic_scan.py:311-341``)."""
    loss_key = rcfg.get("loss", "ce")
    if loss_key not in LOSS_KEYS:
        raise KeyError(f"Unknown loss key {loss_key!r}; known: {sorted(LOSS_KEYS)}")
    svd_info_mode = rcfg.get("svd_info", "full")
    if svd_info_mode not in SVD_INFO_MODES:
        raise ValueError(f"svd_info must be one of {SVD_INFO_MODES}, got {svd_info_mode!r}")
    return {
        "loss_key": loss_key,
        # Non-legacy loss keys are encoded in the run_id so a `loss=` override can
        # never dedup against that config's original-loss results.
        "loss_suffix": "" if loss_key in LEGACY_LOSS_KEYS else f"_loss{loss_key}",
        "track_acc": loss_key in ("ce", "brier") or ("label_regression" in loss_key),
        "is_lm": loss_key == "lm_ce",
        "track_param_norm": rcfg.get("track_param_norm", False),
        "svd_info_mode": svd_info_mode,
        "svd_spectra_every": int(rcfg.get("svd_spectra_every", 20)),
        # Signed-residual Sven rows; scalar-output regression only. Not in the run_id
        # (the update is identical wherever the loss path is finite).
        "signed_residual": (bool(rcfg.get("signed_residual", True))
                            and loss_key in SIGNED_RESIDUAL_LOSS_KEYS),
    }


def resolve_svd_settings(rcfg):
    """Scan-level Sven/Gram settings, validated (``generic_scan.py:429-460``)."""
    variable_k = rcfg.get("variable_k", False)
    use_gram = rcfg.get("use_gram", False)
    if use_gram and variable_k:
        raise ValueError("use_gram is incompatible with variable_k")
    gram_capture = rcfg.get("gram_capture", "hooks")
    gram_freeze_norm_stats = bool(rcfg.get("gram_freeze_norm_stats", True))
    gram_chunk_numel = int(rcfg.get("gram_chunk_numel", 2 ** 22))
    if use_gram and not gram_freeze_norm_stats and gram_capture not in ("chunked", "full"):
        raise ValueError("gram_freeze_norm_stats=false requires gram_capture: chunked or full")
    return {
        "variable_k": variable_k,
        "use_gram": use_gram,
        "gram_capture": gram_capture,
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
                    if not sv["gram_freeze_norm_stats"]:
                        run_id += "_bnbatch"  # batch-statistics BatchNorm
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
                    run_id += seed_str + loss_suffix

                    specs.append(RunSpec(
                        family="standard", run_id=run_id, model_seed=model_seed,
                        loader_seed=loader_seed, batch_size=batch_size,
                        hparams={"optim_name": optim_name, "lr": lr,
                                 "weight_decay": weight_decay},
                        record_extra={
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
                        },
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
                        f"{loss_suffix}"
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
                        f"{loss_suffix}"
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
                    f"_lr{lr}_agg{aggregator_name}_inner{inner_optim_name}{seed_str}{loss_suffix}"
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
                    },
                ))

        # ---------------- Half-Inverse Gradients ----------------
        if flags["hig"]:
            hig_grid = product(hparams['batch_size'], hparams['lrs_hig'], hparams['tau_hig'])
            for batch_size, lr, tau in hig_grid:
                run_id = f"hig_bs{batch_size}{id_str}_lr{lr}_tau{tau}{seed_str}{loss_suffix}"
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
                    },
                ))

    return specs


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
#: evaluation settings; all absent today, live after C-E1/C-E4.
_HASH_EVAL_KEYS = ("eval_batch_size", "eval_every_steps")


def run_hash(spec, rcfg, *, schema_version=SCHEMA_VERSION):
    """sha256 hex of everything that defines what a run computes.

    Covers the family and its hyperparameters, the seeds, the batch size, the
    resolved dataset / model configs, the loss, the epoch count, the evaluation
    settings and ``schema_version`` (C-R3). ``hash8`` = first 8 characters.

    ``rcfg`` must be the *resolved* container and, for LM datasets, must already
    carry the vocab_size / block_size injected into ``cfg.model`` -- otherwise a
    nanoGPT run hashes its un-injected model config.
    """
    settings = {k: v for k, v in spec.record_extra.items()
                if k not in _HASH_EXCLUDED_RECORD_KEYS}
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
        "eval": {k: rcfg.get(k) for k in _HASH_EVAL_KEYS},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash8(spec_or_hash, rcfg=None):
    """First 8 hex characters of :func:`run_hash` (the filename-safe short form)."""
    if isinstance(spec_or_hash, str):
        return spec_or_hash[:8]
    return run_hash(spec_or_hash, rcfg)[:8]
