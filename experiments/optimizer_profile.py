#!/usr/bin/env python3
"""GPU peak-memory + steady-state step-time profiler for Sven and every baseline.

    PYTHONPATH=. python -m experiments.optimizer_profile --config-name profile_mnist

Unlike the training scans this is NOT a Cartesian grid: learning rate / rtol / kappa do
not change the cost of a step, so each architecture has one *set point* (``base``) and a
list of *studies*, each varying ONE axis around it:

    methods         every baseline + every Sven variant at the set point
    batch_size      B sweep            (Gram is B^2, the Jacobian is B*P)
    chunk_fraction  gram_chunked with parameter groups of f*P elements
    param_fraction  pf sweep x mask_mode, per Sven variant
    microbatch      microbatch_size sweep, per Sven variant
    k               rank sweep (as a fraction of the row count)
    width           model-size sweep (any config key, e.g. mlp_width / model.n_embd)

Sven variants
    gram_hooks    GramSvenWrapper(capture="hooks")  + SvenGram   -- no Jacobian at all
    gram_full     GramSvenWrapper(capture="full")   + SvenGram   -- one jacrev: (B,P) J -> G -> eigh
    gram_chunked  GramSvenWrapper(capture="chunked")+ SvenGram   -- jacrev per group of f*P params
    classic       SvenWrapper + Sven(svd_mode="randomized_v2")   -- (B,P) J -> randomized SVD pinv

Protocol (per configuration): a few batches are staged on the GPU and cycled, so data
loading / H2D copies are outside the timed region; ``warmup_steps`` untimed steps; then
up to ``num_steps`` measured steps (at least ``min_steps``, stopping early once
``max_seconds`` of measured time has elapsed).  Each measured step records CUDA-event
time, wall time, peak allocated / reserved bytes and, for Sven, the capture
(``loss_and_grad``) vs solve+apply (``optimizer.step``) split.  The raw per-step lists
are stored; summary statistics include the median, a 10% trimmed mean, a *steady-state
mean* (first 20% of measured steps dropped, then >3 MAD outliers) and a *cycle mean*
(last 80% of measured steps, truncated to whole 10-step cycles) -- use the LAST one as
the cost of a step: the MAD filter deletes the periodic refresh that a method like SOAP
actually pays for (C-T2).

Out-of-memory is a RESULT (``status: "oom"``), not a crash; methods whose Jacobian cannot
possibly fit (HIG on conv-nets / language models) are recorded as ``"infeasible"`` with
the analytic size instead of being attempted, and a method that left the parameters
non-finite as ``"nonfinite"`` (its timings are still recorded).  One JSON per configuration under
``{output_root}/{config_name}/{run_id}.json``; existing files are skipped (resumable).
Run on an exclusively reserved node: co-tenant jobs skew launch-bound timings by up to 2x.

Where the results go
    ``$SV3_PROFILE_ROOT``, else ``profile.output_dir`` from the config, else
    :data:`DEFAULT_OUTPUT_ROOT` (see :func:`output_root`); the resolved absolute path is
    printed and stored in every record.  The environment variable wins so that a job
    running from a deploy snapshot (cwd = the snapshot) can send its results to an
    absolute root outside it without touching the configs.

What the numbers are measured WITH (the 2026-09-17 profile got both of these wrong, which
is why ``profile_results_v3`` exists next to ``v2``):
    * ``empty_cache`` is :data:`EMPTY_CACHE` = False for every Sven variant, the optimizer's
      own default and the campaign's setting.  The per-step ``torch.cuda.empty_cache()``
      made full-capture Sven up to 4.5x slower on CIFAR (841 -> 187 ms/step) and was the
      sole source of its step-time variance.  ``empty_cache()`` is called only BETWEEN
      configurations (``profile_one``'s set-up and teardown), never inside a measured step.
    * ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``, as every campaign job had
      (``tools/worker_pool.sh``).  The allocator setting in force is recorded per result.
    * ``bn_mode`` is resolved exactly as ``experiments/experiment_code/grid.py`` resolves
      it for the scans, so the Gram variants freeze the norm statistics iff the scan they
      inherit from does.  CIFAR (``bn_mode: batch``, C-E2) therefore profiles with batch
      statistics, like the runs whose step time these tables are compared against.

Every record carries ``provenance`` (both repos' git SHAs via
``experiments/experiment_code/provenance.py``, host, SLURM job id, torch/CUDA) and ``env``
(GPU name and size, allocator setting, ``sven_empty_cache``), so a table can always say
which code and which allocator produced it.
"""
from __future__ import annotations

import copy
import gc
import json
import os
import socket
import time
from typing import Any, Callable

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from experiments.experiment_code import provenance
from experiments.experiment_code.experiment_utils import (
    build_standard_optimizer, set_seed, _is_closure_optimizer,
)
from experiments.experiment_code.generic_scan import (
    SVD_LOSS_FNS, STANDARD_LOSS_FNS, SVD_RESIDUAL_FNS, _JD_AGGREGATORS, _HAS_TORCHJD,
)
from experiments.experiment_code.grid import default_bn_mode, resolve_bn_mode
from experiments.optimizers.hig import HIGWrapper, HIGOptimizer
from sven.nn import SvenWrapper, GramSvenWrapper
from sven.opt import Sven, SvenGram

SVEN_VARIANTS = ("gram_hooks", "gram_full", "gram_chunked", "classic")

#: Where results go when neither ``$SV3_PROFILE_ROOT`` nor ``profile.output_dir`` says.
#: v2 (2026-09-17) is FROZEN as the before-table of the ``empty_cache`` fix: never write
#: into it, so the v2-vs-v3 comparison in the profile notebooks keeps a fixed reference.
DEFAULT_OUTPUT_ROOT = "profile_results_v3"

#: Sven's ``empty_cache`` for every profiled variant: the optimizer's own default and the
#: campaign's setting (see the module docstring). Passed EXPLICITLY, so a future change of
#: the default cannot silently re-introduce the 4.5x penalty, and recorded per result.
EMPTY_CACHE = False

#: This repo and the nested `sven` checkout, from this file -- correct both in the live
#: tree and inside a deploy snapshot (where `provenance` falls back to DEPLOY_INFO.json).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SVEN_ROOT = os.path.join(REPO_ROOT, "sven")

# One fixed, non-diverging setting per baseline: cost does not depend on the learning rate.
BASELINE_SPECS: dict[str, dict[str, Any]] = {
    "Adam": {"optim": "Adam", "lr": 1e-3},
    "AdamW": {"optim": "AdamW", "lr": 1e-3},
    "SGD": {"optim": "SGD", "lr": 1e-2},
    "RMSprop": {"optim": "RMSprop", "lr": 1e-3},
    "Muon": {"optim": "Muon", "lr": 1e-3},
    "SOAP": {"optim": "SOAP", "lr": 1e-3},
    "Shampoo": {"optim": "Shampoo", "lr": 1e-3},
    "KFAC": {"optim": "KFAC", "lr": 1e-3},
    "PolyakSGD": {"optim": "PolyakSGD", "lr": None, "kwargs": {"f_star": 0.0, "max_lr": 1.0, "eps": 1e-8}},
    "LBFGS1": {"optim": "LBFGS", "lr": 0.1, "kwargs": {"max_iter": 1, "history_size": 10, "line_search_fn": "strong_wolfe"}},
    "LBFGS3": {"optim": "LBFGS", "lr": 0.1, "kwargs": {"max_iter": 3, "history_size": 10, "line_search_fn": "strong_wolfe"}},
}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
CYCLE = 10   # SOAP's `precondition_frequency`: the period of the most expensive refresh


def cycle_mean(values, cycle=CYCLE):
    """Mean over the last 80% of a per-step series, truncated to whole ``cycle``-step
    cycles -- the amortised cost of a step (C-T2).

    The first 20% are start-up; the remainder is truncated to a multiple of ``cycle``
    so that a refresh with period ``cycle`` is counted exactly the right number of
    times, whatever the window's phase.  NaN for an empty series.

    Duplicated in ``analysis/profile_helpers.py`` (which must stay torch-free, and this
    module imports torch at the top); ``tests/test_analysis_offline.py`` checks the two
    agree.
    """
    a = np.asarray(values if values is not None else [], dtype=float)
    if a.size == 0:
        return np.nan
    tail = a[int(0.2 * a.size):]
    n = (tail.size // cycle) * cycle
    return float(tail[:n].mean() if n else tail.mean())


def summarize(values: list[float]) -> dict[str, float]:
    """Robust summary of a per-step series (ms or bytes).

    ``cycle_mean`` is the headline statistic (C-T2): ``steady_mean`` drops the >3 MAD
    points, which for a method with a periodic refresh (SOAP every 10 steps, Sven's
    re-factorisations) deletes exactly the cost that has to be paid -- SOAP on
    ``profile_mnist`` reads 5.56 ms steady against 6.14 ms amortised.  ``steady_mean``
    is kept as the reference column.
    """
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {}
    s = np.sort(a)
    cut = int(0.1 * len(s))
    trimmed = s[cut: len(s) - cut] if len(s) - 2 * cut > 0 else s
    steady = a[int(0.2 * len(a)):] if len(a) >= 5 else a          # drop start-up steps
    med = np.median(steady)
    mad = 1.4826 * np.median(np.abs(steady - med))
    if mad > 0:
        steady = steady[np.abs(steady - med) <= 3 * mad]            # drop isolated spikes
    return {
        "n": int(a.size), "mean": float(a.mean()), "std": float(a.std()), "median": float(np.median(a)),
        "trimmed_mean": float(trimmed.mean()), "steady_mean": float(steady.mean()),
        "steady_n": int(steady.size), "cycle_mean": float(cycle_mean(a)), "cycle": int(CYCLE),
        "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
        "min": float(a.min()), "max": float(a.max()),
    }


# ---------------------------------------------------------------------------
# Resolution of the two settings that made v2 wrong
# ---------------------------------------------------------------------------
def output_root(prof: dict, base=None, env=None) -> str:
    """Absolute directory the per-configuration JSONs go under.

    ``$SV3_PROFILE_ROOT`` > ``profile.output_dir`` > :data:`DEFAULT_OUTPUT_ROOT`. The
    environment wins because a job launched from a deploy snapshot runs with cwd = the
    snapshot: a relative root would write results INTO the frozen export, and the sbatch
    must be able to redirect them (a smoke run to a temp root) without editing seven
    configs. The resolved path is printed and stored, so which of the three spoke is never
    a guess.

    A relative root is resolved against ``base`` -- the caller passes hydra's ORIGINAL cwd,
    so the results land where the operator ran the command even if ``hydra.job.chdir``
    moves the process into the run directory.
    """
    env = os.environ if env is None else env
    root = env.get("SV3_PROFILE_ROOT") or prof.get("output_dir") or DEFAULT_OUTPUT_ROOT
    root = os.path.expanduser(str(root))
    if not os.path.isabs(root):
        root = os.path.join(base or os.getcwd(), root)
    return os.path.abspath(root)


def bn_mode_of(rcfg: dict) -> str:
    """The scan's norm-statistics policy, by ``grid.py``'s rule (C-E2).

    ``bn_mode``, else the deprecated ``gram_freeze_norm_stats`` alias, else the per-family
    default -- here always the Gram family's, because this is the Sven branch.
    """
    return resolve_bn_mode(rcfg) or default_bn_mode("svd", bool(rcfg.get("use_gram", True)))


def alloc_conf(env=None) -> str | None:
    """The CUDA caching-allocator configuration in force (both spellings)."""
    env = os.environ if env is None else env
    return env.get("PYTORCH_CUDA_ALLOC_CONF") or env.get("PYTORCH_ALLOC_CONF")


# ---------------------------------------------------------------------------
# Job expansion
# ---------------------------------------------------------------------------
def expand_jobs(rcfg: dict) -> list[dict]:
    base = dict(rcfg["base"])
    defaults = {"param_fraction": 1.0, "mask_mode": None, "microbatch_size": 1,
                "chunk_fraction": None, "width": None, "k_fraction": 1.0}
    jobs: list[dict] = []

    def add(study: str, method: str, **over):
        p = {**defaults, **base, **over}
        if method == "gram_chunked" and p["chunk_fraction"] is None:
            p["chunk_fraction"] = 0.25
        jobs.append({"study": study, "method": method, "params": p})

    sven = list(rcfg.get("sven_variants", []))
    for study, spec in (rcfg.get("studies") or {}).items():
        spec = spec or {}
        variants = list(spec.get("variants", sven))
        if study == "methods":
            for m in list(rcfg.get("baselines", [])) + sven:
                add(study, m)
        elif study == "batch_size":
            for b in spec["values"]:
                for m in spec.get("methods", sven + ["Adam", "SGD"]):
                    add(study, m, batch_size=int(b))
        elif study == "chunk_fraction":
            for cf in spec["values"]:
                add(study, "gram_chunked", chunk_fraction=float(cf))
        elif study == "param_fraction":
            for v in variants:
                add(study, v, param_fraction=1.0)
                for mm in spec.get("mask_modes", ["elementwise"]):
                    for pf in spec["values"]:
                        if float(pf) < 1.0:
                            add(study, v, param_fraction=float(pf), mask_mode=mm)
        elif study == "microbatch":
            for v in variants:
                for mb in spec["values"]:
                    add(study, v, microbatch_size=int(mb))
        elif study == "k":
            for v in variants:
                for kf in spec["values"]:
                    add(study, v, k_fraction=float(kf))
        elif study == "width":
            for w in spec["values"]:
                for m in spec["methods"]:
                    add(study, m, width=w, width_key=spec["key"])
        else:
            raise ValueError(f"unknown study {study!r}")
    return jobs


def run_id_of(job: dict) -> str:
    p = job["params"]
    bits = [job["study"], job["method"], f"B{p['batch_size']}", f"kf{p['k_fraction']:g}"]
    if p["param_fraction"] < 1.0:
        bits.append(f"pf{p['param_fraction']:g}-{p['mask_mode']}")
    if p["microbatch_size"] != 1:
        bits.append(f"mb{p['microbatch_size']}")
    if job["method"] == "gram_chunked":
        bits.append(f"cf{p['chunk_fraction']:g}")
    if p.get("width") is not None:
        bits.append(f"w{p['width']}")
    return "__".join(bits)


# ---------------------------------------------------------------------------
# Method builders: return (step_fn, phase_fn | None, meta)
# ---------------------------------------------------------------------------
def build_method(job, model, loss_key, device, rcfg, n_params, sample_batch):
    method, p = job["method"], job["params"]
    B, mb, pf = p["batch_size"], p["microbatch_size"], p["param_fraction"]
    rows = B // mb
    k = max(1, int(round(p["k_fraction"] * rows)))
    meta: dict[str, Any] = {"rows": rows, "k": k}

    if method in SVEN_VARIANTS:
        loss_fn = SVD_LOSS_FNS[loss_key]
        residual_fn = SVD_RESIDUAL_FNS.get(loss_key) if mb == 1 else None
        mask_mode = p["mask_mode"] if pf < 1.0 else None
        if method == "classic":
            w = SvenWrapper(model, loss_fn, device, microbatch_size=mb, param_fraction=pf,
                            mask_mode=mask_mode, residual_fn=residual_fn)
            svd_mode = rcfg.get("classic_svd_mode", "randomized_v2")
            opt = Sven(w, lr=p["lr"], k=k, rtol=p["rtol"], svd_mode=svd_mode,
                       empty_cache=EMPTY_CACHE)
            meta.update(backend="classic", svd_mode=svd_mode, empty_cache=EMPTY_CACHE)
        else:
            capture = {"gram_hooks": "hooks", "gram_full": "full", "gram_chunked": "chunked"}[method]
            kw: dict[str, Any] = {}
            if method == "gram_chunked":
                kw["chunk_numel"] = max(1, int(p["chunk_fraction"] * n_params))
                meta["chunk_numel"] = kw["chunk_numel"]
            # C-E2: the norm policy is the SCAN's, resolved by grid.py's own two functions,
            # so a profile inherits `bn_mode: batch` (CIFAR) instead of silently freezing
            # the running statistics that the runs being costed keep updating. `hooks`
            # capture requires frozen statistics and is not offered where that clashes.
            bn_mode = bn_mode_of(rcfg)
            freeze = True if capture == "hooks" else (bn_mode == "frozen")
            w = GramSvenWrapper(model, loss_fn, device, microbatch_size=mb, param_fraction=pf,
                                mask_mode=mask_mode, capture=capture, freeze_norm_stats=freeze,
                                residual_fn=residual_fn, **kw)
            opt = SvenGram(w, lr=p["lr"], k=k, rtol=p["rtol"], empty_cache=EMPTY_CACHE)
            meta.update(backend="gram", capture=capture, freeze_norm_stats=freeze,
                        bn_mode=bn_mode, empty_cache=EMPTY_CACHE)
            if capture != "hooks":
                groups = w._param_groups()
                meta["n_groups"] = len(groups)
                meta["max_group_numel"] = max(sum(n for _, _, n in g) for g in groups)
        meta["analytic_jacobian_bytes"] = int(rows * pf * n_params * 4)
        meta["analytic_gram_bytes"] = int(rows * rows * 8)

        def capture_fn(batch):
            with torch.no_grad():
                w.loss_and_grad(batch)

        def solve_fn(batch):
            with torch.no_grad():
                opt.step(batch)

        return None, (capture_fn, solve_fn), meta, (w, opt)

    if method == "HIG":
        loss_fn = SVD_LOSS_FNS[loss_key]
        with torch.no_grad():   # staged = [(xb, yb), ...]; 2 samples so train-mode BatchNorm accepts the probe
            out_numel = int(model.to(device)(sample_batch[0][0][:2]).numel()) // 2
        jac_bytes = B * out_numel * n_params * 4
        meta.update(backend="hig", hig_rows=B * out_numel, analytic_jacobian_bytes=int(jac_bytes))
        total = torch.cuda.get_device_properties(device).total_memory
        if 2.5 * jac_bytes > total:
            raise _Infeasible(f"HIG output Jacobian is {jac_bytes / 1e9:.1f} GB "
                              f"({B}x{out_numel} rows x {n_params} params); GPU has {total / 1e9:.0f} GB", meta)
        w = HIGWrapper(model, loss_fn, device)
        opt = HIGOptimizer(w, lr=0.05, tau=1e-4)

        def step(batch):
            with torch.no_grad():
                w.output_and_loss_grad(batch)
                opt.step()
        return step, None, meta, (w, opt)

    if method == "JD":
        if not _HAS_TORCHJD:
            raise RuntimeError("torchjd not installed")
        from torchjd.autojac import backward as jd_backward, jac_to_grad
        model = model.to(device); model.train()
        loss_fn = SVD_LOSS_FNS[loss_key]
        agg = _JD_AGGREGATORS["UPGrad"]()
        inner = build_standard_optimizer(model, "Adam", 1e-3)
        params = list(model.parameters())
        meta.update(backend="jd", aggregator="UPGrad", inner="Adam")

        def step(batch):
            xb, yb = batch
            inner.zero_grad()
            jd_backward(loss_fn(model(xb), yb))
            jac_to_grad(params, agg)
            inner.step()
        return step, None, meta, (model, inner)

    spec = BASELINE_SPECS[method]
    model = model.to(device); model.train()
    loss_fn = STANDARD_LOSS_FNS[loss_key]
    opt = build_standard_optimizer(model, spec["optim"], spec["lr"], **spec.get("kwargs", {}))
    closure_based = _is_closure_optimizer(opt)
    meta.update(backend="standard", optim=spec["optim"], **{f"opt_{a}": b for a, b in spec.get("kwargs", {}).items()})

    def step(batch):
        xb, yb = batch
        if closure_based:
            def closure():
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                return loss
            opt.step(closure)
        else:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return step, None, meta, (model, opt)


class _Infeasible(Exception):
    def __init__(self, msg, meta):
        super().__init__(msg)
        self.meta = meta


# ---------------------------------------------------------------------------
# One configuration
# ---------------------------------------------------------------------------
def profile_one(job, cfg, rcfg, dataset, device, prof) -> dict:
    p = job["params"]
    dev = torch.device(device)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    result: dict[str, Any] = {"status": "ok"}
    keep: Any = None
    try:
        c = cfg
        if p.get("width") is not None:
            c = copy.deepcopy(cfg)
            OmegaConf.set_struct(c, False)
            OmegaConf.update(c, p["width_key"], p["width"], force_add=True)
        set_seed(int(rcfg.get("model_seed", 0)))
        model = instantiate(c.model)
        n_params = sum(q.numel() for q in model.parameters())
        result["n_params"] = int(n_params)

        loader = DataLoader(dataset.train_dataset, batch_size=p["batch_size"], shuffle=True, drop_last=True,
                            generator=torch.Generator().manual_seed(int(rcfg.get("loader_seed", 0))))
        staged = []
        for xb, yb in loader:
            staged.append((xb.to(dev), yb.to(dev)))
            if len(staged) >= int(prof.get("staged_batches", 8)):
                break
        if not staged:
            raise RuntimeError(f"dataset smaller than one batch of {p['batch_size']}")
        torch.cuda.synchronize(dev)
        pre = torch.cuda.memory_allocated(dev)
        model = model.to(dev)
        torch.cuda.synchronize(dev)
        result["baseline_model_bytes"] = int(torch.cuda.memory_allocated(dev) - pre)
        result["staged_bytes"] = int(pre)

        step_fn, phases, meta, keep = build_method(job, model, rcfg["loss"], device, rcfg, n_params, staged)
        result["meta"] = meta

        def one(batch):
            if phases is None:
                step_fn(batch)
            else:
                phases[0](batch); phases[1](batch)

        # ---- warm-up (untimed; also bounded in wall time) ----
        t0 = time.perf_counter()
        for i in range(int(prof["warmup_steps"])):
            one(staged[i % len(staged)])
            torch.cuda.synchronize(dev)
            if time.perf_counter() - t0 > float(prof.get("max_warmup_seconds", 120)) and i >= 1:
                break
        result["warmup_steps_done"] = i + 1

        # ---- measurement ----
        rec: dict[str, list] = {k: [] for k in ("step_ms", "wall_ms", "capture_ms", "solve_ms", "resident_bytes",
                                                "peak_alloc_bytes", "peak_reserved_bytes",
                                                "peak_capture_bytes", "peak_solve_bytes")}
        measured = 0.0
        for i in range(int(prof["num_steps"])):
            batch = staged[i % len(staged)]
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            rec["resident_bytes"].append(int(torch.cuda.memory_allocated(dev)))
            e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            w0 = time.perf_counter()
            e0.record()
            if phases is None:
                step_fn(batch)
                e2.record(); torch.cuda.synchronize(dev)
                peak = torch.cuda.max_memory_allocated(dev)
            else:
                phases[0](batch)
                e1.record(); torch.cuda.synchronize(dev)
                pk_cap = torch.cuda.max_memory_allocated(dev)
                phases[1](batch)
                e2.record(); torch.cuda.synchronize(dev)
                peak = torch.cuda.max_memory_allocated(dev)      # peak over the WHOLE step (no mid-step reset)
                rec["capture_ms"].append(float(e0.elapsed_time(e1)))
                rec["solve_ms"].append(float(e1.elapsed_time(e2)))
                rec["peak_capture_bytes"].append(int(pk_cap))
            wall = (time.perf_counter() - w0) * 1e3
            rec["step_ms"].append(float(e0.elapsed_time(e2)))
            rec["wall_ms"].append(float(wall))
            rec["peak_alloc_bytes"].append(int(peak))
            rec["peak_reserved_bytes"].append(int(torch.cuda.max_memory_reserved(dev)))
            measured += wall / 1e3
            if measured > float(prof.get("max_seconds", 120)) and i + 1 >= int(prof.get("min_steps", 10)):
                break
        if rec["step_ms"] and not np.all(np.isfinite(rec["step_ms"])):
            raise RuntimeError("non-finite step time")
        result["raw"] = {k: v for k, v in rec.items() if v}
        result["time"] = {k: summarize(rec[k]) for k in ("step_ms", "wall_ms", "capture_ms", "solve_ms") if rec[k]}
        result["memory"] = {
            "resident_bytes_mean": float(np.mean(rec["resident_bytes"])),
            "peak_alloc_bytes_max": int(max(rec["peak_alloc_bytes"])),
            "peak_alloc_bytes_mean": float(np.mean(rec["peak_alloc_bytes"])),
            "peak_reserved_bytes_max": int(max(rec["peak_reserved_bytes"])),
            "peak_capture_bytes_max": int(max(rec["peak_capture_bytes"])) if rec["peak_capture_bytes"] else None,
        }
        # A method that blew the parameters up still produces perfectly good step times,
        # so the timings are kept -- but the configuration is not "ok" (C-T2).
        if not all(torch.isfinite(q).all().item() for q in model.parameters()):
            result["status"] = "nonfinite"
    except _Infeasible as e:
        result.update(status="infeasible", error=str(e), meta=e.meta)
    except torch.cuda.OutOfMemoryError as e:
        result.update(status="oom", error=str(e)[:300],
                      peak_alloc_bytes_at_failure=int(torch.cuda.max_memory_allocated(dev)))
    except Exception as e:  # noqa: BLE001 -- unsupported combos are data, not crashes
        msg = str(e)
        result.update(status="oom" if "out of memory" in msg.lower() else "error",
                      error=f"{type(e).__name__}: {msg[:300]}")
    finally:
        del keep
        gc.collect(); torch.cuda.empty_cache()
        try:
            torch.compiler.reset()
        except Exception:  # noqa: BLE001
            pass
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
@hydra.main(config_path="configs", version_base=None)
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("optimizer_profile needs a CUDA device (memory statistics).")
    OmegaConf.set_struct(cfg, False)
    config_name = HydraConfig.get().job.config_name
    rcfg = OmegaConf.to_container(cfg, resolve=True)
    prof = {"warmup_steps": 10, "num_steps": 50, "min_steps": 10, "max_seconds": 120, **(rcfg.get("profile") or {})}
    rcfg["model_seed"] = (rcfg.get("model_seeds") or [0])[0]
    device = rcfg.get("device", "cuda")
    root = output_root(prof, base=HydraConfig.get().runtime.cwd)
    out_dir = os.path.join(root, config_name)
    os.makedirs(out_dir, exist_ok=True)
    prof["output_root"] = root

    dataset = instantiate(cfg.dataset)
    if hasattr(dataset, "vocab_size"):
        cfg.model.vocab_size = int(dataset.vocab_size)
        if hasattr(dataset, "block_size") and "block_size" in cfg.model:
            cfg.model.block_size = int(dataset.block_size)

    props = torch.cuda.get_device_properties(torch.device(device))
    env = {"gpu": props.name, "gpu_total_bytes": int(props.total_memory), "torch": torch.__version__,
           "cuda": torch.version.cuda, "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           # the two settings v2 got wrong, recorded with every number they produced
           "alloc_conf": alloc_conf(), "sven_empty_cache": EMPTY_CACHE,
           "bn_mode": bn_mode_of(rcfg), "output_root": root}
    # collected ONCE (the git calls fork): both repos' SHAs, host, SLURM id, torch/CUDA/GPU
    prov = provenance.collect(REPO_ROOT, SVEN_ROOT)
    jobs = expand_jobs(rcfg)
    only = rcfg.get("only_studies")
    if only:
        jobs = [j for j in jobs if j["study"] in only]
    print(f"[profile] {config_name}: {len(jobs)} configurations on {env['gpu']} -> {out_dir}/")
    print(f"[profile] sv3 {prov['git_sha']} (dirty={prov['git_dirty']}, {prov['git_source']})  "
          f"sven {prov['sven_git_sha']} (dirty={prov['sven_git_dirty']}, {prov['sven_git_source']})")
    print(f"[profile] alloc_conf={env['alloc_conf']!r}  sven_empty_cache={EMPTY_CACHE}  "
          f"bn_mode={env['bn_mode']}", flush=True)

    for n, job in enumerate(jobs):
        rid = run_id_of(job)
        path = os.path.join(out_dir, rid + ".json")
        if os.path.exists(path):
            continue
        started = provenance.start_stamp()
        res = profile_one(job, cfg, rcfg, dataset, device, prof)
        res.update(run_id=rid, study=job["study"], method=job["method"], params=job["params"],
                   arch=rcfg.get("arch", config_name), config_name=config_name, loss=rcfg["loss"],
                   env=env, profile=prof, provenance=prov,
                   **provenance.end_stamp(started))
        with open(path, "w") as f:
            json.dump(res, f)
        t = res.get("time", {}).get("step_ms", {})
        mem = res.get("memory", {})
        print(f"[{n + 1}/{len(jobs)}] {rid}: {res['status']}"
              + (f"  steady {t['steady_mean']:.2f} ms (median {t['median']:.2f}, n={t['n']})"
                 f"  peak {mem['peak_alloc_bytes_max'] / 1e6:.0f} MB" if t else f"  {res.get('error', '')[:120]}"),
              flush=True)


if __name__ == "__main__":
    main()
