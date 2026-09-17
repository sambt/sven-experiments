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
are stored; summary statistics include the median, a 10% trimmed mean and a
*steady-state mean* (first 20% of measured steps dropped, then >3 MAD outliers) -- use
that one as "mean step time ignoring start-up fluctuations".

Out-of-memory is a RESULT (``status: "oom"``), not a crash; methods whose Jacobian cannot
possibly fit (HIG on conv-nets / language models) are recorded as ``"infeasible"`` with
the analytic size instead of being attempted.  One JSON per configuration under
``{output_dir}/{config_name}/{run_id}.json``; existing files are skipped (resumable).
Run on an exclusively reserved node: co-tenant jobs skew launch-bound timings by up to 2x.
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

from experiments.experiment_code.experiment_utils import (
    build_standard_optimizer, set_seed, _is_closure_optimizer,
)
from experiments.experiment_code.generic_scan import (
    SVD_LOSS_FNS, STANDARD_LOSS_FNS, SVD_RESIDUAL_FNS, _JD_AGGREGATORS, _HAS_TORCHJD,
)
from experiments.optimizers.hig import HIGWrapper, HIGOptimizer
from sven.nn import SvenWrapper, GramSvenWrapper
from sven.opt import Sven, SvenGram

SVEN_VARIANTS = ("gram_hooks", "gram_full", "gram_chunked", "classic")

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
def summarize(values: list[float]) -> dict[str, float]:
    """Robust summary of a per-step series (ms or bytes)."""
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
        "steady_n": int(steady.size), "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
        "min": float(a.min()), "max": float(a.max()),
    }


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
            opt = Sven(w, lr=p["lr"], k=k, rtol=p["rtol"], svd_mode=svd_mode)
            meta.update(backend="classic", svd_mode=svd_mode)
        else:
            capture = {"gram_hooks": "hooks", "gram_full": "full", "gram_chunked": "chunked"}[method]
            kw: dict[str, Any] = {}
            if method == "gram_chunked":
                kw["chunk_numel"] = max(1, int(p["chunk_fraction"] * n_params))
                meta["chunk_numel"] = kw["chunk_numel"]
            freeze = True if capture == "hooks" else bool(rcfg.get("gram_freeze_norm_stats", True))
            w = GramSvenWrapper(model, loss_fn, device, microbatch_size=mb, param_fraction=pf,
                                mask_mode=mask_mode, capture=capture, freeze_norm_stats=freeze,
                                residual_fn=residual_fn, **kw)
            opt = SvenGram(w, lr=p["lr"], k=k, rtol=p["rtol"])
            meta.update(backend="gram", capture=capture, freeze_norm_stats=freeze)
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
    out_dir = os.path.join(prof.get("output_dir", "profile_results_v2"), config_name)
    os.makedirs(out_dir, exist_ok=True)

    dataset = instantiate(cfg.dataset)
    if hasattr(dataset, "vocab_size"):
        cfg.model.vocab_size = int(dataset.vocab_size)
        if hasattr(dataset, "block_size") and "block_size" in cfg.model:
            cfg.model.block_size = int(dataset.block_size)

    props = torch.cuda.get_device_properties(torch.device(device))
    env = {"gpu": props.name, "gpu_total_bytes": int(props.total_memory), "torch": torch.__version__,
           "cuda": torch.version.cuda, "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    jobs = expand_jobs(rcfg)
    only = rcfg.get("only_studies")
    if only:
        jobs = [j for j in jobs if j["study"] in only]
    print(f"[profile] {config_name}: {len(jobs)} configurations on {env['gpu']} -> {out_dir}/")

    for n, job in enumerate(jobs):
        rid = run_id_of(job)
        path = os.path.join(out_dir, rid + ".json")
        if os.path.exists(path):
            continue
        res = profile_one(job, cfg, rcfg, dataset, device, prof)
        res.update(run_id=rid, study=job["study"], method=job["method"], params=job["params"],
                   arch=rcfg.get("arch", config_name), config_name=config_name, loss=rcfg["loss"],
                   env=env, profile=prof)
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
