#!/usr/bin/env python3
"""Measurement plan for the campaign execution-layout probe.

Pure python (no torch, no hydra) so the driver, the analysis and the CPU tests can all
import it in milliseconds.  Every measurement is one process running
``experiments.optimizer_profile`` restricted to a SINGLE configuration, so the careful
staged-batch / CUDA-event / peak-memory protocol that already lives there is reused
verbatim; this module only decides *which* configuration and *how many* copies of it run
at the same time.

A measurement is identified by ``tag``; a group is the ``nproc`` copies of one tag that
run concurrently (they synchronise on a file barrier immediately before the warm-up, so
the timed regions overlap).
"""
from __future__ import annotations

import dataclasses

# ---------------------------------------------------------------------------
# Hydra override fragments
# ---------------------------------------------------------------------------
def _list(xs) -> str:
    return "[" + ",".join(str(x) for x in xs) + "]"


def _methods_ov(baselines, sven_variants) -> list[str]:
    """Run the ``methods`` study with exactly one method in it."""
    return ["+only_studies=[methods]",
            f"baselines={_list(baselines)}",
            f"sven_variants={_list(sven_variants)}"]


def _chunk_ov(fraction: float) -> list[str]:
    """Run the ``chunk_fraction`` study at exactly one chunk fraction."""
    return ["+only_studies=[chunk_fraction]",
            f"studies.chunk_fraction.values={_list([fraction])}"]


# workload -> (profile config name, hydra overrides, the run_id optimizer_profile will emit)
WORKLOADS: dict[str, tuple[str, list[str], str]] = {
    # CIFAR-10 ResNet18, B=128, batch-statistics BatchNorm (profile_cifar inherits
    # gram_freeze_norm_stats: false from the headline scan config).
    "cifar_sven_full":   ("profile_cifar",   _methods_ov([], ["gram_full"]),
                          "methods__gram_full__B128__kf1"),
    "cifar_sven_cf0.5":  ("profile_cifar",   _chunk_ov(0.5),
                          "chunk_fraction__gram_chunked__B128__kf1__cf0.5"),
    "cifar_sven_cf0.25": ("profile_cifar",   _chunk_ov(0.25),
                          "chunk_fraction__gram_chunked__B128__kf1__cf0.25"),
    "cifar_adam":        ("profile_cifar",   _methods_ov(["Adam"], []),
                          "methods__Adam__B128__kf1"),
    # MLP / transformer classes, at the production backend (use_gram + gram_capture: hooks)
    "toy1d_sven":        ("profile_toy_1d",  _methods_ov([], ["gram_hooks"]),
                          "methods__gram_hooks__B32__kf1"),
    "mnist_sven":        ("profile_mnist",   _methods_ov([], ["gram_hooks"]),
                          "methods__gram_hooks__B64__kf1"),
    "mnist_adam":        ("profile_mnist",   _methods_ov(["Adam"], []),
                          "methods__Adam__B64__kf1"),
    "mnist_lbfgs3":      ("profile_mnist",   _methods_ov(["LBFGS3"], []),
                          "methods__LBFGS3__B64__kf1"),
    "nanogpt_sven":      ("profile_nanogpt", _methods_ov([], ["gram_hooks"]),
                          "methods__gram_hooks__B64__kf1"),
}

# profile.* settings per workload class.  ``min_steps`` is what guarantees the "at least
# 60 measured steps" requirement of part (b); ``max_seconds`` caps the launch-bound MLP
# runs, which would otherwise spend minutes on thousands of 10 ms steps.
STEP_SETTINGS: dict[str, dict[str, float]] = {
    "cifar_b":    dict(warmup_steps=12, num_steps=75,   min_steps=60,  max_seconds=180, max_warmup_seconds=150),
    "cifar_co":   dict(warmup_steps=8,  num_steps=60,   min_steps=40,  max_seconds=200, max_warmup_seconds=200),
    "cifar_adam": dict(warmup_steps=20, num_steps=400,  min_steps=200, max_seconds=30,  max_warmup_seconds=60),
    "mlp":        dict(warmup_steps=30, num_steps=4000, min_steps=400, max_seconds=20,  max_warmup_seconds=120),
    "nanogpt":    dict(warmup_steps=10, num_steps=400,  min_steps=80,  max_seconds=25,  max_warmup_seconds=120),
}

CAPTURES = {"full": "cifar_sven_full", "cf0.5": "cifar_sven_cf0.5", "cf0.25": "cifar_sven_cf0.25"}
EXPANDABLE = "expandable_segments:True"
MLP_NPROCS = (1, 4, 6, 8, 12)


@dataclasses.dataclass(frozen=True)
class Measurement:
    part: str                  # b | c | d | e | cifarslice
    tag: str                   # unique, filesystem-safe
    workload: str              # key into WORKLOADS
    nproc: int                 # concurrent copies on the one GPU
    steps: str                 # key into STEP_SETTINGS
    alloc_conf: str = ""       # PYTORCH_CUDA_ALLOC_CONF ("" = leave unset)
    no_empty_cache: bool = False   # monkeypatch torch.cuda.empty_cache to a no-op
    capture: str = ""          # "full" | "cf0.5" | "cf0.25" for the CIFAR parts
    note: str = ""

    @property
    def layout(self) -> str:
        bits = [f"nproc{self.nproc}"]
        if self.capture:
            bits.append(self.capture)
        bits.append("ec-off" if self.no_empty_cache else "ec-on")
        bits.append("expandable" if self.alloc_conf else "alloc-default")
        return " ".join(bits)


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------
def plan_b() -> list[Measurement]:
    """(b) CIFAR ResNet18 Sven: capture x empty_cache x allocator, one process each."""
    out = []
    for cap, wl in CAPTURES.items():
        for ec_off in (False, True):
            for alloc in ("", EXPANDABLE):
                tag = (f"b__{cap}__ec-{'off' if ec_off else 'on'}"
                       f"__alloc-{'expandable' if alloc else 'default'}")
                out.append(Measurement("b", tag, wl, 1, "cifar_b", alloc, ec_off, cap))
    return out


def plan_c(best: dict) -> list[Measurement]:
    """(c) co-tenancy: Sven-ResNet at 2 and 3 processes in the best mode from (b);
    Adam-ResNet at 1 and 4."""
    cap = best.get("capture") or "cf0.25"
    alloc = best.get("alloc_conf") or ""
    ec_off = bool(best.get("no_empty_cache"))
    out = []
    for n in (1, 2, 3):
        tag = f"c__sven-{cap}__n{n}"
        out.append(Measurement("c", tag, CAPTURES[cap], n, "cifar_co", alloc, ec_off, cap,
                               note="best mode from part (b)"))
    for n in (1, 4):
        out.append(Measurement("c", f"c__adam__n{n}", "cifar_adam", n, "cifar_adam", alloc, False))
    return out


def plan_d() -> list[Measurement]:
    """(d) MLP sharding plateau: four workload classes x NPROC in {1,4,6,8,12}."""
    out = []
    for wl in ("toy1d_sven", "mnist_sven", "mnist_adam", "mnist_lbfgs3"):
        for n in MLP_NPROCS:
            out.append(Measurement("d", f"d__{wl}__n{n}", wl, n, "mlp"))
    return out


def plan_e() -> list[Measurement]:
    """(e) nanoGPT Sven at NPROC 1, 2, 3."""
    return [Measurement("e", f"e__nanogpt_sven__n{n}", "nanogpt_sven", n, "nanogpt")
            for n in (1, 2, 3)]


def plan_cifarslice() -> list[Measurement]:
    """MIG job only: CIFAR Sven chunked 0.25 on the slice (20 GB holds its 6.9 GB peak)."""
    return [Measurement("cifarslice", "slice__cifar_sven_cf0.25__n1", "cifar_sven_cf0.25",
                        1, "cifar_b", "", False, "cf0.25")]


PART_FNS = {"b": plan_b, "d": plan_d, "e": plan_e, "cifarslice": plan_cifarslice}


def plan_part(part: str, best: dict | None = None) -> list[Measurement]:
    if part == "c":
        return plan_c(best or {})
    return PART_FNS[part]()


# ---------------------------------------------------------------------------
# "best mode from (b)"
# ---------------------------------------------------------------------------
SPEED_TIE_TOL = 0.03      # 3%: closer than this is measurement noise, not a real difference


def choose_best_b(records: list[dict]) -> dict:
    """The mode part (c) should co-tenant in: fastest of part (b), but among modes within
    `SPEED_TIE_TOL` of the fastest, the one that reserves the least memory.

    Measured on the A100: full/ec-off/default is 186.5 ms reserving 32.8 GB and
    full/ec-off/expandable is 186.7 ms reserving 23.3 GB.  A strict speed ordering picks
    the first and then cannot fit three co-tenants in 80 GB, so the 0.1% "win" costs the
    measurement it was chosen for.  Reserved memory is what decides co-tenancy.

    Falls back to chunked 0.25 with the stock allocator when part (b) produced nothing.
    """
    def wall(r):        # wall time, not CUDA-event time; see analyse.group_stats
        return ((r.get("wall_ms") or {}).get("median")
                or (r.get("step_ms") or {}).get("median"))

    cand = [r for r in records if r.get("part") == "b" and r.get("status") == "ok" and wall(r)]
    if not cand:
        return {"capture": "cf0.25", "alloc_conf": "", "no_empty_cache": False, "source": "fallback"}
    floor = min(wall(r) for r in cand)
    tied = [r for r in cand if wall(r) <= floor * (1.0 + SPEED_TIE_TOL)]
    best = min(tied, key=lambda r: ((r.get("memory") or {}).get("peak_reserved_bytes_max") or 0,
                                    wall(r)))
    return {"capture": best.get("capture") or "cf0.25",
            "alloc_conf": best.get("alloc_conf") or "",
            "no_empty_cache": bool(best.get("no_empty_cache")),
            "source": best.get("tag"),
            "median_ms": wall(best),
            "n_tied_within_tol": len(tied)}


# ---------------------------------------------------------------------------
# Hydra argv for one measurement
# ---------------------------------------------------------------------------
def hydra_overrides(m: Measurement, out_root: str, hydra_run_dir: str) -> list[str]:
    """key=value overrides only (no `--config-name`), so the tests can replay them."""
    _cfg, overrides, _rid = WORKLOADS[m.workload]
    ov = ["print_config=false", "hydra.job.chdir=False", "hydra.output_subdir=null",
          f"hydra.run.dir={hydra_run_dir}", f"profile.output_dir={out_root}"]
    for k, v in STEP_SETTINGS[m.steps].items():
        ov.append(f"profile.{k}={v:g}")
    return ov + list(overrides)


def hydra_args(m: Measurement, out_root: str, hydra_run_dir: str) -> list[str]:
    return ["--config-name", WORKLOADS[m.workload][0]] + hydra_overrides(m, out_root, hydra_run_dir)


def expected_run_id(m: Measurement) -> str:
    return WORKLOADS[m.workload][2]


def config_name(m: Measurement) -> str:
    return WORKLOADS[m.workload][0]
