#!/usr/bin/env python3
"""One probe measurement, in its own process, pinned to a frozen code snapshot.

    PYTHONPATH=$SNAP:$SNAP/sven .venv/bin/python bench/probe_campaign/probe_run.py \
        --snapshot $SNAP --config-name profile_cifar --out-root <dir> --jsonl <file> ...

Everything that measures is ``experiments.optimizer_profile``; this wrapper only

  1. refuses to run unless ``sven`` and ``experiments`` both resolve INSIDE the snapshot
     (the editable install of ``sven`` points at the live working tree, which ~10 agents
     are editing right now, so a silent fall-back would poison every number),
  2. optionally monkeypatches ``torch.cuda.empty_cache`` to a no-op (C-T3: the per-step
     calls in ``sven/sven/opt/sven.py`` are not yet behind the ``empty_cache`` flag),
  3. holds the process at a file barrier immediately before warm-up so that the timed
     regions of the ``nproc`` co-tenants actually overlap,
  4. asserts the hydra overrides expand to EXACTLY ONE configuration (a typo that
     silently expands to 40 would burn the whole allocation), and
  5. appends one JSON line with the measurement and its labels, so a timeout still
     leaves every completed measurement on disk.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import sys
import time

MAX_RAW_STEPS = 400   # keep the per-step lists only when they are small enough to be cheap


def strip_editable_finders() -> list[str]:
    """Drop setuptools' editable-install meta finders so `import sven` cannot silently
    resolve to the live working tree instead of the snapshot on sys.path."""
    removed = []
    keep = []
    for f in sys.meta_path:
        mod = str(getattr(f, "__module__", ""))
        name = str(getattr(f, "__name__", type(f).__name__))
        if mod.startswith("__editable__") or name == "_EditableFinder":
            removed.append(f"{mod}.{name}")
            continue
        keep.append(f)
    sys.meta_path[:] = keep
    return removed


def check_provenance(snapshot: str) -> dict:
    import experiments
    import sven
    snap = os.path.realpath(snapshot)
    got = {"sven": os.path.realpath(sven.__file__), "experiments": os.path.realpath(experiments.__file__)}
    print(f"[probe] snapshot   = {snap}", flush=True)
    print(f"[probe] sven       = {got['sven']}", flush=True)
    print(f"[probe] experiments= {got['experiments']}", flush=True)
    bad = [k for k, v in got.items() if not v.startswith(snap + os.sep)]
    if bad:
        raise SystemExit(f"[probe] ABORT: {', '.join(bad)} resolved outside the snapshot: {got}")
    return got


def barrier_wait(bdir: str, n: int, timeout: float) -> tuple[bool, float]:
    """Rendezvous of `n` processes on a shared directory (one file each)."""
    if n <= 1:
        return True, 0.0
    os.makedirs(bdir, exist_ok=True)
    with open(os.path.join(bdir, f"{os.getpid()}.ready"), "w") as f:
        f.write(socket.gethostname())
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            if len(os.listdir(bdir)) >= n:
                return True, time.perf_counter() - t0
        except OSError:
            pass
        time.sleep(0.1)
    return False, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--out-root", required=True, help="profile.output_dir (absolute)")
    ap.add_argument("--jsonl", required=True, help="where this process appends its one result line")
    ap.add_argument("--labels", default="{}", help="JSON dict merged into the result line")
    ap.add_argument("--expect-run-id", default=None)
    ap.add_argument("--barrier-dir", default=None)
    ap.add_argument("--barrier-n", type=int, default=1)
    ap.add_argument("--barrier-timeout", type=float, default=900.0)
    ap.add_argument("--no-empty-cache", action="store_true")
    ap.add_argument("--hydra-arg", action="append", default=[], help="passed through to optimizer_profile")
    a = ap.parse_args()

    t_start = time.time()
    labels = json.loads(a.labels)
    snap = os.path.realpath(a.snapshot)
    for p in (os.path.join(snap, "sven"), snap):
        if p not in sys.path:
            sys.path.insert(0, p)
    removed = strip_editable_finders()

    import torch
    if a.no_empty_cache:
        torch.cuda.empty_cache = lambda *args, **kw: None       # C-T3 axis, measured not assumed

    import experiments.optimizer_profile as op
    prov = check_provenance(snap)

    # --- fail fast on an override typo -------------------------------------------------
    _expand = op.expand_jobs

    def expand_one(rcfg):
        jobs = _expand(rcfg)
        only = rcfg.get("only_studies")
        if only:
            jobs = [j for j in jobs if j["study"] in only]
        rids = [op.run_id_of(j) for j in jobs]
        print(f"[probe] expanded to {len(jobs)} configuration(s): {rids}", flush=True)
        if len(jobs) != 1:
            raise SystemExit(f"[probe] ABORT: expected exactly 1 configuration, got {len(jobs)}: {rids}")
        if a.expect_run_id and rids[0] != a.expect_run_id:
            raise SystemExit(f"[probe] ABORT: run_id {rids[0]!r} != expected {a.expect_run_id!r}")
        return jobs

    op.expand_jobs = expand_one

    # --- barrier immediately before warm-up + measurement ------------------------------
    _profile_one = op.profile_one
    state = {"barrier_ok": True, "barrier_wait_s": 0.0, "t_release": None}

    def profile_one_barriered(*args, **kw):
        if a.barrier_dir:
            ok, waited = barrier_wait(a.barrier_dir, a.barrier_n, a.barrier_timeout)
            state.update(barrier_ok=ok, barrier_wait_s=waited)
            print(f"[probe] barrier {'ok' if ok else 'TIMEOUT'} after {waited:.1f}s "
                  f"({a.barrier_n} procs)", flush=True)
        state["t_release"] = time.time()
        return _profile_one(*args, **kw)

    op.profile_one = profile_one_barriered

    out_dir = os.path.join(a.out_root, a.config_name)
    if a.expect_run_id:                                  # re-measure rather than resume
        try:
            os.remove(os.path.join(out_dir, a.expect_run_id + ".json"))
        except OSError:
            pass

    # `@hydra.main(config_path="configs")` resolves its search path from the task
    # function's module.  Under `python -m experiments.optimizer_profile` that module is
    # `__main__` and hydra resolves "configs" next to the source file; imported from here
    # it is `experiments.optimizer_profile`, and hydra then looks for an importable
    # `experiments.configs` package, which does not exist.  This env var (hydra's own
    # hook) puts it back on the file-relative branch -> <snapshot>/experiments/configs.
    os.environ["HYDRA_MAIN_MODULE"] = "__main__"
    sys.argv = ["optimizer_profile"] + list(a.hydra_arg)
    print(f"[probe] argv: {' '.join(sys.argv[1:])}", flush=True)
    status = "ok"
    err = ""
    try:
        op.main()
    except BaseException as e:                            # noqa: BLE001 -- record, do not lose the group
        status, err = "driver_error", f"{type(e).__name__}: {e}"
        print(f"[probe] {err}", flush=True)

    # --- collect ----------------------------------------------------------------------
    cands = ([os.path.join(out_dir, a.expect_run_id + ".json")] if a.expect_run_id
             else sorted(glob.glob(os.path.join(out_dir, "*.json"))))
    res: dict = {}
    for p in cands:
        if os.path.exists(p):
            with open(p) as f:
                res = json.load(f)
            break
    if not res:
        res = {"status": status if status != "ok" else "missing", "error": err or "no profiler output"}

    t_end = time.time()
    raw = res.get("raw") or {}
    n_raw = len(raw.get("step_ms") or [])
    rec = {
        "schema": "probe_campaign/1",
        **labels,
        "status": res.get("status"),
        "driver_status": status,
        "driver_error": err or None,
        "run_id": res.get("run_id"),
        "config_name": res.get("config_name", a.config_name),
        "method": res.get("method"),
        "params": res.get("params"),
        "n_params": res.get("n_params"),
        "meta": res.get("meta"),
        "error": res.get("error"),
        "step_ms": (res.get("time") or {}).get("step_ms"),
        "wall_ms": (res.get("time") or {}).get("wall_ms"),
        "capture_ms": (res.get("time") or {}).get("capture_ms"),
        "solve_ms": (res.get("time") or {}).get("solve_ms"),
        "memory": res.get("memory"),
        "profile": res.get("profile"),
        "env": res.get("env"),
        "raw_step_ms": (raw.get("step_ms") if n_raw <= MAX_RAW_STEPS else None),
        "n_measured_steps": n_raw,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "t_start": t_start,
        "t_release": state["t_release"],
        "t_end": t_end,
        "barrier_ok": state["barrier_ok"],
        "barrier_wait_s": state["barrier_wait_s"],
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "no_empty_cache": bool(a.no_empty_cache),
        "provenance": prov,
        "removed_meta_path_finders": removed,
    }
    os.makedirs(os.path.dirname(a.jsonl), exist_ok=True)
    with open(a.jsonl, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    t = rec["step_ms"] or {}
    print(f"[probe] DONE {labels.get('tag')} p{labels.get('proc_index')}: {rec['status']}"
          + (f"  median {t['median']:.3f} ms  n={t['n']}" if t else f"  {str(rec.get('error'))[:160]}"),
          flush=True)
    return 0 if rec["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
