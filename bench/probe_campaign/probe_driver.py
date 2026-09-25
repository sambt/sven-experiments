#!/usr/bin/env python3
"""Driver for the campaign execution-layout probe (see campaign/scout/sharding.md §5).

    PROBE_SNAPSHOT=<snap> .venv/bin/python bench/probe_campaign/probe_driver.py \
        --parts a,b,c,d,e --results <dir> --budget-epoch <unix-ts>

Runs the parts in the order given, most valuable first, and writes one JSON line per
measurement as soon as it finishes, so a wall-clock timeout still leaves usable data.
Nothing is written outside ``--results`` (never ``experiment_results``, never
``profile_results_v2``).  Groups whose result lines already exist are skipped, so the
job is restartable.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_plan as P  # noqa: E402

GROUP_TIMEOUT = {"cifar_b": 900, "cifar_co": 1200, "cifar_adam": 420, "mlp": 600, "nanogpt": 600}

ENV_CMDS = [
    ("gpu_list.txt", ["nvidia-smi", "-L"]),
    ("gpu_query.csv", ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,compute_mode,"
                       "persistence_mode,clocks.sm,clocks.max.sm,clocks.applications.graphics,"
                       "power.draw,power.limit,temperature.gpu", "--format=csv"]),
    ("compute_mode.txt", ["nvidia-smi", "-q", "-d", "COMPUTE"]),
    ("clocks_idle.txt", ["nvidia-smi", "-q", "-d", "PERFORMANCE,CLOCK"]),
    ("mps_which.txt", ["bash", "-lc", "which nvidia-cuda-mps-control; "
                       "ls -l /usr/bin/nvidia-cuda-mps-control 2>&1; "
                       "ls -l /usr/local/cuda*/bin/nvidia-cuda-mps-control 2>&1; "
                       "echo '--- GresTypes:'; grep -i '^GresTypes' /etc/slurm/slurm.conf 2>&1"]),
    ("cpus.txt", ["bash", "-lc", "grep Cpus_allowed_list /proc/self/status; nproc; "
                  "echo SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK; "
                  "echo SLURM_JOB_ID=$SLURM_JOB_ID; hostname; free -g | head -2"]),
    # The MLP steps are launch-bound, so host CPU clock and GPU power state matter more
    # than the GPU itself -- record both or the cross-node comparisons are unreadable.
    ("cpu_model.txt", ["bash", "-lc", "lscpu | grep -E 'Model name|MHz|Socket|Core\\(s\\)|Thread'; "
                       "grep -m2 'cpu MHz' /proc/cpuinfo"]),
    ("power_idle.txt", ["nvidia-smi", "-q", "-d", "POWER"]),
]


def log(results: str, msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(results, "progress.log"), "a") as f:
        f.write(line + "\n")


def part_a(results: str, device_class: str) -> None:
    """(a) environment truth: what silicon this actually is, and whether MPS exists."""
    d = os.path.join(results, "env", device_class)
    os.makedirs(d, exist_ok=True)
    for name, cmd in ENV_CMDS:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            txt = out.stdout + ("\n[stderr]\n" + out.stderr if out.stderr.strip() else "")
        except Exception as e:                                   # noqa: BLE001
            txt = f"[failed] {type(e).__name__}: {e}"
        with open(os.path.join(d, name), "w") as f:
            f.write(txt)
        head = " | ".join(txt.strip().splitlines()[:3])
        log(results, f"[a] {name}: {head[:220]}")


def start_clock_sampler(results: str, device_class: str, part: str) -> subprocess.Popen | None:
    """Sample clocks/power/throttle reasons while the GPU is under the probe's load.

    The historical `profile_results_v2` MLP numbers were taken on an --exclusive, almost
    idle A100; if a low power state is what made them 3x slower than a shared GPU, it
    shows up here as a low SM clock, so sample throughout, not just once.
    """
    d = os.path.join(results, "env", device_class)
    os.makedirs(d, exist_ok=True)
    script = (
        f"( sleep 30; nvidia-smi -q -d PERFORMANCE,CLOCK > {d}/clocks_under_load_{part}.txt 2>&1 ) & "
        f"while true; do "
        f"nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.applications.graphics,temperature.gpu,"
        f"power.draw,utilization.gpu,memory.used,clocks_throttle_reasons.active "
        f"--format=csv,noheader >> {d}/clocks_under_load.csv 2>&1; sleep 5; done"
    )
    try:
        return subprocess.Popen(["bash", "-c", script], start_new_session=True)
    except Exception:                                            # noqa: BLE001
        return None


def stop_clock_sampler(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), 15)
    except Exception:                                            # noqa: BLE001
        pass


def read_records(results: str) -> list[dict]:
    recs = []
    for p in sorted(glob.glob(os.path.join(results, "jsonl", "*.jsonl"))):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return recs


def done_counts(results: str) -> dict[tuple[str, str], int]:
    """Lines already on disk, per (device_class, tag) -- both jobs share the results dir."""
    c: dict[tuple[str, str], int] = {}
    for r in read_records(results):
        k = (str(r.get("device_class")), str(r.get("tag")))
        c[k] = c.get(k, 0) + 1
    return c


def run_group(m: P.Measurement, results: str, snapshot: str, venv_python: str,
              device_class: str) -> None:
    stem = f"{device_class}__{m.tag}"
    jsonl_dir = os.path.join(results, "jsonl")
    log_dir = os.path.join(results, "logs")
    bdir = os.path.join(results, "barriers", stem)
    for d in (jsonl_dir, log_dir):
        os.makedirs(d, exist_ok=True)
    shutil.rmtree(bdir, ignore_errors=True)

    base_env = dict(os.environ)
    base_env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                    PYTHONPATH=f"{snapshot}:{os.path.join(snapshot, 'sven')}",
                    PYTHONUNBUFFERED="1", HYDRA_FULL_ERROR="1")
    base_env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if m.alloc_conf:
        base_env["PYTORCH_CUDA_ALLOC_CONF"] = m.alloc_conf

    procs, logs = [], []
    t0 = time.time()
    for i in range(m.nproc):
        out_root = os.path.join(results, "profiler", stem, f"p{i}")
        hydra_dir = os.path.join(results, "hydra", stem, f"p{i}")
        labels = {"part": m.part, "tag": m.tag, "workload": m.workload, "nproc": m.nproc,
                  "proc_index": i, "layout": m.layout, "capture": m.capture,
                  "steps_key": m.steps, "device_class": device_class, "note": m.note}
        cmd = [venv_python, os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_run.py"),
               "--snapshot", snapshot, "--config-name", P.config_name(m),
               "--out-root", out_root, "--jsonl", os.path.join(jsonl_dir, f"{stem}__p{i}.jsonl"),
               "--labels", json.dumps(labels), "--expect-run-id", P.expected_run_id(m),
               "--barrier-dir", bdir, "--barrier-n", str(m.nproc),
               # well inside GROUP_TIMEOUT: if a sibling dies on import, the survivors
               # must still produce a (flagged) measurement rather than hang.
               "--barrier-timeout", "240"]
        if m.no_empty_cache:
            cmd.append("--no-empty-cache")
        for ha in P.hydra_args(m, out_root, hydra_dir):
            cmd.append(f"--hydra-arg={ha}")   # `=` form: some values start with `--`
        lf = open(os.path.join(log_dir, f"{stem}__p{i}.log"), "w")
        logs.append(lf)
        procs.append(subprocess.Popen(cmd, env=base_env, stdout=lf, stderr=subprocess.STDOUT,
                                      cwd=snapshot))

    timeout = GROUP_TIMEOUT.get(m.steps, 900)
    deadline = t0 + timeout
    while time.time() < deadline and any(p.poll() is None for p in procs):
        time.sleep(2)
    killed = 0
    for p in procs:
        if p.poll() is None:
            killed += 1
            p.terminate()
    if killed:
        time.sleep(10)
        for p in procs:
            if p.poll() is None:
                p.kill()
    for p in procs:
        try:
            p.wait(timeout=30)
        except Exception:                                        # noqa: BLE001
            pass
    for lf in logs:
        lf.close()
    rcs = [p.returncode for p in procs]
    log(results, f"[{m.part}] {m.tag} ({m.layout}) done in {time.time() - t0:.0f}s "
                 f"rc={rcs}" + (f" KILLED={killed} (timeout {timeout}s)" if killed else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="a,b,c,d,e")
    ap.add_argument("--results", required=True)
    ap.add_argument("--snapshot", default=os.environ.get("PROBE_SNAPSHOT", ""))
    ap.add_argument("--venv-python", default="/n/home11/sambt/iaifi/sv3/.venv/bin/python")
    ap.add_argument("--device-class", default="a100_80gb")
    ap.add_argument("--budget-epoch", type=float, default=0.0,
                    help="stop starting new groups after this unix timestamp")
    a = ap.parse_args()
    if not a.snapshot:
        raise SystemExit("--snapshot / PROBE_SNAPSHOT is required")
    os.makedirs(a.results, exist_ok=True)
    log(a.results, f"driver start parts={a.parts} snapshot={a.snapshot} "
                   f"device_class={a.device_class} job={os.environ.get('SLURM_JOB_ID')}")

    for part in [p.strip() for p in a.parts.split(",") if p.strip()]:
        if a.budget_epoch and time.time() > a.budget_epoch:
            log(a.results, f"BUDGET EXHAUSTED before part {part}; stopping")
            break
        if part == "a":
            part_a(a.results, a.device_class)
            continue
        best = (P.choose_best_b([r for r in read_records(a.results)
                                 if r.get("device_class") == a.device_class])
                if part == "c" else None)
        if best:
            log(a.results, f"[c] best mode from (b): {best}")
            with open(os.path.join(a.results, "best_b.json"), "w") as f:
                json.dump(best, f, indent=2)
        ms = P.plan_part(part, best)
        done = done_counts(a.results)
        sampler = start_clock_sampler(a.results, a.device_class, part)
        try:
            for m in ms:
                if a.budget_epoch and time.time() > a.budget_epoch:
                    log(a.results, f"BUDGET EXHAUSTED; skipping {m.tag} and the rest")
                    break
                have = done.get((a.device_class, m.tag), 0)
                if have >= m.nproc:
                    log(a.results, f"[{m.part}] {m.tag}: already have {have} lines, skip")
                    continue
                run_group(m, a.results, a.snapshot, a.venv_python, a.device_class)
        finally:
            stop_clock_sampler(sampler)
    log(a.results, "driver done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
