"""A fixed Adam-MLP step-time microbenchmark: is this node quiet enough to time on?

    python bench/calibrate_step.py --tag start
    python bench/calibrate_step.py --tag end --steps 2000

Prints ONE line of JSON prefixed `[calib]`, so a timing job's log carries a measurement of
the machine at the start and at the end of the pass and a contaminated pass can be found
afterwards instead of being believed.

Why this exists: `EXPERIMENTS.md section 1.5` measured the same MNIST-Adam
configuration at **1.13 ms** a step on a quiet node and **4.58 ms** on a node whose four
GPUs were all busy -- a 4x difference in a number this pass exists to report, caused
entirely by HOST CPU load, on a GPU nobody else was using. A launch-bound MLP step is a
CPU measurement wearing a GPU costume. So: nothing of ours on this GPU (NPROC 1), all of a
scan's methods back to back in one job so they share one machine, and this microbenchmark
before and after so `start` vs `end` shows whether the machine changed underneath the pass.

The model and the work are deliberately FIXED (784 -> 32 -> 10 MLP, batch 64, Adam 1e-3,
synthetic data resident on the GPU) and have nothing to do with any scan: the number is
only ever compared against another run of THIS script. Timing is C-T1's --
`torch.cuda.synchronize()` immediately before and after each step -- and the reported
statistic is the median, which a single preemption cannot move.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

import torch


def load_average():
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:                                  # pragma: no cover (not on linux)
        return None


def measure(steps, warmup, device):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 32), torch.nn.ReLU(), torch.nn.Linear(32, 10),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # one resident batch: this measures the step, not the loader
    x = torch.randn(64, 784, device=device)
    y = torch.randint(0, 10, (64,), device=device)
    loss_fn = torch.nn.CrossEntropyLoss()

    def step():
        opt.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        opt.step()

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fixed Adam-MLP step-time calibration.")
    ap.add_argument("--tag", default="calib", help="'start' / 'end' (goes into the line)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available())
                          else "cpu")
    times = measure(a.steps, a.warmup, device)
    times_sorted = sorted(times)
    out = {
        "tag": a.tag,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": (torch.cuda.get_device_name(0) if device.type == "cuda" else None),
        "device": str(device),
        "n_steps": len(times),
        "median_ms": round(statistics.median(times), 4),
        "mean_ms": round(statistics.fmean(times), 4),
        "p05_ms": round(times_sorted[len(times_sorted) // 20], 4),
        "p95_ms": round(times_sorted[min(len(times_sorted) - 1,
                                         19 * len(times_sorted) // 20)], 4),
        "loadavg": load_average(),
        "cpu_count": os.cpu_count(),
        "cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch": torch.__version__,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    print("[calib] " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
