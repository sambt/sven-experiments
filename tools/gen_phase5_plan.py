#!/usr/bin/env python3
"""Generate `campaign/plan_phase5.yaml` from `bench/best_configs.json` (Phase 5, step E).

    tools/select_best.py                      # (re)derive the selection
    tools/gen_phase5_plan.py                  # -> campaign/plan_phase5.yaml
    tools/launch_campaign.py campaign/plan_phase5.yaml --phase P5 --list p5_diag_mlp_mig

Nothing in Phase 5 is hand-written: a grid-extension round is `tools/select_best.py` plus
this script, and every work item's overrides come out of the selection file unchanged. That
is the only way the timing numbers, the spectrum figures and the confirmation table can be
guaranteed to describe the SAME configuration.

Three passes, one work list per (pass, lane):

* **timing** (C-T1/C-T3): every selected configuration alone on a GPU (NPROC 1) with
  `svd_info: none` and `checkpoints: none`, into `{scan}_timing/` under the scan's own
  run_ids. The list exists here so `tools/reconcile.py --all` can track the pass and the
  launcher can dry-run it, but the pass is SUBMITTED by `bench/submit_timing_phase5.sh`:
  the GPU probe measured an MLP step time to be set by HOST CPU load (1.13 ms on a quiet
  node, 4.58 ms on a node at 4/4 GPUs), so a claim-queue pool -- even one on the other
  GPUs of the same node -- corrupts the numbers. One job per scan, serial over methods.
* **diag**: every selected configuration with the full checkpoint ladder and the dense
  spectra schedule, into `{scan}_diag/` (`experiments/configs/{scan}_diag.yaml`).
* **confirm**: every selected configuration on 5 fresh model seeds, into
  `{scan}_confirm/` (`{scan}_confirm.yaml`), and on 3 DATA seeds where the scan has a
  `data_seed` (toy, polynomial) -- one item per data seed, because `data_seed` is a scalar
  config key, not a grid axis.

Lanes follow the scope update and the GPU probe: MLP diag/confirm go to the `mig` lane with
an A100 overflow list running the identical items, CIFAR and nanoGPT to `a100` only (a MIG
slice is 19.6 GB and CIFAR full capture wants 23.3 GB).

Torch-free; hydra only to read each scan's `data_seed`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import reconcile                                        # noqa: E402

DEFAULT_IN = os.path.join(REPO, "bench", "best_configs.json")
DEFAULT_OUT = os.path.join(REPO, "campaign", "plan_phase5.yaml")
DEFAULT_RESULTS_ROOT = "/n/labstore01/LABS/anon_lab/Users/anon/sven_experiments"

#: the hydra overrides that turn a scan config into a TIMING run (C-T1). `++` and not `=`:
#: no scan config declares `svd_info`, and the MNIST ones declare `checkpoints_svd` but not
#: `checkpoints`, so a plain `=` override is rejected by hydra's struct mode -- exactly the
#: failure `worker_pool.sh` documents for `scheduler=claims`. `checkpoints_svd` must be
#: cleared as well or the svd family would keep following the parent's `log`.
#: None of the three is part of `grid.run_hash` (they decide what is WRITTEN, not what is
#: computed), so a timing run carries the scan's run_id AND the scan's run hash.
TIMING_OVERRIDES = "++svd_info=none ++checkpoints=none ++checkpoints_svd=null"

#: scan -> which cost class it belongs to (the GPU probe's three model families)
def scan_class(scan):
    if scan.startswith("cifar10"):
        return "cifar"
    if "nanogpt" in scan:
        return "nanogpt"
    return "mlp"


#: the second-order baselines whose state makes them a `heavy` item
HEAVY_STANDARD = ("SOAP", "Shampoo", "KFAC")


def nproc_class(selection):
    """`sven` | `first` | `heavy` for one selected method (the probe's NPROC classes)."""
    family = selection["family"]
    if family == "svd":
        return "sven"
    if family in ("lbfgs", "jd", "hig"):
        return "heavy"
    if family == "standard" and selection["method"] in HEAVY_STANDARD:
        return "heavy"
    return "first"                      # first-order standard optimizers, PolyakSGD


#: NPROC per (scan class, nproc class, lane), straight from CONTRACTS.md "Stage 1
#: contracts" / campaign/stage0_reports/gpu.probe.md. MIG values for CIFAR and nanoGPT are
#: there only to keep the mapping total: no CIFAR or nanoGPT item is ever put in that lane.
NPROC = {
    ("mlp", "sven"):      {"a100": 12, "mig": 6},
    ("mlp", "first"):     {"a100": 12, "mig": 6},
    ("mlp", "heavy"):     {"a100": 4,  "mig": 6},
    ("cifar", "sven"):    {"a100": 1,  "mig": 1},
    ("cifar", "first"):   {"a100": 4,  "mig": 1},
    ("cifar", "heavy"):   {"a100": 2,  "mig": 1},
    ("nanogpt", "sven"):  {"a100": 3,  "mig": 1},
    ("nanogpt", "first"): {"a100": 3,  "mig": 1},
    ("nanogpt", "heavy"): {"a100": 3,  "mig": 1},
}


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

class Item:
    """One work item, plus what the cost summary needs."""

    def __init__(self, config, overrides, nproc, note, *, n_runs, seconds, lane_key):
        self.config, self.overrides, self.nproc, self.note = config, overrides, nproc, note
        self.n_runs, self.seconds, self.lane_key = n_runs, seconds, lane_key

    def yaml(self, indent=6):
        pad = " " * indent
        nproc = (self.nproc if isinstance(self.nproc, int)
                 else "{" + ", ".join(f"{k}: {v}" for k, v in sorted(self.nproc.items())) + "}")
        out = [f"{pad}- config: {self.config}",
               f"{pad}  overrides: \"{self.overrides}\"",
               f"{pad}  nproc: {nproc}"]
        if self.note:
            out.append(f"{pad}  note: \"{self.note}\"")
        return "\n".join(out)


def data_seeds_of(scan, loader, n_data_seeds):
    """The data seeds a confirmation pass uses for `scan`, or `[]`.

    Only toy and polynomial have a `data_seed` (it feeds `dataset.seed` there, F25/F27);
    the value is read from the scan's own `_confirm` config rather than hardcoded, so a
    config edit moves the replicates with it. The `_confirm` config is also where
    `data_seed` joins `result_id_fields` -- without that the replicates would share a
    run_id under different run hashes and re-run each other for ever (C-R3).
    """
    try:
        cfg = loader.compose(f"{scan}_confirm", "")
    except Exception:
        return []
    base = cfg.get("data_seed")
    if base is None:
        return []
    if "data_seed" not in (cfg.get("result_id_fields") or []):
        raise SystemExit(
            f"[gen] ERROR: {scan}_confirm.yaml has data_seed={base} but does not list "
            f"`data_seed` in result_id_fields; its replicates would all write the same "
            f"run_id (C-R3 would then have them re-run each other for ever)")
    return [int(base) + i for i in range(n_data_seeds)]


def build_items(payload, loader, *, passes, n_data_seeds):
    """`{pass: {lane_key: [Item]}}` over every (scan, method) in the selection."""
    out = {p: {} for p in passes}
    for scan, entry in payload["scans"].items():
        klass = scan_class(scan)
        lane_key = "mlp" if klass == "mlp" else "heavy"
        seeds = data_seeds_of(scan, loader, n_data_seeds) if "confirm" in passes else []
        for method, sel in sorted(entry["methods"].items()):
            cls = nproc_class(sel)
            nproc = NPROC[(klass, cls)]
            secs = sel.get("mean_wall_time_s") or 0.0
            n_seeds = len(sel["model_seeds"])
            label = sel["label"]
            if "timing" in passes:
                out["timing"].setdefault("timing", []).append(Item(
                    f"{scan}_timing", f"{TIMING_OVERRIDES} {sel['overrides']}", 1,
                    f"{label}: seed-mean val {sel['seed_mean_final_val']:.6g}, "
                    f"~{secs / 60:.1f} min/run sharded",
                    n_runs=n_seeds, seconds=secs * n_seeds, lane_key="timing"))
            if "diag" in passes:
                out["diag"].setdefault(lane_key, []).append(Item(
                    f"{scan}_diag", sel["overrides"], nproc,
                    f"{label} ({cls}); checkpoints + dense spectra",
                    n_runs=n_seeds, seconds=secs * n_seeds, lane_key=lane_key))
            if "confirm" in passes:
                for ds in (seeds or [None]):
                    extra = "" if ds is None else f" data_seed={ds}"
                    out["confirm"].setdefault(lane_key, []).append(Item(
                        f"{scan}_confirm", sel["overrides"] + extra, nproc,
                        f"{label} ({cls}); 5 fresh model seeds"
                        + ("" if ds is None else f", data seed {ds}"),
                        n_runs=n_seeds, seconds=secs * n_seeds, lane_key=lane_key))
    return out


# ---------------------------------------------------------------------------
# The plan file
# ---------------------------------------------------------------------------

HEADER = """\
# Phase 5: the three result-dependent passes, GENERATED -- do not hand-edit.
#
#   tools/select_best.py                       # re-derive bench/best_configs.json
#   tools/gen_phase5_plan.py                   # regenerate THIS file
#   tools/launch_campaign.py campaign/plan_phase5.yaml --phase P5              # dry run
#   tools/launch_campaign.py campaign/plan_phase5.yaml --list p5_diag_mlp_mig --submit
#   tools/reconcile.py --all campaign/plan_phase5.yaml --phase P5
#
# Every item's overrides are `bench/best_configs.json`'s `overrides` field verbatim, and
# `tools/select_best.py` has already PROVEN each of them expands to exactly the five seed
# run_ids (and run hashes) of the configuration it names. After a grid extension: re-run
# the two tools above, in that order, and relaunch.
#
# Generated {when}
# from      {src}
#   selection rule: {rule}
#   results root:   {root}
#   configs:        {cfgdir}
#   selection made: {made}
{status}#
# THE TIMING LIST IS NOT SUBMITTED BY THE LAUNCHER. It is here so `reconcile --all` covers
# the pass and so the launcher can dry-run its run counts. Submit it with
# `bench/submit_timing_phase5.sh` (dry run by default): the GPU probe measured an MLP step
# time to be set by HOST CPU load -- 1.13 ms on a quiet node against 4.58 ms on a node at
# 4/4 GPUs, the same config -- so a worker pool on the other GPUs of the same node
# corrupts the numbers even though it never touches this GPU. One job per scan, all of the
# scan's methods back to back, with an Adam-MLP calibration microbenchmark at the start and
# the end of the job so contamination is detectable after the fact.
version: 1
plan: phase5
results_root: {root}

lanes:
  a100:
    partition: "lab_gpu_priority,lab_gpu,gpu"
    gres: "gpu:1"
    mem: 64G
    time: "24:00:00"
    cpus_extra: 2
    note: "diag / confirm for CIFAR + nanoGPT, and the A100 overflow of the MLP lists"
  mig:
    partition: gpu_test
    gres: "gpu:4"
    cpus: 32
    mem: 128G
    time: "12:00:00"
    max_jobs: 2
    note: "MLP diag/confirm only: 19.6 GB a slice fits toy/poly/MNIST, not CIFAR full capture"
  timing:
    partition: "lab_gpu_priority,lab_gpu,gpu"
    gres: "gpu:1"
    cpus: 16
    mem: 64G
    time: "36:00:00"
    note: "reference shape only -- bench/submit_timing_phase5.sh submits this pass serially,
           optionally with --exclusive (an exclusive job holds a whole node slot of its QOS
           and cannot start while a co-tenant holds a GPU on it: the 2026-09-17 timing job
           sat in QOSMaxNodePerUserLimit for 13 h)"
"""

#: (list name, pass, lane, n_jobs, lane_key, note)
LISTS = (
    ("p5_timing", "timing", "timing", 1, "timing",
     "C-T1/C-T3: one selected configuration at a time, NPROC 1, svd_info none, "
     "checkpoints none. SUBMIT WITH bench/submit_timing_phase5.sh, not the launcher."),
    ("p5_diag_mlp_mig", "diag", "mig", 2, "mlp",
     "toy / polynomial / MNIST diagnostics on the 8 MIG slices"),
    ("p5_diag_mlp_a100", "diag", "a100", 2, "mlp",
     "the SAME items at A100 NPROC; absorbs whatever the MIG slices have not finished "
     "(the claim queue makes that free of duplicated work)"),
    ("p5_diag_heavy_a100", "diag", "a100", 2, "heavy",
     "CIFAR + nanoGPT diagnostics; CIFAR writes ~0.95 GB of checkpoints a run"),
    ("p5_confirm_mlp_mig", "confirm", "mig", 2, "mlp",
     "toy / polynomial / MNIST confirmation seeds; toy and polynomial also on 3 data seeds"),
    ("p5_confirm_mlp_a100", "confirm", "a100", 2, "mlp",
     "the SAME items at A100 NPROC"),
    ("p5_confirm_heavy_a100", "confirm", "a100", 2, "heavy",
     "CIFAR + nanoGPT confirmation seeds"),
)


def render(payload, items, *, src, root, status_lines):
    when = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    status = "".join(f"# {l}\n" for l in status_lines)
    out = [HEADER.format(when=when, src=src, root=root, rule=payload.get("rule_name", "?"),
                         cfgdir=payload.get("config_dir", "?"),
                         made=payload.get("generated_at", "?"), status=status),
           "work_lists:"]
    for name, which, lane, n_jobs, lane_key, note in LISTS:
        got = items.get(which, {}).get(lane_key, [])
        if not got:
            continue
        gpu_h = sum(i.seconds for i in got) / 3600.0
        runs = sum(i.n_runs for i in got)
        out += [f"  - name: {name}",
                f"    phase: P5",
                f"    lane: {lane}",
                f"    n_jobs: {n_jobs}",
                f"    note: \"{note}\"",
                f"    # {len(got)} item(s), {runs} run(s); "
                f"~{gpu_h:.1f} process-h at the scans' own (sharded) step times",
                f"    items:"]
        out += [i.yaml() for i in got]
        out.append("")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate campaign/plan_phase5.yaml.")
    ap.add_argument("--in", dest="src", default=DEFAULT_IN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--results-root", default=None,
                    help="default: the selection file's results_root")
    ap.add_argument("--config-dir", default=None, help="hydra config dir (for data_seed)")
    ap.add_argument("--passes", default="timing,diag,confirm")
    ap.add_argument("--data-seeds", type=int, default=3,
                    help="data-seed replicates where the scan has a data_seed (default 3)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="generate even though a scan's grid is unfinished or a selection "
                         "is unverified (the items would then name a configuration chosen "
                         "over half a grid)")
    ap.add_argument("--print", dest="print_only", action="store_true")
    a = ap.parse_args(argv)

    try:
        with open(a.src) as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        reconcile.err(f"[gen] ERROR: cannot read {a.src}: {exc}")
        return 2
    if payload.get("schema") != 2:
        reconcile.err(f"[gen] ERROR: {a.src} is not a schema-2 selection file "
                      f"(run tools/select_best.py; the pre-campaign file is a list of "
                      f"[method, cfg, ...] rows and has no verified overrides)")
        return 2

    passes = [p.strip() for p in a.passes.split(",") if p.strip()]
    # realpath, not abspath: the selection may have resolved the root through the repo's
    # `experiment_results` symlink, and the plan is what the workers get as
    # SV3_RESULTS_ROOT -- a home-directory path there would put O(10^4) result files on a
    # 95 G NFS home instead of on labstore.
    root = os.path.realpath(a.results_root or payload.get("results_root")
                            or DEFAULT_RESULTS_ROOT)

    # --- refuse to generate from a selection that is not trustworthy yet -------
    problems, status = [], []
    for scan, entry in sorted(payload["scans"].items()):
        bad = sorted(m for m, s in entry["methods"].items() if not s.get("verified"))
        if bad:
            problems.append(f"{scan}: unverified overrides for {', '.join(bad)}")
        if entry.get("n_missing"):
            problems.append(f"{scan}: {entry['n_missing']} run(s) of the grid unfinished")
        thin = sorted(f"{m} ({s['n_ok']}/{s['n_expected']})"
                      for m, s in entry["methods"].items()
                      if s["n_ok"] < s["n_expected"])
        if thin:
            problems.append(f"{scan}: selected on fewer than all seeds: {', '.join(thin)}")
    if problems:
        for p in problems:
            reconcile.err(f"[gen] {'WARNING' if a.allow_incomplete else 'ERROR'}: {p}")
        status = ["GENERATED FROM AN INCOMPLETE SELECTION (--allow-incomplete):"] + \
                 [f"  {p}" for p in problems] + \
                 ["Re-run tools/select_best.py and this script once "
                  "`tools/reconcile.py --all campaign/plan_campaign.yaml` is clean."]
        if not a.allow_incomplete:
            reconcile.err("[gen] refusing to generate; pass --allow-incomplete to write "
                          "the plan anyway (it will carry the warnings in its header)")
            return 1

    config_dir, _ = reconcile.resolve_config_dir(a.config_dir, None)
    with reconcile.ConfigLoader(config_dir) as loader:
        items = build_items(payload, loader, passes=passes, n_data_seeds=a.data_seeds)
    text = render(payload, items, src=a.src, root=root, status_lines=status)

    if a.print_only:
        print(text)
    else:
        with open(a.out, "w") as fh:
            fh.write(text)
        print(f"[gen] wrote {a.out}")

    # --- what it costs -------------------------------------------------------
    print(f"[gen] {'pass':9s} {'items':>6s} {'runs':>6s} {'process-h':>11s}")
    for which in passes:
        lanes = items.get(which, {})
        n_items = sum(len(v) for v in lanes.values())
        runs = sum(i.n_runs for v in lanes.values() for i in v)
        hours = sum(i.seconds for v in lanes.values() for i in v) / 3600.0
        print(f"[gen] {which:9s} {n_items:6d} {runs:6d} {hours:11.1f}")
    print("[gen] 'process-h' is the selected runs' own recorded wall time, i.e. what they "
          "cost UNDER the scan's sharding; a timing run alone on a GPU is faster than that "
          "and a diag run with the full ladder is slower.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
