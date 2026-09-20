#!/usr/bin/env python3
"""Did the Phase-5 timing pass re-run the SAME runs as the scan? (the join check)

    bench/check_timing_join.py                       # every scan in the selection
    bench/check_timing_join.py mnist_scan_ce
    bench/check_timing_join.py --json /tmp/join.json

A timing run only means something if it is the scan's run with a stopwatch on it. Three
things have to hold, and each has a way of failing quietly:

1. **Same run_id.** `{scan}_timing/` carries the parent scan's run_ids by construction
   (the `_timing` config inherits the parent and overrides only things outside the
   run_id), so a timing run_id that is not in the scan means the selection named a
   configuration the scan never ran.
2. **Same run hash.** `svd_info`, `checkpoints` and `checkpoints_svd` are deliberately
   outside `grid.run_hash` -- they decide what is WRITTEN, never the trajectory -- so the
   timing run's `run_hash` must equal the scan run's exactly. A difference means something
   that DOES enter the hash moved between the two passes (a dataset or model config edit,
   `num_epochs`, an eval setting), and the two numbers are not of the same experiment.
3. **Same trajectory.** Same seeds means the same initialisation and the same data order,
   so the final validation loss must agree to GPU non-determinism -- not to the last bit,
   but nowhere near a different configuration's value. A large deviation with a matching
   hash points at nondeterministic kernels, not at bookkeeping.

It also reports what is MISSING: every (scan, method) of `bench/best_configs.json` whose
five timed runs are not all present, which is how a resumed pass is known to be finished.

Torch-free, stdlib only: one `os.listdir` and one first-line read per record.

Exit status: **0** every timing run joins and nothing is missing · **1** a mismatch or a
missing run · **2** the question could not be answered (no selection file, no results root).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))

import reconcile                                        # noqa: E402  (torch-free)

#: a final validation loss this far apart (relative) is more than GPU non-determinism
DEFAULT_TOL = 1e-3


def read_scan(scan_dir):
    """`{run_id: record}` from one results directory (first line of each jsonl)."""
    try:
        names = [n for n in os.listdir(scan_dir) if n.endswith(".jsonl")]
    except OSError:
        return None
    out, _bad = reconcile.read_records(scan_dir, sorted(n[:-len(".jsonl")] for n in names))
    return out


def check_scan(scan, root, selection, tol):
    """The report dict for one scan (`ok` says the pass joins and is complete)."""
    scan_dir = os.path.join(root, scan)
    timing_dir = os.path.join(root, f"{scan}_timing")
    rep = {"scan": scan, "timing_dir": timing_dir, "problems": [], "notes": [],
           "n_timing": 0, "n_matched": 0, "max_rel_dev": None, "median_rel_dev": None,
           "missing": [], "hash_mismatch": [], "unmatched": [], "ok": False}

    timing = read_scan(timing_dir)
    if timing is None:
        rep["notes"].append(f"no results yet in {timing_dir}")
        return rep
    scan_records = read_scan(scan_dir)
    if scan_records is None:
        rep["problems"].append(f"the parent scan {scan_dir} has no results to join onto")
        return rep
    rep["n_timing"] = len(timing)

    devs = []
    for run_id, trec in sorted(timing.items()):
        srec = scan_records.get(run_id)
        if srec is None:
            rep["unmatched"].append(run_id)
            continue
        rep["n_matched"] += 1
        th, sh = trec.get("run_hash"), srec.get("run_hash")
        if th and sh and th != sh:
            rep["hash_mismatch"].append(f"{run_id}: timing {th[:8]} vs scan {sh[:8]}")
        a = reconcile._final((srec.get("losses") or {}).get("val"))
        b = reconcile._final((trec.get("losses") or {}).get("val"))
        if math.isfinite(a) and math.isfinite(b):
            devs.append((abs(a - b) / max(abs(a), 1e-30), run_id, a, b))
        elif math.isfinite(a) != math.isfinite(b):
            rep["problems"].append(
                f"{run_id}: final val loss is finite in one pass only "
                f"(scan {a!r}, timing {b!r})")

    if devs:
        rep["max_rel_dev"] = max(d[0] for d in devs)
        rep["median_rel_dev"] = statistics.median(d[0] for d in devs)
        over = sorted((d for d in devs if d[0] > tol), reverse=True)[:5]
        for dev, run_id, a, b in over:
            rep["problems"].append(f"{run_id}: final val loss {a:.8g} (scan) vs "
                                   f"{b:.8g} (timing), relative {dev:.2e} > {tol:g}")

    # completeness against the selection: the pass is done when every selected run is here
    entry = (selection or {}).get("scans", {}).get(scan)
    if entry is None:
        rep["notes"].append("not in the selection file: completeness not checked")
    else:
        for method, sel in sorted(entry["methods"].items()):
            absent = [r for r in sel["run_ids"] if r not in timing]
            if absent:
                rep["missing"].append(f"{method}: {len(absent)} of {len(sel['run_ids'])} "
                                      f"not timed (e.g. {absent[0]})")

    for bucket, label in (("unmatched", "timing run_id(s) the scan does not have"),
                          ("hash_mismatch", "run hash mismatch(es)"),
                          ("missing", "selected configuration(s) not fully timed")):
        if rep[bucket]:
            rep["problems"].append(f"{len(rep[bucket])} {label}")
    rep["ok"] = not rep["problems"] and rep["n_timing"] > 0
    return rep


def print_report(rep, out=sys.stdout):
    p = lambda *a: print(*a, file=out)
    p(f"\n=== {rep['scan']} ===")
    p(f"  timing dir {rep['timing_dir']}")
    p(f"  {rep['n_timing']} timing run(s), {rep['n_matched']} joined onto the scan")
    if rep["max_rel_dev"] is not None:
        p(f"  final val loss vs the scan: median {rep['median_rel_dev']:.2e}, "
          f"max {rep['max_rel_dev']:.2e} (relative)")
    for bucket in ("unmatched", "hash_mismatch", "missing"):
        for line in rep[bucket][:8]:
            p(f"  [{bucket}] {line}")
        if len(rep[bucket]) > 8:
            p(f"  [{bucket}] ... {len(rep[bucket]) - 8} more")
    for note in rep["notes"]:
        p(f"  [note] {note}")
    for prob in rep["problems"]:
        p(f"  [PROBLEM] {prob}")
    p(f"  => {'JOINS' if rep['ok'] else 'NOT CLEAN'}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Check that {scan}_timing re-ran the scan's own runs.")
    ap.add_argument("scans", nargs="*", help="default: every scan in the selection file")
    ap.add_argument("--root", default=None, help="results root (default $SV3_RESULTS_ROOT)")
    ap.add_argument("--selection", default=os.path.join(REPO, "bench", "best_configs.json"))
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help=f"relative final-val-loss tolerance (default {DEFAULT_TOL:g})")
    ap.add_argument("--json", default=None, help="write the full report(s) here")
    a = ap.parse_args(argv)

    selection = None
    try:
        with open(a.selection) as fh:
            selection = json.load(fh)
    except (OSError, ValueError) as exc:
        if a.scans:
            print(f"[join] note: no selection file ({exc}); completeness not checked")
        else:
            reconcile.err(f"[join] ERROR: cannot read {a.selection}: {exc}")
            return 2
    if selection is not None and selection.get("schema") != 2:
        reconcile.err(f"[join] ERROR: {a.selection} is not a schema-2 selection file")
        return 2

    root, root_from = reconcile.resolve_root(
        a.root or (selection or {}).get("results_root"), None)
    print(f"[join] results root {root}   ({root_from})")
    if not os.path.isdir(root):
        reconcile.err(f"[join] ERROR: results root {root} does not exist")
        return 2

    scans = a.scans or sorted((selection or {}).get("scans", {}))
    if not scans:
        reconcile.err("[join] ERROR: nothing to check")
        return 2

    reports = [check_scan(s, root, selection, a.tol) for s in scans]
    for rep in reports:
        print_report(rep)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"root": root, "tol": a.tol, "reports": reports}, fh, indent=1,
                      default=str)
        print(f"\n[join] wrote {a.json}")

    bad = [r["scan"] for r in reports if not r["ok"]]
    print(f"\n[join] {len(reports)} scan(s), "
          f"{sum(r['n_matched'] for r in reports)} joined run(s); "
          f"not clean: {', '.join(bad) if bad else 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
