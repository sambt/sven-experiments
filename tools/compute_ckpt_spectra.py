#!/usr/bin/env python
"""Probe-set Jacobian spectra along the trajectories in a ``<scan>_diag`` pass (C-L4).

The ``<scan>_diag`` passes reran the selected configuration of every method with
``checkpoints: log``, so the state before steps {0, 1, 2, 4, 8, ...} and at every
epoch end is on disk for Sven *and* for the baselines.  This walks those
checkpoints and, at each one, takes the float64 Jacobian of the rows Sven
differentiates on a FIXED probe set of the training pool -- identical for every
run, every seed and every optimizer of the scan -- and stores its singular
values, the residual's projections onto the left singular vectors, the
probe-set loss, the parameter norm and the distance from initialisation.

That is the measurement the online per-step spectra cannot make (F19): they are
of a different random batch on every logged step, capped at B values, and exist
only along Sven's trajectory.

Results are cached as one npz per (method, seed) under
``analysis/ckpt_spectra/<scan>_diag/`` (git-ignored, path scheme in
``ckpt_tools.spectra_path``), so the phase-B figure notebooks plot without
recomputing anything -- they read them with ``ckpt_tools.load_spectra``.

Examples
--------
    # toy: the whole 10,000-example pool is 593 columns wide -- use all of it
    tools/compute_ckpt_spectra.py toy_1d_scan --probe full

    # MNIST (P = 27,562): 2,000 rows at step 0 and the epoch ends, one seed
    tools/compute_ckpt_spectra.py mnist_scan_labelRegression \
        --probe 2000 --epochs-only --seeds 3000 --methods Sven,Adam

Anything heavier than a couple of minutes goes through
``campaign/run_cpu_tests.sh`` (CPU float64 is fine at these sizes); the 40-minute
limit of that lane is why ``--methods`` / ``--seeds`` exist.  As a cost guide, on
four cores: a 10,000 x 593 toy trajectory is ~28 s per run (34 checkpoints), a
512-row MNIST one ~45 s, and a 2,000-row MNIST one ~11 min (21 checkpoints) --
the SVD of the 2,000 x 27,562 Jacobian is what dominates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "analysis"))

import ckpt_tools as ct                                              # noqa: E402

#: Sven plus the three baselines the mechanism story needs (C-A3 / Codex Low 4):
#: the strongest first-order method, the strongest Muon variant and the other
#: Gauss-Newton-flavoured baseline.  Methods absent from a scan are skipped.
DEFAULT_METHODS = ("Sven", "Adam", "MuonW", "HIG")

#: ``analysis/ckpt_spectra/`` -- git-ignored, one subdirectory per diag pass.
#: The path scheme lives in ``ckpt_tools`` next to its reader (``load_spectra``).
DEFAULT_OUT_DIR = ct.SPECTRA_DIR


def diag_scan_name(scan: str, results_root=None) -> str:
    """``toy_1d_scan`` -> ``toy_1d_scan_diag`` when that is what exists.

    The spectra come from the diag pass (it is the only one with ``log``
    checkpoints for the baselines), but the plan and the notebooks name the
    headline scan, so both spellings are accepted.
    """
    root = ct.resolve_results_root(results_root)
    if (root / scan).is_dir() and scan.endswith("_diag"):
        return scan
    if (root / f"{scan}_diag").is_dir():
        return f"{scan}_diag"
    if (root / scan).is_dir():
        return scan
    raise FileNotFoundError(f"neither {root / scan} nor {root}/{scan}_diag exists")


def _run_table(scan: str, results_root=None) -> list[dict]:
    """``{run_id, method, model_seed, status, has_ckpt}`` for every record in ``scan``."""
    rows = []
    for run_id in ct.find_runs(scan, results_root):
        record = ct.load_record(scan, run_id, results_root)
        optimizer = str(record.get("optimizer"))
        rows.append({
            "run_id": run_id,
            "record": record,
            "method": "Sven" if optimizer == "SVD" else optimizer,
            "model_seed": record.get("model_seed"),
            "status": record.get("status"),
            "has_ckpt": bool(record.get("ckpt_file")),
        })
    return rows


def compute_one(scan: str, row: dict, args, results_root=None, spec=None) -> dict | None:
    """Compute (or reuse) one run's trajectory spectra; return a summary row."""
    destination = ct.spectra_path(scan, row["method"], row["model_seed"], args.probe,
                                  args.epochs_only, out_dir=args.out_dir)
    if destination.is_file() and not args.force:
        with np.load(destination, allow_pickle=False) as z:
            return _summary(scan, row, z, str(destination), reused=True)
    if args.dry_run:
        print(f"  [dry-run] would write {destination}")
        return None
    run = ct.load_run(scan, row["run_id"], results_root, record=row["record"])
    started = time.perf_counter()
    arrays = ct.checkpoint_spectra(
        run, n_probe=args.probe, epochs_only=args.epochs_only,
        with_utr=not args.no_utr, chunk_size=args.chunk, verbose=args.verbose,
        spec=spec,
    )
    if args.verify:
        # A run that blew up inside epoch 0 keeps its checkpoints but has no
        # end-of-epoch value to check them against (C-R1) -- exactly the L-BFGS /
        # KFAC runs a `--methods all` pass sweeps up.  Its spectra are still
        # computable by step, so record WHY the check is missing and keep going
        # rather than discarding the work that was just done and every run after it.
        try:
            detail = ct.verify_checkpoint(run, detail=True)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            print(f"    [warn] verify_checkpoint failed: {exc}")
            detail = {"rel_error": float("nan"), "abs_error": float("nan"),
                      "recorded": float("nan"), "recomputed": float("nan"),
                      "loss_scale": float("nan"), "rel_error_scale": float("nan"),
                      "epoch": -1}
            arrays["verify_error"] = np.asarray(f"{type(exc).__name__}: {exc}")
        for key in ("rel_error", "abs_error", "recorded", "recomputed", "loss_scale",
                    "rel_error_scale", "epoch"):
            arrays[f"verify_{key}"] = np.asarray(detail[key])
    arrays["elapsed_s"] = np.asarray(time.perf_counter() - started)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".tmp{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)
    return _summary(scan, row, arrays, str(destination), reused=False)


#: A singular value below this fraction of sigma_max is numerical zero even in
#: float64 (the float32 Gram's own floor is ~1e-7 sigma_max, style.FLOAT32_NOISE_FLOOR).
RANK_TOL = 1e-12


def _summary(scan: str, row: dict, arrays, path: str, reused: bool) -> dict:
    svals = np.asarray(arrays["svals"])
    first, last = svals[0], svals[-1]

    def ratio(values):
        return float(values[-1] / values[0]) if values[0] > 0 else float("nan")

    def rank(values):
        return int((values > RANK_TOL * values[0]).sum()) if values[0] > 0 else 0

    return {
        "scan": scan, "method": row["method"], "seed": row["model_seed"],
        "n_ckpt": int(svals.shape[0]), "n_rows": int(np.asarray(arrays["n_rows"])),
        "n_sv": int(svals.shape[1]),
        "sigma_max_init": float(first[0]), "sigma_max_final": float(last[0]),
        "tail_init": ratio(first), "tail_final": ratio(last),
        "rank_init": rank(first), "rank_final": rank(last),
        "probe_loss_init": float(np.asarray(arrays["probe_loss"])[0]),
        "probe_loss_final": float(np.asarray(arrays["probe_loss"])[-1]),
        "dist_init_final": float(np.asarray(arrays["dist_init"])[-1]),
        "verify_rel_error": (float(np.asarray(arrays["verify_rel_error"]))
                             if "verify_rel_error" in arrays else float("nan")),
        "elapsed_s": float(np.asarray(arrays["elapsed_s"])) if "elapsed_s" in arrays else 0.0,
        "path": path, "reused": reused,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scans", nargs="+",
                        help="scan names; `<scan>_diag` is used when it exists")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                        help="comma-separated analysis method names, or `all` "
                             f"(default: {','.join(DEFAULT_METHODS)})")
    parser.add_argument("--seeds", default="all",
                        help="comma-separated model seeds, or `all` (default: all)")
    parser.add_argument("--probe", default="512",
                        help="probe-set size, or `full` for the whole training pool "
                             "(default: 512)")
    parser.add_argument("--epochs-only", action="store_true",
                        help="step 0 and the epoch-end checkpoints only (under `log` the "
                             "saved steps are dense in epoch 0 and dominate the cost)")
    parser.add_argument("--no-utr", action="store_true",
                        help="singular values only: skips the left singular vectors "
                             "and the |u_i . r| projections")
    parser.add_argument("--chunk", type=int, default=64,
                        help="rows per jacrev vmap chunk (default: 64)")
    parser.add_argument("--row-spec-run", default=None,
                        help="run_id whose record defines the Jacobian rows for the whole "
                             "scan (default: the scan's svd_ records, which must agree; "
                             "needed on a scan that sweeps kappa)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--threads", type=int, default=None,
                        help="torch intra-op threads (default: leave alone)")
    parser.add_argument("--force", action="store_true", help="recompute existing npz files")
    parser.add_argument("--dry-run", action="store_true", help="list the work, compute nothing")
    parser.add_argument("--no-verify", dest="verify", action="store_false",
                        help="skip the recorded-vs-recomputed validation-loss check")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="one line per checkpoint")
    args = parser.parse_args(argv)

    if args.threads:
        torch.set_num_threads(int(args.threads))
    args.probe = None if str(args.probe).lower() in ("full", "none", "all") else int(args.probe)
    methods = (None if args.methods.lower() == "all"
               else [m.strip() for m in args.methods.split(",") if m.strip()])
    seeds = (None if args.seeds.lower() == "all"
             else [int(s) for s in args.seeds.split(",") if s.strip()])

    summaries: list[dict] = []
    problems = 0
    for name in args.scans:
        scan = diag_scan_name(name, args.results_root)
        rows = _run_table(scan, args.results_root)
        present = sorted({r["method"] for r in rows})
        wanted = present if methods is None else [m for m in methods if m in present]
        missing = [] if methods is None else [m for m in methods if m not in present]
        print(f"\n=== {scan}: {len(rows)} runs, methods present: {', '.join(present)}")
        if missing:
            print(f"    (not in this scan, skipped: {', '.join(missing)})")
        # ONE row definition for the whole scan (Sven's), so the baselines'
        # spectra are of the same matrix; a scan that sweeps kappa has to be told
        # which one with --row-spec-run.
        try:
            spec = ct.row_spec_for_scan(scan, args.results_root, run_id=args.row_spec_run)
        except LookupError as exc:
            print(f"    [warn] {exc}; falling back to each run's own record")
            spec = None
        except (ValueError, FileNotFoundError) as exc:      # ambiguous, or a bad --row-spec-run
            print(f"    [skip] {exc}")
            problems += 1
            continue
        print(f"    rows: {spec.describe() if spec is not None else 'per-run record'}")
        selected = [r for r in rows
                    if r["method"] in wanted
                    and (seeds is None or int(r["model_seed"]) in seeds)]
        # The cache path is (method, seed, probe): a diag pass holds ONE config
        # per method, but a full scan holds the whole grid, and those runs would
        # all write the same npz -- last one wins, silently, and phase B would
        # plot whichever config finished last.
        seen: dict[tuple, str] = {}
        unique = []
        for row in sorted(selected, key=lambda r: (r["method"], r["model_seed"], r["run_id"])):
            key = (row["method"], row["model_seed"])
            if key in seen:
                print(f"  [skip] {row['run_id']}: {row['method']} mseed{row['model_seed']} "
                      f"is already taken by {seen[key]} -- this scan has several configs "
                      "per method, so they would overwrite one cache file. Narrow it with "
                      "--methods / --seeds, or point at the _diag pass.")
                problems += 1
                continue
            seen[key] = row["run_id"]
            unique.append(row)
        for row in unique:
            label = f"{row['method']} mseed{row['model_seed']}"
            if not row["has_ckpt"]:
                print(f"  [skip] {label}: no checkpoint file "
                      f"(status={row['status']!r})")
                continue
            if row["status"] not in ("ok", None):
                print(f"  [warn] {label}: status={row['status']!r}, computing anyway")
            print(f"  {label} ({row['run_id']})")
            summary = compute_one(scan, row, args, args.results_root, spec=spec)
            if summary is not None:
                summaries.append(summary)
                print(f"    -> {Path(summary['path']).name}: {summary['n_ckpt']} ckpt x "
                      f"{summary['n_sv']} sv of {summary['n_rows']} rows, "
                      f"sigma_max {summary['sigma_max_init']:.3e} -> "
                      f"{summary['sigma_max_final']:.3e}, "
                      f"sigma_min/sigma_max {summary['tail_init']:.2e} -> "
                      f"{summary['tail_final']:.2e}, rank(1e-12) "
                      f"{summary['rank_init']} -> {summary['rank_final']}, "
                      f"verify {summary['verify_rel_error']:.2e}, "
                      f"{summary['elapsed_s']:.0f}s"
                      f"{' [cached]' if summary['reused'] else ''}")

    if summaries:
        print("\n" + "-" * 118)
        print(f"{'scan':<34}{'method':<8}{'seed':>6}{'ckpt':>6}{'rows':>7}"
              f"{'smax_0':>11}{'smax_T':>11}{'tail_0':>10}{'tail_T':>10}"
              f"{'rk_0':>6}{'rk_T':>6}{'loss_T':>11}{'d_init':>9}{'verify':>10}")
        for s in summaries:
            print(f"{s['scan']:<34}{s['method']:<8}{s['seed']:>6}{s['n_ckpt']:>6}"
                  f"{s['n_rows']:>7}{s['sigma_max_init']:>11.3e}{s['sigma_max_final']:>11.3e}"
                  f"{s['tail_init']:>10.2e}{s['tail_final']:>10.2e}"
                  f"{s['rank_init']:>6}{s['rank_final']:>6}"
                  f"{s['probe_loss_final']:>11.3e}{s['dist_init_final']:>9.2f}"
                  f"{s['verify_rel_error']:>10.1e}")
        print(f"\n{len(summaries)} trajectories under {args.out_dir}")
        print(json.dumps({"written": [s["path"] for s in summaries if not s["reused"]]},
                         indent=None)[:2000])
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
