#!/usr/bin/env python3
"""Print the campaign execution-layout decision table from the probe's JSON lines.

    .venv/bin/python bench/probe_campaign/analyse.py
    .venv/bin/python bench/probe_campaign/analyse.py --results <other dir>

Reads every ``*.jsonl`` under ``{results}/jsonl/`` (one line per measured process) and
prints, in order: the environment, the CIFAR capture/allocator/empty_cache table from
part (b), the co-tenancy tables from parts (c)/(d)/(e), the MIG-vs-A100 equivalence
factors, and the recommended layout.  Stdlib only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st

DEFAULT_RESULTS = "/n/labstore01/LABS/anon_lab/Users/anon/sv3_campaign_scratch/probe_results"
MB = 1e6


# ---------------------------------------------------------------------------
def load(results: str) -> list[dict]:
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


def table(headers: list[str], rows: list[list], title: str = "") -> str:
    rows = [[("" if c is None else str(c)) for c in r] for r in rows]
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    out = []
    if title:
        out += ["", title, "=" * len(title)]
    out.append("  ".join(h.ljust(w[i]) for i, h in enumerate(headers)))
    out.append("  ".join("-" * w[i] for i in range(len(headers))))
    for r in rows:
        out.append("  ".join(r[i].ljust(w[i]) for i in range(len(headers))))
    return "\n".join(out)


def g(r: dict, *path, default=None):
    cur = r
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def num(x, fmt="{:.1f}"):
    return fmt.format(x) if isinstance(x, (int, float)) else ""


def groups(recs: list[dict]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in recs:
        out.setdefault((r.get("device_class"), r.get("tag")), []).append(r)
    return out


def group_stats(rs: list[dict]) -> dict:
    ok = [r for r in rs if r.get("status") == "ok" and g(r, "step_ms", "median")]
    med = [g(r, "step_ms", "median") for r in ok]
    d = {"n_proc_launched": rs[0].get("nproc"), "n_lines": len(rs), "n_ok": len(ok),
         "statuses": ",".join(sorted({str(r.get("status")) for r in rs})),
         "workload": rs[0].get("workload"), "part": rs[0].get("part"),
         "layout": rs[0].get("layout"), "capture": rs[0].get("capture"),
         "no_empty_cache": rs[0].get("no_empty_cache"), "alloc_conf": rs[0].get("alloc_conf"),
         "errors": [str(r.get("error"))[:110] for r in rs if r.get("status") != "ok"]}
    wal = [g(r, "wall_ms", "median") for r in ok if g(r, "wall_ms", "median")]
    if ok:
        d.update(
            event_ms=st.median(med), min_ms=min(med), max_ms=max(med),
            mean_ms=st.mean([g(r, "step_ms", "mean") for r in ok]),
            p10_ms=st.mean([g(r, "step_ms", "p10") for r in ok]),
            p90_ms=st.mean([g(r, "step_ms", "p90") for r in ok]),
            steady_ms=st.mean([g(r, "step_ms", "steady_mean") for r in ok]),
            # CUDA-event time only counts a process's kernels while the GPU is actually
            # running them: under time-slicing the wait before e0 executes is invisible,
            # so `step_ms` can FALL as NPROC rises (measured: mnist_adam 1.01 -> 0.22 ms
            # from NPROC 1 to 6 while wall time rose 1.13 -> 1.87). Throughput decisions
            # must use wall time; `event_bias` = wall/event exposes the artifact.
            median_ms=(st.median(wal) if wal else st.median(med)),
            wall_ms=(st.median(wal) if wal else None),
            event_bias=((st.median(wal) / st.median(med)) if (wal and st.median(med)) else None),
            capture_ms=st.mean([g(r, "capture_ms", "median", default=0) for r in ok]) or None,
            solve_ms=st.mean([g(r, "solve_ms", "median", default=0) for r in ok]) or None,
            peak_alloc_mb=max(g(r, "memory", "peak_alloc_bytes_max", default=0) for r in ok) / MB,
            peak_res_mb=max(g(r, "memory", "peak_reserved_bytes_max", default=0) for r in ok) / MB,
            steps=min(r.get("n_measured_steps") or 0 for r in ok),
            barrier_ok=all(r.get("barrier_ok", True) for r in ok),
            overlap_s=_overlap(ok),
        )
    return d


def _overlap(rs: list[dict]) -> float | None:
    """Seconds during which every process of the group was past the barrier."""
    rel = [r["t_release"] for r in rs if r.get("t_release") is not None]
    end = [r["t_end"] for r in rs if r.get("t_end") is not None]
    if len(rel) != len(rs) or len(end) != len(rs):
        return None
    return max(0.0, min(end) - max(rel))


# ---------------------------------------------------------------------------
def section_env(recs: list[dict], results: str) -> str:
    out = []
    for envdir in sorted(glob.glob(os.path.join(results, "env", "*"))):
        out.append(f"[{os.path.basename(envdir)}]")
        p = os.path.join(envdir, "gpu_list.txt")
        if os.path.exists(p):
            out.append(open(p).read().strip())
        for name, label in (("compute_mode.txt", "compute mode"), ("mps_which.txt", "MPS / GresTypes")):
            q = os.path.join(envdir, name)
            if not os.path.exists(q):
                continue
            txt = open(q).read()
            if name == "compute_mode.txt":
                hits = [ln.strip() for ln in txt.splitlines() if "Compute Mode" in ln]
                out.append(f"{label}: {'; '.join(hits) or 'n/a'}")
            else:
                out.append(f"{label}: " + " | ".join(ln.strip() for ln in txt.splitlines() if ln.strip())[:300])
        q = os.path.join(envdir, "clocks_under_load.csv")
        if os.path.exists(q):
            rows = [ln.split(",") for ln in open(q).read().splitlines() if ln.strip()]
            sm = [float(r[1].split()[0]) for r in rows
                  if len(r) > 1 and r[1].split()[:1] and r[1].split()[0].isdigit()]
            thr = sorted({r[-1].strip() for r in rows if len(r) > 7})
            if sm:
                out.append(f"SM clock under load: min {min(sm):.0f} / median {st.median(sm):.0f} / "
                           f"max {max(sm):.0f} MHz over {len(sm)} samples; throttle reasons {thr}")
    gpus = sorted({f"{g(r, 'env', 'gpu')} ({g(r, 'env', 'gpu_total_bytes', default=0) / 1e9:.0f} GB)"
                   for r in recs if g(r, "env", "gpu")})
    if gpus:
        out.append("GPUs seen by the measurements: " + "; ".join(gpus))
    hosts = sorted({str(r.get("host")) for r in recs})
    jobs = sorted({str(r.get("slurm_job_id")) for r in recs})
    out.append(f"hosts {hosts}  slurm jobs {jobs}")
    bad = [r.get("tag") for r in recs
           if not str(g(r, "provenance", "sven", default="")).startswith("/n/labstore01")]
    out.append("snapshot provenance: OK for every measurement" if not bad
               else f"snapshot provenance VIOLATED for {sorted(set(bad))}")
    return "\n".join("  " + ln for ln in "\n".join(out).splitlines())


def section_b(gs: dict) -> str:
    rows = []
    for (dev, tag), d in sorted(gs.items()):
        if d.get("part") != "b":
            continue
        rows.append([dev, d["capture"], "off" if d["no_empty_cache"] else "on",
                     "expandable" if d["alloc_conf"] else "default", d["statuses"],
                     d.get("steps"), num(d.get("wall_ms")), num(d.get("event_ms")),
                     num(d.get("mean_ms")),
                     num(d.get("p10_ms")), num(d.get("p90_ms")),
                     num(d.get("capture_ms")), num(d.get("solve_ms")),
                     num(d.get("peak_alloc_mb"), "{:.0f}"), num(d.get("peak_res_mb"), "{:.0f}")])
    rows.sort(key=lambda r: (r[1], r[2], r[3]))
    return table(["device", "capture", "empty_cache", "alloc_conf", "status", "steps",
                  "wall_med_ms", "event_med_ms", "event_mean_ms", "event_p10", "event_p90",
                  "capt_ms", "solve_ms", "peak_alloc_MB", "peak_resvd_MB"], rows,
                 "(b) CIFAR ResNet18 Sven, B=128, batch-stat BN - one process")


def section_cotenancy(gs: dict, parts=("c", "d", "e", "cifarslice")) -> tuple[str, dict]:
    """Per-process step time and aggregate throughput vs NPROC, per workload+device."""
    bywl: dict[tuple, dict[int, dict]] = {}
    for (dev, tag), d in gs.items():
        if d.get("part") not in parts:
            continue
        bywl.setdefault((dev, d["workload"], d["capture"] or ""), {})[d["n_proc_launched"]] = d
    rows = []
    best: dict[tuple, tuple[int, float]] = {}
    for key in sorted(bywl):
        dev, wl, cap = key
        per = bywl[key]
        base = per.get(1, {}).get("median_ms")
        for n in sorted(per):
            d = per[n]
            infl = (d.get("median_ms") / base) if (base and d.get("median_ms")) else None
            # Count the processes that actually RAN, not the ones launched: in the
            # measured `c__sven-full__n3` group one of three OOMed, and crediting the
            # survivors' step time to three processes reported T=1.45 where the real
            # answer (two processes, 2.04x inflation) is T=0.98.
            n_ran = d.get("n_ok") or n
            thr = (n_ran / infl) if infl else None
            if thr is not None:
                cur = best.get(key)
                if cur is None or thr > cur[1]:
                    best[key] = (n, thr)
            flag = "" if d.get("barrier_ok", True) else "BARRIER-TIMEOUT"
            if (d.get("event_bias") or 1) > 1.5:
                flag = (flag + " EVENT-TIME-UNUSABLE").strip()
            rows.append([dev, wl + (f"/{cap}" if cap else ""), n, d["statuses"],
                         f"{d['n_ok']}/{d['n_lines']}", d.get("steps"),
                         num(d.get("median_ms"), "{:.2f}"), num(d.get("event_ms"), "{:.2f}"),
                         num(d.get("event_bias"), "{:.1f}"),
                         num(infl, "{:.2f}"), num(thr, "{:.2f}"),
                         num(d.get("peak_alloc_mb"), "{:.0f}"), num(d.get("peak_res_mb"), "{:.0f}"),
                         num(d.get("overlap_s"), "{:.0f}"), flag])
    return table(["device", "workload", "nproc", "status", "ok", "steps", "wall_ms",
                  "event_ms", "wall/event", "inflation", "aggregate_T", "peak_alloc_MB",
                  "peak_resvd_MB", "overlap_s", "flag"], rows,
                 "(c)/(d)/(e) co-tenancy: per-process WALL step time and aggregate throughput "
                 "(T = nproc / inflation; inflation from wall time -- see event_bias)"), best


def section_mig(gs: dict) -> str:
    cand: dict[tuple, tuple[int, float]] = {}
    for (dev, tag), d in gs.items():
        if d.get("n_proc_launched") != 1 or not d.get("median_ms"):
            continue
        # part (b) at the stock axes is the fall-back single-process A100 reference for a
        # CIFAR capture that part (c) did not happen to pick.
        if d.get("part") == "b" and (d.get("no_empty_cache") or d.get("alloc_conf")):
            continue
        rank = 1 if d.get("part") == "b" else 0
        k = (d["workload"], dev)
        if k not in cand or rank < cand[k][0]:
            cand[k] = (rank, d["median_ms"])
    one = {k: v for k, (_r, v) in cand.items()}
    wls = sorted({w for w, _ in one})
    rows = []
    for w in wls:
        a = next((v for (ww, dd), v in one.items() if ww == w and "a100" in str(dd)), None)
        m = next((v for (ww, dd), v in one.items() if ww == w and "mig" in str(dd)), None)
        rows.append([w, num(a, "{:.2f}"), num(m, "{:.2f}"),
                     num(m / a, "{:.2f}") if (a and m) else "",
                     num(a / m, "{:.2f}") if (a and m) else ""])
    return table(["workload", "A100_ms", "MIG_ms", "MIG/A100 slowdown", "A100-equivalents per slice"],
                 rows, "MIG 3g.20gb vs full A100-80GB, single process")


def section_reco(gs: dict, best: dict, recs: list[dict]) -> str:
    out = ["", "RECOMMENDATIONS", "==============="]
    bs = [d for (dev, tag), d in gs.items() if d.get("part") == "b" and d.get("median_ms")]
    if bs:
        w = min(bs, key=lambda d: d["median_ms"])
        out.append(f"CIFAR capture mode: {w['capture']}, empty_cache "
                   f"{'off' if w['no_empty_cache'] else 'on'}, "
                   f"PYTORCH_CUDA_ALLOC_CONF {w['alloc_conf'] or 'default'} "
                   f"-> {w['median_ms']:.0f} ms/step, peak alloc {w['peak_alloc_mb']:.0f} MB, "
                   f"peak reserved {w['peak_res_mb']:.0f} MB")
        for axis, label in (("no_empty_cache", "empty_cache off"), ("alloc_conf", "expandable_segments")):
            on = [d["median_ms"] for d in bs if d[axis]]
            off = [d["median_ms"] for d in bs if not d[axis]]
            if on and off:
                out.append(f"  effect of {label}: {st.mean(on) / st.mean(off):.3f}x median step time "
                           f"(mean over the other axes)")
        rv_on = [d["peak_res_mb"] for d in bs if d["alloc_conf"]]
        rv_off = [d["peak_res_mb"] for d in bs if not d["alloc_conf"]]
        if rv_on and rv_off:
            out.append(f"  effect of expandable_segments on peak RESERVED: "
                       f"{st.mean(rv_on):.0f} vs {st.mean(rv_off):.0f} MB "
                       f"({st.mean(rv_on) / st.mean(rv_off):.2f}x)")
    if best:
        out.append("")
        out.append("NPROC with the highest aggregate throughput (per workload x device):")
        for (dev, wl, cap), (n, t) in sorted(best.items()):
            out.append(f"  {dev:12s} {wl + ('/' + cap if cap else ''):26s} NPROC* = {n:2d}  "
                       f"aggregate T = {t:.2f} standalone-runs-worth per GPU")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    a = ap.parse_args()
    recs = load(a.results)
    if not recs:
        print(f"no measurements under {a.results}/jsonl/ yet")
        return 1
    gs = {k: group_stats(v) for k, v in groups(recs).items()}
    print(f"{len(recs)} measured processes in {len(gs)} groups under {a.results}")
    print("\nENVIRONMENT\n===========")
    print(section_env(recs, a.results))
    print(section_b(gs))
    cot, best = section_cotenancy(gs)
    print(cot)
    print(section_mig(gs))
    print(section_reco(gs, best, recs))
    fails = [(k[1], d["statuses"], d["errors"][:1]) for k, d in sorted(gs.items())
             if d["n_ok"] < (d["n_lines"] or 0) or d["n_lines"] < (d["n_proc_launched"] or 0)]
    if fails:
        print("\nNON-OK GROUPS\n=============")
        for tag, s, e in fails:
            print(f"  {tag}: {s} {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
