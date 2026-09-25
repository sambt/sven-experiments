#!/usr/bin/env python3
"""What is missing from a scan, and where its grid edges are (C-R2).

    tools/reconcile.py toy_1d_scan
    tools/reconcile.py mnist_scan_ce --config-overrides "mode=svd" \
                                     --config-overrides "mode=standard optimizers_standard=[LBFGS]"
    tools/reconcile.py --all campaign/plan_campaign.yaml --phase P0
    tools/reconcile.py rebuttal_overparam_mnist_scan --config-overrides "mode=svd n_data=50000" --json rep.json

For each scan it answers three questions and nothing else:

1. **What should exist?** The union of the manifest (`{scan}/manifest/*.json`: what the
   jobs declared) and `grid.expand_grid` over every override group given (what the
   config describes today). Either source alone is a half-truth -- the manifest misses a
   scan nobody has launched yet, the grid misses runs whose config has since changed.
2. **What does exist?** ONE `os.listdir` each of `done/`, `started/`, `claims/` and the
   scan directory itself, per the marker contract: `done/{run_id}.{hash8}.{status}` is
   written LAST, `started/{run_id}.started` means "attempted and crashed", a fresh
   `claims/{run_id}.claim*` means "a live worker has it".
3. **Is the answer on an edge?** For every method, the best configuration by seed-mean
   final validation loss (diverged runs excluded, `analysis/lib/style.py` semantics) and
   whether its lr -- k / rtol for Sven, tau for HIG -- sits at the end of its grid, which
   is what C-B3 extension rounds key off. A record of a configuration the grid no longer
   describes never wins (see :func:`best_configs`).

Which configs and which results root: `--config-dir`, else the `--snapshot`'s configs
(`$SV3_DEPLOY_SNAPSHOT` / `$SV3_SNAPSHOT`, which `worker_pool.sh` exports), else this
tree's; `--root`, else the PLAN's `results_root`, else `$SV3_RESULTS_ROOT`, else the
repo's symlink. Both are printed, because reconciling the wrong pair of them is how a
finished scan comes back as "work remains".

Torch-free and hydra-only-for-composition, so it runs in ~2 s on a login node: the
`experiments.experiment_code` package `__init__` imports torch, so `grid.py` is loaded
by path (exactly as `tests/test_grid.py` does).

Exit status: **0** nothing missing · **1** work remains (never-started, crashed,
oom/error, or results under a superseded run hash) · **2** the question could not be
answered (bad config, unreadable plan). A launcher chain must resubmit on 1 only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# --- the on-disk contract (mirrors experiments/experiment_code/claims.py, which cannot
# --- be imported here: its package __init__ pulls torch) -------------------------------
DONE_DIRNAME, STARTED_DIRNAME = "done", "started"
CLAIMS_DIRNAME, MANIFEST_DIRNAME = "claims", "manifest"
STARTED_SUFFIX, CLAIM_SUFFIX = ".started", ".claim"
STATUSES = ("ok", "diverged", "oom", "error")
DONE_STATUSES = frozenset({"ok", "diverged"})
CLAIM_TIMEOUT_S = 600.0          # claims.CLAIM_TIMEOUT_S: older mtime => stale

# --- analysis conventions (copies of analysis/lib/style.py, which imports numpy/pandas) ----
DIVERGED_FACTOR = 10.0           # style.DIVERGED_FACTOR
STATUS_DIVERGED = "diverged"
_SEED_SUFFIX_RE = re.compile(r"_mseed-?\d+(_lseed-?\d+)?")   # style.config_key

#: run_id prefix -> family, for run_ids that are on disk but not in any expected grid
_PREFIX_FAMILY = (("svd_", "svd"), ("std_", "standard"), ("jd_", "jd"), ("hig_", "hig"))

#: the classes every expected run falls into, in report order
CLASSES = ("ok", "diverged", "oom", "error", "started_only", "claimed_live",
           "stale_hash", "jsonl_only", "never_started")
#: classes that mean "the runner still has work to do here"
MISSING_CLASSES = ("oom", "error", "started_only", "stale_hash", "jsonl_only",
                   "never_started")
#: the hyperparameter axes a C-B3 extension round can extend: Sven's k / rtol, HIG's tau
#: (grid_counts.md counts it as an axis of the hig grid) and everybody's lr
AXES = ("lr", "k", "rtol", "tau")


# ---------------------------------------------------------------------------
# style.py conventions, re-implemented in pure python
# ---------------------------------------------------------------------------

def config_key(run_id):
    """A run_id with its `_mseed…_lseed…` suffix removed: a configuration's identity."""
    return _SEED_SUFFIX_RE.sub("", str(run_id))


def _final(curve):
    """style.final_value: the last value of a curve, NaN when it ends non-finite."""
    if not curve:
        return float("nan")
    try:
        v = float(curve[-1])
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def is_diverged(train_curve, val_curve, status=None):
    """style.is_diverged: recorded as diverged, or a curve ending non-finite, or a val
    loss ending more than DIVERGED_FACTOR above the untrained value at val[0]."""
    if status is not None and str(status) == STATUS_DIVERGED:
        return True
    fv, ft = _final(val_curve), _final(train_curve)
    if not (math.isfinite(fv) and math.isfinite(ft)):
        return True
    try:
        v0 = float(val_curve[0])
    except (TypeError, ValueError, IndexError):
        return False
    return bool(math.isfinite(v0) and v0 > 0 and fv > DIVERGED_FACTOR * v0)


def config_eligible(n_ok, n_expected):
    """style.config_eligible: more than half the expected seeds must have finished."""
    return n_ok > n_expected / 2.0


def family_of(run_id):
    for prefix, fam in _PREFIX_FAMILY:
        if run_id.startswith(prefix):
            return fam
    return "other"


# ---------------------------------------------------------------------------
# grid.py, loaded by path (its package __init__ imports torch)
# ---------------------------------------------------------------------------

def load_grid(repo=REPO):
    path = os.path.join(repo, "experiments", "experiment_code", "grid.py")
    spec = importlib.util.spec_from_file_location("_sv3_grid", path)
    if spec is None or spec.loader is None:              # pragma: no cover
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    # must be in sys.modules BEFORE exec: @dataclass resolves its own module there
    sys.modules.setdefault(spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


class ConfigLoader:
    """Composes hydra configs (torch-free) and caches the resolved containers."""

    def __init__(self, config_dir=None):
        self.config_dir = os.path.abspath(
            config_dir or os.path.join(REPO, "experiments", "configs"))
        self._ctx = None
        self._cache = {}

    def __enter__(self):
        from hydra import initialize_config_dir
        self._ctx = initialize_config_dir(config_dir=self.config_dir, version_base=None)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        ctx, self._ctx = self._ctx, None
        if ctx is not None:
            ctx.__exit__(*exc)
        return False

    def compose(self, config_name, overrides):
        """The resolved config container for `config_name` + `overrides` (a string)."""
        key = (config_name, overrides)
        if key not in self._cache:
            from hydra import compose as hydra_compose
            from omegaconf import OmegaConf
            cfg = hydra_compose(config_name=config_name,
                                overrides=[o for o in str(overrides).split() if o])
            self._cache[key] = OmegaConf.to_container(cfg, resolve=True)
        return self._cache[key]


def has_torchjd():
    """Whether the runner would enumerate the JD family at all."""
    try:
        return importlib.util.find_spec("torchjd") is not None
    except (ImportError, ValueError):
        return False


# ---------------------------------------------------------------------------
# What should exist
# ---------------------------------------------------------------------------

def injected_dataset_facts(scan_dir):
    """The `cfg.model` keys the RUNNER injected before hashing, or `{}`.

    `generic_scan.run_grid` mutates `cfg.model` with the dataset's `vocab_size` (and
    `block_size`) before anything is instantiated or hashed, because
    `experiments/configs/model/nanogpt.yaml` deliberately omits them -- the dataset owns
    them (`grid.inject_dataset_facts`).  Composing the config here and hashing it
    unchanged therefore gives EVERY language-model run a different hash8, so a finished
    nanoGPT scan reports as `stale-hash`, i.e. "work remains", for ever -- and a chained
    job renews itself for ever on it.

    The values cannot be derived without loading the data (the char vocabulary IS the
    corpus), so they are read back from the post-mutation config the runner saved in
    `{scan}/configs/*.yaml`.  A scan whose model config never carried them -- every MLP,
    ResNet and GPT-2 scan -- yields `{}` and nothing is injected.
    """
    seen = set()
    for path in sorted(_listdir(os.path.join(scan_dir, "configs"))):
        if not path.endswith(".yaml"):
            continue
        try:
            import yaml
            with open(os.path.join(scan_dir, "configs", path)) as fh:
                saved = yaml.safe_load(fh) or {}
        except (OSError, ValueError, ImportError):
            continue
        model = saved.get("model") or {}
        if not isinstance(model, dict) or model.get("vocab_size") is None:
            continue
        seen.add((int(model["vocab_size"]),
                  None if model.get("block_size") is None else int(model["block_size"])))
    if len(seen) != 1:
        return {}                      # nothing saved, or two generations: do not guess
    vocab_size, block_size = seen.pop()
    return {"vocab_size": vocab_size, "block_size": block_size}


def expected_runs(scan, groups, loader, grid, *, use_grid=True, jd=None, scan_dir=None):
    """`({run_id: info}, warnings)` -- the union over override groups.

    `info` carries the family, the current run hash, the batch size, the optimizer and
    the hyperparameters, i.e. everything the edge report needs without opening a record.

    `scan_dir` is where the runner's own resolved configs live; it is what makes the
    hash of a language-model run reproducible here (:func:`injected_dataset_facts`).
    """
    expected, warnings = {}, []
    if not use_grid:
        return expected, warnings
    jd = has_torchjd() if jd is None else jd
    facts = injected_dataset_facts(scan_dir) if scan_dir else {}
    for overrides in groups:
        try:
            rcfg = loader.compose(scan, overrides)
        except Exception as exc:
            warnings.append(f"cannot compose {scan} with {overrides!r}: "
                            f"{type(exc).__name__}: {exc}")
            continue
        model = rcfg.get("model")
        if facts and isinstance(model, dict) and model.get("vocab_size") is None:
            # exactly `run_grid`'s mutation, so `grid.run_hash` sees the model the runs
            # actually instantiated
            rcfg = dict(rcfg)
            rcfg["model"] = grid.inject_dataset_facts(model, **facts)
        try:
            specs = grid.expand_grid(rcfg, verbose=False, has_torchjd=jd)
        except Exception as exc:
            warnings.append(f"expand_grid failed for {scan} {overrides!r}: "
                            f"{type(exc).__name__}: {exc}")
            continue
        for spec in specs:
            info = expected.get(spec.run_id)
            if info is None:
                expected[spec.run_id] = {
                    "family": spec.family,
                    "hash8": grid.hash8(spec, rcfg),
                    "batch_size": spec.batch_size,
                    "optimizer": spec.record_extra.get("optimizer"),
                    "hparams": dict(spec.hparams),
                    "overrides": overrides,
                }
            elif info["hash8"] != grid.hash8(spec, rcfg):
                # same name, different computation: two override groups disagree about
                # what this run IS, so one of them would move the other's result to
                # _stale/ and re-run it, forever.
                warnings.append(
                    f"run_id {spec.run_id} has two different run hashes "
                    f"({info['overrides']!r} vs {overrides!r}) -- these groups fight "
                    f"over the same result file")
    return expected, warnings


def manifest_run_ids(scan_dir):
    """The union of `{scan}/manifest/*.json` run_ids (style.manifest_run_ids)."""
    out, bad = set(), []
    d = os.path.join(scan_dir, MANIFEST_DIRNAME)
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return out, bad
    for name in names:
        if not name.endswith(".json") or name.startswith(".tmp-"):
            continue
        try:
            with open(os.path.join(d, name)) as fh:
                payload = json.load(fh)
            out.update(payload.get("run_ids") or ())
        except (OSError, ValueError) as exc:
            bad.append(f"unreadable manifest {name}: {exc}")
    return out, bad


# ---------------------------------------------------------------------------
# What does exist: four listdirs
# ---------------------------------------------------------------------------

def disk_state(scan_dir):
    """`done`, `started`, `claims`, `jsonl` from ONE `os.listdir` each."""
    done = {}                       # run_id -> {hash8: {status, ...}}
    for name in _listdir(os.path.join(scan_dir, DONE_DIRNAME)):
        parts = name.rsplit(".", 2)          # run_ids contain '.', hashes do not
        if len(parts) != 3:
            continue
        run_id, hash8, status = parts
        if status not in STATUSES or not run_id:
            continue
        done.setdefault(run_id, {}).setdefault(hash8, set()).add(status)

    started = {n[:-len(STARTED_SUFFIX)]
               for n in _listdir(os.path.join(scan_dir, STARTED_DIRNAME))
               if n.endswith(STARTED_SUFFIX)}

    claims_dir = os.path.join(scan_dir, CLAIMS_DIRNAME)
    claims = {}                     # run_id -> {"gens": n, "mtime": newest}
    for name in _listdir(claims_dir):
        base = None
        if name.endswith(CLAIM_SUFFIX):
            base = name[:-len(CLAIM_SUFFIX)]
        else:
            head, _, tail = name.rpartition(".")
            if tail.isdigit() and head.endswith(CLAIM_SUFFIX):
                base = head[:-len(CLAIM_SUFFIX)]
        if not base:
            continue
        try:
            mtime = os.path.getmtime(os.path.join(claims_dir, name))
        except OSError:
            mtime = 0.0
        rec = claims.setdefault(base, {"gens": 0, "mtime": 0.0})
        rec["gens"] += 1
        rec["mtime"] = max(rec["mtime"], mtime)

    top = _listdir(scan_dir)
    jsonl = {n[:-len(".jsonl")] for n in top if n.endswith(".jsonl")}
    return {"done": done, "started": started, "claims": claims, "jsonl": jsonl,
            "has_done_dir": os.path.isdir(os.path.join(scan_dir, DONE_DIRNAME))}


def _listdir(path):
    try:
        return os.listdir(path)
    except OSError:
        return []


def classify(run_id, info, state, now):
    """One expected run's class (see :data:`CLASSES`)."""
    markers = state["done"].get(run_id, {})
    hash8 = (info or {}).get("hash8")
    mine = markers.get(hash8, set()) if hash8 else set()
    if not hash8 and markers:                 # manifest-only run: accept any hash
        mine = set().union(*markers.values())
    for status in ("ok", "diverged"):
        if status in mine:
            return status
    claim = state["claims"].get(run_id)
    if claim and (now - claim["mtime"]) < CLAIM_TIMEOUT_S:
        return "claimed_live"
    for status in ("oom", "error"):
        if status in mine:
            return status
    if markers and not mine:
        # markers exist, but for another run hash: the result on disk was computed by
        # different code/config and the runner will retire it to _stale/ and re-run.
        return "stale_hash"
    if run_id in state["started"]:
        return "started_only"
    if run_id in state["jsonl"]:
        # a record but no marker: either a legacy scan (no done/ at all) or a crash
        # between the jsonl and the marker. Either way the runner re-runs it.
        return "jsonl_only"
    return "never_started"


# ---------------------------------------------------------------------------
# Best configuration per method, and whether it sits on a grid edge
# ---------------------------------------------------------------------------

def read_records(scan_dir, run_ids):
    """`{run_id: record}` for the run_ids that have a readable jsonl."""
    out, bad = {}, []
    for run_id in run_ids:
        path = os.path.join(scan_dir, run_id + ".jsonl")
        try:
            with open(path) as fh:
                line = fh.readline()
            out[run_id] = json.loads(line)
        except (OSError, ValueError) as exc:
            bad.append(f"unreadable record {run_id}: {exc}")
    return out, bad


def best_configs(records, expected):
    """Per method: its best configuration and the edge verdict for lr / k / rtol / tau.

    "Best" = lowest seed-mean final validation loss over the non-diverged runs of a
    configuration that is eligible (`style.config_eligible`: more than half its expected
    seeds finished). The grid a value is compared against is taken from the EXPECTED
    specs of the same method and batch size, so a per-method lr list (Muon after the
    `match_rms_adamw` change, say) is handled without reading the config again.

    A record whose CONFIGURATION is not in the expected grid cannot win: a method whose
    grid has moved (grid_counts.md records HIG's lr grid being shifted rather than grown)
    leaves records of the old grid on disk, and they have no expected seed count, so
    `config_eligible` would pass a single lucky seed and its off-grid value would then be
    skipped by the edge check -- i.e. exactly the config C-B3 must not extend. Records of
    a method that is not in the expected grid AT ALL (a legacy family, or a reconcile of
    one override group) are kept and marked `in_grid: False`: there is nothing they could
    displace, and they are what the operator asked about.
    """
    n_expected_per_key = defaultdict(int)
    grids = defaultdict(lambda: defaultdict(set))       # (method, bs) -> axis -> values
    methods_expected = set()
    keys_expected = defaultdict(set)                    # method -> {config_key}
    for run_id, info in expected.items():
        key = config_key(run_id)
        n_expected_per_key[key] += 1
        method, bs = info.get("optimizer"), info.get("batch_size")
        methods_expected.add(method)
        keys_expected[method].add(key)
        for axis in AXES:
            value = info.get("hparams", {}).get(axis)
            if value is not None:
                grids[(method, bs)][axis].add(value)

    groups, n_off_grid = {}, 0
    for run_id, rec in records.items():
        method = rec.get("optimizer") or family_of(run_id)
        cfg = config_key(run_id)
        if method in methods_expected and cfg not in keys_expected[method]:
            n_off_grid += 1                  # a configuration the grid no longer has
            continue
        key = (method, cfg)
        losses = rec.get("losses") or {}
        bad = is_diverged(losses.get("train"), losses.get("val"), rec.get("status"))
        g = groups.setdefault(key, {"vals": [], "n": 0, "n_div": 0, "rep": rec})
        g["n"] += 1
        if bad:
            g["n_div"] += 1
        else:
            g["vals"].append(_final(losses.get("val")))

    best = {}
    for (method, key), g in groups.items():
        vals = [v for v in g["vals"] if math.isfinite(v)]
        n_exp = n_expected_per_key.get(key, g["n"])
        if not vals or not config_eligible(len(vals), n_exp):
            continue
        mean = sum(vals) / len(vals)
        cur = best.get(method)
        if cur is None or mean < cur["seed_mean"]:
            rep = g["rep"]
            best[method] = {"config_key": key, "seed_mean": mean, "n_ok": len(vals),
                            "n_expected": n_exp, "n_diverged": g["n_div"],
                            "batch_size": rep.get("batch_size"),
                            "values": {a: rep.get(a) for a in AXES}}

    for method, b in best.items():
        axes = grids.get((method, b["batch_size"]), {})
        edges = {}
        for axis, value in b["values"].items():
            options = sorted(v for v in axes.get(axis, ()) if v is not None)
            if value is None or not options:
                continue
            if value not in options:
                # never silently interior: an axis whose value is not on the grid it is
                # compared against is a verdict of its own, not a missing verdict
                edges[axis] = "OFF-GRID"
            elif len(options) < 2:
                continue                     # a single-point axis cannot be extended
            elif value == options[0]:
                edges[axis] = "EDGE-LOW"
            elif value == options[-1]:
                edges[axis] = "EDGE-HIGH"
        b["edges"] = edges
        b["grid_sizes"] = {a: len(axes.get(a, ())) for a in AXES}
        # a method on disk but outside this reconcile's expected grid (a partial override
        # group, or a legacy family) has no grid to be on the edge of -- say so
        b["in_grid"] = method in methods_expected
    return best, n_off_grid


# ---------------------------------------------------------------------------
# One scan
# ---------------------------------------------------------------------------

def reconcile_scan(scan, groups, root, *, grid, loader, use_grid=True, use_manifest=True,
                   do_best=True, max_list=25, out=sys.stdout, quiet=False):
    """Print the report for one scan; return the report dict (`ok` says nothing is missing)."""
    scan_dir = os.path.join(root, scan)
    now = time.time()
    # `fatal` = "the question could not be answered" (a config that will not compose, an
    # override that describes nothing) -> exit 2, never "work remains". `warnings` are
    # advisory (an unreadable manifest or record) and do not change the verdict.
    fatal, warnings = [], []

    expected, fatal_warn = expected_runs(scan, groups, loader, grid, use_grid=use_grid,
                                         scan_dir=scan_dir)
    fatal += fatal_warn
    state = disk_state(scan_dir)
    manifest, warn = (manifest_run_ids(scan_dir) if use_manifest else (set(), []))
    warnings += warn
    for run_id in manifest:
        expected.setdefault(run_id, {"family": family_of(run_id), "hash8": None,
                                     "batch_size": None, "optimizer": None,
                                     "hparams": {}, "overrides": "<manifest>"})

    counts = defaultdict(lambda: defaultdict(int))
    per_class = defaultdict(list)
    for run_id, info in sorted(expected.items()):
        cls = classify(run_id, info, state, now)
        fam = info["family"]
        counts[fam][cls] += 1
        counts[fam]["expected"] += 1
        per_class[cls].append(run_id)

    unexpected = sorted(state["jsonl"] - set(expected))
    missing_total = sum(len(per_class[c]) for c in MISSING_CLASSES)

    report = {
        "scan": scan, "scan_dir": scan_dir, "groups": list(groups),
        "n_expected": len(expected), "n_from_manifest": len(manifest),
        "n_jsonl_on_disk": len(state["jsonl"]), "has_done_dir": state["has_done_dir"],
        "counts": {f: dict(c) for f, c in counts.items()},
        "totals": {c: len(per_class[c]) for c in CLASSES},
        "never_started": per_class["never_started"],
        "started_only": per_class["started_only"],
        "stale_hash": per_class["stale_hash"],
        "jsonl_only": per_class["jsonl_only"],
        "oom": per_class["oom"], "error": per_class["error"],
        "unexpected_jsonl": unexpected,
        "n_missing": missing_total, "warnings": warnings, "fatal": fatal,
        "best": {}, "n_off_grid_records": 0, "config_dir": getattr(loader, "config_dir", None),
        "ok": missing_total == 0 and len(expected) > 0 and not fatal,
    }
    if not expected and not fatal:
        report["fatal"].append(f"{scan}: nothing expected at all -- the overrides "
                               f"{groups} describe an empty grid and the manifest is empty")

    if do_best:
        records, bad = read_records(scan_dir, sorted(state["jsonl"]))
        report["warnings"] += bad[:5]
        report["best"], report["n_off_grid_records"] = best_configs(records, expected)

    if quiet:
        return report

    p = lambda *a: print(*a, file=out)
    p(f"\n=== {scan} ===")
    p(f"  dir        {scan_dir}")
    p(f"  configs    {report['config_dir']}")
    n_grid = sum(1 for i in expected.values() if i["hash8"] is not None)
    p(f"  expected   {len(expected)} runs   (config grid {n_grid}, "
      f"manifest-only {len(expected) - n_grid})")
    p(f"  on disk    {len(state['jsonl'])} jsonl"
      + ("" if state["has_done_dir"] else "   [no done/ dir: pre-claim-queue results]"))
    for group in groups:
        p(f"  overrides  {group!r}")

    head = (f"  {'family':10s}" + "".join(f"{c.replace('_', '-'):>14s}"
                                          for c in ("expected",) + CLASSES))
    p(head)
    for fam in sorted(counts):
        row = counts[fam]
        p(f"  {fam:10s}" + "".join(f"{row.get(c, 0):>14d}"
                                   for c in ("expected",) + CLASSES))
    if len(counts) > 1:
        p(f"  {'TOTAL':10s}" + "".join(
            f"{sum(counts[f].get(c, 0) for f in counts):>14d}"
            for c in ("expected",) + CLASSES))

    for cls in ("never_started", "started_only", "stale_hash", "jsonl_only"):
        ids = per_class[cls]
        if not ids:
            continue
        p(f"  {cls.replace('_', '-')} ({len(ids)}):")
        for run_id in ids[:max_list]:
            p(f"    {run_id}")
        if len(ids) > max_list:
            p(f"    ... {len(ids) - max_list} more (use --max-list 0 for all, "
              f"--json to write them out)")
    if unexpected:
        p(f"  unexpected jsonl on disk, not in any expected grid ({len(unexpected)}):")
        for run_id in unexpected[:5]:
            p(f"    {run_id}")
        if len(unexpected) > 5:
            p(f"    ... {len(unexpected) - 5} more")

    if report["best"]:
        p("  best config per method (seed-mean final val loss, diverged excluded):")
        p(f"    {'method':12s} {'seed-mean':>12s} {'n_ok/exp':>9s} "
          f"{'lr':>10s} {'k':>7s} {'rtol':>9s} {'tau':>9s}  edges")
        for method in sorted(report["best"], key=lambda m: report["best"][m]["seed_mean"]):
            b = report["best"][method]
            v = b["values"]
            edges = (" ".join(f"{a}:{e}" for a, e in sorted(b["edges"].items()))
                     or ("-" if b.get("in_grid") else "(not in the expected grid)"))
            p(f"    {str(method):12s} {b['seed_mean']:12.4g} "
              f"{b['n_ok']:>4d}/{b['n_expected']:<4d} "
              f"{_fmt(v['lr']):>10s} {_fmt(v['k']):>7s} {_fmt(v['rtol']):>9s} "
              f"{_fmt(v['tau']):>9s}  {edges}")
        p(f"    config keys: " + "; ".join(
            f"{m}={report['best'][m]['config_key']}" for m in sorted(report["best"]))[:400])
    elif do_best:
        p("  best config per method: nothing eligible yet "
          "(needs > half the expected seeds of a config to have finished)")
    if do_best and report["n_off_grid_records"]:
        p(f"  {report['n_off_grid_records']} record(s) of a method in the grid but of a "
          f"configuration that is NOT: ignored for 'best' (an old grid point cannot win)")

    for w in report["warnings"]:
        p(f"  [warn] {w}")
    for w in report["fatal"]:
        p(f"  [FATAL] {w}")
    verdict = ("CANNOT ANSWER" if report["fatal"] else
               "COMPLETE" if report["ok"] else "WORK REMAINS")
    p(f"  => {verdict}: {missing_total} run(s) to do"
      + (f", {len(report['warnings'])} warning(s)" if report["warnings"] else ""))
    return report


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def err(*args):
    """stderr after flushing stdout (see tools/launch_campaign.py:err)."""
    sys.stdout.flush()
    print(*args, file=sys.stderr, flush=True)


def resolve_root(root=None, plan=None):
    """The results root to read, and where it came from.

    `--root` wins, then the PLAN's `results_root` (the launcher exports exactly that to
    the workers, so reconciling a plan against a different root answers a question
    nobody asked -- e.g. with `SV3_RESULTS_ROOT` pointing at the frozen legacy root, as
    EXPERIMENTS.md tells people to do), then $SV3_RESULTS_ROOT, then the
    repo's `experiment_results` symlink.
    """
    env = os.environ.get("SV3_RESULTS_ROOT")
    if root:
        return os.path.abspath(root), "--root"
    if plan is not None and plan.results_root:
        chosen = os.path.abspath(plan.results_root)
        if env and os.path.abspath(env) != chosen:
            err(f"[reconcile] note: using the plan's results_root {chosen}, NOT "
                f"$SV3_RESULTS_ROOT={os.path.abspath(env)} (pass --root to override)")
        return chosen, f"the plan's results_root"
    if env:
        return os.path.abspath(env), "$SV3_RESULTS_ROOT"
    return os.path.abspath(os.path.join(REPO, "experiment_results")), "the repo's symlink"


def resolve_config_dir(config_dir=None, snapshot=None):
    """The hydra config dir, and where it came from.

    A reconcile that composes the LIVE configs while the jobs ran the SNAPSHOT's can
    report every finished run as `stale-hash` (run_hash covers the resolved dataset,
    model, num_epochs, loss and the eval settings), i.e. "work remains" for a scan that
    is complete. When this tool runs from inside a snapshot (the chain does:
    `$SNAP/tools/reconcile.py`), REPO already IS the snapshot.
    """
    if config_dir:
        return os.path.abspath(config_dir), "--config-dir"
    snapshot = snapshot or os.environ.get("SV3_DEPLOY_SNAPSHOT") or os.environ.get("SV3_SNAPSHOT")
    if snapshot:
        path = os.path.join(snapshot, "experiments", "configs")
        if os.path.isdir(path):
            return os.path.abspath(path), "the snapshot's"
        err(f"[reconcile] note: {snapshot} has no experiments/configs; using "
            f"{os.path.join(REPO, 'experiments', 'configs')}")
    return os.path.abspath(os.path.join(REPO, "experiments", "configs")), "this tree's"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Reconcile a scan (or a whole plan) against what should exist.")
    ap.add_argument("scan", nargs="?", help="the scan / hydra config name")
    ap.add_argument("--root", default=None, help="results root (default $SV3_RESULTS_ROOT)")
    ap.add_argument("--config-overrides", action="append", default=[], metavar="STR",
                    help="one override group, repeatable; the expected grid is their union")
    ap.add_argument("--config-dir", default=None, help="hydra config dir")
    ap.add_argument("--snapshot", default=None,
                    help="a deploy snapshot whose experiments/configs to compose "
                         "(default: $SV3_DEPLOY_SNAPSHOT / $SV3_SNAPSHOT, else this tree)")
    ap.add_argument("--all", dest="plan", default=None, metavar="PLAN",
                    help="reconcile every scan of a campaign plan file")
    ap.add_argument("--phase", default=None, help="with --all: only this phase (P0, or P0,P1)")
    ap.add_argument("--lane", default=None, help="with --all: only this lane")
    ap.add_argument("--list", dest="lists", action="append", default=None,
                    help="with --all: only these work lists")
    ap.add_argument("--no-grid", action="store_true",
                    help="expect only what the manifest declares")
    ap.add_argument("--no-manifest", action="store_true",
                    help="expect only what the config grid describes")
    ap.add_argument("--no-best", action="store_true",
                    help="skip reading records (counts only; fastest)")
    ap.add_argument("--max-list", type=int, default=25,
                    help="how many run_ids to print per class (0 = all)")
    ap.add_argument("--json", default=None, help="write the full report(s) here")
    ap.add_argument("--quiet", action="store_true",
                    help="print only the one-line verdict (for a job chain)")
    a = ap.parse_args(argv)

    if not a.scan and not a.plan:
        ap.error("give a scan name or --all <plan>")

    plan = None
    if a.plan:
        sys.path.insert(0, HERE)
        try:
            import campaign_plan
            plan = campaign_plan.load_plan(a.plan)
        except Exception as exc:
            err(f"[reconcile] ERROR: {type(exc).__name__}: {exc}")
            return 2
        todo = plan.scan_overrides(phase=a.phase, lane=a.lane, names=a.lists)
        if a.scan:
            todo = {a.scan: todo.get(a.scan, [])}
        if not todo:
            err("[reconcile] ERROR: the plan selection is empty")
            return 2
    else:
        todo = {a.scan: list(a.config_overrides) or [""]}

    root, root_from = resolve_root(a.root, plan)
    config_dir, cfg_from = resolve_config_dir(a.config_dir, a.snapshot)
    # grid.py must come from the same tree as the configs: run_hash is computed by it,
    # so pairing a snapshot's configs with another tree's grid.py is not the hash the
    # workers wrote either.
    code_root = REPO
    if cfg_from == "the snapshot's":
        cand = os.path.dirname(os.path.dirname(config_dir))
        if os.path.isfile(os.path.join(cand, "experiments", "experiment_code", "grid.py")):
            code_root = cand
    if not a.quiet:
        print(f"[reconcile] results root {root}   ({root_from})")
        print(f"[reconcile] configs      {config_dir}   ({cfg_from})")
        print(f"[reconcile] grid.py      {code_root}")
    if not os.path.isdir(root):
        err(f"[reconcile] ERROR: results root {root} does not exist")
        return 2

    grid = load_grid(code_root)
    reports, rc = [], 0
    max_list = a.max_list if a.max_list > 0 else 10 ** 9
    try:
        with ConfigLoader(config_dir) as loader:
            for scan, groups in todo.items():
                rep = reconcile_scan(
                    scan, groups or [""], root, grid=grid, loader=loader,
                    use_grid=not a.no_grid, use_manifest=not a.no_manifest,
                    do_best=not a.no_best, max_list=max_list, quiet=a.quiet)
                reports.append(rep)
                if rep["fatal"]:
                    rc = 2
                elif not rep["ok"] and rc != 2:
                    rc = 1
    except Exception as exc:                                  # a broken config etc.
        err(f"[reconcile] ERROR: {type(exc).__name__}: {exc}")
        return 2

    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"root": root, "reports": reports}, fh, indent=1, default=str)
        if not a.quiet:
            print(f"\n[reconcile] wrote {a.json}")

    missing = sum(r["n_missing"] for r in reports)
    scans_bad = [r["scan"] for r in reports if not r["ok"]]
    unanswerable = [r["scan"] for r in reports if r["fatal"]]
    print(f"\n[reconcile] {len(reports)} scan(s), {missing} run(s) to do; "
          f"incomplete: {', '.join(scans_bad) if scans_bad else 'none'}"
          + (f"; CANNOT ANSWER: {', '.join(unanswerable)}" if unanswerable else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())
