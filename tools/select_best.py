#!/usr/bin/env python3
"""The best configuration of every method in every headline scan (Phase 5, step A).

    tools/select_best.py                       # the seven headline scans -> bench/best_configs.json
    tools/select_best.py --print               # table only, write nothing
    tools/select_best.py toy_1d_scan --print
    tools/select_best.py --compare-analysis    # also cross-check analysis/lib/scan_analysis.py

This is the ONE place Phase 5 decides what "the best configuration" is; the timing,
diagnostics and confirmation passes are generated from its output
(`tools/gen_phase5_plan.py`), never from a second selection. A grid-extension round
therefore costs one re-run of this tool plus one re-run of the generator.

**The rule** (CHANGES_NEEDED.md section 1, "binding -- do not reopen"), in full:

    eligible  ->  fewest diverged seeds  ->  lowest seed-mean final VALIDATION loss

`eligible` = more than half of a configuration's expected seeds finished; diverged =
failed (`style.is_diverged`: recorded `diverged`, a non-finite end, or a final val loss
above 10x the untrained value), excluded from the mean and counted. **Test metrics are
never read** -- :func:`assert_no_test_metric` checks that as an assertion, not a comment.
Grid membership is reconcile's: a record of a configuration the current grid no longer
describes can never win.

Note that `tools/reconcile.py:best_configs` implements only the LAST tier (seed mean), so
the two disagree exactly where a configuration wins on the mean by diverging on seeds its
rivals survive -- on the real results that is LBFGS and SOAP, 6 of 85 selections. This
tool therefore ranks itself and reports the difference (`--rule seed_mean` reproduces
reconcile's answer); `analysis/lib/scan_analysis.py:Scan.configs`, which makes the paper's
plots, implements the same full rule and is cross-checked with `--compare-analysis`.

What this tool adds on top of the rule is the step neither of the others takes -- turning
the winning *configuration key* back into the **exact hydra overrides** that reproduce it,
and then proving they do by expanding the grid again and checking the run_ids and run
hashes come back byte-identical (:func:`verify_overrides`). A selection that cannot be
reproduced by an override string is reported as an error and written to the json with
`"verified": false`, because an unverified override string is how a timing pass silently
times a different configuration than the one the paper plots.

Torch-free (hydra for composition, `grid.py` loaded by path, stdlib for everything else),
so it runs in a few seconds on a login node over the whole results root.

Exit status: **0** every scan produced a verified selection · **1** something could not be
selected or verified · **2** the question could not be answered (bad config, missing root).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import reconcile                                        # noqa: E402  (same directory)

#: the seven scans Phase 5 covers (CONTRACTS.md scope update: six headline scans + nanoGPT)
HEADLINE_SCANS = (
    "toy_1d_scan",
    "polynomial_scan",
    "mnist_scan_labelRegression",
    "mnist_scan_ce",
    "cifar10_resnet_scan_labelRegression",
    "cifar10_resnet_ce_scan",
    "exp_nanogpt_speedrun",
)

#: the override groups whose union is a scan's full grid. `mode=all` enumerates every
#: family the config describes (`grid.mode_flags`: jd/hig only when the config carries
#: their hyperparameters), which is exactly the set of methods Phase 5 must cover. On the
#: two CIFAR scans it also enumerates the PARKED jd/hig grids -- those have no records, so
#: they can never be selected, and their only effect is a larger `expected` set.
DEFAULT_GROUPS = ("mode=all",)

#: `optimizer` values, in report order: Sven first, then everything else alphabetically.
SVEN_METHOD = "SVD"

#: record keys that would leak the test split into a selection. Asserted absent from the
#: scored quantity, not merely unused: "never select on test" is a rule about this file.
TEST_KEYS = ("test", "test_acc", "test_step")

#: `full` = CHANGES_NEEDED.md section 1 (eligible -> fewest diverged -> seed mean), which
#: is what `analysis/lib/scan_analysis.py` plots; `seed_mean` = `reconcile.best_configs`'s
#: single tier, kept so the two tables can be diffed without editing reconcile.
RULES = ("full", "seed_mean")


# ---------------------------------------------------------------------------
# Expanding the grid, keeping the specs
# ---------------------------------------------------------------------------

def expand_scan(scan, groups, loader, grid, scan_dir):
    """`(expected, by_key, warnings)` for one scan.

    `expected` is exactly what `reconcile.best_configs` wants ({run_id: info}); `by_key`
    maps a configuration key (the run_id without its `_mseed…_lseed…` suffix) to the
    RunSpecs and the resolved config behind it, which is what an override string has to be
    built from and verified against.
    """
    expected, by_key, warnings = {}, {}, []
    jd = reconcile.has_torchjd()
    facts = reconcile.injected_dataset_facts(scan_dir)
    for overrides in groups:
        try:
            rcfg = compose_with_facts(loader, grid, scan, overrides, facts)
        except Exception as exc:
            warnings.append(f"cannot compose {scan} with {overrides!r}: "
                            f"{type(exc).__name__}: {exc}")
            continue
        try:
            specs = grid.expand_grid(rcfg, verbose=False, has_torchjd=jd)
        except Exception as exc:
            warnings.append(f"expand_grid failed for {scan} {overrides!r}: "
                            f"{type(exc).__name__}: {exc}")
            continue
        for spec in specs:
            h8 = grid.hash8(spec, rcfg)
            expected.setdefault(spec.run_id, {
                "family": spec.family, "hash8": h8, "batch_size": spec.batch_size,
                "optimizer": spec.record_extra.get("optimizer"),
                "hparams": dict(spec.hparams), "overrides": overrides,
            })
            key = reconcile.config_key(spec.run_id)
            entry = by_key.setdefault(key, {
                "family": spec.family, "method": spec.record_extra.get("optimizer"),
                "batch_size": spec.batch_size, "rcfg": rcfg,
                "group": overrides, "specs": {}, "hash8": {},
            })
            entry["specs"][spec.run_id] = spec
            # per RUN, not per configuration: `grid.run_hash` hashes the model seed, so
            # the five seeds of one configuration have five different hashes.
            entry["hash8"][spec.run_id] = h8
    return expected, by_key, warnings


def compose_with_facts(loader, grid, scan, overrides, facts):
    """The resolved config a run of `scan` really instantiates.

    `generic_scan.run_grid` injects the dataset's `vocab_size` / `block_size` into
    `cfg.model` before anything is hashed, because `model/nanogpt.yaml` deliberately omits
    them. Composing without that mutation gives every language-model run a different
    hash8 (see `reconcile.injected_dataset_facts`).
    """
    rcfg = loader.compose(scan, overrides)
    model = rcfg.get("model")
    if facts and isinstance(model, dict) and model.get("vocab_size") is None:
        rcfg = dict(rcfg)
        rcfg["model"] = grid.inject_dataset_facts(model, **facts)
    return rcfg


# ---------------------------------------------------------------------------
# Configuration key -> hydra overrides
# ---------------------------------------------------------------------------

def fmt(value):
    """A hydra override literal that parses back to exactly `value`.

    `repr` on a float round-trips (0.0001 -> "0.0001", 1e-05 -> "1e-05") and the run_id is
    built from `str(float)` of whatever hydra parsed, so a value that round-trips here
    reproduces the run_id character for character. :func:`verify_overrides` is what
    actually proves it for each selection.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _listify(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _swept(rcfg, key, default=None):
    """Whether the config sweeps `key` (more than one value), i.e. whether an override
    pinning it is needed at all. A one-point axis is left alone so the override string
    stays the minimal statement of what was chosen."""
    return len(_listify(rcfg.get(key, default))) > 1


def build_overrides(spec, rcfg):
    """The hydra overrides that make `spec`'s configuration the WHOLE grid.

    Every axis the config sweeps is pinned; an axis with a single value is left to the
    config, so the string says what was selected and nothing else. The family's `mode=`
    comes first because that is what decides which axes exist at all.
    """
    hp, family = spec.hparams, spec.family
    parts = []
    if _swept(rcfg, "batch_size", 32):
        parts.append(f"batch_size=[{fmt(spec.batch_size)}]")

    if family == "svd":
        parts.insert(0, "mode=svd")
        if "k_values" not in rcfg and "k_fractions" in rcfg:
            # process_hparam_config asserts k_values XOR k_fractions, so `k_values=[...]`
            # on a k_fractions config raises instead of pinning anything.
            raise ValueError(
                f"{spec.run_id}: this config sweeps k_fractions, not k_values; pin the "
                f"fraction by hand (k={hp['k']}, B={spec.batch_size})")
        parts += [f"k_values=[{fmt(hp['k'])}]",
                  f"lrs=[{fmt(hp['lr'])}]",
                  f"rtol=[{fmt(hp['rtol'])}]"]
        if _swept(rcfg, "svd_mode", "randomized"):
            parts.append(f"svd_mode=[{fmt(hp['svd_mode'])}]")
        if hp.get("microbatch_size") is not None:
            parts.append(f"microbatch_sizes=[{fmt(hp['microbatch_size'])}]")
        if hp.get("param_fraction") is not None:
            parts.append(f"param_fractions=[{fmt(hp['param_fraction'])}]")
        if _swept(rcfg, "kappa", 2.0):
            parts.append(f"kappa=[{fmt(hp['kappa'])}]")
    elif family == "standard":
        parts.insert(0, "mode=standard")
        parts += [f"optimizers_standard=[{hp['optim_name']}]",
                  f"lrs_standard=[{fmt(hp['lr'])}]"]
        if _swept(rcfg, "weight_decays", [None]):
            parts.append(f"weight_decays=[{fmt(hp['weight_decay'])}]")
    elif family == "lbfgs":
        parts.insert(0, "mode=standard")
        parts += ["optimizers_standard=[LBFGS]",
                  f"lrs_lbfgs=[{fmt(hp['lr'])}]",
                  f"lbfgs_max_iter=[{fmt(hp['max_iter'])}]",
                  f"lbfgs_history_size=[{fmt(hp['history_size'])}]"]
        if _swept(rcfg, "lbfgs_line_search_fn", "strong_wolfe"):
            parts.append(f"lbfgs_line_search_fn=[{fmt(hp['line_search_fn'])}]")
    elif family == "polyak":
        parts.insert(0, "mode=standard")
        parts.append("optimizers_standard=[PolyakSGD]")
        for key, hkey in (("polyak_f_star", "f_star"), ("polyak_max_lr", "max_lr"),
                          ("polyak_eps", "eps")):
            if _swept(rcfg, key):
                parts.append(f"{key}=[{fmt(hp[hkey])}]")
    elif family == "jd":
        parts.insert(0, "mode=jd")
        parts += [f"lrs_jd=[{fmt(hp['lr'])}]",
                  f"aggregators_jd=[{hp['aggregator']}]",
                  f"inner_optimizers_jd=[{hp['inner_optimizer']}]"]
    elif family == "hig":
        parts.insert(0, "mode=hig")
        parts += [f"lrs_hig=[{fmt(hp['lr'])}]", f"tau_hig=[{fmt(hp['tau'])}]"]
    else:                                                # pragma: no cover
        raise ValueError(f"unknown family {family!r}")
    return " ".join(parts)


def verify_overrides(scan, overrides, want_hash8, loader, grid, facts):
    """`(ok, message)`: do `overrides` expand to EXACTLY the runs of `want_hash8`?

    This is the whole point of the tool. An override string that expands to a superset
    would make the timing pass run the neighbouring grid points too; one that expands to a
    different point would time a configuration nobody plots; one that expands to the right
    run_ids under a different run hash would be re-run by the next job as `stale_hash`.

    `want_hash8` is `{run_id: hash8}` -- per run, because `grid.run_hash` hashes the model
    seed, so one configuration has one hash per seed.
    """
    try:
        rcfg = compose_with_facts(loader, grid, scan, overrides, facts)
        specs = grid.expand_grid(rcfg, verbose=False, has_torchjd=reconcile.has_torchjd())
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    got = {s.run_id for s in specs}
    want = set(want_hash8)
    if got != want:
        extra, missing = sorted(got - want)[:3], sorted(want - got)[:3]
        return False, (f"expands to {len(got)} run(s), wanted {len(want)}"
                       + (f"; unexpected {extra}" if extra else "")
                       + (f"; missing {missing}" if missing else ""))
    bad = sorted(f"{s.run_id} {grid.hash8(s, rcfg)} != {want_hash8[s.run_id]}"
                 for s in specs if grid.hash8(s, rcfg) != want_hash8[s.run_id])
    if bad:
        return False, f"same run_ids, different run hash: {bad[:2]}"
    return True, ""


# ---------------------------------------------------------------------------
# One scan
# ---------------------------------------------------------------------------

def _mean(values):
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(values) / len(values) if values else None


def score_configs(records, expected):
    """`({method: [config, ...] ranked best-first}, n_off_grid)` -- the binding rule.

    Scoring and grid membership are `reconcile.best_configs`'s, line for line (a record of
    a configuration the current grid no longer describes cannot win; a method that is not
    in the grid at all is kept, because it is what the operator asked about). The ranking
    is the full section-1 one; see :func:`rank_key`.
    """
    n_expected_per_key = defaultdict(int)
    methods_expected, keys_expected = set(), defaultdict(set)
    for run_id, info in expected.items():
        n_expected_per_key[reconcile.config_key(run_id)] += 1
        methods_expected.add(info.get("optimizer"))
        keys_expected[info.get("optimizer")].add(reconcile.config_key(run_id))

    groups, n_off_grid = {}, 0
    for run_id, rec in records.items():
        method = rec.get("optimizer") or reconcile.family_of(run_id)
        cfg = reconcile.config_key(run_id)
        if method in methods_expected and cfg not in keys_expected[method]:
            n_off_grid += 1
            continue
        losses = rec.get("losses") or {}
        bad = reconcile.is_diverged(losses.get("train"), losses.get("val"),
                                    rec.get("status"))
        g = groups.setdefault((method, cfg), {"vals": [], "n": 0, "n_div": 0})
        g["n"] += 1
        if bad:
            g["n_div"] += 1
        else:
            g["vals"].append(reconcile._final(losses.get("val")))

    ranked = defaultdict(list)
    for (method, key), g in groups.items():
        vals = [v for v in g["vals"] if math.isfinite(v)]
        n_exp = n_expected_per_key.get(key, g["n"])
        if not vals or not reconcile.config_eligible(len(vals), n_exp):
            continue
        ranked[method].append({
            "config_key": key, "seed_mean": sum(vals) / len(vals), "n_ok": len(vals),
            "n_expected": n_exp, "n_diverged": g["n_div"],
        })
    return dict(ranked), n_off_grid


def axis_grids_of(expected):
    """`{(method, batch_size): {axis: sorted values}}` over `reconcile.AXES`.

    Per method AND batch size, so a method with its own lr list (Muon after the
    `match_rms_adamw` change) is compared against its own grid, not the union.
    """
    grids = defaultdict(lambda: defaultdict(set))
    for info in expected.values():
        for axis in reconcile.AXES:
            value = info.get("hparams", {}).get(axis)
            if value is not None:
                grids[(info.get("optimizer"), info.get("batch_size"))][axis].add(value)
    return {k: {a: sorted(v) for a, v in axes.items()} for k, axes in grids.items()}


def edge_verdicts(axis_grids, method, batch_size, hparams):
    """Which of lr / k / rtol / tau sits at the end of its grid (C-B3's question).

    A one-point axis cannot be extended and gets no verdict; a value that is not on the
    grid it is compared against gets `OFF-GRID`, which is a verdict of its own.
    """
    axes = axis_grids.get((method, batch_size), {})
    out = {}
    for axis in reconcile.AXES:
        value, options = hparams.get(axis), axes.get(axis) or []
        if value is None or not options:
            continue
        if value not in options:
            out[axis] = "OFF-GRID"
        elif len(options) < 2:
            continue
        elif value == options[0]:
            out[axis] = "EDGE-LOW"
        elif value == options[-1]:
            out[axis] = "EDGE-HIGH"
    return out


def rank_key(cfg, hparams, rule="full"):
    """The sort key of one eligible configuration: lower is better.

    `full` puts **fewest diverged seeds** ahead of the seed mean, per CHANGES_NEEDED.md
    section 1: dropping a configuration's failures from its mean flatters it, so a
    configuration that blows up on 2 of 5 seeds must not beat one that finishes all 5.
    `seed_mean` drops that tier and reproduces `reconcile.best_configs`.

    Everything after the score is a TIE-BREAK, and exact ties are the normal case, not a
    corner case: whenever Sven's rtol-rank stays below k, every larger k is the same
    trajectory, and every HIG tau below the numerical floor cuts nothing. The order is
    `analysis/lib/scan_analysis.py:Scan.configs`'s -- smallest k, largest rtol, smallest lr,
    i.e. the cheapest equivalent configuration -- extended with tau (which that function
    leaves to a stable sort over the dataframe, i.e. to file order) and finally the
    configuration key, so the answer does not depend on which machine globbed the results.
    """
    def num(name, default):
        v = hparams.get(name)
        return default if v is None else float(v)
    tail = (num("k", math.inf), -num("rtol", -math.inf), num("lr", math.inf),
            num("tau", math.inf), cfg["config_key"])
    if rule == "seed_mean":
        return (cfg["seed_mean"],) + tail
    return (cfg["n_diverged"], cfg["seed_mean"]) + tail


def select_scan(scan, groups, root, *, grid, loader, rule="full"):
    """`{method: selection}` plus a report dict, for one scan."""
    scan_dir = os.path.join(root, scan)
    expected, by_key, warnings = expand_scan(scan, groups, loader, grid, scan_dir)
    state = reconcile.disk_state(scan_dir)
    records, bad = reconcile.read_records(scan_dir, sorted(state["jsonl"]))
    warnings += bad[:5]

    ranked, n_off_grid = score_configs(records, expected)
    axis_grids = axis_grids_of(expected)
    # A selection made while a grid extension is still running is a selection over half a
    # grid: `config_eligible` passes a configuration with 3 of 5 seeds, so a half-finished
    # extension point can win outright. The count is reported (and `--require-complete`
    # turns it into an error) rather than silently tolerated.
    now = time.time()
    n_missing = sum(1 for run_id, info in expected.items()
                    if reconcile.classify(run_id, info, state, now)
                    in reconcile.MISSING_CLASSES)
    # the same table under reconcile's single-tier rule, so the difference is reported
    # rather than discovered later (see the module docstring)
    rec_best, _ = reconcile.best_configs(records, expected)
    facts = reconcile.injected_dataset_facts(scan_dir)

    def hparams_of(key):
        entry = by_key.get(key)
        if entry is None:
            return {}
        return dict(next(iter(entry["specs"].values())).hparams)

    best, disagreements = {}, []
    for method, candidates in ranked.items():
        order = sorted(candidates,
                       key=lambda c: rank_key(c, hparams_of(c["config_key"]), rule))
        best[method] = order[0]
        best[method]["n_eligible_configs"] = len(order)
        # exact ties on the score: the tie-break decided, so say which runners-up it beat
        best[method]["tied_with"] = [c["config_key"] for c in order[1:]
                                     if c["n_diverged"] == order[0]["n_diverged"]
                                     and c["seed_mean"] == order[0]["seed_mean"]][:4]
        theirs = rec_best.get(method)
        if theirs and theirs["config_key"] != order[0]["config_key"]:
            disagreements.append(
                f"{method}: reconcile picks {theirs['config_key']} "
                f"(mean {theirs['seed_mean']:.6g}, {theirs['n_diverged']} diverged), "
                f"this tool picks {order[0]['config_key']} "
                f"(mean {order[0]['seed_mean']:.6g}, "
                f"{order[0]['n_diverged']} diverged)")

    methods, errors = {}, []
    for method, b in sorted(best.items()):
        key = b["config_key"]
        entry = by_key.get(key)
        if entry is None:
            # a method whose records are on disk but whose configuration the current grid
            # no longer describes (reconcile marks it `in_grid: False`): there is no
            # override string that reproduces it from today's config.
            errors.append(f"{scan}/{method}: winning config {key!r} is not in the "
                          f"current grid -- no reproducible overrides")
            continue
        run_ids = sorted(entry["specs"])
        spec = entry["specs"][run_ids[0]]
        try:
            overrides = build_overrides(spec, entry["rcfg"])
        except ValueError as exc:
            errors.append(f"{scan}/{method}: {exc}")
            continue
        ok, why = verify_overrides(scan, overrides, entry["hash8"], loader, grid, facts)
        if not ok:
            errors.append(f"{scan}/{method}: overrides {overrides!r} do not reproduce "
                          f"the selected runs ({why})")

        seed_vals, wall, train_t = {}, [], []
        for run_id in run_ids:
            rec = records.get(run_id)
            if rec is None:
                continue
            losses = rec.get("losses") or {}
            diverged = reconcile.is_diverged(losses.get("train"), losses.get("val"),
                                             rec.get("status"))
            seed_vals[str(rec.get("model_seed"))] = {
                "final_val": None if diverged else reconcile._final(losses.get("val")),
                "status": rec.get("status"), "diverged": bool(diverged),
                "wall_time_s": rec.get("wall_time_s"),
            }
            if not diverged:
                wall.append(rec.get("wall_time_s"))
                train_t.append(_mean(losses.get("train_times") or []))
        methods[method] = {
            "method": method,
            "label": "Sven" if method == SVEN_METHOD else method,
            "family": entry["family"],
            "config_key": key,
            "overrides": overrides,
            "verified": bool(ok),
            "verify_error": None if ok else why,
            "seed_mean_final_val": b["seed_mean"],
            "n_ok": b["n_ok"], "n_expected": b["n_expected"],
            "n_diverged": b["n_diverged"],
            "n_eligible_configs": b["n_eligible_configs"],
            "tied_with": b["tied_with"],
            "edges": edge_verdicts(axis_grids, method, entry["batch_size"], spec.hparams),
            "batch_size": entry["batch_size"],
            "hash8": dict(entry["hash8"]),
            "hparams": dict(spec.hparams),
            "model_seeds": sorted({s.model_seed for s in entry["specs"].values()}),
            "loader_seed": spec.loader_seed,
            "run_ids": run_ids,
            "seed_values": seed_vals,
            "mean_wall_time_s": _mean(wall),
            "n_records": len(seed_vals),
        }

    report = {
        "scan": scan, "scan_dir": scan_dir, "groups": list(groups), "rule": rule,
        "n_expected": len(expected), "n_records": len(records),
        "n_off_grid_records": n_off_grid, "n_missing": n_missing,
        "warnings": warnings, "errors": errors,
        "reconcile_disagreements": disagreements,
        "methods": methods,
    }
    return report


def assert_no_test_metric(report):
    """`selection uses validation only` -- as an assertion, not as a comment.

    Every number this tool ranks on is `seed_mean_final_val`, taken from `losses['val']`.
    This walks the selection it is about to write and refuses any test key that has crept
    into the scored fields.
    """
    for method, sel in report["methods"].items():
        for key in TEST_KEYS:
            if key in sel:
                raise AssertionError(
                    f"{report['scan']}/{method}: selection carries a test key {key!r}; "
                    f"CHANGES_NEEDED.md section 1: selection uses validation only")


# ---------------------------------------------------------------------------
# Cross-checks
# ---------------------------------------------------------------------------

def compare_with_analysis(reports, root):
    """Disagreements with `analysis/lib/scan_analysis.py`'s `best_sven` / `best_baseline`.

    That module implements one rule this tool's does NOT: it ranks `eligible ->
    n_diverged -> seed mean` and then breaks exact ties deterministically (smallest k,
    largest rtol). `tools/reconcile.py:best_configs` ranks on the seed mean alone. The two
    can therefore disagree on (a) a configuration that wins on the mean but diverges on
    more seeds and (b) an exact tie -- which is common for Sven, where every k above the
    numerical rank is the same trajectory. Disagreements are reported, never silently
    resolved: which one is right is a decision for the orchestrator.

    Needs numpy/pandas (not torch); returns a list of human-readable differences plus the
    error, if the import or a scan fails.
    """
    out = []
    sys.path.insert(0, os.path.join(REPO, "analysis", "lib"))
    sys.path.insert(0, os.path.join(REPO, "analysis"))
    try:
        import scan_analysis                                     # noqa: WPS433
    except Exception as exc:
        return [f"cannot import analysis/lib/scan_analysis.py: {type(exc).__name__}: {exc}"]
    for rep in reports:
        try:
            s = scan_analysis.load_scan(rep["scan"], rep["scan"], "/tmp/_select_best",
                                        results_root=root)
        except Exception as exc:
            out.append(f"{rep['scan']}: load_scan failed: {type(exc).__name__}: {exc}")
            continue
        for method, sel in sorted(rep["methods"].items()):
            try:
                cfg = (s.best_sven() if method == SVEN_METHOD
                       else s.best_baseline(method))
            except Exception as exc:
                out.append(f"{rep['scan']}/{method}: {type(exc).__name__}: {exc}")
                continue
            if cfg is None:
                out.append(f"{rep['scan']}/{method}: scan_analysis finds nothing eligible, "
                           f"this tool picked {sel['config_key']}")
                continue
            mine = sel["hparams"]
            diff = []
            for axis, mine_key in (("k", "k"), ("lr", "lr"), ("rtol", "rtol"),
                                   ("tau", "tau"), ("weight_decay", "weight_decay"),
                                   ("lbfgs_max_iter", "max_iter"),
                                   ("lbfgs_history_size", "history_size")):
                if axis not in cfg:
                    continue
                theirs = cfg[axis]
                ours = mine.get(mine_key)
                if theirs is None or (isinstance(theirs, float) and math.isnan(theirs)):
                    continue
                if ours is None or abs(float(theirs) - float(ours)) > 1e-12 * max(
                        1.0, abs(float(theirs))):
                    diff.append(f"{axis}: scan_analysis {theirs!r} vs select_best {ours!r}")
            if diff:
                out.append(f"{rep['scan']}/{method}: " + "; ".join(diff))
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def to_json(reports, *, root, config_dir, groups, rule):
    return {
        "schema": 2,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generated_by": "tools/select_best.py",
        "results_root": root,
        "config_dir": config_dir,
        "rule_name": rule,
        "rule": ("eligible (more than half the expected seeds finished) -> fewest "
                 "diverged seeds -> lowest seed-mean final VALIDATION loss over the "
                 "non-diverged runs; diverged = style.is_diverged; test metrics are "
                 "never read (CHANGES_NEEDED.md section 1)"
                 if rule == "full" else
                 "lowest seed-mean final VALIDATION loss among eligible configurations "
                 "(tools/reconcile.py:best_configs; NOT the full section-1 rule)"),
        "default_groups": list(groups),
        "scans": {r["scan"]: {
            "groups": r["groups"],
            "scan_dir": r["scan_dir"],
            "n_expected": r["n_expected"],
            "n_records": r["n_records"],
            "n_missing": r["n_missing"],
            "warnings": r["warnings"],
            "errors": r["errors"],
            "reconcile_disagreements": r["reconcile_disagreements"],
            "methods": r["methods"],
        } for r in reports},
    }


def print_table(rep, out=sys.stdout):
    p = lambda *a: print(*a, file=out)
    p(f"\n=== {rep['scan']} ===   {rep['n_records']} record(s), "
      f"{rep['n_expected']} expected, groups {rep['groups']}")
    if rep["n_missing"]:
        p(f"  INCOMPLETE: {rep['n_missing']} run(s) of this grid are not finished -- "
          f"a configuration with 3 of 5 seeds is already eligible, so re-select when "
          f"`tools/reconcile.py {rep['scan']}` is clean")
    if not rep["methods"]:
        p("  nothing eligible")
    else:
        p(f"  {'method':12s} {'seed-mean val':>14s} {'ok/exp':>8s} {'div':>4s}  "
          f"{'':1s}overrides")
        for method in sorted(rep["methods"],
                             key=lambda m: rep["methods"][m]["seed_mean_final_val"]):
            sel = rep["methods"][method]
            mark = " " if sel["verified"] else "!"
            p(f"  {sel['label']:12s} {sel['seed_mean_final_val']:14.6g} "
              f"{sel['n_ok']:>3d}/{sel['n_expected']:<4d} {sel['n_diverged']:>4d}  "
              f"{mark}{sel['overrides']}")
            if sel["edges"]:
                p(f"               edges: "
                  + " ".join(f"{a}:{e}" for a, e in sorted(sel["edges"].items())))
    for d in rep["reconcile_disagreements"]:
        p(f"  [vs reconcile] {d}")
    for w in rep["warnings"]:
        p(f"  [warn] {w}")
    for e in rep["errors"]:
        p(f"  [ERROR] {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Select the best configuration of every method in every headline scan.")
    ap.add_argument("scans", nargs="*", default=None,
                    help=f"scans to select (default: the seven headline scans)")
    ap.add_argument("--root", default=None, help="results root (default $SV3_RESULTS_ROOT)")
    ap.add_argument("--config-dir", default=None, help="hydra config dir")
    ap.add_argument("--snapshot", default=None,
                    help="a deploy snapshot whose experiments/configs to compose")
    ap.add_argument("--groups", action="append", default=None, metavar="STR",
                    help="override group, repeatable (default: 'mode=all')")
    ap.add_argument("--out", default=os.path.join(REPO, "bench", "best_configs.json"),
                    help="where the selection goes (default bench/best_configs.json)")
    ap.add_argument("--require-complete", action="store_true",
                    help="fail when a scan still has unfinished runs (a half-finished "
                         "grid extension can win with 3 of 5 seeds)")
    ap.add_argument("--rule", default="full", choices=RULES,
                    help="'full' = CHANGES_NEEDED section 1 (eligible -> fewest diverged "
                         "-> seed mean, the default); 'seed_mean' = reconcile's tier only")
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="print the table, write nothing")
    ap.add_argument("--compare-analysis", action="store_true",
                    help="also run analysis/lib/scan_analysis.py's selection and diff it "
                         "(needs numpy/pandas)")
    a = ap.parse_args(argv)

    scans = a.scans or list(HEADLINE_SCANS)
    groups = tuple(a.groups) if a.groups else DEFAULT_GROUPS
    root, root_from = reconcile.resolve_root(a.root, None)
    config_dir, cfg_from = reconcile.resolve_config_dir(a.config_dir, a.snapshot)
    print(f"[select] results root {root}   ({root_from})")
    print(f"[select] configs      {config_dir}   ({cfg_from})")
    if not os.path.isdir(root):
        reconcile.err(f"[select] ERROR: results root {root} does not exist")
        return 2

    code_root = REPO
    if cfg_from == "the snapshot's":
        cand = os.path.dirname(os.path.dirname(config_dir))
        if os.path.isfile(os.path.join(cand, "experiments", "experiment_code", "grid.py")):
            code_root = cand
    grid = reconcile.load_grid(code_root)

    reports, rc = [], 0
    try:
        with reconcile.ConfigLoader(config_dir) as loader:
            for scan in scans:
                rep = select_scan(scan, groups, root, grid=grid, loader=loader,
                                  rule=a.rule)
                assert_no_test_metric(rep)
                reports.append(rep)
                print_table(rep)
    except Exception as exc:
        reconcile.err(f"[select] ERROR: {type(exc).__name__}: {exc}")
        return 2

    unverified = [(r["scan"], m) for r in reports for m, s in r["methods"].items()
                  if not s["verified"]]
    n_err = sum(len(r["errors"]) for r in reports)
    empty = [r["scan"] for r in reports if not r["methods"]]
    incomplete = [(r["scan"], r["n_missing"]) for r in reports if r["n_missing"]]
    if incomplete:
        print("[select] INCOMPLETE grids: "
              + ", ".join(f"{s} ({n} run(s) to do)" for s, n in incomplete))

    if a.compare_analysis:
        print("\n[select] cross-check against analysis/lib/scan_analysis.py:")
        diffs = compare_with_analysis(reports, root)
        for d in diffs:
            print(f"[select]   DIFF {d}")
        if not diffs:
            print("[select]   no disagreements")

    if not a.print_only:
        payload = to_json(reports, root=root, config_dir=config_dir, groups=groups,
                          rule=a.rule)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(payload, fh, indent=1, default=str)
        print(f"\n[select] wrote {a.out}")

    n_methods = sum(len(r["methods"]) for r in reports)
    n_dis = sum(len(r["reconcile_disagreements"]) for r in reports)
    print(f"[select] rule '{a.rule}'; {len(reports)} scan(s), {n_methods} selection(s), "
          f"{len(unverified)} unverified, {n_err} error(s), "
          f"{n_dis} disagreement(s) with reconcile"
          + (f"; NO selection for: {', '.join(empty)}" if empty else ""))
    if unverified or n_err or empty or (a.require_complete and incomplete):
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
