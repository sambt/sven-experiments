"""Order + shard equivalence of ``grid.expand_grid`` against the REAL pre-C-R2 loops.

This is the *producer* of ``tests/golden/<scan>.order.txt`` (the ordered legacy
enumerations that ``tests/test_grid.py`` asserts against) and the only thing that
can re-freeze them. Run it whenever a scan config that has an ``.order.txt``
changes -- see "RE-FREEZING" in ``tests/test_grid.py``.

How it works: it runs ``scan()`` exactly as it stands at the pinned pre-refactor
commit :data:`LEGACY_SHA` (no source edits -- the module is exec'd inside the real
``experiments.experiment_code`` package so its relative imports work), with only
the *execution* stubbed out:

  * ``instantiate``            -> a fake dataset / model (no download, no CUDA),
  * ``set_seed``/``_scan_facts`` -> no-ops,
  * ``os``                     -> a shim whose ``path.exists`` RECORDS the run_id
    and returns True, so every run looks "already on disk" and the legacy loop
    only enumerates.

``_shard_skip()`` runs *before* that check, so what is recorded is the legacy
shard assignment; comparing it with ``grid.shard(specs, n, i)`` is what proves
``specs[shard_id::n_shards]`` picks the same runs as the legacy modulo counter.

Configs are read from the WORKING TREE, only the runner comes from
:data:`LEGACY_SHA`: after a config edit (e.g. C-B4 dropping the ``weight_decays``
grid) a re-freeze therefore still compares legacy-enumeration-of-the-new-config
against new-enumeration-of-the-new-config, which is the property under test.

Usage (needs torch, so run it through ``campaign/run_cpu_tests.sh``)::

    .venv/bin/python tests/golden/legacy_grid_equiv.py            # verify only
    .venv/bin/python tests/golden/legacy_grid_equiv.py --freeze   # rewrite *.order.txt

Known fragility: the pinned runner imports names from the *current*
``experiment_utils.py``. If another track stops re-exporting one of them the exec
fails with an ImportError; extract the whole pinned tree instead, e.g.
``git archive 58fc8e4 | tar -x -C /tmp/legacy_sv3`` and run this file from there.
"""
import argparse
import os as real_os
import subprocess
import sys

REPO = real_os.path.dirname(real_os.path.dirname(real_os.path.dirname(
    real_os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from omegaconf import OmegaConf  # noqa: E402

#: the commit whose ``generic_scan.py`` is the pre-C-R2 (six inline product loops)
#: runner. NOT "HEAD": the orchestrator commits the refactor on top of it.
LEGACY_SHA = "58fc8e4"

CONFIGS = real_os.path.join(REPO, "experiments/configs")
GOLDEN = real_os.path.join(REPO, "tests/golden")
REL = "experiments/experiment_code/generic_scan.py"

CASES = [  # (scan, mode, overrides)
    ("toy_1d_scan", "all", ()),
    ("toy_1d_scan", "both", ()),
    ("toy_1d_scan", "svd", ()),
    ("mnist_scan_ce", "all", ()),
    ("cifar10_resnet_scan_labelRegression", "all", ()),
    ("rebuttal_batchsize_polynomial_scan", "both", ()),
    ("mnist_paramfrac_labelreg_scan", "svd", ()),
    ("mnist_kappaScan_labelRegression", "svd", ()),
    ("toy_1d_microbatch_scan", "svd", ()),
    ("exp_finetune_cifar_smallN", "both", ("n_data=250",)),
    ("exp_finetune_cifar_smallN", "svd", ("n_data=2000",)),
]
SHARD_CASES = [(1, 0), (2, 0), (2, 1), (3, 2), (6, 4), (7, 0)]
#: which (scan, mode) pairs `--freeze` writes to `<scan>.order.txt`; these are the
#: files tests/test_grid.py::test_enumeration_order_matches_the_legacy_loops reads.
FREEZE = {"toy_1d_scan": "all", "rebuttal_batchsize_polynomial_scan": "both"}
#: run_id tokens the pinned runner CANNOT emit, because they postdate it: C-E2's
#: `bn_mode` replaced `gram_freeze_norm_stats`, so a config that sets `bn_mode`
#: gets `_bnbatch` / `_bnfrozen` from `grid.bn_mode_suffix` while the legacy loop,
#: which only ever read the boolean, gets its default. The comparison below strips
#: them and reports how many ids differed only by one, so what is still under test
#: is what this file exists for -- the ORDER and the shard membership.
POST_LEGACY_TOKENS = ("_bnbatch", "_bnfrozen")
#: which token each scan is EXPECTED to carry, and on which FAMILIES -- pinned here,
#: not read from the config, so this file stays an oracle. A blanket strip would report
#: `OK` for a CIFAR headline scan flipped to `bn_mode: frozen`, which is exactly the
#: setting that collapsed Sven to ~28% accuracy (probe 2026-09-12): every id would gain
#: `_bnfrozen`, the strip would remove it and the order would still match. It would also
#: report `OK` for a config that LOST `bn_mode` altogether.
#:
#: The carrier families differ per token because `grid.bn_mode_suffix` marks the
#: departure from each family's default (`grid.default_bn_mode`): the Gram svd family
#: defaulted to frozen, so asking for `batch` is what shows up there, while every other
#: family defaulted to batch, so only `frozen` shows up on those.
EXPECTED_BN_TOKEN = {
    # C-E2: batch statistics for every optimizer -> the svd ids say so
    "cifar10_resnet_scan_labelRegression": ("_bnbatch", ("svd",)),
    # O2: frozen pretrained statistics for every optimizer -> the baselines say so
    "exp_finetune_cifar_smallN": ("_bnfrozen", ("standard", "lbfgs", "polyak",
                                                "jd", "hig")),
}


def expected_bn(scan_name):
    """``(token, carrier_families)`` for ``scan_name``, or ``(None, ())``."""
    return EXPECTED_BN_TOKEN.get(scan_name, (None, ()))


def strip_post_legacy(run_id, scan_name):
    """Strip only the token ``scan_name`` is expected to carry (may be none)."""
    token, _ = expected_bn(scan_name)
    return run_id.replace(token, "") if token else run_id


def unexpected_bn_tokens(scan_name, run_ids):
    """run_ids carrying a post-legacy token other than this scan's expected one."""
    expected, _ = expected_bn(scan_name)
    others = [t for t in POST_LEGACY_TOKENS if t != expected]
    return [i for i in run_ids if any(t in i for t in others)]


def load_cfg(name, overrides=()):
    """The config as hydra would compose it (defaults list + dotlist overrides)."""
    cfg = OmegaConf.load(real_os.path.join(CONFIGS, f"{name}.yaml"))
    defaults = cfg.pop("defaults", [])
    merged = OmegaConf.create({})
    for entry in defaults:
        if entry == "_self_":
            continue
        if isinstance(entry, str):                       # a parent config (*_timing)
            merged = OmegaConf.merge(merged, load_cfg(entry))
            continue
        for group, option in OmegaConf.to_container(entry).items():
            merged = OmegaConf.merge(merged, OmegaConf.create(
                {group: OmegaConf.load(real_os.path.join(CONFIGS, group, f"{option}.yaml"))}))
    merged = OmegaConf.merge(merged, cfg)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
    return merged


class _FakeModel:
    def state_dict(self):
        return {}


class _FakeDataset:
    train_dataset = [0]
    val_dataset = [0]


class _PathShim:
    def __init__(self, sink):
        self.sink = sink
        self.join = real_os.path.join

    def exists(self, p):
        base = real_os.path.basename(p)
        if base.endswith(".jsonl"):
            self.sink.append(base[:-len(".jsonl")])
        return True


class _OSShim:
    def __init__(self, sink):
        self.path = _PathShim(sink)

    def makedirs(self, *a, **k):
        pass


def legacy_module():
    """The pinned pre-refactor ``generic_scan`` module, exec'd (never imported)."""
    code = subprocess.run(["git", "-C", REPO, "show", f"{LEGACY_SHA}:{REL}"],
                          capture_output=True, text=True, check=True).stdout
    assert "def _build_id_string" in code and "_shard_skip" in code, (
        f"{LEGACY_SHA}:{REL} is not the pre-C-R2 runner")
    ns = {"__name__": "legacy_generic_scan",
          "__package__": "experiments.experiment_code",
          "__file__": real_os.path.join(REPO, REL)}
    try:
        exec(compile(code, f"{REL}@{LEGACY_SHA}", "exec"), ns)
    except ImportError as e:
        raise SystemExit(
            f"the pinned runner no longer imports against the working tree ({e}).\n"
            "See 'Known fragility' in this file's docstring.")
    return ns


def legacy_run_ids(ns, cfg, scan_name, mode, n_shards=1, shard_id=0):
    """The run_ids the legacy loop would have executed, in its own order."""
    sink = []
    cfg = OmegaConf.merge(cfg, OmegaConf.create(
        {"mode": mode, "n_shards": n_shards, "shard_id": shard_id}))
    ns["instantiate"] = lambda c: (_FakeDataset() if "Dataset" in str(c.get("_target_", ""))
                                   else _FakeModel())
    ns["set_seed"] = lambda *a, **k: None
    ns["_scan_facts"] = lambda *a, **k: {}
    ns["os"] = _OSShim(sink)
    ns["HydraConfig"] = type("H", (), {"get": staticmethod(
        lambda: type("J", (), {"job": type("K", (), {"config_name": scan_name})()})())})
    devnull = open(real_os.devnull, "w")
    out, sys.stdout = sys.stdout, devnull
    try:
        ns["scan"](cfg)
    finally:
        sys.stdout = out
        devnull.close()
    return sink


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", action="store_true",
                    help="rewrite tests/golden/<scan>.order.txt from the legacy runner")
    args = ap.parse_args()

    from experiments.experiment_code.grid import expand_grid, shard

    ns = legacy_module()
    ok = True
    for scan_name, mode, ov in CASES:
        cfg = load_cfg(scan_name, ov)
        rcfg = OmegaConf.to_container(cfg, resolve=True)
        specs = expand_grid(rcfg, mode=mode, verbose=False)
        new = [s.run_id for s in specs]
        old = legacy_run_ids(ns, cfg, scan_name, mode)
        stripped = [strip_post_legacy(i, scan_name) for i in new]
        n_bn = sum(a != b for a, b in zip(new, stripped))
        wrong = unexpected_bn_tokens(scan_name, new)
        # EXACTLY the carrier-family ids of an EXPECTED_BN_TOKEN scan must carry it: too
        # few means the config lost `bn_mode` (or gained it on the wrong family) and
        # silently fell back to the legacy default, which no other check here would see
        token, carriers = expected_bn(scan_name)
        n_carry = sum(1 for s in specs if s.family in carriers)
        same = stripped == old and not wrong and n_bn == n_carry
        ok &= same
        note = f" ({n_bn} ids carry {token})" if n_bn else ""
        if n_bn != n_carry:
            note += f" (EXPECTED {n_carry} x {token} on families {carriers})"
        if wrong:
            note += f" ({len(wrong)} ids carry an UNEXPECTED bn_mode token, " \
                    f"e.g. {wrong[0]!r})"
        print(f"{'OK ' if same else 'BAD'} {scan_name:38s} mode={mode:9s} "
              f"legacy={len(old):5d} new={len(new):5d} order_identical="
              f"{stripped == old}{note}")
        if stripped != old:
            for i, (a, b) in enumerate(zip(old, stripped)):
                if a != b:
                    print(f"     first difference at {i}: legacy={a!r} new={b!r}")
                    break
            continue
        for n, i in SHARD_CASES:
            old_s = legacy_run_ids(ns, cfg, scan_name, mode, n_shards=n, shard_id=i)
            new_s = [strip_post_legacy(s.run_id, scan_name) for s in shard(specs, n, i)]
            if old_s != new_s:
                ok = False
                print(f"     BAD shard n={n} id={i}: legacy={len(old_s)} new={len(new_s)}")
        if args.freeze and FREEZE.get(scan_name) == mode and not ov:
            # the frozen file is the LEGACY enumeration, which test_grid.py compares
            # against the UNstripped new ids -- so a FREEZE scan must carry no bn_mode
            # token at all. (One would already have failed the order comparison above,
            # because only an EXPECTED_BN_TOKEN scan gets stripped; this keeps the
            # reason attached to the freeze.)
            assert scan_name not in EXPECTED_BN_TOKEN and not any(
                t in i for i in new for t in POST_LEGACY_TOKENS), (
                f"{scan_name} now carries a bn_mode token; either drop it from "
                "FREEZE or teach test_enumeration_order_matches_the_legacy_loops to "
                "strip them (tests/test_grid.py owns that assertion)")
            path = real_os.path.join(GOLDEN, f"{scan_name}.order.txt")
            with open(path, "w") as f:
                f.write(f"# ordered legacy enumeration of {scan_name} (mode={mode}), produced by\n"
                        f"# tests/golden/legacy_grid_equiv.py --freeze: scan() from {LEGACY_SHA}\n"
                        f"# ({REL}, pre-C-R2) with execution stubbed out.\n")
                f.write("\n".join(old) + "\n")
            print(f"     froze {len(old)} ids -> {path}")

    print("\nALL EQUIVALENT" if ok else "\nMISMATCHES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
