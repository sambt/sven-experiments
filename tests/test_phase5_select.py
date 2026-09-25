"""`tools/select_best.py`: the binding rule, and overrides that provably reproduce it.

CPU only, temp dirs only, no torch, no SLURM: `select_best` and `reconcile` are loaded by
path, exactly as `tests/test_tools_reconcile.py` does it (`tools/` is not a package and
must not become one -- the campaign runs these as scripts from an exported snapshot).

The two acceptance tests the contract asks for are
:func:`test_overrides_reproduce_exactly_the_selected_run_ids` -- every selected
configuration's override string expands to precisely its own seed run_ids, same run
hashes, nothing more -- and :func:`test_selection_ignores_the_test_split`, which builds
records whose test loss ranks the configurations in the OPPOSITE order to their validation
loss and requires the validation order to win.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")


def _load(name):
    path = os.path.join(TOOLS, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_tools_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


reconcile = _load("reconcile")
select_best = _load("select_best")

#: A standalone scan config (no `defaults:`, so the dataset/model groups are not needed and
#: another track's config edit cannot break this test) that carries EVERY family: svd,
#: standard, lbfgs, polyak, jd and hig.
TINY_CONFIG = """
name: scan
mode: all
device: cpu
print_config: false
num_epochs: 3
batch_size: 8
result_id_fields: []
loss: mse
use_gram: true
k_values: [2, 4]
lrs: [0.1, 1.0]
rtol: [0.001, 0.01]
svd_mode: [torch]
variable_k: false
lrs_standard: [0.001, 0.01]
optimizers_standard: [Adam, SOAP, LBFGS, PolyakSGD]
lrs_lbfgs: [0.1, 0.5]
lbfgs_max_iter: [1, 2]
lbfgs_history_size: [2]
lbfgs_line_search_fn: [strong_wolfe]
lrs_jd: [0.001, 0.01]
aggregators_jd: [UPGrad]
inner_optimizers_jd: [Adam]
lrs_hig: [0.05, 0.1]
tau_hig: [0.0001, 0.01]
loader_seed: 7
model_seeds: [0, 1, 2]
data_seed: 7
"""

SCAN = "tiny_phase5_scan"
GROUPS = ("mode=all",)
N_SEEDS = 3


def _curve(final):
    """A healthy 3-epoch val curve (index 0 = untrained) ending at `final`."""
    return [1.0, 0.5, 0.1, final]


def _write_scan(base, grid, *, final_of, test_of=None, status_of=None, extra_records=()):
    """A finished synthetic scan in `base`; `final_of(spec) -> final val loss`.

    `test_of` writes a `test` value per spec (used to prove it is never selected on);
    `status_of` can mark a run `diverged`. `extra_records` are `(run_id, record)` pairs
    written on top -- how an off-grid leftover gets onto the disk.
    """
    cfg_dir = base / "configs"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / f"{SCAN}.yaml").write_text(TINY_CONFIG)
    root = base / "results"
    scan_dir = root / SCAN
    for sub in ("done", "started", "claims", "manifest"):
        (scan_dir / sub).mkdir(parents=True, exist_ok=True)

    with reconcile.ConfigLoader(str(cfg_dir)) as loader:
        rcfg = loader.compose(SCAN, "mode=all")
        specs = grid.expand_grid(rcfg, verbose=False)
        hashes = {s.run_id: grid.hash8(s, rcfg) for s in specs}

    for spec in specs:
        status = (status_of(spec) if status_of else None) or "ok"
        final = float("nan") if status == "diverged" else final_of(spec)
        record = {
            "run_id": spec.run_id, "schema_version": 2, "status": status,
            "run_hash": hashes[spec.run_id], "model_seed": spec.model_seed,
            "wall_time_s": 10.0,
            "losses": {"train": _curve(final), "val": _curve(final),
                       "train_times": [1.0, 1.0, 1.0]},
            **spec.record_extra,
        }
        if test_of is not None:
            record["test"] = test_of(spec)
        (scan_dir / f"{spec.run_id}.jsonl").write_text(json.dumps(record) + "\n")
        (scan_dir / "done" / f"{spec.run_id}.{hashes[spec.run_id]}.{status}").touch()
    for run_id, record in extra_records:
        (scan_dir / f"{run_id}.jsonl").write_text(json.dumps(record) + "\n")
    return str(cfg_dir), str(root), specs, hashes


@pytest.fixture(scope="module")
def grid():
    return reconcile.load_grid()


def _select(cfg_dir, root, grid, **kw):
    with reconcile.ConfigLoader(cfg_dir) as loader:
        return select_best.select_scan(SCAN, GROUPS, root, grid=grid, loader=loader, **kw)


# ---------------------------------------------------------------------------
# The acceptance tests
# ---------------------------------------------------------------------------

def test_overrides_reproduce_exactly_the_selected_run_ids(tmp_path, grid):
    """Every method's override string expands to its own seed runs and nothing else.

    This is what the whole of Phase 5 rests on: the timing, diagnostics and confirmation
    passes run these strings, so a string that expands to a neighbouring grid point (or to
    the same run_ids under a different run hash, which the runner would retire to
    `_stale/` and re-run) silently measures a configuration nobody plots.
    """
    # a distinct, unambiguous winner per family: the LAST grid point of each axis
    def final_of(spec):
        hp = spec.hparams
        best = {
            "svd": hp.get("lr") == 1.0 and hp.get("k") == 4 and hp.get("rtol") == 0.01,
            "standard": hp.get("lr") == 0.01,
            "lbfgs": hp.get("lr") == 0.5 and hp.get("max_iter") == 2,
            "polyak": True,
            "jd": hp.get("lr") == 0.01,
            "hig": hp.get("lr") == 0.1 and hp.get("tau") == 0.01,
        }[spec.family]
        return 1e-6 if best else 1e-2

    cfg_dir, root, specs, hashes = _write_scan(tmp_path, grid, final_of=final_of)
    rep = _select(cfg_dir, root, grid)
    assert not rep["errors"], rep["errors"]
    # every family is represented: svd, Adam, SOAP, LBFGS, PolyakSGD, JD_UPGrad, HIG
    assert set(rep["methods"]) == {"SVD", "Adam", "SOAP", "LBFGS", "PolyakSGD",
                                   "JD_UPGrad", "HIG"}, sorted(rep["methods"])

    with reconcile.ConfigLoader(cfg_dir) as loader:
        for method, sel in sorted(rep["methods"].items()):
            assert sel["verified"], (method, sel["verify_error"])
            assert len(sel["run_ids"]) == N_SEEDS, (method, sel["run_ids"])
            rcfg = loader.compose(SCAN, sel["overrides"])
            got = grid.expand_grid(rcfg, verbose=False)
            assert sorted(s.run_id for s in got) == sel["run_ids"], method
            assert {s.run_id: grid.hash8(s, rcfg) for s in got} == \
                   {r: hashes[r] for r in sel["run_ids"]}, method
            # and it really is the configuration that won
            assert sel["seed_mean_final_val"] == pytest.approx(1e-6), method


def test_selection_ignores_the_test_split(tmp_path, grid):
    """Validation decides; a test loss ranking the configurations the other way does not.

    EXPERIMENTS.md section 5: "Selection still uses validation only; test numbers are
    for reporting the selected configuration."
    """
    def final_of(spec):
        if spec.family == "polyak":         # no lr axis: its single config is the winner
            return 1e-6
        return 1e-6 if spec.hparams.get("lr") in (1.0, 0.01, 0.5, 0.1) else 1e-2

    def test_of(spec):
        # exactly inverted: the val winner is the test loser, by four orders of magnitude
        return 1e-2 if final_of(spec) == 1e-6 else 1e-6

    cfg_dir, root, _specs, _h = _write_scan(tmp_path, grid, final_of=final_of,
                                            test_of=test_of)
    rep = _select(cfg_dir, root, grid)
    select_best.assert_no_test_metric(rep)          # the rule, as an assertion
    for method, sel in rep["methods"].items():
        assert sel["seed_mean_final_val"] == pytest.approx(1e-6), method
        for key in select_best.TEST_KEYS:
            assert key not in sel, (method, key)
        assert not any(k in json.dumps(sel) for k in ('"test":', '"test_acc":'))


# ---------------------------------------------------------------------------
# The ranking
# ---------------------------------------------------------------------------

def test_fewest_diverged_beats_a_lower_seed_mean(tmp_path, grid):
    """A configuration that blows up on 2 of 3 seeds does not beat one that finishes all.

    Dropping a configuration's failures from its mean flatters it, which is why section 1
    puts `fewest diverged` ahead of the mean -- and why this tool does not simply reuse
    `reconcile.best_configs`, which has only the mean tier. `--rule seed_mean` reproduces
    reconcile's answer, and the disagreement is reported either way.
    """
    def final_of(spec):
        if spec.family != "standard" or spec.hparams["optim_name"] != "Adam":
            return 1e-2
        return 1e-6 if spec.hparams["lr"] == 0.01 else 1e-4

    def status_of(spec):
        # the low-mean Adam configuration survives only one seed out of three
        if (spec.family == "standard" and spec.hparams["optim_name"] == "Adam"
                and spec.hparams["lr"] == 0.01 and spec.model_seed != 0):
            return "diverged"
        return "ok"

    cfg_dir, root, _s, _h = _write_scan(tmp_path, grid, final_of=final_of,
                                        status_of=status_of)
    full = _select(cfg_dir, root, grid, rule="full")
    # 1 of 3 finished is NOT eligible (config_eligible needs more than half), so the
    # blow-up cannot win under either rule; the survivor is the lr 0.001 configuration
    assert full["methods"]["Adam"]["seed_mean_final_val"] == pytest.approx(1e-4)
    assert full["methods"]["Adam"]["n_diverged"] == 0

    # now make it eligible (2 of 3) and still worse on divergences
    def status_2of3(spec):
        if (spec.family == "standard" and spec.hparams["optim_name"] == "Adam"
                and spec.hparams["lr"] == 0.01 and spec.model_seed == 2):
            return "diverged"
        return "ok"

    cfg_dir, root, _s, _h = _write_scan(tmp_path, grid, final_of=final_of,
                                        status_of=status_2of3)
    full = _select(cfg_dir, root, grid, rule="full")
    mean_only = _select(cfg_dir, root, grid, rule="seed_mean")
    assert full["methods"]["Adam"]["n_diverged"] == 0
    assert full["methods"]["Adam"]["seed_mean_final_val"] == pytest.approx(1e-4)
    assert mean_only["methods"]["Adam"]["n_diverged"] == 1
    assert mean_only["methods"]["Adam"]["seed_mean_final_val"] == pytest.approx(1e-6)
    # and the difference is reported, not silent
    assert any("Adam" in d for d in full["reconcile_disagreements"]), \
        full["reconcile_disagreements"]


def test_exact_ties_break_deterministically_to_the_cheapest_config(tmp_path, grid):
    """Sven ties are the normal case; the tie-break must be stable and cheap.

    Whenever the rtol-rank stays below k, every larger k is the same trajectory, so the
    grid is full of exact ties. The tie-break is `scan_analysis.Scan.configs`'s -- smallest
    k, then largest rtol -- extended so the answer never depends on file order.
    """
    cfg_dir, root, _s, _h = _write_scan(tmp_path, grid,
                                        final_of=lambda spec: 1e-6)   # everything ties
    first = _select(cfg_dir, root, grid)["methods"]["SVD"]
    second = _select(cfg_dir, root, grid)["methods"]["SVD"]
    assert first["config_key"] == second["config_key"]
    assert first["hparams"]["k"] == 2, first["config_key"]      # smallest k
    assert first["hparams"]["rtol"] == 0.01, first["config_key"]  # largest rtol
    assert first["tied_with"], "a 24-way tie should be recorded as such"


def test_a_config_the_grid_no_longer_has_cannot_win(tmp_path, grid):
    """An off-grid leftover is ignored, however good it looks.

    A method whose grid has moved leaves records of the old points on disk. They have no
    expected seed count, so eligibility would pass a single lucky seed, and Phase 5 would
    then time a configuration the current config cannot even describe.
    """
    stale = []
    for seed in range(N_SEEDS):
        run_id = f"std_bs8_lr0.5_optimAdam_mseed{seed}_lseed7"
        stale.append((run_id, {
            "run_id": run_id, "schema_version": 2, "status": "ok", "optimizer": "Adam",
            "model_seed": seed, "lr": 0.5, "wall_time_s": 1.0,
            "losses": {"train": _curve(1e-12), "val": _curve(1e-12)},
        }))
    cfg_dir, root, _s, _h = _write_scan(tmp_path, grid,
                                        final_of=lambda spec: 1e-2,
                                        extra_records=stale)
    rep = _select(cfg_dir, root, grid)
    assert rep["n_off_grid_records"] == N_SEEDS, rep["n_off_grid_records"]
    assert "lr0.5" not in rep["methods"]["Adam"]["config_key"]
    assert rep["methods"]["Adam"]["seed_mean_final_val"] == pytest.approx(1e-2)


# ---------------------------------------------------------------------------
# Override literals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [0.05, 1e-05, 3e-05, 0.0001, 3e-08, 0.3, 1.0, 2.0,
                                   0.02, 0.015, 0.0005])
def test_fmt_round_trips_through_hydra_and_the_run_id(tmp_path, grid, value):
    """`fmt` must produce a literal hydra parses back to the same float.

    The run_id embeds `str(float)` of whatever hydra parsed (`lr1e-05`, `tau0.0001`), so a
    value that does not round-trip renames the run and the timing pass writes a run_id the
    scan does not have.
    """
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / f"{SCAN}.yaml").write_text(TINY_CONFIG)
    with reconcile.ConfigLoader(str(cfg_dir)) as loader:
        rcfg = loader.compose(SCAN, f"mode=standard optimizers_standard=[Adam] "
                                    f"lrs_standard=[{select_best.fmt(value)}]")
        assert rcfg["lrs_standard"] == [value]
        specs = grid.expand_grid(rcfg, verbose=False)
        assert len(specs) == N_SEEDS
        assert all(f"_lr{value}_" in s.run_id for s in specs), specs[0].run_id


# ---------------------------------------------------------------------------
# The real configs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scan", ["toy_1d_scan", "cifar10_resnet_ce_scan",
                                  "exp_nanogpt_speedrun"])
def test_every_family_of_a_real_headline_config_builds_a_verified_override(scan, grid):
    """One representative run per family of a real scan, through the real code path.

    Covers what the synthetic config cannot: `use_gram` + `bn_mode: batch` (the `_gram`
    and `_bnbatch` run_id tokens), the CIFAR `gram_capture: full` settings, and the
    language-model `vocab_size` injection that decides a nanoGPT run's hash.
    """
    config_dir = os.path.join(REPO, "experiments", "configs")
    with reconcile.ConfigLoader(config_dir) as loader:
        rcfg = select_best.compose_with_facts(loader, grid, scan, "mode=all", {})
        specs = grid.expand_grid(rcfg, verbose=False)
        seen = set()
        for spec in specs:
            if spec.family in seen:
                continue
            seen.add(spec.family)
            overrides = select_best.build_overrides(spec, rcfg)
            want = {s.run_id: grid.hash8(s, rcfg) for s in specs
                    if reconcile.config_key(s.run_id) == reconcile.config_key(spec.run_id)}
            ok, why = select_best.verify_overrides(scan, overrides, want, loader, grid, {})
            assert ok, f"{scan} {spec.family}: {overrides!r} -> {why}"
        assert seen, scan
