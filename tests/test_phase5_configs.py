"""The Phase-5 companion configs: `*_timing`, `*_diag`, `*_confirm`.

CPU only, hydra composition only, no torch. What each pass has to guarantee:

* **timing** must be the scan's own run with a stopwatch on it -- same run_ids AND the same
  run hashes, with only logging turned off (C-T1);
* **diag** must be the same runs again, with the full checkpoint ladder and the dense
  spectra schedule -- neither of which enters the run hash, by design (C-L2/C-L3);
* **confirm** must be DIFFERENT runs: fresh model seeds, and for toy and polynomial three
  data seeds whose results cannot collide. `data_seed` is in `grid.run_hash` but NOT in
  the run_id, so without a fix three replicates would write one run_id under three hashes
  and the C-R3 rule would have them retire and re-run each other for ever;
  :func:`test_data_seed_replicates_cannot_collide` is that fix's acceptance test.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")
CONFIG_DIR = os.path.join(REPO, "experiments", "configs")


def _load(name):
    path = os.path.join(TOOLS, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_tools_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


reconcile = _load("reconcile")
select_best = _load("select_best")
gen_plan = _load("gen_phase5_plan")

HEADLINE = select_best.HEADLINE_SCANS
#: scans whose dataset is generated from `data_seed` (F25/F27: 3 data-seed replicates)
DATA_SEED_SCANS = ("toy_1d_scan", "polynomial_scan")
#: one cheap svd grid point, so every composition expands to exactly the seed count
PIN = "mode=svd k_values=[1] lrs=[0.1] rtol=[0.001]"


@pytest.fixture(scope="module")
def grid():
    return reconcile.load_grid()


@pytest.fixture(scope="module")
def loader():
    with reconcile.ConfigLoader(CONFIG_DIR) as l:
        yield l


def _specs(loader, grid, name, overrides):
    rcfg = loader.compose(name, overrides)
    return rcfg, grid.expand_grid(rcfg, verbose=False)


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scan", HEADLINE)
def test_timing_is_the_same_run_with_the_logging_off(scan, loader, grid):
    """Same run_ids, same run hashes; `svd_info`, `checkpoints`, `checkpoints_svd` off.

    The run hash must survive because that is what makes a timing run comparable to the
    scan run at all (`bench/check_timing_join.py` asserts it on the real results); the
    three keys must be off because C-T1 says timing runs write no spectra and no states.
    """
    base_cfg, base = _specs(loader, grid, scan, PIN)
    t_cfg, timing = _specs(loader, grid, f"{scan}_timing",
                           f"{gen_plan.TIMING_OVERRIDES} {PIN}")
    assert t_cfg["svd_info"] == "none"
    assert t_cfg["checkpoints"] == "none"
    assert t_cfg["checkpoints_svd"] is None
    assert [s.run_id for s in base] == [s.run_id for s in timing]
    assert [grid.hash8(s, base_cfg) for s in base] == \
           [grid.hash8(s, t_cfg) for s in timing]


def test_timing_overrides_need_the_append_form(loader):
    """`++`, not `=`. No scan config declares `svd_info`, and the MNIST scans declare
    `checkpoints_svd` but not `checkpoints`, so hydra's struct mode refuses the plain
    form -- the same failure `tools/worker_pool.sh` documents for `scheduler=claims`,
    which took out every runner process of every campaign job."""
    assert gen_plan.TIMING_OVERRIDES.split() == ["++svd_info=none", "++checkpoints=none",
                                                 "++checkpoints_svd=null"]
    with pytest.raises(Exception):
        loader.compose("mnist_scan_ce_timing", "svd_info=none")


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scan", HEADLINE)
def test_diag_logs_everything_without_changing_the_run(scan, loader, grid):
    """Full ladder for EVERY family plus the dense spectra schedule, same runs.

    `checkpoints_svd` has to be cleared or the svd family would keep following the parent
    scan's `log` while the baselines got the new policy; the CIFAR scans use `epochs`
    rather than `log` because a ResNet18 state is 45 MB.
    """
    base_cfg, base = _specs(loader, grid, scan, PIN)
    d_cfg, diag = _specs(loader, grid, f"{scan}_diag", PIN)
    want = "epochs" if scan.startswith("cifar10") else "log"
    assert d_cfg["checkpoints"] == want
    assert d_cfg["checkpoints_svd"] is None
    assert d_cfg["svd_info"] == "full"
    schedule = grid.resolve_spectra_schedule(d_cfg)
    assert schedule["dense_first"] >= 1000, schedule
    assert schedule["every"] == 20, schedule
    # what is LOGGED is outside the run hash (C-L2/C-L3), so these are the scan's runs
    assert [s.run_id for s in base] == [s.run_id for s in diag]
    assert [grid.hash8(s, base_cfg) for s in base] == \
           [grid.hash8(s, d_cfg) for s in diag]


# ---------------------------------------------------------------------------
# confirmation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scan", HEADLINE)
def test_confirm_uses_five_fresh_model_seeds(scan, loader, grid):
    base_cfg, base = _specs(loader, grid, scan, PIN)
    c_cfg, conf = _specs(loader, grid, f"{scan}_confirm", PIN)
    assert len(c_cfg["model_seeds"]) == 5, c_cfg["model_seeds"]
    assert not set(c_cfg["model_seeds"]) & set(base_cfg["model_seeds"]), scan
    # base + 100 .. base + 104
    assert sorted(c_cfg["model_seeds"]) == \
        [min(base_cfg["model_seeds"]) + 100 + i for i in range(5)], scan
    assert not {s.run_id for s in conf} & {s.run_id for s in base}, scan
    # cheap: confirmation reports numbers, not trajectories
    assert c_cfg["checkpoints"] == "final"
    assert c_cfg["checkpoints_svd"] is None
    assert c_cfg["svd_info"] == "summary"


@pytest.mark.parametrize("scan", DATA_SEED_SCANS)
def test_data_seed_replicates_cannot_collide(scan, loader, grid):
    """Three data seeds, three sets of run_ids -- the C-R3 collision, fixed.

    On the PARENT config `data_seed` changes the run HASH and not the run_id, which is
    precisely the state that makes two jobs retire each other's results to `_stale/` and
    re-run them for ever. The `_confirm` config puts `data_seed` into `result_id_fields`,
    which is one line against one extra results directory and one extra config per
    replicate.
    """
    base = loader.compose(scan, "")
    ds = int(base["data_seed"])

    # the hazard, stated: same run_id, different hash
    cfg_a, specs_a = _specs(loader, grid, scan, f"{PIN} data_seed={ds}")
    cfg_b, specs_b = _specs(loader, grid, scan, f"{PIN} data_seed={ds + 1}")
    assert [s.run_id for s in specs_a] == [s.run_id for s in specs_b], \
        "parent config: data_seed is not in the run_id (this is the hazard)"
    assert [grid.hash8(s, cfg_a) for s in specs_a] != \
           [grid.hash8(s, cfg_b) for s in specs_b], \
        "parent config: data_seed IS in the run hash (this is why they fight)"

    # the fix
    c_cfg = loader.compose(f"{scan}_confirm", "")
    assert "data_seed" in (c_cfg["result_id_fields"] or []), scan
    ids = []
    for i in range(3):
        cfg, specs = _specs(loader, grid, f"{scan}_confirm",
                            f"{PIN} data_seed={ds + i}")
        assert all(f"_data_seed{ds + i}" in s.run_id for s in specs), specs[0].run_id
        ids.append({s.run_id for s in specs})
    assert len(ids[0] | ids[1] | ids[2]) == sum(len(s) for s in ids), \
        "the three data-seed replicates share a run_id"


def test_generator_refuses_a_confirm_config_that_would_collide(loader):
    """`data_seeds_of` is the guard: a `data_seed` that is not in `result_id_fields`
    stops the generator rather than producing items that overwrite each other."""
    seeds = gen_plan.data_seeds_of("toy_1d_scan", loader, 3)
    base = int(loader.compose("toy_1d_scan", "")["data_seed"])
    assert seeds == [base, base + 1, base + 2]
    # MNIST has no data_seed at all: no replicates, and no error
    assert gen_plan.data_seeds_of("mnist_scan_ce", loader, 3) == []
