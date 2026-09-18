"""`tools/reconcile.py` on a synthetic scan built from `grid.expand_grid`.

CPU only, temp dirs only, no SLURM, no torch: `reconcile` is loaded by path because
`tools/` is not a package (and must not become one -- the campaign runs these as
scripts from an exported snapshot).

The acceptance test the contract asks for is
:func:`test_deleting_one_result_reports_exactly_that_run_never_started`: build a FINISHED
toy scan in a temp root, delete one result and its marker, and require that reconcile
names exactly that run as never-started and exits non-zero.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time

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

# A standalone scan config: no `defaults:`, so it composes without the dataset/model
# groups and cannot be broken by another track editing the real configs.
TINY_CONFIG = """
name: scan
mode: both
device: cpu
print_config: false
num_epochs: 3
batch_size: 8
result_id_fields: []
loss: mse
use_gram: true
k_values: [2, 4]
lrs: [0.1, 1.0]
rtol: [0.001]
svd_mode: [torch]
variable_k: false
lrs_standard: [0.001, 0.01]
optimizers_standard: [Adam, LBFGS]
lrs_lbfgs: [0.1]
lbfgs_max_iter: [1]
lbfgs_history_size: [2]
lbfgs_line_search_fn: [strong_wolfe]
loader_seed: 7
model_seeds: [0, 1, 2]
data_seed: 7
"""

SCAN = "tiny_scan"
#: 12 svd (2 k x 2 lr x 3 seeds) + 6 standard (Adam, 2 lr x 3 seeds) + 3 lbfgs
N_EXPECTED = 21


def _val_curve(final):
    """A healthy 3-epoch val curve (index 0 = untrained) ending at `final`."""
    return [1.0, 0.5, 0.1, final]


@pytest.fixture(scope="module")
def grid():
    return reconcile.load_grid()


@pytest.fixture
def scan(tmp_path_factory, grid):
    """A finished synthetic scan: records, done markers and a manifest for every spec.

    The best SVD configuration is deliberately the one at the TOP of both the k and the
    lr grid, so the edge report has something to find.
    """
    base = tmp_path_factory.mktemp("recon")
    cfg_dir = base / "configs"
    cfg_dir.mkdir()
    (cfg_dir / f"{SCAN}.yaml").write_text(TINY_CONFIG)
    root = base / "results"
    scan_dir = root / SCAN
    for sub in ("done", "started", "claims", "manifest", "diag"):
        (scan_dir / sub).mkdir(parents=True)

    loader = reconcile.ConfigLoader(str(cfg_dir))
    with loader:
        rcfg = loader.compose(SCAN, "")
        specs = grid.expand_grid(rcfg, verbose=False)
        hashes = {s.run_id: grid.hash8(s, rcfg) for s in specs}
    assert len(specs) == N_EXPECTED, len(specs)

    for spec in specs:
        # SVD at lr 1.0 / k 4 is the winner; everything else is an order of magnitude worse
        if spec.family == "svd":
            good = spec.hparams["lr"] == 1.0 and spec.hparams["k"] == 4
            final = 1e-6 if good else 1e-3
        else:
            final = 1e-2
        record = {"run_id": spec.run_id, **spec.record_extra, "status": "ok",
                  "losses": {"train": [0.4, 0.2, final], "val": _val_curve(final)}}
        (scan_dir / f"{spec.run_id}.jsonl").write_text(json.dumps(record) + "\n")
        (scan_dir / "done" / f"{spec.run_id}.{hashes[spec.run_id]}.ok").touch()
    (scan_dir / "manifest" / "job1.shard0.json").write_text(
        json.dumps({"run_ids": [s.run_id for s in specs], "job": "job1", "n_runs": len(specs)}))

    return {"base": base, "cfg_dir": str(cfg_dir), "root": str(root),
            "scan_dir": scan_dir, "specs": specs, "hashes": hashes}


def _run(scan, **kwargs):
    """reconcile_scan on the fixture, with its own ConfigLoader."""
    grid_mod = reconcile.load_grid()
    with reconcile.ConfigLoader(scan["cfg_dir"]) as loader:
        return reconcile.reconcile_scan(SCAN, [""], scan["root"], grid=grid_mod,
                                        loader=loader, quiet=True, **kwargs)


def test_a_finished_scan_reconciles_clean(scan):
    rep = _run(scan)
    assert rep["n_expected"] == N_EXPECTED
    assert rep["totals"]["ok"] == N_EXPECTED
    assert rep["n_missing"] == 0
    assert rep["ok"] is True
    assert rep["never_started"] == []
    assert rep["warnings"] == [] and rep["fatal"] == []
    assert rep["counts"]["svd"]["expected"] == 12
    assert rep["counts"]["standard"]["expected"] == 6
    assert rep["counts"]["lbfgs"]["expected"] == 3


def test_deleting_one_result_reports_exactly_that_run_never_started(scan):
    """THE acceptance test (spec): one deleted result + marker, nothing else."""
    victim = next(s for s in scan["specs"] if s.family == "svd")
    (scan["scan_dir"] / f"{victim.run_id}.jsonl").unlink()
    (scan["scan_dir"] / "done" / f"{victim.run_id}.{scan['hashes'][victim.run_id]}.ok").unlink()

    rep = _run(scan)
    assert rep["never_started"] == [victim.run_id]
    assert rep["n_missing"] == 1
    assert rep["ok"] is False
    assert rep["totals"]["ok"] == N_EXPECTED - 1
    for cls in ("started_only", "stale_hash", "jsonl_only", "oom", "error"):
        assert rep["totals"][cls] == 0, cls
    assert rep["counts"]["svd"]["never_started"] == 1

    # ... and the CLI says so with a non-zero exit code
    proc = subprocess.run(
        [sys.executable, os.path.join(TOOLS, "reconcile.py"), SCAN,
         "--root", scan["root"], "--config-dir", scan["cfg_dir"]],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert victim.run_id in proc.stdout
    assert "WORK REMAINS" in proc.stdout
    assert "1 run(s) to do" in proc.stdout


def test_a_result_without_its_marker_is_jsonl_only_not_ok(scan):
    """The jsonl is no longer the dedup key: the marker is (C-R3)."""
    victim = next(s for s in scan["specs"] if s.family == "lbfgs")
    (scan["scan_dir"] / "done" / f"{victim.run_id}.{scan['hashes'][victim.run_id]}.ok").unlink()
    rep = _run(scan)
    assert rep["jsonl_only"] == [victim.run_id]
    assert rep["never_started"] == []
    assert rep["ok"] is False


def test_a_marker_under_another_hash_is_stale_not_done(scan):
    victim = next(s for s in scan["specs"] if s.family == "standard")
    old = scan["scan_dir"] / "done" / f"{victim.run_id}.{scan['hashes'][victim.run_id]}.ok"
    old.rename(old.with_name(f"{victim.run_id}.deadbeef.ok"))
    rep = _run(scan)
    assert rep["stale_hash"] == [victim.run_id]
    assert rep["totals"]["ok"] == N_EXPECTED - 1
    assert rep["ok"] is False


def test_oom_error_and_started_markers_are_each_their_own_class(scan):
    a, b, c = scan["specs"][0], scan["specs"][1], scan["specs"][2]
    for spec, status in ((a, "oom"), (b, "error")):
        done = scan["scan_dir"] / "done"
        (done / f"{spec.run_id}.{scan['hashes'][spec.run_id]}.ok").unlink()
        (done / f"{spec.run_id}.{scan['hashes'][spec.run_id]}.{status}").touch()
    (scan["scan_dir"] / "done" / f"{c.run_id}.{scan['hashes'][c.run_id]}.ok").unlink()
    (scan["scan_dir"] / f"{c.run_id}.jsonl").unlink()
    (scan["scan_dir"] / "started" / f"{c.run_id}.started").touch()

    rep = _run(scan)
    assert rep["oom"] == [a.run_id]
    assert rep["error"] == [b.run_id]
    assert rep["started_only"] == [c.run_id]
    assert rep["n_missing"] == 3
    assert rep["ok"] is False


def test_a_fresh_claim_is_live_and_a_stale_claim_is_not(scan):
    victim = scan["specs"][5]
    (scan["scan_dir"] / "done" / f"{victim.run_id}.{scan['hashes'][victim.run_id]}.ok").unlink()
    (scan["scan_dir"] / f"{victim.run_id}.jsonl").unlink()
    claim = scan["scan_dir"] / "claims" / f"{victim.run_id}.claim"
    claim.write_text("host pid jobid")

    rep = _run(scan)
    assert rep["totals"]["claimed_live"] == 1
    assert rep["n_missing"] == 0        # a live worker has it: nothing to launch for
    assert rep["ok"] is True

    old = time.time() - 2 * reconcile.CLAIM_TIMEOUT_S
    os.utime(claim, (old, old))
    rep = _run(scan)
    assert rep["totals"]["claimed_live"] == 0
    assert rep["never_started"] == [victim.run_id]
    assert rep["ok"] is False

    # a takeover generation is also a claim
    claim.rename(claim.with_name(claim.name + ".1"))
    os.utime(claim.with_name(claim.name + ".1"), None)
    rep = _run(scan)
    assert rep["totals"]["claimed_live"] == 1


def test_diverged_runs_are_counted_and_left_out_of_the_seed_mean(scan):
    """A finite blow-up is diverged (style.is_diverged) and must not win a comparison."""
    winners = [s for s in scan["specs"]
               if s.family == "svd" and s.hparams["lr"] == 1.0 and s.hparams["k"] == 4]
    assert len(winners) == 3
    for spec in winners[:2]:                 # 2 of 3 seeds blow up -> not eligible
        record = json.loads((scan["scan_dir"] / f"{spec.run_id}.jsonl").read_text())
        record["losses"]["val"] = [1.0, 2.0, 5.0, 100.0]        # > 10 x val[0]
        record["losses"]["train"] = [0.4, 2.0, 50.0]
        (scan["scan_dir"] / f"{spec.run_id}.jsonl").write_text(json.dumps(record) + "\n")
        done = scan["scan_dir"] / "done"
        (done / f"{spec.run_id}.{scan['hashes'][spec.run_id]}.ok").unlink()
        (done / f"{spec.run_id}.{scan['hashes'][spec.run_id]}.diverged").touch()

    rep = _run(scan)
    assert rep["totals"]["diverged"] == 2
    assert rep["n_missing"] == 0             # diverged is a finished run, not missing
    # 1 of 3 seeds left => the winning configuration is no longer ELIGIBLE, so the best
    # SVD config is one of the 1e-3 ones -- a method is never ranked on one lucky seed.
    best = rep["best"]["SVD"]
    assert best["config_key"] != reconcile.config_key(winners[0].run_id)
    assert best["seed_mean"] == pytest.approx(1e-3)
    assert best["n_ok"] == 3


def test_best_config_and_edge_report(scan):
    rep = _run(scan)
    best = rep["best"]
    assert set(best) == {"SVD", "Adam", "LBFGS"}
    svd = best["SVD"]
    # tau is reported for every method (HIG is the one that has it) -- grid_counts.md
    # counts tau as an axis of the hig grid, so C-B3 must see its verdict too
    assert svd["values"] == {"lr": 1.0, "k": 4, "rtol": 0.001, "tau": None}
    assert svd["n_ok"] == 3 and svd["n_expected"] == 3
    # lr 1.0 is the top of [0.1, 1.0] and k 4 the top of [2, 4]; rtol has ONE value,
    # so it is not an edge -- a single-point axis cannot be extended by C-B3.
    assert svd["edges"] == {"lr": "EDGE-HIGH", "k": "EDGE-HIGH"}
    assert svd["in_grid"] is True
    # Adam's best lr is 0.001 = the bottom of [0.001, 0.01]
    assert best["Adam"]["edges"] == {"lr": "EDGE-LOW"}
    assert best["SVD"]["seed_mean"] < best["Adam"]["seed_mean"]
    assert rep["n_off_grid_records"] == 0


def test_a_record_off_the_current_grid_cannot_be_a_methods_best_config(scan):
    """A grid that MOVED leaves old records on disk; they must not win.

    They have no expected seed count, so `config_eligible` would pass a single lucky
    seed, and its off-grid lr would then be skipped by the edge check -- reported as an
    interior best config that the extension round would key off (C-B3).
    """
    stray = "svd_bs8_k4_lr0.9_rtol0.001_svdtorch_mseed0_lseed7"
    (scan["scan_dir"] / f"{stray}.jsonl").write_text(json.dumps(
        {"run_id": stray, "optimizer": "SVD", "lr": 0.9, "k": 4, "rtol": 0.001,
         "batch_size": 8, "status": "ok",
         "losses": {"train": [0.4, 1e-9], "val": _val_curve(1e-9)}}) + "\n")

    rep = _run(scan, use_manifest=False)
    assert rep["n_off_grid_records"] == 1
    assert rep["best"]["SVD"]["values"]["lr"] == 1.0          # the on-grid winner
    assert rep["best"]["SVD"]["seed_mean"] == pytest.approx(1e-6)
    assert stray in rep["unexpected_jsonl"]
    assert rep["ok"] is True                                  # nothing to LAUNCH for it


def _hig_expected(taus=(1e-4, 1e-3), lr=0.1):
    out = {}
    for tau in taus:
        for seed in (0, 1, 2):
            out[f"hig_bs8_lr{lr}_tau{tau}_mseed{seed}_lseed7"] = {
                "family": "hig", "optimizer": "HIG", "batch_size": 8,
                "hparams": {"lr": lr, "tau": tau}, "hash8": "h"}
    return out


def test_tau_is_edge_checked_for_hig():
    """HIG's tau is an axis C-B3 extends, so its best must carry a verdict.

    (`grid_counts.md` counts tau as an axis of the hig grid; before this it was the one
    swept axis with no edge report, so a HIG best sitting at the top tau looked interior.)
    """
    expected = _hig_expected()
    records = {rid: {"optimizer": "HIG", "lr": 0.1, "tau": info["hparams"]["tau"],
                     "batch_size": 8,
                     "losses": {"train": [0.4, 0.1],
                                "val": _val_curve(1e-5 if info["hparams"]["tau"] == 1e-3
                                                  else 1e-2)}}
               for rid, info in expected.items()}
    best, n_off = reconcile.best_configs(records, expected)
    assert n_off == 0
    assert best["HIG"]["values"]["tau"] == 1e-3
    assert best["HIG"]["edges"] == {"tau": "EDGE-HIGH"}      # lr has ONE value: no edge


def test_a_value_the_grid_does_not_contain_is_reported_off_grid_not_interior():
    """An axis whose value is not on the grid it is compared against gets a verdict of
    its own ("OFF-GRID"), never a silently missing one."""
    expected = _hig_expected(taus=(1e-4,))
    # the records are the expected runs, but every one of them RECORDS a tau the config
    # no longer lists (a config edited after the runs were made)
    records = {rid: {"optimizer": "HIG", "lr": 0.1, "tau": 7e-3, "batch_size": 8,
                     "losses": {"train": [0.4, 0.1], "val": _val_curve(1e-5)}}
               for rid in expected}
    best, _ = reconcile.best_configs(records, expected)
    assert best["HIG"]["edges"] == {"tau": "OFF-GRID"}


def test_manifest_only_runs_are_expected_even_when_the_config_no_longer_has_them(scan):
    """A run the jobs declared but the config has since dropped is still expected."""
    ghost = "svd_bs8_k99_lr0.1_rtol0.001_svdtorch_mseed0_lseed7"
    path = scan["scan_dir"] / "manifest" / "job2.shard0.json"
    path.write_text(json.dumps({"run_ids": [ghost], "job": "job2"}))
    rep = _run(scan)
    assert rep["n_expected"] == N_EXPECTED + 1
    assert rep["n_from_manifest"] == N_EXPECTED + 1
    assert rep["never_started"] == [ghost]
    assert rep["ok"] is False
    # ... and --no-manifest goes back to the config's own grid
    rep2 = _run(scan, use_manifest=False)
    assert rep2["n_expected"] == N_EXPECTED and rep2["ok"] is True


def test_unexpected_records_are_reported_but_are_not_missing_work(scan):
    (scan["scan_dir"] / "hig_bs8_lr0.5_tau0.0001_mseed0_lseed7.jsonl").write_text(
        json.dumps({"run_id": "x", "optimizer": "HIG", "losses": {"val": [1.0, 0.1],
                                                                  "train": [0.5]}}) + "\n")
    rep = _run(scan, use_manifest=False)
    assert len(rep["unexpected_jsonl"]) == 1
    assert rep["n_missing"] == 0 and rep["ok"] is True
    assert rep["best"]["HIG"]["in_grid"] is False       # nothing to be on the edge of


def test_one_listdir_per_directory(scan, monkeypatch):
    """The counts come from four listings, whatever the grid size (C-R2)."""
    real = os.listdir
    seen = []

    def counting(path):
        seen.append(str(path))
        return real(path)

    monkeypatch.setattr(os, "listdir", counting)
    reconcile.disk_state(str(scan["scan_dir"]))
    assert len(seen) == 4, seen
    assert sorted(os.path.basename(p) for p in seen[:3]) == ["claims", "done", "started"]


def test_missing_scan_directory_is_all_never_started_not_a_crash(scan):
    grid_mod = reconcile.load_grid()
    with reconcile.ConfigLoader(scan["cfg_dir"]) as loader:
        rep = reconcile.reconcile_scan("tiny_scan", [""], os.path.join(scan["root"], "nope"),
                                       grid=grid_mod, loader=loader, quiet=True)
    assert rep["totals"]["never_started"] == N_EXPECTED and rep["ok"] is False


def test_the_plan_decides_the_results_root_not_the_ambient_env(scan, tmp_path, monkeypatch):
    """`--all <plan>` must reconcile the root the launcher exports to the workers.

    CAMPAIGN_STATUS.md tells people to work with `SV3_RESULTS_ROOT` exported at the
    FROZEN legacy root; reconciling the fresh campaign plan against it would report every
    P0 scan as never-started.
    """
    plan = tmp_path / "plan_root.yaml"
    plan.write_text(
        f"version: 1\nplan: t\nresults_root: {scan['root']}\n"
        "lanes:\n  a100: {partition: p, gres: 'gpu:1'}\n"
        "work_lists:\n  - {name: l, phase: P0, lane: a100, items: "
        f"[{{config: {SCAN}, overrides: '', nproc: 1}}]}}\n")
    legacy = tmp_path / "legacy-root"
    legacy.mkdir()
    monkeypatch.setenv("SV3_RESULTS_ROOT", str(legacy))

    assert reconcile.resolve_root(None, _plan(plan))[0] == str(scan["root"])
    assert reconcile.resolve_root(str(legacy), _plan(plan))[0] == str(legacy)

    proc = subprocess.run(
        [sys.executable, os.path.join(TOOLS, "reconcile.py"), "--all", str(plan),
         "--config-dir", scan["cfg_dir"], "--no-best"],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"results root {scan['root']}" in proc.stdout
    assert "the plan's results_root" in proc.stdout
    assert str(legacy) in proc.stderr          # the disagreement is said out loud


def _plan(path):
    sys.path.insert(0, TOOLS)
    import campaign_plan
    return campaign_plan.load_plan(path)


def test_the_snapshot_supplies_the_configs_when_one_is_named(scan, tmp_path, monkeypatch):
    """Composing the live configs for jobs that run the snapshot's is how a complete
    scan comes back as `stale-hash`: the config dir follows the snapshot and is printed."""
    snap = tmp_path / "snap"
    (snap / "experiments" / "configs").mkdir(parents=True)
    (snap / "experiments" / "configs" / f"{SCAN}.yaml").write_text(TINY_CONFIG)
    got, source = reconcile.resolve_config_dir(None, str(snap))
    assert got == str(snap / "experiments" / "configs") and source == "the snapshot's"
    # worker_pool.sh exports SV3_SNAPSHOT, so a reconcile inside a job finds it too
    monkeypatch.setenv("SV3_SNAPSHOT", str(snap))
    assert reconcile.resolve_config_dir()[1] == "the snapshot's"
    monkeypatch.delenv("SV3_SNAPSHOT")
    assert reconcile.resolve_config_dir()[1] == "this tree's"

    proc = subprocess.run(
        [sys.executable, os.path.join(TOOLS, "reconcile.py"), SCAN,
         "--root", scan["root"], "--snapshot", str(snap), "--no-best"],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"configs      {snap}/experiments/configs   (the snapshot's)" in proc.stdout
    assert f"configs    {snap}/experiments/configs" in proc.stdout      # per scan


def test_a_broken_override_is_a_warning_not_a_silent_empty_grid(scan):
    rep = _run(scan, use_manifest=False)
    assert rep["ok"] is True
    grid_mod = reconcile.load_grid()
    with reconcile.ConfigLoader(scan["cfg_dir"]) as loader:
        rep = reconcile.reconcile_scan(SCAN, ["mode=nonsense"], scan["root"], grid=grid_mod,
                                       loader=loader, quiet=True, use_manifest=False)
    assert rep["n_expected"] == 0
    assert any("nonsense" in w for w in rep["fatal"])
    assert rep["ok"] is False
    # "cannot answer" is exit 2, NOT "work remains" -- a chain must not loop on it
    proc = subprocess.run(
        [sys.executable, os.path.join(TOOLS, "reconcile.py"), SCAN, "--root", scan["root"],
         "--config-dir", scan["cfg_dir"], "--config-overrides", "mode=nonsense",
         "--no-manifest"], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "CANNOT ANSWER" in proc.stdout
