"""`tools/campaign_plan.py` + `tools/launch_campaign.py`: plan parsing, the printed
sbatch lines, the MIG-lane cap (against a FAKE squeue) and the chain script.

CPU only, temp dirs only. Nothing is ever submitted: every test runs the launcher in its
default dry-run mode, except the cap tests, which are only allowed to reach a fake
`squeue` on a temporary PATH -- there is no fake `sbatch` anywhere in this file, so a
regression that submitted something would fail with FileNotFoundError rather than
touching the real queue.
"""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")
PLAN_FILE = os.path.join(REPO, "campaign", "plan_campaign.yaml")


def _load(name):
    path = os.path.join(TOOLS, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_tools_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


campaign_plan = _load("campaign_plan")
launch = _load("launch_campaign")

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
lrs_standard: [0.001, 0.01]
optimizers_standard: [Adam, LBFGS]
lrs_lbfgs: [0.1]
lbfgs_max_iter: [1]
lbfgs_history_size: [2]
loader_seed: 7
model_seeds: [0, 1, 2]
"""

TINY_PLAN = """
version: 1
plan: tiny
results_root: {root}
lanes:
  a100:
    partition: "iaifi_gpu_priority,iaifi_gpu,gpu"
    gres: "gpu:1"
    mem: 64G
    time: "24:00:00"
    cpus_extra: 2
  mig:
    partition: gpu_test
    gres: "gpu:4"
    cpus: 32
    mem: 128G
    time: "12:00:00"
    max_jobs: 2
work_lists:
  - name: l_a100
    phase: P0
    lane: a100
    n_jobs: 2
    items:
      - {{config: tiny_scan, overrides: "mode=svd", nproc: {{a100: 12, mig: 6}}}}
      - {{config: tiny_scan, overrides: "mode=standard optimizers_standard=[LBFGS]", nproc: 4}}
      - {{config: tiny_scan, overrides: "mode=jd", nproc: 4, enabled: false, note: parked}}
  - name: l_mig
    phase: P1
    lane: mig
    n_jobs: 2
    items:
      - {{config: tiny_scan, overrides: "mode=svd", nproc: {{a100: 12, mig: 6}}}}
"""


@pytest.fixture
def sandbox(tmp_path):
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "tiny_scan.yaml").write_text(TINY_CONFIG)
    snap = tmp_path / "snap"                  # a snapshot the worker-pool guard accepts
    (snap / "tools").mkdir(parents=True)
    (snap / "tools" / "worker_pool.sh").write_text("#!/bin/bash\nexit 0\n")
    (snap / "experiments").mkdir()
    (snap / "experiments" / "__init__.py").touch()
    (snap / "sven" / "sven").mkdir(parents=True)
    (snap / "sven" / "sven" / "__init__.py").touch()
    (snap / "run.py").write_text("raise SystemExit(0)\n")
    (snap / ".deploy_complete").write_text("now")
    root = tmp_path / "results"
    root.mkdir()
    plan = tmp_path / "plan_tiny.yaml"
    plan.write_text(TINY_PLAN.format(root=root))
    work = tmp_path / "work"
    logs = tmp_path / "logs"
    return {"cfg": cfg, "snap": snap, "root": root, "plan": plan, "work": work,
            "logs": logs, "tmp": tmp_path}


def _fake_squeue(tmp_path, n_jobs, *, partition="gpu_test", fail=False, names=()):
    """A `squeue` on PATH that reports `n_jobs` of ours in `partition`.

    `names` are reported for the job-NAME query (`-o '%i %T %j'`, no partition), which is
    how the launcher notices that a work list is already running.
    """
    d = tmp_path / f"bin{n_jobs}{'f' if fail else ''}{'-'.join(names)}"
    d.mkdir(exist_ok=True)
    lines = "\n".join(f"{47000000 + i} RUNNING" for i in range(n_jobs))
    named = "\n".join(f"{47000100 + i} RUNNING {n}" for i, n in enumerate(names))
    script = "#!/bin/bash\n"
    if fail:
        script += "echo 'squeue: error: Invalid partition' >&2\nexit 1\n"
    else:
        script += f"""case "$*" in
  *{partition}*) printf '%s' {json.dumps(lines)}; [ -n {json.dumps(lines)} ] && echo ;;
  *%j*) printf '%s' {json.dumps(named)}; [ -n {json.dumps(named)} ] && echo ;;
esac
exit 0
"""
    p = d / "squeue"
    p.write_text(script)
    p.chmod(0o755)
    return d


def _launch(sandbox, *args, extra_path=None, monkeypatch=None):
    argv = [str(sandbox["plan"]), "--snapshot", str(sandbox["snap"]),
            "--work-dir", str(sandbox["work"]), "--log-dir", str(sandbox["logs"]),
            "--config-dir", str(sandbox["cfg"]), *args]
    if extra_path is not None:
        monkeypatch.setenv("PATH", f"{extra_path}:{os.environ['PATH']}")
    return launch.main(argv)


# ---------------------------------------------------------------------------
# The real campaign plan
# ---------------------------------------------------------------------------

def test_the_campaign_plan_parses_and_covers_the_kept_scans():
    plan = campaign_plan.load_plan(PLAN_FILE)
    # P9 = the one combined, priority-ordered MIG list (gpu_test admits only 2 jobs per user)
    assert {wl.phase for wl in plan.work_lists} == {"P0", "P1", "P2", "P3", "P9"}
    assert set(plan.lanes) == {"a100", "mig"}
    names = [wl.name for wl in plan.work_lists]
    assert len(names) == len(set(names))

    p0 = set(plan.scan_overrides(phase="P0"))
    assert p0 == {"toy_1d_scan", "polynomial_scan", "mnist_scan_labelRegression",
                  "mnist_scan_ce", "cifar10_resnet_scan_labelRegression",
                  "cifar10_resnet_ce_scan", "exp_nanogpt_speedrun"}
    all_scans = set(plan.scan_overrides())
    # the CUT scans must not be anywhere in the plan (CONTRACTS.md "Scope update")
    for cut in ("exp_critbatch_mnist", "exp_critbatch_nanogpt",
                "exp_gpt2_small_comparison", "mnist_scan_brier",
                "cifar10_resnet_kappaScan_labelReg", "cifar10_resnet_ce_kappaScan",
                "cifar10_resnet_paramFrac_scan_labelReg",
                "cifar10_resnet_ce_paramFrac_scan"):
        assert cut not in all_scans, cut
    # P1 keeps the three overparam sweeps, Fig-5 and the batch-size scan
    assert set(plan.scan_overrides(phase="P1")) == {
        "rebuttal_overparam_toy_1d_scan", "rebuttal_overparam_polynomial_scan",
        "rebuttal_overparam_mnist_scan", "rebuttal_batchsize_polynomial_scan",
        "rebuttal_fig5_cifar_paramfrac_scan"}


def test_every_mode_and_family_the_contract_requires_has_an_item():
    plan = campaign_plan.load_plan(PLAN_FILE)
    per_scan = plan.scan_overrides()
    for scan in ("toy_1d_scan", "polynomial_scan", "mnist_scan_labelRegression",
                 "mnist_scan_ce"):
        joined = " ".join(per_scan[scan])
        assert "mode=svd" in joined
        assert "mode=jd" in joined and "mode=hig" in joined, scan
        assert "SOAP,Shampoo,KFAC" in joined, f"{scan} has no second-order item"
        assert "LBFGS" in joined
        assert "AdamW" in joined and "Muon" in joined, f"{scan} lost AdamW/Muon"
    # overparam: one item per n_data, MNIST topping out at 50000 and never 60000
    mnist = " ".join(per_scan["rebuttal_overparam_mnist_scan"])
    for n in (2500, 5000, 10000, 20000, 40000, 50000):
        assert f"n_data={n}" in mnist
    assert "n_data=60000" not in mnist


def test_nproc_follows_the_probe_table():
    plan = campaign_plan.load_plan(PLAN_FILE)
    by = {}
    for wl in plan.work_lists:
        for item in wl.items:
            by[(wl.lane, item.config, item.overrides)] = item.nproc
    assert by[("mig", "toy_1d_scan", "mode=svd")] == 6          # MIG MLP Sven
    assert by[("a100", "toy_1d_scan", "mode=svd")] == 12        # A100 MLP Sven
    assert by[("a100", "cifar10_resnet_scan_labelRegression", "mode=svd")] == 1
    assert by[("a100", "exp_nanogpt_speedrun", "mode=svd")] == 3
    assert by[("a100", "toy_1d_scan", "mode=standard optimizers_standard=[LBFGS]")] == 4
    # CIFAR work never lands in the MIG lane (full capture needs 23 GB, a slice has 19.6)
    for wl in plan.work_lists:
        if wl.lane != "mig":
            continue
        for item in wl.items:
            assert "cifar" not in item.config and "nanogpt" not in item.config


def _grid_counts_table():
    """The per-scan totals of `campaign/grid_counts.md` (the orchestrator's ground truth,
    itself generated from the configs and pinned by tests/test_configs.py)."""
    path = os.path.join(REPO, "campaign", "grid_counts.md")
    if not os.path.exists(path):
        pytest.skip("campaign/grid_counts.md not present")
    out = {}
    for line in open(path):
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 9:
            continue
        scan = cells[0].strip("`")
        try:
            total = int(cells[8].strip("* "))
        except ValueError:
            continue
        out[scan] = total
    assert out, "could not parse campaign/grid_counts.md"
    return out


#: (scan, family) grids that the plan deliberately does NOT launch, with the reason.
#: Anything else missing from a scan's own full grid is work left unlaunched.
DELIBERATE_GAPS = {
    # p3_cifar_jd_hig, parked pending a user decision: HIG on ResNet18 costs a full
    # per-sample Jacobian (~50 GPU-h per scan) and no launcher has ever run these.
    ("cifar10_resnet_ce_scan", "jd"), ("cifar10_resnet_ce_scan", "hig"),
    ("cifar10_resnet_scan_labelRegression", "jd"),
    ("cifar10_resnet_scan_labelRegression", "hig"),
    # Sven-only knob sweeps: the ablation configs inherit a `standard` grid from their
    # parent that is not part of the ablation (grid_counts.md prints `-` for it). The
    # plan launches them with `mode=svd`, which is the whole point of the scan.
    ("mnist_kappaScan_labelRegression", "standard"),
    ("mnist_microbatch_ce_scan", "standard"),
    ("mnist_microbatch_labelreg_scan", "standard"),
    ("mnist_paramfrac_ce_scan", "standard"),
    ("mnist_paramfrac_labelreg_scan", "standard"),
    ("polynomial_microbatch_scan", "standard"),
    ("polynomial_paramfrac_scan", "standard"),
    ("toy_1d_microbatch_scan", "standard"),
    ("toy_1d_paramfrac_scan", "standard"),
    # Fig-5 is a param-fraction sweep of the Sven set point only (3 seeds, user).
    ("rebuttal_fig5_cifar_paramfrac_scan", "standard"),
}


def _plan_and_full_grids(plan):
    """`({scan: {(family, run_id)}} launched, {scan: {...}} described)`.

    The launched side is `expand_grid` over the plan's ENABLED items (de-duplicated: the
    MIG lane and its A100 overflow run the SAME items -- one claim queue, one grid). The
    described side is each config's OWN full grid: `mode=all`, no `optimizers_standard`
    override, summed over the `n_data` values the plan sweeps (n_data changes the dataset
    and is in the run_id, so it cannot be inferred from the config alone).
    """
    import re
    reconcile = _load("reconcile")
    grid = reconcile.load_grid()
    jd = reconcile.has_torchjd()
    launched, n_data = {}, {}
    with reconcile.ConfigLoader(None) as loader:
        for wl in plan.work_lists:
            for it in wl.enabled_items():
                m = re.search(r"n_data=(\S+)", it.overrides)
                if m:
                    n_data.setdefault(it.scan, set()).add(m.group(1))
        seen = set()
        for wl in plan.work_lists:
            for it in wl.enabled_items():
                if (it.scan, it.overrides) in seen:
                    continue
                seen.add((it.scan, it.overrides))
                rcfg = loader.compose(it.config, it.overrides)
                specs = grid.expand_grid(rcfg, verbose=False, has_torchjd=jd)
                assert specs, f"{it.config} {it.overrides} expands to nothing"
                launched.setdefault(it.scan, set()).update(
                    (s.family, s.run_id) for s in specs)
        described = {}
        for scan in launched:
            groups = [f"mode=all n_data={n}" for n in sorted(n_data.get(scan, ()))] \
                or ["mode=all"]
            for group in groups:
                rcfg = loader.compose(scan, group)
                described.setdefault(scan, set()).update(
                    (s.family, s.run_id)
                    for s in grid.expand_grid(rcfg, verbose=False, has_torchjd=jd))
    return launched, described


def test_the_plan_covers_every_run_its_own_configs_describe():
    """The whole point of the plan: no in-scope run is left unlaunched.

    Intrinsic -- it compares the plan against the CONFIGS, run_id by run_id, so it fails
    only when the plan and the configs really disagree (an added family, an optimizer the
    plan does not name -- it found SOAP missing from the CIFAR baselines when it was
    first written) and not when another track's bookkeeping is behind.
    """
    plan = campaign_plan.load_plan(PLAN_FILE)
    launched, described = _plan_and_full_grids(plan)

    missing = {}
    for scan, full in described.items():
        gaps = {}
        for family, run_id in sorted(full - launched[scan]):
            if (scan, family) in DELIBERATE_GAPS:
                continue
            gaps.setdefault(family, []).append(run_id)
        if gaps:
            missing[scan] = {f: (len(r), r[0]) for f, r in gaps.items()}
    assert not missing, f"the plan launches no item for: {missing}"

    # ... and nothing the plan launches is outside its config's own grid (a stale
    # override, an optimizer the config dropped)
    stray = {scan: sorted(launched[scan] - full)[:2]
             for scan, full in described.items() if launched[scan] - full}
    assert not stray, f"plan items expand to runs the config does not describe: {stray}"

    # every DELIBERATE gap must still BE a gap, or the exclusion is stale
    unused = {(s, f) for (s, f) in DELIBERATE_GAPS if s in described
              and not any(fam == f for fam, _ in described[s] - launched[s])}
    assert not unused, f"DELIBERATE_GAPS entries that no longer describe a gap: {unused}"


def test_grid_counts_md_is_in_sync_with_the_plan():
    """`campaign/grid_counts.md` (the configs track's ground truth, read by the cost
    plan) against the plan's own totals. A failure here means grid_counts.md is STALE --
    the coverage property is asserted intrinsically by the test above, so this one never
    means "the plan lost runs"."""
    plan = campaign_plan.load_plan(PLAN_FILE)
    table = _grid_counts_table()
    launched, _described = _plan_and_full_grids(plan)
    per_scan = {scan: len(runs) for scan, runs in launched.items()}

    mismatched = {s: (n, table[s]) for s, n in per_scan.items()
                  if s in table and n != table[s]}
    assert not mismatched, (
        f"campaign/grid_counts.md is stale (plan, table): {mismatched} -- regenerate it "
        f"(tests/test_configs.py owns it); the plan itself is checked by "
        f"test_the_plan_covers_every_run_its_own_configs_describe")
    # every scan the plan launches is one grid_counts knows about ...
    assert not set(per_scan) - set(table)
    # ... and the only in-scope scans the plan does NOT launch are the parked ones
    parked = {it.scan for wl in plan.work_lists for it in wl.disabled_items()}
    assert set(table) - set(per_scan) <= parked | {"exp_finetune_cifar_smallN"}


def test_timing_is_a_documented_stub_not_a_work_list():
    plan = campaign_plan.load_plan(PLAN_FILE)
    assert plan.timing["status"] == "not_in_this_plan"
    assert "exclusive" in plan.timing["shape"].lower()
    for wl in plan.work_lists:
        for item in wl.items:
            assert not item.config.endswith("_timing")


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mutation,message", [
    ("lanes:\n  a100: {gres: 'gpu:1'}\n", "needs `partition`"),
    ("work_lists:\n  - {name: x, phase: P0, lane: nope, items: []}\n", "unknown lane"),
    ("work_lists:\n  - {name: x, phase: later, lane: a100, items: []}\n", "must look like P0"),
    ("work_lists:\n  - {name: x, phase: P0, lane: a100, itmes: []}\n", "unknown key"),
    ("work_lists:\n  - {name: x, phase: P0, lane: a100, items: [{config: c}]}\n",
     "needs `nproc`"),
    ("work_lists:\n  - {name: x, phase: P0, lane: a100, items: [{nproc: 1}]}\n",
     "needs a `config`"),
    ("work_lists:\n  - {name: x, phase: P0, lane: a100, items: [{config: c, nproc: 0}]}\n",
     "nproc must be >= 1"),
    ("work_lists:\n  - {name: x, phase: P0, lane: a100, "
     "items: [{config: c, nproc: 1, overrides: 'a=b | c=d'}]}\n", "must not contain"),
    ("work_lists:\n  - {name: x, phase: P0, lane: mig, items: [{config: c, "
     "nproc: {a100: 4}}]}\n", "no entry for lane 'mig'"),
])
def test_a_doubtful_plan_is_refused_with_a_useful_message(tmp_path, mutation, message):
    base = ("version: 1\nlanes:\n  a100: {partition: p, gres: 'gpu:1'}\n"
            "  mig: {partition: gpu_test, gres: 'gpu:4'}\n"
            "work_lists:\n  - {name: ok, phase: P0, lane: a100, items: []}\n")
    text = base
    if mutation.startswith("lanes:"):
        text = "version: 1\n" + mutation + \
               "work_lists:\n  - {name: ok, phase: P0, lane: a100, items: []}\n"
    else:
        text = ("version: 1\nlanes:\n  a100: {partition: p, gres: 'gpu:1'}\n"
                "  mig: {partition: gpu_test, gres: 'gpu:4'}\n") + mutation
    path = tmp_path / "plan_bad.yaml"
    path.write_text(text)
    with pytest.raises(campaign_plan.PlanError) as exc:
        campaign_plan.load_plan(path)
    assert message in str(exc.value)


def test_yaml_anchor_sections_are_ignored(tmp_path):
    path = tmp_path / "plan_anchor.yaml"
    path.write_text(
        "version: 1\n_items:\n  x: &x\n    - {config: c, nproc: 1}\n"
        "lanes:\n  a100: {partition: p, gres: 'gpu:1'}\n"
        "work_lists:\n  - {name: a, phase: P0, lane: a100, items: *x}\n")
    plan = campaign_plan.load_plan(path)
    assert [wl.name for wl in plan.work_lists] == ["a"]
    assert plan.work_lists[0].items[0].config == "c"


def test_cpus_per_task_is_max_nproc_plus_two_unless_the_lane_fixes_it(sandbox):
    plan = campaign_plan.load_plan(sandbox["plan"])
    a100, mig = plan.lanes["a100"], plan.lanes["mig"]
    items = plan.work_lists[0].enabled_items()
    assert [i.nproc for i in items] == [12, 4]
    assert a100.cpus_for(items) == 14
    assert mig.cpus_for(items) == 32                 # fixed by the lane


# ---------------------------------------------------------------------------
# The printed sbatch lines
# ---------------------------------------------------------------------------

def test_dry_run_prints_the_sbatch_line_and_writes_nothing(sandbox, capsys):
    rc = _launch(sandbox, "--phase", "P0")
    out = capsys.readouterr().out
    assert rc == 0
    assert "dry run (nothing is submitted)" in out
    lines = [l for l in out.splitlines() if "$ sbatch" in l]
    assert len(lines) == 2                            # n_jobs: 2
    for i, line in enumerate(lines):
        assert "-p iaifi_gpu_priority,iaifi_gpu,gpu" in line
        assert "--gres=gpu:1" in line
        assert "-c 14" in line                        # max(12, 4) + 2
        assert "--mem=64G" in line
        assert "-t 24:00:00" in line
        assert f"-J P0.a100.j{i}.l_a100" in line      # phase + lane + index
        assert f"{sandbox['snap']}/tools/worker_pool.sh" in line
        assert "l_a100.items.txt" in line
        assert f"SV3_RESULTS_ROOT={sandbox['root']}" in line
    # the run counts of each item come from the live config grid
    assert "12 runs" in out and "3 runs" in out
    assert "15 runs in this list" in out
    # parked items are named, never launched
    assert "mode=jd" not in "".join(lines)
    assert not sandbox["work"].exists() and not sandbox["logs"].exists()


def test_the_check_composes_the_command_line_the_pool_will_really_run(sandbox, capsys):
    """A dry run must compose the worker pool's own overrides too.

    `worker_pool.sh` prepends one override (`scheduler=...`) to every runner command.
    While the check composed only the plan's own overrides, a spelling of that override
    which hydra's struct mode refuses passed the dry run and then failed EVERY runner
    process of EVERY job -- 24 of 24 on the Gate-1 smoke, before any training.
    """
    # the real pool script is the source of truth, and it must be the working spelling
    assert launch.pool_override(REPO) == "++scheduler=claims"

    # the stub snapshot in the sandbox has no SCHED line -> the safe default
    assert launch.pool_override(sandbox["snap"]) == launch.DEFAULT_POOL_OVERRIDE
    rc = _launch(sandbox, "--phase", "P0")
    assert rc == 0
    assert f"pool adds {launch.DEFAULT_POOL_OVERRIDE}" in capsys.readouterr().out

    # a pool that emits the struct-mode-invalid spelling is now caught by the check
    (sandbox["snap"] / "tools" / "worker_pool.sh").write_text(
        "#!/bin/bash\nSCHED=${WORKER_SCHEDULER_OVERRIDE-scheduler=claims}\n")
    assert launch.pool_override(sandbox["snap"]) == "scheduler=claims"
    rc = _launch(sandbox, "--phase", "P0")
    out = capsys.readouterr().out
    assert rc == 0
    assert "[warn] cannot compose" in out and "scheduler" in out


def test_the_items_file_is_the_work_item_format_worker_pool_parses(sandbox, tmp_path):
    plan = campaign_plan.load_plan(sandbox["plan"])
    wl = plan.work_lists[0]
    path = tmp_path / "items.txt"
    launch.write_items(str(path), wl, wl.enabled_items())
    body = path.read_text().splitlines()
    assert body[0].startswith("#") and body[2].startswith("# config_name |")
    assert body[3:] == ["tiny_scan | mode=svd | 12",
                        "tiny_scan | mode=standard optimizers_standard=[LBFGS] | 4"]
    # ... and worker_pool.sh accepts exactly that file (its own parser, --dry-run)
    proc = subprocess.run(["bash", os.path.join(TOOLS, "worker_pool.sh"),
                           str(sandbox["snap"]), str(path), "--dry-run"],
                          capture_output=True, text=True, timeout=120,
                          env={**os.environ, "WORKER_PY": sys.executable,
                               "CUDA_VISIBLE_DEVICES": ""})
    assert "2 work item(s)" in proc.stdout, proc.stdout + proc.stderr


def test_the_items_file_and_the_report_name_the_snapshot_and_its_shas(sandbox, tmp_path):
    """Nothing else records which code a running job's items belong to."""
    (sandbox["snap"] / "DEPLOY_INFO.json").write_text(json.dumps(
        {"git_sha": "a" * 40, "sven_git_sha": "b" * 40, "git_dirty": False,
         "exported_at": "2026-09-18T21:00:00+00:00"}))
    plan = campaign_plan.load_plan(sandbox["plan"])
    wl = plan.work_lists[0]
    path = tmp_path / "items_stamped.txt"
    launch.write_items(str(path), wl, wl.enabled_items(), snapshot=str(sandbox["snap"]),
                       facts=launch.snapshot_facts(str(sandbox["snap"])))
    body = path.read_text()
    assert f"# snapshot {sandbox['snap']}" in body
    assert f"# sv3  {'a' * 40}" in body and f"# sven {'b' * 40}" in body
    # ... and the work items themselves are still what worker_pool.sh parses
    assert "tiny_scan | mode=svd | 12" in body


def test_the_dry_run_prints_the_snapshot_shas_next_to_the_live_heads(sandbox, capsys):
    (sandbox["snap"] / "DEPLOY_INFO.json").write_text(json.dumps(
        {"git_sha": "a" * 40, "sven_git_sha": "b" * 40, "git_dirty": False,
         "exported_at": "2026-09-18T21:00:00+00:00"}))
    _launch(sandbox, "--phase", "P0", "--no-count")
    out = capsys.readouterr().out
    assert "snap sha  sv3 aaaaaaaaaaaa  sven bbbbbbbbbbbb" in out
    assert "live HEAD sv3 " in out
    # this repo's HEAD is not aaaa..., so the mismatch must be said out loud
    assert "the snapshot is NOT the live HEAD" in out
    # ... and the job name carries the sha8, so `squeue` alone shows a snapshot split
    assert "-J P0.a100.j0.l_a100.aaaaaaaa" in out


def test_a_list_live_from_another_snapshot_is_refused_unless_forced(sandbox, monkeypatch,
                                                                    capsys):
    """The C-R3 ping-pong: two lanes of one list from two snapshots give every run two
    hash8s, and each lane retires the other's results to _stale/ and re-runs them."""
    rec = launch.record_path(str(sandbox["work"]), "tiny", "l_a100")
    os.makedirs(os.path.dirname(rec), exist_ok=True)
    with open(rec, "w") as fh:
        json.dump({"list": "l_a100", "snapshot": "/old/snap/deadbeef_cafe1234",
                   "sv3": "d" * 40, "submitted_at": "2026-09-18T20:00:00+0000"}, fh)
    # a job of this list is queued from that other snapshot
    fake = _fake_squeue(sandbox["tmp"], 0, names=("P0.a100.j0.l_a100",))

    rc = _launch(sandbox, "--phase", "P0", "--no-count", "--submit",
                 extra_path=fake, monkeypatch=monkeypatch)
    cap = capsys.readouterr()
    assert rc == 2
    assert "already running from ANOTHER snapshot" in cap.err
    assert "/old/snap/deadbeef_cafe1234" in cap.err
    assert "P0.a100.j0.l_a100" in cap.err
    assert "sbatch" not in cap.err                 # nothing was submitted
    # the same state with NO live job of that list is not a conflict at all
    fake_idle = _fake_squeue(sandbox["tmp"], 0, names=("P0.a100.j0.other_list",))
    monkeypatch.setenv("PATH", f"{fake_idle}:{os.environ['PATH']}")
    assert launch.snapshot_conflict(launch.read_record(rec), str(sandbox["snap"]),
                                    "l_a100") is None
    # and a dry run only NOTES it (the dry run is how you find out)
    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    rc = _launch(sandbox, "--phase", "P0", "--no-count")
    assert rc == 0 and "last launched from /old/snap/deadbeef_cafe1234" in \
        capsys.readouterr().out


def test_a_selection_of_nothing_is_refused(sandbox, capsys):
    assert _launch(sandbox, "--phase", "P9") == 2
    assert "nothing selected" in capsys.readouterr().out


def test_an_empty_work_list_is_skipped_with_its_note(tmp_path, sandbox, capsys):
    path = tmp_path / "plan_stub.yaml"
    path.write_text(
        f"version: 1\nresults_root: {sandbox['root']}\n"
        "lanes:\n  a100: {partition: p, gres: 'gpu:1'}\n"
        "work_lists:\n  - name: stub\n    phase: P2\n    lane: a100\n"
        "    note: fill me from reconcile\n    items:\n"
        "      - {config: c, nproc: 1, enabled: false, note: waiting for set points}\n")
    sandbox["plan"] = path
    rc = _launch(sandbox, "--no-count")
    out = capsys.readouterr().out
    assert rc == 0
    assert "SKIP stub" in out and "fill me from reconcile" in out
    assert "parked: c |" in out and "waiting for set points" in out
    assert "$ sbatch" not in out


# ---------------------------------------------------------------------------
# The MIG-lane cap (fake squeue)
# ---------------------------------------------------------------------------

def test_mig_lane_cap_allows_two_jobs(sandbox, monkeypatch, capsys):
    rc = _launch(sandbox, "--phase", "P1", "--no-count",
                 extra_path=_fake_squeue(sandbox["tmp"], 0), monkeypatch=monkeypatch)
    out = capsys.readouterr().out
    assert rc == 0
    assert "lane cap mig: 0 queued + 2 wanted <= 2 OK" in out
    assert len([l for l in out.splitlines() if "$ sbatch" in l]) == 2


def test_mig_lane_cap_refuses_when_a_job_is_already_queued(sandbox, monkeypatch, capsys):
    rc = _launch(sandbox, "--phase", "P1", "--no-count",
                 extra_path=_fake_squeue(sandbox["tmp"], 1), monkeypatch=monkeypatch)
    cap = capsys.readouterr()
    assert rc == 2
    assert "allows 2 queued job(s); 1 already queued and this selection wants 2" in cap.err
    assert "refusing to exceed the cap" in cap.err
    assert "$ sbatch" not in cap.out


def test_one_mig_job_still_fits_next_to_one_queued(sandbox, monkeypatch, capsys):
    rc = _launch(sandbox, "--phase", "P1", "--no-count", "--n-jobs", "1",
                 extra_path=_fake_squeue(sandbox["tmp"], 1), monkeypatch=monkeypatch)
    out = capsys.readouterr().out
    assert rc == 0 and "1 queued + 1 wanted <= 2 OK" in out


def test_a_broken_squeue_stops_the_launch_rather_than_guessing(sandbox, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _launch(sandbox, "--phase", "P1", "--no-count",
                extra_path=_fake_squeue(sandbox["tmp"], 0, fail=True), monkeypatch=monkeypatch)
    assert "squeue failed" in str(exc.value)


def test_the_uncapped_a100_lane_needs_no_squeue(sandbox, monkeypatch, capsys):
    """No `squeue` on PATH at all: the a100 lane must still print its jobs."""
    monkeypatch.setenv("PATH", str(sandbox["tmp"] / "empty-bin"))
    (sandbox["tmp"] / "empty-bin").mkdir(exist_ok=True)
    rc = _launch(sandbox, "--phase", "P0", "--no-count")
    assert rc == 0 and "$ sbatch" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --chain
# ---------------------------------------------------------------------------

def test_chain_forces_one_job_and_prints_the_script_path(sandbox, monkeypatch, capsys):
    rc = _launch(sandbox, "--phase", "P1", "--chain", "--no-count",
                 extra_path=_fake_squeue(sandbox["tmp"], 0), monkeypatch=monkeypatch)
    out = capsys.readouterr().out
    assert rc == 0
    assert "--chain forces n_jobs=1" in out
    assert "l_mig.chain.sbatch" in out
    assert len([l for l in out.splitlines() if "$ sbatch" in l]) == 1


def test_a_chain_leaves_a_free_slot_in_a_capped_lane(sandbox, monkeypatch, capsys):
    """Two chains would fill gpu_test's 2 MaxSubmit slots, and then NEITHER can renew --
    the module docstring promises the opposite, so a selection of two chains is refused."""
    plan = sandbox["plan"]
    plan.write_text(plan.read_text() + """  - name: l_mig2
    phase: P1
    lane: mig
    n_jobs: 1
    items:
      - {config: tiny_scan, overrides: "mode=svd", nproc: {a100: 12, mig: 6}}
""")
    rc = _launch(sandbox, "--phase", "P1", "--chain", "--no-count",
                 extra_path=_fake_squeue(sandbox["tmp"], 0), monkeypatch=monkeypatch)
    cap = capsys.readouterr()
    assert rc == 2
    assert "a chain must leave one free to renew into" in cap.err
    assert "this selection wants 2" in cap.err
    assert "$ sbatch" not in cap.out
    # one chain in an empty lane is fine, and says which slot it is keeping free
    rc = _launch(sandbox, "--phase", "P1", "--chain", "--no-count", "--list", "l_mig",
                 extra_path=_fake_squeue(sandbox["tmp"], 0), monkeypatch=monkeypatch)
    out = capsys.readouterr().out
    assert rc == 0
    assert "0 queued + 1 wanted <= 1 OK" in out
    assert "slot kept free so the chain can renew" in out


def test_the_chain_gets_control_back_before_the_wall_clock(sandbox):
    """A batch shell whose FOREGROUND child is SIGTERMed never runs the next line, so a
    chain that only waits for the wall clock can never renew. The script therefore asks
    for --signal, traps it, and runs the pool in the BACKGROUND."""
    plan = campaign_plan.load_plan(sandbox["plan"])
    wl = plan.work_lists[1]
    path = sandbox["tmp"] / "chain_signal.sbatch"
    launch.chain_script(str(path), lane=plan.lanes["mig"], work_list=wl,
                        items=wl.enabled_items(), snapshot=str(sandbox["snap"]),
                        items_file="/tmp/items.txt", results_root=str(sandbox["root"]),
                        plan_file=str(sandbox["plan"]), log_dir=str(sandbox["logs"]),
                        name="P1.mig.j0.l_mig", python=sys.executable)
    text = path.read_text()
    subprocess.run(["bash", "-n", str(path)], check=True, timeout=60)
    assert f"#SBATCH --signal=B:USR1@{launch.CHAIN_SIGNAL_S}" in text
    assert "trap on_usr1 USR1" in text
    assert "worker_pool.sh" in text and "POOL_PID=$!" in text     # background, not fg
    assert 'kill -TERM "$POOL_PID"' in text
    assert '[ "$SIGNALLED" = 1 ] && rc=0' in text                 # not a failure
    # the renew block must come AFTER the wait, and be reachable
    assert text.index("POOL_PID=$!") < text.index("reconcile.py") \
        < text.index("sbatch --export=ALL")

    # the same pattern, executed for real: a trapped signal must not eat the tail
    probe = sandbox["tmp"] / "probe.sh"
    probe.write_text("#!/bin/bash\nset -u\n"
                     'stop() { kill -TERM "$P" 2>/dev/null; }\ntrap stop USR1\n'
                     "sleep 30 & P=$!\n"
                     'while :; do wait "$P"; rc=$?; '
                     'if [ "$rc" -gt 128 ] && kill -0 "$P" 2>/dev/null; then continue; '
                     "fi; break; done\n"
                     'echo "RENEW BLOCK REACHED rc=$rc"\n')
    proc = subprocess.Popen(["bash", str(probe)], stdout=subprocess.PIPE, text=True)
    time.sleep(1)
    proc.send_signal(signal.SIGUSR1)
    out, _ = proc.communicate(timeout=60)
    assert "RENEW BLOCK REACHED" in out, out


def test_logs_default_to_holystore_not_to_the_nfs_home(sandbox):
    """One log per (job x pool x item x NPROC) plus a hydra dir each is O(10^4) files for
    the full plan; /n/home11 is a 95 G NFS home at 84%."""
    assert launch.DEFAULT_LOG_DIR.startswith("/n/holystore01/")
    assert "/n/home" not in launch.DEFAULT_LOG_DIR
    pool = open(os.path.join(TOOLS, "worker_pool.sh")).read()
    assert "WORKER_LOG_ROOT:-/n/holystore01/" in pool
    assert "PYTHONDONTWRITEBYTECODE=1" in pool      # nothing writes into the snapshot


def test_the_config_dir_follows_the_snapshot(sandbox, capsys):
    """The dry run must describe the configs the JOBS will run, not the live tree's."""
    cfg = sandbox["snap"] / "experiments" / "configs"
    cfg.mkdir(parents=True)
    # the snapshot's tiny_scan has ONE k value, so its grid is half the live one's
    (cfg / "tiny_scan.yaml").write_text(TINY_CONFIG.replace("k_values: [2, 4]",
                                                            "k_values: [2]"))
    assert launch.snapshot_config_dir(str(sandbox["snap"])) == str(cfg)
    argv = [str(sandbox["plan"]), "--snapshot", str(sandbox["snap"]),
            "--work-dir", str(sandbox["work"]), "--log-dir", str(sandbox["logs"]),
            "--phase", "P0"]
    assert launch.main(argv) == 0
    out = capsys.readouterr().out
    assert f"configs   {cfg}   (the snapshot's" in out
    assert "6 runs" in out and "9 runs in this list" in out     # 6 svd + 3 lbfgs
    # an explicit --config-dir still wins, and says so
    assert launch.main(argv + ["--config-dir", str(sandbox["cfg"])]) == 0
    out = capsys.readouterr().out
    assert "(--config-dir;" in out and "15 runs in this list" in out


def test_the_chain_script_is_valid_bash_and_renews_only_on_exit_code_1(sandbox):
    plan = campaign_plan.load_plan(sandbox["plan"])
    wl = plan.work_lists[1]
    path = sandbox["tmp"] / "chain.sbatch"
    launch.chain_script(str(path), lane=plan.lanes["mig"], work_list=wl,
                        items=wl.enabled_items(), snapshot=str(sandbox["snap"]),
                        items_file="/tmp/items.txt", results_root=str(sandbox["root"]),
                        plan_file=str(sandbox["plan"]), log_dir=str(sandbox["logs"]),
                        name="P1.mig.j0.l_mig", python=sys.executable)
    text = path.read_text()
    subprocess.run(["bash", "-n", str(path)], check=True, timeout=60)
    assert text.startswith("#!/bin/bash")
    assert "#SBATCH -p gpu_test" in text and "#SBATCH --gres=gpu:4" in text
    assert "#SBATCH -t 12:00:00" in text and "#SBATCH -c 32" in text
    assert "worker_pool.sh" in text
    assert "reconcile.py" in text and "--list l_mig" in text and "--quiet" in text
    # renewal is conditional on reconcile's "work remains" code, and bounded
    assert 'if [ "$remaining" -ne 1 ]; then' in text
    assert 'if [ "$DEPTH" -ge "$MAXDEPTH" ]; then' in text
    assert "CHAIN_DEPTH=$((DEPTH+1))" in text
    assert "MaxSubmit" not in text or "2 queued-or-running" in text or "cap" in text
    assert os.access(path, os.X_OK)


def test_submitting_requires_a_real_snapshot(sandbox, capsys):
    """Fatal with --submit, a warning in a dry run (tools/ may not be committed yet)."""
    bad = sandbox["tmp"] / "not-a-snapshot"
    bad.mkdir()
    args = [str(sandbox["plan"]), "--snapshot", str(bad), "--phase", "P0",
            "--no-count", "--work-dir", str(sandbox["work"]),
            "--log-dir", str(sandbox["logs"])]
    with pytest.raises(SystemExit) as exc:
        launch.main(args + ["--submit"])
    assert "not a complete snapshot" in str(exc.value)

    (bad / ".deploy_complete").write_text("x")
    with pytest.raises(SystemExit) as exc:
        launch.main(args + ["--submit"])
    assert "no tools/worker_pool.sh" in str(exc.value)

    assert launch.main(args) == 0                 # the dry run still prints
    out = capsys.readouterr().out
    assert "WARNING" in out and "no tools/worker_pool.sh" in out
    assert "$ sbatch" in out


def test_an_override_naming_an_unlisted_optimizer_is_warned_about(sandbox, capsys):
    plan = sandbox["plan"]
    plan.write_text(plan.read_text().replace(
        "optimizers_standard=[LBFGS]", "optimizers_standard=[LBFGS,Shampoo]"))
    rc = _launch(sandbox, "--phase", "P0")
    out = capsys.readouterr().out
    assert rc == 0
    assert "not in the config's optimizers_standard: Shampoo" in out
    assert "1 warning(s)" in out


def test_an_override_that_expands_to_nothing_is_warned_about(sandbox, capsys):
    plan = sandbox["plan"]
    plan.write_text(plan.read_text().replace('overrides: "mode=svd"', 'overrides: "mode=hig"'))
    rc = _launch(sandbox, "--phase", "P0")
    out = capsys.readouterr().out
    assert rc == 0 and "expands to ZERO runs" in out
