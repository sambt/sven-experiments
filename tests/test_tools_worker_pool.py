"""`tools/worker_pool.sh` and `tools/deploy_snapshot.sh`, exercised for real.

No GPU, no SLURM, no training: the pool is pointed at a FAKE python that records its
argv, cwd and environment and exits with a controllable status, and at a FAKE
`nvidia-smi` on PATH. `deploy_snapshot.sh` is run against two throwaway git repos in a
temp dir, never the real ones.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")
POOL = os.path.join(TOOLS, "worker_pool.sh")
DEPLOY = os.path.join(TOOLS, "deploy_snapshot.sh")

#: a python that is real enough for the snapshot guard (it must execute the guard
#: script it gets on stdin) and fake enough to record what the pool asked for
FAKE_PY = '''#!{exe}
import json, os, sys, time
if len(sys.argv) > 1 and sys.argv[1] == "-":        # the pool's snapshot guard
    src = sys.stdin.read()
    g = {{"__name__": "__main__"}}
    sys.argv = ["-"]
    exec(compile(src, "<guard>", "exec"), g)
    raise SystemExit(0)
with open(os.environ["FAKE_PY_LOG"], "a") as fh:
    fh.write(json.dumps({{
        "argv": sys.argv[1:], "cwd": os.getcwd(), "pid": os.getpid(),
        "env": {{k: os.environ.get(k) for k in
                ("PYTHONPATH", "SV3_RESULTS_ROOT", "PYTORCH_CUDA_ALLOC_CONF",
                 "PYTORCH_ALLOC_CONF",
                 "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
                 "PYTHONDONTWRITEBYTECODE")}}}}) + "\\n")
    fh.flush()
time.sleep(float(os.environ.get("FAKE_PY_SLEEP", "0")))
raise SystemExit(int(os.environ.get("FAKE_PY_EXIT", "0")))
'''

NVIDIA_SMI_MIG = """#!/bin/bash
cat <<'OUT'
GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-82e6ee72-1e38-a4c2-dbe0-950d157f51ac)
  MIG 3g.20gb     Device  0: (UUID: MIG-aaaa0001-ed8c-5150-9e8c-9df08fbb7484)
  MIG 3g.20gb     Device  1: (UUID: MIG-bbbb0002-6e2c-5440-8809-dc2a91ec1cf8)
  MIG 3g.20gb     Device  2: (UUID: MIG-cccc0003-6e2c-5440-8809-dc2a91ec1cf8)
  MIG 3g.20gb     Device  3: (UUID: MIG-dddd0004-6e2c-5440-8809-dc2a91ec1cf8)
OUT
"""

#: `command -v nvidia-smi` succeeds but nothing is listed = a machine with no GPU
NVIDIA_SMI_NONE = """#!/bin/bash
exit 1
"""

NVIDIA_SMI_A100 = """#!/bin/bash
cat <<'OUT'
GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-ae749001-e8b8-fb41-616d-ecfcc23f22f9)
GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-ae749002-e8b8-fb41-616d-ecfcc23f22f9)
OUT
"""

ITEMS = """# a comment, and a blank line follow

toy_1d_scan | mode=svd | 2
polynomial_scan | mode=standard optimizers_standard=[Adam,SGD] | 1
"""


@pytest.fixture
def env(tmp_path):
    """A fake snapshot, a fake python and an empty log root."""
    snap = tmp_path / "snap"
    (snap / "experiments").mkdir(parents=True)
    (snap / "experiments" / "__init__.py").touch()
    (snap / "sven" / "sven").mkdir(parents=True)
    (snap / "sven" / "sven" / "__init__.py").touch()
    (snap / "run.py").write_text("raise SystemExit(0)\n")
    (snap / "DEPLOY_INFO.json").write_text(json.dumps(
        {"git_sha": "a" * 40, "git_dirty": False, "exported_at": "2026-09-18T00:00:00"}))
    (snap / ".deploy_complete").write_text("now")
    results = tmp_path / "results"
    results.mkdir()
    (snap / "experiment_results").symlink_to(results)

    py = tmp_path / "fakepy"
    py.write_text(FAKE_PY.format(exe=sys.executable))
    py.chmod(0o755)

    items = tmp_path / "items.txt"
    items.write_text(ITEMS)
    log = tmp_path / "argv.jsonl"
    return {"snap": snap, "py": py, "items": items, "log": log,
            "logs": tmp_path / "logs", "results": results, "tmp": tmp_path}


def _bin(tmp_path, name, body):
    d = tmp_path / f"bin-{name}"
    d.mkdir(exist_ok=True)
    p = d / "nvidia-smi"
    p.write_text(body)
    p.chmod(0o755)
    return d


def _run(env, *args, cvd=None, extra_env=None, path_dir=None, items=None, timeout=180):
    e = dict(os.environ)
    e.update({"WORKER_PY": str(env["py"]), "WORKER_LOG_ROOT": str(env["logs"]),
              "FAKE_PY_LOG": str(env["log"])})
    e.pop("SV3_RESULTS_ROOT", None)
    if cvd is None:
        e.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        e["CUDA_VISIBLE_DEVICES"] = cvd
    # a shim `nvidia-smi` always comes first, so the test never sees the real GPUs of
    # whatever node it runs on (the rest of PATH is kept: the script needs sed, seq, ...)
    e["PATH"] = f"{path_dir or _bin(env['tmp'], 'none', NVIDIA_SMI_NONE)}:{e['PATH']}"
    if extra_env:
        e.update(extra_env)
    cmd = ["bash", POOL, str(env["snap"]), str(items or env["items"]), *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=e)


def _calls(env):
    return [json.loads(l) for l in env["log"].read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# syntax
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script", [POOL, DEPLOY])
def test_the_shell_scripts_are_valid_bash(script):
    subprocess.run(["bash", "-n", script], check=True, timeout=60)
    assert os.access(script, os.X_OK)


# ---------------------------------------------------------------------------
# work items
# ---------------------------------------------------------------------------

def test_items_are_parsed_and_walked_in_order_with_nproc_processes(env):
    proc = _run(env, "--label", "t")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "2 work item(s)" in proc.stdout
    calls = _calls(env)
    assert len(calls) == 3                       # NPROC 2 then NPROC 1
    assert [c["argv"][2] for c in calls] == ["toy_1d_scan", "toy_1d_scan",
                                             "polynomial_scan"]
    assert calls[0]["argv"][:4] == [str(env["snap"] / "run.py"), "--config-name",
                                    "toy_1d_scan", _pool_scheduler_token()]
    assert calls[0]["argv"][-1] == "mode=svd"
    # the list-valued override survives as ONE argv word (set -f, no globbing)
    assert calls[2]["argv"][-1] == "optimizers_standard=[Adam,SGD]"
    assert "2 item(s), 0 failed" in proc.stdout


def test_hydra_never_writes_into_the_snapshot(env):
    """hydra.run.dir is pinned under the job log dir: the snapshot stays immutable."""
    before = sorted(os.listdir(env["snap"]))
    _run(env, "--label", "t")
    hydra_args = [a for a in _calls(env)[0]["argv"] if a.startswith("hydra.run.dir=")]
    assert len(hydra_args) == 1
    assert hydra_args[0].startswith(f"hydra.run.dir={env['logs']}")
    assert str(env["snap"]) not in hydra_args[0]
    assert sorted(os.listdir(env["snap"])) == before
    env["log"].unlink()
    _run(env, extra_env={"WORKER_HYDRA_RUN_DIR": ""})
    assert not any(a.startswith("hydra.run.dir") for a in _calls(env)[0]["argv"])


def test_the_runner_environment_is_the_contract(env):
    _run(env, "--label", "t")
    call = _calls(env)[0]
    snap = str(env["snap"])
    assert call["env"]["PYTHONPATH"] == f"{snap}:{snap}/sven"
    assert call["env"]["SV3_RESULTS_ROOT"] == f"{snap}/experiment_results"
    # both spellings: this torch warns that the CUDA-prefixed name is deprecated, and
    # `expandable_segments` is what halves peak reserved memory and what lets CIFAR Sven
    # fit a MIG slice, so it must survive the version that stops honouring the old name
    assert call["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert call["env"]["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert call["env"]["OMP_NUM_THREADS"] == "1"
    assert call["env"]["MKL_NUM_THREADS"] == "1"
    # the snapshot is shared and immutable: no job may write __pycache__ into it
    assert call["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert call["cwd"] == snap                   # so `experiments` resolves in-snapshot


def test_logs_default_to_the_scratch_tree_not_to_the_repo():
    """O(10^4) runner logs + hydra dirs for the full plan: they go under $SV3_SCRATCH,
    and a failed `> $log` on a full home looks like a training failure."""
    body = open(POOL).read()
    assert "WORKER_LOG_ROOT:-${SV3_SCRATCH" in body
    assert "slurm_logs" not in body


# ---------------------------------------------------------------------------
# the wall clock
# ---------------------------------------------------------------------------

def test_no_new_item_is_started_inside_the_reserve_and_that_is_exit_zero(env):
    """What gives a chained job control back BEFORE slurm's SIGTERM: a batch shell whose
    foreground child is SIGTERMed never runs the line after it, so the pool must end on
    its own. Winding down is SUCCESS -- the chain renews on reconcile's verdict, not on
    the pool's exit code."""
    proc = _run(env, extra_env={"SLURM_JOB_END_TIME": str(int(time.time()) + 10),
                                "WORKER_RESERVE_S": "900"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "within 900s of the wall clock" in proc.stdout
    assert "wound down at the wall clock" in proc.stdout
    assert not env["log"].exists()               # not one runner was started


def test_the_reserve_leaves_time_for_the_items_it_does_start(env):
    """The deadline stops items, it does not abort them: an item that starts inside the
    budget runs to completion."""
    proc = _run(env, "--until", f"+{3600}")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no new item after" in proc.stdout
    assert len(_calls(env)) == 3                  # both items, all NPROC processes
    assert "wound down" not in proc.stdout


def test_the_deadline_can_come_from_any_of_the_three_sources(env):
    soon = str(int(time.time()) + 10)
    for extra, args in (({"WORKER_DEADLINE_EPOCH": soon}, ()),
                        ({"WORKER_TIME_BUDGET": "10"}, ()),
                        ({}, ("--until", soon))):
        if env["log"].exists():
            env["log"].unlink()
        proc = _run(env, *args, extra_env={**extra, "WORKER_RESERVE_S": "900"})
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "wound down at the wall clock" in proc.stdout, (extra, args)
        assert not env["log"].exists()
    proc = _run(env, "--until", "not-a-time")
    assert proc.returncode == 2 and "unix epoch" in proc.stderr
    proc = _run(env, extra_env={"WORKER_TIME_BUDGET": "soon"})
    assert proc.returncode == 2 and "number of seconds" in proc.stderr


def test_a_slurm_end_time_that_is_not_an_epoch_costs_the_winddown_not_the_job(env):
    """Older slurm exports a formatted date: that must never fail a real job."""
    proc = _run(env, extra_env={"SLURM_JOB_END_TIME": "2026-09-19T12:00:00"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "is not a unix epoch" in proc.stdout
    assert "no wall-clock wind-down" in proc.stdout
    assert len(_calls(env)) == 3                  # every item still ran


def test_an_explicit_results_root_wins(env):
    other = env["tmp"] / "other-root"
    other.mkdir()
    _run(env, extra_env={"SV3_RESULTS_ROOT": str(other)})
    assert _calls(env)[0]["env"]["SV3_RESULTS_ROOT"] == str(other)


def test_the_scheduler_override_the_pool_emits_composes_against_a_real_config():
    """The pool's override must survive hydra's struct mode.

    The fake python of every other test in this file never composes a config, which is
    how a plain `scheduler=claims` reached a GPU: no scan config declares `scheduler`
    (the default is `grid.resolve_scan_settings`'s and `tests/test_configs.py` forbids
    the key in a config), so struct mode answers "Could not override 'scheduler' ...
    not in struct" and EVERY runner process of EVERY campaign job exits 1 having
    trained nothing. Compose the token the pool actually emits, against real configs.
    """
    sys.path.insert(0, TOOLS)
    import reconcile                                    # torch-free
    sys.path.insert(0, REPO)
    grid = reconcile.load_grid()

    token = _pool_scheduler_token()
    with reconcile.ConfigLoader() as loader:
        for config in ("toy_1d_scan", "exp_nanogpt_speedrun",
                       "cifar10_resnet_ce_scan"):
            rcfg = loader.compose(config, token)         # raises on a struct-mode clash
            assert rcfg["scheduler"] == "claims"
            assert grid.resolve_scan_settings(rcfg)["scheduler"] == "claims"


def _pool_scheduler_token():
    """The default of `SCHED` in worker_pool.sh, i.e. what a job really passes."""
    import re
    match = re.search(r"^SCHED=\$\{WORKER_SCHEDULER_OVERRIDE-(\S+)\}\s*$",
                      open(POOL).read(), re.M)
    assert match, "worker_pool.sh no longer defines SCHED the way this test reads it"
    return match.group(1)


def test_the_scheduler_override_is_default_on_and_can_be_dropped(env):
    _run(env)
    assert _pool_scheduler_token() in _calls(env)[0]["argv"][3:]
    env["log"].unlink()
    _run(env, extra_env={"WORKER_SCHEDULER_OVERRIDE": ""})
    # an EMPTY override drops the word entirely (Gate-1 smoke, before `scheduler`
    # exists as a config key); the tmp dir is named after this test, so compare the
    # hydra arguments only -- never the run.py path
    argv = [a for a in _calls(env)[0]["argv"][1:] if not a.startswith("hydra.run.dir=")]
    assert argv == ["--config-name", "toy_1d_scan", "mode=svd"]


def test_comments_and_blank_lines_are_ignored_and_a_bad_nproc_is_fatal(env):
    bad = env["tmp"] / "bad.txt"
    bad.write_text("# only comments\n\ntoy_1d_scan | mode=svd | none\n")
    proc = _run(env, items=bad)
    assert proc.returncode == 2
    assert "NPROC must be a positive integer, got 'none'" in proc.stderr
    assert not env["log"].exists()

    empty = env["tmp"] / "empty.txt"
    empty.write_text("# nothing but a comment\n")
    proc = _run(env, items=empty)
    assert proc.returncode == 2 and "no work items" in proc.stderr


def test_the_items_file_snapshot_stamp_is_logged_and_a_mismatch_shouts(env):
    """The job log is the only place the pair (snapshot running, snapshot the items were
    written for) can be seen after the fact."""
    stamped = env["tmp"] / "stamped.txt"
    stamped.write_text(f"# snapshot {env['snap']}\n# sv3  {'a' * 40}\n"
                       f"# sven {'b' * 40}\n" + ITEMS)
    proc = _run(env, items=stamped, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"items for   {env['snap']}" in proc.stdout
    assert f"items sv3  {'a' * 40}" in proc.stdout
    assert "WARNING: this items file was written for ANOTHER snapshot" not in proc.stdout

    other = env["tmp"] / "other.txt"
    other.write_text("# snapshot /scratch/elsewhere/deadbeef_cafe1234\n" + ITEMS)
    proc = _run(env, items=other, timeout=180)
    assert "written for ANOTHER snapshot (/scratch/elsewhere/deadbeef_cafe1234)" \
        in proc.stdout
    assert "_stale/" in proc.stdout
    assert proc.returncode == 0          # a warning, not a refusal: the operator decides


def test_a_missing_config_name_is_fatal(env):
    bad = env["tmp"] / "nocfg.txt"
    bad.write_text(" | mode=svd | 2\n")
    proc = _run(env, items=bad)
    assert proc.returncode == 2 and "empty config name" in proc.stderr


# ---------------------------------------------------------------------------
# devices
# ---------------------------------------------------------------------------

def test_every_mig_slice_gets_its_own_pool(env):
    proc = _run(env, path_dir=_bin(env["tmp"], "mig", NVIDIA_SMI_MIG))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "4 pool(s)" in proc.stdout
    devices = {c["env"]["CUDA_VISIBLE_DEVICES"] for c in _calls(env)}
    assert devices == {"MIG-aaaa0001-ed8c-5150-9e8c-9df08fbb7484",
                       "MIG-bbbb0002-6e2c-5440-8809-dc2a91ec1cf8",
                       "MIG-cccc0003-6e2c-5440-8809-dc2a91ec1cf8",
                       "MIG-dddd0004-6e2c-5440-8809-dc2a91ec1cf8"}
    assert len(_calls(env)) == 4 * 3             # every pool walks every item


def test_cuda_visible_devices_selects_by_index_into_the_mig_list(env):
    proc = _run(env, cvd="1,3", path_dir=_bin(env["tmp"], "mig", NVIDIA_SMI_MIG))
    assert "2 pool(s)" in proc.stdout
    assert {c["env"]["CUDA_VISIBLE_DEVICES"] for c in _calls(env)} == {
        "MIG-bbbb0002-6e2c-5440-8809-dc2a91ec1cf8",
        "MIG-dddd0004-6e2c-5440-8809-dc2a91ec1cf8"}


def test_cuda_visible_devices_selects_by_uuid_too(env):
    proc = _run(env, cvd="MIG-cccc0003-6e2c-5440-8809-dc2a91ec1cf8",
                path_dir=_bin(env["tmp"], "mig", NVIDIA_SMI_MIG))
    assert "1 pool(s)" in proc.stdout
    assert {c["env"]["CUDA_VISIBLE_DEVICES"] for c in _calls(env)} == {
        "MIG-cccc0003-6e2c-5440-8809-dc2a91ec1cf8"}


def test_whole_gpus_are_used_when_the_node_has_no_mig(env):
    proc = _run(env, path_dir=_bin(env["tmp"], "a100", NVIDIA_SMI_A100))
    assert "2 pool(s)" in proc.stdout
    assert {c["env"]["CUDA_VISIBLE_DEVICES"] for c in _calls(env)} == {
        "GPU-ae749001-e8b8-fb41-616d-ecfcc23f22f9",
        "GPU-ae749002-e8b8-fb41-616d-ecfcc23f22f9"}


def test_no_gpu_at_all_still_runs_one_pool(env):
    proc = _run(env)                              # no nvidia-smi on PATH
    assert proc.returncode == 0
    assert "no GPU visible" in proc.stdout and "1 pool(s)" in proc.stdout
    assert len(_calls(env)) == 3


def test_dry_run_launches_nothing(env):
    proc = _run(env, "--dry-run", path_dir=_bin(env["tmp"], "mig", NVIDIA_SMI_MIG))
    assert proc.returncode == 0
    assert "--dry-run" in proc.stdout and "gpu3 x1" in proc.stdout
    assert not env["log"].exists()
    assert not env["logs"].exists()


# ---------------------------------------------------------------------------
# failures, the snapshot guard and SIGTERM
# ---------------------------------------------------------------------------

def test_a_crashed_runner_makes_the_job_fail(env):
    proc = _run(env, extra_env={"FAKE_PY_EXIT": "3"})
    assert proc.returncode == 1
    assert "a runner exited 3" in proc.stdout
    assert "3 failed runner process(es)" in proc.stdout
    assert len(_calls(env)) == 3                 # a failure does not stop the list


def test_the_snapshot_guard_refuses_a_tree_that_is_not_the_snapshot(env, tmp_path):
    fake = tmp_path / "not-a-snapshot"
    fake.mkdir()
    (fake / "run.py").write_text("raise SystemExit(0)\n")
    e = dict(os.environ, WORKER_PY=str(env["py"]), WORKER_LOG_ROOT=str(env["logs"]),
             FAKE_PY_LOG=str(env["log"]))
    proc = subprocess.run(["bash", POOL, str(fake), str(env["items"])],
                          capture_output=True, text=True, timeout=180, env=e)
    assert proc.returncode == 2
    assert "resolve OUTSIDE the snapshot" in proc.stdout
    assert "snapshot guard failed" in proc.stderr
    assert not env["log"].exists()


def test_sigterm_kills_the_runners_and_stops_the_list(env):
    e = dict(os.environ, WORKER_PY=str(env["py"]), WORKER_LOG_ROOT=str(env["logs"]),
             FAKE_PY_LOG=str(env["log"]), FAKE_PY_SLEEP="60",
             PATH=os.environ["PATH"], CUDA_VISIBLE_DEVICES="")
    proc = subprocess.Popen(["bash", POOL, str(env["snap"]), str(env["items"])],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env=e, start_new_session=True)
    try:
        deadline = time.time() + 120
        while time.time() < deadline and len(_calls(env) if env["log"].exists() else []) < 2:
            time.sleep(0.2)
        assert len(_calls(env)) == 2, "the first item's 2 runners did not start"
        pids = [c["pid"] for c in _calls(env)]
        proc.send_signal(signal.SIGTERM)
        out = proc.stdout.read()
        rc = proc.wait(timeout=60)
    finally:
        if proc.poll() is None:                   # never leave the pool running
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=30)
    assert rc == 143, out
    assert "caught SIGTERM" in out and "terminated by signal" in out
    assert len(_calls(env)) == 2, "item 1 must not have been started after the signal"
    for pid in pids:
        with pytest.raises(OSError):              # the sleeping runners are gone
            os.kill(pid, 0)


# ---------------------------------------------------------------------------
# deploy_snapshot.sh
# ---------------------------------------------------------------------------

def _git_repo(path, sven=False):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True, timeout=60)
    for key, value in (("user.email", "t@example.com"), ("user.name", "T"),
                       ("commit.gpgsign", "false")):
        subprocess.run(["git", "-C", str(path), "config", key, value], check=True)
    if sven:
        (path / "sven").mkdir(exist_ok=True)
        (path / "sven" / "__init__.py").write_text("VERSION = 1\n")
    else:
        (path / "run.py").write_text("raise SystemExit(0)\n")
        (path / "experiments" / "experiment_code").mkdir(parents=True)
        (path / "experiments" / "experiment_code" / "grid.py").write_text("SCHEMA = 2\n")
        (path / "tools").mkdir()
        (path / "tools" / "worker_pool.sh").write_text("#!/bin/bash\nexit 0\n")
        # the real sv3 ignores its nested sven checkout; without this the sven/ dir
        # would make every `git status --porcelain` dirty
        (path / ".gitignore").write_text("sven/\n")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True, timeout=60)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "init"], check=True, timeout=60)
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repos(tmp_path):
    sv3 = tmp_path / "sv3"
    sha = _git_repo(sv3)
    sven_sha = _git_repo(sv3 / "sven", sven=True)
    results = tmp_path / "results-root"
    results.mkdir()
    return {"sv3": sv3, "sha": sha, "sven_sha": sven_sha, "results": results,
            "base": tmp_path / "deploy"}


def _deploy(repos, *args):
    return subprocess.run(
        ["bash", DEPLOY, "--repo", str(repos["sv3"]), "--base", str(repos["base"]),
         "--results-root", str(repos["results"]), *args],
        capture_output=True, text=True, timeout=300)


def test_deploy_exports_both_repos_with_provenance_and_a_results_symlink(repos):
    proc = _deploy(repos)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    snap = proc.stdout.strip().splitlines()[-1]
    assert os.path.basename(snap) == f"{repos['sha'][:8]}_{repos['sven_sha'][:8]}"
    assert os.path.isfile(os.path.join(snap, "run.py"))
    assert os.path.isfile(os.path.join(snap, "sven", "sven", "__init__.py"))
    assert os.path.isfile(os.path.join(snap, ".deploy_complete"))
    assert os.path.realpath(os.path.join(snap, "experiment_results")) == \
        os.path.realpath(str(repos["results"]))
    # the schema provenance.py falls back to, in BOTH roots
    info = json.load(open(os.path.join(snap, "DEPLOY_INFO.json")))
    assert info["git_sha"] == repos["sha"] and info["git_dirty"] is False
    assert info["sven_git_sha"] == repos["sven_sha"]
    assert info["exported_at"] and info["results_root"] == str(repos["results"])
    sven_info = json.load(open(os.path.join(snap, "sven", "DEPLOY_INFO.json")))
    assert sven_info["git_sha"] == repos["sven_sha"] and sven_info["git_dirty"] is False
    # no .git in the export: provenance must take the sidecar, not a parent checkout
    assert not os.path.exists(os.path.join(snap, ".git"))
    assert not os.path.exists(os.path.join(snap, "sven", ".git"))


def test_deploy_is_idempotent(repos):
    first = _deploy(repos).stdout.strip().splitlines()[-1]
    marker = os.path.join(first, "MARKER")
    open(marker, "w").close()
    proc = _deploy(repos)
    assert proc.returncode == 0
    assert proc.stdout.strip().splitlines()[-1] == first
    assert "already exported" in proc.stdout
    assert os.path.exists(marker)                 # untouched, not re-exported

    proc = _deploy(repos, "--force")
    assert proc.returncode == 0
    assert proc.stdout.strip().splitlines()[-1] == first
    assert not os.path.exists(marker)             # --force really re-exported


def test_deploy_refuses_a_dirty_tree_unless_allowed(repos):
    (repos["sv3"] / "run.py").write_text("raise SystemExit(1)   # uncommitted\n")
    proc = _deploy(repos)
    assert proc.returncode == 2
    assert "uncommitted tracked changes in: sv3" in proc.stderr
    assert not os.path.isdir(str(repos["base"])) or not os.listdir(str(repos["base"]))

    proc = _deploy(repos, "--allow-dirty")
    assert proc.returncode == 0
    snap = proc.stdout.strip().splitlines()[-1]
    info = json.load(open(os.path.join(snap, "DEPLOY_INFO.json")))
    assert info["git_dirty"] is True and info["git_dirty_tracked"] is True
    assert info["git_dirty_worktree"] is True
    # HEAD is exported, NOT the dirty working tree
    assert "uncommitted" not in open(os.path.join(snap, "run.py")).read()


def test_deploy_refuses_a_dirty_sven_too(repos):
    (repos["sv3"] / "sven" / "sven" / "__init__.py").write_text("VERSION = 2\n")
    proc = _deploy(repos)
    assert proc.returncode == 2 and "sven" in proc.stderr


def test_untracked_files_are_a_note_not_a_refusal(repos):
    (repos["sv3"] / "scratch_notes.md").write_text("an agent's scratch file\n")
    proc = _deploy(repos)
    assert proc.returncode == 0
    assert "untracked path(s) in sv3 are NOT exported" in proc.stdout
    snap = proc.stdout.strip().splitlines()[-1]
    assert not os.path.exists(os.path.join(snap, "scratch_notes.md"))
    info = json.load(open(os.path.join(snap, "DEPLOY_INFO.json")))
    # `git_dirty` describes the EXPORT, which is `git archive HEAD` and therefore
    # cannot contain the untracked file: a record made from this snapshot must not
    # claim it ran uncommitted code (the live tree always carries agents' scratch
    # files, so the other reading marks every campaign record dirty and says nothing).
    assert info["git_dirty"] is False and info["git_dirty_tracked"] is False
    # the working tree's own state is still on the record, separately
    assert info["git_dirty_worktree"] is True
    assert info["git_untracked_paths"] == 1


def test_deploy_refuses_a_results_root_that_does_not_exist(repos):
    proc = subprocess.run(
        ["bash", DEPLOY, "--repo", str(repos["sv3"]), "--base", str(repos["base"]),
         "--results-root", str(repos["results"]) + "-nope"],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 1 and "is not a directory" in proc.stderr


def test_a_deployed_snapshot_is_what_the_worker_pool_accepts(repos, env):
    """The two scripts agree: a real export passes the pool's snapshot guard."""
    snap = _deploy(repos).stdout.strip().splitlines()[-1]
    (os.path.join(snap, "experiments", "__init__.py"))
    open(os.path.join(snap, "experiments", "__init__.py"), "w").close()
    e = dict(os.environ, WORKER_PY=str(env["py"]), WORKER_LOG_ROOT=str(env["logs"]),
             FAKE_PY_LOG=str(env["log"]), CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(["bash", POOL, snap, str(env["items"]), "--dry-run"],
                          capture_output=True, text=True, timeout=180, env=e)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "resolves    sven" in proc.stdout
    assert str(repos["results"]) in os.path.realpath(
        os.path.join(snap, "experiment_results"))
