"""CPU tests for ``experiments/experiment_code/provenance.py`` (C-R3).

The contract: every record can say which code produced it, on which machine and
when -- **inside** a git checkout (the dev tree: both ``sv3`` and the nested,
separate ``sven`` repo) and **outside** one (a campaign run executes from an
exported snapshot with no ``.git``, which falls back to ``DEPLOY_INFO.json``).

Torch is imported lazily by ``provenance.torch_facts``, so these tests never
import the real torch (1-2 min cold on this cluster): the torch branch is
exercised with a stub module in ``sys.modules``, and the rest with
``include_torch=False``. ``provenance.py`` is loaded from its path so the
``experiments.experiment_code`` package ``__init__`` (which imports torch) is not
involved.
"""

import importlib.util
import json
import re
import sys
import time
import types
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROV_PATH = REPO / "experiments" / "experiment_code" / "provenance.py"
SVEN = REPO / "sven"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


provenance = _load("sv3_provenance_under_test", PROV_PATH)

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


# ---------------------------------------------------------------------------
# Inside a git checkout: the development tree, both repos
# ---------------------------------------------------------------------------

def test_collect_inside_git_checkout():
    facts = provenance.collect(REPO, SVEN, n_shards=4, shard_id=2,
                               include_torch=False)
    for prefix in ("", "sven_"):
        assert SHA_RE.match(facts[f"{prefix}git_sha"] or ""), facts
        assert isinstance(facts[f"{prefix}git_dirty"], bool)
        assert facts[f"{prefix}git_source"] == "git"
    # the two repos are genuinely different repos (F16: sven left no trace)
    assert facts["git_sha"] != facts["sven_git_sha"]
    assert facts["host"] and facts["python_version"]
    assert facts["n_shards"] == 4 and facts["shard_id"] == 2
    assert facts["torch_version"] is None            # include_torch=False
    datetime.fromisoformat(facts["collected_at"])


def test_git_dirty_reflects_the_working_tree(tmp_path):
    import subprocess

    root = tmp_path / "repo"
    root.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True,
                                    capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    (root / "a.txt").write_text("one\n")
    run("add", "a.txt")
    run("commit", "-qm", "first")
    sha, dirty, source = provenance.git_facts(root)
    assert SHA_RE.match(sha) and dirty is False and source == "git"

    (root / "a.txt").write_text("two\n")
    sha2, dirty2, _ = provenance.git_facts(root)
    assert sha2 == sha and dirty2 is True


# ---------------------------------------------------------------------------
# Outside a git checkout: the exported snapshot the campaign runs from
# ---------------------------------------------------------------------------

def test_collect_outside_git_uses_deploy_info(tmp_path):
    repo, sven = tmp_path / "export", tmp_path / "export" / "sven"
    sven.mkdir(parents=True)
    sha_a = "a" * 40
    sha_b = "b" * 40
    (repo / provenance.DEPLOY_INFO_NAME).write_text(json.dumps(
        {"git_sha": sha_a, "git_dirty": True, "exported_at": provenance.now_iso()}))
    (sven / provenance.DEPLOY_INFO_NAME).write_text(json.dumps(
        {"sha": sha_b, "dirty": False}))          # accepted aliases

    facts = provenance.collect(repo, sven, include_torch=False)
    assert (facts["git_sha"], facts["git_dirty"], facts["git_source"]) == \
        (sha_a, True, "deploy_info")
    assert (facts["sven_git_sha"], facts["sven_git_dirty"],
            facts["sven_git_source"]) == (sha_b, False, "deploy_info")


def test_collect_outside_git_without_deploy_info(tmp_path):
    """No .git and no DEPLOY_INFO: all None, no exception, and *not* the sha of
    whatever checkout the directory happens to sit inside (git rev-parse walks up)."""
    plain = tmp_path / "plain"
    plain.mkdir()
    inside_repo = REPO / "campaign"                # inside the sv3 checkout
    for root in (plain, inside_repo):
        assert provenance.git_facts(root) == (None, None, None), root
    facts = provenance.collect(plain, plain / "sven", include_torch=False)
    assert facts["git_sha"] is None and facts["sven_git_sha"] is None
    assert facts["git_dirty"] is None and facts["git_source"] is None
    assert facts["host"]                           # the rest still collects


def test_deploy_info_is_ignored_when_git_is_present(tmp_path):
    """A snapshot that kept its .git must not be labelled by a stale sidecar."""
    import subprocess

    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "t"], check=True)
    (root / "a.txt").write_text("one\n")
    subprocess.run(["git", "-C", str(root), "add", "a.txt"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "first"], check=True)
    (root / provenance.DEPLOY_INFO_NAME).write_text('{"git_sha": "%s"}' % ("c" * 40))
    sha, _, source = provenance.git_facts(root)
    assert source == "git" and sha != "c" * 40


# ---------------------------------------------------------------------------
# Environment facts
# ---------------------------------------------------------------------------

def test_torch_facts_are_read_lazily(monkeypatch):
    stub = types.ModuleType("torch")
    stub.__version__ = "2.9.1+cu128"
    stub.version = types.SimpleNamespace(cuda="12.8")
    stub.cuda = types.SimpleNamespace(is_available=lambda: True,
                                      get_device_name=lambda i: "NVIDIA A100-SXM4-80GB")
    monkeypatch.setitem(sys.modules, "torch", stub)
    facts = provenance.torch_facts()
    assert facts == {"torch_version": "2.9.1+cu128", "cuda_version": "12.8",
                     "gpu_name": "NVIDIA A100-SXM4-80GB"}

    stub.cuda = types.SimpleNamespace(is_available=lambda: False)   # CPU node
    assert provenance.torch_facts()["gpu_name"] is None

    def boom():
        raise RuntimeError("no driver")
    stub.cuda = types.SimpleNamespace(is_available=boom)            # driver hiccup
    assert provenance.torch_facts()["torch_version"] == "2.9.1+cu128"


def test_slurm_job_id_with_and_without_slurm(monkeypatch):
    for key in ("SLURM_JOB_ID", "SLURM_JOBID"):
        monkeypatch.delenv(key, raising=False)
    assert provenance.slurm_job_id() is None
    monkeypatch.setenv("SLURM_JOB_ID", "46123456")
    assert provenance.slurm_job_id() == "46123456"
    facts = provenance.collect(REPO, SVEN, include_torch=False)
    assert facts["slurm_job_id"] == "46123456" and facts["n_shards"] is None


def test_start_and_end_stamps():
    start = provenance.start_stamp()
    datetime.fromisoformat(start["start_time"])
    time.sleep(0.05)
    stamp = provenance.end_stamp(start)
    assert stamp["start_time"] == start["start_time"]
    datetime.fromisoformat(stamp["end_time"])
    assert stamp["end_unix"] >= stamp["start_unix"]
    assert 0.04 <= stamp["wall_time_s"] < 60
    assert provenance.end_stamp({})["end_time"]          # tolerates no start
