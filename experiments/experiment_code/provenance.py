"""Provenance facts for every run record (C-R3).

"Which code, on which machine, when" -- the fields F16 found missing: the git
sha and dirty flag of **both** repos (``sv3`` and the separate, editable-installed
``sven``), the torch / CUDA versions, the GPU name, the host, the SLURM job id,
the shard count and the run's start / end timestamps.

Torch is imported **inside** :func:`torch_facts`, never at module import, so a
launcher or ``tools/reconcile.py`` can collect provenance without paying for a
cold ``import torch`` (1-2 min on this cluster); pass ``include_torch=False`` to
skip it entirely.

Campaign runs execute from an exported snapshot rather than the live working
tree (``EXPERIMENTS.md``), which has no ``.git``. Each root is
therefore resolved as:

1. ``{root}/.git`` exists -> ask git. The check is on the root **itself** on
   purpose: ``git rev-parse`` walks *up* the tree, so an export that happens to
   sit inside some other checkout would otherwise be labelled with that
   checkout's sha.
2. else ``{root}/DEPLOY_INFO.json`` -> read it. Schema (written by whoever makes
   the export)::

       {"git_sha": "<40 hex>", "git_dirty": false, "exported_at": "<iso>"}

   ``sha`` / ``dirty`` are accepted as aliases.
3. else all three fields are ``None`` -- collecting provenance never fails.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone

DEPLOY_INFO_NAME = "DEPLOY_INFO.json"
#: `git status --porcelain` on a big working tree is the slow call here
GIT_TIMEOUT_S = 60.0


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def now_iso():
    """Current UTC time, ISO-8601 (``datetime.fromisoformat``-parseable)."""
    return datetime.now(timezone.utc).isoformat()


def start_stamp():
    """Timestamp fields for the start of a run; pass the result to :func:`end_stamp`."""
    return {"start_time": now_iso(), "start_unix": time.time()}


def end_stamp(start):
    """``start_stamp()`` plus the end timestamps and the wall time, as one dict,
    ready for ``record.update(...)``."""
    out = dict(start or {})
    end = time.time()
    out["end_time"] = now_iso()
    out["end_unix"] = end
    if "start_unix" in out:
        out["wall_time_s"] = round(end - float(out["start_unix"]), 3)
    return out


# ---------------------------------------------------------------------------
# Repo identity
# ---------------------------------------------------------------------------

def _git(root, *args):
    """``git -C root <args>`` stdout, or ``None`` if git is unavailable / failed."""
    try:
        proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                              text=True, timeout=GIT_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _deploy_info(root):
    try:
        with open(os.path.join(str(root), DEPLOY_INFO_NAME)) as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def git_facts(root):
    """``(sha, dirty, source)`` for one repo root; ``source`` is
    ``"git"`` | ``"deploy_info"`` | ``None`` (see the module docstring)."""
    if root is None:
        return None, None, None
    root = str(root)
    if os.path.lexists(os.path.join(root, ".git")):
        sha = _git(root, "rev-parse", "HEAD")
        if sha and sha.strip():
            status = _git(root, "status", "--porcelain")
            dirty = None if status is None else bool(status.strip())
            return sha.strip(), dirty, "git"
    info = _deploy_info(root)
    if info is not None:
        sha = info.get("git_sha", info.get("sha"))
        dirty = info.get("git_dirty", info.get("dirty"))
        return (str(sha) if sha else None,
                None if dirty is None else bool(dirty),
                "deploy_info")
    return None, None, None


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def slurm_job_id():
    """The SLURM job id, or ``None`` outside SLURM (a rented pod, a laptop)."""
    for key in ("SLURM_JOB_ID", "SLURM_JOBID"):
        value = os.environ.get(key)
        if value:
            return str(value)
    return None


def torch_facts():
    """torch / CUDA versions and the GPU name. Imports torch lazily and tolerates
    it being absent or CUDA-less, so this is safe in a torch-free process."""
    facts = {"torch_version": None, "cuda_version": None, "gpu_name": None}
    try:
        import torch
    except Exception:
        return facts
    facts["torch_version"] = getattr(torch, "__version__", None)
    facts["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        if torch.cuda.is_available():
            facts["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass                              # driver hiccup: the rest is still useful
    return facts


def collect(repo_root, sven_root, *, n_shards=None, shard_id=None, include_torch=True):
    """The provenance fields every record of this job carries (C-R3).

    Collect once per job (the git calls fork a subprocess) and ``record.update``
    it into each record, together with :func:`end_stamp` for the per-run times.
    """
    facts = {}
    facts["git_sha"], facts["git_dirty"], facts["git_source"] = git_facts(repo_root)
    (facts["sven_git_sha"], facts["sven_git_dirty"],
     facts["sven_git_source"]) = git_facts(sven_root)
    facts["host"] = socket.gethostname()
    facts["slurm_job_id"] = slurm_job_id()
    facts["n_shards"] = None if n_shards is None else int(n_shards)
    facts["shard_id"] = None if shard_id is None else int(shard_id)
    facts["python_version"] = platform.python_version()
    facts.update(torch_facts() if include_torch else
                 {"torch_version": None, "cuda_version": None, "gpu_name": None})
    facts["collected_at"] = now_iso()
    return facts
