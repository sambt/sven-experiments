"""Work claiming, done-marking and manifests for :func:`generic_scan.scan`.

The bookkeeping side of the campaign runner (C-R1, C-R2, C-R3 and the "Dedup" /
"Scheduling" decisions in ``campaign/CONTRACTS.md``). Like ``grid.py`` this
module is **stdlib only** (no torch, no hydra, no numpy) so a launcher and
``tools/reconcile.py`` can use it without paying for a CUDA import.

Four independent mechanisms, all living in subdirectories of the scan dir so
the scan dir's own listing stays small:

``{scan}/claims/{run_id}.claim`` (``.claim.1``, ``.claim.2``, ... after a takeover)
    *Who is running this right now.* A dynamic work queue replaces static
    shards (CONTRACTS "Scheduling"): every worker walks the **whole** grid and
    runs the specs it can claim, so a resubmitted job mops up whatever is left
    instead of re-running its own static slice. A claim is an
    ``os.open(O_CREAT|O_EXCL)`` create -- atomic on Lustre and on a local file
    system -- holding host, pid, SLURM job id (``None`` off SLURM, e.g. on a
    rented pod), restart count and start time. A :class:`Heartbeat` touches it
    every 60 s; a claim whose mtime is older than ``CLAIM_TIMEOUT_S`` (10 min)
    belongs to a dead worker and is taken over by creating the **next
    generation** file, again with ``O_EXCL``, so a takeover is one atomic
    operation and cannot be won twice (:func:`try_claim` explains why a rename
    of the stale claim is not enough). The lock is held by the highest existing
    generation; the older, stale ones stay behind as the record of how many
    workers died on that run. Claims are locks, not results: releasing one
    deletes it.

``{scan}/done/{run_id}.{hash8}.{status}``
    *What is finished, and for which code/config generation.* A zero-byte
    marker written LAST, after the ckpt, the npz and the jsonl. One
    ``os.listdir`` per process start (:func:`done_index`) answers "skip or run"
    for the whole grid with no JSON reads -- the point of putting the hash in
    the filename rather than reading ~14k records off Lustre. ``ok`` and
    ``diverged`` count as done; ``oom`` and ``error`` are retried. A marker with
    a *different* hash8 means the config or the code changed: that generation's
    files are **moved** to ``{scan}/_stale/{old_hash8}/`` (never deleted) and the
    run executes again.

``{scan}/started/{run_id}.started``
    *What was started but never finished.* Written at run start and removed on
    completion, whatever the status (C-R1). A marker with no record is a timeout
    or a hard crash -- the one failure mode that cannot write its own record.
    Distinct from a claim: a claim is live state, a started marker is evidence.

``{scan}/manifest/{job}[.shard{i}].json``
    *What the scan intends to contain.* Each **worker** records the run_ids it is
    responsible for; the union over workers is the intended grid, which is what
    ``tools/reconcile.py`` counts against (C-R2). Workers of one SLURM job share
    ``$SLURM_JOB_ID``, so the file name must carry the shard id (see
    :func:`write_manifest`) or the last writer's slice replaces the others' and
    the "union" is a strict subset of the grid.

The order the runner must use
-----------------------------
The done index is ONE ``os.listdir`` per process start, so by the time a worker
reaches a grid point another worker may have finished it. The index is therefore
only a *cheap filter*; the decision to run, to retire an old generation or to
skip is taken **under the claim**, with a fresh :func:`is_done_now`::

    index = done_index(scan_dir)                      # once per process
    for spec in specs:                                 # the whole grid
        if is_done(index, spec.run_id, h8):            # cheap skip, no stat
            continue
        claim = try_claim(scan_dir, spec.run_id)
        if claim is None:                              # somebody is on it
            continue
        try:
            if is_done_now(scan_dir, spec.run_id, h8):  # finished since the listdir
                continue
            for old in stale_hashes(index, spec.run_id, h8):
                move_to_stale(scan_dir, spec.run_id, old)
            with Heartbeat(claim):
                mark_started(scan_dir, spec.run_id)
                ...                                     # execute, write ckpt/npz/jsonl
                mark_done(scan_dir, spec.run_id, h8, status)   # LAST write
        finally:
            clear_started(scan_dir, spec.run_id)
            claim.release()

Claiming *before* the fresh check and before :func:`move_to_stale` is what makes
the sequence safe: a claim is the only thing that stops a second worker from
moving results a first worker has just written (a released claim is gone, so a
finished run is re-claimable -- the done marker, not the claim, is the record).
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import socket
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Layout and constants
# ---------------------------------------------------------------------------

CLAIMS_DIRNAME = "claims"
DONE_DIRNAME = "done"
STARTED_DIRNAME = "started"
MANIFEST_DIRNAME = "manifest"
STALE_DIRNAME = "_stale"

CLAIM_SUFFIX = ".claim"
STARTED_SUFFIX = ".started"

#: a claim not touched for this long is stale and may be taken over (CONTRACTS)
CLAIM_TIMEOUT_S = 600.0
#: the heartbeat interval; must stay well inside CLAIM_TIMEOUT_S
HEARTBEAT_INTERVAL_S = 60.0
#: how many workers may die on one run before it is left to `tools/reconcile.py`
MAX_CLAIM_GENERATIONS = 64

#: the statuses a record / done marker can carry (C-R1)
STATUSES = ("ok", "diverged", "oom", "error")
#: the statuses that count as done; `oom` and `error` are retried (CONTRACTS "Dedup")
DONE_STATUSES = frozenset({"ok", "diverged"})

#: a run's artefacts, relative to the scan dir, in write order (ckpt, npz, jsonl);
#: the done marker is appended per status by :func:`move_to_stale`.
ARTEFACT_TEMPLATES = (
    os.path.join("ckpt", "{run_id}.pt"),
    os.path.join("diag", "{run_id}.npz"),
    "{run_id}.jsonl",
)


def _slurm_job_id():
    """The SLURM job id, or ``None`` when running outside SLURM."""
    for key in ("SLURM_JOB_ID", "SLURM_JOBID"):
        value = os.environ.get(key)
        if value:
            return str(value)
    return None


def claim_path(scan_dir, run_id):
    """The run's generation-0 claim file; later generations append ``.1``, ``.2``."""
    return os.path.join(scan_dir, CLAIMS_DIRNAME, run_id + CLAIM_SUFFIX)


def started_path(scan_dir, run_id):
    return os.path.join(scan_dir, STARTED_DIRNAME, run_id + STARTED_SUFFIX)


def done_path(scan_dir, run_id, hash8, status):
    return os.path.join(scan_dir, DONE_DIRNAME, f"{run_id}.{hash8}.{status}")


def manifest_path(scan_dir, job_name, shard_id=None):
    """One worker's manifest file; ``shard_id`` separates the workers of one job."""
    name = _safe_name(job_name)
    if shard_id is not None:
        name = f"{name}.shard{int(shard_id)}"
    return os.path.join(scan_dir, MANIFEST_DIRNAME, name + ".json")


def stale_dir(scan_dir, hash8):
    return os.path.join(scan_dir, STALE_DIRNAME, hash8)


def _safe_name(name):
    """A file-name-safe form of a job name (path separators are the real risk).

    Sanitising alone would map distinct job names onto one file (``scan/a`` and
    ``scan_a``), which under C-R2 silently drops a worker's slice from the union,
    so a sanitised name gets a digest of the original appended.
    """
    keep = "-_."
    text = str(name)
    safe = "".join(c if (c.isalnum() or c in keep) else "_" for c in text).strip(".")
    if safe != text:
        safe = f"{safe or 'job'}-{hashlib.sha1(text.encode()).hexdigest()[:6]}"
    return safe


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path):
    """Parse a JSON file, or ``None`` if it is missing or not (yet) valid JSON."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_json_atomic(path, payload):
    """Write JSON via a temp file in the same directory + ``os.replace``, so a
    reader never sees a half-written file (manifests are read by other jobs)."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True, default=repr)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _move(src, dst):
    """Rename, falling back to a copy+delete across file systems.

    Returns ``dst``, or ``None`` when ``src`` vanished between the caller's
    existence check and the rename: another worker moved it first, which is the
    normal case when several workers of a resumed job retire the same stale
    generation. Crashing there would kill most workers of an allocation seconds
    after it starts, so an already-moved file is not an error.
    """
    try:
        os.rename(src, dst)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        try:
            shutil.move(src, dst)
        except FileNotFoundError:
            return None
    return dst


def _unique(path):
    """``path`` if free, else ``path.1``, ``path.2``, ... -- nothing is overwritten."""
    if not os.path.lexists(path):
        return path
    for i in range(1, 1000):
        candidate = f"{path}.{i}"
        if not os.path.lexists(candidate):
            return candidate
    return f"{path}.{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Claims: the dynamic work queue
# ---------------------------------------------------------------------------

def claim_info(**extra):
    """The identity a worker writes into its claim / started marker.

    ``restarts`` starts from ``SLURM_RESTART_COUNT`` (set by SLURM on a requeue)
    and is incremented by every stale takeover, so it counts "how many workers
    died on this run" -- the number reconcile needs to spot a poison run.
    """
    try:
        restarts = int(os.environ.get("SLURM_RESTART_COUNT") or 0)
    except ValueError:
        restarts = 0
    info = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": _slurm_job_id(),
        "restarts": restarts,
        "start_time": time.time(),
        "start_time_iso": _now_iso(),
    }
    info.update(extra)
    return info


class Claim:
    """A held claim file. Returned by :func:`try_claim`; ``release()`` drops it.

    ``gen`` is the claim's generation: 0 for the first worker on this run, 1 for
    the worker that took it over from a dead gen-0 worker, and so on.
    """

    __slots__ = ("path", "base", "run_id", "info", "took_over", "gen")

    def __init__(self, base, run_id, info, gen=0):
        self.base = base
        self.gen = int(gen)
        self.path = _gen_path(base, self.gen)
        self.run_id = run_id
        self.info = info
        self.took_over = self.gen > 0

    def touch(self):
        """Refresh the claim's mtime (one heartbeat). False if it is gone."""
        try:
            os.utime(self.path, None)
            return True
        except OSError:
            return False

    def _is_mine(self):
        """Do we still hold the lock? Our file must still carry our identity and
        no later generation may exist: a worker that hung past the timeout and
        then woke up must not delete anything, or the worker that took its run
        over would lose its lock while still running."""
        if os.path.lexists(_gen_path(self.base, self.gen + 1)):
            return False
        held = _read_json(self.path) or {}
        return all(held.get(k) == self.info.get(k)
                   for k in ("host", "pid", "start_time"))

    def release(self):
        """Delete our claim file while we still hold the lock (see
        :meth:`_is_mine`). True if this call removed it."""
        if not self._is_mine():
            return False
        try:
            os.unlink(self.path)
            return True
        except OSError:
            return False

    def heartbeat(self, interval=HEARTBEAT_INTERVAL_S):
        """A (not yet started) :class:`Heartbeat` for this claim."""
        return Heartbeat(self, interval=interval)

    def __repr__(self):
        return (f"Claim({self.run_id!r}, host={self.info.get('host')!r}, "
                f"pid={self.info.get('pid')}, took_over={self.took_over})")


def release(claim):
    """Release a :class:`Claim` (ownership-checked) or, given a path, that file."""
    if isinstance(claim, Claim):
        return claim.release()
    if claim is None:
        return False
    try:
        os.unlink(claim)
        return True
    except OSError:
        return False


def _gen_path(base, gen):
    """The claim file of one generation: ``{run_id}.claim``, then ``.1``, ``.2``, ..."""
    return base if int(gen) == 0 else f"{base}.{int(gen)}"


def _create_claim(base, run_id, info, gen=0):
    """``O_CREAT|O_EXCL`` create of one generation's claim file; ``None`` if that
    generation already exists (somebody else got there first)."""
    try:
        fd = os.open(_gen_path(base, gen), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return None
    payload = dict(info, run_id=run_id, gen=int(gen), took_over=int(gen) > 0)
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh, sort_keys=True, default=repr)
        fh.write("\n")
    return Claim(base, run_id, payload, gen=gen)


def _claim_age(path):
    """Seconds since the claim was last touched, or ``None`` if it vanished.

    Staleness is mtime age, which is what works both with and without SLURM (a
    rented pod has no ``squeue``); the heartbeat is what makes the mtime of a
    live worker trustworthy. Cluster clocks are NTP-synced, so comparing a
    remote node's mtime against the local clock is safe at a 10-minute scale.
    Only the mtime is read on the hot path -- a worker walking a claimed grid
    collides thousands of times and must not open a file per collision.
    """
    try:
        return time.time() - os.stat(path).st_mtime
    except FileNotFoundError:
        return None


def try_claim(scan_dir, run_id, info=None, *, timeout=CLAIM_TIMEOUT_S,
              max_generations=MAX_CLAIM_GENERATIONS):
    """Claim ``run_id`` for this worker. Returns a :class:`Claim`, or ``None`` if
    somebody else holds it (or won the takeover race).

    The lock is the highest existing generation of ``{run_id}.claim``. Walk the
    generations: a missing one is free and is created with ``O_EXCL`` (exactly one
    of N workers wins); a fresh one means a live worker, so give up; a stale one
    (mtime older than ``timeout``) means its worker died, and the *next*
    generation is the takeover slot.

    Taking over by **creating** the next generation, rather than renaming the
    stale claim away and re-creating it, is what makes the takeover safe:
    ``os.rename`` is atomic but not conditional on *which* file sits at the path,
    so between one worker's "this claim is stale" and its rename, the winner of
    the race can already have put a **fresh** claim there -- the loser then
    renames away a live lock and both workers run the same spec. (Observed: an
    8-process race plus a 4-process takeover race on one core reproduced it.)
    An ``O_EXCL`` create has no such window: it is one atomic operation that
    either wins or loses.
    """
    base = claim_path(scan_dir, run_id)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    my_info = claim_info() if info is None else dict(info)
    previous = {}
    gen = 0
    while gen < int(max_generations):
        age = _claim_age(_gen_path(base, gen))
        if age is None:                   # free generation: try to take it
            info_gen = dict(my_info)
            if gen:
                info_gen["restarts"] = int(previous.get("restarts") or 0) + 1
                info_gen["took_over_from"] = {k: previous.get(k) for k
                                              in ("host", "pid", "slurm_job_id")}
            claim = _create_claim(base, run_id, info_gen, gen=gen)
            if claim is not None:
                return claim
            continue                      # lost the create race: re-examine this gen
        if age <= float(timeout):
            return None                   # a live worker holds this generation
        previous = _read_json(_gen_path(base, gen)) or {}
        gen += 1                          # its worker died: the next gen takes over
    return None                           # too many dead workers: leave it to reconcile


class Heartbeat:
    """Daemon thread that touches a claim every ``interval`` seconds.

    Context-manager style, so a crash inside the run still stops the thread and
    the claim then ages out::

        with claims.Heartbeat(claim):
            execute(spec, ctx)
    """

    def __init__(self, claim, interval=HEARTBEAT_INTERVAL_S):
        self.path = claim.path if isinstance(claim, Claim) else str(claim)
        self.interval = float(interval)
        self.beats = 0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="claim-heartbeat",
                                            daemon=True)
            self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                os.utime(self.path, None)
                self.beats += 1
            except OSError:
                pass        # claim released or taken over; nothing useful to do

    def stop(self, timeout=None):
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(self.interval if timeout is None else timeout)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


# ---------------------------------------------------------------------------
# Done markers and dedup (CONTRACTS "Dedup", C-R3)
# ---------------------------------------------------------------------------

def mark_done(scan_dir, run_id, hash8, status):
    """Write the zero-byte done marker. Call this LAST, after ckpt/npz/jsonl."""
    assert status in STATUSES, status
    assert "." not in str(hash8), hash8   # the name is parsed by rsplit('.', 2)
    path = done_path(scan_dir, run_id, hash8, status)
    os.makedirs(os.path.join(scan_dir, DONE_DIRNAME), exist_ok=True)
    with open(path, "wb"):
        pass
    return path


def done_index(scan_dir):
    """``{run_id: {hash8: {status, ...}}}`` from ONE ``os.listdir`` of ``done/``.

    One directory listing per process start answers dedup for the whole grid;
    no record is opened. Several statuses can share a hash (an ``error`` that was
    later retried into an ``ok``), hence the set of statuses per hash.
    Non-conforming file names are ignored.
    """
    index = {}
    try:
        names = os.listdir(os.path.join(scan_dir, DONE_DIRNAME))
    except FileNotFoundError:
        return index
    for name in names:
        parts = name.rsplit(".", 2)       # run_ids contain '.' (lr0.01), hashes do not
        if len(parts) != 3:
            continue
        run_id, hash_, status = parts
        if status not in STATUSES or not run_id:
            continue
        index.setdefault(run_id, {}).setdefault(hash_, set()).add(status)
    return index


def is_done(index, run_id, hash8):
    """True iff this run has a marker with *this* hash and status ok/diverged."""
    statuses = index.get(run_id, {}).get(hash8, ())
    return any(s in DONE_STATUSES for s in statuses)


def is_done_now(scan_dir, run_id, hash8):
    """:func:`is_done` re-checked on disk for ONE run: has this generation of
    this run finished *since* the index was built?

    Two stats, paid only for the runs a worker is about to execute (after it won
    the claim), not per grid point -- which is why the cheap index skip stays.
    Mandatory before :func:`move_to_stale` / executing: the index is a snapshot
    from process start, and acting on it hours later moves a fresh result into
    ``_stale/`` and re-runs finished work.
    """
    return any(os.path.lexists(done_path(scan_dir, run_id, hash8, status))
               for status in sorted(DONE_STATUSES))


def stale_hashes(index, run_id, hash8):
    """The run's markers that belong to another generation (sorted hash8s)."""
    return sorted(h for h in index.get(run_id, {}) if h != hash8)


def move_to_stale(scan_dir, run_id, old_hash8):
    """Move one generation of a run out of the way, into ``_stale/{old_hash8}/``.

    Moves the jsonl, the diag npz, the ckpt and that generation's done markers,
    keeping their relative layout (``_stale/{h}/diag/{run_id}.npz`` etc.) so the
    analysis can still read them. **Nothing is deleted**: a destination that
    already exists gets a ``.1``, ``.2``, ... suffix. Returns the paths moved.

    Call it **while holding the run's claim** and only after a fresh
    :func:`is_done_now` says this generation is not already finished (see the
    module docstring's recipe). The artefact paths carry no hash, so a worker
    acting on an out-of-date index would otherwise move the *current*
    generation's freshly written jsonl/npz/ckpt.

    As a second line of defence nothing is moved unless one of this run's
    ``done/{run_id}.{old_hash8}.*`` markers still exists: that marker is the
    proof that the files in the scan dir still belong to the old generation, and
    it is moved last, so a worker that finds it gone is looking at results
    somebody else already retired (and possibly re-ran). A legacy scan with
    records but no ``done/`` dir is therefore never touched -- back-fill its
    markers first.
    """
    marker_rels = [os.path.join(DONE_DIRNAME, f"{run_id}.{old_hash8}.{s}")
                   for s in STATUSES]
    if not any(os.path.lexists(os.path.join(scan_dir, rel)) for rel in marker_rels):
        return []
    dest_root = stale_dir(scan_dir, old_hash8)
    rels = [t.format(run_id=run_id) for t in ARTEFACT_TEMPLATES] + marker_rels
    moved = []
    for rel in rels:
        src = os.path.join(scan_dir, rel)
        if not os.path.lexists(src):
            continue
        dst = _unique(os.path.join(dest_root, rel))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        dst = _move(src, dst)
        if dst is not None:               # None = another worker moved it first
            moved.append(dst)
    return moved


# ---------------------------------------------------------------------------
# Started markers (C-R1): a marker without a record means timeout or crash
# ---------------------------------------------------------------------------

def mark_started(scan_dir, run_id, info=None):
    """Write ``{scan}/started/{run_id}.started`` at run start."""
    return _write_json_atomic(started_path(scan_dir, run_id),
                              dict(claim_info() if info is None else info,
                                   run_id=run_id))


def clear_started(scan_dir, run_id):
    """Remove the started marker on completion (any status). True if it existed."""
    try:
        os.unlink(started_path(scan_dir, run_id))
        return True
    except OSError:
        return False


def started_ids(scan_dir):
    """The run_ids with a started marker, from ONE ``os.listdir``."""
    try:
        names = os.listdir(os.path.join(scan_dir, STARTED_DIRNAME))
    except FileNotFoundError:
        return set()
    return {n[:-len(STARTED_SUFFIX)] for n in names if n.endswith(STARTED_SUFFIX)}


# ---------------------------------------------------------------------------
# Manifests (C-R2): the scan's intended grid is the union over jobs
# ---------------------------------------------------------------------------

def write_manifest(scan_dir, job_name, run_ids, *, shard_id=None):
    """Record the run_ids this **worker** is responsible for, in
    ``manifest/{job}[.shard{shard_id}].json``.

    The file is overwritten, so ``(job_name, shard_id)`` must identify the
    worker, not the scan: the NPROC workers forked inside one SLURM job share
    ``$SLURM_JOB_ID``, and if they write only their own slice under one name the
    last writer wins and :func:`read_manifest_union` returns a strict subset of
    the intended grid -- exactly what C-R2 needs the union for. So pass
    ``shard_id`` (the launcher's ``+shard_id``) whenever more than one worker
    writes a manifest, or have every worker declare the *whole* grid.
    Overwriting a manifest that covered run_ids this call does not is the
    symptom of getting that wrong, and is warned about.
    """
    run_ids = list(run_ids)
    path = manifest_path(scan_dir, job_name, shard_id)
    previous = _read_json(path) or {}
    dropped = set(previous.get("run_ids") or ()) - set(run_ids)
    if dropped:
        print(f"  [warn] manifest {os.path.basename(path)}: rewriting drops "
              f"{len(dropped)} run_id(s) declared before; pass shard_id (or a "
              f"per-worker job_name) so workers of one job keep separate manifests")
    payload = {
        "job": str(job_name),
        "shard_id": None if shard_id is None else int(shard_id),
        "written_at": _now_iso(),
        "host": socket.gethostname(),
        "slurm_job_id": _slurm_job_id(),
        "n_runs": len(run_ids),
        "run_ids": run_ids,
    }
    return _write_json_atomic(path, payload)


def read_manifest_union(scan_dir):
    """The union of every manifest's run_ids = the scan's intended grid."""
    out = set()
    try:
        names = sorted(os.listdir(os.path.join(scan_dir, MANIFEST_DIRNAME)))
    except FileNotFoundError:
        return out
    for name in names:
        if not name.endswith(".json") or name.startswith(".tmp-"):
            continue
        payload = _read_json(os.path.join(scan_dir, MANIFEST_DIRNAME, name))
        if payload is None:
            print(f"  [warn] unreadable manifest: {name}")
            continue
        out.update(payload.get("run_ids") or ())
    return out
