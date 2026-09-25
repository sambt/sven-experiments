"""CPU tests for ``experiments/experiment_code/claims.py``.

The contracts under test (CONTRACTS "Dedup" / "Scheduling", C-R1, C-R2, C-R3):

* a claim is won by **exactly one** worker -- checked with an 8-process race for
  2,000 run_ids on a local temp dir, and (opt-in, see ``LUSTRE_ROOT_ENV``) the
  same race once on Lustre, which is the file system the campaign runs on;
* a **stale** claim is taken over by exactly one of several workers that all see
  it expire at the same moment;
* a :class:`Heartbeat` keeps a claim out of takeover range, and stopping it puts
  the claim back in range;
* done markers: one ``os.listdir``, ``ok``/``diverged`` are done while
  ``oom``/``error`` are retried, another hash8 is stale, and ``is_done_now``
  re-checks one run against the disk;
* ``move_to_stale`` moves every artefact of one generation and deletes nothing,
  concurrent movers do not kill each other, and a generation whose done marker
  is gone is not touched at all;
* the **composed** runner recipe (index -> claim -> fresh re-check ->
  move_to_stale -> execute -> mark_done) over two hash generations, both with a
  stale index and under a 4-process race: every run executes exactly once and
  no fresh result ends up under ``_stale/``;
* started markers and manifests round-trip, with and without SLURM env vars,
  including the workers of one SLURM job writing disjoint slices.

``claims.py`` is loaded straight from its path: importing
``experiments.experiment_code`` would pull in ``generic_scan`` and hence torch.
"""

import importlib.util
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CLAIMS_PATH = REPO / "experiments" / "experiment_code" / "claims.py"

#: set this to a Lustre directory to also run the race there (see the campaign
#: report): SV3_CLAIMS_LUSTRE_ROOT=/n/labstore01/.../claims_probe pytest -s -k lustre
LUSTRE_ROOT_ENV = "SV3_CLAIMS_LUSTRE_ROOT"

N_RACE_IDS = 2000
N_RACE_PROCS = 8


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses / pickling need it registered
    spec.loader.exec_module(mod)
    return mod


claims = _load("sv3_claims_under_test", CLAIMS_PATH)

# A realistic run_id: dots (lr0.01, rtol0.1) are exactly what the marker parser
# must not choke on.
RUN_ID = "svd_bs128_k4_lr0.01_rtol0.1_svdrandomized_mseed0_lseed1"


def _age(path):
    return time.time() - os.stat(path).st_mtime


# ---------------------------------------------------------------------------
# Exactly-once under a race (local temp dir and, opt-in, Lustre)
# ---------------------------------------------------------------------------

def _claim_worker(scan_dir, run_ids, shuffle_seed, out_path, timeout, barrier):
    """Claim everything this worker can, in its own order, and report the wins."""
    import random

    ids = list(run_ids)
    random.Random(shuffle_seed).shuffle(ids)
    info = claims.claim_info()            # a worker's identity is fixed
    won = []
    barrier.wait()                        # all workers hit the directory together
    for run_id in ids:
        if claims.try_claim(scan_dir, run_id, info, timeout=timeout) is not None:
            won.append(run_id)
    with open(out_path, "w") as fh:
        json.dump(won, fh)


def _run_claim_race(scan_dir, n_ids=N_RACE_IDS, n_procs=N_RACE_PROCS):
    """Run the race; return (run_ids, [won per worker], elapsed seconds)."""
    run_ids = [f"{RUN_ID}_i{i:04d}" for i in range(n_ids)]
    out_dir = os.path.join(scan_dir, "_race_out")
    os.makedirs(out_dir, exist_ok=True)
    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(n_procs)
    procs, outs = [], []
    t0 = time.perf_counter()
    for i in range(n_procs):
        out = os.path.join(out_dir, f"won_{i}.json")
        proc = ctx.Process(target=_claim_worker,
                           args=(scan_dir, run_ids, i, out, claims.CLAIM_TIMEOUT_S,
                                 barrier))
        proc.start()
        procs.append(proc)
        outs.append(out)
    for proc in procs:
        proc.join(300)
        assert proc.exitcode == 0, proc.exitcode
    elapsed = time.perf_counter() - t0
    won = []
    for out in outs:
        with open(out) as fh:
            won.append(json.load(fh))
    return run_ids, won, elapsed


def _assert_exactly_once(scan_dir, run_ids, won, elapsed, label):
    flat = [r for w in won for r in w]
    n_attempts = len(run_ids) * len(won)
    print(f"[{label}] {len(run_ids)} ids x {len(won)} procs = {n_attempts} attempts "
          f"in {elapsed:.2f}s = {n_attempts / elapsed:.0f} try_claim/s "
          f"({len(run_ids) / elapsed:.0f} claims/s); "
          f"wins per worker: {[len(w) for w in won]}")
    assert len(flat) == len(set(flat)), "a run_id was claimed twice"
    assert set(flat) == set(run_ids), "a run_id was claimed by nobody"
    claim_files = os.listdir(os.path.join(scan_dir, claims.CLAIMS_DIRNAME))
    assert len(claim_files) == len(run_ids)
    assert sum(1 for w in won if w) >= 2, "no contention -- the race did not race"


def test_eight_process_race_claims_each_run_once(tmp_path):
    scan_dir = str(tmp_path / "toy_scan")
    run_ids, won, elapsed = _run_claim_race(scan_dir)
    _assert_exactly_once(scan_dir, run_ids, won, elapsed, "local")


@pytest.mark.skipif(not os.environ.get(LUSTRE_ROOT_ENV),
                    reason=f"set {LUSTRE_ROOT_ENV} to run the race on Lustre")
def test_eight_process_race_on_lustre():
    root = os.environ[LUSTRE_ROOT_ENV]
    os.makedirs(root, exist_ok=True)
    scan_dir = os.path.join(root, f"race_{uuid.uuid4().hex[:8]}")
    try:
        run_ids, won, elapsed = _run_claim_race(scan_dir)
        _assert_exactly_once(scan_dir, run_ids, won, elapsed, "lustre")
    finally:
        shutil.rmtree(scan_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Stale takeover
# ---------------------------------------------------------------------------

def _takeover_worker(scan_dir, run_id, out_path, timeout, barrier):
    barrier.wait()
    claim = claims.try_claim(scan_dir, run_id, timeout=timeout)
    with open(out_path, "w") as fh:
        json.dump({"won": claim is not None,
                   "took_over": bool(claim and claim.took_over),
                   "restarts": claim.info.get("restarts") if claim else None,
                   "pid": os.getpid()}, fh)


def test_stale_takeover_has_exactly_one_winner(tmp_path):
    scan_dir = str(tmp_path / "toy_scan")
    first = claims.try_claim(scan_dir, RUN_ID)
    assert first is not None and not first.took_over
    assert claims.try_claim(scan_dir, RUN_ID) is None, "a live claim is not re-claimable"

    old = time.time() - 10 * claims.CLAIM_TIMEOUT_S      # the holder died long ago
    os.utime(first.path, (old, old))

    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(4)
    procs, outs = [], []
    for i in range(4):
        out = str(tmp_path / f"takeover_{i}.json")
        proc = ctx.Process(target=_takeover_worker,
                           args=(scan_dir, RUN_ID, out, claims.CLAIM_TIMEOUT_S, barrier))
        proc.start()
        procs.append(proc)
        outs.append(out)
    for proc in procs:
        proc.join(60)
        assert proc.exitcode == 0, proc.exitcode
    results = [json.load(open(o)) for o in outs]

    winners = [r for r in results if r["won"]]
    assert len(winners) == 1, results
    assert winners[0]["took_over"] is True
    assert winners[0]["restarts"] == 1, "the takeover counts the dead worker"
    # the lock is generation 1; generation 0 stays behind as the dead worker's record
    gen1 = first.path + ".1"
    files = sorted(os.listdir(os.path.join(scan_dir, claims.CLAIMS_DIRNAME)))
    assert files == sorted([os.path.basename(first.path),
                            os.path.basename(gen1)]), files
    assert _age(gen1) < claims.CLAIM_TIMEOUT_S
    with open(gen1) as fh:
        held = json.load(fh)
    assert held["pid"] == winners[0]["pid"] and held["gen"] == 1
    assert held["took_over_from"]["pid"] == os.getpid()
    # a live generation 1 is not re-claimable; when it too dies, gen 2 takes over
    assert claims.try_claim(scan_dir, RUN_ID) is None
    os.utime(gen1, (old, old))
    third = claims.try_claim(scan_dir, RUN_ID)
    assert third is not None and third.gen == 2 and third.info["restarts"] == 2


def test_stale_takeover_race_over_many_runs(tmp_path):
    """The takeover must be exactly-once *reliably*, not only when the timing is
    kind: 4 workers race over 300 stale claims at once, so a window between
    "this is stale" and "it is mine" shows up as a duplicate here.
    """
    scan_dir = str(tmp_path / "toy_scan")
    n_ids = 600
    run_ids = [f"{RUN_ID}_i{i:03d}" for i in range(n_ids)]
    old = time.time() - 10 * claims.CLAIM_TIMEOUT_S
    for run_id in run_ids:                             # a whole node's worth of deaths
        claim = claims.try_claim(scan_dir, run_id)
        os.utime(claim.path, (old, old))

    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir)
    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(4)
    procs, outs = [], []
    for i in range(4):
        out = os.path.join(out_dir, f"won_{i}.json")
        proc = ctx.Process(target=_claim_worker,
                           args=(scan_dir, run_ids, i, out, claims.CLAIM_TIMEOUT_S,
                                 barrier))
        proc.start()
        procs.append(proc)
        outs.append(out)
    for proc in procs:
        proc.join(300)
        assert proc.exitcode == 0, proc.exitcode
    won = [r for out in outs for r in json.load(open(out))]

    assert len(won) == len(set(won)), "a stale run was taken over twice"
    assert set(won) == set(run_ids), "a stale run was taken over by nobody"
    # on disk: generation 0 (dead) + generation 1 (the one winner), never a gen 2
    claims_dir = os.path.join(scan_dir, claims.CLAIMS_DIRNAME)
    names = set(os.listdir(claims_dir))
    assert len(names) == 2 * n_ids, len(names)
    for run_id in run_ids:
        base = os.path.basename(claims.claim_path(scan_dir, run_id))
        assert base in names and f"{base}.1" in names and f"{base}.2" not in names


def test_release_frees_the_claim(tmp_path):
    scan_dir = str(tmp_path / "toy_scan")
    claim = claims.try_claim(scan_dir, RUN_ID)
    assert claims.try_claim(scan_dir, RUN_ID) is None
    assert claims.release(claim) is True
    assert claims.release(claim) is False                # idempotent
    again = claims.try_claim(scan_dir, RUN_ID)
    assert again is not None and not again.took_over


def test_release_does_not_drop_a_claim_taken_over_by_somebody_else(tmp_path):
    """A worker that hung past the timeout and woke up must not delete the claim
    of the worker now running its spec."""
    scan_dir = str(tmp_path / "toy_scan")
    hung = claims.try_claim(scan_dir, RUN_ID)
    old = time.time() - 10 * claims.CLAIM_TIMEOUT_S
    os.utime(hung.path, (old, old))
    successor = claims.try_claim(scan_dir, RUN_ID,
                                 dict(claims.claim_info(), pid=999999))
    assert successor is not None and successor.took_over and successor.gen == 1

    assert hung.release() is False
    assert os.path.exists(successor.path), "the successor's claim was deleted"
    assert claims.try_claim(scan_dir, RUN_ID) is None, "the successor still holds it"
    assert successor.release() is True


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

def test_heartbeat_keeps_a_claim_fresh(tmp_path):
    scan_dir = str(tmp_path / "toy_scan")
    timeout = 1.0                          # 600 s in production; scaled for the test
    claim = claims.try_claim(scan_dir, RUN_ID)
    with claims.Heartbeat(claim, interval=0.1) as hb:
        time.sleep(1.5)                    # >> timeout: without beats it would expire
        assert _age(claim.path) < timeout
        assert claims.try_claim(scan_dir, RUN_ID, timeout=timeout) is None
        assert hb.beats >= 3, hb.beats
    time.sleep(timeout + 0.3)              # heartbeat stopped -> the claim ages out
    taken = claims.try_claim(scan_dir, RUN_ID, timeout=timeout)
    assert taken is not None and taken.took_over


# ---------------------------------------------------------------------------
# Done markers, dedup and _stale
# ---------------------------------------------------------------------------

def test_done_index_is_one_listdir(tmp_path, monkeypatch):
    scan_dir = str(tmp_path / "toy_scan")
    for i in range(5):
        claims.mark_done(scan_dir, f"{RUN_ID}_i{i}", "abcd1234", "ok")
    calls = []
    real_listdir = os.listdir

    def counting_listdir(path):
        calls.append(path)
        return real_listdir(path)

    monkeypatch.setattr(os, "listdir", counting_listdir)
    index = claims.done_index(scan_dir)
    assert len(calls) == 1, calls
    assert len(index) == 5


def test_done_index_status_and_hash_semantics(tmp_path):
    scan_dir = str(tmp_path / "toy_scan")
    assert claims.done_index(scan_dir) == {}          # no done/ dir yet
    hash_a, hash_b = "abcd1234", "ffff0000"
    claims.mark_done(scan_dir, RUN_ID, hash_a, "ok")
    claims.mark_done(scan_dir, RUN_ID + "_div", hash_a, "diverged")
    claims.mark_done(scan_dir, RUN_ID + "_oom", hash_a, "oom")
    claims.mark_done(scan_dir, RUN_ID + "_err", hash_a, "error")
    claims.mark_done(scan_dir, RUN_ID + "_retried", hash_a, "error")
    claims.mark_done(scan_dir, RUN_ID + "_retried", hash_a, "ok")
    claims.mark_done(scan_dir, RUN_ID + "_old", hash_b, "ok")
    # files that are not markers are ignored
    Path(scan_dir, claims.DONE_DIRNAME, "README").write_text("x")
    Path(scan_dir, claims.DONE_DIRNAME, f"{RUN_ID}.jsonl").write_text("x")

    index = claims.done_index(scan_dir)
    assert claims.is_done(index, RUN_ID, hash_a) is True
    assert claims.is_done(index, RUN_ID + "_div", hash_a) is True
    assert claims.is_done(index, RUN_ID + "_oom", hash_a) is False   # retried
    assert claims.is_done(index, RUN_ID + "_err", hash_a) is False   # retried
    assert claims.is_done(index, RUN_ID + "_retried", hash_a) is True
    assert claims.is_done(index, RUN_ID + "_missing", hash_a) is False
    # the same run under a different hash is NOT done: the config or code changed
    assert claims.is_done(index, RUN_ID, hash_b) is False
    assert claims.stale_hashes(index, RUN_ID, hash_b) == [hash_a]
    assert claims.stale_hashes(index, RUN_ID, hash_a) == []
    assert claims.stale_hashes(index, RUN_ID + "_old", hash_a) == [hash_b]
    assert claims.stale_hashes(index, "never_ran", hash_a) == []
    assert "README" not in index and f"{RUN_ID}" in index


def test_is_done_now_rechecks_disk_per_run(tmp_path):
    """The fresh re-check a worker does after winning the claim: same semantics
    as ``is_done``, but from the file system rather than the snapshot."""
    scan_dir = str(tmp_path / "toy_scan")
    hash_a, hash_b = "abcd1234", "ffff0000"
    assert claims.is_done_now(scan_dir, RUN_ID, hash_a) is False   # no done/ dir
    claims.mark_done(scan_dir, RUN_ID, hash_a, "error")
    assert claims.is_done_now(scan_dir, RUN_ID, hash_a) is False   # retried
    claims.mark_done(scan_dir, RUN_ID, hash_a, "ok")
    assert claims.is_done_now(scan_dir, RUN_ID, hash_a) is True
    assert claims.is_done_now(scan_dir, RUN_ID, hash_b) is False   # other generation
    assert claims.is_done_now(scan_dir, "never_ran", hash_a) is False
    claims.mark_done(scan_dir, RUN_ID + "_div", hash_a, "diverged")
    assert claims.is_done_now(scan_dir, RUN_ID + "_div", hash_a) is True
    # what the index could not know: the marker appeared after the listdir
    index = claims.done_index(scan_dir)
    claims.mark_done(scan_dir, RUN_ID + "_late", hash_a, "ok")
    assert claims.is_done(index, RUN_ID + "_late", hash_a) is False
    assert claims.is_done_now(scan_dir, RUN_ID + "_late", hash_a) is True


def _write_artefacts(scan_dir, run_id, hash8):
    """The three files a run writes; every body names its generation's hash."""
    paths = {}
    for rel, body in (
            (f"{run_id}.jsonl", f'{{"run_id": "{run_id}", "hash": "{hash8}"}}\n'),
            (os.path.join("diag", f"{run_id}.npz"), f"npz-{hash8}"),
            (os.path.join("ckpt", f"{run_id}.pt"), f"ckpt-{hash8}")):
        path = Path(scan_dir, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        paths[rel] = body
    return paths


def _make_run_artefacts(scan_dir, run_id, hash8, status="ok"):
    """The three files a finished run leaves, plus its done marker (written last)."""
    paths = _write_artefacts(scan_dir, run_id, hash8)
    paths[os.path.join(claims.DONE_DIRNAME, f"{run_id}.{hash8}.{status}")] = ""
    claims.mark_done(scan_dir, run_id, hash8, status)
    return paths


def test_move_to_stale_moves_every_artefact(tmp_path):
    scan_dir = str(tmp_path / "toy_scan")
    old, new = "aaaa1111", "bbbb2222"
    written = _make_run_artefacts(scan_dir, RUN_ID, old)
    other = _make_run_artefacts(scan_dir, RUN_ID + "_other", old)
    claims.mark_done(scan_dir, RUN_ID, new, "ok")        # this generation stays

    moved = claims.move_to_stale(scan_dir, RUN_ID, old)
    assert len(moved) == 4, moved
    for rel, body in written.items():
        assert not Path(scan_dir, rel).exists(), f"{rel} was left behind"
        dest = Path(claims.stale_dir(scan_dir, old), rel)
        assert dest.exists(), f"{rel} did not arrive under _stale/"
        assert dest.read_text() == body, f"{rel} lost its content"
    # the current generation's marker and every other run are untouched
    assert Path(claims.done_path(scan_dir, RUN_ID, new, "ok")).exists()
    for rel in other:
        assert Path(scan_dir, rel).exists(), f"{rel} of another run was moved"
    index = claims.done_index(scan_dir)
    assert claims.stale_hashes(index, RUN_ID, new) == []
    assert claims.is_done(index, RUN_ID, new) is True

    # a second generation with the same hash must not overwrite the first
    _make_run_artefacts(scan_dir, RUN_ID, old)
    moved2 = claims.move_to_stale(scan_dir, RUN_ID, old)
    assert len(moved2) == 4 and not set(moved2) & set(moved)
    assert all(p.endswith(".1") for p in moved2), moved2


def test_move_to_stale_tolerates_missing_files(tmp_path):
    scan_dir = str(tmp_path / "toy_scan")
    Path(scan_dir).mkdir(parents=True)
    assert claims.move_to_stale(scan_dir, RUN_ID, "aaaa1111") == []


def test_move_to_stale_requires_the_old_generations_marker(tmp_path):
    """The artefact names carry no hash, so the old generation's done marker is
    the only proof that the files on disk are the old ones. Without it (another
    worker already retired that generation, or a legacy scan has no ``done/``)
    move_to_stale must be a no-op instead of moving whatever is there."""
    scan_dir = str(tmp_path / "toy_scan")
    old, new = "aaaa1111", "bbbb2222"
    _write_artefacts(scan_dir, RUN_ID, new)              # the current generation
    claims.mark_done(scan_dir, RUN_ID, new, "ok")

    assert claims.move_to_stale(scan_dir, RUN_ID, old) == []
    assert Path(scan_dir, f"{RUN_ID}.jsonl").read_text().count(new) == 1
    assert not Path(scan_dir, claims.STALE_DIRNAME).exists()


def _mover_worker(scan_dir, run_ids, old_hash8, out_path, barrier):
    """Retire the same stale generation as every other worker, and report."""
    err = None
    barrier.wait()
    try:
        for run_id in run_ids:
            claims.move_to_stale(scan_dir, run_id, old_hash8)
    except BaseException as exc:                          # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    Path(out_path).write_text(json.dumps(err))


def test_concurrent_move_to_stale_does_not_kill_workers(tmp_path):
    """All workers of a resumed job build the same index, walk the grid in the
    same order and reach the first stale run at the same moment. A vanished
    source is then normal, not an error (it used to raise FileNotFoundError and
    kill 3 of 4 workers on the very first run)."""
    scan_dir = str(tmp_path / "toy_scan")
    old = "aaaa1111"
    run_ids = [f"{RUN_ID}_i{i:03d}" for i in range(200)]
    for run_id in run_ids:
        _make_run_artefacts(scan_dir, run_id, old)

    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(4)
    procs, outs = [], []
    for i in range(4):
        out = str(tmp_path / f"mover_{i}.json")
        proc = ctx.Process(target=_mover_worker,
                           args=(scan_dir, run_ids, old, out, barrier))
        proc.start()
        procs.append(proc)
        outs.append(out)
    for proc in procs:
        proc.join(120)
        assert proc.exitcode == 0, proc.exitcode
    errs = [json.load(open(o)) for o in outs]
    assert errs == [None] * 4, errs

    # every artefact of every run arrived exactly once, nothing was left behind
    dest = Path(claims.stale_dir(scan_dir, old))
    names = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
    assert len(names) == 4 * len(run_ids), len(names)
    assert not any(n.endswith(".1") for n in names), "an artefact was moved twice"
    for run_id in run_ids:
        assert not Path(scan_dir, f"{run_id}.jsonl").exists()
        assert not Path(claims.done_path(scan_dir, run_id, old, "ok")).exists()


# ---------------------------------------------------------------------------
# The composed runner recipe (CONTRACTS "Dedup", C-R3): index -> claim ->
# fresh re-check -> move_to_stale -> execute -> mark_done, across two hash
# generations. The individual primitives can all be correct while the
# composition destroys results, which is what these two tests cover.
# ---------------------------------------------------------------------------

def _recipe_pass(scan_dir, run_ids, index, hash8, out_path=None):
    """One worker's pass over the whole grid, exactly as the module docstring of
    ``claims.py`` prescribes. Returns the run_ids this worker executed."""
    executed = []
    for run_id in run_ids:
        if claims.is_done(index, run_id, hash8):       # cheap skip on the snapshot
            continue
        claim = claims.try_claim(scan_dir, run_id)
        if claim is None:                              # another worker has it
            continue
        try:
            if claims.is_done_now(scan_dir, run_id, hash8):
                continue                               # finished since the listdir
            for old in claims.stale_hashes(index, run_id, hash8):
                claims.move_to_stale(scan_dir, run_id, old)
            claims.mark_started(scan_dir, run_id)
            _write_artefacts(scan_dir, run_id, hash8)  # "the run"
            claims.mark_done(scan_dir, run_id, hash8, "ok")   # LAST write
            executed.append(run_id)
        finally:
            claims.clear_started(scan_dir, run_id)
            claim.release()
    if out_path is not None:
        Path(out_path).write_text(json.dumps(executed))
    return executed


def _assert_one_clean_generation(scan_dir, run_ids, old, new, executed):
    flat = [r for w in executed for r in w]
    assert len(flat) == len(set(flat)), "a run was executed twice"
    assert set(flat) == set(run_ids), "a run was executed by nobody"
    for run_id in run_ids:
        jsonl = Path(scan_dir, f"{run_id}.jsonl")
        assert jsonl.exists(), f"{run_id}: the fresh record is gone"
        assert new in jsonl.read_text(), f"{run_id}: record is not this generation"
        assert Path(scan_dir, "diag", f"{run_id}.npz").read_text() == f"npz-{new}"
        assert Path(scan_dir, "ckpt", f"{run_id}.pt").read_text() == f"ckpt-{new}"
        assert Path(claims.done_path(scan_dir, run_id, new, "ok")).exists()
    # _stale holds exactly the old generation: one copy each, nothing fresh
    dest = Path(claims.stale_dir(scan_dir, old))
    stale_files = [p for p in dest.rglob("*") if p.is_file()]
    assert len(stale_files) == 4 * len(run_ids), len(stale_files)
    for path in Path(scan_dir, claims.STALE_DIRNAME).rglob("*"):
        if path.is_file():
            assert not path.name.endswith(".1"), f"moved twice: {path}"
            assert new not in path.read_text(), f"fresh result under _stale: {path}"
    # bookkeeping is clean: no claims held, no started markers, no stale hashes
    assert os.listdir(os.path.join(scan_dir, claims.CLAIMS_DIRNAME)) == []
    assert claims.started_ids(scan_dir) == set()
    index = claims.done_index(scan_dir)
    for run_id in run_ids:
        assert claims.is_done(index, run_id, new) is True
        assert claims.stale_hashes(index, run_id, new) == []


def test_recipe_over_two_generations_with_a_stale_index(tmp_path):
    """Two workers sharing a process-start index over a scan that already holds
    an old generation: the second must neither re-run finished work nor move the
    first worker's fresh results into _stale/ (C-R3 "change num_epochs and
    resubmit")."""
    scan_dir = str(tmp_path / "toy_scan")
    old, new = "aaaa1111", "bbbb2222"
    run_ids = [f"{RUN_ID}_i{i}" for i in range(6)]
    for run_id in run_ids:
        _make_run_artefacts(scan_dir, run_id, old)

    index_a = claims.done_index(scan_dir)             # both workers start together
    index_b = claims.done_index(scan_dir)
    executed_a = _recipe_pass(scan_dir, run_ids, index_a, new)
    executed_b = _recipe_pass(scan_dir, run_ids, index_b, new)   # hours later
    assert executed_b == [], "a finished run was re-executed on a stale index"
    _assert_one_clean_generation(scan_dir, run_ids, old, new, [executed_a, executed_b])


def _recipe_worker(scan_dir, run_ids, index, hash8, shuffle_seed, out_path, barrier):
    import random

    ids = list(run_ids)
    random.Random(shuffle_seed).shuffle(ids)          # different order per worker
    barrier.wait()
    _recipe_pass(scan_dir, ids, index, hash8, out_path=out_path)


def test_recipe_race_four_workers_over_two_generations(tmp_path):
    """The same sequence run by 4 processes at once, all holding the same
    process-start index."""
    scan_dir = str(tmp_path / "toy_scan")
    old, new = "aaaa1111", "bbbb2222"
    run_ids = [f"{RUN_ID}_i{i:03d}" for i in range(150)]
    for run_id in run_ids:
        _make_run_artefacts(scan_dir, run_id, old)
    index = claims.done_index(scan_dir)

    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(4)
    procs, outs = [], []
    for i in range(4):
        out = str(tmp_path / f"recipe_{i}.json")
        proc = ctx.Process(target=_recipe_worker,
                           args=(scan_dir, run_ids, index, new, i, out, barrier))
        proc.start()
        procs.append(proc)
        outs.append(out)
    for proc in procs:
        proc.join(180)
        assert proc.exitcode == 0, proc.exitcode
    executed = [json.load(open(o)) for o in outs]
    print(f"[recipe race] executions per worker: {[len(e) for e in executed]}")
    assert sum(1 for e in executed if e) >= 2, "no contention -- the race did not race"
    _assert_one_clean_generation(scan_dir, run_ids, old, new, executed)


# ---------------------------------------------------------------------------
# Started markers (C-R1)
# ---------------------------------------------------------------------------

def test_started_marker_lifecycle(tmp_path):
    scan_dir = str(tmp_path / "toy_scan")
    assert claims.started_ids(scan_dir) == set()          # no started/ dir yet
    path = claims.mark_started(scan_dir, RUN_ID)
    claims.mark_started(scan_dir, RUN_ID + "_crashed")
    assert claims.started_ids(scan_dir) == {RUN_ID, RUN_ID + "_crashed"}
    info = json.loads(Path(path).read_text())
    assert info["run_id"] == RUN_ID and info["pid"] == os.getpid()
    assert info["host"] and "start_time" in info

    # the run that finished clears its marker; the crashed one does not, so
    # "started but no record" isolates exactly it (what reconcile reports)
    claims.mark_done(scan_dir, RUN_ID, "abcd1234", "ok")
    assert claims.clear_started(scan_dir, RUN_ID) is True
    assert claims.clear_started(scan_dir, RUN_ID) is False
    index = claims.done_index(scan_dir)
    assert {r for r in claims.started_ids(scan_dir) if r not in index} == \
        {RUN_ID + "_crashed"}


# ---------------------------------------------------------------------------
# Manifests (C-R2) and claim identity with / without SLURM
# ---------------------------------------------------------------------------

def test_manifest_union(tmp_path):
    scan_dir = str(tmp_path / "toy_scan")
    assert claims.read_manifest_union(scan_dir) == set()   # no manifest/ dir yet
    a = [f"{RUN_ID}_i{i}" for i in range(5)]
    b = [f"{RUN_ID}_i{i}" for i in range(3, 9)]            # overlapping slices
    path_a = claims.write_manifest(scan_dir, "toy_scan_shard0_of2_12345", a)
    claims.write_manifest(scan_dir, "toy_scan/shard1_of2", b)   # sanitised name
    assert claims.read_manifest_union(scan_dir) == set(a) | set(b)
    payload = json.loads(Path(path_a).read_text())
    assert payload["n_runs"] == 5 and payload["run_ids"] == a
    assert payload["job"] == "toy_scan_shard0_of2_12345" and payload["written_at"]
    assert "/" not in os.path.basename(
        claims.manifest_path(scan_dir, "toy_scan/shard1_of2"))
    # rewriting a job's manifest replaces it (a requeued job re-declares its work)
    claims.write_manifest(scan_dir, "toy_scan_shard0_of2_12345", a[:2])
    assert claims.read_manifest_union(scan_dir) == set(a[:2]) | set(b)


def test_manifest_union_over_workers_of_one_slurm_job(tmp_path, monkeypatch, capsys):
    """C-R2's union is the intended grid, so the NPROC workers forked inside ONE
    SLURM job (same ``$SLURM_JOB_ID``, disjoint slices) must not share a manifest
    file -- the last writer would otherwise replace the others' slices."""
    scan_dir = str(tmp_path / "toy_scan")
    monkeypatch.setenv("SLURM_JOB_ID", "46123456")          # shared by the workers
    job = "toy_scan_46123456"
    a = [f"{RUN_ID}_i{i}" for i in range(4)]
    b = [f"{RUN_ID}_i{i}" for i in range(4, 9)]              # disjoint slices
    path_a = claims.write_manifest(scan_dir, job, a, shard_id=0)
    path_b = claims.write_manifest(scan_dir, job, b, shard_id=1)
    assert path_a != path_b
    assert claims.read_manifest_union(scan_dir) == set(a) | set(b)
    payload = json.loads(Path(path_b).read_text())
    assert payload["shard_id"] == 1 and payload["slurm_job_id"] == "46123456"
    assert capsys.readouterr().out == ""                     # nothing to warn about

    # forgetting shard_id is the bug: one file, the second slice replaces the
    # first -- which write_manifest warns about instead of losing it silently
    claims.write_manifest(scan_dir, job + "_noshard", a)
    claims.write_manifest(scan_dir, job + "_noshard", b)
    assert claims.read_manifest_union(scan_dir) == set(a) | set(b)   # via the shards
    warning = capsys.readouterr().out
    assert "shard_id" in warning and "drops 4" in warning, warning

    # distinct job names must not collide after sanitisation
    assert (claims.manifest_path(scan_dir, "toy_scan/a")
            != claims.manifest_path(scan_dir, "toy_scan_a"))


def test_claim_identity_with_and_without_slurm(tmp_path, monkeypatch):
    for key in ("SLURM_JOB_ID", "SLURM_JOBID", "SLURM_RESTART_COUNT"):
        monkeypatch.delenv(key, raising=False)
    info = claims.claim_info()
    assert info["slurm_job_id"] is None and info["restarts"] == 0
    assert info["host"] and isinstance(info["pid"], int)
    assert info["start_time"] <= time.time()

    monkeypatch.setenv("SLURM_JOB_ID", "46123456")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "2")
    claim = claims.try_claim(str(tmp_path / "toy_scan"), RUN_ID)
    held = json.loads(Path(claim.path).read_text())
    assert held["slurm_job_id"] == "46123456" and held["restarts"] == 2
    assert held["run_id"] == RUN_ID and held["took_over"] is False
