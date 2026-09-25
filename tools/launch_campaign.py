#!/usr/bin/env python3
"""Turn a campaign plan into sbatch commands. Prints by default; submits only with --submit.

    tools/launch_campaign.py campaign/plan_campaign.yaml --phase P0
    tools/launch_campaign.py campaign/plan_campaign.yaml --phase P0 --lane a100 --submit
    tools/launch_campaign.py campaign/plan_campaign.yaml --phase P0 --lane mig --chain --submit

Every job is one `tools/worker_pool.sh <snapshot> <work-items-file>` -- fewer, longer jobs
pulling from the claim queue instead of one job per shard (campaign/scout/queue.md: a
84-job burst took 15 h to drain, a single job starts in seconds).

Lanes come from the plan; the two the campaign uses are

  a100  -p iaifi_gpu_priority,iaifi_gpu,gpu --gres=gpu:1, cpus = max NPROC + 2, 64G, 24 h
  mig   -p gpu_test --gres=gpu:4 -c 32 --mem=128G -t 12:00:00, AT MOST 2 JOBS

`--chain` (mig lane) submits a self-renewing job: as its last act it asks
`tools/reconcile.py` whether work remains and resubmits itself if so.

    MaxSubmit=2 caveat: gpu_test allows a user 2 QUEUED-OR-RUNNING jobs. A job that
    resubmits itself while still running counts as one of those two, so a chain can only
    renew itself if a slot is free at that moment. This tool therefore refuses to submit a
    chain unless a slot would stay free (1 chained job + at most 1 other), and the chain
    script retries `sbatch` for ~10 min before giving up, logging loudly either way. A
    chain that does die leaves the queue empty, not corrupted: relaunch it and the claim
    queue picks up exactly where it stopped.

    For the renewal to happen at all, the job must get control back BEFORE slurm's
    wall-clock SIGTERM (a batch shell whose foreground child is SIGTERMed never runs the
    next line). Two independent mechanisms make that so: the pool stops starting items
    ~15 min before `$SLURM_JOB_END_TIME` (`worker_pool.sh --until`), and the chain script
    asks for `--signal=B:USR1@900` and traps it.

Every job is bound to ONE snapshot: its SHAs go into the items-file header and into a
`<list>.launched.json` next to it, and a list whose jobs are still queued from a
DIFFERENT snapshot is refused (`--force` overrides). Two snapshots whose configs differ
in anything `grid.run_hash` covers give the same run_id different hash8s, and the C-R3
rule then has each lane retire the other's finished results to `_stale/` and re-run them.

Default dry run prints, per work list: the work items with the run count of each
(composed from the SNAPSHOT's configs -- the ones the jobs will actually run), any
override that names an optimizer the config does not list, and the exact sbatch command.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import campaign_plan                                  # noqa: E402  (same directory)

DEFAULT_DEPLOY_BASE = "/n/holystore01/LABS/iaifi_lab/Users/sambt/sv3_deploy"
SCRATCH_BASE = "/n/holystore01/LABS/iaifi_lab/Users/sambt/sv3_campaign_scratch"
DEFAULT_WORK_BASE = os.path.join(SCRATCH_BASE, "work")
#: logs live on holystore with everything else: the full plan is O(10^4) runner logs and
#: /n/home11 is a 95 G NFS home at 84% (a failed `> $log` redirect looks like a training
#: failure, not a disk-full one).
DEFAULT_LOG_DIR = os.path.join(SCRATCH_BASE, "logs", "campaign")
SNAPSHOT_PLACEHOLDER = "<snapshot>"
CHAIN_MAX_DEPTH = 8
CHAIN_SBATCH_RETRIES = 10
CHAIN_SBATCH_SLEEP_S = 60
#: seconds before the wall clock at which a chained job stops starting items and renews
CHAIN_SIGNAL_S = 900


def err(*args):
    """stderr, but after flushing stdout: piped output otherwise shows the error
    before the report it belongs to (stdout is block-buffered, stderr is not)."""
    sys.stdout.flush()
    print(*args, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def newest_snapshot(base=DEFAULT_DEPLOY_BASE):
    """The most recently exported COMPLETE snapshot under `base`, or None."""
    try:
        names = os.listdir(base)
    except OSError:
        return None
    done = [os.path.join(base, n) for n in names
            if os.path.exists(os.path.join(base, n, ".deploy_complete"))]
    if not done:
        return None
    return max(done, key=lambda p: os.path.getmtime(os.path.join(p, ".deploy_complete")))


def resolve_snapshot(arg, *, required):
    """The snapshot to run from. Its defects are FATAL when submitting and a warning in a
    dry run -- printing the plan must work before `tools/` has been committed and
    re-exported, which is exactly when somebody is reading the plan."""
    snap = arg or os.environ.get("SV3_DEPLOY_SNAPSHOT") or newest_snapshot()
    if snap:
        snap = os.path.abspath(snap)
        problems = []
        if not os.path.exists(os.path.join(snap, ".deploy_complete")):
            problems.append(f"{snap} is not a complete snapshot (no .deploy_complete)")
        if not os.path.exists(os.path.join(snap, "tools", "worker_pool.sh")):
            problems.append(f"{snap} has no tools/worker_pool.sh -- commit tools/ and "
                            f"re-run tools/deploy_snapshot.sh")
        if problems and required:
            raise SystemExit("[launch] ERROR: " + "; ".join(problems))
        for problem in problems:
            print(f"[launch] WARNING: {problem}")
        return snap
    if required:
        raise SystemExit("[launch] ERROR: no snapshot given and none found under "
                         f"{DEFAULT_DEPLOY_BASE}. Run tools/deploy_snapshot.sh first, "
                         "or pass --snapshot")
    print(f"[launch] note: no snapshot found; printing with {SNAPSHOT_PLACEHOLDER}")
    return SNAPSHOT_PLACEHOLDER


def snapshot_facts(snapshot):
    """`{sv3, sven, dirty, exported_at}` from the snapshot's DEPLOY_INFO.json.

    The full SHAs, not the directory name: `<sv3sha8>_<svensha8>` is not enough to tell
    two snapshots apart in a report, and the snapshot identity is what decides whether
    two lanes agree about what a run IS (`grid.run_hash` hashes the resolved config).
    """
    facts = {"sv3": None, "sven": None, "dirty": None, "exported_at": None,
             "path": snapshot}
    try:
        with open(os.path.join(snapshot, "DEPLOY_INFO.json")) as fh:
            info = json.load(fh)
    except (OSError, ValueError):
        return facts
    facts["sv3"] = info.get("git_sha") or info.get("sha")
    facts["sven"] = info.get("sven_git_sha")
    facts["dirty"] = info.get("git_dirty_tracked", info.get("git_dirty"))
    facts["exported_at"] = info.get("exported_at")
    return facts


def live_heads(repo=REPO):
    """`(sv3 HEAD, sven HEAD)` of the LIVE tree, or (None, None) -- for the dry run to
    show that the snapshot is (or is not) the code being edited right now."""
    out = []
    for path in (repo, os.path.join(repo, "sven")):
        try:
            proc = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=60)
            out.append(proc.stdout.strip() or None if proc.returncode == 0 else None)
        except (OSError, subprocess.SubprocessError):
            out.append(None)
    return tuple(out)


def snapshot_config_dir(snapshot):
    """The snapshot's hydra config dir, or None when there is no usable snapshot.

    Composing the LIVE configs to describe jobs that will run the SNAPSHOT's configs is
    how a dry run (and reconcile's expected grid) ends up describing a scan nobody runs.
    """
    if not snapshot or snapshot == SNAPSHOT_PLACEHOLDER:
        return None
    path = os.path.join(snapshot, "experiments", "configs")
    return path if os.path.isdir(path) else None


# ---------------------------------------------------------------------------
# Plan checking against the live configs (dry run only, unless --check)
# ---------------------------------------------------------------------------

_OPTIM_RE = re.compile(r"optimizers_standard=\[([^\]]*)\]")
_POOL_SCHED_RE = re.compile(r"^SCHED=\$\{WORKER_SCHEDULER_OVERRIDE-(\S+)\}\s*$", re.M)
#: what `tools/worker_pool.sh` adds to every runner command line when it cannot be read
DEFAULT_POOL_OVERRIDE = "++scheduler=claims"


def pool_override(snapshot):
    """The override `worker_pool.sh` prepends to every runner command, read from the
    snapshot that will run.

    Composing only the plan's own overrides is how `scheduler=claims` -- which no scan
    config declares, so hydra's struct mode refuses it -- passed a clean dry run and
    then failed every runner process of every job. The check has to compose the command
    line a job will really use, so it is read from the pool script itself.
    """
    path = os.path.join(str(snapshot or ""), "tools", "worker_pool.sh")
    try:
        with open(path) as fh:
            match = _POOL_SCHED_RE.search(fh.read())
    except OSError:
        return DEFAULT_POOL_OVERRIDE
    return match.group(1) if match else DEFAULT_POOL_OVERRIDE


def check_items(items, *, config_dir=None, extra_overrides=""):
    """`{item_index: (n_runs, [problems])}` by composing each item's config.

    Cheap (hydra only, no torch) and worth it: an item whose overrides expand to zero
    runs, or name an optimizer the config does not list, would otherwise burn a whole
    job before anybody noticed.

    `extra_overrides` is what the worker pool adds to every command (see
    :func:`pool_override`), so what gets composed here is the command line a job runs.
    """
    import reconcile                                  # torch-free, same directory
    out = {}
    grid = reconcile.load_grid()
    jd = reconcile.has_torchjd()
    prefix = f"{extra_overrides} " if extra_overrides else ""
    with reconcile.ConfigLoader(config_dir) as loader:
        for idx, item in enumerate(items):
            problems = []
            try:
                rcfg = loader.compose(item.config, prefix + item.overrides)
            except Exception as exc:
                out[idx] = (None, [f"cannot compose: {type(exc).__name__}: {exc} "
                                   f"(composed with the worker pool's "
                                   f"{extra_overrides!r})"])
                continue
            wanted = _OPTIM_RE.search(item.overrides)
            if wanted:
                # the config's OWN list, i.e. composed without this item's
                # `optimizers_standard=` override (which would otherwise be what we
                # compare against, and the check would be vacuous)
                base = prefix + " ".join(t for t in item.overrides.split()
                                         if not t.startswith("optimizers_standard="))
                try:
                    listed = loader.compose(item.config, base).get("optimizers_standard") or []
                except Exception:
                    listed = rcfg.get("optimizers_standard") or []
                missing = [o.strip() for o in wanted.group(1).split(",")
                           if o.strip() and o.strip() not in listed]
                if missing:
                    problems.append(f"not in the config's optimizers_standard: "
                                    f"{','.join(missing)}")
            try:
                specs = grid.expand_grid(rcfg, verbose=False, has_torchjd=jd)
            except Exception as exc:
                out[idx] = (None, problems + [f"expand_grid: {type(exc).__name__}: {exc}"])
                continue
            if not specs:
                problems.append("expands to ZERO runs")
            out[idx] = (len(specs), problems)
    return out


# ---------------------------------------------------------------------------
# squeue
# ---------------------------------------------------------------------------

def _squeue(args, *, what):
    user = os.environ.get("USER") or ""
    cmd = ["squeue", "-h", "-u", user] + list(args)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"[launch] ERROR: cannot run squeue ({exc}); refusing to "
                         f"{what} blind")
    if proc.returncode != 0:
        raise SystemExit(f"[launch] ERROR: squeue failed: {proc.stderr.strip()}")
    return [l for l in proc.stdout.splitlines() if l.strip()]


def count_queued(partition, user=None):
    """Jobs of `user` queued or running in `partition` (the MaxSubmitPU quantity)."""
    lines = _squeue(["-p", partition, "-o", "%i %T"],
                    what="submit into a capped partition")
    return len(lines), lines


def live_jobs_of_list(list_name):
    """`[(jobid, state, name)]` of our queued-or-running jobs that belong to this work
    list. Job names are `{phase}.{lane}.j{i}.{list}` (only the tail is ever truncated,
    and no plan name comes near the 120-character limit), so a substring match is exact
    enough to notice "this list is already running"."""
    out = []
    for line in _squeue(["-o", "%i %T %j"], what="submit a list that may already run"):
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        if list_name in parts[2]:
            out.append((parts[0], parts[1], parts[2]))
    return out


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------

def job_name(work_list, index, sha8=None):
    """Job names encode phase + lane + index (CONTRACTS.md), then the snapshot's sv3
    sha8 -- so a list running from two snapshots is visible in plain `squeue`."""
    name = f"{work_list.phase}.{work_list.lane}.j{index}.{work_list.name}"
    if sha8:
        name += f".{sha8}"
    return name[:120]


def items_path(work_dir, plan_name, list_name):
    return os.path.join(work_dir, plan_name, f"{list_name}.items.txt")


def record_path(work_dir, plan_name, list_name):
    """Where the snapshot this list was last launched from is recorded."""
    return os.path.join(work_dir, plan_name, f"{list_name}.launched.json")


def write_items(path, work_list, items, *, snapshot=None, facts=None):
    """The work-items file, stamped with the snapshot it was written for.

    The stamp is a comment (worker_pool.sh strips `#`), and it is the only on-disk
    record of which code a running job's items belong to.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    facts = facts or {}
    lines = [f"# {work_list.name}  phase={work_list.phase} lane={work_list.lane}",
             f"# {work_list.note.strip()}" if work_list.note else "#"]
    if snapshot:
        lines += [f"# snapshot {snapshot}",
                  f"# sv3  {facts.get('sv3') or '?'}",
                  f"# sven {facts.get('sven') or '?'}",
                  f"# written {time.strftime('%Y-%m-%dT%H:%M:%S%z')}"]
    lines += ["# config_name | hydra overrides | NPROC"]
    lines += [item.line() for item in items]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def read_record(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_record(path, *, work_list, snapshot, facts, items_file, jobs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"list": work_list.name, "phase": work_list.phase, "lane": work_list.lane,
               "snapshot": snapshot, "sv3": facts.get("sv3"), "sven": facts.get("sven"),
               "items_file": items_file, "jobs": jobs,
               "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    return payload


def snapshot_conflict(record, snapshot, list_name):
    """The refusal message when this list is live from ANOTHER snapshot, else None.

    Two jobs of one list running from two snapshots is the C-R3 ping-pong: whichever
    lane finishes a run writes `done/{run_id}.{hash8}.ok`, the other sees a marker under
    a foreign hash, moves the finished jsonl/npz/ckpt to `_stale/{hash8}/` and re-runs
    it -- for the life of both jobs.
    """
    if not record or not record.get("snapshot"):
        return None
    if os.path.realpath(record["snapshot"]) == os.path.realpath(snapshot):
        return None
    live = live_jobs_of_list(list_name)
    if not live:
        return None
    msg = [f"work list '{list_name}' is already running from ANOTHER snapshot:",
           f"  running: {record['snapshot']}  (sv3 {str(record.get('sv3'))[:8]}, "
           f"sven {str(record.get('sven'))[:8]}, submitted {record.get('submitted_at')})",
           f"  now:     {snapshot}"]
    for jobid, state, name in live:
        msg.append(f"  job {jobid} {state} {name}")
    msg.append("  the two snapshots give the same run_id different run hashes, so each "
               "lane would retire the other's finished results to _stale/ and re-run "
               "them. Wait for those jobs, or re-launch both lanes from one snapshot; "
               "--force submits anyway.")
    return "\n[launch] ERROR: " + "\n[launch] ".join(msg)


def sbatch_args(lane, work_list, items, *, name, log):
    cpus = work_list.cpus or lane.cpus_for(items)
    mem = work_list.mem or lane.mem
    time = work_list.time or lane.time
    args = ["sbatch", "-p", lane.partition, f"--gres={lane.gres}",
            f"--nodes={lane.nodes}", "--ntasks-per-node=1", "-c", str(cpus),
            f"--mem={mem}", "-t", time, "-J", name, "-o", log]
    args += list(lane.extra_sbatch)
    return args


def wrap_command(snapshot, items_file, results_root, name):
    inner = (f"SV3_RESULTS_ROOT={shlex.quote(results_root)} "
             f"bash {shlex.quote(os.path.join(snapshot, 'tools', 'worker_pool.sh'))} "
             f"{shlex.quote(snapshot)} {shlex.quote(items_file)} "
             f"--label {shlex.quote(name)}")
    return inner


def chain_script(path, *, lane, work_list, items, snapshot, items_file, results_root,
                 plan_file, log_dir, name, python):
    """Write a self-renewing sbatch script for the mig lane (see the module docstring
    for the MaxSubmit=2 caveat this implements)."""
    cpus = work_list.cpus or lane.cpus_for(items)
    mem = work_list.mem or lane.mem
    time = work_list.time or lane.time
    log = os.path.join(log_dir, f"{name}-%j.out")
    body = f"""#!/bin/bash
#SBATCH -p {lane.partition}
#SBATCH --gres={lane.gres}
#SBATCH --nodes={lane.nodes}
#SBATCH --ntasks-per-node=1
#SBATCH -c {cpus}
#SBATCH --mem={mem}
#SBATCH -t {time}
#SBATCH -J {name}
#SBATCH -o {log}
#SBATCH --signal=B:USR1@{CHAIN_SIGNAL_S}
# Self-renewing worker job for work list '{work_list.name}' ({work_list.phase}, lane
# {lane.name}). Written by tools/launch_campaign.py --chain; re-submit by hand with
#     sbatch {path}
#
# The renewal is the LAST act of the job: run the pool, ask reconcile whether work
# remains (exit 1 = yes), and only then resubmit. gpu_test allows 2 queued-or-running
# jobs per user and THIS job still holds one of them at that moment, so the sbatch
# needs a free slot; it retries {CHAIN_SBATCH_RETRIES} x {CHAIN_SBATCH_SLEEP_S}s and
# then gives up with a loud message. Relaunching by hand always resumes cleanly -- the
# claim queue, not the job, holds the position.
#
# GETTING CONTROL BACK BEFORE THE WALL CLOCK is what makes the renewal possible at all.
# A batch shell whose FOREGROUND child is SIGTERMed never runs the next line (bash
# defers the signal until the child ends, then exits 128+15), so the pool runs in the
# BACKGROUND and two independent mechanisms end it early:
#   * the pool itself starts no new item within 15 min of $SLURM_JOB_END_TIME (--until),
#   * `--signal=B:USR1@{CHAIN_SIGNAL_S}` reaches THIS shell {CHAIN_SIGNAL_S} s before the wall and the trap
#     below stops the pool -- the backstop for an item that is still running.
set -u
SNAP={shlex.quote(snapshot)}
ITEMS={shlex.quote(items_file)}
PLAN={shlex.quote(plan_file)}
ROOT={shlex.quote(results_root)}
PY={shlex.quote(python)}
DEPTH=${{CHAIN_DEPTH:-1}}
MAXDEPTH=${{CHAIN_MAX_DEPTH:-{CHAIN_MAX_DEPTH}}}
echo "[chain] depth $DEPTH/$MAXDEPTH  job ${{SLURM_JOB_ID:-none}}  $(date -Is)"

SIGNALLED=0
POOL_PID=""
on_usr1() {{
  SIGNALLED=1
  echo "[chain] USR1 {CHAIN_SIGNAL_S}s before the wall clock -- stopping the pool to reconcile and renew"
  [ -n "$POOL_PID" ] && kill -TERM "$POOL_PID" 2>/dev/null
}}
trap on_usr1 USR1

SV3_RESULTS_ROOT="$ROOT" bash "$SNAP/tools/worker_pool.sh" "$SNAP" "$ITEMS" \\
    --label {shlex.quote(name)} &
POOL_PID=$!
while :; do
  wait "$POOL_PID"; rc=$?
  # a trapped signal makes `wait` return >128 while the pool is still alive
  if [ "$rc" -gt 128 ] && kill -0 "$POOL_PID" 2>/dev/null; then continue; fi
  break
done
echo "[chain] worker pool exited $rc (signalled=$SIGNALLED)"
[ "$SIGNALLED" = 1 ] && rc=0

"$PY" "$SNAP/tools/reconcile.py" --all "$PLAN" --list {shlex.quote(work_list.name)} \\
    --root "$ROOT" --no-best --quiet
remaining=$?
echo "[chain] reconcile exit $remaining (0 = complete, 1 = work remains, 2 = error)"

if [ "$remaining" -ne 1 ]; then
  echo "[chain] not renewing (nothing left to do, or reconcile could not answer)"
  exit $rc
fi
if [ "$DEPTH" -ge "$MAXDEPTH" ]; then
  echo "[chain] NOT renewing: chain depth limit $MAXDEPTH reached -- resubmit by hand"
  exit $rc
fi
for try in $(seq 1 {CHAIN_SBATCH_RETRIES}); do
  if out=$(sbatch --export=ALL,CHAIN_DEPTH=$((DEPTH+1)) {shlex.quote(path)} 2>&1); then
    echo "[chain] renewed: $out"
    exit $rc
  fi
  echo "[chain] sbatch attempt $try failed ($out); the 2-job gpu_test cap is probably full"
  sleep {CHAIN_SBATCH_SLEEP_S}
done
echo "[chain] GIVING UP on renewal after {CHAIN_SBATCH_RETRIES} attempts -- resubmit by hand:"
echo "[chain]   sbatch {path}"
exit $rc
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(body)
    os.chmod(path, 0o755)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Print (default) or submit the sbatch jobs of a campaign plan.")
    ap.add_argument("plan")
    ap.add_argument("--phase", default=None, help="only this phase (P0, or 'P0,P1')")
    ap.add_argument("--lane", default=None, help="only this lane (a100 | mig)")
    ap.add_argument("--list", dest="lists", action="append", default=None,
                    help="only these work lists (repeatable)")
    ap.add_argument("--snapshot", default=None,
                    help="the deploy snapshot to run from (default: newest complete one)")
    ap.add_argument("--results-root", default=None,
                    help="SV3_RESULTS_ROOT for the workers (default: the plan's)")
    ap.add_argument("--work-dir", default=DEFAULT_WORK_BASE,
                    help="where the work-items files and chain scripts go")
    ap.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    ap.add_argument("--python", default=os.path.join(REPO, ".venv", "bin", "python"))
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="override the plan's n_jobs for every selected list")
    ap.add_argument("--time", default=None, help="override the wall clock")
    ap.add_argument("--chain", action="store_true",
                    help="self-renewing jobs (one per list; see the MaxSubmit caveat)")
    ap.add_argument("--submit", action="store_true", help="actually submit")
    ap.add_argument("--force", action="store_true",
                    help="submit even though this list is live from another snapshot")
    ap.add_argument("--config-dir", default=None,
                    help="hydra config dir for the run counts "
                         "(default: the SNAPSHOT's, else the repo's)")
    ap.add_argument("--no-count", action="store_true",
                    help="skip composing the configs to count runs per item")
    a = ap.parse_args(argv)

    try:
        plan = campaign_plan.load_plan(a.plan)
    except campaign_plan.PlanError as exc:
        err(f"[launch] ERROR: {exc}")
        return 2

    selected = plan.select(phase=a.phase, lane=a.lane, names=a.lists)
    if not selected:
        print(f"[launch] nothing selected (phase={a.phase} lane={a.lane} "
              f"lists={a.lists}); the plan has: "
              f"{', '.join(f'{w.phase}/{w.lane}/{w.name}' for w in plan.work_lists)}")
        return 2

    snapshot = resolve_snapshot(a.snapshot, required=a.submit)
    results_root = a.results_root or plan.results_root or os.environ.get("SV3_RESULTS_ROOT")
    if not results_root:
        err("[launch] ERROR: no results root (plan `results_root`, --results-root "
            "or $SV3_RESULTS_ROOT)")
        return 2
    results_root = os.path.abspath(results_root)
    if a.submit and not os.path.isdir(results_root):
        err(f"[launch] ERROR: results root {results_root} does not exist")
        return 2

    facts = snapshot_facts(snapshot)
    config_dir = a.config_dir or snapshot_config_dir(snapshot)
    source = ("--config-dir" if a.config_dir else
              "the snapshot's" if config_dir else "the LIVE repo's")

    print(f"[launch] plan      {plan.path}")
    print(f"[launch] snapshot  {snapshot}")
    print(f"[launch] snap sha  sv3 {str(facts['sv3'])[:12]}  sven {str(facts['sven'])[:12]}"
          f"  exported {facts['exported_at']}")
    head_sv3, head_sven = live_heads()
    print(f"[launch] live HEAD sv3 {str(head_sv3)[:12]}  sven {str(head_sven)[:12]}")
    if facts["sv3"] and head_sv3 and (facts["sv3"] != head_sv3
                                      or facts["sven"] != head_sven):
        print("[launch] WARNING: the snapshot is NOT the live HEAD -- the jobs run the "
              "snapshot's code and configs. Re-run tools/deploy_snapshot.sh if that is "
              "not what you want.")
    print(f"[launch] configs   {config_dir or os.path.join(REPO, 'experiments', 'configs')}"
          f"   ({source}; run counts and warnings below describe THESE configs)")
    print(f"[launch] results   {results_root}")
    print(f"[launch] pool adds {pool_override(snapshot)}   "
          f"(composed together with every item's own overrides)")
    print(f"[launch] mode      {'SUBMIT' if a.submit else 'dry run (nothing is submitted)'}"
          f"{'  +chain' if a.chain else ''}")

    # --- per-lane caps -----------------------------------------------------
    planned = []                                  # (work_list, lane, items, n_jobs)
    for wl in selected:
        lane = plan.lanes[wl.lane]
        items = wl.enabled_items()
        if not items:
            print(f"\n[launch] SKIP {wl.name} ({wl.phase}/{wl.lane}): no enabled items"
                  + (f" -- {wl.note.strip().splitlines()[0]}" if wl.note else ""))
            for it in wl.disabled_items():
                print(f"[launch]      parked: {it.line()}"
                      + (f"   # {it.note}" if it.note else ""))
            continue
        n_jobs = a.n_jobs or wl.n_jobs
        if a.chain and wl.lane == "mig":
            if n_jobs != 1:
                print(f"[launch] note: --chain forces n_jobs=1 for {wl.name} "
                      f"(was {n_jobs}); a chain must leave a free slot to renew itself")
                n_jobs = 1
        planned.append((wl, lane, items, n_jobs))

    if not planned:
        print("\n[launch] nothing to do: every selected list is empty")
        return 0

    capped = {}
    for wl, lane, items, n_jobs in planned:
        if lane.max_jobs is None:
            continue
        entry = capped.setdefault(lane.name, {"lane": lane, "want": 0, "chains": 0})
        entry["want"] += n_jobs
        if a.chain and wl.lane == lane.name:
            entry["chains"] += 1
    for _name, entry in capped.items():
        lane, want, chains = entry["lane"], entry["want"], entry["chains"]
        existing, lines = count_queued(lane.partition)
        # A chain renews itself while it still HOLDS one of the cap's slots, so filling
        # the cap with chains means no chain can ever renew (the module docstring's
        # promise): keep one slot free whenever this selection contains a chain.
        budget = lane.max_jobs - (1 if chains else 0)
        if existing + want > budget:
            allows = (f"allows {lane.max_jobs} queued job(s), and a chain must leave one "
                      f"free to renew into" if chains else
                      f"allows {lane.max_jobs} queued job(s)")
            err(f"\n[launch] ERROR: lane '{lane.name}' ({lane.partition}) {allows}; "
                f"{existing} already queued and this selection wants {want}.")
            for l in lines:
                err(f"[launch]   queued: {l}")
            err("[launch] refusing to exceed the cap. Wait, cancel by hand, or select "
                "fewer lists.")
            return 2
        print(f"[launch] lane cap {lane.name}: {existing} queued + {want} wanted "
              f"<= {budget} OK"
              + (f" ({lane.max_jobs} - 1 slot kept free so the chain can renew)"
                 if chains else ""))

    checks = {}
    if not a.no_count:
        try:
            all_items = [it for _wl, _l, its, _n in planned for it in its]
            flat = check_items(all_items, config_dir=config_dir,
                               extra_overrides=pool_override(snapshot))
            off = 0
            for wl, _l, its, _n in planned:
                for i in range(len(its)):
                    checks[(wl.name, i)] = flat[off + i]
                off += len(its)
        except Exception as exc:
            print(f"[launch] note: could not count runs per item "
                  f"({type(exc).__name__}: {exc}); continuing")

    # --- print / submit ----------------------------------------------------
    if a.submit:
        os.makedirs(a.log_dir, exist_ok=True)
    total_jobs, problems, grand_total = 0, 0, 0
    for wl, lane, items, n_jobs in planned:
        path = items_path(a.work_dir, plan.name, wl.name)
        print(f"\n[launch] == {wl.phase} / {wl.lane} / {wl.name}: {len(items)} item(s), "
              f"{n_jobs} job(s) ==")
        if wl.note:
            print(f"[launch]    {' '.join(wl.note.split())}")
        subtotal = 0
        for i, item in enumerate(items):
            n_runs, probs = checks.get((wl.name, i), (None, []))
            subtotal += n_runs or 0
            runs = "     ?" if n_runs is None else f"{n_runs:6d}"
            print(f"[launch]   {runs} runs  NPROC={item.nproc:<3d} {item.config:38s} "
                  f"{item.overrides}")
            for p in probs:
                problems += 1
                print(f"[launch]          [warn] {p}")
            if item.note:
                print(f"[launch]          note: {item.note}")
        grand_total += subtotal
        print(f"[launch]   {subtotal} runs in this list"
              f"{' (counts unavailable)' if not checks else ''}")

        # --- one list, one snapshot (see snapshot_conflict) -----------------
        rec_path = record_path(a.work_dir, plan.name, wl.name)
        previous = read_record(rec_path)
        if previous and os.path.realpath(previous.get("snapshot") or "") != \
                os.path.realpath(snapshot):
            print(f"[launch]   last launched from {previous.get('snapshot')} "
                  f"(sv3 {str(previous.get('sv3'))[:8]}, {previous.get('submitted_at')})")
            if a.submit:
                clash = snapshot_conflict(previous, snapshot, wl.name)
                if clash and not a.force:
                    err(clash)
                    return 2
                if clash:
                    print("[launch]   --force: submitting anyway (snapshot mismatch)")

        if a.submit:
            write_items(path, wl, items, snapshot=snapshot, facts=facts)
            print(f"[launch]   items file {path}")
        else:
            print(f"[launch]   items file {path}  (would be written)")

        submitted_ids = []
        for j in range(n_jobs):
            name = job_name(wl, j, (facts.get("sv3") or "")[:8] or None)
            log = os.path.join(a.log_dir, f"{name}-%j.out")
            if a.chain and wl.lane == "mig":
                script = os.path.join(a.work_dir, plan.name, f"{wl.name}.chain.sbatch")
                if a.submit:
                    chain_script(script, lane=lane, work_list=wl, items=items,
                                 snapshot=snapshot, items_file=path,
                                 results_root=results_root,
                                 plan_file=_plan_copy(a.work_dir, plan, submit=True),
                                 log_dir=a.log_dir, name=name, python=a.python)
                cmd = ["sbatch", script]
                print(f"[launch]   $ {shlex.join(cmd)}"
                      f"{'' if a.submit else f'   (chain script {script} would be written)'}")
            else:
                cmd = sbatch_args(lane, wl, items, name=name, log=log)
                cmd += ["--wrap", wrap_command(snapshot, path, results_root, name)]
                print(f"[launch]   $ {shlex.join(cmd)}")
            total_jobs += 1
            if a.submit:
                if a.time:
                    cmd = _override_time(cmd, a.time)
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    err(f"[launch]   SUBMIT FAILED: {proc.stderr.strip()}")
                    return 2
                print(f"[launch]   -> {proc.stdout.strip()}")
                submitted_ids.append({"name": name, "sbatch": proc.stdout.strip()})

        if a.submit and submitted_ids:
            write_record(rec_path, work_list=wl, snapshot=snapshot, facts=facts,
                         items_file=path, jobs=submitted_ids)
            print(f"[launch]   recorded {rec_path}")

    print(f"\n[launch] {total_jobs} job(s) {'submitted' if a.submit else 'printed'}, "
          f"{grand_total} run(s) of grid in scope, {problems} warning(s)")
    if not a.submit:
        print("[launch] re-run with --submit to submit (and --chain for the mig lane)")
    return 0


def _plan_copy(work_dir, plan, *, submit):
    """A copy of the plan next to the work-items files: the chain script reads the plan
    on a compute node, and the live tree is being edited while jobs run."""
    dest = os.path.join(work_dir, plan.name, os.path.basename(plan.path))
    if submit:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(plan.path) as src, open(dest, "w") as out:
            out.write(src.read())
    return dest


def _override_time(cmd, time):
    out = list(cmd)
    if "-t" in out:
        out[out.index("-t") + 1] = time
    return out


if __name__ == "__main__":
    sys.exit(main())
