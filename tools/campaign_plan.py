#!/usr/bin/env python3
"""Reading and validating a campaign plan (`campaign/plan_*.yaml`).

A plan is the campaign's *work order*: which scans run, split into work items by
family (so NPROC follows the GPU-probe table), grouped into **work lists**, each
assigned to a **lane** (a SLURM job shape) and a **phase** (P0..P3 launch order).

Shared by `tools/launch_campaign.py` (turns lists into sbatch commands) and
`tools/reconcile.py` (`--all`: the expected grid of every scan the plan touches).
Stdlib + PyYAML only -- no torch, no hydra, so it stays instant.

Schema (unknown keys are refused, because a silently ignored typo in a plan costs
GPU-days):

    version: 1
    results_root: <path>                 # optional; SV3_RESULTS_ROOT for the workers
    lanes:
      <name>:
        partition: "a,b,c"               # sbatch -p
        gres: "gpu:1"                    # sbatch --gres
        mem: 64G
        time: "24:00:00"                 # default wall clock for the lane
        cpus: 32                         # fixed cpus-per-task, OR
        cpus_extra: 2                    #   max(NPROC over the list's items) + this
        max_jobs: 2                      # refuse to exceed this many queued jobs
        note: "..."
    work_lists:
      - name: p0_mlp_mig
        phase: P0
        lane: mig
        n_jobs: 2                        # identical worker jobs on the same list
        time: "12:00:00"                 # optional per-list override
        chain: true                      # optional: this list is meant to self-resubmit
        note: "..."
        items:
          - {config: toy_1d_scan, overrides: "mode=svd", nproc: {a100: 12, mig: 6}}
          - {config: toy_1d_scan, overrides: "mode=hig", nproc: 4, enabled: false,
             note: "why it is parked"}

Any top-level key starting with `_` is ignored, which is how a plan keeps YAML
anchors (`_item_groups: &mlp ...`) without them becoming work lists.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

PHASE_RE = re.compile(r"^P\d+$")

_LANE_KEYS = {"partition", "gres", "mem", "time", "cpus", "cpus_extra", "max_jobs",
              "nodes", "extra_sbatch", "note"}
_LIST_KEYS = {"name", "phase", "lane", "n_jobs", "time", "items", "note", "chain",
              "cpus", "mem"}
_ITEM_KEYS = {"config", "overrides", "nproc", "note", "enabled", "scan"}
_PLAN_KEYS = {"version", "plan", "results_root", "lanes", "work_lists", "timing", "note"}


class PlanError(ValueError):
    """A plan file that cannot be trusted to launch: always fatal."""


@dataclass(frozen=True)
class WorkItem:
    config: str                 # hydra --config-name; also the results directory name
    overrides: str              # hydra overrides, as one shell-ready string
    nproc: int                  # runner processes per GPU for this item (probe table)
    note: str = ""
    enabled: bool = True

    @property
    def scan(self) -> str:
        """The results directory this item writes into (== the config name)."""
        return self.config

    def line(self) -> str:
        """The work-items-file line `config | overrides | NPROC` (worker_pool.sh)."""
        return f"{self.config} | {self.overrides} | {self.nproc}"


@dataclass(frozen=True)
class Lane:
    name: str
    partition: str
    gres: str
    mem: str = "64G"
    time: str = "24:00:00"
    cpus: int | None = None
    cpus_extra: int = 2
    max_jobs: int | None = None
    nodes: int = 1
    extra_sbatch: tuple = ()
    note: str = ""

    def cpus_for(self, items) -> int:
        """cpus-per-task: the lane's fixed value, else max NPROC over the items +
        `cpus_extra` (CONTRACTS.md; the probe measured cores, not the GPU, as the
        limit for the cheap MLPs)."""
        if self.cpus:
            return int(self.cpus)
        peak = max([i.nproc for i in items], default=1)
        return int(peak) + int(self.cpus_extra)


@dataclass(frozen=True)
class WorkList:
    name: str
    phase: str
    lane: str
    items: tuple
    n_jobs: int = 1
    time: str | None = None
    note: str = ""
    chain: bool = False
    cpus: int | None = None
    mem: str | None = None

    def enabled_items(self) -> list:
        return [i for i in self.items if i.enabled]

    def disabled_items(self) -> list:
        return [i for i in self.items if not i.enabled]


@dataclass
class Plan:
    path: str
    version: int
    name: str
    results_root: str | None
    lanes: dict
    work_lists: tuple
    timing: dict = field(default_factory=dict)
    note: str = ""

    def select(self, phase=None, lane=None, names=None) -> list:
        """The work lists matching a phase (`P0`, or `P0,P1`), a lane and/or names."""
        phases = {p.strip() for p in phase.split(",")} if phase else None
        lanes = {l.strip() for l in lane.split(",")} if lane else None
        want = {n.strip() for n in names} if names else None
        out = []
        for wl in self.work_lists:
            if phases and wl.phase not in phases:
                continue
            if lanes and wl.lane not in lanes:
                continue
            if want and wl.name not in want:
                continue
            out.append(wl)
        return out

    def scan_overrides(self, phase=None, lane=None, names=None) -> dict:
        """`{scan: [overrides, ...]}` over the selected lists' enabled items, in plan
        order and de-duplicated -- the expected grid of a scan is the union of these
        (`tools/reconcile.py --all`)."""
        out: dict[str, list] = {}
        for wl in self.select(phase=phase, lane=lane, names=names):
            for item in wl.enabled_items():
                seen = out.setdefault(item.scan, [])
                if item.overrides not in seen:
                    seen.append(item.overrides)
        return out


def _expand_root(text):
    """``$SV3_SCRATCH`` / ``~`` in a plan's ``results_root`` are expanded here, so the plan
    file itself never names a machine. When ``$SV3_SCRATCH`` is unset the launcher's own
    default (``~/scratch/sven``) is used."""
    scratch = os.environ.get("SV3_SCRATCH", os.path.expanduser("~/scratch/sven"))
    text = text.replace("${SV3_SCRATCH}", scratch).replace("$SV3_SCRATCH", scratch)
    return os.path.expandvars(os.path.expanduser(text))


def _check_keys(where, mapping, allowed):
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise PlanError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _parse_item(where, raw, lane_name):
    if not isinstance(raw, dict):
        raise PlanError(f"{where}: a work item must be a mapping, got {type(raw).__name__}")
    _check_keys(where, raw, _ITEM_KEYS)
    config = raw.get("config") or raw.get("scan")
    if not config:
        raise PlanError(f"{where}: needs a `config` (the hydra config name)")
    nproc = raw.get("nproc")
    if isinstance(nproc, dict):
        if lane_name not in nproc:
            raise PlanError(f"{where}: nproc has no entry for lane '{lane_name}' "
                            f"(has {sorted(nproc)})")
        nproc = nproc[lane_name]
    if nproc is None:
        raise PlanError(f"{where}: needs `nproc` (int, or a mapping lane -> int)")
    try:
        nproc = int(nproc)
    except (TypeError, ValueError):
        raise PlanError(f"{where}: nproc must be an integer, got {nproc!r}")
    if nproc < 1:
        raise PlanError(f"{where}: nproc must be >= 1, got {nproc}")
    overrides = raw.get("overrides") or ""
    if not isinstance(overrides, str):
        raise PlanError(f"{where}: overrides must be a single string, got {overrides!r}")
    if "|" in overrides or "\n" in overrides:
        raise PlanError(f"{where}: overrides must not contain '|' or a newline "
                        f"(the work-items file is '|'-separated): {overrides!r}")
    return WorkItem(config=str(config), overrides=overrides.strip(), nproc=nproc,
                    note=str(raw.get("note") or ""),
                    enabled=bool(raw.get("enabled", True)))


def load_plan(path) -> Plan:
    """Parse and validate a plan file. Raises :class:`PlanError` on anything doubtful."""
    import yaml                        # PyYAML ships with omegaconf; keeps the import lazy

    path = os.path.abspath(str(path))
    try:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
    except OSError as exc:
        raise PlanError(f"cannot read plan {path}: {exc}")
    except yaml.YAMLError as exc:
        raise PlanError(f"plan {path} is not valid YAML: {exc}")
    if not isinstance(raw, dict):
        raise PlanError(f"plan {path}: top level must be a mapping")

    raw = {k: v for k, v in raw.items() if not str(k).startswith("_")}
    _check_keys(f"plan {os.path.basename(path)}", raw, _PLAN_KEYS)

    lanes_raw = raw.get("lanes") or {}
    if not isinstance(lanes_raw, dict) or not lanes_raw:
        raise PlanError(f"plan {path}: `lanes` must be a non-empty mapping")
    lanes = {}
    for name, spec in lanes_raw.items():
        where = f"lane '{name}'"
        if not isinstance(spec, dict):
            raise PlanError(f"{where}: must be a mapping")
        _check_keys(where, spec, _LANE_KEYS)
        for req in ("partition", "gres"):
            if not spec.get(req):
                raise PlanError(f"{where}: needs `{req}`")
        extra = spec.get("extra_sbatch") or ()
        if isinstance(extra, str):
            extra = (extra,)
        lanes[str(name)] = Lane(
            name=str(name), partition=str(spec["partition"]), gres=str(spec["gres"]),
            mem=str(spec.get("mem", "64G")), time=str(spec.get("time", "24:00:00")),
            cpus=(int(spec["cpus"]) if spec.get("cpus") else None),
            cpus_extra=int(spec.get("cpus_extra", 2)),
            max_jobs=(int(spec["max_jobs"]) if spec.get("max_jobs") else None),
            nodes=int(spec.get("nodes", 1)), extra_sbatch=tuple(str(e) for e in extra),
            note=str(spec.get("note") or ""))

    lists_raw = raw.get("work_lists") or []
    if not isinstance(lists_raw, list) or not lists_raw:
        raise PlanError(f"plan {path}: `work_lists` must be a non-empty list")
    work_lists, seen = [], set()
    for idx, spec in enumerate(lists_raw):
        where = f"work_lists[{idx}]"
        if not isinstance(spec, dict):
            raise PlanError(f"{where}: must be a mapping")
        _check_keys(where, spec, _LIST_KEYS)
        name = str(spec.get("name") or "")
        if not name:
            raise PlanError(f"{where}: needs a `name`")
        if name in seen:
            raise PlanError(f"{where}: duplicate work-list name '{name}'")
        seen.add(name)
        where = f"work list '{name}'"
        phase = str(spec.get("phase") or "")
        if not PHASE_RE.match(phase):
            raise PlanError(f"{where}: `phase` must look like P0/P1/P2/P3, got {phase!r}")
        lane = str(spec.get("lane") or "")
        if lane not in lanes:
            raise PlanError(f"{where}: unknown lane {lane!r} (have {sorted(lanes)})")
        items_raw = spec.get("items")
        if items_raw is None:
            items_raw = []
        if not isinstance(items_raw, list):
            raise PlanError(f"{where}: `items` must be a list")
        # an entry may itself be a list (a YAML alias of an item group): flatten one level, so a
        # combined list can be written `items: [*items_p0_mlp, *items_p1_mlp]` and keep the order
        flat = []
        for it in items_raw:
            flat.extend(it if isinstance(it, list) else [it])
        items = tuple(_parse_item(f"{where} item[{i}]", it, lane)
                      for i, it in enumerate(flat))
        n_jobs = int(spec.get("n_jobs", 1))
        if n_jobs < 1:
            raise PlanError(f"{where}: n_jobs must be >= 1, got {n_jobs}")
        work_lists.append(WorkList(
            name=name, phase=phase, lane=lane, items=items, n_jobs=n_jobs,
            time=(str(spec["time"]) if spec.get("time") else None),
            note=str(spec.get("note") or ""), chain=bool(spec.get("chain", False)),
            cpus=(int(spec["cpus"]) if spec.get("cpus") else None),
            mem=(str(spec["mem"]) if spec.get("mem") else None)))

    timing = raw.get("timing") or {}
    if not isinstance(timing, dict):
        raise PlanError(f"plan {path}: `timing` must be a mapping (it is a stub section)")

    return Plan(path=path, version=int(raw.get("version", 1)),
                name=str(raw.get("plan") or os.path.splitext(os.path.basename(path))[0]),
                results_root=(_expand_root(str(raw["results_root"]))
              if raw.get("results_root") else None),
                lanes=lanes, work_lists=tuple(work_lists), timing=timing,
                note=str(raw.get("note") or ""))
