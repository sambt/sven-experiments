#!/usr/bin/env python3
"""CPU-only tests for the execution-layout probe.

    .venv/bin/python -m pytest bench/probe_campaign/test_probe_campaign.py -x -q

The probe has no acceptance test in `CHANGES_NEEDED.md` (it measures, it does not change
behaviour), so what is worth testing is the part that would silently waste the GPU
allocation: whether each planned set of hydra overrides really selects EXACTLY ONE
configuration of `experiments/optimizer_profile.py`, and with the run_id the driver
expects.  That is checked against the real configs (hydra compose) and the real
`expand_jobs` / `run_id_of` (extracted from the source with `ast`), so neither torch nor
a GPU is needed.  These tests live next to the probe rather than in `tests/` because the
probe track owns only `bench/probe_campaign/`.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SV3 = os.path.dirname(os.path.dirname(HERE))
CONFIG_DIR = os.path.join(SV3, "experiments", "configs")
PROFILER = os.path.join(SV3, "experiments", "optimizer_profile.py")
sys.path.insert(0, HERE)

import analyse                      # noqa: E402
import probe_plan as P              # noqa: E402
import probe_run                    # noqa: E402


# ---------------------------------------------------------------------------
def _real_expand_jobs():
    """`expand_jobs` + `run_id_of` lifted out of the profiler without importing torch."""
    tree = ast.parse(open(PROFILER).read())
    want = {"expand_jobs", "run_id_of"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
    assert {n.name for n in body} == want, "profiler no longer defines expand_jobs / run_id_of"
    ns: dict = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), PROFILER, "exec"), ns)
    return ns["expand_jobs"], ns["run_id_of"]


def _compose(config_name: str, overrides: list[str]) -> dict:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
        OmegaConf.set_struct(cfg, False)
        del cfg["hydra"]                       # hydra's own ${hydra:...} refs do not resolve here
        return OmegaConf.to_container(cfg, resolve=True)
    GlobalHydra.instance().clear()


ALL_MEASUREMENTS = (P.plan_b() + P.plan_c({}) + P.plan_d() + P.plan_e() + P.plan_cifarslice())


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("workload", sorted(P.WORKLOADS))
def test_workload_overrides_select_exactly_one_configuration(workload):
    """The whole point of the guard in probe_run.py: one process, one configuration."""
    m = next(x for x in ALL_MEASUREMENTS if x.workload == workload)
    expand, run_id_of = _real_expand_jobs()
    overrides = [o for o in P.hydra_overrides(m, "/tmp/out", "/tmp/hydra")
                 if not o.startswith("hydra.")]
    rcfg = _compose(P.config_name(m), overrides)
    jobs = [j for j in expand(rcfg) if j["study"] in rcfg["only_studies"]]
    assert len(jobs) == 1, [run_id_of(j) for j in jobs]
    assert run_id_of(jobs[0]) == P.expected_run_id(m)


def test_cifar_workloads_keep_batch_statistics_batchnorm():
    """Part (b) must profile the headline BN mode, not frozen running stats."""
    rcfg = _compose("profile_cifar", [])
    assert rcfg["gram_freeze_norm_stats"] is False
    assert rcfg["base"]["batch_size"] == 128


def test_mlp_and_nanogpt_workloads_use_the_production_backend():
    for cfg_name in ("profile_toy_1d", "profile_mnist", "profile_nanogpt"):
        rcfg = _compose(cfg_name, [])
        assert rcfg["use_gram"] is True
        assert rcfg.get("gram_capture", "hooks") == "hooks"


def test_profile_step_settings_give_at_least_60_measured_steps_in_part_b():
    for m in P.plan_b():
        s = P.STEP_SETTINGS[m.steps]
        assert s["min_steps"] >= 60 and s["num_steps"] >= s["min_steps"]
        assert s["warmup_steps"] >= 10


def test_part_b_covers_the_full_three_by_two_by_two_grid():
    b = P.plan_b()
    axes = {(m.capture, m.no_empty_cache, bool(m.alloc_conf)) for m in b}
    assert len(b) == 12 and len(axes) == 12
    assert {c for c, _, _ in axes} == {"full", "cf0.5", "cf0.25"}
    assert all(m.nproc == 1 for m in b)


def test_part_d_and_e_nproc_ladders():
    d = P.plan_d()
    assert {m.workload for m in d} == {"toy1d_sven", "mnist_sven", "mnist_adam", "mnist_lbfgs3"}
    for wl in {m.workload for m in d}:
        assert sorted(m.nproc for m in d if m.workload == wl) == [1, 4, 6, 8, 12]
    assert sorted(m.nproc for m in P.plan_e()) == [1, 2, 3]
    assert all(m.workload == "nanogpt_sven" for m in P.plan_e())


def test_part_c_uses_the_best_mode_from_part_b_and_includes_a_single_process_reference():
    best = {"capture": "cf0.5", "alloc_conf": P.EXPANDABLE, "no_empty_cache": True}
    c = P.plan_c(best)
    sven = [m for m in c if m.workload.startswith("cifar_sven")]
    assert sorted(m.nproc for m in sven) == [1, 2, 3]
    assert all(m.capture == "cf0.5" and m.alloc_conf == P.EXPANDABLE and m.no_empty_cache
               for m in sven)
    adam = [m for m in c if m.workload == "cifar_adam"]
    assert sorted(m.nproc for m in adam) == [1, 4]


def test_tags_are_unique_and_filesystem_safe():
    tags = [m.tag for m in ALL_MEASUREMENTS]
    assert len(tags) == len(set(tags))
    assert all(set(t) <= set("abcdefghijklmnopqrstuvwxyz0123456789_-.") for t in tags)


def test_nothing_is_written_to_a_protected_results_directory():
    for m in ALL_MEASUREMENTS:
        args = " ".join(P.hydra_args(m, "/n/labstore01/probe_results/x", "/n/labstore01/h"))
        assert "profile.output_dir=/n/labstore01/probe_results/x" in args
        assert "experiment_results" not in args and "profile_results_v2" not in args


def test_choose_best_b_picks_the_fastest_and_falls_back_when_empty():
    recs = [
        {"part": "b", "status": "ok", "tag": "slow", "capture": "full",
         "wall_ms": {"median": 900.0}, "memory": {"peak_reserved_bytes_max": 3e10}},
        {"part": "b", "status": "ok", "tag": "fast", "capture": "cf0.25", "alloc_conf": P.EXPANDABLE,
         "no_empty_cache": True, "wall_ms": {"median": 400.0},
         "memory": {"peak_reserved_bytes_max": 7e9}},
        {"part": "b", "status": "oom", "tag": "dead", "capture": "cf0.5"},
        {"part": "d", "status": "ok", "tag": "other", "wall_ms": {"median": 1.0}},
    ]
    best = P.choose_best_b(recs)
    assert best["capture"] == "cf0.25" and best["no_empty_cache"] is True
    assert best["alloc_conf"] == P.EXPANDABLE and best["source"] == "fast"
    fb = P.choose_best_b([])
    assert fb == {"capture": "cf0.25", "alloc_conf": "", "no_empty_cache": False,
                  "source": "fallback"}


def test_choose_best_b_prefers_less_reserved_memory_when_speeds_tie():
    """The real A100 numbers: 186.5 ms / 32.8 GB vs 186.7 ms / 23.3 GB.  Only the second
    lets three processes share an 80 GB card, so a 0.1% speed win must not decide it."""
    recs = [
        {"part": "b", "status": "ok", "tag": "full-default", "capture": "full",
         "alloc_conf": "", "no_empty_cache": True, "wall_ms": {"median": 186.5},
         "memory": {"peak_reserved_bytes_max": 32_833_000_000}},
        {"part": "b", "status": "ok", "tag": "full-expandable", "capture": "full",
         "alloc_conf": P.EXPANDABLE, "no_empty_cache": True, "wall_ms": {"median": 186.7},
         "memory": {"peak_reserved_bytes_max": 23_325_000_000}},
        {"part": "b", "status": "ok", "tag": "cf0.25-expandable", "capture": "cf0.25",
         "alloc_conf": P.EXPANDABLE, "no_empty_cache": True, "wall_ms": {"median": 205.4},
         "memory": {"peak_reserved_bytes_max": 7_363_000_000}},
    ]
    best = P.choose_best_b(recs)
    assert best["source"] == "full-expandable"          # not full-default, not cf0.25
    assert best["capture"] == "full" and best["alloc_conf"] == P.EXPANDABLE
    assert best["n_tied_within_tol"] == 2               # cf0.25 is 10% slower, so excluded


# ---------------------------------------------------------------------------
def test_strip_editable_finders_removes_only_the_editable_install_shim():
    class _EditableFinder:            # what setuptools appends to sys.meta_path
        pass
    _EditableFinder.__module__ = "__editable___sven_0_1_0_finder"

    class Keeper:
        pass
    saved = list(sys.meta_path)
    try:
        sys.meta_path.append(_EditableFinder)
        sys.meta_path.append(Keeper)
        removed = probe_run.strip_editable_finders()
        assert any("_EditableFinder" in r for r in removed)
        assert Keeper in sys.meta_path and _EditableFinder not in sys.meta_path
        assert any(getattr(f, "__name__", "") == "PathFinder" for f in sys.meta_path)
    finally:
        sys.meta_path[:] = saved


def test_this_venv_really_installs_sven_through_a_meta_path_finder():
    """Why strip_editable_finders exists: the editable install of `sven` is a meta-path
    finder pointing at the live working tree, which the probe must never import."""
    names = [f"{getattr(f, '__module__', '')}.{getattr(f, '__name__', type(f).__name__)}"
             for f in sys.meta_path]
    assert any("__editable__" in n and "sven" in n for n in names), names


def test_barrier_releases_only_once_every_process_has_arrived(tmp_path):
    bdir = str(tmp_path / "b")
    ok, waited = probe_run.barrier_wait(bdir, 1, 1.0)
    assert ok and waited == 0.0
    ok, waited = probe_run.barrier_wait(bdir, 3, 0.5)      # only this process arrives
    assert not ok and waited >= 0.4
    open(os.path.join(bdir, "other1.ready"), "w").close()
    open(os.path.join(bdir, "other2.ready"), "w").close()
    ok, _ = probe_run.barrier_wait(bdir, 3, 1.0)
    assert ok


# ---------------------------------------------------------------------------
def _write(results, recs):
    d = os.path.join(results, "jsonl")
    os.makedirs(d, exist_ok=True)
    for i, r in enumerate(recs):
        with open(os.path.join(d, f"r{i}.jsonl"), "w") as f:
            f.write(json.dumps(r) + "\n")


def _rec(**kw):
    r = {"part": "d", "tag": "t", "workload": "mnist_sven", "nproc": 1, "proc_index": 0,
         "layout": "nproc1", "capture": "", "device_class": "a100_80gb", "status": "ok",
         "host": "h", "slurm_job_id": "1", "n_measured_steps": 400, "barrier_ok": True,
         "t_release": 0.0, "t_end": 30.0, "no_empty_cache": False, "alloc_conf": "",
         "step_ms": {"n": 400, "mean": 10.0, "median": 10.0, "p10": 9.0, "p90": 11.0,
                     "steady_mean": 10.0},
         "wall_ms": {"n": 400, "mean": 10.4, "median": 10.4, "p10": 9.4, "p90": 11.4,
                     "steady_mean": 10.4},
         "memory": {"peak_alloc_bytes_max": 21_000_000, "peak_reserved_bytes_max": 25_000_000},
         "provenance": {"sven": "/n/labstore01/x/sven/sven/__init__.py"}}
    r.update(kw)
    return r


def test_analyse_computes_inflation_and_aggregate_throughput(tmp_path):
    """4 processes each 2x slower (in wall time) than one => inflation 2, aggregate T = 2."""
    results = str(tmp_path)
    recs = [_rec(tag="d__mnist_sven__n1", nproc=1)]
    for i in range(4):
        recs.append(_rec(tag="d__mnist_sven__n4", nproc=4, proc_index=i,
                         step_ms={"n": 400, "mean": 20.0, "median": 20.0, "p10": 19.0,
                                  "p90": 21.0, "steady_mean": 20.0},
                         wall_ms={"n": 400, "mean": 20.8, "median": 20.8, "p10": 19.8,
                                  "p90": 21.8, "steady_mean": 20.8}))
    _write(results, recs)
    gs = {k: analyse.group_stats(v) for k, v in analyse.groups(analyse.load(results)).items()}
    txt, best = analyse.section_cotenancy(gs)
    assert "2.00" in txt                                     # inflation and aggregate T
    assert best[("a100_80gb", "mnist_sven", "")] == (4, pytest.approx(2.0))


def test_aggregate_throughput_counts_only_the_processes_that_ran(tmp_path):
    """Measured: 3 concurrent full-capture CIFAR Sven runs, one OOMs, the two survivors
    take 385 ms against 186.7 ms alone.  That is 2 processes at 2.06x inflation => 0.97,
    not 3 at 2.06x => 1.46."""
    results = str(tmp_path)
    t1 = {"n": 60, "mean": 186.7, "median": 186.7, "p10": 186, "p90": 188, "steady_mean": 186.7}
    t3 = {"n": 60, "mean": 385.0, "median": 385.0, "p10": 384, "p90": 386, "steady_mean": 385.0}
    recs = [_rec(part="c", tag="c__sven-full__n1", workload="cifar_sven_full", capture="full",
                 nproc=1, step_ms=t1, wall_ms=t1)]
    recs.append(_rec(part="c", tag="c__sven-full__n3", workload="cifar_sven_full", capture="full",
                     nproc=3, proc_index=0, status="oom", step_ms=None, wall_ms=None,
                     memory=None, error="CUDA out of memory"))
    for i in (1, 2):
        recs.append(_rec(part="c", tag="c__sven-full__n3", workload="cifar_sven_full",
                         capture="full", nproc=3, proc_index=i, step_ms=t3, wall_ms=t3))
    _write(results, recs)
    gs = {k: analyse.group_stats(v) for k, v in analyse.groups(analyse.load(results)).items()}
    _txt, best = analyse.section_cotenancy(gs)
    n, t = best[("a100_80gb", "cifar_sven_full", "full")]
    assert n == 1 and t == pytest.approx(1.0)          # co-tenancy is not worth it
    assert t < 1.1


def test_inflation_uses_wall_time_not_cuda_event_time(tmp_path):
    """Measured on the MIG slice: mnist_adam CUDA-event time FELL from 1.01 to 0.22 ms
    between NPROC 1 and 6 while wall time rose from 1.13 to 1.87 ms, because event
    timing cannot see a process waiting for its turn on a time-sliced GPU.  Using
    `step_ms` would report an aggregate throughput of 28x; wall time reports 3.6x."""
    results = str(tmp_path)
    recs = [_rec(tag="d__mnist_adam__n1", workload="mnist_adam", nproc=1,
                 step_ms={"n": 4000, "mean": 1.01, "median": 1.014, "p10": 1.0, "p90": 1.1,
                          "steady_mean": 1.01},
                 wall_ms={"n": 4000, "mean": 1.13, "median": 1.134, "p10": 1.1, "p90": 1.2,
                          "steady_mean": 1.13})]
    for i in range(6):
        recs.append(_rec(tag="d__mnist_adam__n6", workload="mnist_adam", nproc=6, proc_index=i,
                         step_ms={"n": 4000, "mean": 0.215, "median": 0.215, "p10": 0.21,
                                  "p90": 0.22, "steady_mean": 0.215},
                         wall_ms={"n": 4000, "mean": 1.867, "median": 1.867, "p10": 1.8,
                                  "p90": 1.95, "steady_mean": 1.867}))
    _write(results, recs)
    gs = {k: analyse.group_stats(v) for k, v in analyse.groups(analyse.load(results)).items()}
    txt, best = analyse.section_cotenancy(gs)
    n, t = best[("a100_80gb", "mnist_adam", "")]
    assert n == 6 and t == pytest.approx(6 * 1.134 / 1.867, rel=1e-3)   # 3.64, not 28
    assert t < 5.0
    assert "EVENT-TIME-UNUSABLE" in txt                      # the artifact is flagged


def test_analyse_part_b_table_reports_mean_median_p10_p90_and_both_memory_numbers(tmp_path):
    results = str(tmp_path)
    _write(results, [_rec(part="b", tag="b__cf0.25__ec-on__alloc-default",
                          workload="cifar_sven_cf0.25", capture="cf0.25",
                          step_ms={"n": 70, "mean": 470.0, "median": 465.0, "p10": 450.0,
                                   "p90": 495.0, "steady_mean": 468.0},
                          memory={"peak_alloc_bytes_max": 6_923_000_000,
                                  "peak_reserved_bytes_max": 18_474_000_000})])
    gs = {k: analyse.group_stats(v) for k, v in analyse.groups(analyse.load(results)).items()}
    txt = analyse.section_b(gs)
    for want in ("cf0.25", "465.0", "470.0", "450.0", "495.0", "6923", "18474"):
        assert want in txt, want


def test_analyse_runs_end_to_end_as_one_command(tmp_path):
    results = str(tmp_path)
    _write(results, [_rec(part="b", tag="b__full__ec-on__alloc-default",
                          workload="cifar_sven_full", capture="full"),
                     _rec(part="d", tag="d__mnist_sven__n1"),
                     _rec(part="d", tag="d__mnist_sven__n1", device_class="mig_3g20gb",
                          step_ms={"n": 400, "mean": 25.0, "median": 25.0, "p10": 24.0,
                                   "p90": 26.0, "steady_mean": 25.0},
                          wall_ms={"n": 400, "mean": 26.0, "median": 26.0, "p10": 25.0,
                                   "p90": 27.0, "steady_mean": 26.0})])
    out = subprocess.run([sys.executable, os.path.join(HERE, "analyse.py"), "--results", results],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "RECOMMENDATIONS" in out.stdout
    assert "MIG 3g.20gb vs full A100-80GB" in out.stdout
    assert "2.50" in out.stdout                              # MIG/A100 slowdown 25/10
    assert "snapshot provenance: OK" in out.stdout


def test_analyse_reports_no_data_instead_of_crashing(tmp_path):
    out = subprocess.run([sys.executable, os.path.join(HERE, "analyse.py"), "--results", str(tmp_path)],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 1 and "no measurements" in out.stdout
