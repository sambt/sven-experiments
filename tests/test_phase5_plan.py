"""`tools/gen_phase5_plan.py`: the plan it writes must parse, and the launcher must run it.

CPU only, no torch, no SLURM. Two things are asserted about the generated plan: that
`tools/campaign_plan.load_plan` accepts it (the schema refuses unknown keys, so a typo in
the generator is fatal rather than silently ignored), and that
`tools/launch_campaign.py:check_items` -- the launcher's own dry-run check, composed
together with the worker pool's `++scheduler=claims` -- reports no problems and a non-zero
run count for every item.

`check_items` is not free (one hydra composition per item), so the round-trip test uses a
small synthetic selection file; the REAL `campaign/plan_phase5.yaml`, if it has been
generated, is checked for parse and for the existence of every config it names.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")
CONFIG_DIR = os.path.join(REPO, "experiments", "configs")
REAL_PLAN = os.path.join(REPO, "campaign", "plan_phase5.yaml")


def _load(name):
    path = os.path.join(TOOLS, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_tools_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


campaign_plan = _load("campaign_plan")
gen_plan = _load("gen_phase5_plan")
launch = _load("launch_campaign")


def _selection(tmp_path):
    """A schema-2 selection file: two real scans, one MLP and one language model, with a
    complete (5 of 5 seeds, verified) selection so the generator's readiness guard passes.
    """
    def method(name, family, overrides, seeds):
        return {
            "method": name, "label": "Sven" if name == "SVD" else name, "family": family,
            "config_key": f"{name}_key", "overrides": overrides, "verified": True,
            "verify_error": None, "seed_mean_final_val": 0.5, "n_ok": 5, "n_expected": 5,
            "n_diverged": 0, "n_eligible_configs": 1, "tied_with": [], "edges": {},
            "batch_size": 32, "hash8": {}, "hparams": {}, "model_seeds": seeds,
            "loader_seed": seeds[0], "run_ids": [], "seed_values": {},
            "mean_wall_time_s": 60.0, "n_records": 5,
        }

    payload = {
        "schema": 2, "generated_at": "2026-09-19T00:00:00+0000",
        "results_root": str(tmp_path / "results"), "config_dir": CONFIG_DIR,
        "rule_name": "full", "rule": "test", "default_groups": ["mode=all"],
        "scans": {
            "toy_1d_scan": {
                "groups": ["mode=all"], "scan_dir": "-", "n_expected": 5,
                "n_records": 5, "n_missing": 0, "warnings": [], "errors": [],
                "reconcile_disagreements": [],
                "methods": {
                    "SVD": method("SVD", "svd",
                                  "mode=svd k_values=[1] lrs=[0.1] rtol=[0.001]",
                                  [1000, 1001, 1002, 1003, 1004]),
                    "LBFGS": method(
                        "LBFGS", "lbfgs",
                        "mode=standard optimizers_standard=[LBFGS] lrs_lbfgs=[0.1] "
                        "lbfgs_max_iter=[1] lbfgs_history_size=[2]",
                        [1000, 1001, 1002, 1003, 1004]),
                },
            },
            "exp_nanogpt_speedrun": {
                "groups": ["mode=all"], "scan_dir": "-", "n_expected": 5,
                "n_records": 5, "n_missing": 0, "warnings": [], "errors": [],
                "reconcile_disagreements": [],
                "methods": {
                    "AdamW": method(
                        "AdamW", "standard",
                        "mode=standard optimizers_standard=[AdamW] lrs_standard=[0.0003]",
                        [5000, 5001, 5002, 5003, 5004]),
                },
            },
        },
    }
    path = tmp_path / "best_configs.json"
    path.write_text(json.dumps(payload))
    (tmp_path / "results").mkdir(exist_ok=True)
    return str(path)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """`(plan_path, Plan)` for a plan generated from the synthetic selection."""
    tmp = tmp_path_factory.mktemp("phase5plan")
    src = _selection(tmp)
    out = str(tmp / "plan_phase5.yaml")
    rc = gen_plan.main(["--in", src, "--out", out, "--config-dir", CONFIG_DIR])
    assert rc == 0
    return out, campaign_plan.load_plan(out)


# ---------------------------------------------------------------------------
# The generated plan
# ---------------------------------------------------------------------------

def test_the_generated_plan_parses_and_covers_the_three_passes(generated):
    _path, plan = generated
    assert plan.name == "phase5"
    assert plan.results_root
    names = {wl.name for wl in plan.work_lists}
    assert {"p5_timing", "p5_diag_mlp_mig", "p5_diag_mlp_a100", "p5_diag_heavy_a100",
            "p5_confirm_mlp_mig", "p5_confirm_mlp_a100",
            "p5_confirm_heavy_a100"} <= names, sorted(names)
    assert {wl.phase for wl in plan.work_lists} == {"P5"}
    # the MIG lane is MLP-only: a 19.6 GB slice does not hold CIFAR full capture, and
    # nanoGPT is NPROC 1 there
    for wl in plan.work_lists:
        if wl.lane == "mig":
            assert all("nanogpt" not in i.config and not i.config.startswith("cifar10")
                       for i in wl.items), wl.name


def test_every_item_names_a_config_that_exists_and_a_pass_directory(generated):
    _path, plan = generated
    for wl in plan.work_lists:
        for item in wl.items:
            assert os.path.isfile(os.path.join(CONFIG_DIR, f"{item.config}.yaml")), \
                f"{wl.name}: {item.config}"
            assert item.config.endswith(("_timing", "_diag", "_confirm")), item.config
            assert item.nproc >= 1
            # the results directory is the config name, so the three passes cannot
            # write into the parent scan's directory
            assert item.scan == item.config


def test_the_mlp_and_a100_overflow_lists_are_the_same_work(generated):
    """The MIG list and its A100 overflow must be the identical items at different NPROC:
    the claim queue makes running both free of duplicated work, but only if they describe
    the same runs."""
    _path, plan = generated
    for mig, a100 in (("p5_diag_mlp_mig", "p5_diag_mlp_a100"),
                      ("p5_confirm_mlp_mig", "p5_confirm_mlp_a100")):
        m = {(i.config, i.overrides) for wl in plan.select(names=[mig]) for i in wl.items}
        a = {(i.config, i.overrides) for wl in plan.select(names=[a100]) for i in wl.items}
        assert m == a, (mig, a100, m ^ a)


def test_timing_items_are_serial_and_log_nothing(generated):
    _path, plan = generated
    lists = plan.select(names=["p5_timing"])
    assert lists and lists[0].lane == "timing"
    for item in lists[0].items:
        assert item.nproc == 1, item.line()
        for token in gen_plan.TIMING_OVERRIDES.split():
            assert token in item.overrides, item.overrides


def test_confirmation_gets_one_item_per_data_seed_for_the_generated_datasets(generated):
    """toy and polynomial get 3 data seeds; everything else gets none.

    `data_seed` is a scalar config key, not a grid axis, so a replicate is an ITEM and not
    a grid point -- and the `_confirm` config is what keeps the three from colliding
    (tests/test_phase5_configs.py: test_data_seed_replicates_cannot_collide).
    """
    _path, plan = generated
    toy = [i for wl in plan.work_lists for i in wl.items
           if i.config == "toy_1d_scan_confirm"]
    seeds = {tok.split("=")[1] for i in toy for tok in i.overrides.split()
             if tok.startswith("data_seed=")}
    assert seeds == {"1000", "1001", "1002"}, seeds
    gpt = [i for wl in plan.work_lists for i in wl.items
           if i.config == "exp_nanogpt_speedrun_confirm"]
    assert gpt and not any("data_seed=" in i.overrides for i in gpt)


def test_the_launcher_dry_runs_every_item_without_a_warning(generated):
    """`check_items` composes each item's config together with the worker pool's
    `++scheduler=claims` -- the command line a job really runs -- and expands the grid.
    Zero problems and a non-zero run count for every item is the launch gate."""
    _path, plan = generated
    items = [i for wl in plan.work_lists for i in wl.items]
    # de-duplicate: the MIG lists and their A100 overflow carry the same items
    seen, unique = set(), []
    for item in items:
        key = (item.config, item.overrides)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    checks = launch.check_items(unique, config_dir=CONFIG_DIR,
                                extra_overrides=launch.DEFAULT_POOL_OVERRIDE)
    for idx, item in enumerate(unique):
        n_runs, problems = checks[idx]
        assert not problems, f"{item.config} {item.overrides}: {problems}"
        assert n_runs == 5, f"{item.config} {item.overrides}: {n_runs} runs, wanted 5"


# ---------------------------------------------------------------------------
# The generator's guards
# ---------------------------------------------------------------------------

def test_the_generator_refuses_the_pre_campaign_selection_format(tmp_path):
    """`bench/best_configs.json` used to be `{scan: [[method, cfg, ...], ...]}` with no
    verified override strings. Consuming it as if it were the new one would generate items
    with empty overrides, i.e. the whole grid on every pass."""
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"toy_1d_scan": [["Sven", {"k": 16.0}, 1e-6, 5, 10.0]]}))
    assert gen_plan.main(["--in", str(legacy), "--out", str(tmp_path / "p.yaml"),
                          "--config-dir", CONFIG_DIR]) == 2
    assert not (tmp_path / "p.yaml").exists()


def test_the_generator_refuses_an_unfinished_selection(tmp_path):
    """A configuration with 3 of 5 seeds is already `eligible`, so a half-finished grid
    extension can win outright -- and every downstream pass would then measure it."""
    src = _selection(tmp_path)
    payload = json.loads(open(src).read())
    payload["scans"]["toy_1d_scan"]["n_missing"] = 40
    payload["scans"]["toy_1d_scan"]["methods"]["SVD"]["n_ok"] = 3
    (tmp_path / "thin.json").write_text(json.dumps(payload))
    out = tmp_path / "thin_plan.yaml"
    assert gen_plan.main(["--in", str(tmp_path / "thin.json"), "--out", str(out),
                          "--config-dir", CONFIG_DIR]) == 1
    assert not out.exists()
    # ... and says so in the header when it is forced
    assert gen_plan.main(["--in", str(tmp_path / "thin.json"), "--out", str(out),
                          "--config-dir", CONFIG_DIR, "--allow-incomplete"]) == 0
    text = out.read_text()
    assert "INCOMPLETE SELECTION" in text
    assert "3/5" in text
    campaign_plan.load_plan(str(out))            # still a valid plan


def test_the_generator_refuses_unverified_overrides(tmp_path):
    src = _selection(tmp_path)
    payload = json.loads(open(src).read())
    payload["scans"]["toy_1d_scan"]["methods"]["SVD"]["verified"] = False
    payload["scans"]["toy_1d_scan"]["methods"]["SVD"]["verify_error"] = "expands to 10"
    (tmp_path / "bad.json").write_text(json.dumps(payload))
    out = tmp_path / "bad_plan.yaml"
    assert gen_plan.main(["--in", str(tmp_path / "bad.json"), "--out", str(out),
                          "--config-dir", CONFIG_DIR]) == 1
    assert not out.exists()


def test_the_nproc_table_is_total_over_the_classes_it_is_asked_for():
    """Every (scan class, nproc class) pair the generator can produce has a probe value,
    for both lanes: a missing entry is a KeyError in the middle of generating a plan."""
    for klass in ("mlp", "cifar", "nanogpt"):
        for cls in ("sven", "first", "heavy"):
            assert set(gen_plan.NPROC[(klass, cls)]) == {"a100", "mig"}
    assert gen_plan.scan_class("cifar10_resnet_ce_scan") == "cifar"
    assert gen_plan.scan_class("exp_nanogpt_speedrun") == "nanogpt"
    assert gen_plan.scan_class("mnist_scan_ce") == "mlp"
    assert gen_plan.nproc_class({"family": "standard", "method": "SOAP"}) == "heavy"
    assert gen_plan.nproc_class({"family": "standard", "method": "Adam"}) == "first"
    assert gen_plan.nproc_class({"family": "polyak", "method": "PolyakSGD"}) == "first"
    assert gen_plan.nproc_class({"family": "svd", "method": "SVD"}) == "sven"
    assert gen_plan.nproc_class({"family": "hig", "method": "HIG"}) == "heavy"


# ---------------------------------------------------------------------------
# The plan that is actually checked in
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isfile(REAL_PLAN), reason="plan_phase5.yaml not generated")
def test_the_checked_in_plan_parses_and_names_configs_that_exist():
    plan = campaign_plan.load_plan(REAL_PLAN)
    assert plan.name == "phase5"
    assert plan.results_root and plan.results_root.startswith("/n/labstore"), \
        f"the workers must not write to the NFS home: {plan.results_root}"
    for wl in plan.work_lists:
        assert wl.items, wl.name
        for item in wl.items:
            assert os.path.isfile(os.path.join(CONFIG_DIR, f"{item.config}.yaml")), \
                f"{wl.name}: {item.config}"
