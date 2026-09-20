"""CPU tests for the campaign's scan configs (`experiments/configs/*.yaml`).

``tests/test_grid.py`` pins the *enumeration code* against nine frozen oracles.
This file pins the *configs* against the campaign decisions in
``campaign/CONTRACTS.md``: the scope update (what is cut, what is kept, no
weight-decay grid), the Stage-1 config keys and their valid values, the baseline
sets (C-B1 MuonW / C-B2 SGDm / C-B5 Muon-on-ResNet), and C-B3 "grids must contain
their optimum". A failure here means a config drifted from a decision, not that
the code is wrong.

The run counts themselves live in ``campaign/grid_counts.md``, which
:func:`test_grid_counts_md_matches_the_configs` parses and compares against
``expand_grid`` -- so the launcher cost table cannot silently go stale.

Torch-free and fast: ``grid.py`` is loaded from its path (the
``experiments.experiment_code`` package ``__init__`` may still pull torch) and
the configs are composed with OmegaConf, the same way ``tests/test_grid.py``
resolves its defaults lists.
"""

import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "experiments" / "configs"
GRID_COUNTS = REPO / "campaign" / "grid_counts.md"


def _load_grid():
    path = REPO / "experiments" / "experiment_code" / "grid.py"
    spec = importlib.util.spec_from_file_location("sv3_grid_for_configs", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod            # dataclasses needs the module registered
    spec.loader.exec_module(mod)
    return mod


grid = _load_grid()


# ---------------------------------------------------------------------------
# Config loading (identical resolution to tests/test_grid.py::load_rcfg)
# ---------------------------------------------------------------------------

def _compose(name):
    cfg = OmegaConf.load(CONFIGS / f"{name}.yaml")
    defaults = cfg.pop("defaults", [])
    merged = OmegaConf.create({})
    for entry in defaults:
        if entry == "_self_":
            continue
        if isinstance(entry, str):                       # a parent config (*_timing)
            merged = OmegaConf.merge(merged, _compose(entry))
            continue
        for group, option in OmegaConf.to_container(entry).items():
            merged = OmegaConf.merge(merged, OmegaConf.create(
                {group: OmegaConf.load(CONFIGS / group / f"{option}.yaml")}))
    return OmegaConf.merge(merged, cfg)                  # _self_ is last everywhere


def load_rcfg(name, *overrides):
    merged = _compose(name)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_container(merged, resolve=True)


# ---------------------------------------------------------------------------
# The campaign's scope (CONTRACTS.md "Scope update", 2026-09-18 evening)
# ---------------------------------------------------------------------------

#: CUT from the campaign. These configs are deliberately left at their
#: pre-campaign state, so no rule below may be asserted about them.
#: (`exp_gpt2_small_comparison` was in this set until 2026-09-19, when the user
#: re-admitted it; it is now an in-plan P1 scan launched by campaign/plan_gpt2.yaml.)
CUT = frozenset({
    "exp_critbatch_mnist", "exp_critbatch_nanogpt",
    "cifar10_resnet_kappaScan_labelReg", "cifar10_resnet_ce_kappaScan",
    "cifar10_resnet_paramFrac_scan_labelReg", "cifar10_resnet_ce_paramFrac_scan",
    "mnist_scan_brier",
})
#: Stale, in no launcher and in no results dir (grid_inventory.md section 4).
STALE = frozenset({"mnist_microbatch_scan", "rebuttal_mnist_batchk_probe"})

#: The six headline scans (P0), in the order CONTRACTS.md lists them.
HEADLINE = ("toy_1d_scan", "polynomial_scan", "mnist_scan_labelRegression",
            "mnist_scan_ce", "cifar10_resnet_scan_labelRegression",
            "cifar10_resnet_ce_scan")
#: Scans whose model is a ResNet or a fine-tuned ResNet (checkpoints: final; the
#: two headline ones and Fig-5 also carry bn_mode: batch).
CIFAR_SCANS = ("cifar10_resnet_scan_labelRegression", "cifar10_resnet_ce_scan",
               "rebuttal_fig5_cifar_paramfrac_scan", "exp_finetune_cifar_smallN")

#: Every in-scope scan, with the mode(s) a launcher runs it in and the `n_data`
#: sweep the launcher passes (() = the config's own single value).
#: `mode` here is what must ENUMERATE, not what one job runs: a scan launched as
#: svd + standard is listed as "both".
IN_SCOPE = {
    # P0: six headline scans + nanoGPT
    "toy_1d_scan": ("all", [()]),
    "polynomial_scan": ("all", [()]),
    "mnist_scan_labelRegression": ("all", [()]),
    "mnist_scan_ce": ("all", [()]),
    "cifar10_resnet_scan_labelRegression": ("both", [()]),
    "cifar10_resnet_ce_scan": ("both", [()]),
    "exp_nanogpt_speedrun": ("both", [()]),
    # P1: rebuttal studies
    "rebuttal_overparam_toy_1d_scan": ("both", [(f"n_data={n}",)
                                                for n in (150, 300, 600, 1200)]),
    "rebuttal_overparam_polynomial_scan": ("both", [(f"n_data={n}",)
                                                    for n in (170, 340, 675, 1350)]),
    "rebuttal_overparam_mnist_scan": ("both", [(f"n_data={n}",) for n in (
        2500, 5000, 10000, 20000, 40000, 50000)]),
    "rebuttal_fig5_cifar_paramfrac_scan": ("svd", [()]),
    "rebuttal_batchsize_polynomial_scan": ("both", [()]),
    # P1, re-admitted by the user on 2026-09-19; launched by campaign/plan_gpt2.yaml
    # (its own plan file, its own lane) rather than by plan_campaign.yaml.
    "exp_gpt2_small_comparison": ("both", [()]),
    # P2: standalone timing companions (thin aliases; same grid, own results dir)
    "toy_1d_scan_timing": ("all", [()]),
    "polynomial_scan_timing": ("all", [()]),
    "mnist_scan_ce_timing": ("all", [()]),
    "mnist_scan_labelRegression_timing": ("all", [()]),
    "exp_nanogpt_speedrun_timing": ("both", [()]),
    "cifar10_resnet_ce_scan_timing": ("both", [()]),
    "cifar10_resnet_scan_labelRegression_timing": ("both", [()]),
    # P3: ablations, kappa retune, fine-tune
    "mnist_kappaScan_labelRegression": ("svd", [()]),
    "toy_1d_microbatch_scan": ("svd", [()]),
    "toy_1d_paramfrac_scan": ("svd", [()]),
    "polynomial_microbatch_scan": ("svd", [()]),
    "polynomial_paramfrac_scan": ("svd", [()]),
    "mnist_microbatch_labelreg_scan": ("svd", [()]),
    "mnist_paramfrac_labelreg_scan": ("svd", [()]),
    "mnist_microbatch_ce_scan": ("svd", [()]),
    "mnist_paramfrac_ce_scan": ("svd", [()]),
    "exp_finetune_cifar_smallN": ("both", [(f"n_data={n}",)
                                           for n in (250, 500, 1000, 2000)]),
}
#: In scope as a CONFIG, but deliberately NOT in the launch plan: the user moved
#: `exp_finetune_cifar_smallN` to the extension phase on 2026-09-18 ~21:30 (CONTRACTS.md
#: "Stage 1 contracts", last bullet; `p3_finetune` in campaign/plan_campaign.yaml keeps
#: its items complete and `enabled: false`). Its config edits stand -- bn_mode: frozen,
#: SGDm, the lr extension -- but its runs are not part of the campaign total.
PARKED = frozenset({"exp_finetune_cifar_smallN"})

#: Families a config describes that NO launcher submits, per scan: the two CIFAR headline
#: configs carry `lrs_jd` / `lrs_hig` / `tau_hig`, which `mode=all` would turn into 100
#: unbudgeted runs each (`p3_cifar_jd_hig`, all items `enabled: false`, awaiting a user
#: decision). They are counted in grid_counts.md's table as `(N)` -- stated, not budgeted.
PARKED_MODES = {
    "cifar10_resnet_scan_labelRegression": ("jd", "hig"),
    "cifar10_resnet_ce_scan": ("jd", "hig"),
}

SCANS = sorted(IN_SCOPE)
#: the five scans that inherit another config and override nothing
TIMING_ALIASES = {n: n[: -len("_timing")] for n in SCANS if n.endswith("_timing")}


def specs_for(scan):
    mode, overrides = IN_SCOPE[scan]
    out = []
    for ov in overrides:
        out.extend(grid.expand_grid(load_rcfg(scan, *ov), mode=mode, verbose=False))
    return out


def standard_optimizers(scan):
    """The `optimizers_standard` list a scan carries, resolved (may be a default)."""
    return list(grid.process_hparam_config(load_rcfg(scan))["optimizers_standard"])


# ---------------------------------------------------------------------------
# Scope bookkeeping
# ---------------------------------------------------------------------------

def test_every_repo_config_is_classified():
    """A new or renamed config must be put in IN_SCOPE, CUT or STALE deliberately.

    Without this, adding a config silently exempts it from every rule below.
    """
    on_disk = {p.stem for p in CONFIGS.glob("*.yaml")
               if not p.stem.startswith("profile_")}          # profile_* is out of scope
    classified = set(IN_SCOPE) | CUT | STALE
    assert on_disk - classified == set(), f"unclassified: {sorted(on_disk - classified)}"
    assert classified - on_disk == set(), f"classified but absent: {sorted(classified - on_disk)}"


@pytest.mark.parametrize("scan", SCANS)
def test_in_scope_config_composes_and_expands(scan):
    """Every in-scope config resolves and enumerates a non-empty, unique grid."""
    mode, overrides = IN_SCOPE[scan]
    for ov in overrides:
        rcfg = load_rcfg(scan, *ov)
        assert rcfg.get("name") == "scan", rcfg.get("name")
        specs = grid.expand_grid(rcfg, mode=mode, verbose=False)
        assert specs, (scan, ov)
        ids = [s.run_id for s in specs]
        assert len(set(ids)) == len(ids), Counter(ids).most_common(3)
        assert len({grid.run_hash(s, rcfg) for s in specs}) == len(specs)


@pytest.mark.parametrize("scan", SCANS)
def test_every_mode_the_config_carries_expands(scan):
    """`mode=` is a command-line override, so every family the config describes must
    enumerate -- including the jd / hig grids the old launchers never submitted
    (grid_inventory.md 5.1: a "full relaunch" silently dropped 400 JD/HIG runs)."""
    rcfg = load_rcfg(scan, *IN_SCOPE[scan][1][0])
    carried = [m for m in grid.VALID_MODES
               if any(grid.mode_flags(rcfg, m).values())]
    assert "svd" in carried or "standard" in carried, carried
    for mode in carried:
        specs = grid.expand_grid(rcfg, mode=mode, verbose=False)
        assert specs, f"{scan} carries mode={mode} but enumerates nothing"


@pytest.mark.parametrize("alias,parent", sorted(TIMING_ALIASES.items()))
def test_timing_alias_is_grid_identical_to_its_parent(alias, parent):
    """A `_timing` config exists only to give the standalone-timing runs their own
    results directory: same settings, same seeds, same run_ids, so
    `attach_standalone_times` can join them onto the scan rows by run_id."""
    assert parent in IN_SCOPE, parent
    mode = IN_SCOPE[parent][0]
    a = [s.run_id for s in grid.expand_grid(load_rcfg(alias), mode=mode, verbose=False)]
    p = [s.run_id for s in grid.expand_grid(load_rcfg(parent), mode=mode, verbose=False)]
    assert a == p, f"{alias} is not grid-identical to {parent}"


# ---------------------------------------------------------------------------
# Scope update: no weight-decay grid anywhere (C-B4 dropped)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scan", SCANS)
def test_no_in_scope_config_sweeps_weight_decay(scan):
    """AdamW runs at the torch default 0.01 and MuonW at 0.1, ONE setting each, while
    Adam and plain Muon stay the wd=0 baselines. A `weight_decays` list with more than
    one entry both doubles three optimizers and forces MuonW off its own default."""
    wd = grid.listify(load_rcfg(scan).get("weight_decays", [None]))
    assert len(wd) == 1, f"{scan} sweeps weight_decays={wd}"
    by_optim = {s.hparams["optim_name"]: s.hparams["weight_decay"]
                for s in specs_for(scan) if s.family == "standard"}
    for name, expected in (("AdamW", 0.01), ("MuonW", 0.1), ("Adam", 0.0),
                           ("Muon", 0.0), ("SGD", 0.0), ("SGDm", 0.0)):
        if name in by_optim:
            assert by_optim[name] == expected, (scan, name, by_optim[name])


# ---------------------------------------------------------------------------
# Baseline sets (C-B1 MuonW, C-B2 SGDm, C-B5/O6 Muon on ResNet)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scan", SCANS)
def test_sgdm_accompanies_sgd_and_muonw_accompanies_muon(scan):
    """C-B2: SGD without momentum is not a 2026 baseline, so every scan that runs SGD
    also runs SGDm. C-B1: MuonW goes wherever Muon does (the pair is the wd ablation
    now that there is no wd grid)."""
    if "optimizers_standard" not in load_rcfg(scan):
        # a Sven-only ablation (mode: svd). process_hparam_config would hand back the
        # stock default list, which nothing runs -- assert that rather than its contents.
        assert not grid.mode_flags(load_rcfg(scan), IN_SCOPE[scan][0])["standard"], scan
        pytest.skip(f"{scan} names no baseline set (Sven-only)")
    optims = standard_optimizers(scan)
    if "SGD" in optims:
        assert "SGDm" in optims, f"{scan} runs SGD without SGDm"
    if "Muon" in optims:
        assert "MuonW" in optims, f"{scan} runs Muon without MuonW"
    # ... and neither may appear alone, which would silently drop the wd=0 arm
    if "MuonW" in optims:
        assert "Muon" in optims, f"{scan} runs MuonW without Muon"


@pytest.mark.parametrize("scan", ("cifar10_resnet_scan_labelRegression",
                                  "cifar10_resnet_ce_scan"))
def test_cifar_headline_baseline_set(scan):
    """The CIFAR headline scans had never run AdamW, Muon, MuonW or SOAP at all
    (grid_inventory.md 5.2: the launcher used CORE_FIRST only, and Muon/MuonW were
    not even in the config). O6 keeps Muon on ResNet because the conv kernels now go
    through the vendored MuonConv; Shampoo/KFAC stay out at 11.18M params."""
    optims = set(standard_optimizers(scan))
    assert {"Adam", "AdamW", "SGD", "SGDm", "RMSprop", "Muon", "MuonW", "SOAP",
            "LBFGS", "PolyakSGD"} <= optims, sorted(optims)
    assert not {"Shampoo", "KFAC"} & optims, sorted(optims)


def test_mnist_ce_keeps_its_full_baseline_list():
    """grid_inventory.md 5.2: submit_fresh_suite.sh:79 used CORE_FIRST for this scan,
    so AdamW, Muon, MuonW, SOAP, Shampoo and KFAC were never submitted although the
    config lists them -- and no launcher ever ran mode=jd or mode=hig (5.1). The new
    launcher must run all of it, so the config has to keep describing all of it."""
    optims = set(standard_optimizers("mnist_scan_ce"))
    assert {"Adam", "AdamW", "SGD", "SGDm", "RMSprop", "Muon", "MuonW", "SOAP",
            "Shampoo", "KFAC", "LBFGS", "PolyakSGD"} == optims, sorted(optims)
    families = {s.family for s in specs_for("mnist_scan_ce")}
    assert {"svd", "standard", "lbfgs", "polyak", "jd", "hig"} == families


# ---------------------------------------------------------------------------
# Stage-1 config keys (CONTRACTS.md "Config keys")
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scan", SCANS)
def test_stage1_keys_have_valid_values(scan):
    """Every Stage-1 key a config sets must be one `grid.resolve_scan_settings` and
    `grid.resolve_svd_settings` accept -- they raise, so a typo would otherwise only
    surface when the job starts on a GPU."""
    rcfg = load_rcfg(scan, *IN_SCOPE[scan][1][0])
    st = grid.resolve_scan_settings(rcfg)
    assert st["checkpoints"] in grid.CHECKPOINT_POLICIES
    assert st["checkpoints_svd"] in (None,) + grid.CHECKPOINT_POLICIES
    assert st["eval_batch_size"] >= 1 and st["train_eval_size"] >= 1
    assert st["bn_mode"] in (None,) + grid.BN_MODES
    sched = st["svd_spectra_schedule"]
    assert sched["dense_first"] >= 0 and sched["every"] >= 1
    if rcfg.get("use_gram"):
        grid.resolve_svd_settings(rcfg)          # raises on hooks + bn_mode: batch
    # keys whose default is already right must NOT be restated (CONTRACTS.md), so a
    # reader can tell a decision from an echo
    raw = OmegaConf.to_container(OmegaConf.load(CONFIGS / f"{scan}.yaml"))
    for key, default in (("empty_cache", False), ("stop_on_nonfinite", True),
                         ("scheduler", "claims"), ("train_eval_size", 10_000),
                         ("eval_batch_size", 2048), ("checkpoints", "final")):
        assert raw.get(key, default) != default or key not in raw, (
            f"{scan} restates the default {key}: {default!r}")


#: `checkpoints` / `checkpoints_svd` per scan family (CONTRACTS.md C-L3 + the
#: storage table): toy / polynomial are small enough to checkpoint every run;
#: MNIST gives the ladder to the svd family only; nanoGPT is one state per epoch;
#: ResNet keeps `final` (a `log` ladder is ~1.6 GB of RAM per run).
EXPECTED_CHECKPOINTS = {
    "toy_1d_scan": ("log", None), "polynomial_scan": ("log", None),
    "toy_1d_microbatch_scan": ("log", None), "toy_1d_paramfrac_scan": ("log", None),
    "polynomial_microbatch_scan": ("log", None), "polynomial_paramfrac_scan": ("log", None),
    "rebuttal_overparam_toy_1d_scan": ("log", None),
    "rebuttal_overparam_polynomial_scan": ("log", None),
    "rebuttal_batchsize_polynomial_scan": ("log", None),
    "mnist_scan_labelRegression": ("final", "log"), "mnist_scan_ce": ("final", "log"),
    "mnist_microbatch_labelreg_scan": ("final", "log"),
    "mnist_paramfrac_labelreg_scan": ("final", "log"),
    "mnist_microbatch_ce_scan": ("final", "log"),
    "mnist_paramfrac_ce_scan": ("final", "log"),
    "mnist_kappaScan_labelRegression": ("final", "log"),
    "rebuttal_overparam_mnist_scan": ("final", "log"),
    "exp_nanogpt_speedrun": ("epochs", None),
    # one 163M-param state is 652 MB, so this scan keeps `final` for every family
    "exp_gpt2_small_comparison": ("final", None),
    "cifar10_resnet_scan_labelRegression": ("final", None),
    "cifar10_resnet_ce_scan": ("final", None),
    "rebuttal_fig5_cifar_paramfrac_scan": ("final", None),
    "exp_finetune_cifar_smallN": ("final", None),
}


@pytest.mark.parametrize("scan", SCANS)
def test_checkpoint_policy_matches_the_scan_family(scan):
    scan = TIMING_ALIASES.get(scan, scan)                    # aliases inherit
    st = grid.resolve_scan_settings(load_rcfg(scan))
    assert (st["checkpoints"], st["checkpoints_svd"]) == EXPECTED_CHECKPOINTS[scan], scan


@pytest.mark.parametrize("scan", SCANS)
def test_eval_batch_size_is_overridden_only_for_gpt_style_models(scan):
    """C-E1/C-E3: the 2048 default is the point of the key; only a GPT-style model
    (block_size tokens per sample) cannot hold it."""
    rcfg = load_rcfg(scan)
    is_lm = rcfg.get("loss") == "lm_ce"
    ebs = grid.resolve_scan_settings(rcfg)["eval_batch_size"]
    if is_lm:
        assert ebs < 2048, f"{scan} is an LM scan and must cap eval_batch_size"
    else:
        assert ebs == 2048, f"{scan} overrides eval_batch_size ({ebs}) without needing to"


def test_bn_mode_policy_per_scan():
    """C-E2. `batch` on the CIFAR headline scans and Fig-5: frozen running stats never
    update from init and collapsed Sven to ~28% accuracy (probe 2026-09-12), and it
    replaces the old `gram_freeze_norm_stats: false`. `frozen` on the fine-tune scan
    (O2): the ImageNet running statistics are part of the pretrained model, so they are
    frozen for EVERY optimizer, not just for the Sven wrappers. Everything else names
    no policy and keeps its family default."""
    for scan in ("cifar10_resnet_scan_labelRegression", "cifar10_resnet_ce_scan",
                 "rebuttal_fig5_cifar_paramfrac_scan"):
        assert grid.resolve_bn_mode(load_rcfg(scan)) == "batch", scan
        # batch statistics under the Gram backend need a replayable capture ...
        assert load_rcfg(scan)["gram_capture"] in ("chunked", "full"), scan
        # ... and the svd run_ids keep the `_bnbatch` token they have always had
        assert all(s.run_id.endswith("_gram_bnbatch") for s in grid.expand_grid(
            load_rcfg(scan), mode="svd", verbose=False)), scan

    assert grid.resolve_bn_mode(load_rcfg("exp_finetune_cifar_smallN")) == "frozen"
    ft = specs_for("exp_finetune_cifar_smallN")
    assert all(s.record_extra["bn_mode"] == "frozen" for s in ft)
    # every non-svd family says so in its run_id; the Gram backend always froze, so
    # the svd ids are unchanged
    assert all(("_bnfrozen" in s.run_id) == (s.family != "svd") for s in ft)

    for scan in SCANS:
        if scan in CIFAR_SCANS or TIMING_ALIASES.get(scan) in CIFAR_SCANS:
            continue
        assert grid.resolve_bn_mode(load_rcfg(scan)) is None, scan


# ---------------------------------------------------------------------------
# C-B3: grids must contain their optimum
# ---------------------------------------------------------------------------

#: (scan, key, values that must be IN the grid) -- the optima
#: ``bench/best_configs.json`` reports on a grid edge, plus the known cases listed
#: under C-B3 in CHANGES_NEEDED.md. "In the grid" is the weak form of the
#: requirement; the strong form (strictly interior) is what tools/reconcile.py
#: checks against the finished runs.
MUST_CONTAIN = [
    # Adam/RMSprop/Muon/SOAP/JD on the old 1e-4 bottom edge on MNIST-CE; KFAC on it
    # on polynomial; Muon/MuonW moved down by match_rms_adamw (optim.fix.md (c)).
    ("mnist_scan_ce", "lrs_standard", [1e-5, 3e-5]),
    ("mnist_scan_labelRegression", "lrs_standard", [1e-5, 3e-5]),
    ("polynomial_scan", "lrs_standard", [1e-5, 3e-5]),
    ("mnist_scan_ce", "lrs_jd", [1e-5, 3e-5]),
    ("toy_1d_scan", "lrs_jd", [1e-5, 3e-5]),
    # SGD and Shampoo on the old 1e-1 top edge in several scans
    ("toy_1d_scan", "lrs_standard", [3e-1, 1.0]),
    ("polynomial_scan", "lrs_standard", [3e-1, 1.0]),
    ("mnist_scan_labelRegression", "lrs_standard", [3e-1, 1.0]),
    # RMSprop's optimum on CIFAR-CE IS the old 1e-1 top edge (seed-mean final val loss
    # over this scan's 290 legacy records: 2.1002 / 1.3324 / 1.3725 / 1.1983 at
    # 1e-4 / 1e-3 / 1e-2 / 1e-1), so that edge gets its two half-decade points too.
    # The label-regression scan does NOT: nothing sits on its top edge there (RMSprop
    # and Adam peak at 1e-2, SGD diverges above 1e-3), which is why the two lists differ.
    ("cifar10_resnet_ce_scan", "lrs_standard", [3e-1, 1.0]),
    # L-BFGS's best shape sits on the lr top edge on BOTH CIFAR headline scans
    # (label-reg per-lr means 3.0635 / 0.7855 / 0.7534; CE best shape mi2/hs5 1.3533 at
    # 0.5 vs 1.3201 at 1.0). One bracketing point: lr > 1 under strong_wolfe is
    # over-relaxation, and the CE mi3 arms already blow up going 0.5 -> 1.0.
    ("cifar10_resnet_ce_scan", "lrs_lbfgs", [2.0]),
    ("cifar10_resnet_scan_labelRegression", "lrs_lbfgs", [2.0]),
    # SOAP on the old bottom edge on nanoGPT; Muon needs headroom below 3e-4
    ("exp_nanogpt_speedrun", "lrs_standard", [1e-5, 3e-5]),
    # GPT-2-small: the legacy list started at 3e-4 and topped out at 1e-2, which under
    # match_rms_adamw (5.5x on the 768x768 matrices, 11.1x on the 3072x768 ones) is a
    # grid that cannot hold Muon's optimum. It gets the nanoGPT list.
    ("exp_gpt2_small_comparison", "lrs_standard", [1e-5, 3e-5]),
    # Sven at k=B with the smallest rtol on polynomial
    ("polynomial_scan", "rtol", [1e-5]),
    # Sven at the smallest lr on CIFAR-CE
    ("cifar10_resnet_ce_scan", "lrs", [0.02, 0.05]),
    # HIG's tau on the old 1e-2 top edge on 3 of the 4 MLP scans
    ("mnist_scan_ce", "tau_hig", [3e-2, 1e-1]),
    ("mnist_scan_labelRegression", "tau_hig", [3e-2, 1e-1]),
    ("polynomial_scan", "tau_hig", [3e-2, 1e-1]),

    # --- the C-B3 EXTENSION ROUND (user-approved, 2026-09-19) -------------------
    # One round, sized from the FINISHED campaign rather than from legacy records:
    # every point below closes an edge that `tools/reconcile.py` flagged over the
    # 15,735 runs in campaign/reconcile_2026-09-19.txt (the `edges` column of "best
    # config per method"). Sven's `k` is deliberately not among them -- k = B is a
    # method boundary, not a grid edge -- and CIFAR's Sven grids are left alone
    # (each run is ~0.5 GPU-h; grid_counts.md "Open items" 1 states that edge).
    #
    # The three P1 scans that had kept the pre-extension 1e-4..1e-1 `lrs_standard`
    # (grid_counts.md "Open items" 3) now use the headline list: eight of ten
    # optimizers were best on its 1e-1 top edge on overparam-toy, five on
    # overparam-polynomial, and KFAC sat on the 1e-4 BOTTOM edge on both
    # overparam-MNIST and the batch-size scan.
    ("rebuttal_overparam_toy_1d_scan", "lrs_standard", [1e-5, 3e-5, 3e-1, 1.0]),
    ("rebuttal_overparam_polynomial_scan", "lrs_standard", [1e-5, 3e-5, 3e-1, 1.0]),
    ("rebuttal_overparam_mnist_scan", "lrs_standard", [1e-5, 3e-5, 3e-1, 1.0]),
    ("rebuttal_batchsize_polynomial_scan", "lrs_standard", [1e-5, 3e-5, 3e-1, 1.0]),
    # L-BFGS best at its lr TOP edge (1.0) on four MLP scans -> two half-decades up.
    # Its MNIST-label-regression and overparam-MNIST optima are lr 0.5, interior, so
    # those two lists are deliberately NOT extended.
    ("polynomial_scan", "lrs_lbfgs", [2.0, 4.0]),
    ("mnist_scan_ce", "lrs_lbfgs", [2.0, 4.0]),
    ("rebuttal_overparam_toy_1d_scan", "lrs_lbfgs", [2.0, 4.0]),
    ("rebuttal_overparam_polynomial_scan", "lrs_lbfgs", [2.0, 4.0]),
    ("rebuttal_batchsize_polynomial_scan", "lrs_lbfgs", [2.0, 4.0]),
    # ... and at its BOTTOM edge (0.1) on toy_1d, the one scan where it goes the
    # other way (0.0002459 at 0.1, 4/5 seeds finite).
    ("toy_1d_scan", "lrs_lbfgs", [0.03, 0.01]),
    # CIFAR L-BFGS: the 2.0 added in the first round IS the new optimum on both scans
    # (0.4380 label-reg, 1.014 CE -- second only to MuonW there). ONE point again, at
    # ~0.04 GPU-h per run.
    ("cifar10_resnet_ce_scan", "lrs_lbfgs", [4.0]),
    ("cifar10_resnet_scan_labelRegression", "lrs_lbfgs", [4.0]),
    # CIFAR-CE: SGD's optimum is the 1.0 top point the first round added (1.338). One
    # point: no other optimizer on that scan is above 3e-1.
    ("cifar10_resnet_ce_scan", "lrs_standard", [3.0]),
    # KFAC is STILL on the bottom edge of the extended polynomial list: seed-mean final
    # val 0.1728 at 1e-5, 0.1872 at 3e-5, 0.3405 at 1e-4, monotone down.
    ("polynomial_scan", "lrs_standard", [1e-6, 3e-6]),
    # Sven's own non-k axes, where the reconcile flagged an edge. toy_1d is the
    # headline Sven scan and was on BOTH its lr and its rtol bottom edge; polynomial
    # and MNIST-CE are on the rtol TOP edge (0.1175 at 1e-2 / 0.1231 at 1e-1); the
    # batch-size scan is on the rtol BOTTOM edge; overparam-polynomial is on both
    # (at its best cell the rtol seed-means are 6.646 / 0.5989 / 0.2829 going up).
    # MNIST-CE gets one point only: rtol is a relative singular-value cut, so the
    # axis runs out at 1.0 (only the leading direction survives) and 3e-1 is the last
    # informative point. The batch-size scan likewise stops one half-decade down.
    ("toy_1d_scan", "lrs", [0.02, 0.01]),
    ("toy_1d_scan", "rtol", [1e-5, 1e-6]),
    ("polynomial_scan", "rtol", [3e-2, 1e-1]),
    ("mnist_scan_ce", "rtol", [3e-1]),
    ("rebuttal_batchsize_polynomial_scan", "rtol", [1e-5]),
    ("rebuttal_overparam_polynomial_scan", "lrs", [0.02, 0.01]),
    ("rebuttal_overparam_polynomial_scan", "rtol", [3e-2, 1e-1]),
    # HIG's lr on toy_1d: the C-B3 shift landed its optimum (4.707e-09, the best
    # number on any MLP scan) on the NEW bottom edge, 0.005.
    ("toy_1d_scan", "lrs_hig", [1.5e-3, 5e-4]),
]


@pytest.mark.parametrize("scan,key,values", MUST_CONTAIN,
                         ids=[f"{s}-{k}" for s, k, _ in MUST_CONTAIN])
def test_extended_grids_contain_the_new_points(scan, key, values):
    grid_values = [float(v) for v in grid.listify(load_rcfg(scan)[key])]
    for v in values:
        assert any(abs(g - v) < 1e-12 * max(1.0, abs(v)) for g in grid_values), (
            f"{scan}.{key} is missing {v}; have {sorted(grid_values)}")


@pytest.mark.parametrize("scan", ("toy_1d_scan", "polynomial_scan",
                                  "mnist_scan_labelRegression", "mnist_scan_ce"))
def test_hig_grid_is_shifted_down_not_grown(scan):
    """C-B3 for HIG: the grid moves DOWN rather than outward, because its optimum is the
    old bottom edge (lr 0.05) on both MNIST scans while its lr >= 0.5 points earn nothing.
    The evidence differs per scan, and only one scan has any: lr >= 0.5 produced NO legacy
    records on toy_1d, polynomial or mnist_scan_labelRegression, and on mnist_scan_ce all
    40 of those runs completed but are dominated (seed-mean final val 0.1035 at lr 0.05 /
    tau 1e-2 against 0.1735 at lr 0.5 and 0.2023 at lr 1.0). "Always crashes" was the
    wrong reason for the right decision -- see _explain_mnist_scan_ce in tests/test_grid.py.

    The shift held: on the finished campaign HIG's optimum is lr 0.05 on polynomial and
    both MNIST scans, i.e. interior, and nothing wants lr >= 0.5 anywhere. Only toy_1d
    moved, and downward again (0.005, the new bottom edge) -- see MUST_CONTAIN. So the
    top-end bound below is the invariant, and the bottom is a floor, not a fixed point.
    """
    lrs = [float(v) for v in grid.listify(load_rcfg(scan)["lrs_hig"])]
    assert max(lrs) < 0.5, f"{scan} still runs HIG at lr >= 0.5: {sorted(lrs)}"
    assert min(lrs) <= 0.005 + 1e-12, f"{scan} HIG grid does not reach 0.005: {sorted(lrs)}"


def test_muon_lr_grids_reach_below_the_legacy_optimum():
    """C-B5 + C-B3. `adjust_lr_fn=match_rms_adamw` multiplies Muon's effective step by
    0.2*sqrt(max(A,B)) of the (flattened) weight shape -- 1.1-5.6x on the MNIST MLP,
    2.3-4.5x on nanoGPT, up to 13.6x on the ResNet18 layer4 kernels (optim.fix.md (c)).
    The shared `lrs_standard` list must therefore reach two half-decades below the
    optimum the OLD Muon had, or the new Muon's optimum falls off the bottom.
    """
    legacy_muon_optimum = {           # bench/best_configs.json, pre-match_rms_adamw
        "mnist_scan_labelRegression": 1e-3,
        "mnist_scan_ce": 1e-4,
        "exp_nanogpt_speedrun": 3e-4,
    }
    for scan, optimum in legacy_muon_optimum.items():
        lrs = [float(v) for v in grid.listify(load_rcfg(scan)["lrs_standard"])]
        # two half-decade points below `optimum` ~ a factor 10
        assert min(lrs) <= optimum / 10 * (1 + 1e-9), (
            f"{scan}: lrs_standard bottoms out at {min(lrs)}, needs <= {optimum / 10}")
    # CIFAR and GPT-2-small: Muon has never run there at all (the GPT-2 scan was
    # cancelled in flight on 2026-09-18 and has no records), so the requirement is the
    # grid edge rather than a measured optimum.
    for scan in ("cifar10_resnet_scan_labelRegression", "cifar10_resnet_ce_scan",
                 "exp_gpt2_small_comparison"):
        lrs = [float(v) for v in grid.listify(load_rcfg(scan)["lrs_standard"])]
        assert min(lrs) <= 1e-5 * (1 + 1e-9), sorted(lrs)


# ---------------------------------------------------------------------------
# Study-specific decisions
# ---------------------------------------------------------------------------

#: the effective steps 2*lr/kappa that the C-X1 grid realises at EVERY kappa. These
#: three triples are what makes the study a kappa comparison rather than an lr sweep:
#: without them F29 ("the kappa scan is an lr sweep in disguise") still applies, since
#: at k = B = 64 and full row rank the kappa step is exactly 2/kappa times the kappa=2
#: step. A full lr x kappa match would need S/2, S, 3S/2 in the grid for every S, i.e.
#: 15 lrs and 450 runs.
KAPPA_MATCHED_STEPS = (0.25, 0.5, 1.0)


def test_kappa_scan_separates_kappa_from_the_effective_step():
    """C-X1 (F29): at k = B and one fixed lr the kappa step is exactly 2/kappa times
    the kappa=2 step, so the old 3-run scan was an lr sweep in disguise. The grid now
    retunes lr per kappa AND carries a slice where truncation binds (k=32 < B=64)."""
    rcfg = load_rcfg("mnist_kappaScan_labelRegression")
    kappas = sorted(grid.listify(rcfg["kappa"]))
    assert kappas == [1, 2, 3]
    assert sorted(grid.listify(rcfg["k_values"])) == [32, 64]
    assert rcfg["batch_size"] == 64                     # so k=32 really is k < B
    assert len(grid.listify(rcfg["model_seeds"])) == 5
    lrs = sorted(float(v) for v in grid.listify(rcfg["lrs"]))
    assert lrs == [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5]
    assert 0.5 in lrs                                   # the legacy set point on disk

    def has(lr):
        return any(abs(lr - o) < 1e-12 for o in lrs)

    # the strong form of C-X1's "at minimum include lr * kappa / 2": each step in
    # KAPPA_MATCHED_STEPS is reachable at kappa 1, 2 AND 3 (lr = step * kappa / 2), so
    # the "same effective step, different kappa" rows exist for all three arms ...
    for step in KAPPA_MATCHED_STEPS:
        for kappa in kappas:
            assert has(step * kappa / 2), (step, kappa, lrs)
    # ... and those are exactly the matched steps, so the comment cannot overclaim
    realised = {k: {2 * lr / k for lr in lrs} for k in kappas}
    matched = sorted(set.intersection(*realised.values()))
    assert matched == pytest.approx(list(KAPPA_MATCHED_STEPS)), matched

    specs = specs_for("mnist_kappaScan_labelRegression")
    assert len(specs) == 7 * 3 * 2 * 5 == 210
    assert Counter(s.hparams["kappa"] for s in specs) == {1: 70, 2: 70, 3: 70}


#: the pre-Gram "classic" Fig-5 set point the config currently carries, and the values
#: EXPERIMENTS.md:108-109 documents for the same figure. They disagree, `best_configs.json`
#: has no CIFAR entry, and the re-point has to come from the BN-fixed
#: `cifar10_resnet_scan_labelRegression` headline scan (grid_inventory.md 5.5).
FIG5_TENTATIVE_SETPOINT = {"k_values": [64], "lrs": [1.0], "rtol": [1e-3]}
FIG5_TENTATIVE_MARKER = "STILL TENTATIVE"


def test_fig5_setpoint_is_flagged_tentative_exactly_while_it_is_tentative():
    """Fig-5 is a headline figure and its k / lr / rtol are NOT measured values, so the
    failure mode is a wrong figure rather than a small waste of 15 runs (~5.5 GPU-h).

    The two halves of the set point -- the values and the "must be re-pointed" marker --
    can therefore only move together: while the marker is in the header the values must
    be exactly the classic triple (nobody may edit one line of three and leave the
    warning), and once the marker goes the values must actually have changed.

    This is the half of the guard that lives in a file this track owns. The other half
    belongs to the launcher: `p1_cifar_fig5` in campaign/plan_campaign.yaml is currently
    `enabled` (by omission), so a P1 launch WOULD run the tentative values -- see the
    report and campaign/grid_counts.md "Open items".
    """
    text = (CONFIGS / "rebuttal_fig5_cifar_paramfrac_scan.yaml").read_text()
    header = "\n".join(l for l in text.splitlines() if l.lstrip().startswith("#"))
    rcfg = load_rcfg("rebuttal_fig5_cifar_paramfrac_scan")
    actual = {k: [float(v) for v in grid.listify(rcfg[k])]
              for k in FIG5_TENTATIVE_SETPOINT}
    classic = {k: [float(v) for v in v_] for k, v_ in FIG5_TENTATIVE_SETPOINT.items()}
    if FIG5_TENTATIVE_MARKER in header:
        assert actual == classic, (
            "the Fig-5 set point was edited but the header still says it is tentative; "
            f"have {actual}, classic {classic}")
        assert "re-point" in header.lower() or "RE-POINTED" in header
    else:
        assert actual != classic, (
            "the tentative marker was removed but the set point is still the pre-Gram "
            "classic triple")


def test_batchsize_scan_pins_the_lbfgs_shape_and_sweeps_only_its_lr():
    """O5: the swept axis of this scan is the BATCH SIZE. max_iter / history_size are
    pinned at the legacy polynomial-headline best (bench/best_configs.json: max_iter 3,
    history_size 2), which takes the L-BFGS family from 810 runs to 90."""
    rcfg = load_rcfg("rebuttal_batchsize_polynomial_scan")
    assert grid.listify(rcfg["lbfgs_max_iter"]) == [3]
    assert grid.listify(rcfg["lbfgs_history_size"]) == [2]
    # 5 lrs since the C-B3 extension round put 2.0 / 4.0 above the 1.0 top edge the
    # finished scan came back on -- the SHAPE stays pinned, which is what O5 is about.
    assert len(grid.listify(rcfg["lrs_lbfgs"])) == 5
    counts = Counter(s.family for s in specs_for("rebuttal_batchsize_polynomial_scan"))
    assert counts["lbfgs"] == 150, counts


def test_overparam_mnist_header_documents_the_50000_cap():
    """The MNIST train part is 50,000 under C-E1 and `n_train` above the pool now
    RAISES instead of clamping, so the old n_data=60000 top point is an error. The
    sweep is documented in the yaml header because the launcher passes it, not the
    config (data.fix.md: three launchers still say 60000)."""
    header = "\n".join(
        l for l in (CONFIGS / "rebuttal_overparam_mnist_scan.yaml").read_text()
        .splitlines() if l.startswith("#"))
    assert "<= 50000" in header
    assert "n_data=60000" not in header       # the mislabelled top point is gone
    swept = re.search(r"^#\s+2500\s+5000\s+10000\s+20000\s+40000\s+50000\s*$",
                      header, re.M)
    assert swept, "the n_data sweep is not listed in the header"
    # and the sweep this file tests with is the documented one
    assert [ov[0] for ov in IN_SCOPE["rebuttal_overparam_mnist_scan"][1]] == [
        "n_data=2500", "n_data=5000", "n_data=10000",
        "n_data=20000", "n_data=40000", "n_data=50000"]


def test_gpt2_small_is_a_single_epoch_scan_with_step_based_evaluation():
    """C-E4 is what makes this scan readable at all: its token budget is ONE pass over
    train.bin, so the epoch-end curves have two points (untrained + final) and every
    "loss vs steps" figure would come from `val_step` / `test_step`.

    Also pins the three facts a reader of the yaml cannot check locally: the budget
    really is 13,125 optimizer steps (210,000 blocks / B=16), one evaluation is a few
    dozen forward batches rather than a second pass over the corpus, and the dataset
    carries the v2 val AND test splits (F4: the v1 bins had val as a prefix of train).
    """
    rcfg = load_rcfg("exp_gpt2_small_comparison")
    st = grid.resolve_scan_settings(rcfg)
    assert rcfg["num_epochs"] == 1 and rcfg["batch_size"] == 16
    steps = rcfg["dataset"]["n_train_blocks"] // rcfg["batch_size"]
    assert steps == 13_125, steps
    # ~26 mid-epoch points: enough to draw a curve, few enough to stay a rounding error
    assert st["eval_every_steps"] == 500
    assert 20 <= steps // st["eval_every_steps"] <= 40

    # an evaluation is val + test + train_eval; all three must be small, because a
    # step-based schedule pays for them 26 times per run (_record_evals has no
    # per-split schedule)
    ebs = st["eval_batch_size"]
    blocks = (rcfg["dataset"]["val_blocks"], rcfg["dataset"]["test_blocks"],
              st["train_eval_size"])
    assert all(b > 0 for b in blocks), blocks
    assert sum(-(-b // ebs) for b in blocks) <= 64, blocks      # forward batches
    # the eval batch may not exceed the training batch's token count: B x 1024 x 50304
    # logits is 206 MB per block and the Sven state already holds ~33 GB
    assert ebs <= rcfg["batch_size"], ebs

    # the v2 token directory, with a test split (tools/check_token_split.py passes on it)
    assert rcfg["dataset"]["ROOT"].endswith("fineweb_edu_gpt2_v2")
    assert rcfg["dataset"]["block_size"] == rcfg["model"]["block_size"] == 1024

    # the Sven set point is the legacy one (k = B, full Gram rank, soft-cut by rtol)
    assert grid.listify(rcfg["k_fractions"]) == [1.0]
    assert [float(v) for v in grid.listify(rcfg["lrs"])] == [0.1, 0.5, 1.0]
    assert rcfg["gram_capture"] == "hooks"          # untied + dropout 0
    assert not rcfg["model"]["tie_weights"]
    # one seed, and no SGD -> the C-B2 SGDm requirement does not apply
    assert grid.listify(rcfg["model_seeds"]) == [6000]
    assert "SGD" not in standard_optimizers("exp_gpt2_small_comparison")
    counts = Counter(s.family for s in specs_for("exp_gpt2_small_comparison"))
    assert counts == {"svd": 3, "standard": 24}, counts


def test_cut_scans_are_untouched_and_stale_ones_stay_out():
    """The scope update cuts seven configs (eight until GPT-2-small was re-admitted on
    2026-09-19); they keep their pre-campaign grids so that
    reviving one is a config decision, not an archaeology exercise. `mnist_scan_brier`
    in particular is a second full MNIST headline grid."""
    for scan in CUT | STALE:
        rcfg = load_rcfg(scan, *(("n_data=1000",) if "critbatch" in scan else ()))
        assert rcfg.get("name") == "scan", scan
        assert grid.resolve_bn_mode(rcfg) is None or scan in CUT, scan
        # no cut config may have picked up a Stage-1 key
        raw = OmegaConf.to_container(OmegaConf.load(CONFIGS / f"{scan}.yaml"))
        assert "bn_mode" not in raw and "checkpoints_svd" not in raw, scan


# ---------------------------------------------------------------------------
# campaign/grid_counts.md is generated from these configs -- keep it honest
# ---------------------------------------------------------------------------

def _parse_grid_counts():
    """{scan: (fams, parked_fams, total, pri)} from the table in grid_counts.md.

    A cell is `-` (the config does not describe that family for the mode a launcher
    runs it in), an integer (launched, and part of the total), or `(N)` -- described by
    the config but submitted by nothing and excluded from the total, which is how the
    CIFAR jd/hig liability is stated rather than left to prose.
    """
    rows = {}
    for line in GRID_COUNTS.read_text().splitlines():
        m = re.match(r"^\|\s*`([a-zA-Z0-9_]+)`\s*\|(.+)\|\s*\*\*(\d+)\*\*\s*\|", line)
        if not m:
            continue
        scan, cells, total = m.group(1), m.group(2).split("|"), int(m.group(3))
        fams, parked = {}, {}
        for fam, cell in zip(grid.FAMILIES, cells[1:]):      # cells[0] = priority
            cell = cell.strip().strip("*")
            if not cell or cell in ("-", "--", "—"):
                continue
            if cell.startswith("(") and cell.endswith(")"):
                parked[fam] = int(cell[1:-1])
            else:
                fams[fam] = int(cell.split()[0])
        rows[scan] = (fams, parked, total, cells[0].strip())
    return rows


def _parked_family_counts(scan):
    """What the modes no launcher submits would enumerate for `scan`."""
    out = Counter()
    for mode in PARKED_MODES.get(scan, ()):
        rcfg = load_rcfg(scan, *IN_SCOPE[scan][1][0])
        out.update(Counter(s.family for s in grid.expand_grid(
            rcfg, mode=mode, verbose=False)))
    return out


def test_grid_counts_md_matches_the_configs():
    """The launcher and the cost plan read campaign/grid_counts.md; it must be exactly
    what expand_grid says today, or a config edit silently invalidates the plan.

    Three totals have to appear, because "in scope" and "in the launch plan" are no
    longer the same set: the in-plan total, the parked total (exp_finetune, moved to the
    extension phase by the user) and their sum.
    """
    table = _parse_grid_counts()
    expected = {s: Counter(x.family for x in specs_for(s)) for s in SCANS
                if s not in TIMING_ALIASES}
    assert set(table) == set(expected), (
        f"only in md: {sorted(set(table) - set(expected))}; "
        f"only in configs: {sorted(set(expected) - set(table))}")
    for scan, (fams, parked, total, pri) in sorted(table.items()):
        assert fams == dict(expected[scan]), (scan, fams, dict(expected[scan]))
        assert total == sum(expected[scan].values()), (scan, total)
        assert parked == dict(_parked_family_counts(scan)), (scan, parked)
        # a scan outside the launch plan must say so where the launcher reads it
        assert ("parked" in pri.lower()) == (scan in PARKED), (scan, pri)
    totals = {s: sum(c.values()) for s, c in expected.items()}
    in_plan = sum(n for s, n in totals.items() if s not in PARKED)
    parked_total = sum(n for s, n in totals.items() if s in PARKED)
    text = GRID_COUNTS.read_text()
    for label, n in (("in-plan", in_plan), ("parked", parked_total),
                     ("grid", in_plan + parked_total)):
        assert re.search(rf"\*\*{n}\*\*", text), (
            f"the {label} total {n} is not in grid_counts.md")
