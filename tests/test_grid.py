"""CPU tests for ``experiments/experiment_code/grid.py`` (C-R2, refactor only).

The contract under test is "same grid, same order, same names": ``expand_grid``
must reproduce the enumeration the six copy-pasted ``itertools.product`` blocks
in ``generic_scan.scan`` produced before the refactor. The oracles are

* ``tests/golden/<scan>.txt`` -- the top-level ``*.jsonl`` filenames of nine real
  result directories under ``experiment_results/`` (run_ids actually on disk),
* ``tests/golden/<scan>.order.txt`` -- the *ordered* legacy enumeration, frozen by
  running ``scan()`` from the pre-C-R2 commit with execution stubbed out
  (``instantiate``/``set_seed``/``_scan_facts`` neutered and ``os.path.exists``
  recording each run_id and returning True, so the legacy loop only enumerates
  and its ``_shard_skip`` counter still runs). This is what pins the *order*, and
  hence that ``specs[shard_id::n_shards]`` picks the legacy shard. The producer is
  ``tests/golden/legacy_grid_equiv.py`` (in the repo, pinning the pre-refactor
  commit), which also checks order and six (n_shards, shard_id) settings against
  the legacy code for 11 config/mode combinations,
* the per-config, per-family run counts in ``campaign/scout/grid_inventory.md``.

Everything here is torch-free and runs in a second: ``grid.py`` is loaded straight
from its path so that the ``experiments.experiment_code`` package ``__init__``
(which imports ``generic_scan`` and hence torch) is not involved. The other half
of C-R2 -- ``scan()`` / ``execute()``, which needs torch -- is pinned here
*structurally* (``test_execute_keeps_the_per_family_asymmetries``) and end to end
by ``tests/golden/scan_smoke.sh``.

RE-FREEZING (read before editing a scan config that has an ``.order.txt``)
-------------------------------------------------------------------------
Dropping the weight-decay grid (C-B4: ``rebuttal_batchsize_polynomial_scan.yaml``
loses ``weight_decays: [0.0, 0.01]``) or any other edit to ``toy_1d_scan.yaml`` /
``rebuttal_batchsize_polynomial_scan.yaml`` invalidates, in this order:

1. ``tests/golden/rebuttal_batchsize_polynomial_scan.order.txt`` (2640 ids) and
   ``tests/golden/toy_1d_scan.order.txt`` (780) -- regenerate with
   ``campaign/run_cpu_tests.sh .venv/bin/python tests/golden/legacy_grid_equiv.py --freeze``
   (it reads the config from the working tree and the runner from the pinned SHA,
   so a re-freeze still compares legacy vs. new enumeration of the SAME config),
2. ``INVENTORY["rebuttal_batchsize_polynomial_scan"]["standard"] == 1440``
   (halves to 720 when the two-point wd grid becomes one point),
3. ``test_weight_decay_grid_filter``'s ``per_optim`` counts (240 -> 120 for
   AdamW / Muon / MuonW),
4. ``_explain_rebuttal_batchsize_polynomial_scan``: class (1) maps the 120 on-disk
   ``_optimAdamW_mseed...`` ids onto today's ``_wd0.0`` ones, which the config will
   no longer produce -- that class then needs the same treatment as class (2)
   (assert they come back under an explicit ``weight_decays=[0.0]`` override).

Do NOT loosen an assertion instead; the point of these oracles is that a config
edit is visible.
"""

import ast
import importlib.util
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
GRID_PATH = REPO / "experiments" / "experiment_code" / "grid.py"
SCAN_PATH = REPO / "experiments" / "experiment_code" / "generic_scan.py"
UTILS_PATH = REPO / "experiments" / "experiment_code" / "experiment_utils.py"
FACTORY_PATH = REPO / "experiments" / "experiment_code" / "optim_factory.py"
CONFIGS = REPO / "experiments" / "configs"
GOLDEN = Path(__file__).parent / "golden"


def _load_grid():
    spec = importlib.util.spec_from_file_location("sv3_grid_under_test", GRID_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses needs the module registered
    spec.loader.exec_module(mod)
    return mod


grid = _load_grid()


# ---------------------------------------------------------------------------
# Config loading without hydra (and therefore without torch)
# ---------------------------------------------------------------------------

def _compose(name):
    """The defaults list of ``<name>.yaml``, merged, with ``_self_`` last.

    Covers the two entry shapes these configs use: ``{group: option}`` (load
    ``configs/<group>/<option>.yaml`` under the key ``group``) and a bare string
    (a parent config in the same directory -- the ``*_timing`` aliases).
    """
    cfg = OmegaConf.load(CONFIGS / f"{name}.yaml")
    defaults = cfg.pop("defaults", [])
    merged = OmegaConf.create({})
    for entry in defaults:
        if entry == "_self_":
            continue
        if isinstance(entry, str):
            merged = OmegaConf.merge(merged, _compose(entry))
            continue
        for group, option in OmegaConf.to_container(entry).items():
            merged = OmegaConf.merge(merged, OmegaConf.create(
                {group: OmegaConf.load(CONFIGS / group / f"{option}.yaml")}))
    return OmegaConf.merge(merged, cfg)            # _self_ is last in every config


def load_rcfg(name, *overrides):
    """Resolve a scan config the way ``hydra.compose`` would, with OmegaConf only
    (dotlist overrides, and full interpolation resolution: ``${mlp_width}``,
    ``${data_seed}``, ``${n_data}``, ...)."""
    merged = _compose(name)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_container(merged, resolve=True)


# ---------------------------------------------------------------------------
# The nine golden scans: the mode(s) they were launched with and the command-line
# overrides that produced the directory (see submit_fresh_suite.sh).
# ---------------------------------------------------------------------------

# `mode` is what the launchers used, not blindly "all": the two headline MLP dirs
# also hold ad-hoc mode=jd / mode=hig runs (grid_inventory.md 5.1), the CIFAR dir
# carries the jd/hig grids in its config but they were never launched, and the
# three ablation configs are `mode: svd` (a plain "all" would enumerate a
# meaningless default baseline grid for them).
MODE = {
    "toy_1d_scan": "all",
    "mnist_scan_ce": "all",
    "cifar10_resnet_scan_labelRegression": "all",
    "rebuttal_batchsize_polynomial_scan": "both",
    "exp_critbatch_nanogpt": "both",
    "mnist_paramfrac_labelreg_scan": "svd",
    "mnist_kappaScan_labelRegression": "svd",
    "toy_1d_microbatch_scan": "svd",
    "exp_finetune_cifar_smallN": "both",
}

# exp_finetune sweeps n_data from the launcher (submit_fresh_suite.sh:125); its
# run_ids carry it via result_id_fields, so all four subsets share one directory.
OVERRIDES = {
    "exp_finetune_cifar_smallN": [(f"n_data={n}",) for n in (250, 500, 1000, 2000)],
}

SCANS = sorted(MODE)

# Per-family run counts from campaign/scout/grid_inventory.md section 1 (all seeds).
# The inventory's "first-order" and "2nd-order" columns are both the `standard`
# family here; `Sven` is `svd`. Counts are per-config, i.e. summed over the
# n_data overrides for exp_finetune (its table row reads "60 x 4 N = 240").
INVENTORY = {
    "toy_1d_scan": {"svd": 360, "standard": 180, "lbfgs": 135, "polyak": 5,
                    "jd": 20, "hig": 80},
    "mnist_scan_ce": {"svd": 640, "standard": 180, "lbfgs": 135, "polyak": 5,
                      "jd": 20, "hig": 80},
    "cifar10_resnet_scan_labelRegression": {"svd": 90, "standard": 60, "lbfgs": 135,
                                            "polyak": 5, "jd": 20, "hig": 80},
    "rebuttal_batchsize_polynomial_scan": {"svd": 360, "standard": 1440,
                                           "lbfgs": 810, "polyak": 30},
    "exp_critbatch_nanogpt": {"svd": 120, "standard": 90},
    "mnist_paramfrac_labelreg_scan": {"svd": 100},
    "mnist_kappaScan_labelRegression": {"svd": 15},
    "toy_1d_microbatch_scan": {"svd": 120},
    "exp_finetune_cifar_smallN": {"svd": 48, "standard": 192},
}


def specs_for(scan):
    """Every spec a scan's directory could contain, over all its n_data overrides."""
    out = []
    for ov in OVERRIDES.get(scan, [()]):
        rcfg = load_rcfg(scan, *ov)
        out.extend(grid.expand_grid(rcfg, mode=MODE[scan], verbose=False))
    return out


def golden_ids(scan):
    lines = (GOLDEN / f"{scan}.txt").read_text().split()
    return {l[:-len(".jsonl")] for l in lines if l.endswith(".jsonl")}


# ---------------------------------------------------------------------------
# Legacy run_ids that today's configs no longer describe, with the reason.
# Anything NOT explained here must be reproduced byte-identically.
# ---------------------------------------------------------------------------

def _explain_rebuttal_batchsize_polynomial_scan(missing, produced):
    """Two known classes in ``rebuttal_batchsize_polynomial_scan``, both from
    launcher/config edits made after those runs were written."""
    # (1) 120 AdamW ids with no `_wd` suffix: written before AdamW/MuonW started
    #     carrying their weight decay in the run_id (generic_scan.py:620-623, so
    #     default-wd runs cannot be skipped as the old wd=0 ones). The config
    #     sweeps weight_decays: [0.0, 0.01], so the same run is named
    #     `..._optimAdamW_wd0.0_...` today -- assert exactly that correspondence.
    pre_wd_suffix = {i for i in missing if "_optimAdamW_mseed" in i}
    assert len(pre_wd_suffix) == 120, len(pre_wd_suffix)
    for i in sorted(pre_wd_suffix):
        assert i.replace("_optimAdamW_mseed", "_optimAdamW_wd0.0_mseed") in produced, i

    # (2) 120 `_optimMuonW_wd0.1_` ids: the MuonW backfill in
    #     submit_reruns_2026-09-17_part2.sh runs this config with the override
    #     `weight_decays=[0.1]` (MuonW's own default), which the config does not
    #     carry. expand_grid reproduces them exactly under that override.
    muonw_backfill = {i for i in missing if "_optimMuonW_wd0.1_" in i}
    assert len(muonw_backfill) == 120, len(muonw_backfill)
    rcfg = load_rcfg("rebuttal_batchsize_polynomial_scan", "weight_decays=[0.1]")
    with_override = {s.run_id for s in grid.expand_grid(rcfg, mode="standard", verbose=False)}
    assert muonw_backfill <= with_override

    return missing - pre_wd_suffix - muonw_backfill


EXPLAINED = {
    "rebuttal_batchsize_polynomial_scan": _explain_rebuttal_batchsize_polynomial_scan,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scan", SCANS)
def test_golden_run_ids_are_reproduced(scan):
    """Every run_id on disk is produced by expand_grid for that config."""
    produced = {s.run_id for s in specs_for(scan)}
    missing = golden_ids(scan) - produced
    if scan in EXPLAINED:
        missing = EXPLAINED[scan](missing, produced)
    assert not missing, f"{len(missing)} unexplained legacy run_ids, e.g. {sorted(missing)[:3]}"


@pytest.mark.parametrize("scan", SCANS)
def test_family_counts_match_grid_inventory(scan):
    counts = Counter(s.family for s in specs_for(scan))
    assert dict(counts) == INVENTORY[scan]


@pytest.mark.parametrize("scan", SCANS)
def test_run_ids_are_unique(scan):
    """A run_id is the dedup key: two grid points may never share one."""
    ids = [s.run_id for s in specs_for(scan)]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    assert not dupes, dupes[:3]


@pytest.mark.parametrize("scan", SCANS)
def test_enumeration_is_seed_major(scan):
    """Order = seeds outer, then the fixed family order, as the legacy loops ran
    (this is what makes specs[shard_id::n_shards] the legacy shard assignment)."""
    for ov in OVERRIDES.get(scan, [()]):
        rcfg = load_rcfg(scan, *ov)
        specs = grid.expand_grid(rcfg, mode=MODE[scan], verbose=False)
        seeds = list(rcfg["model_seeds"])
        # seeds appear in config order, in one contiguous block each
        assert [s.model_seed for s in specs] == sorted(
            (s.model_seed for s in specs), key=seeds.index)
        for seed in seeds:
            fams = [s.family for s in specs if s.model_seed == seed]
            order = [f for f in grid.FAMILIES if f in fams]
            assert fams == sorted(fams, key=order.index)


def _frozen_order(scan):
    """The ordered legacy enumeration of a scan (see RE-FREEZING in the docstring)."""
    lines = [l for l in (GOLDEN / f"{scan}.order.txt").read_text().splitlines()
             if l and not l.startswith("#")]
    assert lines, f"{scan}.order.txt is empty"
    return lines


@pytest.mark.parametrize("scan,mode", [("toy_1d_scan", "all"),
                                       ("rebuttal_batchsize_polynomial_scan", "both")])
def test_enumeration_order_matches_the_legacy_loops(scan, mode):
    """Same run_ids in the same order as the six inline product() loops."""
    rcfg = load_rcfg(scan)
    assert ([s.run_id for s in grid.expand_grid(rcfg, mode=mode, verbose=False)]
            == _frozen_order(scan))


@pytest.mark.parametrize("n_shards", [1, 2, 3, 4, 6])
def test_shards_are_disjoint_and_cover_the_grid(n_shards):
    specs = specs_for("toy_1d_scan")
    slices = [grid.shard(specs, n_shards, i) for i in range(n_shards)]
    ids = [[s.run_id for s in sl] for sl in slices]
    flat = [i for sl in ids for i in sl]
    assert len(flat) == len(specs)                      # covers
    assert set(flat) == {s.run_id for s in specs}
    for i in range(n_shards):                           # disjoint
        for j in range(i + 1, n_shards):
            assert not set(ids[i]) & set(ids[j])
    assert max(len(sl) for sl in ids) - min(len(sl) for sl in ids) <= 1


@pytest.mark.parametrize("scan,mode", [("toy_1d_scan", "all"),
                                       ("rebuttal_batchsize_polynomial_scan", "both")])
@pytest.mark.parametrize("n_shards", [1, 2, 3, 7])
def test_shard_matches_the_legacy_modulo_counter(scan, mode, n_shards):
    """generic_scan.py:366-369 took a run iff a counter incremented once per
    enumerated grid point satisfied ``idx % n_shards == shard_id``.

    Checked against the *frozen legacy* enumeration, not against a second copy of
    the slice expression: the legacy counter ran over the legacy order, so this
    asserts "expand_grid order == legacy order" and "shard membership == legacy
    shard membership" together. ``legacy_grid_equiv.py`` cross-checked six
    (n_shards, shard_id) settings against the legacy loop itself.
    """
    frozen = _frozen_order(scan)
    specs = grid.expand_grid(load_rcfg(scan), mode=mode, verbose=False)
    for shard_id in range(n_shards):
        legacy = frozen[shard_id::n_shards]
        assert [s.run_id for s in grid.shard(specs, n_shards, shard_id)] == legacy


# The record scaffold each family carried between "run_id" and "losses" in the
# legacy result dicts (generic_scan.py:551-577, 656-671, 732-749, 804-821,
# 876-888, 929-940) -- key order included, so the written JSON is unchanged.
LEGACY_RECORD_KEYS = {
    "svd": ("optimizer", "loss", "batch_size", "k_fraction", "k", "lr", "rtol",
            "model_seed", "loader_seed", "svd_mode", "decomposition",
            "microbatch_size", "param_fraction", "variable_k", "use_gram",
            "gram_capture", "gram_freeze_norm_stats", "gram_chunk_numel",
            "mask_mode", "kappa", "signed_residual"),
    "standard": ("optimizer", "loss", "batch_size", "k_fraction", "k", "lr", "rtol",
                 "weight_decay", "model_seed", "loader_seed", "svd_mode", "svd_info"),
    "lbfgs": ("optimizer", "loss", "batch_size", "k_fraction", "k", "lr", "rtol",
              "model_seed", "loader_seed", "svd_mode", "svd_info", "lbfgs_max_iter",
              "lbfgs_history_size", "lbfgs_line_search_fn"),
    "polyak": ("optimizer", "loss", "batch_size", "k_fraction", "k", "lr", "rtol",
               "model_seed", "loader_seed", "svd_mode", "svd_info", "polyak_f_star",
               "polyak_max_lr", "polyak_eps"),
    "jd": ("optimizer", "loss", "batch_size", "lr", "aggregator", "inner_optimizer",
           "model_seed", "loader_seed", "svd_info"),
    "hig": ("optimizer", "loss", "batch_size", "lr", "tau", "model_seed",
            "loader_seed", "svd_info"),
}


def test_record_extra_matches_the_legacy_result_dicts():
    specs = specs_for("toy_1d_scan")
    seen = {}
    for s in specs:
        seen.setdefault(s.family, s)
    assert set(seen) == set(LEGACY_RECORD_KEYS)
    for family, s in seen.items():
        assert tuple(s.record_extra) == LEGACY_RECORD_KEYS[family], family


def test_svd_record_and_run_id_details():
    """The svd run_id / record pieces that are easy to break: k from k_values vs
    k_fractions, the use_gram svd_mode collapse and the optional suffixes."""
    # k_values are used verbatim; k_fractions are multiplied by the batch size.
    rcfg = load_rcfg("toy_1d_scan")
    s = grid.expand_grid(rcfg, mode="svd", verbose=False)[0]
    assert s.hparams["k"] == 1 and s.record_extra["k_fraction"] == 1 / 32
    assert s.run_id == ("svd_bs32_mlp_width16_k1_lr0.05_rtol0.0001_svdtorch"
                        "_mseed1000_lseed1000_gram")
    rcfg = load_rcfg("rebuttal_batchsize_polynomial_scan")
    ks = {(s.batch_size, s.hparams["k"])
          for s in grid.expand_grid(rcfg, mode="svd", verbose=False)}
    assert ks == {(b, b) for b in (8, 16, 32, 64, 128, 256)}   # k_fractions: [1.0]

    # use_gram makes svd_mode meaningless: a list of modes must not multiply the
    # grid, and the run_id must say "torch".
    rcfg = load_rcfg("toy_1d_scan", "svd_mode=[torch,randomized,randomized_v2]")
    specs = grid.expand_grid(rcfg, mode="svd", verbose=False)
    assert len(specs) == INVENTORY["toy_1d_scan"]["svd"]
    assert {s.hparams["svd_mode"] for s in specs} == {"torch"}
    assert all(s.record_extra["decomposition"] == "gram_eigh" for s in specs)

    # optional suffixes: _mb / _pf[_mask] / _gram / _bnbatch / _kappa / _loss{key}
    ids = {s.run_id for s in specs_for("toy_1d_microbatch_scan")}
    assert ("svd_bs32_mlp_width16_k32_lr0.05_rtol0.001_svdtorch"
            "_mseed1000_lseed1000_mb4_gram") in ids
    ids = {s.run_id for s in specs_for("mnist_paramfrac_labelreg_scan")}
    assert ("svd_bs64_mlp_width32_k64_lr0.05_rtol0.0001_svdtorch"
            "_mseed3000_lseed3000_pf0.25_elementwise_gram") in ids
    assert ("svd_bs64_mlp_width32_k64_lr0.05_rtol0.0001_svdtorch"
            "_mseed3000_lseed3000_pf1.0_gram") in ids        # pf == 1 -> no mask token
    ids = {s.run_id for s in specs_for("mnist_kappaScan_labelRegression")}
    assert sum(i.endswith("_kappa1") for i in ids) == 5       # kappa == 2 -> no token
    assert sum("_kappa" in i for i in ids) == 10
    rcfg = load_rcfg("cifar10_resnet_scan_labelRegression")   # gram_freeze_norm_stats: false
    assert all(s.run_id.endswith("_gram_bnbatch")
               for s in grid.expand_grid(rcfg, mode="svd", verbose=False))
    rcfg = load_rcfg("mnist_scan_ce", "loss=brier")           # non-legacy loss key
    assert all(s.run_id.endswith("_lossbrier")
               for s in grid.expand_grid(rcfg, mode="all", verbose=False))


def test_weight_decay_grid_filter():
    """Only AdamW / Muon / MuonW sweep weight decay; others keep their single wd=0
    run (generic_scan.py:614-623)."""
    rcfg = load_rcfg("rebuttal_batchsize_polynomial_scan")   # weight_decays: [0.0, 0.01]
    specs = grid.expand_grid(rcfg, mode="standard", verbose=False)
    per_optim = Counter(s.hparams["optim_name"] for s in specs if s.family == "standard")
    assert per_optim["Adam"] == 120 and per_optim["AdamW"] == 240
    assert per_optim["Muon"] == 240 and per_optim["MuonW"] == 240
    assert {s.hparams["weight_decay"] for s in specs
            if s.hparams.get("optim_name") == "SGD"} == {0.0}
    # weight_decays: [None] -> each optimizer's own default, in the run_id for the
    # two "W" optimizers only.
    rcfg = load_rcfg("toy_1d_scan")
    wd = {s.hparams["optim_name"]: s.hparams["weight_decay"]
          for s in grid.expand_grid(rcfg, mode="standard", verbose=False)
          if s.family == "standard"}
    assert wd == {"Adam": 0.0, "AdamW": 0.01, "SGD": 0.0, "RMSprop": 0.0,
                  "Muon": 0.0, "MuonW": 0.1, "SOAP": 0.0, "Shampoo": 0.0, "KFAC": 0.0}


ALL_SCAN_CONFIGS = sorted(p.stem for p in CONFIGS.glob("*.yaml"))


@pytest.mark.parametrize("name", ALL_SCAN_CONFIGS)
def test_every_config_in_the_repo_expands(name):
    """Every config key still works: each of the repo's scan configs enumerates
    with no exception, no duplicate run_id and no duplicate run_hash."""
    overrides = ("n_data=1000",) if ("overparam" in name or "finetune" in name) else ()
    rcfg = load_rcfg(name, *overrides)
    if rcfg.get("name") != "scan":
        pytest.skip(f"not a scan config (name={rcfg.get('name')!r})")
    specs = grid.expand_grid(rcfg, mode="all", verbose=False)
    assert specs
    assert len({s.run_id for s in specs}) == len(specs)
    assert len({grid.run_hash(s, rcfg) for s in specs}) == len(specs)


def test_mode_flags_select_families():
    rcfg = load_rcfg("toy_1d_scan")
    fams = {m: {s.family for s in grid.expand_grid(rcfg, mode=m, verbose=False)}
            for m in ("svd", "standard", "both", "jd", "hig", "all")}
    assert fams["svd"] == {"svd"}
    assert fams["standard"] == {"standard", "lbfgs", "polyak"}
    assert fams["both"] == fams["svd"] | fams["standard"]
    assert fams["jd"] == {"jd"} and fams["hig"] == {"hig"}
    assert fams["all"] == fams["both"] | {"jd", "hig"}
    # a missing torchjd drops the JD family from the enumeration entirely, as the
    # legacy loop did (its shard counter never advanced for JD runs)
    assert not [s for s in grid.expand_grid(rcfg, mode="all", has_torchjd=False,
                                            verbose=False) if s.family == "jd"]
    with pytest.raises(AssertionError):
        grid.expand_grid(rcfg, mode="nonsense", verbose=False)


def test_run_hash():
    rcfg = load_rcfg("toy_1d_scan")
    specs = grid.expand_grid(rcfg, mode="all", verbose=False)
    hashes = {s.run_id: grid.run_hash(s, rcfg) for s in specs}
    assert all(len(h) == 64 and all(c in "0123456789abcdef" for c in h)
               for h in hashes.values())
    assert len(set(hashes.values())) == len(specs)          # one per grid point
    assert grid.run_hash(specs[0], rcfg) == hashes[specs[0].run_id]   # deterministic
    assert grid.hash8(specs[0], rcfg) == hashes[specs[0].run_id][:8]

    # every identity input moves the hash
    for override in ("num_epochs=21", "loss=brier", "dataset.n_train=5000",
                     "model.activation=relu", "data_seed=7", "eval_batch_size=128"):
        other = load_rcfg("toy_1d_scan", override)
        s2 = grid.expand_grid(other, mode="all", verbose=False)[0]
        assert grid.run_hash(s2, other) != grid.run_hash(specs[0], rcfg), override
    # ... while a pure diagnostics knob does not
    quiet = load_rcfg("toy_1d_scan", "svd_spectra_every=1")
    s3 = grid.expand_grid(quiet, mode="all", verbose=False)[0]
    assert grid.run_hash(s3, quiet) == grid.run_hash(specs[0], rcfg)


def test_runspec_is_compared_and_keyed_by_run_id_not_hashed():
    """RunSpec holds dicts, so it is deliberately unhashable: a consumer keying on
    a spec must use run_id (or run_hash), and gets a clear error if it does not."""
    specs = grid.expand_grid(load_rcfg("toy_1d_scan"), mode="svd", verbose=False)
    assert specs[0] == specs[0] and specs[0] != specs[1]        # eq still works
    with pytest.raises(TypeError, match="unhashable type: 'RunSpec'"):
        {specs[0]}
    assert len({s.run_id: s for s in specs}) == len(specs)      # the intended key


def test_expand_grid_is_silent_when_not_verbose(capsys):
    """A torch-free consumer enumerating many scans must not be buried in the
    copied process_hparam_config's unconditional "defaulting to" notes."""
    rcfg = load_rcfg("mnist_kappaScan_labelRegression")   # no lrs_standard etc. -> notes
    grid.expand_grid(rcfg, mode="all", verbose=False)
    assert capsys.readouterr().out == ""
    grid.expand_grid(rcfg, mode="all", verbose=True)
    assert "defaulting to" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The runner half of C-R2. `execute()` collapsed six near-identical blocks into
# one, and the per-family asymmetries inside it are the easiest thing to lose in
# that collapse (they are silent: a wrong wrapper or a missing drop_last still
# produces a record). This file must stay torch-free, so they are pinned
# structurally, from the AST; `tests/golden/scan_smoke.sh` runs the real thing
# (1 epoch of toy_1d_scan, all six families, second pass skips, shard 1/2).
# ---------------------------------------------------------------------------

def _scan_function(name):
    tree = ast.parse(SCAN_PATH.read_text())
    return tree, next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == name)


def _scan_constant(name):
    tree, _ = _scan_function("execute")
    return ast.literal_eval(next(
        n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == name for t in n.targets)))


def _family_of(test):
    """``"svd"`` for ``spec.family == "svd"``, else None."""
    if (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and ast.unparse(test.left) == "spec.family"
            and isinstance(test.comparators[0], ast.Constant)):
        return test.comparators[0].value
    return None


def _family_dispatch():
    """``({family: branch source}, else-source)`` of ``execute``'s if/elif chain."""
    _, fn = _scan_function("execute")
    node = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and _family_of(n.test))
    branches, tail = {}, ""
    while node is not None:
        branches[_family_of(node.test)] = "\n".join(ast.unparse(s) for s in node.body)
        nxt = node.orelse[0] if len(node.orelse) == 1 else None
        if isinstance(nxt, ast.If) and _family_of(nxt.test):
            node = nxt
        else:
            tail = "\n".join(ast.unparse(s) for s in node.orelse)
            node = None
    return branches, tail


def _branch(branches, family):
    assert family in branches, (
        f"execute() has no {family!r} branch -- see the dispatch test")
    return branches[family]


LOOP_OF_FAMILY = {"svd": "train_loop_svd", "standard": "train_loop_standard",
                  "lbfgs": "train_loop_standard", "polyak": "train_loop_standard",
                  "jd": "train_loop_jd", "hig": "train_loop_hig"}


def test_execute_dispatches_every_family_to_its_own_loop():
    branches, tail = _family_dispatch()
    assert set(branches) == set(grid.FAMILIES)          # no family runs as another
    assert "raise AssertionError" in tail              # ... and none falls through
    for family, loop in LOOP_OF_FAMILY.items():
        assert f"{loop}(" in _branch(branches, family), family
    svd = _branch(branches, "svd")
    assert "SvenWrapper(" in svd and "GramSvenWrapper(" in svd
    assert "HIGWrapper(" in _branch(branches, "hig")


def test_execute_keeps_the_per_family_asymmetries():
    """The legacy blocks were not uniform; the single copy must keep the differences
    (legacy generic_scan.py: .to(device) at 638/710/786/863 only, track_param_norm
    at 548/653/874/927 only, drop_last at 540 only, empty_cache at 679-947 only)."""
    branches, _ = _family_dispatch()
    # the svd and hig wrappers move the model themselves
    assert set(_scan_constant("_TO_DEVICE_FAMILIES")) == set(grid.FAMILIES) - {"svd", "hig"}
    # LBFGS / PolyakSGD deliberately do not track the parameter norm
    for family in ("svd", "standard", "jd", "hig"):
        assert "track_param_norm=" in _branch(branches, family), family
    for family in ("lbfgs", "polyak"):
        assert "track_param_norm=" not in _branch(branches, family), family
    # only the svd train loader drops the last, short batch (microbatching)
    assert "drop_last=" in _branch(branches, "svd")
    for family in set(grid.FAMILIES) - {"svd"}:
        assert "drop_last" not in _branch(branches, family), family
    # svd-only diagnostics + compile cache; empty_cache on the error path for the rest
    _, fn = _scan_function("execute")
    node = next(n for n in ast.walk(fn) if isinstance(n, ast.Try))
    final, handlers = (ast.unparse(node.finalbody[0]),
                       "\n".join(ast.unparse(h) for h in node.handlers))
    assert "torch.compiler.reset()" in final and "spec.family == 'svd'" in final
    assert "empty_cache" in handlers and "spec.family != 'svd'" in handlers
    assert "torch.compiler.reset" not in handlers
    after = "\n".join(ast.unparse(s) for s in node.body)
    assert "if spec.family == 'svd':\n    result['svd_info']" in after


def test_scan_is_expand_shard_dedup_execute_with_a_per_seed_init_state():
    """scan() is a driver only: the grid comes from grid.py, and the per-seed base
    model is built ONCE per model seed (the legacy per-seed preamble, which is what
    reproduces the legacy RNG stream: set_seed -> instantiate -> deepcopy)."""
    _, fn = _scan_function("scan")
    called = {ast.unparse(n.func).split(".")[-1] for n in ast.walk(fn)
              if isinstance(n, ast.Call)}
    assert {"expand_grid", "shard", "execute", "_ScanContext"} <= called
    assert not [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                and ast.unparse(n.func).endswith("product")]     # no inline grid left
    _, seed_state = _scan_function("seed_state")
    src = "\n".join(ast.unparse(s) for s in seed_state.body)
    assert "if self._seed != model_seed:" in src
    assert "set_seed(model_seed)" in src and "instantiate(self.cfg.model)" in src
    assert "copy.deepcopy(base_model.state_dict())" in src


# grid.py must stay torch-free, so it carries COPIES of these pure helpers from
# the runner-side modules (owned by other tracks; the originals stay in place and
# every other call site still uses them). If an original changes, the copy has to
# change with it or the enumerated grid silently stops matching the runner.
# Where the original lives is deliberately not pinned: C-B2/C-B5 move the weight
# decay helpers from experiment_utils.py into optim_factory.py (which experiment_utils
# then re-exports), so every definition found in either file must match the copy.
COPIED_HELPERS = ("listify", "resolve_weight_decay", "process_hparam_config")
ORIGIN_PATHS = (UTILS_PATH, FACTORY_PATH)


def _normalised_function(path, name):
    """A function's signature + body as AST dumps, ignoring its docstring.

    ``None`` if ``path`` does not define ``name`` (it may have moved or be a
    re-export).
    """
    tree = ast.parse(path.read_text())
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if fn is None:
        return None
    body = fn.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return ast.dump(fn.args), [ast.dump(n) for n in body]


@pytest.mark.parametrize("name", COPIED_HELPERS)
def test_copied_helpers_match_the_original(name):
    copy = _normalised_function(GRID_PATH, name)
    assert copy is not None, f"grid.py lost its copy of {name}"
    originals = {p.name: _normalised_function(p, name) for p in ORIGIN_PATHS}
    found = {k: v for k, v in originals.items() if v is not None}
    assert found, f"{name} defined in none of {[p.name for p in ORIGIN_PATHS]}"
    for where, original in found.items():
        assert copy == original, f"grid.py's copy of {name} drifted from {where}"


def test_copied_default_weight_decay_matches_the_original():
    def dump(path):
        tree = ast.parse(path.read_text())
        return [ast.dump(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "_DEFAULT_WEIGHT_DECAY" for t in n.targets)]
    copy = dump(GRID_PATH)
    assert copy != []
    found = {p.name: dump(p) for p in ORIGIN_PATHS if dump(p) != []}
    assert found, "_DEFAULT_WEIGHT_DECAY defined in neither origin module"
    for where, original in found.items():
        assert copy == original, f"_DEFAULT_WEIGHT_DECAY drifted from {where}"


def test_wd_optimizers_filter_tracks_the_default_weight_decay_registry():
    """``grid._WD_OPTIMIZERS`` is the run-id-shaping filter: any other optimizer
    with a non-zero weight decay is dropped from the grid (legacy
    generic_scan.py:614-623). It must stay the two "W" optimizers that have their
    own default wd, plus plain Muon (whose wd is swept from the config) -- so a new
    default-wd baseline from C-B2/C-B5 cannot be silently excluded from the grid.
    """
    assert grid._WD_OPTIMIZERS == ("AdamW", "Muon", "MuonW")
    assert set(grid._WD_OPTIMIZERS) == set(grid._DEFAULT_WEIGHT_DECAY) | {"Muon"}


def test_grid_module_imports_without_torch_hydra_or_sven():
    """grid.py must be importable (fast) in a process that has no torch."""
    code = textwrap.dedent(f"""
        import importlib.util, sys, time
        spec = importlib.util.spec_from_file_location("g", r"{GRID_PATH}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["g"] = mod
        t0 = time.perf_counter()
        spec.loader.exec_module(mod)
        dt = time.perf_counter() - t0
        heavy = sorted({{m.split(".")[0] for m in sys.modules}}
                       & {{"torch", "hydra", "sven", "omegaconf", "numpy", "pandas"}})
        print(heavy, dt)
        assert not heavy, heavy
        assert dt < 1.0, dt
        assert mod.expand_grid and mod.RunSpec and mod.run_hash
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
