#!/usr/bin/env python3
"""Gate 1: check the smoke matrix's results root against CHANGES_NEEDED.md 4.2 / 4.5.

    campaign/run_cpu_tests.sh .venv/bin/python campaign/verify_smoke.py \
        --plan campaign/plan_smoke.yaml --snapshot <smoke snapshot>

Torch IS used (checks (d) and (f) reload checkpoints and evaluate), so run it through
`campaign/run_cpu_tests.sh`, not on the shared dev node.

The checks are lettered as in the Gate-1 task and each one prints PASS/FAIL with the
numbers it measured. Nothing here is allowed to pass by being lenient: a check that
cannot find its evidence FAILS rather than skipping.

  (a) every expected run has a schema-2 record, status ok or diverged (the deliberate
      lr-1e3 / f_star=-1e5 points), diverged runs keep partial curves and a
      diverged_at_step, and nothing is left in started/ or claims/
  (b) val / test / train_eval curves present and finite for ok runs, index 0 =
      untrained, test accuracy for classification, and the schema-2 identity and
      provenance columns on EVERY record
  (c) Sven diag npz: svs full width (B, or B/microbatch), utr the same shape, svs_step
      follows the configured schedule, sv_min_kept / update_norm / resid_norm present,
      no legacy sv_min
  (d) checkpoints reload and reproduce the recorded final val AND test loss
      (MLP 1e-5, CIFAR 1e-4), and a `log` policy file holds {0,1,2,4,...} + epoch ends
  (e) one model seed -> one effective_loader_seed across optimizers; two seeds -> two
  (f) CIFAR Sven step time, peak memory, and num_batches_tracked == optimizer steps in
      the final checkpoint of EVERY CIFAR optimizer
  (j) two concurrent jobs on one scan ran each run exactly once: one record per run, no
      `.1` duplicates under _stale/, attempts/ empty for ok runs

(g) reconcile, (h) resubmission skips, (i) a changed num_epochs retires the old
generation are separate, they need a launcher run rather than a look at the tree, and
are driven from outside this script.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS = []                        # (letter, name, ok, evidence)


def check(letter, name, ok, evidence):
    RESULTS.append((letter, name, bool(ok), str(evidence)))
    print(f"  [{'PASS' if ok else 'FAIL'}] ({letter}) {name}: {evidence}", flush=True)
    return bool(ok)


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


# ---------------------------------------------------------------------------
# Reading the tree
# ---------------------------------------------------------------------------

def scan_dirs(root):
    return sorted(d for d in glob.glob(os.path.join(root, "*"))
                  if os.path.isdir(d) and glob.glob(os.path.join(d, "*.jsonl")))


def read_records(scan_dir):
    """`{run_id: record}` for every jsonl in one scan (last line wins)."""
    out = {}
    for path in glob.glob(os.path.join(scan_dir, "*.jsonl")):
        lines = [l for l in open(path).read().splitlines() if l.strip()]
        if not lines:
            continue
        rec = json.loads(lines[-1])
        out[rec.get("run_id") or os.path.basename(path)[:-len(".jsonl")]] = rec
        rec["_n_lines"] = len(lines)
        rec["_scan_dir"] = scan_dir
    return out


def markers(scan_dir):
    """`{run_id: {(hash8, status)}}` from done/."""
    done_dir = os.path.join(scan_dir, "done")
    out = {}
    for name in (os.listdir(done_dir) if os.path.isdir(done_dir) else []):
        parts = name.rsplit(".", 2)
        if len(parts) == 3:
            out.setdefault(parts[0], set()).add((parts[1], parts[2]))
    return out


def load_plan(plan_path):
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import campaign_plan
    return campaign_plan.load_plan(plan_path)


def expected(plan, snapshot, root):
    """`{scan: {run_id: spec-ish}}` -- the plan's own grid, composed from the snapshot.

    Same machinery as `tools/reconcile.py`, so "expected" here means exactly what the
    launcher submitted, and a run_id nobody asked for is as much a failure as a missing
    one.
    """
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import reconcile
    grid = reconcile.load_grid()
    jd = reconcile.has_torchjd()
    config_dir = os.path.join(snapshot, "experiments", "configs")
    out = {}
    with reconcile.ConfigLoader(config_dir) as loader:
        for scan, groups in plan.scan_overrides().items():
            runs = {}
            # the dataset facts `run_grid` injects into `cfg.model` before hashing, so a
            # language-model run's hash8 is reproducible here (`{}` for every other scan)
            facts = reconcile.injected_dataset_facts(os.path.join(root, scan))
            for overrides in groups:
                rcfg = loader.compose(scan, "++scheduler=claims " + overrides)
                if facts and (rcfg.get("model") or {}).get("vocab_size") is None:
                    rcfg = dict(rcfg)
                    rcfg["model"] = grid.inject_dataset_facts(rcfg["model"], **facts)
                for spec in grid.expand_grid(rcfg, verbose=False, has_torchjd=jd):
                    runs[spec.run_id] = {"family": spec.family, "hparams": spec.hparams,
                                         "batch_size": spec.batch_size,
                                         "model_seed": spec.model_seed,
                                         "hash8": grid.hash8(spec, rcfg),
                                         "num_epochs": rcfg["num_epochs"],
                                         "rcfg": rcfg}
            out[scan] = runs
    return out


#: the grid points the plan makes diverge on purpose (plan_smoke.yaml's comments)
def is_deliberate(info):
    """A grid point plan_smoke.yaml chose to blow up: lr 1e3, or Polyak's f_star=-1e5."""
    hp = info["hparams"]
    lr = hp.get("lr")
    if lr is not None and float(lr) >= 100.0:
        return True
    if hp.get("f_star") is not None and float(hp["f_star"]) < 0:
        return True
    return False


# ---------------------------------------------------------------------------
# (a) every expected run has a record; nothing left claimed or started
# ---------------------------------------------------------------------------

def check_a(exp, root):
    missing, bad_status, no_partial, leftovers, unexpected = [], [], [], [], []
    n_ok = n_div = 0
    for scan, runs in exp.items():
        scan_dir = os.path.join(root, scan)
        recs = read_records(scan_dir) if os.path.isdir(scan_dir) else {}
        marks = markers(scan_dir) if os.path.isdir(scan_dir) else {}
        for run_id, info in runs.items():
            rec = recs.get(run_id)
            if rec is None:
                missing.append(f"{scan}/{run_id}")
                continue
            status = rec.get("status")
            if rec.get("schema_version") != 2:
                bad_status.append(f"{scan}/{run_id} schema_version={rec.get('schema_version')}")
            allowed = {"ok", "diverged"} if is_deliberate(info) else {"ok"}
            if status not in allowed:
                bad_status.append(f"{scan}/{run_id} status={status} "
                                  f"error={(rec.get('error') or {}).get('message', '')[:120]}")
            n_ok += status == "ok"
            n_div += status == "diverged"
            if status == "diverged":
                curves = rec.get("losses") or {}
                if not (curves.get("train") or curves.get("val")):
                    no_partial.append(f"{scan}/{run_id}: no partial curve")
                # "diverged_at_step where known": only the non-finite-loss early stop
                # (DivergedError) knows the step. A linear-algebra failure inside an
                # optimizer is also `diverged` (CONTRACTS / _classify_failure) and has
                # no step, so requiring one there would be requiring a wrong answer.
                err_type = (rec.get("error") or {}).get("type")
                if rec.get("diverged_at_step") is None and err_type == "DivergedError":
                    no_partial.append(f"{scan}/{run_id}: DivergedError without a "
                                      f"diverged_at_step")
                if rec.get("diverged_at_step") is None:
                    print(f"         note: {scan}/{run_id} diverged via {err_type} "
                          f"(no step is knowable): "
                          f"{(rec.get('error') or {}).get('message', '')[:90]}")
            if (info["hash8"], status) not in marks.get(run_id, set()):
                bad_status.append(f"{scan}/{run_id}: no done marker "
                                  f"{info['hash8']}.{status} (have {marks.get(run_id)})")
        for run_id in set(recs) - set(runs):
            unexpected.append(f"{scan}/{run_id}")
        for sub in ("started", "claims"):
            left = [f for f in os.listdir(os.path.join(scan_dir, sub))
                    if not f.startswith(".")] if os.path.isdir(
                        os.path.join(scan_dir, sub)) else []
            leftovers += [f"{scan}/{sub}/{f}" for f in left]
    n_exp = sum(len(r) for r in exp.values())
    ok = not (missing or bad_status or no_partial or leftovers or unexpected)
    check("a", "every expected run has a schema-2 record with the right status",
          ok, f"{n_exp} expected, {n_ok} ok + {n_div} diverged = {n_ok + n_div} "
              f"records; missing {len(missing)}, wrong status/marker {len(bad_status)}, "
              f"no partial curve {len(no_partial)}, left in started+claims "
              f"{len(leftovers)}, unexpected {len(unexpected)}")
    for label, items in (("missing", missing), ("status/marker", bad_status),
                         ("partial", no_partial), ("leftover", leftovers),
                         ("unexpected", unexpected)):
        for item in items[:12]:
            print(f"         {label}: {item}")
        if len(items) > 12:
            print(f"         {label}: ... and {len(items) - 12} more")
    return ok


# ---------------------------------------------------------------------------
# (b) curves and the schema-2 columns
# ---------------------------------------------------------------------------

_IDENTITY_KEYS = ("n_train", "n_val", "n_test", "steps_per_epoch",
                  "effective_loader_seed", "run_hash", "n_params")
_PROV_KEYS = ("git_sha", "sven_git_sha", "git_dirty", "gpu_name", "slurm_job_id",
              "host", "torch_version")


def check_b(exp, root, snapshot):
    info_path = os.path.join(snapshot, "DEPLOY_INFO.json")
    want = json.load(open(info_path))
    want_sv3, want_sven = want["git_sha"], want["sven_git_sha"]
    problems, n_ok, n_class = [], 0, 0
    for scan, runs in exp.items():
        scan_dir = os.path.join(root, scan)
        if not os.path.isdir(scan_dir):
            continue
        for run_id, rec in read_records(scan_dir).items():
            for key in _IDENTITY_KEYS + _PROV_KEYS:
                if rec.get(key) is None:
                    problems.append(f"{scan}/{run_id}: {key} is None")
            # `split_seed` is the seed that decided the held-out examples; CONTRACTS.md
            # states it is None for the POSITIONALLY split text corpora (Shakespeare
            # 80/10/10 by position has no seed) and a number everywhere else. A None
            # anywhere else is a missing column.
            if rec.get("split_seed") is None and rec.get("loss") != "lm_ce":
                problems.append(f"{scan}/{run_id}: split_seed is None on a "
                                f"non-positional split (loss={rec.get('loss')})")
            if rec.get("git_sha") != want_sv3 or rec.get("sven_git_sha") != want_sven:
                problems.append(f"{scan}/{run_id}: sha {str(rec.get('git_sha'))[:8]}/"
                                f"{str(rec.get('sven_git_sha'))[:8]} != the snapshot's")
            if rec.get("git_dirty") is not False or rec.get("sven_git_dirty") is not False:
                problems.append(f"{scan}/{run_id}: git_dirty="
                                f"{rec.get('git_dirty')}/{rec.get('sven_git_dirty')}")
            if rec.get("status") != "ok":
                continue
            n_ok += 1
            curves = rec.get("losses") or {}
            n_epochs = rec.get("num_epochs")
            for key in ("val", "test", "train_eval"):
                curve = curves.get(key)
                if not curve:
                    problems.append(f"{scan}/{run_id}: no {key} curve")
                    continue
                # index 0 = untrained, then one point per epoch
                if len(curve) != n_epochs + 1:
                    problems.append(f"{scan}/{run_id}: len({key})={len(curve)}, "
                                    f"expected num_epochs+1={n_epochs + 1}")
                if not all(finite(v) for v in curve):
                    problems.append(f"{scan}/{run_id}: non-finite in {key}: {curve}")
            if not finite(rec.get("test")) or not finite(rec.get("val_final")):
                problems.append(f"{scan}/{run_id}: test/val_final not finite")
            # the untrained point must be the SAME for every optimizer of a seed only
            # if the loss is; what is checked here is that it is not a trained value
            if rec.get("loss") in ("ce", "brier", "label_regression"):
                n_class += 1
                if not finite(rec.get("test_acc")):
                    problems.append(f"{scan}/{run_id}: classification run without "
                                    f"test_acc ({rec.get('test_acc')})")
                if not curves.get("val_acc") or not curves.get("test_acc"):
                    problems.append(f"{scan}/{run_id}: no val_acc/test_acc curve")
    ok = not problems
    check("b", "three-split curves finite, index 0 untrained, identity+provenance",
          ok, f"{n_ok} ok records checked ({n_class} of them classification, i.e. "
              f"test_acc required); {len(problems)} problem(s); snapshot sha "
              f"{want_sv3[:8]}/{want_sven[:8]}, git_dirty false")
    for p in problems[:15]:
        print(f"         {p}")
    if len(problems) > 15:
        print(f"         ... and {len(problems) - 15} more")
    return ok


# ---------------------------------------------------------------------------
# (c) the Sven diagnostics
# ---------------------------------------------------------------------------

def check_c(exp, root):
    import numpy as np
    problems, n_checked = [], 0
    widths = {}
    for scan, runs in exp.items():
        scan_dir = os.path.join(root, scan)
        if not os.path.isdir(scan_dir):
            continue
        recs = read_records(scan_dir)
        for run_id, info in runs.items():
            if info["family"] != "svd":
                continue
            rec = recs.get(run_id)
            if rec is None or rec.get("status") != "ok":
                continue
            path = os.path.join(scan_dir, rec.get("diag_file") or "")
            if not (rec.get("diag_file") and os.path.exists(path)):
                problems.append(f"{scan}/{run_id}: no diag npz")
                continue
            z = np.load(path)
            n_checked += 1
            if "sv_min" in z.files:
                problems.append(f"{scan}/{run_id}: legacy sv_min is present")
            for key in ("svs", "utr", "svs_step", "sv_min_kept", "update_norm",
                        "resid_norm", "sv_noise_floor"):
                if key not in z.files:
                    problems.append(f"{scan}/{run_id}: diag has no {key}")
            if "svs" not in z.files or "utr" not in z.files:
                continue
            mb = info["hparams"].get("microbatch_size") or 1
            want_m = info["batch_size"] // mb
            svs, utr, steps = z["svs"], z["utr"], z["svs_step"]
            widths[f"{scan}/{run_id}"] = (svs.shape, want_m)
            if svs.ndim != 2 or svs.shape[1] != want_m:
                problems.append(f"{scan}/{run_id}: svs {svs.shape}, want width "
                                f"B/microbatch = {info['batch_size']}/{mb} = {want_m}")
            # "full width on EVERY logged step": no row may be short (pad = NaN)
            n_short = int(np.sum(~np.isfinite(svs)))
            if n_short:
                problems.append(f"{scan}/{run_id}: {n_short} NaN-padded svs entries, "
                                f"i.e. a logged step with fewer than {want_m} values")
            if utr.shape != svs.shape:
                problems.append(f"{scan}/{run_id}: utr {utr.shape} != svs {svs.shape}")
            for key in ("sv_min_kept", "update_norm", "resid_norm"):
                if key in z.files and z[key].shape[0] != steps.shape[0]:
                    problems.append(f"{scan}/{run_id}: {key} {z[key].shape} does not "
                                    f"follow svs_step {steps.shape}")
            # the schedule: every one of the first `dense_first` steps, then every `every`
            sched = rec.get("svd_spectra_schedule") or {}
            dense, every = int(sched.get("dense_first", 0)), int(sched.get("every", 1))
            total = len(z["train_batch"]) if "train_batch" in z.files else None
            if total is not None:
                want = [s for s in range(total) if s < dense or s % every == 0]
                if list(steps) != want:
                    problems.append(
                        f"{scan}/{run_id}: svs_step does not follow "
                        f"{{dense_first:{dense}, every:{every}}} over {total} steps "
                        f"(got {len(steps)}, want {len(want)}; "
                        f"first mismatch at {next((i for i, (a, b) in enumerate(zip(steps, want)) if a != b), None)})")
    ok = not problems
    shapes = sorted({f"{v[0][1]}(=B/mb {v[1]})" for v in widths.values()})
    check("c", "Sven spectra full width on every logged step, utr matched, schedule",
          ok, f"{n_checked} svd npz checked, widths {shapes}; {len(problems)} problem(s)")
    for p in problems[:15]:
        print(f"         {p}")
    return ok


# ---------------------------------------------------------------------------
# (d) checkpoints reload and reproduce the recorded losses
# ---------------------------------------------------------------------------

def _load_ctx(scan_dir, rcfg_path):
    """A `_ScanContext` for one scan, from the resolved config the RUNNER saved."""
    from omegaconf import OmegaConf
    from experiments.experiment_code import generic_scan as gs
    from experiments.experiment_code import grid as gridmod
    from hydra.utils import instantiate
    cfg = OmegaConf.load(rcfg_path)
    cfg.device = "cpu"
    rcfg = OmegaConf.to_container(cfg, resolve=True)
    dataset = instantiate(cfg.dataset)
    ctx = gs._ScanContext(cfg, rcfg, dataset, scan_dir,
                          gridmod.resolve_scan_settings(rcfg),
                          gridmod.resolve_svd_settings(rcfg))
    return cfg, rcfg, ctx


def check_d(exp, root, n_mlp=3, seed=20260918):
    import random
    import torch
    from hydra.utils import instantiate
    from experiments.experiment_code.checkpointing import load_checkpoint, load_state_at
    from experiments.experiment_code.experiment_utils import evaluate

    rng = random.Random(seed)
    # pick runs: n_mlp random MLP runs with a checkpoint, plus one CIFAR run.
    # Only CONVERGED runs are eligible: the deliberately divergent grid points that
    # stayed finite end at a val loss of 1e21-1e27, where float32 spacing is 1e14 and
    # "reproduces the recorded loss to 1e-5" is not a statement about the checkpoint.
    pool_mlp, pool_cifar = [], []
    for scan, runs in exp.items():
        scan_dir = os.path.join(root, scan)
        if not os.path.isdir(scan_dir):
            continue
        for run_id, rec in read_records(scan_dir).items():
            if rec.get("status") != "ok" or not rec.get("ckpt_file"):
                continue
            if not os.path.exists(os.path.join(scan_dir, rec["ckpt_file"])):
                continue
            if not finite(rec.get("val_final")) or abs(rec["val_final"]) > 1e6:
                continue
            (pool_cifar if "cifar" in scan else pool_mlp).append((scan, run_id, rec))
    picked = rng.sample(pool_mlp, min(n_mlp, len(pool_mlp)))
    if pool_cifar:
        picked.append(rng.choice(pool_cifar))
    if len(picked) < n_mlp + 1:
        return check("d", "checkpoints reload and reproduce val+test", False,
                     f"only {len(pool_mlp)} MLP and {len(pool_cifar)} CIFAR "
                     f"checkpoints available, wanted {n_mlp}+1")

    problems, evidence = [], []
    ctx_cache = {}
    for scan, run_id, rec in picked:
        scan_dir = os.path.join(root, scan)
        if scan not in ctx_cache:
            cfgs = sorted(glob.glob(os.path.join(scan_dir, "configs", "*.yaml")))
            if not cfgs:
                problems.append(f"{scan}: no saved resolved config")
                continue
            ctx_cache[scan] = cfgs
        # The saved config of the JOB that produced this run. Matching on the seed alone
        # is not enough: `rebuttal_overparam_mnist_scan` is one scan directory with one
        # job per n_data, i.e. one DATASET per job, so the wrong config would evaluate
        # the right weights on the wrong validation set. `build_id_string` is exactly
        # the part of the run_id that carries `result_id_fields`.
        from experiments.experiment_code.grid import build_id_string
        from omegaconf import OmegaConf as _OC
        cfg_path = None
        for cand in ctx_cache[scan]:
            saved = _OC.to_container(_OC.load(cand), resolve=True)
            # "" when the config has no result_id_fields (CIFAR, nanoGPT), else e.g.
            # "_mlp_width32_n_data2500" -- which is in the run_id verbatim
            if build_id_string(saved) not in run_id:
                continue
            if int(saved.get("num_epochs", -1)) != int(rec["num_epochs"]):
                continue
            cfg_path = cand
            break
        if cfg_path is None:
            problems.append(f"{scan}/{run_id}: no saved config matches "
                            f"(num_epochs={rec['num_epochs']})")
            continue
        cfg, rcfg, ctx = _load_ctx(scan_dir, cfg_path)
        atol = 1e-4 if "cifar" in scan else 1e-5
        # plus float32 round-off at the loss's own magnitude: the absolute tolerance is
        # the statement, the relative term only keeps it meaningful for a loss far from 1
        rtol = 1e-6

        model = instantiate(cfg.model)
        state = load_state_at(os.path.join(scan_dir, rec["ckpt_file"]))
        missing = model.load_state_dict(state, strict=True)
        model = model.to("cpu").eval()

        val_loader = ctx._eval_loader(getattr(ctx.dataset, "val_dataset", None))
        test_loader = ctx._eval_loader(getattr(ctx.dataset, "test_dataset", None))
        got_val = evaluate(model, ctx.loss_fn_svd, val_loader, "cpu",
                           track_acc=ctx.track_acc, is_lm=ctx.is_lm)
        got_test = evaluate(model, ctx.loss_fn_svd, test_loader, "cpu",
                            track_acc=ctx.track_acc, is_lm=ctx.is_lm)
        want_val = float((rec["losses"]["val"])[-1])
        want_test = float(rec["test"])
        d_val = abs(got_val["loss"] - want_val)
        d_test = abs(got_test["loss"] - want_test)
        # the RELATIVE error too: a run that ends at a loss of 4e4 cannot be reproduced
        # to an absolute 1e-5 in float32 (the spacing there is ~4e-3), so an absolute
        # number alone would look like a fudge when it is round-off
        evidence.append(f"{scan}/{run_id[:30]} dval={d_val:.2e} dtest={d_test:.2e} "
                        f"(rel {d_val / max(abs(want_val), 1e-30):.1e}/"
                        f"{d_test / max(abs(want_test), 1e-30):.1e}; atol {atol:g}, "
                        f"loss {want_val:.4g})")
        if d_val > atol + rtol * abs(want_val) or d_test > atol + rtol * abs(want_test):
            problems.append(f"{scan}/{run_id}: |val|={d_val:.3e} |test|={d_test:.3e} "
                            f"> {atol:g} (recorded {want_val:.6g}/{want_test:.6g}, "
                            f"reloaded {got_val['loss']:.6g}/{got_test['loss']:.6g})")

        # the log policy's step ladder, on the runs that have it
        if rec.get("checkpoint_policy") == "log":
            ck = load_checkpoint(os.path.join(scan_dir, rec["ckpt_file"]))
            steps, spe = list(ck["step"]), int(rec["steps_per_epoch"])
            total = spe * int(rec["num_epochs"])
            want = sorted({0} | {2 ** i for i in range(64) if 2 ** i <= total}
                          | {spe * (e + 1) for e in range(int(rec["num_epochs"]))})
            if steps != want:
                problems.append(f"{scan}/{run_id}: log ladder {steps} != {want}")
            else:
                evidence.append(f"{run_id[:20]} log ladder {len(steps)} states OK")
    ok = not problems
    check("d", "checkpoints reload and reproduce the recorded val AND test loss",
          ok, "; ".join(evidence[:6]) + (f"; {len(problems)} problem(s)" if problems else ""))
    for p in problems[:8]:
        print(f"         {p}")
    return ok


# ---------------------------------------------------------------------------
# (e) the effective loader seed
# ---------------------------------------------------------------------------

def check_e(exp, root):
    per_seed = {}                    # (scan, model_seed) -> {effective seed: [run_ids]}
    for scan, runs in exp.items():
        scan_dir = os.path.join(root, scan)
        if not os.path.isdir(scan_dir):
            continue
        for run_id, rec in read_records(scan_dir).items():
            eff = rec.get("effective_loader_seed")
            if eff is None:
                continue
            per_seed.setdefault((scan, rec["model_seed"]), {}).setdefault(
                int(eff), []).append((rec.get("optimizer"), run_id))
    problems = []
    for (scan, seed), by_eff in sorted(per_seed.items()):
        if len(by_eff) != 1:
            problems.append(f"{scan} mseed{seed}: {len(by_eff)} different effective "
                            f"loader seeds {sorted(by_eff)}")
    # two seeds in one scan must differ
    multi = {}
    for (scan, seed), by_eff in per_seed.items():
        multi.setdefault(scan, {})[seed] = sorted(by_eff)[0]
    pairs = {s: v for s, v in multi.items() if len(v) > 1}
    if not pairs:
        problems.append("no scan in this matrix ran two model seeds, so the "
                        "'different across seeds' half of (e) is unproven")
    for scan, by_seed in pairs.items():
        if len(set(by_seed.values())) != len(by_seed):
            problems.append(f"{scan}: two model seeds share an effective loader seed "
                            f"{by_seed}")
    n_opt = max((len({o for o, _ in v}) for by in per_seed.values()
                 for v in by.values()), default=0)
    ok = not problems
    check("e", "one model seed = one effective loader seed across optimizers",
          ok, f"{len(per_seed)} (scan, model_seed) group(s), up to {n_opt} optimizers "
              f"sharing one seed; {len(pairs)} scan(s) with 2 model seeds: "
              f"{ {s: v for s, v in pairs.items()} }")
    for p in problems[:8]:
        print(f"         {p}")
    return ok


# ---------------------------------------------------------------------------
# (f) CIFAR: step time, memory, BatchNorm buffers
# ---------------------------------------------------------------------------

def check_f(exp, root):
    import torch
    from experiments.experiment_code.checkpointing import load_state_at

    problems, times, evidence = [], {}, []
    bn_report = []
    for scan, runs in exp.items():
        if "cifar" not in scan:
            continue
        scan_dir = os.path.join(root, scan)
        if not os.path.isdir(scan_dir):
            continue
        for run_id, rec in read_records(scan_dir).items():
            if rec.get("status") != "ok":
                continue
            opt = rec.get("optimizer")
            spe, ne = rec.get("steps_per_epoch"), rec.get("num_epochs")
            steps = None if spe is None else int(spe) * int(ne)
            # step time: the loops' own summed, synchronised batch time
            curves = rec.get("losses") or {}
            tt = curves.get("train_times")
            if tt and steps:
                per_step = sum(float(t) for t in tt) / steps
                times.setdefault(f"{scan}:{opt}", []).append(per_step)
            # BatchNorm: num_batches_tracked == optimizer steps, for EVERY optimizer
            ck = rec.get("ckpt_file")
            if not (ck and os.path.exists(os.path.join(scan_dir, ck))):
                problems.append(f"{scan}/{run_id}: no checkpoint, cannot check BN")
                continue
            state = load_state_at(os.path.join(scan_dir, ck))
            tracked = {int(v) for k, v in state.items()
                       if k.endswith("num_batches_tracked")}
            bn_report.append(f"{opt}:{sorted(tracked)}")
            if not tracked:
                problems.append(f"{scan}/{run_id}: no num_batches_tracked buffers")
            elif tracked != {steps}:
                problems.append(f"{scan}/{run_id} ({opt}): num_batches_tracked "
                                f"{sorted(tracked)} != optimizer steps {steps}")
    # The probe's 186.7 ms is the HEADLINE configuration: full Gram capture over all
    # parameters, unmasked. `rebuttal_fig5_cifar_paramfrac_scan` masks 75% of the
    # parameters and is a different (slower) computation, so averaging it in would hide
    # a real headline regression behind a number nobody measured. Report both.
    headline = [t for k, v in times.items() for t in v
                if k.endswith(":SVD") and k.startswith("cifar10_resnet")]
    masked = {k: sum(v) / len(v) * 1000 for k, v in times.items()
              if k.endswith(":SVD") and not k.startswith("cifar10_resnet")}
    if not headline:
        problems.append("no headline CIFAR Sven run with a train_times curve")
        avg = float("nan")
    else:
        avg = sum(headline) / len(headline)
        if avg > 0.4:
            problems.append(f"CIFAR Sven {avg * 1000:.1f} ms/step > 400 ms: the "
                            f"empty_cache / allocator settings did NOT carry through "
                            f"(probe: 186.7 ms with empty_cache off + "
                            f"expandable_segments, 841 ms with it on)")
    other = {k.split(":")[-1]: f"{sum(v) / len(v) * 1000:.1f}"
             for k, v in sorted(times.items()) if not k.endswith(":SVD")}
    ok = not problems
    check("f", "CIFAR Sven step time, and num_batches_tracked == optimizer steps",
          ok, f"headline CIFAR Sven {avg * 1000:.1f} ms/step over {len(headline)} run(s) "
              f"(probe 186.7); masked Fig-5 Sven "
              f"{ {k.split(':')[0][:26]: round(v, 1) for k, v in masked.items()} }; "
              f"other CIFAR ms/step {other}; BN num_batches_tracked "
              f"{sorted(set(bn_report))}; {len(problems)} problem(s)")
    for p in problems[:10]:
        print(f"         {p}")
    return ok


# ---------------------------------------------------------------------------
# (j) two concurrent jobs, each run exactly once
# ---------------------------------------------------------------------------

def check_j(exp, root, plan):
    """Every run executed once, by whichever worker claimed it first."""
    problems, n_dup, hosts, jobs = [], 0, set(), set()
    shared = set()
    lists_by_item = {}
    for wl in plan.work_lists:
        for item in wl.enabled_items():
            lists_by_item.setdefault((item.config, item.overrides), set()).add(wl.name)
    for key, names in lists_by_item.items():
        if len(names) > 1:
            shared.add(key)
    shared_scans = {config for config, _ovr in shared}
    #: per shared scan, how many records each slurm job wrote -- the partition is the
    #: property under test ("each run exactly once"), not who got which run
    split = {}
    #: each job's whole record-writing lifetime, to show the two jobs really did overlap
    #: (a per-SCAN window is too strict: a job can finish its share of one scan before
    #: the other starts on it and still be running the whole time)
    windows = {}
    for scan, runs in exp.items():
        scan_dir = os.path.join(root, scan)
        if not os.path.isdir(scan_dir):
            continue
        for run_id, rec in read_records(scan_dir).items():
            if rec["_n_lines"] != 1:
                n_dup += 1
                problems.append(f"{scan}/{run_id}: {rec['_n_lines']} records in one "
                                f"jsonl (a second worker re-ran it)")
            hosts.add(rec.get("host"))
            jobs.add(rec.get("slurm_job_id"))
            job = rec.get("slurm_job_id")
            if scan in shared_scans:
                split.setdefault(scan, collections.Counter())[job] += 1
            if rec.get("start_unix") and rec.get("end_unix"):
                w = windows.setdefault(job, [rec["start_unix"], rec["end_unix"]])
                w[0] = min(w[0], rec["start_unix"])
                w[1] = max(w[1], rec["end_unix"])
        # a `.1` file under _stale means two generations of one run collided
        stale = glob.glob(os.path.join(scan_dir, "_stale", "*", "*"))
        dotted = [s for s in stale if re.search(r"\.\d+($|\.)", os.path.basename(s))]
        problems += [f"{scan}: suspicious _stale file {os.path.basename(s)}"
                     for s in dotted]
        # attempts/ must be empty for runs that finished ok
        att_dir = os.path.join(scan_dir, "attempts")
        for name in (os.listdir(att_dir) if os.path.isdir(att_dir) else []):
            run_id = name.split(".")[0]
            rec = read_records(scan_dir).get(run_id)
            if rec and rec.get("status") == "ok":
                problems.append(f"{scan}: attempts/{name} for an ok run")
    if len(shared) < 1:
        problems.append("the plan has no item in two work lists, so nothing tested "
                        "two jobs on one scan")
    # the partition: a scan whose runs two jobs BOTH wrote is the case under test
    partitioned = {s: dict(c) for s, c in sorted(split.items()) if len(c) > 1}
    if not partitioned:
        problems.append("no scan had its runs written by more than one slurm job, so "
                        "nothing tested two jobs claiming from one scan (the shared "
                        f"items are {sorted(shared_scans)})")
    # ... and that those jobs were alive at the same time
    overlaps = []
    pairs = sorted(windows.items())
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            (ja, wa), (jb, wb) = pairs[i], pairs[j]
            shared_s = min(wa[1], wb[1]) - max(wa[0], wb[0])
            if shared_s > 0:
                overlaps.append(f"{ja}+{jb} {shared_s:.0f}s")
    if not overlaps:
        problems.append("no two jobs were writing records at the same time, so the "
                        "claim queue was never under real concurrency")
    ok = not problems
    check("j", "two jobs on one scan ran every run exactly once",
          ok, f"{len(shared)} item(s) in two work lists; {len(hosts)} host(s) / "
              f"{len(jobs)} slurm job(s); scans split between jobs: {partitioned}; "
              f"job lifetimes overlapping: {overlaps[:4]}; {n_dup} duplicated jsonl; "
              f"{len(problems)} problem(s)")
    for p in problems[:10]:
        print(f"         {p}")
    return ok


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default=os.path.join(REPO, "campaign", "plan_smoke.yaml"))
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--root", default=None,
                    help="results root (default: the plan's)")
    ap.add_argument("--only", default=None, help="only these letters, e.g. 'adf'")
    a = ap.parse_args(argv)

    sys.path.insert(0, REPO)
    plan = load_plan(a.plan)
    root = os.path.abspath(a.root or plan.results_root)
    exp = expected(plan, a.snapshot, root)
    print(f"[verify] plan     {a.plan}")
    print(f"[verify] snapshot {a.snapshot}")
    print(f"[verify] root     {root}")
    print(f"[verify] {len(exp)} scan(s), {sum(len(r) for r in exp.values())} expected "
          f"run(s); {sum(1 for s in exp for r in exp[s].values() if is_deliberate(r))} "
          f"deliberately divergent")
    letters = a.only or "abcdefj"
    runners = {"a": lambda: check_a(exp, root),
               "b": lambda: check_b(exp, root, a.snapshot),
               "c": lambda: check_c(exp, root),
               "d": lambda: check_d(exp, root),
               "e": lambda: check_e(exp, root),
               "f": lambda: check_f(exp, root),
               "j": lambda: check_j(exp, root, plan)}
    for letter in letters:
        print(f"\n[verify] --- ({letter}) ---")
        try:
            runners[letter]()
        except Exception as exc:                        # a check that cannot run FAILS
            import traceback
            traceback.print_exc()
            check(letter, "the check itself", False, f"{type(exc).__name__}: {exc}")

    print("\n[verify] ================ SUMMARY ================")
    for letter, name, ok, ev in RESULTS:
        print(f"[verify] ({letter}) {'PASS' if ok else 'FAIL'}  {name}")
    failed = [r[0] for r in RESULTS if not r[2]]
    print(f"[verify] {len(RESULTS) - len(failed)}/{len(RESULTS)} checks pass"
          + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
