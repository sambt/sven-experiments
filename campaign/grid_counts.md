# Campaign grid counts (2026-09-19, Stage-1 configs + the C-B3 extension round)

Run counts per in-scope scan, **generated from the configs** by `expand_grid`, not by hand.
`tests/test_configs.py::test_grid_counts_md_matches_the_configs` parses the table below and
fails if a config edit moves a number without moving this file — so the launcher and the
cost plan can read it as ground truth.

Scope is `campaign/CONTRACTS.md` "Scope update": seven configs are CUT and are deliberately
left at their pre-campaign state (`exp_critbatch_mnist`, `exp_critbatch_nanogpt`,
the four one-seed `cifar10_resnet_{kappa,paramFrac}*`, and `mnist_scan_brier`);
`mnist_microbatch_scan` and `rebuttal_mnist_batchk_probe` are stale and in no launcher;
`profile_*.yaml` is out of scope.

`exp_gpt2_small_comparison` was the eighth CUT config until **2026-09-19**, when the user
re-admitted it. It is now in scope and in the launch plan — but in a plan file of its own,
`campaign/plan_gpt2.yaml` (a100 lane, NPROC 1, one GPU per job), because its runs are an
order of magnitude longer than anything in `plan_campaign.yaml` and it must be launchable
without touching the campaign plan while the extension round is in flight.

"In scope" and "in the launch plan" are **not** the same set any more:
`exp_finetune_cifar_smallN` was moved to the extension phase by the user on 2026-09-18
~21:30 (CONTRACTS.md "Stage 1 contracts", last bullet), and all eight of its
`p3_finetune` items in `campaign/plan_campaign.yaml` are `enabled: false`. Its config
edits stand — `bn_mode: frozen`, SGDm, the lr extension — but its 408 runs are **not**
part of the campaign total. The row below says `P3-parked`.

## Per-scan counts

Counts include **all seeds** and are summed over the `n_data` sweep a launcher passes
(`rebuttal_overparam_*`, `exp_finetune_cifar_smallN`). Families are the `RunSpec.family`
values: `svd` = Sven, `standard` = every first- and second-order baseline except the two
with their own grids, then `lbfgs`, `polyak`, `jd`, `hig`.

| scan | pri | svd | standard | lbfgs | polyak | jd | hig | **total** | GPU-h floor |
|---|---|---|---|---|---|---|---|---|---|
| `cifar10_resnet_ce_scan` | P0 | 150 | 360 | 225 | 5 | (20) | (80) | **740** | 69 |
| `cifar10_resnet_scan_labelRegression` | P0 | 90 | 280 | 225 | 5 | (20) | (80) | **600** | 47 |
| `exp_nanogpt_speedrun` | P0 | 20 | 120 | - | - | - | - | **140** | 3.7 |
| `mnist_scan_ce` | P0 | 800 | 400 | 225 | 5 | 30 | 150 | **1610** | 27 |
| `mnist_scan_labelRegression` | P0 | 640 | 400 | 135 | 5 | 30 | 150 | **1360** | 22 |
| `polynomial_scan` | P0 | 720 | 500 | 225 | 5 | 30 | 150 | **1630** | 10 |
| `toy_1d_scan` | P0 | 900 | 400 | 225 | 5 | 30 | 210 | **1770** | 14 |
| `rebuttal_batchsize_polynomial_scan` | P1 | 480 | 2400 | 150 | 30 | - | - | **3060** | 7.8 |
| `rebuttal_fig5_cifar_paramfrac_scan` | P1 | 15 | - | - | - | - | - | **15** | 5.5 |
| `rebuttal_overparam_mnist_scan` | P1 | 360 | 2400 | 810 | 30 | - | - | **3600** | 20 |
| `rebuttal_overparam_polynomial_scan` | P1 | 1800 | 1600 | 900 | 20 | - | - | **4320** | 0.9 |
| `rebuttal_overparam_toy_1d_scan` | P1 | 720 | 1600 | 900 | 20 | - | - | **3240** | 0.5 |
| `exp_gpt2_small_comparison` | P1 | 3 | 24 | - | - | - | - | **27** | 125 |
| `mnist_kappaScan_labelRegression` | P3 | 210 | - | - | - | - | - | **210** | 2.2 |
| `mnist_microbatch_ce_scan` | P3 | 140 | - | - | - | - | - | **140** | 1.5 |
| `mnist_microbatch_labelreg_scan` | P3 | 140 | - | - | - | - | - | **140** | 1.5 |
| `mnist_paramfrac_ce_scan` | P3 | 100 | - | - | - | - | - | **100** | 1.1 |
| `mnist_paramfrac_labelreg_scan` | P3 | 100 | - | - | - | - | - | **100** | 1.1 |
| `polynomial_microbatch_scan` | P3 | 120 | - | - | - | - | - | **120** | 0.3 |
| `polynomial_paramfrac_scan` | P3 | 100 | - | - | - | - | - | **100** | 0.3 |
| `toy_1d_microbatch_scan` | P3 | 120 | - | - | - | - | - | **120** | 0.3 |
| `toy_1d_paramfrac_scan` | P3 | 100 | - | - | - | - | - | **100** | 0.3 |
| `exp_finetune_cifar_smallN` | P3-parked | 48 | 360 | - | - | - | - | **408** | 1.4 |

* In the launch plan: **23242** runs across 22 scans, GPU-h floor ~362 (see the caveats
  below — the number to reserve against is 1,100–1,500). 23,215 of those runs and ~237
  GPU-h are `plan_campaign.yaml`; the other 27 runs and ~125 GPU-h are the GPT-2 scan in
  `plan_gpt2.yaml`, which is a third of the floor in 0.1% of the runs.
* Parked (extension phase, `exp_finetune_cifar_smallN`): **408** runs, floor 1.4 GPU-h.
* Described by the in-scope configs in total: **23650** runs across 23 scans.

Cell conventions:

* `-` = the config does not describe that family for the mode a launcher runs it in.
* `(N)` = **described by the config, submitted by nothing, and not in the total.** Both
  CIFAR headline configs still carry `lrs_jd` / `lrs_hig` / `tau_hig`, i.e. 20 JD + 80 HIG
  enumerable runs each, 200 in total. No launcher has ever run `mode=jd` or `mode=hig` on
  a ResNet; `p3_cifar_jd_hig` in `plan_campaign.yaml` keeps all four items
  `enabled: false` pending a user decision, and by the plan's own note HIG on ResNet18
  needs a full per-sample Jacobian, ~50 GPU-h **per scan**. The keys are kept so that
  reviving the study is one flag rather than a config edit; the cost of keeping them is
  that **`mode=all` on either CIFAR config would claim 100 unbudgeted runs** under
  `scheduler=claims`. Launch those two scans as `mode=svd` / `mode=standard`, which is
  what the plan does. `tests/test_configs.py` pins the 20/80 against the configs, so the
  liability cannot grow unnoticed.

On the four MLP headline scans JD and HIG **are** in scope and the new launcher must
submit them — `grid_inventory.md` 5.1: the old `submit_fresh_suite.sh` emitted only
`mode=svd` and `mode=standard`, silently dropping 400 JD/HIG runs that
`best_configs.json` nonetheless reports best configs for.

### Phase-5 companions (not in the total)

Seven scans — `toy_1d_scan`, `polynomial_scan`, `mnist_scan_ce`,
`mnist_scan_labelRegression`, `exp_nanogpt_speedrun` and the two `cifar10_resnet_*`
headline scans — each carry **three** companion configs, 21 in all:

* `*_timing.yaml` (pass B) inherits the parent and overrides nothing, so it describes the
  parent's full grid with the parent's run_ids in its own results directory;
* `*_diag.yaml` (pass C) turns on `svd_info: full`, the checkpoint ladder for *every*
  family and a dense spectra head. Grid-identical to the parent — same run_ids **and**
  same run hashes, because none of what it changes is hashed (`run_hash` ignores what is
  logged or scheduled), so a diagnostics run is bit-for-bit the scan's run with more of it
  recorded;
* `*_confirm.yaml` (pass D) moves to fresh model seeds (base + 100..104) for the numbers
  the paper reports. *Not* grid-identical: different seeds, and toy/polynomial also put
  `data_seed` in `result_id_fields` so the three data-seed replicates get distinct
  run_ids.

All 21 are excluded from the table above, because every pass runs **only the per-method
selected configs** (~13 methods x 5 seeds per scan) — a launcher selection, not a config
one. Counting them at their nominal grid size would roughly treble the campaign total for
work that is about 1,600 runs. Phase-5 counts live in `campaign/plan_phase5.yaml`'s
generated header instead: **425 timing + 425 diag + 725 confirm**.

`tests/test_configs.py` still holds them to every other rule (they are in `IN_SCOPE`):
`COMPANION_PARENT` names all 21 explicitly rather than matching a suffix, so a companion
cannot leave this cost table by being named a certain way, and `EXPECTED_CHECKPOINTS`
pins each pass's own checkpoint policy — `log`/`epochs` with `checkpoints_svd` cleared for
diag, `final` for confirm — since that is precisely what passes C and D exist to change.

## Cost: the "GPU-h floor" column

The column is a **floor**, useful for ranking scans, not for reserving a budget:

```
GPU-h(scan, family) = runs x steps_per_run x ms_per_step / 3.6e6 / T
```

* `ms_per_step` and `T` (standalone-runs-worth of work per GPU per wall-second at the chosen
  NPROC) are the measured values in `campaign/stage0_reports/gpu.probe.md`: Sven-hooks 11.8 ms
  (toy/poly) and 11.1 ms (MNIST) at T=7.18/4.56 with NPROC 12; CIFAR Sven `full` **186.7 ms**
  at T=1.00, NPROC 1; CIFAR first-order 24.0 ms at T=3.85; nanoGPT Sven 71.7 ms at T=2.33;
  MNIST first-order 4.6 ms at T=12.
* per-family multipliers on the first-order step come from the `total_time` ratios in
  `bench/best_configs.json` on `toy_1d_scan` (Adam = 1): `standard` 1.5 (six cheap optimizers
  plus SOAP 1.4x, Shampoo 3.2x, KFAC 3.8x), `lbfgs` 3.0 (max_iter 1/2/3), `jd` 1.8, `hig` 4.1,
  `polyak` 1.05.
* `steps_per_run` uses the **post-C-E1** split sizes (`data.impl.md`): MNIST 782 steps/epoch
  at B=64, CIFAR 352 at B=128, shakespeare 108 at B=64. GPT-2-small is **13,125 steps in a
  single epoch** (210,000 blocks of 1024 tokens at B=16), which is why 27 runs cost a third
  of the whole floor: `ms_per_step` there is ~2,550 for Sven (the ~9.3 h/run recorded in
  `experiments/configs/dataset/fineweb_edu.yaml`) and ~1,100 for a first-order baseline, at
  T = 1.0 (NPROC 1, one GPU per job). **Estimate, not a probe** — re-derive it from the
  2026-09-19 GPU smoke of this scan before reserving against it.
* the nine rows the C-B3 extension round moved were **re-derived, not re-measured**: the
  per-run unit cost of a scan's first-order family is solved from its Stage-1 floor and
  its Stage-1 counts (so the old number is reproduced exactly), Sven's unit is that times
  `(ms_sven/T_sven)/(ms_fo/T_fo)` from the probe values above — 6.35x on MLP, 29.9x on
  CIFAR — and the new counts are put through the same sum. The round adds **~27 GPU-h**
  of floor, of which ~8 are the three P1 `lrs_standard` lists and ~7 are `toy_1d_scan`'s
  Sven grid (360 -> 900 runs).

It excludes evaluation (three loaders now: val, test and the fixed 10k `train_eval` subset),
data loading, checkpoint I/O, process start-up, queue wait and every failed or requeued job.

**Do not plan with 362 GPU-h** (237 for `plan_campaign.yaml` + 125 for the GPT-2 scan).
`campaign/scout/sharding.md` puts the honest figure at **1,100-1,500 GPU-h**, and that is
still the number to reserve against. The two are not in conflict; the gap is mostly three
things:

1. the scout deflated *historical* process-hours, which were measured with
   `torch.cuda.empty_cache()` on every Sven step — 841 ms/step on CIFAR instead of 186.7.
   C-T3 defaulting `empty_cache` to `False` is worth roughly **310 GPU-h on the two CIFAR
   headline scans alone**, and it is the single largest cost change in the campaign;
2. cheap-MLP step times are set by the host CPU, not the GPU (gpu.probe surprise 2: the same
   MNIST-Adam config is 1.13 ms on a quiet node and **4.58 ms** on a 4/4-occupied one). The
   floor uses the busy-node number for first-order and the quiet-node number for Sven, so the
   MLP rows can be off by 2-4x either way;
3. the floor assumes every process gets the probe's NPROC and none of the 12-hour-wall
   timeouts that killed 34 jobs in the last round.

None of the three applies to `exp_gpt2_small_comparison`: it runs at NPROC 1 on a whole
A100 (no co-tenant, no host-CPU contention), its `empty_cache` is already `false`, and its
27 runs are 8 jobs with no wall-clock crowding. Its floor is therefore close to its true
cost once `ms_per_step` is measured rather than estimated — the one row of this table where
the floor is meant to be planned with.

Two rows deserve attention regardless of the model: **CIFAR-CE at 67 GPU-h and CIFAR-label-reg
at 45** are between them more than half of the floor and roughly a quarter of the honest
budget, and they are the only scans where a single run costs ~0.5 GPU-h. Every other scan is
noise by comparison — which is why the C-B3 extension round was made generous on the MLP
scans and is sized point-by-point on CIFAR, each addition justified from **that scan's own
legacy records** rather than from the MLP evidence (see below).

## What changed in the C-B3 extension round (2026-09-19)

The Stage-1 round below baked in one *predicted* extension, from legacy records. This
round is the *measured* one the user approved after the campaign finished: every added
point closes an edge that `tools/reconcile.py` flagged over the 15,735 completed runs
(`campaign/reconcile_2026-09-19.txt`, the `edges` column of "best config per method"),
and nothing is added anywhere the optimum came back interior.

Three rules decided what did **not** grow. Sven's `k` is never extended — `k = B` is a
method boundary, not a grid edge, and it is the reported optimum on six scans. CIFAR's
Sven grids are left alone at ~0.5 GPU-h per run (the knowingly accepted edge of "Open
items" 1 below). And an axis stops early where the next point is meaningless rather than
merely expensive: `rtol` is a *relative* singular-value cut, so at 1.0 only the leading
direction survives — which is why MNIST-CE gets 3e-1 and no more, and the batch-size
scan gets one point below its bottom edge rather than two.

| scan | family | axis | added | edge it closes (seed-mean final val) | new runs |
|---|---|---|---|---|---|
| `toy_1d_scan` | svd | `lrs` | 0.02, 0.01 | Sven `lr:EDGE-LOW` at 0.05 (4.799e-07) | +540 |
| | svd | `rtol` | 1e-5, 1e-6 | Sven `rtol:EDGE-LOW` at 1e-4 (same point) | (with the above) |
| | hig | `lrs_hig` | 0.0015, 0.0005 | HIG `lr:EDGE-LOW` at 0.005 (4.707e-09) | +60 |
| | lbfgs | `lrs_lbfgs` | 0.03, 0.01 | L-BFGS `lr:EDGE-LOW` at 0.1 (2.459e-04) | +90 |
| `polynomial_scan` | svd | `rtol` | 3e-2, 1e-1 | Sven `rtol:EDGE-HIGH` at 1e-2 (0.1175) | +240 |
| | standard | `lrs_standard` | 3e-6, 1e-6 | KFAC `lr:EDGE-LOW` at 1e-5 (0.1728) | +100 |
| | lbfgs | `lrs_lbfgs` | 2.0, 4.0 | L-BFGS `lr:EDGE-HIGH` at 1.0 (0.2395) | +90 |
| `mnist_scan_ce` | svd | `rtol` | 3e-1 | Sven `rtol:EDGE-HIGH` at 1e-1 (0.1231) | +160 |
| | lbfgs | `lrs_lbfgs` | 2.0, 4.0 | L-BFGS `lr:EDGE-HIGH` at 1.0 (0.1277) | +90 |
| `cifar10_resnet_scan_labelRegression` | lbfgs | `lrs_lbfgs` | 4.0 | L-BFGS `lr:EDGE-HIGH` at 2.0 (0.4380) | +45 |
| `cifar10_resnet_ce_scan` | lbfgs | `lrs_lbfgs` | 4.0 | L-BFGS `lr:EDGE-HIGH` at 2.0 (1.014) | +45 |
| | standard | `lrs_standard` | 3.0 | SGD `lr:EDGE-HIGH` at 1.0 (1.338) | +40 |
| `rebuttal_overparam_toy_1d_scan` | standard | `lrs_standard` | the headline list | 8 of 10 optimizers on the 1e-1 top edge | +800 |
| | lbfgs | `lrs_lbfgs` | 2.0, 4.0 | L-BFGS `lr:EDGE-HIGH` at 1.0 (3.219e-06) | +360 |
| `rebuttal_overparam_polynomial_scan` | standard | `lrs_standard` | the headline list | MuonW/Muon/SGDm/SGD/Shampoo on the top edge | +800 |
| | lbfgs | `lrs_lbfgs` | 2.0, 4.0 | L-BFGS `lr:EDGE-HIGH` at 1.0 (0.1244) | +360 |
| | svd | `lrs`, `rtol` | 0.01/0.02, 3e-2/1e-1 | Sven `lr:EDGE-LOW` + `rtol:EDGE-HIGH` (0.2829) | +1080 |
| `rebuttal_overparam_mnist_scan` | standard | `lrs_standard` | the headline list | KFAC on 1e-4, SGD/Shampoo on 1e-1 | +1200 |
| `rebuttal_batchsize_polynomial_scan` | standard | `lrs_standard` | the headline list | KFAC on 1e-4, Shampoo on 1e-1 | +1200 |
| | lbfgs | `lrs_lbfgs` | 2.0, 4.0 | L-BFGS `lr:EDGE-HIGH` at 1.0 (0.1303) | +60 |
| | svd | `rtol` | 1e-5 | Sven `rtol:EDGE-LOW` at 1e-4 (0.1040) | +120 |
| **in-plan total** | | | | | **15735 -> 23215 (+7480)** |

(`plan_campaign.yaml` only. The 27 runs of `exp_gpt2_small_comparison`, re-admitted on
2026-09-19 and launched from `plan_gpt2.yaml`, are not part of this round and are not in
the 23,215; the per-scan table and the totals above include them.)

Notes on the judgement calls:

* **The three P1 scans and the batch-size scan get the headline `lrs_standard` list**
  (`[1e-5, 3e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0]`). They had kept the pre-extension
  1e-4..1e-1 grid as a deliberate deviation ("Open items" 3 below), and the finished runs
  confirmed the prediction made there: they are on an edge, on both ends at once on the two
  MNIST-and-polynomial ones. This is +4,000 of the +7,480 runs and about 8 of the 27 added
  GPU-h, because the shared list multiplies by ten optimizers, five seeds and (for the
  overparam scans) four `n_data` points at once.
* **`rebuttal_overparam_polynomial_scan`'s Sven axes** are the one addition not named in
  the approved list; it is the only scan besides the headline four where the reconcile
  flagged a Sven `lr`/`rtol` edge, and the rtol trend is steep rather than flat (at the
  best cell the seed-means are 6.646 / 0.5989 / 0.2829 going 1e-4 -> 1e-3 -> 1e-2). At
  ~0.2 ms-scale full-batch steps on a 673-parameter MLP the 1,080 runs are ~0.5 GPU-h,
  the cheapest scan in the campaign.
* **`mnist_scan_labelRegression` is untouched.** Its only flag is Sven `k:EDGE-HIGH` at
  k = B, and its L-BFGS and HIG optima are interior (0.5 and 0.05) — so the twin of the
  most-extended MNIST scan gets nothing, which is the point of sizing per scan.

Existing `run_id`s are unchanged everywhere: the round only appends to grid lists, so the
finished 15,735 runs are skipped by the done-marker dedup (`run_hash` reads per-spec
values, never the grid lists) and only the new points execute.

## What changed in the Stage-1 round

Net per scan, against `campaign/scout/grid_inventory.md` section 1 (the pre-campaign
configs). Per-scan nets rather than per-change deltas, because the changes multiply: SGDm
lands on the *extended* lr grid, so "+SGDm" and "+lrs" cannot be added independently.

| scan | before | after | net |
|---|---|---|---|
| `toy_1d_scan` | 780 | 1080 | +300 |
| `polynomial_scan` | 780 | 1200 | +420 |
| `mnist_scan_labelRegression` | 1060 | 1360 | +300 |
| `mnist_scan_ce` | 1060 | 1360 | +300 |
| `cifar10_resnet_scan_labelRegression` | 290 | 555 | +265 |
| `cifar10_resnet_ce_scan` | 290 | 655 | +365 |
| `exp_nanogpt_speedrun` | 100 | 140 | +40 |
| `rebuttal_overparam_toy_1d_scan` | 2240 | 2080 | **-160** |
| `rebuttal_overparam_polynomial_scan` | 2240 | 2080 | **-160** |
| `rebuttal_overparam_mnist_scan` | 2640 | 2400 | **-240** |
| `rebuttal_batchsize_polynomial_scan` | 2640 | 1680 | **-960** |
| `mnist_kappaScan_labelRegression` | 15 | 210 | +195 |
| `exp_finetune_cifar_smallN` (parked) | 240 | 408 | +168 |
| the 8 micro-batch / param-fraction scans, `rebuttal_fig5` | 935 | 935 | 0 |
| **in-scope total** | **15310** | **16143** | **+833** |

What produced those nets:

* **C-B2** `SGDm` beside every `SGD`, in all 11 scans that list SGD. `SGD` with no momentum
  is not a 2026 baseline; `SGDm` is `torch.optim.SGD(momentum=0.9)`.
* **C-B4 dropped** (scope update): no `weight_decays` sweep anywhere. `rebuttal_overparam_*`
  and `rebuttal_batchsize_polynomial_scan` lose `[0.0, 0.01]`, which had been doubling
  AdamW / Muon / MuonW *and* forcing MuonW to 0.01 instead of its own 0.1 default -- the
  thing `submit_reruns_2026-09-17_part2.sh` had to patch with `weight_decays=[0.1]`.
  Worth -1,200 runs on its own.
* **O5**: the L-BFGS shape is pinned at the legacy polynomial-headline best in the
  batch-size scan, 810 -> 90 runs. The largest single saving in the round.
* **C-B1 / C-B5 / O6**: the two CIFAR headline scans gain AdamW, SGDm, SOAP, Muon and MuonW
  (60 -> 280/320 standard runs each). Muon on a ResNet is legitimate now that conv kernels go
  through the vendored MuonConv; the old "Muon" there was AdamW plus one `fc` layer (F13).
* **C-B3**, one extension round baked in, per the evidence in `bench/best_configs.json`, the
  known cases under C-B3 in `CHANGES_NEEDED.md`, and — for the two CIFAR scans, which have no
  `best_configs.json` entry at all — the 290 legacy `*.jsonl` per scan read directly:
  * the shared `lrs_standard` goes 4 -> 8 points on the MLP headline scans, 4 -> 6 on
    nanoGPT, 4 -> 8 on CIFAR-CE and 4 -> 7 on CIFAR-label-reg. The two CIFAR lists differ on
    purpose: **RMSprop's optimum on CIFAR-CE IS the old 1e-1 top edge** (seed-mean final val
    2.1002 / 1.3324 / 1.3725 / 1.1983 at 1e-4 / 1e-3 / 1e-2 / 1e-1), so that edge gets its
    two half-decade points, 3e-1 and 1.0 (+40 runs); on CIFAR-label-reg nothing sits on the
    top edge (RMSprop and Adam peak at 1e-2, SGD diverges above 1e-3 — 1 of 5 seeds finite
    at 1e-2, 0 of 5 at 1e-1), so it stops at 3e-1.
  * `lrs_lbfgs` gains **2.0 on both CIFAR headline scans** (+45 runs each, ~1.7 GPU-h each).
    L-BFGS's best shape is on the lr top edge on both: label-reg per-lr seed-mean falls
    monotonically 3.0635 -> 0.7855 -> 0.7534 at lr 0.1 / 0.5 / 1.0 and its best shape
    (mi3/hs5) is 0.7010 at 0.5 vs 0.6991 at 1.0; CE's best shape (mi2/hs5) is 1.3533 at 0.5
    vs 1.3201 at 1.0. One bracketing point rather than two: the gain across the last
    half-decade is 0.3% (label-reg) and 2.5% (CE), i.e. a plateau, and lr > 1 under
    `strong_wolfe` over-relaxes an already-accepted step — CE's mi3 arms already blow up
    from 1.78 to 3.04 going 0.5 -> 1.0.
  * `lrs_jd` 4 -> 6; HIG's lr grid is **shifted down** rather than grown and its tau
    extended up two half-decades; `rtol` gains 1e-5 on polynomial; CIFAR-CE's Sven lrs gain
    0.05 and 0.02 — that last line alone is ~30 GPU-h and is the most expensive addition in
    the round. Every added point is asserted in `tests/test_configs.py::MUST_CONTAIN` with
    its reason attached.
  * HIG's rationale, stated precisely because it is the one place the first pass got the
    *reason* wrong: lr >= 0.5 produced **no legacy records at all** on `toy_1d_scan`,
    `polynomial_scan` and `mnist_scan_labelRegression`, while on `mnist_scan_ce` all 40 of
    those runs completed and are on disk — they are simply dominated (0.1035 at lr 0.05 /
    tau 1e-2 against 0.1735 at lr 0.5 and 0.2023 at lr 1.0, up to 47.4 at tau 1e-8). "Always
    crashes" was the wrong reason for the right decision.
* **C-X1**: the kappa scan goes 15 -> 210 runs (3 kappa x 7 lr x 2 k x 5 seeds). The lr list
  `[0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5]` is chosen so that **three** effective steps
  `2*lr/kappa` — 0.25, 0.5 and 1.0 — are realised by all three kappas, which is what makes
  the study a kappa comparison instead of an lr sweep in disguise (F29). The first pass had
  five lrs whose matched set was the single step 0.5, i.e. 1 of 15 (kappa, lr) points; a
  *full* match would need `S/2, S, 3S/2` in the grid for every step and cost 15 lrs / 450
  runs. lr 0.5 is kept because it is the legacy set point of the 15 runs on disk.
* **C-E2**: `bn_mode: batch` on the CIFAR headline scans and Fig-5 (replacing
  `gram_freeze_norm_stats: false`), `bn_mode: frozen` on the fine-tune scan. No count change,
  but the fine-tune baselines' run_ids gain `_bnfrozen`: their 192 legacy runs trained
  BatchNorm on 250-2000 images and are superseded.
* **MNIST overparam top point 60000 -> 50000**: no count change. `n_train` above the pool now
  raises instead of clamping, so the old point was an error rather than a 60k run.

The extension round is baked in **now**, in one shot, because there is no time for the
measure-then-extend loop C-B3 describes.

## Checkpoint storage

`checkpoints` / `checkpoints_svd` per scan family (C-L3), and what the `log` ladder costs.
`log` keeps every state **in RAM until flush** (`ckpt-sampler.impl.md`), which is why the
ResNet scans keep `final`:

| family | policy | states/run | bytes/run | runs | total |
|---|---|---|---|---|---|
| toy / polynomial MLP, 20 epochs (593-673 params) | `log` | ~35 | ~94 KB | 6,900 | ~0.6 GB |
| ... the two 200-epoch full-batch overparam scans | `log` | ~209 | ~563 KB | 7,560 | ~4.3 GB |
| MNIST MLP (27,562 params) | `final` + `checkpoints_svd: log` | ~34 (svd only) | ~3.7 MB | 2,490 svd | ~9.2 GB |
| nanoGPT (826,368 params) | `epochs` | 50 | ~165 MB | 140 | ~23 GB |
| GPT-2 small (163.0M params) | `final` | 1 | ~652 MB | 27 | ~18 GB |
| ResNet18 (11.18M params) | `final` | 1 | ~45 MB | 1,763 | ~79 GB |

About **134 GB** for everything the configs describe, of which the parked fine-tune scan is
408 ResNet states ~18 GB — so **~116 GB in the launch plan** — plus one
`ckpt/init_mseed{seed}.pt` per model seed under `final`. The nanoGPT and ResNet rows are the
ones to check against the quota before launch; the MNIST baselines and both CIFAR scans
deliberately keep `final` because `log` holds every state in RAM until flush
(`ckpt-sampler.impl.md`) — ~1.6 GB per ResNet run.

## Open items for the orchestrator

1. **CIFAR-label-reg Sven lr is not extended.** C-B3 in `CHANGES_NEEDED.md` lists "the largest
   [lr] on CIFAR label-reg" as a known edge, but `lrs: [0.1, 0.5, 1.0]` was left alone: lr=1 is
   the full min-norm (Gauss-Newton) step, so points above it are over-relaxation rather than a
   finer search, and two more points cost 2 k x 3 rtol x 5 seeds x 2 = **60 runs ~ 30 GPU-h**.
   `bench/best_configs.json` has no CIFAR entry at all, so the "optimum at the top edge" claim
   rests on the pre-Gram classic runs. Decide: extend to 2.0/3.0, or accept the edge and say so
   in the paper. (The *baseline* edges on the same two scans — RMSprop's lr on CE and L-BFGS's
   lr on both — were measured from the legacy records and are extended; see above. Sven is the
   only family on these two scans where an edge is knowingly accepted.)
   **Still open after the extension round (2026-09-19), and now with real evidence:** the
   finished scans report Sven on `k:EDGE-HIGH` (= B, a method boundary) on both, plus
   `rtol:EDGE-HIGH` at 1e-2 on CIFAR-CE, while `lr` came back interior — 0.5 on label-reg,
   0.1 on CE — so the *lr* edge this item was written about has closed on its own. What is
   left is CIFAR-CE's rtol top edge, deliberately not extended at 2 k x 5 lr x 5 seeds =
   **50 runs ~ 25 GPU-h**. Same decision as before: extend, or accept and say so in the paper.
2. **`rebuttal_fig5_cifar_paramfrac_scan` set point is still tentative.** k=64, lr=1.0,
   rtol=1e-3 are the pre-Gram classic best; `EXPERIMENTS.md` documents k=128, lr=0.1, rtol=1e-4
   for the same figure (`grid_inventory.md` 5.5). It must be re-pointed at the BN-fixed
   `cifar10_resnet_scan_labelRegression` results before phase 4 launches.
   `tests/test_configs.py::test_fig5_setpoint_is_flagged_tentative_exactly_while_it_is_tentative`
   now couples the values to the "STILL TENTATIVE" header marker, so neither can move without
   the other. What is still **unguarded** is the launch itself: `p1_cifar_fig5` in
   `plan_campaign.yaml` is enabled by omission, so a P1 launch today would spend ~5.5 GPU-h on
   the tentative values and produce a wrong headline figure. The launcher track should set
   `enabled: false` on that item until the set point is re-derived.
3. ~~**Extensions were applied to headline scans only.**~~ **CLOSED** by the extension round
   (user-approved, 2026-09-19). The prediction held: on all four of those scans the shared
   `lrs_standard` optimum was an edge — the 1e-1 top one for eight of ten optimizers on
   overparam-toy, the 1e-4 bottom one for KFAC on overparam-MNIST and the batch-size scan —
   and they now carry the headline list. The cost estimate did not hold: it is +4,000 runs,
   not +1,900 (the list multiplies by ten optimizers x five seeds x four `n_data` points),
   and ~8 GPU-h rather than "under 2". See "What changed in the C-B3 extension round".
4. **The two CIFAR jd/hig grids (200 runs) are kept but unbudgeted**, and `mode=all` on either
   config would claim them. See the `(N)` note under the table. Either enable
   `p3_cifar_jd_hig` and budget ~100 GPU-h, or decide the JD/HIG comparison stays on the four
   MLP scans, in which case the three keys can be deleted from both CIFAR configs.
5. **GPT-2-small is one seed and one grid-extension round short.** Re-admitted 2026-09-19
   and budgeted at 27 runs / ~125 GPU-h, which is already a third of the floor, so a
   second seed or a wider `lrs_standard` costs more than every P3 ablation put together.
   Two consequences the orchestrator should plan for: (a) nothing here can be seed-averaged,
   so a tie between Sven and a baseline is unresolvable without doubling the scan; (b) if
   the reconcile comes back with an lr edge, one half-decade for one optimizer is ~4 GPU-h
   and for all four is ~16. Its floor is an estimate, not a probe — see the cost section.
