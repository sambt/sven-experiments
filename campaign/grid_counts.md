# Campaign grid counts (2026-09-18, Stage-1 configs)

Run counts per in-scope scan, **generated from the configs** by `expand_grid`, not by hand.
`tests/test_configs.py::test_grid_counts_md_matches_the_configs` parses the table below and
fails if a config edit moves a number without moving this file — so the launcher and the
cost plan can read it as ground truth.

Scope is `campaign/CONTRACTS.md` "Scope update": eight configs are CUT and are deliberately
left at their pre-campaign state (`exp_critbatch_mnist`, `exp_critbatch_nanogpt`,
`exp_gpt2_small_comparison`, the four one-seed `cifar10_resnet_{kappa,paramFrac}*`, and
`mnist_scan_brier`); `mnist_microbatch_scan` and `rebuttal_mnist_batchk_probe` are stale and
in no launcher; `profile_*.yaml` is out of scope.

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
| `cifar10_resnet_ce_scan` | P0 | 150 | 320 | 180 | 5 | (20) | (80) | **655** | 67 |
| `cifar10_resnet_scan_labelRegression` | P0 | 90 | 280 | 180 | 5 | (20) | (80) | **555** | 45 |
| `exp_nanogpt_speedrun` | P0 | 20 | 120 | - | - | - | - | **140** | 3.7 |
| `mnist_scan_ce` | P0 | 640 | 400 | 135 | 5 | 30 | 150 | **1360** | 22 |
| `mnist_scan_labelRegression` | P0 | 640 | 400 | 135 | 5 | 30 | 150 | **1360** | 22 |
| `polynomial_scan` | P0 | 480 | 400 | 135 | 5 | 30 | 150 | **1200** | 7.3 |
| `toy_1d_scan` | P0 | 360 | 400 | 135 | 5 | 30 | 150 | **1080** | 7.0 |
| `rebuttal_batchsize_polynomial_scan` | P1 | 360 | 1200 | 90 | 30 | - | - | **1680** | 4.8 |
| `rebuttal_fig5_cifar_paramfrac_scan` | P1 | 15 | - | - | - | - | - | **15** | 5.5 |
| `rebuttal_overparam_mnist_scan` | P1 | 360 | 1200 | 810 | 30 | - | - | **2400** | 16 |
| `rebuttal_overparam_polynomial_scan` | P1 | 720 | 800 | 540 | 20 | - | - | **2080** | 0.4 |
| `rebuttal_overparam_toy_1d_scan` | P1 | 720 | 800 | 540 | 20 | - | - | **2080** | 0.4 |
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

* In the launch plan: **15735** runs across 21 scans, GPU-h floor ~210 (see the caveats
  below — the number to reserve against is 1,100–1,500).
* Parked (extension phase, `exp_finetune_cifar_smallN`): **408** runs, floor 1.4 GPU-h.
* Described by the in-scope configs in total: **16143** runs across 22 scans.

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

### P2 timing companions (not in the total)

Seven `*_timing.yaml` configs (`toy_1d_scan`, `polynomial_scan`, `mnist_scan_ce`,
`mnist_scan_labelRegression`, `exp_nanogpt_speedrun`, and the two new
`cifar10_resnet_ce_scan_timing` / `cifar10_resnet_scan_labelRegression_timing`) inherit their
parent and override nothing, so each describes the parent's full grid with the parent's
run_ids in its own results directory. A timing pass runs **only the per-method best configs**
(~13 methods x 5 seeds per scan, i.e. of order 500 runs over the seven scans), which is a
launcher selection, not a config one — hence they are excluded from the total above rather
than counted at their nominal grid size.

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
  at B=64, CIFAR 352 at B=128, shakespeare 108 at B=64.

It excludes evaluation (three loaders now: val, test and the fixed 10k `train_eval` subset),
data loading, checkpoint I/O, process start-up, queue wait and every failed or requeued job.

**Do not plan with 210 GPU-h.** `campaign/scout/sharding.md` puts the honest figure at
**1,100-1,500 GPU-h**, and that is still the number to reserve against. The two are not in
conflict; the gap is mostly three things:

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

Two rows deserve attention regardless of the model: **CIFAR-CE at 67 GPU-h and CIFAR-label-reg
at 45** are between them more than half of the floor and roughly a quarter of the honest
budget, and they are the only scans where a single run costs ~0.5 GPU-h. Every other scan is
noise by comparison — which is why the C-B3 extension round was made generous on the MLP
scans and is sized point-by-point on CIFAR, each addition justified from **that scan's own
legacy records** rather than from the MLP evidence (see below).

## What changed in this round

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
| toy / polynomial MLP, 20 epochs (593-673 params) | `log` | ~35 | ~94 KB | 4,400 | ~0.4 GB |
| ... the two 200-epoch full-batch overparam scans | `log` | ~209 | ~563 KB | 4,160 | ~2.3 GB |
| MNIST MLP (27,562 params) | `final` + `checkpoints_svd: log` | ~34 (svd only) | ~3.7 MB | 2,330 svd | ~8.7 GB |
| nanoGPT (826,368 params) | `epochs` | 50 | ~165 MB | 140 | ~23 GB |
| ResNet18 (11.18M params) | `final` | 1 | ~45 MB | 1,633 | ~73 GB |

About **107 GB** for everything the configs describe, of which the parked fine-tune scan is
408 ResNet states ~18 GB — so **~89 GB in the launch plan** — plus one
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
3. **Extensions were applied to headline scans only.** `rebuttal_overparam_*` and
   `rebuttal_batchsize_polynomial_scan` keep `lrs_standard: [1e-4 .. 1e-1]`, so SGD, Shampoo and
   the new match_rms_adamw Muon may well sit on an edge there too. Extending them costs ~+1,900
   runs but under 2 GPU-h (they are 200-step full-batch or tiny-MLP runs). Cheap to say yes to.
4. **The two CIFAR jd/hig grids (200 runs) are kept but unbudgeted**, and `mode=all` on either
   config would claim them. See the `(N)` note under the table. Either enable
   `p3_cifar_jd_hig` and budget ~100 GPU-h, or decide the JD/HIG comparison stays on the four
   MLP scans, in which case the three keys can be deleted from both CIFAR configs.
