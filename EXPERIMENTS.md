# Sven experiment taxonomy — what the campaign actually ran

Every fact below is derived from the repo and from the result records. Sources:
`experiments/configs/*.yaml` (grids, seeds, epochs, batch sizes, evaluation keys),
`campaign/grid_counts.md` (run counts, in-plan vs parked), `campaign/plan_campaign.yaml` /
`plan_gpt2.yaml` / `plan_phase5.yaml` (what was submitted), `bench/best_configs.json` (the
selection of record), and the `schema_version: 2` records under `experiment_results/`. Every
run count, divergence count and parameter count was reproduced by running `tools/reconcile.py`,
`tools/select_best.py`, `bench/check_timing_join.py` and a first-line read of every record.
The settled total is **24,894** records; storage figures are `du`/`stat` on the files.

**The campaign in one line.** 23,215 in-plan runs across 21 scans plus 29 runs of
`exp_gpt2_small_comparison`, then 1,575 runs in three result-dependent passes over the seven
headline scans. Every one of those finished: **0 `oom`, 0 `error`, 0 stale-hash, 0 poisoned,
0 runs never started**; 1,402 of the 23,215 in-plan runs carry `status: diverged` and are
recorded as such rather than being absent. The **analysis** definition of failure
(`analysis/style.is_diverged`, which also catches a finite blow-up — §5) is wider and counts
**2,553** of them; that wider count is the one that governs selection and every analysis table,
so no claim about a method's robustness may be read off the status field alone (§7).

One additive follow-up, `p2_cifar_ce_rtol` — a 45-run CIFAR-CE Sven `rtol` extension (§5) —
pushed `cifar10_resnet_ce_scan`'s expected count from 740 to **785**, all `ok`, 0 diverged. The
extension was off-grid by design (its `rtol` values are deliberately not in the config's `rtol`
list) and purely additive, so no existing `run_id` or `run_hash` moved. It *did* move the pick,
which is why three further phases follow it: the Fig-5 re-run at the selected configuration
(§3.3), the CIFAR-CE Sven timing / diag / confirm re-runs (§5), and the step-time / memory
re-profile into `profile_results_v3/` (§1.6).

| phase | runs | wall-clock window |
|---|---|---|
| main campaign (P0–P3) | 15,735 | ~26 h |
| grid-extension round | 7,480 | ~4 h |
| GPT-2 small | 29 of 29 | ~17 h |
| phase 5: timing + diag + confirm | 1,575 | ~7 h |
| CIFAR-CE `rtol` extension | 45 of 45 | ~3.5 h |
| Fig-5 re-run at the selected config | 15 of 15 | ~2.5 h |
| phase 5 re-run: CIFAR-CE Sven only | 15 (5 timing + 5 diag + 5 confirm) | ~2 h |
| step-time / memory re-profile → `profile_results_v3/` | 720 of 720 configurations | ~1 h, one exclusive A100-80GB |

`git_dirty` is `false` on every record: all campaign processes ran from an exported snapshot
(`tools/deploy_snapshot.sh`, under `$SV3_SCRATCH/deploy/<sha8>_<sha8>/`), never from the
working tree.

---

## 1. Shared conventions

### 1.1 Three splits, fixed sizes

Every dataset class exposes `train_dataset`, `val_dataset`, `test_dataset` and records
`split_seed` (C-E1). The three splits are the same for every optimizer in a scan.

| dataset | train | val | test | how |
|---|---|---|---|---|
| MNIST | 50,000 | 10,000 | 10,000 | official train split 50k/10k by `split_seed: 1234`; official test set |
| CIFAR-10 | 45,000 | 5,000 | 10,000 | official train split 45k/5k by `split_seed: 1234`; official test set. No augmentation; published channel statistics |
| toy-1D | pool of 10,000 | 10,000 | 10,000 | three *separate* generators seeded from `data_seed`; targets normalised by the **pool** mean/std |
| random polynomial | pool of 10,000 | 10,000 | 10,000 | same construction; `data_seed` also fixes the coefficient draw |
| tiny-shakespeare (char) | 6,971 blocks | 871 | 871 | contiguous 80/10/10 by position, `block_size: 128` |
| FineWeb-edu (GPT-2 BPE) | 210,000 blocks | 200 | 200 | `block_size: 1024`; val/test blocks from **disjoint documents** of one shared iterator (checked by `tools/check_token_split.py`) |

`n_train` (or the launcher's `n_data`) subsamples the *train* part only, so validation, test
and the target scale never move with the training-set size (C-D3 — this is what makes the
overparameterisation study comparable). `n_train` above the available pool **raises**; it
never clamps silently, which is why the top point of `rebuttal_overparam_mnist_scan` is
N = 50,000 and not 60,000.

### 1.2 Evaluation protocol

* **Selection uses validation only.** The best configuration of a method is chosen by the
  seed-mean final validation loss under the binding rule (§5). **Test metrics are outcomes and
  are never selection inputs** — `analysis/lib/scan_analysis.py`, `analysis/lib/analysis_helpers.py`
  and `tools/select_best.py` assert this.
* One `evaluate()` for val, test and `train_eval`: **example-weighted** (token-weighted for
  language models), in **eval mode** with every submodule's previous training flag restored
  afterwards, and it **mutates no buffer**.
* `train_eval` is a fixed training subset of `min(n_train, train_eval_size)` examples, chosen
  with `split_seed` and **shared by all optimizers** of a scan. `train_eval_size` is 10,000 by
  default, 1,000 on `exp_nanogpt_speedrun`, 200 blocks on GPT-2. Index 0 of the `train_eval`
  curve is the untrained model.
* `eval_batch_size` is a separate knob and is **never** the training batch size: 2,048 by
  default, 256 on nanoGPT, 16 on GPT-2.
* Curve keys per record: `train` (example-weighted online loss), `val`, `val_acc`, `test`,
  `test_acc`, `train_eval`, `epoch_times`, `train_times` (sum of synchronised batch times).
  Summaries: `val_final`, `val_best`, `val_best_index`, `val_last3_mean`, `train_eval_final`,
  final `test` / `test_acc`, `peak_gpu_mem_mb`.
* Step-based evaluation (`eval_every_steps`) is used only by `exp_gpt2_small_comparison`
  (every 500 steps → 26 points in `val_step` / `test_step`, indexed by `eval_step_idx`),
  because that scan is a single 13,125-step epoch and epoch-level curves would be two points.
* Accuracy is always argmax over the **raw** outputs (softmax leaves it unchanged).

### 1.3 BatchNorm policy

`bn_mode: batch | frozen` (C-E2) replaces the old `gram_freeze_norm_stats`, which survives
only as a deprecated alias. Any model with norm layers that carry running statistics
**requires** an explicit `bn_mode`.

* `bn_mode: batch` — used by **both CIFAR headline scans and Fig-5, for every optimizer**:
  train with batch statistics; running statistics are updated from the *training* batch
  exactly once per optimizer step; evaluation uses the running statistics and mutates
  nothing. Sven's capture / jvp / delta passes run under a no-write context (implemented by
  temporarily setting `track_running_stats=False`, **not** by `.eval()`, which would change
  the normalisation and hence the Gram matrix), and the step performs one explicit
  `torch.no_grad()` train-mode forward that updates the statistics.
* `bn_mode: frozen` — norm layers in eval mode always, for every optimizer. Used by
  `exp_finetune_cifar_smallN` (parked).
* HIG refuses `bn_mode: batch` on a model with running statistics, so HIG is MLP-only in this
  campaign.
* The MLP and transformer models have no norm layer with running statistics, so the recorded
  `bn_mode` is immaterial there; it differs between families on those scans (cosmetic: MLP,
  nanoGPT and GPT-2 svd rows record `frozen` while their `standard` rows record `batch`).

Why it matters: with frozen running statistics the stats never update from their
initialisation, the net is effectively un-normalised, and Sven collapsed to ~28% validation
accuracy (probe 2026-09-12). The legacy "Sven ≈ Adam on CIFAR" result came from evaluating
Sven with **validation-batch** statistics; it does not survive the corrected protocol.

### 1.4 Seeding and data order

* `model_seeds` are explicit per scan (`[1000..1004]` toy, `[2000..2004]` polynomial,
  `[3000..3004]` MNIST, `[4000..4004]` CIFAR, `[5000..5004]` nanoGPT, `[6000]` GPT-2,
  `[4000..4002]` Fig-5, `[7000..7002]` fine-tune).
* `loader_seed` in a config is the **base** loader seed; the effective seed is
  `derive_loader_seed(loader_seed, model_seed)` and is recorded as `effective_loader_seed`.
  It is identical across optimizers for a given model seed, so a paired comparison at fixed
  model seed shares both the initialisation and the data order (C-S2 / C-A6), and the seed
  band is no longer initialisation variance alone.
* Data order: `EpochPermutationSampler(n, loader_seed, batch_size, drop_last=True)`; the epoch
  permutation is `torch.randperm(n, generator=Generator().manual_seed(mix(loader_seed, e)))`,
  and `sampler.batch_indices(n, loader_seed, batch_size, step)` reconstructs any batch offline
  — which is what makes the checkpoint-based spectrum analysis possible.
* Per-run stochasticity uses `seed_for_run(model_seed, run_id) = model_seed ^ crc32(run_id)`.
* Confirmation seeds are the tuning seeds **+ 100..104** (e.g. `[1100..1104]`), disjoint from
  the tuning seeds and with fresh data orders.
* `data_seed` fixes the synthetic target; `split_seed` fixes which x's land in pool / val /
  test; on MNIST and CIFAR `split_seed: 1234` is fixed and independent of model, loader and
  subsample seeds.

### 1.5 Sven backend and what a step costs

Every Sven run in the campaign uses the exact **Gram** backend (`use_gram: true`): the
`M × M` Gram matrix `G = J Jᵀ` accumulated in float64, then one `torch.linalg.eigh(G)`, and
only *afterwards* the `rtol` cut (`sigma > rtol · sigma_max`) and the rank cap `k`. `M` is the
number of Jacobian rows = batch size / micro-batch size. The update is identical to the
classic truncated-SVD pipeline. `svd_mode: torch` is recorded but irrelevant under
`use_gram` — there is no randomized SVD on this path.

> **Cost statement (C-Z1 / F28 / F35).** Under the Gram backend a Sven step costs **the
> capture plus one `M × M` eigendecomposition, independent of `k`**. `k` and `rtol` decide how
> many eigenpairs are *inverted* after the decomposition; they do not change the cost of the
> decomposition, and the flat step-time-vs-`k` curve in the timing pass is the expected
> result, not a measurement artefact. The familiar `O(k N |D|)` figure describes **only the
> classic randomized-SVD path**, which no campaign run uses. Under `gram_capture: full` the
> capture additionally materialises the `(B, P)` Jacobian, so *memory* — not the
> eigendecomposition — is the binding constraint at ResNet scale.

Capture modes actually used: `hooks` (one weighted backward; all MLP, nanoGPT and GPT-2
scans), `full` (one `jacrev` over all parameters → the dense `(B, P)` Jacobian; all
ResNet scans, because batch-statistics BatchNorm cannot go through the hooks path). `chunked`
is `full` with more than one group and is **not** used by any campaign scan.

`empty_cache` is `False` for every campaign run (C-T3). The GPU probe measured the per-step
`torch.cuda.empty_cache()` at **841 → 186.7 ms/step** on CIFAR Sven with
`gram_capture: full` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (23.3 GB reserved)
— a 4.5× speedup and the entire source of step-time variance (p10–p90 span 643 ms with it on,
1.7 ms with it off). `empty_cache=True` stays reachable for reproducing legacy runs.
Chunked capture only ever helped because it gave `empty_cache` less to churn: with it off,
`full` (186.5 ms) beats `cf0.5` (192.0) and `cf0.25` (205.0).

#### The step-time / memory profile: v2 (before) → v3 (of record)

The numbers of record are **`profile_results_v3`** (one exclusive A100-SXM4-80GB,
torch 2.9.1+cu128, 720/720 configurations). `profile_results_v2` (720/720) is kept only as
the **"before"**: it
was measured with the per-step `empty_cache()` above and without `expandable_segments`, so its
Sven step times are upper bounds and must never be mixed into a v3 table except as this
comparison. v3 records the settings in every file (`env.alloc_conf`, `env.sven_empty_cache`,
`env.bn_mode`, `provenance.git_sha`); v2 recorded none of them, which is why it could not be
told apart from a correct pass by reading the records. Set points, `study=methods`:

| arch | Sven variant | step ms v2 → v3 | ×    | peak MB v2 → v3 | ×    |
|---|---|---|---|---|---|
| cifar_resnet18 | Gram, full `J`   | 842.0 → **188.0**  | 0.22 | 22978 → 22976 | 1.00 |
| cifar_resnet18 | Gram, chunked    | 470.2 → **207.9**  | 0.44 | 6923 → 6921   | 1.00 |
| cifar_resnet18 | classic rand-SVD | 1126.2 → **1040.7**| 0.92 | 23033 → 23030 | 1.00 |
| nanogpt        | Gram, hooks      | 83.2 → **59.5**    | 0.72 | 615 → 614     | 1.00 |
| nanogpt        | Gram, full `J`   | 257.4 → **229.2**  | 0.89 | 2893 → 2892   | 1.00 |
| mnist          | Gram, full `J`   | 19.4 → **14.3**    | 0.74 | 49 → 48       | 0.99 |
| mnist          | classic rand-SVD | 15.5 → **10.1**    | 0.65 | 48 → 48       | 0.99 |
| toy_1d / polynomial | Gram (3 variants) | within ±2%     | ~1   | 19 → 19       | 1.00 |
| toy_1d / polynomial | classic rand-SVD | within ±2%      | ~1   | 22.2 → 21.4   | 0.96 |

Two things to read out of it. **The fix buys step time and costs essentially no memory** —
`empty_cache` changes *when* the caching allocator hands blocks back, not how many are live at
once, so the mem ratio is 1.00 wherever it is the only thing that changed. It is **not exactly
1.00 everywhere**, and the exceptions are the second fix, not the first: `expandable_segments` is
a process-wide allocator setting, so the classic randomized-SVD set point on toy-1D and polynomial
falls 22.2 → 21.4 MiB (**×0.965**, the same ×0.964 that SOAP shows on those two architectures),
MNIST full `J` and classic both read ×0.99, and among the baselines Muon on CIFAR moves ×0.82.
None of these is a Sven-specific effect and none changes a cost conclusion — they are a smaller
allocator footprint at the same live set. And **it changes the capture ranking on CIFAR**:
under v3 `full`
(188.0 ms) beats `chunked` (207.9), the same order the GPU probe found, whereas v2 had
`chunked` ahead by 1.8×. The Gram cost statement above is unaffected; the profile is a cost
measurement, not a selection input, and the campaign's own timing pass puts the selected
CIFAR-CE Sven step at 177.6 ms with 22.96 GB peak, consistent with the v3 profile.

**The baselines are the control**, since only Sven ever called `empty_cache`: median `v3/v2`
step time is 0.993 (toy-1D), 1.010 (polynomial), 0.969 (MNIST), 0.999 (nanoGPT), 1.008
(CIFAR). 5 of 63 baseline set points moved by more than 10%, each accounted for in
`analysis/notebooks/profiling/profile_overview.ipynb`: two L-BFGS entries whose strong-Wolfe line search switches
regime mid-measurement (p90/p10 up to 9.1), one whose polynomial run diverged in one pass, and
Muon on MNIST (×0.85) and CIFAR (×1.20) — real, and allowed, because
`PYTORCH_CUDA_ALLOC_CONF` is a process-wide setting and the second fix is not Sven-only. Eight
of the 720 configurations changed `status` between the passes; none is a cost result (same
notebook).

Measured NPROC per GPU (processes sharing one device without losing throughput), from the
pre-campaign GPU probe:

| device | CIFAR Sven | CIFAR first-order | MLP Sven (hooks) | MLP first-order | L-BFGS (mi≥2) | nanoGPT Sven |
|---|---|---|---|---|---|---|
| A100-SXM4-80GB | 1 | 4+ | 12 (still rising) | = cores | 4–6 | 3 |
| A100-40GB MIG 3g.20gb | 1 (at `cf0.5` only) | — | 6 (toy-1D 8) | 6 | 6–8 | 1 |

Two probe findings that shaped the campaign: **CUDA-event timing is unusable for co-tenancy
decisions** (on a time-sliced GPU the wait before the first event is invisible, so `step_ms`
*fell* while wall time rose); and **cheap-MLP step time is set by host CPU load** — the same
MNIST-Adam config is 1.13 ms on a quiet node and 4.58 ms on one with 4/4 GPUs busy. Both are
why the timing pass runs one job per scan with a calibration microbenchmark at each end.

### 1.6 Losses

`loss:` in a config is one of:

* `mse` — scalar regression (toy-1D, polynomial).
* `label_regression` — the paper's Sec. 4 definition, per-sample `‖f(x) − onehot(y)‖²` on the
  **raw** outputs, no softmax. Every `*labelReg*` / `*labelRegression*` scan uses it.
* `ce` — cross-entropy.
* `lm_ce` — per-token cross-entropy (nanoGPT, GPT-2).
* `brier` — `‖softmax(f(x)) − onehot(y)‖²`. It exists in the registry with its own config
  (`mnist_scan_brier`) but **no campaign scan uses it**, and its numbers are not comparable
  with `label_regression`.

**Signed residuals.** For `mse` on a one-output model Sven builds its Jacobian rows from the
signed residual `sign(r)|r|^κ` rather than `loss^(κ/2)`, for every κ (`signed_residual: true`,
the default; it is recorded on every row). The per-row sign cancels in the pseudo-inverse, so
the update is the paper's; what changes is that `κ < 2` no longer NaNs at `r = 0`. Multi-output
losses (label-reg, CE, LM-CE) have no scalar signed residual and keep the `loss^(κ/2)` rows —
and, as measured, κ = 1 completes there too (§7).

### 1.7 Muon grouping rule

Recorded per run as `muon_variant`. Applied to the parameter-bearing modules in registration
order (`experiments/experiment_code/optim_factory.muon_param_groups`, C-B5):

1. `nn.Embedding` weights → **AdamW** (a lookup table is not a linear map between feature
   spaces).
2. Among the recognised projection layers (`nn.Linear`, any `nn.ConvNd`), the **last** one is
   the output head → **AdamW**.
3. Every remaining projection weight is hidden: 2-D → **Muon**, >2-D conv kernels →
   **MuonConv** (flatten-to-2D, validated exact against stock Muon). This *includes the input
   layer*.
4. Everything else — all 1-D parameters (biases, LayerNorm / BatchNorm weights) and any ≥2-D
   parameter not belonging to a recognised projection layer — → **AdamW**.

`adjust_lr_fn = "match_rms_adamw"` (Moonshot's rule) so one learning rate means the same
effective step size in both groups; this makes Muon's effective step 1.1–13.6× larger than at
the same nominal lr without it, which is why the Muon/MuonW lr grids reach lower than the
others. Muon and MuonW on a ResNet are legitimate only because of MuonConv: the pre-campaign
"Muon on ResNet18" was AdamW plus one `fc` layer (F13).

**Weight decay is not swept** (C-B4 dropped by the 09-18 scope update). Each name runs at its
own default, one setting, in every scan: `AdamW` 0.01, `MuonW` 0.1, `Adam` and plain `Muon`
0.0. AdamW's `run_id` carries `_wd0.01` and MuonW's `_wd0.1`, so they cannot collide with the
superseded wd = 0 runs.

### 1.8 Record schema and results layout

`schema_version: 2` on every new record. Each run writes a light `<run_id>.jsonl` (curves,
hyperparameters, provenance, summaries) plus `diag/<run_id>.npz` (per-step arrays). Schema 2
adds `status` (`ok|diverged|oom|error`), `error`, `diverged_at_step`, `run_hash`, a provenance
block (both repos' SHA + dirty flag, torch, CUDA, GPU name, host, SLURM job id, start/end),
`n_train`, `n_val`, `n_test`, `steps_per_epoch`, `n_params`, `actual_param_fraction`,
`muon_variant`, `split_seed`, `effective_loader_seed`, `eval_batch_size`, `bn_mode`,
`checkpoint_policy`, `svd_spectra_schedule`, and the final `test` / `test_acc`.

Layout under `{root}/{scan}/`: `*.jsonl`, `diag/`, `ckpt/{run_id}.pt` (+
`ckpt/init_mseed{seed}.{model_generation}.pt`), `done/`, `claims/`, `started/`, `manifest/`,
`configs/` (the resolved Hydra config per job), and — if ever needed — `_stale/{hash8}/` and
`attempts/`. **No scan in this campaign produced a `_stale/` or `attempts/` directory**
(verified: neither exists under any of the 43 result directories).

`SV3_RESULTS_ROOT` selects the root (default `experiment_results`, in the campaign a symlink
to `$SV3_SCRATCH/results`); the runner and the analysis both honour it.

**Dedup and scheduling.** `run_id` stays human-readable. A zero-byte marker
`{scan}/done/{run_id}.{hash8}.{status}` is written **last**, after ckpt/npz/jsonl; a run is
skipped iff a marker with the current `hash8` and a status in `{ok, diverged}` exists. A
marker with a different `hash8` moves that run's files to `{scan}/_stale/{old_hash8}/` and
re-runs. Work is claimed dynamically: `os.open(O_CREAT|O_EXCL)` on
`{scan}/claims/{run_id}.claim` with a 60 s heartbeat; a claim older than 10 min is stale and
may be taken over. `run_hash` reads per-spec values and never the grid *lists*, which is why
the extension round could append points to a grid and have the 15,735 finished runs dedup out
untouched.

---

## 2. Headline convergence scans (P0)

The central claim: in the over-parametrised regime Sven drives the loss down faster than
first- and second-order baselines. All seven carry the three result-dependent passes of §4.

Optimizer sets as submitted: **STD** = Adam, AdamW, SGD, SGDm, RMSprop, Muon, MuonW, SOAP,
Shampoo, KFAC (family `standard`); **L-BFGS** and **PolyakSGD** have their own grids and
families; **JD** = torchjd UPGrad with an Adam inner optimizer; **HIG** = half-inverse
gradients. SGDm is `torch.optim.SGD(momentum=0.9)` (C-B2); plain SGD without momentum is not a
2026 baseline and is kept only alongside it.

Throughout this document **`ok / attempted` means the recorded `status`**, i.e. `attempted`
minus the runs whose record says `diverged`. It is the *lifecycle* count. The analysis
definition of failure is wider — `analysis/style.is_diverged` also drops finite blow-ups — and
it is the one selection and every table use. Where the two differ materially (toy-1D most of
all: 1,688 recorded `ok` but 1,481 usable) §7 gives both, per scan and per method. A table in a
notebook must show the wide `finished / attempted`, not the column below.

| scan | data (loss) | model · params | Sven grid | baselines | seeds · epochs | runs (recorded ok / attempted) |
|---|---|---|---|---|---|---|
| `toy_1d_scan` | toy-1D 10k/10k/10k (mse), B = 32 | MLP 1→[16,16,16]→1 GELU · **593** | k ∈ {1,2,4,8,16,32} × lr ∈ {.01,.02,.05,.1,.5,1} × rtol ∈ {1e-6…1e-2} = 900 | STD on lr ∈ {1e-5,3e-5,1e-4,1e-3,1e-2,1e-1,3e-1,1} (400) · L-BFGS lr ∈ {.01,.03,.1,.5,1} × mi {1,2,3} × hs {2,5,10} (225) · Polyak (5) · JD 6 lr (30) · HIG 7 lr × 6 tau (210) | 5 · 20 | **1688 / 1770** |
| `polynomial_scan` | random polynomial 10k/10k/10k (mse), B = 32 | MLP 6→[16,16,16]→1 · **673** | k ∈ {1…32} × lr ∈ {.05,.1,.5,1} × rtol ∈ {1e-5,1e-4,1e-3,1e-2,3e-2,1e-1} = 720 | STD on 10 lrs 1e-6…1 (500) · L-BFGS lr ∈ {.1,.5,1,2,4} × mi × hs (225) · Polyak (5) · JD (30) · HIG 5 lr × 6 tau (150) | 5 · 20 | **1386 / 1630** |
| `mnist_scan_labelRegression` | MNIST 50k/10k/10k (label-reg), B = 64 | MLP 784→[32,32,32]→10 · **27,562** | k ∈ {1,2,4,8,16,32,48,64} × lr ∈ {.05,.1,.5,1} × rtol ∈ {1e-4…1e-1} = 640 | STD 8 lrs (400) · L-BFGS lr ∈ {.1,.5,1} × mi × hs (135) · Polyak (5) · JD (30) · HIG (150) | 5 · 20 | **1231 / 1360** |
| `mnist_scan_ce` | MNIST 50k/10k/10k (cross-entropy), B = 64 | same · **27,562** | k ∈ {1…64, 8 values} × lr ∈ {.05,.1,.5,1} × rtol ∈ {1e-4,1e-3,1e-2,1e-1,3e-1} = 800 | STD 8 lrs (400) · L-BFGS 5 lrs × mi × hs (225) · Polyak (5) · JD (30) · HIG (150) | 5 · 20 | **1490 / 1610** |
| `cifar10_resnet_scan_labelRegression` | CIFAR-10 45k/5k/10k (label-reg), B = 128 | ResNet18, torch.func-compatible BN · **11,181,642** | k ∈ {64,128} × lr ∈ {.1,.5,1} × rtol ∈ {1e-4,1e-3,1e-2}, κ = 2 = 90 · `gram_capture: full`, `bn_mode: batch` | Adam, AdamW, SGD, SGDm, RMSprop, Muon, MuonW, SOAP on 7 lrs 1e-5…3e-1 (280) · L-BFGS 5 lrs × mi × hs (225) · Polyak (5) | 5 · 20 | **572 / 600** |
| `cifar10_resnet_ce_scan` | CIFAR-10 45k/5k/10k (cross-entropy), B = 128 | same · **11,181,642** | k ∈ {64,128} × lr ∈ {.02,.05,.1,.5,1} × rtol ∈ {1e-4,1e-3,1e-2}, κ = 2 = 150 | same 8 optimizers on 9 lrs 1e-5…3.0 (360) · L-BFGS (225) · Polyak (5) | 5 · 20 | **740 / 740** on the config grid, **+45 / 45** off-grid `rtol` runs (§5) = **785 records** |
| `exp_nanogpt_speedrun` | tiny-shakespeare char (lm_ce), B = 64 | nanoGPT 4L·4H·128d, block 128, untied, dropout 0 · **826,368** | k = 64 × lr ∈ {.05,.1,.5,1} × rtol 1e-3 = 20 · `gram_capture: hooks` | AdamW, Muon, MuonW, SOAP on 6 lrs 1e-5…3e-3 (120) | 5 · 50 | **140 / 140** |

`steps_per_epoch` as recorded: 312 (toy, polynomial), 781 (MNIST), 351 (CIFAR), 108 (nanoGPT).

**JD and HIG were submitted on the four MLP headline scans and on neither CIFAR scan.** Both
CIFAR configs still *enumerate* 20 JD + 80 HIG runs (200 in total); no launcher has ever run
`mode=jd` or `mode=hig` on a ResNet, `p3_cifar_jd_hig` in `campaign/plan_campaign.yaml` keeps
all four items `enabled: false`, and HIG on ResNet18 would need a full per-sample Jacobian
(~50 GPU-h per scan). The keys are kept so reviving the study is one flag, at the cost that
`mode=all` on either CIFAR config would claim 100 unbudgeted runs — which is why the plan
launches them as `mode=svd` / `mode=standard`. This is the source of the "100 runs of the grid
unfinished" warning on both CIFAR entries of `bench/best_configs.json`.

### The grid-extension round (2026-09-19, +7,480 runs)

Every axis was extended only where `tools/reconcile.py` flagged the reported optimum **on a
grid edge** over the 15,735 finished runs, and nowhere the optimum came back interior. Three
rules decided what did *not* grow: Sven's `k` is never extended (`k = B` is a method boundary,
not a grid edge, and it is the reported optimum on six scans); the CIFAR Sven grids are left
alone at ~0.5 GPU-h per run; and an axis stops where the next point is meaningless rather
than merely expensive (`rtol` is a *relative* cut, so at 1.0 only the leading direction
survives). The per-scan table with each closed edge is in `campaign/grid_counts.md`. The
round only appends to grid lists, so existing `run_id`s are unchanged and the finished runs
were skipped by the done-marker dedup.

---

## 3. Reviewer-requested scans

### 3.1 Dataset-level over-parameterisation, P > N (P1, R1 crux)

Subsample the training set to trace performance across the P/N = 1 boundary. The synthetics
run full-batch (B = N, `batch_size: ${n_data}`, 200 epochs → 1 step/epoch at the top point);
MNIST keeps the paper's minibatch setup. `n_data` is a launcher axis, one job item per value,
and enters the `run_id` via `result_id_fields: [mlp_width, n_data]`.

| scan | N sweep | Sven grid | baselines | runs (ok / attempted) |
|---|---|---|---|---|
| `rebuttal_overparam_toy_1d_scan` | 150, 300, 600, 1200 (full batch) | k/B ∈ {.25,.5,1} × lr ∈ {.05,.1,.5,1} × rtol ∈ {1e-4,1e-3,1e-2} = 720 | STD on the headline 8-lr list (1600) · L-BFGS 5 lrs × mi × hs (900) · Polyak (20) | **3186 / 3240** |
| `rebuttal_overparam_polynomial_scan` | 170, 340, 675, 1350 (full batch) | k/B ∈ {.25,.5,1} × lr ∈ {.01,.02,.05,.1,.5,1} × rtol ∈ {1e-4,1e-3,1e-2,3e-2,1e-1} = 1800 | STD (1600) · L-BFGS (900) · Polyak (20) | **4282 / 4320** |
| `rebuttal_overparam_mnist_scan` | 2.5k, 5k, 10k, 20k, 40k, **50k** (B = 64) | k ∈ {16,32,48,64} × lr ∈ {.1,.5,1} × rtol 1e-4 = 360 | STD (2400) · L-BFGS 3 lrs × mi × hs (810) · Polyak (30) | **3188 / 3600** |

### 3.2 Batch-size sensitivity (P1, R2 Q1)

| scan | grid | runs (ok / attempted) |
|---|---|---|
| `rebuttal_batchsize_polynomial_scan` | B ∈ {8,16,32,64,128,256}; Sven k/B = 1 × lr ∈ {.05,.1,.5,1} × rtol ∈ {1e-5,1e-4,1e-3,1e-2} = 480 · STD on the headline 8-lr list (2400) · L-BFGS **shape pinned** at mi 3 / hs 2, 5 lrs (150; the O5 cut, 810 → 150 runs) · Polyak (30) · 5 seeds · 20 epochs | **2804 / 3060** |

### 3.3 Fig-5 at scale — ResNet parameter fraction (P1, R1 Q2)

| scan | grid | runs |
|---|---|---|
The scan holds **two Sven configurations**, 15 runs each, both on disk under their own
`run_id`s. The **selected** one is the result; the legacy set point is the comparison.

| configuration | grid | runs |
|---|---|---|
| **selected** (of record) — `p2_cifar_fig5_selected`, 09-21 | pf ∈ {.05,.1,.25,.5,1} at **k = 128, lr = 0.5, rtol = 1e-3**, κ = 2, `gram_capture: full`, `bn_mode: batch`, 3 seeds (4000–4002), 20 epochs | **15 / 15** recorded `ok`, **0 diverged** under `is_diverged` |
| legacy set point — `p1_cifar_fig5`, 09-19 | the same pf grid at **k = 64, lr = 1.0, rtol = 1e-3**, otherwise identical | **15 / 15** recorded `ok`; **2 of the 15 are diverged under `is_diverged`** (pf 0.05 and pf 0.1, both at that configuration's single lr = 1.0: final val 244 and 19.8 against `val[0]` ≈ 2.0 and 1.3) |

Masks are resampled every step and `actual_param_fraction` is recorded per run. The legacy
blow-ups are finite, so they carry `status: ok` and are dropped only by the wider analysis rule
— a legacy Fig-5 point at pf ≤ 0.1 therefore rests on 2 surviving seeds of 3, which any seed
band on that figure must show as `finished / attempted`. At the selected configuration every
point rests on 3 of 3.

**Caveat, now closed.** The legacy set point is the *pre-Gram classic* best, still flagged
`SET POINT, STILL TENTATIVE` in the config, and was **not** re-derived from the BN-fixed
headline scan (`cifar10_resnet_scan_labelRegression`, Sven optimum k = 128, lr = 0.5,
rtol = 1e-3) before the first launch. Its single lr = 1.0 is also the lr where *all five* of that
headline scan's Sven divergences sit (5 of its 90 Sven runs). **Read precisely** — the earlier
wording overstated this: all five sit at lr = 1.0 **and rtol = 1e-4** (2 at k = 128, 3 at k = 64;
final val 31–124 against `val[0]` 1.3–2.0, all carrying `status: ok`, i.e. finite blow-ups caught
only by the wider rule). At the legacy Fig-5's own `rtol = 1e-3` the headline grid has **0 of 10**
Sven divergences at lr = 1.0, and 0 at every other rtol; the whole lr = 1.0 column is clean except
that one (lr, rtol) cell. So the confound the legacy figure carried is weaker than "lr = 1 is
unstable here": lr = 1.0 is the *only* lr that blows up anywhere, but it does so only in
combination with the tightest truncation, which Fig-5 did not use. What the legacy figure could
still not separate is "masking at fraction `f` hurts" from "lr = 1, never re-tuned per `f`, is a
bad operating point" — its own two blow-ups at pf 0.05 and 0.1 are the evidence for the second
reading. The whole figure was therefore re-run at the selected configuration, and **it separates the two**: the divergences are gone, and the
low-`f` collapse survives (seed-mean final val / test accuracy at pf 1 → 0.05:
**0.471 → 0.503 → 0.876 → 2.40 → 4.94** and **69.1% → 68.8% → 67.8% → 25.6% → 19.2%**, against
chance 10%). Cost is unchanged by the re-point, as the Gram statement of §1.5 requires:
187.5 ms/step and 22.96 GB at pf = 1, and masking makes the *full-Jacobian capture no cheaper*
— the slowest point is pf = 0.5 at 453.2 ms (×2.42) and memory falls only to ×0.51 at
pf = 0.05 while its step is ×1.71. `analysis/notebooks/large_models/cifar_analysis.ipynb` §6 draws both
configurations, reading which is selected out of `bench/best_configs.json` rather than having
it typed in.

### 3.4 κ (residual-exponent) ablation (P3, R1)

| scan | grid | runs |
|---|---|---|
| `mnist_kappaScan_labelRegression` | κ ∈ {1,2,3} × lr ∈ {.125,.25,.375,.5,.75,1,1.5} × k ∈ {32,64} × rtol 1e-4, 5 seeds, 20 epochs | **210 / 210** recorded `ok` (0 `status: diverged`); **10 of the 210 are diverged under `is_diverged`** — all at the top of the lr grid (lr 1.5: 7, lr 1.0: 2, lr 0.75: 1; k = 64: 8, k = 32: 2; κ = 1: 6, κ = 2: 4, κ = 3: 0) |

The lr list is chosen so that **three** effective steps `2·lr/κ` — 0.25, 0.5 and 1.0 — are
realised by all three κ values (C-X1 / F29). Without that the study is an lr sweep in
disguise, because in the untruncated full-row-rank solve the κ step is exactly `2/κ` times the
κ = 2 step. k = 32 is the truncating arm (k < B = 64), k = 64 is k = B.

Of the 10 blow-ups, **nine sit outside the three matched effective steps** (κ = 1 at lr 0.75 /
1.0 / 1.5 → effective 1.5 / 2.0 / 3.0; κ = 2 at lr 1.5 → 1.5). Exactly one lands in a matched
cell: κ = 2, lr 1.0, k = 64, i.e. effective step 1.0, which therefore has 4 usable seeds of 5.
The κ-vs-effective-step comparison itself is unaffected; the lr sweep beyond it is not.

### 3.5 Micro-batch scaling (P3, Sven only)

Aggregating samples into micro-batches shrinks the Gram row dimension M — the memory /
update-rank trade-off. `mode=svd`, 5 seeds, 20 epochs.

| scan | grid | runs |
|---|---|---|
| scan | grid | runs (recorded `ok` / attempted · wide-diverged) |
|---|---|---|
| `toy_1d_microbatch_scan` | mb ∈ {1,2,4,8,16,32} × k = 32 × lr ∈ {.05,.1,.5,1} × rtol 1e-3 | **120 / 120** · 0 |
| `polynomial_microbatch_scan` | mb ∈ {1,2,4,8,16,32} × k = 32 × 4 lrs × rtol 1e-3 | **120 / 120** · **2** (lr 1.0) |
| `mnist_microbatch_labelreg_scan` | mb ∈ {1,2,4,8,16,32,64} × k = 64 × 4 lrs × rtol 1e-4 | **140 / 140** · **1** (lr 1.0) |
| `mnist_microbatch_ce_scan` | mb ∈ {1,…,64} × k/B = 1 × 4 lrs × rtol 1e-1 | **140 / 140** · 0 |

### 3.6 Parameter fraction (P3, Sven only)

| scan | grid | runs (recorded `ok` / attempted · wide-diverged) |
|---|---|---|
| `toy_1d_paramfrac_scan` | pf ∈ {.1,.25,.5,.75,1} × k = 32 × 4 lrs × rtol 1e-3 | **99 / 100** · **14** (lr 0.5: 5, lr 1.0: 9) |
| `polynomial_paramfrac_scan` | same shape, rtol 1e-3 | **85 / 100** · **20** (lr 0.5: 7, lr 1.0: 13) |
| `mnist_paramfrac_labelreg_scan` | pf ∈ {.1…1} × k = 64 × 4 lrs × rtol 1e-4 | **86 / 100** · **24** (lr 0.5: 10, lr 1.0: 14) |
| `mnist_paramfrac_ce_scan` | pf ∈ {.1…1} × k = 64 × 4 lrs × rtol 1e-1 | **91 / 100** · **12** (lr 0.5: 5, lr 1.0: 7) |

Masking is `elementwise` (the default `mask_mode`), which is the only mode that can split a
BatchNorm layer. The runs that are *absent from the `ok` count* are Sven divergences that
**raised** — the masked-Gram `eigh` guard — and they are all at lr ≥ 0.5. Under the wider
analysis definition (§7) so is every additional blow-up: **none of these four scans has a
single Sven failure of either kind below lr 0.5**, so the pf trend at lr 0.05 / 0.1 is
complete, and only the two top lrs lose seeds. These are the only scans in the campaign where
Sven raises at all (39 runs in total).

### 3.7 GPT-2 small (P1, re-admitted 2026-09-19)

| scan | data | model | grid | runs |
|---|---|---|---|---|
| `exp_gpt2_small_comparison` | FineWeb-edu GPT-2 BPE, 210,000 blocks of 1024 tokens train / 200 val / 200 test from disjoint documents | GPT-2 small, untied embeddings, dropout 0, `block_size: 1024` · **163,109,376** | Sven k/B = 1 (= **k = B = 16**) × lr ∈ {.02,.05,.1,.5,1} × rtol 1e-3 (5) · AdamW, Muon, MuonW, SOAP on 6 lrs 1e-5…3e-3 (24) · **1 seed, 1 epoch** = 13,125 steps at B = 16, `eval_every_steps: 500` | **29 / 29**, all `ok`, 0 diverged (the last SOAP run at lr 3e-3 finished 09-20 23:13Z) |

Measured on the lane's own hardware (A100-SXM4-80GB, NPROC 1, one GPU per run): Sven
**2.52 s/step** → 9.2 h/run and **36,050 MB** peak allocated; AdamW 776 ms, Muon 794 ms,
SOAP 1,126 ms. Its 29 runs are ~122 GPU-h — a third of the whole campaign's GPU-h floor in
0.1% of the runs, which is why it has its own plan file, `campaign/plan_gpt2.yaml`.

**Framing this scan honestly:** one seed, one epoch, an untuned `k = B = 16`, and Sven at
~3× the time per run. Nothing here can be seed-averaged, so a tie is unresolvable without
doubling the scan. Sven's best validation loss is at lr 0.1 (interior optimum) against
baselines around 3.8–4.0 — a **negative** scaling data point, to be reported as such with the
untuned-`k`/`B` caveat. The headline numbers belong to the analysis notebooks
(`analysis/notebooks/large_models/gpt2_analysis.ipynb`), not here.

---

## 4. The three result-dependent passes (phase 5)

Seven headline scans × three companion configs = 21 companions, all complete
(**1,575 runs: 425 timing + 425 diag + 725 confirm**, verified). Each pass runs **only the
per-method selected configurations** (~11–15 methods × 5 seeds per scan) — a launcher
selection driven by `bench/best_configs.json`, not a config grid, which is why
`campaign/grid_counts.md` excludes them from the campaign total.

| pass | config | what it changes | why |
|---|---|---|---|
| **B — timing** | `<scan>_timing.yaml` | nothing (inherits the parent); `checkpoints: none`, one job per scan, one run at a time, NPROC = 1 | wall-clock numbers that are not inflated by shard contention. Submitted by `bench/submit_timing_phase5.sh`, **not** by the launcher, with an Adam-MLP calibration microbenchmark at the start and end of each job so host-load contamination is detectable after the fact |
| **C — diagnostics** | `<scan>_diag.yaml` | `svd_info: full`, `svd_spectra_schedule: {dense_first: 1000, every: 20}`, the checkpoint ladder for **every** family (`checkpoints: log` with `checkpoints_svd` cleared; **`epochs` on CIFAR — 21 states, 0.94 GB per run as measured — because the `log` ladder over 7,020 steps would be 34 states / ~1.5 GB**) | full spectra, `utr`, `update_norm`, `resid_norm`, `sv_min_kept`, `sv_noise_floor`, and checkpoints for the offline Jacobian analysis |
| **D — confirmation** | `<scan>_confirm.yaml` | `model_seeds` = base + 100..104; `checkpoints: final`, `svd_info: summary`; toy and polynomial additionally get **3 data seeds** via `result_id_fields: [mlp_width, data_seed]` | the numbers the paper reports, read off seeds the hyperparameters were **not** tuned on. The tuning-vs-confirmation gap is the selection optimism |

Passes B and C are **grid-identical to the parent**, with the same `run_id`s *and* the same
`run_hash`es: neither `checkpoints` nor `svd_spectra_schedule` nor `svd_info` enters
`run_hash` (they decide what is *written*, never the trajectory), so a diagnostics run is
bit-for-bit the scan's run with more of it recorded. Pass D is deliberately *not*
grid-identical (different seeds; `data_seed` in the run_id so the three replicates do not
collide under one `run_id` with two hashes).

Verified with `bench/check_timing_join.py`: **425 timing runs, 425 joined onto their scan by
`run_id` and `run_hash`, 0 missing.** Trajectory agreement (relative difference in final
validation loss) is exact on nanoGPT (median 0.00e+00) and at the 1e-9 / 1e-7 level on
polynomial / toy; the deviations are confined to hardware-sensitive methods — see §8.

**The passes are not failure-free.** 20 of the 1,575 records carry `status: diverged` and 21
are diverged under `is_diverged`; the §7 table covers only the 21 campaign scans, so these are
listed here instead. The selected configuration of a method can diverge on fresh seeds even
though it survived the tuning seeds it was selected on — which is exactly the kind of
selection optimism pass D exists to expose.

| companion | diverged / attempted (status · wide) | who |
|---|---|---|
| `polynomial_scan_confirm` | **13 / 15 · 13 / 15** | L-BFGS, the selected `lr 1.0, mi 1, hs 2` — 4 of 5 model seeds diverged at data seed 2000, 4 of 5 at 2001, **5 of 5** at 2002 |
| `polynomial_scan_diag` | 2 / 5 · 2 / 5 | L-BFGS, same config (mseeds 2000, 2004) |
| `polynomial_scan_timing` | 2 / 5 · 2 / 5 | L-BFGS, same config, same two seeds |
| `toy_1d_scan_confirm` | 1 / 15 · 1 / 15 | L-BFGS `lr 0.03, mi 3, hs 10` (data seed 1001) |
| `mnist_scan_ce_confirm` | 1 / 5 · 1 / 5 (L-BFGS `lr 0.5`) + 0 / 5 · 1 / 5 (SOAP `lr 1e-4`, a finite blow-up) | L-BFGS, SOAP |
| `mnist_scan_labelRegression_timing` | 1 / 5 · 1 / 5 | SOAP (the `lr 0.01, mseed 3001` record of §8) |
| every other companion | 0 · 0 | — |

**Polynomial L-BFGS is the one case that makes a confirmation mean unusable:** 13 of its 15
confirmation records diverged, so its "mean over 5 model seeds × 3 data seeds" rests on **2**
runs. Report it as `2 / 15`, or report L-BFGS on polynomial as a failure; do not print a
two-run mean without the fraction beside it. This is the method, not the pass: the same
configuration was already `n_ok: 3, n_diverged: 2` on the tuning seeds (it won anyway, because
every other L-BFGS configuration on that scan was worse or less eligible — L-BFGS is 204 / 225
diverged on `polynomial_scan`, §7), and it reproduces at 3 / 5 in both grid-identical passes.

---

## 5. Selection, and what "best" means

The **selection of record** is `bench/best_configs.json`, produced by `tools/select_best.py`
(schema 2, `rule_name: "full"`, generated 2026-09-20T03:58:14-0400). The binding rule:

> **eligible** (more than half of the scan's expected seeds finished — not diverged, not
> missing) → **fewest diverged seeds** → **lowest seed-mean final VALIDATION loss** over the
> non-diverged runs. Diverged = `analysis/style.is_diverged`. Test metrics are never read.

`analysis/lib/scan_analysis.py` implements the same rule and agrees with it on all 85 (scan,
method) picks. **`tools/reconcile.py`'s quick "best config per method" table omits the
fewest-diverged tier and differs on 6 of the 85 — never quote it.** `select_best.py` proves
that each entry's `overrides` string expands to exactly the five seed `run_id`s (and run
hashes) of the configuration it names; `tools/gen_phase5_plan.py` then copies those overrides
verbatim into `campaign/plan_phase5.yaml`.

`is_diverged` is the single definition used everywhere: recorded `status == "diverged"`, **or**
a non-finite final train/val value, **or** a final validation loss more than `DIVERGED_FACTOR`
(10×) above `val[0]`, the untrained value — a finite blow-up, which the param-fraction scans
produce at 1e7…1e15 without ever reaching a NaN. It is **wider than the recorded `status`** —
2,553 runs against 1,402 — and §7 gives both counts per scan and per method; a record's
`status` alone must never stand in for it. Every analysis table shows `finished / attempted`
under *this* definition, and seed bands are mean ± 1 std (ddof = 1), labelled "± 1 std over
seeds", never a confidence interval.

Sven's selected configuration per headline scan, with the flagged grid edges:

| scan | selected Sven config | seed-mean final val | edges |
|---|---|---|---|
| `toy_1d_scan` | k = 32, lr = 0.01, rtol = 1e-4 | 2.873e-07 | `lr:EDGE-LOW`, `k:EDGE-HIGH` |
| `polynomial_scan` | k = 16, lr = 0.5, rtol = 3e-2 | 0.10948 | interior |
| `mnist_scan_labelRegression` | k = 64, lr = 0.5, rtol = 1e-3 | 0.052150 | `k:EDGE-HIGH` |
| `mnist_scan_ce` | k = 32, lr = 0.5, rtol = 3e-1 | 0.114936 | `rtol:EDGE-HIGH` |
| `cifar10_resnet_scan_labelRegression` | k = 128, lr = 0.5, rtol = 1e-3 | 0.480345 | `k:EDGE-HIGH` |
| `cifar10_resnet_ce_scan` | k = 128, lr = 0.5, **rtol = 0.3** | 1.35515 | `k:EDGE-HIGH`, `rtol:EDGE-HIGH` |
| `exp_nanogpt_speedrun` | k = 64, lr = 0.1, rtol = 1e-3 | 1.72363 | interior |

`k:EDGE-HIGH` always means `k = B`, a **method boundary rather than a grid edge**, and is not
extended. **Three** open edges remain:

* `toy_1d_scan`'s Sven `lr` at 0.01, the new bottom point after the extension round — accepted.
* `mnist_scan_ce`'s Sven `rtol` at 0.3, which is the top of that scan's `rtol` grid
  [1e-4, 1e-3, 1e-2, 1e-1, 3e-1] — flagged `rtol:EDGE-HIGH` in `bench/best_configs.json`'s own
  `edges` dict for that pick, and the same knob whose CIFAR-CE twin was judged worth ~20 GPU-h
  to extend, with "MNIST-CE prefers 0.1–0.3" as the stated rationale.
  **Decision: accepted, not extended**, on two grounds. First, the pick at that edge is an
  **exact three-way tie**: k = 32, 48 and 64 at lr = 0.5 / rtol = 0.3 all give the identical
  seed-mean 0.11493559425553576 over 5/5 seeds, because at rtol = 0.3 the rtol-rank (3.4 of
  B = 64, `analysis/notebooks/spectra/comparisons.ipynb` §2) truncates far below every one of those k, so k does
  not bind and the tie-break on smallest k picks 32; the optimum is a plateau in k, not a point.
  Second, the direction of the edge is a *weaker* truncation, and the CIFAR-CE extension is the
  measurement of what lies past it: pushing `rtol` from 1e-2 to 0.3 there bought 3.4 % of
  validation loss and did not change the rank (10th of 11 either way). Extending MNIST-CE would
  therefore buy a number, not a conclusion. The honest statement is that **Sven's `rtol` optimum
  on both cross-entropy scans sits at the largest truncation the grid offers**, i.e. Sven does
  best on CE when it keeps the fewest singular values — which is §7's story, not a tuning gap.
* `cifar10_resnet_ce_scan`'s Sven `rtol`, which was at 1e-2 while MNIST-CE prefers 0.1–0.3.
  **Extended, and it moved the pick** — `p2_cifar_ce_rtol`: `rtol ∈ {0.03, 0.1, 0.3}`
  restricted to `k = 128` and
  `lr ∈ {0.05, 0.1, 0.5}` = **45 runs, ~20 GPU-h**, against 150 runs / ~70 GPU-h for the full
  `2 k × 5 lr × 3 rtol` version. It reconciles clean (45/45 `ok`, 0 diverged) and re-selection
  over the union of both override groups moved Sven from **k = 128, lr = 0.1, rtol = 1e-2
  (1.40281)** to **k = 128, lr = 0.5, rtol = 0.3 (1.35515)**, 5/5 seeds and 0 diverged either
  way — 3.4% of validation loss, and the rank is 10th of 11 on the tuning seeds either way
  (RMSprop 1.33404 above, PolyakSGD 1.78242 below). `rtol` is **still on the high edge**: 0.3
  is the top of the extended set. The extension is **off-grid by design** — those `rtol` values
  are deliberately *not* added to the config's `rtol` list, so run counts,
  `campaign/grid_counts.md` and the frozen goldens in `tests/golden/` do not move; the
  consequence is that `tools/select_best.py` only sees them when the extension is named as a
  second `--groups` (read the `p2_cifar_ce_rtol` note in `campaign/plan_campaign.yaml` first —
  the obvious command overwrites the other six scans' selections). Because the pick moved,
  `bench/best_configs.json` was re-spliced for that one scan (`selection_provenance` records
  which scan was selected when), `campaign/plan_phase5.yaml` regenerated, and the **timing,
  diag and confirm passes re-run for CIFAR-CE Sven only** (5 runs each, beside the old pick's,
  which keep their own `run_id`s; `analysis/headline.selected_runs` matches on hyperparameter
  values, so the old pick's runs are never mixed in).

Headline numbers come from the **confirmation** seeds, with the tuning-seed numbers shown
beside them, so that the selection optimism is visible rather than absorbed.

---

## 6. Checkpoints and spectra logging

`checkpoints: none | final | epochs | log` per scan, with `checkpoints_svd` as an override for
the svd family (C-L3). `log` = steps {0, 1, 2, 4, 8, …} plus every epoch end; step 0 is the
initialisation. A checkpoint file is
`torch.save({"step": [...], "epoch": [...], "state": [state_dict on CPU, fp32, weights AND
buffers]})`. Under the `final` policy each model seed also gets a shared
`ckpt/init_mseed{seed}.{model_generation}.pt` (verified: 5 of them under
`cifar10_resnet_ce_scan/ckpt/`, one per seed); under `log` the initialisation is step 0 of
every run's own ladder, so no separate file is written.

Storage **as measured on disk** (`stat` per file, `du -sb` per directory, 2026-09-20), not as
estimated before launch. `campaign/grid_counts.md` §"checkpoint storage" is a pre-launch
estimate and runs about 2.1× low on the MLP ladders; where the two disagree, the column below
is the file system.

| family | policy in the scan | measured | notes |
|---|---|---|---|
| toy / polynomial MLP (593–673 params) | `checkpoints: log` | **195–203 KiB** per completed run (34 states, ~5.9 KB each; median 199,445 B toy / 208,037 B polynomial). `toy_1d_scan/ckpt` 342 MB over 1,770 files, `polynomial_scan/ckpt` 312 MB over 1,630 | the whole point is offline spectra. `grid_counts.md`'s "~94 KB" is 2.1× low: it counts only the fp32 parameter bytes (593 × 4 × 35 ≈ 83 KB) and misses the per-state `torch.save` / buffer overhead, which roughly doubles it |
| the two 200-epoch full-batch overparam scans | `log` | **1.14–1.20 MiB** per run (201 states, ~6.0–6.3 KB each; median 1,190,679 B toy / 1,256,613 B polynomial); 3.79 GB + 5.36 GB per `ckpt` dir | same cause; `grid_counts.md`'s "~563 KB" is 2.1× low (its state count, ~209, is right) |
| MNIST MLP (27,562 params) | `final` + `checkpoints_svd: log` | **3.79 MiB** per svd run (3,979,860 B mean over the 640 svd files), 112 KiB per `final` run; `mnist_scan_labelRegression/ckpt` = **2.63 GB** total (2.55 GB svd ladders + 81 MB final/init) | the "~9.2 GB" in `grid_counts.md` is the campaign-wide svd-ladder total over ~2,490 runs, not one scan |
| nanoGPT (826,368 params) | `epochs` | **175 MiB** per run (51 states; median 183,277,320 B), 25.7 GB for the scan | — |
| ResNet18 (11.18M) | `final` | **42.7 MiB** per run (44.82 MB); 33.8 GB for `cifar10_resnet_ce_scan` | `log` would keep every state in RAM until flush — 34 states ≈ **1.5 GB** per run — which is why the diag pass uses `epochs` (21 states, **0.94 GB**, measured: 941,005,669 B median over the 55 diag runs) |
| GPT-2 small (163.0M) | `final` | **670 MiB** per run (702,845,528 B); 20.4 GB for the scan | — |

Campaign-wide: **243 GB** of checkpoints and **4.6 GB** of `diag/*.npz` under
`experiment_results/` (248 GB for the whole root). The two CIFAR diag companions are the
largest single directories at 51.8 GB each.

Sven diagnostics: `svd_spectra_schedule: {dense_first: 200, every: 20}` by default,
`{dense_first: 1000, every: 20}` in the diag pass. On a logged step the optimizer appends
`step`, `svs` (**all M singular values, before the k/rtol cut**), `utr` (= `Uᵀr`, all M),
`update_norm` (the norm of the *applied* change, including lr), `resid_norm`, `sv_min_kept` and
`sv_noise_floor`; scheduled arrays carry their own `svs_step` index array. `num_nonzero_svs`
stays per step. Legacy `sv_min` is **not** written for schema-2 records: it changed meaning
between old records (smallest kept SV) and new ones (σ_B, i.e. numerical noise), so the name
was retired rather than reused.

Records predating 2026-09-10 carry the spectra inline in the JSONL; `analysis/lib/style.py` reads
both layouts. Legacy spectra were truncated at `rtol` and their tails are survivorship
averages — **nothing in the fresh analysis may read them**.

---

## 7. Failures, and how they are recorded now

**The most important change from the pre-campaign documentation: a failing run is no longer a
missing file.** A failing run is *recorded* and a done-marker is placed, so it is counted,
never silently absent, and never retried. `stop_on_nonfinite: true` everywhere. Across the
whole campaign: **0 `oom`, 0 `error`, 0 poisoned runs, 0 stale hashes.**

Two paths produce `status: "diverged"`, and they differ in what else the record carries:

* A non-finite training-batch loss raises `DivergedError(step)` → `status: "diverged"` with
  `diverged_at_step` set. **1,147 records**, all with a step (SGDm 203, SGD 152, L-BFGS 726,
  SOAP 40, HIG 19, Sven 7).
* `_classify_failure` (`experiments/experiment_code/generic_scan.py:498–510`) also maps
  `torch.linalg` exceptions and the "no singular value above rtol" / "failed to converge" /
  "ill-conditioned" messages to `diverged`, with `diverged_at_step: None`. **275 records**
  (19% of the 1,422): **243 K-FAC `_LinAlgError`** ("linalg.eigh: The algorithm failed to
  converge because the input matrix is ill-conditioned") and **32 Sven `RuntimeError`**
  ("SvenGram: no singular value above rtol · sigma_max (sigma_max=0); the Gram matrix is
  zero", the masked-Gram guard in the param-fraction scans).

That second path is why `status: "error"` is 0 campaign-wide — the failures are classified, not
absent. It also means **any analysis that histograms `diverged_at_step` silently drops 275
diverged runs**; use `status` for "did it fail", `diverged_at_step` only for "when".

### Two counts, and which one is which

`status == "diverged"` is the **lifecycle** count: it decides whether a run is retried.
`analysis/style.is_diverged` is the **analysis** count: recorded-diverged **or** a non-finite
final value **or** a finite blow-up (final val > 10 × `val[0]`). It is wider, it is the one
that governs selection (§5) and every analysis table, and campaign-wide it is **2,553** against
1,402 recorded. The gap is not a rounding detail — on `toy_1d_scan` it is 289 vs 82, and 194 of
those 207 extra runs are Sven. **No robustness claim may be read off the status field alone.**

Both counts per scan, with the per-method breakdown given as the **wide** count and the
recorded count in parentheses where they differ (verified by reading every record's first line
on 2026-09-20; the wide totals reproduce `tools/select_best.py`'s own
"N diverged run(s), excluded from seed means" lines):

| scan | recorded | **wide** / attempted | by method (wide, recorded in parens) |
|---|---|---|---|
| `toy_1d_scan` | 82 | **289** / 1770 | **Sven** 194/900 (0 recorded), L-BFGS 52/225, K-FAC 19/40 (6 recorded), HIG 11/210, SGD 5/40, SGDm 5/40, SOAP 3/40 |
| `polynomial_scan` | 244 | **260** / 1630 | L-BFGS 204/225, K-FAC 21/50 (15), SGDm 12/50, SGD 8/50, SOAP 4/50 (2), **Sven** 4/720 (0), HIG 3/150, RMSprop 3/50 (0), MuonW 1/50 (0) |
| `mnist_scan_labelRegression` | 129 | **149** / 1360 | L-BFGS 68/135, **K-FAC 40/40**, SOAP 9/40 (3), HIG 8/150 (5), SGDm 8/40, RMSprop 5/40 (0), SGD 5/40, MuonW 3/40 (0), AdamW 1/40 (0), Muon 1/40 (0), **Sven** 1/640 (0) |
| `mnist_scan_ce` | 120 | **131** / 1610 | L-BFGS 70/225, **K-FAC 40/40**, SOAP 9/40 (2), SGDm 5/40, MuonW 3/40 (0), SGD 3/40, RMSprop 1/40 (0) — **Sven 0/800** |
| `cifar10_resnet_scan_labelRegression` | 28 | **33** / 600 | SGDm 15/35, SGD 13/35, **Sven** 5/90 (0) |
| `cifar10_resnet_ce_scan` | 0 | **11** / 740 | RMSprop 8/45 (0), SOAP 1/45 (0), Muon 1/45 (0), SGDm 1/45 (0) — **Sven 0/150** |
| `exp_nanogpt_speedrun` | 0 | **0** / 140 | — |
| `rebuttal_overparam_toy_1d_scan` | 54 | **202** / 3240 | **Sven** 111/720 (0), RMSprop 33/160 (0), SGDm 32/160 (31), SGD 25/160 (23), Muon 1/160 (0) |
| `rebuttal_overparam_polynomial_scan` | 38 | **369** / 4320 | **Sven** 252/1800 (0), RMSprop 49/160 (0), SGD 20/160, SGDm 18/160, Muon 12/160 (0), Adam 10/160 (0), SOAP 7/160 (0), MuonW 1/160 (0) |
| `rebuttal_overparam_mnist_scan` | 412 | **649** / 3600 | L-BFGS 225/810, K-FAC 127/240 (87), SOAP 70/240 (19), RMSprop 55/240 (0), SGDm 51/240, MuonW 32/240 (0), SGD 30/240, Adam 23/240 (0), Muon 23/240 (0), AdamW 10/240 (0), Shampoo 2/240 (0), **Sven** 1/360 (0) |
| `rebuttal_batchsize_polynomial_scan` | 256 | **375** / 3060 | K-FAC 94/240 (55), L-BFGS 88/150, SGDm 59/240 (58), SGD 45/240, **Sven** 35/480 (0), RMSprop 25/240 (0), SOAP 16/240 (10), MuonW 7/240 (0), Muon 5/240 (0), AdamW 1/240 (0) |
| `mnist_kappaScan_labelRegression` | 0 | **10** / 210 | **Sven** 10/210 (0) — §3.4 |
| `toy_1d_microbatch_scan` | 0 | **0** / 120 | — |
| `polynomial_microbatch_scan` | 0 | **2** / 120 | **Sven** 2/120 (0) |
| `mnist_microbatch_labelreg_scan` | 0 | **1** / 140 | **Sven** 1/140 (0) |
| `mnist_microbatch_ce_scan` | 0 | **0** / 140 | — |
| `toy_1d_paramfrac_scan` | 1 | **14** / 100 | **Sven** 14/100 (1) |
| `polynomial_paramfrac_scan` | 15 | **20** / 100 | **Sven** 20/100 (15) |
| `mnist_paramfrac_labelreg_scan` | 14 | **24** / 100 | **Sven** 24/100 (14) |
| `mnist_paramfrac_ce_scan` | 9 | **12** / 100 | **Sven** 12/100 (9) |
| `rebuttal_fig5_cifar_paramfrac_scan` | 0 | **2** / 15 | **Sven** 2/15 (0) — the **on-grid** legacy set point only; the 15 off-grid runs at the selected configuration add **0** (§3.3) |

(`cifar10_resnet_ce_scan` is shown on its **740 on-grid** runs, so the row does not move with
the off-grid `rtol` extension. That extension is now complete — all **45** of its Sven runs are
`ok` and **none** is diverged under either definition — so both counts stand as written over
the scan's 785 records.)

The 21 phase-5 companions add 20 recorded / 21 wide divergences of their own — §4.

Four things worth naming:

1. **K-FAC dies deterministically on MNIST.** 40 of 40 runs on *both* MNIST headline scans end
   `diverged` — `torch.linalg.eigh` fails to converge on the rank-deficient Kronecker-factored
   Fisher — so K-FAC has **no eligible configuration** on either scan and is absent from their
   entries in `bench/best_configs.json` (14 methods, not 15). This is on-message rather than a
   harness bug: K-FAC only "exists" in the over-parametrised regime via damping, i.e. a biased,
   non-min-norm update. Only the full-suite MLP configs include K-FAC; the CIFAR and
   Sven-only scans do not.
2. **Sven raises in the param-fraction scans alone — but it blows up in 16 of the 21 scans,
   and only the wide count sees it.** *As recorded*, Sven fails in the param-fraction scans
   alone (39 of those 400 runs, all at lr ≥ 0.5, where the `eigh` on the masked Gram hits a
   zero Gram matrix once training has already diverged) and `status: "diverged"` is 0 in every
   other scan. *Under `is_diverged`* — the definition that
   governs selection and every table — Sven has **688 of its 7,825 on-grid
   campaign runs diverged**, spread over 16 of the 21 scans (39 recorded, 688 wide):

   | scan | Sven wide / attempted | where |
   |---|---|---|
   | `rebuttal_overparam_polynomial_scan` | **252 / 1800** | rtol 1e-4 (119), 1e-3 (114), 1e-2 (19); lr 0.5 (103) / 1.0 (136) |
   | `toy_1d_scan` | **194 / 900** | rtol 1e-6 (94), 1e-5 (70), 1e-4 (30) — **none at 1e-3 or 1e-2**; every lr from 0.01 to 1.0; k ≥ 4 |
   | `rebuttal_overparam_toy_1d_scan` | **111 / 720** | all at rtol 1e-4; lr 0.5 (51) / 1.0 (60) |
   | `rebuttal_batchsize_polynomial_scan` | **35 / 480** | rtol 1e-5 (22), 1e-4 (11), 1e-3 (2); lr 0.5 / 1.0 |
   | the four param-fraction scans (toy / poly / MNIST-LR / MNIST-CE) | **14 / 20 / 24 / 12 of 100 each** | all at lr ≥ 0.5 — the only scans where Sven also *raises* |
   | `mnist_kappaScan_labelRegression` | **10 / 210** | lr ≥ 0.75, κ ∈ {1, 2} — §3.4 |
   | `cifar10_resnet_scan_labelRegression` | **5 / 90** | all at lr 1.0, rtol 1e-4 |
   | `polynomial_scan` | **4 / 720** | lr 1.0; rtol 1e-5 (3), 1e-3 (1) |
   | `polynomial_microbatch_scan` | **2 / 120** | lr 1.0 |
   | `rebuttal_fig5_cifar_paramfrac_scan` | **2 / 15** | lr 1.0, pf 0.05 and 0.1 — §3.3; **0 / 15** at the re-pointed lr 0.5 |
   | `mnist_scan_labelRegression`, `mnist_microbatch_labelreg_scan`, `rebuttal_overparam_mnist_scan` | **1 each** | the same k = 64, lr 1.0, rtol 1e-4 point |
   | `mnist_scan_ce`, `cifar10_resnet_ce_scan`, `exp_nanogpt_speedrun`, `toy_1d_microbatch_scan`, `mnist_microbatch_ce_scan` | **0** | the only scans where Sven has 0 under *both* definitions |

   The pattern is mechanistic and worth stating rather than hiding: Sven blows up when `rtol`
   is at or near the **bottom** of the grid, i.e. when the cut is loose enough to invert
   near-noise Gram directions, and the risk grows with lr. On `toy_1d_scan` the selected
   configuration (rtol 1e-4) sits one point above the worst band, and the two largest rtol
   values produce no divergence at all. **Sven has 0 divergences in either sense on
   `mnist_scan_ce`, `cifar10_resnet_ce_scan`, `exp_nanogpt_speedrun` and two of the four
   micro-batch scans** — that is the defensible version of the claim.
3. **κ = 1 no longer fails.** `mnist_kappaScan_labelRegression` is 210/210 *recorded* `ok` with
   κ ∈ {1, 2, 3} — no NaN, no raise, at any κ. Multi-output label-regression rows keep the
   `loss^(κ/2)` form and complete; the κ < 2 NaN at zero residual was a *scalar*-MSE problem
   and is removed there by `signed_residual`. Under the wide rule 10 of the 210 blow up
   (κ = 1: 6, κ = 2: 4, κ = 3: 0), all at lr ≥ 0.75 and nine of them outside the three matched
   effective steps — see §3.4. The pre-campaign claim that the CIFAR CE κ scan was "missing
   κ = 1 and 1.5" refers to scans that are **cut** (§9) and no longer describes anything that
   ran.
4. **The SOAP high-LR CUDA deadlock did not recur.** All 40 SOAP runs per MNIST headline scan
   have records (2–3 of them `diverged`), and no job hung; `stop_on_nonfinite` plus the claim
   queue mean a bad point can no longer block a shard.

---

## 8. Known non-reproducibility across GPU types

Scans ran mostly on A100-40GB **MIG 3g.20gb** slices; the timing, diagnostics and confirmation
passes ran partly or wholly on **A100-SXM4-80GB** (CIFAR, nanoGPT and GPT-2 ran on the 80GB
part throughout). `bench/check_timing_join.py` compares each timing run's final validation
loss with its scan twin at identical `run_id` **and** `run_hash`:

The `median` / `max` columns are over **all** methods of the scan, so they are aggregates over 14
(MLP), 11 (CIFAR) or 5 (nanoGPT) optimizers and must not be read as any one method's number —
which is exactly the mistake corrected below. **Sven's own** median / max is given beside them.

| scan | pairs | GPU pair | median (all methods) | max (all methods) | **Sven** median / max |
|---|---|---|---|---|---|
| `exp_nanogpt_speedrun` | 25 | same (80GB→80GB) | 0.00e+00 | 0.00e+00 | **0.0e+00 / 0.0e+00** |
| `polynomial_scan` | 75 | MIG-40→80GB | 8.7e-09 | 2.5e-01 (Muon / MuonW at lr 0.01) | 6.0e-09 / 1.7e-08 |
| `toy_1d_scan` | 75 | MIG-40→80GB | 4.5e-07 | 2.1e-01 (HIG at 1e-9-scale losses) | 1.9e-05 / 2.6e-05 |
| `mnist_scan_ce` | 70 | MIG-40→80GB | 5.7e-05 | 3.1e-01 (L-BFGS line search, SOAP) | **1.40e-02 / 3.80e-02** |
| `mnist_scan_labelRegression` | 70 | MIG-40→80GB | 8.3e-03 | **2.0e+02** (one SOAP run at lr 0.01) | 1.4e-04 / 1.3e-03 |
| `cifar10_resnet_scan_labelRegression` | 55 | **same** (80GB→80GB) | 2.5e-02 | 2.3e-01 | 1.1e-02 / 8.3e-02 |
| `cifar10_resnet_ce_scan` | 60 | **same** (80GB→80GB) | 3.0e-02 | 2.1e-01 | 1.4e-02 / 3.2e-02 |

**Corrected 2026-09-21.** An earlier version of this section said "Sven and the plain first-order
MLP runs reproduce" and listed Sven among the bit-reproducible methods. **That is false, and the
per-scan medians above hide it** — each is an aggregate over 14 methods. Joining every timing run
to its scan twin at identical `run_id` **and** `run_hash` and splitting by method and by GPU pair
gives the honest statement:

* **No method is bit-reproducible across GPU types.** Sven's own cross-GPU deviation is
  method- *and* scan-dependent: `polynomial_scan` median 6.0e-09 / max 1.7e-08 · `toy_1d_scan`
  1.9e-05 / 2.6e-05 · `mnist_scan_labelRegression` 1.4e-04 / 1.3e-03 · **`mnist_scan_ce`
  1.40e-02 / 3.80e-02** over its 4 MIG-40GB → A100-80GB pairs (0.1165537 in the scan vs
  0.1121249 in the timing pass). That 3.8 % is *larger* than the ±1.4 % confirm-vs-tuning gap
  the MNIST-CE table reports for Sven, and comparable to Sven's own 4.5 % confirmation seed std
  — so a single MNIST-CE Sven seed is not a reproducible number across hardware.
* **The "plain first-order" half fails too.** On `mnist_scan_labelRegression` the cross-GPU
  median is Adam 1.01e-02, AdamW 5.05e-03 and RMSprop 2.38e-02 — all of them **larger** than
  Sven's 1.4e-04 on the same scan. Only SGD and SGD+momentum are near-zero there (1.6e-07 and
  2.0e-08), and even they are not exactly 0.
* **Same-GPU pairs on the MLP scans and nanoGPT are exactly bit-identical** (relative deviation
  `0.000e+00`), which is the control that makes the attribution to hardware rather than to a
  lifecycle bug: the same code, seeds and hash on the same GPU model reproduce to the last bit.
  Where a MIG-slice run was re-timed on an A100-80GB it does not.
* **The two CIFAR scans are the exception to that control**, and the mechanism there is *not*
  cross-type: both passes ran wholly on A100-SXM4-80GB, and Sven's same-GPU pairs still deviate
  by a median 1.1e-02 (label regression) and 1.4e-02 (CE), with every baseline in the same 1e-02
  range. That is within-hardware cuDNN nondeterminism (kernel selection and atomics), so **no
  CIFAR seed-level result is bit-reproducible at all**, on any GPU.
* Worst offenders overall remain **Muon (bf16 Newton–Schulz)**, up to 2.5e-01 on polynomial,
  **L-BFGS** (`strong_wolfe` line search, up to 3.1e-01 on MNIST-CE) and **SOAP / Shampoo / HIG**
  (eigendecompositions at 1e-9-scale losses, up to 2.1e-01 on toy).

The hashes match in every case, so the two numbers are of the same experiment, and none of this
affects the validity of the timing measurements. The paper sentence is therefore not "Muon and
L-BFGS are not reproducible" but: **seed-level results are not bit-reproducible across GPU types
for any method, the magnitude is method- and task-dependent (1e-08 to 1e-01), and on CIFAR they
are not bit-reproducible even on identical hardware** — which is why every number reported is a
mean over 5 seeds with its spread, never a single run.

**One caveat to carry forward:** on `mnist_scan_labelRegression` a single SOAP run
(`lr 0.01, mseed 3001`) ends at 0.901 in the scan and **184.8** in the timing pass. That is
not a rounding difference — the timing copy of that run took a different trajectory. Timing
claims must therefore come from `<scan>_timing` **step times**, and loss values from the scan
or confirmation passes, never from a timing record. Check the calibration lines in the timing
job logs (`$SV3_SCRATCH/campaign/logs/`, `bench/calibrate_step.py`) for host-load
contamination before quoting MLP step times at all.

---

## 9. Cut, parked, and stale

**Cut from the campaign** (their configs are not included in this repository):

| config | why |
|---|---|
| `exp_critbatch_mnist`, `exp_critbatch_nanogpt` | user cut; the critical-batch study confounds retained rank with batch size (F23) and crossings were measured once per epoch |
| `cifar10_resnet_kappaScan_labelReg`, `cifar10_resnet_ce_kappaScan` | 1 seed, tentative set point; the κ story runs on MNIST at 5 seeds with a matched-effective-step grid |
| `cifar10_resnet_paramFrac_scan_labelReg`, `cifar10_resnet_ce_paramFrac_scan` | superseded by `rebuttal_fig5_cifar_paramfrac_scan` at 3 seeds |
| `mnist_scan_brier` | a different objective; not comparable with label-regression and not needed for any claim |
| the second grid-extension round; confirmation seeds beyond the headline scans | time |

**Parked to the extension phase, config edits standing:** `exp_finetune_cifar_smallN`
(408 runs, `bn_mode: frozen`, SGDm, the extended lr grid; all eight `p3_finetune` items in
`campaign/plan_campaign.yaml` are `enabled: false`). Its 192 legacy runs trained BatchNorm on
250–2000 images and are superseded.

**Also parked:** the 200 enumerated JD + HIG runs on the two CIFAR configs (§2).

**Never run, and not answered anywhere — reviewer 2's question 2b.** *"Can you give results on a
classifier model using CCE loss with Sven used only on intermediate layers? Can you also compare
your result with Muon?"* The second half is answered on every scan (Muon and MuonW are baselines
throughout, and §1.7 documents the grouping rule that decides which tensors Muon gets). **The
first half has no experiment.** It is easy to mistake three things for it, and none of them is it:
§1.7's Muon grouping rule describes what *Muon* is applied to, not Sven; `param_fraction` (§3.6,
and Fig-5 in §3.3) masks a random subset of parameter **elements**, resampled every step, not a
layer group — so `paramfrac_analysis` and Fig-5 do not answer it either; and `mnist_scan_ce` /
`cifar10_resnet_ce_scan` run Sven on *all* parameters. The mirrored ablation — Sven restricted to
exactly the hidden-matrix group Muon receives (hidden 2-D weights and flattened convs), with
AdamW on embeddings / head / 1-D tensors — was never configured or launched, and
this is a scope gap inherited from the plan rather than an execution failure. It would need a new optimizer-group option in the runner, not
just a grid edit. **State it as an open question, not as answered.**

`profile_*.yaml` is out of scope for the scan machinery — see `README.md` for the profiler.

**Superseded whole-directory:** the pre-campaign (legacy) results are frozen read-only at
`experiment_results_legacy_2026-09-18/`; the campaign's `experiment_results/` was started empty.

---

## 10. The polynomial target was redefined

`RandomPolynomialDataset` (C-D1, `experiments/datasets/all_datasets.py`) is **all** monomials
of total degree ≤ `degree` in `num_vars` variables — constant and linear included, 210 of them
for degree 4 in 6 variables — with the per-variable factors **multiplied**, x ~ N(0, I), and
the coefficient of monomial `m` drawn as `N(0,1) / sqrt(E[m(x)²])` where
`E[m²] = Π_j (2p_j − 1)!!` ("variance-normalised monomials"), so that no single degree
dominates the target.

The pre-2026-09 generator is kept, correctly named, as **`AdditiveCubicDataset`**: despite
being called a "random polynomial" it *added* the per-variable factors rather than multiplying
them, and its `product(range(d), …)` enumeration excluded every exponent ≥ d, so the constant
and all linear monomials were missing (185 tuples for degree 4 in 6 variables).

**Consequence for the analysis: the polynomial benchmark is a different target from the one in
the submitted paper.** Absolute losses are not comparable across the redefinition; only ranks
are. All polynomial scans in this campaign (`polynomial_scan`,
`rebuttal_overparam_polynomial_scan`, `rebuttal_batchsize_polynomial_scan`,
`polynomial_microbatch_scan`, `polynomial_paramfrac_scan`) use the new generator at
`degree: 4`, `input_dim: 6`, `data_seed: 2000`.

---

## 11. How to reproduce the campaign

```bash
# 0. environment: SV3_REPO, SV3_SCRATCH, SV3_DATA_ROOT (README.md "Environment"); tests green
.venv/bin/python -m pytest tests/ -q          # sv3
(cd sven && ../.venv/bin/python -m pytest tests/ -q)

# 1. freeze BOTH repos at HEAD (never run jobs from the working tree)
SNAP=$(tools/deploy_snapshot.sh | tail -1)
#   -> $SV3_SCRATCH/deploy/<sha8>_<sha8>/ with DEPLOY_INFO.json
#      and an experiment_results symlink to the real results root

# 2. dry-run, then submit, one work list at a time (P0 -> P1 -> P3)
.venv/bin/python tools/launch_campaign.py campaign/plan_campaign.yaml --snapshot "$SNAP" --phase P0
.venv/bin/python tools/launch_campaign.py campaign/plan_campaign.yaml --snapshot "$SNAP" --list <name> --submit
#   the launcher refuses to mix snapshots for one list: drain or cancel first
#   GPT-2 has its own plan:  campaign/plan_gpt2.yaml

# 3. progress / completeness (the authority on "is this scan done")
.venv/bin/python tools/reconcile.py --all campaign/plan_campaign.yaml --no-best

# 4. once reconcile is clean: the selection of record
.venv/bin/python tools/select_best.py --require-complete   # -> bench/best_configs.json
#   --require-complete refuses a half-finished grid extension (3-of-5 seeds can win)
#   --compare-analysis also diffs against analysis/lib/scan_analysis.py's selection

# 5. regenerate the phase-5 plan from that selection and launch the passes
.venv/bin/python tools/gen_phase5_plan.py             # -> campaign/plan_phase5.yaml
.venv/bin/python tools/launch_campaign.py campaign/plan_phase5.yaml --list p5_diag_mlp_mig --submit
bench/submit_timing_phase5.sh                         # timing: dry run by default
.venv/bin/python tools/reconcile.py --all campaign/plan_phase5.yaml --phase P5

# 6. did the timing pass re-run the same runs?
.venv/bin/python bench/check_timing_join.py

# 7. analysis
./make_plots.sh
```

Workers set `PYTHONPATH=<snapshot>:<snapshot>/sven`, `SV3_RESULTS_ROOT`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `OMP_NUM_THREADS=1`. After a grid
extension, re-run steps 4 and 5 **in that order** before relaunching any pass.

Everything is resumable: re-running a launcher is safe (done-marker dedup), stale claims
expire after 10 minutes, and a run with ≥ 3 `oom`/`error` attempts under one hash is marked
poisoned and no longer retried (delete its `attempts/` files to retry).


---

## 12. Binding conventions and interfaces

The decisions the runner, the records and the analysis are held to. Where a docstring cites
"the contracts", this is the section it means.

### Decisions
* Polynomial target: all monomials of total degree ≤ d incl. constant and linear (210 for d=4,
  6 vars), factors MULTIPLIED, x ~ N(0,1), coefficient of monomial m = N(0,1) / sqrt(E[m(x)^2])
  with E[m^2] = prod_j (2 p_j - 1)!! ("variance-normalised monomials"; §10). The old generator
  is kept as `AdditiveCubicDataset`.
* CIFAR: 5 headline seeds; Fig-5 3 seeds.
* BatchNorm: `bn_mode: batch | frozen`. `batch` = train with batch statistics, running stats
  updated from the TRAINING batch exactly once per optimizer step, evaluation uses running
  stats and mutates nothing. `frozen` = norm layers in eval mode always (train and eval) for
  EVERY optimizer (the fine-tune study). Suppressing running-stat writes is done by temporarily
  setting `track_running_stats=False` on norm modules (train-mode forward then still normalises
  with batch statistics and writes no buffer), NOT by `.eval()`, which would change the
  normalisation and hence the Gram matrix; per-module flags are saved and restored.
* Sven wrappers under `bn_mode=batch`: every capture / jvp / delta pass runs under the no-write
  context and the step performs ONE explicit `torch.no_grad()` train-mode forward of the
  training batch that updates the stats (skipped when the model has no norm layer with running
  stats). `wrapper.evaluate()` = eval mode, side-effect-free. `evaluate_and_loss()` (used only
  by the `variable_k` line search, which no campaign config uses) = train-mode normalisation,
  no buffer write.
* Dedup: `run_id` stays human-readable. A zero-byte marker `{scan}/done/{run_id}.{hash8}.{status}`
  is written LAST (after ckpt, npz, jsonl). Skip iff a marker with the current hash8 and status
  in {ok, diverged} exists. A marker with another hash8 => move that run's jsonl/npz/ckpt/marker
  to `{scan}/_stale/{old_hash8}/` and run. One `os.listdir(done/)` per process start.
* Scheduling: dynamic claim files are the default; `+n_shards/+shard_id` slicing of the spec
  list stays as a fallback (`scheduler=static`). Claim = `os.open(O_CREAT|O_EXCL)` of
  `{scan}/claims/{run_id}.claim` containing host, pid, SLURM job id, start time; a heartbeat
  thread touches it every 60 s; a claim whose mtime is older than 10 min is stale and may be
  taken over. This is distinct from the `{run_id}.started` marker. A run with ≥ 3 `oom`/`error`
  attempts under one hash (`{scan}/attempts/{run_id}.{hash8}.{status}.{token}`) is `poisoned`
  and no longer retried (delete its `attempts/` files to retry).
* Results root: env var `SV3_RESULTS_ROOT` (default `experiment_results`), honoured by the
  runner and the analysis. `schema_version: 2` on every record.
* No weight-decay grid: AdamW runs at the torch default wd 0.01 and MuonW at 0.1, one setting
  each, in EVERY scan; Adam (wd 0) and plain Muon (wd 0) are separate named baselines.
* `empty_cache` defaults to `False` (the per-step `torch.cuda.empty_cache()` made CIFAR Sven
  4.5x slower, 841 -> 187 ms/step, and caused all of its step-time variance; `empty_cache=True`
  stays reachable for reproducing legacy runs). Launchers export
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. CIFAR keeps `gram_capture: full`, NPROC=1.
* MNIST train part is 50,000: `n_train` larger than the available pool RAISES, never clamps.
* Any model with norm running stats REQUIRES an explicit `bn_mode`; HIG refuses `bn_mode=batch`
  on such models (HIG is MLP-only in the campaign). The shared init checkpoint is
  `ckpt/init_mseed{seed}.{model_generation}.pt`.
* The runner records `actual_param_fraction = train_model.mean_actual_param_fraction` for
  svd-family runs (1.0 unmasked).

### Interfaces
**`experiments/experiment_code/grid.py`** (torch-free, importable in < 1 s): `RunSpec` frozen
dataclass: `family` in {svd, standard, lbfgs, polyak, jd, hig}; `run_id`; `model_seed`;
`loader_seed`; `batch_size`; `hparams: dict` (named grid point); `record_extra: dict`.
`expand_grid(rcfg, ...) -> list[RunSpec]`, seed-major, same order and byte-identical run_ids
as the legacy loops. `run_hash(spec, resolved_cfg) -> str` (sha256 hex; hash8 = first 8 chars).

**Training loops (`experiment_utils.py`)** — keyword arguments, all optional: `losses=None`
(caller-owned dict, filled in place so partial curves survive an exception), `test_loader=None`,
`train_eval_loader=None`, `eval_every_steps=None`, `checkpointer=None`, `log_schedule=None`
(callable step -> bool; svd loop sets `optimizer.log_this_step`), `stop_on_nonfinite=True`,
`bn_mode="batch"`. A non-finite training batch loss raises `DivergedError(step)`. Curve keys:
`train` (example-weighted online loss), `val`, `val_acc`, `test`, `test_acc`, `train_eval`
(index 0 = untrained), `epoch_times`, `train_times` (sum of synchronised batch times),
step-based evals in `val_step` / `test_step` with indices `eval_step_idx`.
`evaluate(forward_fn, per_sample_loss_fn, loader, device, *, track_acc=False, is_lm=False)
-> dict(loss, acc, n)` is example-weighted (token-weighted for lm), eval-mode, restores the
previous mode, mutates no buffer.

**Checkpointer (`experiments/experiment_code/checkpointing.py`)**: `Checkpointer(path, policy,
steps_per_epoch, num_epochs, rewrite_each_epoch=False)`; `.maybe_save(step, epoch, module)`
(called before the update of `step`, and step 0 = init); `.epoch_end(epoch, step, module)`;
`.flush()`. File = `torch.save({"step": [...], "epoch": [...], "state": [state_dict on CPU,
fp32, weights AND buffers]})`. Policies none|final|epochs|log (log = steps {0,1,2,4,8,...} +
every epoch end).

**Sampler (`experiments/experiment_code/sampler.py`)**: `EpochPermutationSampler(n, loader_seed,
batch_size, drop_last=True)` with `set_epoch(e)`; epoch permutation = `torch.randperm(n,
generator=Generator().manual_seed(mix(loader_seed, e)))`; `batch_indices(n, loader_seed,
batch_size, step) -> LongTensor` reconstructs any batch offline. `derive_loader_seed(
base_loader_seed, model_seed)` (identical across optimizers for a model seed).
`seed_for_run(model_seed, run_id) = model_seed ^ zlib.crc32(run_id.encode())`.

**Sven optimizer logging**: attribute `optimizer.log_this_step: bool` (default True). On logged
steps append to `svd_info`: `step`, `svs` (all M = B/microbatch values before the k/rtol cut),
`utr` (= U^T r, all M), `update_norm` (norm of the APPLIED change, includes lr), `resid_norm`,
`sv_min_kept`, `sv_noise_floor`. `num_nonzero_svs` stays per step.

**Datasets**: every class exposes `train_dataset`, `val_dataset`, `test_dataset` and records
`split_seed`. MNIST 50k/10k + official test; CIFAR-10 45k/5k + official test; Shakespeare
contiguous 80/10/10; token bins val/test/train from disjoint documents; toy/polynomial: fixed
pool of 10,000 + val + test from SEPARATE generators, targets normalised by pool mean/std,
`n_train` subsamples the pool.

### Config keys (top-level in each scan yaml; every key optional with these defaults)
`eval_batch_size: 2048` (validation/test/train_eval loaders; never the training batch size) ·
`train_eval_size: 10000` (fixed train subset, chosen with `split_seed`, shared by all
optimizers) · `checkpoints: final` (none|final|epochs|log) and `checkpoints_svd: null`
(override for the svd family, e.g. `log`) · `svd_spectra_schedule: {dense_first: 200, every: 20}`
· `bn_mode: batch` (batch|frozen; supersedes `gram_freeze_norm_stats`, a deprecated alias) ·
`empty_cache: false` · `stop_on_nonfinite: true` · `scheduler: claims` (claims|static) ·
`loader_seed` stays the BASE loader seed; the effective seed is
`derive_loader_seed(loader_seed, model_seed)`. run_id keeps its human-readable form (with
`_lseed{base}`); new behaviour is captured by the run hash, not by new suffixes, except
`bn_mode=frozen`, which appends `_bnfrozen`.

### Record (schema 2)
Adds: `schema_version`, `status` (ok|diverged|oom|error), `error`, `diverged_at_step`,
`run_hash`, a provenance dict (both repos' sha/dirty, torch, cuda, gpu, host, slurm_job_id,
start/end), `n_train`, `n_val`, `n_test`, `steps_per_epoch`, `n_params`,
`actual_param_fraction`, `muon_variant`, `split_seed`, `effective_loader_seed`,
`eval_batch_size`, `bn_mode`, `checkpoint_policy`, `svd_spectra_schedule`, the curve summaries
from `summarize_curves`, and final `test`/`test_acc`. Diag npz: per-step arrays keep their
names; scheduled Sven diagnostics are stored with their own `svs_step` index array. Layout
under `{root}/{scan}/`: `*.jsonl`, `diag/`, `ckpt/{run_id}.pt` (+ `ckpt/init_mseed{seed}.pt`),
`done/`, `claims/`, `started/`, `manifest/`, `configs/` (resolved Hydra config per job),
`_stale/{hash8}/`, `attempts/`.

### Runner CLI (Hydra overrides)
`mode=svd|standard|jd|hig|all`, `optimizers_standard=[...]`, `model_seeds=[...]`, `n_data=...`;
`+n_shards/+shard_id` only with `scheduler=static`. With `scheduler=claims` every process
expands the full grid for its mode/optimizer subset, shuffles nothing, and walks it claiming
runs; many processes and many jobs may serve the same scan concurrently. Exit code 0 when
nothing is left to claim.

### Execution
Campaign processes run from an exported snapshot, never the live tree: `tools/deploy_snapshot.sh`
exports both repos at their HEAD into `$SV3_SCRATCH/deploy/<sha8>_<sha8>/` with
`DEPLOY_INFO.json` (provenance falls back to it) and an `experiment_results` symlink to the real
results root; workers set `PYTHONPATH=<snapshot>:<snapshot>/sven`, `SV3_RESULTS_ROOT`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `OMP_NUM_THREADS=1`. NPROC per GPU from the
probe (§1.5): A100 — MLP Sven 12, first-order MLP 12, LBFGS/HIG/Shampoo 4, nanoGPT 3, CIFAR Sven
1, CIFAR baselines 4; MIG slice — MLP Sven 6, first-order 6, LBFGS 6, nanoGPT 1. The MIG lane
(`gpu_mig`, `--gres=gpu:4`, 12 h) runs one worker pool per slice.

### Analysis conventions
Selection = seed-mean final VALIDATION loss under the full rule (eligible -> fewest diverged ->
seed mean), implemented in `analysis/lib/scan_analysis.py` and `tools/select_best.py` ->
`bench/best_configs.json` (the selection of record; never `tools/reconcile.py`'s quick table).
Test metrics are OUTCOMES, never selection inputs. Diverged = failed (status or 10x rule),
excluded from means and counted; every table shows `finished / attempted`. Seed band = mean
+/- 1 std (ddof=1) labelled "+/- 1 std over seeds". Headline numbers = CONFIRMATION seeds
(`<scan>_confirm`), tuning-seed numbers shown beside them. Colours `style.METHOD_COLORS` (Sven
black), names `style.DATASET_TITLES`. The results roots are read-only for the analysis; the
loader cache under `experiment_results/_cache/` is the one allowed write.
