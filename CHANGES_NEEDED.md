# Changes needed, and the plan for re-executing the experiments

Written 2026-09-18 (sv3 branch `rebuttals` @ `13f65f7`, sven @ `ca8742b`). This is a handoff
document: it is meant to be read by someone (or an instance of Claude) with **no context from
the session that produced it**, working on the cluster with SLURM access.

It synthesises three sources:
* `CODEX_CRITIQUES.md` — Codex's methodological review;
* `FABLE_CRITIQUES.md` — the point-by-point response, with measurements, plus new findings;
* the decisions the user made after reading both (section 1).

Evidence for every finding is in those two files and is not repeated here. This file says
**what to change, where, how to tell it worked, and in what order to rerun.**

## 0. Start here

**Read, in order:** this file; `FABLE_CRITIQUES.md` (evidence, and the logging spec);
`analysis/ANALYSIS_FIXES.md` (analysis conventions already decided); `analysis/RERUNS_NEEDED.md`
(what was launched on 2026-09-17).

**Two repositories.** `sv3` (this one) holds the runner, configs, baselines and analysis.
`sven/` is a *separate git repo*, gitignored here and installed editable; it holds the optimizer
and wrappers and has its own tests (`sven/tests`, 69 Gram tests passing at `ca8742b`). Several
changes below touch both. Work on a feature branch in each; record both SHAs in every result.

**First hour on the cluster, before writing code:**
1. `squeue -u $USER`. The 2026-09-17 launchers (`submit_reruns_2026-09-17.sh`, `..._part2.sh`)
   started jobs that carry bugs described here: **GPT-2-small** (train/val overlap, finding F4)
   and **`exp_finetune_cifar_smallN`** (BatchNorm statistics learned from validation batches,
   F3). Report to the user what is still running. Their output must not reach a plot; whether
   to cancel is the user's call. Everything else from that launch will be superseded by this
   campaign but is harmless.
2. Run the ten-second overlap check on the FineWeb token files (section C-D2).
3. Establish the compute facts this plan only estimates: partitions, per-user GPU cap, time
   limit, GPU types (`sinfo`, `sacctmgr show assoc user=$USER`). The launchers assume
   `lab_gpu_priority,lab_gpu,gpu`, 12 h, and a cap of 8 concurrent single-GPU jobs.
4. Never delete results. Superseded files are moved aside (section 4.1).

## 1. Decisions already made (binding — do not reopen)

From the 2026-09-17 analysis audit (`ANALYSIS_FIXES.md`):
* One selection metric everywhere: **seed-mean final validation loss** of a configuration.
* **Diverged = failed**: non-finite end, or final val loss > 10x its pre-training value
  (`style.is_diverged`). Diverged runs are excluded from means and counted in `n_diverged`.
  Ranking: eligible (more than half the seeds finished) -> fewest diverged -> seed mean.
* Seed band = arithmetic mean +/- 1 std (ddof=1), lower edge clipped at the lowest seed.
* Critical batch: a config counts only if every seed reaches the target.
* Colours from `style.METHOD_COLORS` (Sven black); names from `style.DATASET_TITLES`.
* AdamW runs at PyTorch's default wd 0.01; `MuonW` = Muon at wd 0.1; plain Muon stays at 0.

From the user on 2026-09-18, after the two critiques:
* **Test split.** Every dataset gets a test split independent of train and validation. Test
  loss is computed **during the scan** by the training loops, not offline. **Selection still
  uses validation only**; test numbers are for reporting the selected configuration.
* **Checkpoints and full SVD information are the top priority** of the logging work.
* The batch-weighting and timing defects are small (measured in `FABLE_CRITIQUES.md`) but cheap,
  so they get fixed, including the offline repair of existing results.
* **All experiments are re-executed** on the fixed code.

## 2. Findings and the change that addresses each

Source: **C** = Codex (T = top, M = mid, L = low), **F** = Fable. Status: **V** = verified by
running something, **R** = established by reading code. Change IDs refer to section 3.

| # | Finding | Source | St. | Change | Rerun? |
|---|---|---|---|---|---|
| F1 | "Degree-4 polynomial" is an additive cubic: no interactions, no power above 3; 19 features fit it to 7e-8 | C-T1 | V | C-D1 | all polynomial scans |
| F2 | Sven validates with train-mode BatchNorm (val-batch statistics, buffers mutated); baselines use running stats | C-T2 | V | C-E2 | CIFAR |
| F3 | With `freeze_norm_stats=True` the running stats are updated **only** by validation batches (`evaluate` is outside `_frozen_norm_stats`) | F | V | C-E2 | fine-tune |
| F4 | `prepare_tokens.py` restarts the stream: `val.bin` is a prefix of `train.bin`; unfilled file tail left as zeros; contamination is optimizer-dependent | C-T3 + F | R | C-D2 | GPT-2 |
| F5 | Official test sets double as validation; toy's test split is never evaluated | C-T4, remedy per user | R | C-E1 | all |
| F6 | Exceptions are printed and dropped. Sven / HIG / KFAC blow-ups raise in a linalg call (file missing); first-order blow-ups record NaN (counted diverged). HIG has 35–40 of 80 grid runs on three scans, 0 recorded failures. Partial curves are lost | C-T5 + F | V | C-R1, C-R2 | all |
| F7 | Batch timers stop before the CUDA sync for baselines; Sven syncs inside the timed region | C-T6 | R | C-T1 | timing runs |
| F8 | Profile `steady_mean` drops >3 MAD points, removing SOAP's periodic refresh (10%); same filter moves Sven `gram_full` 9%, `classic` 7% | C-T6, amended F | V | C-T2 | none (offline) |
| F9 | Historical AdamW == Adam (wd forced to 0). Code fixed 2026-09-17. `submit_fresh_suite.sh` omits `MuonW`; one default wd is not a tuned AdamW | C-T7, amended F | V | C-B1, C-B4 | covered by rerun |
| F10 | Epoch metrics average batch means, not examples. Median error 0.07–0.4% in headline scans; up to 9–33% for isolated runs in large-batch / full-batch scans | C-T8, amended F | V | C-E1, C-A1 | covered; legacy repaired offline |
| F11 | Online train loss spans changing parameters; LBFGS records its *last* closure evaluation | C-T8 | R | C-E3 | covered |
| F12 | Tuning budget: Sven 72/128 grid points (48–90 distinct trajectories) vs 4 lrs for most baselines; SGD without momentum; no schedules; many optima on grid edges | C-M1, amended F | V | C-B2, C-B3, C-A4 | extensions |
| F13 | Muon: all 2-D params incl. embeddings/heads go to Muon; conv kernels (4-D) go to AdamW, so on ResNet "Muon" is AdamW plus one `fc` layer; one lr shared by both halves | C-M1 + F | R | C-B5 | Muon runs |
| F14 | Local schedule-free classes differ from the reference algorithms (extra EMA / momentum). Not in saved results | C-M1 | R | C-B6 | none |
| F15 | RNG is seeded once per model seed; masks, randomized SVD, model construction and val-loader iterators draw from the global stream, so a masked run depends on its position in the process | C-M2 | R | C-S1 | paramfrac scans |
| F16 | `run_id` omits epochs, dataset seed, `n_train`, code version; dedup is by file existence, so fixed code silently keeps stale results. The editable `sven` install leaves no trace in results | C-M2 + F | R | C-R3 | all |
| F17 | No untruncated spectrum exists on disk (all 28 scan dirs audited). Spectra are trajectory-dependent and the best configs mostly have `k < B`, so the `k = B` rerun is not sufficient | C-M2 + F | V | C-L1, C-L2 | best configs |
| F18 | No checkpoints are saved anywhere | F / user | R | C-L3 | all |
| F19 | Per-step spectra come from a different random batch each time (confounds evolution with batch noise; capped at B values); float32 inputs give a noise floor near 1e-7 sigma_max | F | R | C-L4 | none (offline from checkpoints) |
| F20 | `sv_min` changes meaning between old records (smallest kept SV) and new ones (sigma_B, numerical noise) | F | R | C-L1 | covered |
| F21 | Updates, examples processed and synchronised time are not separate axes; wall-time winners are chosen by final-epoch loss | C-M3 | R | C-A5 | none |
| F22 | Overparam studies: changing `n_train` also changes the validation inputs and the target normalisation (one shared generator; train-set mean/std); batch = N changes rank and steps together | C-M3 | V | C-D3 | overparam scans |
| F23 | Critical-batch: retained rank is tied to batch size; crossings measured once per epoch | C-M3 | R | C-X2, C-E4 | critbatch |
| F24 | Workloads are small; CIFAR uses an ImageNet stem, no augmentation, 20 epochs | C-M4 | R | tier 2 (4.6) | new |
| F25 | Uncertainty: no confirmation seeds, no paired differences, 2–3 seed studies | C-M5 | R | C-A6, plan | confirmation runs |
| F26 | `loader_seed` is shared by all model seeds: seed bands contain initialisation variance only | F | R | C-S2 | covered |
| F27 | Each synthetic benchmark is a single draw (`data_seed` fixed) | F | R | plan 4.5 | replicates |
| F28 | Cost claim "scales with retained rank" does not describe the Gram backend (B x B `eigh` regardless of k; `full` capture materialises J) | C-M5 | V | C-Z1 | none |
| F29 | kappa ablation at `k = B`, fixed lr: the kappa step is exactly `2/kappa` times the kappa = 2 step in the untruncated full-row-rank solve, so it is nearly an lr sweep | C-L1 + F | R | C-X1 | kappa scans |
| F30 | Missing mechanism studies: noise / under-parameterised / loss scaling / additive constant; hard vs damped truncation; masks at matched cost; implicit regularisation | C-L1–L4 | — | tier 3 (4.6) | new |
| F31 | `actual_param_fraction` is computed by the wrappers and never recorded | C-L3 + F | R | C-R4 | covered |
| F32 | On MNIST-CE the models overfit within 20 epochs; last-epoch val loss then selects the lowest lr for 7 of 11 baselines. The metric is decided; the consequence is that those grids are truncated from below | F | V | C-B3, C-E5 | extensions |
| F33 | `torch.cuda.empty_cache()` on every Sven step; "LBFGS" is minibatch `torch.optim.LBFGS`; Sven-microbatch uses `drop_last=True`, baselines do not; eval batch size = train batch size | F | R | C-T3, C-B7, C-S3, C-E1 | covered |
| F34 | GPT-2 config runs one epoch and validates once per epoch: two validation points, no curve | C-M4 + F | R | C-E4 | GPT-2 |
| F35 | Docs drift: `EXPERIMENTS.md` says 10 seeds and `chunked` capture, configs say 5 and `full`; `CLAUDE.md` states O(kN\|D\|) | F | R | C-Z1 | none |

## 3. The changes

Each item gives the location, the specification and an **acceptance test**. Tests marked (CPU)
need no GPU and should become unit tests.

### 3.1 Run records and lifecycle (`experiments/experiment_code/generic_scan.py`)

**C-R1. Failures are results.** Every `except` block (lines 583, 676, 754, 826, 892, 944 today)
writes a normal record with:
* `status`: `ok` | `diverged` | `oom` | `error`. Classify: `RuntimeError` raised by
  `SvenGram.step` for an empty spectrum, `torch.linalg.LinAlgError`, and "failed to converge"
  messages -> `diverged`; `torch.cuda.OutOfMemoryError` -> `oom`; anything else -> `error`;
* `error` (exception type and message) and the **partial curves** collected so far. The loops
  must therefore fill a `losses` dict owned by the caller rather than returning it at the end;
* `diverged_at_step` where known.
Also stop early, with `status: diverged`, when a first-order run's batch loss goes non-finite
(today those runs burn all remaining epochs producing NaN).
Write a `{run_id}.started` marker at run start and remove it on completion: a marker without a
record means timeout or crash.
Dedup: `ok` and `diverged` records count as done; `oom` and `error` are retried.
Analysis: `style.is_diverged` also returns True for `status == "diverged"`.
*Test (CPU):* a toy config with `lrs: [1e3]` for Sven, HIG and SGD produces three records with
`status: diverged`, non-empty partial curves, and `n_diverged = 1` each in `config_table`.

**C-R2. Manifest and reconciliation.** Refactor the five copy-pasted grid loops into
`expand_grid(rcfg) -> list[RunSpec]` (family, `run_id`, hparams) plus one `execute(spec)`;
sharding becomes `specs[shard_id::n_shards]` with the same ordering as today. Each job writes
the run_ids it is responsible for to `{scan}/manifest/{job}.json`; the scan's intended grid is
the union. Add `tools/reconcile.py <scan>`: per method, counts of expected / ok / diverged /
oom / error / started-only / never-started, and for each method's best config whether its lr
(and k, rtol for Sven) lies on a grid edge. Tables in the notebooks show `finished / attempted`.
*Test (CPU):* delete one result file from a finished toy scan; `reconcile` reports exactly that
run as never-started and exits non-zero.

**C-R3. Identity and provenance.** Every record stores `run_hash`: a hash of the resolved
dataset and model configs, loss, epochs, batch size, all optimizer hparams, the three seeds,
eval settings, and `schema_version`. Dedup skips only when the existing record's hash matches;
on mismatch the old files are **moved** to `{scan}/_stale/{old_hash}/` and the run executes.
Also record `git_sha` and `git_dirty` for **both** repos, torch and CUDA versions, GPU name,
host, SLURM job id, `n_shards`, start/end timestamps. Save the resolved Hydra config once per
job under `{scan}/configs/`.
*Test (CPU):* change `num_epochs` and resubmit; every run re-executes and the old files are
under `_stale/`.

**C-R4. Facts on every record:** `n_train`, `n_val`, `n_test`, `steps_per_epoch`, `n_params`,
and for masked runs the mean `actual_param_fraction`.

### 3.2 Splits, evaluation and metrics

**C-E1. Three splits and one evaluation function.**
* `experiments/datasets/all_datasets.py`: every class exposes `train_dataset`, `val_dataset`,
  `test_dataset`. MNIST: official train -> 50,000 train / 10,000 val by a fixed `split_seed`
  (a dataset-config value, independent of model and loader seeds, recorded); test = official
  test set. CIFAR-10: 45,000 / 5,000, test = official. `n_train` subsampling draws from the
  train part only. Shakespeare: contiguous 80 / 10 / 10 by position. Token bins: see C-D2.
  Synthetic: see C-D1 and C-D3.
* `experiment_utils.py`: one `evaluate(forward, per_sample_loss, loader, ...)` used by all four
  loops for validation and test, before training and after every epoch. It returns the
  **example-weighted** loss and accuracy (sum over examples / N), runs the model in **eval
  mode**, restores the previous mode, and mutates nothing. It uses its own `eval_batch_size`
  (config key), not the training batch size.
* Record per-epoch `val`, `val_acc`, `test`, `test_acc` (index 0 = untrained). The online train
  loss is also example-weighted.
* Analysis: `final_test_loss` / `final_test_acc` are **outcome** columns, shown beside the
  selected config. No selection function may take them as its metric; add an assertion.
*Test (CPU):* with 10,000 validation examples and eval batch 128, the recorded value equals the
mean of per-example losses to 1e-6; evaluating twice changes no buffer.

**C-E2. One BatchNorm policy for every optimizer.** Goal: train with batch statistics; update
running statistics from **training** batches **exactly once per optimizer step**; evaluate with
running statistics; never touch them during evaluation.
* `sven/sven/nn/sven_wrapper.py` `evaluate()` (and `HIGWrapper.evaluate` in
  `experiments/optimizers/hig.py`): run the module in eval mode.
* Repeated forwards within a step must not update the stats again: Sven's `delta_from_w`, the
  second and later groups of chunked capture, `SvenGramReg._rows_jvp`, `variable_k` re-evaluations,
  and LBFGS closure calls after the first. A context that temporarily disables running-stat
  tracking on all `_NormBase` modules does this.
* The standard loop's pre-training validation must run in eval mode (it currently runs in train
  mode because `.eval()` is first called after epoch 1).
* Frozen-stats path (`freeze_norm_stats=True`, hooks capture): frozen means frozen for
  evaluation too. For the fine-tune study apply the same `bn_mode` to the baselines (default:
  frozen pretrained statistics for all optimizers; see open decision O2) and put `bn_mode` in
  the `run_id`.
*Test (CPU):* on `SmallResNet`, after one optimizer step `running_mean` equals
`(1 - m) * old + m * batch_mean` exactly once, for Sven (`full` and `chunked` capture) and for
LBFGS with `max_iter=3`; `evaluate` leaves every buffer bit-identical; a fixed example's
prediction does not depend on its batch companions. This is the probe from the appendix of
`FABLE_CRITIQUES.md`, inverted.

**C-E3. Train-loss semantics.** LBFGS records the loss of its **first** closure call (the
pre-update loss, as for SGD and Sven). Add `train_eval`: an end-of-epoch, eval-mode loss on a
fixed subset of `min(n_train, 10_000)` training examples. That is the quantity the paper's
convergence claim is about, comparable across optimizers at a fixed parameter vector.

**C-E4. Step-based evaluation.** Optional `eval_every_steps`: validation (and test) at step
multiples as well as epoch ends, stored with their step indices. Needed by the one-epoch GPT-2
study (F34) and by critical-batch crossings (F23).

**C-E5. Alternatives to "final" are logged, not adopted.** Record the best-epoch validation
loss and the mean of the last three epochs beside the last-epoch value. The selection metric
does not change (section 1); this lets the open decision D28 be settled later without reruns
and shows how much of a gap is overfitting (F32).

### 3.3 Datasets

**C-D1. Polynomial.** Enumerate all monomials of total degree <= d, including the constant and
linear terms (C(6+4, 4) = 210 for the configured case), and **multiply** the factors. Keep the
old generator as `AdditiveCubicDataset` so historical results stay reproducible and correctly
named. Add a test split.
*Test (CPU):* known monomials evaluate correctly; the 19-feature additive fit now has relative
RMSE > 0.1; the old class still reproduces 7e-8.

**C-D2. Token preparation (`experiments/data_prep/prepare_tokens.py`).** Consume **one shared
iterator**, writing val, then test, then train from disjoint documents; truncate each file to
the tokens written; print and store counts. Add `tools/check_token_split.py` that asserts no
overlap and reports sizes. The check for the existing files:
```python
v = np.memmap(".../val.bin", dtype=np.uint16, mode="r")
t = np.memmap(".../train.bin", dtype=np.uint16, mode="r")
print(np.array_equal(t[:len(v)], v))      # True => val is a prefix of train
```

**C-D3. Synthetic data that does not move with `n_train`.** Toy and polynomial: generate a fixed
training *pool* (10,000), a validation set and a test set from **separate generators**;
normalise targets by the pool's mean and std; subsample `n_train` from the pool. Validation,
test and the target scale are then identical at every P/N.
*Test (CPU):* `val_dataset` tensors are equal for `n_train` = 150 and 1200.

### 3.4 Seeding and data order

**C-S1. Position-independent randomness.** Immediately before building each run (after the
model has been constructed and `init_state` loaded), call
`set_seed(model_seed XOR crc32(run_id))`. Use `zlib.crc32`, not Python's salted `hash`.
*Test (CPU):* a `param_fraction = 0.5` run gives bit-identical curves when run alone and when
run after another configuration in the same process.

**C-S2. Data order as a pure function.** Replace `shuffle=True` with a sampler whose epoch-`e`
permutation is `randperm(n, generator=Generator().manual_seed(f(loader_seed, e)))`. The batch at
any step can then be reconstructed offline with no stored state, which the checkpoint tools
need (C-L4). Derive `loader_seed` from `model_seed`, so seed bands include data-order variance,
while keeping it **identical across optimizers** for a given model seed so comparisons stay
paired.

**C-S3.** `drop_last=True` for the training loader of **every** optimizer, so all methods take
the same number of steps on the same batches.

### 3.5 Logging: spectra and checkpoints (both repos)

Full rationale and storage arithmetic: first section of `FABLE_CRITIQUES.md`.

**C-L1. Optimizer (`sven/sven/opt/sven.py`).**
* A per-step flag set by the training loop decides whether this step is logged; the device sync
  for `sigma_full.cpu()` happens only on logged steps. `num_nonzero_svs` stays per-step.
* On logged steps `SvenGram` / `SvenGramReg` store: `step`; `svs` (all B values, before the
  k / rtol cut, as now); **`utr = U.T @ r`** (all B); `update_norm`; `resid_norm`;
  **`sv_min_kept`** = the smallest singular value actually inverted.
* The classic `Sven` path computes the full spectrum on logged steps from a float64 `eigh` of
  `J J^T`; it cannot be recovered from `pinv()`.
*Tests (CPU, in `sven/tests`):* in float64 on a small MLP, `svs` equals
`torch.linalg.svdvals` of the explicit Jacobian; the applied update equals
`J^T U diag(1/s^2) utr` restricted to the kept directions.

**C-L2. Runner.** `svd_spectra_schedule: {dense_first: 200, every: 20}` replaces
`svd_spectra_every`; `_split_diagnostics` stores what the optimizer logged with its step
indices and no longer subsamples. `svd_summary` records the schedule. `sv_min` in old and new
records must not share a column name (F20).

**C-L3. Checkpoints.** New `experiments/experiment_code/checkpointing.py`, used by all four
loops. Config key `checkpoints: none | final | epochs | log`, where `log` = steps
{0, 1, 2, 4, 8, ...} plus every epoch end.
* Content: the module `state_dict` cloned to CPU in float32 — **weights and buffers**.
* Layout: **one file per run**, `{scan}/ckpt/{run_id}.pt` = `{step: [...], epoch: [...],
  state: [...]}`. Per-checkpoint files would create 10^5 small files per scan on Lustre.
  Long runs (ResNet, GPT) rewrite the file at each epoch end so a timeout loses little. Under
  `final`, the shared initial state is saved once per seed as `ckpt/init_mseed{seed}.pt`.
* Write order: checkpoint, then the npz, then the jsonl, preserving "jsonl last = dedup marker".
* A failed run (C-R1) writes whatever checkpoints it collected; the last state before a blow-up
  is diagnostic.
* Defaults: toy / polynomial `log` for every run (~90 KB per run); MNIST `log` for Sven and
  `final` for baselines in the grids (3.9 MB vs 110 KB per run); nanoGPT `epochs`; ResNet18
  `final` (45 MB per run, ~13 GB per 290-run scan); `log` everywhere in the diagnostic reruns
  of phase 5.
*Test (CPU):* reload the final checkpoint of a toy run; its validation and test losses
reproduce the recorded final values to 1e-6.

**C-L4. Offline tools (`analysis/ckpt_tools.py`).** Load a checkpoint into a model; reconstruct
the batch of a given step from C-S2; compute the Jacobian spectrum in float64 on (a) a **fixed
probe set** shared by every run and every optimizer of a scan, and (b) the **full training
set** for toy and polynomial (10,000 x ~600, seconds) and a few thousand examples for MNIST.
This gives spectra along the trajectories of Adam and the other baselines too, and right
singular vectors, which cannot be logged online.

### 3.6 Timing and profiling

**C-T1.** `torch.cuda.synchronize()` immediately before the start and end reads of every batch
timer, in all four loops. Record per-epoch `train_times` (sum of synchronised batch times) next
to `epoch_times`; wall-time curves use cumulative **training** time, since evaluation is an
identical overhead for every method and now larger (val + test + `train_eval`). Timing runs set
`svd_info: none` and `checkpoints: none`.

**C-T2. Profile summaries (offline — no new runs).** Add `cycle_mean` to
`optimizer_profile.summarize`: the plain mean over the last 80% of measured steps, truncated to
a whole number of 10-step cycles (SOAP's `precondition_frequency`). `analysis/profile_helpers.py`
(lines 74–77) switches `step_ms`, `wall_ms`, `capture_ms`, `solve_ms` to it, **recomputed from
the stored `raw` lists** so existing profiles are repaired without rerunning. Keep
`steady_mean` for reference. After the measured steps, check parameters are finite and record
`status: nonfinite` otherwise.
*Test:* SOAP on `profile_mnist` reads 6.14 ms (was 5.56), `gram_hooks` stays at 10.40.

**C-T3.** Make the per-step `torch.cuda.empty_cache()` calls in `sven.py` optional (default off
for timing and profile runs) and measure the difference once.

### 3.7 Baselines

**C-B1.** `submit_fresh_suite.sh:49`: add `MuonW` (and `SGDm`, below) to `FIRST`.
**C-B2.** Add SGD with momentum 0.9 as a named baseline `SGDm`; keep plain `SGD`.
**C-B3. Grids must contain their optimum.** After each scan `tools/reconcile.py` lists best
configs on a grid edge. Extend by half-decades beyond the edge, at most two rounds, for
baselines **and** for Sven. Known cases: on MNIST-CE, Adam, RMSprop, Muon, SOAP, JD and HIG at
their lowest lr; SGD and Shampoo at their highest in several scans; KFAC at its lowest on
polynomial; Sven at `k = B` with the smallest rtol on polynomial, the smallest lr on CIFAR-CE
and the largest on CIFAR label-reg.
**C-B4.** AdamW: lr x wd grid with wd in {0, 0.01, 0.1} in the headline scans.
**C-B5. Muon.** Hidden 2-D weights only: embeddings, the output head and all 1-D parameters go
to AdamW. For conv nets either vendor a small Muon that flattens kernels to 2-D (as SOAP is
vendored) or leave Muon out of the CIFAR studies; do not report the current construction as
"Muon" on ResNet. Use `adjust_lr_fn="match_rms_adamw"` if the cluster's torch provides it, so
the shared lr is principled; otherwise sweep the two learning rates separately. Record the
variant in the result.
**C-B6.** Schedule-free: use the reference `schedulefree` package or remove the two classes
from the registry.
**C-B7.** Label LBFGS as stochastic (minibatch) L-BFGS in plots and text.

### 3.8 Study-specific changes

**C-X1. kappa.** Retune lr for each kappa (at minimum include `lr * kappa / 2`), and add a slice
where truncation binds (`k < B` or a larger rtol). Plot against the effective step `2 lr / kappa`.
**C-X2. Critical batch.** Sweep `k_fractions` in {0.25, 0.5, 1.0} at each batch size so rank and
batch size separate; use `eval_every_steps` for the crossings; report optimizer steps and
examples processed, not epochs (now possible from C-R4).

### 3.9 Analysis (`analysis/`)

**C-A1. Legacy repair (offline).** `analysis/repair_legacy.py`: recompute example-weighted
validation losses from `val_batch` and the known batch lengths for the legacy results root, as
new columns. Accuracy cannot be repaired (not stored per batch).
**C-A2.** Loaders understand `status`, the manifest and the test columns; `n_missing` comes from
the manifest, so configurations with no result file are no longer invisible.
**C-A3.** Spectrum plots: full spectrum with the rtol line, the k cut and the float32 noise
floor marked; new `utr` plots; probe-set and full-dataset spectra from C-L4.
**C-A4. Tuning-budget disclosure, from existing data.** A table of distinct trajectories per
method, and a best-of-n curve: the expected best validation loss when n configurations are
drawn at random from each method's grid. This answers the budget objection without new runs.
**C-A5.** Present optimizer steps, examples processed and synchronised training time as separate
axes; add time-to-target at several pre-declared targets, counting runs that never reach them.
**C-A6.** Paired differences (same model seed, hence same init and data order) with intervals;
legends say "+/- 1 std over seeds", not confidence interval; completion counts on every table.

### 3.10 Documentation

**C-Z1.** `EXPERIMENTS.md` (seed counts, capture mode, splits, evaluation protocol),
`CLAUDE.md` and the paper text on cost: under the Gram backend a step costs a B x B `eigh` plus
the capture, independent of k; O(kN|D|) describes only the classic randomized path.

## 4. Re-execution plan

### 4.1 Results layout

On the cluster `experiment_results` is a symlink to labstore, and the path is hard-coded
(`generic_scan.py:347`; `RESULTS_ROOT` in the analysis modules). Once the in-flight jobs have
drained or been cancelled: rename the existing directory to
`experiment_results_legacy_2026-09-18/` (read-only from then on) and start an empty
`experiment_results/`. Every notebook path keeps working; the legacy-repair analysis points at
the legacy root explicitly. Make the root a config key and an analysis argument while there.

### 4.2 Phases and gates

**Phase 0 — offline, no GPU, can start immediately.** C-T2 (profiles), C-A1 (legacy validation
repair), a first `reconcile` of the legacy scans against their grids, C-A4 (budget curves).
Deliverable: corrected legacy tables, which are the baseline for judging what the fixes change.

**Phase 1 — development.** Suggested order, which minimises rework: C-R2 (the refactor
everything else plugs into) -> C-R1, C-R3, C-R4 -> C-E1 -> C-E2 -> C-S1–S3 -> C-L1–L3 -> C-T1,
C-T3 -> C-D1–D3 -> C-B1–B7 -> C-E3–E5, C-X1–X2 -> C-A2–A6, C-L4. CPU tests listed above go into
`sven/tests` and a new `tests/` here.

**Gate 1 — smoke tests on one GPU** (each a 2-epoch, 1-seed run): every family (`svd`,
`standard`, LBFGS, PolyakSGD, `jd`, `hig`) on every dataset class; one deliberately divergent
run per family; one masked run; one CIFAR run per `bn_mode`. Check: record schema; test columns
present; spectra full width with `utr`; checkpoint reloads and reproduces the final losses;
failure records written; `reconcile` clean; resubmission skips everything; resubmission after a
config change reruns everything. Do not launch a full scan until this passes.

**Phase 2 — pilot: `toy_1d_scan` and `polynomial_scan`, full suites** (~13 GPU-hours). Then run
`reconcile` and `./make_plots.sh` end to end. Expect toy to be close to the legacy numbers
(different split and data order, same problem) and polynomial to differ (different target).
Anything else that moves needs explaining before going on. **Gate 2:** the user reviews the
pilot tables.

**Phase 3 — headline.** Both MNIST scans, both CIFAR scans. CIFAR Sven dominates the whole
campaign (90 runs of ~2 h per scan), so submit it first within the phase.

**Phase 4 — ablations, rebuttal and tier-3 scans**, using set points from phase 3: micro-batch,
param-fraction (paired with C-S1), kappa (C-X1), overparam (C-D3), batch-size, critical batch
(C-X2), nanoGPT, fine-tune (C-E2), CIFAR ablations, and GPT-2 only after C-D2 has rebuilt the
token files and the overlap check passes.

**Phase 5 — best configurations, three separate passes.**
* *Timing:* NPROC = 1, exclusive node, logging and checkpoints off. Regenerate
  `bench/best_configs.json` with `bench/select_best_configs.py`, which must call
  `Scan.best_sven` / `best_baseline` from the current `scan_analysis.py`.
* *Diagnostics:* the best config of **every** method (not only Sven) with `checkpoints: log`
  and the dense spectra schedule; toy and polynomial additionally in float64. These runs are
  what the spectrum figures read (F17).
* *Confirmation:* the selected config of each method on **5 fresh model seeds**, and on **3
  data seeds** for toy and polynomial (F25, F27). Report these, not the tuning seeds.

**Phase 6 — grid extensions** where `reconcile` still shows an optimum on an edge (C-B3), then
the AdamW lr x wd grid (C-B4). Re-select, and repeat the affected part of phase 5.

**Phase 7 — analysis.** `./make_plots.sh`; update the documents (C-Z1); write down what changed
against the legacy tables.

### 4.3 Inventory and cost

Measured from the legacy records: "process-h" is the sum of each run's recorded wall time under
NPROC-way sharding; "GPU-h" divides by the NPROC the launcher uses (6 toy / polynomial, 4 MNIST,
2 ResNet / nanoGPT). These are estimates to be re-derived on the cluster. Evaluation now
includes a test pass and `train_eval`, which adds roughly 10–20% for the cheap first-order runs
and little for Sven.

| scan | runs | process-h | NPROC | ~GPU-h | what dominates |
|---|---|---|---|---|---|
| `toy_1d_scan` | 714 | 34 | 6 | 6 | LBFGS 16 h, Sven 13 h |
| `polynomial_scan` | 711 | 41 | 6 | 7 | LBFGS 23 h, Sven 13 h |
| `mnist_scan_ce` | 1020 | 169 | 4 | 42 | HIG 67 h (80 runs), Sven 56 h |
| `mnist_scan_labelRegression` | 979 | 184 | 4 | 46 | Sven 59 h, LBFGS 55 h, HIG 50 h (39 runs) |
| `cifar10_resnet_ce_scan` | 290 | 215 | 2 | 108 | Sven 201 h (90 runs, ~2 h each) |
| `cifar10_resnet_scan_labelRegression` | 290 | 215 | 2 | 107 | Sven 200 h |
| micro-batch scans (4) | 464 | 86 | 6 / 4 | 20 | Sven only |
| param-fraction scans (4) | 323 | 80 | 6 / 4 | 19 | Sven only |
| `mnist_kappaScan_labelRegression` | 15 | 7 | 4 | 2 | grows with C-X1 |
| `rebuttal_batchsize_polynomial_scan` | 2376 | 285 | 6 | 47 | **LBFGS 238 h (810 runs, 439 diverged)** |
| `rebuttal_overparam_mnist_scan` | 2357 | 161 | 4 | 40 | LBFGS 88 h, Shampoo 32 h |
| `rebuttal_overparam_{toy,polynomial}` | 4160 | 50 | 6 | 8 | Sven |
| `exp_critbatch_{mnist,nanogpt}` | 168 | 28 | 4 / 2 | 11 | grows with C-X2 |
| `exp_nanogpt_speedrun` | 48 | 4 | 2 | 2 | |
| standalone timing (5 dirs) | 262 | 19 | 1 | 19 | |
| **subtotal, scans present locally** | | **~1,580** | | **~490** | |
| CIFAR ablations, fine-tune (240 runs), GPT-2, Brier | | not local | | ~150–250 | measure on the cluster |

At 8 concurrent GPUs, 490 GPU-hours is about 2.5 days of pure compute; with queueing, the 12 h
limit and the phase gates, plan on one to two weeks. Phase 5 adds little (tens of runs per
scan). Checkpoint storage under the defaults above is on the order of 50 GB, mostly ResNet
finals.

**Savings worth taking:**
* Batch-size scan: LBFGS is half the cost of the scan and mostly diverges. Fix
  (`max_iter`, `history`) at the headline scan's best values and sweep lr only: 810 -> ~90 runs,
  about 35 GPU-hours saved. With C-R1's early stop the diverging ones also end sooner.
* C-R1's early stop removes the cost of every run that currently trains on NaN to the end.
* HIG on MNIST costs 50–67 process-hours for runs that mostly crash at lr >= 0.5; with failure
  records they end at the blow-up, and its grid should move down (C-B3).

### 4.4 Launch mechanics

Reuse `submit_fresh_suite.sh` (groups `headline | ablations | rebuttal | tier3`, `DRY=1`,
`ONLY=regex`) and `submit_rebuttal_parallel.sh` (NPROC shards per GPU). Keep the existing job
splitting — Sven, first-order, second-order and LBFGS as separate jobs — so a Sven job never
waits on an LBFGS line search; that gating caused the 12 h timeouts before. Dedup plus the run
hash make every job safely resubmittable. After each phase: `reconcile` for every scan, resubmit
until nothing is missing, then move on.

### 4.5 Verification after each phase

* `tools/reconcile.py`: zero never-started or started-only runs; failures classified; edge
  optima listed.
* Spot checks: reload three random checkpoints per scan and reproduce their recorded losses;
  one Sven run's `svs` has width B (or B / microbatch) at every logged step; test columns are
  finite wherever validation is.
* Paired reproducibility: rerun one finished configuration alone and compare curves (bit-exact
  for MLPs, to GPU non-determinism for conv nets).
* Record both git SHAs for the phase in `analysis/RERUNS_NEEDED.md`'s launch log.

### 4.6 After the campaign (not part of the rerun)

* **Tier 2 (F24):** a CIFAR-style ResNet stem, standard augmentation and schedules on CIFAR-10,
  then CIFAR-100; one correctly split BPE language-model workload with equal token budgets,
  tuned AdamW and Muon, and step-based validation.
* **Tier 3 (F30):** label and observation noise plus a small under-parameterised case;
  multiplicative loss scaling and an additive loss constant (leaves gradients unchanged, changes
  Sven's step); hard truncation vs `SvenGramReg` damping at matched tuning budget; element /
  tensor / row masks at matched measured cost using the achieved fraction; distance from
  initialisation, margins and function-space change under network rescaling. Most of these
  become analyses of the phase-5 checkpoints rather than new training.

## 5. Open decisions, each with the default to use if nobody answers

| # | Question | Default |
|---|---|---|
| O1 | In-flight GPT-2 and fine-tune jobs from 2026-09-17: cancel or let finish? | Ask the user; either way quarantine the output |
| O2 | Fine-tune BatchNorm policy for all optimizers: frozen pretrained statistics, or batch statistics (needs `chunked` / `full` capture at P = 11 M) | Frozen for all |
| O3 | Seeds in the tuning grids | 5, as now, plus 5 fresh confirmation seeds for the selected configs (phase 5) |
| O4 | Polynomial: fix and rerun, or keep and relabel | Fix and rerun; legacy results relabelled "additive cubic" |
| O5 | Reduce the LBFGS grid in the batch-size scan | Yes (section 4.3) |
| O6 | Muon on conv nets: vendor a flattening Muon, or drop Muon from the CIFAR studies | Drop from CIFAR unless the vendored version is quick to validate |
| O7 | D28, what "final" means | Unchanged (last epoch); C-E5 logs the alternatives |

## 6. Definition of done

* Every scan reconciles against its manifest with nothing missing; every failure is a record
  with a status.
* Every record carries test metrics, provenance for both repos, and a run hash; every Sven
  record carries full-width spectra and `utr` on its logged steps; every run has at least a
  final checkpoint that reproduces its recorded losses.
* Validation and test evaluation are eval-mode, example-weighted and side-effect-free for every
  optimizer; the CPU tests in section 3 pass in both repos.
* The best config of every method has an interior optimum, a standalone timing, a diagnostic
  run with dense checkpoints, and confirmation seeds.
* The notebooks re-execute without error cells, tables show `finished / attempted` and test
  beside validation, and the differences from the legacy tables are written down.
