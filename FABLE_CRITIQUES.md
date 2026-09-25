# Sven experiments: response to the Codex critique, plus new findings

Written 2026-09-18 on branch `rebuttals` (sv3 `13f65f7`, sven `ca8742b`). Scope: the experiment
runner, training loops, datasets, models, baselines, the `sven` optimizer/wrappers, configs and
launchers, the analysis helpers and notebooks, `ANALYSIS_FIXES.md` / `RERUNS_NEEDED.md`, and the
results in the local `experiment_results/` (1.8 GB, 28 scan directories) and `profile_results_v2/`.

No experiment, optimizer or analysis code was changed. Claims marked **[verified]** were checked
by running something (CPU probes, or audits of the result files); **[read]** means established
from the code only. The checks are listed in the appendix so they can be re-run.

## Summary

**Codex's critique is largely correct.** I reproduced its three quantitative claims exactly
(polynomial fit to 7e-8, the BatchNorm mutation, SOAP 6.14 ms vs 5.56 ms). I disagree on
*priority* for two of its eight top items, where I measured the effect and it is small, and I
think two others are more urgent than it says because the affected jobs are already running.

| Codex item | Verdict | One-line reason |
|---|---|---|
| Top 1 polynomial is an additive cubic | **Concur** [verified] | 185 terms, no power above 3, no interactions; 19 features fit it to 7e-8 |
| Top 2 CIFAR evaluates different models | **Concur, and it is worse than stated** [verified] | on the frozen-stats path Sven's BN statistics are learned *only from validation batches* |
| Top 3 GPT-2 train/val overlap | **Concur, more urgent** [read] | the GPT-2 jobs were launched 2026-09-17; contamination is optimizer-dependent |
| Top 4 test set used for selection | **Concur**, mild disagreement on remedy | checkpoints make the fix free; "fresh confirmation tasks" is more than is needed |
| Top 5 exceptions vanish; missing beats diverged | **Concur, and it has a direction** [verified] | HIG — the method that beats Sven — is missing 40–45 of 80 grid runs per scan, none counted as failures |
| Top 6 timing summaries | **Concur on the defects, disagree on rank** [verified] | the MAD filter moves SOAP by 10% but also Sven `gram_full` by 9%; method gaps are 2–30x |
| Top 7 AdamW == Adam | **Concur, but already handled** | fixed and rerun under D29 / RERUNS item 0; the live remnant is the launcher |
| Top 8 batch-weighted epoch means | **Concur on the defect, disagree on rank** [verified] | median error 0.07–0.4% in the headline scans; exactly repairable offline |
| Mid 1–5, Low 1–4 | **Concur**, with additions | see below; the kappa point is an exact identity |

**My priority order** (detail in the sections below):

1. **Save checkpoints and full SVD information** (your max-priority item). Nothing on disk has
   an untruncated spectrum, and no run saves a model. Checkpoints also retroactively repair
   Codex Top 2, 4 and 8 for every future run. Spec in the next section.
2. **Quarantine the two studies that are running with known bugs**: GPT-2 (Top 3) and the
   CIFAR fine-tune (Top 2, frozen-stats variant). Both were launched 2026-09-17 per the launch
   log in `RERUNS_NEEDED.md`.
3. **Record failures as results** (Top 5 + new finding N1). Cheap, and it changes how the
   HIG-vs-Sven comparison reads.
4. Fix BatchNorm evaluation, fix or relabel the polynomial, then the baseline-tuning and
   reproducibility items.

---

## Max priority: checkpoints and full SVD logging

### What the code does today

* **Gram path** (`SvenGram.step`, `sven/sven/opt/sven.py:263-290`): since sven `ca8742b` it
  records the full B-vector `sigma_full` before the k / rtol cut. This is the right fix.
* **Classic path** (`Sven.step`, `sven.py:194`): still records `1/S_inv[S_inv > 0]`, i.e. only
  the singular values that survived both k and rtol. `pinv()` has already discarded the rest
  (`pinv.py:63-68`). Every scan config uses `use_gram: true`, so this affects only
  `variable_k`, the profiler's `classic` variant and notebook use.
* **Runner** (`generic_scan._split_diagnostics`, `generic_scan.py:172-226`): keeps every 20th
  spectrum (`svd_spectra_every`, default 20, line 335), plus per-step `sv_max` / `sv_min` /
  `num_nonzero_svs`.
* **Checkpoints**: none. `init_state` is held in memory (`generic_scan.py:392`) and no
  `state_dict` is ever written. The final model is discarded when the run returns.

### What is on disk [verified]

I audited every Sven `diag/*.npz` in all 28 scan directories: **no run has an untruncated
spectrum.** Where `k < B` the stored width equals k (e.g. MNIST `k=16`: 16 of 64). Where
`k = B` the width is the rtol-rank (polynomial B=128: 126; overparam polynomial B=1350: 411;
overparam toy B=1200: 24). The local results predate `ca8742b`; the `k = B` rerun launched on
2026-09-17 has not been synced.

That rerun is also not sufficient on its own, for a reason worth stating: **the spectrum is a
property of the trajectory, and the best configs are mostly not `k = B`.** Best Sven is `k=16`
on toy, `k=4` on MNIST-CE, `k=32` of 64 on MNIST label-reg. The `k = B` slice shows the
spectrum along a different, worse trajectory than the one in the headline plots.

### What is still missing even with `ca8742b`

1. **The residual's projection onto the singular directions.** The update is
   `sum_i (u_i . r / sigma_i) v_i`. The spectrum alone cannot say why truncation helps; the
   size of `u_i . r` in the tail can. `U` is already computed in `SvenGram.step`, so logging
   `U.T @ r` costs one B-vector per saved step.
2. **Step-level scalars**: `||delta theta||`, `||r||`, and the linearised predicted loss change
   next to the actual one. These separate "truncation suppresses noise" from "truncation is
   step-size control" (Codex Low 2).
3. **A semantic break in `sv_min`.** `_split_diagnostics` takes `np.min` of whatever was
   recorded (`generic_scan.py:211`). For old records that is the smallest *kept* singular
   value; for new records it is `sigma_B`, which is numerical noise. A scan directory that
   mixes pre- and post-`ca8742b` runs (exactly what the `k = B` rerun creates) has two meanings
   in one column. Record `sv_min_kept = sigma[nnz - 1]` explicitly.
4. **Dense early sampling.** One spectrum per 20 steps misses the first ~100 steps, where the
   spectrum moves fastest. Use a schedule: every step for the first N, then every M.
5. **A numerical floor on the tail.** The Gram is accumulated in float64 but from float32
   activations and grad-outputs, so singular values below roughly `1e-7 * sigma_max` are
   round-off, not signal. Mark that floor on every spectrum plot. For toy and polynomial, a
   float64 diagnostic rerun is nearly free and gives a true tail.

### Why per-step batch spectra are not enough (new)

Each logged spectrum is of a *different random batch*. Comparing spectra across training
therefore mixes batch-to-batch variation with genuine evolution, and the spectrum is capped at
B values by construction. With checkpoints this is solved offline:

* evaluate the Jacobian on a **fixed probe set** at every checkpoint, identical across runs
  *and across optimizers*, so spectra along Adam's trajectory can be compared with Sven's;
* for the small models use the **full dataset**. Toy has P = 593 and polynomial P = 673, so the
  10,000 x P Jacobian is 24–27 MB and its complete SVD takes seconds. That is the actual NTK
  spectrum over training, not a B-row sketch of it. MNIST (P = 27,562) works on a few thousand
  examples;
* right singular vectors, which can never be logged online (P-dimensional), become available,
  which is what Codex Low 4 (function-space vs parameter-space movement) needs.

### Proposed spec

**Optimizer (`sven`)**
* `SvenGram` / `SvenGramReg`: on logged steps also store `utr = U.T @ r` (full B),
  `update_norm`, `resid_norm`, `sv_min_kept`.
* `Sven` (classic): on logged steps only, compute the full spectrum from an fp64 `eigh` of
  `J J^T` (B x B, the cost of one Gram contraction). Do not try to recover it from `pinv()`.
* Give the optimizer a `log_this_step` flag. Today `sigma_full.cpu()` forces a device sync on
  *every* step although 19 of 20 results are thrown away by the runner.

**Runner (`generic_scan`)**
* `svd_spectra_schedule: {dense_first: 200, every: 20}`, recorded in `svd_summary`.
* `checkpoints: none | final | epochs | log`, where `log` = steps {0, 1, 2, 4, 8, ...} plus
  every epoch end. Save weights **and buffers** (BatchNorm running stats), as fp32, written
  before the npz and the jsonl so the existing "jsonl last = dedup marker" invariant holds.
* Save the init once per seed, and the per-epoch index permutations once per scan. `loader_seed`
  is shared by every run in a scan, so one small file lets any step's batch be reconstructed
  exactly.

**Storage, at roughly 35 checkpoints per run under `log`:**

| model | P | per checkpoint | per run (`log`) | suggested default |
|---|---|---|---|---|
| toy / polynomial MLP | 593 / 673 | 2.4 / 2.7 KB | ~90 KB | `log` for every run |
| MNIST MLP | 27,562 | 110 KB | 3.9 MB (4 GB per 1,020-run scan) | `log` for Sven and the best config per baseline; `final` elsewhere |
| nanoGPT (4 x 128) | ~0.8 M | 3.3 MB | 115 MB | `epochs` |
| ResNet18 | 11.2 M | 45 MB | 1.6 GB | `final` in scans (13 GB per 290 runs); `log` only in a best-config diagnostic rerun |

**What `final` alone buys, for free:** a held-out test evaluation of locked configs (Top 4), a
BatchNorm-correct re-evaluation (Top 2), example-weighted metrics (Top 8), distance from
initialisation and parameter norms for every optimizer (Low 4). Three of Codex's eight top items
stop requiring retraining.

---

## Codex's top-priority items, point by point

### 1. "Degree-4 polynomial" is an additive cubic — concur [verified]

`all_datasets.py:97-119`. `product(range(d), repeat=num_vars)` caps each exponent at `d - 1`
and yields nothing for d = 0 or 1; `term += x**power` adds where it should multiply. With the
configured degree 4 and 6 variables I get 185 terms, total degrees {2, 3, 4}, largest single
exponent 3. Each term is `1 + sum_j x_j^p_j` with `x^0 = 1`, so the target collapses to
`c + sum_j (a_j x_j + b_j x_j^2 + c_j x_j^3)`. A least-squares fit on those 19 features gives a
validation relative RMSE of **6.99e-8** (val MSE 5e-15), matching Codex's 8e-8.

Two additions. The benchmark is not useless: it is a legitimate smooth regression target, and
relabelling is a defensible alternative to rerunning, as Codex says. But the target is dominated
by cubes of Gaussian inputs, so it is heavy-tailed, and per-sample-residual methods (Sven, HIG)
are plausibly more sensitive to that than Adam. Second, `data_seed` is fixed, so every
polynomial result is for **one draw of coefficients** (see N4). If it is rerun, fix both.

### 2. CIFAR validation evaluates different models — concur, and it is worse [verified]

`train_loop_svd` never calls `.eval()` and `SvenWrapper.evaluate` (`sven_wrapper.py:150-153`)
passes live buffers into a train-mode model. My probe on `SmallResNet` reproduces both of
Codex's observations on both configurations: running means change during a validation batch,
and the prediction for a fixed example changes by O(1) when only its batch companions change.

The part Codex did not spell out concerns `freeze_norm_stats=True`, the hooks path.
`_frozen_norm_stats()` wraps `loss_and_grad` and `delta_from_w` (`gram_wrapper.py:218, 232,
267`) but **not `evaluate`**. The probe shows the consequence directly:

```
freeze_norm_stats=True, capture=hooks
  after 1 TRAIN step : running_mean changed = False
  after 1 VAL  batch : running_mean changed = True
```

So on that path the training passes normalise with running statistics that are updated *only by
validation batches*. The comment in `generic_scan.py:442-446` says frozen stats "are never
updated from their init"; in fact they are updated, from the wrong data. This is the
configuration of `exp_finetune_cifar_smallN` (hooks, freeze default true), where it quietly
adapts ImageNet BN statistics to CIFAR using the validation inputs. That is a leak of validation
data into Sven's training that no baseline has. Its net direction against the baselines is
unclear, since they adapt BN from training data, which frozen-stats Sven never does; the point
is that the comparison is not interpretable. The headline CIFAR scans use
`gram_freeze_norm_stats: false`, so there Sven is "only" evaluated with validation-batch
statistics while baselines use running stats.

Also confirmed: the standard loop's initial validation runs before `model.train()`/`.eval()` is
ever called, i.e. in train mode (`experiment_utils.py:181-200`). The effect washes out within an
epoch at momentum 0.1, so it is cosmetic next to the above.

**Action.** Give `evaluate` an eval-mode context, wrap it in `_frozen_norm_stats`-equivalent
logic, and use one normalisation policy for all optimizers. Rerun CIFAR and the fine-tune study.

### 3. GPT-2 corpus: validation is a prefix of training — concur, more urgent [read]

`prepare_tokens.py:32-47`. Both `write_split` calls run `for ex in ds` over the same streaming
`IterableDataset`, which restarts from the first document each time. I could not run the HF
stream here, so this is from reading, but the code leaves no other behaviour available. The
unfilled tail of the preallocated memmap is also real (zeros decode as token 0, `!`).

Codex found no local GPT-2 runs. However the launch log at the end of `RERUNS_NEEDED.md` says
GPT-2-small was launched on 2026-09-17 via `./submit_gpt2.sh`, against `dataset/fineweb_edu.yaml`
whose header names this script. Those results should be treated as contaminated until checked.
A ten-second check on the cluster:

```python
v = np.memmap(".../val.bin", dtype=np.uint16, mode="r")
t = np.memmap(".../train.bin", dtype=np.uint16, mode="r")
print(np.array_equal(t[:len(v)], v))   # True => overlap
```

One point beyond Codex: **the contamination is not optimizer-neutral.** Each validation block is
seen once in training. Sven at lr up to 1.0 solves the linearised per-sample problem in a single
step, so it plausibly memorises single exposures more strongly than AdamW at 3e-4. The bias
would favour Sven, which is the wrong direction for a claim to survive review.

Separately, that config has `num_epochs: 1` and validation runs once per epoch, so its stated
deliverable ("val loss vs steps and wall-time") cannot be produced: there are two validation
points, before and after.

### 4. Official test sets used for selection — concur, mild disagreement on the remedy

`all_datasets.py:47-48, 142-143` assign `train=False` to `val_dataset`; the toy dataset builds a
test split that nothing evaluates (`:16-36`). Correct, and already acknowledged as
ANALYSIS_FIXES A10 / RERUNS item 8 (deferred on 2026-09-17).

On magnitude: selection is over at most 128 configs on 10,000 examples, so the optimism is small
in absolute terms. But the MNIST tables have methods separated by 1–2 seed standard deviations
(Muon 0.0485, HIG 0.0492, Sven 0.0502 on label-reg, each +/- 0.002), which is the scale at which
it matters. So I agree it needs doing.

Where I part ways: Codex asks for fresh confirmation tasks because the test sets "have already
informed development". For an optimizer paper, carving a validation split out of the training
set, selecting on it, and reporting the official test set once for locked configs is standard
and sufficient. The leak from having looked at 20-epoch MLP scans is negligible. With `final`
checkpoints this needs no retraining beyond the reruns already planned.

### 5. Exceptions disappear; missing beats diverged — concur, and it has a direction [verified]

`generic_scan.py:583-584` (and 676, 754, 826, 892, 944) print and continue. `config_table` ranks
eligible, then fewest *diverged*, then mean (`analysis_helpers.py:116-126`); missing seeds cost
nothing. Codex's polynomial-LBFGS example checks out: the selected config has 4 finished seeds
and 1 diverged.

What Codex did not say is **which methods benefit.** Whether a blow-up becomes a recorded NaN or
a vanished file depends on the optimizer:

* first-order methods produce NaN losses, the run completes, and it is recorded as *diverged*;
* Sven, HIG and KFAC run a linear-algebra decomposition on the blown-up matrix, which raises.
  `SvenGram.step` raises explicitly (`sven.py:271-276`), `pinv` fails on an empty `.max()`
  (`pinv.py:63`), HIG's `torch.linalg.svd` and KFAC's `eigh` throw. The run is *missing*.

Counting files against the grid (4 lrs x 4 taus x 5 seeds = 80 HIG runs per scan):

| scan | HIG files | lrs present | recorded as diverged | KFAC files |
|---|---|---|---|---|
| `toy_1d_scan` | 35 / 80 | 0.05, 0.1 | 0 | 19 / 20 |
| `polynomial_scan` | 40 / 80 | 0.05, 0.1 | 0 | 11 / 20 |
| `mnist_scan_labelRegression` | 39 / 80 | 0.05, 0.1 | 0 | 0 / 20 |
| `mnist_scan_ce` | 80 / 80 | all four | 0 | 0 / 20 |

HIG is the method that beats Sven on toy (3.1e-7 vs 3.0e-6) and polynomial (3.0e-3 vs 1.2e-2).
Its best row reads `n_seeds=5, n_diverged=0, n_missing=0`, while half its grid crashed — a
config with no files at all is invisible to `config_table`. Sven's headline grids are complete
(360 / 360 / 640 / 640), so today the asymmetry flatters Sven's closest competitor on
robustness, not Sven. In the param-fraction scans it is Sven's own blow-ups at f <= 0.25 that
vanish. Either way the robustness story cannot be told from the current records.

A second loss: when a run raises at epoch 15, the first 14 epochs of curves are discarded too.

**Action.** In every `except`, write a record with `status` (`diverged` / `oom` / `timeout` /
`error`), the exception text and the partial curves. Reconcile against an explicit manifest, as
Codex says. Show `finished / attempted` beside every "best config" row.

### 6. Timing summaries — concur on the defects, disagree on the rank [verified]

Both defects are real. The batch timers stop before `loss.item()`
(`experiment_utils.py:237-239`), while Sven syncs inside the timed region through `.item()` and
`.cpu()`, so `avg_batch_time_train` measures launch latency for baselines and completed work for
Sven. And `summarize` drops points beyond 3 MAD (`optimizer_profile.py:97-101`), which removes
SOAP's every-10-steps preconditioner refresh.

I reproduced Codex's SOAP numbers on `profile_mnist` (6.142 ms unfiltered vs 5.556 ms filtered).
But the same table shows the filter is not a pro-Sven bias:

| method | filtered `steady_mean` (ms) | mean, last 80% (ms) | ratio |
|---|---|---|---|
| SOAP | 5.56 | 6.14 | 1.105 |
| Sven `gram_full` | 17.62 | 19.37 | 1.099 |
| Sven `classic` | 14.45 | 15.51 | 1.073 |
| LBFGS3 | 15.28 | 17.17 | 1.124 |
| Sven `gram_hooks` (headline) | 10.40 | 10.40 | 1.000 |
| Adam / SGD / KFAC / HIG | | | 1.00–1.01 |

A 10% correction, applied about equally to Sven variants and baselines, against method gaps of
2x to 30x. It should be fixed — report the amortised mean over whole update cycles — but it is a
reprocessing of the saved raw lists, not new data, and it changes no conclusion.

The unsynchronised batch timer matters more for ResNet and GPT than for the launch-bound MLPs,
and it feeds only the `avg_batch_time_train` panels in the four scan notebooks. The headline
wall-time plots use epoch and total times, which include the syncs, as Codex itself notes.

Verdict: mid priority. Add `torch.cuda.synchronize()` before each timer read and re-summarise.

### 7. "AdamW" is Adam — concur on the facts, but this is already handled

Confirmed in the tables: AdamW and Adam rows agree to every printed digit in all four headline
scans. This is ANALYSIS_FIXES D29. `resolve_weight_decay` fixes the code, AdamW `run_id`s now
carry `_wd`, and the reruns were launched 2026-09-17 with the old files moved to `_adamw_wd0/`.
It is a correct observation about the local files rather than a new finding.

Two live remnants. `submit_fresh_suite.sh:49` still has `FIRST='[Adam,AdamW,SGD,RMSprop,Muon,
PolyakSGD]'` with no `MuonW`, so a fresh suite launch silently drops it; the one-off rerun
launcher covered it, the permanent one does not. And a single default weight decay is not a
tuned AdamW; Codex is right that it needs an lr x wd search to count as a regularised baseline.

### 8. Epoch metrics weight batches, not examples — concur on the defect, disagree on the rank [verified]

`np.mean` over batch means (`experiment_utils.py:276, 375`). The stored per-batch validation
losses let me recompute the example-weighted final validation loss and compare:

| scan | batch sizes | median rel. error | 95th pct | max |
|---|---|---|---|---|
| `toy_1d_scan` | 32 | 0.07% | 0.14% | 0.34% |
| `polynomial_scan` | 32 | 0.10% | 0.38% | 1.7% |
| `mnist_scan_ce` | 64 | 0.41% | 0.48% | 0.79% |
| `mnist_scan_labelRegression` | 64 | 0.25% | 0.45% | 0.58% |
| `cifar10_resnet_ce_scan` | 128 | 0.17% | 0.71% | 1.1% |
| `exp_critbatch_mnist` | 8–512 | 0.07% | 1.8% | 2.2% |
| `rebuttal_batchsize_polynomial_scan` | 8–256 | 0.15% | 2.3% | 9.0% |
| `rebuttal_overparam_polynomial_scan` | 170–1350 | 0.19% | 3.3% | 33% |

In the headline scans the error is a fraction of a percent, far inside the seed spread, and at
fixed batch size every method sees the same examples with the same weights, so it largely
cancels in comparisons. It deserves attention only in the large-batch and full-batch studies,
where a few runs move by 10–30%. It is **exactly repairable offline** from `val_batch` and the
known batch lengths (accuracy is not stored per batch and needs checkpoints). Codex's closing
caveat is right and worth keeping: online train loss spans changing parameters, and LBFGS
records its last closure evaluation. Verdict: fix it going forward, repair with a script, low
priority.

---

## Codex's mid-priority items

**Mid 1, baseline recipes and tuning budget — concur, with one correction to the arithmetic.**
Sven's grid is not 72 / 128 independent configurations. Whenever the rtol-rank stays below k,
larger k is the identical trajectory. Counting distinct final-loss values per (lr, seed)
[verified]: toy 12 of 18, polynomial 11.3 of 18, MNIST-CE 22.4 of 32, MNIST label-reg 16 of 32.
So the effective budget is roughly 48 / 45 / 90 / 64 against 4, i.e. 11–22x rather than 18–32x.
The conclusion stands.

Boundary optima [verified]: on MNIST-CE, **seven of eleven baselines** select the lowest lr in
the grid (Adam, AdamW, RMSprop, Muon, SOAP, JD at 1e-4; HIG at 0.05; AdamW being the Adam
duplicate of Top 7, so six distinct). Elsewhere SGD, Shampoo and
KFAC sit on an edge in several scans. Sven is at a corner too: polynomial picks `k = B` with
the smallest rtol, CIFAR-CE the smallest lr, CIFAR label-reg the largest. See N5 for why the
MNIST-CE pattern is more than a grid artefact.

On Muon, Codex is right (`experiment_utils.py:626-635`) and there is more:
* conv kernels are 4-D, so on ResNet18 "Muon" applies Muon to the final `fc` layer only and
  AdamW to everything else. The CIFAR fine-tune "Muon" and "MuonW" baselines are AdamW in all
  but name. Standard practice flattens kernels to 2-D;
* the Muon and AdamW halves share one lr. With PyTorch's default `adjust_lr_fn` the two
  typically want very different learning rates, so a shared value handicaps one of them. Use
  `adjust_lr_fn="match_rms_adamw"` (check it exists in the cluster's torch version) or sweep
  the two separately.

Schedule-free: confirmed, the local classes add a first-moment EMA (AdamW) and a Nesterov
momentum buffer (SGD) that the reference algorithms do not have. Inactive in saved results.

**Mid 2, reproducibility — concur [read].** `set_seed` runs once per model seed
(`generic_scan.py:390`). Masks draw `torch.randperm` from the global CPU generator
(`sven_wrapper.py:397`), randomized SVD draws `torch.randn` from the global CUDA generator
(`pinv.py:105`), every `instantiate(cfg.model)` consumes init draws, and each validation
`DataLoader` iterator takes a base seed from the global RNG. A masked run's randomness therefore
depends on how many runs preceded it in its process, which sharding and dedup-skipping change.
Unmasked Gram runs consume no RNG in training, which is why the timing reruns reproduced at
0.00 relative deviation; that result does not extend to the param-fraction scans. On run
identity and stale dedup, see N6.

**Mid 3, separate efficiency from problem changes — concur.** I confirmed the specific claim
about synthetic data: toy and polynomial draw train then validation inputs from one generator
and normalise targets by the training mean and std (`all_datasets.py:25-32, 107-124`), so
changing `n_train` changes the validation inputs *and* the target scale. The overparam scans
therefore compare losses measured in different units on different points at each P/N. Generate
the evaluation set from its own seed and fix the normalisation constants.

**Mid 4, strengthen existing workloads — concur.** torchvision's ResNet18 applies a stride-2
7x7 stem and a max-pool to 32x32 inputs, which is a weak CIFAR recipe, with no augmentation.
All methods share it, so it is fair, but it is not evidence of competitiveness. How much to
invest here is the authors' call; I would do it after the infrastructure fixes.

**Mid 5, uncertainty and claim scope — concur.** The task-dependence is visible in the current
tables: HIG leads on toy, polynomial and MNIST-CE; Muon on MNIST label-reg; RMSprop and Adam on
CIFAR-CE, where Sven is 5th of 6; Adam on CIFAR label-reg, where Sven is 2nd. On complexity, `RERUNS_NEEDED` item 3 already
records that wall time is flat in k under the Gram backend, because the `eigh` is B x B
regardless. `CLAUDE.md` still states the cost as O(kN|D|); that describes the classic randomized
path, not what the experiments run. See N3 on why the seed bands understate variance.

## Codex's low-priority items

**Low 1 — concur, and the kappa point is an exact identity.** With rows `r = l^(kappa/2)`, the
row Jacobian is `(kappa/2) l^(kappa/2 - 1) J_l`. In the full-row-rank, untruncated solve each
row gives `J_l delta = -(2/kappa) l`, so **the kappa step is exactly `2/kappa` times the
kappa = 2 step.** `mnist_kappaScan_labelRegression` runs `k = B = 64`, `rtol = 1e-4`, fixed
`lr = 0.5`: close to that regime. There kappa = 1 is lr = 1.0 and kappa = 3 is lr = 0.33. As
configured the ablation is nearly a three-point learning-rate sweep. Retune lr per kappa, or run
it where truncation binds. The additive-constant diagnostic is a good idea: it leaves gradients
untouched and changes Sven's step.

**Low 2 — concur.** This is what the `U.T @ r` logging above is for. A damped solve already
exists as `SvenGramReg` (`damping`), so the hard-vs-soft truncation comparison needs only a
config, not new optimizer code.

**Low 3 — concur.** `actual_param_fraction` is computed by the wrappers but never written to the
result record; one line in `generic_scan` fixes that. Note that under chunked capture the mask
does not reduce compute (the module docstring says so), so "matched time" needs measuring.

**Low 4 — concur.** Needs checkpoints. `track_param_norm` is on only in the fine-tune config.

---

## New critiques

Ordered by priority. The checkpoint and SVD logging item above is the first of these.

**N1. Failure recording is asymmetric by optimizer family.** Detailed under Top 5. It is a
distinct finding because it determines who benefits, and because the comparison it distorts
(HIG vs Sven) is the one where Sven currently loses.

**N2. Two studies are running with bugs that are already understood.** Per the 2026-09-17
launch log: GPT-2-small (train/val overlap, Top 3) and `exp_finetune_cifar_smallN` (BN
statistics learned from validation batches, Top 2). The GPT-2 bias plausibly favours Sven; the
fine-tune one makes Sven's training depend on validation inputs, direction unclear. I cannot see
the cluster, so check whether they are still running; either way, do not let their output reach
a plot before the fixes.

**N3. Seed bands measure initialisation variance only.** `loader_seed` is a single value per
config and every run builds `torch.Generator().manual_seed(loader_seed)`
(`generic_scan.py:537-541`). All five model seeds, and every optimizer, see the identical batch
sequence. That is good for paired comparisons and should be kept across *methods*. But the
"+/- 1 std over seeds" band then excludes data-order variance, so it understates run-to-run
spread, and every result is conditional on one shuffle. Derive the loader seed from the model
seed.

**N4. Each synthetic benchmark is one draw.** `data_seed` is fixed, so "Random Polynomial" is a
single set of coefficients and toy is a single sample. Claims about the family need a few
dataset seeds, at least for the best configs.

**N5. On MNIST-CE, last-epoch validation loss selects for slow training.** Seven of eleven
baselines choose the smallest lr in the grid, and train losses sit far below validation (Polyak:
0.022 train, 0.137 val). The models overfit within 20 epochs, so the metric rewards whichever
setting has trained least. I am not reopening the decision to select on seed-mean final
validation loss. The consequence within that decision is that the MNIST-CE baseline grids are
truncated from below and must extend until each optimum is interior, otherwise the table
compares Sven against baselines the metric says are mis-tuned. It also bears on the open
decision D28: a best-epoch or early-stopped value, logged alongside, would show how much of each
gap is optimisation and how much is incidental regularisation.

**N6. Dedup-by-existence will silently keep stale results after these fixes.** `run_id` omits
`num_epochs`, the dataset seed, `n_train`, the spectra schedule and the code version. Once
BatchNorm evaluation is fixed, a resubmit of the CIFAR configs skips all 580 existing files.
The 2026-09-17 reruns handled this by moving files aside by hand, which does not scale. Store a
hash of the resolved config and the git SHAs of **both** repos (plus a dirty flag) in every
record, and have dedup compare the hash. The editable `sven` install makes the second SHA
essential: the spectra semantics changed at `ca8742b` with no trace in any result file.

**N7. Smaller items.**
* `torch.cuda.empty_cache()` runs on every Sven step, twice in the classic path
  (`sven.py:175-176, 202-203, 295-296`). It forces a sync and an allocator flush, which
  penalises Sven's own timings and, under NPROC sharding, its neighbours'. Profile without it.
* "LBFGS" is `torch.optim.LBFGS` on minibatches, keeping curvature pairs across different
  batches. That is a known-unstable construction; its 439 divergences in the batch-size scan say
  more about the baseline than about L-BFGS. Label it as stochastic L-BFGS.
* Sven with microbatching uses `drop_last=True` (`generic_scan.py:540`); baselines do not. The
  data differs slightly between the arms.
* The validation batch size equals the training batch size. Under train-mode BN evaluation this
  makes Sven's validation numbers batch-size-dependent, which contaminates the batch-size study
  beyond Top 8.

---

## Suggested order

1. Land the logging work: checkpoints (`final` everywhere, `log` for the MLP scans), `U.T @ r`,
   `sv_min_kept`, the dense-early schedule, failure records, config hash and git SHAs. One
   change to the runner, one to `sven`. Everything after this is rerun once, not twice.
2. Check the two in-flight studies (N2) and the `val.bin` prefix test.
3. Fix BatchNorm evaluation and `prepare_tokens.py`; fix or relabel the polynomial.
4. Offline, no new runs: re-summarise the profiles, repair batch-weighted validation losses,
   reconcile every scan against its grid and report `finished / attempted`.
5. Rerun in this order: best configs of every method with `log` checkpoints (this is what the
   spectrum plots should read), CIFAR, the polynomial scans, then the baseline lr/wd extensions
   where optima sit on a boundary.
6. Then Codex's mechanism experiments, which by that point need mostly analysis of checkpoints
   rather than new training.

## Appendix: what was run

All on CPU from the repo root; scripts are in the session scratchpad and are a few lines each.

* **Spectrum audit**: for every `svd_*.jsonl`, load `diag/*.npz` and compare the count of finite
  entries per `svs` row with `batch_size // microbatch_size` and `k`.
* **Polynomial**: instantiate `RandomPolynomialDataset(degree=4, num_vars=6, seed=2000)`, inspect
  `power_combinations`, least-squares fit on `[1, x, x^2, x^3]` per coordinate.
* **BatchNorm probe**: `SmallResNet(width=4, num_blocks=1)` under `GramSvenWrapper` with
  (`hooks`, freeze) and (`full`, no freeze); one train step, one `evaluate`, then the same
  example with different batch companions. Needs `PYTHONPATH=. .venv/bin/python`.
* **Batch-weighting audit**: recompute the final-epoch validation loss from `val_batch` with
  true batch lengths; asserted that the equal-weight mean reproduces the stored value.
* **Profile filter**: recompute the unfiltered mean of the last 80% of `raw.step_ms` for every
  method in `profile_results_v2/profile_mnist/methods__*.json`.
* **Best-config tables**: `analysis_helpers.best_per_method` on the six headline scans, with a
  check for the selected lr lying on a grid edge.
* **Distinct Sven trajectories**: count distinct final validation-loss triples per (lr, seed).
* **HIG / KFAC manifest**: count result files against the configured grid.

Not run: anything on GPU, the HF streaming path in `prepare_tokens.py`, and any check of the
cluster's state or of results not synced to this checkout.
