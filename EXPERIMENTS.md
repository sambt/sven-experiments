# Sven Experiment Taxonomy

Every experiment run for the paper, grouped by purpose. All optimizer runs now use the
exact **Gram-trick** backend; the hyperparameter columns give the full swept grid.

**28 experiments · 10 families · 3 tiers · 6 architectures · params 593 → 11.18M**

*(The extra-baseline scans have been consolidated into the headline convergence configs —
those now run the full optimizer suite. See "Headline convergence" and the note below.)*

## Shared conventions

- **Sven backend.** Every Sven run uses the exact Gram trick (`use_gram: true`) — the `B×B`
  Gram matrix `G = JJᵀ`, an identical update to the classic truncated-SVD pipeline. Capture is
  `hooks` (one weighted backward) unless a row is marked **[chunked]** (required only to mask a
  conv-net's BatchNorm). The update rank is capped at the (micro)batch size: `k ≤ B`. Default
  `κ = 2`.
- **Scans.** Each experiment sweeps the full Cartesian product of the listed hyperparameters ×
  seeds. `k` = SVD truncation rank (absolute, or a fraction of `B`); `rtol` = singular-value
  cutoff; `pf` = parameter fraction; `mb` = micro-batch size; `κ` = residual exponent.
- **Optimizer sets.** **Sven-Gram** is the method under test. **CORE** = Adam, SGD, PolyakSGD,
  RMSprop, LBFGS. **FULL** = CORE + AdamW, Muon, SOAP, Shampoo, K-FAC. Standard baselines sweep
  `lr ∈ {1e-4 … 1e-1}`. Rows with no baseline set are Sven-only (`mode: svd`).

### Architectures & parameter counts

| Key | Architecture | Params |
|---|---|---|
| Toy-1D MLP | `MLP 1→[16,16,16]→1`, GELU | 593 |
| Poly MLP | `MLP 6→[16,16,16]→1`, GELU | 673 |
| MNIST MLP | `MLP 784→[32,32,32]→10`, GELU | 27,562 |
| CIFAR ResNet18 | ResNet18 with torch.func-compatible BatchNorm | 11,181,642 |
| ResNet18 (pretrained) | ImageNet-pretrained ResNet18, fresh 10-way head | 11,181,642 |
| nanoGPT | GPT 4 layers · 4 heads · 128-d, block 128, untied, dropout 0 | 826,368 |

---

## Tier 1 — Core paper experiments

The experiments behind the submitted manuscript, re-run on the Gram backend so every result
sits on one footing.

### Headline convergence
*The central claim — in the over-parametrized (P>B) regime Sven drives training loss down faster
than first- and second-order baselines. Full data.*

| Experiment | Dataset (loss) | Model · params | Hyperparameter scan | Optimizers |
|---|---|---|---|---|
| `toy_1d_scan` | Toy 1-D (MSE) | Toy-1D MLP · 593 | k∈{1,2,4,8,16,32}, lr∈{.05,.1,.5,1}, rtol∈{1e-4,1e-3,1e-2}, 10 seeds, 20 ep | Sven-Gram + FULL |
| `polynomial_scan` | Random polynomial (MSE) | Poly MLP · 673 | k∈{1,2,4,8,16,32}, lr∈{.05,.1,.5,1}, rtol∈{1e-4,1e-3,1e-2}, 10 seeds, 20 ep | Sven-Gram + FULL |
| `mnist_scan_labelRegression` | MNIST (label-reg MSE) | MNIST MLP · 27,562 | k∈{1…64, 8 vals}, lr∈{.05,.1,.5,1}, rtol∈{1e-4…1e-1}, 10 seeds, 20 ep | Sven-Gram + FULL |
| `mnist_scan_ce` | MNIST (cross-entropy) | MNIST MLP · 27,562 | k∈{1…64, 8 vals}, lr∈{.05,.1,.5,1}, rtol∈{1e-4…1e-1}, 10 seeds, 20 ep | Sven-Gram + FULL |
| `cifar10_resnet_scan_labelRegression` | CIFAR-10 (label-reg) | CIFAR ResNet18 · 11.18M | k∈{64,128}, lr∈{.1,.5,1}, rtol∈{1e-4,1e-3,1e-2}, κ=2, 1 seed, 20 ep · **[hooks]** | Sven-Gram + CORE |
| `cifar10_resnet_ce_scan` | CIFAR-10 (cross-entropy) | CIFAR ResNet18 · 11.18M | k∈{64,128}, lr∈{.1,.5,1}, rtol∈{1e-4,1e-3,1e-2}, κ=2, 1 seed, 20 ep · **[hooks]** | Sven-Gram + CORE |

### Micro-batch scaling
*Aggregating samples into micro-batches shrinks the Gram row dimension M — the memory / update-rank
trade-off. Sven only.*

| Experiment | Dataset (loss) | Model · params | Hyperparameter scan | Optimizers |
|---|---|---|---|---|
| `toy_1d_microbatch_scan` | Toy 1-D (MSE) | Toy-1D MLP · 593 | mb∈{1,2,4,8,16,32}, k∈{16,32}, lr∈{.05,.1,.5,1}, rtol 1e-3, 10 seeds | Sven-Gram |
| `polynomial_microbatch_scan` | Random polynomial (MSE) | Poly MLP · 673 | mb∈{1…32}, k∈{16,32}, lr∈{.05,.1,.5,1}, rtol 1e-2, 10 seeds | Sven-Gram |
| `mnist_microbatch_labelreg_scan` | MNIST (label-reg) | MNIST MLP · 27,562 | mb∈{1…64, 7 vals}, k∈{32,64}, lr∈{.05,.1,.5,1}, rtol 1e-4, 10 seeds | Sven-Gram |
| `mnist_microbatch_ce_scan` | MNIST (cross-entropy) | MNIST MLP · 27,562 | mb∈{1…64, 7 vals}, k·frac∈{.25,.5,1}, lr∈{.05,.1,.5,1}, rtol 1e-2, 3 seeds | Sven-Gram |

### Parameter-fraction (stochastic subset)
*Updating a random subset of parameters each step (Fig-5). Tests that Sven barely degrades at
fractions ≥ 25% and that masked-Gram makes any fraction cheap.*

| Experiment | Dataset (loss) | Model · params | Hyperparameter scan | Optimizers |
|---|---|---|---|---|
| `toy_1d_paramfrac_scan` | Toy 1-D (MSE) | Toy-1D MLP · 593 | pf∈{.1,.25,.5,.75,1}, k∈{16,32}, lr∈{.05,.1,.5,1}, rtol 1e-3, 10 seeds | Sven-Gram |
| `polynomial_paramfrac_scan` | Random polynomial (MSE) | Poly MLP · 673 | pf∈{.1…1, 5 vals}, k∈{16,32}, lr∈{.05,.1,.5,1}, rtol 1e-2, 10 seeds | Sven-Gram |
| `mnist_paramfrac_labelreg_scan` | MNIST (label-reg) | MNIST MLP · 27,562 | pf∈{.1…1, 5 vals}, k∈{32,64}, lr∈{.05,.1,.5,1}, rtol 1e-4, 10 seeds | Sven-Gram |
| `mnist_paramfrac_ce_scan` | MNIST (cross-entropy) | MNIST MLP · 27,562 | pf∈{.1…1, 5 vals}, k∈{8,16,32,64}, lr∈{.05,.1,.5,1}, rtol 1e-2, 3 seeds | Sven-Gram |
| `cifar10_resnet_paramFrac_scan_labelReg` | CIFAR-10 (label-reg) | CIFAR ResNet18 · 11.18M | pf∈{.05,.1,.25,.5,1}, k=128, lr .1, rtol 1e-4, 1 seed · **[chunked]** | Sven-Gram |
| `cifar10_resnet_ce_paramFrac_scan` | CIFAR-10 (cross-entropy) | CIFAR ResNet18 · 11.18M | pf∈{.05,.1,.25,.5,1}, k=128, lr .1, rtol 1e-4, 1 seed · **[chunked]** | Sven-Gram |

### κ (residual-exponent) ablation
*Sensitivity to the residual-decomposition exponent κ — the κ=1 theory vs the κ=2 default
implementation that avoids NaNs.*

| Experiment | Dataset (loss) | Model · params | Hyperparameter scan | Optimizers |
|---|---|---|---|---|
| `mnist_kappaScan_labelRegression` | MNIST (label-reg) | MNIST MLP · 27,562 | κ∈{1,1.5,2,2.5,3}, k=64, lr .5, rtol 1e-4, 10 seeds | Sven-Gram |
| `cifar10_resnet_kappaScan_labelReg` | CIFAR-10 (label-reg) | CIFAR ResNet18 · 11.18M | κ∈{1,1.5,2,2.5,3}, k=128, lr .1, rtol 1e-4, 1 seed · **[hooks]** | Sven-Gram |
| `cifar10_resnet_ce_kappaScan` | CIFAR-10 (cross-entropy) | CIFAR ResNet18 · 11.18M | κ∈{1,1.5,2,2.5,3}, k=128, lr .1, rtol 1e-4, 1 seed · **[hooks]** | Sven-Gram |

---

## Tier 2 — Rebuttal additions

New experiments answering the NeurIPS reviews — the requested baselines, and the
dataset-overparam regime the theory actually targets.

### Extra baselines (R1 / R2) — consolidated into the headline scans
*The reviewer-requested full optimizer suite (AdamW, Muon, SOAP, Shampoo, K-FAC on top of
CORE) has been **folded into the headline convergence configs** above — `toy_1d_scan`,
`polynomial_scan`, and `mnist_scan_labelRegression` now run Sven-Gram + FULL. Their result
directories hold the union of the original CORE runs and the added baselines; for polynomial
and MNIST that union spans two loader seeds / LBFGS grids (the added-baseline runs used a
different loader seed and, for MNIST, a reduced k/lr/rtol×5-seed sub-grid), so the directory
documents all runs rather than a single reproducible Cartesian grid. The former
`rebuttal_baselines_*` configs and dirs are archived under `_backup_2026-08-31/…__premerge`.*

### Dataset-overparam, P > N (R1 crux)
*The decisive rebuttal experiment: subsample the training set to trace performance across the
P/N = 1 boundary. Synthetics run full-batch (B = N); MNIST keeps the paper's minibatch setup.*

| Experiment | Dataset (loss) | Model · params | Hyperparameter scan | Optimizers |
|---|---|---|---|---|
| `rebuttal_overparam_toy_1d_scan` | Toy 1-D, N∈{150,300,600,1200}, full-batch | Toy-1D MLP · 593 | k·frac∈{.25,.5,1}, lr∈{.05,.1,.5,1}, rtol∈{1e-4,1e-3,1e-2}, 5 seeds, 200 ep | Sven-Gram + FULL |
| `rebuttal_overparam_polynomial_scan` | Poly, N∈{170,340,675,1350}, full-batch | Poly MLP · 673 | k·frac∈{.25,.5,1}, lr∈{.05,.1,.5,1}, rtol∈{1e-4,1e-3,1e-2}, 5 seeds, 200 ep | Sven-Gram + FULL |
| `rebuttal_overparam_mnist_scan` | MNIST, N∈{2.5k,5k,10k,20k,40k,60k}, B=64 | MNIST MLP · 27,562 | k∈{16,32,48,64}, lr∈{.1,.5,1}, rtol∈{1e-3,1e-2}, 5 seeds, 20 ep | Sven-Gram + FULL |

### Batch-size sensitivity (R2)
*How Sven and the baselines respond to batch size on a fixed task — Sven's rank tracks B.*

| Experiment | Dataset (loss) | Model · params | Hyperparameter scan | Optimizers |
|---|---|---|---|---|
| `rebuttal_batchsize_polynomial_scan` | Random polynomial (MSE) | Poly MLP · 673 | B∈{8,16,32,64,128,256}, k·frac∈{.5,1}, lr∈{.05,.1,.5,1}, rtol∈{1e-4,1e-3,1e-2}, 5 seeds | Sven-Gram + FULL |

### Fig-5 at scale — ResNet param-fraction (R1)
*The paper's Fig-5 re-run on the exact Gram backend with 3 seeds, now reporting accuracy, peak
memory and step time. Chunked capture masks the conv-net's BatchNorm.*

| Experiment | Dataset (loss) | Model · params | Hyperparameter scan | Optimizers |
|---|---|---|---|---|
| `rebuttal_fig5_cifar_paramfrac_scan` | CIFAR-10 (label-reg) | CIFAR ResNet18 · 11.18M | pf∈{.05,.1,.25,.5,1}, k=128, lr .1, rtol 1e-4, 3 seeds, 20 ep · **[chunked]** | Sven-Gram |

---

## Tier 3 — Large-scale (Gram-enabled)

The new frontier the Gram trick unlocks — Sven at Adam-class cost on transformers and real
fine-tuning. Benchmarked against the modern optimizer set.

### nanoGPT language modeling
*Places Sven in the modern optimizer arena (transformer pretraining), the extreme-P regime the
theory targets. Untied embeddings + dropout 0 keep it on the fast hooks path. Reports val loss →
perplexity vs steps and wall-time.*

| Experiment | Dataset (loss) | Model · params | Hyperparameter scan | Optimizers |
|---|---|---|---|---|
| `exp_nanogpt_speedrun` | tiny-shakespeare, char (LM CE) | nanoGPT · 826,368 | k∈{32,64}, lr∈{.05,.1,.5,1}, rtol 1e-3, 3 seeds, 50 ep · **[hooks]** | Sven-Gram + AdamW, Muon, SOAP |

### Fine-tuning, genuine P ≫ N
*Fine-tune an ImageNet-pretrained ResNet18 on a small labelled set — the honest P≫N regime R1
demanded. Sweeps the training-set size and tracks the parameter norm for the
min-norm / generalization story.*

| Experiment | Dataset (loss) | Model · params | Hyperparameter scan | Optimizers |
|---|---|---|---|---|
| `exp_finetune_cifar_smallN` | CIFAR-10, N∈{250,500,1000,2000} (label-reg) | ResNet18 pretrained · 11.18M | k∈{32,64}, lr∈{.01,.05,.1,.5}, rtol 1e-3, 3 seeds, 30 ep, ‖θ‖ tracked · **[hooks]** | Sven-Gram + AdamW, SGD, Muon |

### Critical batch size
*McCandlish-style steps-to-target vs batch size — distinctive for Sven because the update rank is
capped at B. k = B/2 so the rank scales with the batch.*

| Experiment | Dataset (loss) | Model · params | Hyperparameter scan | Optimizers |
|---|---|---|---|---|
| `exp_critbatch_nanogpt` | tiny-shakespeare, char (LM CE) | nanoGPT · 826,368 | B∈{8,16,32,64,128,256}, k=B/2, lr∈{.05,.1,.5,1}, rtol 1e-3, 2 seeds, 40 ep · **[hooks]** | Sven-Gram + AdamW |
| `exp_critbatch_mnist` | MNIST (label-reg) | MNIST MLP · 27,562 | B∈{8…512, 7 vals}, k=B/2, lr∈{.1,.5,1}, rtol 1e-3, 2 seeds, 20 ep | Sven-Gram + AdamW |

---

## Caught numerical failures at aggressive settings (expected; not harness bugs)

A handful of runs are **permanently absent** because the optimizer blows up or hits a
singular matrix at aggressive/degenerate hyperparameters. Every such run is caught by a
`try/except`, logged as `[error] … failed`, and skipped **without writing a result file**;
the failure is deterministic for a given (seed, data), so re-submitting reproduces it. In
every case the *converging* grid points are present — only the divergent/singular corners
are missing, and the analysis notebooks simply plot what survived. Three sources:

**1. K-FAC — singular Kronecker-factored Fisher.** In the over-parametrized / small-data
regimes the Fisher is rank-deficient, so the `torch.linalg.eigh` that inverts it fails to
converge. This is on-message: K-FAC only "exists" there via damping (a biased, non-min-norm
update) — the point made in the R3 response. Only the **FULL**-suite configs include K-FAC;
**CORE** (headline / κ / micro-batch / all CIFAR configs) does not, so those are unaffected.
Caught-failure counts, as measured:

| Config | K-FAC runs missing |
|---|---|
| `toy_1d_scan` (headline) | ~6 |
| `polynomial_scan` (headline) | ~10 |
| `mnist_scan_labelRegression` (headline) | ~20 |
| `rebuttal_batchsize_polynomial_scan` | ~15 |
| `rebuttal_overparam_mnist_scan` | ~44 |
| `rebuttal_overparam_toy_1d_scan`, `rebuttal_overparam_polynomial_scan` | 0 (full-batch synthetic Fisher stays well-conditioned) |

**2. SOAP — high-LR hang.** At `lr = 0.1` (~30× SOAP's sane LR) SOAP diverges and *deadlocks*
a CUDA op (rather than raising), which hangs the whole scan process. Confirmed deterministic
(hangs identically at NPROC=1, so not GPU contention). Affected the MNIST FULL-suite configs
(`mnist_scan_labelRegression`, `rebuttal_overparam_mnist`); ~3 SOAP points each are absent. SOAP
is present and valid at its lower LRs. (Because a hang blocks the shard, the standard runs
queued *after* it were recovered by re-running that phase with SOAP excluded.)

**3. Sven (Gram) — `eigh` non-convergence at high LR or low κ.** Sven's own `torch.linalg.eigh`
on the `B×B` Gram fails when training diverges and the (masked) Gram goes NaN / ill-conditioned.
Two triggers, both expected: **high LR** — `mnist_paramfrac_labelreg` (~34), `mnist_paramfrac_ce`
(~12), `polynomial_paramfrac` (~19) missing, all at `lr = 0.5 / 1.0`; and **low κ** —
`cifar10_resnet_ce_kappaScan` missing κ = 1 and 1.5 (the κ=1 theory value is the numerically
unstable one that κ=2 exists to avoid — exactly the κ=1-vs-κ=2 tradeoff the paper discusses).

The same `try/except` also catches the occasional Shampoo NaN (rare, not systematic).

---

## Not re-run

Present in the repo but excluded from the fresh Gram suite:

- `mnist_scan`, `mnist_microbatch_scan` — legacy configs (large LRs up to 100), superseded by
  `mnist_scan_ce` and the label-regression
  micro-batch scan.
- `rebuttal_mnist_batchk_probe` — a one-off batch/k diagnostic that already served its purpose.
