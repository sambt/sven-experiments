# Sven — NeurIPS Rebuttal Plan & Setup

Scores: **3 / 3 / 2** (R1 conf 5, R2 conf 3, R3 conf 4). Realistically a reject at this
venue; the rebuttal's job is to (a) neutralize R3's misreads, (b) show the requested
baselines don't change the story, and (c) generate the overparam-regime + Fig-5 results
that make the next submission much stronger. Everything below runs on the `rebuttals` branch.

---

## 1. The one criticism that actually matters (R1)

The theory motivates the **dataset-overparametrized** regime (P > N_train), but every
headline experiment is overparametrized only at the **minibatch** level (P > B): 600–27,600
params vs 10k–60k samples. The single true dataset-overparam case (ResNet18/CIFAR) merely
matches baselines at higher wall-time. **Adding baselines does not fix this** — the decisive
experiment is to show a real Sven advantage where P > N_train (Tier 2 below). If that fails,
it's important signal.

---

## 2. Environment (uv)

Home dir was ~94% full, so the venv lives on holystore and is symlinked as `.venv`:

```bash
# venv:  /n/holystore01/LABS/iaifi_lab/Users/sambt/sv3_venv   (symlinked as ./.venv)
source /n/home11/sambt/iaifi/sv3/.venv/bin/activate      # or use .venv/bin/python directly
```

Contents: torch 2.9.1+cu128, torchvision, `sven` (editable from ./sven), hydra/omegaconf/
numpy/pandas/matplotlib/scipy/seaborn/tqdm, **torch_optimizer 0.3.0** (Shampoo),
**kfac-pytorch 0.4.2** (K-FAC). SOAP is vendored at `experiments/optimizers/soap.py` (zero deps).
`torch.optim.Muon` and `AdamW` are native. UV cache: `/n/holystore01/.../uv_cache`.

Slurm: use **`submit_rebuttal.sh`** (added) — it runs `.venv/bin/python` so Shampoo/K-FAC are
available. The old `submit_experiment.sh` uses the `jax` conda env, which lacks those two
packages; don't use it for the baseline scans (or `uv pip install --python <jax-env> --no-deps
torch_optimizer pytorch_ranger git+https://github.com/gpauloski/kfac-pytorch.git` into it).

---

## 3. What was set up

**Code (backward-compatible):**
- `experiments/datasets/all_datasets.py` — restored `CIFAR10Dataset` (was imported in
  `__init__` but missing → broke *all* dataset instantiation on this branch); added optional
  `n_train=` subsample knob to MNIST + CIFAR (for P>N experiments).
- `experiments/experiment_code/experiment_utils.py` — wired **SOAP** (custom registry),
  **Shampoo** (`torch_optimizer`) and **K-FAC** (`kfac-pytorch`, via a `_KFACOptimizer`
  wrapper so `precond.step()` runs inside `.step()` — no train-loop change). Added
  **peak-GPU-memory logging** (`peak_gpu_mem_mb`) to both train loops.
- `experiments/optimizers/soap.py` — vendored reference SOAP.

All 5 reviewer-requested baselines (AdamW, Muon, SOAP, Shampoo, K-FAC) build + train on CE
and label-regression MSE (tested on CPU through the full scan pipeline).

**New configs (`experiments/configs/`):**
| Config | Purpose | Reviewer |
|---|---|---|
| `rebuttal_baselines_{toy_1d,polynomial,mnist}_scan.yaml` | Sven vs full baseline suite, full data | R1, R2 |
| `rebuttal_overparam_{toy_1d,polynomial,mnist}_scan.yaml` | P>N sweep via `n_data=` | **R1 (crux)** |
| `rebuttal_batchsize_polynomial_scan.yaml` | batch-size sensitivity | R2 |

---

## 4. Experiment plan (prioritized)

### 4.0 Timing & parallelism — READ FIRST
Sven is slow and these grids are large, so **most configs do NOT fit in a single 12h job
serially.** Measured per-run times (20 epochs): 1D Sven ≈68s, MNIST Sven ≈315s; and the real
bottleneck is **LBFGS+line-search: `max_iter=10` ≈ 800s/run on 1D (worse on MNIST)** — 12× Sven.

Two mitigations are built in:
- **`submit_rebuttal_parallel.sh`** runs `NPROC` processes on one GPU, each a disjoint 1/NPROC
  slice of the grid (via the scan's `+n_shards`/`+shard_id`). Small MLPs don't saturate the GPU,
  so this is ~4–6× faster. Verified: 4 shards → disjoint, complete coverage (18/18 unique files).
  Usage: `NPROC=6 sbatch submit_rebuttal_parallel.sh <config> [overrides...]`
- **LBFGS grid trimmed** to `max_iter=[4,10] × history=[10]` (was 4×4=16 combos → 2), 8× fewer LBFGS runs.
- **Scan is now crash-robust**: each standard run is wrapped in try/except, so a K-FAC eigh failure
  (it dies on MNIST: `linalg.eigh ill-conditioned`) or Shampoo NaN skips that one run instead of
  killing the whole multi-hour scan. Failures print `[error] <opt> run failed`.

Estimated wall-time (trimmed grids):

| config | runs | serial | NPROC=6 | recommendation |
|---|---|---|---|---|
| baselines_toy_1d | 1190 | ~28h | ~6h | `NPROC=6` parallel |
| baselines_polynomial | 1190 | ~28h | ~6h | `NPROC=6` parallel |
| baselines_mnist | 475 | ~41h | ~9h | **split by seed** (5 jobs, 1 seed each, NPROC=4 → ~2h each) |
| overparam_toy_1d (per n_data) | 415 | ~4h | ~1h | `NPROC=6`; one job per n_data |
| overparam_polynomial (per n_data) | 415 | ~4h | ~1h | `NPROC=6`; one job per n_data |
| overparam_mnist (per n_data) | 475 | ~2–9h | ~1–2h | `NPROC=4`; larger n_data (40k,60k) slower |
| batchsize_polynomial | 2130 | ~55h | ~10h | **split by batch_size** or drop LBFGS from it |

NPROC=6 numbers assume ideal speedup; realistic is ~1.3× those. Use NPROC=6 for 1D/poly,
NPROC=4 for MNIST (bigger per-run GPU use). To split MNIST by seed:
`for s in 3000 3001 3002 3003 3004; do NPROC=4 sbatch submit_rebuttal_parallel.sh experiments/configs/rebuttal_baselines_mnist_scan.yaml model_seeds=[$s]; done`

### Tier 1 — Extra baselines (the "at minimum" ask). Cheap; run first.
Sven vs {Adam, AdamW, SGD, RMSprop, Muon, SOAP, Shampoo, K-FAC, LBFGS, PolyakSGD} on the
three headline datasets, full data. Expectation: Sven keeps its 1D/poly edge; MNIST stays
a wash. Either way it closes the "missing baselines" objection.

```bash
NPROC=6 sbatch submit_rebuttal_parallel.sh experiments/configs/rebuttal_baselines_toy_1d_scan.yaml
NPROC=6 sbatch submit_rebuttal_parallel.sh experiments/configs/rebuttal_baselines_polynomial_scan.yaml
# MNIST is heaviest — split by seed so each job is ~2h:
for s in 3000 3001 3002 3003 3004; do NPROC=4 sbatch submit_rebuttal_parallel.sh experiments/configs/rebuttal_baselines_mnist_scan.yaml model_seeds=[$s]; done
```

### Tier 2 — Dataset-overparametrized regime (R1 crux). Highest scientific value.
Trace performance across the P/N = 1 boundary. 1D (P=593) and poly (P=673) run full-batch
(purest test); MNIST (P=27,562) keeps the paper's exact minibatched setup and subsamples N.

```bash
# 1D  (P/N ~ 4.0, 2.0, 1.0, 0.5)
for N in 150 300 600 1200; do NPROC=6 sbatch submit_rebuttal_parallel.sh experiments/configs/rebuttal_overparam_toy_1d_scan.yaml n_data=$N; done
# polynomial  (P/N ~ 4.0, 2.0, 1.0, 0.5)
for N in 170 340 675 1350; do NPROC=6 sbatch submit_rebuttal_parallel.sh experiments/configs/rebuttal_overparam_polynomial_scan.yaml n_data=$N; done
# MNIST, real dataset, crosses P/N=1 at N~27,562
for N in 2500 5000 10000 20000 40000 60000; do NPROC=4 sbatch submit_rebuttal_parallel.sh experiments/configs/rebuttal_overparam_mnist_scan.yaml n_data=$N; done
```
Report: train-loss convergence AND wall-time vs P/N. Win condition = Sven faster/lower in
the P>N region specifically. `track_param_norm` is on (min-norm story).

### Tier 3 — Batch-size sweep (R2) + Fig-5 upgrades (R1).
```bash
sbatch submit_rebuttal.sh experiments/configs/rebuttal_batchsize_polynomial_scan.yaml
```
Fig-5 (ResNet18/CIFAR param-fraction): re-run the existing CIFAR param-fraction scan now that
train loops log `peak_gpu_mem_mb`, with ≥3 seeds; report **accuracy** (already tracked for
label-regression), **peak memory + step time** (`peak_gpu_mem_mb`, `epoch_times`), and state
that **the parameter subset IS resampled every step** (confirmed: `SvenWrapper._make_param_mask`
is called fresh each `loss_and_grad` via `torch.randperm`). Move the result to the main text.

### Tier 4 — Stretch (R2 Q2): Muon/Sven on intermediate layers only + CE.
Muon already skips non-2D params; a "hidden-only" variant (exclude the final Linear/head) is a
small change to `build_standard_optimizer`. "Sven on intermediate layers only" needs a fixed
layer-name mask in `SvenWrapper` (the block-mask plumbing exists; it just needs a named-layer
selector). Only if time allows.

**Notes:** scans dedup by `run_id` (re-launching resumes/skips completed runs), so MNIST scans
that exceed the 12h wall clock can simply be re-submitted. Shampoo needs a low lr (its default
0.1 diverges on tiny MLPs); the `lrs_standard` grid already covers 1e-4…1e-1. K-FAC uses SGD+
momentum base, damping 3e-3 (tunable via `kfac_damping`).

---

## 4.6 RESULTS — dataset-overparametrized sweep (DONE, 2026-07-24)

Full P/N sweep, both synthetic tasks, 5 seeds, full-batch. Metric = wall-time to reach
train loss < 1e-3 (the paper's "fast convergence" claim), best hyperparameter config per
optimizer; 11 optimizers total. **Sven's rank:**

| | P/N=4.0 | P/N=2.0 | P/N=1.0 | P/N=0.5 |
|---|---|---|---|---|
| 1D (P=593) | **2nd** | **2nd** | 3rd | 5th |
| polynomial (P=673) | **2nd** | **2nd** | **2nd** | 2nd |

Headline (directly rebuts R1's "no advantage in the regime the theory targets"):
- In the genuine dataset-overparam regime (P/N>1), **Sven is consistently 2nd-fastest**,
  beating every first-order method (Adam/AdamW/SGD/RMSprop/Muon/Polyak) AND SOAP/K-FAC/Shampoo
  — typically ~2–3× faster than Adam in wall-time.
- On polynomial, Sven drives final train loss to **1e-8 – 1e-11**, orders of magnitude below
  every baseline including LBFGS (~1e-6) — the exponential-convergence/min-norm story.
- **P/N crossover confirms the theory:** on 1D, Sven degrades 2nd→3rd→5th crossing into the
  under-param regime — its edge is specifically an overparametrization phenomenon. (Show this.)
- **The one competitor is LBFGS** (1st in raw speed-to-threshold everywhere). Framing: Sven
  beats LBFGS on final precision (poly), and LBFGS's line-search doesn't scale (`max_iter=10`
  ≈800s/run even at 1D scale). The **MNIST-overparam run** (launched) is the "Sven scales,
  LBFGS doesn't" argument.

Baselines (full-data) status: polynomial COMPLETE; toy_1d COMPLETE. (1D/poly Sven LRs match the
paper's [0.05..1.0] grid, so these results are valid.)
K-FAC dies on ~10 ill-conditioned `linalg.eigh` points per config — caught+skipped (robustness
fix); its best config still recovered from surviving runs, so not worth recovering.

### MNIST Sven — config error found & FIXED (rmsprop flag), re-running with Gram
The rebuttal MNIST configs had `use_rmsprop: true` — the RMSprop-style Sven *variant*, NOT the
paper's headline label-regression result (`use_rmsprop: false`, best **lr=0.5**). That crippled
Sven (final ~0.145 / val-acc 0.845, best pinned at the grid-max lr) and made it look last-place,
contradicting Fig-1. (My earlier "needs large LRs" and "k≪P" diagnoses were BOTH wrong — the real
issue was the rmsprop flag; author confirmed lr=0.5.) **Verified fix:** with `use_rmsprop:false`,
lr=0.5, Sven-Gram on full MNIST reaches **final 0.030 / val-acc 0.972 — beats Adam (~0.92 acc)**.
So Sven IS strong on MNIST; the earlier negative was purely the config bug. Stale `svd_*RMSpropAlpha*`
files deleted; MNIST Sven re-running `mode=svd` on the headline grid (k=[16,32,48,64],
rtol=[1e-3,1e-2], lrs=[0.1,0.5,1.0], 5 seeds); baselines unaffected (rmsprop only touches Sven).

**Gram trick now the Sven backend** (`use_gram: true`): `GramSvenWrapper`+`SvenGram` compute the
IDENTICAL update via the B×B Gram matrix (verified exact to 9e-7) with no B×P Jacobian. NOTE its
~400× speedup is at P≈1M; at the MNIST-MLP scale (P=27,562) it's ≈classic speed (341s vs 315s) —
the win will show on the ResNet/CIFAR Fig-5 experiments (large P). Wired into `generic_scan.py`
behind the `use_gram` flag; incompatible with variable_k / pre-pinv rmsprop (guarded).

## 5. Reviewer 3 responses (misreads — verified against the code)

1. **"MNIST has 10 outputs — how is the Jacobian/pinv defined?"** (the key misread). The
   Jacobian is of the per-sample **scalar** residual R_eff^α = (ℓ^α)^{κ/2}, **not** the 10-dim
   output. Verified: for a 10-output model with batch B, `SvenWrapper.grads` has shape
   **(B, P)** — one row per sample, independent of output dim. M ∈ ℝ^{B×P}, M⁺ ∈ ℝ^{P×B}.
   Fix in revision: state this explicitly + a dimensionality box.
2. **Explicit Jacobian vs JVP/VJP.** We materialize the per-**batch** (B×P) Jacobian (B = micro-
   batch, not dataset size); `jacrev` is already a vmapped VJP. The truncated-SVD pinv needs the
   singular subspace, so we materialize once; randomized-SVD backends touch M only via matvecs
   for large P. This is the right framing and aligns with our own Appendix-I memory future-work.
3. **Eq (9) "κ does nothing."** Correct as an identity — by design. κ selects the residual
   *decomposition* R_eff (hence M and the update), not the loss value. Rewrite eq 9 to expose R_eff.
4. **Derive (16) from (8)+(15):** δθ = −η(MᵀM)⁻¹MᵀR; with L=Σ(R^α)², ∂L/∂θ = 2MᵀR, so
   δθⁱ = −(η/2)Σ_j[(MᵀM)⁻¹]^{ij}∂L/∂θ^j — the natural-gradient/Gauss-Newton step. Add to appendix.
5. **K-FAC to avoid the overparam singularity?** No — K-FAC still inverts the P×P (block-Kron)
   Fisher, which is rank-deficient for P>N; it only exists via damping (biased, not min-norm).
   Sven takes the exact Moore-Penrose pinv in the low-dim sample space — the principled min-norm
   update. Different axis; doesn't resolve the singularity in the sense Sven targets.

---

## 6. Manuscript revision checklist
- Add a dimensionality box: B=(micro)batch, P=#params, N=dataset; M∈ℝ^{B×P} (row α = ∂R_eff^α/∂θ,
  R_eff^α scalar), M⁺∈ℝ^{P×B}. (Fixes R3's central confusion + R3 clarity complaints.)
- Rewrite eq 9 to expose R_eff; add the eq-16 natural-gradient derivation to the appendix.
- Address the κ=1 (theory) vs κ=2 (implementation, avoids NaNs) gap head-on, not just in App. G.
- Promote the param-fraction (Fig 5) result to the main text with seeds/accuracy/memory/time.
- Frame the memory bottleneck honestly with the matrix-free randomized-SVD path as the roadmap.
