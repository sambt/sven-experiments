# Datasets, models, baselines — scout report

Everything below is **read** from the files cited unless marked **[RAN]**.

## 1. Dataset classes (`experiments/datasets/all_datasets.py`)

| class | ctor args | splits today | RNG | normalisation | test? |
|---|---|---|---|---|---|
| `Toy1DRegressionDataset` :9 | `n_train=10k, n_val=10k, n_test=10k, seed` | all three sampled sequentially from **one** generator (:16, :25-27) | `torch.Generator().manual_seed(seed)` | `y` by **y_train** mean/std (:29) | **yes**, built, never evaluated (F5) |
| `MNISTDataset` :38 | `ROOT, digits, n_train, subsample_seed` | official train → train; **official test → `val_dataset`** (:48) | `Generator().manual_seed(subsample_seed)` for subsample only | `Normalize((0.1307,),(0.3081,))`, flatten 784 | no |
| `RandomPolynomialDataset` :93 | `degree, num_vars, seed, n_train, n_val` | coeffs → x_train → x_val from the **same** `default_rng` (:95,107-108) | numpy `default_rng(seed)` | `y` by y_train mean/std (:121) | no |
| `CIFAR10Dataset` :132 | `for_mlp, ROOT, n_train, subsample_seed` | official train → train; **official test → val** (:143) | as MNIST | CIFAR channel mean/std, **no augmentation** | no |
| `CharTextDataset` :171 | `ROOT, block_size=128, val_fraction=0.1, n_train, subsample_seed` | contiguous by position, val = **last 10 %** (:195-196) | subsample only | none (char ids) | no |
| `TokenBinDataset` :236 | `ROOT, block_size=1024, n_train_blocks, val_blocks=200, vocab_size` | reads `train.bin` / `val.bin` | none (deterministic blocks) | none | no |

**Token files** `/n/labstore01/LABS/anon_lab/Users/anon/datasets/fineweb_edu_gpt2/`: `train.bin` 500,000,000 B = 250 M uint16 tokens, `val.bin` 16,000,000 B = 8 M tokens (both Sep 8 15:16–15:20).
**[RAN] C-D2 check: `np.array_equal(t[:len(v)], v)` → `True`; 8,000,000/8,000,000 tokens identical.** F4 confirmed. **No** trailing zeros in either file (both budgets exactly filled), so the "unfilled tail" half of F4 does not apply here. Because `val_blocks=200`, only the first 204,800 tokens of `val.bin` are used — these are literally train blocks 0–199, i.e. the validation set *is* the first 200 training batches. `datasets/tokenize.log` also shows a `[Errno 9] Bad file descriptor` retry and a fatal interpreter error after both writes flushed (files are complete, the tool is not crash-clean).

## 2. C-E1 / C-D1 / C-D2 / C-D3 insertion points

Loader sites, one per grid loop: `generic_scan.py:537-542, 643-647, 719-723, 791-795, 866-870, 919-923` — all six do `DataLoader(dataset.val_dataset, batch_size=batch_size, shuffle=False)`: **eval batch = train batch** (F33), no test loader, `drop_last` only for microbatch (:540). `_scan_facts` :251-261 records `n_train`/`n_val` only (add `n_test`; its docstring says `ceil`, which becomes wrong once C-S3 forces `drop_last`). Pre-training val: `experiment_utils.py:182-200` runs **before** `model.train()` (:210) and before the first `.eval()` (:249) → train mode, exactly the C-E2 bullet; svd equivalent :303-321; `_initial_val` for jd/hig. Batch-mean epoch averaging: `np.mean(v)` at :276, :463, :519 (F10).

Sizes: **C-D1 = S** (~25 lines: `combinations_with_replacement` is already imported and unused; `*=` instead of `+=`); **C-E1 = M** (six classes + six loader sites + `_scan_facts` + 44 configs); **C-D3 = M**; **C-D2 = S** (hold `it = iter(ds)` outside `write_split` — the bug is that `for ex in ds` on a streaming `IterableDataset` restarts at document 0 — plus `os.truncate(path, n*2)`).

**[RAN] C-D1 arithmetic.** Today's enumeration yields **185** power tuples with degrees only {2,3,4} — `range(d)` excludes any exponent ≥ d, so the constant *and all linear* monomials are absent. Combined with `term += x[:,j]**p[j]` the function collapses to `c0 + Σ_j (a1 x_j + a2 x_j² + a3 x_j³)` = **19 effective features** (matches F1's 7e-8 fit). True count `C(10,4) = 210`.

**Steps/epoch changes [RAN]:** MNIST 60k→50k @B=64: 938→**782** (−16.7 %); CIFAR 50k→45k @B=128: 391→**352** (−10 %); Shakespeare (1,115,394 chars, vocab 65) blocks 7842/871 → **6971/871/871**, steps/epoch @B=64 122→**108**. Legacy curves are therefore not epoch-comparable; MNIST/CIFAR scans also get 10–17 % cheaper.

**Re-tokenising:** ~**5–6 min** wall for 266 M tokens, inferred from the existing run's mtimes (8 M at 15:16, 250 M at 15:20). It **needs HuggingFace network access**: **[RAN]** `curl https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu` → **HTTP 200** from this compute node, and `/n/labstore01/.../hf_cache/datasets/` holds only `burkelibbey___colors` (streaming never caches), so ~1.1 GB is re-downloaded. Budget 30 min with one retry. Note train needs ≥ 215,040,000 tokens for `n_train_blocks: 210000`, so write 250 M / 8 M / 8 M.

## 3. Baselines

**Muon today** (`experiment_utils.py:621-635`): `muon_params = [p for p in model.parameters() if p.ndim == 2]` — every 2-D tensor including `tok_emb`, `pos_emb`, `lm_head`; everything else (all 4-D convs, all 1-D) → `AdamW`, wrapped in `_CombinedOptimizer` (:565) with **one shared `lr`**. `adjust_lr_fn` is never passed. F13 confirmed; on `resnet18_functional` only `fc.weight` reaches Muon.

**torch 2.9.1+cu128**, read from `site-packages/torch/optim/_muon.py` (not imported): `adjust_lr_fn` **is supported** — validated against `["original","match_rms_adamw"]` at :110-115, `_adjust_lr` :74-87 gives `0.2*sqrt(max(A,B))` for `match_rms_adamw`. Defaults `lr=1e-3, weight_decay=0.1, momentum=0.95, nesterov=True`. Critically, `__init__` **raises `ValueError` for any `p.ndim != 2`** (:132-133), so conv kernels cannot be handed in at all — C-B5's flattening variant must be vendored (view grad as `(out, in*kh*kw)`), or Muon is dropped from CIFAR (O6).

**`schedulefree` is NOT installed**: absent from `site-packages`, absent from `pyproject.toml` deps, **0 hits in `uv.lock`**. Removal (C-B6) is free: `_CUSTOM_OPTIMIZERS` (:526-531) holds the two local classes but no live headline config lists them (`toy_1d_scan.yaml:25` is commented out).

**C-B2 `SGDm`**: `build_standard_optimizer` falls through to `getattr(torch.optim, name)` (:666), so `SGDm` needs an explicit branch → `torch.optim.SGD(momentum=0.9)`, plus the name in `submit_fresh_suite.sh:49` `FIRST`, `style.METHOD_COLORS`, and any family routing. **S**.

**C-B4 AdamW lr×wd**: machinery already exists — `weight_decays` grid (`experiment_utils.py:122`), `resolve_weight_decay` (:601), and the standard loop already puts `_wd{v}` in the `run_id` and on the record (`generic_scan.py:609-623, 665`), gated to `AdamW|Muon|MuonW` at :616. C-B4 is a **config-only** change: `weight_decays: [0, 0.01, 0.1]`. **S**.

**C-B7** is a label change in `analysis/style.py`. **C-B1** is one line.

## 4. Spec problems and risks

1. **Polynomial coefficient scale is unspecified and matters.** With `x ~ N(0,1)^6` and `c_i ~ N(0,1)` over 210 genuine monomials, degree-4 terms (`x_i⁴`, `x_i²x_j²`) dominate the variance and the standardised target is strongly heavy-tailed — `max|y|` of order tens of σ in 10 k draws. That changes what MSE means and interacts with the label-regression `sign(r)|r|^κ` path. Options: scale `c_i` by degree or by monomial multiplicity, or draw `x ~ U[-1,1]^6`. **Not covered by CHANGES_NEEDED.md — needs a user decision before the phase-2 pilot**, since it defines the new benchmark.
2. **Normalisation constants** (0.1307/0.3081, CIFAR triples) are full-official-train statistics and become slightly stale under 50k/45k. Harmless and no leakage, but must be stated in `EXPERIMENTS.md`; recomputing them would break every legacy comparison for nothing.
3. **C-D3's pool normalisation** uses statistics of 10,000 examples the model never sees at `n_train=150` — intended, but it is a mild information flow; the analytic generator moments would be cleaner.
4. **LM epochs are genuine, not nanoGPT-style.** Both `CharTextDataset` and `_BlockDataset` build **fixed non-overlapping** blocks and the `DataLoader(shuffle=True)` permutes them, so there is no random-with-replacement sampling and `drop_last` has its ordinary meaning — C-S2/C-S3 apply unchanged. But `_BlockDataset.n = min(n_blocks, max_blocks)` takes a **prefix**, not a random subset. `exp_gpt2_small_comparison.yaml`: `num_epochs: 1`, B=16, 210 k blocks → 13,125 steps with **two** val points (F34) → C-E4 mandatory.
5. **`HIGWrapper.evaluate`** (`hig.py:88-91`) calls `functional_call` with the module's current `training` flag and the wrapper never calls `.eval()`/`.train()` — it evaluates in whatever mode it was left in. Only MLP scans use HIG today; must be fixed before HIG touches any BN model.
6. **`_frozen_norm_stats`** (`sven/sven/nn/gram_wrapper.py:357-370`) restores with an unconditional `mod.train()`, so calling it while the model is legitimately in eval mode leaves it in **train** mode. C-E2 must save/restore per-module flags (and `track_running_stats`), not call `.train()`.
7. **C-B5 will move the Muon baseline.** On nanoGPT/GPT-2 today Muon orthogonalises a 50304×768 embedding and a 768×50304 head; moving them to AdamW plus `match_rms_adamw` will most likely make Muon *better* there. This is the change most likely to weaken a headline claim — flag it to the user now.
8. **CIFAR = torchvision `resnet18`** (`nets.py:115-125`): 7×7/stride-2 stem + maxpool on 32×32, no augmentation (F24). `SmallResNet` (`nets.py:231`) is used by **no** config and is the right vehicle for the C-E2 CPU test and for tier 2.
9. **`exp_finetune_cifar_smallN.yaml`** sets `dataset.subsample_seed: ${data_seed}`, so `split_seed` must be a **third** key; `result_id_fields: [n_data]` needs `bn_mode` (C-E2) and `split_seed`. Under C-E1 its val becomes 5 k held-out train images and test the official 10 k. Also note `k_fractions` (GPT-2) vs `k_values` (MNIST/CIFAR) — `expand_grid` (C-R2) must keep both.
10. `experiment_results/exp_gpt2_small_comparison/` is **empty** (mtime 09-18 08:33) — the in-flight GPT-2 jobs have written no records yet, so cancelling costs nothing. `_backup_2026-08-31`, `_backup_2026-09-11` and `_cache` will ride along in the 4.1 rename.

## 5. Work packages and dependencies

| WP | content | size | depends on |
|---|---|---|---|
| **WP-T** token rebuild: one-iterator + truncate + `tools/check_token_split.py`, re-tokenise 250/8/8 M | S | nothing — **do first, today**; gates GPT-2 (phase 4) |
| **WP-P** true polynomial + `AdditiveCubicDataset` + coefficient-scale decision | S | user answer on risk 1 |
| **WP-S3** three splits + `split_seed` on all six classes, `test_dataset` everywhere | M | nothing |
| **WP-FP** fixed pool/val/test from separate generators (C-D3) | M | WP-S3 |
| **WP-EV** one `evaluate(...)`, example-weighted, eval-mode, `eval_batch_size` + its CPU test | M | helper itself independent; **wiring** waits on C-R2 (six loader sites collapse to one) |
| **WP-BL** SGDm, `MuonW` in `FIRST`, Muon hidden-2D grouping + `match_rms_adamw` + vendored flattening Muon, AdamW wd grid (config only), drop ScheduleFree*, LBFGS relabel | M (Muon vendor is the only real work) | C-R3 for the `muon_variant` record field |
| **WP-CFG** 44 configs: `eval_batch_size`, `split_seed`, `checkpoints`, `svd_spectra_schedule`, `bn_mode`, `weight_decays` | M, mechanical | last, after C-R3 so the run hash covers them |

WP-T, WP-P, WP-S3, WP-FP and the `evaluate` helper + CPU tests are all **independent of the C-R2 runner refactor** and should be built in parallel with it; only the loader wiring, the record fields and the configs wait on it.