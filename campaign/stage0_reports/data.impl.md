# TRACK data — final report

## Files created / changed
| file | change |
|---|---|
| `/n/home/anon/sven-experiments/experiments/datasets/all_datasets.py` | rewritten: 6 classes + `AdditiveCubicDataset` + 8 pure helpers |
| `/n/home/anon/sven-experiments/experiments/datasets/__init__.py` | exports `AdditiveCubicDataset` and the helpers |
| `/n/home/anon/sven-experiments/experiments/data_prep/prepare_tokens.py` | one shared iterator, val→test→train, `os.truncate`, `token_counts.json`, `--test_tokens`, clean `os._exit(0)` |
| `/n/home/anon/sven-experiments/tools/check_token_split.py` | **new** |
| `/n/home/anon/sven-experiments/tests/test_datasets.py` | **new**, 19 tests |
| `/n/home/anon/sven-experiments/experiments/configs/dataset/*.yaml` | all 7 updated |

**Token rebuild DONE and verified.** `/n/labstore01/.../datasets/fineweb_edu_gpt2_v2/` = 250,000,000 train / 8,000,000 val / 8,000,000 test uint16 + `token_counts.json`. `check_token_split.py` → `DISJOINT`, exit 0 (15 s). Old `fineweb_edu_gpt2/` untouched (mtimes still Sep 8) and now reported `OVERLAP DETECTED`, exit 1 (32/32 sampled val blocks found in train — F4 reproduced then eliminated). v2 `val.bin` is byte-identical to v1 `val.bin` (same stream order); v2 `train.bin` no longer contains it.

## Acceptance tests — `19 passed in 18.60s` (observed, `pytest tests/test_datasets.py -q`)
- `test_known_monomials_evaluate_correctly` — `eval_polynomial` multiplies factors: `3x₀x₁−2x₀²+5`, `x₀²x₁³`, constant, `rtol=1e-15`.
- `test_monomial_enumeration_is_complete` — 210 unique tuples, equals brute-force `{sum(p)≤4}`, contains constant + all linear + `x₀⁴`; `C(v+d,d)` for 6 (d,v) pairs.
- `test_monomial_rms_matches_gaussian_moments` — `(2p−1)!!` = 1,1,3,15,105; Monte-Carlo unit second moment.
- `test_new_polynomial_is_not_additive_cubic` — 19-feature fit rel-RMSE **> 0.1** (measured 0.864 / 0.904 / 0.884 at seeds 0/1/2000); `num_terms == 210`.
- `test_additive_cubic_is_still_additive_cubic` — rel-RMSE **< 1e-6** (measured **5.6e-8 / 6.3e-8 / 6.5e-8**, matches F1's 7e-8); `num_terms == 185`.
- `test_additive_cubic_reproduces_legacy_targets_bit_for_bit` — inlined legacy body; `torch.equal` on x and y for train and val.
- `test_val_and_test_are_identical_across_n_train[toy|poly]` — `torch.equal` on val/test tensors at n_train 150 vs 1200; train draws nested.
- `test_holdout_split_is_disjoint_and_covering[60000-10000|50000-5000|11-3]`, `test_holdout_split_depends_only_on_split_seed` (deterministic, seed-dependent, **does not touch the global RNG**), `test_split_is_independent_of_subsample_seed` (held-out set fixed across subsample seeds 0/1/7, subsample never intersects val), `test_subsample_is_nested_and_deterministic`.
- `test_contiguous_split_bounds_are_disjoint_and_cover`, `test_shakespeare_style_contiguous_split_has_no_overlap` — synthetic 4000-char corpus, every block equals its corpus slice, three touched position sets pairwise disjoint.
- `test_token_bin_dataset_reads_three_bins`, `test_check_token_split_accepts_disjoint_and_rejects_overlap` (prefix case **and** mid-file case).

**Extra end-to-end check** on real files via `run_cpu_tests.sh` (log `slurm_logs/devtests/devtest-mOhfN2.out`): MNIST 50000/10000/10000, CIFAR 45000/5000/10000, shakespeare 6971/871/871 bounds `(0,892316)(892316,1003855)(1003855,1115394)`, fineweb 210000/200/200; train∩val = 0, subsample ⊂ train, val/test unchanged at n_train=1000.

## Deviations from CONTRACTS.md
1. `AdditiveCubicDataset` does **not** get C-D3 (one shared `default_rng`, y_train-based normalisation) — bit-for-bit legacy reproduction requires it. Its test split is drawn after x_val so train/val are unperturbed.
2. `CharTextDataset` / `TokenBinDataset` record `split_seed = None` (their split is positional/file-based, no randomness).
3. Synthetic classes take no `split_seed` argument; per the task text the three generator seeds derive from `seed`, and `split_seed` is recorded as `int(seed)`.
4. Fixed a latent bug I own: MNIST `digits` relabelling now maps from the original labels (old code double-remapped for e.g. `digits=[1,0]`). No live config uses `digits`.

## Notes for the integrator
**New public API** in `experiments/datasets/all_datasets.py` (all exported from `experiments.datasets`):
`derive_seeds(seed, *names) -> tuple[int,...]`; `holdout_split_indices(n, n_holdout, split_seed) -> (train_idx, holdout_idx)` (LongTensors); `subsample_indices(n_pool, n_train, subsample_seed) -> LongTensor` (nested prefix; `None`/≥pool → `arange`); `contiguous_split_bounds(n, val_fraction=0.1, test_fraction=0.1) -> ((s,e),(s,e),(s,e))`; `monomial_powers(degree, num_vars)`; `monomial_rms(powers)`; `eval_polynomial(x, coeffs, powers)`; `legacy_additive_powers(degree, num_vars)`; class `AdditiveCubicDataset(degree, num_vars, seed, n_train, n_val, n_test)`. New ctor kwargs: MNIST `n_val=10_000, split_seed=1234`; CIFAR `n_val=5_000, split_seed=1234`; CharText `test_fraction=0.1`; TokenBin `test_blocks=None`; Toy/Poly `n_test`, `pool_size=10_000`, `subsample_seed=0`. Every class sets `n_train`/`n_val`/`n_test`/`split_seed`; `CharTextDataset` also exposes `.stoi`/`.bounds`, `RandomPolynomialDataset` `.target_mean`/`.target_std`, `TokenBinDataset` `.token_counts`.

**Call sites I do not own that must change:**
- `experiments/experiment_code/generic_scan.py:255` `_scan_facts` — add `('n_test', 'test_dataset')`; record `dataset.split_seed`; its `ceil` docstring is wrong under C-S3.
- `generic_scan.py:542,647,723,795,870,923` — six `DataLoader(dataset.val_dataset, ...)`; need a `test_loader` from `dataset.test_dataset` and `eval_batch_size`.
- `exp_finetune_cifar_smallN.yaml` `result_id_fields` should gain `split_seed` (and `bn_mode`).
- **`rebuttal_overparam_mnist_scan` `n_data=60000` now silently yields 50,000** (train part is 50k): the launcher's top point is mislabelled — cap the sweep at 50000 (`submit_fresh_suite.sh:108`, `submit_reruns_2026-09-17_part2.sh:30`).
- To exceed 10,000 synthetic training examples, raise `pool_size` as well as `n_train`.
- `TokenBinDataset` now *requires* `test.bin`; anything still pointed at `fineweb_edu_gpt2/` (v1) will raise `FileNotFoundError`. `bench/bench_sharding.py` / `bench/cifar_chunk_probe.py` still work unchanged.
- Steps/epoch change: MNIST 938→782 @B=64, CIFAR 391→352 @B=128, shakespeare 122→108 @B=64.
- New polynomial target stats (variance-normalised, after standardisation): `max|y|` ≈ 10–13σ, `E[y⁴]` ≈ 16–19 (additive cubic: 7.5σ, 6.2) — heavy-tailed but far from the "tens of σ" the scout warned about.
- Two of my `devtest` SLURM jobs (47042032, 47043397) ran a pre-optimisation version of the checker and are self-expiring at their 40-min limit; I did not cancel them (ground rules). The committed checker takes 7–15 s on 250M tokens.
- `EXPERIMENTS.md` (not mine) should state that the 0.1307/0.3081 and CIFAR channel constants are full-official-train statistics, deliberately not recomputed for the 50k/45k splits.