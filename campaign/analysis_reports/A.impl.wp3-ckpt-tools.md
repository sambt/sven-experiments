# WP3 tool half — final report

## Files
**New:** `/n/home/anon/sven-experiments/analysis/ckpt_tools.py` (960 l), `/n/home/anon/sven-experiments/tools/compute_ckpt_spectra.py` (260 l), `/n/home/anon/sven-experiments/tests/test_ckpt_tools.py` (520 l, 28 tests, 4.8 s).
**Changed (authorised by the task):** `analysis/.gitignore` +3 lines (`ckpt_spectra/`). Nothing else touched; results read-only (verified: no file under `experiment_results/*_diag` modified).
**Cache (git-ignored, 54 MB, 64 npz):** `analysis/ckpt_spectra/{toy_1d,polynomial}_scan_diag/*_probefull.npz` (20 each = Sven/Adam/MuonW/HIG × 5 seeds, 34 ckpts, full 10 000-row pool), `mnist_scan_labelRegression_diag/*_probe512.npz` (20, 35 ckpts) and `*_probe2000_epochs.npz` (4, 21 ckpts, seed 3000).

**API:** `load_run` → `Run` (record + the matching `{scan}/configs/*.yaml`, picked by loss/`result_id_fields`/seed then run_id-prefix→`mode`; disagreeing `dataset`/`model` subtrees **raise**), `state_at/model_at(step|epoch)`, `verify_checkpoint(run) -> rel_error` (`detail=True` adds `abs_error`, `loss_scale`; token-weighted for `lm_ce`), `verifiable_epochs`, `probe_indices/probe_set` (= `subsample_indices(n_train, n, split_seed)`, the runner's own C-E3 draw), `batch_of_step` (checks `derive_loader_seed` against the recorded `effective_loader_seed`), `RowSpec` (+`row_spec_for_scan`), `jacobian_rows`/`spectrum` (float64, `jacrev` row-chunks, `MAX_JAC_BYTES` guard), `checkpoint_spectra`, `param_vector`/`distance_from_init`, `spectra_path`/`load_spectra`.

## Verified by running
* **`verify_checkpoint` on real runs — 7 diag passes × 4-5 methods** (final epoch, float32, CPU). abs error ≤ 2.2e-8 everywhere: toy Sven `1.64295e-07 → 1.64302e-07` (rel 3.9e-05, **abs 6.4e-12**), Adam rel 1.0e-05, HIG rel 7.4e-03 (abs 4.4e-11 on a 6e-9 loss — fp32 floor, hence `abs_error`); polynomial Sven 7.4e-08 / Adam 3.8e-08 / MuonW 5.2e-09 / SOAP 4.8e-08; mnist-labelreg Sven 5.3e-08 / Adam 2.0e-08 / HIG 6.3e-09; mnist-CE Sven 4.7e-08 / SOAP 7.7e-07; **CIFAR-labelreg (BatchNorm buffers) Sven 2.4e-05, Adam 4.1e-05, MuonW 1.6e-05**; CIFAR-CE Sven 1.3e-05; nanoGPT Sven 2.2e-08. **All 20 epochs × 5 seeds × 4 methods** on toy+polynomial (800 checks): rel max 2.7e-04, abs max 5.2e-08.
* **Cross-check vs the spectra `SvenGram` logged online** (reconstructed batch + reloaded checkpoint + `RowSpec`): max rel diff over SVs above 1e-6 σ_max = **1.0e-08…3.3e-07** (steps 0-781, all three scans); at late toy steps it grows to 2e-4, which is the float32 Gram losing precision, not the tool (F19).
* **Adversarial recompute of cached npz** with one `autograd.grad` per row: toy full pool 10 000×593 **5.2e-14**, polynomial 10 000×673 1.1e-15, MNIST 512×27 562 1.4e-15; `|utr|` agrees to 1e-15 on resolved directions, `‖utr‖ = ‖r‖` exactly.
* **Batch reconstruction:** 28 100 steps (3 scans × all epochs) vs `EpochPermutationSampler` — **0 mismatches**.
* **Probe set:** bit-identical across all 20 runs per scan and nested (512 ⊂ 2000); `row_spec` identical for Sven and every baseline; step 0 σ_max/probe-loss identical across methods (MNIST `6.10906338e+01` / `1.174937`).
* `tests/` green on a CPU job: **1001 passed, 33 skipped** (116 s).

## Spectra (seed means, float64)
| scan (probe) | σ_max init → final | σ_B/σ_1 init → final | rank>1e-12 σ_max | ‖θ_T−θ_0‖ |
|---|---|---|---|---|
| toy full 10 000, P=593, B=k=32 | 219.5 → Sven 1.83 / Adam 3.60 / MuonW 15.8 / HIG 0.16 | 8.6e-17 → 1.3e-08 (Sven) / 1.3e-07 (Adam) | 14 → 45/48 | 7.82 / 7.02 / 6.26 / 8.52 |
| polynomial full 10 000, P=673, k=16 | 222.7 → Sven 748 / Adam 491 | 9.3e-03 → 0.121 / 0.162 | 673 (full) | 7.52 / 14.54 |
| MNIST-labelreg 512, P=27 562, B=k=64 | 30.5 → Sven 36.2 / Adam 32.5 | 7.5e-02 → 3.4e-02 / 3.9e-02 | 512 (full) | 12.48 / 20.38 (MuonW 17.6, HIG 12.5) |

Mechanism findings for phase B: on the **full pool** only 8 (init) → 27-31 (final) of 593 toy singular values sit above the 1e-7 float32 floor, and **100.0000 %** of the probe residual lies in the top-32 left singular directions, so toy's k=32 discards nothing (online `num_nonzero_svs` mean 14.5/32 — `rtol` binds, not k). MNIST keeps 99.93 % → 92.9 % in the top 64 with 64/64 directions used online. **Polynomial is the exception: only 74 % of the residual is inside Sven's k=16 (89 % inside k=B=32), and rtol=0.03 keeps 12→105 of 673** — the one scan where truncation drops real signal. Sven and HIG reach their loss with ~40 % less parameter movement than Adam (Codex Low 4).

## Deviations / notes
* `resolve_results_root` is repo-root-relative (`style`'s `'../experiment_results'` only works from `analysis/`); precedence unchanged.
* toy/polynomial probe (a) and full pool (b) coincide — the whole 10 000-example pool *is* the fixed probe set there.
* `--epochs-only` also keeps step 0; the first two 2000-row MNIST files were computed before that fix and were **deleted and regenerated** (all four now have 21 ckpts from step 0).
* Two `polynomial_scan_diag` L-BFGS runs are `diverged` with partial curves; `verify_checkpoint` now refuses them with a diagnosis instead of an `IndexError` (spectra still computable by step). Tested.
* Full-pool MNIST (50 000×27 562 = 10.3 GB fp64) is refused by `MAX_JAC_BYTES=3 GB`; feasible on a big-memory node if wanted.
* Cost on 4 cores: toy/polynomial 28 s per run, MNIST-512 45 s, MNIST-2000 11 min.
* **Phase B / other WPs:** read the cache with `ckpt_tools.load_spectra(scan, method=, model_seed=, n_probe=)`; hyperparameters (k, rtol, lr, B) come from `load_run(scan, str(entry['run_id'])).record`, deliberately not duplicated in the npz. No name clashes with `sv_diagnostics` (which covers only the online per-step spectra). Scratch scripts kept at `/n/home/anon/sv3_wp3_scratch/`.