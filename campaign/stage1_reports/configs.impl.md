## Configs track — final report

### Files changed (all absolute paths under `/n/home/anon/sven-experiments/`)

**`experiments/configs/` (22 edited, 2 created)**
- **Headline MLP** `toy_1d_scan`, `polynomial_scan`, `mnist_scan_{ce,labelRegression}`: `checkpoints: log` (toy/poly) / `checkpoints_svd: log` (MNIST); `+SGDm`; shared `lrs_standard: [1e-5,3e-5,1e-4,1e-3,1e-2,1e-1,3e-1,1.0]`; `lrs_jd: [1e-5…1e-1]`; `lrs_hig: [0.005,0.015,0.05,0.1,0.25]` (≥0.5 removed — always crashes); `tau_hig` +`3e-2,1e-1`; polynomial `rtol` +`1e-5`.
- **CIFAR headline** `cifar10_resnet_{ce_scan,scan_labelRegression}`: `bn_mode: batch` (alias `gram_freeze_norm_stats: false` removed — the integrator landed `resolve_bn_mode` mid-session, `_bnbatch` run_ids unchanged); `optimizers_standard: [Adam, AdamW, SGD, SGDm, RMSprop, Muon, MuonW, SOAP, LBFGS, PolyakSGD]`; `lrs_standard: [1e-5…3e-1]`; CIFAR-CE Sven `lrs: [0.02,0.05,0.1,0.5,1.0]`. No `checkpoints` key (default `final` is right).
- `exp_nanogpt_speedrun`: `eval_batch_size: 256`, `checkpoints: epochs`, `lrs_standard: [1e-5,3e-5,1e-4,3e-4,1e-3,3e-3]`.
- `rebuttal_overparam_{toy_1d,polynomial,mnist}`, `rebuttal_batchsize_polynomial_scan`: `weight_decays` removed, `+SGDm`, checkpoints. Batch-size scan: `lbfgs_max_iter: [3]`, `lbfgs_history_size: [2]` (O5, 810→90). MNIST overparam header documents `n_data <= 50000` and the sweep `2500 5000 10000 20000 40000 50000`.
- `rebuttal_fig5_cifar_paramfrac_scan`: `bn_mode: batch`; set point flagged **TENTATIVE, re-point before launch**.
- `mnist_kappaScan_labelRegression` (C-X1): `k_values [32,64]` × `lrs [0.1,0.25,0.5,0.75,1.0]` × `kappa [1,2,3]` × 5 seeds = **150**.
- `exp_finetune_cifar_smallN`: `bn_mode: frozen`, `+SGDm`, `lrs_standard` down to 1e-5.
- 8 micro-batch/param-fraction scans: checkpoint keys only.
- **New**: `cifar10_resnet_ce_scan_timing.yaml`, `cifar10_resnet_scan_labelRegression_timing.yaml`.

**`tests/test_configs.py`** (new, 239 tests) · **`campaign/grid_counts.md`** (new) · **`tests/golden/{toy_1d_scan,rebuttal_batchsize_polynomial_scan}.order.txt`** re-frozen (1080 / 1680 ids) · **`tests/golden/legacy_grid_equiv.py`** strips the two post-legacy `bn_mode` tokens before comparing (order + shard membership still fully compared; refuses to `--freeze` a scan carrying one).

**`tests/test_grid.py`** (integrator's — 5 minimal edits, detailed in the message I sent to `main`): RE-FREEZING docstring; INVENTORY rows for 6 scans; `EXPLAINED` rewritten for `rebuttal_batchsize` + new entries for `mnist_scan_ce` (40 HIG crashers) and `exp_finetune` (192 ids gain `_bnfrozen`); kappa counts 5→50 / 10→100; `test_weight_decay_grid_filter` now uses an explicit override and asserts no config sweeps wd.

### Verified by running
- `.venv/bin/python -m pytest tests/test_configs.py tests/test_grid.py -q` → **`345 passed, 10 skipped in 17.86s`**
- `campaign/run_cpu_tests.sh .venv/bin/python -m pytest tests/ -q` → **`599 passed, 29 skipped, 25 warnings in 88.38s`**, `exit=0`
- `campaign/run_cpu_tests.sh .venv/bin/python tests/golden/legacy_grid_equiv.py --freeze` → `OK toy_1d_scan mode=all legacy=1080 new=1080 order_identical=True`, `OK rebuttal_batchsize_polynomial_scan mode=both legacy=1680 new=1680`, both froze; re-verify run → `ALL EQUIVALENT`, `exit=0`, with `(90 ids carry a post-legacy bn_mode token)` on CIFAR and fine-tune.

### Counts
**15953 runs / 22 in-scope scans** (was 15310). ~206 GPU-h *floor*; plan against the scout's 1,100–1,500. CIFAR-CE 570 runs/65 GPU-h and CIFAR-label-reg 510/43 are a third of the cost; everything else is noise.

### Deviations from CONTRACTS.md
1. **CIFAR-label-reg Sven lr not extended** though C-B3 lists "largest lr on CIFAR label-reg": lr=1 is the full Gauss-Newton step, two more points cost 60 runs ≈ 30 GPU-h, and the brief said keep CIFAR Sven minimal. Flagged in the config and in `grid_counts.md`.
2. **tau_hig extended** and **HIG lr grid shifted** beyond the literal known-case list — justified from `best_configs.json` (tau on the top edge in 3 of 4 MLP scans; HIG lr ≥ 0.5 crashes).
3. **Extensions applied to headline scans only**; `rebuttal_overparam_*` / `rebuttal_batchsize` keep `lrs_standard: [1e-4…1e-1]`.
4. `exp_finetune_cifar_smallN` `result_id_fields` did *not* gain `split_seed`/`bn_mode` (data.impl.md asked) — it would orphan all 240 on-disk ids without an oracle. Orchestrator call.

### Open questions
1. Extend CIFAR-label-reg Sven lr above 1.0? (+60 runs, ~30 GPU-h)
2. `rebuttal_fig5_cifar_paramfrac_scan` set point (k=64/lr=1.0/rtol=1e-3 vs EXPERIMENTS.md's k=128/lr=0.1/rtol=1e-4) — nothing enforces re-pointing; no CIFAR entry exists in `best_configs.json`.
3. Extend the P1 rebuttal lr grids too? ≈+1,900 runs but <2 GPU-h.
4. nanoGPT `eval_batch_size: 256` was chosen by reasoning (2048 blocks × 128 tokens), not measured.

### Notes for other tracks
- **launcher**: read `campaign/grid_counts.md`, not `grid_inventory.md`. `mnist_scan_ce` must submit all 12 baselines **and** `mode=jd`/`mode=hig` (the old `CORE_FIRST` dropped 6 optimizers; no launcher ever ran jd/hig → 400 runs). MNIST overparam sweep is capped at 50000. The 7 `*_timing` configs describe the full parent grid — the best-config selection is yours. CIFAR needs NPROC=1 for Sven, 4 for baselines, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **integrator**: `exp_finetune`'s 192 baseline run_ids now carry `_bnfrozen`, so its results dir is superseded, not resumable. Storage ≈100 GB, 67 GB of it ResNet `final`.