**WP3 figure half — final report**

## Files
**New:** `/n/home/anon/sven-experiments/analysis/spectra_figs.py` (927 l), `/n/home/anon/sven-experiments/analysis/spectra_analysis.ipynb` (42 cells / 25 code), `/n/home/anon/sven-experiments/tests/test_spectra_figs.py` (456 l, 31 tests).
**Changed:** `/n/home/anon/sven-experiments/analysis/comparisons.ipynb` (regenerated, 22 cells / 11 code; old version at `/tmp/claude-66176/.../scratchpad/wp3/comparisons.ipynb.bak`), `/n/home/anon/sven-experiments/analysis/sv_diagnostics.py` (+78/−10).
**Cache extended (git-ignored):** `analysis/ckpt_spectra/mnist_scan_ce_diag/` — 20 new npz (Sven/Adam/MuonW/HIG × 5 seeds, 35 ckpts, 512-row probe), one 40-min CPU job, verify_rel_error ≤ 2e-8. Results roots untouched (only `_cache/`).

## Shared-helper fix (the one I made; must be listed)
`sv_diagnostics.plot_epoch_spectra` drew `style.FLOAT32_NOISE_FLOOR = 1e-7` labelled "float32 floor". **Wrong by 3.5 decades for the Gram backend**: `sven.opt.sven.SvenGram` records `sv_noise_floor = sqrt(eps_fp32)·σ_max`, measured constant at **3.453e-4·σ_max** on all 7 diag passes (matches phase A: online-vs-offline agree to 5e-7 *above* 1e-2·σ_max only). Added `noise_floor_rel(diag)` / `noise_floor(rows)` (returns `None` for legacy records) and the line now reads the record, falling back to the old 1e-7 line only when nothing is recorded. `spectrum_floor`'s clip is deliberately **unchanged** (frozen `scan_analysis.plot_sv_spectra` calls it separately; changing it would leave 3.5 decades of empty axis in WP2/WP4 notebooks). No other frozen helper touched.

## Notebooks
| notebook | code cells | error cells | runtime |
|---|---|---|---|
| `spectra_analysis.ipynb` (new) | 25 | **0** | 89 s |
| `comparisons.ipynb` | 11 | **0** | 39 s |

`tests/` green on a CPU job: **1155 passed, 33 skipped** (122 s).

## Key numbers the figures show
**ONLINE (per-batch Gram, all 7 headline scans, 5 seeds each; `min(k,rtol-rank)` == the optimizer's own `num_nonzero_svs` on every logged step of every seed — asserted in-notebook):**

| scan | B/k/rtol | SVs used first→final | binds | batch residual kept first→final |
|---|---|---|---|---|
| Toy 1D | 32/32/1e-4 | 4.2 → 14.4 | rtol | 92.08 % → 91.53 % |
| Polynomial | 32/16/0.03 | 7.0 → 13.4 | rtol | 92.25 % → 99.59 % |
| MNIST label-reg | 64/64/1e-3 | 64 → 63.6 | neither (k=B) | 100 % → 100 % |
| MNIST CE | 64/32/0.3 | 10.0 → **2.8** | rtol | **31.0 %** → 95.2 % |
| CIFAR label-reg | 128/128/1e-3 | 128 → 81.6 | rtol | 100 % → 99.999 % |
| CIFAR CE | 128/128/0.01 | 128 → 117 | rtol | 100 % → 99.999 % |
| nanoGPT | 64/64/1e-3 | 64 → 64 | neither (k=B) | 100 % → 100 % |

**Headline mechanism result: on 5 of 7 scans `rtol`, not `k`, is the binding cut**, and the discarded *batch* residual spans five decades (1e-8 for MNIST-labelreg/nanoGPT, 4e-3 polynomial, 8.5e-2 toy, 7e-2 MNIST-CE). MNIST-CE inverts ~3 of 64 directions at the end and still reaches val 0.11. CIFAR-CE's failure is **not** a truncation failure (~100 % kept, 117/128 used). Cross-scan rtol-rank at k=B (light records, whole grid): MNIST-CE 0.14→0.05 of B over rtol 1e-4→0.3; CIFAR-CE 0.97→0.56; MNIST-labelreg 1.00→0.21. Norms: applied `‖Δθ‖` shrinks on every scan (ratio 8e-5 toy … 0.50 CIFAR-labelreg) while `‖r‖` shrinks by very different factors. Seed spread of final used-rank: 0.0 % (nanoGPT) to 69 % (MNIST-CE), CIFAR-labelreg 24 %.

**PROBE SET (offline float64, fixed rows, identical across optimizers — step-0 spread between methods 0.0e+00):** the same `k` answers differently. Final `top_k` of the *probe* residual at the selected k: toy 100.000 %, **polynomial 58.82 % (k=16; 73.71 % at k=32, 89.59 % at k=105 — 41 % of the residual is outside k=16)**, MNIST-labelreg 99.868 %, MNIST-CE 96.09 %. `in_span = ‖P_U r‖²/‖r‖²` = 0.9949 (polynomial), 1.0 elsewhere. Effective rank `exp(H(σ))` init→final: toy 1.2→Sven 4.54/Adam 5.52/MuonW 4.69/HIG 5.30; polynomial 21.4→124.6/172.6/134.8/128.2; MNIST-labelreg 286.5→75.4/102.4/114.6/60.3; MNIST-CE 286.8→61.6/59.0/51.9/47.9 — Sven moves the Jacobian geometry *earliest* in both directions. σ_B/σ_1 falls on MNIST (0.075→0.034 Sven) and rises on polynomial (0.0093→0.121).

**Codex Low 4, computed not asserted:** Sven has the smallest `‖θ_T−θ_0‖` on **2 of 4** scans (polynomial 7.52, MNIST-labelreg 12.48) and ranks 3rd of 4 on toy (7.82 vs MuonW 6.26) and MNIST-CE (9.47 vs HIG 7.69); smallest `‖θ_T‖` on **1 of 4**. The notebook prints "state it per scan, never as *Sven stays closer to init*".

## Deviations / open issues
* No `savefig` PDF+PNG helper existed (`Scan.save` is PDF-only and needs a `Scan`); added `spectra_figs.savefig` and used it in both notebooks. 36 files under `plots_v2/spectra_analysis/`, 16 under `plots_v2/comparisons/`; I deleted 4 stale PDFs from the old comparisons (`sv_rank_comparison_lr0.05`, `sv_used_comparison_<scan>`×3).
* Probe spectra for CIFAR (both) and nanoGPT are **not** computed — cost-bound (full-pool MNIST is already refused by `MAX_JAC_BYTES`); the notebook skips them cleanly and says so.
* CIFAR-CE is live-provisional: `freshness_report()` fires (763→765 records, 23 post-selection, 5 live claims) and the notebook prints a PROVISIONAL banner. The diag pass still matches today's pick (k=128, lr=0.1, rtol=0.01); if `select_best.py` changes it, `SvenDiag.selection_note` prints the mismatch instead of mislabelling the figure — no CIFAR-CE number is hard-coded. Comparisons' rank table shows `n_seeds=3` for the new rtol rows.
* `used_rank_grid` shows toy rtol=1e-6 with `n_seeds=1` (194 of toy's diverged runs are Sven at the bottom of the rtol grid — consistent with 688/7,825; `n_seeds` is displayed on every row).
* **For other WPs:** re-running any notebook that calls `scan_analysis.plot_sv_spectra` now draws the *correct* 3.45e-4 floor line instead of 1e-7 — expect that legend/line to change in `toy_1d/polynomial/mnist*/cifar_analysis`. `test_analysis_core.py`'s 1e-7 assertion still passes (its fixture records no `sv_noise_floor`).

## For `make_plots.sh` (do not edit — orchestrator)
Uncomment `# WP3 spectra    : spectra_analysis` in the `NEW=( … )` block. `comparisons` is already in `SCANS`.