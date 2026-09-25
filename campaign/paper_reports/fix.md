Done. Final report:

## Files changed

**Asset code** (`/n/home/anon/sven-experiments/analysis/paper_assets/`): `spectra.py` (added `dry_run`; `_name_list`; `ENERGY_PINNED_TOL`; new kept-energy/binds macros; panel (b) redrawn as discarded-fraction on a log axis over all 7 scans; panel (a) legend recoloured), `common.py` (`MODULES` += `campaign`), `main.py` (`FIRST_ORDER_METHODS`, first-order + "ahead of Sven" macro families, `numCostMemory*`, `numNBaselinesMlp`, `numPolyLbfgsRank/Fin/Att`, grid-edge ↑/↓ flags in `grid_<scan>.tex`, F3 arch ticks derived from data, per-panel Fig-1 legends), `large.py` (`BestNameAcc`/`BestAccName`/`BestNameTrainEval`, same-source `CifarLR{Adam,MuonW}MsStep`/`SvenMsStepVsAdam`/`SvenMemVsMuonW`/`SvenWallVsMuonW`, `profile_protocol()` → `numProfStagedBatches/WarmupSteps/MeasuredSteps/AmortSteps/NRecords`, `num<key>Params`, `_grid_runner_up` → `CifarCERunnerUp*`, `numFigFiveRunsSelected/Other`, nanoGPT ahead-lists, T19 columns dropped).
**Tests**: `/n/home/anon/sven-experiments/tests/test_paper_assets_large.py` (T19 assertion updated).
**Manuscript** (`/n/home/anon/sven-experiments/iclr_manuscript/`): `iclr2026_conference_v2.tex`, `sections_v2/{app_cifar,app_overparam,app_spectra,app_profiling,app_gram,app_budget,app_additional,app_batchsize,app_fig5,app_mnist_ce,app_transformers,app_exp_details,app_grids_tables,app_kappa}.tex`, all `numbers_v2*.tex`, `tables_v2/*`, `figures_iclr/*`, `CHANGES_v2.md` (new §4 recording every finding, plus corrected stale claims in §1/§2/§3). Nothing committed in either repo; `iclr2026_conference.tex` untouched.

## Verified by running

- `python -m paper_assets` **exits 0**: `main 7/31/418 · reviewer 8/9/343 · large 17/10/422 · spectra 13/5/158 · campaign 0/0/11`; `numbers_v2.tex` now inputs `numbers_v2_campaign`.
- LaTeX 4-pass: **0 errors, 0 undefined refs/citations, 0 overfull >10 pt, 96 pages, `sec:code` on page 9** (main text at the limit). Proof at `/n/home/anon/sven-experiments/agent_lab/paper_build/iclr2026_conference_v2_PROOF.pdf`.
- Gates: 535 `\num` macros used, **0 undefined**; **0 orphan `fig:`/`tab:` labels**, 0 undefined refs; elided word-diff vs the original is clean (Alg-1 caption, Shampoo/SOAP sentence and Conclusion ¶2 confirmed byte-identical).
- `pytest tests/test_paper_assets_*`: **244 passed, 1 skipped**.
- Data re-checked from records (`/n/home/anon/sven-experiments/agent_lab/paper_fixer/verify.py`): CIFAR-CE grid = 39 configs/195 records, rtol axis max 0.3, lr at rtol=0.3 ∈ {0.05,0.1,0.5}; CE train-eval Sven 0.1586 vs MuonW 0.3106 vs median 0.0962; timing-pass Sven 176.9 ms vs Adam 6.5 ms (**27.2×**); profile protocol 8/10/50 unanimous over 720; K-FAC 35/50 poly, 34/40 1D, 40/40 diverged on each MNIST; Fig-5 = 15 + 15; batch-size field = 13 (no HIG/JD); overparam MNIST N=40000 → P/N 0.689.

## Findings rejected

- **"27× in CHANGES_v2 unsupported"** — the 27× is right; it was the *tex* that was wrong (cross-source 7.5×). Fixed in the tex, CHANGES_v2 kept.
- **"Rename N→P inside App. A/B"** — would make two byte-identical appendices edited. Instead: blue notation paragraph at App. A + translation in §2.1 + App. Q warning.
- **"Promote Fig-5 figures to §4"** — the numbers (seeds/accuracy/memory/time) are now in §4.1; the figures stay in App. K because the main text is exactly 9 pages. Recorded as a conscious trade.
- **"Run the tensor-mask arm for R2"** — out of scope for a paper pass; answered as a named non-experiment in App. I.

## Notes for the integrator

1. **Overleaf is authoritative.** The local build needs `agent_lab/paper_build/shim/` (algpseudocode/bbm/nicefrac) and the shim maps `\State`→`\STATE`, so the four algorithm floats render at a different height there. `sec:code` sits on page 9 with ~0 slack — re-check the page count on Overleaf first.
2. **§4 still has one float** (Fig. 1). `tables_v2/headline_confirm_compact.tex` is one `\input` from returning as Table 1 if anything else is cut.
3. **Not done:** Figs. 1/2 MNIST panels are still dominated by SOAP's failed run and HIG's 58 s/epoch on the time axis (axis clipping/insets not attempted).
4. `numCifarLRSvenB` is used for the "$128\times128$" Gram-matrix phrasing; if the CIFAR batch size ever changes, that sentence follows automatically.