**`analysis/paper_assets/reviewer.py`** (2285 lines) — module was ~complete from the interrupted attempt; I audited it against PAPER_PLAN §5, rebuilt it three times against the now-final inputs, eyeballed every figure, and fixed four defects. **`tests/test_paper_assets_reviewer.py`** — +4 tests (91 pass).

## Delivered (all re-runnable: `cd analysis && ../.venv/bin/python -m paper_assets --only reviewer`)
`iclr_manuscript/figures_iclr/reviewer/`: `budget` (F4), `overparam` (F5), `overparam_outcomes` (F5b, test+train loss vs P/N), `batchsize` (F6), `kappa` (F8), `divergence` (F10), `divergence_grids` (F10b), `knobs` (F14).
`iclr_manuscript/tables_v2/`: `budget`, `equal_budget` (T6), `divergence` (T11), `overparam`+`overparam_loss` (T12), `batchsize`+`batchsize_loss` (T13), `kappa` (T14, 3 blocks), `knobs` (T21).
`numbers_v2_reviewer.tex`: **343 macros** (Microbatch 63, Overparam 61, Paramfrac 53, Batchsize 43, Kappa 40, Div 40, Budget 35, Optimism 8), `provisional: []`. 18 `*.provenance.json` sidecars under `agent_lab/paper_assets/provenance/` (never in the Overleaf repo); PNG twins in `agent_lab/paper_assets/reviewer/`.

## Fixes I made
1. **`` `rtol' `` rendered literally** (backtick+apostrophe) in matplotlib titles/labels/legends of `batchsize.pdf`, `divergence.pdf`, `divergence_grids.pdf`. New constants `RTOL_TEX = \texttt{rtol}` (tables/captions, matching the manuscript's 14 existing `\texttt{rtol}`) and `RTOL_FIG = $\mathtt{rtol}$` (figures); verified in the rebuilt PNGs.
2. **`tables_v2/optimism.tex` was written by both `main` and `reviewer`** — contents depended on which module ran last. `reviewer` no longer writes it (plan §5.2 assigns it to `main`, whose `_t6_optimism` claims it); `_table_optimism` kept, uncalled and documented. Macros unaffected.
3. **Macro-file provenance under-reported its scans**: `\numDivSvenNRuns` etc. are campaign-wide, so the sidecar now unions `headline_figs.on_grid_scans()` (22 dirs incl. both CIFAR grids) instead of the 17 the module loads itself. Its stale "CIFAR-CE is still landing runs" note is now conditional on `C.provisional_scans()`.
4. Knobs caption takes the seed-band wording from `paired.SEED_SPREAD_LABEL` instead of hard-coding it.

## Verified by running
- Three full builds on CPU jobs: `figures=8 tables=9 macros=343`, exit 0. `--check` after the input refresh moved exactly `divergence.tex`, `optimism.tex` and the macro file — every other reviewer table is byte-identical, so the CIFAR-CE/Fig-5/profile refresh touched only what it should.
- 91 tests: macro names letters-only and injective over every swept value, no NaN/None/inf in macros or cells, no mid-line `%`, no text-mode `_`/`^`, every row ends `\\`, caption+label present, PDF+PNG+provenance for every declared figure, no cross-module macro collision, no sidecar in the manuscript repo.
- Looked at all 8 PNGs at print size: legible at 0.32/0.49, legends below the panels (never over data), log axes, Sven black, `style.method_label` names, `arm_ticks` correctly blanks the crowded 0.689 P/N label.
- Sample facts now in macros: Sven 688/7890 on-grid runs diverged (8.72 %), 0/195 on CIFAR-CE; equal budget n=8 — Sven 15 of 15 on toy 1D, 4 of 15 on polynomial, "never" matches the leader on 3 of 4 scans; κ=1 at k=64, step 1 → 0.41 vs 0.078 (439 % spread).

## Open issues for the integrator
- **`MACRO CLASH: numCifarCESvenK, numCifarLRSvenK`** — defined by both `large` and `spectra` (same value, 128). Not my files.
- `numbers_v2_large.tex` / `numbers_v2_spectra.tex` still carry `[PROVISIONAL]` CIFAR-CE stamps from pre-refresh builds: **both modules need re-running** now the inputs are final.
- `optimism.tex` on disk is still reviewer's rendering; re-run `--only main` (or the whole command) so main's version lands.
- Plan §5.2 (module `main`) contradicts §5.4/§8 (module `reviewer`) on T6/`optimism.tex`; resolved as above.
- `overparam_outcomes.pdf` and `divergence_grids.pdf` are extra panels beyond the inventory rows — wire them into App. G / P or drop them.