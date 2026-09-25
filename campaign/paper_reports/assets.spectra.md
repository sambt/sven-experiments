Everything verified. Writing the report.

**TASK "assets-spectra" — complete.** Resumed prior partial work, refreshed against the now-final inputs, fixed a build-breaking macro clash, polished every figure, added one missing asset.

## Files changed
- `/n/home11/sambt/iaifi/sv3/analysis/paper_assets/spectra.py` (1631 → 1764 l) — F2, F9, T8, T9, G4.
- `/n/home11/sambt/iaifi/sv3/tests/test_paper_assets_spectra.py` — +3 tests (38 total; **37 passed, 1 skipped**).
- Written into `iclr_manuscript/` (no tex edited): `figures_iclr/spectra/*.pdf` (13), `tables_v2/spectra_{mechanism,mechanism_main,probe_energy,low4,probe_widths}.tex`, `numbers_v2_spectra.tex` (**149 macros, 0 provisional**), `numbers_v2.tex` (index regenerated).
- Eyeball PNGs: `agent_lab/paper_assets/spectra/*.png`; provenance sidecars: `agent_lab/paper_assets/provenance/**` (one per generated file, outside the Overleaf repo); LaTeX harness `agent_lab/paper_assets/scratch/spectra_texcheck/check.tex`.

## Verified by running (all via `campaign/run_cpu_tests.sh`)
- `python -m paper_assets.spectra`: 13 figures, 5 tables, 149 macros, **0 provisional** (CIFAR-CE diag now 5/5 seeds at k=128, lr=0.5, rtol=0.3). Rank law `min(k, rtol-rank) == num_nonzero_svs` holds on **49,040** logged steps of 7 scans.
- `macro_collisions() == {}` — **fixed a build-breaker**: `\numCifar{LR,CE}SvenK` was defined by both `large` and `spectra`. Mine are now `\num<Scan>SvenKCap` (7 scans, semantically the rank *cap*); `large` keeps `...SvenK` for App. J. `numbers_v2.tex` no longer carries the clash warning.
- pdflatex ×2 over all 5 tables + 149 macros + 6 figures: **0 errors, 0 undefined macros**, no table overfull (the 15pt overfulls are my harness's `\parindent`).
- Looked at all 13 PNGs at print size. Fixed: clipped rightmost titles in every 4-column grid (two-line `_panel_title`); step/`k`/rtol/noise-floor labels sitting on data or on each other (new `_emptiest_corner`, translucent label boxes, deterministic placement — `k` label left of its line at the top, floor label at the right edge, step labels lower-left); in-axes scan label in `probe_spectra` → suptitle; clipped `k` labels in the new figure.

## Claims discipline — three PAPER_PLAN claims the refreshed data contradict
1. **C13 is wrong as written.** With CIFAR-CE re-selected to rtol=0.3, **rtol binds, not k**: used rank 106.0 → 36.6 of 128 (`\numCifarCESvenBinds{}`=rtol, `...BindsFirst`=rtol). CIFAR-LR: 128 → 81.6, nothing binds at step 0 and rtol binds by the end. "On CIFAR-10 k binds rather than rtol… 112.5/128 and 123.3/128" must be rewritten.
2. **C10's "the fraction rises on every scan" is now false.** CIFAR-CE kept energy **falls 90.1 % → 76.7 %** (was 100 → 99.999 at the old rtol=0.01); discarded 0.26. The rise still holds on poly (92.3→99.6) and MNIST-CE (31.0→95.2).
3. **C11's probe-width caveat conflates rows with width.** Probe *rows* are 10,000/10,000/512/512; the *spectrum width* min(rows, P) is 593/673/512/512.

Everything above is a macro or a table cell — re-running updates the prose.

## Notes for the integrator
- **Profiling (scope item 2) is `large.py`'s, per PAPER_PLAN §5.4**, and is done there on v3 (`\numProfRoot`, `\numProfComplete`=complete, CIFAR rows present, `profile_{methods,scaling,v2_v3}.tex`, `gram_cost.tex`, 6 `profile_*.pdf`). I did not duplicate it — doing so would have re-created ~100 macro clashes.
- **New asset beyond the inventory**: `figures_iclr/spectra/mnist_lr_vs_ce.pdf` (1×3 @0.32) — the regenerated drop-in for the retired single-seed App. I figure (old `figures_v2/comparisons/sv_spectra_lines_compare_mnist_ce_vs_labelReg.pdf`): spectrum at 5 matched training fractions, used rank, kept energy, seed bands, both rtol/k cuts drawn. MNIST-CE's scan colour changed to dark neutral `#4A4A4A` (orange-vs-brown was indistinguishable at 0.32 linewidth); affects F2(b,c) and `online_mechanism` legends too.
- New macros to use: `\num<Scan>SvenBindsFirst` (verdict at the first logged step) beside `...SvenBinds` (last step). `\numSpectraRtolBinds{}`=5, `\numSpectraKBinds{}`=2, `\numSpectraRtolBelowFloor{}`=1 (toy's rtol sits below the recorded float32 floor).
- Captions are drafts in `% caption:` comments at the head of each table; T8's notes carry the binding caveats verbatim.
- `PAPER_ASSETS_STRICT=1` turns any provisional scan into a hard failure; nothing is provisional now.