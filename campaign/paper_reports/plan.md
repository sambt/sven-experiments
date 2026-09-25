**Created:** `/n/home/anon/sven-experiments/campaign/PAPER_PLAN.md` (526 lines, ~10.5k words). Nothing else written; `campaign/grid_counts.md` was already modified by another WP. No commit, nothing touched in `iclr_manuscript/`.

## What the plan decides

**Headline (§1):** main results = **polynomial + MNIST-LR + nanoGPT**; 1D → App. E. Justified on 1D's seed variance (5.09e-07 ± 1.49e-06, same-instance gap −63 %) and on the three retained tasks spanning regression / real classification / transformer LM with the full 15-method field.

**§2 CLAIMS MAP:** 30 forward claims (C1–C30), each with asset id + analysis function of record + today's number + a **binding** caveat; plus 18 retired claims (X1–X18) from the current tex with line numbers and replacement direction. The load-bearing reversals: `O(kN|D|)` → Gram cost (5 locations); "memory on par under mild restrictions" → hooks-capture parity (MNIST 0.99×, nanoGPT 1.9×) **vs** full-capture 47× on the ResNet; Fig-1's L-BFGS/HIG caption (HIG is *first* on both synthetics, L-BFGS 204/225 diverged); "k ≈ B ⇒ many significant directions" → `used = min(k, rtol-rank)`, `rtol` binds on 5/7; CIFAR "similar loss" → 9/11 both losses; Fig-5 "unchanged down to 5 %" → collapse to chance accuracy; κ=1-crashes → 210/210 complete with signed residuals.

**§3–4:** section-by-section main-text edits (incl. the R3 dimensionality paragraph, the Gram paragraph, the new protocol paragraph, a new §4.2 spectra subsection, a new limitations paragraph) with a named deletion paying for each addition; 18 appendices A–R, 9 of them new, each with what changes / assets / length.

**§5:** 15 figure ids + 21 table ids + 4 macro groups — output path, functions, panel layout, owning module (`main|reviewer|large|spectra`), main-vs-appendix, caption message, and which existing figure each replaces. Plus the `analysis/paper_assets/` package spec (CLI, `--check`, freshness gate, `tests/test_paper_assets.py`).

**§6:** page budget. **§7:** 14 risks. **§8:** sequencing (`paper-assets-core` must land before the four module agents).

## Verified by running

- **All 84 named analysis functions/attributes exist** across `headline`, `headline_figs`, `budget`, `reviewer_figs`, `large_figs`, `spectra_figs`, `profile_helpers`, `paired`, `ckpt_tools`, `style` → `MISSING: none`.
- **Page budget is measured, not guessed**: compiled the current tex in scratch. Body = **≈8.6 pp** (§4 p6, Fig1 p7, Fig2 p8, §5/§6 p9, references start ~60 % down p9), so real slack ≈0.4 pp. Corrected my first draft's 9.45 → 8.60.
- **Build gate problem found and solved**: this TeX Live 2018 lacks `algpseudocode`/`algorithmicx`, `bbm`, `nicefrac` (and `siunitx`). Decision: an **out-of-repo `TEXINPUTS` shim** (exact stub contents in §6, verified to build all four passes with 0 undefined refs/citations) — never vendor into the Overleaf repo, since a local `algpseudocode.sty` would shadow the real one.
- **[MOVING] state at 22:33 EDT**: CIFAR-CE re-selected to `k=128, lr=0.5, rtol=0.3`; each of `_confirm/_timing/_diag` holds **1 of 5** records at the new config → every CIFAR-CE number in the plan is stale. Fig-5 re-run under way at `k=128, lr=0.5, rtol=1e-3` (**3 of 15**, 5 claims live) — this removes the `lr=1.0` confound both old blow-ups sat at, so the collapse threshold may move or vanish. `profile_results_v3` = **578/720 and has no `cifar_resnet18` directory**.

## Notes for the integrator

1. Macro names must be **letters only** (TeX forbids digits) — hence `\numFigFive…`, `\numGptTwo…`.
2. F3's and App. Q's CIFAR cost points come from `headline.efficiency_table` (timing pass: 177 ms/step, 22,960 MB) **not** the profiler, so they survive v3 never getting CIFAR.
3. F1's bottom row: confirmation-seed **losses** against timing-pass **times** — never a timing run's loss (one SOAP MNIST run differs 200×).
4. Nothing under `analysis/tables/` may be `\input` (fallback renderer, no `tabulate`); `paper_assets._common` emits its own booktabs.
5. Open: `references_v2.bib` needs ≥7 new entries (nanoGPT, GPT-2, FineWeb-edu, LoRA, Muon scaling, `torch.func`, SGD+momentum).