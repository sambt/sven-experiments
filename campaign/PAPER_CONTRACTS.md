# Manuscript-update contracts (2026-09-20) — binding for every paper agent

> **Figure API, 2026-09-21.** The figures are no longer drawn-and-written in one call:
> each one is a `FIGURE_SPECS` entry in its `paper_assets` module (`draw(ctx, opts) ->
> (fig, meta)`, writing nothing) and `build()` is the only save path, so the same builder
> can be driven from `analysis/notebooks/paper/`. Cosmetic choices made there are pinned in
> `analysis/paper_assets/figure_overrides.yaml` and replayed by the CLI, so **the override
> file is part of how the paper looks** — read it before concluding a figure's builder
> produces what you see. `campaign/FIGURE_API_CONTRACT.md` is binding for any change to a
> figure; `tools/cmp_figures.py` checks whether a rebuild moved anything.
>
> **Layout change 2026-09-21.** The analysis layer this document points at was reorganised:
> the helper modules are now `analysis/lib/*.py` (`headline.py`, `headline_figs.py`,
> `large_figs.py`, `spectra_figs.py`, `reviewer_figs.py`, `legacy_diff.py`,
> `profile_helpers.py`, ...) and the notebooks are grouped under
> `analysis/notebooks/<group>/`. `analysis/paper_assets/` and its entry point
> (`cd analysis && ../.venv/bin/python -m paper_assets`) are unchanged, and so is every
> output path. Map: `analysis/README.md`.

## What the user asked for (verbatim priorities, condensed)
Update the ICLR manuscript in `iclr_manuscript/` in a **NEW tex file, starting from a copy of the existing manuscript**
(`iclr2026_conference.tex`), with **every edit in blue text** so it is distinguishable from the old text. Priorities:
1. Get all the new results in, including the added baselines. Consider making the headline results **polynomial, MNIST
   and nanoGPT** (transformer/LM support now exists). Consider reworking the singular-value-spectrum discussion to reflect
   what the new analysis shows (full spectra, how they evolve over training, what truncation discards).
2. Update every description of the optimizer and the experiments so it matches what was actually run. Add a main-text
   mention plus a **full appendix** on the memory-optimised **Gram-trick** version of Sven that makes larger models tractable.
3. Add plots/tables for all the new studies done in response to the NeurIPS reviews (mostly appendix; new baselines as
   new lines on main-text plots). The paper should contain a **comprehensive summary of all results obtained**.
4. Update the profiling study to the new profile.
5. Any other update needed to tell the full story.
Style: match the existing writing — concise, not verbose, but no important detail omitted.

## Files and conventions
* New main file: `iclr_manuscript/iclr2026_conference_v2.tex` (a copy of `iclr2026_conference.tex`; the original is
  never edited). `iclr_manuscript/` is its own git repo synced with Overleaf: **never commit, push, pull or change
  branches there**; do not delete or rename existing files. New figures go to `iclr_manuscript/figures_iclr/<group>/`
  (PDF), generated tables to `iclr_manuscript/tables_v2/*.tex`, generated number macros to
  `iclr_manuscript/numbers_v2.tex`. Everything the new tex needs must live inside `iclr_manuscript/` (Overleaf).
* **Blue markup** (defined once in the preamble of the v2 file): `\new{...}` for inline/short edits and
  `\begin{newtext} ... \end{newtext}` for whole new paragraphs/sections (both = `\color{blue}`); new or changed
  figure/table captions are blue; a replaced figure keeps its label but its caption is blue. Text that is removed is
  deleted (not struck through), except that a short `% OLD: ...` comment may keep a removed sentence when a claim changed
  materially. Unchanged text stays black and byte-identical.
* **Numbers are never typed by hand.** Every campaign-derived number in the prose is a macro from `numbers_v2.tex`
  (e.g. `\numSvenPolyVal`), and every results table is `\input` from `tables_v2/`. Both are produced by ONE re-runnable
  script, `analysis/paper_assets.py` (figures too), from the analysis layer (`analysis/headline.py`,
  `headline_figs.py`, `reviewer_figs.py`, `large_figs.py`, `spectra_figs.py`, `legacy_diff.py`, `profile_helpers.py`,
  `bench/best_configs.json`). Reason: CIFAR-CE, Fig-5 and the profile are being refreshed while the paper is written;
  one command must update the paper. Macro names: `\num<Scan><Method><Quantity>`, documented in a comment header.
* Figures: regenerate from the analysis layer in the manuscript's visual language (single-column-width panels used 3
  across at `0.32\linewidth`, log-y validation loss vs epoch and vs wall time, Sven black, `style.METHOD_COLORS`,
  `style.method_label` names, seed band = mean +/- 1 std over seeds). Headline numbers come from the CONFIRMATION seeds;
  wall time from the standalone `<scan>_timing` runs. Read `campaign/ANALYSIS_CONTRACTS.md` for the binding analysis
  conventions and `campaign/analysis_reports/B.fix.*.md` for the facts and caveats the reviews established.
* **Claims discipline.** The paper must tell the truth the data support (see the caveat list in
  `campaign/CAMPAIGN_STATUS.md`, entry "Analysis phase B committed", and `analysis/WHAT_CHANGED.md` section 4):
  Sven is 2nd (behind HIG) on 1D and polynomial regression, top-tier on MNIST (MuonW first), within ~1% of AdamW on
  nanoGPT, clearly behind on CIFAR-10/ResNet18 and GPT-2-small; it needs fewer epochs than Adam on the regression tasks
  but not less wall time on all of them; its larger tuning grid is partly redundant and at a small equal budget it is
  not ahead; it diverges at very small rtol; results are not bit-reproducible across GPU types for several methods.
  Negative results are reported plainly (appendix + a limitations paragraph), never hidden and never apologetic.
  The paper does NOT discuss the legacy bugs or the campaign's internal history; it simply describes the protocol that
  was run (train/val/test splits, selection on validation, confirmation seeds, failure accounting).
* Anonymity is preserved (no names, cluster names, paths, or repo URLs other than the existing anonymised ones).
* LaTeX: TeX Live 2018 is at `/usr/bin/pdflatex` + `bibtex`; the v2 file must compile from inside
  `iclr_manuscript/` with `pdflatex -interaction=nonstopmode iclr2026_conference_v2 && bibtex iclr2026_conference_v2
  && pdflatex ... && pdflatex ...` with no errors and no undefined references/citations. New citations go to a NEW bib
  file `iclr_manuscript/references_v2.bib` loaded alongside the existing one. ICLR main text limit: 9 pages
  (references and appendices unlimited) — check the page count of the main text.
* Ground rules: repo `/n/home/anon/sven-experiments` (branch robustness-campaign): never commit there either unless your
  task says so; results roots are read-only; heavy CPU work through `campaign/run_cpu_tests.sh`; python
  `.venv/bin/python`. Edit only the files your task owns.
