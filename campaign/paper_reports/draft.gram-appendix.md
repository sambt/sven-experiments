**Files written** (both rewritten from scratch; prior drafts in place were unverified and carried 18 non-existent macros)

- `/n/home11/sambt/iaifi/sv3/iclr_manuscript/sections_v2/app_gram.tex` (418 lines) — App. D, §§ D.1 Gram identity · D.2 assembling `G` from hooks · D.3 restrictions + `full`/`chunked` fallbacks · D.4 truncation & numerical floor · D.5 micro-batches/masks · D.6 exactness tests · D.7 Algorithm (`alg:gram`) + Table (`tab:gram_cost`, `\input tables_v2/gram_cost`) · D.8 measured cost · D.9 open problem. `\section` and all prose inside `newtext`; both floats outside it, captions via `\new{}`.
- `/n/home11/sambt/iaifi/sv3/iclr_manuscript/sections_v2/main_gram_paragraph.tex` (69 lines) — BLOCK A = corrected cost paragraph replacing `iclr2026_conference.tex:290`; BLOCK B = new closing §2.3 paragraph.

**Verified by running**
- `campaign/run_cpu_tests.sh` on the 7 Gram/logging test files → **123 passed, 3 skipped**. Real tolerances are `DELTA_RTOL=1e-9` (relative, applied update) and `GRAM_ATOL=1e-10` (max abs, `G` vs `MMᵀ`), **float64 on CPU** — the task brief's "1e-6 float32 / 1e-13 float64" does not exist in the suite; D.6 states the real ones (they are code constants, not campaign numbers, so not macros).
- Four-pass-style `pdflatex` of both fragments in a scratch harness with the out-of-repo TeX shim: **0 errors, 0 undefined references/citations, 0 hyperref warnings, 0 overfull boxes**. App. D = **~4.3 pp** (plan estimated 2.5), main paragraphs ~0.45 pp. Probe PDF: `/tmp/claude-66176/-n-home11-sambt-iaifi-sv3/30b30b16-e05d-4e46-8a25-371ea6181950/scratchpad/app_gram_probe.pdf`.
- **All 46 `\num…` macros used are defined** in the current `numbers_v2_{main,large,spectra}.tex` (checked by set-diff). No new macros required.
- **PAPER_PLAN's shim in §6 is broken**: the vendored `algorithmic.sty` defines `\STATE` *inside* the environment, so `\let\State\STATE` at package load leaves `\State` undefined and every algorithm errors. Working replacement (`\newcommand{\State}{\STATE}`, `\newcommand{\If}[1]{\IF{#1}}`, …) is at `…/scratchpad/shim/algpseudocode.sty`.

**Corrections made against the code/data (the earlier draft was wrong on these)**
1. Macro names: `numProf<Arch><Backend>{CaptureFrac,KSpread,KMin,KMax,VsAdam,VsSgd}` (`VsAdam` = step time, `VsSgd` = peak memory) and `numProfMnistWidth{Hooks,Full}{Step,Mem}Exp` — not the `…Sven…Exponent` forms guessed before.
2. **Dropped "unlike the classic path whose cost is ∝ k"**: measured classic `k` spread is 1.01–1.11, because its cost is dominated by materialising `J` (87.2 % of its ResNet18 step). D.8 now says the `O(kN|D|)` factor is invisible in the classic path at these ranks, and distinguishes algorithm from regime.
3. **Reversed the rtol/floor claim**: `numSpectraRtolBelowFloor` = 1, i.e. one scan's *selected* rtol (1D, 1e-4) sits below the float32 Gram floor (`numSpectraNoiseFloorRel` = 3.5e-4·σ₁) — stated plainly instead of "rtol settings below the floor are precisely the ones that diverge".
4. `chunked` is **not used by any campaign scan** (EXPERIMENTS.md §1.5); `full` is what every ResNet run uses. Fallbacks reordered accordingly.
5. Micro-batching: capture cost is unchanged only under `hooks`; under `full`/`chunked` the row count (and cost) falls with μ.
6. Masked `rows` mode excludes normalisation layers (`_check_rows_supported`), unlike `tensor`/`elementwise` — now stated correctly.
7. Removed the unsupported online-vs-offline Gram numbers (~5e-07 / 2.4e-04 exist only in `A.fix.wp3-ckpt-tools.md §7`, with no emitter); D.6 makes the qualitative claim and delegates the figures to App. O.
8. Added the allocator caveat (`numProfSvenStepSpeedupMax` 4.48×, `numProfBaselineRatioWorst` control) and the rank-law check (`numSpectraRankLawSteps` 49,040 steps).

**Notes for the integrator**
- Must supply `\label{fig:cost_memory}` (F3). `app:spectra`/`app:robustness` come from `app_reproducibility.tex`; everything else exists today.
- `tables_v2/gram_cost.tex` (T19) is 11 columns and overruns `\textwidth` by ~270 pt at `\small`; I wrapped it in `\resizebox{\textwidth}{!}{…}` so the build is clean, but **`large.py` should shed two columns** ("k range", "×SGD") for legibility.
- Optional macros that would tighten prose: `numMnistLRSvenMemVsAdam`, `numCifarLRSven{MsStepVsAdam,MemVsAdam}` (ratios are currently avoided by printing both MB/ms values).
- **Notation conflict**: I kept the manuscript's `N` = parameter count, `|\mathcal{D}|` = rows, per the task brief; `PAPER_PLAN.md` §3 proposes `B/P/N` for the new methodology paragraph. If §2 adopts B/P/N, this appendix must be renamed with it (flagged in the file header).
- App. D landed at ~4.3 pp vs the plan's 2.5 pp; appendices are unlimited, but a further cut would have to drop content (D.5 masks or D.8's scaling paragraph are the least load-bearing).
- `%` comments in both files are integrator notes naming repo-internal modules; header says to strip them before camera-ready.