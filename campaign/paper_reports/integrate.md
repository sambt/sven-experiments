Appendix lettering matches the plan exactly (A–R, `app:kappaScans` = M). Final report:

---

## `integrate` — complete. Main text is **9 pages**; build is clean.

### Files created / changed (all in `iclr_manuscript/`, nothing committed in either repo; `iclr2026_conference.tex` byte-identical)

| path | what |
|---|---|
| `iclr2026_conference_v2.tex` | **the deliverable.** Copy of the original + blue markup, `\input{numbers_v2}`, `references_v2.bib`, `\svnfittable`, main-text float parameters. All §1–§5 edits. |
| `references_v2.bib` | **new**: AdamW, Muon-scalable, LoRA, per-example gradients, nanoGPT, GPT-2, FineWeb-edu, Transformer, momentum, `torch.func` |
| `CHANGES_v2.md` | **new**: section-by-section rationale, 15 weakened/corrected claims + 4 strengthened with their evidence, 3 claims the refresh contradicted, 9 author decisions |
| `sections_v2/app_{additional,budget,overparam,batchsize,mnist_ce,cifar,fig5,transformers,kappa,lims_results,spectra,robustness,profiling,grids_tables}.tex` | **new** (14 fragments): App. E–R written or rewritten |
| `sections_v2/app_{gram,main_gram_paragraph}.tex` | renamed `|D|→B`, `N→P` to match the main text and every generated table; added `\cite{goodfellow2015efficient}`; fixed `\timesB` |
| `sections_v2/app_{exp_details,reproducibility}.tex` | wired in; dataset citations added; wide tables guarded |
| `agent_lab/paper_build/` | out-of-repo build harness: `build.sh`, `base.sh`, TeX shims, `iclr2026_conference_v2_PROOF.pdf` |

### Verified by running (everything via `campaign/run_cpu_tests.sh`)
- **4-pass build: 0 errors, 0 undefined references, 0 undefined citations, 0 overfull boxes >10 pt anywhere** (92 pp). `sec:code` on **p9**, references from p10. Like-for-like: the same harness reproduces the original's committed layout (27 pp, `sec:code` p9).
- **477 `\num…` macros used, all defined** (set-diff against the 1,249 generated).
- Appendix lettering A–R matches `PAPER_PLAN` exactly; every `\includegraphics` target exists; `newtext` balanced in all 17 files; **no float inside `newtext`** (risk 9); no `figures_v2/` or `figures/` references remain.
- Anonymity scan clean (one repo-path mention survives in an `app_gram.tex` *comment*).
- Measured every generated table's natural width in a probe build and guarded the 13 that overflow; every figure include is at `\linewidth`, which is their design width (397 pt).

### To reach 9 pages I moved out of §4 (all one `\input` from returning, listed in CHANGES_v2 §3.3)
**Table 1** → App. E (it duplicates the per-scan confirmation tables already there; the prose quotes every number); **Fig. 2** → App. O; **Fig. 3** → App. Q; the two-column spectra summary dropped. Prose was also compressed ~25 % throughout; nothing mandated (dimensionality, protocol, limitations, §4.2) was cut.

### Open issues for the next agent
1. **Table 1 in the appendix is the weakest point** of the layout — §4 now has one float. Recovering ~0.5 pp elsewhere would put it back.
2. **Three claims the refresh contradicted** are handled in-text but are worth a review lens: kept energy *falls* on 2 of 7 scans (CIFAR-CE 90.1→76.7 %, not "rises everywhere"); on CIFAR-CE **`rtol` binds, not `k`** (used rank 48.8/128); batch-size loss is **not** monotone.
3. I dropped all "R1 asked…" phrasings (fresh submission, not a response letter) — an author decision flagged in CHANGES_v2.
4. Unused generated assets: `time_to_target.tex` (112 rows, taller than a page — App. E uses the median-target version), `headline_confirm_compact.tex`, `confirmation_exp_nanogpt_speedrun.tex`.
5. `PAPER_PLAN` §6's `algpseudocode` shim is broken (`\let\State\STATE` at load time captures nothing); the working version is `agent_lab/paper_build/shim/`.