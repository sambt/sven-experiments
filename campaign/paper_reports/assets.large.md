**TASK "assets-large" — complete.** Resumed prior partial work (which was sound), verified it against the now-final inputs, fixed two defects, filled one scope gap.

## Files changed
- `/n/home/anon/sven-experiments/analysis/paper_assets/large.py` (kept; extended) — new `_headroom()`, `selected_lrs()`, `fig_nanogpt()`, `table_nanogpt_lr()`, 7 new nanoGPT lr macros, registry + provenance entries, corrected a stale docstring claim (v3 *does* profile the ResNet).
- `/n/home/anon/sven-experiments/tests/test_paper_assets_large.py` — stub gained a selection payload + `nanogpt_lr` frame; 4 new tests (74 pass).
- Regenerated (all by `python -m paper_assets.large`, nothing hand-typed): `iclr_manuscript/figures_iclr/large/*.pdf` (17), `iclr_manuscript/tables_v2/{cifar,fig5,transformers,transformers_nanogpt_lr,transformers_gpt2,transformers_gpt2_lr,profile_methods,gram_cost,profile_v2_v3,profile_scaling}.tex`, `iclr_manuscript/numbers_v2_large.tex` (**370 macros, 0 provisional, 0 skipped**), PNG twins in `agent_lab/paper_assets/large/`, `*.provenance.json` per file. Nothing committed anywhere; manuscript repo untouched except its untracked generated dirs.

## Verified by running (all via `campaign/run_cpu_tests.sh`)
- Full build twice end-to-end: 17 figures, 10 tables, 370 macros, `profile_results_v3` **720/720 incl. `cifar_resnet18`** → PAPER_PLAN risk #3 is resolved; no v2/v3 mixing, no provisional stamps anywhere.
- Inputs are picked up live, no caching: CIFAR-CE selection reads **k=128, lr=0.5, rtol=0.3** (landscape's red ring, rank-used panel, all macros); Fig-5 shows **both** on-disk configs (selected `k=128,lr=0.5,rtol=1e-3` solid+ringed, old `k=64,lr=1` dashed "not selected").
- LaTeX: all 10 tables + all 370 macros compile in a scratch doc (11 pp, 0 errors/undefined); all 17 figure PDFs `\includegraphics` cleanly at 0.32/0.49/1.0.
- Looked at every PNG. Fixed: data clipped by the top spine in `profile_batchsize`/`k_sweep`/`scaling` (added log-space headroom), and colliding tick labels in the new nanoGPT lr panel (decade ticks). Re-checked after: legible at target width, legends below/clear of data, Sven black, log axes sane.

## Results the refresh moved (integrator: re-read App. J/K/L/Q)
- **Fig-5 collapse threshold moved**: at the *selected* config accuracy is 19.2 / 25.6 / **67.8** / 68.8 / 69.1 % at f = .05/.1/.25/.5/1 → `\numFigFiveCollapseFrac{}` = **0.1** (was ≈0.5 under the `lr=1` confound). Cost finding unchanged: **0.515×** memory at **1.71–2.42×** step time.
- **CIFAR-CE** (5 confirm seeds): val 1.355±0.016, test 1.339, test acc 58.2 %, rank 9, train-eval 0.1586, 178 ms/step, 22 965 MB; `rtol=0.3` now binds — used rank **48.8 = 38.1 %** of k=128 (was at the cap).
- nanoGPT lr sensitivity is now an asset: Sven's optimum is **interior**, worst/best val spread **2.71×**.

## Open issues / notes
1. `analysis/large_figs.py` carries uncommitted helpers this module depends on (`paramfrac_groups`, `headline_sven_config`, `mark_selected`) — must be committed with `large.py`.
2. `numFigFive*` describe the **selected** group only; the k=64 block exists in `fig5.tex`/the figure but has no macros. Fine for the planned App.-K prose; say "selected configuration" when quoting.
3. `transformers_nanogpt_lr.tex` (T17a2) and `figures_iclr/large/nanogpt.pdf` are **additions** beyond PAPER_PLAN's F11 row (scope asked for nanoGPT curves + lr sensitivity); main's F1 still owns the main-text nanoGPT panels — don't double-place.
4. `profile_phases`: the MNIST hooks bar shows no solid capture segment (missing phase breakdown in v3 for that cell) — data gap, not a plotting bug.
5. Scratch evidence (probes, tex checks) in `agent_lab/scratch/large_task/`; delete freely.