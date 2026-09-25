Done. Final report:

---

## assets-main — complete

**Resume finding:** the interrupted attempt had already written the whole module; it was built against pre-refresh data (2026-09-20 23:28, `cifar10_resnet_ce_scan` stamped `PROVISIONAL`). I audited it, fixed two real defects, and rebuilt everything against the now-final inputs.

### Files changed
| path | change |
|---|---|
| `/n/home/anon/sven-experiments/analysis/paper_assets/main.py` | F1 top row and F13 columns 1/3 moved from `versus='step'` to `versus='epoch'`. Both docstrings already said "vs epoch", `PAPER_CONTRACTS.md` names "log-y validation loss vs epoch and vs wall time", and C7 is quoted in epochs — the step axis was an oversight, not a decision. `_ALIGN`-comment documents it. |
| `/n/home/anon/sven-experiments/analysis/paper_assets/common.py` | `booktabs` now left-aligns `p{}` cells (header and body) via `\parbox[t]{\linewidth}{\raggedright …\strut}`. A justified 1-inch column stretched `k=128, lr=0.5, rtol=0.3` across half the cell. `array`'s `\arraybackslash` is unavailable and `PAPER_PLAN` §6 forbids a new `\usepackage`; the naive `{\raggedright …\par}` costs one extra line per row (**measured: T1 141 pt → 242 pt**, a third of a page). The parbox form is byte-identical in height to the justified original (24.06 pt both) and preserves the column spec. |
| `/n/home/anon/sven-experiments/tests/test_paper_assets_main.py` | +2 tests: `test_booktabs_left_aligns_a_p_column_without_the_array_package`, `test_booktabs_never_puts_a_bare_par_in_a_p_cell` (the regression guard for the extra-line cost). 42 pass. |
| `iclr_manuscript/{numbers_v2_main.tex, numbers_v2.tex, tables_v2/*.tex (31), figures_iclr/main/*.pdf (7)}` | rebuilt; provenance JSONs under `agent_lab/paper_assets/provenance/`. No commits in `iclr_manuscript/`. |

### Verified by running (all via `campaign/run_cpu_tests.sh`)
- `headline.freshness_report()`: 21/21 passes fresh, **0 provisional** — CIFAR-CE confirm/timing now hold 5 records at the new selection, `profile_results_v3/profile_cifar` exists.
- `python -m paper_assets --only main --check`: **7 figures, 31 tables, 376 macros**, no `PROVISIONAL` stamp anywhere in `numbers_v2_main.tex`. Work list complete: F1(+`headline_curves_all`), F3, F12(`k_sweeps`+`hparam_landscape`), F13(`allseed_curves`+`allseed_curves_ce_lm`, covering 1D and MNIST-CE), T1,T2,T3,T4,T5(×14),T6-optimism,T10(×8),T18,T20.
- **Looked at all 7 PNGs** in `agent_lab/paper_assets/main/`: legible at 0.32\linewidth, legends below the figure (or white-framed in a data-free corner), log axes correct, Sven black and thick, `style.method_label` names, truncated panels annotated "axis truncated". rtol sweeps include the extension points (poly→0.1, MNIST-CE→0.3).
- **Real LaTeX compile** of all 31 tables + all 376 macros (`agent_lab/paper_assets/scratch/texcheck/`): **0 errors, 24 pages**; rendered and eyeballed T1, T2-main, T10, the CIFAR confirmation tables. Automated scan of my 32 output files: 0 NaN/None/inf, 0 bare `%`, 0 unescaped `_`.
- Full suite `tests/test_paper_assets_{main,large,reviewer,spectra}.py`: **241 passed, 1 skipped**, so the `common.py` change is safe for the other three modules (only `main` uses `p{}` columns today).

### Open issues / integrator notes
1. **BLOCKING macro clash (not mine):** `\numCifarLRSvenK` and `\numCifarCESvenK` are `\newcommand`ed in **both** `numbers_v2_large.tex` and `numbers_v2_spectra.tex` (same value, 128). `numbers_v2.tex` carries the warning; the build **will** fail. `large` or `spectra` must drop them.
2. **C24 is now false.** The CIFAR-CE refresh moved Sven to val `1.355 ± 0.016`, test acc `58.21 %`, **train-eval `0.1586`** (was 0.819, "the highest of 11"). It is now mid-field; the highest is SGD+momentum (0.6511). Rank is still 9/11 val, 8/11 test. App. J's "optimisation failure" sentence needs rewriting. Selected config `k=128, lr=0.5, rtol=0.3`, 5/5, optimism +0.0 %, grid 39 points.
3. T1 measures **135 pt ≈ 0.28 pp**, under the 0.40 pp budgeted in §6.
4. nanoGPT and both CIFAR macros are deferred to `numbers_v2_large.tex` by design (`MACRO_DEFERRED`), so the abstract's AdamW-tie numbers live there; `write_numbers_index` re-checks the split every build.
5. Cosmetic: T1's span rows read "Random Polynomial" (title case) beside "MNIST (label reg.)" — comes from `hl.scan_title`/`style.DATASET_TITLES`, shared, left alone.
6. `--check` always reports figure PDFs as changed (creation date in the bytes); read the `.tex` lines only.