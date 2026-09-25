**Files created/changed**

| path | what |
|---|---|
| `/n/home/anon/sven-experiments/iclr_manuscript/sections_v2/app_exp_details.tex` | finalised: campaign totals now macro-driven, seed-count and confirmation-scope claims corrected, compute paragraph rewritten from measured records, ablation-table caption disambiguated |
| `/n/home/anon/sven-experiments/iclr_manuscript/sections_v2/app_reproducibility.tex` | finalised: failure-record wording (275 of 1,422 divergences carry no step), added the exported-snapshot / commit-hash fact |
| `/n/home/anon/sven-experiments/analysis/paper_assets/campaign.py` | **new** generator (module protocol `build()`, also `python -m paper_assets.campaign`); reads the first line of every record, emits 11 macros |
| `/n/home/anon/sven-experiments/iclr_manuscript/numbers_v2_campaign.tex` | **generated** (+ provenance sidecar in `agent_lab/paper_assets/provenance/`) |

**Verified by running**

- `expand_grid()` on 20 scan configs: every value in Tables `tab:sven-grids`, `tab:sven-ablation-grids`, `tab:baseline-grids` matches (Sven 900/720/640/800/90/150/20/5; L-BFGS `max_iter{1,2,3}`×`history_size{2,5,10}`; HIG τ 6 values, lr 7 on 1D / 5 elsewhere; JD 6 lrs; standard lr lists 8/10/8/8/7/9/6/6). `rebuttal_overparam_{toy_1d,polynomial}` cannot expand outside Hydra (`batch_size: ${n_data}`); their counts come from `EXPERIMENTS.md` §3.1 arithmetic and reconcile.
- `optim_factory.py` / `soap.py` / `polyak.py` / `hig.py`: AdamW 0.01, MuonW 0.1 (0.01 on its AdamW group), SGDm 0.9, SOAP β=(0.95,0.95) every 10 steps, Shampoo freq 1 ε=1e-4, K-FAC SGDm base damping 3e-3 factors every step, Polyak `(L−f*)/(‖∇L‖²+ε)` capped at 1, HIG `S.pow(-0.5)` above `τ·S_max`, Muon grouping + `match_rms_adamw`. All as printed.
- `all_datasets.py`: 210 monomials, `E[m²]=Π(2d_j−1)!!`, pool-standardised targets, 50k/10k/10k, 45k/5k/10k, Shakespeare **vocab 65** (counted from the corpus), FineWeb 50,304/block 1024. Record fields confirm `n_params` 593/673/27,562/11,181,642/826,368/163,109,376, `steps_per_epoch` 312/781/351/108/13,125, `eval_batch_size` 2048/256/16, `train_eval_size` 10k/1k/200.
- **Compute, computed from 24,894 records**: 23,304 grid runs on 22 grids + 1,590 in 21 passes; **1,567 GPU-h** = 955 partitioned + 612 whole-device; two device types only; MIG confined to the MLP families (every CIFAR/nanoGPT/GPT-2 run whole-device); GPT-2 29 runs = **124 GPU-h** vs CIFAR-CE 785 runs = **123** (7.9% each).
- LaTeX: both fragments compile standalone (probe in scratchpad, `TEXINPUTS` → `iclr_manuscript`): **0 errors, 0 undefined refs/citations, 0 overfull boxes**, ~6.5 pages. Macro guard tested with the generated file both present and absent.

**Open issues**

1. **This cluster's TeX Live 2018 cannot build the manuscript**: `algpseudocode.sty`, `bbm.sty`, `nicefrac.sty` are absent — `iclr2026_conference.tex` itself fails at line 21. The repo's `.log` is from a co-author's MiKTeX. PAPER_CONTRACTS' build gate needs these installed (or `algorithmic` substituted).
2. `'campaign'` is not in `paper_assets.common.MODULES` (not my file). Until it is, the fragment self-inputs `numbers_v2_campaign.tex` under `\ifdefined…\IfFileExists`. Add the one token and those six guard lines plus the `\providecommand` fallbacks can be deleted.
3. `\numCampaignGptTwoGpuHours` (124) duplicates `\numGptTwoTotalGpuH` in `numbers_v2_large.tex` — same value, different owner; pick one.
4. Refresh state: CIFAR-CE extension complete (785 records), Fig-5 re-run landed (30 = 15 legacy + 15 selected; the table row is the **selected** k=128/lr=0.5/rtol=1e-3). Any later refresh: re-run `paper_assets.campaign`; nothing else in these fragments is result-dependent.

**Integrator notes.** Preamble must supply `\new`, `newtext`, `\input{numbers_v2}`. Labels owed by other fragments: `app:gram`, `app:overparam`, `app:transformers`, `app:robustness`, `app:spectra`. `\svnfittable`/`\svntblbox` are defined idempotently in both fragments. Black text is byte-identical to `iclr2026_conference.tex` lines 604–642; the replaced A100/one-day sentence is kept as an `% OLD:` comment.