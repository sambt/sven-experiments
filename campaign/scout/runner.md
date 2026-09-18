## 1. `generic_scan.py` — structure (read; grep-verified line numbers)

One function, `scan(cfg)` (`:276-949`). Preamble `:300-394`: mode flags `:304-309`; loss keys + `loss_suffix` + `track_acc`/`is_lm` `:311-323`; `svd_info`/`svd_spectra_every` `:332-335`; `signed_residual` `:340-341`; `scan_name = HydraConfig...job.config_name` `:344`; **`output_dir = "experiment_results"` hard-coded `:347`**; `hparams = process_hparam_config(rcfg)` `:352`; `id_str` `:353`; `seeds`/`loader_seed` `:356-357`; shard counter `:363-369`; `dataset = instantiate(cfg.dataset)` `:372`; **`cfg.model` mutated for LM datasets `:376-380`**.

Outer `for model_seed in seeds` `:382`; per seed `set_seed(model_seed)` `:390`, `init_state = deepcopy(base_model.state_dict())` `:392`, `common = _scan_facts(...)` `:393` (n_params/n_train/n_val only, `:251-261`).

Six copy-pasted grid blocks, fixed order inside the seed loop: **svd** `:399-586`, **standard non-LBFGS** `:604-679`, **LBFGS** `:682-757`, **PolyakSGD** `:760-829`, **jd** `:834-895`, **hig** `:900-947`. Each: `itertools.product(...)` → build `run_id` → `_shard_skip()` → `os.path.exists(scan_dir/run_id+".jsonl")` → `try:` instantiate model, `load_state_dict(init_state)`, build wrapper/optimizer, build 2 DataLoaders, call a train loop, build `result` dict, `_write_run` → `except Exception: print` only (`:583,676,754,826,892,944` — grep-verified, exactly the six the spec names).

**Sharding**: one mutable counter `_run_idx=[0]`; take iff `idx % n_shards == shard_id`, incremented unconditionally *before* dedup `:366-369,486`. Index space is global over (seed, family, product order). `specs[shard_id::n_shards]` is mathematically identical, so C-R2 preserves it — but only for a fixed `mode`, since `run_svd/run_standard/run_jd/run_hig` change what is enumerated (`submit_gpt2.sh:9` shards per-mode). Launcher: `submit_rebuttal_parallel.sh:35-39` forks NPROC python processes with `+n_shards/+shard_id`, one GPU, 8 CPUs, 48 GB, 12 h.

**run_id per family**: all carry `seed_str="_mseed{ms}_lseed{ls}"` `:387` and `loss_suffix`. svd `:466-484` (`svd_bs{B}{id_str}_k{k}_lr_rtol_svd{mode}` + optional `_mb`,`_pf{pf}[_{mask}]`,`_variablek`,`_gram[_bnbatch]`,`_kappa`); std `:619-624`; LBFGS `:692-696`; Polyak `:769-773`; jd `:850-853`; hig `:907`.

**Record**: the six `result` dicts (`:551-577, 656-671, 732-749, 804-821, 876-888, 929-940`) — run_id, optimizer, loss, batch_size, hparams, seeds, `losses` (whole curve dict), `svd_info = getattr(optimizer,"svd_info",{})` `:576`, plus `result_id_fields` echo, plus `common` via `setdefault` in `_write_run` `:231-234`. `_split_diagnostics` `:172-226` pops per-batch keys `_DIAG_LOSS_KEYS :148` into the npz, builds `svd_summary`, and subsamples spectra every `svd_spectra_every` `:215-218`. Write order npz (`{scan}/diag/{run_id}.npz` `:246`) then jsonl `:248` = dedup marker.

## 2. The four loops in `experiment_utils.py` (read)

| loop | line | signature tail | returns | pre-train val | BN / mode | notes |
|---|---|---|---|---|---|---|
| `train_loop_standard` | 173 | `(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device, track_acc, track_param_norm, is_lm)` | `(model, losses)` | inline `:182-196`, **no `.eval()`** | `.train()` `:210`, `.eval()` `:249` | scalar `loss_fn`; closure `:219-229` stores the **last** call (F11); schedule-free toggles `:250-251,268-269` |
| `train_loop_svd` | 298 | same tail | `(model, losses, optimizer)` | inline `:303-321` via `model.evaluate` | **never `.eval()`d** → train-mode BN for train *and* val | whole body in `torch.no_grad()` `:328`; `loss_and_grad` `:337` + `optimizer.step(batch)` `:338` |
| `train_loop_hig` | 424 | `(..., track_acc, track_param_norm)` | `(model, losses)` | `_initial_val` `:408` | same defect | `_finish_losses` `:395` |
| `train_loop_jd` | 469 | `(model, inner_optimizer, aggregator, per_sample_loss_fn, ...)` | `(model, losses)` | `_initial_val` after `model.eval()` `:482` | only correct loop | torchjd backward |

All four create `losses = defaultdict(list)` **locally** and return it at the end → C-R1 forces a signature change in all four. Timers: `time.perf_counter()` around each batch with **no `torch.cuda.synchronize()`** (`:212/237, 334/339, 441/445, 492/500`). Epoch values are `np.mean` over per-batch means `:275-276, 374-375, 462-463, 518-519` (F10). `val_loader` batch size = train batch size everywhere; `eval_batch_size` does not exist anywhere in the repo (grep-verified). `drop_last=(microbatch_size is not None)` only on the svd train loader `:540`. `SvenWrapper.evaluate → _func_call` (`sven_wrapper.py:138-153`) is a `functional_call` in whatever mode the module is in and is *not* wrapped in `GramSvenWrapper._frozen_norm_stats` (`gram_wrapper.py:358-370`) — this is F2/F3 mechanically.

**Verified by reading `_tie_parameters_to_flat` (`sven_wrapper.py:372-389`)**: wrapper parameters are rebound as *views* into the flat `params` vector, so `wrapper.model.state_dict()` does reflect Sven's updates. C-L3 is therefore implementable uniformly as `wrapper.model` / `model`.

## 3. Per-change insertion points

| ID | Where | Size | Difficulties / ambiguity |
|---|---|---|---|
| C-R1 | `execute()` try/except; all four loops must accept a caller-owned `losses` | **L**, ~180 | `optimizer`/`train_model` must be pre-bound to `None` before the `try` or the failure path cannot read `svd_info` `:576`; `_finish_losses :399-401` does `np.mean([])` → NaN/warning on early failure; early-stop check is free (`.item()` already syncs) but for LBFGS the NaN appears inside the line search, so check after `step`; classification of "failed to converge" by message string is brittle |
| C-R2 | replace `:399-947` with `expand_grid` + `execute` | **L**, ~400 moved | per-seed `init_state` and `common` are computed in the seed loop `:390-393`; RunSpec must either carry the seed and let `execute` own an init-state cache, or `expand_grid` must be seed-major (keep seed-major to preserve order) |
| C-R3 | `expand_grid` (hash) + dedup site + one-time job header | **M**, ~120 | hash must be taken **after** the `cfg.model` mutation `:376-380`; "skip only if hash matches" turns dedup from one `stat` into a JSON read per grid point (~14k Lustre reads per 2400-run job) — put an 8-char hash in the filename or a sidecar `{run_id}.hash` instead; `_stale` move must take the npz too |
| C-R4 | `_scan_facts :251-261` + `execute` | **S**, ~30 | `n_test` needs C-E1 first; `actual_param_fraction` lives on the wrapper (`sven_wrapper.py:115`), only defined for svd family |
| C-E1 | `all_datasets.py` (5 of 6 classes lack `test_dataset`; grep-verified only `Toy1D` has one, `:36`) + new `evaluate()` in utils + all four loops | **L**, ~350 | `train_loop_standard` is handed a **scalar** `STANDARD_LOSS_FNS` (`generic_scan.py:98-104`) — example weighting needs per-sample losses, so the runner must pass `SVD_LOSS_FNS[loss_key]` (or `reduction='none'`) to that loop too; `is_multi` (M,B,C) and `is_lm` (B,T,V) paths need their own weighting rule (per-sequence for lm_ce); test evaluation roughly doubles eval cost |
| C-E2 | `sven/nn/gram_wrapper.py:358`, `sven_wrapper.py:151`, `optimizers/hig.py:89`, `experiment_utils.py:182-196` | **M**, ~120 both repos | **Spec is imprecise**: a context that `.eval()`s `_NormBase` changes the *normalisation* (running vs batch stats) and hence the update. The correct primitive is temporarily setting `track_running_stats=False` (train-mode forward then uses batch stats and writes no buffers). `bn_mode` in the run_id changes every CIFAR run_id |
| C-E3 | closure `:219-229` → record first call; new `train_eval` pass | **S/M**, ~60 | fixed 10k train subset must be split-seed-derived and shared across optimizers |
| C-E4 | inner batch loop of all four loops | **M**, ~100 | mid-epoch eval must toggle `optimizer.eval()/train()` for schedule-free (`:250,268`) and `model.eval()/train()` for jd, and must be excluded from `epoch_times`/`train_times` |
| C-E5 | `_finish_losses` / record build | **S**, ~20 | none |
| C-S1 | `execute()`, right after `load_state_dict(init_state)` | **S**, ~15 | changes every masked/randomized result; must be after mask construction ordering is decided (`_make_param_mask` is called in the wrapper ctor, `sven_wrapper.py:354`) so the seed must be set *before* wrapper construction, not after |
| C-S2 | loader construction (6 sites today, 1 after C-R2) | **M**, ~60 | deriving `loader_seed` from `model_seed` invalidates the current `seed_str`/run_id convention (`_lseed{ls}`) — decide whether `_lseed` stays in the id |
| C-S3 | same site | **S**, ~5 | changes baseline step counts → all baselines rerun |
| C-L2 | `_split_diagnostics :172-226`, `svd_summary` | **M**, ~80 | with sparse scheduled logging, `sv_max`/`sv_min` derived at `:210-212` lose per-step alignment; they need their own step-index array, which the spec does not state. `num_nonzero_svs` stays per-step |
| C-L3 | new `checkpointing.py` + all four loops + write order in `_write_run :229-248` | **M/L**, ~200 | rewrite-per-epoch of one file is a whole-file rewrite (45 MB for ResNet) — acceptable but do it on epoch end only; failure path must flush |
| C-T1 | 8 timer sites in the four loops | **S**, ~30 | adds a sync per batch; for toy/poly (sub-ms steps) this measurably slows the run, so `train_times` and `epoch_times` will disagree with legacy numbers |
| C-B1/B2/B4/B5 wiring | `build_standard_optimizer :609-666`, `_DEFAULT_WEIGHT_DECAY :598`, grid filter `:616` | **S/M**, ~80 | `:616` currently *skips* non-zero wd for non-AdamW/Muon — the B4 wd grid works, but `SGDm` needs a registry entry, not `getattr(torch.optim,...)` |
| C-X1/X2 wiring | config-level mostly; `k_fractions` already in the svd product `:404,417-426` | **S**, ~20 | C-X2 needs C-E4 |

## 4. Collisions and work packages

Everything in §3 except C-E1's dataset half, C-L1, C-T2 and the analysis items touches **one of two files**, so parallel developers in the same file is the default failure mode. Concretely: C-R1/R2/R3/R4/S1/S2/S3/L2/L3 all rewrite the same `try` block; C-E1/E2/E3/E4/T1 all rewrite the same inner loops.

**The spec's ordering (C-R2 first) is right** — do not parallelise across it. `expand_grid()` collapses six near-identical 80-line blocks into one, and every later change then lands in exactly one place instead of six. Doing C-R1 or C-T1 first means writing the same edit six times and then deleting five of them.

Proposed RunSpec (frozen dataclass): `family` (`svd|std|lbfgs|polyak|jd|hig`), `run_id`, `run_hash`, `model_seed`, `loader_seed`, `batch_size`, `hparams: dict` (the family's product tuple, named), `record_extra: dict` (the constant fields each family's `result` dict carries today), plus scan-level `loss_key`, `num_epochs`, `eval` settings, `checkpoint` policy, `svd` settings. `execute(spec, ctx)` where `ctx` holds dataset, `cfg.model`, per-seed `init_state` cache, `scan_dir`, `common`, provenance header. Keep `expand_grid` seed-major so the order is bit-identical to today's.

Sequential packages (each a PR, each with its own CPU tests):

| WP | Content | Parallel with |
|---|---|---|
| **A** | C-R2 refactor only, no behaviour change (byte-identical run_ids and shard slices; assert against a dumped legacy grid) | nothing — blocks all of B, C, D |
| **B1** | C-R1 + `.started` markers + dedup status rules | B2, B3 (different files) |
| **B2** | dataset layer of C-E1 (`all_datasets.py`, C-D1, C-D3) + `tools/check_token_split.py` (C-D2) | B1, B3 |
| **B3** | `sven` repo: C-L1, C-E2 wrapper half, C-T3 | B1, B2 |
| **C1** | loops: C-E1 `evaluate()`, C-E2 runner half, C-E3, C-T1 (one author, one file) | C2 |
| **C2** | `execute()`: C-R3, C-R4, C-S1, C-S2, C-S3, results-root config key | C1 |
| **D** | C-L2, C-L3, C-E4, C-E5 (needs C1+C2 merged) | C-B*, C-X* configs; analysis WP E |
| **E** | `tools/reconcile.py`, C-A1–A6, C-L4, C-T2 — `analysis/` only, fully parallel from day 0 | all |

C1 and C2 both edit `experiment_utils.py`/`generic_scan.py` but disjoint regions (loop bodies vs `execute` scaffold); if that is still too hot, serialise C1 → C2.

## 5. Where I think the spec is wrong or risky

* **Sharding order preservation is a non-goal.** §4.1 starts an empty results root, so there is nothing to resume against; the only reason to keep the order is the WP-A equivalence test. Say so, or WP-A will be over-engineered.
* **Static `specs[shard_id::n_shards]` is the wrong scheduler and the spec keeps it.** Cost inside one scan varies ~20x (Sven `k=1` vs `k=B`; LBFGS `max_iter=1` vs `3`), so a 6-shard job finishes 5 shards and then runs one serially for hours — this, not queueing, is what produced the 12 h timeouts. C-R1's `{run_id}.started` marker is already a claim file: make it the claim (`os.open(..., O_CREAT|O_EXCL)`, write host/jobid/timestamp), have every worker iterate the *full* `expand_grid` list and skip ids with a record or a live claim, and reclaim claims older than the wall limit. `execute(spec)` is unchanged; only the driver loop changes, `n_shards`/`shard_id` become optional, and resubmission mops up automatically. This is ~40 LOC on top of C-R1/R2 and should be in WP-B1. O_EXCL create is atomic on Lustre; put claims in `{scan}/claim/` to keep the scan dir's directory listing small.
* **`eval_every_steps` (C-E4) is not drop-in.** Three hazards the spec does not mention: schedule-free optimizers need `optimizer.eval()/train()` around every mid-epoch eval; `train_loop_jd` needs `model.eval()/train()`; and mid-epoch eval time must be excluded from `epoch_times`/`train_times` or the C-T1 timing story breaks. For LBFGS, evaluating between steps is safe (closure re-evaluates), but with `line_search_fn="strong_wolfe"` step counts are not fixed, so "step multiples" must mean optimizer-step count, recorded explicitly.
* **C-E2's prescription (`.eval()` on norm layers) would change the algorithm**, not just the bookkeeping — see §3. Use `track_running_stats=False` for the "repeated forward" context, and reserve `.eval()` for actual evaluation.
* **C-R3's hash-based dedup as written is a Lustre hazard** (one JSON read per grid point per job start). Recommend hash-in-filename or a sidecar; this is a decision the plan should make now because it determines whether run_ids stay human-readable.
* **The results-root hard-coding is in seven places, not one.** Verified by grep: `generic_scan.py:347`, `analysis/sv_diagnostics.py:44`, `analysis/scan_analysis.py:35`, `analysis/style.py:47,142,224`, `submit_reruns_2026-09-17.sh:21`, `submit_timing_runs.sh:14` (plus `RESULTS_ROOT` references in the notebooks). §4.1's "make the root a config key" must cover the analysis defaults or the legacy-repair scripts will silently read the new root.
* **C-E1 makes a full rerun unavoidable, and the plan should say it outright.** Only `Toy1DRegressionDataset` has a test split today; MNIST/CIFAR/polynomial/CharText/TokenBin all pass the *official test set* as `val_dataset` (`all_datasets.py:91,168,128,215,252`). Every selection on disk was made on the test set, and after the 50k/10k and 45k/5k resplit the validation distribution, `n_train`, and steps per epoch all change. No existing scan result survives as a selection input.
* **C-L2 vs C-L1 coupling is underspecified**: if the optimizer only logs on scheduled steps, `sv_max`/`sv_min`/`utr` become sparse series and need step indices; `num_nonzero_svs` stays dense. Decide the npz schema (one `*_step` array per density class) before WP-B3 and WP-D start, or they will disagree.
* **C-R1 + `svd_info` lifetime**: `svd_info` is read off the optimizer *after* the loop returns (`:576`). On failure the optimizer may not exist yet; also for a run killed by the 12 h limit nothing is written at all. The `.started` marker plus C-L3's per-epoch checkpoint rewrite covers this only if the partial `losses`/`svd_info` are also flushed on epoch end — consider making C-R1 write a provisional record each epoch (cheap for the jsonl, expensive for the npz), or accept that timeouts lose the curve.
* **C-T1's syncs will change the very numbers §4.3 estimates from.** Toy/polynomial steps are sub-millisecond; a per-batch sync at ~10-20 µs each is a few percent, but `train_times` and legacy `batch_times_train` are then not comparable, which matters for the "expect toy to be close to legacy" gate in Phase 2.