# Campaign development contracts (2026-09-18)

Binding for every development track. The spec is `CHANGES_NEEDED.md`; evidence in `FABLE_CRITIQUES.md`;
scout reports with exact insertion points in `campaign/scout/*.md`. Where this file and the spec disagree,
**this file wins** (it records corrections found by the scouts and decisions the user made on 2026-09-18).

## Ground rules
* Deadline is 5 days. Build the simplest thing that satisfies the acceptance test. No speculative generality.
* Branch `robustness-campaign` exists in BOTH repos (`sv3` and the nested, separate repo `sven/`). **Do not
  commit, stash, reset, checkout or switch branches** — the orchestrator commits. Other agents are editing
  other files in the same working tree at the same time: touch ONLY the files your track owns. If you need a
  change in a file you do not own, write it down in your report instead.
* Never touch `experiment_results/` (symlink to irreplaceable results) and never submit/cancel SLURM jobs,
  except through `campaign/run_cpu_tests.sh` (CPU test jobs) or where your track says otherwise.
* Python: `/n/home11/sambt/iaifi/sv3/.venv/bin/python`. The dev node has 4 cores / 16 GB shared by ~10 agents
  and `import torch` takes 1-2 min cold. Run small test files locally (`-x -q`, one file at a time); run
  anything heavy via `campaign/run_cpu_tests.sh <cmd...>` (sbatch --wait on a CPU partition, prints the log).
* Tests: sv3 tests go in `tests/` (pytest, CPU-only, float64 where exactness is asserted, no dataset
  downloads, each file < 60 s). sven tests go in `sven/tests/test_torch_*.py`.
* Match the surrounding code style and comment density. Keep legacy behaviour reachable where the spec says so.
* Final report (your return value): files created/changed, what each acceptance test asserts and its result
  (paste the pytest summary line), every deviation from this contract, and open issues for the integrator.

## Decisions (user, 2026-09-18)
* Polynomial (C-D1): all monomials of total degree <= d incl. constant and linear (210 for d=4, 6 vars),
  factors MULTIPLIED, x ~ N(0,1), coefficient of monomial m = N(0,1) / sqrt(E[m(x)^2]) with
  E[m^2] = prod_j (2 p_j - 1)!!  ("variance-normalised monomials"). Old generator kept as `AdditiveCubicDataset`.
* CIFAR: 5 headline seeds; kappa / param-fraction ablations stay 1 seed; Fig-5 3 seeds.
* BatchNorm (C-E2): `bn_mode: batch | frozen`. `batch` = train with batch statistics, running stats updated
  from the TRAINING batch exactly once per optimizer step, evaluation uses running stats and mutates nothing.
  `frozen` = norm layers in eval mode always (train and eval) for EVERY optimizer (fine-tune study; O2).
* Suppressing running-stat writes is done by temporarily setting `track_running_stats=False` on norm modules
  (train-mode forward then still normalises with batch statistics and writes no buffer). NOT by `.eval()`,
  which would change the normalisation and hence the Gram matrix. Save/restore per-module flags; never call a
  blanket `.train()` on restore.
* Sven wrappers under `bn_mode=batch`: every capture / jvp / delta pass runs under the no-write context and
  the step performs ONE explicit `torch.no_grad()` train-mode forward of the training batch that updates the
  stats (skipped entirely when the model has no norm layer with running stats).
* `wrapper.evaluate()` = eval mode, side-effect-free. `evaluate_and_loss()` (used only by the `variable_k`
  line search) = train-mode normalisation, no buffer write.
* Dedup (C-R3): `run_id` stays human-readable. A zero-byte marker `{scan}/done/{run_id}.{hash8}.{status}` is
  written LAST (after ckpt, npz, jsonl). Skip iff a marker with the current hash8 and status in {ok, diverged}
  exists. A marker with another hash8 => move that run's jsonl/npz/ckpt/marker to `{scan}/_stale/{old_hash8}/`
  and run. One `os.listdir(done/)` per process start; no JSON reads for dedup.
* Scheduling: dynamic claim files replace static shards as the default; `+n_shards/+shard_id` slicing of the
  spec list stays as a fallback. Claim = `os.open(O_CREAT|O_EXCL)` of `{scan}/claims/{run_id}.claim` containing
  host, pid, SLURM job id, start time; a heartbeat thread touches it every 60 s; a claim whose mtime is older
  than 10 min is stale and may be taken over (works with and without SLURM, e.g. on a rented pod).
  This is distinct from the C-R1 `{run_id}.started` marker.
* Results root: env var `SV3_RESULTS_ROOT` (default `experiment_results`), honoured by the runner and analysis.
* `schema_version: 2` on every new record.

## Interfaces
**`experiments/experiment_code/grid.py`** (torch-free, importable in < 1 s):
`RunSpec` frozen dataclass: `family` in {svd, standard, lbfgs, polyak, jd, hig}; `run_id`; `model_seed`;
`loader_seed`; `batch_size`; `hparams: dict` (named grid point); `record_extra: dict`. `expand_grid(rcfg, ...)
-> list[RunSpec]`, seed-major, same order and byte-identical run_ids as the legacy loops. `run_hash(spec,
resolved_cfg) -> str` (sha256 hex; hash8 = first 8 chars).

**Training loops (`experiment_utils.py`)** — new keyword arguments, all optional with defaults that keep the
old call sites working: `losses=None` (caller-owned dict, filled in place so partial curves survive an
exception), `test_loader=None`, `train_eval_loader=None`, `eval_every_steps=None`, `checkpointer=None`,
`log_schedule=None` (callable step -> bool; svd loop sets `optimizer.log_this_step`), `stop_on_nonfinite=True`,
`bn_mode="batch"`. A non-finite training batch loss raises `DivergedError(step)` (defined in
`experiment_utils.py`). Curve keys: `train` (example-weighted online loss), `val`, `val_acc`, `test`,
`test_acc`, `train_eval` (index 0 = untrained), `epoch_times`, `train_times` (sum of synchronised batch times),
step-based evals in `val_step` / `test_step` with indices `eval_step_idx`. Per-batch keys keep their names.
`evaluate(forward_fn, per_sample_loss_fn, loader, device, *, track_acc=False, is_lm=False) -> dict(loss, acc, n)`
is example-weighted (token-weighted for lm), eval-mode, restores the previous mode, mutates no buffer.

**Checkpointer (`experiments/experiment_code/checkpointing.py`)**: `Checkpointer(path, policy, steps_per_epoch,
num_epochs, rewrite_each_epoch=False)`; `.maybe_save(step, epoch, module)` (called before the update of
`step`, and step 0 = init); `.epoch_end(epoch, step, module)`; `.flush()`. File = `torch.save({"step": [...],
"epoch": [...], "state": [state_dict on CPU, fp32, weights AND buffers]})`. Policies none|final|epochs|log
(log = steps {0,1,2,4,8,...} + every epoch end).

**Sampler (`experiments/experiment_code/sampler.py`)**: `EpochPermutationSampler(n, loader_seed, batch_size,
drop_last=True)` with `set_epoch(e)`; epoch permutation = `torch.randperm(n, generator=Generator().manual_seed(
mix(loader_seed, e)))`; `batch_indices(n, loader_seed, batch_size, step) -> LongTensor` reconstructs any batch
offline. `derive_loader_seed(base_loader_seed, model_seed)` (identical across optimizers for a model seed).
`seed_for_run(model_seed, run_id) = model_seed ^ zlib.crc32(run_id.encode())` (C-S1).

**Sven optimizer logging (C-L1)**: attribute `optimizer.log_this_step: bool` (default True = legacy behaviour).
On logged steps append to `svd_info`: `step`, `svs` (all M = B/microbatch values before the k/rtol cut), `utr`
(= U^T r, all M), `update_norm`, `resid_norm`, `sv_min_kept`. `num_nonzero_svs` stays per step. Constructor
flag `empty_cache: bool = True` (C-T3). The optimizer keeps its own step counter.

**Datasets**: every class exposes `train_dataset`, `val_dataset`, `test_dataset` and records `split_seed`.
MNIST 50k/10k + official test; CIFAR-10 45k/5k + official test; Shakespeare contiguous 80/10/10; token bins
val/test/train from disjoint documents; toy/polynomial: fixed pool of 10,000 + val + test from SEPARATE
generators, targets normalised by pool mean/std, `n_train` subsamples the pool.

## Scope update (user, 2026-09-18 evening) — overrides anything above and in CHANGES_NEEDED.md
* Cluster only (no rented GPUs). Headline scans first.
* **No weight-decay grid (C-B4 dropped):** AdamW runs at the torch default wd 0.01 and MuonW at 0.1, one setting
  each, in EVERY scan. Configs that sweep `weight_decays` (overparam, batch-size: `[0.0, 0.01]`) go back to the
  single default; Adam (wd 0) and plain Muon (wd 0) stay as separate named baselines.
* CUT from the campaign: `exp_critbatch_mnist`, `exp_critbatch_nanogpt` (and C-X2), `exp_gpt2_small_comparison`
  (deferred; in-flight jobs cancelled), `cifar10_resnet_kappaScan_labelReg`, `cifar10_resnet_ce_kappaScan`,
  `cifar10_resnet_paramFrac_scan_labelReg`, `cifar10_resnet_ce_paramFrac_scan` (Fig-5 at 3 seeds replaces them),
  `mnist_scan_brier`, the second grid-extension round, confirmation seeds beyond the headline scans.
  `eval_every_steps` (C-E4) and `analysis/ckpt_tools.py` (C-L4) are off the critical path.
* KEPT, in launch order: (P0) six headline scans + `exp_nanogpt_speedrun`; (P1) `rebuttal_overparam_*`,
  `rebuttal_fig5_cifar_paramfrac_scan`, `rebuttal_batchsize_polynomial_scan` (LBFGS grid cut per O5);
  (P2) standalone timing of headline best configs, one extension round on headline scans; (P3)
  `mnist_kappaScan_labelRegression` with per-kappa lr retune (C-X1), micro-batch / param-fraction MLP scans
  (toy + MNIST label-reg first), `exp_finetune_cifar_smallN`, diagnostics + confirmation seeds for headline
  scans, polynomial data-seed replicates.
* **`empty_cache` default is now `False`** (probe 2026-09-18: the per-step `torch.cuda.empty_cache()` made CIFAR
  Sven 4.5x slower, 841 -> 187 ms/step, and caused all of its step-time variance). Keep `empty_cache=True`
  reachable for reproducing legacy runs. Launchers export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  CIFAR keeps `gram_capture: full`, NPROC=1.
* MNIST train part is 50,000: `n_train` larger than the available pool must RAISE, never clamp silently; the top
  point of `rebuttal_overparam_mnist_scan` becomes N=50000.
* `evaluate_and_loss()` = train-mode-no-write under `bn_mode=batch`; under `frozen` the frozen decision wins
  (eval-stat normalisation, no write). No campaign config uses `variable_k`.
* Runner records `actual_param_fraction = train_model.mean_actual_param_fraction` for svd-family runs (1.0 unmasked).
