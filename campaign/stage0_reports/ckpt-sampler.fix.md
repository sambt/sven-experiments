**Track ckpt-sampler — fixer report**

**Files changed** (only track-owned): `/n/home/anon/sven-experiments/experiments/experiment_code/checkpointing.py`, `/n/home/anon/sven-experiments/experiments/experiment_code/sampler.py`, `/n/home/anon/sven-experiments/tests/test_checkpointing.py`, `/n/home/anon/sven-experiments/tests/test_sampler.py`. Nothing outside the track imports either module (verified by grep: only docs reference them), so all changes are safe.

**Findings: all five confirmed and fixed, none rejected.**
1. *pid-only temp name* — `_atomic_save` now writes `f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"`. uuid4 draws from `os.urandom`, so cross-node uniqueness needs no hostname. The `save_init_state` TOCTOU is left as is and documented: with per-writer temp files the racing writers write *identical* bytes (one seed, one init), so last-`os.replace`-wins is correct.
2. *flush swallows failures* — kept the swallow (must not mask an in-flight exception) and added `Checkpointer.last_error: str | None` (`repr(exc)` on failure, cleared on a successful write). Did **not** make `flush()` re-raise by default: the integrator would have to remember `quiet=True` in every `except`, reinstating the exact hazard the design avoids. Contract for the integrator below.
3. *no init under `epochs`* — `maybe_save` now records step 0 under `epochs` as well as `log`. Verified: policy `epochs`, spe=3, epochs=2 → steps `[0, 3, 6]`, epochs `[0, 0, 1]` (was `[3, 6]`). Costs one extra state (+3.3 MB of a 115 MB nanoGPT budget) and needs no cross-run coordination.
4. *base-vs-effective loader seed* — added `batch_indices_for_run(n, base_loader_seed, model_seed, batch_size, step, drop_last=True)` and classmethod `EpochPermutationSampler.for_run(n, base_loader_seed, model_seed, batch_size, drop_last=True)`, which stores `.base_loader_seed` / `.model_seed` (both `None` on the plain ctor) so the runner can record the effective seed.
5. *false-positive warning* — added `_epoch_set`; `__iter__` warns only when `not self._epoch_set and self.epoch == 0 and self._last_iterated == 0`, i.e. only for a `set_epoch` never called at all. A deliberate second pass at one epoch (per-epoch train/test eval through the loader) is silent.

**Acceptance tests / observed summary lines.** `tests/test_checkpointing.py` → `10 passed in 4.52s`; `tests/test_sampler.py` → `10 passed in 7.12s`; together → `20 passed in 8.78s`. Unchanged assertions are as in the impl report. Changed/new:
- `test_log_policy_...`: now also asserts `epochs` gives `[0] + epoch ends` with labels `[0] + range(E)` and that state 0 equals an independent step-0 snapshot.
- `test_flush_after_an_exception...`: adds `blocked.last_error is not None`, `checkpointer.last_error is None` after a good write, and that re-pointing `.path` to a writable file clears it.
- `test_atomic_temp_name_is_unique_per_writer` (new): spies `torch.save`, asserts two writes of one path use two distinct temp names, neither equal to the legacy `{path}.tmp.{pid}`, the promoted file loads and equals the last writer's payload, and no `.tmp` residue.
- `test_batch_indices_for_run_derives_the_effective_seed` (new): `for_run().loader_seed == derive_loader_seed(base, ms) != base`; every DataLoader batch over 2 epochs equals `batch_indices_for_run(...)`; the base seed yields a valid but different batch for steps 0–2; different model seeds give different batches.
- `test_warns_only_when_set_epoch_is_never_called` (replaces the old warn test): warns on a second pass without `set_epoch`; under `warnings.simplefilter("error")`, `set_epoch(0)` + two passes + `set_epoch(1)` + two passes raise nothing.

**Deviations from CONTRACTS.md** (additive only; the impl report's 1–6, 8 stand, item 7 is reversed): `epochs` now includes step 0 (spec text said epoch ends only); `Checkpointer.last_error`; `for_run` / `batch_indices_for_run`; narrowed warning condition.

**Integrator notes (in addition to the impl report's).**
- Build the train sampler with `EpochPermutationSampler.for_run(len(train_dataset), spec.loader_seed, spec.model_seed, spec.batch_size)`; `DataLoader(train_dataset, batch_sampler=bs)`, no `batch_size/shuffle/sampler/drop_last`.
- **Requirement:** record `effective_loader_seed = bs.loader_seed` in `record_extra` (schema_version 2) alongside the base seed kept in the run_id. Analysis must reconstruct batches only via `batch_indices_for_run` or the recorded effective seed.
- **Requirement:** `train_eval_loader` must be a *separate* sequential `DataLoader(train_dataset, batch_size=..., shuffle=False)` — never the train loader object (as `tests/test_loops_contract.py:524` does). Reusing it measures the `drop_last`-truncated set in permuted order.
- **Contract:** `flush()` never raises; after it, `if cp.policy != "none" and cp.flush() is None: record_extra["ckpt_error"] = cp.last_error`, else `record_extra["ckpt_path"] = cp.path`. Then analysis can assert every `ok` run has a checkpoint.
- Under `epochs` no `save_init_state` call is needed (step 0 is in-file); `final` still needs `save_init_state(f"{scan_dir}/ckpt/init_mseed{model_seed}.pt", init_state)` once per seed.

Aside for the orchestrator: nothing in this track serializes review — the fixer tracks touch disjoint files, so review/fix agents can run concurrently.