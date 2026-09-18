Both files pass. Final report:

**Files created** (all new, none else touched):
- `/n/home11/sambt/iaifi/sv3/experiments/experiment_code/checkpointing.py`
- `/n/home11/sambt/iaifi/sv3/experiments/experiment_code/sampler.py`
- `/n/home11/sambt/iaifi/sv3/tests/test_checkpointing.py`
- `/n/home11/sambt/iaifi/sv3/tests/test_sampler.py`

**Observed pytest lines** (via `campaign/run_cpu_tests.sh`): `tests/test_checkpointing.py` → `9 passed in 15.99s`; `tests/test_sampler.py` → `9 passed in 6.62s`; together `18 passed in 8.95s`.

**Acceptance tests, what each asserts**
- *Final reload* (`test_final_checkpoint_reloads_and_reproduces_the_recorded_loss`): MLP+BatchNorm trained 3 epochs, `final` policy; fresh differently-initialised model loads the flushed state and its eval-mode val/train losses match the recorded ones to <1e-6 (differ >1e-3 before loading); `running_mean`/`num_batches_tracked` restored exactly.
- *Log step set* (parametrised ×2): `steps_per_epoch=3,epochs=4` → steps `[0,1,2,3,4,6,8,9,12]`, epoch labels `[0,0,0,0,1,1,2,2,3]`; `spe=2,epochs=3` → `[0,1,2,4,6]` (epoch ends coincide with powers of two → dedup). Each on-disk state equals an independent snapshot at that step; `epochs` policy saves epoch ends only.
- *Flush after exception*: `log` policy, `RuntimeError` at step 5 → nothing on disk; `flush()` in the handler writes `[0,1,2,3,4]`, is idempotent; empty checkpointer writes no file; an unwritable path returns `None` instead of raising (cannot mask the original exception).
- *State content*: fp64 model → CPU, float32, cloned (different `data_ptr`), buffers included, `num_batches_tracked` int64 = 4; further training does not mutate the stored copy.
- *Batch reconstruction*: `n=20,bs=6,drop_last=True`, 3 epochs — every batch the `DataLoader(batch_sampler=...)` yields equals `batch_indices(n, loader_seed, bs, step)`.
- *Permutations*: differ across 4 epochs and 4 loader seeds, unaffected by the global RNG, and identical to two subprocesses' values. `seed_for_run` equals `model_seed ^ zlib.crc32(run_id)` and is identical in subprocesses run with `PYTHONHASHSEED=0` and `=4242` (asserted `sys.flags.hash_randomization` actually differed). `drop_last` default drops the tail (3 batches), `False` keeps a 2-example tail.

**Deviations from CONTRACTS.md** (all additive; no interface renamed)
1. The file carries three extra keys beyond `step/epoch/state`: `policy`, `steps_per_epoch`, `num_epochs` — C-L4 needs `steps_per_epoch` to map a step to its batch. That is also the only use of the `steps_per_epoch`/`num_epochs` ctor args; they do not select steps.
2. `load_state_at(path_or_loaded_ckpt, step=None, epoch=None)` also accepts an already-loaded dict; `epoch=e` = **last** state labelled `e` (= end of epoch e), enforced by letting only `epoch_end` relabel a deduplicated step.
3. `save_init_state(path, state_dict, overwrite=False)` writes the same envelope (`step=[0]`) so `load_checkpoint`/`load_state_at` read it unchanged, and skips an existing file.
4. `policy=None` accepted as `"none"`; hooks return bools, `flush()` returns the path or `None`.
5. Unspecified formulas chosen: `mix(ls,e)=crc32(f"{ls}:{e}")`, `derive_loader_seed(b,ms)=b ^ crc32(f"mseed{ms}")`.
6. Extra sampler helpers: `epoch_permutation`, `steps_per_epoch(n,bs,drop_last)`, `set_loader_epoch(loader,epoch)`, `EpochPermutationSampler.batch_indices(step)`, `batch_indices(..., drop_last=True)` kwarg; a `RuntimeWarning` if the sampler is iterated twice at one epoch (guards a forgotten `set_epoch`).
7. `epochs` policy saves epoch ends only (no step 0) — per the literal spec; call `save_init_state` if nanoGPT needs its init.
8. Tests exec both modules by path (`importlib`) instead of importing `experiments.experiment_code`, whose `__init__` pulls in `generic_scan`/sven/hydra.

**For the integrator**
- `generic_scan.py:345-352` (`Ctx.loaders`): replace `shuffle=True, generator=..., drop_last=drop_last` with
`bs = EpochPermutationSampler(len(self.dataset.train_dataset), derive_loader_seed(spec.loader_seed, spec.model_seed), spec.batch_size)` and `DataLoader(train_dataset, batch_sampler=bs)` — do **not** also pass `batch_size/shuffle/sampler/drop_last` (torch raises). `len(train_loader)` is still `steps_per_epoch`. The `drop_last=` parameter of `loaders()` becomes dead (C-S3: always True).
- `experiment_utils.py`: add `set_loader_epoch(train_loader, epoch)` as the first statement inside each epoch loop — lines **610** (standard), **709** (svd), **781** (hig), **842** (jd). Without it every epoch replays epoch 0's order (the warning fires).
- C-S1: `set_seed(seed_for_run(spec.model_seed, spec.run_id))` after `load_state_dict(init_state)` and **before** wrapper construction (`_make_param_mask` runs in the ctor).
- `run_id` keeps the *base* `_lseed{loader_seed}`; the effective seed is derived, so names stay byte-identical (scout's open decision on `grid.py`).
- Checkpointing: `Checkpointer(f"{scan_dir}/ckpt/{run_id}.pt", cfg.checkpoints, len(train_loader), num_epochs, rewrite_each_epoch=<ResNet/GPT>)`; it creates `ckpt/` itself. Call `flush()` **before** the npz and jsonl and again in the `except`/`finally` of `execute`. Under `final` use `save_init_state(f"{scan_dir}/ckpt/init_mseed{model_seed}.pt", init_state)` once per seed. Caveat: `log` holds every state in RAM until flush (~1.6 GB for ResNet18 × 35) — keep `final` for ResNet scans as the storage table says. A run that dies inside epoch 0 under `final` writes no file (nothing collected).