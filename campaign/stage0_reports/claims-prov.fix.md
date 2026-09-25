## Fixer report — track `claims-prov`

**Changed** (nothing else touched): `/n/home11/sambt/iaifi/sv3/experiments/experiment_code/claims.py`, `/n/home11/sambt/iaifi/sv3/tests/test_claims.py`. `provenance.py` / `test_provenance.py` unchanged (no findings against them).

### Findings: all four real, all fixed
Reproduced the reviewer's probe verbatim first (`scratchpad/probe_race.py`): PROBE 1 → 3 of 4 workers died with `FileNotFoundError`; PROBE 2 → `fresh jsonl still present: False`, fresh jsonl/npz/ckpt under `_stale/aaaa1111/...` as `.1` copies, finished run re-claimable.

1. **(high) stale index eats fresh results.** Added `is_done_now(scan_dir, run_id, hash8)` (2 `lexists`, only for runs about to execute) and made the corrected order the module docstring's prescribed recipe: cheap `is_done(index,…)` skip → `try_claim` → `is_done_now` → `move_to_stale` → execute → `mark_done` (last) → `release`. Second line of defence: `move_to_stale` now returns `[]` unless a `done/{run_id}.{old_hash8}.*` marker still exists — that marker is the only proof the hash-free artefact paths still hold the old generation, and it is moved last, so its absence means somebody already retired (and possibly re-ran) that run.
2. **(high) concurrent `move_to_stale` crash.** `_move` returns `None` instead of raising when the source vanished (also inside the EXDEV `shutil.move` fallback); `move_to_stale` appends only non-`None` results.
3. **(medium) manifest overwrite under one `$SLURM_JOB_ID`.** `write_manifest(scan_dir, job_name, run_ids, *, shard_id=None)` → `manifest/{job}.shard{i}.json`; payload carries `shard_id`; rewriting a manifest that *drops* previously declared run_ids prints a `[warn]`. `_safe_name` now appends a 6-hex digest when sanitisation changed the name, so `scan/a` and `scan_a` no longer collide. Deliberately **not** put `$SLURM_JOB_ID` in the name: it would accumulate a manifest per resubmission and make the union include run_ids from grids no longer intended. `analysis/style.py:manifest_run_ids` globs `manifest/*.json`, so the suffix is compatible.
4. **(medium) no composed test.** Added below.

### Acceptance tests (observed)
`.venv/bin/python -m pytest tests/test_claims.py tests/test_provenance.py -q` → **`27 passed, 1 skipped in 5.16s`**; same on a CPU job with `SV3_CLAIMS_LUSTRE_ROOT` set → **`28 passed in 7.14s`** (Lustre race: **20,242 try_claim/s, 2,530 claims/s**, exactly-once held; probe dir created and removed, nothing left on holystore).

New tests, and what each asserts:
- `test_recipe_over_two_generations_with_a_stale_index` — two workers, one process-start index, `hash_old` generation on disk: worker B executes **nothing**, every `{run_id}.jsonl/npz/ckpt` present with `hash_new` content, `_stale/hash_old/` holds exactly 4 files per run, none containing `hash_new`, no `.1` duplicates, claims dir empty, no started markers, `stale_hashes == []`.
- `test_recipe_race_four_workers_over_two_generations` — same recipe, 4 forked workers, shuffled orders, 150 runs: exit 0, each run executed exactly once (`[38,39,37,36]`), same end-state assertions.
- `test_concurrent_move_to_stale_does_not_kill_workers` — 4 workers × 200 stale runs: no exception, exit 0, every artefact exactly once under `_stale/`, nothing left behind.
- `test_move_to_stale_requires_the_old_generations_marker`, `test_is_done_now_rechecks_disk_per_run`, `test_manifest_union_over_workers_of_one_slurm_job` (shared `SLURM_JOB_ID`, disjoint slices, `shard_id=0/1` → union complete; missing `shard_id` warns).

Non-vacuity checked: against a reverted copy, the two `move_to_stale` tests fail with the reviewer's `FileNotFoundError`; with the *old recipe* both composed tests fail (`a finished run was re-executed on a stale index`, 6 extra executions).

### Deviations from CONTRACTS.md
None new. The implementer's two (generation-based takeover instead of renaming; ownership-checked `release`) stand — the generation scheme is what keeps takeover exactly-once (test `…race_over_many_runs`). `is_done_now` and the marker gate are additions, not departures.

### Integrator notes
- New/changed API: `is_done_now(scan_dir, run_id, hash8) -> bool`; `move_to_stale(...)` unchanged signature but **must be called while holding the claim, after `is_done_now`** (else no-op or wrong generation); `write_manifest(scan_dir, job_name, run_ids, *, shard_id=None)`; `manifest_path(scan_dir, job_name, shard_id=None)`.
- `campaign/stage0_reports/claims-prov.impl.md:58` (not my file) prescribes the **broken** order — replace with the recipe in `claims.py`'s docstring: claim first, then `is_done_now`, then `move_to_stale`. `generic_scan.py` must pass `shard_id=<+shard_id>` to `write_manifest` when NPROC>1.
- A second mop-up pass in the same process should call `done_index(scan_dir)` again (one listdir) rather than reuse the snapshot.