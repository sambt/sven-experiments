Final state: 4 new files, all tests green, nothing outside my track touched.

## Files created
- `/n/home11/sambt/iaifi/sv3/experiments/experiment_code/claims.py` — stdlib-only (no torch/hydra/numpy, 0.19 s import)
- `/n/home11/sambt/iaifi/sv3/experiments/experiment_code/provenance.py` — torch imported lazily inside `torch_facts()`
- `/n/home11/sambt/iaifi/sv3/tests/test_claims.py`, `/n/home11/sambt/iaifi/sv3/tests/test_provenance.py` — both load the modules by path (the package `__init__` imports torch)

## Acceptance tests (observed summaries)
`.venv/bin/python -m pytest tests/test_claims.py tests/test_provenance.py -q` → **`21 passed, 1 skipped in 4.93s`** (locally; 3x on a CPU job via `campaign/run_cpu_tests.sh`: `21 passed, 1 skipped` in 6.05/4.79/4.74 s). With the Lustre probe enabled: **`22 passed in 28.47s`**.

| test | asserts |
|---|---|
| `test_eight_process_race_claims_each_run_once` | 8 forked workers, 2,000 run_ids, shuffled orders, released simultaneously by a barrier: union of wins = all ids, no id twice, 2,000 claim files, ≥2 workers won. **local tmp: 43,986 try_claim/s, 5,498 claims/s** |
| `test_eight_process_race_on_lustre` (skipped unless `SV3_CLAIMS_LUSTRE_ROOT` is set) | same race on `/n/holystore01/.../sv3_campaign_scratch/claims_probe/`: **1,236 try_claim/s, 155 claims/s** (2,000 claims in 12.9 s). Exactly-once held. Dir created, race dir removed by the test, probe dir `rmdir`ed — no leftovers. 15k campaign claims ≈ 2 min of aggregate metadata work |
| `test_stale_takeover_has_exactly_one_winner` | 4-process race on one backdated claim: exactly 1 winner, `took_over`, `restarts==1`, `took_over_from.pid` = dead holder, live claim not re-claimable, a second death escalates to gen 2 / `restarts==2` |
| `test_stale_takeover_race_over_many_runs` | 4 workers over **600** stale claims at once: no double takeover, none missed, on disk exactly gen 0 + gen 1 per run and never a gen 2 |
| `test_release_frees_the_claim`, `test_release_does_not_drop_a_claim_taken_over_by_somebody_else` | release is idempotent; a hung worker that wakes up cannot delete its successor's claim |
| `test_heartbeat_keeps_a_claim_fresh` | 1.5 s at interval 0.1 with timeout 1.0: claim stays un-takeable, `beats≥3`; after `stop()` it ages out and the next `try_claim` takes over |
| `test_done_index_is_one_listdir` | `os.listdir` monkeypatched and counted: exactly **1** call for the whole grid |
| `test_done_index_status_and_hash_semantics` | run_ids containing dots parse; `is_done` true only for ok/diverged (oom/error retried), false for the same run under another hash; `stale_hashes` finds the other generation; non-marker files ignored |
| `test_move_to_stale_moves_every_artefact` / `_tolerates_missing_files` | jsonl + diag npz + ckpt + that hash's done marker all land under `_stale/{old}/` with layout and contents preserved; current-generation marker and other runs untouched; a repeat move gets `.1` suffixes (nothing overwritten); missing files → `[]` |
| `test_started_marker_lifecycle` | marker written/removed, holds host/pid/run_id/start_time; "started ∖ done-index" isolates exactly the crashed run |
| `test_manifest_union` | union over two overlapping jobs, `n_runs`/`run_ids`/`job`/`written_at`, `/` sanitised out of the filename, rewrite replaces a job's manifest |
| `test_claim_identity_with_and_without_slurm` | `slurm_job_id None`/`restarts 0` off SLURM; `SLURM_JOB_ID`/`SLURM_RESTART_COUNT` picked up |
| provenance (8 tests) | both repos' 40-hex sha + bool dirty + `source=="git"` in the dev tree and they differ; real dirty flag flips on an edit in a scratch repo; `DEPLOY_INFO.json` fallback incl. `sha`/`dirty` aliases; **no `.git` and no sidecar → all `None`, and notably `git_facts(REPO/"campaign") == (None,None,None)` (no walking up into the enclosing checkout)**; a present `.git` beats a stale sidecar; torch/CUDA/GPU from a stub module incl. CPU-node and driver-error paths; start/end stamps ISO-parseable with `wall_time_s` |

Real-torch check on a CPU job (not in the suite): `torch_version 2.9.1+cu128, cuda_version 12.8, gpu_name null, slurm_job_id 47048926, git_sha 58fc8e41…, git_dirty true, sven_git_sha ca8742bc…, sven_git_dirty true`.

## Deviations from CONTRACTS.md
1. **Takeover is by claim *generation*, not by renaming the stale claim.** CONTRACTS/the task suggested "e.g. atomic rename … before re-claiming". I implemented that first and my own 4-process test caught it failing: `os.rename` is atomic but **not conditional on which file sits at the path**, so between worker B's "this claim is stale" and its rename, winner A can already have re-created a *fresh* claim there — B renames a live lock away and both run the spec. Measured on the rename version with the 4×300 race: `301 wins for 300 stale claims, 1 double-takeover` (1 of 3 trials; scratch repro deleted). Now the lock is the **highest existing generation** of `{run_id}.claim` → `.claim.1` → `.claim.2`, and a takeover is a single `O_EXCL` create of the next generation: no TOCTOU window at all, no renames, no unlinking of another worker's file. Stale generations stay behind as the record of how many workers died. `MAX_CLAIM_GENERATIONS=64` then leaves a poison run to reconcile.
2. `release()` is **ownership-checked** (our identity in the file *and* no higher generation exists), so a hung worker that wakes up cannot unlock a run another worker is running.
3. Scout `sharding.md` §4 risk (1) suggested reclaiming via `squeue -j` rather than mtime. Kept mtime + heartbeat as CONTRACTS specifies — it is the only mechanism that works off SLURM — but the claim records the SLURM job id so reconcile can cross-check.
4. Additions (not in the letter of the contract): `shard_id` alongside `n_shards` in `collect()`; `include_torch=False` (tests and any torch-free caller); `start_time_iso` in claim info; `python_version`/`collected_at` in provenance.

## Notes for the integrator
Signatures added (all paths are relative to `scan_dir`):

```
claims: claim_path/started_path/done_path/manifest_path/stale_dir(scan_dir, ...)
        claim_info(**extra) -> dict(host,pid,slurm_job_id,restarts,start_time,start_time_iso)
        try_claim(scan_dir, run_id, info=None, *, timeout=600.0, max_generations=64) -> Claim|None
        Claim(.path,.base,.gen,.run_id,.info,.took_over; .touch(), .release(), .heartbeat(interval))
        release(claim_or_path); Heartbeat(claim, interval=60.0)  # context manager, .beats
        mark_done(scan_dir, run_id, hash8, status) -> path      # status in STATUSES, write LAST
        done_index(scan_dir) -> {run_id: {hash8: {status,...}}}  # ONE listdir
        is_done(index, run_id, hash8); stale_hashes(index, run_id, hash8) -> [hash8]
        move_to_stale(scan_dir, run_id, old_hash8) -> [moved paths]
        mark_started/clear_started(scan_dir, run_id); started_ids(scan_dir) -> set
        write_manifest(scan_dir, job_name, run_ids) -> path; read_manifest_union(scan_dir) -> set
        constants: CLAIM_TIMEOUT_S, HEARTBEAT_INTERVAL_S, STATUSES, DONE_STATUSES,
                   ARTEFACT_TEMPLATES (ckpt/{run_id}.pt, diag/{run_id}.npz, {run_id}.jsonl)
provenance: collect(repo_root, sven_root, *, n_shards=None, shard_id=None, include_torch=True) -> dict
            git_facts(root) -> (sha, dirty, source); torch_facts(); slurm_job_id()
            now_iso(); start_stamp() -> dict; end_stamp(start) -> dict  # adds end_*, wall_time_s
```

Call sites in files I do **not** own:
- `generic_scan.py:618` — replace `os.path.exists(scan_dir/run_id+".jsonl")` with, once per process: `index = claims.done_index(scan_dir)`, `union = claims.write_manifest(scan_dir, job_name, [s.run_id for s in mine])`, `prov = provenance.collect(REPO, REPO/"sven", n_shards=n_shards, shard_id=shard_id)`; per spec: `h8 = grid.hash8(spec, rcfg)`; skip if `claims.is_done(index, run_id, h8)`; else for each `h` in `claims.stale_hashes(index, run_id, h8)` call `claims.move_to_stale(scan_dir, run_id, h)`; then `claim = claims.try_claim(scan_dir, run_id)` — `None` means another worker has it, `continue`. Wrap the run in `with claims.Heartbeat(claim):`, `try: claims.mark_started(...)` … `finally: claims.clear_started(...); claim.release()`. `mark_done` must be the **last** write, after ckpt/npz/jsonl, and must also fire for `oom`/`error` (so reconcile can tell "attempted and failed" from "never started"). `record.update(prov)` + `record.update(provenance.end_stamp(t0))` with `t0 = provenance.start_stamp()`.
- Static shards stay as the fallback: `grid.shard(...)` for the worker's slice, dynamic queue = walk the full list.
- **Existing results have no `done/` dir**, so `done_index` is empty and everything re-runs. That matches §4.1 (fresh results root); if any legacy scan must be resumed in place, back-fill markers from the jsonl files first.
- `MAX_CLAIM_GENERATIONS` exhaustion and `try_claim` returning `None` are both silent by design — `tools/reconcile.py` (WP-E) is the place that must report "manifest ∖ done ∖ started" and runs whose claims dir shows many generations.
- `DEPLOY_INFO.json` schema the export/launcher track must write into each snapshot root: `{"git_sha": "<40 hex>", "git_dirty": false, "exported_at": "<iso>"}` (`sha`/`dirty` accepted).