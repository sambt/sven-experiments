**TRACK launcher — done.** 68 tests green (`68 passed in 22.33s` local, `68 passed in 20.52s` on a CPU job via `campaign/run_cpu_tests.sh`).

## Files created (nothing outside the track touched)
| file | what |
|---|---|
| `/n/home/anon/sven-experiments/tools/deploy_snapshot.sh` | `git archive` both repos at HEAD → `…/sv3_deploy/<sv3sha8>_<svensha8>/` with nested `sven/`, `DEPLOY_INFO.json` in **both** roots (provenance.git_facts never walks up, so sven needs its own), `experiment_results` symlink, `.deploy_complete` written last; idempotent; refuses tracked dirt unless `--allow-dirty` |
| `/n/home/anon/sven-experiments/tools/worker_pool.sh` | one pool per visible GPU (MIG UUIDs, `CUDA_VISIBLE_DEVICES` as index-or-UUID filter), walks items in order, NPROC runners per item, snapshot guard, SIGTERM forwarding, per-process logs under `slurm_logs/campaign/<label>.<jobid>/` |
| `/n/home/anon/sven-experiments/tools/campaign_plan.py` | plan schema + validation (shared by launcher and reconcile) |
| `/n/home/anon/sven-experiments/tools/launch_campaign.py` | plan → sbatch; dry run by default; lanes, mig cap via `squeue`, `--chain` |
| `/n/home/anon/sven-experiments/tools/reconcile.py` | C-R2; torch-free, `grid.py` path-loaded, 4 listdirs, per-family classes, never-started list, best-config + grid-edge report |
| `/n/home/anon/sven-experiments/campaign/plan_campaign.yaml` | 14 work lists, P0–P3 (already committed by the orchestrator as 6fd8fd9; my later edits are uncommitted) |
| `/n/home/anon/sven-experiments/tests/test_tools_{reconcile,launch,worker_pool}.py` | 13 + 30 + 25 tests |

## Verified by running
* **Real snapshot**: `tools/deploy_snapshot.sh --allow-dirty` → `…/sv3_deploy/6fd8fd9e_203a4e61` (17 MB, 15 s), `WARNING: tools/worker_pool.sh is not in this snapshot (not committed yet)`, second call `already exported`. Pool guard against it with the **real** venv: `resolves experiments = …/6fd8fd9e_203a4e61/experiments/__init__.py`, `resolves sven = …/6fd8fd9e_203a4e61/sven/sven/__init__.py` — the editable `sven` install (which points at the live tree) loses to PYTHONPATH because its finder is *appended* to `sys.meta_path`.
* **Counts agree with `campaign/grid_counts.md` scan-by-scan** (new test; it caught **SOAP missing from my CIFAR baselines**, now item 7/8). Dry runs: `P0/a100 12 job(s), 6220 runs`, `P0/mig 2 job(s), 5000 runs`, `P1 a100 6/8255, mig 2/8240`, `P3 2/1070`, `0 warning(s)` everywhere.
* `reconcile.py --all plan --phase P0 --no-best` → 7 scans in 3.2 s, exit 1, totals 1080/1200/1360/1360/475/535/140 (= the launcher's 6150+70). On the frozen legacy root: 734 jsonl → `jsonl-only 679`, `never-started 221`, best config per method incl. `SVD 2.967e-06 5/5 lr0.1 k16 rtol0.001`; with a shrunken grid it prints `k:EDGE-HIGH lr:EDGE-LOW rtol:EDGE-HIGH`.
* Whole-plan selection is refused: `ERROR: lane 'mig' (gpu_test) allows 2 queued job(s); 0 already queued and this selection wants 6`.

## Deviations from CONTRACTS.md
1. **Dirty = tracked changes only** for the refusal (untracked files are never archived → note, not error); `DEPLOY_INFO.json` still records `git_dirty` with provenance's own (untracked-inclusive) meaning, plus `git_dirty_tracked`.
2. **`--chain` forces n_jobs=1** and the chain retries `sbatch` 10×60 s: MaxSubmit=2 counts the still-running job, so a chain can only renew into a free slot (documented in the tool and the plan).
3. Pool adds `hydra.run.dir=<logdir>/…` so nothing writes into the shared snapshot (`WORKER_HYDRA_RUN_DIR=` restores hydra's default).
4. Reconcile exit codes are **0 / 1 (work remains) / 2 (cannot answer)**; the chain renews on 1 only. Extra classes beyond the spec: `stale-hash`, `jsonl-only`, `claimed-live`.
5. NPROC inferred (not in the probe): MLP JD 4/6, CIFAR LBFGS **2**, CIFAR SOAP 2, fine-tune Sven 1 — all marked `# inferred` in the plan.
6. `exp_finetune_cifar_smallN` items are present but `enabled: false` per the 21:30 user decision (extension phase); `p2_extension_round`, `p3_diagnostics_confirmation` are empty stubs; `p3_cifar_jd_hig` parked.

## For the orchestrator (decisions)
1. **Commit `tools/`, then re-run `deploy_snapshot.sh`** — jobs run `$SNAP/tools/worker_pool.sh`, which today's snapshot lacks.
2. **CIFAR JD/HIG** (20+80 runs per scan, never run, HIG ≈ Sven cost ⇒ ~50 GPU-h/scan): parked in `p3_cifar_jd_hig`, needs a yes/no.
3. **Polynomial data-seed replicates are impossible as configured**: `data_seed` is not in the run_id, so two seeds collide on one result file — needs `data_seed` in `result_id_fields` or its own config.
4. Fig-5 CIFAR set points must be re-derived from the reconciled headline scan before `p1_cifar_fig5` is launched.
5. n_jobs are guesses (P0 = 12 a100 + 2 mig); say if you want more/fewer A100 jobs.

## For the other tracks
* Integrator: `scheduler=claims` is passed unconditionally (`WORKER_SCHEDULER_OVERRIDE=` disables it for Gate-1 before the key exists); `SV3_RESULTS_ROOT` is exported *and* cwd = snapshot, so the hard-coded relative `"experiment_results"` still lands in the real root either way. Reconcile assumes `mark_done` also fires for `oom`/`error` and that `started/` is cleared on success.
* Configs: reconcile/launcher read the **live** configs, so counts follow your edits; keep `grid_counts.md` in sync or my coverage test fails. Any optimizer you add to a config but not to my plan's `_overrides` anchors will be *unlaunched* — that test is what catches it.