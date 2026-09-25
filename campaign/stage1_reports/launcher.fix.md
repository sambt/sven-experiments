**FIXER track "launcher" — all 7 findings accepted and fixed; 87 tests green** (`87 passed in 36.01s` via `campaign/run_cpu_tests.sh`; locally 38+18+30).

## Changed (nothing outside the track)
`/n/home/anon/sven-experiments/tools/launch_campaign.py`, `tools/worker_pool.sh`, `tools/reconcile.py`, `campaign/plan_campaign.yaml`, `tests/test_tools_{launch,reconcile,worker_pool}.py`.

| # | fix |
|---|---|
| 1 | **Snapshot binding.** `snapshot_facts()`/`live_heads()`; header prints both snapshot SHAs next to live HEAD + a warning on mismatch; items file stamped `# snapshot/# sv3/# sven`; `<list>.launched.json` records it; before submitting, `squeue -o "%i %T %j"` + that record refuse a list live from another snapshot (`--force` overrides, dry run only notes it); job names gain the sv3 sha8 (`-J P0.a100.j0.p0_nanogpt.6fd8fd9e`) so a split is visible in `squeue`; the pool logs the items stamp and shouts on mismatch |
| 2 | **`--chain` can now renew.** Confirmed the bash behaviour myself (`fg script exit=143`, no line after the child). Pool gained `--until`/`WORKER_DEADLINE_EPOCH`/`WORKER_TIME_BUDGET`/`$SLURM_JOB_END_TIME` with `WORKER_RESERVE_S=900`: starts no item inside the reserve, exits **0**. Chain script gained `#SBATCH --signal=B:USR1@900`, a USR1 trap, the pool in the **background** + re-`wait`, and `SIGNALLED ⇒ rc=0` |
| 3 | **Config dir follows the snapshot** in both tools (`--snapshot`, `$SV3_SNAPSHOT`), plus `grid.py` from the same tree; printed with its source, per run and per scan |
| 4 | **Chain cap leaves a slot**: `existing+want > max_jobs-1` when any planned job chains |
| 5 | **`resolve_root(root, plan)`**: plan's `results_root` beats `$SV3_RESULTS_ROOT`, with a loud note; root printed |
| 6 | **`best_configs`**: a record whose *configuration* is off the expected grid can never win (counted + reported); `tau` is now an axis; a value not in its grid prints `OFF-GRID` instead of nothing |
| 7 | Coverage split: `test_the_plan_covers_every_run_its_own_configs_describe` (intrinsic — run_id by run_id vs `mode=all` + the plan's `n_data`, with a commented `DELIBERATE_GAPS` set that must still *be* gaps) and `test_grid_counts_md_is_in_sync_with_the_plan` ("grid_counts.md is stale") |

## Observed
- `--phase P0 --lane a100`: **snapshot configs 4640 runs / 10 warnings** (`SGDm not in optimizers_standard` ×4) vs **live 6350 / 0** — finding 3's divergence, now printed, not silent. nanoGPT: snapshot 100 runs vs live 140.
- `--phase P0,P1 --lane mig --chain` → `ERROR: … allows 2 queued job(s), and a chain must leave one free to renew into; … wants 2.` One chain: `0 queued + 1 wanted <= 1 OK (2 - 1 slot kept free…)`.
- Legacy root, grid shrunk to `k_values=[8] lrs=[0.5]`: best SVD is now the on-grid `lr0.5 k8` (9.652e-06) with `rtol:EDGE-HIGH`, `345 record(s) … ignored for 'best'` — previously the off-grid `lr0.1 k16` (2.967e-06) won with no lr/k verdict. Unshrunk legacy output unchanged (`SVD 2.967e-06 5/5 lr0.1 k16 rtol0.001`, HIG `tau 0.0001 (not in the expected grid)`).
- `reconcile --all … --phase P0 --no-best` → `the plan's results_root`; with a foreign `$SV3_RESULTS_ROOT` it says so on stderr. Pool against the real snapshot: guard OK, `wall clock no new item after …`, items stamp echoed.
- Finding 7's asserted failure no longer reproduces: the configs track re-synced `grid_counts.md` (555/655/210), so it was 68→green before I touched it; the design defect was real and is fixed.

## Deviations (additions to the earlier list)
- `PYTHONDONTWRITEBYTECODE=1` and default log root moved to `…/sv3_campaign_scratch/logs` in **both** the pool and `launch_campaign.DEFAULT_LOG_DIR` (home is 95 G NFS at 84%, 17 G free).
- A non-epoch `$SLURM_JOB_END_TIME` warns and disables the wind-down (never fails a job); an explicit bad `--until`/budget is fatal.

## Decisions needed
1. Re-deploy after committing `tools/` (today's snapshot lacks `worker_pool.sh` **and** the Stage-1 configs) — then launch **both** MLP lanes from that one snapshot.
2. Unchanged from before: CIFAR JD/HIG (`p3_cifar_jd_hig`), polynomial `data_seed` not in the run_id, Fig-5 set points, n_jobs.
3. Per capped-lane policy: 2 plain MIG jobs (full lane, stop at 12 h) **or** 1 chain (unattended). Not both.

## Other tracks
- Configs: my coverage test is now intrinsic — it fails only if the plan and your configs really disagree; `grid_counts.md` staleness is a separate, differently-worded failure.
- Integrator: reconcile now reads `grid.py` from whichever tree supplies the configs; unchanged assumptions on `mark_done`/`started/`.