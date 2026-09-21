# Experiment Repository for Sven

This repository contains code for the experiments presented in **Sven: Singular Value Descent
as a Computationally Efficient Natural Gradient Method**. You will need the `sven` package
from [this repository](https://github.com/sambt/sven) to use the Sven optimizer; in this tree
it lives as the nested repo `sven/`.

## Where things are

| what | where |
|---|---|
| **What was actually run**, with grids, splits, seeds, evaluation protocol and run counts | [`EXPERIMENTS.md`](EXPERIMENTS.md) |
| Campaign state, timeline, snapshots, SLURM job ids | [`campaign/CAMPAIGN_STATUS.md`](campaign/CAMPAIGN_STATUS.md) |
| Binding decisions: splits, BatchNorm policy, record schema 2, dedup, layout | [`campaign/CONTRACTS.md`](campaign/CONTRACTS.md) |
| Exact run counts per scan, in-plan vs cut vs parked, GPU-h estimates | [`campaign/grid_counts.md`](campaign/grid_counts.md) |
| Analysis plan and the work packages under way | [`campaign/ANALYSIS_PLAN.md`](campaign/ANALYSIS_PLAN.md), [`campaign/ANALYSIS_CONTRACTS.md`](campaign/ANALYSIS_CONTRACTS.md) |
| Review findings the campaign answers | [`CHANGES_NEEDED.md`](CHANGES_NEEDED.md), `FABLE_CRITIQUES.md`, `CODEX_CRITIQUES.md` |
| What each component does, per development track | `campaign/stage0_reports/*.impl.md`, `campaign/stage1_reports/*` |
| GPU probe: step times, NPROC per workload, the `empty_cache` finding | `campaign/stage0_reports/gpu.probe.md` |
| Launch / bookkeeping tools | `tools/` (see below) |
| Analysis notebooks and helpers | `analysis/` (see below) |

## Setup

```bash
uv sync --inexact            # or: pip install -r requirements.txt
```

The project venv is `.venv/`; use `.venv/bin/python` and `.venv/bin/jupyter` — not whatever is
first on `PATH`.

## Running a single experiment

Experiments use [Hydra](https://hydra.cc/); config files live in `experiments/configs/`.

```bash
python run.py --config-name toy_1d_scan
python run.py --config-name toy_1d_scan num_epochs=50 device=cpu
python run.py --config-name mnist_scan_ce mode=svd k_values=[64] lrs=[0.5]
```

`mode` selects the optimizer family: `svd` (Sven), `standard` (every first- and second-order
baseline), `jd`, `hig`, `all`. `+n_shards` / `+shard_id` slicing works only with
`scheduler=static`; the default `scheduler=claims` lets any number of processes and jobs serve
one scan concurrently by claiming runs from a file queue.

For the campaign the runner is driven by `tools/launch_campaign.py` from an exported snapshot
rather than by hand — see `EXPERIMENTS.md` §12.

**Config inventory.** `EXPERIMENTS.md` lists every in-plan scan with its grid; the short
version is four MLP headline scans (toy-1D, random polynomial, MNIST label-regression, MNIST
cross-entropy), two CIFAR-10 / ResNet18 headline scans, nanoGPT and GPT-2-small, three
dataset-overparameterisation scans, a batch-size scan, a κ scan, and eight micro-batch /
parameter-fraction ablations. Each headline scan additionally has `_timing`, `_diag` and
`_confirm` companion configs.

**Loss keys** (`loss:`): `mse`; `label_regression` (the paper's Sec. 4 definition, per-sample
`‖f(x) − onehot(y)‖²` on the **raw** outputs — every `*labelReg*` result uses this); `ce`;
`lm_ce` (per-token cross-entropy); and `brier` (`‖softmax(f(x)) − onehot(y)‖²`, a *different*
objective whose results are not comparable with `label_regression`, and which no campaign scan
uses). Accuracy is always argmax over the raw outputs.

**Signed residuals.** For scalar-output `mse` Sven builds its Jacobian rows from the signed
residual rather than `loss^(κ/2)`, for every κ (`signed_residual: true`, the default). The
update is the same; the κ < 2 NaN at zero residual is gone. Multi-output losses keep the
`loss^(κ/2)` rows.

## Cost of a Sven step

Every campaign run uses the exact **Gram** backend (`use_gram: true`): the `M × M` Gram matrix
`G = J Jᵀ` accumulated in float64, one `torch.linalg.eigh(G)`, and only then the `rtol` cut and
the rank cap `k` (`M` = batch size / micro-batch size).

> A Sven step therefore costs **the capture plus one `M × M` eigendecomposition, independent of
> `k`**. `k` and `rtol` decide how many eigenpairs get inverted *after* the decomposition, not
> how expensive the decomposition is. The `O(k N |D|)` figure describes **only the classic
> randomized-SVD path**, which no campaign run takes. Under `gram_capture: full` the capture
> materialises the dense `(B, P)` Jacobian, so at ResNet scale *memory*, not the
> eigendecomposition, is the binding cost.

`empty_cache` defaults to `False`: the per-step `torch.cuda.empty_cache()` cost 4.5× on CIFAR
Sven (841 → 187 ms/step) and was the entire source of step-time variance. Launchers export
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Results layout

Results live under `$SV3_RESULTS_ROOT` (default `experiment_results`, a symlink to holystore).
Each run writes, under `{root}/{scan}/`:

```
<run_id>.jsonl        light record: hyperparameters, per-epoch curves, timings, provenance,
                      summaries (val_final / val_best / test / test_acc / train_eval_final /
                      peak_gpu_mem_mb), status, run_hash.  schema_version: 2
diag/<run_id>.npz     per-batch arrays; for Sven the scheduled spectra (svs, utr,
                      update_norm, resid_norm, sv_min_kept, sv_noise_floor + svs_step)
ckpt/<run_id>.pt      checkpoint ladder per the scan's `checkpoints` policy
                      (+ ckpt/init_mseed<seed>.<gen>.pt, the shared initialisation)
done/, claims/, started/, manifest/, configs/     bookkeeping; `configs/` holds the resolved
                      Hydra config of each job.  `_stale/`, `attempts/` appear only on trouble
```

In the notebooks `load_results(name)` reads only the light files; `load_results(name,
slim=False)` or `load_diagnostics(row)` pulls in the npz on demand. The slim caches under
`experiment_results/_cache/` invalidate themselves on any file change.

The frozen pre-campaign results are read-only at `experiment_results_legacy_2026-09-18/`
(`SV3_RESULTS_ROOT=../experiment_results_legacy_2026-09-18` for the analysis).

## Tools

| script | what it does |
|---|---|
| `tools/deploy_snapshot.sh` | freeze both repos at HEAD into a content-addressed snapshot on holystore; prints the path as its last line |
| `tools/launch_campaign.py` | turn a `campaign/plan_*.yaml` into sbatch commands; prints by default, submits with `--submit` |
| `tools/reconcile.py` | the authority on "is this scan done": expected vs on-disk per family, with `ok / diverged / oom / error / claimed-live / stale-hash` |
| `tools/select_best.py` | the **selection of record** → `bench/best_configs.json`, under the binding rule (eligible → fewest diverged → seed-mean final validation loss). Never quote `reconcile.py`'s quick best-config table: it omits the fewest-diverged tier |
| `tools/gen_phase5_plan.py` | regenerate `campaign/plan_phase5.yaml` from that selection |
| `tools/worker_pool.sh`, `tools/campaign_plan.py` | the per-GPU worker pool and the plan loader the launcher uses |
| `tools/check_token_split.py` | verify the GPT-2 token bins come from disjoint documents |
| `bench/check_timing_join.py` | did the timing pass re-run the *same* runs as the scan (run_id, run_hash, trajectory)? |
| `bench/submit_timing_phase5.sh` | submit the timing pass, one job per scan, with a calibration microbenchmark at each end |
| `bench/calibrate_step.py` | the microbenchmark that makes host-load contamination detectable after the fact |

## Memory / step-time profiling

`experiments/optimizer_profile.py` measures peak GPU memory and steady-state step time for
every baseline and every Sven variant (Gram/hooks, Gram/full-Jacobian, Gram/chunked, classic
randomized SVD) on each architecture, one `profile_<arch>.yaml` per architecture. It is
study-based, not a grid: a set point plus single-axis sweeps (methods, batch size, chunk
fraction, parameter fraction, micro-batch, rank, model width). Out-of-memory is recorded as a
result. Run it on one exclusively reserved node:

```bash
sbatch bench/profile_serial.sbatch                      # all architectures, serially, resumable
PROFILE_CONFIGS="profile_mnist" sbatch bench/profile_serial.sbatch
```

Results land in `{output_root}/<config>/<run_id>.json`, resolved as `$SV3_PROFILE_ROOT` >
`profile.output_dir` in the config > `profile_results_v3`; existing files are skipped, so the
sweep is resumable. `analysis/profile_helpers.py` flattens them into one table (reading
`$SV3_PROFILE_ROOT`, else v3 once it has results, else v2) and the `analysis/profile_*.ipynb`
notebooks produce the tables and figures.

**`profile_results_v2` is pessimistic for Sven and is kept only as the "before" table.** It
was measured on 2026-09-17 with the per-step `torch.cuda.empty_cache()` that the campaign has
since turned off — worth up to 4.5× on CIFAR full-capture Sven. The re-measurement at the
campaign code writes `profile_results_v3/` (`campaign/ANALYSIS_PLAN.md` §7.4), and **it has
landed: 720/720 configurations, 2026-09-21, job 47396284** — so v3 is the root of record, the
notebooks read it, and the v2 Sven step times are upper bounds kept only for the before/after
table in `EXPERIMENTS.md` §1.6. Never mix the two roots in one table except as that explicit
v2-vs-v3 comparison.

## Analysis

Notebooks and helpers are in `analysis/`; shared style and loaders are `analysis/style.py`,
`analysis/analysis_helpers.py` and `analysis/scan_analysis.py`. `./make_plots.sh` re-executes
every notebook in place (figures under `analysis/plots_v2/`); pass notebook names to run a
subset, or `ONLY_SCANS=1` for the headline set. It needs a compute node — a cold scan load
from Lustre is ~20 s.

Conventions the analysis is held to (`analysis/ANALYSIS_FIXES.md`,
`campaign/ANALYSIS_CONTRACTS.md`): selection on validation only, **test metrics are outcomes
and never selection inputs**; diverged runs excluded from means and counted, with
`finished / attempted` on every table; seed bands are mean ± 1 std (ddof = 1), labelled
"± 1 std over seeds"; headline numbers come from the confirmation seeds with the tuning-seed
numbers beside them.

**Diverged has two counts, and the analysis uses the wider one.** `status == "diverged"` in a
record is the *lifecycle* count (it decides whether a run is retried): 1,402 campaign runs.
`analysis/style.is_diverged` — recorded, **or** a non-finite final value, **or** a final
validation loss more than 10× `val[0]` — is the *analysis* count and governs selection: 2,553
over the same runs. A method with 0 recorded divergences can have hundreds of finite blow-ups
(Sven: 39 recorded, 688 wide). `EXPERIMENTS.md` §7 has both, per scan and per method; never
make a robustness claim from the status field alone.

`analysis/RERUNS_NEEDED.md` ends with the launch log of the campaign (dates, phases, snapshots,
both repos' SHAs, job ids, outcome); its numbered items above that log are historical and were
superseded by the campaign.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q          # sv3, CPU-only
(cd sven && ../.venv/bin/python -m pytest tests/ -q)
```

`tests/test_configs.py` pins the config grids against `campaign/grid_counts.md`, so a config
edit that moves a run count fails until the cost table moves with it.
