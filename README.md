# Experiment Repository for Sven

This repository contains the code for the experiments presented in **Sven: Singular Value
Descent as a Computationally Efficient Natural Gradient Method**. You will need the `sven`
package to use the Sven optimizer; in this tree it lives as the nested repo `sven/`.

## Where things are

| what | where |
|---|---|
| **What was actually run**, with grids, splits, seeds, evaluation protocol, run counts, and the binding conventions | [`EXPERIMENTS.md`](EXPERIMENTS.md) |
| Exact run counts per scan, in-plan vs parked, GPU-h estimates | [`campaign/grid_counts.md`](campaign/grid_counts.md) |
| What was submitted, as plan files the launcher reads | `campaign/plan_*.yaml` |
| The runner and its configs | `run.py`, `experiments/` |
| Launch / bookkeeping tools | `tools/` (see below) |
| Timing and profiling passes | `bench/` (see below) |
| Analysis notebooks, helpers, and the paper's figures/tables | `analysis/` (see below) |

## Setup

```bash
uv sync --inexact
```

The project venv is `.venv/`; use `.venv/bin/python` and `.venv/bin/jupyter` — not whatever is
first on `PATH`. `pyproject.toml` / `uv.lock` pin the environment the campaign ran in.

## Environment

Nothing in the repo names a machine. Four variables place things:

| variable | meaning | default |
|---|---|---|
| `SV3_REPO` | the checkout, for the SLURM scripts | `$HOME/sven-experiments` |
| `SV3_SCRATCH` | deploy snapshots, work lists, logs, and the default results root | `~/scratch/sven` |
| `SV3_RESULTS_ROOT` | where runs are written and read | `experiment_results` (a symlink in the campaign) |
| `SV3_DATA_ROOT` | datasets (`torch/mnist`, `torch/cifar10`, `shakespeare`, `fineweb_edu_gpt2_v2`, ...) | `./torch_datasets` |

The plan files say `${SV3_SCRATCH}/results` for the results root and the plan loader expands
it. SLURM partitions are the generic `gpu` (a whole GPU per job) and `gpu_mig` (a MIG-sliced
partition, four slices per job); edit them in `campaign/plan_*.yaml` and the `#SBATCH` lines of
`bench/*.sbatch` for another cluster.

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

**Config inventory.** `EXPERIMENTS.md` lists every in-plan scan with its grid; the short
version is four MLP headline scans (toy-1D, random polynomial, MNIST label-regression, MNIST
cross-entropy), two CIFAR-10 / ResNet18 headline scans, nanoGPT and GPT-2-small, three
dataset-overparameterisation scans, a batch-size scan, a κ scan, and eight micro-batch /
parameter-fraction ablations. Each headline scan additionally has `_timing`, `_diag` and
`_confirm` companion configs. `profile_*.yaml` drive the memory / step-time profiler.

**Loss keys** (`loss:`): `mse`; `label_regression` (the paper's definition, per-sample
`‖f(x) − onehot(y)‖²` on the **raw** outputs — every `*labelReg*` result uses this); `ce`;
`lm_ce` (per-token cross-entropy). Accuracy is always argmax over the raw outputs.

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

Each run writes, under `$SV3_RESULTS_ROOT/{scan}/`:

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

## Reproducing the campaign on a SLURM cluster

`EXPERIMENTS.md` §11 is the step-by-step recipe. In short: freeze the checkout into a snapshot,
dry-run then submit the plan one work list at a time, reconcile, select, generate and run the
phase-5 passes, then analyse.

| script | what it does |
|---|---|
| `tools/deploy_snapshot.sh` | freeze both repos at HEAD into a content-addressed snapshot under `$SV3_SCRATCH/deploy`; prints the path as its last line |
| `tools/launch_campaign.py` | turn a `campaign/plan_*.yaml` into sbatch commands; prints by default, submits with `--submit` |
| `tools/worker_pool.sh`, `tools/campaign_plan.py` | the per-GPU worker pool and the plan loader the launcher uses |
| `tools/reconcile.py` | the authority on "is this scan done": expected vs on-disk per family, with `ok / diverged / oom / error / claimed-live / stale-hash` |
| `tools/select_best.py` | the **selection of record** → `bench/best_configs.json`, under the binding rule (eligible → fewest diverged → seed-mean final validation loss). Never quote `reconcile.py`'s quick best-config table: it omits the fewest-diverged tier |
| `tools/gen_phase5_plan.py` | regenerate `campaign/plan_phase5.yaml` from that selection |
| `tools/compute_ckpt_spectra.py` | probe-set Jacobian spectra along the checkpoints of a `_diag` pass |
| `tools/check_token_split.py` | verify the GPT-2 token bins come from disjoint documents |
| `tools/build_fig_notebooks.py` | regenerate the paper-figure notebooks from the figure registry |
| `bench/submit_timing_phase5.sh`, `bench/timing_phase5.sbatch` | submit the timing pass, one job per scan, with a calibration microbenchmark at each end |
| `bench/calibrate_step.py` | the microbenchmark that makes host-load contamination detectable after the fact |
| `bench/check_timing_join.py` | did the timing pass re-run the *same* runs as the scan (run_id, run_hash, trajectory)? |
| `bench/profile_serial.sbatch` | the memory / step-time profile, all architectures serially, resumable |

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
sweep is resumable. `analysis/lib/profile_helpers.py` flattens them into one table and the
`analysis/notebooks/profiling/profile_*.ipynb` notebooks produce the tables and figures.
`profile_results_v3` is the root of record; `profile_results_v2` was measured with the
per-step `empty_cache()` since turned off and is kept only for the before/after table in
`EXPERIMENTS.md` §1.6.

## Analysis

The notebooks are grouped under `analysis/notebooks/` (`headline/`, `spectra/`,
`mlp_studies/`, `large_models/`, `profiling/`, `paper/`) and the helper modules they import
are in `analysis/lib/`; `analysis/README.md` maps the directory and
`analysis/notebooks/README.md` indexes the notebooks. Every notebook's first cell chdir's to
`analysis/` and puts `analysis/lib/` on `sys.path`, so the analysis paths
(`../experiment_results`, `plots_v2/`, `tables/`) mean the same thing wherever it is opened
from. `./make_plots.sh` re-executes every notebook in place (figures under
`analysis/plots_v2/`); pass notebook names (no group, no suffix) or a group directory to run
a subset, or `ONLY_SCANS=1` for the headline set.

**The paper's own figures.** `analysis/paper_assets/` builds every figure, table and number
macro the manuscript uses (`cd analysis && ../.venv/bin/python -m paper_assets`). Each figure
is declared in its module's `FIGURE_SPECS` as a draw function that writes nothing, plus a dict
of its own knobs; `analysis/notebooks/paper/fig_{main,reviewer,large,spectra}.ipynb` draw them
with `paper_assets.notebook`, and `pf.pin(...)` persists a choice to
`analysis/paper_assets/figure_overrides.yaml`, which the CLI reads too. The contract for adding
or changing a figure is `analysis/paper_assets/README.md`.

Conventions the analysis is held to (`EXPERIMENTS.md` §12): selection on validation only,
**test metrics are outcomes and never selection inputs**; diverged runs excluded from means and
counted, with `finished / attempted` on every table; seed bands are mean ± 1 std (ddof = 1);
headline numbers come from the confirmation seeds with the tuning-seed numbers beside them.

**Diverged has two counts, and the analysis uses the wider one.** `status == "diverged"` in a
record is the *lifecycle* count (it decides whether a run is retried): 1,402 campaign runs.
`style.is_diverged` — recorded, **or** a non-finite final value, **or** a final validation loss
more than 10× `val[0]` — is the *analysis* count and governs selection: 2,553 over the same
runs. `EXPERIMENTS.md` §7 has both, per scan and per method; never make a robustness claim
from the status field alone.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q          # CPU-only
(cd sven && ../.venv/bin/python -m pytest tests/ -q)
```

`tests/test_configs.py` pins the config grids against `campaign/grid_counts.md`, so a config
edit that moves a run count fails until the cost table moves with it. A handful of tests need
the generated paper figures or a results root and skip or fail without them.
