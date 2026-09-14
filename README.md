# Experiment Repository for Sven

This repository contains code for the experiments presented in **Sven: Singular Value Descent as a Computationally Efficient Natural Gradient Method**. You will need to install the `sven` package from [this repository](https://github.com/sambt/sven) to use the Sven optimizer.

## Setup

Install the `sven` package first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Running Experiments

Experiments use [Hydra](https://hydra.cc/) for configuration. Config files live in `experiments/configs/`.

Run an experiment from the repository root:

```bash
python run.py --config-name <config>
```

For example:

```bash
python run.py --config-name toy_1d_scan
python run.py --config-name mnist_scan
python run.py --config-name polynomial_scan
```

You can override any config parameter from the command line:

```bash
python run.py --config-name toy_1d_scan num_epochs=50 device=cpu
```

Available configs include scans over Sven hyperparameters (`k`, learning rate, `rtol`) and comparisons against standard optimizers (Adam, SGD, etc.) for several datasets:

| Config | Dataset | Notes |
|--------|---------|-------|
| `toy_1d_scan` | 1D regression | Basic test problem |
| `polynomial_scan` | Random polynomial | |
| `mnist_scan` | MNIST | Cross-entropy loss |
| `mnist_scan_labelRegression` | MNIST | Label regression loss |
| `*_microbatch_*` | Various | Microbatched Jacobian variants |
| `*_paramfrac_*` | Various | Partial-parameter Jacobian variants |

**Loss keys** (`loss:` in a config): `ce` (cross-entropy), `mse`, `label_regression`
(the paper's Sec. 4 definition: per-sample `||f(x) − onehot(y)||²` on the **raw** network
outputs — every `*labelReg*` / `*labelRegression*` result uses this), `brier`
(`||softmax(f(x)) − onehot(y)||²`, the multiclass Brier score — a *different* objective whose
results are not comparable with `label_regression`; its run IDs carry a `_loss…` suffix and every
result row records its `loss`), and `lm_ce` (per-token cross-entropy for language models).
Accuracy is always argmax over the raw outputs, which softmax leaves unchanged.

**Output format.** Each run writes two files under `experiment_results/<config>/`: a light
`<run_id>.jsonl` (hyperparameters, per-epoch curves, timings, and a per-epoch `svd_summary`;
a few KB, all the loss-curve notebooks need) and `diag/<run_id>.npz` (compressed per-batch
losses/times and, for Sven, the singular-value spectra). Two config knobs control the Sven
diagnostics: `svd_info: none | summary | full` (default `full`) and `svd_spectra_every: N`
(default 20 — keep every N-th step's spectrum; set 1 to keep all). In the notebooks,
`load_results(name)` reads only the light files; `load_results(name, slim=False)` or
`load_diagnostics(row)` pulls in the npz on demand.

**Signed residuals.** For scalar-output `mse` Sven builds its Jacobian rows from the signed
residual `pred − y` (`signed_residual: true`, default) instead of `loss^(κ/2)`, for every `κ`. The update is the
same; the `κ < 2` NaN at zero residual is gone. Multi-output losses keep the `loss^(κ/2)` rows.

## Analysis

Analysis notebooks are in `analysis/`. They load experiment results and produce plots. The shared style configuration is in `analysis/style.py`.