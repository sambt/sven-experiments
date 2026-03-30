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

## Analysis

Analysis notebooks are in `analysis/`. They load experiment results and produce plots. The shared style configuration is in `analysis/style.py`.