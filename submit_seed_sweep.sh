#!/bin/bash
# Submit one SLURM job per model seed for an experiment config.
#
# Usage:
#   ./submit_seed_sweep.sh experiments/configs/mnist_scan_labelRegression.yaml
#   ./submit_seed_sweep.sh experiments/configs/mnist_scan_labelRegression.yaml --extra-hydra-overrides
#
# Seeds are defined here; override as needed.

SEEDS=(1 2 3 4 5 6 7 8 9)

config=$1
shift

config_name=$(basename "$config" .yaml)

for seed in "${SEEDS[@]}"; do
    echo "Submitting seed=$seed for config=$config_name"
    sbatch --time=1-12:00:00 --job-name="${config_name}_s${seed}" \
        submit_experiment.sh "$config" "model_seeds=[$seed]" "$@"
done
