#!/bin/bash
# Master launcher for the fresh Gram-backend rebuttal suite (2026-08-31).
# Submits every scan as an intra-GPU-sharded job. SLURM queues them under the
# per-user concurrency limit; run_id dedup makes any job safely re-submittable.
set -u
cd /n/home11/sambt/iaifi/sv3
C=experiments/configs
sub() { NPROC=$1 sbatch --parsable submit_rebuttal_parallel.sh "$2" "${@:3}"; }

echo "== synthetic baselines (NPROC=6) =="
sub 6 $C/rebuttal_baselines_toy_1d_scan.yaml
sub 6 $C/rebuttal_baselines_polynomial_scan.yaml

echo "== overparam 1D  P/N = 4,2,1,0.5  (NPROC=6) =="
for N in 150 300 600 1200; do sub 6 $C/rebuttal_overparam_toy_1d_scan.yaml n_data=$N; done

echo "== overparam polynomial  P/N = 4,2,1,0.5  (NPROC=6) =="
for N in 170 340 675 1350; do sub 6 $C/rebuttal_overparam_polynomial_scan.yaml n_data=$N; done

echo "== MNIST baselines (NPROC=4) =="
sub 4 $C/rebuttal_baselines_mnist_scan.yaml

echo "== overparam MNIST  crosses P/N=1 at N~27.5k  (NPROC=4) =="
for N in 2500 5000 10000 20000 40000 60000; do sub 4 $C/rebuttal_overparam_mnist_scan.yaml n_data=$N; done

echo "== batch-size sweep, polynomial (NPROC=6) =="
sub 6 $C/rebuttal_batchsize_polynomial_scan.yaml

echo "== Fig-5 ResNet18/CIFAR param-fraction, chunked Gram (NPROC=4) =="
sub 4 $C/rebuttal_fig5_cifar_paramfrac_scan.yaml
