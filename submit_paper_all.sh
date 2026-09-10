#!/bin/bash
# Re-run the full PAPER suite on the Gram backend, same footing as
# the rebuttal suite. Old results already moved to _backup_2026-08-31/.
# SLURM queues these under the per-user cap; run_id dedup => re-submittable.
set -u
cd /n/home11/sambt/iaifi/sv3
C=experiments/configs
sub() { NPROC=$1 sbatch --parsable submit_rebuttal_parallel.sh "$2" "${@:3}"; }

echo "== toy_1d (NPROC=6) =="
for c in toy_1d_scan toy_1d_microbatch_scan toy_1d_paramfrac_scan; do sub 6 $C/$c.yaml; done

echo "== polynomial (NPROC=6) =="
for c in polynomial_scan polynomial_microbatch_scan polynomial_paramfrac_scan; do sub 6 $C/$c.yaml; done

echo "== MNIST (NPROC=4) =="
for c in mnist_scan_labelRegression mnist_scan_ce mnist_kappaScan_labelRegression \
         mnist_microbatch_labelreg_scan mnist_microbatch_ce_scan \
         mnist_paramfrac_labelreg_scan mnist_paramfrac_ce_scan; do sub 4 $C/$c.yaml; done

echo "== CIFAR-10 ResNet18 (NPROC=4) =="
for c in cifar10_resnet_scan_labelRegression cifar10_resnet_ce_scan \
         cifar10_resnet_kappaScan_labelReg cifar10_resnet_ce_kappaScan \
         cifar10_resnet_paramFrac_scan_labelReg cifar10_resnet_ce_paramFrac_scan; do sub 4 $C/$c.yaml; done
