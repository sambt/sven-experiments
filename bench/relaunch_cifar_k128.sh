#!/bin/bash
# Relaunch the unfinished CIFAR k=128 Sven runs alone on a GPU (NPROC=1) with
# capture "full" (single jacrev over all parameters), as set in the CIFAR configs.
# Usage: bench/relaunch_cifar_k128.sh [lrs...]      (DRY=1 to print only)
set -u; set -f; cd /n/home/anon/sven-experiments
LRS=${@:-0.1 0.5 1.0}; n=0
for c in cifar10_resnet_scan_labelRegression cifar10_resnet_ce_scan; do
  for s in 4000 4001 4002 4003 4004; do for lr in $LRS; do
    ov="mode=svd model_seeds=[$s] k_values=[128] lrs=[$lr]"
    if [ "${DRY:-0}" = 1 ]; then echo "  [dry] NPROC=1 $c $ov"; else
      jid=$(NPROC=1 sbatch --parsable --job-name="$c:svd_bnbatch_full_s${s}_k128_lr${lr}" submit_rebuttal_parallel.sh experiments/configs/$c.yaml $ov); echo "  $jid $c $ov"; fi
    n=$((n+1))
  done; done
done; echo "$n jobs"
