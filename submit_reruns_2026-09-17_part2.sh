#!/bin/bash
# Part 2 of the 2026-09-17 reruns (decided after part 1 was submitted):
#   * exp_finetune_cifar_smallN: ImageNet-pretrained ResNet18 on N in {250,500,1000,2000} CIFAR
#     samples, hooks capture with BN frozen at its ImageNet running stats (the model's design),
#     Sven + [AdamW, SGD, Muon, MuonW] at their default wd, 3 seeds. Layout as submit_fresh_suite tier3.
#   * MuonW in the overparam and batch-size scans at ITS default wd (0.1; AdamW side 0.01), passed
#     explicitly because those configs sweep weight_decays [0.0, 0.01] for AdamW / Muon.
#   * GPT-2-small: ./submit_gpt2.sh (its own launcher; one full GPU per Sven run).
# Usage: ./submit_reruns_2026-09-17_part2.sh     DRY=1 to print only. Dedup makes it re-submittable.
set -u; set -f
cd /n/home/anon/sven-experiments
C=experiments/configs; DRY=${DRY:-0}; NJOBS=0
sub() {
  local np=$1 cfg=$2; shift 2
  local name; name=$(basename "$cfg" .yaml); NJOBS=$((NJOBS+1))
  local jname; jname="$name:$(echo "$*" | sed -E 's/optimizers_standard=//; s/mode=//; s/weight_decays=/wd/; s/[][]//g; s/ +/_/g')"
  if [ "$DRY" = 1 ]; then printf "  [dry] NPROC=%s %-40s %s\n" "$np" "$name" "$*"
  else local jid; jid=$(NPROC=$np sbatch --parsable --job-name="${jname:0:120}" submit_rebuttal_parallel.sh "$cfg" "$@")
       printf "  %-9s NPROC=%s %-40s %s\n" "$jid" "$np" "$name" "$*"; fi
}
echo "== Fine-tune: pretrained ResNet18, small N (NPROC=2) =="
for N in 250 500 1000 2000; do
  sub 2 $C/exp_finetune_cifar_smallN.yaml mode=svd n_data=$N
  sub 2 $C/exp_finetune_cifar_smallN.yaml mode=standard n_data=$N
done
echo "== MuonW at its default wd in the overparam / batch-size scans =="
MW="optimizers_standard=[MuonW]"; WD="weight_decays=[0.1]"
for N in 150 300 600 1200;  do sub 6 $C/rebuttal_overparam_toy_1d_scan.yaml     mode=standard "$MW" "$WD" n_data=$N; done
for N in 170 340 675 1350;  do sub 6 $C/rebuttal_overparam_polynomial_scan.yaml mode=standard "$MW" "$WD" n_data=$N; done
for N in 2500 5000 10000 20000 40000 60000; do sub 4 $C/rebuttal_overparam_mnist_scan.yaml mode=standard "$MW" "$WD" n_data=$N; done
sub 6 $C/rebuttal_batchsize_polynomial_scan.yaml mode=standard "$MW" "$WD"
echo "== GPT-2-small =="
if [ "$DRY" = 1 ]; then echo "  [dry] ./submit_gpt2.sh  (3 Sven + 6 baseline-shard jobs)"; else ./submit_gpt2.sh; NJOBS=$((NJOBS+9)); fi
[ "$DRY" = 1 ] && echo "== $NJOBS jobs (dry run) ==" || echo "== $NJOBS jobs submitted =="
