#!/bin/bash
# Fresh paper + rebuttal suite (2026-09-11): 5 seeds, light outputs (jsonl + diag/npz),
# signed-residual Sven rows for scalar MSE. Old results: experiment_results/_backup_2026-09-11/.
#
# Usage:  ./submit_fresh_suite.sh [headline|ablations|tier1|rebuttal|tier3|all]   (default: all)
#         Run `headline` first; its scans fix the per-dataset k / rtol set points
#         that `ablations`, `rebuttal` and `tier3` assume.
#         DRY=1 ./submit_fresh_suite.sh tier1                       # print, don't submit
#         ONLY='mnist|cifar' ./submit_fresh_suite.sh headline          # subset by config-name regex
# Jobs go to whichever of lab_gpu_priority / lab_gpu / gpu starts them first (see the
# #SBATCH --partition line in submit_rebuttal_parallel.sh); the two lab partitions each cap
# a user at 2 nodes (8 single-GPU jobs), `gpu` has no cap but a lower priority tier.
#
# Layout (from the gpu_test sharding benchmark, bench/results_46028086.jsonl):
#   * NPROC shards per GPU: 6 for toy/poly MLPs, 4 for MNIST MLPs, 2 for ResNet/nanoGPT
#     (2-3.5x throughput per GPU; the per-user cap is 8 concurrent GPU jobs).
#   * Baselines are split into their own jobs so a Sven job never waits on an LBFGS
#     line search or a K-FAC eigh (that gating caused the 12h TIMEOUTs last time):
#     Sven | first-order | second-order (SOAP/Shampoo/KFAC) | LBFGS.
#   * Long grids are split by seed (MNIST headline Sven) or by batch size (batch-size
#     LBFGS) to stay under 12h. run_id dedup makes every job safely re-submittable:
#     if one times out, just run this script again -- finished runs are skipped.
set -u
set -f   # no globbing: hydra list overrides use [...]
cd /n/home/anon/sven-experiments
C=experiments/configs
GROUP=${1:-all}
DRY=${DRY:-0}
NJOBS=0

ONLY=${ONLY:-}   # optional regex: only submit configs whose name matches (e.g. ONLY='mnist|cifar')

sub() {  # sub NPROC config.yaml [hydra overrides...]
  local np=$1 cfg=$2; shift 2
  local name; name=$(basename "$cfg" .yaml)
  if [ -n "$ONLY" ] && ! [[ "$name" =~ $ONLY ]]; then return; fi
  NJOBS=$((NJOBS+1))
  # Descriptive SLURM job name (config + overrides, squashed) so `squeue -o %j` is readable
  # and a job can be recognised without scontrol.
  local jname; jname="$name:$(echo "$*" | sed -E 's/optimizers_standard=//; s/mode=//; s/model_seeds=/s/; s/[][]//g; s/ +/_/g')"
  if [ "$DRY" = 1 ]; then
    printf "  [dry] NPROC=%s %-46s %s\n" "$np" "$name" "$*"
  else
    local jid; jid=$(NPROC=$np sbatch --parsable --job-name="${jname:0:120}" submit_rebuttal_parallel.sh "$cfg" "$@")
    printf "  %-9s NPROC=%s %-46s %s\n" "$jid" "$np" "$name" "$*"
  fi
}

FIRST='[Adam,AdamW,SGD,RMSprop,Muon,PolyakSGD]'   # FULL suite, first-order part
CORE_FIRST='[Adam,SGD,RMSprop,PolyakSGD]'          # CORE suite, first-order part
SECOND='[SOAP,Shampoo,KFAC]'

sven_only()  { local np=$1 cfg=$2; shift 2; sub "$np" "$cfg" mode=svd "$@"; }
full_suite() { # Sven + FIRST + SECOND + LBFGS, each its own job
  local np=$1 cfg=$2; shift 2
  sub "$np" "$cfg" mode=svd "$@"
  sub "$np" "$cfg" mode=standard "optimizers_standard=$FIRST" "$@"
  sub "$np" "$cfg" mode=standard "optimizers_standard=$SECOND" "$@"
  sub "$np" "$cfg" mode=standard "optimizers_standard=[LBFGS]" "$@"
}
core_suite() { # Sven + CORE_FIRST + LBFGS
  local np=$1 cfg=$2; shift 2
  sub "$np" "$cfg" mode=svd "$@"
  sub "$np" "$cfg" mode=standard "optimizers_standard=$CORE_FIRST" "$@"
  sub "$np" "$cfg" mode=standard "optimizers_standard=[LBFGS]" "$@"
}
seeds_of() { grep -E '^model_seeds:' "$1" | grep -o '[0-9]\+' | tr '\n' ' '; }

headline() {  # the six broad scans: these fix the per-dataset k / rtol set points for everything else
  echo "== Headline: toy-1D and polynomial (NPROC=6) =="
  full_suite 6 $C/toy_1d_scan.yaml
  full_suite 6 $C/polynomial_scan.yaml
  echo "== Headline: MNIST (NPROC=4; Sven split by seed, ~3h each) =="
  for s in $(seeds_of $C/mnist_scan_labelRegression.yaml); do sub 4 $C/mnist_scan_labelRegression.yaml mode=svd "model_seeds=[$s]"; done
  sub 4 $C/mnist_scan_labelRegression.yaml mode=standard "optimizers_standard=$FIRST"
  sub 4 $C/mnist_scan_labelRegression.yaml mode=standard "optimizers_standard=$SECOND"
  sub 4 $C/mnist_scan_labelRegression.yaml mode=standard "optimizers_standard=[LBFGS]"
  for s in $(seeds_of $C/mnist_scan_ce.yaml); do sub 4 $C/mnist_scan_ce.yaml mode=svd "model_seeds=[$s]"; done
  sub 4 $C/mnist_scan_ce.yaml mode=standard "optimizers_standard=$CORE_FIRST"
  sub 4 $C/mnist_scan_ce.yaml mode=standard "optimizers_standard=[LBFGS]"
  echo "== Headline: CIFAR-10 ResNet18 (NPROC=2; Sven and LBFGS split by seed, ~6h each) =="
  for c in cifar10_resnet_scan_labelRegression cifar10_resnet_ce_scan; do
    for s in $(seeds_of $C/$c.yaml); do
      sub 2 $C/$c.yaml mode=svd "model_seeds=[$s]"
      sub 2 $C/$c.yaml mode=standard "optimizers_standard=[LBFGS]" "model_seeds=[$s]"
    done
    sub 2 $C/$c.yaml mode=standard "optimizers_standard=$CORE_FIRST"
  done
}

ablations() {  # k = B and the per-dataset rtol set point (tentative until the headline scans are in)
  echo "== Ablations: toy-1D / polynomial micro-batch + param-fraction (NPROC=6) =="
  for c in toy_1d_microbatch_scan toy_1d_paramfrac_scan polynomial_microbatch_scan polynomial_paramfrac_scan; do sven_only 6 $C/$c.yaml; done
  echo "== Ablations: MNIST kappa / micro-batch / param-fraction (NPROC=4) =="
  for c in mnist_kappaScan_labelRegression mnist_microbatch_labelreg_scan mnist_microbatch_ce_scan \
           mnist_paramfrac_labelreg_scan mnist_paramfrac_ce_scan; do sven_only 4 $C/$c.yaml; done
  echo "== Ablations: CIFAR kappa / param-fraction (NPROC=2) =="
  for c in cifar10_resnet_kappaScan_labelReg cifar10_resnet_ce_kappaScan \
           cifar10_resnet_paramFrac_scan_labelReg cifar10_resnet_ce_paramFrac_scan; do sven_only 2 $C/$c.yaml; done
}

tier1() { headline; ablations; }

rebuttal() {
  echo "== Rebuttal: dataset-overparam P>N (one job set per n_data) =="
  for N in 150 300 600 1200;  do full_suite 6 $C/rebuttal_overparam_toy_1d_scan.yaml n_data=$N; done
  for N in 170 340 675 1350;  do full_suite 6 $C/rebuttal_overparam_polynomial_scan.yaml n_data=$N; done
  for N in 2500 5000 10000 20000 40000 60000; do full_suite 4 $C/rebuttal_overparam_mnist_scan.yaml n_data=$N; done
  echo "== Rebuttal: batch-size sweep (LBFGS split by batch size) =="
  sub 6 $C/rebuttal_batchsize_polynomial_scan.yaml mode=svd
  sub 6 $C/rebuttal_batchsize_polynomial_scan.yaml mode=standard "optimizers_standard=$FIRST"
  sub 6 $C/rebuttal_batchsize_polynomial_scan.yaml mode=standard "optimizers_standard=$SECOND"
  for B in 8 16 32 64 128 256; do sub 6 $C/rebuttal_batchsize_polynomial_scan.yaml mode=standard "optimizers_standard=[LBFGS]" "batch_size=[$B]"; done
  echo "== Rebuttal: Fig-5 ResNet param-fraction (chunked capture, NPROC=2) =="
  sven_only 2 $C/rebuttal_fig5_cifar_paramfrac_scan.yaml
}

tier3() {
  echo "== Tier 3: nanoGPT / fine-tune / critical batch (NPROC=2-4) =="
  sub 2 $C/exp_nanogpt_speedrun.yaml mode=svd
  sub 2 $C/exp_nanogpt_speedrun.yaml mode=standard
  sub 2 $C/exp_critbatch_nanogpt.yaml mode=svd
  sub 2 $C/exp_critbatch_nanogpt.yaml mode=standard
  sub 4 $C/exp_critbatch_mnist.yaml
  for N in 250 500 1000 2000; do sub 2 $C/exp_finetune_cifar_smallN.yaml mode=svd n_data=$N; sub 2 $C/exp_finetune_cifar_smallN.yaml mode=standard n_data=$N; done
  echo "   (GPT-2-small has its own launcher: ./submit_gpt2.sh)"
}

case "$GROUP" in
  headline) headline ;;
  ablations) ablations ;;
  tier1) tier1 ;;
  rebuttal) rebuttal ;;
  tier3) tier3 ;;
  all) tier1; rebuttal; tier3 ;;
  *) echo "unknown group '$GROUP' (headline|ablations|tier1|rebuttal|tier3|all)"; exit 1 ;;
esac
if [ "$DRY" = 1 ]; then echo "== $NJOBS jobs (dry run) =="; else echo "== $NJOBS jobs submitted -- the per-user cap runs ~8 at a time; the rest queue =="; fi
