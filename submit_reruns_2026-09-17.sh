#!/bin/bash
# Reruns called for by analysis/RERUNS_NEEDED.md (2026-09-17), after the ANALYSIS_FIXES audit.
#   item 0  AdamW at its default wd (0.01) + new MuonW, in every scan that lists them
#   item 1  Sven k = B slice of the four headline scans with full-spectrum logging (sven ca8742b)
#   item 2  standalone timing: re-select bench/best_configs.json, re-time what changed (dependent job)
#   item 4  missing seeds: the headline standard / hig / jd modes are resubmitted (run_id dedup
#           skips everything that exists; deterministic KFAC/HIG failures just fail again, fast)
#   item 5  5 seeds for nanoGPT speedrun, both critical-batch studies, the two MNIST-CE knob sweeps
#   item 7  the CIFAR kappa / param-fraction / Fig-5 ablations that were never submitted
#           (exp_finetune_cifar_smallN and GPT-2 are NOT here: separate decision)
# Superseded result files are MOVED ASIDE (never deleted) into a subdirectory of the scan --
# the loader reads only top-level *.jsonl, exactly like the CIFAR `_frozen_bn/` precedent --
# so the scan's dedup re-runs them:  <scan>/_adamw_wd0/ (item 0), <scan>/_spectra_truncated/ (item 1).
#
# Usage:  ./submit_reruns_2026-09-17.sh            DRY=1 ./submit_reruns_2026-09-17.sh (print only)
#         ONLY='cifar' ...                          subset by config-name regex (like submit_fresh_suite.sh)
# Re-running is safe (dedup); the move-asides are idempotent (nothing to move the second time).
set -u
set -f   # hydra list overrides use [...]
cd /n/home/anon/sven-experiments
C=experiments/configs; R=experiment_results
DRY=${DRY:-0}; ONLY=${ONLY:-}; NJOBS=0; DEPS=""; LAST_JID=""

sub() {  # sub NPROC config.yaml [hydra overrides...]  -> LAST_JID
  local np=$1 cfg=$2; shift 2
  local name; name=$(basename "$cfg" .yaml); LAST_JID=""
  if [ -n "$ONLY" ] && ! [[ "$name" =~ $ONLY ]]; then return; fi
  NJOBS=$((NJOBS+1))
  local jname; jname="$name:$(echo "$*" | sed -E 's/optimizers_standard=//; s/mode=//; s/model_seeds=/s/; s/param_fractions=/pf/; s/[][]//g; s/ +/_/g')"
  if [ "$DRY" = 1 ]; then
    printf "  [dry] NPROC=%s %-40s %s\n" "$np" "$name" "$*"; LAST_JID="dry$NJOBS"
  else
    LAST_JID=$(NPROC=$np sbatch --parsable --job-name="${jname:0:120}" submit_rebuttal_parallel.sh "$cfg" "$@")
    printf "  %-9s NPROC=%s %-40s %s\n" "$LAST_JID" "$np" "$name" "$*"
  fi
}
dep() { [ -n "$LAST_JID" ] && DEPS="$DEPS:$LAST_JID"; }   # collect a job the timing pass must wait for

aside() {  # aside <scan> <subdir> <glob>: move matching top-level jsonl (+ diag npz) into <scan>/<subdir>/
  local d=$R/$1 sd=$2 pat=$3 files n
  files=$(cd "$d" && set +f && ls $pat 2>/dev/null); n=$(printf "%s" "$files" | grep -c .)
  if [ "$n" = 0 ]; then echo "  (nothing to move aside: $1/$pat)"; return; fi
  if [ "$DRY" = 1 ]; then echo "  [dry] would move $n files $1/$pat -> $1/$sd/"; return; fi
  mkdir -p "$d/$sd/diag"
  for f in $files; do
    mv "$d/$f" "$d/$sd/"; b=${f%.jsonl}
    [ -f "$d/diag/$b.npz" ] && mv "$d/diag/$b.npz" "$d/$sd/diag/"
  done
  echo "  moved $n files $1/$pat -> $1/$sd/"
}
seeds_of() { grep -E '^model_seeds:' "$1" | grep -o '[0-9]\+' | tr '\n' ' '; }

echo "== Items 0 + 1 + 4: the four headline scans =="
for spec in toy_1d_scan:32:6 polynomial_scan:32:6 mnist_scan_labelRegression:64:4 mnist_scan_ce:64:4; do
  IFS=: read -r scan B np <<< "$spec"; cfg=$C/$scan.yaml
  echo "-- $scan (B=$B, NPROC=$np)"
  aside "$scan" _adamw_wd0         "*_optimAdamW_mseed*.jsonl"   # old wd=0 AdamW (== Adam); new run_ids carry _wd0.01
  aside "$scan" _spectra_truncated "svd_bs${B}_*_k${B}_*.jsonl"  # k = B Sven: spectra were cut at rtol
  if [ "$np" = 4 ]; then   # MNIST: k = B slice per seed (~16 runs x 4 min each, sharded)
    for s in $(seeds_of "$cfg"); do sub $np "$cfg" mode=svd "k_values=[$B]" "model_seeds=[$s]"; dep; done
  else
    sub $np "$cfg" mode=svd "k_values=[$B]"; dep
  fi
  # item 0 (+ item 4 gap fill where a file can actually appear): KFAC has 0/20 files on MNIST
  # (dies at the eigh every time) and 19/20, 11/20 on toy/poly; HIG only ever finished at
  # lr <= 0.1 on toy / MNIST label-reg; JD is 20/20 everywhere, so it is not resubmitted.
  if [ "$np" = 4 ]; then sub $np "$cfg" mode=standard "optimizers_standard=[AdamW,MuonW]"; dep
  else                   sub $np "$cfg" mode=standard "optimizers_standard=[AdamW,MuonW,KFAC]"; dep; fi
  case $scan in
    toy_1d_scan)                sub $np "$cfg" mode=hig "lrs_hig=[0.05,0.1]"; dep ;;   # 5 missing
    mnist_scan_labelRegression) sub $np "$cfg" mode=hig "lrs_hig=[0.1]"; dep ;;        # 1 missing
  esac
done

echo "== Items 0 + 5: nanoGPT speedrun (5 seeds; AdamW wd + MuonW; NPROC=2) =="
aside exp_nanogpt_speedrun _adamw_wd0 "*_optimAdamW_mseed*.jsonl"
sub 2 $C/exp_nanogpt_speedrun.yaml mode=svd "model_seeds=[5003,5004]"; dep
sub 2 $C/exp_nanogpt_speedrun.yaml mode=standard; dep

echo "== Items 0 + 5: critical batch, 5 seeds =="
aside exp_critbatch_nanogpt _adamw_wd0 "*_optimAdamW_mseed*.jsonl"
sub 2 $C/exp_critbatch_nanogpt.yaml mode=standard                       # AdamW(wd) x 5 seeds, ~4 h
for s in 5002 5003 5004; do sub 2 $C/exp_critbatch_nanogpt.yaml mode=svd "model_seeds=[$s]"; done   # ~3.6 h each
for s in 3002 3003 3004; do sub 4 $C/exp_critbatch_mnist.yaml "model_seeds=[$s]"; done              # mode: both, ~1.3 h each

echo "== Item 5: MNIST-CE knob sweeps, 2 more seeds (NPROC=4) =="
sub 4 $C/mnist_microbatch_ce_scan.yaml mode=svd
sub 4 $C/mnist_paramfrac_ce_scan.yaml  mode=svd

echo "== Item 7: CIFAR ResNet18 ablations, full-J capture, one run per job (NPROC=1, ~2 h each) =="
for k in 1 1.5 2 2.5 3; do sub 1 $C/cifar10_resnet_kappaScan_labelReg.yaml "kappa=[$k]"; done
for k in 1 2 3;         do sub 1 $C/cifar10_resnet_ce_kappaScan.yaml       "kappa=[$k]"; done
for f in 0.05 0.1 0.25 0.5 1.0;      do sub 1 $C/cifar10_resnet_paramFrac_scan_labelReg.yaml "param_fractions=[$f]"; done
for f in 0.05 0.1 0.25 0.5 0.75 1.0; do sub 1 $C/cifar10_resnet_ce_paramFrac_scan.yaml       "param_fractions=[$f]"; done
for s in 4000 4001 4002; do for f in 0.05 0.1 0.25 0.5 1.0; do
  sub 1 $C/rebuttal_fig5_cifar_paramfrac_scan.yaml "param_fractions=[$f]" "model_seeds=[$s]"; done; done

echo "== Item 2: standalone timing pass (exclusive node) after the headline + nanoGPT jobs =="
DEPS=${DEPS#:}
if [ "$DRY" = 1 ]; then
  echo "  [dry] sbatch --dependency=afterany:$DEPS --export=ALL,RESELECT=1,GROUP=all --job-name=timing_serial_RERUNS bench/timing_serial.sbatch"
elif [ -n "$DEPS" ]; then
  jid=$(sbatch --parsable --dependency=afterany:$DEPS --export=ALL,RESELECT=1,GROUP=all --job-name=timing_serial_RERUNS bench/timing_serial.sbatch)
  echo "  $jid  timing_serial_RERUNS (RESELECT=1; waits for $DEPS)"; NJOBS=$((NJOBS+1))
else
  echo "  (no headline jobs submitted under ONLY='$ONLY' -> timing pass not chained; run it by hand:"
  echo "   sbatch --export=ALL,RESELECT=1,GROUP=all --job-name=timing_serial_RERUNS bench/timing_serial.sbatch)"
fi
[ "$DRY" = 1 ] && echo "== $NJOBS jobs (dry run) ==" || echo "== $NJOBS jobs submitted =="
