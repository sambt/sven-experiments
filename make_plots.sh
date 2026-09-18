#!/bin/bash
# Re-execute the analysis notebooks IN PLACE (outputs + the PDFs under analysis/plots_v2/).
#
#   ./make_plots.sh                 every notebook, in dependency-free order
#   ./make_plots.sh toy_1d_analysis comparisons     just these (name without .ipynb)
#   ONLY_SCANS=1 ./make_plots.sh    the four headline scan notebooks + comparisons
#
# A notebook whose data is not in experiment_results/ prints "[skip] ... not found" /
# "missing ..." and carries on (see analysis/RERUNS_NEEDED.md, item 7); a notebook that
# errors stops the script and leaves its partially executed copy in place.
#
# Every notebook loads the shared helpers (style.py, scan_analysis.py, ...) fresh, so this
# is also how the saved outputs are brought back in line after a helper change -- they are
# stale until it is run (analysis/ANALYSIS_FIXES.md).
set -eu
cd "$(dirname "$0")/analysis"

SCANS=(toy_1d_analysis polynomial_analysis mnist_analysis mnist_analysis_labelRegression comparisons)
STUDIES=(baselines_analysis batchsize_analysis overparam_analysis critbatch_analysis nanogpt_analysis
         cifar_analysis kappa_analysis finetune_analysis microbatch_analysis paramfrac_analysis)
PROFILES=(profile_overview profile_scaling profile_paramfrac_microbatch profile_sven_backends)

if [ $# -gt 0 ]; then NBS=("$@")
elif [ "${ONLY_SCANS:-0}" = 1 ]; then NBS=("${SCANS[@]}")
else NBS=("${SCANS[@]}" "${STUDIES[@]}" "${PROFILES[@]}"); fi

for nb in "${NBS[@]}"; do
  printf '== %s ==\n' "$nb"
  jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb" \
      --ExecutePreprocessor.timeout=3600 --log-level=ERROR
done
echo "done: ${#NBS[@]} notebook(s); plots under analysis/plots_v2/"
