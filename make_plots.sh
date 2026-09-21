#!/bin/bash
# Re-execute the analysis notebooks IN PLACE (outputs + the PDFs under analysis/plots_v2/).
#
#   ./make_plots.sh                 every notebook, in dependency-free order
#   ./make_plots.sh toy_1d_analysis comparisons     just these (name without .ipynb)
#   ONLY_SCANS=1 ./make_plots.sh    the four headline scan notebooks + comparisons
#   NO_PROFILES=1 ./make_plots.sh   everything except the four profile_* notebooks
#                                   (those read profile_results_v2/ -> v3, not the scans)
#
# A notebook whose data is not in experiment_results/ prints "[skip] ... not found" /
# "missing ..." and carries on (see analysis/RERUNS_NEEDED.md, item 7); a notebook that
# errors stops the script and leaves its partially executed copy in place.
#
# Every notebook loads the shared helpers (style.py, scan_analysis.py, ...) fresh, so this
# is also how the saved outputs are brought back in line after a helper change -- they are
# stale until it is run (analysis/ANALYSIS_FIXES.md).
set -eu
REPO=$(cd "$(dirname "$0")" && pwd)
# The venv's OWN jupyter: the `jupyter` on PATH (~/.local/bin) cannot import
# jupyter_core, and nbconvert is a dev dependency of this project (pyproject.toml),
# not of whatever interpreter happens to come first.  `uv sync` installs it.
JUPYTER=$REPO/.venv/bin/jupyter
if [ ! -x "$JUPYTER" ] || ! "$JUPYTER" nbconvert --version >/dev/null 2>&1; then
  echo "error: $JUPYTER nbconvert is missing; run 'uv sync --inexact' (or" >&2
  echo "       'uv pip install --python .venv/bin/python nbconvert')" >&2
  exit 1
fi
cd "$REPO/analysis"

SCANS=(toy_1d_analysis polynomial_analysis mnist_analysis mnist_analysis_labelRegression comparisons)
STUDIES=(baselines_analysis batchsize_analysis overparam_analysis critbatch_analysis nanogpt_analysis
         cifar_analysis kappa_analysis finetune_analysis microbatch_analysis paramfrac_analysis)
PROFILES=(profile_overview profile_scaling profile_paramfrac_microbatch profile_sven_backends)

# >>> NEW NOTEBOOKS OF THE 2026-09-20 ANALYSIS WORK PACKAGES -- ADD YOURS HERE <<<
# One line per work package (campaign/ANALYSIS_PLAN.md section 6).  A notebook stays
# commented out until it exists and executes clean, so `./make_plots.sh` keeps working
# while the packages land one at a time; uncomment the entry in the same commit that
# adds the notebook.  Keep the names sorted by package, no .ipynb suffix.
NEW=(
  # WP2 headline   : headline_tables
  # WP3 spectra    : spectra_analysis
  # WP4b large     : gpt2_analysis
  # WP5 legacy diff: legacy_vs_fresh
)

if [ $# -gt 0 ]; then NBS=("$@")
elif [ "${ONLY_SCANS:-0}" = 1 ]; then NBS=("${SCANS[@]}")
elif [ "${NO_PROFILES:-0}" = 1 ]; then NBS=("${SCANS[@]}" "${STUDIES[@]}" ${NEW[@]+"${NEW[@]}"})
else NBS=("${SCANS[@]}" "${STUDIES[@]}" ${NEW[@]+"${NEW[@]}"} "${PROFILES[@]}"); fi

for nb in "${NBS[@]}"; do
  printf '== %s ==\n' "$nb"
  "$JUPYTER" nbconvert --to notebook --execute --inplace "$nb.ipynb" \
      --ExecutePreprocessor.timeout=3600 --log-level=ERROR
done
echo "done: ${#NBS[@]} notebook(s); plots under analysis/plots_v2/"
