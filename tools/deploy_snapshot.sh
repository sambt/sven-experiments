#!/bin/bash
# Freeze BOTH repos at HEAD into a content-addressed snapshot on labstore, so that
# campaign jobs never import from the live working tree (~10 agents edit it while jobs
# run; EXPERIMENTS.md section 12 "Stage 1 contracts / Execution").
#
#   tools/deploy_snapshot.sh [--allow-dirty] [--force] [--results-root PATH]
#                            [--base DIR] [--repo DIR] [--quiet]
#
# Prints the snapshot path as its LAST line, so a launcher can do
#   SNAP=$(tools/deploy_snapshot.sh | tail -1)
#
# Layout of <base>/<sv3sha8>_<svensha8>/:
#   run.py, experiments/, analysis/, tools/, ...   (sv3 @ HEAD, `git archive`)
#   sven/                                          (sven @ HEAD, nested, `git archive`)
#   DEPLOY_INFO.json                               (sv3 identity; provenance.py fallback)
#   sven/DEPLOY_INFO.json                          (sven identity; same fallback)
#   experiment_results -> <results root>           (so the runner's relative default works)
#   .deploy_complete                               (written last = "this export is whole")
#
# Idempotent: a second call with the same two HEADs prints the existing path and exits 0
# without touching it. `--force` re-exports (the old tree is removed only after the new
# one is complete). Nothing is ever written inside the results root itself.
set -u
set -o pipefail

REPO=${SV3_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BASE=${SV3_DEPLOY_BASE:-${SV3_SCRATCH:-$HOME/scratch/sven}/deploy}
RESULTS_ROOT=""
ALLOW_DIRTY=0
FORCE=0
QUIET=0

die() { echo "[deploy] ERROR: $*" >&2; exit 1; }
say() { [ "$QUIET" = 1 ] || echo "[deploy] $*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --force) FORCE=1; shift ;;
    --quiet) QUIET=1; shift ;;
    --results-root) RESULTS_ROOT=${2:?--results-root needs a path}; shift 2 ;;
    --base) BASE=${2:?--base needs a path}; shift 2 ;;
    --repo) REPO=${2:?--repo needs a path}; shift 2 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) die "unknown argument '$1' (try --help)" ;;
  esac
done

REPO=$(cd "$REPO" 2>/dev/null && pwd) || die "repo not found"
SVEN=$REPO/sven
[ -d "$SVEN/.git" ] || [ -f "$SVEN/.git" ] || die "$SVEN is not a git repo (the nested sven checkout is required)"

# ---------------------------------------------------------------------------
# Identity and cleanliness of both repos
# ---------------------------------------------------------------------------
# Two notions of "dirty", deliberately:
#   * tracked changes CHANGE WHAT `git archive HEAD` WOULD CONTAIN vs. what you are
#     testing -> they are what makes an export a lie, so they are what we refuse on.
#   * `git status --porcelain` (untracked included) is what provenance.git_facts()
#     records, so DEPLOY_INFO.json carries THAT flag, byte-compatible with a run made
#     from the live tree. Untracked files (campaign/, notebooks, ...) are never
#     archived, so they only earn a warning.
repo_facts() {              # repo_facts <root> -> "<sha> <dirty_any> <dirty_tracked> <n_untracked>"
  local root=$1 sha porc tracked untracked
  sha=$(git -C "$root" rev-parse HEAD) || die "git rev-parse failed in $root"
  porc=$(git -C "$root" status --porcelain) || die "git status failed in $root"
  tracked=$(git -C "$root" status --porcelain --untracked-files=no | grep -c . || true)
  untracked=$(printf '%s\n' "$porc" | grep -c '^??' || true)
  local any=false
  [ -n "$porc" ] && any=true
  local dt=false
  [ "$tracked" -gt 0 ] && dt=true
  echo "$sha $any $dt $untracked"
}

read -r SV3_SHA SV3_DIRTY SV3_DIRTY_TRACKED SV3_UNTRACKED <<<"$(repo_facts "$REPO")"
read -r SVEN_SHA SVEN_DIRTY SVEN_DIRTY_TRACKED SVEN_UNTRACKED <<<"$(repo_facts "$SVEN")"

say "sv3  HEAD $SV3_SHA  (tracked changes: $SV3_DIRTY_TRACKED, untracked files: $SV3_UNTRACKED)"
say "sven HEAD $SVEN_SHA  (tracked changes: $SVEN_DIRTY_TRACKED, untracked files: $SVEN_UNTRACKED)"

if [ "$ALLOW_DIRTY" != 1 ]; then
  bad=""
  [ "$SV3_DIRTY_TRACKED" = true ] && bad="$bad sv3"
  [ "$SVEN_DIRTY_TRACKED" = true ] && bad="$bad sven"
  if [ -n "$bad" ]; then
    echo "[deploy] ERROR: uncommitted tracked changes in:$bad" >&2
    git -C "$REPO" status --short --untracked-files=no >&2
    git -C "$SVEN" status --short --untracked-files=no >&2
    echo "[deploy] the snapshot would NOT be the code you are testing." >&2
    echo "[deploy] commit first, or re-run with --allow-dirty" >&2
    exit 2
  fi
fi
[ "$SV3_UNTRACKED" -gt 0 ] && say "note: $SV3_UNTRACKED untracked path(s) in sv3 are NOT exported (git archive HEAD)"
[ "$SVEN_UNTRACKED" -gt 0 ] && say "note: $SVEN_UNTRACKED untracked path(s) in sven are NOT exported"

# ---------------------------------------------------------------------------
# Where the results live
# ---------------------------------------------------------------------------
if [ -z "$RESULTS_ROOT" ]; then
  RESULTS_ROOT=${SV3_RESULTS_ROOT:-}
fi
if [ -z "$RESULTS_ROOT" ]; then
  [ -e "$REPO/experiment_results" ] || die "no --results-root given and $REPO/experiment_results does not exist"
  RESULTS_ROOT=$(readlink -f "$REPO/experiment_results")
fi
[ -d "$RESULTS_ROOT" ] || die "results root '$RESULTS_ROOT' is not a directory"
say "results root $RESULTS_ROOT"

SNAP=$BASE/${SV3_SHA:0:8}_${SVEN_SHA:0:8}

if [ -e "$SNAP/.deploy_complete" ] && [ "$FORCE" != 1 ]; then
  say "snapshot already exported (use --force to redo it)"
  cur=$(readlink "$SNAP/experiment_results" 2>/dev/null || true)
  if [ "$cur" != "$RESULTS_ROOT" ]; then
    say "WARNING: existing snapshot points experiment_results at '$cur', not '$RESULTS_ROOT'"
  fi
  echo "$SNAP"
  exit 0
fi

# ---------------------------------------------------------------------------
# Export into a temp dir, then move into place (so an interrupted export is never
# mistaken for a usable snapshot: .deploy_complete only exists in a whole tree)
# ---------------------------------------------------------------------------
mkdir -p "$BASE" || die "cannot create $BASE"
TMP=$(mktemp -d "$BASE/.tmp-XXXXXXXX") || die "cannot create a temp dir under $BASE"
cleanup() { [ -n "${TMP:-}" ] && [ -d "$TMP" ] && rm -rf "$TMP"; }
trap cleanup EXIT

git -C "$REPO" archive --format=tar HEAD | tar -x -C "$TMP" || die "export of sv3 failed"
mkdir -p "$TMP/sven"
git -C "$SVEN" archive --format=tar HEAD | tar -x -C "$TMP/sven" || die "export of sven failed"

STAMP=$(date -Is)
# The schema provenance.py falls back to: git_sha / git_dirty / exported_at (sha / dirty
# are accepted aliases). The sven identity goes in sven/DEPLOY_INFO.json because
# provenance.git_facts(<snap>/sven) looks there, and deliberately never walks up.
#
# `git_dirty` describes THIS EXPORT, not the working tree it came from: the tree is
# `git archive HEAD`, so untracked files are not in it and cannot make it differ from
# HEAD, while an uncommitted TRACKED change means the code under test is not HEAD.
# Every record made from a snapshot carries this flag as its "was the code committed"
# answer (C-R3 / F16), and ~10 agents keep untracked scratch files in the live tree --
# so the untracked-inclusive reading would stamp `dirty` on essentially every record of
# the campaign and say nothing. The working tree's own state is still recorded, under
# `git_dirty_worktree` / `git_untracked_paths`.
cat > "$TMP/DEPLOY_INFO.json" <<EOF
{
  "git_sha": "$SV3_SHA",
  "git_dirty": $SV3_DIRTY_TRACKED,
  "git_dirty_tracked": $SV3_DIRTY_TRACKED,
  "git_dirty_worktree": $SV3_DIRTY,
  "git_untracked_paths": $SV3_UNTRACKED,
  "exported_at": "$STAMP",
  "repo": "$REPO",
  "sven_git_sha": "$SVEN_SHA",
  "sven_git_dirty": $SVEN_DIRTY_TRACKED,
  "sven_git_dirty_tracked": $SVEN_DIRTY_TRACKED,
  "sven_git_dirty_worktree": $SVEN_DIRTY,
  "sven_git_untracked_paths": $SVEN_UNTRACKED,
  "results_root": "$RESULTS_ROOT",
  "exported_by": "tools/deploy_snapshot.sh",
  "host": "$(hostname)"
}
EOF
cat > "$TMP/sven/DEPLOY_INFO.json" <<EOF
{
  "git_sha": "$SVEN_SHA",
  "git_dirty": $SVEN_DIRTY_TRACKED,
  "git_dirty_tracked": $SVEN_DIRTY_TRACKED,
  "git_dirty_worktree": $SVEN_DIRTY,
  "git_untracked_paths": $SVEN_UNTRACKED,
  "exported_at": "$STAMP",
  "repo": "$SVEN",
  "exported_by": "tools/deploy_snapshot.sh"
}
EOF

# `experiment_results` is untracked in sv3 (a symlink), so the archive never carries it;
# create it explicitly, and overwrite whatever might be there.
rm -rf "$TMP/experiment_results"
ln -s "$RESULTS_ROOT" "$TMP/experiment_results" || die "cannot create the experiment_results symlink"

for f in run.py experiments/experiment_code/grid.py sven/sven/__init__.py; do
  [ -e "$TMP/$f" ] || die "export is missing $f -- wrong repo layout?"
done
if [ ! -x "$TMP/tools/worker_pool.sh" ] && [ ! -e "$TMP/tools/worker_pool.sh" ]; then
  say "WARNING: tools/worker_pool.sh is not in this snapshot (not committed yet):"
  say "         campaign jobs run \$SNAPSHOT/tools/worker_pool.sh -- commit tools/ and re-deploy."
fi

date -Is > "$TMP/.deploy_complete"
chmod -R u+rwX "$TMP" 2>/dev/null || true

if [ -e "$SNAP" ]; then
  OLD=$SNAP.superseded.$$
  mv "$SNAP" "$OLD" || die "cannot move the old snapshot aside"
  mv "$TMP" "$SNAP" || die "cannot move the new snapshot into place"
  rm -rf "$OLD"
else
  mv "$TMP" "$SNAP" || die "cannot move the new snapshot into place"
fi
TMP=""
trap - EXIT

say "exported $(du -sh "$SNAP" 2>/dev/null | cut -f1) to:"
say "workers must set PYTHONPATH=$SNAP:$SNAP/sven and cd there"
echo "$SNAP"
