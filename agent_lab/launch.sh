#!/bin/bash
# Launch the autonomous research manager in a tmux session.
#
# Usage:
#   bash agent_lab/launch.sh
#
# This starts a tmux session called "agent-lab" with the manager running.
# You can detach (Ctrl-B, D) and reattach later with:
#   tmux attach -t agent-lab
#
# To check progress without attaching, look at:
#   - agent_lab/taskboard.md    (task statuses)
#   - agent_lab/notes.md        (detailed findings)
#   - agent_lab/report.md       (synthesized conclusions)
#   - agent_lab/results/        (raw experiment data)
#   - agent_lab/plots/          (visualizations)

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Make sure we're on the right branch
git checkout agent-lab

# Read the orchestrator prompt from file
PROMPT=$(cat agent_lab/orchestrator_prompt.md)

# Launch in a new tmux session
tmux new-session -d -s agent-lab -c "$REPO_ROOT" \
  "claude --verbose -p '$PROMPT' 2>&1 | tee agent_lab/logs/session_$(date +%Y%m%d_%H%M%S).log; echo 'Session ended. Press Enter to close.'; read"

echo "Agent lab started in tmux session 'agent-lab'"
echo ""
echo "Commands:"
echo "  tmux attach -t agent-lab     # watch live"
echo "  tmux kill-session -t agent-lab  # stop"
echo ""
echo "Check progress:"
echo "  cat agent_lab/taskboard.md"
echo "  cat agent_lab/report.md"
echo "  tail agent_lab/notes.md"
echo "  ls agent_lab/results/"
echo "  ls agent_lab/plots/"
