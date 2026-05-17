#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  start.sh  —  PuzzleBot LLM-VLM Simulation  (one-click launch)
#  Usage:  bash start.sh
# ═══════════════════════════════════════════════════════════════

set -eo pipefail

# ── Colour helpers ───────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
section() { echo -e "\n${BOLD}${BLUE}══ $* ══${NC}\n"; }

# ── Project root = directory containing this script ─────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLD_FILE="$PROJECT_ROOT/worlds/sim_world.sdf"
LAUNCH_FILE="$PROJECT_ROOT/launch/sim_launch.py"
PIPELINE_NODE="$PROJECT_ROOT/ros2/pipeline_node.py"
WEB_UI="$PROJECT_ROOT/web_ui/index.html"
ROS_SETUP="/opt/ros/humble/setup.bash"
SESSION="puzzlebot"

# ── Fix WSL2 DDS discovery: use CycloneDDS (works reliably on WSL2)
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# ── Cleanup on exit ──────────────────────────────────────────────
cleanup() {
  echo ""
  warn "Shutting down all processes…"
  # Kill tmux session (kills all panes)
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  # Kill any stray ROS2 / ign processes we may have started
  pkill -f "ign gazebo"      2>/dev/null || true
  pkill -f "ros2 launch"     2>/dev/null || true
  pkill -f "pipeline_node"   2>/dev/null || true
  pkill -f "rosbridge"       2>/dev/null || true
  ok "All stopped. Bye!"
}
trap cleanup EXIT INT TERM

# ════════════════════════════════════════════════════════════════
section "Pre-flight checks"
# ════════════════════════════════════════════════════════════════

FAIL=0

check() {
  local label="$1"; shift
  if "$@" &>/dev/null; then
    ok "$label"
  else
    error "$label  →  not found / failed"
    FAIL=1
  fi
}

check "ROS2 Humble setup.bash"   test -f "$ROS_SETUP"
check "worlds/sim_world.sdf"     test -f "$WORLD_FILE"
check "launch/sim_launch.py"     test -f "$LAUNCH_FILE"
check "ros2/pipeline_node.py"    test -f "$PIPELINE_NODE"
check "web_ui/index.html"        test -f "$WEB_UI"
check "tmux installed"           command -v tmux
check "python3 installed"        command -v python3

# Source ROS2
# shellcheck source=/dev/null
source "$ROS_SETUP"
check "ros2 CLI available"           command -v ros2
check "ign gazebo available"         command -v ign
check "ros_gz_bridge installed"      ros2 pkg list | grep -q ros_gz_bridge
check "rosbridge_server installed"   ros2 pkg list | grep -q rosbridge_server

# Python dependencies
check "transformers installed"   python3 -c "import transformers"
check "torch installed"          python3 -c "import torch"
check "opencv installed"         python3 -c "import cv2"
check "PIL installed"            python3 -c "from PIL import Image"

if [[ $FAIL -ne 0 ]]; then
  error "Fix the items above, then re-run."
  echo ""
  echo "  ROS2 packages:  sudo apt install -y ros-humble-rosbridge-server ros-humble-rmw-cyclonedds-cpp"
  echo "  Python deps:    pip install transformers torch torchvision opencv-python pillow ollama"
  exit 1
fi

# ── Ollama (optional — regex fallback used if unavailable) ───────
if command -v ollama &>/dev/null; then
  if ! pgrep -x ollama &>/dev/null; then
    info "Starting Ollama service…"
    ollama serve &>/dev/null &
    sleep 2
  fi
  if ollama list 2>/dev/null | grep -q mistral; then
    ok "Ollama + Mistral ready (LLM mode)"
  else
    warn "Mistral not found — run: ollama pull mistral"
    warn "Falling back to regex intent extraction"
  fi
else
  warn "Ollama not installed — using regex intent extraction"
  warn "Install: curl -fsSL https://ollama.ai/install.sh | sh && ollama pull mistral"
fi

ok "All checks passed."

# ════════════════════════════════════════════════════════════════
section "Launching simulation"
# ════════════════════════════════════════════════════════════════

# Kill stale session if it exists
tmux kill-session -t "$SESSION" 2>/dev/null || true

# ── Create timestamped log file for pipeline_node output ────────────
LOG_FILE="$PROJECT_ROOT/project_log/$(date +%Y%m%d%H%M).log"
mkdir -p "$PROJECT_ROOT/project_log"
info "Pipeline log → $LOG_FILE"

# Create a new tmux session (detached) with 3 panes
#
#  ┌──────────────────────┬──────────────────────┐
#  │  PANE 0              │  PANE 1              │
#  │  Gazebo + bridge     │  pipeline_node       │
#  │  (ros2 launch)       │  LLM + VLM + servo   │
#  ├──────────────────────┴──────────────────────┤
#  │  PANE 2                                     │
#  │  rosbridge  (for Web UI)                    │
#  └─────────────────────────────────────────────┘

info "Creating tmux session '${SESSION}'…"

# Pane 0 — base pane
tmux new-session -d -s "$SESSION" -x 220 -y 50 \
     -e "ROS_SETUP=$ROS_SETUP" \
     -e "PROJECT_ROOT=$PROJECT_ROOT"

# Split right → Pane 1
tmux split-window -h -t "$SESSION:0"

# Split bottom of full width → Pane 2
tmux split-window -v -t "$SESSION:0.0"   # split pane-0 vertically → pane 2 below

# Even the layout
tmux select-layout -t "$SESSION" main-vertical 2>/dev/null || true

# ── Labels (title in each pane) ──────────────────────────────────
title() { tmux send-keys -t "$SESSION:0.$1" "printf '\033]2;$2\033\\'" Enter; }
title 0 "🌍 Gazebo + Bridge"
title 1 "🤖 Pipeline Node"
title 2 "🌐 Rosbridge"

# ── PANE 0: Gazebo simulation + ros_gz_bridge ────────────────────
tmux send-keys -t "$SESSION:0.0" \
  "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   source $ROS_SETUP && \
   echo -e '\033[1;36m[PANE 0] Starting Gazebo + ros_gz_bridge…\033[0m' && \
   sleep 1 && \
   ros2 launch $LAUNCH_FILE" \
  Enter

# ── PANE 2: rosbridge (for Web UI) ──────────────────────────────
tmux send-keys -t "$SESSION:0.2" \
  "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   source $ROS_SETUP && \
   echo -e '\033[1;36m[PANE 2] Waiting 8s then starting rosbridge…\033[0m' && \
   sleep 8 && \
   ros2 launch rosbridge_server rosbridge_websocket_launch.xml" \
  Enter

# ── PANE 1: pipeline_node (LLM + VLM + visual servo) ────────────
tmux send-keys -t "$SESSION:0.1" \
  "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   source $ROS_SETUP && \
   cd $PROJECT_ROOT && \
   echo -e '\033[1;36m[PANE 1] Waiting 20s for Gazebo to start…\033[0m' && \
   sleep 20 && \
   echo -e '\033[1;32m[PANE 1] Starting pipeline_node — log: $LOG_FILE\033[0m' && \
   python3 ros2/pipeline_node.py --ros-args -p sim_mode:=true 2>&1 | tee $LOG_FILE" \
  Enter

# ════════════════════════════════════════════════════════════════
section "Opening Web UI"
# ════════════════════════════════════════════════════════════════

# Try to open the browser automatically
if command -v xdg-open &>/dev/null; then
  info "Opening web_ui/index.html in browser…"
  xdg-open "$WEB_UI" &
elif command -v firefox &>/dev/null; then
  firefox "$WEB_UI" &
elif command -v google-chrome &>/dev/null; then
  google-chrome "$WEB_UI" &
else
  warn "Could not auto-open browser."
  echo "  Open manually:  $WEB_UI"
fi

# ════════════════════════════════════════════════════════════════
section "All done — attaching to tmux"
# ════════════════════════════════════════════════════════════════

echo ""
echo -e "  ${BOLD}Web UI${NC}     →  open ${CYAN}web_ui/index.html${NC} in browser"
echo -e "  ${BOLD}Command${NC}    →  publish to ${CYAN}/llm_command${NC}  or type in the Web UI"
echo ""
echo -e "  ${YELLOW}Press Ctrl-B then D to detach from tmux (processes keep running).${NC}"
echo -e "  ${YELLOW}Press Ctrl-C here to stop everything.${NC}"
echo ""

# Attach to the tmux session so the user can see all panes
tmux select-pane -t "$SESSION:0.1"   # focus pipeline pane on attach
tmux attach-session -t "$SESSION"
