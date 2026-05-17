#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  robot.sh  —  PuzzleBot LLM-VLM  (real robot, PC side)
#
#  Architecture:
#    PC (this script):
#      • rosbridge  — Web UI ↔ ROS2
#      • pipeline_node (sim_mode=False)
#          SUB /video_source/raw  ← Jetson camera
#          PUB /nav_target        → Jetson navigation_node
#          PUB /detection_result  → Jetson navigation_node
#
#    Jetson Nano (start manually):
#      source /opt/ros/humble/setup.bash
#      python3 ros2/navigation_node.py
#
#  Usage:  bash robot.sh [JETSON_IP]
#    JETSON_IP defaults to 192.168.0.100
# ═══════════════════════════════════════════════════════════════

set -eo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
section() { echo -e "\n${BOLD}${BLUE}══ $* ══${NC}\n"; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_NODE="$PROJECT_ROOT/ros2/pipeline_node.py"
WEB_UI="$PROJECT_ROOT/web_ui/index.html"
ROS_SETUP="/opt/ros/humble/setup.bash"
SESSION="puzzlebot_robot"
JETSON_IP="${1:-192.168.0.100}"

# ── Fix WSL2 DDS discovery ───────────────────────────────────────
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

cleanup() {
  warn "Shutting down…"
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  ok "Stopped. Bye!"
}
trap cleanup EXIT INT TERM

# ════════════════════════════════════════════════════════════════
section "Pre-flight checks"
# ════════════════════════════════════════════════════════════════

FAIL=0
check() {
  local label="$1"; shift
  if "$@" &>/dev/null; then ok "$label"
  else error "$label  →  not found"; FAIL=1; fi
}

source "$ROS_SETUP"
check "ROS2 Humble"           test -f "$ROS_SETUP"
check "ros2/pipeline_node.py" test -f "$PIPELINE_NODE"
check "web_ui/index.html"     test -f "$WEB_UI"
check "tmux"                  command -v tmux
check "python3"               command -v python3
check "rosbridge_server"      ros2 pkg list | grep -q rosbridge_server
check "transformers"          python3 -c "import transformers"
check "torch"                 python3 -c "import torch"
check "opencv"                python3 -c "import cv2"

[[ $FAIL -ne 0 ]] && {
  error "Fix the items above."
  echo "  pip install transformers torch torchvision opencv-python pillow ollama"
  exit 1
}

# Ollama
if command -v ollama &>/dev/null; then
  pgrep -x ollama &>/dev/null || { ollama serve &>/dev/null & sleep 2; }
  ollama list 2>/dev/null | grep -q mistral \
    && ok "Ollama + Mistral ready" \
    || warn "Mistral not found — regex fallback active"
else
  warn "Ollama not installed — regex fallback active"
fi

info "Jetson IP: ${JETSON_IP}"
info "Camera topic: /video_source/raw (CompressedImage from Jetson)"
info "RMW: ${RMW_IMPLEMENTATION}"
ok "All checks passed."

# ════════════════════════════════════════════════════════════════
section "Launching robot pipeline"
# ════════════════════════════════════════════════════════════════

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 220 -y 50
tmux split-window -h -t "$SESSION:0"

# ── PANE 0: rosbridge ────────────────────────────────────────────
tmux send-keys -t "$SESSION:0.0" \
  "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   source $ROS_SETUP && \
   echo -e '\033[1;36m[PANE 0] Starting rosbridge…\033[0m' && \
   ros2 launch rosbridge_server rosbridge_websocket_launch.xml" \
  Enter

# ── PANE 1: pipeline_node (real robot mode) ──────────────────────
tmux send-keys -t "$SESSION:0.1" \
  "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   source $ROS_SETUP && \
   cd $PROJECT_ROOT && \
   echo -e '\033[1;36m[PANE 1] Waiting 5s for rosbridge…\033[0m' && \
   sleep 5 && \
   echo -e '\033[1;32m[PANE 1] Starting pipeline_node (robot mode)…\033[0m' && \
   python3 ros2/pipeline_node.py --ros-args -p sim_mode:=false" \
  Enter

# ════════════════════════════════════════════════════════════════
section "Opening Web UI"
# ════════════════════════════════════════════════════════════════

if command -v xdg-open &>/dev/null; then
  xdg-open "$WEB_UI" &
elif command -v firefox &>/dev/null; then
  firefox "$WEB_UI" &
else
  warn "Open manually: $WEB_UI"
fi

echo ""
echo -e "  ${BOLD}PC side${NC}     → rosbridge + LLM/VLM pipeline (this machine)"
echo -e "  ${BOLD}Jetson side${NC} → run on Jetson Nano:"
echo -e "    ${CYAN}source /opt/ros/humble/setup.bash${NC}"
echo -e "    ${CYAN}export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp${NC}"
echo -e "    ${CYAN}python3 ros2/navigation_node.py${NC}"
echo ""
echo -e "  ${YELLOW}Ctrl-C to stop everything.${NC}"
echo ""

tmux select-pane -t "$SESSION:0.1"
tmux attach-session -t "$SESSION"
