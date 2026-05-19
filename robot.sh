#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  robot.sh  —  PuzzleBot LLM-VLM  (full pipeline launcher)
#
#  PC side:
#    PANE 0: rosbridge       — Web UI ↔ ROS2
#    PANE 1: pipeline_node   — LLM + VLM (calls NCC for GPU inference)
#    PANE 2: camera_node     — Jetson CSI camera
#    PANE 3: cmd_vel_bridge  — Jetson wheel control
#    PANE 4: nav_node        — Jetson navigation
#    PANE 5: ncc_server      — NCC GPU inference (DINO + Depth Anything)
#    PANE 6: ssh_tunnel      — SSH tunnel PC:5001 → NCC GPU node:5001
#
#  Usage:  bash robot.sh [JETSON_IP]
#    JETSON_IP defaults to 172.20.10.2
# ═══════════════════════════════════════════════════════════════

set -e

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

LOGDIR="$PROJECT_ROOT/project_log"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/$(date +%Y%m%d%H%M).log"

JETSON_IP="${1:-172.20.10.2}"
JETSON_USER="jetson"
JETSON_PROJECT="/home/jetson/semantic_navigation"
JETSON_ROS_SETUP="/opt/ros/foxy/setup.bash"

# NCC settings
NCC_USER="bnwp12"
NCC_HOST="ncc1.clients.dur.ac.uk"
NCC_SCRIPT="/home3/bnwp12/project/ncc_inference_server.py"
NCC_PYTHON="/home3/bnwp12/project/ncc_venv/bin/python"
NCC_PORT=5001
NCC_GPU=0
NCC_JOB_NAME="puzzlebot_ncc"
NCC_PARTITION="tpg-gpu-small"
NCC_GRES="gpu:pascal:1"

unset RMW_IMPLEMENTATION
export FASTRTPS_DEFAULT_PROFILES_FILE="$PROJECT_ROOT/fastdds_tailscale.xml"
export RCUTILS_COLORIZED_OUTPUT=0

cleanup() {
  warn "Shutting down…"
  # Kill SSH tunnel if running
  pkill -f "ssh.*-L ${NCC_PORT}:.*:${NCC_PORT}.*${NCC_HOST}" 2>/dev/null || true
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
check "rosbridge_server"      bash -c "ros2 pkg list | grep -q rosbridge_server"
check "transformers"          python3 -c "import transformers"
check "torch"                 python3 -c "import torch"
check "opencv"                python3 -c "import cv2"

# Jetson SSH check
info "Checking SSH to Jetson (${JETSON_USER}@${JETSON_IP})…"
if ssh -o ConnectTimeout=5 -o BatchMode=yes \
       "${JETSON_USER}@${JETSON_IP}" exit 2>/dev/null; then
  ok "Jetson SSH"
else
  error "Cannot SSH to Jetson — run: ssh-copy-id ${JETSON_USER}@${JETSON_IP}"
  FAIL=1
fi

# NCC SSH check
info "Checking SSH to NCC (${NCC_USER}@${NCC_HOST})…"
if ssh -o ConnectTimeout=5 -o BatchMode=yes \
       "${NCC_USER}@${NCC_HOST}" exit 2>/dev/null; then
  ok "NCC SSH"
else
  error "Cannot SSH to NCC — run: ssh-copy-id ${NCC_USER}@${NCC_HOST}"
  FAIL=1
fi

# NCC script check
info "Checking NCC inference script…"
if ssh -o ConnectTimeout=5 -o BatchMode=yes \
       "${NCC_USER}@${NCC_HOST}" "test -f ${NCC_SCRIPT}" 2>/dev/null; then
  ok "NCC script found: ${NCC_SCRIPT}"
else
  error "NCC script not found: ${NCC_SCRIPT}"
  error "Upload ncc_inference_server.py to NCC first"
  FAIL=1
fi

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

info "Jetson IP:  ${JETSON_IP}"
info "NCC Host:   ${NCC_HOST}  (${NCC_GRES}, port ${NCC_PORT})"
info "RMW:        ${RMW_IMPLEMENTATION}"
ok "All checks passed."
info "Log  →  $LOGFILE"

# ════════════════════════════════════════════════════════════════
section "Launching pipeline"
# ════════════════════════════════════════════════════════════════

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 240 -y 55

# Layout: 7 panes
#   PANE 0: rosbridge      (PC)
#   PANE 1: pipeline_node  (PC)
#   PANE 2: camera_node    (Jetson)
#   PANE 3: cmd_vel_bridge (Jetson)
#   PANE 4: nav_node       (Jetson)
#   PANE 5: ncc_server     (NCC GPU)
#   PANE 6: ssh_tunnel     (PC — tunnel to NCC GPU node)
tmux split-window -h -t "$SESSION:0"
tmux split-window -v -t "$SESSION:0.0"
tmux split-window -v -t "$SESSION:0.1"
tmux split-window -h -t "$SESSION:0.2"
tmux split-window -v -t "$SESSION:0.4"
tmux split-window -v -t "$SESSION:0.5"

JENV="source ${JETSON_ROS_SETUP} && export ROS_DOMAIN_ID=42 && export FASTRTPS_DEFAULT_PROFILES_FILE=/home/jetson/fastdds_tailscale.xml && export RCUTILS_COLORIZED_OUTPUT=0 && cd ${JETSON_PROJECT}"
SSH_OPTS="-o ServerAliveInterval=10 -o ServerAliveCountMax=6 -o ConnectTimeout=10"

STRIP_ANSI="sed 's/\x1b\[[0-9;]*[mGKHFJK]//g'"

# ── PANE 0: rosbridge ────────────────────────────────────────────
tmux send-keys -t "$SESSION:0.0" \
  "source $ROS_SETUP && echo -e '\033[1;36m[PANE 0] rosbridge starting…\033[0m' && ros2 launch rosbridge_server rosbridge_websocket_launch.xml 2>&1 | $STRIP_ANSI | awk '{print \"[rosbridge] \" \$0; fflush()}' | tee -a $LOGFILE" \
  Enter

# ── PANE 1: pipeline_node (PC — LLM + VLM, calls NCC) ───────────
tmux send-keys -t "$SESSION:0.1" \
  "source $ROS_SETUP && cd $PROJECT_ROOT && echo -e '\033[1;36m[PANE 1] Waiting 8s for rosbridge + NCC tunnel…\033[0m' && sleep 8 && echo -e '\033[1;32m[PANE 1] pipeline_node starting…\033[0m' && python3 ros2/pipeline_node.py --ros-args -p sim_mode:=false 2>&1 | $STRIP_ANSI | awk '{print \"[pipeline] \" \$0; fflush()}' | tee -a $LOGFILE" \
  Enter

# ── PANE 2: SSH → Jetson camera_node ────────────────────────────
tmux send-keys -t "$SESSION:0.2" \
  "echo -e '\033[1;36m[PANE 2] Jetson camera_node…\033[0m' && sleep 3 && ssh -tt $SSH_OPTS ${JETSON_USER}@${JETSON_IP} '${JENV} && echo [JETSON] camera_node starting && python3 ros2/camera_node.py --ros-args --remap /camera/image_raw/compressed:=/video_source/raw' 2>&1 | $STRIP_ANSI | awk '{print \"[camera] \" \$0; fflush()}' | tee -a $LOGFILE" \
  Enter

# ── PANE 3: SSH → Jetson cmd_vel_bridge ─────────────────────────
tmux send-keys -t "$SESSION:0.3" \
  "echo -e '\033[1;36m[PANE 3] Jetson cmd_vel_bridge…\033[0m' && sleep 4 && ssh -tt $SSH_OPTS ${JETSON_USER}@${JETSON_IP} '${JENV} && echo [JETSON] cmd_vel_bridge starting && python3 ros2/cmd_vel_bridge.py' 2>&1 | $STRIP_ANSI | awk '{print \"[cmdvel] \" \$0; fflush()}' | tee -a $LOGFILE" \
  Enter

# ── PANE 4: SSH → Jetson navigation_node (auto-restart on crash) ─
tmux send-keys -t "$SESSION:0.4" \
  "echo -e '\033[1;36m[PANE 4] Jetson navigation_node…\033[0m' && sleep 5 && ssh -t $SSH_OPTS ${JETSON_USER}@${JETSON_IP} '${JENV} && echo [JETSON] navigation_node starting && until python3 ros2/navigation_node.py; do echo [JETSON] nav_node crashed, restarting in 3s...; sleep 3; done' 2>&1 | $STRIP_ANSI | awk '{print \"[nav_node] \" \$0; fflush()}' | tee -a $LOGFILE" \
  Enter

# ── PANE 5: SSH → NCC GPU inference server ───────────────────────
tmux send-keys -t "$SESSION:0.5" \
  "echo -e '\033[1;35m[PANE 5] NCC GPU inference server starting…\033[0m' && sleep 2 && ssh -tt $SSH_OPTS ${NCC_USER}@${NCC_HOST} 'srun --job-name=${NCC_JOB_NAME} -c 1 --gres=${NCC_GRES} --partition=${NCC_PARTITION} --pty bash -c \"${NCC_PYTHON} ${NCC_SCRIPT} --port ${NCC_PORT} --gpu ${NCC_GPU}\"' 2>&1 | $STRIP_ANSI | awk '{print \"[ncc] \" \$0; fflush()}' | tee -a $LOGFILE" \
  Enter

# ── PANE 6: SSH tunnel PC:5001 → NCC GPU node:5001 ────────────────
# Wait for Slurm to allocate a GPU node, then tunnel through the login host.
tmux send-keys -t "$SESSION:0.6" \
  "echo -e '\033[1;35m[PANE 6] SSH tunnel to NCC GPU node…\033[0m' && NODE='' && until [ -n \"\$NODE\" ]; do NODE=\$(ssh $SSH_OPTS ${NCC_USER}@${NCC_HOST} \"squeue -h -u ${NCC_USER} -n ${NCC_JOB_NAME} -t R -o '%N' | head -n 1\" 2>/dev/null); [ -n \"\$NODE\" ] || { echo '[tunnel] waiting for Slurm GPU node...'; sleep 5; }; done && echo -e \"\033[1;32m[PANE 6] Opening tunnel localhost:${NCC_PORT} → \${NODE}:${NCC_PORT}\033[0m\" && until ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 $SSH_OPTS -N -L ${NCC_PORT}:\${NODE}:${NCC_PORT} ${NCC_USER}@${NCC_HOST}; do echo '[tunnel] reconnecting in 5s...'; sleep 5; done 2>&1 | $STRIP_ANSI | awk '{print \"[tunnel] \" \$0; fflush()}' | tee -a $LOGFILE" \
  Enter

# ════════════════════════════════════════════════════════════════
section "Opening Web UI"
# ════════════════════════════════════════════════════════════════

if grep -qi microsoft /proc/version 2>/dev/null && command -v explorer.exe &>/dev/null; then
  explorer.exe "$(wslpath -w "$WEB_UI")" &
elif command -v xdg-open &>/dev/null; then
  xdg-open "$WEB_UI" &
elif command -v firefox &>/dev/null; then
  firefox "$WEB_UI" &
else
  warn "Open manually: $WEB_UI"
fi

echo ""
echo -e "  ${BOLD}PANE 0${NC}  rosbridge         (PC)"
echo -e "  ${BOLD}PANE 1${NC}  pipeline_node     (PC — LLM + calls NCC)"
echo -e "  ${BOLD}PANE 2${NC}  camera_node       (Jetson)"
echo -e "  ${BOLD}PANE 3${NC}  cmd_vel_bridge    (Jetson)"
echo -e "  ${BOLD}PANE 4${NC}  navigation_node   (Jetson)"
echo -e "  ${BOLD}PANE 5${NC}  ncc_server        (NCC — ${NCC_GRES})"
echo -e "  ${BOLD}PANE 6${NC}  ssh_tunnel        (PC → NCC GPU node port ${NCC_PORT})"
echo ""
echo -e "  ${YELLOW}NCC server needs ~30s to load models after GPU allocation.${NC}"
echo -e "  ${YELLOW}Ctrl-C to stop everything (including NCC and Jetson).${NC}"
echo ""

tmux select-pane -t "$SESSION:0.1"
tmux attach-session -t "$SESSION"
