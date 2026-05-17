# Semantic Navigation LLM-VLM PuzzleBot

A semantic navigation system for PuzzleBot using LLM intent extraction (Mistral + CoT) and VLM object detection (Grounding DINO), enabling natural language commands such as "Move to the sofa" over ROS2.

---

## Project Structure

```
├── llm/                    # LLM intent extraction (Mistral via Ollama)
├── vlm/                    # VLM object detection (Grounding DINO)
├── ros2/
│   ├── pipeline_node.py    # PC side: LLM + VLM + visual servo
│   └── navigation_node.py  # Jetson side: cmd_vel control
├── tests/
│   ├── test_intent.py      # LLM accuracy evaluation (ALFRED dataset)
│   └── test_vlm.py         # VLM accuracy evaluation (COCO 2017)
├── web_ui/                 # Browser-based command interface
├── launch/                 # ROS2 launch files
├── worlds/                 # Gazebo SDF world files
├── test_log/               # Auto-generated test results
├── project_log/            # Auto-generated simulation logs
├── gazebo.sh               # One-click simulation launcher
└── robot.sh                # One-click real robot launcher
```

---

## Dependencies

```bash
# Python
pip install transformers torch torchvision opencv-python pillow ollama requests beautifulsoup4

# ROS2 packages (Ubuntu / WSL2)
sudo apt install -y ros-humble-rosbridge-server ros-humble-rmw-cyclonedds-cpp

# LLM model
ollama pull mistral
```

---

## 1. Run Simulation (Gazebo)

Launches Gazebo, ros_gz_bridge, pipeline_node, and rosbridge in a tmux session.

```bash
bash gazebo.sh
```

**What it does:**
- Pane 0: Gazebo + ros_gz_bridge
- Pane 1: pipeline_node (LLM + VLM + visual servo) — log saved to `project_log/`
- Pane 2: rosbridge (for Web UI)
- Opens `web_ui/index.html` automatically

**Send a command manually (without Web UI):**
```bash
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 topic pub --once /llm_command std_msgs/String "data: 'Move to the sofa'"
```

**tmux shortcuts:**
```
Ctrl-B D        detach (processes keep running)
Ctrl-B arrow    switch panes
Ctrl-C          stop everything
```

---

## 2. Run Real Robot (PuzzleBot)

Two-machine setup: PC runs LLM/VLM pipeline, Jetson Nano runs navigation controller.

**PC side:**
```bash
bash robot.sh                   # default Jetson IP: 192.168.0.100
bash robot.sh 192.168.1.50      # custom Jetson IP
```

**Jetson Nano side (run manually on Jetson):**
```bash
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
python3 ros2/navigation_node.py
```

**Topic overview:**
| Topic | Direction | Description |
|-------|-----------|-------------|
| `/llm_command` | Web UI → PC | Natural language command |
| `/video_source/raw` | Jetson → PC | Camera stream |
| `/nav_target` | PC → Jetson | Extracted target label |
| `/detection_result` | PC → Jetson | Bounding box JSON |
| `/cmd_vel` | Jetson internal | Motor commands |

---

## 3. Test LLM (ALFRED Dataset)

Evaluates Mistral + CoT intent extraction on 100 ALFRED `valid_seen` instructions.
Pass threshold: **85% accuracy**.

**Prerequisites:** Download ALFRED annotation data first (one-time, ~500 MB).

```bash
# Download (one-time, ~500 MB — JSON annotations only, no images needed)
git clone https://github.com/askforalfred/alfred.git
cd alfred
bash data/download_data.sh json
# Data lands at alfred/data/json_2.1.0/
cd ..
```

> Note: ignore the `torch==1.1.0` error from `requirements.txt` — do NOT run
> `pip install -r requirements.txt`. The download script only needs bash + wget.

```bash
# Full evaluation (requires Ollama running)
python3 tests/test_intent.py \
  --alfred-path "alfred/data/json_2.1.0" \
  --split valid_seen

# Regex fallback only (no Ollama needed)
python3 tests/test_intent.py \
  --alfred-path "alfred/data/json_2.1.0" \
  --split valid_seen \
  --regex-only

# Other splits
python3 tests/test_intent.py \
  --alfred-path "alfred/data/json_2.1.0" \
  --split valid_unseen
```

**Output:** One summary line in terminal. Full results saved to:
```
test_log/alfred_<split>_<YYYYMMDDHHMMSS>.log
```

**Log contains:** per-sample instruction / GT label / LLM raw JSON output / predicted label / latency.

---

## 4. Test VLM (COCO 2017 Dataset)

Evaluates Grounding DINO on 100 COCO 2017 indoor scene images.
Pass threshold: **75% accuracy** at IoU ≥ 0.5.

**Prerequisites:** Download COCO val2017 data first.

```bash
# Download (one-time, ~1.2 GB total)
mkdir -p coco/images coco/annotations
cd coco
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d images/ && rm val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip && rm annotations_trainval2017.zip
cd ..
```

```bash
# Run evaluation
python3 tests/test_vlm.py --coco-dir coco/

# Custom IoU threshold
python3 tests/test_vlm.py --coco-dir coco/ --iou-threshold 0.5
```

**Output:** One summary line in terminal. Full results saved to:
```
test_log/coco_val2017_<YYYYMMDDHHMMSS>.log
```

**Log contains:** per-image category / found / confidence score / predicted box / best IoU / per-category accuracy breakdown.

**Note:** Grounding DINO runs on CPU. 100 images takes approximately 20–30 minutes.

---

## Evaluation Results

| Module | Dataset | Metric | Result | Threshold | Status |
|--------|---------|--------|--------|-----------|--------|
| LLM (Mistral + CoT) | ALFRED valid_seen (100 samples) | Target label accuracy | **95.0%** | 85% | PASS |
| VLM (Grounding DINO tiny) | COCO 2017 val2017 (100 samples) | IoU ≥ 0.5 accuracy | **82.0%** | 75% | PASS |

**VLM per-category breakdown:**

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| chair | 16 | 20 | 80.0% |
| couch | 13 | 15 | 86.7% |
| dining table | 18 | 20 | 90.0% |
| bed | 8 | 15 | 53.3% |
| toilet | 8 | 10 | 80.0% |
| tv | 10 | 10 | 100.0% |
| sink | 4 | 5 | 80.0% |
| refrigerator | 5 | 5 | 100.0% |

---

## Acknowledgements

- **PuzzleBot ROS2 package** — [ManchesterRoboticsLtd/puzzlebot_ros](https://github.com/ManchesterRoboticsLtd/puzzlebot_ros) (MIT License, © 2023 Manchester-Robotics). Used as the base ROS2 driver for PuzzleBot hardware control.
- **Grounding DINO** — [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- **ALFRED dataset** — [askforalfred/alfred](https://github.com/askforalfred/alfred)
