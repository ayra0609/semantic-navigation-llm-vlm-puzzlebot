# Source code of LVM-Nav

This repository contains the source code for LVM-Nav.

The dissertation title is below.

```text
LVM-Nav: A Modular Language-Vision-Motion Pipeline for Semantic Navigation on a ROS2 PuzzleBot
```

LVM-Nav is a semantic robot navigation system. It uses natural language commands and RGB camera images to detect target objects and estimate target distance. Finally, it sends motion commands through ROS2.

Example commands are below.

```text
go to the bag
move to the cup
navigate to the shoe
approach the notebook
stop
```

## Main files

`llm/intent_extraction.py`

This file extracts the target object from a command.
It uses Qwen through Ollama.
It also has a regex fallback.

`vlm/grounding_dino.py`

This file runs Grounding DINO.
It detects objects from text labels.

`ros2/pipeline_node.py`

This is the main PC side ROS2 node.
It receives user commands.
It runs language reasoning.
It sends images to the perception system.
It publishes target labels and detection results.

`ros2/navigation_node.py`

This is the Jetson side navigation node.
It receives target labels and detection results.
It generates `/cmd_vel`.

`ros2/cmd_vel_bridge.py`

This file converts `/cmd_vel` into PuzzleBot wheel commands.
It sends wheel commands by UDP.
It publishes odometry from wheel encoder feedback.

`ncc_inference_server.py`

This file runs the NCC inference server.
It runs Grounding DINO and Depth Anything V2.
It returns detection results and distance estimates.

`web_ui/index.html`

This file provides the browser interface.
It sends commands to ROS2.
It shows robot status and camera frames.

`robot.sh`

This script starts the real robot pipeline.

`tests/test_intent.py`

This script tests the language module with ALFRED data.

`tests/test_vlm.py`

This script tests the vision module with COCO data.

## ROS2 topics

`/llm_command`

User command from the web interface.

`/video_source/raw`

Camera image stream.

`/nav_target`

Target object label.

`/detection_result`

Detection result.

`/cmd_vel`

Robot velocity command.

`/odom`

Odometry feedback.

`/puzzlebot/status`

Robot status message.

## Installation

The PC side used ROS2 Humble.
The Jetson side used ROS2 Foxy.
Python 3.10 is recommended.

Install Python packages.

```bash
pip install transformers torch torchvision opencv-python pillow ollama requests flask numpy
```

Install ROS2 packages.

```bash
sudo apt install -y ros-humble-rosbridge-server ros-humble-rmw-cyclonedds-cpp
```

Install the language model.

```bash
ollama pull qwen2.5:0.5b
```

ALFRED data is not included.
COCO data is not included.

## Real robot run

Start the pipeline on the PC.

```bash
bash robot.sh
```

Use a custom Jetson address if needed.

```bash
bash robot.sh 192.168.1.50
```

Open the web interface.

```text
web_ui/index.html
```

Use this default ROS bridge address.

```text
ws://localhost:9090
```

The launcher expects a Jetson camera node at this path.

```text
ros2/camera_node.py
```

If the camera node is separate, it should publish compressed images here.

```text
/video_source/raw
```

## Evaluation

Test the language module.

```bash
python3 tests/test_intent.py \
  --alfred-path alfred/data/json_2.1.0 \
  --split valid_seen
```

Test the regex fallback.

```bash
python3 tests/test_intent.py \
  --alfred-path alfred/data/json_2.1.0 \
  --split valid_seen \
  --regex-only
```

Test the vision module.

```bash
python3 tests/test_vlm.py --coco-dir coco/ --iou-threshold 0.5
```

Logs are written here.

```text
test_log/
```

SUMMARY OF 75 TRIALS.
```
puzzlebot_test_plan.xlsx
```

