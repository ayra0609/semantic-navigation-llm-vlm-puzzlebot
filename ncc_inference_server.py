#!/usr/bin/env python3
"""
ncc_inference_server.py  —  GPU inference server for PuzzleBot LLM-VLM pipeline

Runs on Durham NCC (Newton Computing Cluster) under Slurm.
Started by robot.sh via:
  srun --gres=gpu:pascal:1 --partition=tpg-gpu-small \
       python ncc_inference_server.py --port 5001 --gpu 0

Endpoint
--------
POST /detect
  Form fields:  target (str), image (JPEG file)
  Response:     JSON (see _detect)

Distance estimation
-------------------
Primary:  ground-plane geometry
            d = h_cam * f_y / (y2_px - y_horizon)
  where y2 = bottom edge of DINO bounding box (object base on floor),
        y_horizon = image row of the geometric horizon given camera pitch,
        h_cam     = camera height above floor (m),
        f_y       = vertical focal length (px).

Fallback: height-based pinhole  d = H_obj * f_y / (y2 - y1)
  Used when the ground-plane denominator is too small (far/ambiguous object).
"""

import argparse
import io
import math
import time
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import (AutoImageProcessor, AutoModelForDepthEstimation,
                          AutoModelForZeroShotObjectDetection, AutoProcessor)

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────
# Camera geometry  (PuzzleBot, Jetson camera, fixed mount)
# Measure h_cam with a ruler from the floor to the lens.
# Set pitch_deg > 0 if the camera tilts downward.
# ──────────────────────────────────────────────────────────
CAM_HEIGHT_M  = 0.222  # m  — camera height above floor, measured 2026-05-20
CAM_PITCH_DEG = 12.7   # °  — positive = tilting UP (horizon moves below image center)
                        # calibrated 2026-05-20: reported 0.68m vs actual 2.204m → 12.7°

# Pinhole parameters (intrinsics from camera calibration)
FOCAL_PX     = 644.69  # f_x ≈ f_y (assume square pixels)
FOCAL_Y_PX   = 644.69  # f_y (set independently if camera is non-square)

# Fallback height prior for chair-like objects (seat height 0.43–0.48 m)
CHAIR_HEIGHT_M = 0.45

# Grounding DINO
DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DINO_THRESHOLD = 0.25

# Depth Anything V2
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# ──────────────────────────────────────────────────────────
# Distance helpers
# ──────────────────────────────────────────────────────────

def _ground_plane_dist(y2_px, img_h,
                        cam_h=CAM_HEIGHT_M,
                        pitch_deg=CAM_PITCH_DEG,
                        fy=FOCAL_Y_PX):
    # type: (float, int, float, float, float) -> Optional[float]
    """Horizontal distance to a ground-contact point at image row y2_px.

    Geometry: camera at height h, tilted UP by θ from horizontal.
    The geometric horizon appears BELOW image centre:
        y_horizon = cy + fy * tan(θ)   [θ > 0 → tilting up → horizon moves down]
        d         = h_cam * fy / (y2 - y_horizon)

    Returns None when y2 is at or above the horizon (object too far or bad bbox).
    """
    cy        = img_h / 2.0
    y_horizon = cy + fy * math.tan(math.radians(pitch_deg))
    y_below   = y2_px - y_horizon
    if y_below < 10:
        return None
    return cam_h * fy / y_below


def _height_based_dist(y1_px, y2_px,
                        obj_h=CHAIR_HEIGHT_M,
                        fy=FOCAL_Y_PX):
    # type: (float, float, float, float) -> Optional[float]
    """Pinhole distance using known object height.
    d = H_real * fy / (y2 - y1)
    """
    bbox_h = y2_px - y1_px
    if bbox_h < 5:
        return None
    return obj_h * fy / bbox_h


def estimate_distance(box, img_h):
    # type: (List[float], int) -> Tuple[Optional[float], str]
    """Return (distance_m, method_name) using best available method."""
    x1, y1, x2, y2 = box

    # Primary: ground-plane geometry
    d = _ground_plane_dist(y2, img_h)
    if d is not None:
        return d, "ground_plane"

    # Secondary: height-based pinhole (chair seat ≈ 0.45 m)
    d = _height_based_dist(y1, y2)
    if d is not None:
        return d, "height_pinhole"

    return None, "none"


# ──────────────────────────────────────────────────────────
# Model loading (lazy, cached)
# ──────────────────────────────────────────────────────────

_dino_processor = _dino_model = _dino_device = None
_depth_processor = _depth_model = None


def _load_dino():
    global _dino_processor, _dino_model, _dino_device
    if _dino_model is None:
        t0 = time.time()
        print("[SERVER] Loading Grounding DINO...", flush=True)
        _dino_device = "cuda" if torch.cuda.is_available() else "cpu"
        _dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
        _dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            DINO_MODEL_ID).to(_dino_device)
        _dino_model.eval()
        print(f"[SERVER] DINO ready in {(time.time()-t0)*1000:.0f}ms", flush=True)


def _load_depth():
    global _depth_processor, _depth_model
    if _depth_model is None:
        t0 = time.time()
        print("[SERVER] Loading Depth Anything V2...", flush=True)
        _depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        _depth_model = AutoModelForDepthEstimation.from_pretrained(
            DEPTH_MODEL_ID).to(_dino_device)
        _depth_model.eval()
        print(f"[SERVER] Depth Anything ready on {_dino_device} in {(time.time()-t0)*1000:.0f}ms", flush=True)


# ──────────────────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────────────────

def _run_dino(pil_img, target):
    text   = f"{target}."
    inputs = _dino_processor(images=pil_img, text=text,
                              return_tensors="pt").to(_dino_device)
    with torch.no_grad():
        outputs = _dino_model(**inputs)
    # transformers <4.38 uses box_threshold; >=4.38 accepts threshold as alias
    try:
        results = _dino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=DINO_THRESHOLD, text_threshold=DINO_THRESHOLD,
            target_sizes=[pil_img.size[::-1]])
    except TypeError:
        results = _dino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=DINO_THRESHOLD, text_threshold=DINO_THRESHOLD,
            target_sizes=[pil_img.size[::-1]])
    boxes  = results[0]["boxes"]
    scores = results[0]["scores"]
    if len(scores) == 0:
        return {"found": False}
    idx   = scores.argmax()
    box   = boxes[idx].tolist()
    score = scores[idx].item()
    x1, y1, x2, y2 = box
    return {
        "found":        True,
        "score":        round(score, 3),
        "box":          [round(v, 1) for v in box],
        "center_x":     round((x1 + x2) / 2, 1),
        "center_y":     round((y1 + y2) / 2, 1),
        "image_width":  pil_img.size[0],
        "image_height": pil_img.size[1],
    }


# ──────────────────────────────────────────────────────────
# Depth Anything V2 helper
# ──────────────────────────────────────────────────────────

def _get_depth_map(pil_img):
    inputs = _depth_processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(_dino_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _depth_model(**inputs)
    import torch.nn.functional as F
    depth = F.interpolate(
        outputs.predicted_depth.unsqueeze(1),
        size=pil_img.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    return depth


def _sample_bbox_depth(depth_map, box):
    """Median relative depth in a 30×30 ROI at the bbox centre."""
    x1, y1, x2, y2 = [int(v) for v in box]
    cx = max(15, min(depth_map.shape[1] - 15, (x1 + x2) // 2))
    cy = max(15, min(depth_map.shape[0] - 15, (y1 + y2) // 2))
    roi = depth_map[cy - 15:cy + 15, cx - 15:cx + 15]
    return float(np.median(roi))


# Per-session depth scale: absolute_dist / relative_depth_value.
# Updated whenever ground-plane gives a valid distance.
# Lets Depth Anything bridge the gap when the camera angle pushes the floor
# contact point below the image bottom (<1.63 m for this camera setup).
_depth_scale = None  # type: Optional[float]


# ──────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/detect", methods=["POST"])
def _detect():
    global _depth_scale

    target = request.form.get("target", "object")
    img_file = request.files.get("image")
    if img_file is None:
        return jsonify({"error": "no image"}), 400

    pil = Image.open(io.BytesIO(img_file.read())).convert("RGB")

    # ── DINO detection ────────────────────────────────────
    t0 = time.time()
    det = _run_dino(pil, target)
    dino_ms = (time.time() - t0) * 1000

    if not det["found"]:
        return jsonify({"found": False, "target": target,
                        "dino_ms": round(dino_ms)})

    # ── Depth Anything V2 ─────────────────────────────────
    t1 = time.time()
    try:
        depth_map = _get_depth_map(pil)
        d_rel = _sample_bbox_depth(depth_map, det["box"])
    except Exception:
        depth_map = None
        d_rel = 0.0
    depth_ms = (time.time() - t1) * 1000

    # ── Distance estimation ───────────────────────────────
    box   = det["box"]
    img_h = pil.size[1]

    # Primary: ground-plane geometry (valid when floor contact is on-screen)
    d_gp, gp_method = estimate_distance(box, img_h)

    dist_m      = None
    dist_method = "none"

    if d_gp is not None:
        dist_m      = d_gp
        dist_method = gp_method
        # Calibrate depth scale while we have a reliable anchor
        if d_rel > 1e-3:
            _depth_scale = d_gp / d_rel
    elif _depth_scale is not None and d_rel > 1e-3:
        # Ground-plane failed (object too close, contact off-screen):
        # use Depth-Anything scaled by the last valid calibration.
        # This keeps dist_m flowing so DIST_STOP_DISTANCE can trigger.
        dist_m      = _depth_scale * d_rel
        dist_method = "depth_scaled"

    resp = {
        "found":        True,
        "target":       target,
        "score":        det["score"],
        "box":          det["box"],
        "center_x":     det["center_x"],
        "center_y":     det["center_y"],
        "image_width":  det["image_width"],
        "image_height": det["image_height"],
        "dino_ms":      round(dino_ms),
        "depth_ms":     round(depth_ms),
        "dist_method":  dist_method,
    }
    if dist_m is not None:
        resp["distance_m"] = round(dist_m, 2)

    return jsonify(resp)


@app.route("/health", methods=["GET"])
def _health():
    return jsonify({"status": "ok", "device": str(_dino_device)})


# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--gpu",  type=int, default=0,
                        help="CUDA device index (ignored if no GPU)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"[SERVER] Device: cuda:{args.gpu}", flush=True)
    else:
        print("[SERVER] Device: cpu", flush=True)

    _load_dino()
    _load_depth()
    print(f"[SERVER] All models loaded. Starting Flask on 0.0.0.0:{args.port}",
          flush=True)
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
