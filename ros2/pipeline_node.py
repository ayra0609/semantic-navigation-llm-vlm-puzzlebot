#!/usr/bin/env python3
"""
pipeline_node.py  --  LLM + VLM Pipeline

Command tiers (processed in order, shortest path wins):
  1. STOP words   -> immediate halt, retried 3x for DDS reliability
  2. DIRECT MOTION -> token sent to navigation_node, no LLM/DINO needed
  3. NAVIGATE TO X -> LLM extracts target, DINO locates it frame-by-frame

Dual-mode operation (ROS2 param 'sim_mode', default False):

  sim_mode = False  (real PuzzleBot on Jetson)
    SUB  /video_source/raw   sensor_msgs/CompressedImage  -- Jetson camera
    PUB  /nav_target         std_msgs/String  -- target / motion token -> Jetson
    PUB  /detection_result   std_msgs/String  -- VLM bbox result -> Jetson

  sim_mode = True   (Gazebo simulation)
    SUB  /video_source/raw   sensor_msgs/Image       -- Gazebo camera
    PUB  /cmd_vel            geometry_msgs/Twist     -- direct robot control

Both modes:
    SUB  /llm_command        std_msgs/String  -- command from Web UI
    PUB  /puzzlebot/status   std_msgs/String  -- status to Web UI
"""

import sys
import os
import threading
import numpy as np
import cv2
import rclpy
import torch
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from llm.intent_extraction import extract_navigation_intent
from vlm.grounding_dino import detect_object, load_model

try:
    import requests
except ImportError:
    requests = None
    
# Visual servo parameters (sim_mode only)
LINEAR_SPEED      = 0.18
ANGULAR_GAIN      = 0.0015
ANGULAR_DEAD_ZONE = 25
STOP_AREA_RATIO   = 0.30
MIN_SCORE_HSV     = 0.07
MIN_SCORE_DINO    = 0.20
MIN_SCORE         = MIN_SCORE_HSV

DINO_CACHE_MAX_ERROR = 150

# NCC inference server
NCC_SERVER_URL = 'http://localhost:5001/detect'
USE_NCC        = True

FOCAL_PX       = 644.69
TARGET_WIDTH_M = 0.50   # fallback only
STOP_DIST_M    = 0.50

# Ground-plane distance estimation (camera extrinsics on PuzzleBot)
# Measure CAM_HEIGHT_M with a ruler; set CAM_PITCH_DEG to 0 if camera is level.
CAM_HEIGHT_M  = 0.222  # camera height above floor (m)  — measured 2026-05-20
CAM_PITCH_DEG = 12.7   # positive = tilting UP (horizon moves below image center)
                        # calibrated 2026-05-20: reported 0.68m vs actual 2.204m → 12.7°


def _ground_plane_dist(y2_px, img_h,
                        cam_h=None, pitch_deg=None, fy=FOCAL_PX):
    """Estimate horizontal distance to an object whose base touches the floor.

    Camera tilted UP by pitch_deg: the geometric horizon appears BELOW image
    centre (y_horizon > cy). Distance is inversely proportional to how far
    y2 (bbox bottom = object's ground contact point) falls below the horizon.
    """
    import math
    if cam_h is None:
        cam_h = CAM_HEIGHT_M
    if pitch_deg is None:
        pitch_deg = CAM_PITCH_DEG
    cy        = img_h / 2.0
    y_horizon = cy + fy * math.tan(math.radians(pitch_deg))
    y_below   = y2_px - y_horizon
    if y_below < 10:   # base near / above horizon → object too far or bad bbox
        return None
    return cam_h * fy / y_below

_COLOR_HSV = {
    'red':    [((0,   30, 40), (10,  255, 255)),
               ((160, 30, 40), (180, 255, 255))],
    'blue':   [((100, 80, 40), (130, 255, 255))],
    'green':  [((40,  80, 40), (80,  255, 255))],
    'yellow': [((20, 100, 80), (35,  255, 255))],
    'orange': [((10,  80, 60), (20,  255, 255))],
    'white':  [((0,   0, 180), (180, 30,  255))],
}

_depth_processor = None
_depth_model     = None

def load_depth_model():
    global _depth_processor, _depth_model
    if _depth_model is None:
        _depth_processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf")
        _depth_model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf")
        _depth_model.eval()
    return _depth_processor, _depth_model


def get_depth_map(pil_image):
    processor, model = load_depth_model()
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    depth = torch.nn.functional.interpolate(
        outputs.predicted_depth.unsqueeze(1),
        size=pil_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze().numpy()
    return depth

class PipelineNode(Node):

    def __init__(self):
        super().__init__('pipeline_node')

        self.declare_parameter('sim_mode', False)
        self.sim_mode = self.get_parameter('sim_mode').value
        mode_str = 'SIMULATION' if self.sim_mode else 'REAL ROBOT'
        self.get_logger().info(f'Pipeline node started -- mode: {mode_str}')

        # Publishers
        self.pub_status = self.create_publisher(String, '/puzzlebot/status', 10)
        if self.sim_mode:
            self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        else:
            self.pub_target    = self.create_publisher(String, '/nav_target',       10)
            self.pub_detection = self.create_publisher(String, '/detection_result', 10)

        # Subscribers
        self.create_subscription(String, '/llm_command', self._on_command, 10)
        self.create_subscription(String, '/puzzlebot/status', self._on_nav_status, 10)
        if self.sim_mode:
            self.create_subscription(Image,    '/video_source/raw', self._on_image_raw,        10)
            self.create_subscription(Odometry, '/odom',             self._on_odom,             10)
        else:
            self.create_subscription(CompressedImage, '/video_source/raw', self._on_image_compressed, 10)

        # State
        self.target          = None
        self.navigating      = False
        self._nav_start_time = None
        self._detection      = None
        self.latest_frame    = None
        self._frame_lock     = threading.Lock()
        self._det_lock       = threading.Lock()
        self._odom_log_time  = 0.0
        self._smooth_cx      = None
        self._last_frame_ts  = 0.0

        # Stop-retry: after a stop command publish empty string 3 more times
        # (200 ms apart) to survive DDS packet loss between Humble and Foxy
        self._stop_retries_left = 0

        # VLM background thread
        self._vlm_ready        = False
        self._detection_thread = None

        self.create_timer(0.5, self._detection_loop)
        self.create_timer(0.2, self._stop_retry_tick)
        if self.sim_mode:
            self.create_timer(0.1, self._control_loop)

        if USE_NCC and not self.sim_mode:
            self._vlm_ready = True
            self.get_logger().info(
                f'Using NCC inference server at {NCC_SERVER_URL}; '
                'skipping local Grounding DINO/Depth preload')
            self._publish_status('IDLE: Ready -- using NCC inference server')
        else:
            self._publish_status('LOADING: Grounding DINO model...')
            threading.Thread(target=self._preload_vlm, daemon=True).start()

    # -------------------------------------------------------------------------
    # Stop-retry heartbeat
    # -------------------------------------------------------------------------

    def _stop_retry_tick(self):
        if self._stop_retries_left <= 0:
            return
        if self.sim_mode:
            self._stop_retries_left = 0
            return
        m = String(); m.data = ''
        self.pub_target.publish(m)
        self._stop_retries_left -= 1
        self.get_logger().info(
            f'[STOP RETRY] -> /nav_target ""  retries_left={self._stop_retries_left}')

    # -------------------------------------------------------------------------
    # VLM preload
    # -------------------------------------------------------------------------

    def _preload_vlm(self):
        try:
            load_model()
            load_depth_model()
            self._vlm_ready = True
            self.get_logger().info('Grounding DINO ready')
            self._publish_status('IDLE: Ready -- type a navigation command')
        except Exception as e:
            self.get_logger().error(f'VLM preload failed: {e}')
            self._publish_status(f'ERROR: VLM load failed -- {e}')

    # -------------------------------------------------------------------------
    # Command dispatch
    # -------------------------------------------------------------------------

    # Tier 1: stop words
    _STOP_WORDS = {
            'stop', 'halt', 'cancel', 'abort', 'emergency stop',
        }

    # Tier 2: direct motion
    # Map lowercase trigger phrase -> (nav_target token, status label)
    _DIRECT_MOTIONS = {
            # forward
            'forward':       ('__forward', 'Moving forward'),
            'go forward':    ('__forward', 'Moving forward'),
            'move forward':  ('__forward', 'Moving forward'),
            'straight':      ('__forward', 'Moving forward'),
            'go straight':   ('__forward', 'Moving forward'),
            # backward
            'back':          ('__backward', 'Moving backward'),
            'backward':      ('__backward', 'Moving backward'),
            'go back':       ('__backward', 'Moving backward'),
            'reverse':       ('__backward', 'Moving backward'),
            'move back':     ('__backward', 'Moving backward'),
            # left
            'left':          ('__left', 'Turning left'),
            'turn left':     ('__left', 'Turning left'),
            'rotate left':   ('__left', 'Turning left'),
            'spin left':     ('__left', 'Turning left'),
            # right
            'right':         ('__right', 'Turning right'),
            'turn right':    ('__right', 'Turning right'),
            'rotate right':  ('__right', 'Turning right'),
            'spin right':    ('__right', 'Turning right'),
        }
    
    def _on_command(self, msg: String):
        user_input = msg.data.strip()
        if not user_input:
            return

        text = user_input.lower().strip()

        # Tier 1: STOP
        if any(w in text for w in self._STOP_WORDS):
            self.navigating = False
            self.target     = None
            with self._det_lock:
                self._detection = None
            if self.sim_mode:
                self.pub_cmd_vel.publish(Twist())
            else:
                m = String(); m.data = ''
                self.pub_target.publish(m)
                self._stop_retries_left = 3
            self.get_logger().info(f'[CMD] Stop: "{user_input}"')
            self._publish_status('STOPPED')
            return

        # Tier 2: DIRECT MOTION (no LLM, no DINO)
        # Longest-match first to prefer "turn left" over "left"
        motion_token = None
        motion_label = None
        for phrase in sorted(self._DIRECT_MOTIONS, key=len, reverse=True):
            if phrase in text:
                motion_token, motion_label = self._DIRECT_MOTIONS[phrase]
                break

        if motion_token:
            self.navigating = True
            self.target     = motion_token
            with self._det_lock:
                self._detection = None
            if not self.sim_mode:
                m = String(); m.data = motion_token
                self.pub_target.publish(m)
            self.get_logger().info(
                f'[CMD] Direct motion: "{user_input}" -> {motion_token}')
            self._publish_status(f'MOVING: {motion_label}')
            return

        # Tier 3: NAVIGATE TO X (LLM + DINO)
        self.get_logger().info(f'Command: "{user_input}"')
        self._publish_status(f'PROCESSING: "{user_input}"')

        try:
            import time as _time
            _t0 = _time.time()
            result = extract_navigation_intent(user_input)
            _llm_ms = (_time.time() - _t0) * 1000
            self.get_logger().info(
                f'[LATENCY] LLM intent extraction: {_llm_ms:.0f}ms')
        except Exception as e:
            self.get_logger().error(f'LLM error: {e}')
            self._publish_status(f'ERROR: LLM failed -- {e}')
            return

        self.get_logger().info(f'[LLM] result={result}')

        if result and result.get('action') == 'navigate_to':
            self.target          = result['target']
            self.navigating      = True
            self._nav_start_time = self.get_clock().now()
            self._smooth_cx      = None
            with self._det_lock:
                self._detection = None
            self.get_logger().info(
                f'[LLM] navigate_to target="{self.target}"')
            self._publish_status(f'NAVIGATING: Heading to "{self.target}"')
            if not self.sim_mode:
                m = String(); m.data = self.target
                self.pub_target.publish(m)
                self.get_logger().info(f'[PUB] /nav_target "{self.target}"')
        else:
            self.get_logger().warn(f'[LLM] unrecognised result: {result}')
            self._publish_status(f'ERROR: Could not parse "{user_input}"')

    # -------------------------------------------------------------------------
    # Image callbacks
    # -------------------------------------------------------------------------

    def _on_image_raw(self, msg: Image):
        import time as _time
        enc = msg.encoding.lower()
        channels = 4 if enc in ('rgba8', 'bgra8') else 3
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, channels)
        frame = raw[:, :, :3].copy()
        if enc in ('rgb8', 'rgba8'):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        with self._frame_lock:
            self.latest_frame   = frame
            self._last_frame_ts = _time.time()

    def _on_image_compressed(self, msg: CompressedImage):
        arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        with self._frame_lock:
            self.latest_frame = frame

    def _on_odom(self, msg: Odometry):
        if not self.navigating:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._odom_log_time < 2.0:
            return
        self._odom_log_time = now
        x  = msg.pose.pose.position.x
        y  = msg.pose.pose.position.y
        vx = msg.twist.twist.linear.x
        wz = msg.twist.twist.angular.z
        self.get_logger().info(
            f'[ODOM] pos=({x:.3f},{y:.3f})  vel=(lin={vx:.3f},ang={wz:.3f})')

    def _on_nav_status(self, msg: String):
        if not self.navigating:
            return
        s = msg.data
        if (s.startswith('ARRIVED') or
                s.startswith('ERROR: Target lost') or
                s.startswith('ERROR: Could not find') or
                s.startswith('ERROR: Obstacle') or
                s.startswith('ERROR: Collision')):
            self.navigating = False
            self.target = None
            with self._det_lock:
                self._detection = None
            self.get_logger().info(f'[NAV_STATUS] navigation ended: {s[:60]}')

    # -------------------------------------------------------------------------
    # Detection loop (0.5 Hz)
    # -------------------------------------------------------------------------

    def _detection_loop(self):
        if not self.navigating or self.target is None:
            return
        # Skip keepalive for direct-motion tokens (no DINO needed)
        if self.target.startswith('__'):
            if not self.sim_mode:
                m = String(); m.data = self.target
                self.pub_target.publish(m)
            return
        # Republish nav_target as keepalive so late-joining navigation_node gets it
        if not self.sim_mode:
            m = String(); m.data = self.target
            self.pub_target.publish(m)
            self.get_logger().info(f'[PUB] /nav_target "{self.target}" (keepalive)')
        if not self._vlm_ready:
            self._publish_status('NAVIGATING: Loading VLM model...')
            return
        if self._detection_thread and self._detection_thread.is_alive():
            return
        with self._frame_lock:
            if self.latest_frame is None:
                self._publish_status('NAVIGATING: Waiting for camera...')
                return
            frame = self.latest_frame.copy()
        target = self.target
        self._detection_thread = threading.Thread(
            target=self._run_detection, args=(frame, target), daemon=True)
        self._detection_thread.start()

    def _run_detection(self, frame: np.ndarray, target: str):
        import time as _time
        import json as _json
        import requests as _requests
        from PIL import Image as PILImage

        pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = None

        if USE_NCC:
            try:
                import io
                buf = io.BytesIO()
                pil.save(buf, format='JPEG', quality=85)
                buf.seek(0)
                _t0 = _time.time()
                resp = _requests.post(
                    NCC_SERVER_URL,
                    files={'image': ('frame.jpg', buf, 'image/jpeg')},
                    data={'target': target},
                    timeout=10,
                )
                total_ms = (_time.time() - _t0) * 1000
                result = resp.json()
                if result.get('found'):
                    self.get_logger().info(
                        f'[DINO] found=True  score={result["score"]:.2f}'
                        f'  cx={result.get("center_x",0):.0f}'
                        f'  dist={result.get("distance_m","?")}m')
                    self.get_logger().info(
                        f'[LATENCY] DINO={result.get("dino_ms",0):.0f}ms'
                        f'  Depth={result.get("depth_ms",0):.0f}ms'
                        f'  Network+total={total_ms:.0f}ms')
                else:
                    self.get_logger().info(
                        f'[DINO] found=False for "{target}"')
            except Exception as e:
                self.get_logger().error(f'[NCC] Request failed: {e}')
                result = {'found': False, 'target': target}
        else:
            try:
                _t0 = _time.time()
                result = detect_object(pil, target)
                _dino_ms = (_time.time() - _t0) * 1000
                self.get_logger().info(
                    f'[LATENCY] DINO detection: {_dino_ms:.0f}ms')
            except Exception as e:
                self.get_logger().error(f'VLM error: {e}')

            if result and result.get('found'):
                b  = result.get('box', [0,0,0,0])
                iw = result.get('image_width', 640)
                try:
                    _t1 = _time.time()
                    depth_map    = get_depth_map(pil)
                    _depth_ms    = (_time.time() - _t1) * 1000
                    self.get_logger().info(
                        f'[LATENCY] Depth estimation: {_depth_ms:.0f}ms')
                    x1,y1,x2,y2 = [int(v) for v in b]
                    ih           = pil.size[1]
                    gp_dist      = _ground_plane_dist(y2, ih)
                    if gp_dist is not None:
                        result['distance_m'] = round(gp_dist, 2)
                        result['dist_method'] = 'ground_plane'
                    else:
                        # Fallback: width-based pinhole scaled by relative depth
                        bbox_w_px = x2 - x1
                        rough_dist = (TARGET_WIDTH_M * FOCAL_PX) / bbox_w_px \
                            if bbox_w_px > 0 else 1.0
                        cx_i = max(15, min(depth_map.shape[1]-15, (x1+x2)//2))
                        cy_i = max(15, min(depth_map.shape[0]-15, (y1+y2)//2))
                        roi  = depth_map[cy_i-15:cy_i+15, cx_i-15:cx_i+15]
                        med  = float(np.median(roi))
                        result['distance_m'] = round(med * (rough_dist/(med+1e-6)), 2)
                        result['dist_method'] = 'pinhole_width'
                except Exception as e:
                    self.get_logger().warn(f'[DEPTH] failed: {e}')
            else:
                self.get_logger().info(f'[DINO] found=False for "{target}"')

        if self.sim_mode and (result is None or not result.get('found')):
            result = self._color_detect(frame, target)

        if result is None or not self.navigating:
            return

        with self._det_lock:
            self._detection = result

        if result.get('found') and not self.sim_mode:
            import json
            m = String(); m.data = json.dumps(result)
            self.pub_detection.publish(m)
            self.get_logger().info(
                f'[PUB] /detection_result  found=True'
                f'  score={result["score"]:.2f}'
                f'  cx={result.get("center_x",0):.0f}'
                f'  dist={result.get("distance_m","?")}m')

    # -------------------------------------------------------------------------
    # HSV color fallback (sim only)
    # -------------------------------------------------------------------------

    def _color_detect(self, frame: np.ndarray, target: str) -> dict:
        target_l = target.lower()
        color    = next((c for c in _COLOR_HSV if c in target_l), None)
        if color is None:
            return {'found': False, 'target': target}
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in _COLOR_HSV[color]:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        k    = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {'found': False, 'target': target}
        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        h, w = frame.shape[:2]
        if area < 300:
            return {'found': False, 'target': target}
        x, y, bw, bh = cv2.boundingRect(best)
        score = float(min(0.90, area / (w * h * 0.05)))
        return {
            'found':        True,
            'target':       target,
            'score':        round(score, 3),
            'box':          [float(x), float(y), float(x+bw), float(y+bh)],
            'center_x':     float(x + bw/2),
            'center_y':     float(y + bh/2),
            'image_width':  float(w),
            'image_height': float(h),
        }

    # -------------------------------------------------------------------------
    # Visual servo control loop (sim mode, 10 Hz)
    # -------------------------------------------------------------------------

    def _control_loop(self):
        if not self.navigating or self.target is None:
            return

        import time as _time

        with self._frame_lock:
            if self.latest_frame is None:
                self.get_logger().warn('[CTRL] no camera frame yet')
                return
            frame        = self.latest_frame.copy()
            frame_age_ms = int((_time.time() - self._last_frame_ts) * 1000)

        if frame_age_ms > 500:
            self.get_logger().warn(f'[CTRL] STALE FRAME {frame_age_ms}ms')

        # Direct motion tokens are handled by navigation_node; in sim_mode
        # publish cmd_vel directly
        if self.target.startswith('__'):
            cmd_map = {
                '__forward':  ( 0.15,  0.0),
                '__backward': (-0.15,  0.0),
                '__left':     ( 0.0,   0.40),
                '__right':    ( 0.0,  -0.40),
            }
            if self.target in cmd_map:
                lin, ang = cmd_map[self.target]
                cmd = Twist()
                cmd.linear.x  = lin
                cmd.angular.z = ang
                self.pub_cmd_vel.publish(cmd)
            return

        hsv_det = self._color_detect(frame, self.target)
        det     = hsv_det
        det_src = 'HSV'

        if not det.get('found'):
            with self._det_lock:
                dino_fallback = self._detection
            if dino_fallback and dino_fallback.get('found'):
                dino_cx    = dino_fallback.get('center_x', 320)
                dino_error = abs(dino_cx - 320)
                if dino_error <= DINO_CACHE_MAX_ERROR:
                    det     = dino_fallback
                    det_src = 'DINO-cached'

        cmd = Twist()
        cx    = det.get('center_x', 0.0) if (det and det.get('found')) else None
        box   = det.get('box', [0,0,0,0]) if (det and det.get('found')) else None
        img_w = det.get('image_width',  640.0) if det else 640.0
        img_h = det.get('image_height', 480.0) if det else 480.0

        min_score = MIN_SCORE_HSV if det_src == 'HSV' else MIN_SCORE_DINO
        target_visible = (
            det is not None
            and det.get('found')
            and det.get('score', 0.0) >= min_score
        )

        if not target_visible:
            self._smooth_cx = None
            elapsed = 0.0
            if self._nav_start_time is not None:
                elapsed = (self.get_clock().now() - self._nav_start_time).nanoseconds / 1e9
            if elapsed < 1.0:
                self.pub_cmd_vel.publish(Twist())
                return
            cmd.angular.z = 0.30
            cmd.linear.x  = 0.0
            self.pub_cmd_vel.publish(cmd)
            return

        area     = ((box[2]-box[0])*(box[3]-box[1])) / (img_w*img_h)
        if self._smooth_cx is None:
            self._smooth_cx = cx
        else:
            self._smooth_cx = 0.85 * self._smooth_cx + 0.15 * cx

        error     = self._smooth_cx - img_w / 2.0
        bbox_w_px = box[2] - box[0]
        est_dist  = (TARGET_WIDTH_M * FOCAL_PX / bbox_w_px) if bbox_w_px > 0 else 99.0

        if area > STOP_AREA_RATIO or est_dist < STOP_DIST_M:
            self.pub_cmd_vel.publish(Twist())
            self.navigating = False
            with self._det_lock:
                self._detection = None
            self.target = None
            self.get_logger().info(
                f'[CTRL] ARRIVED -- area={area:.3f}  dist~{est_dist:.2f}m')
            self._publish_status('ARRIVED: Reached target!')
            return

        max_error   = 320.0
        error_ratio = min(abs(error) / max_error, 1.0)
        cmd.linear.x = LINEAR_SPEED * (1.0 - 0.6 * error_ratio)
        if abs(error) < ANGULAR_DEAD_ZONE:
            cmd.angular.z = 0.0
        else:
            cmd.angular.z = -error * ANGULAR_GAIN

        self.pub_cmd_vel.publish(cmd)

    # -------------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------------

    def _publish_status(self, text: str):
        msg = String(); msg.data = text
        self.pub_status.publish(msg)
        self.get_logger().info(f'[STATUS] {text}')


def main(args=None):
    rclpy.init(args=args)
    node = PipelineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.sim_mode:
            node.pub_cmd_vel.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
