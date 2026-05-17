#!/usr/bin/env python3
"""
pipeline_node.py  —  LLM + VLM Pipeline

Dual-mode operation (set via ROS2 param 'sim_mode', default True):

  sim_mode = True   (Gazebo simulation)
    SUB  /video_source/raw   sensor_msgs/Image       — Gazebo camera
    PUB  /cmd_vel            geometry_msgs/Twist     — direct robot control

  sim_mode = False  (real PuzzleBot on Jetson)
    SUB  /video_source/raw   sensor_msgs/CompressedImage  — Jetson camera
    PUB  /nav_target         std_msgs/String         — target → Jetson
    PUB  /detection_result   std_msgs/String         — VLM result → Jetson

Both modes:
    SUB  /llm_command        std_msgs/String         — command from Web UI
    PUB  /puzzlebot/status   std_msgs/String         — status to Web UI
"""

import sys, os, threading
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from llm.intent_extraction import extract_navigation_intent
from vlm.grounding_dino import detect_object, load_model

# ── Visual servo parameters ────────────────────────────────────────────────────
LINEAR_SPEED       = 0.18   # m/s forward
ANGULAR_GAIN       = 0.0015 # rad/s per pixel error (reduced to avoid path drift)
ANGULAR_DEAD_ZONE  = 25     # px — no steering correction inside this band
STOP_AREA_RATIO    = 0.30   # stop when box covers 30% of frame area
MIN_SCORE_HSV   = 0.07   # HSV detection is precise — threshold can be loose
MIN_SCORE_DINO  = 0.20   # DINO may hallucinate — apply stricter filter
MIN_SCORE       = MIN_SCORE_HSV  # backward-compat alias

# ── DINO cached result limit ──────────────────────────────────────────────────
# Reject cached DINO result if cx is too far from centre (>150px) to avoid
# large spurious steering corrections
DINO_CACHE_MAX_ERROR = 150  # px

# ── Distance estimation from bounding box ─────────────────────────────────────
# Camera horizontal FOV=60° (1.047 rad), image width=640px
# focal_px = (640/2) / tan(30°) ≈ 554
# distance_m = TARGET_WIDTH_M * FOCAL_PX / bbox_width_px
FOCAL_PX        = 554.0
TARGET_WIDTH_M  = 0.50   # target box width in metres
STOP_DIST_M     = 0.50   # stop when estimated distance < 0.5 m

# ── HSV color ranges for simulation fallback detection ─────────────────────────
# Format: list of ((H_lo, S_lo, V_lo), (H_hi, S_hi, V_hi))
_COLOR_HSV = {
    'red':    [((0,   30, 40), (10,  255, 255)),
               ((160, 30, 40), (180, 255, 255))],
    'blue':   [((100, 80, 40), (130, 255, 255))],
    'green':  [((40,  80, 40), (80,  255, 255))],
    'yellow': [((20, 100, 80), (35,  255, 255))],
    'orange': [((10,  80, 60), (20,  255, 255))],
    'white':  [((0,   0, 180), (180, 30,  255))],
}


class PipelineNode(Node):

    def __init__(self):
        super().__init__('pipeline_node')

        # ── Mode parameter ────────────────────────────────────────────────────
        self.declare_parameter('sim_mode', False)
        self.sim_mode = self.get_parameter('sim_mode').value
        mode_str = 'SIMULATION' if self.sim_mode else 'REAL ROBOT'
        self.get_logger().info(f'Pipeline node started — mode: {mode_str}')

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_status = self.create_publisher(String, '/puzzlebot/status', 10)
        if self.sim_mode:
            self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        else:
            self.pub_target    = self.create_publisher(String, '/nav_target',       10)
            self.pub_detection = self.create_publisher(String, '/detection_result', 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(String, '/llm_command', self._on_command, 10)
        if self.sim_mode:
            self.create_subscription(
                Image, '/video_source/raw', self._on_image_raw, 10)
            self.create_subscription(
                Odometry, '/odom', self._on_odom, 10)
        else:
            self.create_subscription(
                CompressedImage, '/video_source/raw',
                self._on_image_compressed, 10)

        # ── State ─────────────────────────────────────────────────────────────
        self.target          = None
        self.navigating      = False
        self._nav_start_time = None   # set when navigation begins
        self._detection      = None   # guarded by _det_lock
        self.latest_frame    = None
        self._frame_lock     = threading.Lock()
        self._det_lock       = threading.Lock()
        self._odom_log_time  = 0.0    # throttle /odom logging to every 2 s
        self._smooth_cx      = None   # exponentially smoothed target centre x
        self._last_frame_ts  = 0.0    # timestamp of last received frame

        # VLM runs in a background thread so it never blocks ROS2 timers
        self._vlm_ready        = False
        self._detection_thread = None

        self.create_timer(0.5, self._detection_loop)
        if self.sim_mode:
            self.create_timer(0.1, self._control_loop)

        # Preload Grounding DINO in background
        self._publish_status('LOADING: Grounding DINO model…')
        threading.Thread(target=self._preload_vlm, daemon=True).start()

    # ── VLM preload (background thread) ───────────────────────────────────────

    def _preload_vlm(self):
        try:
            load_model()
            self._vlm_ready = True
            self.get_logger().info('Grounding DINO ready')
            self._publish_status('IDLE: Ready — type a navigation command')
        except Exception as e:
            self.get_logger().error(f'VLM preload failed: {e}')
            self._publish_status(f'ERROR: VLM load failed — {e}')

    # ── Command callback ──────────────────────────────────────────────────────

    def _on_command(self, msg: String):
        user_input = msg.data.strip()
        if not user_input:
            return
        self.get_logger().info(f'Command: "{user_input}"')
        self._publish_status(f'PROCESSING: "{user_input}"')

        try:
            result = extract_navigation_intent(user_input)
        except Exception as e:
            self.get_logger().error(f'LLM error: {e}')
            self._publish_status(f'ERROR: LLM failed — {e}')
            return

        self.get_logger().info(f'[LLM] result={result}')

        if result and result.get('action') == 'navigate_to':
            self.target          = result['target']
            self.navigating      = True
            self._nav_start_time = self.get_clock().now()
            self._smooth_cx      = None   # reset smoothing for new target
            with self._det_lock:
                self._detection = None
            self.get_logger().info(
                f'[LLM]: action=navigate_to  target="{self.target}"')
            self._publish_status(f'NAVIGATING: Heading to "{self.target}"')

            if not self.sim_mode:
                m = String(); m.data = self.target
                self.pub_target.publish(m)
        else:
            self.get_logger().warn(
                f'[LLM]: unrecognised result: {result}')
            self._publish_status(f'ERROR: Could not parse "{user_input}"')

    # ── Image callbacks ───────────────────────────────────────────────────────

    def _on_image_raw(self, msg: Image):
        enc = msg.encoding.lower()
        channels = 4 if enc in ('rgba8', 'bgra8') else 3
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, channels)
        frame = raw[:, :, :3].copy()
        if enc in ('rgb8', 'rgba8'):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        import time as _time
        with self._frame_lock:
            self.latest_frame    = frame
            self._last_frame_ts  = _time.time()

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
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        vx = msg.twist.twist.linear.x
        wz = msg.twist.twist.angular.z
        self.get_logger().info(
            f'[ODOM] pos=({x:.3f}, {y:.3f})  vel=(lin={vx:.3f}, ang={wz:.3f})')

    # ── Detection loop (0.5 Hz timer → spawns background thread) ─────────────

    def _detection_loop(self):
        if not self.navigating or self.target is None:
            return
        if not self._vlm_ready:
            self._publish_status('NAVIGATING: Loading VLM model…')
            return
        if self._detection_thread and self._detection_thread.is_alive():
            return  # previous detection still running

        with self._frame_lock:
            if self.latest_frame is None:
                self._publish_status('NAVIGATING: Waiting for camera…')
                return
            frame = self.latest_frame.copy()

        target = self.target
        self._detection_thread = threading.Thread(
            target=self._run_detection, args=(frame, target), daemon=True)
        self._detection_thread.start()

    def _run_detection(self, frame: np.ndarray, target: str):
        from PIL import Image as PILImage
        pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        result = None
        try:
            result = detect_object(pil, target)
        except Exception as e:
            self.get_logger().error(f'VLM error: {e}')

        # Log raw VLM output — filter out obvious false positives
        if result and result.get('found'):
            b = result.get('box', [0,0,0,0])
            iw = result.get('image_width', 640)
            ih = result.get('image_height', 480)
            raw_area = (b[2]-b[0])*(b[3]-b[1])/(iw*ih)
            box_w_frac = (b[2]-b[0]) / iw
            # Reject: box fills >85% of image width or >60% of image area
            # (DINO hallucination — typically returns near-full-frame bbox)
            if box_w_frac > 0.85 or raw_area > 0.60:
                self.get_logger().warn(
                    f'[VLM-DINO] Rejected false positive:'
                    f'  area={raw_area:.3f}  box_w={box_w_frac:.2f}  box={[round(v) for v in b]}')
                result = {'found': False, 'target': target}
            else:
                self.get_logger().info(
                    f'[VLM-DINO] found=True  score={result["score"]:.2f}'
                    f'  cx={result.get("center_x",0):.0f}'
                    f'  box={[round(v) for v in b]}'
                    f'  area={raw_area:.3f}')
        else:
            self.get_logger().info(
                f'[VLM-DINO] found=False  (no detection for "{target}")')

        # Simulation fallback: if VLM finds nothing, use HSV color detection.
        if self.sim_mode and (result is None or not result.get('found')):
            result = self._color_detect(frame, target)
            if result and result.get('found'):
                b = result.get('box', [0,0,0,0])
                iw = result.get('image_width', 640)
                ih = result.get('image_height', 480)
                hsv_area = (b[2]-b[0])*(b[3]-b[1])/(iw*ih)
                self.get_logger().info(
                    f'[VLM-HSV]  found=True  score={result["score"]:.2f}'
                    f'  cx={result.get("center_x",0):.0f}'
                    f'  box={[round(v) for v in b]}'
                    f'  area={hsv_area:.3f}  '
                    f'  {">> WILL ARRIVE" if hsv_area > STOP_AREA_RATIO else "-> moving"}')
            else:
                self.get_logger().info('[VLM-HSV]  found=False')

        if result is None:
            return

        # Abort if navigation ended while detection was running
        if not self.navigating:
            return

        with self._det_lock:
            self._detection = result

        if result.get('found'):
            score = result.get('score', 0.0)
            cx    = result.get('center_x', 0)
            self.get_logger().info(
                f'[VLM] final: score={score:.2f}  cx={cx:.0f}')
            self._publish_status(
                f'NAVIGATING: "{target}" detected score={score:.2f}')

            if not self.sim_mode:
                import json
                m = String(); m.data = json.dumps(result)
                self.pub_detection.publish(m)
        else:
            self.get_logger().info(f'[VLM] Searching for "{target}"…')

    # ── HSV color fallback (simulation only) ──────────────────────────────────

    def _color_detect(self, frame: np.ndarray, target: str) -> dict:
        target_l = target.lower()
        color    = next((c for c in _COLOR_HSV if c in target_l), None)
        if color is None:
            return {'found': False, 'target': target}

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in _COLOR_HSV[color]:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

        # Remove noise
        k    = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {'found': False, 'target': target}

        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        h, w = frame.shape[:2]

        if area < 300:  # ignore tiny blobs
            return {'found': False, 'target': target}

        x, y, bw, bh = cv2.boundingRect(best)
        score = float(min(0.90, area / (w * h * 0.05)))

        return {
            'found':        True,
            'target':       target,
            'score':        round(score, 3),
            'box':          [float(x), float(y), float(x + bw), float(y + bh)],
            'center_x':     float(x + bw / 2),
            'center_y':     float(y + bh / 2),
            'image_width':  float(w),
            'image_height': float(h),
        }

    # ── Visual servo control loop (sim mode, 10 Hz) ────────────────────────────
    #
    # In sim_mode the control loop uses INLINE HSV colour detection on every
    # tick (< 1 ms) so it is never blocked by the slow DINO inference thread
    # (DINO can take 15-20 s on CPU).  DINO still runs in the background for
    # logging and verification.

    def _control_loop(self):
        if not self.navigating or self.target is None:
            return

        # ── Get current frame ────────────────────────────────────────────────
        import time as _time
        with self._frame_lock:
            if self.latest_frame is None:
                self.get_logger().warn('[CTRL] no camera frame yet')
                return
            frame        = self.latest_frame.copy()
            frame_age_ms = int((_time.time() - self._last_frame_ts) * 1000)

        if frame_age_ms > 500:
            self.get_logger().warn(
                f'[CTRL] STALE FRAME — last update {frame_age_ms}ms ago'
                f'  (Gazebo camera may have frozen)')

        # ── Fast inline HSV detection (primary, real-time) ───────────────────
        hsv_det  = self._color_detect(frame, self.target)
        det      = hsv_det
        det_src  = 'HSV'

        # If HSV found nothing, fall back to last DINO result
        if not det.get('found'):
            with self._det_lock:
                dino_fallback = self._detection
            if dino_fallback and dino_fallback.get('found'):
                # reject cached result if cx deviation is too large
                dino_cx    = dino_fallback.get('center_x', 320)
                dino_error = abs(dino_cx - 320)
                if dino_error <= DINO_CACHE_MAX_ERROR:
                    det     = dino_fallback
                    det_src = 'DINO-cached'
                else:
                    det_src = f'DINO-rejected(err={dino_error:.0f})'
            else:
                det_src = 'NONE'

        # log full status line every 5 ticks (0.5 s)
        self._ctrl_log_n = getattr(self, '_ctrl_log_n', 0) + 1
        log_now = (self._ctrl_log_n % 5 == 1)

        cmd = Twist()

        cx    = det.get('center_x', 0.0) if (det and det.get('found')) else None
        box   = det.get('box', [0, 0, 0, 0]) if (det and det.get('found')) else None
        img_w = det.get('image_width',  640.0) if det else 640.0
        img_h = det.get('image_height', 480.0) if det else 480.0

        min_score = MIN_SCORE_HSV if det_src == 'HSV' else MIN_SCORE_DINO
        target_visible = (
            det is not None
            and det.get('found')
            and det.get('score', 0.0) >= min_score
        )

        # ── Case 1: target not visible — rotate to search ────────────────────────
        if not target_visible:
            self._smooth_cx = None   # reset smoothing when target is lost
            elapsed = 0.0
            if self._nav_start_time is not None:
                elapsed = (self.get_clock().now() - self._nav_start_time).nanoseconds / 1e9
            if elapsed < 1.0:
                self.pub_cmd_vel.publish(Twist())
                return

            # search phase: count red pixels at different saturation thresholds
            hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h_ch    = hsv_img[:, :, 0]
            s_ch    = hsv_img[:, :, 1]
            v_ch    = hsv_img[:, :, 2]

            # red pixel count at three saturation levels (S>10 / S>30 / S>60)
            h_red = (h_ch <= 10) | (h_ch >= 170)   # H range covering red
            red10 = int(np.sum(h_red & (s_ch > 10) & (v_ch > 40)))
            red30 = int(np.sum(h_red & (s_ch > 30) & (v_ch > 40)))
            red60 = int(np.sum(h_red & (s_ch > 60) & (v_ch > 40)))

            # max saturation among H<=15 pixels (diagnostic for near-red hues)
            h_near_red = h_ch <= 15
            max_s = int(s_ch[h_near_red].max()) if h_near_red.any() else 0

            if log_now:
                self.get_logger().info(
                    f'[CTRL] SEARCH  elapsed={elapsed:.1f}s  src={det_src}'
                    f'  red_px(S>10)={red10}  (S>30)={red30}  (S>60)={red60}'
                    f'  max_S_in_H0-15={max_s}')
            elif red10 > 200:
                # red signal detected at any saturation — log immediately
                self.get_logger().info(
                    f'[CTRL] SEARCH  red signal!  elapsed={elapsed:.1f}s'
                    f'  red(S>10)={red10}  (S>30)={red30}  max_S={max_s}')

            cmd.angular.z = 0.30
            cmd.linear.x  = 0.0
            self.pub_cmd_vel.publish(cmd)
            return

        area  = ((box[2] - box[0]) * (box[3] - box[1])) / (img_w * img_h)

        # exponential smoothing α=0.15 — single-frame jump of 229px → error ~34px (near dead zone)
        if self._smooth_cx is None:
            self._smooth_cx = cx
        else:
            self._smooth_cx = 0.85 * self._smooth_cx + 0.15 * cx

        error = self._smooth_cx - img_w / 2.0

        # ── Estimate distance to target ───────────────────────────────────────────
        bbox_w_px = box[2] - box[0]
        est_dist  = (TARGET_WIDTH_M * FOCAL_PX / bbox_w_px) if bbox_w_px > 0 else 99.0

        # ── Case 2: arrived — stop (area threshold OR distance < 0.5 m) ─────────
        arrived = area > STOP_AREA_RATIO or est_dist < STOP_DIST_M
        if arrived:
            self.pub_cmd_vel.publish(Twist())
            self.navigating = False
            with self._det_lock:
                self._detection = None
            self.target = None
            self.get_logger().info(
                f'[CTRL] ARRIVED — area={area:.3f}  dist≈{est_dist:.2f}m')
            self._publish_status('ARRIVED: Reached target!')
            return

        # ── Case 3: target visible — proportional control, always moving forward ──
        max_error   = 320.0
        error_ratio = min(abs(error) / max_error, 1.0)

        cmd.linear.x = LINEAR_SPEED * (1.0 - 0.6 * error_ratio)

        # dead zone: suppress steering below ANGULAR_DEAD_ZONE px to avoid noise-induced drift
        if abs(error) < ANGULAR_DEAD_ZONE:
            cmd.angular.z = 0.0
        else:
            cmd.angular.z = -error * ANGULAR_GAIN

        if log_now:
            self.get_logger().info(
                f'[CTRL] TRACK  src={det_src}'
                f'  cx_raw={cx:.0f}  cx_smooth={self._smooth_cx:.0f}'
                f'  error={error:.0f}  area={area:.3f}'
                f'  dist≈{est_dist:.2f}m  score={det.get("score",0):.2f}'
                f'  cmd=(lin={cmd.linear.x:.2f}, ang={cmd.angular.z:.3f})')

        self.pub_cmd_vel.publish(cmd)

    # ── Helper ────────────────────────────────────────────────────────────────

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
