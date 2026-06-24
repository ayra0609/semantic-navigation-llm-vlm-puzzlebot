#!/usr/bin/env python3
"""
navigation_node.py - ROS2 Navigation Node for PuzzleBot
Runs on Jetson Nano. Receives detection results from PC and controls robot.

Command tiers (via /nav_target topic):
  ""            -> immediate stop
  "__forward"   -> move forward  (until stop / new command)
  "__backward"  -> move backward (until stop / new command)
  "__left"      -> rotate left   (until stop / new command)
  "__right"     -> rotate right  (until stop / new command)
  "<object>"    -> navigate to object using DINO detections from PC
"""

import time
import json
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class PidController:
    def __init__(self, Kp=0.001, Ti=float('inf'), Td=0.0):
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        self.err_prev = 0.0
        self.err_int  = 0.0

    def reset(self):
        self.err_prev = 0.0
        self.err_int  = 0.0

    def compute(self, error, dt):
        if dt <= 0:
            return 0.0
        self.err_int  += dt * error
        derr           = (error - self.err_prev) / dt
        self.err_prev  = error
        return self.Kp * (error + (1.0 / self.Ti) * self.err_int + self.Td * derr)


class NavigationNode(Node):

    # Visual servoing parameters
    IMAGE_WIDTH    = 640
    IMAGE_HEIGHT   = 480
    IMAGE_CENTER_X = IMAGE_WIDTH / 2

    LINEAR_SPEED   = 0.10   # m/s -- forward speed once aligned

    # Alignment threshold: must be centered within this many pixels
    # 80px is more forgiving of DINO bounding-box jitter on the real robot
    CENTER_TOLERANCE = 80   # pixels (was 40 -- too tight for 0.5 Hz DINO)

    # Stop conditions
    STOP_AREA_RATIO      = 0.35   # stop when bbox covers 35% of image
    STOP_AREA_MIN_SCORE  = 0.60   # area-based stop requires this confidence
    AREA_STOP_MAX_DISTANCE = 0.80 # area stop only counts when depth also says close
    CLOSE_RANGE_AREA     = 0.20   # target is visually close even if depth is noisy
    BBOX_BOTTOM_STOP_RATIO = 0.92 # stop if bbox bottom is near image bottom
    CLOSE_RANGE_DEPTH    = 1.20   # after this, target identity may become partial
    USE_LIDAR            = False  # real PuzzleBot setup has no LiDAR
    LIDAR_STOP_DISTANCE  = 0.40   # unused on the real robot if no /scan data is available
    MIN_DETECTION_SCORE  = 0.25   # ignore low-confidence detections
    DR_MIN_SCORE         = 0.50   # min score to update dead-reckoning distance
    DIST_STOP_DISTANCE   = 0.50   # stop when fresh depth says target is close
    DR_ARRIVE_MARGIN     = 0.20   # backup clearance; live depth is the primary stop
    DR_BACKUP_STOP       = False  # keep DR for logs only; stop primarily by live depth
    DR_RECALIBRATE_INTERVAL_S = 2.0  # re-anchor DR after this much forward motion
    FINAL_APPROACH_DISTANCE = 1.60   # below this, depth tends to plateau on chair legs
    FINAL_APPROACH_EXTRA_M  = 0.30   # short backup crawl after entering final approach
    DR_OUTLIER_JUMP_M       = 1.50   # reject sudden depth jumps away from target
    DR_OUTLIER_RATIO        = 1.80   # reject distance much larger than current DR remainder
    LOCKED_TARGET_FALLBACK_TIMEOUT = 1.5  # use old target pose only after fresh vision drops out

    # Collision / stall detection from odometry. This is not a physical bumper:
    # it detects "I am commanding motion, but the robot is not actually moving".
    COLLISION_DETECTION       = True
    COLLISION_CHECK_WINDOW_S  = 1.8
    COLLISION_MIN_LINEAR_CMD  = 0.04
    COLLISION_MIN_ANGULAR_CMD = 0.18
    COLLISION_MIN_TRANSLATION = 0.025
    COLLISION_MIN_ROTATION    = 0.08
    ODOM_STALE_MOTION_STOP_S  = 3.0
    NO_PROGRESS_TIMEOUT_S     = 10.0
    NO_PROGRESS_MIN_LINEAR_CMD = 0.04
    NO_PROGRESS_MIN_DIST_DROP_M = 0.20
    NO_PROGRESS_MIN_AREA_GAIN = 0.04
    EDGE_STUCK_TIMEOUT_S      = 6.0
    EDGE_STUCK_MARGIN_PX      = 80
    EDGE_STUCK_MIN_ERROR_PX   = 220
    PERCEPTION_RISK_TIMEOUT_S = 6.0
    PERCEPTION_RISK_THRESHOLD = 0.60

    # Detection hysteresis
    # How long (s) to keep using the last positive detection before switching
    # to SEARCH mode. Without this, a single missed DINO frame immediately
    # stops forward motion and starts the robot spinning.
    DETECTION_TIMEOUT = 4.0   # seconds
    SEARCH_TIMEOUT    = 25.0  # seconds without detection before stopping
    SEARCH_ANGULAR    = 0.20  # rad/s, enough to scan most of a room before timeout
    FIRST_DETECTION_WAIT_S = 15.0  # wait for slow NCC/VLM startup before rotating
    CLOSE_RANGE_LOST_STOP_S = 2.5  # DINO is ~0.5 Hz; stop if close target misses a frame
    NCC_UNAVAILABLE_TIMEOUT_S = 15.0  # report failure if NCC stays unavailable this long
    FOCAL_PX         = 644.69
    ARRIVE_THRESHOLD = 0.40
    # Direct-motion speeds
    DIRECT_LINEAR  = 0.15   # m/s   (forward / backward)
    DIRECT_ANGULAR = 0.40   # rad/s (left / right in-place turn)

    # Map direct-motion tokens to (linear_x, angular_z)
    DIRECT_CMD_MAP = {
        '__forward':  ( 0.15,  0.0),
        '__backward': (-0.15,  0.0),
        '__left':     ( 0.0,   0.40),
        '__right':    ( 0.0,  -0.40),
    }

    STATE_ARRIVED        = 'arrived'
    STATE_OBSTACLE       = 'obstacle'
    STATE_DIRECT         = 'direct'
    STATE_CLOSE_LOST     = 'close_lost'
    STATE_LOCKED_MEMORY  = 'locked_memory'
    STATE_NCC_UNAVAILABLE_WAIT = 'ncc_unavailable_wait'
    STATE_NCC_UNAVAILABLE_STOP = 'ncc_unavailable_stop'
    STATE_WAITING_FIRST_DETECTION = 'waiting_first_detection'
    STATE_SEARCHING      = 'searching'
    STATE_TRACKING       = 'tracking'

    def __init__(self):
        super().__init__('navigation_node')

        # PID controller for angular steering
        # Kp=0.001: at 0.5 Hz DINO, max 0.32 rad/s -> ~18 deg per 2-s window
        # (was 0.003 which caused 57 deg overshoot and wild oscillation)
        self.pid = PidController(Kp=0.001, Ti=float('inf'), Td=0.0)
        self._last_control_time = None

        # State
        self.target               = None
        self.detection            = None
        self._last_detection_time = 0.0   # time of last positive detection
        self._direct_cmd          = None  # (lin, ang) for direct motion tokens
        self.lidar_distance       = float('inf')
        self.navigating           = False
        self.start_pose           = None
        self.current_pose         = None
        self._odom_encoder_fresh  = False
        self.current_yaw        = 0.0
        self.target_world_pos   = None
        self.target_dist        = None
        self._last_fresh_odom_wall = None
        self._nav_start_wall    = None
        self._wait_logged       = False
        self._search_start_wall = None
        self._ever_detected     = False
        # Dead-reckoning stop (used when odometry unavailable but depth available)
        self._dr_dist      = None  # latest distance estimate from depth
        self._dr_advance_s = 0.0   # seconds spent advancing since last dist update
        self._dr_last_recal_s = 0.0
        self._final_approach_locked = False
        self._final_approach_total = None
        self._dist_arrived = False  # set when live depth reading reaches DIST_STOP_DISTANCE
        self._visual_arrived = False
        self._close_range_seen = False
        self._ncc_unavailable_since = None
        self._ncc_unavailable_last = None
        self._collision_ref_pose = None
        self._collision_ref_time = None
        self._odom_stale_motion_since = None
        self._progress_ref_time = None
        self._progress_ref_dist = None
        self._progress_ref_area = None
        self._edge_stuck_since = None
        self._edge_stuck_last_dist = None
        self._perception_risk_since = None
        self._last_perception_risk_log = 0.0
        self._last_cmd_linear = 0.0
        self._last_cmd_angular = 0.0
        self._collision_stop_reason = 'collision'
        # Subscribers
        self.create_subscription(String,   '/nav_target',       self.target_callback,    10)
        self.create_subscription(String,   '/detection_result', self.detection_callback, 10)
        self.create_subscription(LaserScan,'/scan',             self.scan_callback,      10)
        self.create_subscription(Odometry, '/odom',             self.odom_callback,      10)

        # Publisher
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_status  = self.create_publisher(String, '/puzzlebot/status', 10)

        # Control loop: 10 Hz
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Navigation node started, waiting for target...')

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def target_callback(self, msg):
        new_target = msg.data.strip()

        # Empty string -> immediate stop
        if new_target == '':
            self.stop_robot('command_stop')
            return

        # Direct motion token (__forward / __backward / __left / __right)
        if new_target in self.DIRECT_CMD_MAP:
            if new_target == self.target and self.navigating:
                return   # keepalive -- already executing this command
            lin, ang = self.DIRECT_CMD_MAP[new_target]
            self.target             = new_target
            self.navigating         = True
            self.detection          = None
            self._direct_cmd        = (lin, ang)
            self._last_control_time = None
            self._nav_start_wall    = time.time()
            self._wait_logged        = False
            self._search_start_wall = None
            self._ever_detected     = False
            self._dr_dist           = None
            self._dr_advance_s      = 0.0
            self._dr_last_recal_s   = 0.0
            self._final_approach_locked = False
            self._final_approach_total = None
            self._dist_arrived      = False
            self._visual_arrived    = False
            self._close_range_seen  = False
            self._ncc_unavailable_since = None
            self._ncc_unavailable_last = None
            self._reset_collision_watchdog()
            self.pid.reset()
            self.get_logger().info(f'[NAV_TARGET] new goal "{self.target}"')
            self.get_logger().info(
                f'[NAV_TARGET] direct cmd "{new_target}"  '
                f'lin={lin:.2f}  ang={ang:.2f}')
            return

        # Keepalive for existing navigation goal -- ignore
        if new_target == self.target and self.navigating:
            return

        # New object navigation goal
        self.target             = new_target
        self.navigating         = True
        self.detection          = None
        self._direct_cmd        = None
        self.start_pose         = self.current_pose
        self._last_control_time = None
        self._nav_start_wall    = time.time()
        self._wait_logged       = False
        self._search_start_wall = None
        self._ever_detected     = False
        self._dr_dist           = None
        self._dr_advance_s      = 0.0
        self._dr_last_recal_s   = 0.0
        self._final_approach_locked = False
        self._final_approach_total = None
        self._dist_arrived      = False
        self._visual_arrived    = False
        self._close_range_seen  = False
        self._ncc_unavailable_since = None
        self._ncc_unavailable_last = None
        self._reset_collision_watchdog()
        self.pid.reset()
        self.get_logger().info(f'[NAV_TARGET] new goal "{self.target}"')
        self._publish_status(f'NAVIGATING: Searching for "{self.target}"')

    def detection_callback(self, msg):
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                return

            found = data.get('found', False)
            score = data.get('score', 0.0)

            if data.get('ncc_unavailable') or data.get('vlm_unavailable'):
                now = time.time()
                if self._ncc_unavailable_since is None:
                    self._ncc_unavailable_since = self._nav_start_wall or now
                    self._publish_status('WAITING: NCC unavailable, holding still')
                self._ncc_unavailable_last = now
                self.get_logger().warn(
                    f'[NCC_UNAVAILABLE] holding still; error={data.get("error", "?")}')
                return

            if found and score >= self.MIN_DETECTION_SCORE:
                # Positive detection: accept and refresh timestamp
                self.detection            = data
                self._last_detection_time = time.time()
                self._search_start_wall   = None
                self._ncc_unavailable_since = None
                self._ncc_unavailable_last = None
                if not self._ever_detected:
                    self._publish_status(f'NAVIGATING: Target detected "{self.target}"')
                self._ever_detected       = True
                self.get_logger().info(
                    f'[DETECTION] found=True  score={score:.2f}  '
                    f'cx={data.get("center_x", 0):.0f}  '
                    f'dist={data.get("distance_m", "?")}m  '
                    f'method={data.get("dist_method", "?")}')

                dist_m = data.get('distance_m', None)
                box = data.get('box', [0, 0, 0, 0])
                img_w = data.get('image_width', self.IMAGE_WIDTH)
                img_h = data.get('image_height', self.IMAGE_HEIGHT)
                try:
                    box_area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
                    area_ratio = box_area / float(img_w * img_h) if img_w > 0 and img_h > 0 else 0.0
                    bottom_ratio = box[3] / float(img_h) if img_h > 0 else 0.0
                except Exception:
                    area_ratio = 0.0
                    bottom_ratio = 0.0

                # Immediate stop when live depth says we are close enough.
                # Use MIN_DETECTION_SCORE (not DR_MIN_SCORE): depth accuracy is
                # independent of DINO classification confidence, and at close range
                # the score often drops as the camera sees only chair legs.
                if (dist_m and dist_m <= self.DIST_STOP_DISTANCE
                        and score >= self.MIN_DETECTION_SCORE and self.navigating):
                    self.get_logger().info(
                        f'[DIST_ARRIVED] live dist={dist_m:.2f}m <= {self.DIST_STOP_DISTANCE}m -- flagging stop')
                    self._dist_arrived = True

                if ((dist_m and dist_m <= self.CLOSE_RANGE_DEPTH)
                        or area_ratio >= self.CLOSE_RANGE_AREA
                        or bottom_ratio >= 0.85):
                    if not self._close_range_seen:
                        self.get_logger().info(
                            f'[CLOSE_RANGE] dist={dist_m if dist_m else "?"}m  '
                            f'area={area_ratio:.2f}  bottom={bottom_ratio:.2f}')
                    self._close_range_seen = True

                if (area_ratio >= self.STOP_AREA_RATIO
                        and dist_m
                        and dist_m <= self.AREA_STOP_MAX_DISTANCE
                        and score >= self.STOP_AREA_MIN_SCORE):
                    self.get_logger().info(
                        f'[VISUAL_ARRIVED] area={area_ratio:.2f} >= {self.STOP_AREA_RATIO:.2f} '
                        f'and dist={dist_m:.2f}m <= {self.AREA_STOP_MAX_DISTANCE:.2f}m')
                    self._visual_arrived = True
                elif (bottom_ratio >= self.BBOX_BOTTOM_STOP_RATIO
                        and self._close_range_seen
                        and score >= self.MIN_DETECTION_SCORE):
                    self.get_logger().info(
                        f'[VISUAL_ARRIVED] bbox_bottom={bottom_ratio:.2f} '
                        f'>= {self.BBOX_BOTTOM_STOP_RATIO:.2f}')
                    self._visual_arrived = True

                if self._no_progress_detected(dist_m, area_ratio):
                    self.get_logger().warn(
                        '[NO_PROGRESS] commanded forward motion but target distance/area did not improve')
                    self.stop_robot('collision')
                    return

                if self._edge_stuck_detected(data, dist_m):
                    self.get_logger().warn(
                        '[EDGE_STUCK] target stayed at image edge while robot kept moving')
                    self.stop_robot('collision')
                    return

                if self._perception_collision_risk_detected(
                        data, dist_m, area_ratio, bottom_ratio):
                    self.get_logger().warn(
                        '[PERCEPTION_RISK] pretrained visual/depth cues indicate collision risk')
                    self.stop_robot('collision')
                    return

                # Update dead-reckoning estimate (high-confidence detections only).
                # Total distance can only decrease — prevents inflated estimates from
                # a bbox that is wider than TARGET_WIDTH_M from extending the journey.
                if dist_m and dist_m > self.ARRIVE_THRESHOLD and score >= self.DR_MIN_SCORE:
                    covered = self._dr_advance_s * self.LINEAR_SPEED
                    dr_remaining = None
                    if self._dr_dist is not None:
                        dr_remaining = max(0.0, self._dr_dist - covered)
                    if self._final_approach_locked and dist_m > self.FINAL_APPROACH_DISTANCE:
                        self.get_logger().warn(
                            f'[DR_OUTLIER] final approach locked; ignoring '
                            f'dist={dist_m:.2f}m  covered={covered:.2f}m')
                        return
                    if (dr_remaining is not None
                            and dist_m > dr_remaining + self.DR_OUTLIER_JUMP_M
                            and dist_m > dr_remaining * self.DR_OUTLIER_RATIO):
                        self.get_logger().warn(
                            f'[DR_OUTLIER] ignoring dist={dist_m:.2f}m  '
                            f'dr_remaining={dr_remaining:.2f}m  covered={covered:.2f}m')
                        return
                    new_total = covered + dist_m
                    if dist_m <= self.FINAL_APPROACH_DISTANCE:
                        # At close range the low camera often sees chair legs,
                        # and the distance estimate can stop decreasing around
                        # 1.5 m. Do not blindly drive all of that reported
                        # distance; creep a short calibrated extra distance.
                        if not self._final_approach_locked:
                            self._final_approach_locked = True
                            self._final_approach_total = (
                                covered + self.FINAL_APPROACH_EXTRA_M + self.DR_ARRIVE_MARGIN)
                            self.get_logger().info(
                                f'[DR_FINAL_LOCK] dist={dist_m:.2f}m  '
                                f'covered={covered:.2f}m  '
                                f'total={self._final_approach_total:.2f}m')
                        # Use the locked creep distance directly. If the depth
                        # model underestimates the chair distance, min() would
                        # make _dr_dist smaller than DR_ARRIVE_MARGIN and the
                        # robot would declare arrival without moving.
                        new_total = self._final_approach_total
                        self.get_logger().info(
                            f'[DR_FINAL_CAP] dist={dist_m:.2f}m  '
                            f'covered={covered:.2f}m  total_cap={new_total:.2f}m')
                    if self._dr_dist is None:
                        self._dr_dist = new_total
                        self._dr_last_recal_s = self._dr_advance_s
                        self.get_logger().info(
                            f'[DR_INIT] remaining={dist_m:.2f}m  covered={covered:.2f}m  '
                            f'total={self._dr_dist:.2f}m')
                    elif self._dr_advance_s - self._dr_last_recal_s >= self.DR_RECALIBRATE_INTERVAL_S:
                        old_total = self._dr_dist
                        self._dr_dist = min(self._dr_dist, new_total)
                        self._dr_last_recal_s = self._dr_advance_s
                        self.get_logger().info(
                            f'[DR_RECAL] remaining={dist_m:.2f}m  covered={covered:.2f}m  '
                            f'total={old_total:.2f}->{self._dr_dist:.2f}m')
                    else:
                        self._dr_dist = min(self._dr_dist, new_total)
                        self.get_logger().info(
                            f'[DR] remaining={dist_m:.2f}m  covered={covered:.2f}m  '
                            f'total={self._dr_dist:.2f}m')

                if (dist_m and self.current_pose and self.target_world_pos is None and dist_m > self.ARRIVE_THRESHOLD):
                    cx        = data.get('center_x', self.IMAGE_CENTER_X)
                    img_w     = data.get('image_width', self.IMAGE_WIDTH)
                    px_offset = cx - img_w / 2.0
                    angle     = math.atan(px_offset / self.FOCAL_PX)
                    world_ang = self.current_yaw + angle
                    self.target_world_pos = (
                        self.current_pose[0] + dist_m * math.cos(world_ang),
                        self.current_pose[1] + dist_m * math.sin(world_ang),
                    )
                    self.target_dist = dist_m
                    self._publish_status(
                        f'NAVIGATING: Target locked "{self.target}" at {dist_m:.2f}m')
                    self.get_logger().info(
                        f'[TARGET_LOCKED] world=({self.target_world_pos[0]:.2f},'
                        f'{self.target_world_pos[1]:.2f})  dist={dist_m:.2f}m')
            else:
                self._ncc_unavailable_since = None
                self._ncc_unavailable_last = None
                # Not-found or low score: only overwrite after timeout.
                # This prevents a single missed DINO frame from immediately
                # spinning the robot into SEARCH mode.
                age = time.time() - self._last_detection_time
                if age > self.DETECTION_TIMEOUT:
                    self.detection = data
                    self.get_logger().info(
                        f'[DETECTION] found=False  '
                        f'(age={age:.1f}s > timeout -> switching to SEARCH)')
                else:
                    self.get_logger().info(
                        f'[DETECTION] found=False  ignored  '
                        f'(last positive {age:.1f}s ago, holding cache)')
        except Exception as e:
            self.get_logger().warn(f'detection_callback error: {e}')

    def scan_callback(self, msg):
        n = len(msg.ranges)
        sector = list(range(0, n // 24)) + list(range(n - n // 24, n))
        valid = [msg.ranges[i] for i in sector
                 if not math.isinf(msg.ranges[i]) and not math.isnan(msg.ranges[i])]
        self.lidar_distance = min(valid) if valid else float('inf')

    def odom_callback(self, msg):
        encoder_fresh = msg.pose.covariance[0] >= 0.0
        now = time.time()
        if not encoder_fresh and self._last_fresh_odom_wall is not None:
            if now - self._last_fresh_odom_wall < 1.0:
                return
        self._odom_encoder_fresh = encoder_fresh
        if encoder_fresh:
            self._last_fresh_odom_wall = now
        pos = msg.pose.pose.position
        self.current_pose = (pos.x, pos.y)

        q = msg.pose.pose.orientation
        siny_cosp        = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp        = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    # -------------------------------------------------------------------------
    # Control loop (10 Hz)
    # -------------------------------------------------------------------------

    def control_loop(self):
        try:
            self._control_loop_impl()
        except Exception as e:
            self.get_logger().error(f'control_loop exception: {e}')

    def _control_loop_impl(self):
        if not self.navigating or self.target is None:
            return

        cmd = Twist()
        nav_state = self._assess_navigation_state()
        state = nav_state['state']

        if state == self.STATE_ARRIVED:
            self.get_logger().info(
                f'[STATE] ARRIVED -- {nav_state["reason"]}')
            self.stop_robot('arrived')
            return

        if state == self.STATE_OBSTACLE:
            self.get_logger().info(
                f'Obstacle at {self.lidar_distance:.2f}m -- stopping!')
            self.stop_robot('obstacle_stop')
            return

        if state == self.STATE_NCC_UNAVAILABLE_STOP:
            self.get_logger().warn(
                f'[STATE] NCC unavailable -- {nav_state["reason"]}')
            self.stop_robot('ncc_unavailable')
            return

        if state == self.STATE_NCC_UNAVAILABLE_WAIT:
            self.get_logger().warn(
                f'[NCC_UNAVAILABLE_WAIT] {nav_state["reason"]}')
            self.pub_cmd_vel.publish(Twist())
            return

        # Direct motion command (__forward / __backward / __left / __right)
        # Executed every tick until a stop or new command arrives.
        if state == self.STATE_DIRECT:
            lin, ang = self._direct_cmd
            cmd.linear.x  = lin
            cmd.angular.z = ang
            self.get_logger().info(
                f'[CMD_VEL] DIRECT  lin={lin:.2f}  ang={ang:.2f}')
            self._publish_cmd(cmd)
            return

        if state == self.STATE_CLOSE_LOST:
            self.get_logger().info(
                '[CLOSE_RANGE_LOST] target was close and fresh vision is stale -- stopping conservatively')
            self.stop_robot('arrived')
            return

        if state == self.STATE_WAITING_FIRST_DETECTION:
            if not self._wait_logged:
                self.get_logger().info(
                    f'[WAIT_FIRST_DETECTION] holding still for "{self.target}" before SEARCH')
                self._publish_status(f'NAVIGATING: Checking current view for "{self.target}"')
                self._wait_logged = True
            self.pub_cmd_vel.publish(Twist())
            return

        # If the target was detected earlier with depth, keep a short spatial
        # memory only after fresh vision has dropped out. While the target is
        # visible, visual servoing below remains the primary "am I there yet?"
        # loop.
        if state == self.STATE_LOCKED_MEMORY:
            tx, ty = self.target_world_pos
            rx, ry = self.current_pose
            dist   = math.sqrt((tx - rx)**2 + (ty - ry)**2)

            if dist < self.ARRIVE_THRESHOLD:
                self.get_logger().info(
                    f'[ARRIVED] dist={dist:.2f}m < threshold')
                self.stop_robot('arrived')
                return

            desired_yaw = math.atan2(ty - ry, tx - rx)
            yaw_error   = desired_yaw - self.current_yaw
            while yaw_error >  math.pi: yaw_error -= 2 * math.pi
            while yaw_error < -math.pi: yaw_error += 2 * math.pi

            cmd.linear.x  = min(self.LINEAR_SPEED, dist * 0.5)
            cmd.angular.z = max(-0.25, min(0.25, yaw_error * 1.5))
            self.get_logger().info(
                f'[NAV_TO_LOCKED_TARGET] dist={dist:.2f}m  '
                f'yaw_err={math.degrees(yaw_error):.1f}deg  '
                f'lin={cmd.linear.x:.2f}  ang={cmd.angular.z:.3f}')
            self._publish_cmd(cmd)
            return

        # No detection (or timed-out): rotate to search
        if state == self.STATE_SEARCHING:
            now = time.time()
            if self._search_start_wall is None:
                self._search_start_wall = now
            search_age = now - self._search_start_wall
            if search_age > self.SEARCH_TIMEOUT:
                reason = 'target_lost' if self._ever_detected else 'search_timeout'
                self.get_logger().warn(
                    f'[SEARCH_TIMEOUT] no detection for "{self.target}" '
                    f'after {search_age:.1f}s -- stopping')
                self.stop_robot(reason)
                return
            cmd.angular.z = self.SEARCH_ANGULAR
            self.get_logger().info(
                f'[CMD_VEL] SEARCH "{self.target}"  ang={self.SEARCH_ANGULAR:+.2f}')
            self._publish_cmd(cmd)
            return
        
        # Target found: visual servoing
        center_x     = self.detection.get('center_x', self.IMAGE_CENTER_X)
        box          = self.detection.get('box', [0, 0, 0, 0])
        image_width  = self.detection.get('image_width',  self.IMAGE_WIDTH)
        image_height = self.detection.get('image_height', self.IMAGE_HEIGHT)
        score        = self.detection.get('score', 0.0)
        dist_m       = self.detection.get('distance_m', None)

        box_area   = (box[2] - box[0]) * (box[3] - box[1])
        image_area = image_width * image_height
        area_ratio = box_area / image_area if image_area > 0 else 0

        # Area is only a supporting cue. With the low PuzzleBot camera, tall
        # objects such as chairs can fill the image while still being far away.
        if (area_ratio > self.STOP_AREA_RATIO
                and dist_m
                and dist_m <= self.AREA_STOP_MAX_DISTANCE
                and score >= self.STOP_AREA_MIN_SCORE):
            self.get_logger().info(
                f'Reached "{self.target}"!  area={area_ratio:.2f}  '
                f'dist={dist_m:.2f}m  score={score:.2f}')
            self.stop_robot('success')
            return

        error = center_x - (image_width / 2.0)

        now = time.time()
        dt  = (now - self._last_control_time) if self._last_control_time else 0.1
        self._last_control_time = now

        # Phase 1: heading error too large → half speed + gentle angular correction.
        # Never stop for pure rotation: DINO at 0.5 Hz is too slow — the robot
        # overshoots while spinning and the stool swings left/right every frame.
        if abs(error) > self.CENTER_TOLERANCE:
            cmd.linear.x  = self.LINEAR_SPEED * 0.5
            cmd.angular.z = -self.pid.compute(error * 0.5, dt)
            if self.target_world_pos is None and self._dr_dist is not None:
                self._dr_advance_s += dt * 0.5  # half speed → half DR credit
                dist_covered = self._dr_advance_s * self.LINEAR_SPEED
                backup_arrival = max(0.0, self._dr_dist - self.DR_ARRIVE_MARGIN)
                if self._final_approach_locked:
                    backup_arrival = max(
                        backup_arrival,
                        self._final_approach_total - self.DR_ARRIVE_MARGIN)
                if self.DR_BACKUP_STOP and dist_covered >= backup_arrival:
                    self.get_logger().info(
                        f'[DR_ARRIVED] covered={dist_covered:.2f}m  '
                        f'est={self._dr_dist:.2f}m')
                    self.stop_robot('arrived')
                    return
            self.get_logger().info(
                f'[CMD_VEL] ALIGNING  error={error:+.0f}px  '
                f'area={area_ratio:.2f}  lin={cmd.linear.x:.2f}  ang={cmd.angular.z:+.3f}')

        # Phase 2: move forward once aligned
        else:
            if self.target_world_pos is None and self._dr_dist is not None:
                self._dr_advance_s += dt
                dist_covered = self._dr_advance_s * self.LINEAR_SPEED
                backup_arrival = max(0.0, self._dr_dist - self.DR_ARRIVE_MARGIN)
                if self._final_approach_locked:
                    backup_arrival = max(
                        backup_arrival,
                        self._final_approach_total - self.DR_ARRIVE_MARGIN)
                if self.DR_BACKUP_STOP and dist_covered >= backup_arrival:
                    self.get_logger().info(
                        f'[DR_ARRIVED] covered={dist_covered:.2f}m  '
                        f'est={self._dr_dist:.2f}m')
                    self.stop_robot('arrived')
                    return
            cmd.linear.x  = self.LINEAR_SPEED
            cmd.angular.z = -self.pid.compute(error * 0.5, dt)
            if self._dr_dist is not None:
                dist_covered = self._dr_advance_s * self.LINEAR_SPEED
                self.get_logger().info(
                    f'[CMD_VEL] DR_ADVANCE  {dist_covered:.2f}/{self._dr_dist:.2f}m  '
                    f'lin={self.LINEAR_SPEED:.2f}  ang={cmd.angular.z:+.3f}')
            else:
                self.get_logger().info(
                    f'[CMD_VEL] ADVANCING  error={error:+.0f}px  '
                    f'area={area_ratio:.2f}  lin={cmd.linear.x:.2f}  ang={cmd.angular.z:+.3f}')

        # Cap at +/-0.25 rad/s (was 0.5 -- caused 57 deg overshoot per DINO frame)
        cmd.angular.z = max(-0.25, min(0.25, cmd.angular.z))
        self._publish_cmd(cmd)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def stop_robot(self, reason: str = 'stop'):
        stopped_target = self.target
        final_dist = None
        if isinstance(self.detection, dict):
            try:
                final_dist = float(self.detection.get('distance_m'))
            except (TypeError, ValueError):
                final_dist = None
        nav_ms = None
        if self._nav_start_wall and self.navigating:
            nav_ms = (time.time() - self._nav_start_wall) * 1000
            self.get_logger().info(
                f'[LATENCY] Navigation total: {nav_ms:.0f}ms  reason={reason}')
        final_dist_text = f'{final_dist:.2f}' if final_dist is not None else 'NA'
        nav_ms_text = f'{nav_ms:.0f}' if nav_ms is not None else 'NA'
        dist_suffix = f' final_dist_m={final_dist_text}' if final_dist is not None else ''
        nav_suffix = f' nav_ms={nav_ms_text} reason={reason}{dist_suffix}' if nav_ms is not None else ''

        if reason == 'arrived' or reason == 'success':
            self._publish_status(
                f'ARRIVED: I have arrived and stopped.{nav_suffix}'
                if nav_ms is not None else 'ARRIVED: I have arrived and stopped.')
        elif reason == 'target_lost':
            self._publish_status(
                f'ERROR: Target lost -- I lost sight of "{stopped_target}". I saw it before, but I cannot see it now.'
                + nav_suffix)
        elif reason == 'search_timeout':
            self._publish_status(
                f'ERROR: Could not find -- I could not find "{stopped_target}", so I stopped.'
                + nav_suffix)
        elif reason == 'collision':
            self._publish_status(
                'ERROR: Collision detected -- I may have bumped into something or got stuck, so I stopped.'
                + nav_suffix)
        elif reason == 'encoder_lost':
            self._publish_status(
                'ERROR: Encoder feedback lost -- I lost wheel encoder feedback while moving, so I stopped for safety.'
                + nav_suffix)
        elif reason == 'ncc_unavailable':
            self._publish_status('ERROR: ncc unavailable' + nav_suffix)
        elif reason == 'obstacle_stop':
            self._publish_status(
                'ERROR: Obstacle detected -- There is something in front of me, so I stopped.'
                + nav_suffix)
        elif reason == 'command_stop':
            self._publish_status('STOPPED: Okay, I have stopped.')

        self.get_logger().info(
            f'[TRIAL_NAV_RESULT] target="{stopped_target}" reason={reason} '
            f'nav_ms={nav_ms_text} final_dist_m={final_dist_text}')
        self.get_logger().info(f'[CMD_VEL] STOP  reason={reason}')
        self.pub_cmd_vel.publish(Twist())
        self.navigating  = False
        self.detection   = None
        self._direct_cmd = None
        self._last_cmd_linear = 0.0
        self._last_cmd_angular = 0.0
        self.target_world_pos = None
        self.target_dist      = None
        self._wait_logged       = False
        self._search_start_wall = None
        self._ever_detected     = False
        self._dr_dist      = None
        self._dr_advance_s = 0.0
        self._dr_last_recal_s = 0.0
        self._final_approach_locked = False
        self._final_approach_total = None
        self._dist_arrived = False
        self._visual_arrived = False
        self._close_range_seen = False
        self._ncc_unavailable_since = None
        self._ncc_unavailable_last = None
        self._reset_collision_watchdog()
        self._log_result(reason)
        self.target = None
        self._nav_start_wall  = None 
        

    def _log_result(self, reason: str):
        dist_str = '?'
        if self.start_pose and self.current_pose:
            dx = self.current_pose[0] - self.start_pose[0]
            dy = self.current_pose[1] - self.start_pose[1]
            dist_str = f'{math.sqrt(dx**2 + dy**2):.2f}m'
        self.get_logger().info(
            f'Navigation ended | reason={reason} | '
            f'distance={dist_str} | lidar={self.lidar_distance:.2f}m')

    def _assess_navigation_state(self):
        """Decide what the robot should believe before choosing a motion."""
        if self._dist_arrived:
            return {
                'state': self.STATE_ARRIVED,
                'reason': 'live depth threshold reached',
            }

        if self._visual_arrived:
            return {
                'state': self.STATE_ARRIVED,
                'reason': 'visual close-range threshold reached',
            }

        if self.USE_LIDAR and self.lidar_distance < self.LIDAR_STOP_DISTANCE:
            return {
                'state': self.STATE_OBSTACLE,
                'reason': 'front obstacle detected',
            }

        if self._direct_cmd is not None:
            return {
                'state': self.STATE_DIRECT,
                'reason': 'manual/direct motion command',
            }

        if self._ncc_unavailable_since is not None:
            unavailable_age = time.time() - self._ncc_unavailable_since
            if unavailable_age >= self.NCC_UNAVAILABLE_TIMEOUT_S:
                return {
                    'state': self.STATE_NCC_UNAVAILABLE_STOP,
                    'reason': f'ncc unavailable for {unavailable_age:.1f}s',
                }
            return {
                'state': self.STATE_NCC_UNAVAILABLE_WAIT,
                'reason': f'ncc unavailable for {unavailable_age:.1f}s; holding still',
            }

        last_positive_age = time.time() - self._last_detection_time
        detection_valid = (
            self.detection is not None
            and self.detection.get('found', False)
            and last_positive_age < self.DETECTION_TIMEOUT
        )

        if self._close_range_seen and last_positive_age >= self.CLOSE_RANGE_LOST_STOP_S:
            return {
                'state': self.STATE_CLOSE_LOST,
                'reason': 'target was close but fresh vision is stale',
            }

        if (
            self.target_world_pos is not None
            and self.current_pose is not None
            and not detection_valid
            and last_positive_age >= self.LOCKED_TARGET_FALLBACK_TIMEOUT
        ):
            return {
                'state': self.STATE_LOCKED_MEMORY,
                'reason': 'temporarily using remembered target position',
            }

        if (not self._ever_detected
                and self._nav_start_wall is not None
                and time.time() - self._nav_start_wall < self.FIRST_DETECTION_WAIT_S):
            return {
                'state': self.STATE_WAITING_FIRST_DETECTION,
                'reason': 'waiting for first perception result',
            }

        if not detection_valid:
            return {
                'state': self.STATE_SEARCHING,
                'reason': 'target not currently visible',
            }

        return {
            'state': self.STATE_TRACKING,
            'reason': 'fresh visual detection available',
        }

    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.pub_status.publish(msg)
        self.get_logger().info(f'[STATUS] {text}')

    def _reset_collision_watchdog(self):
        self._collision_ref_pose = None
        self._collision_ref_time = None
        self._odom_stale_motion_since = None
        self._progress_ref_time = None
        self._progress_ref_dist = None
        self._progress_ref_area = None
        self._edge_stuck_since = None
        self._edge_stuck_last_dist = None

    def _no_progress_detected(self, dist_m, area_ratio) -> bool:
        if not self.navigating or self.target is None:
            self._progress_ref_time = None
            return False
        if self._direct_cmd is not None:
            self._progress_ref_time = None
            return False
        if abs(self._last_cmd_linear) < self.NO_PROGRESS_MIN_LINEAR_CMD:
            self._progress_ref_time = None
            return False
        if dist_m is None:
            self._progress_ref_time = None
            return False

        now = time.time()
        if self._progress_ref_time is None:
            self._progress_ref_time = now
            self._progress_ref_dist = float(dist_m)
            self._progress_ref_area = float(area_ratio)
            return False

        dist_drop = self._progress_ref_dist - float(dist_m)
        area_gain = float(area_ratio) - self._progress_ref_area
        if (dist_drop >= self.NO_PROGRESS_MIN_DIST_DROP_M
                or area_gain >= self.NO_PROGRESS_MIN_AREA_GAIN):
            self._progress_ref_time = now
            self._progress_ref_dist = float(dist_m)
            self._progress_ref_area = float(area_ratio)
            return False

        elapsed = now - self._progress_ref_time
        if elapsed >= self.NO_PROGRESS_TIMEOUT_S:
            self.get_logger().warn(
                f'[NO_PROGRESS] elapsed={elapsed:.1f}s '
                f'dist {self._progress_ref_dist:.2f}->{float(dist_m):.2f}m '
                f'area {self._progress_ref_area:.2f}->{float(area_ratio):.2f} '
                f'cmd_lin={self._last_cmd_linear:.2f}')
            return True
        return False

    def _edge_stuck_detected(self, data, dist_m) -> bool:
        if not self.navigating or self.target is None:
            self._edge_stuck_since = None
            return False
        if self._direct_cmd is not None:
            self._edge_stuck_since = None
            return False
        if abs(self._last_cmd_linear) < self.NO_PROGRESS_MIN_LINEAR_CMD:
            self._edge_stuck_since = None
            return False

        img_w = float(data.get('image_width', self.IMAGE_WIDTH) or self.IMAGE_WIDTH)
        cx = float(data.get('center_x', self.IMAGE_CENTER_X))
        px_error = abs(cx - img_w * 0.5)
        near_edge = (
            cx <= self.EDGE_STUCK_MARGIN_PX
            or cx >= img_w - self.EDGE_STUCK_MARGIN_PX
        )
        if not near_edge or px_error < self.EDGE_STUCK_MIN_ERROR_PX:
            self._edge_stuck_since = None
            self._edge_stuck_last_dist = None
            return False

        now = time.time()
        if self._edge_stuck_since is None:
            self._edge_stuck_since = now
            self._edge_stuck_last_dist = float(dist_m) if dist_m is not None else None
            self.get_logger().warn(
                f'[EDGE_STUCK] watch start cx={cx:.0f}/{img_w:.0f} '
                f'err={px_error:.0f}px dist={dist_m if dist_m is not None else "?"}')
            return False

        if dist_m is not None and self._edge_stuck_last_dist is not None:
            dist_drop = self._edge_stuck_last_dist - float(dist_m)
            if dist_drop >= self.NO_PROGRESS_MIN_DIST_DROP_M:
                self._edge_stuck_since = now
                self._edge_stuck_last_dist = float(dist_m)
                return False

        elapsed = now - self._edge_stuck_since
        if elapsed >= self.EDGE_STUCK_TIMEOUT_S:
            self.get_logger().warn(
                f'[EDGE_STUCK] elapsed={elapsed:.1f}s cx={cx:.0f}/{img_w:.0f} '
                f'err={px_error:.0f}px cmd_lin={self._last_cmd_linear:.2f}')
            return True
        return False

    def _perception_collision_risk_detected(self, data, dist_m, area_ratio, bottom_ratio) -> bool:
        if not self.navigating or self.target is None:
            self._perception_risk_since = None
            return False
        if self._direct_cmd is not None:
            self._perception_risk_since = None
            return False
        if abs(self._last_cmd_linear) < self.NO_PROGRESS_MIN_LINEAR_CMD:
            self._perception_risk_since = None
            return False

        img_w = float(data.get('image_width', self.IMAGE_WIDTH) or self.IMAGE_WIDTH)
        cx = float(data.get('center_x', self.IMAGE_CENTER_X))
        px_error = abs(cx - img_w * 0.5)

        edge_score = min(1.0, max(0.0, (px_error - self.CENTER_TOLERANCE) / max(1.0, img_w * 0.5 - self.CENTER_TOLERANCE)))
        close_score = 0.0
        if dist_m is not None:
            close_score = min(1.0, max(0.0, (2.0 - float(dist_m)) / 1.2))
        area_score = min(1.0, max(0.0, (float(area_ratio) - 0.03) / 0.12))
        bottom_score = min(1.0, max(0.0, (float(bottom_ratio) - 0.70) / 0.25))

        risk = (
            0.60 * edge_score
            + 0.20 * close_score
            + 0.10 * area_score
            + 0.10 * bottom_score
        )

        now = time.time()
        if now - self._last_perception_risk_log > 1.0:
            self._last_perception_risk_log = now
            self.get_logger().info(
                f'[PERCEPTION_RISK] risk={risk:.2f} edge={edge_score:.2f} '
                f'close={close_score:.2f} area={area_score:.2f} bottom={bottom_score:.2f} '
                f'cx={cx:.0f}/{img_w:.0f} dist={dist_m if dist_m is not None else "?"}')

        if risk < self.PERCEPTION_RISK_THRESHOLD:
            self._perception_risk_since = None
            return False

        if self._perception_risk_since is None:
            self._perception_risk_since = now
            return False

        elapsed = now - self._perception_risk_since
        if elapsed >= self.PERCEPTION_RISK_TIMEOUT_S:
            self.get_logger().warn(
                f'[PERCEPTION_RISK] high risk for {elapsed:.1f}s >= {self.PERCEPTION_RISK_TIMEOUT_S:.1f}s')
            return True
        return False

    def _publish_cmd(self, cmd: Twist):
        self._last_cmd_linear = cmd.linear.x
        self._last_cmd_angular = cmd.angular.z
        self._collision_stop_reason = 'collision'
        if self._collision_detected(cmd):
            self.get_logger().warn('[COLLISION] commanded motion but odometry is stalled')
            self.stop_robot(self._collision_stop_reason)
            return
        self.pub_cmd_vel.publish(cmd)

    def _collision_detected(self, cmd: Twist) -> bool:
        if not self.COLLISION_DETECTION:
            return False

        linear_cmd = abs(cmd.linear.x)
        angular_cmd = abs(cmd.angular.z)
        moving = (
            linear_cmd >= self.COLLISION_MIN_LINEAR_CMD
            or angular_cmd >= self.COLLISION_MIN_ANGULAR_CMD
        )
        if not moving:
            self._reset_collision_watchdog()
            return False

        now = time.time()

        if self.current_pose is None or not self._odom_encoder_fresh:
            self._collision_ref_pose = None
            self._collision_ref_time = None
            if self._odom_stale_motion_since is None:
                self._odom_stale_motion_since = now
                self.get_logger().warn(
                    '[ODOM_STALE] commanded motion but encoder feedback is unavailable')
                return False
            stale_age = now - self._odom_stale_motion_since
            if stale_age >= self.ODOM_STALE_MOTION_STOP_S:
                self.get_logger().warn(
                    f'[ODOM_STALE] no fresh encoder feedback for {stale_age:.1f}s while moving')
                self._collision_stop_reason = 'encoder_lost'
                return True
            return False

        self._odom_stale_motion_since = None

        if self._collision_ref_pose is None or self._collision_ref_time is None:
            self._collision_ref_pose = (
                self.current_pose[0],
                self.current_pose[1],
                self.current_yaw,
            )
            self._collision_ref_time = now
            return False

        elapsed = now - self._collision_ref_time
        if elapsed < self.COLLISION_CHECK_WINDOW_S:
            return False

        x0, y0, yaw0 = self._collision_ref_pose
        dx = self.current_pose[0] - x0
        dy = self.current_pose[1] - y0
        translation = math.sqrt(dx * dx + dy * dy)
        dyaw = self.current_yaw - yaw0
        while dyaw > math.pi:
            dyaw -= 2 * math.pi
        while dyaw < -math.pi:
            dyaw += 2 * math.pi
        rotation = abs(dyaw)

        stalled_linear = (
            linear_cmd >= self.COLLISION_MIN_LINEAR_CMD
            and translation < self.COLLISION_MIN_TRANSLATION
        )
        stalled_angular = (
            linear_cmd < self.COLLISION_MIN_LINEAR_CMD
            and angular_cmd >= self.COLLISION_MIN_ANGULAR_CMD
            and rotation < self.COLLISION_MIN_ROTATION
        )

        self._collision_ref_pose = (
            self.current_pose[0],
            self.current_pose[1],
            self.current_yaw,
        )
        self._collision_ref_time = now

        return stalled_linear or stalled_angular


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
