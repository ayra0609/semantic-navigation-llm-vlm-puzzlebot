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

    LINEAR_SPEED   = 0.15   # m/s -- forward speed once aligned

    # Alignment threshold: must be centered within this many pixels
    # 80px is more forgiving of DINO bounding-box jitter on the real robot
    CENTER_TOLERANCE = 80   # pixels (was 40 -- too tight for 0.5 Hz DINO)

    # Stop conditions
    STOP_AREA_RATIO      = 0.40   # stop when bbox covers 40% of image
    STOP_AREA_MIN_SCORE  = 0.60   # area-based stop requires this confidence
    LIDAR_STOP_DISTANCE  = 0.40   # stop if obstacle within 0.4 m
    MIN_DETECTION_SCORE  = 0.25   # ignore low-confidence detections
    DR_MIN_SCORE         = 0.50   # min score to update dead-reckoning distance
    DIST_STOP_DISTANCE   = 0.30   # stop immediately when depth reading is this close

    # Detection hysteresis
    # How long (s) to keep using the last positive detection before switching
    # to SEARCH mode. Without this, a single missed DINO frame immediately
    # stops forward motion and starts the robot spinning.
    DETECTION_TIMEOUT = 4.0   # seconds
    SEARCH_TIMEOUT    = 10.0  # seconds without detection before stopping
    FOCAL_PX         = 644.69
    ARRIVE_THRESHOLD = 0.25
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
        self.current_yaw        = 0.0
        self.target_world_pos   = None
        self.target_dist        = None
        self._nav_start_wall    = None
        self._search_start_wall = None
        self._ever_detected     = False
        # Dead-reckoning stop (used when odometry unavailable but depth available)
        self._dr_dist      = None  # latest distance estimate from depth
        self._dr_advance_s = 0.0   # seconds spent advancing since last dist update
        self._dist_arrived = False  # set when live depth reading reaches DIST_STOP_DISTANCE
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
            self._search_start_wall = None
            self._ever_detected     = False
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
        self._search_start_wall = None
        self._ever_detected     = False
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

            if found and score >= self.MIN_DETECTION_SCORE:
                # Positive detection: accept and refresh timestamp
                self.detection            = data
                self._last_detection_time = time.time()
                self._search_start_wall   = None
                if not self._ever_detected:
                    self._publish_status(f'NAVIGATING: Target detected "{self.target}"')
                self._ever_detected       = True
                self.get_logger().info(
                    f'[DETECTION] found=True  score={score:.2f}  '
                    f'cx={data.get("center_x", 0):.0f}')

                dist_m = data.get('distance_m', None)

                # Immediate stop when live depth says we are close enough.
                # DR tracking can lag behind the actual distance due to noise,
                # so this direct check prevents the robot from ramming the target.
                if (dist_m and dist_m <= self.DIST_STOP_DISTANCE
                        and score >= self.DR_MIN_SCORE and self.navigating):
                    self.get_logger().info(
                        f'[DIST_ARRIVED] live dist={dist_m:.2f}m <= {self.DIST_STOP_DISTANCE}m -- flagging stop')
                    self._dist_arrived = True

                # Update dead-reckoning estimate (high-confidence detections only).
                # Total distance can only decrease — prevents inflated estimates from
                # a bbox that is wider than TARGET_WIDTH_M from extending the journey.
                if dist_m and dist_m > self.ARRIVE_THRESHOLD and score >= self.DR_MIN_SCORE:
                    covered = self._dr_advance_s * self.LINEAR_SPEED
                    new_total = covered + dist_m
                    if self._dr_dist is None:
                        self._dr_dist = new_total
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

        # Immediate stop: live depth reading reached DIST_STOP_DISTANCE
        if self._dist_arrived:
            self.get_logger().info('[DIST_ARRIVED] stopping -- live depth threshold reached')
            self.stop_robot('arrived')
            return

        # Safety: LiDAR override
        if self.lidar_distance < self.LIDAR_STOP_DISTANCE:
            self.get_logger().info(
                f'Obstacle at {self.lidar_distance:.2f}m -- stopping!')
            self.stop_robot('obstacle_stop')
            return

        # Direct motion command (__forward / __backward / __left / __right)
        # Executed every tick until a stop or new command arrives.
        if self._direct_cmd is not None:
            lin, ang = self._direct_cmd
            cmd.linear.x  = lin
            cmd.angular.z = ang
            self.get_logger().info(
                f'[CMD_VEL] DIRECT  lin={lin:.2f}  ang={ang:.2f}')
            self.pub_cmd_vel.publish(cmd)
            return

        # If the target was detected earlier with depth, keep driving to the
        # locked world position even if the low-mounted camera now only sees legs.
        if self.target_world_pos is not None and self.current_pose is not None:
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
            self.pub_cmd_vel.publish(cmd)
            return

        # No detection (or timed-out): rotate to search
        detection_valid = (
            self.detection is not None
            and self.detection.get('found', False)
            and (time.time() - self._last_detection_time) < self.DETECTION_TIMEOUT
        )
        if not detection_valid:
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
            cmd.angular.z = 0.15
            self.get_logger().info(
                f'[CMD_VEL] SEARCH "{self.target}"  ang=+0.15')
            self.pub_cmd_vel.publish(cmd)
            return
        
        # Target found: visual servoing
        center_x     = self.detection.get('center_x', self.IMAGE_CENTER_X)
        box          = self.detection.get('box', [0, 0, 0, 0])
        image_width  = self.detection.get('image_width',  self.IMAGE_WIDTH)
        image_height = self.detection.get('image_height', self.IMAGE_HEIGHT)
        score        = self.detection.get('score', 0.0)

        box_area   = (box[2] - box[0]) * (box[3] - box[1])
        image_area = image_width * image_height
        area_ratio = box_area / image_area if image_area > 0 else 0

        # Area-based stop only when no depth estimate available (DR not active).
        # When DR is active, distance-based stop takes over in Phase 2.
        if (area_ratio > self.STOP_AREA_RATIO and score >= self.STOP_AREA_MIN_SCORE
                and self._dr_dist is None):
            self.get_logger().info(
                f'Reached "{self.target}"!  area={area_ratio:.2f}  score={score:.2f}')
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
                if dist_covered >= self._dr_dist - self.ARRIVE_THRESHOLD:
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
                if dist_covered >= self._dr_dist - self.ARRIVE_THRESHOLD:
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
        self.pub_cmd_vel.publish(cmd)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def stop_robot(self, reason: str = 'stop'):
        stopped_target = self.target
        if self._nav_start_wall and self.navigating:
            nav_ms = (time.time() - self._nav_start_wall) * 1000
            self.get_logger().info(
                f'[LATENCY] Navigation total: {nav_ms:.0f}ms  reason={reason}')
        self.get_logger().info(f'[CMD_VEL] STOP  reason={reason}')
        self.pub_cmd_vel.publish(Twist())
        if reason == 'arrived' or reason == 'success':
            self._publish_status('ARRIVED: Reached target!')
        elif reason == 'target_lost':
            self._publish_status(
                f'ERROR: Target lost -- "{stopped_target}" left the camera view')
        elif reason == 'search_timeout':
            self._publish_status(
                f'ERROR: Could not find "{stopped_target}" after searching')
        elif reason == 'obstacle_stop':
            self._publish_status('ERROR: Obstacle detected -- navigation stopped')
        elif reason == 'command_stop':
            self._publish_status('STOPPED')
        self.navigating  = False
        self.detection   = None
        self._direct_cmd = None
        self.target_world_pos = None
        self.target_dist      = None
        self._search_start_wall = None
        self._ever_detected     = False
        self._dr_dist      = None
        self._dr_advance_s = 0.0
        self._dist_arrived = False
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

    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.pub_status.publish(msg)
        self.get_logger().info(f'[STATUS] {text}')


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
