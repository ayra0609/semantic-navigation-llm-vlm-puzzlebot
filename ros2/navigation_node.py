#!/usr/bin/env python3
"""
navigation_node.py - ROS2 Navigation Node for PuzzleBot
Runs on Jetson Nano. Receives detection results from PC and controls robot.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import json
import math


class PidController:
    def __init__(self, Kp=0.003, Ti=1.0, Td=0.0):
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

    # ── Visual servoing parameters ────────────────────────────────────────────
    IMAGE_WIDTH    = 640
    IMAGE_HEIGHT   = 480
    IMAGE_CENTER_X = IMAGE_WIDTH / 2

    LINEAR_SPEED   = 0.15   # m/s  — forward speed once aligned

    # Alignment threshold — must be centered within this many pixels
    # before the robot moves forward
    CENTER_TOLERANCE = 40   # pixels

    # Stop conditions
    STOP_AREA_RATIO      = 0.20   # stop when bbox covers 20% of image
    LIDAR_STOP_DISTANCE  = 0.40   # stop if obstacle within 0.4m
    MIN_DETECTION_SCORE  = 0.25   # ignore low-confidence detections

    def __init__(self):
        super().__init__('navigation_node')

        # PID controller for angular steering
        self.pid = PidController(Kp=0.003, Ti=1.0, Td=0.0)
        self._last_control_time = None

        # State
        self.target          = None
        self.detection       = None
        self.lidar_distance  = float('inf')
        self.navigating      = False
        self.start_pose      = None
        self.current_pose    = None

        # Subscribers
        self.sub_target = self.create_subscription(
            String, '/nav_target', self.target_callback, 10)
        self.sub_detection = self.create_subscription(
            String, '/detection_result', self.detection_callback, 10)
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # Publisher
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        # Control loop: 10 Hz
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Navigation node started, waiting for target...')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def target_callback(self, msg):
        self.target              = msg.data
        self.navigating          = True
        self.detection           = None
        self.start_pose          = self.current_pose
        self._last_control_time  = None
        self.pid.reset()                # reset PID integrator for new target
        self.get_logger().info(f'New target: "{self.target}"')

    def detection_callback(self, msg):
        try:
            data = json.loads(msg.data)
            # Ignore low-confidence detections
            if data.get('score', 0.0) >= self.MIN_DETECTION_SCORE or not data.get('found', False):
                self.detection = data
        except json.JSONDecodeError:
            self.get_logger().warn('Invalid detection JSON')

    def scan_callback(self, msg):
        n = len(msg.ranges)
        sector = list(range(0, n // 24)) + list(range(n - n // 24, n))
        valid = [msg.ranges[i] for i in sector
                 if not math.isinf(msg.ranges[i]) and not math.isnan(msg.ranges[i])]
        self.lidar_distance = min(valid) if valid else float('inf')

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        self.current_pose = (pos.x, pos.y)

    # ── Control Loop ──────────────────────────────────────────────────────────

    def control_loop(self):
        if not self.navigating or self.target is None:
            return

        cmd = Twist()

        # ── Safety: LiDAR override ────────────────────────────────────────────
        if self.lidar_distance < self.LIDAR_STOP_DISTANCE:
            self.get_logger().info(
                f'Obstacle at {self.lidar_distance:.2f}m — stopping!')
            self.stop_robot('obstacle_stop')
            return

        # ── No detection: rotate to search ────────────────────────────────────
        if self.detection is None or not self.detection.get('found', False):
            self.get_logger().info(
                f'Searching for "{self.target}"...',
                throttle_duration_sec=1.0)
            cmd.angular.z = 0.3   # rad/s — slow search rotation
            self.pub_cmd_vel.publish(cmd)
            return

        # ── Target found: visual servoing ─────────────────────────────────────
        center_x     = self.detection.get('center_x', self.IMAGE_CENTER_X)
        box          = self.detection.get('box', [0, 0, 0, 0])
        image_width  = self.detection.get('image_width',  self.IMAGE_WIDTH)
        image_height = self.detection.get('image_height', self.IMAGE_HEIGHT)
        score        = self.detection.get('score', 0.0)

        box_area   = (box[2] - box[0]) * (box[3] - box[1])
        image_area = image_width * image_height
        area_ratio = box_area / image_area if image_area > 0 else 0

        # Stop condition
        if area_ratio > self.STOP_AREA_RATIO:
            self.get_logger().info(
                f'Reached "{self.target}"! '
                f'area={area_ratio:.2f}  score={score:.2f}')
            self.stop_robot('success')
            return

        error = center_x - (image_width / 2.0)

        # ── Compute dt ────────────────────────────────────────────────────────
        import time as _time
        now = _time.time()
        dt  = (now - self._last_control_time) if self._last_control_time else 0.1
        self._last_control_time = now

        # ── Phase 1: align first (pure rotation, no forward motion) ───────────
        if abs(error) > self.CENTER_TOLERANCE:
            cmd.linear.x  = 0.0
            cmd.angular.z = -self.pid.compute(error, dt)
            self.get_logger().info(
                f'ALIGNING  error={error:+.0f}px  '
                f'area={area_ratio:.2f}  ang={cmd.angular.z:+.3f}',
                throttle_duration_sec=0.3)

        # ── Phase 2: move forward once aligned ────────────────────────────────
        else:
            cmd.linear.x  = self.LINEAR_SPEED
            cmd.angular.z = -self.pid.compute(error * 0.5, dt)  # gentler correction
            self.get_logger().info(
                f'ADVANCING  error={error:+.0f}px  '
                f'area={area_ratio:.2f}  score={score:.2f}',
                throttle_duration_sec=0.3)

        self.pub_cmd_vel.publish(cmd)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def stop_robot(self, reason: str = 'stop'):
        self.pub_cmd_vel.publish(Twist())
        self.navigating = False
        self.detection  = None
        self._log_result(reason)
        self.target = None

    def _log_result(self, reason: str):
        dist_str = '?'
        if self.start_pose and self.current_pose:
            dx = self.current_pose[0] - self.start_pose[0]
            dy = self.current_pose[1] - self.start_pose[1]
            dist_str = f'{math.sqrt(dx**2+dy**2):.2f}m'
        self.get_logger().info(
            f'Navigation ended | reason={reason} | '
            f'distance={dist_str} | lidar={self.lidar_distance:.2f}m')


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
