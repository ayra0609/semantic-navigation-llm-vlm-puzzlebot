#!/usr/bin/env python3
"""
puzzlebot_odom_node.py -- PuzzleBot UDP encoder feedback -> ROS2 /odom

Runs on Jetson Nano. The PuzzleBot Hackerboard periodically sends Float32
messages over UDP with topics:
  VelocityEncR  right wheel angular velocity (rad/s)
  VelocityEncL  left wheel angular velocity (rad/s)

This node receives those packets, integrates differential-drive odometry, and
publishes nav_msgs/Odometry on /odom for navigation_node.py.
"""

import errno
import math
import socket
import struct
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


PUZZLEBOT_IP = "192.168.1.1"
PUZZLEBOT_PORT = 3142

WHEEL_BASE = 0.18
WHEEL_RADIUS = 0.05
ENCODER_STALE_TIMEOUT_S = 0.5

ODOM_FRAME = "odom"
BASE_FRAME = "base_link"


def _yaw_to_quaternion(yaw):
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


class PuzzleBotOdomNode(Node):

    def __init__(self):
        super().__init__("puzzlebot_odom_node")

        self.declare_parameter("puzzlebot_ip", PUZZLEBOT_IP)
        self.declare_parameter("puzzlebot_port", PUZZLEBOT_PORT)
        self.declare_parameter("wheel_base", WHEEL_BASE)
        self.declare_parameter("wheel_radius", WHEEL_RADIUS)
        self.declare_parameter("encoder_stale_timeout_s", ENCODER_STALE_TIMEOUT_S)
        self.declare_parameter("publish_tf", True)

        self.puzzlebot_ip = self.get_parameter("puzzlebot_ip").value
        self.puzzlebot_port = int(self.get_parameter("puzzlebot_port").value)
        self.wheel_base = float(self.get_parameter("wheel_base").value)
        self.wheel_radius = float(self.get_parameter("wheel_radius").value)
        self.encoder_stale_timeout_s = float(
            self.get_parameter("encoder_stale_timeout_s").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)

        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect((self.puzzlebot_ip, self.puzzlebot_port))
        self.sock.setblocking(False)

        self.buffer = bytearray()
        self.wr = 0.0
        self.wl = 0.0
        self.last_enc_time = None
        self.last_update_time = time.time()
        self.last_log_time = 0.0

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.create_timer(0.01, self._spin_once)
        self.get_logger().info(
            f"[UDP] Listening for VelocityEncR/L from "
            f"{self.puzzlebot_ip}:{self.puzzlebot_port}")
        self.get_logger().info("[PUB] /odom ready")

    def _spin_once(self):
        self._read_udp_packets()

        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now
        if dt <= 0.0 or dt > 0.5:
            return

        v = self.wheel_radius * (self.wr + self.wl) * 0.5
        omega = self.wheel_radius * (self.wr - self.wl) / self.wheel_base

        if abs(omega) < 1e-6:
            self.x += v * dt * math.cos(self.theta)
            self.y += v * dt * math.sin(self.theta)
        else:
            self.x += (v / omega) * (
                math.sin(self.theta + omega * dt) - math.sin(self.theta))
            self.y -= (v / omega) * (
                math.cos(self.theta + omega * dt) - math.cos(self.theta))
        self.theta = math.atan2(
            math.sin(self.theta + omega * dt),
            math.cos(self.theta + omega * dt),
        )

        enc_age = (
            now - self.last_enc_time
            if self.last_enc_time is not None else float("inf")
        )
        encoder_fresh = enc_age <= self.encoder_stale_timeout_s

        self._publish_odom(v, omega, encoder_fresh)

        if now - self.last_log_time > 2.0:
            self.last_log_time = now
            self.get_logger().info(
                f"[ODOM] x={self.x:.3f} y={self.y:.3f} "
                f"theta={self.theta:.2f} wr={self.wr:.3f} wl={self.wl:.3f} "
                f"enc_age={enc_age:.2f}s fresh={encoder_fresh}")

    def _read_udp_packets(self):
        while True:
            try:
                data = self.sock.recv(2048)
                if not data:
                    return
                self.buffer.extend(data)
            except socket.error as exc:
                if exc.args and exc.args[0] in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                self.get_logger().warn(f"[UDP] recv failed: {exc}")
                break

        while True:
            msg = self._try_parse_one_message()
            if msg is None:
                break
            topic, value, stamp = msg
            if topic == "VelocityEncR":
                self.wr = value
                self.last_enc_time = time.time()
            elif topic == "VelocityEncL":
                self.wl = value
                self.last_enc_time = time.time()

    def _try_parse_one_message(self):
        magic = bytes([254, 238])
        start = self.buffer.find(magic)
        if start < 0:
            if len(self.buffer) > 2:
                del self.buffer[:-1]
            return None
        if start > 0:
            del self.buffer[:start]

        header_len = 15
        if len(self.buffer) < header_len:
            return None

        msg_type = self.buffer[2]
        name_size = struct.unpack("i", self.buffer[3:7])[0]
        data_size = struct.unpack("i", self.buffer[7:11])[0]
        checksum = struct.unpack("i", self.buffer[11:15])[0]
        expected = msg_type + sum(self.buffer[3:7]) + sum(self.buffer[7:11])

        if checksum != expected or name_size < 0 or data_size < 0:
            del self.buffer[0]
            return None

        total_len = header_len + name_size + data_size + 4
        if len(self.buffer) < total_len:
            return None

        body_start = header_len
        body_end = header_len + name_size + data_size
        body = self.buffer[body_start:body_end]
        body_checksum = struct.unpack("i", self.buffer[body_end:total_len])[0]
        if body_checksum != sum(body):
            del self.buffer[0]
            return None

        topic = body[:name_size].decode("utf-8", errors="replace")
        payload = body[name_size:]
        del self.buffer[:total_len]

        if msg_type != 1 or data_size < 8:
            return None

        value = struct.unpack("f", payload[0:4])[0]
        stamp = struct.unpack("f", payload[4:8])[0]
        return topic, float(value), float(stamp)

    def _publish_odom(self, v, omega, encoder_fresh):
        stamp = self.get_clock().now().to_msg()
        qx, qy, qz, qw = _yaw_to_quaternion(self.theta)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = ODOM_FRAME
        odom.child_frame_id = BASE_FRAME
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = omega

        # navigation_node treats covariance[0] < 0 as "encoder feedback stale".
        # This prevents false collision/stall stops when /odom is alive but the
        # Hackerboard is not actually sending VelocityEncR/L packets.
        odom.pose.covariance[0] = 0.02 if encoder_fresh else -1.0
        odom.pose.covariance[7] = 0.02
        odom.pose.covariance[35] = 0.05
        odom.twist.covariance[0] = 0.02
        odom.twist.covariance[35] = 0.05

        self.pub_odom.publish(odom)

        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = ODOM_FRAME
            t.child_frame_id = BASE_FRAME
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.translation.z = 0.0
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(t)

    def destroy_node(self):
        self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PuzzleBotOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
