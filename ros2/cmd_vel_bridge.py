#!/usr/bin/env python3
"""
cmd_vel_bridge.py  —  ROS2 /cmd_vel → PuzzleBot UDP wheel velocities
Runs on Jetson Nano.

Subscribes: /cmd_vel  (geometry_msgs/Twist)
Sends UDP:  VelocitySetR, VelocitySetL  (float32, rad/s)  → 192.168.1.1:3142
"""

import errno
import math
import socket
import struct
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

PUZZLEBOT_IP   = "192.168.1.1"
PUZZLEBOT_PORT = 3142
RESEND_HZ      = 20.0
CMD_STALE_S    = 0.50
ODOM_HZ        = 100.0
ENCODER_STALE_TIMEOUT_S = 0.5
ENCODER_LOST_STOP_S = 3.0

WHEEL_BASE   = 0.18   # m — distance between wheels
WHEEL_RADIUS = 0.05   # m
CMD_MAX      = 8.0    # rad/s — max wheel speed
ODOM_FRAME   = "odom"
BASE_FRAME   = "base_link"


def _yaw_to_quaternion(yaw):
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def _build_packet(topic: str, value: float):
    """Return (head_bytes, body_bytes) for a PuzzleBot Float32 message."""
    topic_b   = topic.encode('utf-8')
    name_size = len(topic_b)
    data_size = 8                     # float32 value (4) + float32 stamp (4)
    stamp     = float(time.time() % 1000.0)

    head = bytearray([254, 238, 1])   # magic + type=1 (Float32)
    head.extend(struct.pack("i", name_size))
    head.extend(struct.pack("i", data_size))
    head.extend(struct.pack("i", sum(head[2:])))   # checksum1

    body = bytearray()
    body.extend(topic_b)
    body.extend(struct.pack("f", value))
    body.extend(struct.pack("f", stamp))
    body.extend(struct.pack("i", sum(body)))       # checksum2

    return bytes(head), bytes(body)


class CmdVelBridge(Node):

    def __init__(self):
        super().__init__('cmd_vel_bridge')

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.connect((PUZZLEBOT_IP, PUZZLEBOT_PORT))
        self._sock.setblocking(False)
        self.get_logger().info(
            f'[UDP] Socket ready -> {PUZZLEBOT_IP}:{PUZZLEBOT_PORT}')

        self._log_count  = 0
        self._last_vr    = None
        self._last_vl    = None
        self._cmd_vr     = 0.0
        self._cmd_vl     = 0.0
        self._last_cmd_time = 0.0
        self._zero_sent_after_stale = False
        self._udp_buffer = bytearray()
        self._wr = 0.0
        self._wl = 0.0
        self._last_enc_time = None
        self._last_odom_update = time.time()
        self._last_odom_log = 0.0
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0

        self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)
        self.pub_odom = self.create_publisher(Odometry, '/odom', 10)
        self.pub_status = self.create_publisher(String, '/puzzlebot/status', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(1.0 / RESEND_HZ, self._resend_latest_cmd)
        self.create_timer(1.0 / ODOM_HZ, self._odom_tick)
        self._encoder_stale_status_sent = False
        self.get_logger().info('[SUB] /cmd_vel  bridge ready')
        self.get_logger().info('[PUB] /odom ready from VelocityEncR/L feedback')

    def _on_cmd_vel(self, msg: Twist):
        v = msg.linear.x
        w = msg.angular.z

        vr = (v + w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
        vl = (v - w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
        vr = max(-CMD_MAX, min(CMD_MAX, vr))
        vl = max(-CMD_MAX, min(CMD_MAX, vl))
        self._cmd_vr = vr
        self._cmd_vl = vl
        self._last_cmd_time = time.time()
        self._zero_sent_after_stale = False

        # Log every receive — throttle to once every 5 calls (~2 Hz at 10 Hz loop)
        self._log_count += 1
        changed = (vr != self._last_vr or vl != self._last_vl)
        if self._log_count % 5 == 1 or changed:
            self.get_logger().info(
                f'[SUB] /cmd_vel  lin={v:.2f}  ang={w:.3f}'
                f'  ->  VR={vr:.2f}  VL={vl:.2f}')
            self._last_vr = vr
            self._last_vl = vl

        self._send_wheels(
            vr, vl,
            log=(self._log_count % 5 == 1 or changed),
            prefix='PuzzleBot')

    def _resend_latest_cmd(self):
        if self._last_cmd_time <= 0.0:
            return
        age = time.time() - self._last_cmd_time
        if age > CMD_STALE_S:
            if not self._zero_sent_after_stale:
                self._cmd_vr = 0.0
                self._cmd_vl = 0.0
                self._send_wheels(0.0, 0.0, log=True, prefix='STALE STOP')
                self._zero_sent_after_stale = True
            return
        self._send_wheels(self._cmd_vr, self._cmd_vl)

    def _send_wheels(self, vr: float, vl: float, log: bool = False, prefix: str = 'PuzzleBot'):
        try:
            head_r, body_r = _build_packet("VelocitySetR", vr)
            head_l, body_l = _build_packet("VelocitySetL", vl)
            self._sock.send(head_r)
            self._sock.send(body_r)
            self._sock.send(head_l)
            self._sock.send(body_l)
            if log:
                self.get_logger().info(
                    f'[UDP->] {prefix}  VR={vr:.2f}  VL={vl:.2f}')
        except Exception as e:
            self.get_logger().error(f'[UDP] Send failed: {e}')

    def _odom_tick(self):
        self._read_udp_packets()

        now = time.time()
        dt = now - self._last_odom_update
        self._last_odom_update = now
        if dt <= 0.0 or dt > 0.5:
            return

        enc_age = (
            now - self._last_enc_time
            if self._last_enc_time is not None else float('inf')
        )
        encoder_fresh = enc_age <= ENCODER_STALE_TIMEOUT_S
        commanded_moving = (
            abs(self._cmd_vr) > 0.05 or abs(self._cmd_vl) > 0.05
        )

        if encoder_fresh:
            self._encoder_stale_status_sent = False
            v = WHEEL_RADIUS * (self._wr + self._wl) * 0.5
            omega = WHEEL_RADIUS * (self._wr - self._wl) / WHEEL_BASE
        else:
            v = 0.0
            omega = 0.0
            if (commanded_moving
                    and enc_age >= ENCODER_LOST_STOP_S
                    and not self._encoder_stale_status_sent):
                self.get_logger().warn(
                    f'[ENCODER_STALE] commanded wheels but no fresh feedback for {enc_age:.2f}s')
                self._publish_status(
                    'ERROR: Encoder feedback lost -- wheel encoder feedback stopped while moving')
                self._encoder_stale_status_sent = True

        if abs(omega) < 1e-6:
            self._x += v * dt * math.cos(self._theta)
            self._y += v * dt * math.sin(self._theta)
        else:
            self._x += (v / omega) * (
                math.sin(self._theta + omega * dt) - math.sin(self._theta))
            self._y -= (v / omega) * (
                math.cos(self._theta + omega * dt) - math.cos(self._theta))
        self._theta = math.atan2(
            math.sin(self._theta + omega * dt),
            math.cos(self._theta + omega * dt),
        )

        self._publish_odom(v, omega, encoder_fresh)

        if now - self._last_odom_log > 2.0:
            self._last_odom_log = now
            self.get_logger().info(
                f'[ODOM] x={self._x:.3f} y={self._y:.3f} '
                f'theta={self._theta:.2f} wr={self._wr:.3f} wl={self._wl:.3f} '
                f'enc_age={enc_age:.2f}s fresh={encoder_fresh}')

    def _read_udp_packets(self):
        while True:
            try:
                data = self._sock.recv(2048)
                if not data:
                    return
                self._udp_buffer.extend(data)
            except socket.error as exc:
                if exc.args and exc.args[0] in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                self.get_logger().warn(f'[UDP] recv failed: {exc}')
                break

        while True:
            msg = self._try_parse_one_message()
            if msg is None:
                break
            topic, value, stamp = msg
            if topic == 'VelocityEncR':
                self._wr = value
                self._last_enc_time = time.time()
            elif topic == 'VelocityEncL':
                self._wl = value
                self._last_enc_time = time.time()

    def _try_parse_one_message(self):
        magic = bytes([254, 238])
        start = self._udp_buffer.find(magic)
        if start < 0:
            if len(self._udp_buffer) > 2:
                del self._udp_buffer[:-1]
            return None
        if start > 0:
            del self._udp_buffer[:start]

        header_len = 15
        if len(self._udp_buffer) < header_len:
            return None

        msg_type = self._udp_buffer[2]
        name_size = struct.unpack('i', self._udp_buffer[3:7])[0]
        data_size = struct.unpack('i', self._udp_buffer[7:11])[0]
        checksum = struct.unpack('i', self._udp_buffer[11:15])[0]
        expected = msg_type + sum(self._udp_buffer[3:7]) + sum(self._udp_buffer[7:11])

        if checksum != expected or name_size < 0 or data_size < 0:
            del self._udp_buffer[0]
            return None

        total_len = header_len + name_size + data_size + 4
        if len(self._udp_buffer) < total_len:
            return None

        body_start = header_len
        body_end = header_len + name_size + data_size
        body = self._udp_buffer[body_start:body_end]
        body_checksum = struct.unpack('i', self._udp_buffer[body_end:total_len])[0]
        if body_checksum != sum(body):
            del self._udp_buffer[0]
            return None

        topic = body[:name_size].decode('utf-8', errors='replace')
        payload = body[name_size:]
        del self._udp_buffer[:total_len]

        if msg_type != 1 or data_size < 8:
            return None

        value = struct.unpack('f', payload[0:4])[0]
        stamp = struct.unpack('f', payload[4:8])[0]
        return topic, float(value), float(stamp)

    def _publish_odom(self, v, omega, encoder_fresh):
        stamp = self.get_clock().now().to_msg()
        qx, qy, qz, qw = _yaw_to_quaternion(self._theta)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = ODOM_FRAME
        odom.child_frame_id = BASE_FRAME
        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = omega
        odom.pose.covariance[0] = 0.02 if encoder_fresh else -1.0
        odom.pose.covariance[7] = 0.02
        odom.pose.covariance[35] = 0.05
        odom.twist.covariance[0] = 0.02
        odom.twist.covariance[35] = 0.05
        self.pub_odom.publish(odom)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = ODOM_FRAME
        t.child_frame_id = BASE_FRAME
        t.transform.translation.x = self._x
        t.transform.translation.y = self._y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.pub_status.publish(msg)

    def _send_zero(self):
        try:
            head_r, body_r = _build_packet("VelocitySetR", 0.0)
            head_l, body_l = _build_packet("VelocitySetL", 0.0)
            self._sock.send(head_r); self._sock.send(body_r)
            self._sock.send(head_l); self._sock.send(body_l)
            self.get_logger().info('[UDP->] PuzzleBot  STOP  VR=0.00  VL=0.00')
        except Exception:
            pass

    def destroy_node(self):
        self._send_zero()
        self._sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
