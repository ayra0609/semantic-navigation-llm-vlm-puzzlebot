#!/usr/bin/env python3
"""
cmd_vel_bridge.py  —  ROS2 /cmd_vel → PuzzleBot UDP wheel velocities
Runs on Jetson Nano.

Subscribes: /cmd_vel  (geometry_msgs/Twist)
Sends UDP:  VelocitySetR, VelocitySetL  (float32, rad/s)  → 192.168.1.1:3142
"""

import socket
import struct
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

PUZZLEBOT_IP   = "192.168.1.1"
PUZZLEBOT_PORT = 3142

WHEEL_BASE   = 0.18   # m — distance between wheels
WHEEL_RADIUS = 0.05   # m
CMD_MAX      = 8.0    # rad/s — max wheel speed


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
        self.get_logger().info(
            f'[UDP] Socket ready -> {PUZZLEBOT_IP}:{PUZZLEBOT_PORT}')

        self._log_count  = 0
        self._last_vr    = None
        self._last_vl    = None

        self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)
        self.get_logger().info('[SUB] /cmd_vel  bridge ready')

    def _on_cmd_vel(self, msg: Twist):
        v = msg.linear.x
        w = msg.angular.z

        vr = (v + w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
        vl = (v - w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
        vr = max(-CMD_MAX, min(CMD_MAX, vr))
        vl = max(-CMD_MAX, min(CMD_MAX, vl))

        # Log every receive — throttle to once every 5 calls (~2 Hz at 10 Hz loop)
        self._log_count += 1
        changed = (vr != self._last_vr or vl != self._last_vl)
        if self._log_count % 5 == 1 or changed:
            self.get_logger().info(
                f'[SUB] /cmd_vel  lin={v:.2f}  ang={w:.3f}'
                f'  ->  VR={vr:.2f}  VL={vl:.2f}')
            self._last_vr = vr
            self._last_vl = vl

        try:
            head_r, body_r = _build_packet("VelocitySetR", vr)
            head_l, body_l = _build_packet("VelocitySetL", vl)
            self._sock.send(head_r)
            self._sock.send(body_r)
            self._sock.send(head_l)
            self._sock.send(body_l)
            if self._log_count % 5 == 1 or changed:
                self.get_logger().info(
                    f'[UDP->] PuzzleBot  VR={vr:.2f}  VL={vl:.2f}')
        except Exception as e:
            self.get_logger().error(f'[UDP] Send failed: {e}')

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