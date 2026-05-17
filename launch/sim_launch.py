"""
sim_launch.py
─────────────────────────────────────────────────────────────────
Starts:
  1. Gazebo Fortress (headless, llvmpipe software rendering for WSL)
  2. ros_gz_bridge — eager mode, no lazy activation needed
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    _here = os.path.dirname(os.path.abspath(__file__))
    default_world = os.path.join(_here, '..', 'worlds', 'sim_world.sdf')

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=default_world,
        description='Path to Gazebo world SDF file',
    )

    # ── 1. Gazebo Fortress (server-only, software rendering) ─────
    # IGN_GAZEBO_RESOURCE_PATH includes the worlds/ directory so that
    # relative mesh URIs (./meshes/chassis1.dae) resolve correctly
    # regardless of the process working directory.
    worlds_dir = os.path.join(_here, '..', 'worlds')

    gazebo = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', '-s', '--headless-rendering', '-v1',
             LaunchConfiguration('world')],
        additional_env={
            'LIBGL_ALWAYS_SOFTWARE':    '1',
            'GALLIUM_DRIVER':           'llvmpipe',
            'MESA_GL_VERSION_OVERRIDE': '3.3',
            'MESA_GLSL_CACHE_DISABLE':  '1',
            'IGN_GAZEBO_RESOURCE_PATH': worlds_dir,
        },
        output='screen',
    )

    # ── 2. ros_gz_bridge (eager, non-lazy) ───────────────────────
    #   ros_ign_bridge is deprecated and redirects here; use directly.
    #   lazy:=false forces the bridge to publish immediately without
    #   waiting for a ROS subscriber — fixes WSL DDS discovery issues.
    bridge = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                name='ros_gz_bridge',
                parameters=[{'lazy': False}],
                arguments=[
                    '/video_source/raw@sensor_msgs/msg/Image[ignition.msgs.Image',
                    '/overview/raw@sensor_msgs/msg/Image[ignition.msgs.Image',
                    '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
                    '/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
                    '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
                ],
                output='screen',
            )
        ],
    )

    return LaunchDescription([
        world_arg,
        gazebo,
        bridge,
    ])
