#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    ld = LaunchDescription()


    # Lidar-Camera Fusion Node
    filter_fusion_detection_node = Node(
        package='lidar_camera_fusion_detection',
        executable='filter_fusion_detection',
        name='filter_fusion_detection_node',
        parameters=[
            {'min_range': 0.1, 'max_range': 50.0,
             'lidar_frame': 'livox_frame',
             'camera_frame': 'camera_color_optical_frame'}
        ],
        remappings=[
            ('/scan/points', '/livox/lidar'),
            ('/observer/gimbal_camera_info', '/camera/camera/color/camera_info'),
            ('/observer/gimbal_camera', '/camera/camera/color/image_raw'),
            ('/rgb/tracking', '/rgb/tracking')
        ]
    )

    # YOLO Launch for Lidar-Camera Fusion
    yolo_launch_fusion = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('yolo_bringup'),
                'launch/yolov11.launch.py'
            ])
        ]),
        launch_arguments={
            'model': '/home/goghy/Projects/yolo_models/plate-bowl-spoon-super.pt',
            'threshold': '0.6',
            'input_image_topic': '/camera/camera/color/image_raw',
            'namespace': 'rgb',
            'device': 'cpu'
        }.items()
    )


    # Add all nodes and launches to the launch description


    ld.add_action(filter_fusion_detection_node)
    ld.add_action(yolo_launch_fusion)


    return ld
