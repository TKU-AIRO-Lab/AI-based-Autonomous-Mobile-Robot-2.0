from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='py_code',
            executable='lidar_node',
            name='lidar_node',
            output='screen'
        ),
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name='v4l2_camera_node',
            output='screen'
        ),
        # 3. 啟動 YOLO 辨識 Node
        Node(
            package='py_code',
            executable='yolo_node',
            name='yolo_node',
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'
        )
    ])