import os
from launch import LaunchDescription
from launch_ros.actions import Node

# 設定獨立 Domain ID，避免和其他機器人的 topic 混在一起
os.environ['ROS_DOMAIN_ID'] = '10'
os.environ['TURTLEBOT3_MODEL'] = 'burger'

def generate_launch_description():
    return LaunchDescription([

        Node(
            package='v4l2_camera',           # 如果你是用自己寫的 node，把這裡換成 'py_code'
            executable='v4l2_camera_node',   # 換成你開相機的執行檔名稱
            name='camera_node',
            output='screen',
        ),

        Node(
            package='py_code',
            executable='main_controller',
            name='main_controller_node',
            output='screen',
            emulate_tty=True,
        ),

        Node(
            package='py_code',
            executable='yolo_node',
            name='yolo_node',
            output='screen',
            additional_env={
                'PYTHONPATH': '/home/tkuai/miniconda3/envs/yolo_env/lib/python3.10/site-packages'
                              ':/home/tkuai/ros2_ws/install/py_code/lib/python3.10/site-packages'
                              ':/opt/ros/humble/lib/python3.10/site-packages'
                              ':/opt/ros/humble/local/lib/python3.10/dist-packages',
                'QT_QPA_FONTDIR': '/usr/share/fonts/truetype/dejavu',
            },
        ),
        Node(
            package='py_code',
            executable='lidar_node',
            name='lidar_node',
            output='screen'
        ),

        Node(
            package='py_code',
            executable='motor_driver',
            name='motor_driver',
            output='screen',
        ),
    ])