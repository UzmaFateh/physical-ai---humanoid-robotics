from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    This launch file starts the ROS-TCP-Endpoint, which acts as a bridge
    to communicate with external applications like Unity.
    """
    return LaunchDescription([
        Node(
            package='ros_tcp_endpoint',
            executable='main',
            name='ros_tcp_endpoint',
            output='screen',
            parameters=[{
                'ROS_IP': '127.0.0.1', # IP of the ROS machine
                'ROS_TCP_PORT': 10000
            }]
        )
    ])
