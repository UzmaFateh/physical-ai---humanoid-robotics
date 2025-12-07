import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # This launch file is intended to be run from a package that has this file.
    # For demonstration purposes, we assume the URDF file is in a known path relative
    # to the package that would contain this launch file.
    # In a real package, you would install the URDF file and find it with get_package_share_directory.
    
    # A more robust way in a package would be:
    # urdf_path = os.path.join(
    #     get_package_share_directory('your_package_name_here'),
    #     'urdf', 'simple_bot.urdf')

    # For this standalone example, we assume it's in the same directory.
    # This is NOT best practice for a real ROS package.
    urdf_path = 'simple_bot.urdf'

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': open(urdf_path).read()}])
    ])
