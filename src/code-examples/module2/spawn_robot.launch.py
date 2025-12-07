import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    """
    This launch file starts Gazebo and spawns a robot from a URDF file.
    NOTE: This launch file assumes it is part of a ROS 2 package that has
    a URDF file installed in 'share/<package_name>/urdf'.
    For this example, we point to the URDF created in the module 1 examples.
    A real-world implementation would have its own package.
    """
    
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    
    # In a real package, you would discover your own package share directory
    # For this example, we assume a path. THIS IS NOT ROBUST.
    # A better way:
    # pkg_your_robot = get_package_share_directory('your_robot_pkg_name')
    # urdf_path = os.path.join(pkg_your_robot, 'urdf', 'simple_bot_enhanced.urdf')

    # For this standalone example, we construct a relative path.
    # This assumes you run the launch file from the root of a workspace
    # that contains the 'src' directory with the code examples.
    # This is for demonstration only.
    urdf_path = 'src/code-examples/module1/simple_bot.urdf' # Using the one from module 1

    # Start Gazebo server and client
    gzserver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        )
    )
    gzclient_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    # Spawn the robot entity
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'simple_bot',
            '-file', urdf_path,
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5'
        ],
        output='screen'
    )

    return LaunchDescription([
        gzserver_launch,
        gzclient_launch,
        spawn_entity,
    ])
