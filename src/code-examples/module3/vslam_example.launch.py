from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    This is a conceptual launch file demonstrating how to start an Isaac ROS VSLAM node.
    In a real application, you would use the official launch files from the
    isaac_ros_vslam package, which are more complex and configurable.
    This example simplifies the concept for educational purposes.
    """

    # Declare launch arguments to make the launch file configurable
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo or Isaac Sim) clock.')

    # VSLAM Node
    # This node performs the Visual-SLAM algorithm.
    vslam_node = Node(
        package='isaac_ros_vslam',
        executable='isaac_ros_visual_slam_node',
        name='isaac_ros_visual_slam',
        parameters=[{
            # Using simulation time is crucial when working with simulators
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            
            # Input topics - these must match what your simulator is publishing
            'left_image_topic': '/camera/left/image_raw',
            'left_camera_info_topic': '/camera/left/camera_info',
            'right_image_topic': '/camera/right/image_raw',
            'right_camera_info_topic': '/camera/right/camera_info',
            
            # Enable IMU fusion for better accuracy
            'enable_imu_fusion': True,
            'imu_topic': '/camera/imu',
            
            # Output topics
            'visual_slam/odometry_topic': '/vis/slam/odometry',
            'visual_slam/point_cloud_topic': '/vis/slam/point_cloud',
        }],
        # Remapping can also be used to connect topics
        # remappings=[
        #     ('left/image_raw', '/your/sim/left_cam'),
        #     ('right/image_raw', '/your/sim/right_cam'),
        # ]
    )

    return LaunchDescription([
        use_sim_time,
        vslam_node
    ])
