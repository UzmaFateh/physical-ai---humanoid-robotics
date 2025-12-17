---
sidebar_position: 3
---

# Chapter 3: ROS 2 Packages and Practical Examples

## Creating a Complete Robot Package

Let's create a more comprehensive robot package that includes URDF, launch files, and control interfaces. This will serve as the foundation for our robot simulation in later modules.

First, let's create a URDF (Unified Robot Description Format) file for a simple robot:

```xml
<!-- simple_robot_pkg/urdf/simple_bot.urdf -->
<?xml version="1.0"?>
<robot name="simple_bot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Left Wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right Wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0.15 0.15 -0.05" rpy="1.57075 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0.15 -0.15 -0.05" rpy="1.57075 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

</robot>
```

## Robot State Publisher

To publish the robot's state, we'll use the robot_state_publisher:

```python
# simple_robot_pkg/robot_state_publisher_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import math

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Create publisher for joint states
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Create transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer for publishing
        self.timer = self.create_timer(0.1, self.publish_joint_states)

        # Initialize joint positions
        self.joint_names = ['left_wheel_joint', 'right_wheel_joint']
        self.joint_positions = [0.0, 0.0]
        self.joint_velocities = [0.0, 0.0]
        self.joint_efforts = [0.0, 0.0]

    def publish_joint_states(self):
        # Create joint state message
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        self.joint_state_publisher.publish(msg)

        # Update joint positions (for demonstration)
        self.joint_positions[0] += 0.1
        self.joint_positions[1] += 0.1

def main(args=None):
    rclpy.init(args=args)
    node = RobotStatePublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Twist Controller for Differential Drive

Let's create a controller that accepts velocity commands:

```python
# simple_robot_pkg/twist_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

class TwistController(Node):
    def __init__(self):
        super().__init__('twist_controller')

        # Subscribe to cmd_vel
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)

        # Publish wheel velocities
        self.wheel_cmd_publisher = self.create_publisher(
            Float64MultiArray,
            'wheel_cmd',
            10)

        self.wheel_radius = 0.1  # meters
        self.wheel_separation = 0.3  # meters
        self.get_logger().info('Twist Controller initialized')

    def cmd_vel_callback(self, msg):
        # Convert linear and angular velocity to wheel velocities
        linear_vel = msg.linear.x
        angular_vel = msg.angular.z

        # Calculate wheel velocities for differential drive
        left_wheel_vel = (linear_vel - angular_vel * self.wheel_separation / 2.0) / self.wheel_radius
        right_wheel_vel = (linear_vel + angular_vel * self.wheel_separation / 2.0) / self.wheel_radius

        # Create and publish wheel command
        wheel_cmd = Float64MultiArray()
        wheel_cmd.data = [left_wheel_vel, right_wheel_vel]

        self.wheel_cmd_publisher.publish(wheel_cmd)
        self.get_logger().info(f'Left: {left_wheel_vel:.2f}, Right: {right_wheel_vel:.2f}')

def main(args=None):
    rclpy.init(args=args)
    controller = TwistController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch File for Complete Robot System

Now let's create a comprehensive launch file that brings up our complete robot system:

```python
# simple_robot_pkg/launch/simple_bot.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the launch directory
    pkg_dir = get_package_share_directory('simple_robot_pkg')
    urdf_path = os.path.join(pkg_dir, 'urdf', 'simple_bot.urdf')

    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),

        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': open(urdf_path).read()
            }]),

        # Joint State Publisher (for simulation)
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[{
                'use_sim_time': use_sim_time
            }]),

        # Twist Controller
        Node(
            package='simple_robot_pkg',
            executable='twist_controller',
            name='twist_controller',
            parameters=[{
                'use_sim_time': use_sim_time
            }]),

        # Robot State Publisher Node
        Node(
            package='simple_robot_pkg',
            executable='robot_state_publisher_node',
            name='robot_state_publisher_node',
            parameters=[{
                'use_sim_time': use_sim_time
            }]),
    ])
```

## Creating a Simple Navigation Node

Let's also create a basic navigation node that can move the robot to a goal:

```python
# simple_robot_pkg/simple_navigation.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import math

class SimpleNavigation(Node):
    def __init__(self):
        super().__init__('simple_navigation')

        # Publishers and subscribers
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_subscriber = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)

        # Navigation parameters
        self.current_pose = Point()
        self.target_pose = Point()
        self.target_pose.x = 2.0
        self.target_pose.y = 2.0

        # Timer for navigation
        self.timer = self.create_timer(0.1, self.navigate_to_goal)

        self.get_logger().info(f'Navigating to goal: ({self.target_pose.x}, {self.target_pose.y})')

    def odom_callback(self, msg):
        # Update current pose from odometry
        self.current_pose.x = msg.pose.pose.position.x
        self.current_pose.y = msg.pose.pose.position.y

    def navigate_to_goal(self):
        # Calculate distance to goal
        dx = self.target_pose.x - self.current_pose.x
        dy = self.target_pose.y - self.current_pose.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Create twist message
        twist = Twist()

        if distance > 0.1:  # If not close to goal
            # Calculate angle to goal
            angle_to_goal = math.atan2(dy, dx)

            # Get current orientation (simplified)
            # In a real system, you'd get this from the orientation in the odom message
            current_angle = 0.0  # Simplified for example

            # Simple proportional controller for rotation
            angle_error = angle_to_goal - current_angle
            twist.angular.z = max(-1.0, min(1.0, angle_error * 1.0))

            # Move forward if roughly aligned
            if abs(angle_error) < 0.2:
                twist.linear.x = max(0.0, min(0.5, distance * 0.5))
        else:
            # Stop when close to goal
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info('Reached goal!')

        self.cmd_vel_publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    navigator = SimpleNavigation()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Package Configuration

Don't forget to update your package.xml to include dependencies:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>simple_robot_pkg</name>
  <version>0.1.0</version>
  <description>Simple robot package for educational purposes</description>
  <maintainer email="student@robotics.edu">Student</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>nav_msgs</end>
  <depend>tf2_ros</depend>
  <depend>example_interfaces</depend>

  <exec_depend>ros2launch</exec_depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

And update your setup.py:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'simple_robot_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Include URDF files
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='student@robotics.edu',
    description='Simple robot package for educational purposes',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_publisher = simple_robot_pkg.simple_publisher:main',
            'simple_subscriber = simple_robot_pkg.simple_subscriber:main',
            'robot_state_publisher_node = simple_robot_pkg.robot_state_publisher_node:main',
            'twist_controller = simple_robot_pkg.twist_controller:main',
            'simple_navigation = simple_robot_pkg.simple_navigation:main',
        ],
    },
)
```

## Building and Running

To build and run your complete robot system:

```bash
cd ~/ros2_ws
colcon build --packages-select simple_robot_pkg
source install/setup.bash

# Run the complete robot system
ros2 launch simple_robot_pkg simple_bot.launch.py
```

## Next Steps

Now that we have a comprehensive ROS 2 robot package, we'll move on to Module 2 where we'll integrate this with simulation environments like Gazebo and Unity. The URDF we created here will be used to represent our robot in the digital twin.