---
sidebar_position: 4
---

# Chapter 4: URDF and Launch Files for Robot Integration

## Understanding URDF (Unified Robot Description Format)

URDF is XML-based format used to describe robots in ROS. It contains information about robot's physical properties like joints, links, inertial properties, visual and collision properties. URDF is essential for simulation, visualization, and kinematic analysis.

### URDF Structure

A URDF file typically contains:
- **Links**: Rigid bodies with visual and collision properties
- **Joints**: Connections between links with kinematic properties
- **Materials**: Color and appearance definitions
- **Gazebo plugins**: Simulation-specific configurations

Let's expand our simple robot URDF to include more realistic properties:

```xml
<!-- simple_robot_pkg/urdf/simple_bot_complete.urdf -->
<?xml version="1.0"?>
<robot name="simple_bot_complete" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Materials -->
  <material name="blue">
    <color rgba="0 0 1 0.8"/>
  </material>
  <material name="black">
    <color rgba="0 0 0 1"/>
  </material>
  <material name="red">
    <color rgba="1 0 0 0.8"/>
  </material>

  <!-- Base Footprint (virtual link for ground reference) -->
  <link name="base_footprint">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.001 0.001 0.001"/>
      </geometry>
    </visual>
  </link>

  <joint name="base_footprint_joint" type="fixed">
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <parent link="base_footprint"/>
    <child link="base_link"/>
  </joint>

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.04375" ixy="0.0" ixz="0.0" iyy="0.077083" iyz="0.0" izz="0.10625"/>
    </inertial>
  </link>

  <!-- Left Wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black"/>
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
      <material name="black"/>
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

  <!-- Left Caster Wheel -->
  <link name="left_caster_wheel">
    <visual>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.00005" ixy="0.0" ixz="0.0" iyy="0.00005" iyz="0.0" izz="0.00005"/>
    </inertial>
  </link>

  <!-- Right Caster Wheel -->
  <link name="right_caster_wheel">
    <visual>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.00005" ixy="0.0" ixz="0.0" iyy="0.00005" iyz="0.0" izz="0.00005"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0.15 0.15 -0.075" rpy="1.57075 0 0"/>
    <axis xyz="0 0 1"/>
    <dynamics damping="0.5"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0.15 -0.15 -0.075" rpy="1.57075 0 0"/>
    <axis xyz="0 0 1"/>
    <dynamics damping="0.5"/>
  </joint>

  <joint name="left_caster_joint" type="fixed">
    <parent link="base_link"/>
    <child link="left_caster_wheel"/>
    <origin xyz="-0.2 -0.1 -0.075" rpy="0 0 0"/>
  </joint>

  <joint name="right_caster_joint" type="fixed">
    <parent link="base_link"/>
    <child link="right_caster_wheel"/>
    <origin xyz="-0.2 0.1 -0.075" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo-specific configurations -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
  </gazebo>

  <gazebo reference="left_wheel">
    <material>Gazebo/Black</material>
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>

  <gazebo reference="right_wheel">
    <material>Gazebo/Black</material>
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>

  <gazebo reference="left_caster_wheel">
    <material>Gazebo/Red</material>
    <mu1>0.1</mu1>
    <mu2>0.1</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>

  <gazebo reference="right_caster_wheel">
    <material>Gazebo/Red</material>
    <mu1>0.1</mu1>
    <mu2>0.1</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>

  <!-- Transmission for differential drive -->
  <transmission name="left_wheel_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_wheel_joint">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_wheel_motor">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_wheel_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_wheel_joint">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_wheel_motor">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

</robot>
```

## Advanced Launch File with Gazebo Integration

Now let's create a comprehensive launch file that integrates our robot with Gazebo simulation:

```python
# simple_robot_pkg/launch/simple_bot_gazebo.launch.py
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    world = LaunchConfiguration('world', default='empty.sdf')

    # Package names
    pkg_gazebo_ros = FindPackageShare('gazebo_ros').find('gazebo_ros')
    pkg_simple_robot = FindPackageShare('simple_robot_pkg').find('simple_robot_pkg')

    # Paths
    urdf_path = os.path.join(pkg_simple_robot, 'urdf', 'simple_bot_complete.urdf')
    rviz_config_path = os.path.join(pkg_simple_robot, 'rviz', 'simple_bot.rviz')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'),

        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Whether to start RViz'),

        DeclareLaunchArgument(
            'world',
            default_value='empty.sdf',
            description='Choose one of the world files from `/gazebo_ros/worlds`'),

        # Start Gazebo server and client
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world}.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
            ),
        ),

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

        # Spawn robot in Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'simple_bot',
                '-file', urdf_path,
                '-x', '0', '-y', '0', '-z', '0.2'
            ],
            output='screen'),

        # Joint State Publisher (GUI for manual joint control)
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            parameters=[{'use_sim_time': use_sim_time}]),

        # RViz
        Node(
            condition=IfCondition(use_rviz),
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_path],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'),

        # Twist mux for velocity commands
        Node(
            package='twist_mux',
            executable='twist_mux',
            parameters=[os.path.join(pkg_simple_robot, 'config', 'twist_mux.yaml')],
            remappings=[('/cmd_vel_out', '/simple_bot/cmd_vel')],
            output='screen'),

        # Robot localization (odometry)
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[os.path.join(pkg_simple_robot, 'config', 'ekf.yaml')]),
    ])
```

## RViz Configuration

Create an RViz configuration file to visualize our robot:

```yaml
# simple_robot_pkg/rviz/simple_bot.rviz
Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /RobotModel1
        - /TF1
        - /LaserScan1
        - /PointCloud21
      Splitter Ratio: 0.5
    Tree Height: 549
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Tool Properties
    Expanded:
      - /2D Goal Pose1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.5886790156364441
  - Class: rviz_common/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Alpha: 1
      Class: rviz_default_plugins/RobotModel
      Collision Enabled: false
      Description File: ""
      Description Source: Topic
      Description Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /robot_description
      Enabled: true
      Links:
        All Links Enabled: true
        Expand Joint Details: false
        Expand Link Details: false
        Expand Tree: false
        Link Tree Style: Links in Alphabetic Order
        base_footprint:
          Alpha: 1
          Show Axes: false
          Show Trail: false
        base_link:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        left_caster_wheel:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        left_wheel:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        right_caster_wheel:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        right_wheel:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
      Name: RobotModel
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
    - Class: rviz_default_plugins/TF
      Enabled: true
      Frame Timeout: 15
      Frames:
        All Enabled: true
      Marker Scale: 1
      Name: TF
      Show Arrows: true
      Show Axes: true
      Show Names: false
      Tree:
        {}
      Update Interval: 0
      Value: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: base_link
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
      Line color: 128; 128; 0
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 10
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 0.5
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz)
      Yaw: 0.5
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 846
  Hide Left Dock: false
  Hide Right Dock: false
  QMainWindow State: 000000ff00000000fd000000040000000000000156000002b4fc0200000008fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000005c00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000003d000002b4000000c900fffffffb0000002000730065006c0065006300740069006f006e0020006200750066006600650072005f0069006d006100670065000000003d000000c90000000000000000fb0000000a0049006d00610067006501000001d40000010c0000000000000000fb0000000a0056006900650077007301000003e0000000a4000000a400fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e10000019700000003000004420000003efc0100000002fb0000000800540069006d00650100000000000004420000000000000000fb0000000800540069006d006501000000000000045000000000000000000000023f000002b400000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Width: 1200
  X: 72
  Y: 60
```

## Twist Multiplexer Configuration

Create a configuration file for the twist multiplexer:

```yaml
# simple_robot_pkg/config/twist_mux.yaml
twist_mux:
  ros__parameters:
    topics:
      - name: navigation
        topic: cmd_vel
        timeout: 0.1
        priority: 10
      - name: joystick
        topic: cmd_vel_joy
        timeout: 0.5
        priority: 5
      - name: keyboard
        topic: cmd_vel_key
        timeout: 0.5
        priority: 3
```

## Extended Robot Controller with Gazebo Integration

Let's create a more sophisticated controller that works with Gazebo:

```python
# simple_robot_pkg/gazebo_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Wrench
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty
import math

class GazeboController(Node):
    def __init__(self):
        super().__init__('gazebo_controller')

        # Publishers for Gazebo
        self.left_wheel_pub = self.create_publisher(Float64MultiArray, '/simple_bot/left_wheel_controller/commands', 10)
        self.right_wheel_pub = self.create_publisher(Float64MultiArray, '/simple_bot/right_wheel_controller/commands', 10)

        # Subscriber for velocity commands
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)

        # Timer for control loop
        self.control_timer = self.create_timer(0.02, self.control_loop)  # 50 Hz

        # Robot parameters
        self.wheel_radius = 0.1  # meters
        self.wheel_separation = 0.3  # meters
        self.max_wheel_velocity = 5.0  # rad/s

        # Control variables
        self.desired_linear_vel = 0.0
        self.desired_angular_vel = 0.0

        self.get_logger().info('Gazebo Controller initialized')

    def cmd_vel_callback(self, msg):
        self.desired_linear_vel = msg.linear.x
        self.desired_angular_vel = msg.angular.z

    def control_loop(self):
        # Convert linear and angular velocity to wheel velocities
        linear_vel = self.desired_linear_vel
        angular_vel = self.desired_angular_vel

        # Calculate wheel velocities for differential drive
        left_wheel_vel = (linear_vel - angular_vel * self.wheel_separation / 2.0) / self.wheel_radius
        right_wheel_vel = (linear_vel + angular_vel * self.wheel_separation / 2.0) / self.wheel_radius

        # Limit velocities
        left_wheel_vel = max(-self.max_wheel_velocity, min(self.max_wheel_velocity, left_wheel_vel))
        right_wheel_vel = max(-self.max_wheel_velocity, min(self.max_wheel_velocity, right_wheel_vel))

        # Create and publish wheel commands
        left_cmd = Float64MultiArray()
        left_cmd.data = [left_wheel_vel]
        self.left_wheel_pub.publish(left_cmd)

        right_cmd = Float64MultiArray()
        right_cmd.data = [right_wheel_vel]
        self.right_wheel_pub.publish(right_cmd)

def main(args=None):
    rclpy.init(args=args)
    controller = GazeboController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch File for Complete System

Finally, let's create a complete launch file that includes all components:

```python
# simple_robot_pkg/launch/simple_bot_complete.launch.py
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    world = LaunchConfiguration('world', default='empty.sdf')

    # Package names
    pkg_gazebo_ros = FindPackageShare('gazebo_ros').find('gazebo_ros')
    pkg_simple_robot = FindPackageShare('simple_robot_pkg').find('simple_robot_pkg')
    pkg_ros_gz_sim = FindPackageShare('ros_gz_sim').find('ros_gz_sim')

    # Paths
    urdf_path = os.path.join(pkg_simple_robot, 'urdf', 'simple_bot_complete.urdf')
    rviz_config_path = os.path.join(pkg_simple_robot, 'rviz', 'simple_bot.rviz')

    # Launch description
    ld = LaunchDescription()

    # Declare launch arguments
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'))

    ld.add_action(DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to start RViz'))

    ld.add_action(DeclareLaunchArgument(
        'world',
        default_value='empty.sdf',
        description='Choose one of the world files from `/gazebo_ros/worlds`'))

    # Start Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items(),
    )

    gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        ),
    )

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': open(urdf_path).read()
        }])

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'simple_bot',
            '-file', urdf_path,
            '-x', '0', '-y', '0', '-z', '0.2'
        ],
        output='screen')

    # Joint State Publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}])

    # Gazebo Controller
    gazebo_controller = Node(
        package='simple_robot_pkg',
        executable='gazebo_controller',
        name='gazebo_controller',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    # RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    # Add all actions to the launch description
    ld.add_action(gazebo)
    ld.add_action(gazebo_client)
    ld.add_action(robot_state_publisher)
    ld.add_action(joint_state_publisher)
    ld.add_action(spawn_entity)
    ld.add_action(gazebo_controller)

    # Conditionally add RViz
    ld.add_action(rviz)

    return ld
```

## Running the Complete System

To run the complete system with Gazebo:

```bash
cd ~/ros2_ws
colcon build --packages-select simple_robot_pkg
source install/setup.bash

# Run the complete simulation
ros2 launch simple_robot_pkg simple_bot_complete.launch.py
```

In another terminal, you can send velocity commands:

```bash
# Send velocity commands
ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 0.5}, angular: {z: 0.2}}'
```

## Next Steps

This chapter has covered the integration of ROS 2 with simulation environments. In the next module, we'll dive deeper into the Digital Twin concept, exploring how to create more sophisticated simulations with Gazebo and Unity, and how to implement physics-based modeling and visualization of robotic systems.