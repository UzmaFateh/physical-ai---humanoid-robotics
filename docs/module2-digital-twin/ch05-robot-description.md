---
sidebar_position: 1
---

# Chapter 5: The Digital Twin - Gazebo & Unity Integration

## Introduction to Digital Twins in Robotics

A digital twin is a virtual replica of a physical system that can be used for simulation, testing, and analysis. In robotics, digital twins enable us to test algorithms, validate control systems, and train AI models in a safe, controlled environment before deploying them on real hardware.

For robotics applications, digital twins serve multiple purposes:
- **Testing and Validation**: Verify robot behaviors without risk to physical hardware
- **Algorithm Development**: Develop and refine control algorithms in simulation
- **Training**: Train machine learning models with synthetic data
- **System Design**: Evaluate different robot configurations and capabilities
- **Safety Analysis**: Test edge cases and failure scenarios safely

## Gazebo: The Robot Simulation Engine

Gazebo is a 3D simulation environment that provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. It's widely used in the robotics community for testing and validating robot algorithms.

### Installing Gazebo Garden

For this course, we'll use Gazebo Garden (the latest version at the time of writing):

```bash
# Add Gazebo repository
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/gazebo-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gazebo-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Install Gazebo Garden
sudo apt update
sudo apt install gz-garden
```

### Understanding Gazebo Components

Gazebo consists of several key components:
- **Gazebo GUI**: Visual interface for interacting with the simulation
- **Gazebo Server**: Physics simulation engine
- **SDF (Simulation Description Format)**: XML-based format for describing simulation worlds
- **Plugins**: Extend Gazebo functionality through custom code
- **ROS 2 Integration**: Bridge between Gazebo and ROS 2

### Creating Custom Worlds

Let's create a custom world file for our robot to operate in:

```xml
<!-- simple_robot_pkg/worlds/simple_world.sdf -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_world">
    <!-- Include a ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include a sky -->
    <include>
      <uri>model://sky</uri>
    </include>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.6 0.4 -0.8</direction>
    </light>

    <!-- Simple maze environment -->
    <model name="wall_1">
      <pose>-2 0 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 4 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 4 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="wall_2">
      <pose>2 0 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 4 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 4 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="wall_3">
      <pose>0 -2 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>4 0.1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>4 0.1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="wall_4">
      <pose>0 2 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>4 0.1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>4 0.1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Add some obstacles -->
    <model name="obstacle_1">
      <pose>-1 -1 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>1</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>1</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.8 0.3 0.3 1</ambient>
            <diffuse>1 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="obstacle_2">
      <pose>1 1 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.4 0.4 0.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.4 0.4 0.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.8 1</ambient>
            <diffuse>0.5 0.5 1 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
```

### Advanced Gazebo Plugins

Let's create a custom Gazebo plugin for our robot that provides sensor simulation:

```cpp
// simple_robot_pkg/gazebo_plugins/laser_scanner_plugin.cpp
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/sensors/sensors.hh>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <tf/transform_broadcaster.h>

namespace gazebo
{
  class LaserScannerPlugin : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      this->model = _model;
      this->world = _model->GetWorld();

      // Initialize ROS
      if (!ros::isInitialized())
      {
        int argc = 0;
        char **argv = NULL;
        ros::init(argc, argv, "gazebo_laser_scanner", ros::init_options::NoSigintHandler);
      }

      this->rosNode.reset(new ros::NodeHandle("gazebo_laser_scanner"));

      // Create publisher for laser scan
      this->pub = this->rosNode->advertise<sensor_msgs::LaserScan>("/scan", 1);

      // Listen to the update event
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&LaserScannerPlugin::OnUpdate, this));

      gzdbg << "LaserScannerPlugin loaded\n";
    }

    public: void OnUpdate()
    {
      // Create a simulated laser scan message
      sensor_msgs::LaserScan scan_msg;
      scan_msg.header.stamp = ros::Time::now();
      scan_msg.header.frame_id = "laser_frame";

      // Set laser scan parameters
      scan_msg.angle_min = -M_PI / 2.0;  // -90 degrees
      scan_msg.angle_max = M_PI / 2.0;   // 90 degrees
      scan_msg.angle_increment = M_PI / 180.0;  // 1 degree
      scan_msg.time_increment = 0.0;
      scan_msg.scan_time = 0.1;
      scan_msg.range_min = 0.1;
      scan_msg.range_max = 10.0;

      // Simulate some ranges (in a real implementation, this would come from physics simulation)
      int num_ranges = (scan_msg.angle_max - scan_msg.angle_min) / scan_msg.angle_increment + 1;
      scan_msg.ranges.resize(num_ranges);

      for (int i = 0; i < num_ranges; ++i)
      {
        // Simulate a wall at 2 meters in front
        scan_msg.ranges[i] = 2.0 + 0.1 * sin(i * 0.1);  // Add some noise
      }

      // Publish the laser scan
      this->pub.publish(scan_msg);
    }

    private: physics::ModelPtr model;
    private: physics::WorldPtr world;
    private: ros::NodeHandlePtr rosNode;
    private: ros::Publisher pub;
    private: event::ConnectionPtr updateConnection;
  };

  GZ_REGISTER_MODEL_PLUGIN(LaserScannerPlugin)
}
```

For ROS 2, here's the equivalent plugin using the ROS 2 Gazebo bridge:

```cpp
// simple_robot_pkg/gazebo_plugins/laser_scanner_plugin_ros2.cpp
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/sensors/sensors.hh>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_ros/transform_broadcaster.h>

namespace gazebo
{
  class LaserScannerPluginROS2 : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      this->model = _model;
      this->world = _model->GetWorld();

      // Initialize ROS 2
      if (!rclcpp::ok())
      {
        int argc = 0;
        char **argv = NULL;
        rclcpp::init(argc, argv);
      }

      this->node = std::make_shared<rclcpp::Node>("gazebo_laser_scanner");

      // Create publisher for laser scan
      this->pub = this->node->create_publisher<sensor_msgs::msg::LaserScan>("/scan", 10);

      // Listen to the update event
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&LaserScannerPluginROS2::OnUpdate, this));

      gzdbg << "LaserScannerPluginROS2 loaded\n";
    }

    public: void OnUpdate()
    {
      // Create a simulated laser scan message
      auto scan_msg = std::make_shared<sensor_msgs::msg::LaserScan>();
      scan_msg->header.stamp = this->node->get_clock()->now();
      scan_msg->header.frame_id = "laser_frame";

      // Set laser scan parameters
      scan_msg->angle_min = -M_PI / 2.0;  // -90 degrees
      scan_msg->angle_max = M_PI / 2.0;   // 90 degrees
      scan_msg->angle_increment = M_PI / 180.0;  // 1 degree
      scan_msg->time_increment = 0.0;
      scan_msg->scan_time = 0.1;
      scan_msg->range_min = 0.1;
      scan_msg->range_max = 10.0;

      // Simulate some ranges
      int num_ranges = (scan_msg->angle_max - scan_msg->angle_min) / scan_msg->angle_increment + 1;
      scan_msg->ranges.resize(num_ranges);

      for (int i = 0; i < num_ranges; ++i)
      {
        // Simulate a wall at 2 meters in front
        scan_msg->ranges[i] = 2.0 + 0.1 * sin(i * 0.1);  // Add some noise
      }

      // Publish the laser scan
      this->pub->publish(*scan_msg);
    }

    private: physics::ModelPtr model;
    private: physics::WorldPtr world;
    private: rclcpp::Node::SharedPtr node;
    private: rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pub;
    private: event::ConnectionPtr updateConnection;
  };

  GZ_REGISTER_MODEL_PLUGIN(LaserScannerPluginROS2)
}
```

### Gazebo-ROS 2 Bridge Configuration

Let's create a launch file that properly integrates Gazebo with ROS 2:

```python
# simple_robot_pkg/launch/gazebo_integration.launch.py
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
    world = LaunchConfiguration('world', default='simple_world.sdf')
    robot_name = LaunchConfiguration('robot_name', default='simple_bot')

    # Package names
    pkg_gazebo_ros = FindPackageShare('gazebo_ros').find('gazebo_ros')
    pkg_simple_robot = FindPackageShare('simple_robot_pkg').find('simple_robot_pkg')

    # Paths
    urdf_path = os.path.join(pkg_simple_robot, 'urdf', 'simple_bot_complete.urdf')
    world_path = os.path.join(pkg_simple_robot, 'worlds', 'simple_world.sdf')
    rviz_config_path = os.path.join(pkg_simple_robot, 'rviz', 'simple_bot.rviz')

    # Create launch description
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
        default_value='simple_world.sdf',
        description='Choose one of the world files from `/simple_robot_pkg/worlds`'))

    ld.add_action(DeclareLaunchArgument(
        'robot_name',
        default_value='simple_bot',
        description='Name of the robot to spawn'))

    # Start Gazebo server
    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={
            'world': world_path,
            'verbose': 'true'
        }.items(),
    )

    # Start Gazebo client
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
            '-entity', [robot_name],
            '-file', urdf_path,
            '-x', '0', '-y', '0', '-z', '0.2',
            '-robot_namespace', [robot_name]
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

    # Laser scanner node (if using custom plugin)
    laser_node = Node(
        package='simple_robot_pkg',
        executable='laser_scanner_node',
        name='laser_scanner_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    # Navigation stack (basic)
    navigation_node = Node(
        package='simple_robot_pkg',
        executable='simple_navigation',
        name='simple_navigation',
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
    ld.add_action(gazebo_server)
    ld.add_action(gazebo_client)
    ld.add_action(robot_state_publisher)
    ld.add_action(joint_state_publisher)
    ld.add_action(spawn_entity)
    ld.add_action(gazebo_controller)
    ld.add_action(laser_node)
    ld.add_action(navigation_node)

    # Conditionally add RViz
    ld.add_action(rviz)

    return ld
```

## Unity Integration for Advanced Visualization

Unity provides a powerful platform for creating high-fidelity visualizations and user interfaces for robotics applications. While Unity doesn't directly interface with ROS 2 like Gazebo does, we can create bridges using various approaches.

### Setting up Unity with ROS 2

Unity can connect to ROS 2 through several methods:
1. **ROS# (ROS Sharp)**: A Unity package that provides ROS connectivity
2. **WebSocket bridges**: Connect Unity to ROS 2 via web protocols
3. **Custom TCP/IP bridges**: Direct communication protocols

Let's create a basic Unity C# script for receiving robot data:

```csharp
// Assets/Scripts/RosJointController.cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes;

public class RosJointController : MonoBehaviour
{
    [SerializeField]
    private string topicName = "/joint_states";

    private ROSConnection ros;
    private Dictionary<string, float> jointPositions = new Dictionary<string, float>();

    // Robot parts to control
    public Transform leftWheel;
    public Transform rightWheel;

    // Start is called before the first frame update
    void Start()
    {
        // Get the ROS connection static instance
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<EmptyMsg>(topicName);

        // Subscribe to joint states
        ros.Subscribe<JointStateMsg>(topicName, JointStateCallback);
    }

    // Callback function to receive joint state messages
    void JointStateCallback(JointStateMsg jointState)
    {
        // Update joint positions dictionary
        for (int i = 0; i < jointState.name.Count; i++)
        {
            if (i < jointState.position.Count)
            {
                jointPositions[jointState.name[i]] = (float)jointState.position[i];
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        // Update wheel positions based on joint data
        if (jointPositions.ContainsKey("left_wheel_joint"))
        {
            leftWheel.localRotation = Quaternion.Euler(0, 0, jointPositions["left_wheel_joint"] * Mathf.Rad2Deg);
        }

        if (jointPositions.ContainsKey("right_wheel_joint"))
        {
            rightWheel.localRotation = Quaternion.Euler(0, 0, jointPositions["right_wheel_joint"] * Mathf.Rad2Deg);
        }
    }

    // Publish velocity commands
    public void PublishVelocityCommand(double linearX, double angularZ)
    {
        var twistMsg = new TwistMsg();
        twistMsg.linear = new Vector3Msg(linearX, 0, 0);
        twistMsg.angular = new Vector3Msg(0, 0, angularZ);

        ros.Publish("/cmd_vel", twistMsg);
    }
}
```

### Unity Scene Setup for Robot Visualization

Here's a Unity scene setup script that creates a visualization of our robot:

```csharp
// Assets/Scripts/RobotVisualization.cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RobotVisualization : MonoBehaviour
{
    [Header("Robot Configuration")]
    public float wheelRadius = 0.1f;
    public float wheelSeparation = 0.3f;

    [Header("Visualization")]
    public Transform baseLink;
    public Transform leftWheel;
    public Transform rightWheel;
    public Transform laserScanner;

    [Header("Simulation")]
    public float simulationSpeed = 1.0f;

    // Internal state
    private float leftWheelAngle = 0f;
    private float rightWheelAngle = 0f;
    private Vector3 robotPosition = Vector3.zero;
    private float robotRotation = 0f;

    // Velocity commands
    private float linearVelocity = 0f;
    private float angularVelocity = 0f;

    void Start()
    {
        // Initialize robot position
        robotPosition = baseLink.position;
        robotRotation = baseLink.eulerAngles.y;
    }

    void Update()
    {
        // Update wheel rotations based on velocity
        UpdateWheelRotations();

        // Update robot position based on velocities
        UpdateRobotPosition();

        // Apply transformations
        ApplyTransformations();
    }

    private void UpdateWheelRotations()
    {
        // Update wheel angles based on velocities
        leftWheelAngle += linearVelocity * Time.deltaTime / wheelRadius;
        rightWheelAngle += linearVelocity * Time.deltaTime / wheelRadius;

        // Add differential rotation for angular movement
        float angularOffset = angularVelocity * wheelSeparation * Time.deltaTime / 2.0f;
        leftWheelAngle -= angularOffset / wheelRadius;
        rightWheelAngle += angularOffset / wheelRadius;
    }

    private void UpdateRobotPosition()
    {
        // Calculate forward movement
        float forwardMove = linearVelocity * Time.deltaTime;

        // Update position based on current rotation
        robotPosition.x += forwardMove * Mathf.Sin(robotRotation * Mathf.Deg2Rad);
        robotPosition.z += forwardMove * Mathf.Cos(robotRotation * Mathf.Deg2Rad);

        // Update rotation
        robotRotation += angularVelocity * Time.deltaTime * Mathf.Rad2Deg;
    }

    private void ApplyTransformations()
    {
        // Apply position and rotation to base link
        baseLink.position = robotPosition;
        baseLink.eulerAngles = new Vector3(0, robotRotation, 0);

        // Apply wheel rotations
        leftWheel.localEulerAngles = new Vector3(90, leftWheelAngle * Mathf.Rad2Deg, 0);
        rightWheel.localEulerAngles = new Vector3(90, rightWheelAngle * Mathf.Rad2Deg, 0);

        // Apply laser scanner rotation (if it rotates with the robot)
        if (laserScanner != null)
        {
            laserScanner.eulerAngles = new Vector3(0, robotRotation, 0);
        }
    }

    // Method to set velocity commands from external sources
    public void SetVelocityCommand(float linear, float angular)
    {
        linearVelocity = linear;
        angularVelocity = angular;
    }

    // Method to get current robot state
    public Vector3 GetRobotPosition()
    {
        return robotPosition;
    }

    public float GetRobotRotation()
    {
        return robotRotation;
    }

    // Method to reset robot to initial position
    public void ResetRobot()
    {
        robotPosition = Vector3.zero;
        robotRotation = 0f;
        linearVelocity = 0f;
        angularVelocity = 0f;
    }
}
```

### Creating a TCP Bridge for Unity-ROS 2 Communication

Let's create a Python-based TCP bridge that can relay messages between ROS 2 and Unity:

```python
# simple_robot_pkg/unity_bridge.py
import socket
import json
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import struct

class UnityBridge(Node):
    def __init__(self):
        super().__init__('unity_bridge')

        # ROS 2 publishers and subscribers
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.laser_scan_sub = self.create_subscription(LaserScan, '/scan', self.laser_scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # TCP server setup
        self.host = 'localhost'
        self.port = 12345
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.get_logger().info(f'Unity Bridge listening on {self.host}:{self.port}')
        except Exception as e:
            self.get_logger().error(f'Failed to bind to {self.host}:{self.port} - {e}')
            return

        # Data to send to Unity
        self.joint_states_data = {}
        self.laser_scan_data = []
        self.odom_data = {'position': [0, 0, 0], 'orientation': [0, 0, 0, 1]}

        # Start TCP server thread
        self.tcp_thread = threading.Thread(target=self.tcp_server_loop)
        self.tcp_thread.daemon = True
        self.tcp_thread.start()

        # Timer for sending data to Unity
        self.send_timer = self.create_timer(0.05, self.send_data_to_unity)  # 20 Hz

        self.client_socket = None
        self.client_address = None

    def joint_state_callback(self, msg):
        self.joint_states_data = {
            'names': list(msg.name),
            'positions': [float(p) for p in msg.position],
            'velocities': [float(v) for v in msg.velocity],
            'efforts': [float(e) for e in msg.effort]
        }

    def laser_scan_callback(self, msg):
        self.laser_scan_data = {
            'ranges': [float(r) for r in msg.ranges if not (r != r or r > msg.range_max or r < msg.range_min)],
            'angle_min': float(msg.angle_min),
            'angle_max': float(msg.angle_max),
            'angle_increment': float(msg.angle_increment)
        }

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        self.odom_data = {
            'position': [pos.x, pos.y, pos.z],
            'orientation': [orient.x, orient.y, orient.z, orient.w]
        }

    def tcp_server_loop(self):
        while rclpy.ok():
            try:
                self.get_logger().info('Waiting for Unity connection...')
                self.client_socket, self.client_address = self.socket.accept()
                self.get_logger().info(f'Unity connected from {self.client_address}')

                while rclpy.ok():
                    try:
                        # Receive command from Unity
                        data = self.client_socket.recv(1024)
                        if data:
                            try:
                                cmd = json.loads(data.decode('utf-8'))
                                self.process_unity_command(cmd)
                            except json.JSONDecodeError:
                                self.get_logger().warn('Invalid JSON received from Unity')
                    except ConnectionResetError:
                        self.get_logger().info('Unity client disconnected')
                        break
                    except Exception as e:
                        self.get_logger().error(f'Error receiving data from Unity: {e}')
                        break

            except Exception as e:
                self.get_logger().error(f'Error accepting connection: {e}')

        self.socket.close()

    def process_unity_command(self, cmd):
        if cmd.get('type') == 'cmd_vel':
            twist = Twist()
            twist.linear.x = cmd.get('linear_x', 0.0)
            twist.linear.y = cmd.get('linear_y', 0.0)
            twist.linear.z = cmd.get('linear_z', 0.0)
            twist.angular.x = cmd.get('angular_x', 0.0)
            twist.angular.y = cmd.get('angular_y', 0.0)
            twist.angular.z = cmd.get('angular_z', 0.0)

            self.cmd_vel_pub.publish(twist)
        elif cmd.get('type') == 'reset':
            self.get_logger().info('Reset command received from Unity')

    def send_data_to_unity(self):
        if self.client_socket:
            try:
                # Prepare data to send to Unity
                data_to_send = {
                    'joint_states': self.joint_states_data,
                    'laser_scan': self.laser_scan_data,
                    'odom': self.odom_data
                }

                json_data = json.dumps(data_to_send)
                self.client_socket.send(json_data.encode('utf-8'))
            except Exception as e:
                self.get_logger().error(f'Error sending data to Unity: {e}')
                # Close the socket if there's an error
                try:
                    self.client_socket.close()
                except:
                    pass
                self.client_socket = None

def main(args=None):
    rclpy.init(args=args)
    bridge = UnityBridge()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.socket.close()
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Running the Complete Digital Twin System

To run the complete digital twin system with both Gazebo and Unity visualization:

```bash
# Terminal 1: Start the ROS 2 bridge
cd ~/ros2_ws
source install/setup.bash
ros2 run simple_robot_pkg unity_bridge

# Terminal 2: Start Gazebo simulation
ros2 launch simple_robot_pkg gazebo_integration.launch.py

# Terminal 3: In Unity editor, run the scene with the RosJointController and RobotVisualization scripts
# Or build and run the Unity application
```

## Best Practices for Digital Twin Development

1. **Synchronization**: Ensure simulation time matches real-time when needed
2. **Physics Accuracy**: Use appropriate physics parameters that match real-world behavior
3. **Sensor Simulation**: Model sensor noise and limitations realistically
4. **Performance**: Optimize simulation for real-time performance
5. **Validation**: Regularly compare simulation results with real-world data
6. **Scalability**: Design systems that can handle multiple robots and complex environments

## Next Steps

In the next chapter, we'll explore NVIDIA Isaac, which provides advanced simulation and AI capabilities for robotics applications. We'll see how to integrate our digital twin with Isaac's powerful simulation and perception tools.