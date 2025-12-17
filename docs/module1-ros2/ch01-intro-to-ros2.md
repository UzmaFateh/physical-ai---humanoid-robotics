---
sidebar_position: 1
title: "Introduction to ROS 2: The Robotic Nervous System"
---

# Introduction to ROS 2: The Robotic Nervous System

## Overview

ROS 2 (Robot Operating System 2) serves as the nervous system for robotic applications, providing a framework for communication, coordination, and control between different components of a robot. In this module, we'll explore how ROS 2 enables complex robotic behaviors through distributed computing, message passing, and standardized interfaces.

## What is ROS 2?

Robot Operating System 2 (ROS 2) is the next generation of the Robot Operating System, designed to address the limitations of ROS 1 and provide a more robust, scalable, and production-ready framework for robotics development. Unlike ROS 1, which was built on a single master architecture, ROS 2 uses DDS (Data Distribution Service) as its communication layer, making it more suitable for distributed systems, real-time applications, and industrial deployments.

ROS 2 is built on a modern, modular architecture that addresses key challenges faced in robotics development:

- **Distributed Systems**: No single point of failure with peer-to-peer communication
- **Real-time Capabilities**: Support for real-time applications with deterministic behavior
- **Security**: Built-in authentication, authorization, and encryption capabilities
- **Cross-platform Compatibility**: Runs on Linux, Windows, and macOS with consistent APIs
- **Industrial Standards**: Compliance with industry communication protocols

The core philosophy of ROS 2 is to provide a flexible framework that enables developers to focus on robot-specific logic rather than low-level communication infrastructure.

## Key Differences from ROS 1

ROS 2 represents a fundamental architectural shift from ROS 1, addressing many of its predecessor's limitations:

- **Middleware**: ROS 2 uses DDS (Data Distribution Service) instead of the ROS Master, enabling true distributed computing without a central point of failure
- **Real-time support**: Better support for real-time systems with deterministic timing guarantees
- **Security**: Built-in security features including authentication, authorization, and encryption
- **Cross-platform**: Improved support for Windows, macOS, and various Linux distributions with consistent APIs
- **Quality of Service (QoS)**: Configurable communication policies allowing fine-tuned control over message delivery
- **Package management**: Uses colcon for building packages instead of catkin, providing better performance and flexibility
- **API consistency**: More consistent and intuitive APIs across different programming languages
- **Memory management**: Improved memory management and reduced memory footprint

These improvements make ROS 2 particularly suitable for humanoid robotics applications where reliability, security, and real-time performance are critical.

## Installing ROS 2

For this course, we recommend using ROS 2 Humble Hawksbill, which is an LTS (Long Term Support) version that provides stability for long-term projects. Here's the complete installation process:

### Prerequisites

Before installing ROS 2, ensure your system meets the following requirements:
- Ubuntu 22.04 LTS (recommended) or Ubuntu 20.04 LTS
- At least 10GB of free disk space
- Sufficient RAM (8GB minimum, 16GB recommended)
- Internet connection for package downloads

### Installation Steps

1. **Update your system and install prerequisites:**
```bash
sudo apt update && sudo apt install curl gnupg lsb-release
```

2. **Add the ROS 2 repository:**
```bash
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

3. **Install ROS 2 Humble:**
```bash
sudo apt update
sudo apt install ros-humble-desktop
```

4. **Install additional dependencies:**
```bash
sudo apt install python3-rosdep2
sudo rosdep init
rosdep update
```

5. **Set up environment variables:**
Add the following line to your `~/.bashrc` file to automatically source ROS 2:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

Don't forget to source the ROS 2 environment in each new terminal:
```bash
source /opt/ros/humble/setup.bash
```

### Verification

To verify your installation, run:
```bash
ros2 --version
```

This should display the ROS 2 version information, confirming a successful installation.

## Key Concepts

Understanding the fundamental concepts of ROS 2 is crucial for developing complex robotic systems. Each concept builds upon the others to create a cohesive framework for robot development.

### Nodes
Nodes are processes that perform computation. ROS 2 is designed to be modular, with each node running independently and communicating with other nodes through messages. Nodes are the basic building blocks of a ROS 2 system and can be written in multiple programming languages (C++, Python, etc.).

**Node characteristics:**
- Each node has a unique name within the ROS 2 graph
- Nodes can be started and stopped independently
- Nodes can be distributed across multiple machines
- Nodes can be written in different programming languages
- Nodes can be organized into namespaces for better organization

**Creating nodes with different client libraries:**
- `rclpy`: Python client library for ROS 2
- `rclcpp`: C++ client library for ROS 2
- Other languages: Rust, Java, C#, etc.

### Topics and Messages
Topics are named buses over which nodes exchange messages. Messages are the data packets sent from publishers to subscribers over topics. This publisher-subscriber pattern enables asynchronous communication between nodes.

**Topic communication patterns:**
- **Unidirectional**: Publishers send data, subscribers receive data
- **Many-to-many**: Multiple publishers can send to one topic, multiple subscribers can listen to one topic
- **Asynchronous**: Publishers and subscribers don't need to run simultaneously
- **Type-safe**: Messages have defined data structures and types

**Common message types:**
- `std_msgs`: Basic data types (Int, Float, String, etc.)
- `geometry_msgs`: Geometric primitives (Point, Pose, Twist, etc.)
- `sensor_msgs`: Sensor data (LaserScan, Image, JointState, etc.)
- `nav_msgs`: Navigation-specific messages (Odometry, Path, etc.)

### Services
Services provide a request/response mechanism for communication between nodes, allowing nodes to request specific actions or information. Unlike topics, services are synchronous and provide immediate responses.

**Service characteristics:**
- **Synchronous**: Client waits for response from server
- **Request/Response**: Client sends request, server sends response
- **Stateless**: Each request is independent of others
- **Blocking**: Client blocks until response is received

### Actions
Actions are a more sophisticated form of communication that support long-running tasks with feedback and the ability to cancel. They combine the best features of topics and services for complex operations.

**Action characteristics:**
- **Long-running**: Designed for operations that take significant time
- **Feedback**: Provides continuous feedback during execution
- **Cancelation**: Clients can cancel ongoing actions
- **Goal management**: Server can handle multiple goals simultaneously

## Why ROS 2 for Humanoid Robotics?

Humanoid robots require sophisticated coordination between multiple systems including perception, planning, control, and interaction. ROS 2 provides several key advantages that make it ideal for humanoid robotics development:

### Real-time Capabilities
Humanoid robots need precise timing for:
- **Joint control**: Real-time control of multiple actuators
- **Balance control**: Continuous adjustment of center of mass
- **Sensor fusion**: Processing data from multiple sensors simultaneously
- **Safety systems**: Immediate response to dangerous situations

### Distributed Architecture
Humanoid robots often have:
- **Multiple processing units**: Different subsystems on separate computers
- **Sensor distribution**: Sensors located throughout the robot body
- **Modular design**: Independent subsystems that can be developed separately
- **Scalability**: Ability to add new components without major rework

### Rich Ecosystem
ROS 2 provides extensive libraries for:
- **Computer vision**: OpenCV integration, image processing
- **Path planning**: Navigation2 stack, motion planning
- **Control systems**: PID controllers, trajectory generation
- **Simulation**: Gazebo, Webots integration
- **Perception**: SLAM, object detection, tracking

### Standardized Interfaces
ROS 2 ensures:
- **Component interoperability**: Different components can work together
- **Hardware abstraction**: Same interfaces for different hardware
- **Community support**: Shared packages and solutions
- **Documentation**: Standardized documentation formats

## Python and ROS 2 (rclpy)

ROS 2 supports multiple programming languages, with Python being one of the most popular choices for rapid prototyping and development. The `rclpy` library provides Python bindings for ROS 2, allowing developers to create ROS 2 nodes using familiar Python syntax.

### Advanced Node Configuration

Nodes can be configured with various parameters and settings:

```python
# simple_robot_pkg/advanced_node.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

class AdvancedRobotNode(Node):
    def __init__(self):
        # Initialize with node name and namespace
        super().__init__('advanced_robot_node', namespace='/humanoid_robot')

        # Create a QoS profile for reliable communication
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # Create publisher with custom QoS
        self.publisher = self.create_publisher(
            String,
            'robot_status',
            qos_profile
        )

        # Create timer with custom period
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.counter = 0

        # Log information
        self.get_logger().info('Advanced Robot Node initialized')
        self.get_logger().info(f'Namespace: {self.get_namespace()}')

    def timer_callback(self):
        msg = String()
        msg.data = f'Advanced Robot Status - Counter: {self.counter}'
        self.publisher.publish(msg)
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedRobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Publisher Example with rclpy

Here's how to create a publisher using rclpy:

```python
# simple_robot_pkg/simple_publisher.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SimplePublisher(Node):
    def __init__(self):
        super().__init__('simple_publisher')
        self.publisher = self.create_publisher(String, 'robot_status', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Robot status: operational - {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    publisher = SimplePublisher()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Example with rclpy

And here's a corresponding subscriber:

```python
# simple_robot_pkg/simple_subscriber.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SimpleSubscriber(Node):
    def __init__(self):
        super().__init__('simple_subscriber')
        self.subscription = self.create_subscription(
            String,
            'robot_status',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    subscriber = SimpleSubscriber()
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced rclpy Example: Publisher with Custom Message

For more complex robotic applications, you might need to create custom messages. Here's an example of how to work with custom messages:

```python
# simple_robot_pkg/joint_state_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_joint_states)

        # Initialize joint names and positions
        self.joint_names = ['hip_joint', 'knee_joint', 'ankle_joint',
                           'shoulder_joint', 'elbow_joint', 'wrist_joint']
        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)

    def publish_joint_states(self):
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts

        self.publisher.publish(msg)
        self.get_logger().info(f'Published joint states: {self.joint_positions}')

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Complex Publisher Example: Robot Control Interface

For humanoid robotics, we often need more sophisticated publishers that can handle complex control commands:

```python
# simple_robot_pkg/robot_control_publisher.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time
import math

class RobotControlPublisher(Node):
    def __init__(self):
        super().__init__('robot_control_publisher')

        # Publisher for base velocity commands
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Publisher for joint commands
        self.joint_cmd_publisher = self.create_publisher(JointState, 'joint_commands', 10)

        # Timer for periodic control updates
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        # Robot state tracking
        self.robot_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        self.get_logger().info('Robot Control Publisher initialized')

    def control_loop(self):
        """Main control loop that updates robot state and publishes commands"""
        # Update robot pose based on current velocities
        dt = 0.05  # timer period
        self.robot_pose['x'] += self.linear_velocity * math.cos(self.robot_pose['theta']) * dt
        self.robot_pose['y'] += self.linear_velocity * math.sin(self.robot_pose['theta']) * dt
        self.robot_pose['theta'] += self.angular_velocity * dt

        # Publish velocity commands
        self.publish_velocity_commands()

        # Publish joint commands (example for humanoid joints)
        self.publish_joint_commands()

    def publish_velocity_commands(self):
        """Publish velocity commands to robot base"""
        cmd_vel = Twist()
        cmd_vel.linear.x = self.linear_velocity
        cmd_vel.angular.z = self.angular_velocity
        self.cmd_vel_publisher.publish(cmd_vel)

    def publish_joint_commands(self):
        """Publish joint commands for humanoid robot"""
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.header.frame_id = 'base_link'

        # Example joint names for a simple humanoid model
        joint_cmd.name = ['left_hip', 'right_hip', 'left_knee', 'right_knee',
                         'left_ankle', 'right_ankle', 'left_shoulder', 'right_shoulder']

        # Calculate joint positions based on walking pattern
        t = self.get_clock().now().nanoseconds / 1e9  # time in seconds
        amplitude = 0.2
        frequency = 1.0

        # Generate walking pattern for joints
        joint_cmd.position = [
            amplitude * math.sin(2 * math.pi * frequency * t),      # left_hip
            amplitude * math.sin(2 * math.pi * frequency * t),      # right_hip
            amplitude * math.sin(2 * math.pi * frequency * t + math.pi),  # left_knee
            amplitude * math.sin(2 * math.pi * frequency * t + math.pi),  # right_knee
            amplitude * math.sin(2 * math.pi * frequency * t),      # left_ankle
            amplitude * math.sin(2 * math.pi * frequency * t),      # right_ankle
            0.0,  # left_shoulder
            0.0   # right_shoulder
        ]

        self.joint_cmd_publisher.publish(joint_cmd)

    def set_linear_velocity(self, velocity):
        """Set linear velocity for the robot"""
        self.linear_velocity = velocity

    def set_angular_velocity(self, velocity):
        """Set angular velocity for the robot"""
        self.angular_velocity = velocity

def main(args=None):
    rclpy.init(args=args)
    node = RobotControlPublisher()

    # Example: Set some initial velocities
    node.set_linear_velocity(0.5)  # Move forward at 0.5 m/s
    node.set_angular_velocity(0.1) # Turn slightly right

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Robot Control Publisher interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Your First ROS 2 Package

Let's create a simple ROS 2 package to understand the structure. This package will serve as a foundation for our humanoid robot applications:

```bash
# Create a workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build the workspace initially
colcon build
source install/setup.bash

# Navigate to src and create package
cd src
ros2 pkg create --build-type ament_python simple_robot_pkg --dependencies rclpy std_msgs sensor_msgs geometry_msgs
```

This creates a basic package structure with the following files:
- `package.xml`: Package metadata containing dependencies, version, and maintainers
- `setup.py`: Python package configuration with entry points and dependencies
- `setup.cfg`: Installation configuration for Python package installation
- `simple_robot_pkg/`: Python module directory containing your Python code
- `simple_robot_pkg/__init__.py`: Python package initialization file
- `simple_robot_pkg/simple_robot_pkg/`: Main package directory

### Package Structure Explained

The `package.xml` file contains important metadata:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>simple_robot_pkg</name>
  <version>0.0.0</version>
  <description>Simple robot package for humanoid robotics</description>
  <maintainer email="user@todo.todo">user</maintainer>
  <license>Apache-2.0</license>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <exec_depend>rclpy</exec_depend>
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### Entry Points Configuration

The `setup.py` file configures the executables:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'simple_robot_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files if any
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Simple robot package for humanoid robotics',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_node = simple_robot_pkg.simple_node:main',
            'simple_publisher = simple_robot_pkg.simple_publisher:main',
            'simple_subscriber = simple_robot_pkg.simple_subscriber:main',
            'joint_state_publisher = simple_robot_pkg.joint_state_publisher:main',
            'robot_control_publisher = simple_robot_pkg.robot_control_publisher:main',
        ],
    },
)
```

## Running Your First Example

1. **Build your package:**
```bash
cd ~/ros2_ws
colcon build --packages-select simple_robot_pkg
source install/setup.bash
```

2. **In one terminal, run the publisher:**
```bash
ros2 run simple_robot_pkg simple_publisher
```

3. **In another terminal, run the subscriber:**
```bash
ros2 run simple_robot_pkg simple_subscriber
```

4. **You can also use ROS 2 command-line tools to inspect the system:**
```bash
# List all nodes
ros2 node list

# List all topics
ros2 topic list

# Echo messages on a topic
ros2 topic echo /robot_status std_msgs/msg/String

# Show topic information
ros2 topic info /robot_status
```

You should see the publisher sending messages and the subscriber receiving them.

## ROS 2 in the Context of Humanoid Robotics

For humanoid robots, ROS 2 serves as the integration platform where different subsystems communicate. The complexity of humanoid robots requires sophisticated communication patterns:

### Sensor Integration
- **IMU (Inertial Measurement Unit)**: Provides orientation, acceleration, and angular velocity data
- **Cameras**: RGB, depth, and thermal imaging for perception
- **Force/Torque Sensors**: Measure forces at joints and contact points
- **Joint Encoders**: Provide precise joint position feedback
- **LIDAR**: 3D mapping and obstacle detection

### Actuator Control
- **Joint Controllers**: PID controllers for precise joint positioning
- **Motor Drivers**: Low-level motor control interfaces
- **Servo Systems**: High-precision actuator control
- **Hydraulic/Pneumatic Systems**: For powerful humanoid actuators

### Perception Systems
- **Computer Vision**: Object detection, tracking, and recognition
- **SLAM (Simultaneous Localization and Mapping)**: Environment mapping
- **Sensor Fusion**: Combining data from multiple sensors
- **Depth Perception**: 3D scene understanding

### Planning and Control
- **Motion Planning**: Path planning and trajectory generation
- **Balance Control**: Center of mass control for bipedal locomotion
- **Gait Generation**: Walking pattern generation for humanoid locomotion
- **Task Planning**: High-level action planning

## Basic ROS 2 Architecture

ROS 2 uses a client library architecture where nodes are built using client libraries such as `rclpy` (Python) or `rclcpp` (C++). These client libraries communicate with the ROS 2 middleware (RMW - ROS Middleware) which handles the actual message passing.

```
[Node] --(rclpy/rclcpp)--> [RMW] --(DDS)--> [Network]
```

### DDS (Data Distribution Service) Layer

DDS provides several important capabilities:
- **Discovery**: Automatic discovery of nodes, topics, and services
- **Transport**: Reliable message transport with configurable policies
- **Quality of Service**: Configurable reliability, durability, and liveliness
- **Partitioning**: Logical separation of communication domains
- **Security**: Authentication, access control, and encryption

### Middleware Implementation

ROS 2 supports multiple DDS implementations:
- **Fast DDS**: Default implementation by eProsima
- **Cyclone DDS**: Eclipse implementation
- **RTI Connext DDS**: Commercial implementation
- **OpenSplice DDS**: PrismTech implementation

## Advanced ROS 2 Concepts for Humanoid Robotics

### Quality of Service (QoS) Configuration

For humanoid robotics applications, proper QoS configuration is critical:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# For critical control commands (e.g., joint commands)
control_qos = QoSProfile(
    depth=1,  # Only keep the latest command
    reliability=ReliabilityPolicy.RELIABLE,  # Ensure delivery
    durability=DurabilityPolicy.VOLATILE,  # Don't keep old data
    history=HistoryPolicy.KEEP_LAST  # Keep only recent messages
)

# For sensor data (e.g., IMU, encoders)
sensor_qos = QoSProfile(
    depth=10,  # Keep some history for processing
    reliability=ReliabilityPolicy.BEST_EFFORT,  # Accept some packet loss
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST
)

# For logging/debugging
logging_qos = QoSProfile(
    depth=1000,  # Keep extensive history
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,  # Keep until subscribers connect
    history=HistoryPolicy.KEEP_ALL  # Keep all messages
)
```

### Lifecycle Nodes for Complex Systems

For humanoid robots with complex startup and shutdown sequences:

```python
from rclpy.lifecycle import LifecycleNode, LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn

class HumanoidLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('humanoid_lifecycle_node')
        self.get_logger().info('Lifecycle node created')

    def on_configure(self, state):
        self.get_logger().info('Configuring...')
        # Initialize hardware interfaces, parameters, etc.
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('Activating...')
        # Start publishers, subscribers, timers
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info('Deactivating...')
        # Stop publishers, subscribers, timers
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self.get_logger().info('Cleaning up...')
        # Clean up resources
        return TransitionCallbackReturn.SUCCESS
```

## Summary

ROS 2 provides the essential infrastructure for developing complex robotic systems like humanoid robots. Its distributed architecture, rich ecosystem, and support for multiple programming languages make it an ideal choice for coordinating the various subsystems required in humanoid robotics.

The key advantages of ROS 2 for humanoid robotics include:
- **Robust communication**: Reliable message passing with configurable QoS
- **Real-time capabilities**: Support for time-critical control applications
- **Security**: Built-in security features for safe operation
- **Cross-platform**: Consistent APIs across different operating systems
- **Extensive ecosystem**: Rich libraries and tools for robotics development
- **Modular design**: Independent components that can be developed separately

In the next chapter, we'll dive deeper into the core concepts of ROS 2 and explore how to implement more sophisticated communication patterns between nodes, including services, actions, and advanced topic configurations.