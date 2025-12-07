---
title: 'Chapter 4: URDF & Launch Files'
---

# Chapter 4: URDF & Launch Files

In this chapter, we'll cover two essential tools for managing complex robot systems in ROS 2: URDF for describing the robot's physical structure, and Launch files for starting and configuring multiple nodes at once.

## Describing a Robot with URDF

**URDF** stands for **Unified Robot Description Format**. It's an XML-based format used in ROS to describe all the physical elements of a robot. A URDF file defines:

-   **Links**: The rigid parts of the robot (e.g., the base, an arm link, a gripper finger).
-   **Joints**: The connections between links, which define how they can move relative to each other (e.g., revolute, prismatic, fixed).
-   **Visuals**: The 3D mesh or shape of each link (what the robot looks like).
-   **Collisions**: The geometry of each link used for collision detection.
-   **Inertials**: The mass and inertia properties of each link for physics simulation.

By creating a URDF file, you are building a data-driven model of your robot that many different ROS 2 tools can use.

### The `robot_state_publisher`

A URDF file by itself is just a description. To make it useful, we use a special ROS 2 node called `robot_state_publisher`. This node does the following:

1.  Reads your URDF file.
2.  Subscribes to a topic (usually `/joint_states`) that provides the current state (e.g., angle) of all the robot's joints.
3.  Based on the URDF and the current joint states, it calculates the 3D pose of every link on the robot.
4.  It then publishes these poses as **transformations** using ROS 2's transformation system, `tf2`.

Other nodes in your system (like a visualization tool or a perception node) can then subscribe to `tf2` to get the real-time pose of any part of the robot.

### Example URDF Snippet

Here's a very simple example of a two-link arm:

```xml
<?xml version="1.0"?>
<robot name="simple_arm">

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </visual>
  </link>

  <!-- Arm Link -->
  <link name="arm_link">
    <visual>
      <geometry>
        <box size="0.8 0.1 0.1"/>
      </geometry>
    </visual>
  </link>

  <!-- Joint connecting base to arm -->
  <joint name="base_to_arm_joint" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/>
    <limit effort="1000.0" lower="-1.57" upper="1.57" velocity="0.5"/>
  </joint>

</robot>
```

-   We define two `<link>`s: `base_link` and `arm_link`.
-   We define one `<joint>` of type `revolute` (a rotating joint).
-   The joint connects the `parent` link (`base_link`) to the `child` link (`arm_link`).
-   The `<origin>` tag specifies where the joint is located relative to the parent link.
-   The `<axis>` tag specifies the axis of rotation.

## Managing Complexity with Launch Files

As your robot system grows, you'll find yourself opening many terminals to run many different nodes (`ros2 run ...`). This quickly becomes tedious and error-prone.

**Launch files** are the solution. A ROS 2 launch file is a Python script that allows you to start and configure a whole system of nodes with a single command.

With a launch file, you can:

-   Start multiple nodes at once.
-   Automatically start required helper nodes (like `robot_state_publisher`).
-   Pass parameters and remappings to your nodes.
-   Restart nodes automatically if they crash.
-   Group nodes into namespaces for multi-robot systems.

### A Simple Launch File

ROS 2 launch files are written in Python. They look like this:

```python
# simple_robot.launch.py

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # Path to your URDF file
    urdf_file_path = os.path.join(
        get_package_share_directory('my_robot_pkg'),
        'urdf',
        'my_robot.urdf.xacro'
    )

    # Robot State Publisher Node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': open(urdf_file_path).read()}]
    )

    # Your custom node
    my_custom_node = Node(
        package='my_robot_pkg',
        executable='my_robot_driver',
        name='my_robot_driver'
    )

    # Create the launch description and populate
    ld = LaunchDescription()

    # Add the nodes to the launch description
    ld.add_action(robot_state_publisher_node)
    ld.add_action(my_custom_node)

    return ld
```

-   The `generate_launch_description()` function is the main entry point.
-   We create `Node` objects for each node we want to run.
-   For `robot_state_publisher`, we pass the contents of our URDF file as a parameter named `robot_description`.
-   All `Node` objects are added to a `LaunchDescription` object, which is then returned.

### Running a Launch File

To run this launch file (assuming it's in a package named `my_robot_pkg` and installed in a directory named `launch`), you would use the `ros2 launch` command:

```bash
ros2 launch my_robot_pkg simple_robot.launch.py
```

This single command will start both the `robot_state_publisher` and your custom driver node, correctly configured and connected.

---

### Lab 4.1: Creating a URDF and Launch File

**Problem Statement**: Create a simple URDF for a two-wheeled robot and a launch file to start the `robot_state_publisher` for it.

**Expected Outcome**: You will have a package containing a URDF file and a launch file. When you run the launch file, the `robot_state_publisher` will start, load your URDF, and wait for joint state information. You can verify this using visualization tools like RViz2 (which we will cover in the next module).

**Steps**:

1.  **Create a package**:
    ```bash
    cd ros2_ws/src
    ros2 pkg create --build-type ament_python --license Apache-2.0 simple_robot_pkg
    ```

2.  **Create URDF file**:
    -   Inside `simple_robot_pkg`, create a directory called `urdf`.
    -   Inside `urdf`, create a file named `simple_bot.urdf`.
    -   Add the following content to `simple_bot.urdf`:
      ```xml
      <robot name="simple_bot">
        <link name="base_link">
          <visual>
            <geometry><box size="0.6 0.4 0.2"/></geometry>
          </visual>
        </link>
        <link name="right_wheel_link">
          <visual>
            <geometry><cylinder radius="0.1" length="0.05"/></geometry>
          </visual>
        </link>
        <joint name="base_to_right_wheel" type="continuous">
          <parent link="base_link"/>
          <child link="right_wheel_link"/>
          <origin xyz="0 -0.25 0" rpy="1.5707 0 0"/>
          <axis xyz="0 0 1"/>
        </joint>
      </robot>
      ```

3.  **Create Launch file**:
    -   Inside `simple_robot_pkg`, create a directory called `launch`.
    -   Inside `launch`, create a file named `display.launch.py`.
    -   Add the following content:
      ```python
      import os
      from ament_index_python.packages import get_package_share_directory
      from launch import LaunchDescription
      from launch_ros.actions import Node

      def generate_launch_description():
          urdf_path = os.path.join(
              get_package_share_directory('simple_robot_pkg'),
              'urdf', 'simple_bot.urdf')
          
          return LaunchDescription([
              Node(
                  package='robot_state_publisher',
                  executable='robot_state_publisher',
                  name='robot_state_publisher',
                  output='screen',
                  parameters=[{'robot_description': open(urdf_path).read()}])
          ])
      ```

4.  **Install launch and urdf files**:
    -   Edit your `setup.py` to tell the build system about your new `launch` and `urdf` directories.
      ```python
      # setup.py
      from setuptools import setup
      import os
      from glob import glob

      package_name = 'simple_robot_pkg'

      setup(
          # ... other setup args
          data_files=[
              ('share/ament_index/resource_index/packages',
                  ['resource/' + package_name]),
              ('share/' + package_name, ['package.xml']),
              # Include all launch files
              (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
              # Include all urdf files
              (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
          ],
      )
      ```

5.  **Build and Run**:
    -   Build your workspace: `cd ~/ros2_ws && colcon build`
    -   Source the workspace: `source ~/ros2_ws/install/setup.bash`
    -   Run the launch file: `ros2 launch simple_robot_pkg display.launch.py`

**Conclusion**: If the launch command runs without errors, you have successfully created a URDF model and a launch file to publish its structure to the ROS 2 network. This is the first step towards visualizing and simulating your robot.

---

## References

[1] Robot Operating System (ROS) Website: https://www.ros.org/
[2] ROS 2 Documentation: https://docs.ros.org/en/humble/index.html
[3] Data Distribution Service (DDS) Standard: https://www.omg.org/dds/
[4] Gazebo Documentation: http://gazebosim.org/
[5] Unity Robotics Hub: https://github.com/Unity-Technologies/Unity-Robotics-Hub
[6] NVIDIA Isaac Sim Documentation: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html
[7] NVIDIA Isaac ROS Documentation: https://nvidia-isaac-ros.github.io/
[8] Nav2 Documentation: https://navigation.ros.org/
[9] OpenAI Whisper API Documentation: https://platform.openai.com/docs/guides/speech-to-text
[10] OpenAI API Documentation: https://platform.openai.com/docs/api-reference
[11] IEEE Editorial Style Manual: https://www.ieee.org/content/dam/ieee-org/ieee/web/org/pubs/ieee_style_manual.pdf