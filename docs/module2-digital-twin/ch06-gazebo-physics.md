---
title: 'Chapter 6: Gazebo Physics & Sensors'
---

# Chapter 6: Gazebo Physics & Sensors

Gazebo is more than just a 3D viewer; it's a powerful physics simulator. It uses pluggable physics engines (like ODE, Bullet, Simbody, and DART) to simulate the dynamics of your robot and its interaction with the world. This chapter explores how to define physical properties and add sensors to your robot model.

## Defining Physical Properties

For a realistic simulation, every link in your robot needs physical properties. These are defined within the `<inertial>` and `<collision>` tags in your URDF.

### The `<inertial>` Tag

The `<inertial>` tag specifies the mass and rotational inertia of a link.

```xml
<link name="my_link">
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="1.0"/>
    <inertia 
      ixx="0.01" ixy="0.0" ixz="0.0" 
      iyy="0.01" iyz="0.0" 
      izz="0.01"/>
  </inertial>
</link>
```

-   `<mass>`: The mass of the link in kilograms.
-   `<inertia>`: The 3x3 rotational inertia matrix. `ixx`, `iyy`, and `izz` are moments of inertia, while `ixy`, `ixz`, and `iyz` are products of inertia. For symmetrical shapes centered at the origin, the products of inertia are typically zero.

Calculating these values by hand is difficult. CAD software can often generate them automatically for a given model.

### The `<collision>` and `<contact>` Tags

The `<collision>` tag defines the geometry of a link for the physics engine. This can be different from the `<visual>` geometry, allowing you to use a simpler shape (like a sphere or box) for collision calculations to improve performance.

Within a Gazebo `<gazebo>` block, you can further define surface properties inside a `<contact>` tag.

```xml
<gazebo reference="my_link">
  <collision>
    <surface>
      <contact>
        <ode>
          <kp>1000000.0</kp>
          <kd>100.0</kd>
          <max_vel>1.0</max_vel>
          <min_depth>0.001</min_depth>
        </ode>
      </contact>
      <friction>
        <ode>
          <mu>1.0</mu>
          <mu2>1.0</mu2>
        </ode>
      </friction>
    </surface>
  </collision>
</gazebo>
```

Here, we've defined:
-   **Contact stiffness (`kp`) and damping (`kd`)**: These determine how "hard" or "bouncy" a surface is when it collides with another.
-   **Friction coefficients (`mu`, `mu2`)**: These define static and dynamic friction.

Tuning these parameters is an art and is crucial for achieving stable and realistic simulation behavior.

## Adding Sensors

One of the most powerful features of Gazebo is its ability to simulate a wide variety of sensors. Sensors are added to links within a `<gazebo>` block using the `<sensor>` tag. The data from these simulated sensors is then published to ROS 2 topics, allowing your software stack to behave exactly as it would with real hardware.

All sensors are implemented as plugins, and Gazebo comes with a rich library.

### Common Sensor Types

-   **Camera**: Simulates a standard camera, publishing images to a ROS 2 topic.
-   **Depth Camera (Kinect, RealSense)**: Simulates a camera that provides a point cloud or depth image, measuring the distance to objects in its view.
-   **Lidar (Laser Scanner)**: Simulates a 2D or 3D Lidar, publishing laser scan data.
-   **IMU (Inertial Measurement Unit)**: Simulates an IMU, providing acceleration and angular velocity data.
-   **Contact Sensor**: A simple sensor that reports when its parent link collides with another object.

### Example: Adding a Lidar

Let's add a 2D Lidar to the `base_link` of a robot.

```xml
<gazebo reference="base_link">
  <sensor type="ray" name="lidar_sensor">
    <pose>0 0 0.15 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>20</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-1.570796</min_angle>
          <max_angle>1.570796</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.10</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="gazebo_ros_head_hokuyo_controller" filename="libgazebo_ros_laser.so">
      <ros>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>lidar_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

-   `<sensor type="ray">`: Defines this as a Lidar/laser-based sensor.
-   `<pose>`: The position and orientation of the sensor relative to its parent link (`base_link`).
-   `<visualize>`: If true, Gazebo will render the laser beams.
-   `<scan>`: Defines the properties of the laser scan (number of samples, resolution, field of view).
-   `<range>`: Defines the minimum and maximum detection distance.
-   `<plugin>`: This is the crucial part. We use the `libgazebo_ros_laser.so` plugin to handle the simulation and ROS 2 communication.
-   `<remapping>`: The plugin's default topic is `~/out`. We remap it to a more sensible name, `scan`. The ROS 2 topic will be `/scan`.
-   `<frame_name>`: The coordinate frame in which the laser scan data will be published.

With this block added to your URDF, when you spawn the robot in Gazebo, a simulated Lidar will be created, and it will start publishing `sensor_msgs/LaserScan` messages on the `/scan` topic, ready for a navigation or mapping node to use.

---

### Lab 6.1: Adding a Camera to Your Robot

**Problem Statement**: Add a simulated camera to the `base_link` of your `simple_bot.urdf`.

**Expected Outcome**: When the robot is spawned in Gazebo, a camera sensor will be active and publishing `sensor_msgs/Image` messages to a ROS 2 topic.

**Steps**:

1.  **Open your URDF**: Start with the URDF file you enhanced in the previous lab.

2.  **Add a Camera Sensor Block**: Inside the `<gazebo reference="base_link">` block, add a new `<sensor>` for the camera.
    ```xml
    <sensor type="camera" name="camera_sensor">
      <update_rate>30.0</update_rate>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>800</width>
          <height>600</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.02</near>
          <far>300</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <ros>
          <remapping>image_raw:=camera/image_raw</remapping>
          <remapping>camera_info:=camera/camera_info</remapping>
        </ros>
        <camera_name>my_camera</camera_name>
        <frame_name>camera_link</frame_name>
        <hack_baseline>0.07</hack_baseline>
      </plugin>
    </sensor>
    ```

3.  **Create a Launch File to Start Gazebo**: Spawning a robot in Gazebo requires a more complex launch file. Create a new launch file named `gazebo.launch.py`. This file will:
    -   Start the Gazebo simulation environment.
    -   Find your URDF file.
    -   Use a `spawn_entity.py` script (provided by Gazebo) to add your robot model to the simulation.

    Here is a basic template for such a launch file:
    ```python
    import os
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import IncludeLaunchDescription
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from launch_ros.actions import Node

    def generate_launch_description():
        pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
        pkg_simple_robot = get_package_share_directory('simple_robot_pkg')

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

        # Get the URDF file
        urdf_path = os.path.join(pkg_simple_robot, 'urdf', 'simple_bot.urdf')

        # Spawn the robot
        spawn_entity = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'simple_bot', '-file', urdf_path],
            output='screen'
        )

        return LaunchDescription([
            gzserver_launch,
            gzclient_launch,
            spawn_entity,
        ])
    ```

4.  **Build and Run**:
    -   Build your workspace: `colcon build`
    -   Source your workspace: `source install/setup.bash`
    -   Run your new Gazebo launch file: `ros2 launch simple_robot_pkg gazebo.launch.py`

5.  **Verify**:
    -   Gazebo should open, and you should see your robot model in an empty world.
    -   In a new terminal, check the list of topics: `ros2 topic list`. You should see `/camera/image_raw` and `/camera/camera_info`.
    -   You can try to echo the topic: `ros2 topic echo /camera/image_raw`. You will see a stream of raw image data.

**Conclusion**: You have now added a functional, simulated sensor to your robot. This same process can be used to add Lidars, IMUs, and other sensors, building up a complete digital twin of a physical robot.

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