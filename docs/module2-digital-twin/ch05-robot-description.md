---
title: 'Chapter 5: Robot Description - From URDF to SDF'
---

# Chapter 5: Robot Description - From URDF to SDF

In the previous module, we introduced URDF as the standard way to describe a robot's structure in ROS. While URDF is excellent for kinematics and visualization, it has limitations when it comes to physics simulation. It cannot specify many real-world properties like friction, damping, or detailed contact dynamics.

To address this, the simulation world introduced the **Simulation Description Format (SDF)**. SDF is an XML format designed to describe everything about a simulation environment, from robots and lights to physics and sensors.

## URDF vs. SDF

| Feature              | URDF                               | SDF                                         |
| :------------------- | :--------------------------------- | :------------------------------------------ |
| **Primary Purpose**  | Kinematic and visual description   | Full simulation environment description     |
| **Physics**          | Very limited (mass, inertia only)  | Rich (friction, damping, contact, etc.)     |
| **Sensors**          | No native support                  | Rich, plugin-based sensor models            |
| **World**            | Describes only a single robot      | Describes entire worlds (robots, lights, etc.) |
| **Extensibility**    | Limited                            | Highly extensible via plugins               |
| **ROS Integration**  | Native, via `robot_state_publisher` | Requires Gazebo-specific plugins            |

For robust simulation in Gazebo, SDF is the preferred format. Fortunately, you don't have to throw away your URDFs. The standard workflow is to enhance your URDF with special tags that Gazebo understands.

## Enhancing URDF for Gazebo

You can keep your URDF file as the primary source of your robot's description and add Gazebo-specific tags within a `<gazebo>` block. This keeps your model compatible with standard ROS tools while adding the richness needed for simulation.

### Example: Adding a Color and a Sensor

Let's enhance the `right_wheel_link` from our simple bot's URDF.

```xml
<link name="right_wheel_link">
  <visual>
    <geometry><cylinder radius="0.1" length="0.05"/></geometry>
  </visual>

  <!-- Gazebo-specific tags -->
  <gazebo reference="right_wheel_link">
    <!-- Set the material color -->
    <material>Gazebo/Red</material>

    <!-- Add a contact sensor -->
    <sensor name="right_wheel_contact_sensor" type="contact">
      <contact>
        <collision>right_wheel_collision</collision>
      </contact>
      <update_rate>30</update_rate>
    </sensor>
  </gazebo>
</link>
```

-   The `reference` attribute of the `<gazebo>` tag specifies which link these properties apply to.
-   `<material>`: We've set the wheel's color in the simulation to red using one of Gazebo's built-in materials.
-   `<sensor>`: We've attached a virtual contact sensor to the wheel, which can report when it collides with another object.

### The `<plugin>` Tag

The most powerful Gazebo tag is `<plugin>`. Plugins are shared libraries that can control almost any aspect of the simulation. A common use is to create a **differential drive** plugin that can subscribe to a ROS 2 `Twist` message (like the ones we published in Chapter 2) and convert it into wheel velocities to drive your robot.

```xml
<gazebo>
  <plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
    <ros>
      <namespace>/</namespace>
      <remapping>cmd_vel:=cmd_vel_demo</remapping>
    </ros>
    
    <!-- Wheel joints -->
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    
    <!-- Kinematics -->
    <wheel_separation>0.4</wheel_separation>
    <wheel_diameter>0.2</wheel_diameter>
    
    <!-- Output -->
    <odometry_topic>odom</odometry_topic>
    <odometry_frame>odom</odometry_frame>
    <robot_base_frame>base_link</robot_base_frame>
  </plugin>
</gazebo>
```

This plugin, provided by the `gazebo_plugins` package, does all the hard work of creating a drivable robot for you. It listens on the `/cmd_vel_demo` topic and automatically publishes odometry information (the robot's estimated position) to the `/odom` topic.

## The URDF-to-SDF Conversion Pipeline

When you use a URDF file in a Gazebo simulation, a conversion process happens behind the scenes:

1.  **Launch**: A Gazebo launch file spawns a model using your URDF file.
2.  **Parsing**: Gazebo's parser reads the URDF file.
3.  **Conversion**: It converts the URDF data structures into its internal SDF data structures.
4.  **Enrichment**: It processes all the `<gazebo>` tags and applies the simulation-specific properties.
5.  **Spawning**: The final, complete SDF model is spawned into the simulation world.

Understanding this pipeline is key to debugging simulation issues. If your robot doesn't behave as expected, the problem often lies in either the URDF-SDF conversion or the configuration within your `<gazebo>` tags.

---

### Lab 5.1: Enhancing a URDF for Simulation

**Problem Statement**: Take the `simple_bot.urdf` from the previous module and enhance it with Gazebo-specific tags to prepare it for simulation.

**Expected Outcome**: You will have an updated URDF file that specifies colors, collision geometry, and a differential drive plugin.

**Steps**:

1.  **Copy the URDF**: Start with the `simple_bot.urdf` file from `src/code-examples/module1`. Copy it into a new package for this module, or continue working in your `simple_robot_pkg`.

2.  **Add a `collision` block**: The visual geometry is not used for physics. You need to add a separate `<collision>` block. For simple shapes, it's often identical to the visual geometry.
    ```xml
    <link name="base_link">
      <visual>
        <geometry><box size="0.6 0.4 0.2"/></geometry>
      </visual>
      <collision>
        <geometry><box size="0.6 0.4 0.2"/></geometry>
      </collision>
    </link>
    ```
    Do this for both the `base_link` and the `right_wheel_link`.

3.  **Add a `left_wheel`**: Our robot is a bit unbalanced. Add a new link and joint for a `left_wheel_link` as a mirror of the right wheel. Make sure to update its origin `xyz` to be `0 0.25 0`.

4.  **Add Gazebo Color**:
    -   Inside a `<gazebo>` tag for the `base_link`, add `<material>Gazebo/Blue</material>`.
    -   Inside a `<gazebo>` tag for each wheel link, add `<material>Gazebo/Black</material>`.

5.  **Add the `diff_drive` plugin**:
    -   Add a new `<gazebo>` block at the top level of your `<robot>` tag (not referencing a specific link).
    -   Copy the differential drive plugin example from this chapter into that block.
    -   Make sure the `<left_joint>` and `<right_joint>` tags match the names of the joints you defined for your wheels.

6.  **Validate**: Although we can't fully simulate it yet, you can use the `check_urdf` tool to parse your URDF and check for syntax errors.
    ```bash
    check_urdf simple_bot.urdf
    ```
    If it runs without errors, your URDF is syntactically valid.

**Conclusion**: You have successfully taken a basic kinematic description (URDF) and enriched it with simulation-specific details (SDF properties via `<gazebo>` tags). Your robot model is now much closer to being a true "digital twin."

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