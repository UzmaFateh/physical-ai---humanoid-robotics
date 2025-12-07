---
title: 'Chapter 7: High-Fidelity Visualization with Unity'
---

# Chapter 7: High-Fidelity Visualization with Unity

While Gazebo is an excellent tool for physics simulation, its graphical capabilities, while functional, are not state-of-the-art. For creating photorealistic environments, advanced lighting, and high-quality visual effects, game engines like Unity are unparalleled.

In a "digital twin" workflow, it's common to use Gazebo for the physics and sensor simulation and a game engine like Unity purely for visualization. This gives you the best of both worlds: accurate physics and beautiful graphics.

## Why Use Unity for Visualization?

-   **Photorealism**: Unity's Universal Render Pipeline (URP) and High Definition Render Pipeline (HDRP) can produce stunningly realistic visuals.
-   **Asset Store**: Access to a massive ecosystem of pre-made 3D models, textures, and environments.
-   **Advanced Tools**: A sophisticated editor for building scenes, animating objects, and creating visual effects.
-   **Cross-Platform**: Easily build visualizations that run on Windows, macOS, Linux, and even VR/AR devices.

The goal is not to replace Gazebo, but to augment it. The simulation "source of truth" remains in Gazebo, while Unity acts as a beautiful, real-time "viewer" of that simulation.

## Connecting ROS 2 to Unity

The key to this workflow is connecting your ROS 2 graph to the Unity environment. The official and most effective way to do this is with the **Unity Robotics Hub**. Specifically, we will use the [ROS-TCP-Connector](https://github.com/Unity-Technologies/ROS-TCP-Connector) package.

### How ROS-TCP-Connector Works

The system consists of two main parts:

1.  **A ROS 2 node (`ros_tcp_endpoint`)**: This is a ROS 2 node that you run on the ROS side. It acts as a server that listens for incoming TCP connections. It subscribes to ROS topics you specify and forwards the messages over TCP. It also receives messages over TCP and publishes them onto ROS topics.

2.  **A Unity Plugin**: This is a C# library that you import into your Unity project. It provides the components to connect to the `ros_tcp_endpoint` server and to create publishers and subscribers within your Unity scene.

The workflow is as follows:

1.  You run the `ros_tcp_endpoint` node in your ROS 2 system.
2.  You start your Unity application.
3.  The Unity TCP Connector component connects to the ROS `ros_tcp_endpoint`.
4.  A ROS subscriber in your Unity project (e.g., one that listens to `/joint_states`) receives a message from ROS.
5.  A C# script in Unity takes this joint state data and applies it to the joints of a robot model in the Unity scene.

The result: The robot model in Unity moves in perfect sync with the robot in the Gazebo simulation.

## Setting Up a Unity Project

1.  **Install Unity Hub and Unity Editor**: Download and install a recent version of the Unity Editor (e.g., 2022.3 LTS).

2.  **Create a New 3D Project**.

3.  **Install the ROS-TCP-Connector**:
    -   In Unity, go to `Window -> Package Manager`.
    -   Click the `+` icon and select "Add package from git URL...".
    -   Enter `https://github.com/Unity-Technologies/ROS-TCP-Connector.git`.
    -   Unity will download and install the package.

4.  **Import URDF**:
    -   Install the `URDF-Importer` package from the same Unity Robotics GitHub organization (`https://github.com/Unity-Technologies/URDF-Importer.git`).
    -   This tool allows you to import your robot's URDF file directly into Unity. It will parse the file and create a corresponding hierarchy of GameObjects that represent your robot's links and joints.

5.  **Configure the Connection**:
    -   Add the `ROSConnection` prefab to your scene.
    -   In its Inspector, set the "ROS IP Address" to the IP of the machine running ROS. If it's the same machine, you can use `127.0.0.1`.

## Visualizing Robot State

To make the robot in Unity mirror the simulation, you need a script to apply the joint states.

-   The ROS-TCP-Connector package provides a component called `ROSJointController` (or you can write your own).
-   You add this script to your robot model in Unity.
-   You configure it to subscribe to the `/joint_states` topic.
-   In its `OnMessageReceived` method, it will receive a `sensor_msgs/JointState` message.
-   The script will then iterate through the joint names and positions in the message and apply the corresponding rotations to the robot's joints in the Unity scene.

This creates a powerful one-way visualization stream: `Gazebo (Physics) -> ROS Topics -> Unity (Graphics)`.

---

### Lab 7.1: Visualizing Joint States in Unity

**Problem Statement**: Create a simple Unity scene that can connect to a ROS 2 network and visualize the motion of a single rotating joint.

**Expected Outcome**: You will have a Unity scene with a simple two-link arm. When you publish a `JointState` message from a ROS 2 terminal, the arm in the Unity scene will rotate to the specified angle.

**Steps**:

1.  **Setup Unity**: Create a new 3D Unity project and install the `ROS-TCP-Connector` package as described above.

2.  **Create a Simple "Robot"**:
    -   In the Unity scene, create two simple cubes (`GameObject -> 3D Object -> Cube`).
    -   Arrange them to look like a base link and an arm link, with the arm positioned as if it's attached by a hinge to the base.
    -   Create an empty GameObject to act as the parent for the arm link. This will be our joint.

3.  **Add ROS Components**:
    -   Drag the `ROSConnection` prefab into your scene from the `Assets/RosMessages` folder. Configure the IP address.
    -   Create a new C# script called `SimpleJointController`.

4.  **Write the Controller Script**:
    -   Edit `SimpleJointController.cs`.
      ```csharp
      using UnityEngine;
      using Unity.Robotics.ROSTCPConnector;
      using RosMessageTypes.Sensor; // For JointState message

      public class SimpleJointController : MonoBehaviour
      {
          public GameObject joint; // Assign your joint GameObject in the Inspector

          void Start()
          {
              // Subscribe to the /joint_states topic
              ROSConnection.GetOrCreateInstance().Subscribe<JointStateMsg>("joint_states", OnJointStateReceived);
          }

          void OnJointStateReceived(JointStateMsg jointState)
          {
              // Assuming the first joint in the message is ours
              if (jointState.name.Length > 0 && jointState.position.Length > 0)
              {
                  // The message position is in radians, Unity uses degrees
                  float angleDegrees = (float)jointState.position[0] * Mathf.Rad2Deg;
                  
                  // Apply the rotation to the joint
                  joint.transform.localEulerAngles = new Vector3(0, angleDegrees, 0);
              }
          }
      }
      ```
    -   Attach this script to an object in your scene and drag your joint GameObject onto the `joint` public variable in the Inspector.

5.  **Run the ROS Side**:
    -   In a terminal, start the ROS TCP Endpoint. You'll need to have installed this package first (`pip install roslibpy`).
      ```bash
      ros2 run ros_tcp_endpoint main --ros-args -p ROS_IP:=127.0.0.1
      ```
    -   Press "Play" in the Unity Editor. You should see a log message in the ROS terminal indicating a client has connected.

6.  **Publish a Joint State**:
    -   In a second terminal, publish a `JointState` message.
      ```bash
      ros2 topic pub --once /joint_states sensor_msgs/msg/JointState "{name: ['my_joint'], position: [1.57]}"
      ```

7.  **Verify**: The "arm" cube in your Unity scene should rotate by 1.57 radians (90 degrees).

**Conclusion**: You have successfully bridged the gap between ROS 2 and a high-fidelity game engine. You can now subscribe to any ROS topic from within Unity and use that data to drive a visually rich simulation.

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