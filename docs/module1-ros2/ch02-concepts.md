---
title: 'Chapter 2: Core Concepts - Nodes, Topics, Services, & Actions'
---

# Chapter 2: Core Concepts - Nodes, Topics, Services, & Actions

Now that you have a high-level understanding of what ROS 2 is, let's break down its fundamental components. These are the building blocks you will use to construct any robotics application.

## The ROS 2 Graph

A ROS 2 system is best understood as a **graph** of processes that are all connected and passing data to one another. The elements of this graph are:

-   **Nodes**: The processes that perform computation.
-   **Topics**: A bus for nodes to send and receive data (one-to-many).
-   **Services**: A way for nodes to request information or trigger an action (one-to-one request/response).
-   **Actions**: A way for nodes to trigger a long-running task that provides feedback.

Let's explore each of these in detail.

## Nodes

A **node** is the smallest unit of executable code in ROS 2. Think of a node as a single, self-contained program that performs a specific task. For example, you might have:

-   A node that controls a camera and publishes images.
-   A node that processes those images to detect objects.
-   A node that takes object locations and commands a robot arm to pick them up.
-   A node that controls the wheels of the robot.

Each node in a ROS 2 system should be responsible for a single, well-defined purpose. This modularity makes the system easier to design, debug, and scale. Nodes are written using one of the ROS Client Libraries (RCL), such as `rclpy` for Python or `rclcpp` for C++.

## Topics (Asynchronous, One-to-Many)

**Topics** are the primary method for continuous data flow in ROS 2. They are named buses over which nodes can exchange messages.

-   A node that sends data to a topic is called a **Publisher**.
-   A node that receives data from a topic is called a **Subscriber**.

A key feature of topics is that they are a **one-to-many** communication method. Many publishers can send data to the same topic, and many subscribers can listen to that same topic. The publishers and subscribers are decoupled; they don't know or care about each other's existence. They only care about the topic name and the message type.

**Use Case**: A camera node would *publish* images to an `/image_raw` topic. An image processing node would *subscribe* to `/image_raw` to receive the images for analysis. A logging node could also subscribe to the same topic to save the images to disk.

## Services (Synchronous, One-to-One)

**Services** are used for synchronous, request/response communication. They are ideal for when a node needs to ask another node a direct question and wait for a direct answer.

-   A node that provides a service is called a **Service Server**.
-   A node that requests a service is called a **Service Client**.

Unlike topics, services are a **one-to-one** communication method. A client sends a single request to a server and waits for a single response. This is a blocking operation, meaning the client will pause its execution until the response is received (or a timeout occurs).

**Use Case**: Imagine you have a node that can calculate the inverse kinematics for a robot arm. Another node could act as a service *client* to send a request like "What joint angles are needed to reach position (x, y, z)?" The inverse kinematics node, acting as a service *server*, would receive the request, perform the calculation, and send back a response with the required joint angles.

## Actions (Asynchronous, One-to-One with Feedback)

**Actions** are designed for long-running, non-blocking tasks that need to provide feedback during their execution. They are similar to services but provide more structure for tasks that take time.

An action involves three parties:

-   An **Action Client** sends a goal to an action server.
-   An **Action Server** receives the goal and works to achieve it.
-   The client can receive continuous **feedback** from the server.
-   Once the task is complete, the server sends a final **result**.

The client can also cancel a goal at any time. This makes actions perfect for tasks like navigation, where a robot needs to travel to a destination.

**Use Case**: A navigation node could send a goal like "Go to coordinates (x, y)" to a robot's motion control action server.
-   The **Action Client** (navigation node) sends the goal.
-   The **Action Server** (motion controller) accepts the goal and starts moving the robot.
-   The server periodically sends **feedback** to the client, such as "Current distance to goal: 5.2 meters".
-   If something blocks the path, the client can choose to **cancel** the goal.
-   When the robot arrives, the server sends a final **result** like "Goal reached successfully".

## Summary of Communication Methods

| Method  | Pattern              | Behavior     | Use For                                          |
| :------ | :------------------- | :----------- | :----------------------------------------------- |
| **Topic**   | Publish / Subscribe  | Asynchronous | Continuous data streams (e.g., sensor data)      |
| **Service** | Client / Server      | Synchronous  | Quick, transactional requests (e.g., "get status") |
| **Action**  | Client / Server      | Asynchronous | Long-running, feedback-driven tasks (e.g., "navigate") |

---

### Lab 2.1: Exploring the ROS 2 Graph

**Problem Statement**: Use ROS 2 command-line tools to inspect a running system and understand the relationships between nodes, topics, services, and actions.

**Expected Outcome**: You can list the active nodes, topics, services, and actions of a running demo and inspect their properties.

**Steps**:

1.  Open a new terminal and source your ROS 2 setup.
2.  Start the `turtlesim` demo application. This is a classic ROS demo that provides a simple simulator of a turtle that you can command.
    ```bash
    ros2 run turtlesim turtlesim_node
    ```
    You should see a new window pop up with a blue background and a single turtle in the middle.
3.  Open a **second terminal** and source your ROS 2 setup.
4.  **List the nodes**:
    ```bash
    ros2 node list
    ```
    You should see at least one node listed: `/turtlesim`.
5.  **List the topics**:
    ```bash
    ros2 topic list
    ```
    You will see several topics, such as `/turtle1/cmd_vel`, `/turtle1/color_sensor`, and `/turtle1/pose`. The `/turtlesim` node subscribes to `/turtle1/cmd_vel` to receive motion commands and publishes the turtle's position on `/turtle1/pose`.
6.  **Echo a topic**: Let's see the data being published on the `/turtle1/pose` topic.
    ```bash
    ros2 topic echo /turtle1/pose
    ```
    You'll see a stream of messages showing the turtle's x, y, and theta (orientation). Leave this running.
7.  **Publish to a topic**: Open a **third terminal** (source ROS 2 again). We will manually publish a command to make the turtle move. The `/turtle1/cmd_vel` topic expects a message of type `geometry_msgs/msg/Twist`.
    This command will make the turtle move forward and turn slightly.
    ```bash
    ros2 topic pub --once /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 2.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.8}}"
    ```
    You will see the turtle in the simulator window move. In the second terminal where `topic echo` is running, you will see the pose values change.
8.  **List the services**:
    ```bash
    ros2 service list
    ```
    You will see services like `/spawn`, `/kill`, and `/set_pen`. These allow you to make specific requests.
9.  **Call a service**: Let's call the `/spawn` service to create a new turtle.
    ```bash
    ros2 service call /spawn turtlesim/srv/Spawn "{x: 2, y: 2, theta: 0.2, name: 'turtle2'}"
    ```
    A new turtle will appear in the simulator window.
10. Close all terminals with `Ctrl+C`.

**Conclusion**: You have now used command-line tools to inspect and interact with a live ROS 2 system. You've seen how topics are used for continuous data (pose) and commands (cmd_vel), and how services are used for one-off requests (spawning a turtle).

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