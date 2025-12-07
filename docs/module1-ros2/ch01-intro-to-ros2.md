---
title: 'Chapter 1: Introduction to ROS 2'
---

# Chapter 1: Introduction to ROS 2

Welcome to the world of robotics! This textbook will be your guide to building intelligent systems, starting with the foundational software framework used by roboticists worldwide: the Robot Operating System (ROS).

## What is ROS?

ROS is not a traditional operating system like Windows, macOS, or Linux. Instead, it's a **meta-operating system**. Think of it as a flexible framework of software, libraries, and tools that simplifies the complex task of creating robot applications. It provides services you would expect from an operating system, including:

- Hardware abstraction
- Low-level device control
- Implementation of commonly-used functionality
- Message-passing between processes
- Package management

The primary goal of ROS is to support code reuse in robotics research and development. It provides a standard, language-agnostic way for different parts of a robot's software stack to communicate with each other. A vision system written in Python can seamlessly send data to a navigation system written in C++, all thanks to ROS.

## Why ROS 2?

This textbook focuses on ROS 2, the second major version of ROS. ROS 2 was redesigned from the ground up to address the needs of modern robotics, especially for commercial and mission-critical applications. Key improvements over ROS 1 include:

- **Support for Multi-Robot Systems**: ROS 2 is designed for networks of multiple robots that need to communicate with each other.
- **Real-Time Control**: It offers improved support for real-time control loops, essential for precise and fast-acting robots.
- **Unreliable Networks**: ROS 2 is built to handle the realities of wireless communication, gracefully managing intermittent or lossy connections.
- **Production Environments**: It provides the security, robustness, and scalability needed to move a robot from a research lab to a real-world product.

ROS 2 uses a **Data Distribution Service (DDS)** for its communication layer. DDS is an industry standard for high-performance, real-time data exchange, and it's a key reason for ROS 2's improved performance and reliability.

<h2>Core ROS 2 Philosophy</h2>

The ROS 2 framework is based on a few core philosophical goals:

1.  **Peer-to-Peer**: There is no central master node. All nodes (ROS 2 processes) are equal peers, making the system more robust.
2.  **Tools-Based**: ROS is composed of many small, independent tools that can be combined to build complex systems.
3.  **Language-Agnostic**: ROS 2 APIs are available in multiple languages, with Python and C++ being the most common.
4.  **Thin**: The core of ROS is lightweight. Functionality is added through packages, so you only need to install and run what you need.
5.  **Free and Open-Source**: ROS 2 is developed and maintained by a global community, encouraging collaboration and shared innovation.

<h2>What to Expect in this Module</h2>

In this first module, we will dive deep into the fundamental concepts that make ROS 2 work. By the end of this module, you will understand:

- The core building blocks: Nodes, Topics, Services, and Actions.
- How to write your own ROS 2 applications using Python.
- How to structure your code into reusable packages.
- How to describe a robot's physical structure using URDF.
- How to manage complex robot applications using Launch files.

Let's begin our journey into the nervous system of the modern robot.

---

<h3>Lab 1.1: Setting Up Your Environment</h3>

**Problem Statement**: Before we can write any code, we need to ensure your development environment is set up correctly. This lab will guide you through verifying your ROS 2 installation.

**Expected Outcome**: You can successfully run a basic ROS 2 command and see the expected output, confirming your installation is working.

**Steps**:

1.  Open a new terminal.
2.  Source your ROS 2 installation. If you haven't already added this to your `.bashrc` or `.zshrc`, you'll need to do this every time you open a new terminal.
    ```bash
    source /opt/ros/humble/setup.bash
    ```
3.  Run the `ros2` command to see its options.
    ```bash
    ros2
    ```
    You should see a list of available verbs like `action`, `daemon`, `node`, `run`, etc. This confirms that your system can find the ROS 2 command-line tools.
4.  Run a simple "talker" demo node. This node will start publishing messages.
    ```bash
    ros2 run demo_nodes_py talker
    ```
    You should see output like `[INFO] [167...]: Publishing: "Hello World: 1"`, `[INFO] [167...]: Publishing: "Hello World: 2"`, and so on.
5.  Open a **second terminal**. Make sure to source your ROS 2 setup file in this terminal as well.
6.  In the second terminal, run a "listener" demo node.
    ```bash
    ros2 run demo_nodes_py listener
    ```
    You should see the listener printing the messages it receives from the talker: `[INFO] [167...]: I heard: [Hello World: 1]`, `[INFO] [167...]: I heard: [Hello World: 2]`, etc.
7.  You can stop the nodes by pressing `Ctrl+C` in each terminal.

**Conclusion**: If you saw the talker publishing messages and the listener receiving them, your ROS 2 installation is working correctly! You are now ready to dive into the core concepts.

---

## References
[1] Robot Operating System (ROS) Website: https://www.ros.org/
[2] ROS 2 Documentation: https://docs.ros.org/en/humble/index.html
[3] Data Distribution Service (DDS) Standard: https://www.omg.org/dds/