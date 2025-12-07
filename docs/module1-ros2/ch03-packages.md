---
title: 'Chapter 3: ROS 2 Packages with Python'
---

# Chapter 3: ROS 2 Packages with Python

In the previous chapter, we wrote simple, single-file ROS 2 nodes. While this is great for learning, real-world robotics projects quickly become complex. To manage this complexity, ROS 2 uses a **package** system.

A ROS 2 package is simply a directory with a specific structure and a `package.xml` file that contains metadata about the package. Packages are the fundamental unit for organizing, sharing, and reusing ROS 2 code.

## Why Use Packages?

-   **Organization**: Packages group related nodes, launch files, and configuration files together.
-   **Dependencies**: They explicitly declare their dependencies on other packages.
-   **Build System Integration**: ROS 2's build tools (`colcon`) know how to find, build, and install packages.
-   **Reusability**: Well-defined packages can be easily shared and used across different projects.

## Structure of a Python Package

A minimal Python ROS 2 package has the following structure:

```
my_python_pkg/
├── package.xml
├── setup.py
├── setup.cfg
└── my_python_pkg/
    ├── __init__.py
    └── my_node.py
```

Let's break down each file:

-   **`package.xml`**: This is the most important file. It contains metadata about the package, such as its name, version, author, license, and dependencies. The build system uses this file to figure out how to handle the package.

-   **`setup.py`**: This is a standard Python setup script. It tells the build system how to install the package and where to find its executable nodes.

-   **`setup.cfg`**: This file configures the `setuptools` build process, specifying where to find the executable scripts (our ROS 2 nodes).

-   **`my_python_pkg/` (the inner directory)**: This is the actual Python package that contains your source code. It must have an `__init__.py` file to be recognized as a Python package. Your node files (e.g., `my_node.py`) live inside this directory.

## Creating a Python Package

ROS 2 provides a command-line tool to quickly create a new package with all the necessary boilerplate.

1.  Navigate to the `src` directory of your ROS 2 workspace. A workspace is a directory where you keep your custom packages.
    ```bash
    mkdir -p ros2_ws/src
    cd ros2_ws/src
    ```

2.  Run the `ros2 pkg create` command:
    ```bash
    ros2 pkg create --build-type ament_python --node-name my_first_node my_first_pkg
    ```
    -   `--build-type ament_python`: Specifies that we are creating a Python package.
    -   `--node-name my_first_node`: Creates a sample executable node with this name.
    -   `my_first_pkg`: The name of our new package.

This command will generate a directory named `my_first_pkg` with all the files we discussed above, including a sample "Hello World" style node.

## Building and Running a Package

Once you have created your package, you need to build it using `colcon`, the standard ROS 2 build tool.

1.  Navigate to the root of your workspace (`ros2_ws`).
2.  Run `colcon build`.
    ```bash
    cd ..  # Go up to ros2_ws
    colcon build
    ```
    Colcon will find all the packages in the `src` directory, resolve their dependencies, and build them. It creates `install`, `build`, and `log` directories in your workspace root.

3.  **Source the workspace**: After building, you need to source the workspace's setup file. This tells ROS 2 where to find the executables and resources from your new package.
    ```bash
    source install/setup.bash
    ```
    **Important**: You must do this in every new terminal where you want to use your custom package.

4.  **Run your node**: Now you can run the node from your package using `ros2 run`.
    ```bash
    ros2 run my_first_pkg my_first_node
    ```
    This command tells ROS 2 to look for a package named `my_first_pkg` and execute the node named `my_first_node`.

## The `package.xml` File in Detail

This file is crucial. Here's a minimal example:

```xml
<?xml version="1.0"?>
<package format="3">
  <name>my_first_pkg</name>
  <version>0.0.0</version>
  <description>A simple example package.</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

-   `<name>`: The official name of the package.
-   `<version>`, `<description>`, `<maintainer>`, `<license>`: Standard metadata.
-   `<depend>`: This is where you declare your package's dependencies. If your Python code imports `rclpy` and `std_msgs`, you must list them here. This is how the build system ensures all necessary libraries are available.
-   `<export>`: This section tells the build system what kind of package this is.

---

### Lab 3.1: Creating and Running a Custom Package

**Problem Statement**: Create a ROS 2 package that contains the simple publisher and subscriber nodes from the previous chapter's code examples.

**Expected Outcome**: You will have a custom package that you can build with `colcon`, and you can use `ros2 run` to start your publisher and subscriber nodes.

**Steps**:

1.  **Create a workspace**:
    ```bash
    mkdir -p ros2_ws/src
    cd ros2_ws/src
    ```

2.  **Create a new package**: Let's call it `chatter_pkg`.
    ```bash
    ros2 pkg create --build-type ament_python chatter_pkg
    ```
    This will create the directory `chatter_pkg` with the basic file structure.

3.  **Create the node files**:
    -   Inside the `chatter_pkg/chatter_pkg` directory, create two new files: `publisher_node.py` and `subscriber_node.py`.
    -   Copy the code from `publisher.py` (the example from `src/code-examples/module1`) into `publisher_node.py`.
    -   Copy the code from `subscriber.py` into `subscriber_node.py`.

4.  **Edit `package.xml`**:
    -   Open `chatter_pkg/package.xml`.
    -   Add dependencies for `rclpy` and `std_msgs` since our nodes use them.
      ```xml
      <depend>rclpy</depend>
      <depend>std_msgs</depend>
      ```

5.  **Edit `setup.py`**:
    -   Open `chatter_pkg/setup.py`.
    -   We need to tell the build system about our two new executables. Modify the `entry_points` section to look like this:
      ```python
      'console_scripts': [
          'my_publisher = chatter_pkg.publisher_node:main',
          'my_subscriber = chatter_pkg.subscriber_node:main',
      ],
      ```
      This tells `colcon` that when the package is installed, it should create two executable scripts: `my_publisher` (which runs the `main` function from `publisher_node.py`) and `my_subscriber`.

6.  **Build the package**:
    -   Navigate to the root of your workspace (`ros2_ws`).
    -   Run `colcon build`.
      ```bash
      cd ../..  # Go back to ros2_ws
      colcon build
      ```

7.  **Run your nodes**:
    -   Open a new terminal. Source the main ROS 2 setup file AND your workspace's setup file.
      ```bash
      source /opt/ros/humble/setup.bash
      source install/setup.bash
      ```
    -   Run your publisher node.
      ```bash
      ros2 run chatter_pkg my_publisher
      ```
    -   Open a second terminal and do the same sourcing.
      ```bash
      source /opt/ros/humble/setup.bash
      source install/setup.bash
      ```
    -   Run your subscriber node.
      ```bash
      ros2 run chatter_pkg my_subscriber
      ```
      You should see the same publisher/subscriber communication as before, but now you are running the nodes from your very own ROS 2 package!

**Conclusion**: You have successfully organized standalone nodes into a proper ROS 2 package, making your code more modular, reusable, and easier to build and distribute. This is a critical skill for any real-world robotics project.

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