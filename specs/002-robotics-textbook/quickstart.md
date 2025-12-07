# Quickstart: Environment Setup

This guide provides the steps to set up a development environment capable of running all code examples in the textbook. The primary target is an **NVIDIA Jetson Orin** developer kit, but many examples can be run on a standard Linux machine with an NVIDIA GPU.

## Core Requirements

- **Host Machine**: Ubuntu 22.04 LTS (Jammy Jellyfish).
- **GPU**: NVIDIA GPU with CUDA 11.8+ support.
- **Storage**: At least 100GB of free space.
- **Memory**: At least 16GB RAM.

## Step 1: Install NVIDIA JetPack (Jetson Orin Users)

If you are using an NVIDIA Jetson Orin, your primary step is to flash it with the correct JetPack version.

1.  **Download NVIDIA SDK Manager**.
2.  Log in and select the Jetson Orin device.
3.  Select **JetPack 5.1.2** as the target operating system.
4.  Follow the on-screen instructions to flash the device. This will install Ubuntu 22.04, CUDA 11.8, and other necessary drivers.

## Step 2: Install ROS 2 Humble Hawksbill

Once your Ubuntu 22.04 environment is ready, install ROS 2 Humble.

1.  **Set Locale**:
    ```bash
    sudo apt update && sudo apt install locales
    sudo locale-gen en_US en_US.UTF-8
    sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
    export LANG=en_US.UTF-8
    ```

2.  **Add ROS 2 APT Repository**:
    ```bash
    sudo apt install software-properties-common
    sudo add-apt-repository universe
    sudo apt update && sudo apt install curl
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    ```

3.  **Install ROS 2 Packages**:
    ```bash
    sudo apt update
    sudo apt install ros-humble-desktop
    ```

4.  **Source the Setup File**:
    ```bash
    # Add this to your ~/.bashrc
    source /opt/ros/humble/setup.bash
    ```

## Step 3: Install Gazebo Simulator

Install Gazebo Classic (v11) and the ROS 2 integration plugins.

```bash
sudo apt install gazebo
sudo apt install ros-humble-gazebo-ros-pkgs
```

## Step 4: Install NVIDIA Isaac Sim

Isaac Sim is a separate, powerful simulator that will be used in Modules 3 and 4.

1.  **Download and Install Isaac Sim**: Follow the official NVIDIA documentation to download and install Isaac Sim version `2023.1.1`. It requires the Omniverse Launcher.
2.  **Install Isaac ROS Dev Environment**: Follow the setup scripts provided with Isaac ROS `2.0.0` to create a Docker container or local workspace with all necessary ROS 2 packages for Isaac integration.

## Step 5: Install Project-Specific Tools

These are tools needed for the VLA module and for building the textbook itself.

1.  **Install Docusaurus (for contributing to the textbook)**:
    ```bash
    npm install --global yarn
    # From the root of this repository:
    yarn install
    ```

2.  **Install Python Dependencies**:
    ```bash
    # For Whisper/LLM API access
    pip install openai
    ```

## Verification

After completing these steps, you can verify your installation:

1.  **Check ROS 2**:
    ```bash
    # Should print "hello world"
    ros2 run demo_nodes_py talker
    ```
2.  **Check Gazebo**:
    ```bash
    # Should launch the Gazebo GUI
    gazebo --verbose
    ```
3.  **Check Isaac Sim**: Launch Isaac Sim from the Omniverse Launcher to ensure it starts correctly.
