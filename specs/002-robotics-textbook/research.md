# Research & Decisions for Robotics Textbook

This document outlines the key technical decisions made to resolve ambiguities in the project specification.

## 1. Documentation Platform

- **Decision**: Use **Docusaurus v3**.
- **Rationale**: Docusaurus is a modern static site generator built with React, specifically designed for content-driven websites like documentation and textbooks. It supports MDX, allowing for interactive components if needed in the future. Its versioning and sidebar features are ideal for structuring a textbook with modules and chapters.
- **Alternatives Considered**: 
  - **GitBook**: A strong contender, but Docusaurus offers more customization and is open-source.
  - **MkDocs**: Simpler, but less powerful than Docusaurus for a large, structured project.

## 2. Core Robotics & Simulation Stack

- **Decision**: Standardize on **ROS 2 Humble Hawksbill** with **Python 3.10** for all `rclpy` examples.
- **Rationale**: The spec requires ROS 2 Humble. This is a stable LTS release, and Python 3.10 is its primary supported Python version. This ensures long-term stability for students.
- **Alternatives Considered**: Using a newer ROS 2 release like Iron or Jazzy was considered but rejected to adhere to the stability and widespread support of the LTS version specified.

- **Decision**: Use **Gazebo Classic (v11)** for physics simulation.
- **Rationale**: Gazebo 11 is the standard and most compatible version for ROS 2 Humble, with extensive tutorials and community support.
- **Alternatives Considered**: The new Gazebo (formerly Ignition) offers more advanced features but has a steeper learning curve and less integration material available for ROS 2 Humble specifically.

- **Decision**: Use **Unity with the official ROS TCP Connector v0.7.0**.
- **Rationale**: The ROS TCP Connector is the standard, well-supported method for creating a bidirectional communication link between a ROS 2 system and a Unity simulation. It's robust and sufficient for the visualization purposes of this textbook.
- **Alternatives Considered**: A custom WebSocket or ZeroMQ bridge would add unnecessary complexity.

## 3. NVIDIA AI Stack

- **Decision**: Use **NVIDIA Isaac Sim version `2023.1.1`**.
- **Rationale**: This version has documented compatibility with ROS 2 Humble and is a stable, feature-rich release for simulation, and synthetic data generation.
- **Alternatives Considered**: Newer or older versions could introduce compatibility risks with the specified ROS 2 Humble stack.

- **Decision**: Use **NVIDIA Isaac ROS version `2.0.0`**.
- **Rationale**: This version is built for ROS 2 Humble and provides optimized, hardware-accelerated packages for perception, navigation, and manipulation, which is critical for performance on Jetson Orin.
- **Alternatives Considered**: Building from the main branch was rejected due to potential instability.

## 4. VLA (Vision-Language-Action) Stack

- **Decision**: Integrate with the **OpenAI Whisper API** for speech-to-text.
- **Rationale**: The Whisper API provides high-accuracy, robust speech recognition, which is ideal for a learning environment. It simplifies the setup for students, as it doesn't require managing local models.
- **Alternatives Considered**: A local Whisper model is powerful but would add significant setup overhead and computational requirements for students.

- **Decision**: Use a generic **OpenAI GPT-4/GPT-3.5-Turbo API** for the LLM-based cognitive planning module.
- **Rationale**: Using a standard, powerful, and well-documented API like OpenAI's allows the textbook to focus on the *logic* of converting language to actions, rather than the complexities of hosting and managing a large language model.
- **Alternatives Considered**: Local LLMs (e.g., Llama 3) are excellent but, like local Whisper, would complicate the learning environment for students new to the field.

## 5. Target Hardware

- **Decision**: All code examples will be validated for **NVIDIA Jetson Orin** with **JetPack 5.1.2**.
- **Rationale**: JetPack 5.1.2 provides the necessary drivers and CUDA environment to support the chosen Isaac and ROS versions. Providing a specific target configuration is crucial for reproducibility.
- **Alternatives Considered**: Using a more recent JetPack was considered, but 5.1.2 has proven stability with the rest of the selected software stack.
