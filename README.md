# Physical AI & Humanoid Robotics Course

[![Course](https://img.shields.io/badge/Course-Physical%20AI%20%26%20Humanoid%20Robotics-blue)](https://github.com/your-organization/physical-ai-humanoid-robotics)
[![ROS 2](https://img.shields.io/badge/ROS%202-Humble%20Hawksbill-green)](https://docs.ros.org/en/humble/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

Welcome to the **Physical AI & Humanoid Robotics** course - a comprehensive educational program covering the integration of artificial intelligence with physical robotic systems. This course provides hands-on experience with the four core technologies that define modern robotics:

1. **The Robotic Nervous System (ROS 2)** - Communication and coordination
2. **The Digital Twin (Gazebo & Unity)** - Simulation and visualization
3. **The AI-Robot Brain (NVIDIA Isaac)** - Perception and control
4. **Vision-Language-Action (VLA)** - Intelligent interaction

## Course Structure

The course is organized into four comprehensive modules:

### Module 1: The Robotic Nervous System (ROS 2)
- ROS 2 architecture and communication patterns
- Node development and package management
- URDF and robot modeling
- Launch files and system integration

### Module 2: The Digital Twin (Gazebo & Unity)
- Physics simulation and realistic environments
- Sensor modeling and data generation
- Unity integration for advanced visualization
- System integration and deployment

### Module 3: The AI-Robot Brain (NVIDIA Isaac)
- Perception systems and computer vision
- Path planning and navigation
- Reinforcement learning for robotics
- Sim-to-real transfer techniques

### Module 4: Vision-Language-Action (VLA)
- Large language models for robotic planning
- Vision-language integration
- Action grounding and execution
- Complete system integration

## Prerequisites

- Basic programming experience (Python)
- Understanding of fundamental robotics concepts
- Linux command line familiarity
- Basic mathematics (linear algebra, calculus)

## Technical Requirements

### Hardware
- Multi-core CPU (Intel i7 / AMD Ryzen 7+)
- 16GB+ RAM (32GB recommended)
- NVIDIA GPU with CUDA support (for Isaac Sim)
- 50GB+ storage space

### Software
- Ubuntu 20.04/22.04 or Windows 10/11 + WSL2
- ROS 2 Humble Hawksbill
- Gazebo Garden/Fortress
- Unity 2021.3 LTS
- NVIDIA Isaac Sim (optional but recommended)

## Getting Started

1. **Clone the repository:**
```bash
git clone https://github.com/your-organization/physical-ai-humanoid-robotics.git
cd physical-ai-humanoid-robotics
```

2. **Install ROS 2 Humble:**
```bash
# Follow the official ROS 2 installation guide:
# https://docs.ros.org/en/humble/Installation.html
```

3. **Set up your workspace:**
```bash
mkdir -p ~/physical_ai_ws/src
cd ~/physical_ai_ws
colcon build
source install/setup.bash
```

4. **Follow the course materials:**
Browse the documentation starting with the [Introduction](./docs/intro.md)

## Documentation

Complete course materials are available in the `docs/` directory:

- [Course Introduction](./docs/intro.md)
- [Module 1: ROS 2](./docs/module1-ros2/ch01-intro-to-ros2.md)
- [Module 2: Digital Twin](./docs/module2-digital-twin/ch05-robot-description.md)
- [Module 3: NVIDIA Isaac](./docs/module3-nvidia-isaac/ch09-isaac-sim-fundamentals.md)
- [Module 4: VLA Systems](./docs/module4-vla-systems/ch13-whisper.md)

## Features

- ✅ Comprehensive step-by-step tutorials
- ✅ Real-world examples and applications
- ✅ Integration-focused approach
- ✅ Modern AI and robotics technologies
- ✅ Hands-on practical exercises
- ✅ Complete system architecture

## Contributing

We welcome contributions to improve this educational resource. Please see our contribution guidelines for more information.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Deployment

This documentation site is built with [Docusaurus 3](https://docusaurus.io/) and can be deployed to GitHub Pages.

### Local Development

1. **Install dependencies:**
```bash
npm install
```

2. **Start development server:**
```bash
npm start
```

3. **Build for production:**
```bash
npm run build
```

### GitHub Pages Deployment

The site is configured for automatic deployment via GitHub Actions:
- Push changes to the `main` branch
- The GitHub Actions workflow in `.github/workflows/deploy.yml` will automatically build and deploy the site
- The site will be available at `https://your-organization.github.io/physical-ai-humanoid-robotics/`

### Configuration Notes

- The `baseUrl` in `docusaurus.config.js` is set to `/physical-ai-humanoid-robotics/` for GitHub Pages
- Update `organizationName` and `projectName` in `docusaurus.config.js` to match your repository
- Customize the navbar and footer links in `docusaurus.config.js` as needed

## Support

For questions and support, please open an issue in this repository or join our community discussions.

---

*Building the next generation of intelligent robotic systems, one lesson at a time.*