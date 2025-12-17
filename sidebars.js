// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['intro'],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module1-ros2/ch01-intro-to-ros2',
        'module1-ros2/ch02-concepts',
        'module1-ros2/ch03-packages',
        'module1-ros2/ch04-urdf-launch-files'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module2-digital-twin/ch05-robot-description',
        'module2-digital-twin/ch06-gazebo-physics',
        'module2-digital-twin/ch07-unity-viz',
        'module2-digital-twin/ch08-integration'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module3-nvidia-isaac/ch09-isaac-sim-fundamentals',
        'module3-nvidia-isaac/ch10-isaac-ros',
        'module3-nvidia-isaac/ch11-navigation',
        'module3-nvidia-isaac/ch12-rl-sim-to-real'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module4-vla-systems/ch13-whisper',
        'module4-vla-systems/ch14-llm-planning',
        'module4-vla-systems/ch15-plans-to-actions',
        'module4-vla-systems/ch16-capstone'
      ],
      collapsed: false,
    },
  ],
};

module.exports = sidebars;