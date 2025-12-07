/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Module 1: ROS 2',
      items: [
        'module1-ros2/ch01-intro-to-ros2',
        'module1-ros2/ch02-concepts',
        'module1-ros2/ch03-packages',
        'module1-ros2/ch04-urdf-launch-files',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin',
      items: [
        'module2-digital-twin/ch05-robot-description',
        'module2-digital-twin/ch06-gazebo-physics',
        'module2-digital-twin/ch07-unity-viz',
        'module2-digital-twin/ch08-integration',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: NVIDIA Isaac',
      items: [
        'module3-nvidia-isaac/ch09-isaac-sim-fundamentals',
        'module3-nvidia-isaac/ch10-isaac-ros',
        'module3-nvidia-isaac/ch11-navigation',
        'module3-nvidia-isaac/ch12-rl-sim-to-real',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: VLA Systems',
      items: [
        'module4-vla-systems/ch13-whisper',
        'module4-vla-systems/ch14-llm-planning',
        'module4-vla-systems/ch15-plans-to-actions',
        'module4-vla-systems/ch16-capstone',
      ],
    },
  ],
};

module.exports = sidebars;
