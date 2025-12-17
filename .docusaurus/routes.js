import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '053'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', 'bce'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', '787'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', '66c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '3b2'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '9d5'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', 'ca8'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', '101'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '60f'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '2d9'),
            routes: [
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', 'aed'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module1-ros2/ch01-intro-to-ros2',
                component: ComponentCreator('/docs/module1-ros2/ch01-intro-to-ros2', 'c67'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module1-ros2/ch02-concepts',
                component: ComponentCreator('/docs/module1-ros2/ch02-concepts', 'a9a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module1-ros2/ch03-packages',
                component: ComponentCreator('/docs/module1-ros2/ch03-packages', '769'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module1-ros2/ch04-urdf-launch-files',
                component: ComponentCreator('/docs/module1-ros2/ch04-urdf-launch-files', '46b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module2-digital-twin/ch05-robot-description',
                component: ComponentCreator('/docs/module2-digital-twin/ch05-robot-description', '42c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module2-digital-twin/ch06-gazebo-physics',
                component: ComponentCreator('/docs/module2-digital-twin/ch06-gazebo-physics', '9fb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module2-digital-twin/ch07-unity-viz',
                component: ComponentCreator('/docs/module2-digital-twin/ch07-unity-viz', '5cb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module2-digital-twin/ch08-integration',
                component: ComponentCreator('/docs/module2-digital-twin/ch08-integration', '1d5'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module3-nvidia-isaac/ch09-isaac-sim-fundamentals',
                component: ComponentCreator('/docs/module3-nvidia-isaac/ch09-isaac-sim-fundamentals', 'aec'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module3-nvidia-isaac/ch10-isaac-ros',
                component: ComponentCreator('/docs/module3-nvidia-isaac/ch10-isaac-ros', '2a0'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module3-nvidia-isaac/ch11-navigation',
                component: ComponentCreator('/docs/module3-nvidia-isaac/ch11-navigation', '7d2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module3-nvidia-isaac/ch12-rl-sim-to-real',
                component: ComponentCreator('/docs/module3-nvidia-isaac/ch12-rl-sim-to-real', 'd36'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module4-vla-systems/ch13-whisper',
                component: ComponentCreator('/docs/module4-vla-systems/ch13-whisper', 'a73'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module4-vla-systems/ch14-llm-planning',
                component: ComponentCreator('/docs/module4-vla-systems/ch14-llm-planning', 'e58'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module4-vla-systems/ch15-plans-to-actions',
                component: ComponentCreator('/docs/module4-vla-systems/ch15-plans-to-actions', '26b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/module4-vla-systems/ch16-capstone',
                component: ComponentCreator('/docs/module4-vla-systems/ch16-capstone', '5a9'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '459'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
