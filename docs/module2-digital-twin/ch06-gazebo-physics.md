---
sidebar_position: 2
---

# Chapter 6: Advanced Gazebo Physics and Simulation

## Understanding Gazebo Physics Engine

Gazebo uses ODE (Open Dynamics Engine) as its default physics engine, though it also supports Bullet and DART. The physics engine is responsible for simulating realistic interactions between objects, including collisions, friction, and dynamics.

### Physics Configuration in SDF

The physics engine can be configured in your world file to match real-world conditions:

```xml
<!-- simple_robot_pkg/worlds/physics_world.sdf -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="physics_world">
    <!-- Physics engine configuration -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.000001</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Include a ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include a sky -->
    <include>
      <uri>model://sky</uri>
    </include>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.6 0.4 -0.8</direction>
    </light>

    <!-- Physics test models -->
    <model name="physics_box">
      <pose>0 0 1 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.166667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.166667</iyy>
            <iyz>0</iyz>
            <izz>0.166667</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.2 0.2 0.2</size>
            </box>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>1.0</mu>
                <mu2>1.0</mu2>
                <fdir1>0 0 0</fdir1>
                <slip1>0</slip1>
                <slip2>0</slip2>
              </ode>
            </friction>
            <bounce>
              <restitution_coefficient>0.1</restitution_coefficient>
              <threshold>100000</threshold>
            </bounce>
            <contact>
              <ode>
                <soft_cfm>0</soft_cfm>
                <soft_erp>0.2</soft_erp>
                <kp>1000000000000</kp>
                <kd>1</kd>
                <max_vel>0.01</max_vel>
                <min_depth>0</min_depth>
              </ode>
            </contact>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.2 0.2 0.2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.3 0.3 1</ambient>
            <diffuse>1 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Sphere model -->
    <model name="physics_sphere">
      <pose>1 0 1 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>0.5</mass>
          <inertia>
            <ixx>0.01</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.01</iyy>
            <iyz>0</iyz>
            <izz>0.01</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <sphere>
              <radius>0.1</radius>
            </sphere>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.5</mu>
                <mu2>0.5</mu2>
              </ode>
            </friction>
            <bounce>
              <restitution_coefficient>0.8</restitution_coefficient>
              <threshold>100000</threshold>
            </bounce>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <sphere>
              <radius>0.1</radius>
            </sphere>
          </geometry>
          <material>
            <ambient>0.3 0.8 0.3 1</ambient>
            <diffuse>0.5 1 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Cylinder model -->
    <model name="physics_cylinder">
      <pose>-1 0 1 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>0.7</mass>
          <inertia>
            <ixx>0.005833</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.005833</iyy>
            <iyz>0</iyz>
            <izz>0.00175</izz>
          </inertia>
        </inertial>
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.08</radius>
              <length>0.2</length>
            </cylinder>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.8</mu>
                <mu2>0.8</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.08</radius>
              <length>0.2</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.3 0.3 0.8 1</ambient>
            <diffuse>0.5 0.5 1 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
```

## Advanced Robot Control with Physics

Let's create a more sophisticated controller that takes physics into account:

```python
# simple_robot_pkg/physics_controller.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Wrench
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty
import math
import numpy as np

class PhysicsController(Node):
    def __init__(self):
        super().__init__('physics_controller')

        # Publishers for Gazebo
        self.left_wheel_pub = self.create_publisher(Float64MultiArray, '/simple_bot/left_wheel_controller/commands', 10)
        self.right_wheel_pub = self.create_publisher(Float64MultiArray, '/simple_bot/right_wheel_controller/commands', 10)

        # Subscribers
        self.cmd_vel_sub = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100 Hz

        # Robot parameters
        self.wheel_radius = 0.1  # meters
        self.wheel_separation = 0.3  # meters
        self.max_wheel_velocity = 5.0  # rad/s
        self.max_wheel_torque = 10.0  # Nm
        self.robot_mass = 2.0  # kg
        self.robot_inertia = 0.2  # kg*m^2

        # Control variables
        self.desired_linear_vel = 0.0
        self.desired_angular_vel = 0.0

        # Joint state feedback
        self.left_wheel_pos = 0.0
        self.right_wheel_pos = 0.0
        self.left_wheel_vel = 0.0
        self.right_wheel_vel = 0.0

        # IMU feedback
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # PID controllers
        self.linear_pid = PIDController(kp=2.0, ki=0.1, kd=0.05)
        self.angular_pid = PIDController(kp=3.0, ki=0.1, kd=0.08)

        # Safety limits
        self.max_tilt_angle = 0.3  # radians (about 17 degrees)

        self.get_logger().info('Physics Controller initialized')

    def cmd_vel_callback(self, msg):
        self.desired_linear_vel = msg.linear.x
        self.desired_angular_vel = msg.angular.z

    def joint_state_callback(self, msg):
        try:
            left_idx = msg.name.index('left_wheel_joint')
            right_idx = msg.name.index('right_wheel_joint')

            self.left_wheel_pos = msg.position[left_idx]
            self.right_wheel_pos = msg.position[right_idx]
            self.left_wheel_vel = msg.velocity[left_idx]
            self.right_wheel_vel = msg.velocity[right_idx]
        except ValueError:
            pass  # Joint names not found

    def imu_callback(self, msg):
        # Convert quaternion to Euler angles
        quat = msg.orientation
        self.roll, self.pitch, self.yaw = self.quaternion_to_euler(
            quat.x, quat.y, quat.z, quat.w)

    def quaternion_to_euler(self, x, y, z, w):
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def control_loop(self):
        # Check safety conditions
        if abs(self.roll) > self.max_tilt_angle or abs(self.pitch) > self.max_tilt_angle:
            self.get_logger().warn('Robot tilt exceeded safety limits, stopping')
            self.publish_wheel_commands(0.0, 0.0)
            return

        # Calculate current linear and angular velocities from wheel velocities
        current_linear_vel = (self.left_wheel_vel + self.right_wheel_vel) * self.wheel_radius / 2.0
        current_angular_vel = (self.right_wheel_vel - self.left_wheel_vel) * self.wheel_radius / self.wheel_separation

        # Calculate control efforts using PID
        linear_error = self.desired_linear_vel - current_linear_vel
        angular_error = self.desired_angular_vel - current_angular_vel

        linear_effort = self.linear_pid.update(linear_error, 1.0/100.0)  # dt = 0.01s
        angular_effort = self.angular_pid.update(angular_error, 1.0/100.0)

        # Convert to wheel velocities
        left_wheel_vel = (linear_effort - angular_effort * self.wheel_separation / 2.0) / self.wheel_radius
        right_wheel_vel = (linear_effort + angular_effort * self.wheel_separation / 2.0) / self.wheel_radius

        # Apply velocity limits
        left_wheel_vel = max(-self.max_wheel_velocity, min(self.max_wheel_velocity, left_wheel_vel))
        right_wheel_vel = max(-self.max_wheel_velocity, min(self.max_wheel_velocity, right_wheel_vel))

        # Publish wheel commands
        self.publish_wheel_commands(left_wheel_vel, right_wheel_vel)

    def publish_wheel_commands(self, left_vel, right_vel):
        left_cmd = Float64MultiArray()
        left_cmd.data = [left_vel]
        self.left_wheel_pub.publish(left_cmd)

        right_cmd = Float64MultiArray()
        right_cmd.data = [right_vel]
        self.right_wheel_pub.publish(right_cmd)

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        if dt <= 0:
            return 0.0

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        return output

def main(args=None):
    rclpy.init(args=args)
    controller = PhysicsController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Creating Advanced Gazebo Plugins

Let's create a more advanced Gazebo plugin for IMU simulation:

```cpp
// simple_robot_pkg/gazebo_plugins/imu_plugin.cpp
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/sensors/sensors.hh>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <random>

namespace gazebo
{
  class ImuPlugin : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      this->model = _model;
      this->world = _model->GetWorld();

      // Get the IMU link
      if (_sdf->HasElement("imu_link"))
      {
        std::string imu_link_name = _sdf->Get<std::string>("imu_link");
        this->imu_link = _model->GetLink(imu_link_name);
      }
      else
      {
        this->imu_link = _model->GetLink();
      }

      // Initialize ROS 2
      if (!rclcpp::ok())
      {
        int argc = 0;
        char **argv = NULL;
        rclcpp::init(argc, argv);
      }

      this->node = std::make_shared<rclcpp::Node>("gazebo_imu");

      // Create publisher for IMU
      this->pub = this->node->create_publisher<sensor_msgs::msg::Imu>("/imu", 10);

      // Get update rate from SDF (default 100 Hz)
      double update_rate = _sdf->Get<double>("update_rate", 100.0);
      this->update_period_ = 1.0 / update_rate;

      // Initialize random number generator for noise
      std::random_device rd;
      this->rng.seed(rd());
      this->gaussian_noise = std::normal_distribution<double>(0.0, 0.01); // 1% noise

      // Listen to the update event
      this->update_connection_ = event::Events::ConnectWorldUpdateBegin(
          std::bind(&ImuPlugin::OnUpdate, this));

      gzdbg << "IMU Plugin loaded\n";
    }

    public: void OnUpdate()
    {
      static double last_update_time = 0;
      double current_time = this->world->SimTime().Double();

      // Rate limiting
      if (current_time - last_update_time < this->update_period_)
        return;

      last_update_time = current_time;

      // Get pose and velocity from the IMU link
      ignition::math::Pose3d pose = this->imu_link->WorldPose();
      ignition::math::Vector3d linear_vel = this->imu_link->WorldLinearVel();
      ignition::math::Vector3d angular_vel = this->imu_link->WorldAngularVel();
      ignition::math::Vector3d linear_acc = this->imu_link->WorldLinearAccel();

      // Create IMU message
      auto imu_msg = std::make_shared<sensor_msgs::msg::Imu>();
      imu_msg->header.stamp = this->node->get_clock()->now();
      imu_msg->header.frame_id = "imu_link";

      // Set orientation (for now, we'll use the pose orientation)
      imu_msg->orientation.x = pose.Rot().X();
      imu_msg->orientation.y = pose.Rot().Y();
      imu_msg->orientation.z = pose.Rot().Z();
      imu_msg->orientation.w = pose.Rot().W();

      // Add noise to orientation
      imu_msg->orientation.x += this->gaussian_noise(this->rng);
      imu_msg->orientation.y += this->gaussian_noise(this->rng);
      imu_msg->orientation.z += this->gaussian_noise(this->rng);
      imu_msg->orientation.w += this->gaussian_noise(this->rng);

      // Set angular velocity
      imu_msg->angular_velocity.x = angular_vel.X();
      imu_msg->angular_velocity.y = angular_vel.Y();
      imu_msg->angular_velocity.z = angular_vel.Z();

      // Add noise to angular velocity
      imu_msg->angular_velocity.x += this->gaussian_noise(this->rng);
      imu_msg->angular_velocity.y += this->gaussian_noise(this->rng);
      imu_msg->angular_velocity.z += this->gaussian_noise(this->rng);

      // Set linear acceleration
      imu_msg->linear_acceleration.x = linear_acc.X();
      imu_msg->linear_acceleration.y = linear_acc.Y();
      imu_msg->linear_acceleration.z = linear_acc.Z();

      // Add noise to linear acceleration
      imu_msg->linear_acceleration.x += this->gaussian_noise(this->rng);
      imu_msg->linear_acceleration.y += this->gaussian_noise(this->rng);
      imu_msg->linear_acceleration.z += this->gaussian_noise(this->rng);

      // Set covariance matrices (information about uncertainty)
      for (int i = 0; i < 9; ++i)
      {
        imu_msg->orientation_covariance[i] = 0.01;
        imu_msg->angular_velocity_covariance[i] = 0.01;
        imu_msg->linear_acceleration_covariance[i] = 0.01;
      }

      // Publish the IMU message
      this->pub->publish(*imu_msg);
    }

    private: physics::ModelPtr model;
    private: physics::WorldPtr world;
    private: physics::LinkPtr imu_link;
    private: rclcpp::Node::SharedPtr node;
    private: rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub;
    private: event::ConnectionPtr update_connection_;
    private: double update_period_;
    private: std::mt19937 rng;
    private: std::normal_distribution<double> gaussian_noise;
  };

  GZ_REGISTER_MODEL_PLUGIN(ImuPlugin)
}
```

## Advanced Sensor Simulation

Let's create a more realistic camera sensor plugin:

```cpp
// simple_robot_pkg/gazebo_plugins/camera_plugin.cpp
#include <gazebo/common/Plugin.hh>
#include <gazebo/sensors/sensors.hh>
#include <gazebo/rendering/Camera.hh>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace gazebo
{
  class CameraPlugin : public SensorPlugin
  {
    public: CameraPlugin() {}

    public: virtual void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf)
    {
      // Get the camera sensor
      this->camera_sensor_ = std::dynamic_pointer_cast<sensors::CameraSensor>(_sensor);
      if (!this->camera_sensor_)
      {
        gzerr << "CameraPlugin requires a CameraSensor\n";
        return;
      }

      // Initialize ROS 2
      if (!rclcpp::ok())
      {
        int argc = 0;
        char **argv = NULL;
        rclcpp::init(argc, argv);
      }

      this->node = std::make_shared<rclcpp::Node>("gazebo_camera");

      // Create publisher for camera images
      this->image_pub = this->node->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", 10);
      this->info_pub = this->node->create_publisher<sensor_msgs::msg::CameraInfo>("/camera/camera_info", 10);

      // Connect to camera update event
      this->new_frame_connection_ = this->camera_sensor_->Camera()->ConnectNewFrame(
          std::bind(&CameraPlugin::OnNewFrame, this,
                    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                    std::placeholders::_4, std::placeholders::_5));

      gzdbg << "Camera Plugin loaded\n";
    }

    private: void OnNewFrame(const unsigned char *_image,
                             unsigned int _width, unsigned int _height,
                             unsigned int _depth, const std::string &_format)
    {
      // Create OpenCV Mat from the image data
      cv::Mat cv_image;
      if (_depth == 3 && _format == "R8G8B8")
      {
        cv_image = cv::Mat(_height, _width, CV_8UC3, (void*)_image);
        cv::cvtColor(cv_image, cv_image, cv::COLOR_RGB2BGR);
      }
      else if (_depth == 1 && _format == "L8")
      {
        cv_image = cv::Mat(_height, _width, CV_8UC1, (void*)_image);
      }
      else
      {
        // Unsupported format
        return;
      }

      // Apply some image processing to simulate real camera effects
      // Add noise
      cv::Mat noise = cv::Mat(cv_image.size(), cv_image.type());
      cv::randu(noise, cv::Scalar(0), cv::Scalar(10));
      cv_image += noise;

      // Convert to ROS 2 image message
      cv_bridge::CvImage cv_bridge_msg;
      cv_bridge_msg.header.stamp = this->node->get_clock()->now();
      cv_bridge_msg.header.frame_id = "camera_link";
      cv_bridge_msg.encoding = sensor_msgs::image_encodings::BGR8;
      cv_bridge_msg.image = cv_image;

      // Publish image
      this->image_pub->publish(*cv_bridge_msg.toImageMsg());

      // Publish camera info
      auto info_msg = std::make_shared<sensor_msgs::msg::CameraInfo>();
      info_msg->header = cv_bridge_msg.header;
      info_msg->width = _width;
      info_msg->height = _height;

      // Set camera matrix (example values)
      info_msg->k[0] = 500.0; // fx
      info_msg->k[2] = _width / 2.0; // cx
      info_msg->k[4] = 500.0; // fy
      info_msg->k[5] = _height / 2.0; // cy
      info_msg->k[8] = 1.0;

      this->info_pub->publish(*info_msg);
    }

    private: sensors::CameraSensorPtr camera_sensor_;
    private: rclcpp::Node::SharedPtr node;
    private: rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub;
    private: rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub;
    private: event::ConnectionPtr new_frame_connection_;
  };

  GZ_REGISTER_SENSOR_PLUGIN(CameraPlugin)
}
```

## Advanced Launch File with Physics and Sensors

Now let's create a comprehensive launch file that includes all our advanced components:

```python
# simple_robot_pkg/launch/advanced_gazebo.launch.py
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    world = LaunchConfiguration('world', default='physics_world.sdf')
    robot_name = LaunchConfiguration('robot_name', default='simple_bot')

    # Package names
    pkg_gazebo_ros = FindPackageShare('gazebo_ros').find('gazebo_ros')
    pkg_simple_robot = FindPackageShare('simple_robot_pkg').find('simple_robot_pkg')

    # Paths
    urdf_path = os.path.join(pkg_simple_robot, 'urdf', 'simple_bot_complete.urdf')
    world_path = os.path.join(pkg_simple_robot, 'worlds', 'physics_world.sdf')
    rviz_config_path = os.path.join(pkg_simple_robot, 'rviz', 'simple_bot.rviz')

    # Create launch description
    ld = LaunchDescription()

    # Declare launch arguments
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'))

    ld.add_action(DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to start RViz'))

    ld.add_action(DeclareLaunchArgument(
        'world',
        default_value='physics_world.sdf',
        description='Choose one of the world files from `/simple_robot_pkg/worlds`'))

    ld.add_action(DeclareLaunchArgument(
        'robot_name',
        default_value='simple_bot',
        description='Name of the robot to spawn'))

    # Start Gazebo server with physics world
    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={
            'world': world_path,
            'verbose': 'true'
        }.items(),
    )

    # Start Gazebo client
    gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        ),
    )

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': open(urdf_path).read()
        }])

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', [robot_name],
            '-file', urdf_path,
            '-x', '0', '-y', '0', '-z', '0.2',
            '-robot_namespace', [robot_name]
        ],
        output='screen')

    # Physics Controller
    physics_controller = Node(
        package='simple_robot_pkg',
        executable='physics_controller',
        name='physics_controller',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    # IMU Sensor (if using custom plugin)
    imu_node = Node(
        package='simple_robot_pkg',
        executable='imu_sensor_node',
        name='imu_sensor_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    # Camera Sensor (if using custom plugin)
    camera_node = Node(
        package='simple_robot_pkg',
        executable='camera_sensor_node',
        name='camera_sensor_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    # Navigation stack with physics awareness
    navigation_node = Node(
        package='simple_robot_pkg',
        executable='physics_aware_navigation',
        name='physics_aware_navigation',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    # RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    # Add all actions to the launch description
    ld.add_action(gazebo_server)
    ld.add_action(gazebo_client)
    ld.add_action(robot_state_publisher)
    ld.add_action(spawn_entity)
    ld.add_action(physics_controller)
    ld.add_action(imu_node)
    ld.add_action(camera_node)
    ld.add_action(navigation_node)

    # Conditionally add RViz
    ld.add_action(rviz)

    return ld
```

## Physics-Based Navigation

Let's create a physics-aware navigation node:

```python
# simple_robot_pkg/physics_aware_navigation.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

class PhysicsAwareNavigation(Node):
    def __init__(self):
        super().__init__('physics_aware_navigation')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(Imu, 'imu', self.imu_callback, 10)

        # TF listener for transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Navigation parameters
        self.current_pose = Point()
        self.current_orientation = [0, 0, 0, 1]  # x, y, z, w
        self.target_pose = Point()
        self.target_pose.x = 3.0
        self.target_pose.y = 3.0

        # Robot state
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # Navigation state
        self.navigation_active = True
        self.avoiding_obstacle = False
        self.safety_stop = False

        # Physics parameters
        self.max_linear_vel = 0.5  # m/s
        self.max_angular_vel = 1.0  # rad/s
        self.max_tilt_angle = 0.4  # radians (about 23 degrees)
        self.min_obstacle_distance = 0.5  # meters

        # PID controllers
        self.linear_pid = PIDController(kp=1.5, ki=0.05, kd=0.1)
        self.angular_pid = PIDController(kp=2.0, ki=0.05, kd=0.15)

        # Timer for navigation
        self.nav_timer = self.create_timer(0.05, self.navigate)  # 20 Hz

        self.get_logger().info(f'Starting physics-aware navigation to ({self.target_pose.x}, {self.target_pose.y})')

    def odom_callback(self, msg):
        # Update current pose
        self.current_pose.x = msg.pose.pose.position.x
        self.current_pose.y = msg.pose.pose.position.y
        self.current_pose.z = msg.pose.pose.position.z

        # Update orientation
        self.current_orientation[0] = msg.pose.pose.orientation.x
        self.current_orientation[1] = msg.pose.pose.orientation.y
        self.current_orientation[2] = msg.pose.pose.orientation.z
        self.current_orientation[3] = msg.pose.pose.orientation.w

        # Update velocities
        self.current_linear_vel = msg.twist.twist.linear.x
        self.current_angular_vel = msg.twist.twist.angular.z

    def scan_callback(self, msg):
        # Check for obstacles in front of the robot
        if len(msg.ranges) > 0:
            # Get distances in front (±30 degrees)
            front_start = int(len(msg.ranges) * 0.45)
            front_end = int(len(msg.ranges) * 0.55)
            front_distances = msg.ranges[front_start:front_end]

            # Remove invalid ranges
            valid_distances = [d for d in front_distances if d >= msg.range_min and d <= msg.range_max]

            if valid_distances:
                min_front_distance = min(valid_distances)
                self.avoiding_obstacle = min_front_distance < self.min_obstacle_distance

    def imu_callback(self, msg):
        # Convert quaternion to Euler angles
        x, y, z, w = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w

        # Convert to roll, pitch, yaw
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        self.roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            self.pitch = math.copysign(math.pi / 2, sinp)
        else:
            self.pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

        # Check if robot is tilted too much
        self.safety_stop = abs(self.roll) > self.max_tilt_angle or abs(self.pitch) > self.max_tilt_angle

    def navigate(self):
        if self.safety_stop:
            # Emergency stop if robot is tilted too much
            self.publish_velocity(0.0, 0.0)
            self.get_logger().warn('Safety stop: Robot tilt exceeded limits')
            return

        if not self.navigation_active:
            self.publish_velocity(0.0, 0.0)
            return

        # Calculate distance to goal
        dx = self.target_pose.x - self.current_pose.x
        dy = self.target_pose.y - self.current_pose.y
        distance_to_goal = math.sqrt(dx*dx + dy*dy)

        # Check if we reached the goal
        if distance_to_goal < 0.2:
            self.navigation_active = False
            self.publish_velocity(0.0, 0.0)
            self.get_logger().info('Goal reached!')
            return

        # Calculate desired heading
        desired_yaw = math.atan2(dy, dx)

        # Get current yaw from orientation
        current_yaw = self.yaw  # Updated from IMU

        # Calculate heading error
        heading_error = desired_yaw - current_yaw
        # Normalize angle to [-π, π]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        # If avoiding obstacle, do obstacle avoidance instead of going to goal
        if self.avoiding_obstacle:
            self.perform_obstacle_avoidance()
        else:
            # Calculate control efforts
            linear_effort = self.linear_pid.update(distance_to_goal, 0.05)  # dt = 0.05s
            angular_effort = self.angular_pid.update(heading_error, 0.05)

            # Apply limits
            linear_effort = max(0.0, min(self.max_linear_vel, linear_effort))  # Only forward
            angular_effort = max(-self.max_angular_vel, min(self.max_angular_vel, angular_effort))

            # Publish velocity command
            self.publish_velocity(linear_effort, angular_effort)

    def perform_obstacle_avoidance(self):
        # Simple obstacle avoidance: turn away from obstacles
        self.get_logger().info('Obstacle detected, performing avoidance maneuver')

        # For now, just turn right while slowing down
        linear_vel = max(0.0, self.max_linear_vel * 0.3)  # Reduce speed
        angular_vel = -self.max_angular_vel * 0.5  # Turn right

        self.publish_velocity(linear_vel, angular_vel)

    def publish_velocity(self, linear, angular):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_vel_pub.publish(twist)

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        if dt <= 0:
            return 0.0

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        return output

def main(args=None):
    rclpy.init(args=args)
    navigator = PhysicsAwareNavigation()

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot on shutdown
        stop_msg = Twist()
        navigator.cmd_vel_pub.publish(stop_msg)
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Running the Advanced Physics Simulation

To run the complete physics-aware simulation:

```bash
cd ~/ros2_ws
colcon build --packages-select simple_robot_pkg
source install/setup.bash

# Run the advanced simulation
ros2 launch simple_robot_pkg advanced_gazebo.launch.py

# In another terminal, you can monitor the robot's state:
ros2 topic echo /imu
ros2 topic echo /scan
ros2 topic echo /odom
```

## Physics Validation and Tuning

To validate that your simulation matches real-world physics, you should:

1. **Compare kinematic behavior**: Verify that your robot moves at the expected speeds
2. **Validate sensor data**: Ensure simulated sensors produce realistic values
3. **Test dynamics**: Check that mass, friction, and other physical properties behave correctly
4. **Calibrate parameters**: Adjust physics parameters to match real robot behavior

## Next Steps

In the next chapter, we'll explore Unity integration in more detail, creating advanced visualization and interaction systems that can work alongside our physics simulation. We'll also look at how to synchronize Unity visualization with real-time physics simulation data.