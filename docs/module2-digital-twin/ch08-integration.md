---
sidebar_position: 4
---

# Chapter 8: Integration and Deployment of Digital Twin Systems

## System Integration Architecture

In this chapter, we'll explore how to integrate all the components we've developed into a cohesive digital twin system. The architecture involves multiple layers working together to create a comprehensive simulation and visualization environment.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Unity Visualization Layer            │
├─────────────────────────────────────────────────────────┤
│         ROS TCP Connector Bridge                        │
├─────────────────────────────────────────────────────────┤
│                    ROS 2 Middleware                     │
├─────────────────────────────────────────────────────────┤
│            Gazebo Simulation Engine                     │
├─────────────────────────────────────────────────────────┤
│              Robot Control Stack                        │
└─────────────────────────────────────────────────────────┘
```

### Integration Launch File

Let's create a comprehensive launch file that brings together all components:

```python
# simple_robot_pkg/launch/digital_twin_system.launch.py
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    use_unity = LaunchConfiguration('use_unity', default='false')
    world = LaunchConfiguration('world', default='physics_world.sdf')
    robot_name = LaunchConfiguration('robot_name', default='simple_bot')

    # Package names
    pkg_gazebo_ros = FindPackageShare('gazebo_ros').find('gazebo_ros')
    pkg_simple_robot = FindPackageShare('simple_robot_pkg').find('simple_robot_pkg')
    pkg_nav2_bringup = FindPackageShare('nav2_bringup').find('nav2_bringup') if os.path.exists(FindPackageShare('nav2_bringup').find('nav2_bringup')) else None

    # Paths
    urdf_path = os.path.join(pkg_simple_robot, 'urdf', 'simple_bot_complete.urdf')
    world_path = os.path.join(pkg_simple_robot, 'worlds', 'physics_world.sdf')
    rviz_config_path = os.path.join(pkg_simple_robot, 'rviz', 'simple_bot.rviz')
    unity_bridge_path = os.path.join(pkg_simple_robot, 'scripts', 'unity_bridge.py')

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
        'use_unity',
        default_value='false',
        description='Whether to start Unity bridge'))

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

    # Joint State Publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}])

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

    # Controller Manager for diff drive
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[{
            'robot_description': open(urdf_path).read(),
            'use_sim_time': use_sim_time
        }],
        output='screen'
    )

    # Diff drive controller
    diff_drive_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_drive_controller', '-c', '/controller_manager'],
        output='screen'
    )

    # Joint state broadcaster
    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '-c', '/controller_manager'],
        output='screen'
    )

    # Physics Controller
    physics_controller = Node(
        package='simple_robot_pkg',
        executable='physics_controller',
        name='physics_controller',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    # Physics-aware Navigation
    physics_navigation = Node(
        package='simple_robot_pkg',
        executable='physics_aware_navigation',
        name='physics_aware_navigation',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    # Unity Bridge (if enabled)
    unity_bridge = ExecuteProcess(
        cmd=['python3', unity_bridge_path],
        output='screen'
    )

    # Laser scanner node
    laser_scanner = Node(
        package='simple_robot_pkg',
        executable='laser_scanner_node',
        name='laser_scanner_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    # IMU sensor node
    imu_sensor = Node(
        package='simple_robot_pkg',
        executable='imu_sensor_node',
        name='imu_sensor_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    # RViz (if enabled)
    rviz = Node(
        condition=IfCondition(use_rviz),
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
    ld.add_action(joint_state_publisher)
    ld.add_action(spawn_entity)
    ld.add_action(controller_manager)

    # Add controller spawners after controller manager starts
    ld.add_action(RegisterEventHandler(
        OnProcessStart(
            target_action=controller_manager,
            on_start=[diff_drive_controller]
        )
    ))

    ld.add_action(RegisterEventHandler(
        OnProcessStart(
            target_action=diff_drive_controller,
            on_start=[joint_state_broadcaster]
        )
    ))

    ld.add_action(physics_controller)
    ld.add_action(physics_navigation)
    ld.add_action(laser_scanner)
    ld.add_action(imu_sensor)

    # Conditionally add Unity bridge
    ld.add_action(RegisterEventHandler(
        OnProcessStart(
            target_action=spawn_entity,
            on_start=[unity_bridge],
            condition=IfCondition(use_unity)
        )
    ))

    # Conditionally add RViz
    ld.add_action(rviz)

    return ld
```

### Unity Bridge with Enhanced Features

Let's enhance our Unity bridge to handle more ROS 2 message types:

```python
# simple_robot_pkg/scripts/enhanced_unity_bridge.py
import socket
import json
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, LaserScan, Imu, Image, CameraInfo
from geometry_msgs.msg import Twist, Pose, PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String, Float32, Bool
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import time
import base64
from cv_bridge import CvBridge
import cv2

class EnhancedUnityBridge(Node):
    def __init__(self):
        super().__init__('enhanced_unity_bridge')

        # ROS 2 publishers and subscribers
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.laser_scan_sub = self.create_subscription(LaserScan, '/scan', self.laser_scan_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.path_sub = self.create_subscription(Path, '/plan', self.path_callback, 10)
        self.camera_sub = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)

        # ROS 2 publishers for commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.reset_pub = self.create_publisher(Bool, '/reset_simulation', 10)

        # TCP server setup
        self.host = 'localhost'
        self.port = 12345
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.get_logger().info(f'Enhanced Unity Bridge listening on {self.host}:{self.port}')
        except Exception as e:
            self.get_logger().error(f'Failed to bind to {self.host}:{self.port} - {e}')
            return

        # Data to send to Unity
        self.joint_states_data = {}
        self.laser_scan_data = {}
        self.imu_data = {}
        self.odom_data = {}
        self.path_data = []
        self.camera_data = None
        self.camera_info = None

        # CV Bridge for image processing
        self.cv_bridge = CvBridge()

        # Connection management
        self.client_socket = None
        self.client_address = None
        self.client_connected = False

        # Start TCP server thread
        self.tcp_thread = threading.Thread(target=self.tcp_server_loop, daemon=True)
        self.tcp_thread.start()

        # Timer for sending data to Unity
        self.send_timer = self.create_timer(0.05, self.send_data_to_unity)  # 20 Hz

        # Statistics
        self.msg_count = 0
        self.last_send_time = time.time()

    def joint_state_callback(self, msg):
        self.joint_states_data = {
            'names': list(msg.name),
            'positions': [float(p) for p in msg.position],
            'velocities': [float(v) for v in msg.velocity],
            'efforts': [float(e) for e in msg.effort]
        }

    def laser_scan_callback(self, msg):
        self.laser_scan_data = {
            'ranges': [float(r) for r in msg.ranges if not (r != r or r > msg.range_max or r < msg.range_min)],
            'angle_min': float(msg.angle_min),
            'angle_max': float(msg.angle_max),
            'angle_increment': float(msg.angle_increment),
            'time_increment': float(msg.time_increment),
            'scan_time': float(msg.scan_time),
            'range_min': float(msg.range_min),
            'range_max': float(msg.range_max)
        }

    def imu_callback(self, msg):
        self.imu_data = {
            'orientation': [float(msg.orientation.x), float(msg.orientation.y),
                           float(msg.orientation.z), float(msg.orientation.w)],
            'angular_velocity': [float(msg.angular_velocity.x), float(msg.angular_velocity.y),
                                float(msg.angular_velocity.z)],
            'linear_acceleration': [float(msg.linear_acceleration.x), float(msg.linear_acceleration.y),
                                   float(msg.linear_acceleration.z)]
        }

    def odom_callback(self, msg):
        self.odom_data = {
            'position': {
                'x': float(msg.pose.pose.position.x),
                'y': float(msg.pose.pose.position.y),
                'z': float(msg.pose.pose.position.z)
            },
            'orientation': {
                'x': float(msg.pose.pose.orientation.x),
                'y': float(msg.pose.pose.orientation.y),
                'z': float(msg.pose.pose.orientation.z),
                'w': float(msg.pose.pose.orientation.w)
            },
            'linear_velocity': {
                'x': float(msg.twist.twist.linear.x),
                'y': float(msg.twist.twist.linear.y),
                'z': float(msg.twist.twist.linear.z)
            },
            'angular_velocity': {
                'x': float(msg.twist.twist.angular.x),
                'y': float(msg.twist.twist.angular.y),
                'z': float(msg.twist.twist.angular.z)
            }
        }

    def path_callback(self, msg):
        path_points = []
        for pose in msg.poses:
            path_points.append({
                'x': float(pose.pose.position.x),
                'y': float(pose.pose.position.y),
                'z': float(pose.pose.position.z)
            })
        self.path_data = path_points

    def camera_callback(self, msg):
        try:
            # Convert ROS image to OpenCV image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Resize image for performance (optional)
            cv_image = cv2.resize(cv_image, (320, 240))

            # Encode image to base64 for transmission
            _, buffer = cv2.imencode('.jpg', cv_image, [cv2.IMWRITE_JPEG_QUALITY, 50])
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            self.camera_data = {
                'encoding': msg.encoding,
                'height': msg.height,
                'width': msg.width,
                'data': image_base64
            }
        except Exception as e:
            self.get_logger().error(f'Error processing camera image: {e}')

    def camera_info_callback(self, msg):
        self.camera_info = {
            'width': msg.width,
            'height': msg.height,
            'k': list(msg.k),  # Camera matrix
            'd': list(msg.d),  # Distortion coefficients
            'r': list(msg.r),  # Rectification matrix
            'p': list(msg.p)   # Projection matrix
        }

    def tcp_server_loop(self):
        while rclpy.ok():
            try:
                self.get_logger().info('Waiting for Unity connection...')
                self.client_socket, self.client_address = self.socket.accept()
                self.client_connected = True
                self.get_logger().info(f'Unity connected from {self.client_address}')

                while rclpy.ok() and self.client_connected:
                    try:
                        # Receive command from Unity
                        data = self.client_socket.recv(4096)  # Increased buffer size
                        if data:
                            try:
                                cmd = json.loads(data.decode('utf-8'))
                                self.process_unity_command(cmd)
                            except json.JSONDecodeError:
                                self.get_logger().warn('Invalid JSON received from Unity')
                            except UnicodeDecodeError:
                                self.get_logger().warn('Invalid UTF-8 received from Unity')
                    except ConnectionResetError:
                        self.get_logger().info('Unity client disconnected')
                        self.client_connected = False
                        break
                    except Exception as e:
                        self.get_logger().error(f'Error receiving data from Unity: {e}')
                        self.client_connected = False
                        break

            except Exception as e:
                self.get_logger().error(f'Error accepting connection: {e}')
                time.sleep(1)  # Wait before trying again

        self.socket.close()

    def process_unity_command(self, cmd):
        cmd_type = cmd.get('type', '')

        if cmd_type == 'cmd_vel':
            twist = Twist()
            twist.linear.x = cmd.get('linear_x', 0.0)
            twist.linear.y = cmd.get('linear_y', 0.0)
            twist.linear.z = cmd.get('linear_z', 0.0)
            twist.angular.x = cmd.get('angular_x', 0.0)
            twist.angular.y = cmd.get('angular_y', 0.0)
            twist.angular.z = cmd.get('angular_z', 0.0)

            self.cmd_vel_pub.publish(twist)

        elif cmd_type == 'navigation_goal':
            goal = PoseStamped()
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.header.frame_id = cmd.get('frame_id', 'map')
            goal.pose.position.x = cmd.get('x', 0.0)
            goal.pose.position.y = cmd.get('y', 0.0)
            goal.pose.position.z = cmd.get('z', 0.0)
            goal.pose.orientation.w = 1.0  # Default orientation

            self.goal_pub.publish(goal)

        elif cmd_type == 'reset':
            reset_msg = Bool()
            reset_msg.data = True
            self.reset_pub.publish(reset_msg)

        elif cmd_type == 'request_data':
            # Unity is requesting specific data - we'll send it in the next cycle
            pass

        elif cmd_type == 'emergency_stop':
            # Send emergency stop command
            stop_twist = Twist()
            self.cmd_vel_pub.publish(stop_twist)
            self.get_logger().warn('Emergency stop activated from Unity')

    def send_data_to_unity(self):
        if not self.client_connected or self.client_socket is None:
            return

        try:
            # Prepare data to send to Unity
            data_to_send = {
                'timestamp': time.time(),
                'joint_states': self.joint_states_data,
                'laser_scan': self.laser_scan_data,
                'imu': self.imu_data,
                'odom': self.odom_data,
                'path': self.path_data,
                'camera': self.camera_data,
                'camera_info': self.camera_info
            }

            json_data = json.dumps(data_to_send, separators=(',', ':'))
            self.client_socket.send(json_data.encode('utf-8'))

            self.msg_count += 1
            current_time = time.time()
            if current_time - self.last_send_time >= 1.0:  # Print stats every second
                self.get_logger().info(f'Sent {self.msg_count} messages in the last second')
                self.msg_count = 0
                self.last_send_time = current_time

        except BrokenPipeError:
            self.get_logger().warn('Unity client disconnected (BrokenPipeError)')
            self.client_connected = False
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
        except Exception as e:
            self.get_logger().error(f'Error sending data to Unity: {e}')
            self.client_connected = False
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None

    def destroy_node(self):
        self.client_connected = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    bridge = EnhancedUnityBridge()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Unity Integration Script

Now let's create an updated Unity script that works with the enhanced bridge:

```csharp
// Assets/Scripts/EnhancedUnityRobotInterface.cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Robotics.ROSTCPConnector;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

public class EnhancedUnityRobotInterface : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosIP = "127.0.0.1";
    public int rosPort = 12345;

    [Header("Robot Components")]
    public Transform robotBase;
    public Transform leftWheel;
    public Transform rightWheel;
    public Transform laserScanner;
    public Camera robotCamera;

    [Header("Visualization")]
    public Material validRangeMaterial;
    public Material invalidRangeMaterial;
    public float maxLaserRange = 10.0f;
    public int laserResolution = 360; // Number of laser beams

    [Header("UI Elements")]
    public Text statusText;
    public Text positionText;
    public Text velocityText;
    public Slider linearVelSlider;
    public Slider angularVelSlider;
    public Button resetButton;
    public Button emergencyStopButton;
    public RawImage cameraFeed;

    private ROSConnection ros;
    private bool isConnected = false;
    private float lastReceiveTime = 0f;
    private float connectionTimeout = 5.0f;

    // Robot state
    private Dictionary<string, float> jointPositions = new Dictionary<string, float>();
    private Vector3 robotPosition = Vector3.zero;
    private Quaternion robotRotation = Quaternion.identity;
    private Vector3 robotLinearVel = Vector3.zero;
    private Vector3 robotAngularVel = Vector3.zero;
    private List<float> laserRanges = new List<float>();
    private float[] cameraImage = null; // Placeholder for camera data

    // Laser visualization objects
    private GameObject[] laserRays;

    void Start()
    {
        InitializeROSConnection();
        InitializeLaserVisualization();
        SetupUIEventHandlers();
    }

    void InitializeROSConnection()
    {
        ros = ROSConnection.GetOrCreateInstance();
        StartCoroutine(ConnectToROS());
    }

    IEnumerator ConnectToROS()
    {
        // Attempt to connect to ROS bridge
        while (!isConnected)
        {
            try
            {
                ros.Initialize(rosIP, rosPort);
                isConnected = true;
                Debug.Log("Connected to ROS bridge");
                UpdateStatus("Connected to ROS");
            }
            catch
            {
                isConnected = false;
                Debug.LogWarning("Failed to connect to ROS, retrying...");
                UpdateStatus("Connecting to ROS...");
                yield return new WaitForSeconds(2.0f);
            }
        }
    }

    void InitializeLaserVisualization()
    {
        laserRays = new GameObject[laserResolution];
        for (int i = 0; i < laserResolution; i++)
        {
            GameObject ray = new GameObject($"LaserRay_{i}");
            ray.transform.SetParent(transform);
            LineRenderer lineRenderer = ray.AddComponent<LineRenderer>();
            lineRenderer.material = validRangeMaterial;
            lineRenderer.startWidth = 0.01f;
            lineRenderer.endWidth = 0.01f;
            laserRays[i] = ray;
        }
    }

    void SetupUIEventHandlers()
    {
        if (linearVelSlider != null)
            linearVelSlider.onValueChanged.AddListener(OnLinearVelChanged);

        if (angularVelSlider != null)
            angularVelSlider.onValueChanged.AddListener(OnAngularVelChanged);

        if (resetButton != null)
            resetButton.onClick.AddListener(OnResetClicked);

        if (emergencyStopButton != null)
            emergencyStopButton.onClick.AddListener(OnEmergencyStopClicked);
    }

    void Update()
    {
        // Check for connection timeout
        if (isConnected && Time.time - lastReceiveTime > connectionTimeout)
        {
            isConnected = false;
            UpdateStatus("Connection timeout");
        }

        // Update laser visualization
        UpdateLaserVisualization();

        // Update UI
        UpdateUI();
    }

    void UpdateLaserVisualization()
    {
        if (laserRanges.Count == 0 || laserRays == null) return;

        float angleStep = 360.0f / laserResolution;

        for (int i = 0; i < laserResolution; i++)
        {
            if (i < laserRanges.Count)
            {
                float distance = laserRanges[i];
                float angle = (i * angleStep) * Mathf.Deg2Rad;

                Vector3 direction = new Vector3(
                    Mathf.Cos(angle) * distance,
                    0,
                    Mathf.Sin(angle) * distance
                );

                LineRenderer lineRenderer = laserRays[i].GetComponent<LineRenderer>();
                if (lineRenderer != null)
                {
                    lineRenderer.SetPosition(0, laserScanner.position);
                    lineRenderer.SetPosition(1, laserScanner.position + laserScanner.TransformDirection(direction));

                    // Color based on range validity
                    if (distance < maxLaserRange && distance > 0.1f)
                    {
                        lineRenderer.material = validRangeMaterial;
                    }
                    else
                    {
                        lineRenderer.material = invalidRangeMaterial;
                    }
                }
            }
        }
    }

    void UpdateUI()
    {
        if (positionText != null)
        {
            positionText.text = $"Position: ({robotPosition.x:F2}, {robotPosition.y:F2}, {robotPosition.z:F2})";
        }

        if (velocityText != null)
        {
            velocityText.text = $"Velocity: ({robotLinearVel.x:F2}, {robotAngularVel.z:F2})";
        }
    }

    void UpdateStatus(string status)
    {
        if (statusText != null)
        {
            statusText.text = status;
            statusText.color = isConnected ? Color.green : Color.red;
        }
    }

    void OnLinearVelChanged(float value)
    {
        if (isConnected)
        {
            SendVelocityCommand(value, angularVelSlider.value);
        }
    }

    void OnAngularVelChanged(float value)
    {
        if (isConnected)
        {
            SendVelocityCommand(linearVelSlider.value, value);
        }
    }

    void OnResetClicked()
    {
        if (isConnected)
        {
            var resetCmd = new { type = "reset" };
            ros.Send("unity_commands", JsonConvert.SerializeObject(resetCmd));
        }
    }

    void OnEmergencyStopClicked()
    {
        if (isConnected)
        {
            var emergencyCmd = new { type = "emergency_stop" };
            ros.Send("unity_commands", JsonConvert.SerializeObject(emergencyCmd));

            // Also reset sliders
            if (linearVelSlider != null) linearVelSlider.value = 0;
            if (angularVelSlider != null) angularVelSlider.value = 0;
        }
    }

    void SendVelocityCommand(float linear, float angular)
    {
        var cmd = new
        {
            type = "cmd_vel",
            linear_x = linear,
            linear_y = 0.0,
            linear_z = 0.0,
            angular_x = 0.0,
            angular_y = 0.0,
            angular_z = angular
        };

        ros.Send("unity_commands", JsonConvert.SerializeObject(cmd));
    }

    // Method to send navigation goal
    public void SendNavigationGoal(float x, float y, float z = 0)
    {
        if (isConnected)
        {
            var goalCmd = new
            {
                type = "navigation_goal",
                frame_id = "map",
                x = x,
                y = y,
                z = z
            };

            ros.Send("unity_commands", JsonConvert.SerializeObject(goalCmd));
        }
    }

    // Process incoming data from ROS
    public void ProcessROSData(string jsonData)
    {
        lastReceiveTime = Time.time;

        try
        {
            JObject data = JObject.Parse(jsonData);

            // Update joint states
            var jointStates = data["joint_states"];
            if (jointStates != null)
            {
                var names = jointStates["names"] as JArray;
                var positions = jointStates["positions"] as JArray;

                if (names != null && positions != null && names.Count == positions.Count)
                {
                    jointPositions.Clear();
                    for (int i = 0; i < names.Count; i++)
                    {
                        jointPositions[names[i].ToString()] = (float)positions[i];
                    }
                }
            }

            // Update laser scan
            var laserScan = data["laser_scan"];
            if (laserScan != null)
            {
                var ranges = laserScan["ranges"] as JArray;
                laserRanges.Clear();
                if (ranges != null)
                {
                    foreach (var range in ranges)
                    {
                        laserRanges.Add((float)range);
                    }
                }
            }

            // Update odometry
            var odom = data["odom"];
            if (odom != null)
            {
                var pos = odom["position"];
                if (pos != null)
                {
                    robotPosition = new Vector3(
                        (float)pos["x"],
                        (float)pos["y"],
                        (float)pos["z"]
                    );
                }

                var orient = odom["orientation"];
                if (orient != null)
                {
                    robotRotation = new Quaternion(
                        (float)orient["x"],
                        (float)orient["y"],
                        (float)orient["z"],
                        (float)orient["w"]
                    );
                }

                var linVel = odom["linear_velocity"];
                if (linVel != null)
                {
                    robotLinearVel = new Vector3(
                        (float)linVel["x"],
                        (float)linVel["y"],
                        (float)linVel["z"]
                    );
                }

                var angVel = odom["angular_velocity"];
                if (angVel != null)
                {
                    robotAngularVel = new Vector3(
                        (float)angVel["x"],
                        (float)angVel["y"],
                        (float)angVel["z"]
                    );
                }

                // Update robot transform
                if (robotBase != null)
                {
                    robotBase.position = robotPosition;
                    robotBase.rotation = robotRotation;
                }

                // Update wheel rotations
                if (jointPositions.ContainsKey("left_wheel_joint") && leftWheel != null)
                {
                    leftWheel.localRotation = Quaternion.Euler(90, 0, jointPositions["left_wheel_joint"] * Mathf.Rad2Deg);
                }

                if (jointPositions.ContainsKey("right_wheel_joint") && rightWheel != null)
                {
                    rightWheel.localRotation = Quaternion.Euler(90, 0, jointPositions["right_wheel_joint"] * Mathf.Rad2Deg);
                }
            }

            // Update camera feed
            var camera = data["camera"];
            if (camera != null && cameraFeed != null)
            {
                string imageData = camera["data"].ToString();
                // In a real implementation, you would decode the base64 image data
                // and update the RawImage component
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error processing ROS data: {e.Message}");
        }
    }

    // Cleanup on destroy
    void OnDestroy()
    {
        if (ros != null)
        {
            ros.Close();
        }
    }
}
```

### Deployment Configuration

For deploying the digital twin system in different environments, let's create configuration files:

```yaml
# simple_robot_pkg/config/digital_twin_config.yaml
digital_twin:
  ros_bridge:
    ip: "127.0.0.1"
    port: 12345
    timeout: 5.0
    max_reconnect_attempts: 10

  unity_visualization:
    enabled: true
    resolution: [1280, 720]
    frame_rate: 60
    quality_level: 2  # 0=Fastest, 5=Fantastic

  simulation:
    real_time_factor: 1.0
    physics_update_rate: 1000
    max_step_size: 0.001

  sensors:
    laser:
      enabled: true
      update_rate: 10
      range_min: 0.1
      range_max: 10.0
      angle_min: -1.57  # -90 degrees
      angle_max: 1.57   # 90 degrees
      angle_increment: 0.01745  # 1 degree
    imu:
      enabled: true
      update_rate: 100
    camera:
      enabled: true
      update_rate: 15
      width: 640
      height: 480
      fov: 60

  controllers:
    diff_drive:
      wheel_separation: 0.3
      wheel_radius: 0.1
      max_linear_vel: 0.5
      max_angular_vel: 1.0
      cmd_timeout: 0.5
```

### System Monitoring and Diagnostics

Let's create a monitoring node for the digital twin system:

```python
# simple_robot_pkg/digital_twin_monitor.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import JointState, LaserScan, Imu
from geometry_msgs.msg import Twist
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import time
import psutil
import GPUtil

class DigitalTwinMonitor(Node):
    def __init__(self):
        super().__init__('digital_twin_monitor')

        # Subscribers for system monitoring
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.laser_scan_sub = self.create_subscription(LaserScan, '/scan', self.laser_scan_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Publishers for diagnostics
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.system_status_pub = self.create_publisher(String, '/system_status', 10)

        # Timer for system monitoring
        self.monitor_timer = self.create_timer(1.0, self.monitor_system)

        # State tracking
        self.last_joint_update = 0
        self.last_laser_update = 0
        self.last_imu_update = 0
        self.last_cmd_vel = 0

        # Performance tracking
        self.msg_counts = {
            'joint_state': 0,
            'laser_scan': 0,
            'imu': 0,
            'cmd_vel': 0
        }

        self.get_logger().info('Digital Twin Monitor initialized')

    def joint_state_callback(self, msg):
        self.last_joint_update = time.time()
        self.msg_counts['joint_state'] += 1

    def laser_scan_callback(self, msg):
        self.last_laser_update = time.time()
        self.msg_counts['laser_scan'] += 1

    def imu_callback(self, msg):
        self.last_imu_update = time.time()
        self.msg_counts['imu'] += 1

    def cmd_vel_callback(self, msg):
        self.last_cmd_vel = time.time()
        self.msg_counts['cmd_vel'] += 1

    def monitor_system(self):
        # Create diagnostic array
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        # System resource diagnostics
        system_diag = self.create_system_diagnostics()
        diag_array.status.append(system_diag)

        # Sensor diagnostics
        sensor_diag = self.create_sensor_diagnostics()
        diag_array.status.append(sensor_diag)

        # Performance diagnostics
        perf_diag = self.create_performance_diagnostics()
        diag_array.status.append(perf_diag)

        # Publish diagnostics
        self.diag_pub.publish(diag_array)

        # Publish system status summary
        status_msg = String()
        status_msg.data = self.get_system_summary()
        self.system_status_pub.publish(status_msg)

        # Reset message counts
        for key in self.msg_counts:
            self.msg_counts[key] = 0

    def create_system_diagnostics(self):
        status = DiagnosticStatus()
        status.name = "System Resources"
        status.level = DiagnosticStatus.OK
        status.message = "System resources within normal range"

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        status.values.append(KeyValue(key="CPU Usage (%)", value=f"{cpu_percent}"))

        # Memory usage
        memory = psutil.virtual_memory()
        status.values.append(KeyValue(key="Memory Usage (%)", value=f"{memory.percent}"))
        status.values.append(KeyValue(key="Memory Available (GB)", value=f"{memory.available / (1024**3):.2f}"))

        # Disk usage
        disk = psutil.disk_usage('/')
        status.values.append(KeyValue(key="Disk Usage (%)", value=f"{(disk.used/disk.total)*100:.1f}"))

        # GPU usage (if available)
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Use first GPU
            status.values.append(KeyValue(key="GPU Usage (%)", value=f"{gpu.load*100:.1f}"))
            status.values.append(KeyValue(key="GPU Memory (%)", value=f"{gpu.memoryUtil*100:.1f}"))

        # Check if resources are too high
        if cpu_percent > 90 or memory.percent > 90:
            status.level = DiagnosticStatus.WARN
            status.message = "High resource usage detected"

        return status

    def create_sensor_diagnostics(self):
        status = DiagnosticStatus()
        status.name = "Sensor Status"
        status.level = DiagnosticStatus.OK
        status.message = "All sensors operational"

        current_time = time.time()

        # Check sensor update times
        if current_time - self.last_joint_update > 2.0:
            status.level = DiagnosticStatus.ERROR
            status.message = "Joint state timeout"
        elif current_time - self.last_laser_update > 2.0:
            status.level = DiagnosticStatus.ERROR
            status.message = "Laser scan timeout"
        elif current_time - self.last_imu_update > 2.0:
            status.level = DiagnosticStatus.ERROR
            status.message = "IMU timeout"

        status.values.append(KeyValue(key="Joint State Age (s)", value=f"{current_time - self.last_joint_update:.2f}"))
        status.values.append(KeyValue(key="Laser Scan Age (s)", value=f"{current_time - self.last_laser_update:.2f}"))
        status.values.append(KeyValue(key="IMU Age (s)", value=f"{current_time - self.last_imu_update:.2f}"))

        return status

    def create_performance_diagnostics(self):
        status = DiagnosticStatus()
        status.name = "Performance Metrics"
        status.level = DiagnosticStatus.OK
        status.message = "Performance within acceptable range"

        # Message rates
        for msg_type, count in self.msg_counts.items():
            status.values.append(KeyValue(key=f"{msg_type} Rate (Hz)", value=f"{count}"))

        # Add performance warnings if needed
        if self.msg_counts['joint_state'] > 100:  # Too high frequency
            status.level = DiagnosticStatus.WARN
            status.message = "High message frequency detected"

        return status

    def get_system_summary(self):
        current_time = time.time()
        joint_age = current_time - self.last_joint_update
        laser_age = current_time - self.last_laser_update
        imu_age = current_time - self.last_imu_update

        return f"Joint: {joint_age:.1f}s, Laser: {laser_age:.1f}s, IMU: {imu_age:.1f}s, Status: {self.get_status_string()}"

    def get_status_string(self):
        current_time = time.time()
        max_age = max(
            current_time - self.last_joint_update,
            current_time - self.last_laser_update,
            current_time - self.last_imu_update
        )

        if max_age > 2.0:
            return "ERROR"
        elif max_age > 1.0:
            return "WARNING"
        else:
            return "OK"

def main(args=None):
    rclpy.init(args=args)
    monitor = DigitalTwinMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Running the Complete Digital Twin System

To run the complete integrated system:

```bash
# Terminal 1: Start the enhanced Unity bridge
cd ~/ros2_ws
source install/setup.bash
ros2 run simple_robot_pkg enhanced_unity_bridge

# Terminal 2: Start the monitoring system
ros2 run simple_robot_pkg digital_twin_monitor

# Terminal 3: Start the complete simulation
ros2 launch simple_robot_pkg digital_twin_system.launch.py use_unity:=true

# Terminal 4: In Unity editor, run the scene with EnhancedUnityRobotInterface
```

## Best Practices for Deployment

1. **Containerization**: Use Docker to package the entire system
2. **Configuration Management**: Use YAML files for environment-specific settings
3. **Monitoring**: Implement comprehensive monitoring and logging
4. **Security**: Secure communication channels between components
5. **Scalability**: Design for multiple robots and distributed systems
6. **Failover**: Implement graceful degradation when components fail

## Next Steps

With the digital twin system complete, we now move to Module 3, where we'll explore NVIDIA Isaac - a comprehensive platform for robotics simulation, perception, and AI. We'll see how to integrate Isaac's advanced capabilities with our existing ROS 2 and Unity infrastructure.