---
sidebar_position: 2
---

# Chapter 2: Advanced ROS 2 Concepts

## Services

Services provide a request/response communication pattern in ROS 2. Unlike topics which are asynchronous, services are synchronous - the client waits for a response from the server. This makes services ideal for operations that require a definitive result or acknowledgment.

### When to Use Services

Services are appropriate for:
- **Configuration requests**: Setting parameters or changing robot state
- **One-time operations**: Calculations, data retrieval, or simple commands
- **Synchronous operations**: When the client needs to wait for completion
- **Command acknowledgment**: Confirming that a command was received and processed

### Service Definition

Services are defined using `.srv` files that specify the request and response message formats:

```python
# simple_robot_pkg/srv/RobotCommand.srv
string command
float64[] parameters
---
bool success
string message
float64 result
```

This service definition creates:
- **Request**: `command` (string) and `parameters` (array of floats)
- **Response**: `success` (boolean), `message` (string), and `result` (float)

### Service Server Implementation

Here's a comprehensive service server for robot commands:

```python
# simple_robot_pkg/robot_command_server.py
import rclpy
from rclpy.node import Node
from simple_robot_pkg.srv import RobotCommand  # Assuming you created the service definition
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import time

class RobotCommandServer(Node):
    def __init__(self):
        super().__init__('robot_command_server')

        # Create the service
        self.srv = self.create_service(
            RobotCommand,
            'robot_command',
            self.command_callback
        )

        # Publishers for robot control
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.joint_cmd_publisher = self.create_publisher(JointState, 'joint_commands', 10)

        # Robot state tracking
        self.robot_state = {
            'x': 0.0, 'y': 0.0, 'theta': 0.0,
            'is_moving': False,
            'battery_level': 100.0
        }

        self.get_logger().info('Robot Command Server initialized')

    def command_callback(self, request, response):
        """Handle incoming robot commands"""
        command = request.command.lower()
        params = request.parameters

        self.get_logger().info(f'Received command: {command} with params: {params}')

        try:
            if command == 'move_forward':
                response = self.handle_move_forward(params, response)
            elif command == 'turn':
                response = self.handle_turn(params, response)
            elif command == 'stop':
                response = self.handle_stop(response)
            elif command == 'get_status':
                response = self.handle_get_status(response)
            elif command == 'set_joint_position':
                response = self.handle_set_joint_position(params, response)
            else:
                response.success = False
                response.message = f'Unknown command: {command}'
                response.result = -1.0

        except Exception as e:
            self.get_logger().error(f'Error processing command {command}: {str(e)}')
            response.success = False
            response.message = f'Error processing command: {str(e)}'
            response.result = -1.0

        return response

    def handle_move_forward(self, params, response):
        """Handle move forward command"""
        distance = params[0] if len(params) > 0 else 1.0
        speed = params[1] if len(params) > 1 else 0.5

        # Create velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = speed
        cmd_vel.angular.z = 0.0

        # Publish command
        self.cmd_vel_publisher.publish(cmd_vel)

        # Simulate movement (in real robot, this would be based on encoders)
        time.sleep(abs(distance) / abs(speed))

        # Stop the robot
        cmd_vel.linear.x = 0.0
        self.cmd_vel_publisher.publish(cmd_vel)

        # Update robot state
        self.robot_state['x'] += distance * 0.7  # Simplified kinematics
        self.robot_state['is_moving'] = False

        response.success = True
        response.message = f'Moved forward {distance} meters at {speed} m/s'
        response.result = distance

        return response

    def handle_turn(self, params, response):
        """Handle turn command"""
        angle = params[0] if len(params) > 0 else 90.0  # degrees
        angular_speed = params[1] if len(params) > 1 else 0.5  # rad/s

        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = angular_speed if angle > 0 else -abs(angular_speed)

        # Calculate time to turn (simplified)
        turn_time = abs(angle * 3.14159 / 180.0) / abs(angular_speed)

        # Publish command
        self.cmd_vel_publisher.publish(cmd_vel)

        # Simulate turn
        time.sleep(turn_time)

        # Stop the robot
        cmd_vel.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_vel)

        # Update robot state
        self.robot_state['theta'] += angle * 3.14159 / 180.0
        self.robot_state['is_moving'] = False

        response.success = True
        response.message = f'Turned {angle} degrees at {angular_speed} rad/s'
        response.result = angle

        return response

    def handle_stop(self, response):
        """Handle stop command"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0

        self.cmd_vel_publisher.publish(cmd_vel)
        self.robot_state['is_moving'] = False

        response.success = True
        response.message = 'Robot stopped'
        response.result = 0.0

        return response

    def handle_get_status(self, response):
        """Handle get status command"""
        response.success = True
        response.message = f'Robot at ({self.robot_state["x"]:.2f}, {self.robot_state["y"]:.2f}), battery: {self.robot_state["battery_level"]:.1f}%'
        response.result = self.robot_state['battery_level']

        return response

    def handle_set_joint_position(self, params, response):
        """Handle set joint position command"""
        if len(params) < 2:
            response.success = False
            response.message = 'Need at least joint_id and position'
            response.result = -1.0
            return response

        joint_id = int(params[0])
        position = params[1]

        # Create joint command
        joint_cmd = JointState()
        joint_cmd.header.stamp = self.get_clock().now().to_msg()
        joint_cmd.header.frame_id = 'base_link'

        # For demonstration, we'll assume joint names are indexed
        joint_cmd.name = [f'joint_{joint_id}']
        joint_cmd.position = [position]

        self.joint_cmd_publisher.publish(joint_cmd)

        response.success = True
        response.message = f'Set joint {joint_id} to position {position}'
        response.result = position

        return response

def main(args=None):
    rclpy.init(args=args)
    server = RobotCommandServer()

    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        server.get_logger().info('Service server interrupted')
    finally:
        server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client Implementation

Here's a comprehensive client that can interact with the robot command server:

```python
# simple_robot_pkg/robot_command_client.py
import sys
import rclpy
from rclpy.node import Node
from simple_robot_pkg.srv import RobotCommand  # Assuming you created the service definition
import time

class RobotCommandClient(Node):
    def __init__(self):
        super().__init__('robot_command_client')
        self.cli = self.create_client(RobotCommand, 'robot_command')

        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.req = RobotCommand.Request()

    def send_command(self, command, parameters=None):
        """Send a command to the robot"""
        if parameters is None:
            parameters = []

        self.req.command = command
        self.req.parameters = parameters

        self.get_logger().info(f'Sending command: {command} with params: {parameters}')

        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)

        return future.result()

    def run_demo_sequence(self):
        """Run a demonstration sequence of commands"""
        self.get_logger().info('Starting robot command demo sequence...')

        # Get initial status
        response = self.send_command('get_status')
        if response:
            self.get_logger().info(f'Initial status: {response.message}')

        # Move forward
        response = self.send_command('move_forward', [1.0, 0.3])  # 1m at 0.3 m/s
        if response and response.success:
            self.get_logger().info(f'Move forward result: {response.message}')

        # Turn right
        response = self.send_command('turn', [90.0, 0.2])  # 90 degrees at 0.2 rad/s
        if response and response.success:
            self.get_logger().info(f'Turn result: {response.message}')

        # Get updated status
        response = self.send_command('get_status')
        if response:
            self.get_logger().info(f'Updated status: {response.message}')

        # Stop robot
        response = self.send_command('stop')
        if response and response.success:
            self.get_logger().info(f'Stop result: {response.message}')

def main(args=None):
    rclpy.init(args=args)
    client = RobotCommandClient()

    try:
        # Check command line arguments
        if len(sys.argv) > 1:
            command = sys.argv[1]
            params = [float(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else []

            response = client.send_command(command, params)
            if response:
                if response.success:
                    client.get_logger().info(f'Success: {response.message}, Result: {response.result}')
                else:
                    client.get_logger().error(f'Failed: {response.message}')
            else:
                client.get_logger().error('Service call failed')
        else:
            # Run demo sequence
            client.run_demo_sequence()

    except KeyboardInterrupt:
        client.get_logger().info('Client interrupted')
    except ValueError as e:
        client.get_logger().error(f'Invalid parameter format: {e}')
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Actions

Actions are used for long-running tasks that may take some time to complete. They provide feedback during execution and can be canceled, making them ideal for operations like navigation, manipulation, or complex movements.

### When to Use Actions

Actions are appropriate for:
- **Long-running operations**: Tasks that take seconds to minutes
- **Progress tracking**: Operations where you need to monitor progress
- **Cancelation capability**: Operations that can be interrupted
- **Goal management**: Operations with specific targets or endpoints

### Action Definition

Actions are defined using `.action` files that specify goal, result, and feedback messages:

```python
# simple_robot_pkg/action/MoveToPose.action
# Goal definition
geometry_msgs/Pose target_pose
float64 tolerance
---
# Result definition
bool success
string message
float64 distance_traveled
---
# Feedback definition
float64 distance_to_goal
float64 current_progress
string status
```

### Action Server Implementation

Here's a comprehensive action server for navigation:

```python
# simple_robot_pkg/move_to_pose_server.py
import time
import math
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.node import Node
from simple_robot_pkg.action import MoveToPose  # Assuming you created the action definition
from geometry_msgs.msg import Twist, Pose, Point
from nav_msgs.msg import Odometry
import tf_transformations

class MoveToPoseActionServer(Node):
    def __init__(self):
        super().__init__('move_to_pose_server')

        # Create action server
        self._action_server = ActionServer(
            self,
            MoveToPose,
            'move_to_pose',
            self.execute_callback,
            cancel_callback=self.cancel_callback
        )

        # Publishers and subscribers
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_subscriber = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )

        # Robot state
        self.current_pose = Pose()
        self.is_moving = False

        self.get_logger().info('MoveToPose Action Server initialized')

    def odom_callback(self, msg):
        """Update current robot pose from odometry"""
        self.current_pose = msg.pose.pose

    def cancel_callback(self, goal_handle):
        """Handle goal cancellation"""
        self.get_logger().info('Received cancel request')
        self.is_moving = False

        # Stop the robot
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_vel)

        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute the move to pose action"""
        self.get_logger().info('Executing move to pose goal...')

        target_pose = goal_handle.request.target_pose
        tolerance = goal_handle.request.tolerance

        # Initialize feedback
        feedback_msg = MoveToPose.Feedback()
        feedback_msg.distance_to_goal = self.calculate_distance_to_target(target_pose)
        feedback_msg.current_progress = 0.0
        feedback_msg.status = 'Moving to target'

        self.is_moving = True

        # Main navigation loop
        while self.is_moving:
            # Check for cancellation
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')

                # Create result
                result = MoveToPose.Result()
                result.success = False
                result.message = 'Goal was canceled'
                result.distance_traveled = 0.0

                return result

            # Calculate distance to goal
            distance_to_goal = self.calculate_distance_to_target(target_pose)

            # Check if we've reached the target
            if distance_to_goal <= tolerance:
                self.get_logger().info('Reached target pose')
                break

            # Calculate control commands
            cmd_vel = self.calculate_navigation_commands(target_pose)

            # Publish command
            self.cmd_vel_publisher.publish(cmd_vel)

            # Update feedback
            feedback_msg.distance_to_goal = distance_to_goal
            feedback_msg.current_progress = max(0.0, 1.0 - (distance_to_goal /
                                                            self.calculate_distance_to_target(target_pose)))
            feedback_msg.status = f'Moving, {distance_to_goal:.2f}m to goal'

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)

            # Small delay to prevent busy waiting
            time.sleep(0.1)

        # Stop the robot
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_vel)
        self.is_moving = False

        # Calculate final result
        final_distance = self.calculate_distance_to_target(target_pose)
        success = final_distance <= tolerance

        result = MoveToPose.Result()
        result.success = success
        result.message = f'Reached target: {success}, final distance: {final_distance:.3f}m'
        result.distance_traveled = self.calculate_distance_to_target(target_pose)

        if success:
            goal_handle.succeed()
            self.get_logger().info(f'Goal succeeded: {result.message}')
        else:
            goal_handle.abort()
            self.get_logger().info(f'Goal failed: {result.message}')

        return result

    def calculate_distance_to_target(self, target_pose):
        """Calculate Euclidean distance to target pose"""
        dx = target_pose.position.x - self.current_pose.position.x
        dy = target_pose.position.y - self.current_pose.position.y
        return math.sqrt(dx*dx + dy*dy)

    def calculate_navigation_commands(self, target_pose):
        """Calculate velocity commands to move toward target pose"""
        cmd_vel = Twist()

        # Calculate desired direction
        dx = target_pose.position.x - self.current_pose.position.x
        dy = target_pose.position.y - self.current_pose.position.y

        # Calculate distance and angle
        distance = math.sqrt(dx*dx + dy*dy)
        desired_angle = math.atan2(dy, dx)

        # Get current orientation from quaternion
        current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)

        # Calculate angle difference
        angle_diff = desired_angle - current_yaw
        # Normalize angle difference to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # PID-like control parameters
        linear_kp = 0.5
        angular_kp = 1.0

        # Set commands
        cmd_vel.linear.x = min(linear_kp * distance, 0.5)  # Limit speed
        cmd_vel.angular.z = angular_kp * angle_diff

        return cmd_vel

    def quaternion_to_yaw(self, quaternion):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    server = MoveToPoseActionServer()

    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        server.get_logger().info('Action server interrupted')
    finally:
        server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Action Client Implementation

Here's a client for the move to pose action:

```python
# simple_robot_pkg/move_to_pose_client.py
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from simple_robot_pkg.action import MoveToPose  # Assuming you created the action definition
from geometry_msgs.msg import Pose, Point, Quaternion
import time

class MoveToPoseClient(Node):
    def __init__(self):
        super().__init__('move_to_pose_client')
        self._action_client = ActionClient(self, MoveToPose, 'move_to_pose')

    def send_goal(self, x, y, tolerance=0.1):
        """Send a goal to move to a specific pose"""
        goal_msg = MoveToPose.Goal()

        # Set target pose
        goal_msg.target_pose = Pose()
        goal_msg.target_pose.position.x = float(x)
        goal_msg.target_pose.position.y = float(y)
        goal_msg.target_pose.position.z = 0.0
        goal_msg.target_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        goal_msg.tolerance = float(tolerance)

        self.get_logger().info(f'Sending goal to move to ({x}, {y}) with tolerance {tolerance}')

        # Wait for action server
        self._action_client.wait_for_server()

        # Send goal
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle result callback"""
        result = future.result().result
        self.get_logger().info(f'Result: {result.message}')

        # Shutdown after completion
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        """Handle feedback callback"""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Feedback: {feedback.status}, '
            f'Distance to goal: {feedback.distance_to_goal:.2f}m, '
            f'Progress: {feedback.current_progress*100:.1f}%'
        )

def main(args=None):
    rclpy.init(args=args)

    # Parse command line arguments
    if len(sys.argv) >= 3:
        x = float(sys.argv[1])
        y = float(sys.argv[2])
        tolerance = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    else:
        # Default values
        x, y, tolerance = 1.0, 1.0, 0.1
        print("Usage: ros2 run simple_robot_pkg move_to_pose_client <x> <y> [tolerance]")
        print(f"Using default: move to ({x}, {y}) with tolerance {tolerance}")

    client = MoveToPoseClient()

    # Send the goal
    client.send_goal(x, y, tolerance)

    # Spin to process callbacks
    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        client.get_logger().info('Client interrupted')

if __name__ == '__main__':
    main()
```

## Parameters

Parameters allow you to configure nodes at runtime, providing flexibility without requiring code changes. They're essential for humanoid robotics where different robots may have different configurations.

### Parameter Types and Best Practices

```python
# simple_robot_pkg/robot_parameters_node.py
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rcl_interfaces.srv import SetParameters, GetParameters, ListParameters
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_parameters

class RobotParametersNode(Node):
    def __init__(self):
        super().__init__('robot_parameters_node')

        # Declare parameters with descriptions and constraints
        self.declare_parameter(
            'robot_name',
            'default_robot',
            ParameterDescriptor(
                name='robot_name',
                type=ParameterType.PARAMETER_STRING,
                description='Name of the robot',
                read_only=False
            )
        )

        self.declare_parameter(
            'max_linear_velocity',
            1.0,
            ParameterDescriptor(
                name='max_linear_velocity',
                type=ParameterType.PARAMETER_DOUBLE,
                description='Maximum linear velocity in m/s',
                read_only=False,
                floating_point_range=[ParameterDescriptor(range=[0.0, 5.0, 0.1])]
            )
        )

        self.declare_parameter(
            'max_angular_velocity',
            1.57,  # π/2 radians/second
            ParameterDescriptor(
                name='max_angular_velocity',
                type=ParameterType.PARAMETER_DOUBLE,
                description='Maximum angular velocity in rad/s',
                read_only=False,
                floating_point_range=[ParameterDescriptor(range=[0.0, 3.14, 0.01])]
            )
        )

        self.declare_parameter(
            'joint_limits',
            [1.57, 1.57, 1.57, 1.57, 1.57, 1.57],  # 6 joints with 90-degree limits
            ParameterDescriptor(
                name='joint_limits',
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description='Maximum joint angle limits in radians',
                read_only=False
            )
        )

        self.declare_parameter(
            'enable_safety_mode',
            True,
            ParameterDescriptor(
                name='enable_safety_mode',
                type=ParameterType.PARAMETER_BOOL,
                description='Enable safety mode which limits robot behavior',
                read_only=False
            )
        )

        # Set up parameter callbacks
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Timer to periodically check parameters
        self.timer = self.create_timer(1.0, self.check_parameters)

        self.get_logger().info('Robot Parameters Node initialized')

    def parameter_callback(self, params):
        """Callback for parameter changes"""
        for param in params:
            self.get_logger().info(f'Parameter {param.name} changed to {param.value}')

            # Validate parameters
            if param.name == 'max_linear_velocity':
                if param.value < 0.0 or param.value > 5.0:
                    self.get_logger().warn(f'Invalid max_linear_velocity: {param.value}, clamping to [0, 5]')
                    return rclpy.node.SetParametersResult(successful=False, reason='Velocity out of range')
            elif param.name == 'max_angular_velocity':
                if param.value < 0.0 or param.value > 3.14:
                    self.get_logger().warn(f'Invalid max_angular_velocity: {param.value}, clamping to [0, π]')
                    return rclpy.node.SetParametersResult(successful=False, reason='Angular velocity out of range')

        return rclpy.node.SetParametersResult(successful=True)

    def check_parameters(self):
        """Periodically check and log parameter values"""
        robot_name = self.get_parameter('robot_name').value
        max_lin_vel = self.get_parameter('max_linear_velocity').value
        max_ang_vel = self.get_parameter('max_angular_velocity').value
        joint_limits = self.get_parameter('joint_limits').value
        safety_enabled = self.get_parameter('enable_safety_mode').value

        self.get_logger().debug(
            f'Current parameters: {robot_name}, '
            f'Lin Vel: {max_lin_vel}, Ang Vel: {max_ang_vel}, '
            f'Safety: {safety_enabled}'
        )

    def get_robot_config(self):
        """Get robot configuration as a dictionary"""
        return {
            'robot_name': self.get_parameter('robot_name').value,
            'max_linear_velocity': self.get_parameter('max_linear_velocity').value,
            'max_angular_velocity': self.get_parameter('max_angular_velocity').value,
            'joint_limits': self.get_parameter('joint_limits').value,
            'safety_enabled': self.get_parameter('enable_safety_mode').value
        }

def main(args=None):
    rclpy.init(args=args)
    node = RobotParametersNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Parameters node interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch Files

Launch files allow you to start multiple nodes at once with specific configurations. They're crucial for humanoid robotics where multiple subsystems need to be coordinated.

### Basic Launch File Structure

```python
# simple_robot_pkg/launch/basic_robot.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare launch arguments
    namespace = LaunchConfiguration('namespace', default='humanoid_robot')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    robot_name = LaunchConfiguration('robot_name', default='simple_bot')

    # Define package directories
    pkg_share = get_package_share_directory('simple_robot_pkg')

    # Robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=namespace,
        parameters=[{
            'use_sim_time': use_sim_time,
            'publish_frequency': 50.0,
        }],
        output='screen'
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        namespace=namespace,
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    # Robot command server
    robot_command_server = Node(
        package='simple_robot_pkg',
        executable='robot_command_server',
        name='robot_command_server',
        namespace=namespace,
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_name': robot_name,
        }],
        output='screen'
    )

    # Parameter node
    parameter_node = Node(
        package='simple_robot_pkg',
        executable='robot_parameters_node',
        name='robot_parameters_node',
        namespace=namespace,
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_name': robot_name,
        }],
        output='screen'
    )

    # Launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(DeclareLaunchArgument(
        'namespace',
        default_value='humanoid_robot',
        description='Robot namespace'
    ))

    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    ))

    ld.add_action(DeclareLaunchArgument(
        'robot_name',
        default_value='simple_bot',
        description='Name of the robot'
    ))

    # Add nodes
    ld.add_action(robot_state_publisher)
    ld.add_action(joint_state_publisher)
    ld.add_action(robot_command_server)
    ld.add_action(parameter_node)

    return ld
```

### Advanced Launch File with Conditional Launch

```python
# simple_robot_pkg/launch/advanced_robot.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import GroupAction
from launch_ros.actions import PushRosNamespace

def launch_setup(context, *args, **kwargs):
    """Function to create launch description based on launch arguments"""
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')
    enable_vision = LaunchConfiguration('enable_vision')
    enable_navigation = LaunchConfiguration('enable_navigation')

    # Create nodes that are always launched
    nodes = []

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )
    nodes.append(robot_state_publisher)

    # Parameter node
    parameter_node = Node(
        package='simple_robot_pkg',
        executable='robot_parameters_node',
        name='robot_parameters_node',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_name': robot_name,
        }],
        output='screen'
    )
    nodes.append(parameter_node)

    # Conditional nodes based on launch arguments
    if enable_vision.perform(context) == 'true':
        vision_node = Node(
            package='simple_robot_pkg',
            executable='vision_node',  # This would need to be created
            name='vision_node',
            parameters=[{
                'use_sim_time': use_sim_time,
            }],
            output='screen'
        )
        nodes.append(vision_node)

    if enable_navigation.perform(context) == 'true':
        navigation_node = Node(
            package='simple_robot_pkg',
            executable='navigation_node',  # This would need to be created
            name='navigation_node',
            parameters=[{
                'use_sim_time': use_sim_time,
            }],
            output='screen'
        )
        nodes.append(navigation_node)

    # Robot command server (always launched)
    robot_command_server = Node(
        package='simple_robot_pkg',
        executable='robot_command_server',
        name='robot_command_server',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )
    nodes.append(robot_command_server)

    return nodes

def generate_launch_description():
    """Generate the launch description"""
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'robot_name',
            default_value='simple_bot',
            description='Name of the robot'
        ),
        DeclareLaunchArgument(
            'enable_vision',
            default_value='false',
            description='Enable vision processing nodes'
        ),
        DeclareLaunchArgument(
            'enable_navigation',
            default_value='false',
            description='Enable navigation nodes'
        ),

        # Opaque function to conditionally add nodes
        OpaqueFunction(function=launch_setup),
    ])
```

### Launch File for Humanoid Robot Control

```python
# simple_robot_pkg/launch/humanoid_control.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    robot_description_file = LaunchConfiguration('robot_description_file',
                                               default='simple_humanoid.urdf')
    controller_config_file = LaunchConfiguration('controller_config_file',
                                                default='controllers.yaml')

    # Joint state broadcaster
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # Robot controller
    robot_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['robot_controller'],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': LaunchConfiguration('robot_description'),
        }],
        output='screen'
    )

    # Humanoid control node
    humanoid_control_node = Node(
        package='simple_robot_pkg',
        executable='humanoid_control_node',  # This would need to be created
        name='humanoid_control_node',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    # Event handler to start controllers after robot state publisher starts
    delayed_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=robot_state_publisher,
            on_start=[
                joint_state_broadcaster_spawner,
                robot_controller_spawner,
            ],
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        DeclareLaunchArgument(
            'robot_description_file',
            default_value='simple_humanoid.urdf',
            description='URDF file to use for the robot'
        ),
        DeclareLaunchArgument(
            'controller_config_file',
            default_value='controllers.yaml',
            description='Controller configuration file'
        ),

        robot_state_publisher,
        humanoid_control_node,
        delayed_controller_spawner,
    ])
```

## Quality of Service (QoS)

QoS policies allow you to configure how messages are delivered, which is critical for humanoid robotics where timing and reliability requirements vary significantly across different systems.

### QoS Profiles for Different Applications

```python
# simple_robot_pkg/qos_examples.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, LivelinessPolicy
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class QoSDemoNode(Node):
    def __init__(self):
        super().__init__('qos_demo_node')

        # Different QoS profiles for different use cases

        # 1. Critical Control Commands (Joint commands, velocity commands)
        # Need reliable delivery with minimal latency
        control_qos = QoSProfile(
            depth=1,  # Only keep the latest command
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            liveliness=LivelinessPolicy.AUTOMATIC,
            deadline=(0, 100000000),  # 100ms deadline
            lifespan=(0, 500000000),  # 500ms lifespan
        )

        # 2. Sensor Data (IMU, encoders) - need recent data but can tolerate some loss
        sensor_qos = QoSProfile(
            depth=10,  # Keep some history for processing
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )

        # 3. Logging/Debugging - keep all data
        logging_qos = QoSProfile(
            depth=1000,  # Keep extensive history
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_ALL,
        )

        # 4. Real-time control - bounded history with reliable delivery
        realtime_qos = QoSProfile(
            depth=3,  # Small bounded history
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            deadline=(0, 10000000),  # 10ms deadline for real-time
        )

        # Publishers with different QoS profiles
        self.control_publisher = self.create_publisher(Twist, 'cmd_vel', control_qos)
        self.sensor_publisher = self.create_publisher(Imu, 'imu/data', sensor_qos)
        self.logging_publisher = self.create_publisher(String, 'debug_info', logging_qos)
        self.joint_publisher = self.create_publisher(JointState, 'joint_states', realtime_qos)

        # Subscribers with matching QoS
        self.control_subscriber = self.create_subscription(
            Twist, 'cmd_vel', self.control_callback, control_qos
        )
        self.sensor_subscriber = self.create_subscription(
            Imu, 'imu/data', self.sensor_callback, sensor_qos
        )
        self.laser_subscriber = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, sensor_qos
        )

        # Timer for publishing
        self.timer = self.create_timer(0.1, self.publish_demo_data)

        self.get_logger().info('QoS Demo Node initialized with various QoS profiles')

    def control_callback(self, msg):
        """Handle control commands"""
        self.get_logger().debug(f'Received control command: linear={msg.linear.x}, angular={msg.angular.z}')

    def sensor_callback(self, msg):
        """Handle sensor data"""
        self.get_logger().debug(f'Received sensor data: orientation={msg.orientation}')

    def laser_callback(self, msg):
        """Handle laser scan data"""
        self.get_logger().debug(f'Received laser scan: {len(msg.ranges)} ranges')

    def publish_demo_data(self):
        """Publish demo data with different QoS profiles"""
        # Publish control command
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.5
        cmd_vel.angular.z = 0.1
        self.control_publisher.publish(cmd_vel)

        # Publish sensor data
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.orientation.w = 1.0
        self.sensor_publisher.publish(imu_msg)

        # Publish logging info
        debug_msg = String()
        debug_msg.data = f'Demo data published at {self.get_clock().now().nanoseconds}'
        self.logging_publisher.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    node = QoSDemoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('QoS Demo Node interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced ROS 2 Patterns for Humanoid Robotics

### State Machine Pattern

For humanoid robots that need to transition between different operational states:

```python
# simple_robot_pkg/state_machine.py
import rclpy
from rclpy.node import Node
from enum import Enum
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class RobotState(Enum):
    IDLE = "idle"
    MOVING = "moving"
    STANDBY = "standby"
    EMERGENCY_STOP = "emergency_stop"
    CHARGING = "charging"

class StateMachineNode(Node):
    def __init__(self):
        super().__init__('state_machine_node')

        # Initialize state
        self.current_state = RobotState.IDLE
        self.previous_state = None

        # Publishers
        self.state_publisher = self.create_publisher(String, 'robot_state', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers
        self.state_request_sub = self.create_subscription(
            String, 'state_request', self.state_request_callback, 10
        )

        # Timer for state processing
        self.state_timer = self.create_timer(0.1, self.state_machine_update)

        # Emergency stop subscriber
        self.emergency_sub = self.create_subscription(
            String, 'emergency_stop', self.emergency_stop_callback, 10
        )

        self.get_logger().info(f'Initial state: {self.current_state.value}')

    def state_machine_update(self):
        """Main state machine update loop"""
        # Process based on current state
        if self.current_state == RobotState.IDLE:
            self.handle_idle_state()
        elif self.current_state == RobotState.MOVING:
            self.handle_moving_state()
        elif self.current_state == RobotState.STANDBY:
            self.handle_standby_state()
        elif self.current_state == RobotState.EMERGENCY_STOP:
            self.handle_emergency_state()
        elif self.current_state == RobotState.CHARGING:
            self.handle_charging_state()

        # Publish current state
        state_msg = String()
        state_msg.data = self.current_state.value
        self.state_publisher.publish(state_msg)

    def handle_idle_state(self):
        """Handle idle state behavior"""
        # Stop robot if it was moving
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_vel)

    def handle_moving_state(self):
        """Handle moving state behavior"""
        # Continue current movement or maintain position
        pass

    def handle_standby_state(self):
        """Handle standby state behavior"""
        # Reduce power consumption, maintain minimal awareness
        pass

    def handle_emergency_state(self):
        """Handle emergency state behavior"""
        # Complete stop, maintain safety systems
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_vel)

    def handle_charging_state(self):
        """Handle charging state behavior"""
        # Maintain minimal operations, monitor battery
        pass

    def request_state_change(self, new_state):
        """Request a state change with validation"""
        if self.can_transition_to(new_state):
            self.previous_state = self.current_state
            self.current_state = new_state
            self.get_logger().info(f'State transition: {self.previous_state.value} -> {self.current_state.value}')
            return True
        else:
            self.get_logger().warn(f'Invalid state transition: {self.current_state.value} -> {new_state.value}')
            return False

    def can_transition_to(self, new_state):
        """Determine if state transition is valid"""
        valid_transitions = {
            RobotState.IDLE: [RobotState.MOVING, RobotState.STANDBY, RobotState.EMERGENCY_STOP],
            RobotState.MOVING: [RobotState.IDLE, RobotState.STANDBY, RobotState.EMERGENCY_STOP],
            RobotState.STANDBY: [RobotState.IDLE, RobotState.EMERGENCY_STOP],
            RobotState.EMERGENCY_STOP: [RobotState.IDLE],  # Can only go back to idle
            RobotState.CHARGING: [RobotState.IDLE, RobotState.EMERGENCY_STOP],
        }

        return new_state in valid_transitions.get(self.current_state, [])

    def state_request_callback(self, msg):
        """Handle state change requests"""
        try:
            requested_state = RobotState(msg.data)
            self.request_state_change(requested_state)
        except ValueError:
            self.get_logger().error(f'Invalid state requested: {msg.data}')

    def emergency_stop_callback(self, msg):
        """Handle emergency stop requests"""
        if msg.data.lower() == 'emergency' or msg.data.lower() == 'stop':
            self.request_state_change(RobotState.EMERGENCY_STOP)

def main(args=None):
    rclpy.init(args=args)
    node = StateMachineNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('State machine node interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices

1. **Use meaningful names**: Choose clear, descriptive names for nodes, topics, and services
   - Use consistent naming conventions across your robot system
   - Include robot name in topic names for multi-robot systems
   - Use forward slashes to separate namespaces

2. **Handle errors gracefully**: Always check for service availability and handle exceptions
   - Implement timeout mechanisms for service calls
   - Use try-catch blocks around critical operations
   - Provide fallback behaviors when possible

3. **Use parameters for configuration**: Avoid hardcoding values
   - Use parameter declarations with descriptions and constraints
   - Group related parameters logically
   - Validate parameter values in callbacks

4. **Log appropriately**: Use different log levels (info, warn, error) appropriately
   - Use debug for detailed information during development
   - Use info for important operational messages
   - Use warn for recoverable issues
   - Use error for serious problems

5. **Clean up resources**: Properly destroy nodes and publishers/subscribers
   - Always call destroy_node() in finally blocks
   - Close file handles and network connections
   - Use context managers when available

6. **Design for modularity**: Create nodes that perform single, well-defined functions
   - Each node should have a clear responsibility
   - Use composition rather than monolithic nodes
   - Design nodes to be reusable across different robots

7. **Consider timing and synchronization**: Humanoid robots have strict timing requirements
   - Use appropriate QoS profiles for different data types
   - Implement proper timing constraints for real-time operations
   - Synchronize data from multiple sensors when needed

8. **Plan for debugging**: Make your nodes easy to debug and monitor
   - Provide diagnostic information through topics or services
   - Use parameter servers to enable/disable debug features
   - Implement health checks and status reporting

## Next Steps

In the next chapter, we'll look at creating more complex robot applications and integrating with hardware interfaces. We'll explore how to create URDF models for humanoid robots, implement joint control systems, and connect to real hardware platforms.