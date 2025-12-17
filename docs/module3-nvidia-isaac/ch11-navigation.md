---
sidebar_position: 3
---

# Chapter 11: Isaac Navigation - Path Planning and Control

## Introduction to Isaac Navigation

Isaac Navigation builds upon the traditional ROS navigation stack but with significant enhancements leveraging NVIDIA's hardware acceleration and AI capabilities. The Isaac Navigation system includes:

- **GPU-accelerated path planning algorithms**
- **Deep learning-based obstacle avoidance**
- **Advanced trajectory optimization**
- **Real-time dynamic replanning**
- **Multi-robot coordination capabilities**

The system is designed to work seamlessly with Isaac Sim for simulation and testing, providing a complete pipeline from simulation to deployment.

## Isaac Navigation Architecture

The Isaac Navigation system follows a layered architecture:

```
┌─────────────────────────────────────────┐
│           Mission Planning              │
├─────────────────────────────────────────┤
│        Global Path Planning             │
├─────────────────────────────────────────┤
│        Local Path Planning              │
├─────────────────────────────────────────┤
│        Trajectory Generation            │
├─────────────────────────────────────────┤
│        Motion Control                   │
└─────────────────────────────────────────┘
```

## Global Path Planning with Isaac

Isaac provides GPU-accelerated global path planning algorithms:

```python
# isaac_global_planner.py
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
import numpy as np
import cv2
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
import cupy as cp  # NVIDIA GPU acceleration

class IsaacGlobalPlanner(Node):
    def __init__(self):
        super().__init__('isaac_global_planner')

        # Publishers and subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.start_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.start_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)

        self.path_pub = self.create_publisher(Path, '/plan', 10)
        self.path_marker_pub = self.create_publisher(Marker, '/global_plan_marker', 10)

        # Map data
        self.map_data = None
        self.map_resolution = 0.05
        self.map_origin = [0.0, 0.0]
        self.map_width = 0
        self.map_height = 0

        # Robot pose and goal
        self.start_pose = None
        self.goal_pose = None

        # Path planning parameters
        self.inflation_radius = 0.5  # meters
        self.cost_scaling_factor = 3.0

        self.get_logger().info('Isaac Global Planner initialized')

    def map_callback(self, msg):
        self.map_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_resolution = msg.info.resolution
        self.map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        self.map_width = msg.info.width
        self.map_height = msg.info.height

        self.get_logger().info(f'Map received: {self.map_width}x{self.map_height}, resolution: {self.map_resolution}')

    def start_callback(self, msg):
        self.start_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.get_logger().info(f'Start pose set: {self.start_pose}')

    def goal_callback(self, msg):
        self.goal_pose = [msg.pose.position.x, msg.pose.position.y]
        self.get_logger().info(f'Goal pose set: {self.goal_pose}')

        if self.map_data is not None and self.start_pose is not None:
            self.plan_path()

    def plan_path(self):
        if self.map_data is None or self.start_pose is None or self.goal_pose is None:
            return

        # Inflation of obstacles
        inflated_map = self.inflate_obstacles(self.map_data)

        # Convert poses to map coordinates
        start_map = self.world_to_map(self.start_pose)
        goal_map = self.world_to_map(self.goal_pose)

        if not self.is_valid_cell(start_map) or not self.is_valid_cell(goal_map):
            self.get_logger().warn('Start or goal pose is in invalid cell')
            return

        # GPU-accelerated path planning
        path = self.gpu_astar_planning(inflated_map, start_map, goal_map)

        if path is not None and len(path) > 0:
            # Convert path back to world coordinates
            world_path = [self.map_to_world(pos) for pos in path]
            self.publish_path(world_path)
        else:
            self.get_logger().warn('No valid path found')

    def inflate_obstacles(self, occupancy_grid):
        # Create costmap with inflated obstacles
        inflated = occupancy_grid.copy()

        # Calculate inflation radius in cells
        inflation_cells = int(self.inflation_radius / self.map_resolution)

        # Create inflation kernel
        kernel_size = 2 * inflation_cells + 1
        y, x = np.ogrid[-inflation_cells:inflation_cells+1, -inflation_cells:inflation_cells+1]
        mask = x**2 + y**2 <= inflation_cells**2

        # Apply inflation to obstacles
        obstacle_map = (occupancy_grid > 50).astype(np.uint8)  # Threshold for obstacles
        inflated_obstacles = cv2.dilate(obstacle_map, np.ones((kernel_size, kernel_size), np.uint8))

        # Combine original and inflated obstacles
        result = occupancy_grid.copy()
        result[inflated_obstacles > 0] = 100  # Mark as occupied

        return result

    def gpu_astar_planning(self, costmap, start, goal):
        """GPU-accelerated A* path planning using CuPy"""
        try:
            # Transfer map to GPU
            gpu_costmap = cp.asarray(costmap.astype(cp.float32))

            # Convert to cost values (higher values = higher cost)
            gpu_costmap = cp.where(gpu_costmap > 0, gpu_costmap, 1.0)  # Free space = 1
            gpu_costmap = cp.where(gpu_costmap > 100, cp.inf, gpu_costmap)  # Obstacles = inf

            # A* algorithm on GPU (simplified implementation)
            start_idx = start[1] * self.map_width + start[0]
            goal_idx = goal[1] * self.map_width + goal[0]

            # Initialize GPU arrays for A*
            rows, cols = gpu_costmap.shape
            open_set = cp.zeros((rows, cols), dtype=cp.bool_)
            closed_set = cp.zeros((rows, cols), dtype=cp.bool_)
            g_score = cp.full((rows, cols), cp.inf, dtype=cp.float32)
            f_score = cp.full((rows, cols), cp.inf, dtype=cp.float32)
            came_from = cp.full((rows, cols, 2), -1, dtype=cp.int32)

            # Set start position
            g_score[start[1], start[0]] = 0
            f_score[start[1], start[0]] = self.heuristic(start, goal)
            open_set[start[1], start[0]] = True

            # A* loop (simplified GPU version)
            path_found = False
            max_iterations = rows * cols  # Prevent infinite loops
            iteration = 0

            while cp.any(open_set) and iteration < max_iterations:
                # Find cell with minimum f_score
                open_cells = cp.where(open_set)
                if len(open_cells[0]) == 0:
                    break

                # For simplicity, use CPU to find minimum (in real implementation, optimize this)
                current_f_scores = f_score[open_cells].get()
                current_idx = cp.argmin(f_score[open_set])
                current = [open_cells[1][current_idx], open_cells[0][current_idx]]  # [x, y]

                if current[0] == goal[0] and current[1] == goal[1]:
                    path_found = True
                    break

                open_set[current[1], current[0]] = False
                closed_set[current[1], current[0]] = True

                # Check 8-connected neighbors
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue

                        neighbor = [current[0] + dx, current[1] + dy]

                        if not self.is_valid_cell(neighbor):
                            continue

                        if closed_set[neighbor[1], neighbor[0]]:
                            continue

                        tentative_g = g_score[current[1], current[0]] + gpu_costmap[neighbor[1], neighbor[0]].item()

                        if tentative_g < g_score[neighbor[1], neighbor[0]]:
                            came_from[neighbor[1], neighbor[0]] = current
                            g_score[neighbor[1], neighbor[0]] = tentative_g
                            f_score[neighbor[1], neighbor[0]] = tentative_g + self.heuristic(neighbor, goal)
                            open_set[neighbor[1], neighbor[0]] = True

                iteration += 1

            if path_found:
                # Reconstruct path
                path = []
                current = goal
                while current[0] != -1 and current[1] != -1:
                    path.append(current[:])
                    temp = came_from[current[1], current[0]].get()
                    current = [int(temp[0]), int(temp[1])]

                path.reverse()
                return path
            else:
                return None

        except Exception as e:
            self.get_logger().error(f'GPU path planning error: {e}')
            # Fallback to CPU implementation
            return self.cpu_astar_planning(costmap, start, goal)

    def cpu_astar_planning(self, costmap, start, goal):
        """CPU fallback for A* path planning"""
        import heapq

        def heuristic(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    neighbor = (current[0] + dx, current[1] + dy)

                    if not self.is_valid_cell(neighbor):
                        continue

                    if costmap[neighbor[1], neighbor[0]] > 99:  # Obstacle
                        continue

                    tentative_g = g_score[current] + costmap[neighbor[1], neighbor[0]]

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def heuristic(self, pos1, pos2):
        # Manhattan distance heuristic
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def is_valid_cell(self, pos):
        x, y = pos
        return (0 <= x < self.map_width and 0 <= y < self.map_height and
                self.map_data[y, x] < 99)  # Not an obstacle

    def world_to_map(self, world_pos):
        x, y = world_pos
        map_x = int((x - self.map_origin[0]) / self.map_resolution)
        map_y = int((y - self.map_origin[1]) / self.map_resolution)
        return [map_x, map_y]

    def map_to_world(self, map_pos):
        x, y = map_pos
        world_x = x * self.map_resolution + self.map_origin[0]
        world_y = y * self.map_resolution + self.map_origin[1]
        return [world_x, world_y]

    def publish_path(self, path):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for pos in path:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = pos[0]
            pose_stamped.pose.position.y = pos[1]
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            path_msg.poses.append(pose_stamped)

        self.path_pub.publish(path_msg)

        # Publish visualization marker
        self.publish_path_marker(path)

    def publish_path_marker(self, path):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'
        marker.ns = 'global_plan'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05  # Line width
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for pos in path:
            point = Point()
            point.x = pos[0]
            point.y = pos[1]
            point.z = 0.05  # Slightly above ground for visibility
            marker.points.append(point)

        self.path_marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    planner = IsaacGlobalPlanner()

    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Local Path Planning and Trajectory Generation

```python
# isaac_local_planner.py
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import LaserScan, PointCloud2
from visualization_msgs.msg import Marker
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import math

class IsaacLocalPlanner(Node):
    def __init__(self):
        super().__init__('isaac_local_planner')

        # Publishers and subscribers
        self.global_plan_sub = self.create_subscription(
            Path, '/plan', self.global_plan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.local_plan_pub = self.create_publisher(Path, '/local_plan', 10)
        self.velocity_marker_pub = self.create_publisher(Marker, '/velocity_profile', 10)

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Robot state
        self.current_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.current_vel = np.array([0.0, 0.0])  # linear, angular
        self.global_plan = []
        self.local_plan = []
        self.robot_radius = 0.3  # meters

        # Navigation parameters
        self.local_plan_time = 2.0  # seconds
        self.sim_granularity = 0.05  # meters
        self.angular_sim_granularity = 0.05  # radians
        self.max_vel_x = 0.5  # m/s
        self.max_vel_theta = 1.0  # rad/s
        self.min_vel_x = 0.1  # m/s
        self.min_vel_theta = 0.05  # rad/s
        self.max_acc_x = 2.0  # m/s^2
        self.max_acc_theta = 3.0  # rad/s^2

        # Obstacle avoidance
        self.laser_ranges = []
        self.laser_angle_min = 0.0
        self.laser_angle_max = 0.0
        self.laser_angle_increment = 0.0

        # Control parameters
        self.lookahead_dist = 1.0
        self.kp_linear = 1.0
        self.kp_angular = 2.0

        # Timer for local planning
        self.local_plan_timer = self.create_timer(0.1, self.local_plan_callback)

        self.get_logger().info('Isaac Local Planner initialized')

    def global_plan_callback(self, msg):
        self.global_plan = []
        for pose_stamped in msg.poses:
            self.global_plan.append([
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y
            ])

    def odom_callback(self, msg):
        # Update current pose
        self.current_pose[0] = msg.pose.pose.position.x
        self.current_pose[1] = msg.pose.pose.position.y

        # Convert quaternion to euler
        quat = msg.pose.pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        self.current_pose[2] = math.atan2(siny_cosp, cosy_cosp)

        # Update current velocity
        self.current_vel[0] = msg.twist.twist.linear.x
        self.current_vel[1] = msg.twist.twist.angular.z

    def scan_callback(self, msg):
        self.laser_ranges = np.array(msg.ranges)
        self.laser_angle_min = msg.angle_min
        self.laser_angle_max = msg.angle_max
        self.laser_angle_increment = msg.angle_increment

    def local_plan_callback(self):
        if len(self.global_plan) == 0:
            return

        # Generate local plan
        local_plan = self.generate_local_plan()
        if local_plan is not None:
            self.local_plan = local_plan
            self.publish_local_plan()

            # Calculate velocity command
            cmd_vel = self.calculate_velocity_command()
            if cmd_vel is not None:
                self.cmd_vel_pub.publish(cmd_vel)

    def generate_local_plan(self):
        if len(self.global_plan) == 0:
            return None

        # Find the closest point on global plan
        robot_pos = self.current_pose[:2]
        distances = [np.linalg.norm(robot_pos - np.array(wp)) for wp in self.global_plan]
        closest_idx = np.argmin(distances)

        # Get waypoints ahead of robot
        lookahead_idx = min(closest_idx + 20, len(self.global_plan) - 1)  # Look ahead 20 points
        waypoints = self.global_plan[closest_idx:lookahead_idx]

        if len(waypoints) < 2:
            return None

        # Smooth the path
        smoothed_path = self.smooth_path(waypoints)
        return smoothed_path

    def smooth_path(self, waypoints, smoothing_factor=0.1):
        if len(waypoints) < 3:
            return waypoints

        # Convert to numpy array
        points = np.array(waypoints)

        # Simple smoothing using moving average
        smoothed = np.zeros_like(points)
        for i in range(len(points)):
            start_idx = max(0, i - 1)
            end_idx = min(len(points), i + 2)
            smoothed[i] = np.mean(points[start_idx:end_idx], axis=0)

        return smoothed.tolist()

    def calculate_velocity_command(self):
        if len(self.local_plan) == 0:
            return None

        # Find the goal point on the path
        robot_pos = self.current_pose[:2]

        # Find the point on path that is closest to lookahead distance
        goal_point = None
        for i in range(len(self.local_plan) - 1, -1, -1):
            path_point = np.array(self.local_plan[i][:2])
            dist_to_robot = np.linalg.norm(robot_pos - path_point)

            if dist_to_robot >= self.lookahead_dist:
                goal_point = path_point
                break

        if goal_point is None:
            # If no point is far enough, use the last point
            goal_point = np.array(self.local_plan[-1][:2])

        # Calculate desired heading to goal point
        desired_angle = math.atan2(
            goal_point[1] - robot_pos[1],
            goal_point[0] - robot_pos[0]
        )

        # Calculate angle error
        angle_error = desired_angle - self.current_pose[2]
        # Normalize angle to [-π, π]
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        # Check for obstacles in the path
        safe_to_proceed = self.check_obstacle_free_path(robot_pos, goal_point)

        if not safe_to_proceed:
            # Emergency stop or obstacle avoidance
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = self.avoid_obstacles()
            return cmd_vel

        # PID control for linear and angular velocities
        linear_vel = min(self.max_vel_x, max(self.min_vel_x, self.kp_linear * abs(angle_error)))
        angular_vel = self.kp_angular * angle_error

        # Limit velocities
        linear_vel = min(linear_vel, self.max_vel_x)
        angular_vel = max(-self.max_vel_theta, min(self.max_vel_theta, angular_vel))

        cmd_vel = Twist()
        cmd_vel.linear.x = linear_vel if abs(angle_error) < math.pi/4 else 0.0  # Don't move forward if angle error is too large
        cmd_vel.angular.z = angular_vel

        return cmd_vel

    def check_obstacle_free_path(self, start, end):
        # Check if the path from start to end is obstacle-free using laser data
        if len(self.laser_ranges) == 0:
            return True

        # Calculate path points
        dist = np.linalg.norm(end - start)
        if dist < 0.1:  # Very close, assume safe
            return True

        # Sample points along the path
        num_samples = int(dist / 0.1)  # Sample every 10cm
        path_points = []
        for i in range(1, num_samples + 1):
            t = i / num_samples
            point = start + t * (end - start)
            path_points.append(point)

        # Transform laser points to global frame and check for collisions
        robot_angle = self.current_pose[2]
        robot_pos = self.current_pose[:2]

        for range_idx, range_val in enumerate(self.laser_ranges):
            if range_val < 0.1 or range_val > 10.0:  # Invalid range
                continue

            # Calculate angle of this laser beam
            angle = self.laser_angle_min + range_idx * self.laser_angle_increment + robot_angle
            # Calculate global position of obstacle
            obstacle_global = robot_pos + np.array([
                range_val * math.cos(angle),
                range_val * math.sin(angle)
            ])

            # Check if any path point is too close to this obstacle
            for path_point in path_points:
                if np.linalg.norm(path_point - obstacle_global) < self.robot_radius:
                    return False  # Path is blocked

        return True

    def avoid_obstacles(self):
        # Simple obstacle avoidance: turn away from obstacles
        if len(self.laser_ranges) == 0:
            return 0.0

        # Look at the front 90 degrees
        center_idx = len(self.laser_ranges) // 2
        front_start = max(0, center_idx - len(self.laser_ranges) // 8)  # 45 degrees left
        front_end = min(len(self.laser_ranges), center_idx + len(self.laser_ranges) // 8)  # 45 degrees right

        front_ranges = self.laser_ranges[front_start:front_end]

        # Check for close obstacles
        min_front_dist = np.min(front_ranges[front_ranges > 0]) if np.any(front_ranges > 0) else float('inf')

        if min_front_dist < 0.5:  # Obstacle within 50cm
            # Turn away from the closest obstacle
            min_idx = np.argmin(front_ranges)
            if min_idx < len(front_ranges) // 2:
                # Obstacle on the left, turn right
                return -0.5
            else:
                # Obstacle on the right, turn left
                return 0.5

        return 0.0  # No obstacle detected

    def publish_local_plan(self):
        if len(self.local_plan) == 0:
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for pos in self.local_plan:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = pos[0]
            pose_stamped.pose.position.y = pos[1]
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            path_msg.poses.append(pose_stamped)

        self.local_plan_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    planner = IsaacLocalPlanner()

    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac Navigation Behavior Trees

Isaac Navigation uses behavior trees for complex navigation decision-making:

```python
# navigation_behavior_tree.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import math
import time

class BehaviorNode:
    """Base class for behavior tree nodes"""
    def __init__(self, name):
        self.name = name
        self.status = "IDLE"  # IDLE, RUNNING, SUCCESS, FAILURE

    def tick(self):
        """Execute the behavior and return status"""
        raise NotImplementedError

class SequenceNode(BehaviorNode):
    """Executes children in sequence until one fails"""
    def __init__(self, name, children):
        super().__init__(name)
        self.children = children
        self.current_child_idx = 0

    def tick(self):
        for i in range(self.current_child_idx, len(self.children)):
            child_status = self.children[i].tick()

            if child_status == "FAILURE":
                self.current_child_idx = 0
                self.status = "FAILURE"
                return self.status
            elif child_status == "RUNNING":
                self.current_child_idx = i
                self.status = "RUNNING"
                return self.status
            # If SUCCESS, continue to next child

        # All children succeeded
        self.current_child_idx = 0
        self.status = "SUCCESS"
        return self.status

class SelectorNode(BehaviorNode):
    """Executes children until one succeeds"""
    def __init__(self, name, children):
        super().__init__(name)
        self.children = children
        self.current_child_idx = 0

    def tick(self):
        for i in range(self.current_child_idx, len(self.children)):
            child_status = self.children[i].tick()

            if child_status == "SUCCESS":
                self.current_child_idx = 0
                self.status = "SUCCESS"
                return self.status
            elif child_status == "RUNNING":
                self.current_child_idx = i
                self.status = "RUNNING"
                return self.status
            # If FAILURE, continue to next child

        # All children failed
        self.current_child_idx = 0
        self.status = "FAILURE"
        return self.status

class IsPathValid(BehaviorNode):
    """Check if global path is valid"""
    def __init__(self, name, nav_system):
        super().__init__(name)
        self.nav_system = nav_system

    def tick(self):
        if len(self.nav_system.global_plan) > 0:
            self.status = "SUCCESS"
        else:
            self.status = "FAILURE"
        return self.status

class IsGoalReached(BehaviorNode):
    """Check if robot has reached goal"""
    def __init__(self, name, nav_system):
        super().__init__(name)
        self.nav_system = nav_system
        self.goal_tolerance = 0.2  # meters

    def tick(self):
        if len(self.nav_system.global_plan) == 0:
            self.status = "SUCCESS"  # No path means we're done
            return self.status

        goal = self.nav_system.global_plan[-1]
        robot_pos = self.nav_system.current_pose[:2]
        dist_to_goal = math.sqrt((goal[0] - robot_pos[0])**2 + (goal[1] - robot_pos[1])**2)

        if dist_to_goal <= self.goal_tolerance:
            self.status = "SUCCESS"
        else:
            self.status = "FAILURE"
        return self.status

class FollowPath(BehaviorNode):
    """Execute path following"""
    def __init__(self, name, nav_system):
        super().__init__(name)
        self.nav_system = nav_system

    def tick(self):
        # Calculate velocity command to follow path
        cmd_vel = self.nav_system.calculate_velocity_command()
        if cmd_vel is not None:
            self.nav_system.cmd_vel_pub.publish(cmd_vel)
            self.status = "RUNNING"
        else:
            self.status = "FAILURE"
        return self.status

class ReplanPath(BehaviorNode):
    """Request new path planning"""
    def __init__(self, name, nav_system):
        super().__init__(name)
        self.nav_system = nav_system

    def tick(self):
        # In a real implementation, this would call the global planner
        # For now, just indicate that replanning was requested
        self.nav_system.get_logger().info("Requesting path replan")
        self.status = "SUCCESS"
        return self.status

class IsaacBehaviorTreePlanner(Node):
    def __init__(self):
        super().__init__('isaac_behavior_tree_planner')

        # Publishers and subscribers
        self.global_plan_sub = self.create_subscription(
            Path, '/plan', self.global_plan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Navigation state
        self.global_plan = []
        self.current_pose = [0.0, 0.0, 0.0]
        self.laser_ranges = []

        # Create behavior tree
        self.create_behavior_tree()

        # Timer for behavior tree execution
        self.bt_timer = self.create_timer(0.1, self.execute_behavior_tree)

        self.get_logger().info('Isaac Behavior Tree Planner initialized')

    def create_behavior_tree(self):
        """Create the navigation behavior tree"""
        # Path following sequence
        path_following_sequence = SequenceNode("PathFollowingSequence", [
            IsPathValid("IsPathValid", self),
            FollowPath("FollowPath", self)
        ])

        # Goal checking
        goal_check = IsGoalReached("IsGoalReached", self)

        # Main navigation selector
        self.main_tree = SelectorNode("NavigationTree", [
            goal_check,  # Check if goal is reached first
            path_following_sequence  # If not at goal, follow path
        ])

    def global_plan_callback(self, msg):
        self.global_plan = []
        for pose_stamped in msg.poses:
            self.global_plan.append([
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y
            ])

    def odom_callback(self, msg):
        self.current_pose[0] = msg.pose.pose.position.x
        self.current_pose[1] = msg.pose.pose.position.y

        # Convert quaternion to euler
        quat = msg.pose.pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        self.current_pose[2] = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        self.laser_ranges = list(msg.ranges)

    def execute_behavior_tree(self):
        """Execute the behavior tree"""
        status = self.main_tree.tick()
        self.get_logger().debug(f'Behavior tree status: {status}')

    def calculate_velocity_command(self):
        """Calculate velocity command based on current state"""
        if len(self.global_plan) == 0:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            return cmd_vel

        # Simple path following
        goal = self.global_plan[0]  # Use first waypoint
        robot_pos = self.current_pose[:2]

        # Calculate desired angle to goal
        desired_angle = math.atan2(
            goal[1] - robot_pos[1],
            goal[0] - robot_pos[0]
        )

        # Calculate angle error
        angle_error = desired_angle - self.current_pose[2]
        # Normalize angle
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        # Create velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.3  # Simple constant forward speed
        cmd_vel.angular.z = 1.5 * angle_error  # P controller for rotation

        # Limit angular velocity
        cmd_vel.angular.z = max(-1.0, min(1.0, cmd_vel.angular.z))

        return cmd_vel

def main(args=None):
    rclpy.init(args=args)
    planner = IsaacBehaviorTreePlanner()

    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Navigation Monitoring and Recovery

```python
# navigation_monitor.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Bool
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
import time
import math

class IsaacNavigationMonitor(Node):
    def __init__(self):
        super().__init__('isaac_navigation_monitor')

        # Publishers and subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.path_sub = self.create_subscription(
            Path, '/plan', self.path_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        self.recovery_pub = self.create_publisher(Bool, '/recovery_trigger', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)
        self.diagnostics_pub = self.create_publisher(DiagnosticArray, '/navigation_diagnostics', 10)

        # Robot state
        self.current_pose = [0.0, 0.0, 0.0]
        self.current_vel = [0.0, 0.0]  # linear, angular
        self.last_cmd_vel = [0.0, 0.0]
        self.last_cmd_time = 0.0
        self.global_path = []
        self.laser_ranges = []

        # Navigation monitoring
        self.oscillation_threshold = 0.1  # m/s
        self.oscillation_time_window = 5.0  # seconds
        self.oscillation_history = []

        self.stuck_threshold = 0.05  # m/s
        self.stuck_time_window = 10.0  # seconds
        self.stuck_history = []

        self.goal_distance_threshold = 0.5  # meters

        # Recovery states
        self.recovery_active = False
        self.recovery_start_time = 0.0
        self.recovery_type = None

        # Timer for monitoring
        self.monitor_timer = self.create_timer(0.5, self.monitor_navigation)

        self.get_logger().info('Isaac Navigation Monitor initialized')

    def odom_callback(self, msg):
        self.current_pose[0] = msg.pose.pose.position.x
        self.current_pose[1] = msg.pose.pose.position.y

        # Convert quaternion to euler
        quat = msg.pose.pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        self.current_pose[2] = math.atan2(siny_cosp, cosy_cosp)

        self.current_vel[0] = msg.twist.twist.linear.x
        self.current_vel[1] = msg.twist.twist.angular.z

    def cmd_vel_callback(self, msg):
        self.last_cmd_vel = [msg.linear.x, msg.angular.z]
        self.last_cmd_time = time.time()

    def path_callback(self, msg):
        self.global_path = []
        for pose_stamped in msg.poses:
            self.global_path.append([
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y
            ])

    def scan_callback(self, msg):
        self.laser_ranges = list(msg.ranges)

    def monitor_navigation(self):
        current_time = time.time()

        # Check for oscillation
        self.check_oscillation()

        # Check if robot is stuck
        self.check_stuck()

        # Check for obstacles too close
        self.check_obstacles()

        # Check if robot is near goal but not progressing
        self.check_goal_progress()

        # Publish diagnostics
        self.publish_diagnostics()

    def check_oscillation(self):
        # Track velocity oscillations
        current_vel_linear = abs(self.current_vel[0])
        current_vel_angular = abs(self.current_vel[1])

        self.oscillation_history.append((time.time(), current_vel_angular))

        # Keep only the last time window of data
        cutoff_time = time.time() - self.oscillation_time_window
        self.oscillation_history = [
            (t, v) for t, v in self.oscillation_history if t > cutoff_time
        ]

        # Check for excessive angular oscillation
        if len(self.oscillation_history) > 10:  # Need sufficient data
            avg_angular_vel = sum(v for t, v in self.oscillation_history) / len(self.oscillation_history)
            max_angular_vel = max(v for t, v in self.oscillation_history)

            if max_angular_vel > 2.0 * avg_angular_vel and max_angular_vel > self.oscillation_threshold:
                self.get_logger().warn('Oscillation detected, triggering recovery')
                self.trigger_recovery('oscillation')

    def check_stuck(self):
        # Track if robot is not making progress
        current_speed = abs(self.current_vel[0])
        current_time = time.time()

        self.stuck_history.append((current_time, current_speed))

        # Keep only the last time window of data
        cutoff_time = current_time - self.stuck_time_window
        self.stuck_history = [
            (t, v) for t, v in self.stuck_history if t > cutoff_time
        ]

        # Check if average speed is too low
        if len(self.stuck_history) > 5:
            avg_speed = sum(v for t, v in self.stuck_history) / len(self.stuck_history)
            if avg_speed < self.stuck_threshold and abs(self.last_cmd_vel[0]) > 0.1:
                self.get_logger().warn('Robot appears stuck, triggering recovery')
                self.trigger_recovery('stuck')

    def check_obstacles(self):
        if len(self.laser_ranges) == 0:
            return

        # Check for very close obstacles (emergency stop threshold)
        min_range = min(r for r in self.laser_ranges if r > 0 and r < 10) if any(r > 0 and r < 10 for r in self.laser_ranges) else float('inf')

        if min_range < 0.3:  # Less than 30cm
            self.get_logger().error('Very close obstacle detected, triggering emergency stop')
            self.trigger_emergency_stop()

    def check_goal_progress(self):
        if len(self.global_path) == 0:
            return

        goal = self.global_path[-1]
        robot_pos = self.current_pose[:2]
        dist_to_goal = math.sqrt((goal[0] - robot_pos[0])**2 + (goal[1] - robot_pos[1])**2)

        # If very close to goal but still receiving motion commands, check if making progress
        if dist_to_goal < self.goal_distance_threshold and abs(self.last_cmd_vel[0]) > 0.1:
            # Check if we've been in this area for too long
            # This would require tracking position over time
            pass

    def trigger_recovery(self, recovery_type):
        if not self.recovery_active:
            self.recovery_active = True
            self.recovery_type = recovery_type
            self.recovery_start_time = time.time()

            recovery_msg = Bool()
            recovery_msg.data = True
            self.recovery_pub.publish(recovery_msg)

            self.get_logger().info(f'Triggered {recovery_type} recovery')

    def trigger_emergency_stop(self):
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)

        # Also send zero velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        # This would typically be published to a safety-critical topic

        self.get_logger().error('Emergency stop triggered')

    def publish_diagnostics(self):
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        # Navigation status diagnostic
        nav_diag = DiagnosticStatus()
        nav_diag.name = 'Navigation Status'
        nav_diag.level = DiagnosticStatus.OK
        nav_diag.message = 'Navigation system operational'

        # Add key values
        nav_diag.values.append(KeyValue(key='Linear Velocity', value=f'{self.current_vel[0]:.3f}'))
        nav_diag.values.append(KeyValue(key='Angular Velocity', value=f'{self.current_vel[1]:.3f}'))
        nav_diag.values.append(KeyValue(key='Recovery Active', value=str(self.recovery_active)))
        nav_diag.values.append(KeyValue(key='Recovery Type', value=self.recovery_type or 'None'))

        if self.recovery_active:
            nav_diag.level = DiagnosticStatus.WARN
            nav_diag.message = f'Recovery active: {self.recovery_type}'

        diag_array.status.append(nav_diag)
        self.diagnostics_pub.publish(diag_array)

def main(args=None):
    rclpy.init(args=args)
    monitor = IsaacNavigationMonitor()

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

## Launch File for Complete Navigation System

```python
# launch/isaac_navigation_system.launch.py
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    enable_behavior_tree = LaunchConfiguration('enable_behavior_tree', default='true')
    enable_monitoring = LaunchConfiguration('enable_monitoring', default='true')

    # Package names
    pkg_simple_robot = FindPackageShare('simple_robot_pkg').find('simple_robot_pkg')

    # Create launch description
    ld = LaunchDescription()

    # Declare launch arguments
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'))

    ld.add_action(DeclareLaunchArgument(
        'enable_behavior_tree',
        default_value='true',
        description='Enable behavior tree navigation'))

    ld.add_action(DeclareLaunchArgument(
        'enable_monitoring',
        default_value='true',
        description='Enable navigation monitoring'))

    # Isaac Global Planner
    global_planner = Node(
        package='simple_robot_pkg',
        executable='isaac_global_planner',
        name='isaac_global_planner',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Isaac Local Planner
    local_planner = Node(
        package='simple_robot_pkg',
        executable='isaac_local_planner',
        name='isaac_local_planner',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Isaac Behavior Tree Planner (conditional)
    behavior_tree_planner = Node(
        condition=IfCondition(enable_behavior_tree),
        package='simple_robot_pkg',
        executable='isaac_behavior_tree_planner',
        name='isaac_behavior_tree_planner',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Navigation Monitor (conditional)
    nav_monitor = Node(
        condition=IfCondition(enable_monitoring),
        package='simple_robot_pkg',
        executable='isaac_navigation_monitor',
        name='isaac_navigation_monitor',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Add all nodes to launch description
    ld.add_action(global_planner)
    ld.add_action(local_planner)
    ld.add_action(behavior_tree_planner)
    ld.add_action(nav_monitor)

    return ld
```

## Next Steps

In the next chapter, we'll explore Isaac's capabilities for Sim-to-Real transfer, focusing on how to ensure that behaviors learned in simulation work effectively on real robots. This includes domain randomization, sensor fusion, and adaptive control strategies.