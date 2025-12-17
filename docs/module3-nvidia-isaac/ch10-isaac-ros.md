---
sidebar_position: 2
---

# Chapter 10: Isaac ROS - Perception and Navigation

## Introduction to Isaac ROS

Isaac ROS is NVIDIA's collection of hardware-accelerated perception and navigation packages designed to run on ROS 2. These packages leverage NVIDIA GPUs for accelerated processing, providing significant performance improvements over traditional CPU-based implementations.

The Isaac ROS package ecosystem includes:
- **Stereo Image Pipeline**: Accelerated stereo vision processing
- **Segmentation**: Real-time semantic and instance segmentation
- **Point Cloud Utils**: GPU-accelerated point cloud processing
- **Apriltag**: GPU-accelerated AprilTag detection
- **DNN Inference**: Optimized deep learning inference
- **Occupancy Grids**: GPU-accelerated mapping
- **Visual SLAM**: Visual Simultaneous Localization and Mapping

## Isaac ROS Stereo Image Pipeline

The stereo image pipeline in Isaac ROS provides hardware-accelerated stereo processing for depth estimation:

### Installation and Setup

```bash
# Install Isaac ROS stereo packages
sudo apt install ros-humble-isaac-ros-stereo-image-pipeline
sudo apt install ros-humble-isaac-ros-rectify
sudo apt install ros-humble-isaac-ros-stereo-disparity
```

### Stereo Rectification

Stereo rectification is crucial for accurate depth estimation:

```python
# stereo_rectification_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge
import numpy as np
import cv2
from stereo_image_proc_py import StereoProcessor

class IsaacStereoRectificationNode(Node):
    def __init__(self):
        super().__init__('isaac_stereo_rectification_node')

        # Publishers and subscribers
        self.left_sub = self.create_subscription(
            Image, '/camera/left/image_raw', self.left_callback, 10)
        self.right_sub = self.create_subscription(
            Image, '/camera/right/image_raw', self.right_callback, 10)
        self.left_info_sub = self.create_subscription(
            CameraInfo, '/camera/left/camera_info', self.left_info_callback, 10)
        self.right_info_sub = self.create_subscription(
            CameraInfo, '/camera/right/camera_info', self.right_info_callback, 10)

        self.left_rect_pub = self.create_publisher(Image, '/camera/left/image_rect', 10)
        self.right_rect_pub = self.create_publisher(Image, '/camera/right/image_rect', 10)
        self.disparity_pub = self.create_publisher(DisparityImage, '/disparity_map', 10)

        # CV Bridge
        self.cv_bridge = CvBridge()

        # Camera parameters
        self.left_camera_info = None
        self.right_camera_info = None
        self.rectification_initialized = False
        self.left_map1 = None
        self.left_map2 = None
        self.right_map1 = None
        self.right_map2 = None

        # Latest images
        self.left_image = None
        self.right_image = None

        # Timer for processing
        self.process_timer = self.create_timer(0.1, self.process_stereo)

        self.get_logger().info('Isaac Stereo Rectification Node initialized')

    def left_info_callback(self, msg):
        if self.left_camera_info is None:
            self.left_camera_info = msg
            self.initialize_rectification()

    def right_info_callback(self, msg):
        if self.right_camera_info is None:
            self.right_camera_info = msg
            self.initialize_rectification()

    def initialize_rectification(self):
        if self.left_camera_info is None or self.right_camera_info is None:
            return

        # Extract camera parameters
        left_K = np.array(self.left_camera_info.k).reshape(3, 3)
        right_K = np.array(self.right_camera_info.k).reshape(3, 3)
        left_D = np.array(self.left_camera_info.d)
        right_D = np.array(self.right_camera_info.d)

        # Get R and T from stereo calibration (in a real system, these come from calibration)
        # For this example, we'll use placeholder values
        R = np.eye(3)  # Rotation matrix (identity for now)
        T = np.array([-0.1, 0, 0])  # Translation vector (baseline = 10cm)

        # Image size
        size = (self.left_camera_info.width, self.left_camera_info.height)

        # Compute rectification maps
        R1, R2, P1, P2 = cv2.stereoRectify(
            left_K, left_D, right_K, right_D, size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
        )[:4]

        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            left_K, left_D, R1, P1, size, cv2.CV_32FC1)
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            right_K, right_D, R2, P2, size, cv2.CV_32FC1)

        self.rectification_initialized = True
        self.get_logger().info('Stereo rectification initialized')

    def left_callback(self, msg):
        self.left_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def right_callback(self, msg):
        self.right_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def process_stereo(self):
        if not self.rectification_initialized:
            return

        if self.left_image is None or self.right_image is None:
            return

        # Rectify images
        left_rect = cv2.remap(self.left_image, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(self.right_image, self.right_map1, self.right_map2, cv2.INTER_LINEAR)

        # Publish rectified images
        left_rect_msg = self.cv_bridge.cv2_to_imgmsg(left_rect, encoding='passthrough')
        left_rect_msg.header = self.left_image.header  # Copy header
        self.left_rect_pub.publish(left_rect_msg)

        right_rect_msg = self.cv_bridge.cv2_to_imgmsg(right_rect, encoding='passthrough')
        right_rect_msg.header = self.right_image.header  # Copy header
        self.right_rect_pub.publish(right_rect_msg)

        # Compute disparity using GPU-accelerated method (simulated here)
        # In Isaac ROS, this would use GPU-accelerated stereo matching
        disparity = self.compute_disparity_gpu(left_rect, right_rect)

        # Create and publish disparity image
        disp_msg = DisparityImage()
        disp_msg.header.stamp = self.get_clock().now().to_msg()
        disp_msg.header.frame_id = 'camera_link'
        disp_msg.image = self.cv_bridge.cv2_to_imgmsg(disparity, encoding='32FC1')
        disp_msg.f = P1[0, 0] if 'P1' in locals() else 500.0  # Focal length
        disp_msg.T = abs(T[0]) if 'T' in locals() else 0.1     # Baseline

        self.disparity_pub.publish(disp_msg)

    def compute_disparity_gpu(self, left, right):
        # In Isaac ROS, this would use GPU-accelerated stereo matching
        # For simulation, we'll use a CPU-based method
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=9,
            P1=8 * 3 * 9**2,
            P2=32 * 3 * 9**2,
        )

        # Convert to grayscale if needed
        if len(left.shape) == 3:
            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left
            right_gray = right

        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        return disparity

def main(args=None):
    rclpy.init(args=args)
    node = IsaacStereoRectificationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac ROS Segmentation

Isaac ROS provides GPU-accelerated segmentation capabilities:

### Installation

```bash
# Install Isaac ROS segmentation packages
sudo apt install ros-humble-isaac-ros-segmentation
sudo apt install ros-humble-isaac-ros-dnn-inference
```

### Semantic Segmentation Node

```python
# segmentation_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50

class IsaacSegmentationNode(Node):
    def __init__(self):
        super().__init__('isaac_segmentation_node')

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.segmentation_pub = self.create_publisher(Image, '/segmentation/mask', 10)
        self.detections_pub = self.create_publisher(Detection2DArray, '/segmentation/detections', 10)

        # CV Bridge
        self.cv_bridge = CvBridge()

        # Initialize segmentation model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_segmentation_model()
        self.model.to(self.device)
        self.model.eval()

        # COCO class names (for visualization)
        self.coco_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        self.get_logger().info('Isaac Segmentation Node initialized')

    def load_segmentation_model(self):
        # Load DeepLabV3 model pre-trained on COCO
        model = deeplabv3_resnet50(pretrained=True)
        return model

    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run segmentation
            segmented_image, detections = self.segment_image(cv_image)

            # Publish segmentation mask
            mask_msg = self.cv_bridge.cv2_to_imgmsg(segmented_image, encoding='mono8')
            mask_msg.header = msg.header
            self.segmentation_pub.publish(mask_msg)

            # Publish detections
            detections_msg = self.create_detections_message(detections, msg.header)
            self.detections_pub.publish(detections_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def segment_image(self, image):
        # Preprocess image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((520, 520)),  # DeepLabV3 default size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
            output_predictions = output.argmax(0).cpu().numpy()

        # Create colored segmentation mask
        mask = self.create_segmentation_mask(output_predictions, image.shape[:2])

        # Extract detections
        detections = self.extract_detections(output_predictions)

        return mask, detections

    def create_segmentation_mask(self, predictions, original_shape):
        # Create a colored mask based on class predictions
        h, w = original_shape[:2]
        mask = np.zeros((h, w, 3), dtype=np.uint8)

        # Generate colors for each class
        colors = self.generate_colors(len(self.coco_names))

        for class_id in np.unique(predictions):
            if class_id < len(colors):
                mask[predictions == class_id] = colors[class_id]

        return cv2.resize(mask, (w, h))

    def generate_colors(self, num_classes):
        # Generate distinct colors for each class
        colors = []
        for i in range(num_classes):
            # Simple color generation (in practice, use a more sophisticated method)
            hue = i * 137.5 % 360  # Golden angle for good distribution
            colors.append(self.hsv_to_rgb(hue, 0.8, 0.8))
        return colors

    def hsv_to_rgb(self, h, s, v):
        # Convert HSV to RGB
        h = h / 360.0
        r, g, b = cv2.cvtColor(
            np.array([[[h, s, v]]], dtype=np.float32),
            cv2.COLOR_HSV2BGR
        )[0, 0]
        return (int(b * 255), int(g * 255), int(r * 255))

    def extract_detections(self, predictions):
        # Extract bounding boxes and class information from segmentation
        detections = []
        unique_classes = np.unique(predictions)

        for class_id in unique_classes:
            if class_id == 0:  # Skip background
                continue

            if class_id < len(self.coco_names):
                class_name = self.coco_names[class_id]
                mask = (predictions == class_id).astype(np.uint8)

                # Find contours to get bounding boxes
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = cv2.contourArea(contour)

                    # Only include large enough detections
                    if area > 100:  # Minimum area threshold
                        detections.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'bbox': (x, y, w, h),
                            'area': area
                        })

        return detections

    def create_detections_message(self, detections, header):
        detections_msg = Detection2DArray()
        detections_msg.header = header

        for det in detections:
            detection = Detection2D()
            detection.header = header

            # Bounding box
            detection.bbox.center.x = det['bbox'][0] + det['bbox'][2] / 2
            detection.bbox.center.y = det['bbox'][1] + det['bbox'][3] / 2
            detection.bbox.size_x = det['bbox'][2]
            detection.bbox.size_y = det['bbox'][3]

            # Classification
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(det['class_name'])
            hypothesis.hypothesis.score = 0.9  # Placeholder confidence

            detection.results.append(hypothesis)
            detections_msg.detections.append(detection)

        return detections_msg

def main(args=None):
    rclpy.init(args=args)
    node = IsaacSegmentationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac ROS Navigation Stack

Isaac ROS includes optimized navigation capabilities:

### Installation

```bash
# Install Isaac ROS navigation packages
sudo apt install ros-humble-isaac-ros-nav2
sudo apt install ros-humble-isaac-ros-occupancy-grid-localizer
sudo apt install ros-humble-isaac-ros-visual-slam
```

### Occupancy Grid Localizer

```python
# occupancy_grid_localizer.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from tf2_geometry_msgs import PoseStamped as TF2PoseStamped
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import binary_dilation

class IsaacOccupancyGridLocalizer(Node):
    def __init__(self):
        super().__init__('isaac_occupancy_grid_localizer')

        # Publishers and subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/amcl_pose', 10)

        # TF broadcaster and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Robot state
        self.current_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.current_odom = None
        self.map = None
        self.map_resolution = 0.05
        self.map_origin = np.array([0.0, 0.0])

        # Particle filter parameters
        self.num_particles = 1000
        self.particles = np.zeros((self.num_particles, 3))  # x, y, theta
        self.weights = np.ones(self.num_particles) / self.num_particles

        # Motion model noise
        self.motion_noise = [0.1, 0.1, 0.05]  # x, y, theta

        # Timer for localization
        self.localization_timer = self.create_timer(0.1, self.localize)

        self.get_logger().info('Isaac Occupancy Grid Localizer initialized')

    def map_callback(self, msg):
        self.map = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_resolution = msg.info.resolution
        self.map_origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        self.get_logger().info(f'Map received: {msg.info.width}x{msg.info.height}, resolution: {self.map_resolution}')

    def odom_callback(self, msg):
        self.current_odom = msg

    def scan_callback(self, msg):
        # Process laser scan for localization
        if self.map is not None:
            # Convert laser scan to points in robot frame
            angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
            valid_ranges = []
            valid_angles = []

            for i, r in enumerate(msg.ranges):
                if msg.range_min <= r <= msg.range_max:
                    valid_ranges.append(r)
                    valid_angles.append(angles[i])

            if len(valid_ranges) > 0:
                # Convert to Cartesian coordinates
                x_points = np.array(valid_ranges) * np.cos(valid_angles)
                y_points = np.array(valid_ranges) * np.sin(valid_angles)

                # Transform to map frame using current pose estimate
                self.scan_points_map = np.vstack([x_points, y_points])

    def initialize_particles(self, initial_pose, covariance):
        # Initialize particles around initial pose
        for i in range(self.num_particles):
            self.particles[i, 0] = np.random.normal(initial_pose[0], np.sqrt(covariance[0]))
            self.particles[i, 1] = np.random.normal(initial_pose[1], np.sqrt(covariance[7]))  # cov[7] = y variance
            self.particles[i, 2] = np.random.normal(initial_pose[2], np.sqrt(covariance[35]))  # cov[35] = theta variance

    def motion_model(self, prev_pose, curr_odom, prev_odom):
        if prev_odom is None:
            return prev_pose

        # Calculate motion based on odometry change
        dx = curr_odom.pose.pose.position.x - prev_odom.pose.pose.position.x
        dy = curr_odom.pose.pose.position.y - prev_odom.pose.pose.position.y
        dtheta = 2 * np.arctan2(
            curr_odom.pose.pose.orientation.z,
            curr_odom.pose.pose.orientation.w
        ) - 2 * np.arctan2(
            prev_odom.pose.pose.orientation.z,
            prev_odom.pose.pose.orientation.w
        )

        # Add noise
        dx += np.random.normal(0, self.motion_noise[0])
        dy += np.random.normal(0, self.motion_noise[1])
        dtheta += np.random.normal(0, self.motion_noise[2])

        # Update pose
        new_pose = prev_pose.copy()
        new_pose[0] += dx * np.cos(prev_pose[2]) - dy * np.sin(prev_pose[2])
        new_pose[1] += dx * np.sin(prev_pose[2]) + dy * np.cos(prev_pose[2])
        new_pose[2] += dtheta

        # Normalize angle
        new_pose[2] = (new_pose[2] + np.pi) % (2 * np.pi) - np.pi

        return new_pose

    def sensor_model(self, particle_pose, scan_points):
        if self.map is None or len(scan_points[0]) == 0:
            return 1.0  # Return high probability if no data

        # Transform scan points to map frame
        cos_theta = np.cos(particle_pose[2])
        sin_theta = np.sin(particle_pose[2])

        transformed_x = particle_pose[0] + scan_points[0] * cos_theta - scan_points[1] * sin_theta
        transformed_y = particle_pose[1] + scan_points[0] * sin_theta + scan_points[1] * cos_theta

        # Convert to map coordinates
        map_x = ((transformed_x - self.map_origin[0]) / self.map_resolution).astype(int)
        map_y = ((transformed_y - self.map_origin[1]) / self.map_resolution).astype(int)

        # Check if points are within map bounds
        valid_indices = (map_x >= 0) & (map_x < self.map.shape[1]) & \
                       (map_y >= 0) & (map_y < self.map.shape[0])

        if not np.any(valid_indices):
            return 0.001  # Very low probability if no valid points

        # Calculate probability based on occupancy values
        # Free space (0) should have high probability, occupied space (100) low probability
        valid_map_x = map_x[valid_indices]
        valid_map_y = map_y[valid_indices]

        occupancy_values = self.map[valid_map_y, valid_map_x]

        # Calculate likelihood: lower probability for occupied cells
        probabilities = np.where(occupancy_values > 50, 0.1, 0.9)  # 0.1 for occupied, 0.9 for free
        likelihood = np.mean(probabilities) if len(probabilities) > 0 else 0.5

        return likelihood

    def resample_particles(self):
        # Systematic resampling
        new_particles = np.zeros_like(self.particles)

        # Calculate cumulative weights
        cumulative_weights = np.cumsum(self.weights)

        # Sample particles
        step = 1.0 / self.num_particles
        start = np.random.uniform(0, step)

        i, j = 0, 0
        while i < self.num_particles:
            if start + j * step <= cumulative_weights[i]:
                new_particles[j] = self.particles[i]
                j += 1
            else:
                i += 1

        self.particles = new_particles
        self.weights.fill(1.0 / self.num_particles)

    def localize(self):
        if self.map is None or self.current_odom is None:
            return

        # Update particles with motion model
        if hasattr(self, 'prev_odom'):
            for i in range(self.num_particles):
                self.particles[i] = self.motion_model(
                    self.particles[i],
                    self.current_odom,
                    self.prev_odom
                )

        # Update particle weights based on sensor model
        if hasattr(self, 'scan_points_map'):
            for i in range(self.num_particles):
                weight = self.sensor_model(self.particles[i], self.scan_points_map)
                self.weights[i] *= weight

            # Normalize weights
            total_weight = np.sum(self.weights)
            if total_weight > 0:
                self.weights /= total_weight
            else:
                self.weights.fill(1.0 / self.num_particles)

            # Resample if effective sample size is low
            effective_samples = 1.0 / np.sum(self.weights**2)
            if effective_samples < self.num_particles / 2.0:
                self.resample_particles()

        # Calculate estimated pose (weighted average)
        estimated_pose = np.average(self.particles, axis=0, weights=self.weights)

        # Publish estimated pose
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.pose.position.x = float(estimated_pose[0])
        pose_msg.pose.pose.position.y = float(estimated_pose[1])
        pose_msg.pose.pose.position.z = 0.0

        # Convert angle to quaternion
        from math import sin, cos
        cy = cos(estimated_pose[2] * 0.5)
        sy = sin(estimated_pose[2] * 0.5)
        pose_msg.pose.pose.orientation.z = float(sy)
        pose_msg.pose.pose.orientation.w = float(cy)

        # Set covariance (simplified)
        pose_msg.pose.covariance[0] = 0.1   # x variance
        pose_msg.pose.covariance[7] = 0.1   # y variance
        pose_msg.pose.covariance[35] = 0.1  # theta variance

        self.pose_pub.publish(pose_msg)

        # Store previous odometry for next iteration
        self.prev_odom = self.current_odom

def main(args=None):
    rclpy.init(args=args)
    node = IsaacOccupancyGridLocalizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac ROS Visual SLAM

Isaac ROS includes GPU-accelerated Visual SLAM capabilities:

```python
# visual_slam_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Path
from cv_bridge import CvBridge
import numpy as np
import cv2
from collections import deque

class IsaacVisualSLAMNode(Node):
    def __init__(self):
        super().__init__('isaac_visual_slam_node')

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/rgb/camera_info', self.camera_info_callback, 10)

        self.pose_pub = self.create_publisher(PoseStamped, '/visual_slam/pose', 10)
        self.path_pub = self.create_publisher(Path, '/visual_slam/path', 10)
        self.keypoints_pub = self.create_publisher(PointStamped, '/visual_slam/keypoints', 10)

        # CV Bridge
        self.cv_bridge = CvBridge()

        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None

        # SLAM state
        self.prev_frame = None
        self.prev_kp = None
        self.prev_desc = None
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.path = []

        # Feature detector (using ORB for efficiency)
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Frame buffer for keyframe selection
        self.frame_buffer = deque(maxlen=10)

        self.get_logger().info('Isaac Visual SLAM Node initialized')

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info('Camera parameters initialized')

    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Process frame for SLAM
            self.process_frame(cv_image, msg.header.stamp)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def process_frame(self, frame, timestamp):
        if self.camera_matrix is None:
            return

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect features
        kp = self.detector.detectAndCompute(gray, None)
        if kp is None:
            return

        keypoints, descriptors = kp

        # If this is the first frame, store it as reference
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_kp = keypoints
            self.prev_desc = descriptors
            return

        # Match features with previous frame
        matches = self.matcher.match(self.prev_desc, descriptors)
        if len(matches) < 10:
            # Not enough matches, update previous frame and return
            self.prev_frame = gray
            self.prev_kp = keypoints
            self.prev_desc = descriptors
            return

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract corresponding points
        prev_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate motion using essential matrix
        E, mask = cv2.findEssentialMat(
            curr_pts, prev_pts,
            self.camera_matrix,
            method=cv2.RANSAC,
            threshold=1.0,
            prob=0.999
        )

        if E is not None:
            # Decompose essential matrix to get rotation and translation
            _, R, t, _ = cv2.recoverPose(E, curr_pts, prev_pts, self.camera_matrix)

            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()

            # Update current pose
            self.current_pose = self.current_pose @ np.linalg.inv(T)

            # Store pose in path
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = timestamp
            pose_stamped.header.frame_id = 'map'
            pose_stamped.pose.position.x = float(self.current_pose[0, 3])
            pose_stamped.pose.position.y = float(self.current_pose[1, 3])
            pose_stamped.pose.position.z = float(self.current_pose[2, 3])

            # Convert rotation matrix to quaternion
            from math import sin, cos
            # Simple conversion for rotation matrix to quaternion
            # In practice, use a more robust method
            trace = np.trace(R)
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
                qw = 0.25 * s
                qx = (R[2, 1] - R[1, 2]) / s
                qy = (R[0, 2] - R[2, 0]) / s
                qz = (R[1, 0] - R[0, 1]) / s
            else:
                if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                    s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                    qw = (R[2, 1] - R[1, 2]) / s
                    qx = 0.25 * s
                    qy = (R[0, 1] + R[1, 0]) / s
                    qz = (R[0, 2] + R[2, 0]) / s
                elif R[1, 1] > R[2, 2]:
                    s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                    qw = (R[0, 2] - R[2, 0]) / s
                    qx = (R[0, 1] + R[1, 0]) / s
                    qy = 0.25 * s
                    qz = (R[1, 2] + R[2, 1]) / s
                else:
                    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                    qw = (R[1, 0] - R[0, 1]) / s
                    qx = (R[0, 2] + R[2, 0]) / s
                    qy = (R[1, 2] + R[2, 1]) / s
                    qz = 0.25 * s

            pose_stamped.pose.orientation.x = float(qx)
            pose_stamped.pose.orientation.y = float(qy)
            pose_stamped.pose.orientation.z = float(qz)
            pose_stamped.pose.orientation.w = float(qw)

            self.path.append(pose_stamped)

            # Publish current pose
            pose_pub_msg = PoseStamped()
            pose_pub_msg.header = pose_stamped.header
            pose_pub_msg.pose = pose_stamped.pose
            self.pose_pub.publish(pose_pub_msg)

            # Publish path
            path_msg = Path()
            path_msg.header.stamp = timestamp
            path_msg.header.frame_id = 'map'
            path_msg.poses = self.path[-100:]  # Keep last 100 poses
            self.path_pub.publish(path_msg)

        # Update previous frame data
        self.prev_frame = gray
        self.prev_kp = keypoints
        self.prev_desc = descriptors

def main(args=None):
    rclpy.init(args=args)
    node = IsaacVisualSLAMNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch Files for Isaac ROS Integration

Let's create a launch file that brings together all Isaac ROS components:

```python
# launch/isaac_ros_system.launch.py
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    enable_segmentation = LaunchConfiguration('enable_segmentation', default='true')
    enable_stereo = LaunchConfiguration('enable_stereo', default='true')
    enable_slam = LaunchConfiguration('enable_slam', default='true')

    # Package names
    pkg_simple_robot = FindPackageShare('simple_robot_pkg').find('simple_robot_pkg')

    # Paths
    rviz_config_path = os.path.join(pkg_simple_robot, 'rviz', 'isaac_ros.rviz')

    # Create launch description
    ld = LaunchDescription()

    # Declare launch arguments
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'))

    ld.add_action(DeclareLaunchArgument(
        'enable_segmentation',
        default_value='true',
        description='Enable Isaac ROS segmentation'))

    ld.add_action(DeclareLaunchArgument(
        'enable_stereo',
        default_value='true',
        description='Enable Isaac ROS stereo processing'))

    ld.add_action(DeclareLaunchArgument(
        'enable_slam',
        default_value='true',
        description='Enable Isaac ROS SLAM'))

    # Isaac ROS Stereo Processing
    stereo_rectification = Node(
        condition=IfCondition(enable_stereo),
        package='simple_robot_pkg',
        executable='isaac_stereo_rectification_node',
        name='isaac_stereo_rectification',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Isaac ROS Segmentation
    segmentation_node = Node(
        condition=IfCondition(enable_segmentation),
        package='simple_robot_pkg',
        executable='isaac_segmentation_node',
        name='isaac_segmentation',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Isaac ROS Visual SLAM
    visual_slam_node = Node(
        condition=IfCondition(enable_slam),
        package='simple_robot_pkg',
        executable='isaac_visual_slam_node',
        name='isaac_visual_slam',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Isaac ROS Occupancy Grid Localizer
    occupancy_localizer = Node(
        package='simple_robot_pkg',
        executable='isaac_occupancy_grid_localizer',
        name='isaac_occupancy_localizer',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Add all nodes to launch description
    ld.add_action(stereo_rectification)
    ld.add_action(segmentation_node)
    ld.add_action(visual_slam_node)
    ld.add_action(occupancy_localizer)

    return ld
```

## Performance Optimization with Isaac ROS

To maximize performance with Isaac ROS, consider these optimization techniques:

1. **GPU Memory Management**: Monitor and optimize GPU memory usage
2. **Pipeline Optimization**: Chain processing nodes efficiently
3. **Batch Processing**: Process multiple frames simultaneously when possible
4. **Precision Control**: Use appropriate precision levels for your application

## Next Steps

In the next chapter, we'll explore Isaac Lab, which provides reinforcement learning capabilities for robotics applications. We'll see how to train robot behaviors using simulated environments and transfer them to real robots.