---
sidebar_position: 1
---

# Chapter 13: Vision-Language Integration with AI Models

## Introduction to Vision-Language-Action (VLA) Systems

Vision-Language-Action (VLA) systems represent the next frontier in robotics, combining visual perception, natural language understanding, and robotic action execution. These systems enable robots to understand and respond to human instructions in natural language while perceiving and interacting with the real world.

The key components of VLA systems include:
- **Vision systems**: Processing visual information from cameras and sensors
- **Language models**: Understanding and generating human language
- **Action planning**: Converting high-level instructions into robot actions
- **Integration frameworks**: Coordinating all components for coherent behavior

## Understanding Modern AI Models for VLA

### Large Language Models (LLMs) in Robotics

Large Language Models like GPT, Claude, and open-source alternatives have revolutionized how robots can understand and process natural language commands. These models provide:

1. **Natural language understanding**: Interpreting human commands in natural language
2. **Context awareness**: Maintaining context across multiple interactions
3. **Reasoning capabilities**: Planning and decision-making based on complex instructions
4. **Multimodal integration**: Combining text with visual information

### Vision Transformers and Perception

Vision Transformers (ViTs) and other modern vision models enable robots to:
- Recognize objects and scenes in real-time
- Understand spatial relationships
- Extract meaningful features from visual input
- Integrate visual information with language understanding

## Setting up a VLA System with ROS 2

Let's create a comprehensive VLA system that integrates vision, language, and action:

```python
# vla_system.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import numpy as np
import cv2
import openai  # or use Hugging Face transformers for open-source models
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image as PILImage

class VLASystem(Node):
    def __init__(self):
        super().__init__('vla_system')

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.command_sub = self.create_subscription(
            String, '/robot_command', self.command_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/vla_status', 10)

        # CV Bridge for image processing
        self.cv_bridge = CvBridge()

        # Vision components
        self.current_image = None
        self.image_timestamp = None

        # Language model components
        self.setup_language_model()

        # Action planning
        self.action_queue = []
        self.current_action = None

        # System state
        self.is_executing = False
        self.system_status = "IDLE"

        # Timer for processing commands
        self.process_timer = self.create_timer(0.1, self.process_commands)

        self.get_logger().info('VLA System initialized')

    def setup_language_model(self):
        """Initialize the language model for command processing."""
        try:
            # Using BLIP for vision-language understanding (open-source alternative)
            self.vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

            # For actual LLM integration, you might use:
            # openai.api_key = "your-api-key"
            # Or Hugging Face models for open-source alternatives

            self.get_logger().info('Language model initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize language model: {e}')

    def image_callback(self, msg):
        """Process incoming camera images."""
        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_timestamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def command_callback(self, msg):
        """Process incoming natural language commands."""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Add command to processing queue
        self.action_queue.append(command)

        # Update status
        status_msg = String()
        status_msg.data = f"Received command: {command}"
        self.status_pub.publish(status_msg)

    def process_commands(self):
        """Process commands in the queue."""
        if not self.action_queue or self.is_executing:
            return

        command = self.action_queue.pop(0)
        self.is_executing = True

        try:
            # Analyze the command and current visual scene
            action_plan = self.analyze_command_and_scene(command)

            if action_plan:
                self.execute_action_plan(action_plan)
            else:
                self.get_logger().warn(f'Could not generate action plan for command: {command}')

        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')
        finally:
            self.is_executing = False

    def analyze_command_and_scene(self, command):
        """Analyze command and current scene to generate action plan."""
        if self.current_image is None:
            self.get_logger().warn('No current image available for scene analysis')
            return None

        try:
            # Process image with vision model
            pil_image = PILImage.fromarray(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))

            # Generate image caption
            inputs = self.vision_processor(pil_image, command, return_tensors="pt")
            out = self.vision_model.generate(**inputs)
            caption = self.vision_processor.decode(out[0], skip_special_tokens=True)

            self.get_logger().info(f'Image caption: {caption}')

            # Use LLM to interpret command and generate action plan
            action_plan = self.generate_action_plan(command, caption)

            return action_plan

        except Exception as e:
            self.get_logger().error(f'Error analyzing command and scene: {e}')
            return None

    def generate_action_plan(self, command, scene_description):
        """Generate action plan using LLM."""
        # This is a simplified example - in practice, you'd use a more sophisticated approach
        command_lower = command.lower()

        # Simple rule-based action planning (in practice, use LLM reasoning)
        if 'move to' in command_lower or 'go to' in command_lower:
            if 'red' in command_lower:
                return {'action': 'navigate_to_color', 'color': 'red', 'command': command}
            elif 'blue' in command_lower:
                return {'action': 'navigate_to_color', 'color': 'blue', 'command': command}
            elif 'table' in command_lower:
                return {'action': 'navigate_to_object', 'object': 'table', 'command': command}
        elif 'pick up' in command_lower or 'grasp' in command_lower:
            return {'action': 'pick_up_object', 'command': command}
        elif 'stop' in command_lower:
            return {'action': 'stop', 'command': command}
        elif 'turn' in command_lower or 'rotate' in command_lower:
            if 'left' in command_lower:
                return {'action': 'rotate', 'direction': 'left', 'command': command}
            elif 'right' in command_lower:
                return {'action': 'rotate', 'direction': 'right', 'command': command}
            else:
                return {'action': 'rotate', 'direction': 'around', 'command': command}

        return None

    def execute_action_plan(self, action_plan):
        """Execute the generated action plan."""
        action_type = action_plan['action']

        if action_type == 'navigate_to_color':
            self.navigate_to_color(action_plan['color'])
        elif action_type == 'navigate_to_object':
            self.navigate_to_object(action_plan['object'])
        elif action_type == 'pick_up_object':
            self.pick_up_object()
        elif action_type == 'stop':
            self.stop_robot()
        elif action_type == 'rotate':
            self.rotate_robot(action_plan['direction'])
        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')

    def navigate_to_color(self, color):
        """Navigate to an object of specified color."""
        if self.current_image is None:
            return

        # Convert image to HSV for color detection
        hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)

        # Define color ranges
        color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255])
        }

        if color not in color_ranges:
            self.get_logger().warn(f'Unknown color: {color}')
            return

        lower, upper = color_ranges[color]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Find contours of colored regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the center of the contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Convert to normalized coordinates (center at 0, edges at -1, 1)
                h, w = self.current_image.shape[:2]
                norm_x = (cx - w/2) / (w/2)
                norm_y = (cy - h/2) / (h/2)

                # Generate movement command based on object position
                cmd_vel = Twist()

                # Move forward if object is roughly centered
                if abs(norm_x) < 0.2:
                    cmd_vel.linear.x = 0.3
                else:
                    # Turn toward object
                    cmd_vel.angular.z = -norm_x * 0.5  # Negative because image coordinates are flipped

                self.cmd_vel_pub.publish(cmd_vel)
                self.get_logger().info(f'Navigating to {color} object at ({cx}, {cy})')

                return

        # If no object found, rotate to search
        cmd_vel = Twist()
        cmd_vel.angular.z = 0.5  # Rotate slowly
        self.cmd_vel_pub.publish(cmd_vel)
        self.get_logger().info(f'Searching for {color} object')

    def navigate_to_object(self, obj_name):
        """Navigate to a specific object."""
        # In a real implementation, this would use object detection
        # For now, we'll use a simplified approach
        self.get_logger().info(f'Navigating to {obj_name}')

        # Simple forward movement
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.3
        self.cmd_vel_pub.publish(cmd_vel)

    def pick_up_object(self):
        """Simulate picking up an object."""
        self.get_logger().info('Attempting to pick up object')

        # In a real implementation, this would interface with robot arms/grippers
        # For now, just stop the robot
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)

    def stop_robot(self):
        """Stop the robot."""
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)
        self.get_logger().info('Robot stopped')

    def rotate_robot(self, direction):
        """Rotate the robot."""
        cmd_vel = Twist()

        if direction == 'left':
            cmd_vel.angular.z = 0.5
        elif direction == 'right':
            cmd_vel.angular.z = -0.5
        elif direction == 'around':
            cmd_vel.angular.z = 0.5  # Default to left rotation

        self.cmd_vel_pub.publish(cmd_vel)
        self.get_logger().info(f'Rotating robot to the {direction}')

def main(args=None):
    rclpy.init(args=args)
    vla_system = VLASystem()

    try:
        rclpy.spin(vla_system)
    except KeyboardInterrupt:
        pass
    finally:
        vla_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Vision Processing for VLA Systems

```python
# vision_processor.py
import cv2
import numpy as np
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests

class AdvancedVisionProcessor:
    def __init__(self):
        """Initialize advanced vision processing components."""
        try:
            # Object detection model (DETR - Detection Transformer)
            self.detection_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.detection_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.detection_model.to(self.device)
            self.detection_model.eval()

            print("Advanced vision processor initialized")
        except Exception as e:
            print(f"Failed to initialize vision processor: {e}")
            # Fallback to basic processing
            self.detection_model = None

    def detect_objects(self, image):
        """Detect objects in the image using transformer-based detection."""
        if self.detection_model is None:
            return self.fallback_detection(image)

        try:
            # Convert OpenCV image to PIL
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Process image
            inputs = self.detection_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.detection_model(**inputs)

            # Process results
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.detection_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.9
            )[0]

            # Extract detections
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                detections.append({
                    'label': self.detection_model.config.id2label[label.item()],
                    'score': score.item(),
                    'bbox': box.tolist()  # [x_min, y_min, x_max, y_max]
                })

            return detections

        except Exception as e:
            print(f"Error in object detection: {e}")
            return self.fallback_detection(image)

    def fallback_detection(self, image):
        """Fallback detection using traditional computer vision."""
        detections = []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use simple shape detection
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h

                if aspect_ratio > 1.5:
                    label = "rectangle"
                elif aspect_ratio < 0.7:
                    label = "rectangle"  # Could be square depending on angle
                else:
                    label = "circle_or_square"

                detections.append({
                    'label': label,
                    'score': 0.5,  # Fallback confidence
                    'bbox': [x, y, x + w, y + h]
                })

        return detections

    def extract_features(self, image, region_of_interest=None):
        """Extract visual features from the image."""
        if region_of_interest:
            x, y, w, h = region_of_interest
            image = image[y:y+h, x:x+w]

        # Convert to RGB for consistent processing
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Extract color histogram
        hist_features = self.compute_color_histogram(rgb_image)

        # Extract texture features (using Local Binary Patterns)
        texture_features = self.compute_texture_features(gray)

        # Extract shape features
        shape_features = self.compute_shape_features(image)

        return {
            'color': hist_features,
            'texture': texture_features,
            'shape': shape_features
        }

    def compute_color_histogram(self, image):
        """Compute color histogram features."""
        hist_r = cv2.calcHist([image], [0], None, [8], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [8], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [8], [0, 256])

        # Normalize histograms
        hist_r = hist_r.flatten() / hist_r.sum()
        hist_g = hist_g.flatten() / hist_g.sum()
        hist_b = hist_b.flatten() / hist_b.sum()

        return np.concatenate([hist_r, hist_g, hist_b])

    def compute_texture_features(self, gray_image):
        """Compute simple texture features."""
        # Using Local Binary Pattern (LBP) approach
        # This is a simplified version - in practice, use scikit-image's LBP
        texture_features = []
        h, w = gray_image.shape

        # Compute gradient magnitude as a simple texture measure
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Compute statistics
        texture_features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude),
            np.percentile(gradient_magnitude, 25),
            np.percentile(gradient_magnitude, 75)
        ])

        return np.array(texture_features)

    def compute_shape_features(self, image):
        """Compute shape-related features."""
        # Find contours
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.zeros(4)  # Return zeros if no contours found

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area == 0:
            return np.zeros(4)

        # Compute shape features
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0

        # Extent (ratio of contour area to bounding rectangle area)
        extent = area / (w * h) if w * h > 0 else 0

        # Solidity (ratio of contour area to convex hull area)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        return np.array([circularity, aspect_ratio, extent, solidity])

    def compare_regions(self, image, region1, region2):
        """Compare two regions in the image."""
        features1 = self.extract_features(image, region1)
        features2 = self.extract_features(image, region2)

        # Compute similarity (simple Euclidean distance for demonstration)
        color_diff = np.linalg.norm(features1['color'] - features2['color'])
        texture_diff = np.linalg.norm(features1['texture'] - features2['texture'])
        shape_diff = np.linalg.norm(features1['shape'] - features2['shape'])

        # Weighted similarity score
        similarity = 1.0 / (1.0 + 0.5 * color_diff + 0.3 * texture_diff + 0.2 * shape_diff)

        return similarity
```

## Language Understanding Component

```python
# language_understanding.py
import openai
import re
from typing import Dict, List, Tuple
import numpy as np

class LanguageUnderstanding:
    def __init__(self, api_key=None):
        """Initialize language understanding component."""
        if api_key:
            openai.api_key = api_key

        # Define common command patterns
        self.command_patterns = {
            'navigation': [
                r'go to (?:the )?(.+)',
                r'move to (?:the )?(.+)',
                r'navigate to (?:the )?(.+)',
                r'go (?:to the |toward the |towards the )?(.+)',
            ],
            'grasping': [
                r'pick up (?:the )?(.+)',
                r'grasp (?:the )?(.+)',
                r'get (?:the )?(.+)',
                r'collect (?:the )?(.+)',
            ],
            'manipulation': [
                r'push (?:the )?(.+)',
                r'pull (?:the )?(.+)',
                r'move (?:the )?(.+) (?:to|toward) (.+)',
                r'bring (?:the )?(.+) to (.+)',
            ],
            'orientation': [
                r'look at (?:the )?(.+)',
                r'face (?:the )?(.+)',
                r'turn to (?:the )?(.+)',
                r'rotate to (?:the )?(.+)',
            ],
            'stop': [
                r'stop',
                r'halt',
                r'freeze',
            ]
        }

        # Object categories and synonyms
        self.object_categories = {
            'furniture': ['table', 'chair', 'couch', 'sofa', 'desk', 'bed'],
            'kitchen': ['fridge', 'oven', 'microwave', 'sink', 'stove'],
            'office': ['computer', 'laptop', 'printer', 'book', 'pen'],
            'colors': ['red', 'blue', 'green', 'yellow', 'black', 'white'],
            'shapes': ['box', 'cylinder', 'sphere', 'cube', 'ball']
        }

    def parse_command(self, command: str) -> Dict:
        """Parse a natural language command and extract intent and entities."""
        command_lower = command.lower().strip()

        # Find the best matching pattern
        for action_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, command_lower)
                if match:
                    entities = match.groups()
                    return {
                        'action_type': action_type,
                        'entities': entities,
                        'original_command': command,
                        'parsed_command': self._interpret_command(action_type, entities)
                    }

        # If no pattern matches, return unknown
        return {
            'action_type': 'unknown',
            'entities': [],
            'original_command': command,
            'parsed_command': {'action': 'unknown', 'params': {}}
        }

    def _interpret_command(self, action_type: str, entities: Tuple) -> Dict:
        """Interpret the parsed command and convert to robot action."""
        if action_type == 'navigation':
            target = entities[0] if entities else None
            return {
                'action': 'navigate',
                'target': self._normalize_object(target),
                'params': {}
            }
        elif action_type == 'grasping':
            target = entities[0] if entities else None
            return {
                'action': 'grasp',
                'target': self._normalize_object(target),
                'params': {}
            }
        elif action_type == 'manipulation':
            if len(entities) >= 2:
                target = entities[0]
                destination = entities[1]
                return {
                    'action': 'manipulate',
                    'target': self._normalize_object(target),
                    'destination': self._normalize_object(destination),
                    'params': {}
                }
        elif action_type == 'orientation':
            target = entities[0] if entities else None
            return {
                'action': 'orient',
                'target': self._normalize_object(target),
                'params': {}
            }
        elif action_type == 'stop':
            return {
                'action': 'stop',
                'params': {}
            }

        return {'action': 'unknown', 'params': {}}

    def _normalize_object(self, obj_name: str) -> str:
        """Normalize object names to standard categories."""
        if not obj_name:
            return "unknown"

        obj_lower = obj_name.lower().strip()

        # Check if it's a color
        for category, items in self.object_categories.items():
            if category == 'colors' and obj_lower in items:
                return f"color:{obj_lower}"

        # Check other categories
        for category, items in self.object_categories.items():
            if category != 'colors':  # Colors handled separately
                for item in items:
                    if item in obj_lower or obj_lower in item:
                        return f"{category}:{item}"

        # Return as is if not in any category
        return f"object:{obj_lower}"

    def generate_response(self, command: str, context: Dict = None) -> str:
        """Generate a response to the command."""
        parsed = self.parse_command(command)

        if parsed['action_type'] == 'unknown':
            return f"I'm not sure how to handle: '{command}'. Could you rephrase that?"

        action = parsed['parsed_command']['action']
        entities = parsed['entities']

        if action == 'navigate':
            target = entities[0] if entities else "unknown location"
            return f"Okay, I'll navigate to the {target}."
        elif action == 'grasp':
            target = entities[0] if entities else "unknown object"
            return f"Okay, I'll grasp the {target}."
        elif action == 'stop':
            return "Okay, I'll stop."
        else:
            return f"Okay, I'll {action} the {entities[0] if entities else 'object'}."

    def validate_command(self, command: str, current_scene: List[Dict]) -> bool:
        """Validate if the command is feasible given the current scene."""
        parsed = self.parse_command(command)

        if parsed['action_type'] == 'navigation':
            target = parsed['parsed_command']['target']
            if target.startswith('color:'):
                # Check if color exists in scene
                target_color = target.split(':')[1]
                for obj in current_scene:
                    if obj.get('color') == target_color:
                        return True
                return False
            elif target.startswith('object:') or target.startswith('category:'):
                # Check if object exists in scene
                target_obj = target.split(':')[1]
                for obj in current_scene:
                    if target_obj in obj.get('label', '').lower():
                        return True
                return False

        # For other actions, assume they're valid for now
        return True

    def generate_action_sequence(self, high_level_command: str, scene_objects: List[Dict]) -> List[Dict]:
        """Generate a sequence of low-level actions from a high-level command."""
        parsed = self.parse_command(high_level_command)

        if parsed['action_type'] == 'navigation':
            target = parsed['parsed_command']['target']

            # Find the target object in the scene
            target_obj = None
            for obj in scene_objects:
                if target in obj.get('label', '').lower() or target in obj.get('color', '').lower():
                    target_obj = obj
                    break

            if target_obj:
                # Generate navigation sequence
                actions = [
                    {'action': 'approach_object', 'object_id': target_obj.get('id', 0)},
                    {'action': 'align_with_object', 'object_id': target_obj.get('id', 0)},
                    {'action': 'stop_at_object', 'object_id': target_obj.get('id', 0)}
                ]
                return actions

        # Default: single action
        return [{'action': parsed['parsed_command']['action'], 'params': {}}]
```

## Integration Example: Complete VLA System

```python
# integrated_vla_system.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import numpy as np
import cv2
from vision_processor import AdvancedVisionProcessor
from language_understanding import LanguageUnderstanding

class IntegratedVLASystem(Node):
    def __init__(self):
        super().__init__('integrated_vla_system')

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.command_sub = self.create_subscription(
            String, '/natural_language_command', self.command_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/vla_status', 10)
        self.action_pub = self.create_publisher(String, '/robot_action', 10)

        # Initialize components
        self.cv_bridge = CvBridge()
        self.vision_processor = AdvancedVisionProcessor()
        self.language_understanding = LanguageUnderstanding()

        # System state
        self.current_image = None
        self.current_scene = []
        self.command_queue = []
        self.is_executing = False

        # Action execution parameters
        self.linear_speed = 0.3
        self.angular_speed = 0.5
        self.approach_distance = 0.5  # meters

        # Timer for processing
        self.process_timer = self.create_timer(0.1, self.process_cycle)

        self.get_logger().info('Integrated VLA System initialized')

    def image_callback(self, msg):
        """Process incoming images and update scene understanding."""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_image = cv_image

            # Process image to detect objects
            detections = self.vision_processor.detect_objects(cv_image)
            self.current_scene = self.format_detections(detections, cv_image.shape)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def command_callback(self, msg):
        """Process incoming natural language commands."""
        command = msg.data
        self.get_logger().info(f'Received natural language command: {command}')

        # Add to command queue
        self.command_queue.append(command)

        # Publish status
        status_msg = String()
        status_msg.data = f"Processing: {command}"
        self.status_pub.publish(status_msg)

    def format_detections(self, detections, image_shape):
        """Format detections for internal use."""
        h, w = image_shape[:2]
        formatted_objects = []

        for detection in detections:
            bbox = detection['bbox']
            x_min, y_min, x_max, y_max = bbox

            # Calculate center and normalized position
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            norm_x = (center_x - w/2) / (w/2)  # -1 to 1
            norm_y = (center_y - h/2) / (h/2)  # -1 to 1

            formatted_objects.append({
                'id': len(formatted_objects),
                'label': detection['label'],
                'confidence': detection['score'],
                'bbox': bbox,
                'center': (center_x, center_y),
                'normalized_center': (norm_x, norm_y),
                'size': ((x_max - x_min) / w, (y_max - y_min) / h)
            })

        return formatted_objects

    def process_cycle(self):
        """Main processing cycle."""
        if not self.command_queue or self.is_executing:
            # Update scene understanding even when not executing commands
            self.update_scene_understanding()
            return

        command = self.command_queue.pop(0)
        self.is_executing = True

        try:
            # Validate command against current scene
            if not self.language_understanding.validate_command(command, self.current_scene):
                self.get_logger().warn(f'Command not feasible in current scene: {command}')
                self.publish_status(f"Command not feasible: {command}")
                return

            # Parse command
            parsed_command = self.language_understanding.parse_command(command)

            # Generate action sequence
            action_sequence = self.language_understanding.generate_action_sequence(
                command, self.current_scene
            )

            # Execute action sequence
            for action in action_sequence:
                self.execute_action(action)
                # Add small delay between actions
                self.get_clock().sleep_for(rclpy.time.Duration(seconds=0.1))

            # Generate and publish response
            response = self.language_understanding.generate_response(command)
            self.publish_status(response)

        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')
            self.publish_status(f"Error processing command: {e}")
        finally:
            self.is_executing = False

    def update_scene_understanding(self):
        """Update scene understanding without processing commands."""
        if self.current_image is not None:
            # This could be used to continuously update scene understanding
            # For now, we just log the number of detected objects
            self.get_logger().debug(f'Current scene: {len(self.current_scene)} objects detected')

    def execute_action(self, action):
        """Execute a single action."""
        action_type = action['action']

        if action_type == 'navigate':
            self.execute_navigation(action)
        elif action_type == 'grasp':
            self.execute_grasp(action)
        elif action_type == 'approach_object':
            self.execute_approach_object(action)
        elif action_type == 'align_with_object':
            self.execute_align_with_object(action)
        elif action_type == 'stop':
            self.execute_stop()
        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')

    def execute_navigation(self, action):
        """Execute navigation action."""
        target = action.get('target', 'unknown')
        self.get_logger().info(f'Navigating to {target}')

        # In a real implementation, this would involve path planning
        # For now, we'll use simple obstacle avoidance
        cmd_vel = self.simple_navigation_to_target(target)
        self.cmd_vel_pub.publish(cmd_vel)

    def execute_approach_object(self, action):
        """Approach a specific object."""
        object_id = action.get('object_id', 0)

        if object_id < len(self.current_scene):
            obj = self.current_scene[object_id]
            cmd_vel = self.navigate_to_object_position(obj)
            self.cmd_vel_pub.publish(cmd_vel)

    def execute_align_with_object(self, action):
        """Align with a specific object."""
        object_id = action.get('object_id', 0)

        if object_id < len(self.current_scene):
            obj = self.current_scene[object_id]
            cmd_vel = self.align_with_object(obj)
            self.cmd_vel_pub.publish(cmd_vel)

    def execute_stop(self):
        """Stop the robot."""
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)

    def simple_navigation_to_target(self, target):
        """Simple navigation to target based on current scene."""
        cmd_vel = Twist()

        # Find target object in scene
        target_obj = None
        for obj in self.current_scene:
            if target.lower() in obj['label'].lower():
                target_obj = obj
                break

        if target_obj:
            # Navigate based on object position
            norm_x, norm_y = target_obj['normalized_center']

            # Move toward object if centered enough
            if abs(norm_x) < 0.3:  # Object is roughly centered
                cmd_vel.linear.x = min(self.linear_speed, 0.3)
            else:
                # Rotate to center object
                cmd_vel.angular.z = -norm_x * self.angular_speed

            self.get_logger().info(f'Navigating to {target_obj["label"]} at ({norm_x:.2f}, {norm_y:.2f})')
        else:
            # No target found, perhaps search for it
            cmd_vel.angular.z = 0.3  # Rotate slowly to search
            self.get_logger().info(f'Searching for {target}')

        return cmd_vel

    def navigate_to_object_position(self, obj):
        """Navigate to object position."""
        cmd_vel = Twist()

        # Get normalized position of object
        norm_x, norm_y = obj['normalized_center']

        # Approach strategy: center horizontally first, then approach
        if abs(norm_x) > 0.2:  # Not centered horizontally
            cmd_vel.angular.z = -norm_x * self.angular_speed
        elif obj['size'][0] < 0.3:  # Object is far (small in image)
            cmd_vel.linear.x = self.linear_speed
        else:  # Close enough
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0

        return cmd_vel

    def align_with_object(self, obj):
        """Align robot with object."""
        cmd_vel = Twist()

        # Get normalized position of object
        norm_x, norm_y = obj['normalized_center']

        # Align horizontally (rotate until centered)
        if abs(norm_x) > 0.1:
            cmd_vel.angular.z = -norm_x * self.angular_speed
        else:
            cmd_vel.angular.z = 0.0

        return cmd_vel

    def publish_status(self, status):
        """Publish status message."""
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    vla_system = IntegratedVLASystem()

    try:
        rclpy.spin(vla_system)
    except KeyboardInterrupt:
        pass
    finally:
        vla_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Next Steps

In the next chapter, we'll explore how to integrate large language models like Claude with robotic systems, focusing on the specific APIs and techniques for creating intelligent, conversational robots that can understand complex instructions and provide meaningful responses.