---
sidebar_position: 2
---

# Chapter 14: LLM-Based Task Planning and Reasoning

## Introduction to LLM-Based Planning in Robotics

Large Language Models (LLMs) have emerged as powerful tools for high-level task planning and reasoning in robotics. Unlike traditional planning approaches that rely on symbolic representations and predefined rules, LLMs can understand natural language commands, reason about the environment, and generate complex action sequences based on contextual understanding.

The key advantages of LLM-based planning include:
- **Natural language interface**: Accept commands in natural language
- **Contextual reasoning**: Understand spatial relationships and object affordances
- **Flexible planning**: Adapt plans based on changing conditions
- **Knowledge integration**: Leverage pre-trained world knowledge

## Setting Up LLM Integration

### API Configuration

```python
# llm_integration.py
import openai
import anthropic
import asyncio
import json
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass

@dataclass
class RobotState:
    """Represents the current state of the robot."""
    position: Tuple[float, float, float]  # x, y, z
    orientation: Tuple[float, float, float, float]  # quaternion
    battery_level: float
    attached_objects: List[str]
    available_actions: List[str]

@dataclass
class TaskPlan:
    """Represents a planned sequence of actions."""
    steps: List[Dict]
    estimated_time: float
    confidence: float
    dependencies: List[str]

class LLMPlanner:
    """Integrates LLMs for robotic task planning and reasoning."""

    def __init__(self, provider: str = "openai", api_key: str = None):
        """
        Initialize LLM planner.

        Args:
            provider: "openai", "anthropic", or "huggingface"
            api_key: API key for the LLM service
        """
        self.provider = provider
        self.api_key = api_key

        if provider == "openai":
            openai.api_key = api_key
        elif provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)

        # System prompt for robotic planning
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create system prompt for robotic planning."""
        return """
        You are an expert robotic task planner. Your role is to:

        1. Interpret natural language commands for a mobile robot
        2. Generate detailed, executable action plans
        3. Consider environmental constraints and robot capabilities
        4. Handle ambiguous or incomplete information gracefully
        5. Break down complex tasks into simple, sequential steps
        6. Account for object properties, spatial relationships, and safety

        Robot capabilities:
        - Navigation: move to locations, avoid obstacles
        - Manipulation: pick up, place, push objects
        - Perception: detect and identify objects
        - Communication: report status, ask for clarification

        Action format should be JSON with these types:
        {
            "action": "navigate_to | pick_up | place | push | report | ask_for_clarification",
            "target": "object_id or location",
            "location": "specific coordinates or named location",
            "description": "brief explanation of the action",
            "estimated_duration": "time in seconds",
            "preconditions": ["list of conditions that must be true"],
            "effects": ["list of changes to the world state"]
        }

        Always return valid JSON. Be specific and actionable.
        """

    def plan_task(self,
                  command: str,
                  robot_state: RobotState,
                  environment_description: str) -> Optional[TaskPlan]:
        """
        Generate a task plan for the given command.

        Args:
            command: Natural language command
            robot_state: Current robot state
            environment_description: Description of the environment

        Returns:
            TaskPlan object or None if planning fails
        """
        prompt = self._create_planning_prompt(command, robot_state, environment_description)

        try:
            if self.provider == "openai":
                response = openai.ChatCompletion.create(
                    model="gpt-4-turbo",  # or gpt-4o for latest
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                plan_text = response.choices[0].message.content
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-opus-20240229",  # or latest Claude model
                    max_tokens=1000,
                    temperature=0.3,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                plan_text = response.content[0].text

            # Parse the response
            return self._parse_plan_response(plan_text)

        except Exception as e:
            print(f"Error generating plan: {e}")
            return None

    def _create_planning_prompt(self, command: str, robot_state: RobotState, env_desc: str) -> str:
        """Create the detailed prompt for task planning."""
        prompt = f"""
        Command: {command}

        Current Robot State:
        - Position: {robot_state.position}
        - Battery: {robot_state.battery_level * 100:.1f}%
        - Attached objects: {robot_state.attached_objects}
        - Available actions: {robot_state.available_actions}

        Environment:
        {env_desc}

        Generate a detailed action plan in JSON format. Consider:
        1. The robot's current position and battery level
        2. Environmental constraints and obstacles
        3. Object locations and accessibility
        4. Safety considerations
        5. Most efficient path to complete the task

        Return ONLY valid JSON with the action sequence.
        """
        return prompt

    def _parse_plan_response(self, response: str) -> Optional[TaskPlan]:
        """Parse the LLM response into a TaskPlan object."""
        try:
            # Extract JSON from response if it contains other text
            start_idx = response.find('[')
            if start_idx == -1:
                start_idx = response.find('{')
            if start_idx != -1:
                response = response[start_idx:]
                # Find matching closing bracket
                brace_count = 0
                for i, char in enumerate(response):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            response = response[:i+1]
                            break

            # Parse JSON
            plan_data = json.loads(response)

            # Handle both single action and action list formats
            if isinstance(plan_data, dict):
                steps = [plan_data]
            else:
                steps = plan_data

            # Calculate estimated time
            total_time = sum(step.get('estimated_duration', 5) for step in steps)

            return TaskPlan(
                steps=steps,
                estimated_time=total_time,
                confidence=0.8,  # Default confidence
                dependencies=[]
            )

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response was: {response}")
            return None
        except Exception as e:
            print(f"Error parsing plan response: {e}")
            return None

    def update_plan(self,
                    original_plan: TaskPlan,
                    new_information: str) -> Optional[TaskPlan]:
        """Update an existing plan based on new information."""
        prompt = f"""
        Original plan:
        {json.dumps(original_plan.steps, indent=2)}

        New information:
        {new_information}

        Update the plan considering the new information. Return the revised plan in the same JSON format.
        If the new information makes the original plan impossible, return a new plan.
        """

        try:
            if self.provider == "openai":
                response = openai.ChatCompletion.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                plan_text = response.choices[0].message.content
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1000,
                    temperature=0.3,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                plan_text = response.content[0].text

            return self._parse_plan_response(plan_text)

        except Exception as e:
            print(f"Error updating plan: {e}")
            return original_plan  # Return original plan if update fails
```

## Advanced Reasoning with LLMs

```python
# reasoning_engine.py
import re
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class SpatialRelation(Enum):
    """Spatial relationships between objects."""
    INSIDE = "inside"
    ON_TOP_OF = "on_top_of"
    NEXT_TO = "next_to"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    NEAR = "near"
    FAR_FROM = "far_from"

@dataclass
class Object:
    """Represents an object in the environment."""
    id: str
    name: str
    category: str
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]  # width, height, depth
    properties: Dict[str, Any]
    spatial_relations: Dict[str, SpatialRelation]

class ReasoningEngine:
    """Advanced reasoning engine using LLMs for complex robotic tasks."""

    def __init__(self, llm_planner: LLMPlanner):
        self.llm_planner = llm_planner
        self.objects: Dict[str, Object] = {}
        self.spatial_knowledge = {}

    def add_object(self, obj: Object):
        """Add an object to the environment model."""
        self.objects[obj.id] = obj

    def update_object_position(self, obj_id: str, new_position: Tuple[float, float, float]):
        """Update an object's position."""
        if obj_id in self.objects:
            self.objects[obj_id].position = new_position
            # Recalculate spatial relationships
            self._update_spatial_relations(obj_id)

    def _update_spatial_relations(self, obj_id: str):
        """Update spatial relationships for the given object."""
        if obj_id not in self.objects:
            return

        target_obj = self.objects[obj_id]
        relations = {}

        for other_id, other_obj in self.objects.items():
            if other_id == obj_id:
                continue

            relation = self._calculate_spatial_relation(
                target_obj.position, other_obj.position,
                target_obj.size, other_obj.size
            )
            relations[other_id] = relation

        target_obj.spatial_relations = relations

    def _calculate_spatial_relation(self, pos1, pos2, size1, size2) -> SpatialRelation:
        """Calculate spatial relation between two objects."""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dz = pos1[2] - pos2[2]

        # Calculate distances
        horizontal_dist = (dx**2 + dy**2)**0.5
        vertical_dist = abs(dz)

        # Determine relationship based on distances and sizes
        if vertical_dist < min(size1[2], size2[2]) / 2:  # Similar height
            if horizontal_dist < (size1[0] + size2[0]) / 2:
                return SpatialRelation.NEXT_TO
            elif horizontal_dist < 2.0:  # Within 2 meters
                return SpatialRelation.NEAR
        elif dz > 0 and vertical_dist < size2[1]:  # pos1 is above pos2
            return SpatialRelation.ON_TOP_OF
        elif dz < 0 and vertical_dist < size1[1]:  # pos1 is below pos2
            return SpatialRelation.INSIDE

        return SpatialRelation.FAR_FROM

    def answer_spatial_query(self, query: str) -> str:
        """Answer spatial reasoning queries using LLM."""
        # Extract query components
        object1, object2 = self._extract_objects_from_query(query)

        if object1 and object2 and object1 in self.objects and object2 in self.objects:
            # Check direct spatial relationship
            if object2 in self.objects[object1].spatial_relations:
                relation = self.objects[object1].spatial_relations[object2]
                return f"The {object1} is {relation.value} the {object2}."

        # Use LLM for complex reasoning
        context = self._get_environment_context()
        prompt = f"""
        Environment context:
        {context}

        Question: {query}

        Provide a clear, accurate answer based on the environment information.
        """

        try:
            if self.llm_planner.provider == "openai":
                response = openai.ChatCompletion.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are a spatial reasoning expert for robotics. Answer questions about object locations and relationships based on the provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                return response.choices[0].message.content
            elif self.llm_planner.provider == "anthropic":
                response = self.llm_planner.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=300,
                    temperature=0.1,
                    system="You are a spatial reasoning expert for robotics. Answer questions about object locations and relationships based on the provided context.",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
        except Exception as e:
            return f"Could not answer query: {e}"

    def _extract_objects_from_query(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract object names from spatial query."""
        # Simple extraction - in practice, use NER or more sophisticated parsing
        words = query.lower().split()
        found_objects = []

        for obj_id in self.objects.keys():
            if obj_id.lower() in query.lower():
                found_objects.append(obj_id)

        if len(found_objects) >= 2:
            return found_objects[0], found_objects[1]

        return None, None

    def _get_environment_context(self) -> str:
        """Get current environment context for reasoning."""
        context = "Environment Objects:\n"
        for obj_id, obj in self.objects.items():
            context += f"- {obj_id}: {obj.name} ({obj.category}) at {obj.position}\n"

        context += "\nSpatial Relationships:\n"
        for obj_id, obj in self.objects.items():
            if obj.spatial_relations:
                context += f"- {obj_id}:\n"
                for other_id, relation in obj.spatial_relations.items():
                    context += f"  - {relation.value} {other_id}\n"

        return context

    def handle_complex_command(self, command: str, robot_state: RobotState) -> Optional[TaskPlan]:
        """Handle complex commands that require reasoning."""
        # First, use LLM to understand the command and required reasoning
        reasoning_prompt = f"""
        Command: {command}

        Robot capabilities: {robot_state.available_actions}

        Analyze this command and determine:
        1. What information is needed to execute it
        2. What reasoning steps are required
        3. What questions need to be answered
        4. What plan would be appropriate

        Be specific about spatial reasoning, object identification, and task decomposition needed.
        """

        try:
            if self.llm_planner.provider == "openai":
                response = openai.ChatCompletion.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are a robotic command analyzer. Break down complex commands into reasoning requirements and planning needs."},
                        {"role": "user", "content": reasoning_prompt}
                    ],
                    temperature=0.3
                )
                analysis = response.choices[0].message.content
            elif self.llm_planner.provider == "anthropic":
                response = self.llm_planner.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=500,
                    temperature=0.3,
                    system="You are a robotic command analyzer. Break down complex commands into reasoning requirements and planning needs.",
                    messages=[{"role": "user", "content": reasoning_prompt}]
                )
                analysis = response.content[0].text

            # Based on analysis, determine if we need to ask questions or can proceed with planning
            if "ask for clarification" in analysis.lower() or "unknown" in analysis.lower():
                # Extract question from analysis
                question_match = re.search(r"ask: (.+)", analysis, re.IGNORECASE)
                if question_match:
                    return TaskPlan(
                        steps=[{
                            "action": "ask_for_clarification",
                            "question": question_match.group(1),
                            "description": f"Need clarification: {question_match.group(1)}"
                        }],
                        estimated_time=10.0,
                        confidence=0.5,
                        dependencies=[]
                    )

            # Generate detailed plan based on analysis
            environment_context = self._get_environment_context()
            return self.llm_planner.plan_task(command, robot_state, environment_context)

        except Exception as e:
            print(f"Error in complex command handling: {e}")
            return None
```

## ROS 2 Integration for LLM-Based Planning

```python
# llm_planning_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from llm_integration import LLMPlanner, RobotState
from reasoning_engine import ReasoningEngine
import json
import threading

class LLMPlanningNode(Node):
    def __init__(self):
        super().__init__('llm_planning_node')

        # Publishers and subscribers
        self.command_sub = self.create_subscription(
            String, '/natural_language_command', self.command_callback, 10)
        self.status_pub = self.create_publisher(String, '/llm_status', 10)
        self.plan_pub = self.create_publisher(String, '/generated_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Initialize LLM components
        self.llm_planner = LLMPlanner(provider="anthropic", api_key="your-api-key")
        self.reasoning_engine = ReasoningEngine(self.llm_planner)

        # Robot state
        self.robot_state = RobotState(
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            battery_level=1.0,
            attached_objects=[],
            available_actions=['navigate', 'pick_up', 'place', 'push', 'report']
        )

        # Current plan and execution state
        self.current_plan = None
        self.plan_index = 0
        self.is_executing = False
        self.command_queue = []

        # Environment information
        self.laser_scan = None
        self.map_data = None

        # Subscriptions for environment data
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        # Timer for plan execution
        self.execution_timer = self.create_timer(0.1, self.execute_plan_step)

        # Lock for thread safety
        self.plan_lock = threading.Lock()

        self.get_logger().info('LLM Planning Node initialized')

    def command_callback(self, msg):
        """Handle incoming natural language commands."""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Add to command queue
        self.command_queue.append(command)

        # Process commands in a separate thread to avoid blocking
        threading.Thread(target=self.process_command_queue, daemon=True).start()

    def laser_callback(self, msg):
        """Update laser scan data."""
        self.laser_scan = msg

    def map_callback(self, msg):
        """Update map data."""
        self.map_data = msg

    def process_command_queue(self):
        """Process commands in the queue."""
        while self.command_queue:
            command = self.command_queue.pop(0)

            with self.plan_lock:
                # Generate plan for the command
                env_description = self.get_environment_description()
                plan = self.llm_planner.plan_task(command, self.robot_state, env_description)

                if plan:
                    self.current_plan = plan
                    self.plan_index = 0
                    self.is_executing = True

                    # Publish the plan
                    plan_msg = String()
                    plan_msg.data = json.dumps({
                        'command': command,
                        'plan': plan.steps,
                        'estimated_time': plan.estimated_time
                    })
                    self.plan_pub.publish(plan_msg)

                    self.get_logger().info(f'Generated plan with {len(plan.steps)} steps')
                else:
                    self.get_logger().error('Failed to generate plan')

    def get_environment_description(self) -> str:
        """Get current environment description."""
        description = "Environment:\n"

        # Add map information
        if self.map_data:
            description += f"Map resolution: {self.map_data.info.resolution}m\n"
            description += f"Map size: {self.map_data.info.width}x{self.map_data.info.height}\n"

        # Add laser scan information
        if self.laser_scan:
            # Analyze scan for obstacles
            min_distance = min([r for r in self.laser_scan.ranges if 0 < r < float('inf')], default=float('inf'))
            description += f"Closest obstacle: {min_distance:.2f}m ahead\n"

        # Add robot state
        description += f"Robot position: {self.robot_state.position}\n"
        description += f"Robot battery: {self.robot_state.battery_level * 100:.1f}%\n"

        return description

    def execute_plan_step(self):
        """Execute the current plan step by step."""
        if not self.is_executing or not self.current_plan:
            return

        if self.plan_index >= len(self.current_plan.steps):
            # Plan completed
            self.is_executing = False
            self.current_plan = None
            self.plan_index = 0

            status_msg = String()
            status_msg.data = "Plan completed successfully"
            self.status_pub.publish(status_msg)
            return

        # Get current step
        current_step = self.current_plan.steps[self.plan_index]

        # Execute the step
        success = self.execute_action(current_step)

        if success:
            self.plan_index += 1
            self.get_logger().info(f'Completed step {self.plan_index} of {len(self.current_plan.steps)}')

            if self.plan_index >= len(self.current_plan.steps):
                # Plan completed
                self.is_executing = False
                status_msg = String()
                status_msg.data = "Plan completed successfully"
                self.status_pub.publish(status_msg)
        else:
            # Plan execution failed
            self.is_executing = False
            status_msg = String()
            status_msg.data = f"Plan execution failed at step {self.plan_index + 1}"
            self.status_pub.publish(status_msg)

    def execute_action(self, action: Dict) -> bool:
        """Execute a single action from the plan."""
        action_type = action.get('action', 'unknown')

        if action_type == 'navigate_to':
            return self.execute_navigation(action)
        elif action_type == 'pick_up':
            return self.execute_pickup(action)
        elif action_type == 'place':
            return self.execute_placement(action)
        elif action_type == 'report':
            return self.execute_report(action)
        elif action_type == 'ask_for_clarification':
            return self.execute_ask_clarification(action)
        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')
            return False

    def execute_navigation(self, action) -> bool:
        """Execute navigation action."""
        target = action.get('target', action.get('location'))
        if not target:
            self.get_logger().error('No target specified for navigation')
            return False

        self.get_logger().info(f'Navigating to {target}')

        # In a real implementation, this would use navigation stack
        # For simulation, we'll just publish a simple velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.3  # Move forward
        self.cmd_vel_pub.publish(cmd_vel)

        # Simulate navigation completion after a short time
        # In real implementation, wait for navigation feedback
        return True

    def execute_pickup(self, action) -> bool:
        """Execute pickup action."""
        target = action.get('target')
        if not target:
            self.get_logger().error('No target specified for pickup')
            return False

        self.get_logger().info(f'Attempting to pick up {target}')

        # In real implementation, interface with gripper/arm
        # For simulation, just update robot state
        self.robot_state.attached_objects.append(target)

        return True

    def execute_placement(self, action) -> bool:
        """Execute placement action."""
        target = action.get('target')
        location = action.get('location')

        if not target:
            self.get_logger().error('No target specified for placement')
            return False

        self.get_logger().info(f'Placing {target} at {location}')

        # In real implementation, interface with gripper/arm
        # For simulation, just update robot state
        if target in self.robot_state.attached_objects:
            self.robot_state.attached_objects.remove(target)

        return True

    def execute_report(self, action) -> bool:
        """Execute report action."""
        message = action.get('message', 'Status report')
        self.get_logger().info(f'Report: {message}')

        status_msg = String()
        status_msg.data = message
        self.status_pub.publish(status_msg)

        return True

    def execute_ask_clarification(self, action) -> bool:
        """Execute ask for clarification action."""
        question = action.get('question', 'Unknown question')
        self.get_logger().info(f'Asking for clarification: {question}')

        status_msg = String()
        status_msg.data = f"Clarification needed: {question}"
        self.status_pub.publish(status_msg)

        # In a real implementation, wait for user response
        # For now, return True to continue
        return True

def main(args=None):
    rclpy.init(args=args)
    node = LLMPlanningNode()

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

## Advanced Planning Scenarios

```python
# advanced_scenarios.py
import asyncio
from typing import Dict, List
import time

class AdvancedScenarioPlanner:
    """Handles advanced planning scenarios using LLMs."""

    def __init__(self, llm_planner: LLMPlanner):
        self.llm_planner = llm_planner

    async def handle_multi_robot_coordination(self,
                                            commands: List[str],
                                            robot_states: List[RobotState]) -> List[TaskPlan]:
        """Plan coordinated actions for multiple robots."""
        coordination_prompt = f"""
        Coordinate the following tasks among multiple robots:

        Robot States:
        {json.dumps([{
            'id': i,
            'position': rs.position,
            'battery': rs.battery_level,
            'capabilities': rs.available_actions
        } for i, rs in enumerate(robot_states)], indent=2)}

        Commands:
        {json.dumps(commands, indent=2)}

        Generate coordinated plans that:
        1. Assign tasks based on robot proximity and capabilities
        2. Avoid conflicts and collisions
        3. Optimize overall completion time
        4. Consider battery levels and recharging needs

        Return plans for each robot in JSON format.
        """

        try:
            if self.llm_planner.provider == "openai":
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: openai.ChatCompletion.create(
                        model="gpt-4-turbo",
                        messages=[
                            {"role": "system", "content": "You are a multi-robot coordination expert. Generate coordinated plans that optimize task completion while avoiding conflicts."},
                            {"role": "user", "content": coordination_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=2000
                    )
                )
                plans_text = response.choices[0].message.content
            elif self.llm_planner.provider == "anthropic":
                response = self.llm_planner.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=2000,
                    temperature=0.3,
                    system="You are a multi-robot coordination expert. Generate coordinated plans that optimize task completion while avoiding conflicts.",
                    messages=[{"role": "user", "content": coordination_prompt}]
                )
                plans_text = response.content[0].text

            # Parse response - this would return multiple plans
            # Implementation depends on exact response format
            return self._parse_coordinated_plans(plans_text)

        except Exception as e:
            print(f"Error in multi-robot coordination: {e}")
            return []

    def _parse_coordinated_plans(self, response: str) -> List[TaskPlan]:
        """Parse coordinated plans from LLM response."""
        # Implementation depends on response format
        # This is a simplified version
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return [TaskPlan(
                    steps=plan.get('steps', []),
                    estimated_time=plan.get('estimated_time', 0),
                    confidence=plan.get('confidence', 0.8),
                    dependencies=plan.get('dependencies', [])
                ) for plan in data]
        except:
            pass
        return []

    async def handle_long_term_planning(self,
                                      goals: List[str],
                                      time_horizon: int,
                                      robot_state: RobotState) -> TaskPlan:
        """Plan for long-term goals over extended time periods."""
        long_term_prompt = f"""
        Plan for the following long-term goals over {time_horizon} hours:

        Goals:
        {json.dumps(goals, indent=2)}

        Current Robot State:
        - Position: {robot_state.position}
        - Battery: {robot_state.battery_level * 100:.1f}%
        - Available actions: {robot_state.available_actions}

        Consider:
        1. Battery management and charging schedules
        2. Daily/weekly patterns and routines
        3. Maintenance and calibration needs
        4. Weather and environmental factors
        5. Human interaction schedules
        6. Priority management between goals

        Generate a long-term plan with:
        - Daily schedules
        - Battery charging intervals
        - Task priorities
        - Contingency plans

        Return as detailed action plan with timestamps.
        """

        try:
            if self.llm_planner.provider == "openai":
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: openai.ChatCompletion.create(
                        model="gpt-4-turbo",
                        messages=[
                            {"role": "system", "content": "You are a long-term planning expert for autonomous robots. Create detailed schedules that balance multiple objectives over extended periods."},
                            {"role": "user", "content": long_term_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=1500
                    )
                )
                plan_text = response.choices[0].message.content
            elif self.llm_planner.provider == "anthropic":
                response = self.llm_planner.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1500,
                    temperature=0.3,
                    system="You are a long-term planning expert for autonomous robots. Create detailed schedules that balance multiple objectives over extended periods.",
                    messages=[{"role": "user", "content": long_term_prompt}]
                )
                plan_text = response.content[0].text

            return self.llm_planner._parse_plan_response(plan_text)

        except Exception as e:
            print(f"Error in long-term planning: {e}")
            return None

    def handle_uncertainty_and_contingency(self,
                                         base_plan: TaskPlan,
                                         potential_issues: List[str]) -> Dict[str, TaskPlan]:
        """Generate contingency plans for potential issues."""
        contingency_prompt = f"""
        Base plan:
        {json.dumps(base_plan.steps, indent=2)}

        Potential issues:
        {json.dumps(potential_issues, indent=2)}

        Generate contingency plans for each potential issue:
        1. Object not found
        2. Path blocked
        3. Battery low
        4. Gripper failure
        5. Communication loss

        For each issue, provide an alternative plan that achieves the same goals
        or gracefully handles the failure.
        """

        try:
            if self.llm_planner.provider == "openai":
                response = openai.ChatCompletion.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are a contingency planning expert. Generate alternative plans for handling various failure scenarios while maintaining mission objectives."},
                        {"role": "user", "content": contingency_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1500
                )
                contingency_text = response.choices[0].message.content
            elif self.llm_planner.provider == "anthropic":
                response = self.llm_planner.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1500,
                    temperature=0.3,
                    system="You are a contingency planning expert. Generate alternative plans for handling various failure scenarios while maintaining mission objectives.",
                    messages=[{"role": "user", "content": contingency_prompt}]
                )
                contingency_text = response.content[0].text

            # Parse contingency plans
            return self._parse_contingency_plans(contingency_text)

        except Exception as e:
            print(f"Error in contingency planning: {e}")
            return {}

    def _parse_contingency_plans(self, response: str) -> Dict[str, TaskPlan]:
        """Parse contingency plans from response."""
        # Implementation depends on response format
        # Simplified version
        try:
            data = json.loads(response)
            contingency_plans = {}
            for issue, plan_data in data.items():
                contingency_plans[issue] = TaskPlan(
                    steps=plan_data.get('steps', []),
                    estimated_time=plan_data.get('estimated_time', 0),
                    confidence=plan_data.get('confidence', 0.8),
                    dependencies=plan_data.get('dependencies', [])
                )
            return contingency_plans
        except:
            return {}

# Example usage function
def demonstrate_advanced_planning():
    """Demonstrate advanced planning capabilities."""
    # Initialize LLM planner
    llm_planner = LLMPlanner(provider="anthropic", api_key="your-api-key")
    scenario_planner = AdvancedScenarioPlanner(llm_planner)

    # Example: Multi-robot coordination
    commands = [
        "Robot 1: Go to kitchen and bring coffee to office",
        "Robot 2: Patrol the hallway every 30 minutes",
        "Robot 3: Monitor the entrance and report visitors"
    ]

    robot_states = [
        RobotState((0, 0, 0), (0, 0, 0, 1), 0.8, [], ['navigate', 'pick_up', 'place']),
        RobotState((5, 0, 0), (0, 0, 0, 1), 0.9, [], ['navigate', 'patrol']),
        RobotState((10, 0, 0), (0, 0, 0, 1), 0.7, [], ['navigate', 'monitor', 'report'])
    ]

    # This would be called with asyncio.run() in a real implementation
    # coordinated_plans = asyncio.run(scenario_planner.handle_multi_robot_coordination(commands, robot_states))

    print("Advanced planning scenarios demonstrated")

if __name__ == "__main__":
    demonstrate_advanced_planning()
```

## Next Steps

In the next chapter, we'll explore how to implement the final integration between all the components we've developed - connecting the vision-language-action system with the LLM-based planning to create a complete, intelligent robotic system that can understand natural language commands, perceive its environment, plan complex tasks, and execute them successfully.