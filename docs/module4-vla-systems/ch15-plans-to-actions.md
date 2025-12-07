---
title: 'Chapter 15: From LLM Plans to ROS 2 Actions'
---

# Chapter 15: From LLM Plans to ROS 2 Actions

We've seen how Whisper can convert speech to text and how an LLM can generate a high-level action plan from that text. The final, crucial step in our Vision-Language-Action (VLA) pipeline is to convert these abstract LLM-generated actions into concrete, executable commands for our ROS 2 robot. This chapter focuses on the "Action Executor" node that closes the loop.

## The Action Executor Node

The Action Executor node acts as the bridge between the cognitive planning layer (LLM) and the robot's low-level control systems. Its primary responsibilities are:

1.  **Subscribe to LLM Plans**: It listens to a ROS 2 topic (e.g., `/robot_plan`) where the LLM Planner node publishes its structured action sequences.
2.  **Parse and Validate**: It parses the incoming plan (e.g., a JSON array of actions) and validates each action against the robot's known capabilities and current state. This is a critical safety step to prevent the robot from executing hallucinated or impossible commands.
3.  **Execute Actions**: For each action in the plan, it translates it into the appropriate ROS 2 primitive (publishing to a topic, calling a service, or sending an action goal).
4.  **Monitor Progress**: It monitors the execution of each action, waiting for completion or detecting failures.
5.  **Provide Feedback**: It can publish feedback to a status topic or even back to the LLM Planner for replanning if an action fails.

## Mapping LLM Actions to ROS 2 Primitives

Let's revisit our example LLM plan:

```json
[
  {"action": "say_text", "args": {"text": "Okay, I will pick up the red block and place it on the table."}},
  {"action": "move_to_object", "args": {"object_name": "red_block"}},
  {"action": "pick_object", "args": {"object_name": "red_block"}},
  {"action": "move_to_object", "args": {"object_name": "table"}},
  {"action": "place_object", "args": {"location_name": "table"}},
  {"action": "say_text", "args": {"text": "I have completed your request!"}}
]
```

The Action Executor node needs to know how to handle each of these custom actions:

1.  **`say_text(text)`**: This could map to:
    *   **Topic**: Publishing a `std_msgs/String` message to a `/robot_say` topic, which a text-to-speech (TTS) node subscribes to.
    *   **Service**: Calling a `tts_service` with the text as a request.

2.  **`move_to_object(object_name)`**: This is a more complex action. It would typically involve:
    *   **Perception**: Getting the `object_name`'s current pose from an object detection node (e.g., via a ROS 2 service or a lookup in `tf2`).
    *   **Navigation**: Sending a `geometry_msgs/PoseStamped` goal message to the Nav2 action server (e.g., `/navigate_to_pose`). This is a ROS 2 Action, so the executor would be an Action Client.

3.  **`pick_object(object_name)`**: This is an even more complex manipulation action, likely involving:
    *   **Motion Planning**: Using a motion planning framework like MoveIt 2 to calculate a trajectory to grasp the object.
    *   **Gripper Control**: Sending commands to open/close the robot's gripper.
    *   This would typically be a ROS 2 Action (e.g., `/pick`) provided by a higher-level manipulation controller node.

## A Python Action Executor Example

Here's a conceptual Python ROS 2 node for an Action Executor. It simplifies the actual robot interaction for clarity.

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time

# Assuming you have an Action client for navigation and picking.
# For simplicity, we'll just print messages here.

class ActionExecutor(Node):
    def __init__(self):
        super().__init__('action_executor')
        self.subscription = self.create_subscription(
            String,
            'robot_plan',
            self.plan_callback,
            10)
        self.subscription # prevent unused variable warning
        self.say_publisher = self.create_publisher(String, 'robot_say', 10)
        self.get_logger().info('Action Executor node started. Waiting for plans...')

        # Mock current world state (in a real robot, this would come from sensors/localization)
        self.robot_at_location = "start_area"
        self.held_object = None

    def _execute_say_text(self, text):
        msg = String()
        msg.data = text
        self.say_publisher.publish(msg)
        self.get_logger().info(f"Robot says: {text}")
        time.sleep(1) # Simulate speech duration
        return True

    def _execute_move_to_object(self, object_name):
        self.get_logger().info(f"Moving to {object_name}...")
        # In a real system, this would be a Nav2 action goal
        time.sleep(2) # Simulate movement time
        # Update mock state
        self.robot_at_location = object_name 
        self.get_logger().info(f"Reached {object_name}.")
        return True

    def _execute_pick_object(self, object_name):
        self.get_logger().info(f"Attempting to pick {object_name}...")
        # In a real system, this would be a MoveIt 2 action goal
        if self.robot_at_location == object_name and not self.held_object:
            self.held_object = object_name
            self.get_logger().info(f"Picked up {object_name}.")
            time.sleep(1)
            return True
        else:
            self.get_logger().warn(f"Cannot pick {object_name}. Not at location or already holding something.")
            return False

    def _execute_place_object(self, location_name):
        self.get_logger().info(f"Attempting to place object at {location_name}...")
        # In a real system, this would involve precise manipulation
        if self.held_object and self.robot_at_location == location_name:
            self.get_logger().info(f"Placed {self.held_object} at {location_name}.")
            self.held_object = None
            time.sleep(1)
            return True
        else:
            self.get_logger().warn(f"Cannot place object. Not holding anything or not at location.")
            return False

    def plan_callback(self, msg):
        self.get_logger().info(f"Received plan: {msg.data}")
        try:
            plan = json.loads(msg.data)
            if not isinstance(plan, list):
                self.get_logger().error("Received plan is not a list.")
                return

            self.get_logger().info("Executing plan...")
            for action_data in plan:
                action_type = action_data.get("action")
                args = action_data.get("args", {})
                
                success = False
                if action_type == "say_text":
                    success = self._execute_say_text(args.get("text", "Unknown message"))
                elif action_type == "move_to_object":
                    success = self._execute_move_to_object(args.get("object_name"))
                elif action_type == "pick_object":
                    success = self._execute_pick_object(args.get("object_name"))
                elif action_type == "place_object":
                    success = self._execute_place_object(args.get("location_name"))
                else:
                    self.get_logger().error(f"Unknown action type: {action_type}. Skipping.")
                    success = False
                
                if not success:
                    self.get_logger().error(f"Action '{action_type}' failed. Stopping plan execution.")
                    break # Stop if an action fails
                
            self.get_logger().info("Plan execution finished.")

        except json.JSONDecodeError:
            self.get_logger().error(f"Failed to parse JSON plan: {msg.data}")
        except Exception as e:
            self.get_logger().error(f"Error during plan execution: {e}")

def main(args=None):
    rclpy.init(args=args)
    executor_node = ActionExecutor()
    rclpy.spin(executor_node)
    executor_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Considerations for Real-World Deployment

-   **Robust Error Handling**: Real robots fail. The Action Executor must be able to detect failures (e.g., navigation goal not reached, gripper malfunction) and potentially trigger replanning by the LLM.
-   **Concurrency**: A robot might need to perform multiple actions concurrently (e.g., move while monitoring sensors). This requires careful design of the executor.
-   **Feedback Loop**: Providing continuous feedback to the LLM Planner about the success or failure of actions, and the updated world state, is crucial for dynamic replanning.
-   **Human Supervision**: For safety, an operator should always be able to halt or override the robot's actions.
-   **Security**: Ensure that LLM-generated commands cannot be exploited to perform malicious or unsafe actions.

This Action Executor completes our VLA pipeline, transforming abstract commands into purposeful robot behavior.

---

### Lab 15.1: Executing a Mock Robot Plan

**Problem Statement**: Create a ROS 2 node that simulates an LLM Planner publishing a plan, and run the Action Executor node to process and "execute" this plan.

**Expected Outcome**: The Action Executor node will receive the JSON plan and print messages indicating the execution of each step.

**Steps**:

1.  **Create Action Executor Node**: Save the Python code for the `ActionExecutor` class above into a file named `action_executor_node.py` in your `src/code-examples/module4/` directory.

2.  **Create Mock LLM Planner Node**: Create a new Python file named `mock_llm_planner.py` in `src/code-examples/module4/`. This node will publish a hardcoded plan.
    ```python
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    import json
    import time

    class MockLLMPlanner(Node):
        def __init__(self):
            super().__init__('mock_llm_planner')
            self.publisher_ = self.create_publisher(String, 'robot_plan', 10)
            self.timer = self.create_timer(5.0, self.timer_callback) # Publish plan every 5 seconds
            self.get_logger().info('Mock LLM Planner node started.')
            self.plan_published = False

        def timer_callback(self):
            if not self.plan_published:
                mock_plan = [
                  {"action": "say_text", "args": {"text": "Hello! I will now perform a task."}},
                  {"action": "move_to_object", "args": {"object_name": "red_block"}},
                  {"action": "pick_object", "args": {"object_name": "red_block"}},
                  {"action": "move_to_object", "args": {"object_name": "box"}},
                  {"action": "place_object", "args": {"location_name": "box"}},
                  {"action": "say_text", "args": {"text": "Task completed!"}}
                ]
                msg = String()
                msg.data = json.dumps(mock_plan)
                self.publisher_.publish(msg)
                self.get_logger().info("Published mock plan.")
                self.plan_published = True # Only publish once

    def main(args=None):
        rclpy.init(args=args)
        mock_planner_node = MockLLMPlanner()
        rclpy.spin(mock_planner_node)
        mock_planner_node.destroy_node()
        rclpy.shutdown()

    if __name__ == '__main__':
        main()
    ```

3.  **Run the Nodes**:
    -   Open a terminal, source your ROS 2 setup, and run the Action Executor:
        ```bash
        ros2 run <your_pkg_name> action_executor_node
        ```
        (Replace `<your_pkg_name>` with the actual name of your package if you put these into a ROS 2 package). For this example, you might need to run them as simple Python scripts:
        ```bash
        python src/code-examples/module4/action_executor_node.py
        ```
    -   Open a second terminal, source your ROS 2 setup, and run the Mock LLM Planner:
        ```bash
        python src/code-examples/module4/mock_llm_planner.py
        ```

4.  **Verify**: In the terminal running the `action_executor_node.py`, you should see it receiving the plan and then printing messages about the simulated execution of `say_text`, `move_to_object`, `pick_object`, and `place_object`.

**Conclusion**: You have successfully demonstrated the final stage of the VLA pipeline, where an LLM's plan is translated into a sequence of robot actions. This fundamental architecture can be expanded to control highly complex robotic systems with intuitive natural language commands.
