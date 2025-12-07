---
title: 'Chapter 14: LLM Cognitive Planning'
---

# Chapter 14: LLM Cognitive Planning

Once a robot receives a voice command and transcribes it to text (as we saw with Whisper), the next challenge is to understand the user's intent and convert that high-level request into a series of executable robot actions. This is where **Large Language Models (LLMs)** come into play, providing a powerful capability for cognitive planning.

## What is Cognitive Planning for Robots?

Cognitive planning for robots involves:

1.  **Understanding Natural Language**: Interpreting ambiguous or high-level human instructions.
2.  **State Assessment**: Considering the current environment and the robot's capabilities.
3.  **Task Decomposition**: Breaking down a complex instruction into smaller, manageable sub-tasks.
4.  **Action Sequencing**: Ordering these sub-tasks into a logical sequence of actions the robot can perform.
5.  **Error Handling**: Anticipating potential failures and planning for recovery.

Traditionally, this required complex, hand-coded state machines or symbolic AI planners, which were brittle and difficult to scale. LLMs offer a more flexible and robust approach.

## How LLMs Facilitate Planning

LLMs, with their vast knowledge of language and general reasoning capabilities, can act as the "brain" of a robot's planning system. They can:

-   **Interpret User Intent**: Understand various phrasings of the same command (e.g., "pick up the block," "grab the cube," "take the red object").
-   **Grounding**: Map abstract concepts in human language (e.g., "red block," "table") to specific objects and locations in the robot's environment model. This is a critical step often done by providing the LLM with information about the scene.
-   **Generate Action Sequences**: Output a sequence of primitive actions that the robot can execute, given its available tools and skills.

### Prompt Engineering for Robot Planning

The key to using LLMs for planning is **prompt engineering**. You need to craft a prompt that:

1.  **Defines the Robot's Capabilities**: List the specific, low-level functions (ROS 2 actions, services, or topics) the robot can execute. This is like giving the LLM a "skill set."
    *   Example: `pick_object(object_id)`, `move_to_pose(x, y, z, qx, qy, qz, qw)`, `say_text(text)`.
2.  **Provides Context**: Give the LLM information about the current state of the world, available objects, their locations, and the robot's current pose.
3.  **Instructs the LLM**: Clearly tell the LLM to output a sequence of actions from its skill set that fulfills the user's request.
4.  **Specifies Output Format**: Crucially, instruct the LLM to output the plan in a structured, parseable format (e.g., JSON or a specific command language) so that your ROS 2 system can execute it.

### Example Prompt Structure

```text
You are a helpful robot assistant. Your goal is to generate a sequence of robot commands to fulfill user requests.

Here are the actions you can perform:
- move_to_object(object_name): Move to the vicinity of an object.
- pick_object(object_name): Pick up a specified object. Requires being near the object.
- place_object(location_name): Place the held object at a specified location. Requires being near the location.
- say_text(text): Speak a message to the user.

Current World State:
- Robot is at (0.5, 0.5, 0).
- Available objects: 'red_block' at (1.0, 0.2, 0), 'blue_ball' at (0.3, 0.8, 0).
- Locations: 'table' at (1.5, 0.5, 0), 'box' at (0.0, 0.0, 0).

User Request: "Can you grab the red block and put it on the table?"

Your plan (output as a list of actions in JSON format):
```

The LLM might then respond with something like:

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

## Integrating LLMs with ROS 2

Similar to Whisper, you would have a ROS 2 node that:

1.  **Subscribes to `/speech_to_text`**: Receives the transcribed user command.
2.  **Gathers Context**: Queries other ROS 2 nodes (e.g., an object detection node, a localization node) for the current world state.
3.  **Constructs Prompt**: Combines the user command, robot capabilities, and world state into a prompt for the LLM.
4.  **Calls LLM API**: Sends the prompt to an LLM (e.g., OpenAI's GPT-4).
5.  **Parses Response**: Validates and parses the structured action sequence from the LLM.
6.  **Publishes Actions**: Publishes individual actions (or the entire plan) to another ROS 2 topic (e.g., `/robot_plan`) for an "Action Executor" node to consume.

## Challenges and Future Directions

-   **Hallucinations**: LLMs can sometimes generate incorrect or impossible actions. Robust error checking and replanning mechanisms are crucial.
-   **Real-time Constraints**: API calls introduce latency. For very dynamic environments, faster, more local models might be needed.
-   **Safety**: Ensuring the LLM-generated plans are safe and do not lead to dangerous situations for the robot or its surroundings.
-   **Learning New Skills**: Advanced systems might allow the LLM to dynamically define new, more complex skills by combining existing ones.

Despite these challenges, LLM-based cognitive planning represents a paradigm shift in how we program and interact with intelligent robots, moving towards more intuitive and flexible human-robot collaboration.

---

### Lab 14.1: Generating a Robot Plan with an LLM

**Problem Statement**: Use an LLM (e.g., an OpenAI model) to generate a sequence of robot actions in JSON format, given a natural language command and a simplified world state.

**Expected Outcome**: Your Python script will take a user command and output a valid JSON array of robot actions.

**Steps**:

1.  **Install OpenAI Python Library**: If you haven't already: `pip install openai`.

2.  **Get an OpenAI API Key**: As in the previous lab.

3.  **Create Python Script**: Create a Python file named `llm_planner.py` in `src/code-examples/module4/`.

4.  **Add Script Content**:
    ```python
    import openai
    import os
    import json

    openai.api_key = os.getenv("OPENAI_API_KEY") 

    def get_robot_plan(user_request, current_world_state):
        if not openai.api_key:
            return "Error: OPENAI_API_KEY environment variable not set."

        robot_capabilities = """
        Here are the actions you can perform:
        - move_to_object(object_name): Move to the vicinity of an object.
        - pick_object(object_name): Pick up a specified object. Requires being near the object.
        - place_object(location_name): Place the held object at a specified location. Requires being near the location.
        - say_text(text): Speak a message to the user.
        """

        prompt = f"""
        You are a helpful robot assistant. Your goal is to generate a sequence of robot commands to fulfill user requests.
        {robot_capabilities}

        Current World State:
        {json.dumps(current_world_state, indent=2)}

        User Request: "{user_request}"

        Your plan (output ONLY a JSON array of actions, like [{"action": "say_text", "args": {"text": "..."}}], do NOT include any other text):
        """

        messages = [
            {"role": "system", "content": "You are a robot planning assistant."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo", # Or "gpt-4" for more advanced reasoning
                messages=messages,
                temperature=0.0 # Make the output more deterministic
            )
            plan_json_str = response.choices[0].message.content
            return json.loads(plan_json_str) # Parse the JSON string
        except openai.APIConnectionError as e:
            return f"The server could not be reached: {e.__cause__}"
        except openai.RateLimitError as e:
            return f"A rate limit was exceeded: {e.response}"
        except openai.APIStatusError as e:
            return f"Another non-200-range status code was received: {e.response}"
        except json.JSONDecodeError:
            return f"LLM returned invalid JSON: {plan_json_str}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    if __name__ == '__main__':
        world_state = {
            "robot_location": "kitchen",
            "available_objects": [
                {"name": "red_block", "location": "counter"},
                {"name": "blue_ball", "location": "floor"}
            ],
            "known_locations": ["counter", "table", "box", "kitchen"]
        }

        user_command = "Can you please take the red block from the counter and put it in the box?"
        plan = get_robot_plan(user_command, world_state)
        
        print("\n--- Generated Plan ---")
        if isinstance(plan, list):
            for step in plan:
                print(step)
        else:
            print(plan)
    ```

5.  **Run the Script**:
    ```bash
    export OPENAI_API_KEY='sk-...' # Replace with your actual key
    python llm_planner.py
    ```

6.  **Verify**: The script should output a JSON array of actions similar to the example shown in this chapter.

**Conclusion**: You've successfully used an LLM for cognitive planning. This demonstrates how a robot can translate high-level natural language instructions into a series of structured, executable actions, bridging the gap between human intent and robotic execution.
