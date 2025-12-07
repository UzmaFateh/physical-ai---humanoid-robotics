---
title: 'Chapter 13: Whisper - Integrating Voice Commands'
---

# Chapter 13: Whisper - Integrating Voice Commands

The most natural way for humans to interact is through speech. For robots, enabling voice commands dramatically enhances their usability and intuition. In this chapter, we explore how to integrate **OpenAI Whisper**, a powerful Automatic Speech Recognition (ASR) system, to allow our humanoid robot to understand spoken commands.

## Automatic Speech Recognition (ASR)

ASR is the technology that converts spoken language into text. Traditional ASR systems often struggle with background noise, accents, or domain-specific terminology. OpenAI Whisper, however, is a large, general-purpose ASR model trained on a massive dataset of audio and text, making it highly robust and accurate across diverse conditions and languages.

## Why Whisper?

-   **High Accuracy**: Whisper excels at transcribing speech even in challenging acoustic environments.
-   **Multi-lingual**: It supports transcription in many languages, and can also perform language identification.
-   **Robustness**: Handles various accents, background noise, and technical jargon surprisingly well.
-   **Simplicity of API**: OpenAI provides an easy-to-use API that abstracts away the complexity of running a large machine learning model.

## Integrating Whisper with ROS 2

Since Whisper is typically accessed via an API (cloud-based or local server), the integration with ROS 2 involves a node that handles:

1.  **Audio Capture**: Receiving audio input (e.g., from a microphone).
2.  **Audio Processing**: Packaging the audio for Whisper (e.g., converting to a supported format like `.wav` or `.mp3`).
3.  **API Call**: Sending the audio to the Whisper API.
4.  **Text Publishing**: Receiving the transcribed text and publishing it to a ROS 2 topic (e.g., `/speech_to_text`).

Other ROS 2 nodes can then subscribe to this `/speech_to_text` topic to receive the human's command as plain text, ready for further processing by an LLM or a rule-based system.

### Audio Capture in ROS 2

ROS 2 doesn't have a built-in "microphone node" out of the box, but you can use:

-   **`audio_common` package**: This package (often available for ROS 1 but with efforts for ROS 2) provides nodes for audio capture and playback.
-   **Custom `rclpy` node**: You can write a Python ROS 2 node that uses libraries like `PyAudio` or `Sounddevice` to capture audio from your system's microphone. This node would then publish `audio_msgs/AudioData` or simply save audio chunks to a buffer for Whisper processing.

<h2>Whisper API Usage (Python Example)</h2>

Assuming you have an audio file (`command.wav`), calling the Whisper API is straightforward with the `openai` Python client library:

```python
import openai
import os

# Replace with your OpenAI API key or set as an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY") 

def transcribe_audio(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text" # Or "json" for more details
        )
    return transcript

if __name__ == '__main__':
    # This assumes a 'voice_command.wav' file exists in the current directory
    # In a ROS node, this would come from a microphone input
    transcribed_text = transcribe_audio("voice_command.wav")
    print(f"Transcribed: {transcribed_text}")
```

In a ROS 2 system, your Python node would:
1.  Continuously record small chunks of audio (e.g., 5-10 seconds).
2.  Send these chunks to Whisper.
3.  Publish the resulting text.
4.  Optionally, use Voice Activity Detection (VAD) to only send audio when speech is detected, reducing API calls and improving responsiveness.

<h2>Challenges and Considerations</h2>

-   **Latency**: Cloud-based API calls introduce network latency. For critical, real-time commands, a local Whisper deployment (if possible on the robot's hardware) might be preferable.
-   **Cost**: Cloud APIs incur costs per use. Batching audio or using VAD can help manage this.
-   **Privacy**: Be mindful of privacy concerns when sending audio data to cloud services.
-   **Wake Word Detection**: For truly intuitive interaction, a robot needs to know *when* to listen. This usually involves a "wake word" (e.g., "Hey Robot") that triggers the ASR system, preventing continuous transcription of ambient noise.

Despite these challenges, integrating ASR is a significant step towards enabling natural language interfaces for your humanoid robot.

---

<h3>Lab 13.1: Transcribing a Voice Command</h3>

**Problem Statement**: Use the OpenAI Whisper API to transcribe a recorded voice command into text using a Python script.

**Expected Outcome**: You will provide an audio file, and the script will output the accurately transcribed text.

**Steps**:

1.  **Get an OpenAI API Key**: If you don't have one, create an account on [OpenAI Platform](https://platform.openai.com/) and generate an API key.

2.  **Install OpenAI Python Library**:
    ```bash
    pip install openai
    ```

3.  **Record a Voice Command**:
    -   Use your computer's microphone to record a short audio file.
    -   Speak a clear command, such as "Robot, please pick up the red block."
    -   Save the file as `voice_command.wav` (or another supported format) in your working directory. Ensure it's a relatively high-quality recording.

4.  **Create Python Script**: Create a Python file named `transcribe_command.py` in `src/code-examples/module4/`.

5.  **Add Script Content**:
    ```python
    import openai
    import os

    # Set your OpenAI API key. It's best practice to use environment variables.
    # export OPENAI_API_KEY='your_api_key_here'
    # Or you can paste it directly here (NOT recommended for production):
    openai.api_key = os.getenv("OPENENAI_API_KEY") 

    def transcribe_audio(audio_file_path):
        if not openai.api_key:
            print("Error: OPENAI_API_KEY environment variable not set.")
            print("Please set it using: export OPENAI_API_KEY='your_api_key_here'")
            return "API Key not set."

        try:
            with open(audio_file_path, "rb") as audio_file:
                print(f"Transcribing {audio_file_path} using Whisper...")
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            return transcript
        except FileNotFoundError:
            return f"Error: Audio file not found at {audio_file_path}"
        except openai.APIConnectionError as e:
            return f"The server could not be reached: {e.__cause__}"
        except openai.RateLimitError as e:
            return f"A rate limit was exceeded: {e.response}"
        except openai.APIStatusError as e:
            return f"Another non-200-range status code was received: {e.response}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    if __name__ == '__main__':
        audio_filename = "voice_command.wav"
        transcribed_text = transcribe_audio(audio_filename)
        print(f"Transcribed Text: {transcribed_text}")

    ```

6.  **Run the Script**:
    ```bash
    # Set your API key first (replace 'sk-...' with your actual key)
    export OPENAI_API_KEY='sk-...' 
    python transcribe_command.py
    ```

7.  **Verify**: The script should print the transcribed text of your voice command.

**Conclusion**: You have successfully integrated an advanced ASR system into your workflow. This transcribed text is the crucial input for the next stage: intelligent planning with Large Language Models.

---

<h2>References</h2>

[1] Robot Operating System (ROS) Website: https://www.ros.org/
[2] ROS 2 Documentation: https://docs.ros.org/en/humble/index.html
[3] Data Distribution Service (DDS) Standard: https://www.omg.org/dds/
[4] Gazebo Documentation: http://gazebosim.org/
[5] Unity Robotics Hub: https://github.com/Unity-Technologies/Unity-Robotics-Hub
[6] NVIDIA Isaac Sim Documentation: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html
[7] NVIDIA Isaac ROS Documentation: https://nvidia-isaac-ros.github.io/
[8] Nav2 Documentation: https://navigation.ros.org/
[9] OpenAI Whisper API Documentation: https://platform.openai.com/docs/guides/speech-to-text
[10] OpenAI API Documentation: https://platform.openai.com/docs/api-reference
[11] IEEE Editorial Style Manual: https://www.ieee.org/content/dam/ieee-org/ieee/web/org/pubs/ieee_style_manual.pdf