---
sidebar_position: 3
---

# Chapter 7: Unity Visualization for Robotics

## Introduction to Unity for Robotics Visualization

Unity provides an extremely powerful platform for creating high-fidelity visualizations of robotic systems. Unlike Gazebo, which is primarily focused on physics simulation, Unity excels at creating visually appealing, interactive environments that can be used for operator interfaces, training scenarios, and system monitoring.

The Unity Robotics ecosystem includes several tools and packages that make integration with ROS 2 more straightforward:

1. **Unity Robotics Hub**: Centralized access to robotics packages
2. **ROS# (ROS TCP Connector)**: Communication bridge between Unity and ROS 2
3. **Unity Perception**: Tools for generating synthetic training data
4. **Unity ML-Agents**: Framework for training intelligent agents

## Setting Up Unity for Robotics

To get started with Unity for robotics, you'll need to:

1. Install Unity Hub and Unity 2021.3 LTS or later
2. Install the ROS TCP Connector package
3. Set up your Unity project structure

### Installing ROS TCP Connector

The ROS TCP Connector is the primary communication bridge between Unity and ROS 2. Here's how to install it:

1. Open Unity Hub and create a new 3D project
2. Go to Window → Package Manager
3. Click the + button → Add package from git URL
4. Enter: `https://github.com/Unity-Technologies/ROS-TCP-Connector.git`

### Basic Unity Scene Setup

Let's create a Unity scene that visualizes our robot:

```csharp
// Assets/Scripts/RobotVisualizationManager.cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Nav;

public class RobotVisualizationManager : MonoBehaviour
{
    [Header("Robot Components")]
    public Transform robotBase;
    public Transform leftWheel;
    public Transform rightWheel;
    public Transform laserScanner;
    public GameObject[] jointObjects; // Array of joint objects to animate

    [Header("Visualization Settings")]
    public float wheelRadius = 0.1f;
    public float wheelSeparation = 0.3f;
    public bool useRealtimeData = true;

    [Header("ROS Settings")]
    public string rosIP = "127.0.0.1";
    public int rosPort = 10000;

    // ROS connection
    private ROSConnection ros;

    // Robot state
    private Dictionary<string, float> jointPositions = new Dictionary<string, float>();
    private Vector3 robotPosition = Vector3.zero;
    private Quaternion robotRotation = Quaternion.identity;
    private List<float> laserRanges = new List<float>();
    private Vector3 robotLinearVel = Vector3.zero;
    private Vector3 robotAngularVel = Vector3.zero;

    void Start()
    {
        // Initialize ROS connection
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIP, rosPort);

        // Subscribe to ROS topics
        ros.Subscribe<JointStateMsg>("joint_states", JointStateCallback);
        ros.Subscribe<LaserScanMsg>("scan", LaserScanCallback);
        ros.Subscribe<OdometryMsg>("odom", OdometryCallback);

        // Set up joint objects if not assigned manually
        if (jointObjects.Length == 0)
        {
            SetupJointObjects();
        }
    }

    void SetupJointObjects()
    {
        // Find all joint objects in the scene
        jointObjects = new GameObject[transform.childCount];
        for (int i = 0; i < transform.childCount; i++)
        {
            jointObjects[i] = transform.GetChild(i).gameObject;
        }
    }

    void JointStateCallback(JointStateMsg jointState)
    {
        // Update joint positions dictionary
        for (int i = 0; i < jointState.name.Count; i++)
        {
            if (i < jointState.position.Count)
            {
                jointPositions[jointState.name[i]] = (float)jointState.position[i];
            }
        }
    }

    void LaserScanCallback(LaserScanMsg scan)
    {
        laserRanges.Clear();
        foreach (var range in scan.ranges)
        {
            if (range >= scan.range_min && range <= scan.range_max)
            {
                laserRanges.Add((float)range);
            }
            else
            {
                laserRanges.Add(scan.range_max); // Use max range for invalid readings
            }
        }
    }

    void OdometryCallback(OdometryMsg odom)
    {
        // Update robot position
        robotPosition = new Vector3(
            (float)odom.pose.pose.position.x,
            (float)odom.pose.pose.position.y,
            (float)odom.pose.pose.position.z
        );

        // Update robot rotation
        robotRotation = new Quaternion(
            (float)odom.pose.pose.orientation.x,
            (float)odom.pose.pose.orientation.y,
            (float)odom.pose.pose.orientation.z,
            (float)odom.pose.pose.orientation.w
        );

        // Update velocities
        robotLinearVel = new Vector3(
            (float)odom.twist.twist.linear.x,
            (float)odom.twist.twist.linear.y,
            (float)odom.twist.twist.linear.z
        );

        robotAngularVel = new Vector3(
            (float)odom.twist.twist.angular.x,
            (float)odom.twist.twist.angular.y,
            (float)odom.twist.twist.angular.z
        );
    }

    void Update()
    {
        if (useRealtimeData)
        {
            UpdateRobotVisualization();
        }

        // Visualize laser scan data
        VisualizeLaserScan();
    }

    void UpdateRobotVisualization()
    {
        // Update robot position and rotation
        if (robotBase != null)
        {
            robotBase.position = robotPosition;
            robotBase.rotation = robotRotation;
        }

        // Update wheel rotations based on joint positions
        if (jointPositions.ContainsKey("left_wheel_joint") && leftWheel != null)
        {
            leftWheel.localRotation = Quaternion.Euler(90, 0, jointPositions["left_wheel_joint"] * Mathf.Rad2Deg);
        }

        if (jointPositions.ContainsKey("right_wheel_joint") && rightWheel != null)
        {
            rightWheel.localRotation = Quaternion.Euler(90, 0, jointPositions["right_wheel_joint"] * Mathf.Rad2Deg);
        }

        // Update other joints
        foreach (var joint in jointPositions)
        {
            // Find corresponding joint object and update its rotation
            foreach (GameObject jointObj in jointObjects)
            {
                if (jointObj.name.ToLower().Contains(joint.Key.ToLower().Replace("_joint", "")))
                {
                    // For revolute joints, update rotation
                    jointObj.transform.localRotation = Quaternion.Euler(0, 0, joint.Value * Mathf.Rad2Deg);
                    break;
                }
            }
        }
    }

    void VisualizeLaserScan()
    {
        // Create or update laser scan visualization
        // This is a simplified version - in practice you'd create line renderers or point clouds
        if (laserRanges.Count > 0 && laserScanner != null)
        {
            // Clear previous visualization (in a real implementation)
            // Create new visualization based on current laser data
            VisualizeLaserRays();
        }
    }

    void VisualizeLaserRays()
    {
        // This method creates visual representations of laser scan rays
        // For each laser reading, create a line from the scanner position
        if (laserRanges.Count == 0) return;

        float angleMin = -Mathf.PI / 2; // Assuming our laser scans 180 degrees
        float angleIncrement = Mathf.PI / (laserRanges.Count - 1);

        for (int i = 0; i < laserRanges.Count; i++)
        {
            float angle = angleMin + i * angleIncrement;
            float distance = laserRanges[i];

            Vector3 direction = new Vector3(
                Mathf.Cos(angle) * distance,
                0,
                Mathf.Sin(angle) * distance
            );

            // In a real implementation, you would draw lines or use particle systems
            // This is just a conceptual example
            Debug.DrawRay(laserScanner.position, laserScanner.TransformDirection(direction), Color.red, 0.1f);
        }
    }

    // Method to send velocity commands to ROS
    public void SendVelocityCommand(float linearX, float angularZ)
    {
        var twistMsg = new TwistMsg();
        twistMsg.linear = new Vector3Msg(linearX, 0, 0);
        twistMsg.angular = new Vector3Msg(0, 0, angularZ);

        ros.Send("cmd_vel", twistMsg);
    }

    // Method to reset robot position (for simulation)
    public void ResetRobot()
    {
        var resetMsg = new EmptyMsg();
        ros.Send("reset_simulation", resetMsg);
    }
}
```

### Advanced Visualization Components

Let's create more sophisticated visualization components:

```csharp
// Assets/Scripts/LaserScanVisualizer.cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;

public class LaserScanVisualizer : MonoBehaviour
{
    [Header("Visualization Settings")]
    public Material laserMaterial;
    public float maxDistance = 10.0f;
    public Color validColor = Color.red;
    public Color invalidColor = Color.gray;
    public float pointSize = 0.05f;

    [Header("Performance Settings")]
    public int maxPoints = 1000;
    public bool useLineRenderer = false;

    private LineRenderer lineRenderer;
    private List<GameObject> pointObjects = new List<GameObject>();
    private List<float> currentRanges = new List<float>();
    private float angleMin, angleMax, angleIncrement;

    void Start()
    {
        if (useLineRenderer)
        {
            lineRenderer = gameObject.AddComponent<LineRenderer>();
            lineRenderer.material = laserMaterial;
            lineRenderer.startWidth = 0.02f;
            lineRenderer.endWidth = 0.02f;
        }
    }

    public void UpdateLaserScan(LaserScanMsg scan)
    {
        currentRanges.Clear();

        // Store scan parameters
        angleMin = (float)scan.angle_min;
        angleMax = (float)scan.angle_max;
        angleIncrement = (float)scan.angle_increment;

        // Process ranges
        foreach (var range in scan.ranges)
        {
            if (range >= scan.range_min && range <= scan.range_max)
            {
                currentRanges.Add((float)range);
            }
            else
            {
                currentRanges.Add(float.MaxValue); // Invalid range
            }
        }

        UpdateVisualization();
    }

    void UpdateVisualization()
    {
        if (useLineRenderer)
        {
            UpdateLineRenderer();
        }
        else
        {
            UpdatePointVisualization();
        }
    }

    void UpdateLineRenderer()
    {
        if (lineRenderer == null || currentRanges.Count == 0) return;

        lineRenderer.positionCount = currentRanges.Count;

        for (int i = 0; i < currentRanges.Count; i++)
        {
            float angle = angleMin + i * angleIncrement;
            float distance = currentRanges[i];

            Vector3 point;
            if (distance < maxDistance && distance != float.MaxValue)
            {
                point = new Vector3(
                    Mathf.Cos(angle) * distance,
                    0,
                    Mathf.Sin(angle) * distance
                );
            }
            else
            {
                // For invalid ranges, don't show anything or show max distance
                point = new Vector3(
                    Mathf.Cos(angle) * maxDistance,
                    0,
                    Mathf.Sin(angle) * maxDistance
                );
            }

            lineRenderer.SetPosition(i, transform.TransformPoint(point));
        }
    }

    void UpdatePointVisualization()
    {
        // Clean up old point objects
        foreach (GameObject pointObj in pointObjects)
        {
            if (pointObj != null)
            {
                DestroyImmediate(pointObj);
            }
        }
        pointObjects.Clear();

        // Create new point objects for valid ranges
        for (int i = 0; i < Mathf.Min(currentRanges.Count, maxPoints); i++)
        {
            float angle = angleMin + i * angleIncrement;
            float distance = currentRanges[i];

            if (distance < maxDistance && distance != float.MaxValue)
            {
                Vector3 point = new Vector3(
                    Mathf.Cos(angle) * distance,
                    0,
                    Mathf.Sin(angle) * distance
                );

                GameObject pointObj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                pointObj.transform.position = transform.TransformPoint(point);
                pointObj.transform.localScale = Vector3.one * pointSize;

                // Set material color based on distance
                Renderer renderer = pointObj.GetComponent<Renderer>();
                if (renderer != null)
                {
                    renderer.material = new Material(laserMaterial);
                    float colorIntensity = 1.0f - (distance / maxDistance);
                    renderer.material.color = new Color(validColor.r * colorIntensity,
                                                      validColor.g * colorIntensity,
                                                      validColor.b * colorIntensity,
                                                      validColor.a);
                }

                // Make it a child of this object
                pointObj.transform.parent = transform;

                pointObjects.Add(pointObj);
            }
        }
    }

    // Clear the visualization
    public void ClearVisualization()
    {
        if (lineRenderer != null)
        {
            lineRenderer.positionCount = 0;
        }

        foreach (GameObject pointObj in pointObjects)
        {
            if (pointObj != null)
            {
                DestroyImmediate(pointObj);
            }
        }
        pointObjects.Clear();
    }
}
```

### Environment Visualization

Let's create a script for visualizing the environment and navigation:

```csharp
// Assets/Scripts/EnvironmentVisualizer.cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Nav;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;

public class EnvironmentVisualizer : MonoBehaviour
{
    [Header("Map Visualization")]
    public Material obstacleMaterial;
    public Material freeSpaceMaterial;
    public Material unknownSpaceMaterial;
    public float resolution = 0.1f;

    [Header("Path Visualization")]
    public Material pathMaterial;
    public Material goalMaterial;
    public float pathLineWidth = 0.1f;

    [Header("Grid Settings")]
    public int gridSize = 100;
    public float cellSize = 0.5f;

    private GameObject[,] gridCells;
    private LineRenderer pathRenderer;
    private GameObject goalMarker;

    void Start()
    {
        InitializeGrid();
        InitializePathRenderer();
        InitializeGoalMarker();
    }

    void InitializeGrid()
    {
        gridCells = new GameObject[gridSize, gridSize];

        for (int x = 0; x < gridSize; x++)
        {
            for (int y = 0; y < gridSize; y++)
            {
                GameObject cell = GameObject.CreatePrimitive(PrimitiveType.Quad);
                cell.transform.position = new Vector3(x * cellSize, 0, y * cellSize) - new Vector3(gridSize * cellSize / 2, 0.01f, gridSize * cellSize / 2);
                cell.transform.localScale = Vector3.one * cellSize * 0.9f;
                cell.transform.rotation = Quaternion.Euler(90, 0, 0); // Face upward

                Renderer renderer = cell.GetComponent<Renderer>();
                renderer.material = new Material(freeSpaceMaterial);
                renderer.material.color = Color.green;

                gridCells[x, y] = cell;
            }
        }
    }

    void InitializePathRenderer()
    {
        GameObject pathObj = new GameObject("Path");
        pathRenderer = pathObj.AddComponent<LineRenderer>();
        pathRenderer.material = new Material(pathMaterial);
        pathRenderer.startWidth = pathLineWidth;
        pathRenderer.endWidth = pathLineWidth;
        pathRenderer.useWorldSpace = true;
    }

    void InitializeGoalMarker()
    {
        goalMarker = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        goalMarker.transform.localScale = new Vector3(0.3f, 0.1f, 0.3f);
        goalMarker.GetComponent<Renderer>().material = new Material(goalMaterial);
        goalMarker.GetComponent<Renderer>().material.color = Color.blue;
        goalMarker.SetActive(false);
    }

    public void UpdateOccupancyGrid(sbyte[] data, int width, int height, float resolution, Vector3 origin)
    {
        if (data.Length != width * height) return;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = y * width + x;
                if (index >= data.Length) continue;

                sbyte value = data[index];
                Material cellMaterial = GetMaterialForValue(value);

                if (x < gridSize && y < gridSize && gridCells[x, y] != null)
                {
                    Renderer renderer = gridCells[x, y].GetComponent<Renderer>();
                    if (renderer != null)
                    {
                        renderer.material = new Material(cellMaterial);
                    }
                }
            }
        }
    }

    Material GetMaterialForValue(sbyte value)
    {
        if (value == 100) // Occupied
        {
            return obstacleMaterial;
        }
        else if (value == 0) // Free
        {
            return freeSpaceMaterial;
        }
        else // Unknown (-1) or other values
        {
            return unknownSpaceMaterial;
        }
    }

    public void UpdatePath(List<Vector3> pathPoints)
    {
        if (pathRenderer == null || pathPoints.Count == 0) return;

        pathRenderer.positionCount = pathPoints.Count;
        for (int i = 0; i < pathPoints.Count; i++)
        {
            pathRenderer.SetPosition(i, pathPoints[i]);
        }
    }

    public void SetGoal(Vector3 goalPosition)
    {
        if (goalMarker != null)
        {
            goalMarker.transform.position = new Vector3(goalPosition.x, 0.05f, goalPosition.y);
            goalMarker.SetActive(true);
        }
    }

    public void ClearPath()
    {
        if (pathRenderer != null)
        {
            pathRenderer.positionCount = 0;
        }
    }

    public void ClearGoal()
    {
        if (goalMarker != null)
        {
            goalMarker.SetActive(false);
        }
    }

    // Update the visualization with a simple path
    public void SetSimplePath(Vector3 start, Vector3 goal)
    {
        List<Vector3> path = new List<Vector3>();
        path.Add(new Vector3(start.x, 0.1f, start.y));
        path.Add(new Vector3(goal.x, 0.1f, goal.y));

        UpdatePath(path);
        SetGoal(goal);
    }
}
```

### Unity Scene Controller

Now let's create a scene controller that manages the overall visualization:

```csharp
// Assets/Scripts/UnitySceneController.cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Nav;

public class UnitySceneController : MonoBehaviour
{
    [Header("Visualization Components")]
    public RobotVisualizationManager robotViz;
    public LaserScanVisualizer laserViz;
    public EnvironmentVisualizer envViz;

    [Header("UI Elements")]
    public Text statusText;
    public Text positionText;
    public Text velocityText;
    public Slider linearVelSlider;
    public Slider angularVelSlider;
    public Button resetButton;
    public Button pauseButton;

    [Header("Simulation Settings")]
    public string rosIP = "127.0.0.1";
    public int rosPort = 10000;
    public float simulationSpeed = 1.0f;

    private ROSConnection ros;
    private bool isPaused = false;
    private Vector3 lastRobotPosition = Vector3.zero;

    void Start()
    {
        // Initialize ROS connection
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIP, rosPort);

        // Subscribe to topics
        ros.Subscribe<OdometryMsg>("odom", OdometryCallback);
        ros.Subscribe<LaserScanMsg>("scan", LaserScanCallback);
        ros.Subscribe<JointStateMsg>("joint_states", JointStateCallback);

        // Setup UI event listeners
        if (linearVelSlider != null)
            linearVelSlider.onValueChanged.AddListener(OnLinearVelChanged);

        if (angularVelSlider != null)
            angularVelSlider.onValueChanged.AddListener(OnAngularVelChanged);

        if (resetButton != null)
            resetButton.onClick.AddListener(OnResetClicked);

        if (pauseButton != null)
            pauseButton.onClick.AddListener(OnPauseClicked);

        UpdateUI();
    }

    void OdometryCallback(OdometryMsg odom)
    {
        // Update robot position text
        Vector3 pos = new Vector3((float)odom.pose.pose.position.x,
                                 (float)odom.pose.pose.position.y,
                                 (float)odom.pose.pose.position.z);

        Vector3 vel = new Vector3((float)odom.twist.twist.linear.x,
                                 (float)odom.twist.twist.linear.y,
                                 (float)odom.twist.twist.linear.z);

        if (positionText != null)
        {
            positionText.text = $"Position: ({pos.x:F2}, {pos.y:F2}, {pos.z:F2})";
        }

        if (velocityText != null)
        {
            velocityText.text = $"Velocity: ({vel.x:F2}, {vel.y:F2}, {vel.z:F2})";
        }

        // Update environment visualization with current position
        if (envViz != null)
        {
            // For example, highlight the robot's position on the map
        }
    }

    void LaserScanCallback(LaserScanMsg scan)
    {
        if (laserViz != null)
        {
            laserViz.UpdateLaserScan(scan);
        }
    }

    void JointStateCallback(JointStateMsg jointState)
    {
        // Joint states are handled by the robot visualization manager
    }

    void OnLinearVelChanged(float value)
    {
        if (!isPaused)
        {
            SendVelocityCommand(value, angularVelSlider.value);
        }
    }

    void OnAngularVelChanged(float value)
    {
        if (!isPaused)
        {
            SendVelocityCommand(linearVelSlider.value, value);
        }
    }

    void OnResetClicked()
    {
        // Send reset command to ROS
        var resetMsg = new EmptyMsg();
        ros.Send("reset_simulation", resetMsg);

        // Reset Unity visualization
        if (robotViz != null)
        {
            robotViz.ResetRobot();
        }

        if (laserViz != null)
        {
            laserViz.ClearVisualization();
        }

        if (envViz != null)
        {
            envViz.ClearPath();
            envViz.ClearGoal();
        }
    }

    void OnPauseClicked()
    {
        isPaused = !isPaused;
        pauseButton.GetComponentInChildren<Text>().text = isPaused ? "Resume" : "Pause";

        if (isPaused)
        {
            // Send zero velocity when paused
            SendVelocityCommand(0, 0);
        }
    }

    void SendVelocityCommand(float linear, float angular)
    {
        var twistMsg = new TwistMsg();
        twistMsg.linear = new Vector3Msg(linear, 0, 0);
        twistMsg.angular = new Vector3Msg(0, 0, angular);

        ros.Send("cmd_vel", twistMsg);
    }

    void UpdateUI()
    {
        if (statusText != null)
        {
            statusText.text = isPaused ? "PAUSED" : "RUNNING";
            statusText.color = isPaused ? Color.red : Color.green;
        }
    }

    void Update()
    {
        UpdateUI();
    }

    // Method to set a navigation goal
    public void SetNavigationGoal(Vector3 goal)
    {
        var goalMsg = new PoseStampedMsg();
        goalMsg.header = new HeaderMsg();
        goalMsg.header.frame_id = "map";
        goalMsg.pose.position = new Vector3Msg(goal.x, goal.y, goal.z);
        goalMsg.pose.orientation = new QuaternionMsg(0, 0, 0, 1);

        ros.Send("move_base_simple/goal", goalMsg);

        // Update visualization
        if (envViz != null)
        {
            envViz.SetGoal(goal);
        }
    }

    // Method to send emergency stop
    public void EmergencyStop()
    {
        SendVelocityCommand(0, 0);
        Debug.Log("Emergency stop command sent!");
    }
}
```

### Advanced Unity Features for Robotics

Let's implement some advanced features that make Unity particularly valuable for robotics:

```csharp
// Assets/Scripts/RobotTelemetryDashboard.cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Diagnostic;

public class RobotTelemetryDashboard : MonoBehaviour
{
    [Header("Telemetry Display")]
    public Text batteryText;
    public Text cpuUsageText;
    public Text memoryUsageText;
    public Text temperatureText;
    public Text jointStatusText;
    public Slider batterySlider;

    [Header("Safety Systems")]
    public Text safetyStatusText;
    public Color safeColor = Color.green;
    public Color warningColor = Color.yellow;
    public Color dangerColor = Color.red;

    private float batteryLevel = 100.0f;
    private float cpuUsage = 0.0f;
    private float memoryUsage = 0.0f;
    private float temperature = 25.0f; // Celsius

    // Simulated telemetry data
    private float lastUpdate = 0f;
    private float updateInterval = 1.0f; // Update every second

    void Update()
    {
        // Simulate telemetry updates (in real implementation, this would come from ROS topics)
        if (Time.time - lastUpdate > updateInterval)
        {
            UpdateTelemetry();
            lastUpdate = Time.time;
        }

        UpdateUI();
    }

    void UpdateTelemetry()
    {
        // Simulate realistic telemetry values
        batteryLevel = Mathf.Clamp(batteryLevel - Random.Range(0.01f, 0.05f), 0f, 100f);
        cpuUsage = Mathf.Clamp(Random.Range(10f, 80f), 0f, 100f);
        memoryUsage = Mathf.Clamp(Random.Range(20f, 70f), 0f, 100f);
        temperature = Mathf.Clamp(Random.Range(20f, 40f), 0f, 80f); // Keep within safe limits

        // Simulate joint status (in real implementation, this would come from joint state messages)
        int operationalJoints = Random.Range(8, 12); // Assume 12 total joints
        int totalJoints = 12;

        jointStatusText.text = $"Joints: {operationalJoints}/{totalJoints} operational";
    }

    void UpdateUI()
    {
        if (batteryText != null)
            batteryText.text = $"Battery: {batteryLevel:F1}%";

        if (cpuUsageText != null)
            cpuUsageText.text = $"CPU: {cpuUsage:F1}%";

        if (memoryUsageText != null)
            memoryUsageText.text = $"Memory: {memoryUsage:F1}%";

        if (temperatureText != null)
            temperatureText.text = $"Temp: {temperature:F1}°C";

        if (batterySlider != null)
            batterySlider.value = batteryLevel / 100f;

        // Update safety status based on telemetry
        UpdateSafetyStatus();
    }

    void UpdateSafetyStatus()
    {
        if (safetyStatusText == null) return;

        // Determine safety level based on telemetry
        if (temperature > 70f || batteryLevel < 10f || cpuUsage > 90f)
        {
            safetyStatusText.text = "DANGER";
            safetyStatusText.color = dangerColor;
        }
        else if (temperature > 60f || batteryLevel < 20f || cpuUsage > 75f)
        {
            safetyStatusText.text = "WARNING";
            safetyStatusText.color = warningColor;
        }
        else
        {
            safetyStatusText.text = "SAFE";
            safetyStatusText.color = safeColor;
        }
    }

    // Method to update from real ROS diagnostic messages
    public void UpdateFromDiagnostic(DiagnosticArrayMsg diagArray)
    {
        foreach (var status in diagArray.status)
        {
            switch (status.name)
            {
                case "battery":
                    // Parse battery level from diagnostic message
                    foreach (var value in status.values)
                    {
                        if (value.key == "percentage")
                        {
                            float.TryParse(value.value, out batteryLevel);
                            break;
                        }
                    }
                    break;

                case "cpu":
                    // Parse CPU usage
                    foreach (var value in status.values)
                    {
                        if (value.key == "load")
                        {
                            float.TryParse(value.value, out cpuUsage);
                            break;
                        }
                    }
                    break;

                case "memory":
                    // Parse memory usage
                    foreach (var value in status.values)
                    {
                        if (value.key == "used")
                        {
                            float.TryParse(value.value, out memoryUsage);
                            break;
                        }
                    }
                    break;

                case "temperature":
                    // Parse temperature
                    foreach (var value in status.values)
                    {
                        if (value.key == "current")
                        {
                            float.TryParse(value.value, out temperature);
                            break;
                        }
                    }
                    break;
            }
        }
    }

    // Method to trigger emergency procedures
    public void TriggerEmergencyProcedure()
    {
        Debug.Log("EMERGENCY: Safety threshold exceeded!");

        // In a real system, this would send emergency stop commands
        // and potentially trigger backup safety systems
    }
}
```

## Building Unity Applications for Robotics

To build a Unity application that can interface with your ROS 2 system:

1. **Set up the build**: File → Build Settings → Add Scenes
2. **Configure player settings**:
   - Set the correct IP and port for ROS connection
   - Optimize for your target platform
3. **Build the application**: Select your platform and build

For a headless Linux build (common in robotics):
- Open Build Settings
- Select Linux as the platform
- Choose x64 architecture
- Check "Headless Mode" if running on a server
- Build and run with ROS 2

## Integration Best Practices

When integrating Unity with your robotics system:

1. **Performance**: Keep Unity running at a consistent frame rate while maintaining ROS communication
2. **Synchronization**: Ensure visualization updates don't lag behind the actual robot state
3. **Network reliability**: Handle network interruptions gracefully
4. **Resource management**: Optimize for the computational resources available
5. **Safety**: Include emergency stop capabilities in the visualization interface

## Next Steps

In the next chapter, we'll look at how to integrate Unity visualization with the overall system architecture, including deployment considerations and real-world use cases for robotics visualization.