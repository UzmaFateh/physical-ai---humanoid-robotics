using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor; // For JointState message
using System.Collections.Generic;

/// <summary>
/// This script subscribes to the /joint_states topic and applies the
/// received joint angles to a robot model in the Unity scene.
/// It assumes the robot model has a GameObject for each joint, and these
-/// are mapped by name.
/// </summary>
public class RosJointController : MonoBehaviour
{
    // Dictionary to map joint names to their corresponding GameObjects
    public Dictionary<string, GameObject> jointMap;

    // A list to be populated in the Unity Inspector
    [System.Serializable]
    public class JointMapping
    {
        public string rosJointName;
        public GameObject jointGameObject;
    }
    public List<JointMapping> jointMappings;

    void Start()
    {
        // Initialize the dictionary from the Inspector list
        jointMap = new Dictionary<string, GameObject>();
        foreach (var mapping in jointMappings)
        {
            jointMap[mapping.rosJointName] = mapping.jointGameObject;
        }

        // Subscribe to the /joint_states topic
        ROSConnection.GetOrCreateInstance().Subscribe<JointStateMsg>("joint_states", OnJointStateReceived);
        Debug.Log("Subscribed to /joint_states");
    }

    void OnJointStateReceived(JointStateMsg jointState)
    {
        // Loop through all the joints in the received message
        for (int i = 0; i < jointState.name.Length; i++)
        {
            string jointName = jointState.name[i];
            
            // Check if we have this joint in our map
            if (jointMap.ContainsKey(jointName))
            {
                // Get the target angle in radians
                float targetRadian = (float)jointState.position[i];

                // Convert radians to degrees for Unity's transform
                float targetDegree = targetRadian * Mathf.Rad2Deg;

                // Get the joint's GameObject
                GameObject jointObject = jointMap[jointName];

                // Apply the rotation.
                // This assumes a simple revolute joint rotating around the Y-axis.
                // A more complex robot would need to know the correct axis for each joint.
                jointObject.transform.localEulerAngles = new Vector3(0, targetDegree, 0);
            }
        }
    }
}
