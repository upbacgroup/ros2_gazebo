#!/usr/bin/env python3
import numpy as np
import rospy 
from gazebo_msgs.srv import DeleteModel, DeleteModelRequest, SpawnModel, GetWorldProperties
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelStates


# rospy.init_node('target_points')
spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

            
def spawn(number):
    rospy.wait_for_service("/gazebo/spawn_sdf_model")

    # random positions for 3 goal points
    global positions_matrix
    positions_matrix = np.random.uniform(-3, 3, (4, 3))
    # positions_matrix = np.array([[2.0, 0.0, 1.0],
    #                     [2.0, 2.0, 1.0],
    #                     [4.0, 4.0, 1.0]])
    for i in range(positions_matrix.shape[0]):
        positions_matrix[i, -1] = 1.0

    positions_matrix = np.array(positions_matrix)
    
    model_sdf = open("/home/ecem/.gazebo/models/green_ball/model.sdf", "r").read()
    model_namespace = ""
    # Set the initial pose
    initial_pose = Pose()
    if number == 4:
        model_name = "green_ball_4"
        initial_pose.position.x = 0
        initial_pose.position.y = 0
        initial_pose.position.z = 1.0

    else:
        model_name = f"green_ball_{number}"
        initial_pose.position.x = positions_matrix[number - 1,0]
        initial_pose.position.y = positions_matrix[number - 1,1]
        initial_pose.position.z = positions_matrix[number - 1,2]



    success, status_message = spawn_target_points(model_name, model_sdf, model_namespace, initial_pose)

    if success:
        rospy.loginfo(f"Successfully spawned {model_name} in Gazebo!")
    else:
        rospy.logerr(f"Failed to spawn {model_name} in Gazebo. Error: {status_message}")

    return positions_matrix[number - 1] if number != 4 else np.array([0, 0, 1.0])
  
def model_states_callback(msg):
    global spawned_balls
    spawned_balls = [model_name for model_name in msg.name if "green_ball_" in model_name]

# Function to check if any balls are present in the environment
def check_balls_in_environment():
    return len(spawned_balls) > 0

rospy.Subscriber('/gazebo/model_states', ModelStates, model_states_callback)

def spawn_target_points(model_name, model_sdf, model_namespace, initial_pose):
    try:
        resp = spawn_model(model_name, model_sdf, model_namespace, initial_pose, "world")
        return resp.success, resp.status_message
    except rospy.ServiceException as e:
        return False, f"Service call failed: {str(e)}"

def delete_object_service(object_name):
    rospy.wait_for_service('/gazebo/get_world_properties')
    rospy.wait_for_service('/gazebo/delete_model')    
  
    try:
        get_world_properties = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        # Get the list of models in the Gazebo environment
        model_names = get_world_properties().model_names

        if object_name in model_names:
            # Object exists, proceed with deletion
            req = DeleteModelRequest()
            req.model_name = object_name
            resp = delete_model(req)
            rospy.loginfo(f"Deleted object: {object_name}")
            # if object_name in target_positions:
            #     del target_positions[object_name]
            return True
        else:
            rospy.logwarn(f"Object '{object_name}' not found in the environment, skipping deletion.")
            return False
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

            