#!/usr/bin/env python3
import numpy as np
import rospy 
from gazebo_msgs.srv import DeleteModel, DeleteModelRequest, SpawnModel
from geometry_msgs.msg import Pose


# rospy.init_node('target_points')
spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

# random positions for 3 goal points
# Create a 4x3 matrix with random values between -4 and 4
positions_matrix = np.random.uniform(1, 5, (4, 3))
# Set the last item of each row to 1
for i in range(positions_matrix.shape[0] - 1):
    positions_matrix[i, -1] = 0.5
# Ensure the last row remains 0,0,1
positions_matrix[3, :] = np.array([0, 0, 0.5])
positions_matrix = np.array(positions_matrix)

            
def spawn(number):
    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    
    model_sdf = open("/home/ecem/.gazebo/models/green_ball_1/model.sdf", "r").read()
    model_namespace = ""
    # Set the initial pose
    initial_pose = Pose()
    if number == 1:
        print("1st ball spawned")
        model_name = "green_ball_1"
        initial_pose.position.x = positions_matrix[0,0]
        initial_pose.position.y = positions_matrix[0,1]
        initial_pose.position.z = positions_matrix[0,2]
        number = 0

    elif number == 2:
        print("2nd ball spawned")
        model_name = "green_ball_2"
        initial_pose.position.x = positions_matrix[1,0]
        initial_pose.position.y = positions_matrix[1,1]
        initial_pose.position.z = positions_matrix[1,2]
        number = 0

    elif number == 3:
        print("3rd ball spawned")
        model_name = "green_ball_3"
        initial_pose.position.x = positions_matrix[2,0]
        initial_pose.position.y = positions_matrix[2,1]
        initial_pose.position.z = positions_matrix[2,2]
        number = 0

    elif number == 4:
        print("4th ball spawned")
        model_name = "green_ball_4"
        initial_pose.position.x = positions_matrix[3,0]
        initial_pose.position.y = positions_matrix[3,1]
        initial_pose.position.z = positions_matrix[3,2] 
        number = 0   

    success, status_message = spawn_target_points(model_name, model_sdf, model_namespace, initial_pose)

    if success:
        rospy.loginfo(f"Successfully spawned {model_name} in Gazebo!")
    # else:
    #     rospy.logerr(f"Failed to spawn {model_name} in Gazebo. Error: {status_message}")

def spawn_target_points(model_name, model_sdf, model_namespace, initial_pose):
    try:
        resp = spawn_model(model_name, model_sdf, model_namespace, initial_pose, "world")
        return resp.success, resp.status_message
    except rospy.ServiceException as e:
        return False, f"Service call failed: {str(e)}"

def delete_object_service(object_name):
    rospy.wait_for_service('/gazebo/delete_model')    
  
    req = DeleteModelRequest()
    req.model_name = str(object_name)  # Specify the name of the object to delete
    resp = delete_model(req)
    rospy.loginfo(f"Deleted object: {object_name}")
            
    # except rospy.ServiceException as e:
    #     rospy.logerr(f"Failed to delete object: {object_name}, error: {e}")

          