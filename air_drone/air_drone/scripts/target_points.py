#!/usr/bin/env python3
import numpy as np
import rclpy
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelStates

# rclpy.init_node('target_points')
spawn_entity = None
delete_entity = None

def spawn(number):
    global spawn_entity
    global delete_entity

    rclpy.get_logger().info("Waiting for Gazebo services...")
    spawn_entity = rclpy.create_client(SpawnEntity, "/gazebo/spawn_entity")
    delete_entity = rclpy.create_client(DeleteEntity, "/gazebo/delete_entity")

    if not spawn_entity.wait_for_service(timeout_sec=5.0) or not delete_entity.wait_for_service(timeout_sec=5.0):
        rclpy.get_logger().error("Failed to connect to Gazebo services.")
        return None

    positions_matrix = np.random.uniform(-3, 3, (4, 3))
    for i in range(positions_matrix.shape[0]):
        positions_matrix[i, -1] = 1.0

    positions_matrix = np.array(positions_matrix)

    model_sdf = open("/home/ecem/.gazebo/models/green_ball/model.sdf", "r").read()
    model_namespace = ""

    initial_pose = Pose()
    if number == 4:
        model_name = "green_ball_4"
        initial_pose.position.x = 0
        initial_pose.position.y = 0
        initial_pose.position.z = 1.0
    else:
        model_name = f"green_ball_{number}"
        initial_pose.position.x = positions_matrix[number - 1, 0]
        initial_pose.position.y = positions_matrix[number - 1, 1]
        initial_pose.position.z = positions_matrix[number - 1, 2]

    request = SpawnEntity.Request()
    request.name = model_name
    request.xml = model_sdf
    request.initial_pose = initial_pose
    request.reference_frame = "world"

    future = spawn_entity.call_async(request)
    rclpy.spin_until_future_complete(spawn_entity, future)

    if future.result() is not None:
        rclpy.get_logger().info(f"Successfully spawned {model_name} in Gazebo!")
    else:
        rclpy.get_logger().error(f"Failed to spawn {model_name} in Gazebo.")

    return positions_matrix[number - 1] if number != 4 else np.array([0, 0, 1.0])

def model_states_callback(msg):
    global spawned_balls
    spawned_balls = [model_name for model_name in msg.name if "green_ball_" in model_name]

def check_balls_in_environment():
    return len(spawned_balls) > 0

def delete_object_service(object_name):
    global delete_entity

    try:
        request = DeleteEntity.Request()
        request.name = object_name

        future = delete_entity.call_async(request)
        rclpy.spin_until_future_complete(delete_entity, future)

        if future.result() is not None:
            rclpy.get_logger().info(f"Deleted object: {object_name}")
            return True
        else:
            rclpy.get_logger().warn(f"Object '{object_name}' not found in the environment, skipping deletion.")
            return False
    except Exception as e:
        rclpy.get_logger().error(f"Service call failed: {str(e)}")

def main(args=None):
    global spawn_entity
    global delete_entity
    global spawned_balls

    rclpy.init(args=args)
    spawned_balls = []

    node = rclpy.create_node('target_points')
    node.create_subscription(ModelStates, '/gazebo/model_states', model_states_callback)

    spawn(1)
    # Optionally spawn other balls here

    while rclpy.ok() and not check_balls_in_environment():
        rclpy.spin_once(node)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
