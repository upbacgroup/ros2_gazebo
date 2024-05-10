#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, GetWorldProperties

import numpy as np


class TargetPoints(Node):
    def __init__(self):
        super().__init__('target_points')
        self.spawned_balls = []
        self.positions_matrix = None
        self.spawn_model_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_model_client = self.create_client(DeleteEntity, '/delete_entity')
        #self.get_world_properties_client = self.create_client(GetWorldProperties, '/get_world_properties')
        while not self.client.service_is_ready():
            self.get_logger().info('Waiting for /spawn_entity service...')
            rclpy.spin_once(self)

        self.entity_name = 'my_entity'
        self.urdf_path = '/path/to/your/model.urdf'

        self.spawn_entity()

    def spawn(self):
        req = SpawnEntity.Request()
        req.name = self.entity_name
        req.robot_namespace = ''
        req.initial_pose.position.x = 1.0
        req.initial_pose.position.y = 1.0
        req.initial_pose.position.z = 1.0
        self.future = self.client.send_request(req)
        self.get_logger().info(f'Sent spawn request for entity: {self.entity_name}')

        try:
            response = self.future.result()
            if response.success:
                self.get_logger().info(f'Entity {self.entity_name} spawned successfully!')
            else:
                self.get_logger().error(f'Failed to spawn entity: {response.reason}')
        except Exception as e:
            self.get_logger().error(f'An error occurred: {e}')



def main(args=None):
    rclpy.init(args=args)
    node = TargetPoints()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
