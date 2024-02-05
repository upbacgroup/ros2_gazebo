#!/usr/bin/env python3
import rospy
import gym
import cv2
import imutils
import tensorflow as tf
import os
import math
import numpy as np
import time
import stable_baselines3 as sb3
import tensorflow as tf


from cv_bridge import CvBridge
from gym import spaces
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage
from air_drone.msg import Position, MotorSpeed
from stable_baselines3.common.noise import NormalActionNoise

from target_points import positions_matrix
from target_points import spawn
from target_points import delete_object_service
import sensor_msgs.point_cloud2 as pc2

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
		
        rospy.init_node('drone_env')
        self.bridge = CvBridge()
        self.closest_distance = float('inf')
        self.width = 640
        self.height = 480
        self.distance = 100

        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0]),
                                       high=np.array([1.0, 1.0, 1.0, 1.0]),
                                       dtype=np.float32)  
                      
        # self.observation_space = spaces.Box(low=0, high=255,
		# 									shape=(1, ), 
        #                                     dtype=np.uint8) # height and width
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(self.height, self.width), 
                                            dtype=np.uint8)
        
        self.action_noise = NormalActionNoise(mean=np.zeros(self.action_space.shape), 
                                              sigma=0.3 * np.ones(self.action_space.shape))


        # Publishers, subscribers and services
        self.current_pose_sub = rospy.Subscriber("pose", Position, self.position_callback)  
        self.z_speed = rospy.Publisher("motor_speed_cmd", MotorSpeed, queue_size=1) # motor speed for z direction
        self.speed_motors_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)  # motor speed 
        self.reset_simulation_service = rospy.ServiceProxy('/gazebo/reset_simulation', Empty) # resets gazebo  
        self.image_sub = rospy.Subscriber("/depth_camera/depth/image_raw", Image, self.camera_callback) # depth camera
        self.point_sub = rospy.Subscriber("/depth_camera/depth/points", pc2.PointCloud2, self.point_callback)

    def point_callback(self, data):
        points = np.array(list(pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)))
        # Calculate the distance between the camera and the closest point in the point cloud
        self.distance = np.min(np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2))
        # rospy.loginfo("Distance: %f", self.distance)

    def camera_callback(self, image_msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        # Replace NaN values with zeros in the copied array
        self.depth_image = np.nan_to_num(self.depth_image)

        if np.any(np.isnan(self.depth_image)):
            print("***********nan values in depth_image************")    
        
    def position_callback(self, pos_msg):
        self.pose_x, self.pose_y, self.pose_z = pos_msg.x, pos_msg.y, pos_msg.z
        self.pitch, self.roll = pos_msg.pitch, pos_msg.roll

        self.current_position = [self.pose_x, self.pose_y, self.pose_z]

    def object_name_finder(self):
                
        object_name = None
        target_1 = np.take(positions_matrix, 0, axis=0)
        target_2 = np.take(positions_matrix, 1, axis=0)
        target_3 = np.take(positions_matrix, 2, axis=0)

        self.dist_target_1 = np.linalg.norm(self.current_position - target_1)
        self.dist_target_2 = np.linalg.norm(self.current_position - target_2)
        self.dist_target_3 = np.linalg.norm(self.current_position - target_3)

        if  self.dist_target_1 < 0.5:
            if self.flag_1 == False:
                object_name = 'green_ball_1'
                delete_object_service(object_name)
                # rospy.loginfo(f"object name in finder func:{object_name}")
                self.collected_target_points += 1
                self.flag_1 = True
                
        if  self.dist_target_2 < 0.5: 
            if self.flag_2 == False:
                object_name = 'green_ball_2' 
                delete_object_service(object_name)
                # rospy.loginfo(f"object name in finder func:{object_name}")
                self.collected_target_points += 1
                self.flag_2 = True
                

        if self.dist_target_3 < 0.5: 
            if self.flag_3 == False:
                object_name = 'green_ball_3'
                delete_object_service(object_name)
                # rospy.loginfo(f"object name in finder func:{object_name}")
                self.collected_target_points += 1
                self.flag_3 = True

        return object_name 

    def check_collision(self):

        collision_count = 0
        if collision_count == 0 and self.distance < 1.0 or abs(self.roll) > math.radians(15) or abs(self.pitch) > math.radians(15):

            delete_object_service('green_ball_1')
            delete_object_service('green_ball_2')
            delete_object_service('green_ball_3')
            delete_object_service('green_ball_4')

            self.done = True
           #reset func
            self.flag_1 = False
            self.flag_2 = False
            self.flag_3 = False

            self.linear_x = 0.0
            self.dist_target_1 = 10000
            self.dist_target_2 = 10000
            self.dist_target_3 = 10000

            self.collected_target_points = 0
            self.pose_x, self.pose_y, self.pose_z = 0.0, 0.0, 0.0
            self.current_position = [self.pose_x, self.pose_y, self.pose_z]

            self.spawn_last_target = False

            self.width = 640
            self.height = 480

            self.time_taken = 0
            self.reward = 0

            self.depth_image = np.zeros((self.height, self.width)) 
            self.observation = self.depth_image
  
            spawn(1)
            spawn(2)
            spawn(3)
            self.reset_simulation_service() 
            

            # self.collected_target_points = 0

            # self.time_taken = 0
            # self.reward = 0 

            collision_count += 1
            return self.done
        return self.done    
            
    def take_action(self, action):
        self.object_name_finder()
        self.vel_cmd = Twist()
        self.z_cmd = MotorSpeed()

        self.linear_x = 2.5*(action[0]+1)/2 #scaling (-1,1) to (0,1) because velocity should be positive
        linear_y = action[1]*2.5
        angular_z = action[2]*2.5
        z_vel = action[3]*125

        self.vel_cmd.linear.x = self.linear_x
        self.vel_cmd.linear.y = linear_y
        self.vel_cmd.angular.z = angular_z
        self.speed_motors_pub.publish(self.vel_cmd)

        self.z_cmd.name = ['propeller1', 'propeller2', 'propeller3', 'propeller4', 'propeller5', 'propeller6']
        self.z_cmd.velocity = [z_vel, -z_vel , z_vel, -z_vel, z_vel, -z_vel]
        self.z_speed.publish(self.z_cmd)

    def step(self, action):
        self.take_action(action)
        self.check_collision()

        if self.collected_target_points == 3:
            if self.spawn_last_target == False:
                spawn(4)
                self.spawn_last_target = True             

        self.time_taken += 0.01 
        collecting_reward = 2/(1+math.exp(self.collected_target_points))  

        self.reward = -max(self.linear_x/self.distance,0) - self.time_taken + 100*collecting_reward 

        if self.collected_target_points == 4:
            self.done = True

        if np.isnan(self.reward):
            print("***********nan values in reward*************")    
        
        # print(f"Reward: {self.reward}")
        
        self.observation = self.depth_image
        info = {}

        return self.observation, self.reward, self.done, info
	
    def reset(self):
        print("Reset function called")
        self.done = False
        self.flag_1 = False
        self.flag_2 = False
        self.flag_3 = False

        self.linear_x = 0.0
        self.dist_target_1 = 10000
        self.dist_target_2 = 10000
        self.dist_target_3 = 10000

        self.collected_target_points = 0
        self.pose_x, self.pose_y, self.pose_z = 0.0, 0.0, 0.0
        self.current_position = [self.pose_x, self.pose_y, self.pose_z]

        self.spawn_last_target = False

        self.width = 640
        self.height = 480

        self.time_taken = 0
        self.reward = 0

        self.depth_image = np.zeros((self.height, self.width)) 
        self.observation = self.depth_image

        spawn(1)
        spawn(2)
        spawn(3)


        return self.observation 
    

if __name__ == '__main__':   
    DroneEnv()       
	
