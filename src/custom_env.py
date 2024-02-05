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

from cv_bridge import CvBridge
from gym import spaces
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Image, CompressedImage
from air_drone.msg import Position, MotorSpeed
from gazebo_msgs.srv import DeleteModel, DeleteModelRequest, SpawnModel
from stable_baselines3.common.noise import NormalActionNoise

VERBOSE = False

class DroneNavigationEnv(gym.Env):
    def __init__(self):

        """This part is for spawning 3 target points"""
        self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        # random positions for 3 goal points
        # Create a 4x3 matrix with random values between -4 and 4
        self.positions_matrix = np.random.uniform(-4, 4, (4, 3))
        # Set the last item of each row to 1
        for i in range(self.positions_matrix.shape[0] - 1):
            self.positions_matrix[i, -1] = 0.5
        # Ensure the last row remains 0,0,1
        self.positions_matrix[3, :] = np.array([0, 0, 1])
        self.positions_matrix = np.array(self.positions_matrix)
        
        self.spawn(1)
        self.spawn(2)
        self.spawn(3)  


        
        # self.laser_sub = rospy.Subscriber('/laser/scan', LaserScan, self.laser_scan_callback)

        """Current drone position"""
        self.current_pose_sub = rospy.Subscriber("pose", Position, self.position_callback)  
        
        """Velocity publishers"""
        self.z_speed = rospy.Publisher("motor_speed_cmd", MotorSpeed, queue_size=1) # motor speed for z direction
        self.speed_motors_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)  # motor speed 

        """To reset and delete collected targets"""
        self.reset_simulation_service = rospy.ServiceProxy('/gazebo/reset_simulation', Empty) # resets gazebo  
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        """For camera"""
        self.image_sub = rospy.Subscriber("depth_camera/depth/image_raw", Image, self.camera_callback) # depth camera
        self.image_pub = rospy.Publisher("output/image_raw/compressed",
                                         CompressedImage, queue_size=1)
        self.subscriber = rospy.Subscriber("/camera1/image_raw/compressed",
                                           CompressedImage, self.callback,  queue_size=1)


        self.done = False
        self.flag_1 = False
        self.flag_2 = False
        self.flag_3 = False

        self.collected_target_points = 0
        self.pose_x, self.pose_y, self.pose_z = 0.0, 0.0, 0.0
        self.current_position = [self.pose_x, self.pose_y, self.pose_z]

        self.depth_image = None
        self.ball_detected = False
        self.closest_target_point = None
        self.spawn_last_target = False

        self.width = 640
        self.height = 480

        self.time_taken = 0
        self.reward = 0
        self.bridge = CvBridge()

        self.action_space = spaces.Box(low=np.array([0.0, -1.0, -1.0]),
                                       high=np.array([1.0, 1.0, 1.0]),dtype=np.float32)               
        
        # self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width), dtype=np.uint8)
        self.action_noise = NormalActionNoise(mean=np.zeros(self.action_space.shape), sigma=0.3 * np.ones(self.action_space.shape))

    def position_callback(self, pos_msg):
        self.pose_x, self.pose_y, self.pose_z = pos_msg.x, pos_msg.y, pos_msg.z
        self.pitch, self.roll = pos_msg.pitch, pos_msg.roll

        self.current_position = [self.pose_x, self.pose_y, self.pose_z]
    
    def camera_callback(self, image_msg):
        # Handling for depth camera data 
        try:
            depth_data = np.frombuffer(image_msg.data, dtype=np.float32)
            
            depth_image = depth_data.reshape((image_msg.height, image_msg.width))
            # Handle NaN values by replacing them with zeros
            depth_image = np.nan_to_num(depth_image)
  
            epsilon = 1e-5
            denominator = (depth_image.max() - depth_image.min())
            if denominator == 0:
                denominator = epsilon  # Avoid division by zero
            normalized_depth = ((depth_image - depth_image.min()) / denominator * 255).astype(np.uint8)

            self.current_depth_map = normalized_depth
            self.camera_data = normalized_depth

            center_x, center_y = normalized_depth.shape[1] // 2, normalized_depth.shape[0] // 2
            depth_value = depth_image[center_y, center_x]
            self.distance = depth_value
                  
            if self.current_depth_map is not None:
                self.current_depth_map = normalized_depth
                self.camera_data = normalized_depth
                
            else:
                self.observation = np.zeros((self.height, self.width), dtype=np.float32)  # Default value
                
        except ValueError as ve:
            rospy.logerr(f"ValueError processing camera data: {ve}")

    
    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE:
            print ('received image of type: "%s"' % ros_data.format)

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # OpenCV >= 3.0:

        greenLower = (50, 50, 20)
        greenUpper = (70, 255, 255)

        blurred = cv2.GaussianBlur(image_np, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        #cv2.imshow('mask', mask)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(image_np, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(image_np, center, 5, (0, 0, 255), -1)
                vel = Twist()
                vel.angular.z = -0.002*(center[0]-400)
                self.speed_motors_pub.publish(vel)

        else:
            vel = Twist()
            vel.angular.z = 0.5
            self.speed_motors_pub.publish(vel)

        # update the points queue
        # pts.appendleft(center)
        cv2.imshow('window', image_np)
        cv2.waitKey(2)    

    def target_callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE:
            print ('received image of type: "%s"' % ros_data.format)

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # OpenCV >= 3.0:

        greenLower = (50, 50, 20)
        greenUpper = (70, 255, 255)

        blurred = cv2.GaussianBlur(image_np, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        #cv2.imshow('mask', mask)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        self.ball_detected = False

        if len(cnts) > 0:
            # find the closest ball
            min_distance = float('inf')
            closest_index = None
            for i, c in enumerate(cnts):
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                
                distance = np.sqrt((center[0] - 400) ** 2 + (center[1] - 300) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_index = i

            # select the closest ball
            c = cnts[closest_index]
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            self.ball_detected = True
            self.closest_ball_index = closest_index
            self.closest_ball_center = center
            self.closest_ball_radius = radius

        if self.ball_detected:
            cv2.circle(image_np, (int(x), int(y)), int(radius),
                        (0, 255, 255), 2)
            cv2.circle(image_np, self.closest_ball_center, 5, (0, 0, 255), -1)
            vel = Twist()
            vel.angular.z = -0.002*(self.closest_ball_center[0]-400)
            self.speed_motors_pub.publish(vel)

        else:
            vel = Twist()
            vel.angular.z = 0.5
            self.speed_motors_pub.publish(vel)
            

        # update the points queue
        # pts.appendleft(center)
        cv2.imshow('window', image_np)
        cv2.waitKey(2)

    def object_name_finder(self):
            
        object_name = None
        target_1 = np.take(self.positions_matrix, 0, axis=0)
        target_2 = np.take(self.positions_matrix, 1, axis=0)
        target_3 = np.take(self.positions_matrix, 2, axis=0)

        if np.linalg.norm(self.current_position - target_1) < 0.5:
            if self.flag_1 == False:
                object_name = 'green_ball_1'
                self.delete_object_service(object_name)
                rospy.loginfo(f"object name in finder func:{object_name}")
                self.collected_target_points += 1
                self.flag_1 = True
                
        if np.linalg.norm(self.current_position - target_2) < 0.5: 
            if self.flag_2 == False:
                object_name = 'green_ball_2' 
                self.delete_object_service(object_name)
                rospy.loginfo(f"object name in finder func:{object_name}")
                self.collected_target_points += 1
                self.flag_2 = True
                

        if np.linalg.norm(self.current_position - target_3) < 0.5: 
            if self.flag_3 == False:
                object_name = 'green_ball_3'
                self.delete_object_service(object_name)
                rospy.loginfo(f"object name in finder func:{object_name}")
                self.collected_target_points += 1
                self.flag_3 = True

        return object_name           

    def spawn(self, number):
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        
        model_sdf = open("/home/ecem/.gazebo/models/green_ball_1/model.sdf", "r").read()
        model_namespace = ""
        # Set the initial pose
        initial_pose = Pose()
        if number == 1:
            print("first ball spawned")
            model_name = "green_ball_1"
            initial_pose.position.x = self.positions_matrix[0,0]
            initial_pose.position.y = self.positions_matrix[0,1]
            initial_pose.position.z = self.positions_matrix[0,2]
            number = 0

        elif number == 2:
            print("2nd ball spawned")
            model_name = "green_ball_2"
            initial_pose.position.x = self.positions_matrix[1,0]
            initial_pose.position.y = self.positions_matrix[1,1]
            initial_pose.position.z = self.positions_matrix[1,2]
            number = 0

        elif number == 3:
            print("3rd ball spawned")
            model_name = "green_ball_3"
            initial_pose.position.x = self.positions_matrix[2,0]
            initial_pose.position.y = self.positions_matrix[2,1]
            initial_pose.position.z = self.positions_matrix[2,2]
            number = 0

        elif number == 4:
            print("4th ball spawned")
            model_name = "green_ball_4"
            initial_pose.position.x = self.positions_matrix[3,0]
            initial_pose.position.y = self.positions_matrix[3,1]
            initial_pose.position.z = self.positions_matrix[3,2] 
            number = 0   

        success, status_message = self.spawn_target_points(model_name, model_sdf, model_namespace, initial_pose)

        if success:
            rospy.loginfo(f"Successfully spawned {model_name} in Gazebo!")
        else:
            rospy.logerr(f"Failed to spawn {model_name} in Gazebo. Error: {status_message}")
                
    def spawn_target_points(self, model_name, model_sdf, model_namespace, initial_pose):
        try:
            resp = self.spawn_model(model_name, model_sdf, model_namespace, initial_pose, "world")
            return resp.success, resp.status_message
        except rospy.ServiceException as e:
            return False, f"Service call failed: {str(e)}"

    def delete_object_service(self, object_name):
        rospy.wait_for_service('/gazebo/delete_model')    
        try:
            req = DeleteModelRequest()
            req.model_name = str(object_name)  # Specify the name of the object to delete
            resp = self.delete_model(req)
            rospy.loginfo(f"Deleted object: {object_name}")
            
            
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to delete object: {object_name}, error: {e}")

    def step(self, action):
        self.take_action(action) 
        self.check_collision()
     
        if self.collected_target_points == 1:
            self.reward = 5
        elif self.collected_target_points == 2:
            self.reward = 10
        elif self.collected_target_points == 3:
            self.reward = 20
            if self.spawn_last_target == False:
                self.spawn(4)
                self.spawn_last_target = True        

        self.time_taken += 1
        # Penalize excessive time taken
        penalty = 0.1 * self.time_taken
        self.reward -= penalty

        # rospy.loginfo(f"reward: {self.reward}")

        # Update observations
        observations = self.get_observations()

        # Set done flag
        done = self.collected_target_points == 4

        # Return observations, reward, done, and additional info
        return observations, self.reward, done, {}

    def get_observations(self):

        depth_image = self.depth_image
        observations = depth_image

        return observations
           
    def take_action(self, action):
        self.object_name_finder()
        self.vel_cmd = Twist()
        self.z_cmd = MotorSpeed()

        linear_x = action[0]*5
        angular_vel = action[1]
        z_vel = action[2]*150

        self.vel_cmd.linear.x = linear_x
        self.vel_cmd.angular.z = angular_vel
        self.speed_motors_pub.publish(self.vel_cmd)

        self.z_cmd.name = ['propeller1', 'propeller2', 'propeller3', 'propeller4', 'propeller5', 'propeller6']
        self.z_cmd.velocity = [z_vel, -z_vel , z_vel, -z_vel, z_vel, -z_vel]
        self.z_speed.publish(self.z_cmd)
    
    def check_collision(self):

        collision_count = 0
        if collision_count == 0 and self.distance > 1.0 and self.distance < 2.0 or abs(self.roll) > math.radians(15) or abs(self.pitch) > math.radians(15):
            print("Drone crashed") 
            self.delete_object_service('green_ball_1')
            self.delete_object_service('green_ball_2')
            self.delete_object_service('green_ball_3')
            self.reset_simulation_service()

            self.spawn(1)
            self.spawn(2)
            self.spawn(3)
            self.reset()
            time.sleep(3)
                    
            collision_count += 1
        
    def done(self):
        return self.collected_target_points == 4

    def reset(self):
        
        self.depth_image = np.zeros((self.height, self.width)) 
        self.observation = self.depth_image

        self.done = False
        self.flag_1 = False
        self.flag_2 = False
        self.flag_3 = False

        self.collected_target_points = 0

        self.time_taken = 0
        self.reward = 0
        
            
        return self.observation  


    
if __name__ == '__main__':
    rospy.init_node('custom_env')

    env = DroneNavigationEnv()


