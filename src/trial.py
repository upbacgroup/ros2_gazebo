#!/usr/bin/env python3
import os
import rospy
import numpy as np 
import time
import random
import math
import gym

from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel



from tf.transformations import *
from air_drone.msg import Pose, MotorSpeed
from gym import spaces

from stable_baselines3 import DQN, PPO, DDPG
from std_srvs.srv import Empty
from rospy.exceptions import ROSTimeMovedBackwardsException
from stable_baselines3.common.noise import NormalActionNoise

goal_model_dir = '/home/ecem/drone_ws/src/air_drone/world/drone_env.world'

class DroneNavigationEnv(gym.Env):
    def __init__(self):
        self.vel_cmd = Twist()
        self.z_cmd = MotorSpeed()

        self.speed_motors_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10) 
        self.current_pose_sub = rospy.Subscriber("pose", Pose, self.getCurrentPosition)  
       
        self.image_sub = rospy.Subscriber("depth_camera/depth/image_raw", Image, self.getImage) # depth camera
        self.z_speed = rospy.Publisher("motor_speed_cmd", MotorSpeed, queue_size=1)           # motor speed for z direction


        self.reset_simulation_service = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)       
        self.reset_simulation_service = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)        
        self.reset_simulation_service = rospy.ServiceProxy('/gazebo/pause_physics', Empty)          
        self.reset_simulation_service = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)        
        self.reset_simulation_service = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)   

        self.past_distance = 0.0
        self.arrival_threshold = 0.5
        

        self.goals = [(3.,3.), (-3.,2.), (3.,-3.), (-3., -1.2)]
        self.current_pos = [self.pose_x, self.pose_y, self.pose_z]
        self.current_distance = 0.0

    def generate_goal(self):
        
        if self.target_number == 1:
            self.goal_x = 1.0
            self.goal_y = 0.0
            self.goal_z = 1.0
            print(f"Goal Position: {self.goal_x}, {self.goal_y}, {self.goal_z}")
            #self.goal_reached = False

        elif self.target_number == 2:
            self.goal_x = 0.0
            self.goal_y = 2.0
            self.goal_z = 1.0
            print(f"Goal Position: {self.goal_x}, {self.goal_y}, {self.goal_z}")
            #self.goal_reached = False
            
            
        elif self.target_number == 3:
            self.goal_x = -1.0
            self.goal_y = 0.0
            self.goal_z = 1.0
            print(f"Goal Position: {self.goal_x}, {self.goal_y}, {self.goal_z}")
            #self.goal_reached = False

        elif self.target_number == 4:
            self.goal_x = 0.0
            self.goal_y = 0.0
            self.goal_z = 0.0
            print(f"Goal Position: {self.goal_x}, {self.goal_y}, {self.goal_z}")
            



    def getCurrentPosition(self,pos_msg):
        self.pose_x, self.pose_y, self.pose_z = pos_msg.x, pos_msg.y, pos_msg.z
        self.pitch, self.roll = pos_msg.pitch, pos_msg.roll    
    
    def getGoalDistance(self):
        self.current_distance =  np.linalg.norm(np.array(self.goals) - np.array(self.current_pos))
        if self.goal_distance < self.arrival_threshold:
            arrive = True

        return arrive    

    # def getImage(self, msg):
        

    def detectObstacles(self, depth_image):
        threshold_value = 0.5
        obstacle_mask = (depth_image < threshold_value).astype(np.uint8)
        obstacles_detected = np.any(obstacle_mask)

        return obstacles_detected #like done
    
    def setReward(self, arrive, obstacles_detected):
        distance_rate = self.past_distance - self.current_distance
        reward = 500*distance_rate
        self.past_distance = self.current_distance

        if arrive:
            reward = 120
            self.z_speed.publish(self.z_cmd)     
            self.speed_motors_pub.publish(self.vel_cmd)

        if obstacles_detected:
            reward = -100
            self.z_speed.publish(self.z_cmd)     
            self.speed_motors_pub.publish(self.vel_cmd)

        return reward


    def step(self, action, past_action):
        # Actions: x, y, z velocities
        x_velocity, y_velocity, z_velocity = self.action

        self.vel_cmd.linear.x = x_velocity
        self.vel_cmd.linear.y = y_velocity
        self.z_cmd.name = ['propeller1', 'propeller2', 'propeller3', 'propeller4', 'propeller5', 'propeller6']
        self.z_cmd.velocity = [z_velocity, -z_velocity, z_velocity, -z_velocity, z_velocity, -z_velocity]
        
        # Publish velocity commands
        self.z_speed.publish(self.z_cmd)
        self.speed_motors_pub.publish(self.vel_cmd)

        image = None
        while image is None:
            try:
                image = rospy.wait_for_message("depth_camera/depth/image_raw", Image, timeout=5)
                bridge = CvBridge()
                try:
                    # Convert ROS Image to OpenCV image
                    image = bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
                except Exception as e:
                    raise e
            except:
                pass

    def reset(self):
        rospy.wait_for_service('/gazebo/delete_model')
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")


    def main(self):
        total_timesteps = 0
        self.rate = rospy.Rate(10)
        num_episodes = 10
        total_timesteps_ = 1000000

        initial_ent_coef = 0.1
        # model = PPO("MlpPolicy", env, ent_coef=initial_ent_coef, learning_rate=0.0003)
        # model = DDPG("MlpPolicy", env,  learning_rate=0.003)
        model = DDPG(
            "MlpPolicy",  
            env,
            buffer_size=10000,  # Replay buffer size
            learning_rate=1e-3,  # Learning rate for the actor and critic networks
            batch_size=64,  # Batch size for training
            gamma=0.95,  # Discount factor for future rewards
            tau=0.005,  # Soft update coefficient for target networks
            policy_kwargs=dict(net_arch=[400, 300]),  # Architecture of the policy network
            action_noise=self.action_noise
        )
        
        self.generate_goal()
        self.observation = env.reset()
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")
            self.rate.sleep()
            
            
            # Collect experiences during the episode
            episode_experiences = []

            for timestep in range(total_timesteps_): 

                self.action, _ = model.predict(self.observation, deterministic=True)
                self.observation, self.reward, self.done, self.info = env.step(self.action)
                time.sleep(1)

                experience = {
                    "observation": self.observation,
                    "action": self.action,
                    "reward": self.reward,
                    "done": self.done,
                    "info": self.info,
                }
                episode_experiences.append(experience)


                if self.done:
                    # self.goal_reached_1 = False
                    # self.goal_reached_2 = False
                    # self.goal_reached_3 = False
                    #self.goal_reached_4 = True

                    self.target_reward = 0.0
                    self.time_step = 0  

            # Perform a batch update at the end of each episode
            if len(episode_experiences) > 0:
                total_timesteps += len(episode_experiences)  # Increment the total timesteps
                # model.learn(total_timesteps=len(episode_experiences), reset_num_timesteps=False, batch_size=64)  # Adjust the batch_size value
                model.learn(total_timesteps=len(episode_experiences), reset_num_timesteps=False, batch_size=64)


            if self.goal_reached_4 == True:
                print("All targets reached. Stopping training.")
                break

            self.rate.sleep()
    



if __name__ == '__main__':
    rospy.init_node('ddpg')
    env = DroneNavigationEnv()
    env.main()


       

       










        
        

















