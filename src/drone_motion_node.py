#!/usr/bin/env python3
from os import times
from tabnanny import verbose
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Twist

import rospy
import numpy as np
from tf.transformations import *
from air_drone.msg import Pose, MotorSpeed
import gym
from gym import spaces
import rospy
from sensor_msgs.msg import Image
import cv2
from stable_baselines3 import DQN, PPO
from std_srvs.srv import Empty
from rospy.exceptions import ROSTimeMovedBackwardsException
import random
import math

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
# Add this line to your imports
from collections import deque
### Publishers: 
# MotorSpeed: "motor_speed_cmd"
# Marker: "/goal_marker"

### Subscribers:
# Image: "depth_camera/depth/image_raw"
# Pose: "pose" 

### Service:
# Empty: "/gazebo/reset_simulation"

class DroneNavigationEnv(gym.Env):
    def __init__(self):

        # initializations
        self.prev_position = [0.0, 0.0, 0.0]  # Initialize with the initial position
        self.position_check_time = rospy.Time.now()  # Initialize time for position checking
        self.position_check_duration = rospy.Duration(3.0)  # Specify the duration for position checking (3 seconds in this example)

        self.reward_history = deque(maxlen=1000)  # Store the last 1000 rewards for plotting
        self.reached_target_history = []
        self.reward_plot = None
        self.bar_plot = None
        self.success_count = 0
        self.fail = 0
        self.time_step = 0
        self.collision_count = 0
        self.pose_x, self.pose_y, self.pose_z = 0.0, 0.0, 0.0
        self.goal_x, self.goal_y, self.goal_z = 0.0, 0.0, 0.0 
        self.reset_con = False 
        self.drone_reset = False        
        self.goal_reached_1 = False
        self.goal_reached_2 = False
        self.goal_reached_3 = False
        self.current_depth_map = None
        self.camera_data = None
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=255, shape=(60, 80, 1), dtype=np.uint8)
        self.goal_relative_x = 0.0
        self.goal_relative_y = 0.0
        self.collision = False
        self.reward = 0.0
        self.action = 0
        self.linear_speed = 10.0
        self.target_number = 1
        # Publishers and subscribers
        self.image_sub = rospy.Subscriber("depth_camera/depth/image_raw", Image, self.camera_callback) # depth camera
        #self.current_pose_sub = rospy.Subscriber('odom', Odometry, self.position_callback)             # current drone position 
        self.current_pose_sub = rospy.Subscriber("pose", Pose, self.position_callback)  
        self.z_speed = rospy.Publisher("motor_speed_cmd", MotorSpeed, queue_size=1)           # motor speed for z direction
        self.speed_motors_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)                      # motor speed 
        self.reset_simulation_service = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)          # resets gazebo  
        self.marker_pub = rospy.Publisher("/goal_marker", Marker, queue_size=1)                        # marks goal point 
        self.target_reward = 0.0      

    # def publish_goal_marker(self):
    #     # Create a marker message
    #     marker = Marker()
    #     marker.header.frame_id = "world"  # Set the frame id
    #     marker.id = 0  # Marker ID
    #     marker.type = Marker.SPHERE  # Marker type (SPHERE)
    #     marker.action = Marker.ADD  # Action (ADD)
    #     marker.pose.position = Point(self.goal_x, self.goal_y, 0.1)  # Set the position of the marker
    #     marker.pose.orientation.x = 0.0
    #     marker.pose.orientation.y = 0.0
    #     marker.pose.orientation.z = 0.0
    #     marker.pose.orientation.w = 1.0
    #     marker.scale.x = 0.1  # Set marker scale
    #     marker.scale.y = 0.1
    #     marker.scale.z = 0.1
    #     marker.color.a = 1.0  # Set alpha (transparency)
    #     marker.color.r = 0.0  
    #     marker.color.g = 0.0
    #     marker.color.b = 1.0  # Set color (blue)  
    #     self.marker_pub.publish(marker)
      

    def generate_goal(self):
        
        if self.target_number == 1:
            self.goal_x = 1.0
            self.goal_y = -1.0
            self.goal_z = 1.0
            #print(f"Goal Position: {self.goal_x}, {self.goal_y}, {self.goal_z}")
            self.goal_reached_1 = True

        elif self.target_number == 2:
            self.goal_x = 0.0
            self.goal_y = -2.5
            self.goal_z = 1.0
            #print(f"Goal Position: {self.goal_x}, {self.goal_y}, {self.goal_z}")
            self.goal_reached_2 = True
            
        
            
        elif self.target_number == 3:
            self.goal_x = -1.5
            self.goal_y = -4.0
            self.goal_z = 1.0
            #print(f"Goal Position: {self.goal_x}, {self.goal_y}, {self.goal_z}")
            self.goal_reached_3 = True


    def calculate_reward(self):
        # Calculate reward based on camera data
        if self.camera_data is None:
            return 0.0
       
        self.generate_goal()

        flag_1 = True
        flag_2 = True

        target_distance_threshold_1 = 1.0  # Threshold for considering target reached
        target_distance_threshold_2 = 1.5  # Threshold for considering target reached
        target_distance_threshold_3 = 2.0  # Threshold for considering target reached
            # Reward for reaching the target      
        collision_penalty = 0.0
        #current_position = [self.pose_x, self.pose_y, self.pose_z]
        current_position = [round(self.pose_x), round(self.pose_y), self.pose_z]

        goal_position = [self.goal_x, self.goal_y, self.goal_z]
        self.distance_to_goal = np.linalg.norm(np.array(current_position) - np.array(goal_position))
        distance_reward = - self.distance_to_goal
        #velocity_reward = -np.linalg.norm(np.array(self.vel_cmd.velocity))
        total_reward = 0.0
        collided = self.check_collision()
        #print(f"Distance: {self.distance_to_goal}")
        #Calculate the reward
        if collided:
           collision_penalty = -5000.0
           self.fail += int(1/self.collision_count)
           
           
           self.reached_target_history.append(0)  # Add 0 to indicate failure
           self.reset_con = True
           
        
        

        elif self.distance_to_goal < target_distance_threshold_1:
            
            self.success_count += 1
            self.reached_target_history.append(1)  # Add 1 to indicate success
            
            if self.goal_reached_1:
                print(f"Goal Position: {self.goal_x}, {self.goal_y}, {self.goal_z}")
                self.target_reward = 3000.0
                print("***********************1st Hit *****************************")
                self.target_number = 2
                
                if self.goal_reached_2:
                    self.target_reward += 4000.0
                    print("***********************1st Combo *****************************")
                    self.target_number = 3
                    
                    
                    if self.goal_reached_3:
                        self.target_reward += 15000.0
                        print("***********************2nd Combo *****************************")
                        self.drone_reset = True
                        self.reset_con = True
                        self.target_number = 1
                        

            flag_1 = False


        elif flag_1 and self.distance_to_goal < target_distance_threshold_2:
           #print("***********************CLOSE*****************************")
           self.target_reward = 2000.0
           #self.success_count += 2
           self.reached_target_history.append(1)  # Add 1 to indicate success
           flag_1 = True
           flag_2 = False
          
        elif flag_1 and flag_2 and self.distance_to_goal < target_distance_threshold_3:
           #print("***********************FINE*****************************")
           self.target_reward = 1000.0
           #self.success_count += 1
           self.reached_target_history.append(1)  # Add 1 to indicate success
           flag_1 = True
           

        else:
            self.reset_con = False

        
            
            
   
        # Sum up different reward terms
        total_reward += distance_reward + collision_penalty + self.target_reward - 10*self.time_step
        #print(f"Total {total_reward}")
        return total_reward        

    def position_callback(self,pos_msg):
        self.pose_x, self.pose_y, self.pose_z = pos_msg.x, pos_msg.y, pos_msg.z
        self.pitch, self.roll = pos_msg.pitch, pos_msg.roll
  

     # Add a method to calculate relative goal position
    def calculate_relative_goal_position(self):
        self.goal_relative_x = self.goal_x - self.pose_x
        self.goal_relative_y = self.goal_y - self.pose_y

    
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

            # Resize the image to the target dimensions
             # Resize the image to the target dimensions (smaller dimensions)
            target_width, target_height = 80, 60  # Adjust these dimensions as needed
            resized_depth = cv2.resize(normalized_depth, (target_width, target_height))


            self.current_depth_map = resized_depth
            self.camera_data = resized_depth
            #self.camera_data = cv2.cvtColor(normalized_depth, cv2.COLOR_GRAY2RGB)
            self.current_depth_map = normalized_depth
            self.camera_data = normalized_depth
            #rospy.loginfo(f"Depth image min value: {self.current_depth_map}")
       
            if self.current_depth_map is not None:
                
                self.current_depth_map = resized_depth
                self.camera_data = resized_depth
                
            else:
                # Handle the case when depth map data is not available
                # You can set self.observation to a default value or take other appropriate actions
                self.observation = np.zeros((60, 80, 1), dtype=np.uint8)  # Default value
                

        except ValueError as ve:
            rospy.logerr(f"ValueError processing camera data: {ve}")
        except IndexError as ie:
            rospy.logerr(f"IndexError processing camera data: {ie}")
        except Exception as e:
            rospy.logerr(f"Error processing camera data: {e}")
    

    def step(self, action):
        self.time_step += 1
        #Actions: 
        self.vel_cmd = Twist()
        self.z_cmd = MotorSpeed()
        a = 100
        #self.cmd_vel_msg.angular.z = self.angular_speed

        if action == 0: # x
            self.vel_cmd.linear.x = self.linear_speed
        elif action == 1: # -x
            self.vel_cmd.linear.x = -self.linear_speed
           
        elif action == 1: # y
            self.vel_cmd.linear.y = self.linear_speed
            
        elif action == 3: # -y
            self.vel_cmd.linear.y = -self.linear_speed
            
        elif action == 4: # z
            self.z_cmd.name = ['propeller1', 'propeller2', 'propeller3', 'propeller4', 'propeller5', 'propeller6']
            self.z_cmd.velocity = [a,-a,a,-a,a,-a]
        elif action == 5: # -z
            self.z_cmd.name = ['propeller1', 'propeller2', 'propeller3', 'propeller4', 'propeller5', 'propeller6']
            self.z_cmd.velocity = [-a,a,-a,a,-a,a]  
        # elif action == 4: # rotation
        #     self.vel_cmd.angular.z = self.linear_speed    
        elif action == 6: # stop
            self.vel_cmd.linear.x = 0.0
            self.vel_cmd.linear.y = 0.0

        self.z_speed.publish(self.z_cmd)     
        self.speed_motors_pub.publish(self.vel_cmd)
        
        # Calculate reward based on camera data
        depth_map_reshaped = self.current_depth_map[:, :, np.newaxis]

        # Include depth map in observations
        self.observation = depth_map_reshaped

        # Check if the episode is done
        if self.done:
            self.reward = 0.0  # Reset the reward to zero
        else:
            self.reward = self.calculate_reward()
        self.done = False 
        self.info = {}

        #if self.loaded_model is not None:
        #    action, _ = self.loaded_model.predict(self.observation)
            # Use the action and the loaded model to determine the drone's behavior


        return self.observation, self.reward, self.done, self.info



    def reset(self):
        # Reset the environment
        #if self.loaded_model is None:
        #    self.loaded_model = DQN.load("point_mass_navigation_dqn")
        self.done = False
        self.reward = 0.0  # Reset the reward to zero
        self.calculate_relative_goal_position()
        self.camera_data = np.zeros((60, 80, 1)) 
            
        return self.camera_data


    def check_collision(self):
        if self.current_depth_map is None:
            print("Depth map is not available")
            return True

        if abs(self.roll) > math.radians(60):
            if self.collision_count == 0:
                print("Drone is vertical")
                #self.reset_simulation_service()  # Call the reset simulation service
            self.collision_count += 1
            return True

        if self.pose_z > 3.0:
            if self.collision_count == 0:
                print("Drone height is too much")
            self.collision_count += 1
            return True

        if abs(self.pose_x) > 5.0 or abs(self.pose_y) > 5.0:
            if self.collision_count == 0:
                print("Drone is out of range")
                #self.reset_simulation_service()  # Call the reset simulation service
            self.collision_count += 1
            return True
        
        return False  # No collision detected

    


    
    
    def plot_reward_history(self):
        
        
        #plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    
        if self.bar_plot is None:
            #plt.figure(figsize=(8, 4))
            
            #self.success_count = np.bincount(self.reached_target_history)
            x_labels = ["Not Reached", "Reached"]
            x_positions = np.arange(len(x_labels))
            bar_width = 0.5  # Adjust the bar width as needed
            self.ax1 = plt.subplot(211)
            a_x1 = self.ax1
            self.counts = [self.success_count, self.fail]
            self.bar_plot = a_x1.bar(x_labels, self.counts, width=bar_width, color=['red', 'green'])
           
            self.ax1.set_title("Success History")
            self.ax1.set_xlabel("Outcome")
            self.ax1.set_ylabel("Count")
            self.ax1.set_xticks(x_positions, x_labels)  # Set custom x-axis labels
            
        else:
           max_reward_bar = max(self.counts)
           y_margin = 100  # You can adjust this margin as needed
           self.ax1.set_ylim([0, max_reward_bar + y_margin])
           self.success_count = np.bincount(self.reached_target_history)
           for bar, height in zip(self.bar_plot, self.success_count):
                bar.set_height(height)


        if self.reward_plot is None:
            
            self.ax2 = plt.subplot(212)
            self.reward_plot, = self.ax2.plot(range(len(self.reward_history)), self.reward_history)
            
            self.ax2.set_title("Reward History")
            self.ax2.set_xlabel("Timestep")
            self.ax2.set_ylabel("Reward")
            self.count = 1
            plt.ion()  # Turn on interactive mode for live updating
            #plt.show()
        else:
           min_reward = min(self.reward_history)
           max_reward = max(self.reward_history)
           y_margin = 10  # You can adjust this margin as needed
           self.ax2.set_ylim([min_reward - y_margin, max_reward + y_margin])
           self.ax2.set_xlim([0, 1000])
           self.update_plot()

        
        plt.tight_layout()  # Adjust spacing between subplots
        plt.show()
           
    def update_plot(self):
        self.reward_plot.set_ydata(self.reward_history)
        self.reward_plot.set_xdata(range(len(self.reward_history)))  # Update x data

        # Adjust y-axis limits to make rewards more visible
        
        self.count += 1
        plt.draw()
        plt.pause(0.001)

        
   
        


        

    def main_loop(self):
        
        self.rate = rospy.Rate(10)
        #timesteps=100000
        timesteps=100
        #self.model = DQN('CnnPolicy', env, learning_rate=0.3, buffer_size=1000000, exploration_fraction=0.1, verbose=3, exploration_final_eps=0.000001)
        self.model = PPO('MlpPolicy', env)
        #self.model = PPO.load("/home/ecem/Desktop/dqn/drone_dqn2")

        #self.model.learn(total_timesteps=10) 
        #self.model.save("/home/ecem/Desktop/dqn/drone_dqn2")
        self.observation = env.reset()

        while not rospy.is_shutdown():  
            #self.model = DQN('CnnPolicy', env, learning_rate=0.003, buffer_size=1000000, exploration_fraction=0.001, verbose=3, exploration_final_eps=0.000001)
           
            self.generate_goal()
            for self.timestep in range(timesteps):
                if self.reset_con and self.timestep>0:

                    try:
                        self.reset_simulation_service()  # Call the reset simulation service  
                    except ROSTimeMovedBackwardsException:
                        rospy.logwarn("ROSTimeMovedBackwardsException caught. Delaying reset...")
                        current_time = rospy.Time.now()
                        while rospy.Time.now() - current_time < rospy.Duration(1.0):
                            self.rate.sleep()
                
                    self.goal_reached_1 = False
                    self.goal_reached_2 = False
                    self.goal_reached_3 = False
                    
                    self.target_reward = 0.0
                    self.target_number = 1
                    self.collision_count = 0 
                    self.time_step = 0
                    #self.model = DQN.load("drone_navigation_dqn")
                    self.generate_goal()
                    #self.publish_goal_marker()                  
                    #self.is_upside_down = False  # Reset the flag
                    self.reset_con = False
                    self.drone_reset = False
                    self.last_ros_time = rospy.Time.now()
                    #self.model = DQN.load("drone_navigation_dqn")
                    
            
                else: 
                    if self.current_depth_map is not None:   
                        self.action, _ = self.model.predict(self.observation)
                        self.observation = self.current_depth_map[:, :, np.newaxis]  
                        self.observation, self.reward, self.done, self.info = env.step(self.action)
                        self.model.learn(total_timesteps=1, reset_num_timesteps=False) 
                        #rospy.loginfo(f"Timestep: {self.timestep}, Action: {self.action}, Reward: {self.reward}")
                        # Append the reward to the reward history
                        #self.reward_history.append(self.reward)
                        rospy.Rate(1)
                    else:
                        self.observation = np.zeros((60, 80, 1), dtype=np.uint8)  # Default value

                    if self.done:
                        #self.model.save("/home/ecem/Desktop/dqn/drone_dqn")

                        #self.observation = env.reset()
                        self.goal_reached_1 = False
                        self.goal_reached_2 = False
                        self.goal_reached_3 = False
                        
                        self.target_reward = 0.0
                        self.target_number = 1
                        self.collision_count = 0
                        self.time_step = 0
                        #self.model = DQN.load("drone_navigation_dqn")
                    
                        # Create or update the reward plot
                    #self.plot_reward_history() 
                
            self.rate.sleep()
            #self.model.learn(total_timesteps=10)
            #self.model.save("/home/ecem/Desktop/dqn/drone_dqn2")
            self.model.save("/home/ecem/Desktop/dqn/drone_ppo_cross")
              
       

if __name__ == '__main__':
    rospy.init_node('drone_motion_node')
    

    env = DroneNavigationEnv()
    env.main_loop()
