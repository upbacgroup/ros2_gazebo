
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
# from air_drone.msg import Position
from cv_bridge import CvBridge
import gym
from gym import Env
import math
import cv2
import numpy as np
import time

from air_drone.scripts.target_points import spawn
from air_drone.scripts.target_points import check_balls_in_environment
from air_drone.scripts.target_points import delete_object_service

VERBOSE = False

class DroneEnv(Node, Env):
    def __init__(self):
        super().__init__('drone_env')

        self.width = 80
        self.height = 60
        
        self.bridge = CvBridge()

        self.collected_targets = []
        self.depth_image = []
        self.detection_reward = 0
        
        self.last_position = None
        self.start_time = None
        self.check = False
        self.green_penalty = 0
        self.previous_green_pixels = 0

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(60,80,1), dtype=np.uint8)

        # self.current_pose_sub = self.create_subscription(Position, 'pose', self.position_callback)
        # self.image_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.camera_callback)
        self.speed_motors_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reset_simulation_service = self.create_client(Empty, '/gazebo/reset_simulation')
        # self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback)

    def image_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
    
        # Convert BGR to HSV
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define range of green color in HSV
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv_image, lower_green, upper_green)

        num_green_pixels = cv2.countNonZero(mask)
    
        self.green_pixels = num_green_pixels 

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

        # Filter contours based on size and shape
        min_contour_area = 100
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_contour_area:
                # Draw bounding box around detected green objects
                # print("**************GREEN BALL********************")
                x, y, w, h = cv2.boundingRect(contour)
                self.ball_coords = (x,y)
                current_position_xy = tuple(self.current_position[:2]) 
                self.dist = np.linalg.norm(np.array(current_position_xy) - np.array(self.ball_coords))
                # print(self.dist)
                cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), -1)
                
        # Display the result
        cv2.imshow('Green Ball Detection', cv_image)
        cv2.waitKey(1)

    def camera_callback(self, image_msg):
        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        
        if image.size > 0:
            image = np.nan_to_num(image)
            resized_depth_image = np.expand_dims(image, axis=-1)
            # Convert resized depth image to float32
            self.depth_image = np.array(resized_depth_image, dtype=np.float32)

            depth_array = np.array(self.depth_image, dtype=np.float32)

            non_zero_mask = (depth_array != 0)
            if np.any(non_zero_mask):
                self.min_non_zero_distance = np.min(depth_array[non_zero_mask])

    def position_callback(self, pos_msg):
        pose_x, pose_y, pose_z = pos_msg.x, pos_msg.y, pos_msg.z
        pitch, roll, yaw = pos_msg.pitch, pos_msg.roll, pos_msg.yaw

        self.current_position = [pose_x, pose_y, pose_z]
        self.current_pose = np.array([pose_x, pose_y, pose_z, pitch, roll, yaw])

        tolerance = 0.2

        if self.last_position is None:
            self.last_position = self.current_position
            self.start_time = time.time()
        else:
            if np.linalg.norm(np.array(self.last_position) - np.array(self.current_position)) < tolerance:
                a = time.time() - self.start_time
                
                if  a >= 5.0:
                    print("Drone position has remained stable for 5 seconds.")
                    self.done = True
                    
            else:
                self.last_position = self.current_position
                self.start_time = time.time()

    def object_name_finder(self):
        object_name = None
        current_position_xy = self.current_position[:2] 

        distances = {}

        for name, self.pos in self.target_positions.items():
            if name not in self.collected_targets: # Check if target is not collected
                distance = np.linalg.norm(current_position_xy - self.pos[:2])
                distances[name] = distance


        if distances:
            min_target = min(distances, key=distances.get)
            min_distance = distances[min_target]
            self.min_distance = min_distance
            print(self.min_distance)

            if min_distance < 1.0:
                object_name = f"{min_target}"
                status = delete_object_service(object_name)
                if object_name in self.target_positions:
                    del self.target_positions[object_name]

                if status:
                    self.collected_target_points += 1
                    print(f"Collected balls: {self.collected_target_points}")


        return object_name

    def check_collision(self):
        collision_count = 0
        # print(self.min_non_zero_distance)
        if collision_count == 0 and self.min_non_zero_distance < 0.7 or abs(self.current_position[0])>8 or abs(self.current_position[1])>8:  
            print("*******************collision*********************")
            self.check = True
            self.done = True
            # self.reset()
            collision_count += 1

        return self.done   

    def take_action(self, action):
        self.object_name_finder()

        linear_x = (action[0]+1)/2
        angular_z = action[1]
        
        vel_cmd = Twist()
        vel_cmd.linear.x = 5.0*linear_x
        vel_cmd.angular.z = 10.0*angular_z

        self.speed_motors_pub.publish(vel_cmd)

    def step(self, action):
        print("Step function called")
        self.done = False
        self.take_action(action)
        self.check_collision() 
        
        reward = -1 + 2/(1+np.exp(0.5*self.min_distance/(self.green_pixels+1)))
        print(f"# Green pxl: {self.green_pixels}")
        print(f"Reward: {reward}")
       

        if check_balls_in_environment():
            pass
        else:    
            print("******ALL TARGETS ARE COLLECTED******")
            self.done = True

           
        self.observation = self.depth_image
        # self.observation = {"depth_map": self.depth_image,
        #                     "current_pose": self.current_pose}
       
       
        # print(self.observation)
       

        info = {}

        return self.observation, reward, self.done, info

    def reset(self):
        print(f"Reset function called")
       
        self.last_position = None
        self.start_time = None
       
        self.green_pixels = 0
        self.collected_target_points = 0
        self.current_position = [0,0,0]
        self.detection_reward = 0
        self.min_distance = np.array([100])
        self.current_pose = np.array([0,0,0,0,0,0])

        self.spawn_last_target = False
        self.ball_coords = [0,0]

        image = np.zeros((self.height, self.width)) 

        resized_depth_image = np.expand_dims(image, axis=-1)   
        depth_image = np.array(resized_depth_image, dtype=np.float32)
        self.observation = self.depth_image
        # self.observation = {"depth_map": self.depth_image,
        #                     "current_pose": self.current_pose}
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0

        self.speed_motors_pub.publish(vel_cmd)
       

        delete_object_service('green_ball_1')
        # delete_object_service('green_ball_2')
        # delete_object_service('green_ball_3')
        # delete_object_service('green_ball_4')
        self.reset_simulation_service()
        print("simulation started")

        self.target_1 = spawn(1)
        # self.target_2 = spawn(2)
        # self.target_3 = spawn(3)

        self.target_positions = {
            'green_ball_1': self.target_1[:2],
            # 'green_ball_2': self.target_2[:2],
            # 'green_ball_3': self.target_3[:2],
        }
        
        return self.observation

def main(args=None):
    rclpy.init(args=args)
    drone_env = DroneEnv()
    rclpy.spin(drone_env)
    drone_env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
