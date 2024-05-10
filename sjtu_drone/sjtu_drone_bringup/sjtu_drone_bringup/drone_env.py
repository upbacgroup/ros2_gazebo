
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

from cv_bridge import CvBridge
import gym
from gym import Env
import math
import cv2
import numpy as np
import time
import sys

# from sjtu_drone_bringup.target_points import spawn
# from sjtu_drone_bringup.target_points import check_balls_in_environment
# from sjtu_drone_bringup.target_points import delete_object_service

VERBOSE = False

class DroneEnv(Node, Env):
    def __init__(self):
        super().__init__('drone_env')
        print('**********HELLO FROM MY ENV************')
        self.width = 640
        self.height = 360
        
        self.bridge = CvBridge()

        self.collected_targets = []
        self.depth_image = []
        self.detection_reward = 0
        self.current_position = None
        self.last_position = None
        self.start_time = None
        self.check = False
        self.green_penalty = 0
        self.previous_green_pixels = 0

        self.target_positions = {
            'green_ball_1': [3, 0, 1],
            'green_ball_2': [0, 2, 1],
            'green_ball_3': [0, -2, 1],
        }

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height,self.width,1), dtype=np.uint8)

        self.current_pose_sub = self.create_subscription(Odometry, '/simple_drone/odom', self.position_callback, 1024)
       
        self.image_sub = self.create_subscription(Image, '/simple_drone/front/image_raw', self.camera_callback, 1024)
        
        self.speed_motors_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 1024)
        # self.rst_cli = self.create_client(DeleteEntity, 'delete_entity')
        # self.cli = self.create_client(SpawnEntity, '/spawn_entity')
        self.client = self.create_client(Empty, '/reset_simulation')
        

       
    def reset_simulation(self):
        request = Empty.Request()
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('Simulation reset successful!')
        else:
            self.get_logger().error('Failed to reset simulation')

    def start_training(self):
        iters = 0
        while True:
            iters += 1
            self.model.learn(total_timesteps=self.TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
            # self.model.save(f"{self.models_dir}/{self.TIMESTEPS*iters}")



    def spawn_entity(self):
        content = sys.argv[1]
        namespace = sys.argv[2]

        req = SpawnEntity.Request()
        req.name = namespace
        req.xml = content
        req.robot_namespace = namespace
        req.reference_frame = "world"


    
    def position_callback(self, msg:Odometry):
        # print("*********POSE MESSAGE********")
        self._gt_pose = msg

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        self.current_position = [x, y, z]
        print(self.current_position)
        tolerance = 0.2
        target_position = [3, 0, 1]
        difference = np.linalg.norm(np.array(self.current_position) - np.array(target_position))
        # print("Difference from target position (3, 0, 1):", difference)
        # if difference <= 1.0:
        #     print("**********COLLECT**********")
        #     self.done = True
        #     self.reset()

        if self.last_position is None:
            self.last_position = self.current_position
            self.start_time = time.time()
    
        else:
            if np.linalg.norm(np.array(self.last_position) - np.array(self.current_position)) < tolerance:
                a = time.time() - self.start_time
                
                if  a >= 5.0:
                    # print("Drone position has remained stable for 5 seconds.")
                    self.done = True   
                    
            else:
                self.last_position = self.current_position
                self.start_time = time.time()    

    def camera_callback(self, image_msg):
        # print("*********CAMERA MESSAGE********")
        # Process image message
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        image_height, image_width, num_channels = cv_image.shape
        # print(f"height: {image_height}, width: {image_width}")
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
                x, y, w, h = cv2.boundingRect(contour)
                self.ball_coords = (x, y)
                if self.current_position is not None:
                    current_position_xy = self.current_position[:2]
                    self.dist = np.linalg.norm(np.array(current_position_xy) - np.array(self.ball_coords))
                    cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), -1)

        # Process depth image (assuming it's part of the image message)
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

        # Display the result
        cv2.imshow('Green Ball Detection', cv_image)
        cv2.waitKey(1)


    def object_name_finder(self):
        object_name = None
        current_position_xy = self.current_position[:2] 

        # distances = {}

        # for name, self.pos in self.target_positions.items():
            # if name not in self.collected_targets: # Check if target is not collected
                # distance = np.linalg.norm(current_position_xy - self.pos[:2])
        distance = np.linalg.norm(np.array(current_position_xy) - np.array([3,0]))
        # distances[name] = distance
        self.min_distance = distance
        print(f"Distance: {current_position_xy}")
        print(f"Distance: {distance}")
        if distance <= 1.0:
            self.done = True
            
                # print("*****************************************")
                # print(distance)


        # if distances:
        #     min_target = min(distances, key=distances.get)
        #     min_distance = distances[min_target]
        #     self.min_distance = min_distance
        #     print(self.min_distance)

        #     if min_distance < 1.0:
        #         object_name = f"{min_target}"
        #         # status = delete_object_service(object_name)
        #         if object_name in self.target_positions:
        #             del self.target_positions[object_name]

                # if status:
                #     self.collected_target_points += 1
                #     print(f"Collected balls: {self.collected_target_points}")


        return object_name

    def check_collision(self):
        collision_count = 0
        # print(self.min_non_zero_distance)
        print(self.current_position)
        # if collision_count == 0 and self.min_non_zero_distance < 0.7 or abs(self.current_position[0])>8 or abs(self.current_position[1])>8:  
        if collision_count == 0 and abs(self.current_position[0])>5 or abs(self.current_position[1])>5:      
            print("*******************collision*********************")
            # self.check = True
            self.done = True
            self.reset()
            collision_count += 1

        return self.done   

    def take_action(self, action):
    # def take_action(self):
        self.object_name_finder()

        linear_x = (action[0]+1)/2
        angular_z = action[1]

        # linear_x = 0.5
        # angular_z = 0.1
        
        vel_cmd = Twist()
        vel_cmd.linear.x = 1.0*linear_x
        vel_cmd.angular.z = 1.0*angular_z

        self.speed_motors_pub.publish(vel_cmd)
        # self.get_logger().info('Publishing cmd_vel message: linear_x = %f, angular_z = %f' % (linear_x, angular_z))


    def step(self, action):
        print("Step function called")
        self.done = False
        self.take_action(action)
        self.check_collision() 
        
        reward = -1 + 2/(1+np.exp(0.5*self.min_distance/(0+1)))
        print(f"# Green pxl: {self.green_pixels}")
        print(f"Reward: {reward}")
    
       

        # if check_balls_in_environment():
        #     pass
        # else:    
        #     print("******ALL TARGETS ARE COLLECTED******")
        #     self.done = True

           
        self.observation = self.depth_image
        # self.observation = {"depth_map": self.depth_image,
        #                     "current_pose": self.current_pose}
       
       
        # print(self.observation)
       

        info = {}

        return self.observation, reward, self.done, info

    def reset(self):
        print("Reset function called")
       
        self.last_position = None
        self.start_time = None
       
        self.green_pixels = 0
        self.collected_target_points = 0
        self.current_position = [0,0,0]
        self.detection_reward = 0
        self.min_distance = np.array([100])
        # self.current_pose = np.array([0,0,0,0,0,0])

        self.spawn_last_target = False
        self.ball_coords = [0,0]

        image = np.zeros((self.height, self.width)) 

        resized_depth_image = np.expand_dims(image, axis=-1)   
        self.depth_image = np.array(resized_depth_image, dtype=np.float32)
        self.observation = self.depth_image
        # self.observation = {"depth_map": self.depth_image,
        #                     "current_pose": self.current_pose}
       

        # delete_object_service('green_ball_1')
        # delete_object_service('green_ball_2')
        # delete_object_service('green_ball_3')
        # delete_object_service('green_ball_4')
        # self.reset_simulation_service()

        # self.reset_simulation()

        image = np.zeros((self.height, self.width))
        resized_depth_image = np.expand_dims(image, axis=-1)
        self.depth_image = np.array(resized_depth_image, dtype=np.float32)
        self.observation = self.depth_image

        
        # self.target_1 = spawn(1)
        # self.target_2 = spawn(2)
        # self.target_3 = spawn(3)

        # self.target_positions = {
        #     'green_ball_1': self.target_1[:2],
        #     # 'green_ball_2': self.target_2[:2],
        #     # 'green_ball_3': self.target_3[:2],
        # }
        self.target_positions = {
            'green_ball_1': [3, 0, 1],
            # 'green_ball_2': [0, 2, 1],
            # 'green_ball_3': [0, -2, 1],
        }
        
        return self.observation

def main(args=None):
    rclpy.init(args=args)
    drone_env = DroneEnv()
    rclpy.spin(drone_env)
    # drone_env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
