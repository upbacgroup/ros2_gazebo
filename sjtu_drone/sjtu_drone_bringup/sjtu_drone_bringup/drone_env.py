import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from std_msgs.msg import Empty as EmptyMsg



from cv_bridge import CvBridge
import gym
import cv2
import numpy as np
import time

from spawn_entities import SpawnEntityClient
from delete_entities import DeleteEntityClient
from gazebo_msgs.srv import SpawnEntity


class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        rclpy.init(args=None)
        self.node = Node('drone_env')

        print('**********HELLO FROM MY ENV************')
        self.width = 80
        self.height = 45
        
        self.bridge = CvBridge()
        self.red_lower = (0, 100, 100)  # Lower bounds for red in HSV
        self.red_upper = (10, 255, 255)  # Upper bounds for red in HSV
        self.green_lower = (40, 50, 50)  # Lower bounds for green in HSV
        self.green_upper = (70, 255, 255)  # Upper bounds for green in HSV
        self.focal_length = 150  # Focal length in pixels
        self.known_width = 0.15 
        self.in_flight = False



        self.collected_targets = []
        self.depth_image = []
        self.detection_reward = 0
        self.current_position = None
        self.last_position = None
        self.start_time = None


        self.delete_entity_client = DeleteEntityClient()
        self.spawn_entity_client = SpawnEntityClient()




        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height,self.width,3), dtype=np.uint8)

        self.current_pose_sub = self.node.create_subscription(Odometry, '/simple_drone/odom', self.position_callback, 10)
       
        self.image_sub = self.node.create_subscription(Image, '/simple_drone/front/image_raw', self.camera_callback, 10)
        
        self.speed_motors_pub = self.node.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.reset_client = self.node.create_client(Empty, '/reset_world')
        self.spawn_client = self.node.create_client(SpawnEntity, '/spawn_entity')
        self.takeoff_publisher = self.node.create_publisher(EmptyMsg, '/simple_drone/takeoff', 10)
        self.land_publisher = self.node.create_publisher(EmptyMsg, '/simple_drone/land', 10)
 
       
    def reset_simulation(self):
        print("reset called")
        
        # self.land_publisher.publish(EmptyMsg())
        # self.node.get_logger().info("Land command sent")


        # self.delete_entity_client.send_request('green_ball_0')
        # self.targets = None
        # self.targets = self.spawn_entity_client.send_request('green_ball', 1)

        
        
        request = Empty.Request()
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        if future.result() is not None:
            self.node.get_logger().info('Simulation reset successful!')
        else:
            self.node.get_logger().error('Failed to reset simulation')

        
           

    
    def position_callback(self, msg:Odometry):
        self._gt_pose = msg

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        self.current_position = np.array([x, y, z])

        tolerance=0.2

        # self.node.get_logger().info(f"pos in clb: {self.current_position}")
        

        if self.last_position is None:
            self.last_position = self.current_position
            self.start_time = time.time()
    
        else:
            if np.linalg.norm(np.array(self.last_position) - np.array(self.current_position)) < tolerance:
                a = time.time() - self.start_time
                
                if  a >= 5.0:
                    print("Drone position has remained stable for 5 seconds.")
                    self.done=True
                    
            else:
                self.last_position = self.current_position
                self.start_time = time.time()    

    def camera_callback(self, image_msg):
       
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8') 
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Detect red color
        red_mask = cv2.inRange(hsv_image, self.red_lower, self.red_upper)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detect green color
        green_mask = cv2.inRange(hsv_image, self.green_lower, self.green_upper)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count green pixels
        self.num_green_pixels = cv2.countNonZero(green_mask)

        for cnt in red_contours:
            cv2.drawContours(cv_image, [cnt], 0, (0, 0, 255), 2)  # Red color for red contours
        for cnt in green_contours:
            cv2.drawContours(cv_image, [cnt], 0, (0, 255, 0), 2)  # Green color for green contours


        # Draw contours and (optional) text on the image
        for cnt in red_contours:
            x, y, w, h = cv2.boundingRect(cnt)  # Get bounding rectangle of the contour
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw rectangle for red
            distance = (self.known_width * self.focal_length) / w  # Placeholder for calculated distance
            self.wall_dist = distance
            cv2.putText(cv_image, f'{distance:.2f} m', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Add text with distance

        for cnt in green_contours:
            x, y, w, h = cv2.boundingRect(cnt)  # Get bounding rectangle of the contour
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle for green



        # Display the result
        cv2.imshow('Object Detection', cv_image)
        cv2.waitKey(1)

    def object_name_finder(self):
        object_name = None
        current_position_xy = self.current_position[:2] 
        x = 0

        distance = np.linalg.norm(current_position_xy - self.targets[:2])
        # print(f"closest ball: {distance}")
        if distance < 0.7 and x == 0:
            print("*****TARGET COLLECTED*******")
            self.done = True
            x += 1

        self.min_distance = distance



        return object_name

    def check_collision(self):
        collision_penalty = 0
        # print(self.min_non_zero_distance)
        # print(f"pos in col: {self.current_position}")
        if self.collision_count:
            return
        
        if self.wall_dist < 0.2 or abs(self.current_position[0])>8 or abs(self.current_position[1])>8:  
            # if collision_count == 0 and abs(self.current_position[0])>3.0 or abs(self.current_position[1])>3.0:      
                print("*******************collision*********************")
                collision_penalty = -1
                self.done=True
        else:
            self.collision_penalty = 0 

        return collision_penalty

    def take_action(self, action):
        self.object_name_finder()

        linear_x = (action[0]+1)/2
        angular_z = action[1]
        
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_x
        vel_cmd.angular.z = float(angular_z)
        
        self.speed_motors_pub.publish(vel_cmd)
       

    def step(self, action):
        # print("Step function called")
        self.collision_count = False
        self.done = False
        self.take_action(action)
        rclpy.spin_once(self.node, timeout_sec=0.1)
        collision_penalty = self.check_collision() 
        
        reward = -1 + 2/(1+np.exp(0.5*self.min_distance/(self.num_green_pixels+1))) - collision_penalty

        # reward = -1 + 2/(1+np.exp(0.5*self.min_distance))
        # print(f"# Green pxl: {self.num_green_pixels}")
        # print(f"Reward: {reward}")


           
        self.observation = self.depth_image

        info = {}

        return self.observation, reward, self.done, info

    def reset(self):
        # print("Reset function called")
        

        self.last_position = None
        self.start_time = None
       
        self.green_pixels = 0
        self.collected_target_points = 0
        self.current_position = [0,0,0]

        self.min_distance = np.array([100])


        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.depth_image = np.array(image)
        self.observation = self.depth_image


        
        self.land_publisher.publish(EmptyMsg())
        self.node.get_logger().info("Land command sent")


        self.delete_entity_client.send_request('green_ball_0')
        self.targets = None
        

        
        self.reset_simulation()
        time.sleep(2)
        self.targets = self.spawn_entity_client.send_request('green_ball', 1)
        time.sleep(2)

        if self.targets is not None:
            self.takeoff_publisher.publish(EmptyMsg())
            self.node.get_logger().info("Takeoff command sent")
        else:
            self.node.get_logger().error("No targets available, not taking off") 
        
        print(self.targets)

        
        rclpy.spin_once(self.node, timeout_sec=0.1)
        return self.observation
    

    def close(self):
        # Clean up ROS 2 resources
        self.node.destroy_node()
        rclpy.shutdown()
        self.ros_thread.join()


