import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from std_srvs.srv import Empty as EmptyDrone
from std_msgs.msg import Empty as EmptyMsg



from cv_bridge import CvBridge
import gym
import cv2
import numpy as np
import time

from spawn_entities import SpawnEntityClient
from delete_entities import DeleteEntityClient
from spawn_drone import SpawnDroneNode
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.msg import ContactsState
# import spawn_drone


class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        rclpy.init(args=None)
        self.node = Node('drone_env')

        print('**********HELLO FROM MY ENV************')
        self.width = 80
        self.height = 45
        
        self.bridge = CvBridge()
        
        self.green_lower = (40, 50, 50)  # Lower bounds for green in HSV
        self.green_upper = (70, 255, 255)  # Upper bounds for green in HSV

        self.collected_targets = []
        self.image = []
        self.detection_reward = 0
        self.current_position = None
        self.last_position = None
        self.start_time = None


        self.delete_entity_client = DeleteEntityClient()
        self.spawn_entity_client = SpawnEntityClient()
        self.spawn_drone_client = SpawnDroneNode()


        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height,self.width,3), dtype=np.uint8)

        self.image_sub = self.node.create_subscription(Image, '/simple_drone/front/image_raw', self.camera_callback, 10)
        self.current_pose_sub = self.node.create_subscription(Odometry, '/simple_drone/odom', self.position_callback, 10)
        self.collision_sub = self.node.create_subscription(ContactsState, '/simple_drone/bumper_states', self.collision_callback, 10)
       
        self.speed_motors_pub = self.node.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_publisher = self.node.create_publisher(EmptyMsg, '/simple_drone/takeoff', 10)
        self.land_publisher = self.node.create_publisher(EmptyMsg, '/simple_drone/land', 10)

        self.reset_client = self.node.create_client(Empty, '/reset_world')
        self.spawn_client = self.node.create_client(SpawnEntity, '/spawn_entity')
        self.spawn_drone = self.node.create_client(EmptyDrone, 'spawn_drone_random')
        

 
       
    def reset_simulation(self):
        print("reset called")
        
        request = Empty.Request()
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        if future.result() is not None:
            self.node.get_logger().info('Simulation reset successful!')
        else:
            self.node.get_logger().error('Failed to reset simulation')
        
    def call_spawn_drone_service(self):
        req = EmptyDrone.Request()
        future = self.spawn_drone.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        
        if future.result() is not None:
            if future.result().success:
                self.node.get_logger().info('Drone spawned successfully: ' + future.result().message)
            else:
                self.node.get_logger().warn('Failed to spawn drone: ' + future.result().message)
        else:
            self.node.get_logger().error('Service call failed: ' + str(future.exception()))


        
    def position_callback(self, msg:Odometry):
        self._gt_pose = msg

        position = msg.pose.pose.position
        self.current_position = np.array([position.x, position.y, position.z])


        if self.last_position is None:
            self.last_position = self.current_position
            self.start_time = time.time()
        else:
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 20.0:
                print("Episode time exceeded")
                self.done = True        
   

    def camera_callback(self, image_msg):
        self.node.get_logger().info('Received an image with timestamp: ' + str(image_msg.header.stamp))

        cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8') 
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Detect green color
        green_mask = cv2.inRange(hsv_image, self.green_lower, self.green_upper)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count green pixels
        self.num_green_pixels = cv2.countNonZero(green_mask)

        for cnt in green_contours:
            cv2.drawContours(cv_image, [cnt], 0, (0, 255, 0), 2)  # Green color for green contours

        # grayscale_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # grayscale_image = np.expand_dims(grayscale_image, axis=-1)
        # self.image = grayscale_image

        self.image = cv_image


        # Resize the image
        scale_percent = 400  # Percentage of original size
        width = int(cv_image.shape[1] * scale_percent / 100)
        height = int(cv_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(cv_image, dim, interpolation=cv2.INTER_AREA)

        # Display the resized image
        cv2.imshow('Object Detection', resized_image)
        cv2.waitKey(1)

    def object_name_finder(self):
        object_name = None
        current_position_xy = self.current_position[:2] 
        x = 0
        if self.targets is not None:
            distance = np.linalg.norm(current_position_xy - self.targets[:2])
            # print(f"closest ball: {distance}")
            if distance < 0.7 and x == 0:
                print("*****TARGET COLLECTED*******")
                self.done = True
                x += 1

            self.min_distance = distance



        return object_name

    def collision_callback(self, msg):
        for state in msg.states:
            # Extract the collision names
            collision1_name = state.collision1_name
            collision2_name = state.collision2_name
            
            # Check if either collision name contains "Wall"
            if 'Wall' in collision1_name or 'Wall' in collision2_name:
                print("Collision with a wall detected!")
                self.done = True
       
    def take_action(self, action):
        self.object_name_finder()

        linear_x = (action[0]+1)/2
        angular_z = action[1]
        
        vel_cmd = Twist()
        vel_cmd.linear.x = 4*linear_x
        vel_cmd.angular.z = 2*float(angular_z)
        
        self.speed_motors_pub.publish(vel_cmd)
       
    def step(self, action):

        self.take_action(action)
        rclpy.spin_once(self.node, timeout_sec=0.1)
        
        
        reward = -1 + 2/(1+np.exp(0.5*self.min_distance/(100*self.num_green_pixels+1)))
        # reward = -1 + 2/(1+np.exp(0.5*self.min_distance))
        # print(f"# Green pxl: {self.num_green_pixels}")
        print(f"Reward: {reward}")


           
        self.observation = self.image
        # print(self.observation)
        # pos = self.current_position[:2]
        # self.observation = pos

        info = {}

        return self.observation, reward, self.done, info

    def reset(self):
        # print("Reset function called")
        
        self.done = False
        self.last_position = None
        self.start_time = None
       
        self.green_pixels = 0
        self.collected_target_points = 0
        self.current_position = [0,0,0]

        self.min_distance = np.array([100])


        image = np.zeros((self.height, self.width,3), dtype=np.uint8)
        self.image = np.array(image)
        # pos = self.current_position[:2]
        # self.observation = pos
        self.observation = self.image


        
        self.land_publisher.publish(EmptyMsg())
        self.node.get_logger().info("Land command sent")


        self.delete_entity_client.send_request('green_ball_0')
        self.targets = None
        

        
        self.reset_simulation()
        self.spawn_drone_client.spawn_drone_callback()
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
        # self.ros_thread.join()


