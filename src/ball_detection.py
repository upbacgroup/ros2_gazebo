#!/usr/bin/env python3

# Python libs
import sys
import time

# numpy and scipy
import numpy as np
from scipy.ndimage import filters

import imutils

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from gazebo_msgs.srv import DeleteModel, DeleteModelRequest

import random
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose, Point, Quaternion


VERBOSE = False


class image_feature:
    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        rospy.init_node('image_feature', anonymous=True)
        # topic where we publish
        self.image_sub = rospy.Subscriber("depth_camera/depth/image_raw", Image, self.camera_callback) # depth camera
        self.image_pub = rospy.Publisher("output/image_raw/compressed",
                                         CompressedImage, queue_size=1)
        self.vel_pub = rospy.Publisher("/cmd_vel",
                                       Twist, queue_size=1)

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/camera1/image_raw/compressed",
                                           CompressedImage, self.callback,  queue_size=1)
        self.spawn_model_proxy = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)

        
        self.width = 640
        self.height = 480
        self.counter = 0
        self.spawn_robot()

    def spawn_robot(self):
        # Define URDF model file path
        # urdf_file_path = '$DRONE_MODEL_XML'
        spawn_model = SpawnModel()
        urdf_file_path = '/home/ecem/Documents/drone_ws/src/air_drone/urdf/drone.urdf'

        spawn_request = spawn_model.SpawnModelRequest()
        spawn_request.model_name = 'drone'
        spawn_request.model_xml = urdf_file_path
        spawn_request.robot_namespace = "/"  # Robot namespace
        model_pose = Pose(position=Point(5, 2, 0.5), orientation=Quaternion(0, 0, 0, 1))
        spawn_request.initial_pose = model_pose
        spawn_request.reference_frame = 'odom'

        try:
            # Call Gazebo's spawn model service
            spawn_model.spawn_model(spawn_request)
            rospy.loginfo("Spawned robot at random location: {}".format(model_pose))
        except rospy.ServiceException as e:
            rospy.logerr("Failed to spawn robot: {}".format(e))




    def delete_object_service(self, object_name):
        rospy.wait_for_service('/gazebo/delete_model')
        print("*****************waited**********************")
        
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            req = DeleteModelRequest()
            req.model_name = object_name  # Specify the name of the object to delete
            
            # Call the service to delete the object
            resp = delete_model(req)
            
            rospy.loginfo(f"Deleted object: {object_name}")
            
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to delete object: {object_name}, error: {e}")

    def camera_callback(self, image_msg):
        # Handling for depth camera data 
        try:
            # self.spawn_robot()
            depth_data = np.frombuffer(image_msg.data, dtype=np.float32)
            
            depth_image = depth_data.reshape((image_msg.height, image_msg.width))
            # Handle NaN values by replacing them with zeros
            depth_image = np.nan_to_num(depth_image)
  
            epsilon = 1e-5
            denominator = (depth_image.max() - depth_image.min())
            if denominator == 0:
                denominator = epsilon  # Avoid division by zero
            normalized_depth = ((depth_image - depth_image.min()) / denominator * 255).astype(np.uint8)

           
            target_width, target_height = self.height, self.width  # Adjust these dimensions as needed
            resized_depth = cv2.resize(normalized_depth, (target_width, target_height))
      

            self.current_depth_map = resized_depth
            self.camera_data = resized_depth

            center_x, center_y = resized_depth.shape[1] // 2, resized_depth.shape[0] // 2
            depth_value = depth_image[center_y, center_x]
            self.distance = depth_value
            # rospy.loginfo(f"Distance: {self.distance}")
       
            # if self.current_depth_map is not None:
            #     self.current_depth_map = resized_depth
            #     self.camera_data = resized_depth
                
            # else:
            #     self.observation = np.zeros((self.height, self.width, 1), dtype=np.float32)  # Default value
                

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
                # vel.linear.x = self.distance
                self.vel_pub.publish(vel)
                # if self.counter == 0:
                #     self.delete_object_service('green_ball_1_1')
                #     self.counter = 1
                # elif self.counter == 1:  
                #     self.delete_object_service('green_ball_1')
                #     self.counter = 2
                # else:
                #     self.delete_object_service('green_ball_1_0')
                #     self.counter = 3       

        else:
            vel = Twist()
            vel.angular.z = 0.5
            self.vel_pub.publish(vel)
            

        # update the points queue
        # pts.appendleft(center)
        cv2.imshow('window', image_np)
        cv2.waitKey(2)

        # self.subscriber.unregister()

    

def main(args):
    '''Initializes and cleanup ros node'''
    ic = image_feature()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
