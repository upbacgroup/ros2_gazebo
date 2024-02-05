#!/usr/bin/env python3

import rospy
import random

if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('random_node', anonymous=True)
    
    # Generate random x and y values
    random_x = random.uniform(-10, 10)  # Adjust bounds as needed
    random_y = random.uniform(-10, 10)  # Adjust bounds as needed
    
    # Set random x and y values as ROS parameters
    rospy.set_param('/random_node_x', random_x)
    rospy.set_param('/random_node_y', random_y)
    
    # Log the generated random values
    rospy.loginfo("Random x: %s", random_x)
    rospy.loginfo("Random y: %s", random_y)
    
    # Spin the node to keep it alive
    rospy.spin()

