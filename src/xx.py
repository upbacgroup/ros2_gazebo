#!/usr/bin/env python3

import rospy

if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('random_xy_node', anonymous=True)
    
    # Retrieve random x and y values
    random_x = rospy.get_param('/x', 0.0)
    random_y = rospy.get_param('/y', 0.0)
    
    # Log the retrieved random values
    rospy.loginfo("Retrieved random x: %s", random_x)
    rospy.loginfo("Retrieved random y: %s", random_y)
    
    # Spin the node to keep it alive
    rospy.spin()
