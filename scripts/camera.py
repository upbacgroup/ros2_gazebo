#!/usr/bin/env python3
import rospy
import sensor_msgs.point_cloud2 as pc2
import numpy as np


def callback(data):
    # Convert the point cloud message to a NumPy array
    points = np.array(list(pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)))
    # Calculate the distance between the camera and the closest point in the point cloud
    dist = np.min(np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2))

    rospy.loginfo("Distance: %f", dist)


def listener():
    # Initialize the node
    rospy.init_node('camera_node', anonymous=True)
    # Subscribe to the point cloud topic
    rospy.Subscriber("/depth_camera/depth/points", pc2.PointCloud2, callback)
    # Spin to keep the node alive
    rospy.spin()


if __name__ == '__main__':

    try:
        listener()
    except rospy.ROSInterruptException:
        pass



    

   
    