import rclpy
from std_msgs.msg import Empty

def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('takeoff_node')
    publisher = node.create_publisher(Empty, '/simple_drone/takeoff', 10)

    msg = Empty()
    publisher.publish(msg)

    node.get_logger().info('Publishing empty message on /simple_drone/takeoff')

    # Spin the node for a short duration to ensure the message is published
    rclpy.spin_once(node, timeout_sec=1.0)

    # Shutdown the node
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
