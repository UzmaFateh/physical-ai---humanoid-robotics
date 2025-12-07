import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SimplePublisher(Node):
    """
    A simple ROS 2 publisher node.
    This node publishes a "Hello World" message to the 'chatter' topic every second.
    """
    def __init__(self):
        super().__init__('simple_publisher')
        self.publisher_ = self.create_publisher(String, 'chatter', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.i = 0
        self.get_logger().info('Publisher node started.')

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    
    simple_publisher = SimplePublisher()
    
    try:
        rclpy.spin(simple_publisher)
    except KeyboardInterrupt:
        pass
    
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    simple_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
