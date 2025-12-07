import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SimpleSubscriber(Node):
    """
    A simple ROS 2 subscriber node.
    This node subscribes to the 'chatter' topic and prints the received messages.
    """
    def __init__(self):
        super().__init__('simple_subscriber')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.get_logger().info('Subscriber node started.')

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    
    simple_subscriber = SimpleSubscriber()
    
    try:
        rclpy.spin(simple_subscriber)
    except KeyboardInterrupt:
        pass
    
    # Destroy the node explicitly
    simple_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
