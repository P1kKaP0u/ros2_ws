import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class JoyToCmdVel(Node):
    def __init__(self):
        super().__init__('joy_to_cmd_vel')
        self.sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def joy_callback(self, msg):
        twist = Twist()
        
        # Joystick eksenlerini kontrol et: ros2 topic echo /joy
        twist.linear.x = msg.axes[1] * 0.5   # Sol çubuk yukarı-aşağı → ileri-geri
        twist.linear.z = msg.axes[3] * 0.5   # Sağ çubuk yukarı-aşağı → yukarı-aşağı
        twist.angular.z = msg.axes[0] * 0.5   # Sol çubuk sağa-sola → yaw

        self.pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = JoyToCmdVel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 