import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist  # Mesaj tipi

class ManualControlNode(Node):
    def __init__(self):
        super().__init__('manual_control_node')

        # /cmd_vel topic'ine mesaj yayınlamak için publisher oluştur
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("Manual Control Node başlatıldı.")

        #Timer ile düzenli olarak kullanıcan komut al
        self.timer = self.create_timer(0.1, self.ask_for_command)
        

    def ask_for_command(self):
        cmd = input("Komut gir (ileri, geri, yukarı, asağı, sola, sağa, dur)").lower()
        msg = Twist()

        if cmd == "ileri":
            msg.linear.x = 1.0
        elif cmd == "geri":
            msg.linear.x = -1.0
        elif cmd == "yukarı":
            msg.linear.z = 1.0
        elif cmd == "aşağı":
            msg.linear.z = -1.0
        elif cmd == "sola":
            msg.linear.y = 1.0
        elif cmd == "sağa":
            msg.linear.y = -1.0
        elif cmd == "dur":
            pass
        else:
            self.get_logger().info("Geçersiz komut. Lütfen tekrar deneyin.")
            return
        
        self.publisher.publish(msg)
        self.get_logger().info(f"Yayınlanan cmd_vel: {msg}")


def main(args=None):
    rclpy.init(args=args)
    manual_control_node = ManualControlNode()
    try:
        rclpy.spin(manual_control_node)
    except KeyboardInterrupt:
        pass
    finally:
        manual_control_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()