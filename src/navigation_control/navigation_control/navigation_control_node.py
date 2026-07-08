#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
from movement_control.state_machine import StateMachine


class NavigationControlNode(Node):
    def __init__(self):
        super().__init__('navigation_control_node')

        # ---- Hedef Konum ----
        self.declare_parameter('target_x', 0.0)
        self.declare_parameter('target_y', 0.0)
        self.declare_parameter('target_z', 0.0)

        self.target_x = self.get_parameter('target_x').get_parameter_value().double_value
        self.target_y = self.get_parameter('target_y').get_parameter_value().double_value
        self.target_z = self.get_parameter('target_z').get_parameter_value().double_value

        # ---- Mevcut Konum ----
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_yaw = 0.0

        # ---- PID / kontrol parametreleri ----
        self.linear_speed = 0.5
        self.angular_speed = 0.3
        self.depth_speed = 0.3
        self.distance_tolerance = 0.05
        self.yaw_tolerance = math.radians(10)

        # ---- Mesafe hesaplaması için değişkenler ----,
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        self.target_angle = math.atan2(dy, dx)

        self.total_distance_traveled = 0.0
        self.initial_distance = math.hypot(dx, dy)  # Başlangıçta hedef ile aradaki mesafeyi sakla
        self.last_time = self.get_clock().now()

        # ---- State Machine ----
        self.state_machine = StateMachine()

        # ---- Subscriberlar ----
        self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.depth_callback, 10)
        self.create_subscription(Float64, '/compass/heading', self.compass_callback, 10)

        # ---- Publisher ----
        self.cmd_vel_pub = self.create_publisher(Twist, '/target_cmd_vel', 10)

        # ---- Timer ----
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info("Navigation Control Node Başlatıldı!")

    def quaternion_to_yaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
        return math.atan2(siny_cosp, cosy_cosp)

    def imu_callback(self, msg):
        q = msg.orientation
        self.current_yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

    def depth_callback(self, msg):
        self.current_z = msg.data
        self.get_logger().info(f"Mevcut Derinlik: {self.current_z:.2f} m")

    def compass_callback(self, msg):
        self.current_yaw = math.radians((msg.data -245) % 360)
        self.get_logger().info(f"Mevcut Yaw: {math.degrees(self.current_yaw):.2f}°")

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def control_loop(self):
        # now = self.get_clock().now()
        # dt = (now - self.last_time).nanoseconds * 1e-9
        # self.last_time = now
        twist = Twist()

        # -------------------- INIT --------------------
        if self.state_machine.state == "INIT":
            self.get_logger().info("StateMachine INIT -> DEPTH_CONTROL")
            self.state_machine.transition("DEPTH_CONTROL")

        # ---------------- DEPTH CONTROL ----------------
        elif self.state_machine.state == "DEPTH_CONTROL":
            depth_error = self.target_z - self.current_z
            if abs(depth_error) > 0.1:
                twist.linear.z = self.depth_speed if depth_error > 0 else -self.depth_speed
            else:
                self.state_machine.transition("YAW_CONTROL")

        # ---------------- YAW CONTROL ----------------
        elif self.state_machine.state == "YAW_CONTROL":
            yaw_error = self.normalize_angle(self.target_angle - self.current_yaw)

            if abs(yaw_error) > self.yaw_tolerance:
                twist.angular.z = self.angular_speed if yaw_error > 0 else -self.angular_speed
            else:
                self.state_machine.transition("GO_FORWARD")

        # ---------------- GO FORWARD ----------------
        elif self.state_machine.state == "GO_FORWARD":
            twist.linear.x = self.linear_speed
            self.total_distance_traveled += self.linear_speed * 0.060
            distance_to_target = max(0.0, self.initial_distance - self.total_distance_traveled)

            self.get_logger().info(
                f"Alınan yol: {self.total_distance_traveled:.2f} m | "
                f"Kalan mesafe: {distance_to_target:.2f} m"
            )

            if distance_to_target <= self.distance_tolerance:
                self.get_logger().info("Hedefe ulaşıldı!")
                self.state_machine.transition("STOP")

        # ---------------- STOP ----------------
        elif self.state_machine.state == "STOP":
            twist.linear.x = 0.0
            twist.linear.z = 0.0
            twist.angular.z = 0.0

        # Tek bir yerde publish et
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = NavigationControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
