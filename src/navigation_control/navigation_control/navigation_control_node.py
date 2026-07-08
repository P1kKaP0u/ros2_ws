#!/usr/bin/env python3
"""Coordinate-based navigation controller for the AUV stack."""
import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from sensor_msgs.msg import Imu
from movement_control.state_machine import StateMachine
from .compute_distance import bearing_to_target, euclidean_distance, normalize_angle

class NavigationControlNode(Node):
    def __init__(self):
        super().__init__('navigation_control_node')

        # ---- Current / Target coordinates ----
        self.declare_parameter('current_x', 0.0)
        self.declare_parameter('current_y', 0.0)
        self.declare_parameter('current_z', 0.0)
        self.declare_parameter('target_x', 0.0)
        self.declare_parameter('target_y', 0.0)
        self.declare_parameter('target_z', 0.0)

        self.current_x = self.get_parameter('current_x').get_parameter_value().double_value
        self.current_y = self.get_parameter('current_y').get_parameter_value().double_value
        self.current_z = self.get_parameter('current_z').get_parameter_value().double_value
        self.target_x = self.get_parameter('target_x').get_parameter_value().double_value
        self.target_y = self.get_parameter('target_y').get_parameter_value().double_value
        self.target_z = self.get_parameter('target_z').get_parameter_value().double_value

        # ---- Current heading ----
        self.current_yaw = 0.0

        # ---- Control parameters ----
        self.linear_speed = 0.5
        self.angular_speed = 0.3
        self.depth_speed = 0.3
        self.distance_tolerance = 0.05
        self.depth_tolerance = 0.10
        self.yaw_tolerance = math.radians(10)
        self.max_forward_speed = 0.5

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
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def imu_callback(self, msg):
        q = msg.orientation
        self.current_yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

    def depth_callback(self, msg):
        self.current_z = msg.data
        self.get_logger().debug(f"Mevcut Derinlik: {self.current_z:.2f} m")

    def compass_callback(self, msg):
        self.current_yaw = math.radians((msg.data -245) % 360)
        self.get_logger().debug(f"Mevcut Yaw: {math.degrees(self.current_yaw):.2f}°")

    def control_loop(self):
        twist = Twist()
        distance_to_target = euclidean_distance(
            self.current_x,
            self.current_y,
            self.target_x,
            self.target_y,
        )
        target_bearing = bearing_to_target(
            self.current_x,
            self.current_y,
            self.target_x,
            self.target_y,
        )
        yaw_error = normalize_angle(target_bearing - self.current_yaw)
        depth_error = self.target_z - self.current_z

        self.get_logger().info(
            f"Target ({self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}) | "
            f"Current ({self.current_x:.2f}, {self.current_y:.2f}, {self.current_z:.2f}) | "
            f"distance={distance_to_target:.2f} m | yaw_error={math.degrees(yaw_error):.1f} deg | "
            f"depth_error={depth_error:.2f} m"
        )

        # -------------------- INIT --------------------
        if self.state_machine.state == "INIT":
            self.get_logger().info("StateMachine INIT -> DEPTH_CONTROL")
            self.state_machine.transition("DEPTH_CONTROL")

        # ---------------- DEPTH CONTROL ----------------
        elif self.state_machine.state == "DEPTH_CONTROL":
            if abs(depth_error) > self.depth_tolerance:
                twist.linear.z = self.depth_speed if depth_error > 0 else -self.depth_speed
            else:
                self.state_machine.transition("YAW_CONTROL")

        # ---------------- YAW CONTROL ----------------
        elif self.state_machine.state == "YAW_CONTROL":
            if abs(yaw_error) > self.yaw_tolerance:
                twist.angular.z = self.angular_speed if yaw_error > 0 else -self.angular_speed
            else:
                self.state_machine.transition("GO_FORWARD")

        # ---------------- GO FORWARD ----------------
        elif self.state_machine.state == "GO_FORWARD":
            if distance_to_target > self.distance_tolerance:
                twist.linear.x = self.linear_speed
            else:
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
