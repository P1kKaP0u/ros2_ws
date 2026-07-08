#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from collections import deque
from .pid_controller import PID
from std_msgs.msg import Float64, Bool, Int32
from geometry_msgs.msg import Pose2D
from .state_machine import StateMachine
import math
class MovementTargetNode(Node):
    def __init__(self):
        super().__init__('movement_target_node')

        self.cmd_vel_pub = self.create_publisher(Twist, '/target_cmd_vel', 10)
        self.create_subscription(Pose2D, '/pipe_pose_error', self.pose_callback, 10)
        self.create_subscription(Float64, '/mavros/global_position/rel_alt', self.depth_callback, 10)
        self.create_subscription(Bool, '/pipe_visible', self.search_callback, 10)
        self.create_subscription(Bool, '/mission_stop', self.mission_stop_callback, 10)
        self.create_subscription(Bool, '/pipe_end', self.pipe_end_callback, 10)
        self.set_mode_pub = self.create_publisher(Int32, '/vehicle/set_mode', 10)

        # Daha yumuşak derinlik kontrolü
        self.pid_depth = PID(kp=1.5, ki=0.1, kd=0.3, axis_name='z')
        self.pid_yaw   = PID(kp=0.5, ki=0.01, kd=0.2, axis_name='yaw')
        self.pid_x = PID(kp=1, ki=0.0, kd=0.1, axis_name='x')

        self.x_error_window = deque(maxlen=5)
        self.theta_error_window = deque(maxlen=5)

        # ---- Dikey eksen sınırlamaları ----
        self.max_up_cmd = 0.5       # yukarı yönde max komut (m/s ya da itki birimin)
        self.max_down_cmd = 0.5     # aşağı yönde max komut (daha küçük tut: yavaş batış)
        self.z_slew_step = 0.01      # her döngüde max değişim (slew)
        self.last_z_cmd = 0.0        # önceki komut (slew için)

        self.dead_zone = 0.05
        self.only_forward = True
        self.yaw_align_threshold = 0.15  # rad
        self.target_depth = -0.4
        self.depth_tolerance = 0.1
        self.state_machine = StateMachine()  # StateMachine nesnesi
        self.mission_stop = False
        self.mission_comp = False
        self.pipe_end = False


        self.current_depth = 0.0  # callback gelene kadar güvenli default

    def depth_callback(self, msg: Float64):
        self.current_depth = msg.data

    def mission_stop_callback(self, msg: Bool):
        self.mission_stop = msg.data

    def pipe_end_callback(self, msg:Bool):
        self.pipe_end = msg.data

    def _limit(self, val, lo, hi):
        return max(lo, min(hi, val))

    def _slew(self, desired, last, step):
        delta = desired - last
        if delta > step:
            delta = step
        elif delta < -step:
            delta = -step
        return last + delta


    def search_callback(self, msg: Bool):
        if not msg.data:
            twist = Twist()
            self.get_logger().warn("No pipe found -> Searching pipe")
            twist.angular.z = 0.3
            twist.linear.x = 0.1
            self.get_logger().info(f"[SEARCHING] yaw_cmd={twist.angular.z:.3f}, x_cmd={twist.linear.x:.3f}")
            self.cmd_vel_pub.publish(twist)


    def depth_control(self, twist: Twist, current_depth: float,target_depth: float):
            depth_error = target_depth - current_depth
            if abs(depth_error) < self.depth_tolerance: 
                self.get_logger().info("Hedef derinllikte")
                self.pid_depth.reset()
                self.last_z_cmd=0.0
                self.pid_yaw.reset()
                self.state_machine.transition("TRACKING")
            # PID -> ölçekle -> doyumla -> slew uygula
            u = self.pid_depth.compute(depth_error)
            u = self._limit(u, -self.max_down_cmd, self.max_up_cmd)  # negatif: aşağı, pozitif: yukarı (senin işaret sisteminle tutarlı)
            u = self._slew(u, self.last_z_cmd, self.z_slew_step)
            self.last_z_cmd = u

            twist.linear.z = u
            twist.linear.x = 0.0   # dalışta ileri hız çok düşük
            twist.angular.z = 0.0

            self.get_logger().info(f'[DEPTH_CONTROL] depth_err={depth_error:.2f}, z_cmd={twist.linear.z:.3f}, x={twist.linear.x:.2f}')



    def tracking_control(self, twist: Twist, msg: Pose2D, current_depth: float, target_depth: float):
            # Derinlikte kalmak için küçük düzeltmeler (istersen kapatabilirsin)
            filtered_theta = 0.0 if abs(msg.theta) < self.dead_zone else msg.theta
            depth_error = target_depth - current_depth
            if abs(depth_error)> 0.125:
                self.get_logger().info("Out of depth tolerance -> self.state=DEPTH_CONTROL")
                self.state_machine.transition("DEPTH_CONTROL")

            if abs(filtered_theta) < 0.3:
                twist.angular.z = 0.0
                twist.linear.z = 0.0

                self.x_error_window.append(msg.x)
                filtered_x = sum(self.x_error_window) / len(self.x_error_window)

                if abs(filtered_x) < self.dead_zone:
                    filtered_x = 0.0
                if self.only_forward and filtered_x < 0.0:
                    filtered_x = 0.0

                base_forward_thrust = 0.55
                pid_x_output = self.pid_x.compute(filtered_x)
                twist.linear.x = base_forward_thrust - pid_x_output

                self.get_logger().info(f'[TRACKING] x={twist.linear.x:.2f}, z={twist.linear.z:.3f}')
            # if filtered_theta > 1.4:
            #     twist.angular.z = math.radians(90)
            # elif filtered_theta < -1.4:
            #     twist.angular.z = math.radians(-90)
            elif abs(filtered_theta) > self.yaw_align_threshold:
                self.theta_error_window.append(msg.theta)
                filtered_theta = sum(self.theta_error_window) / len(self.theta_error_window)
                twist.angular.z = self.pid_yaw.compute(filtered_theta)
                self.get_logger().info(f'[ALIGNING] x={twist.linear.x:.2f}, yaw_cmd={twist.angular.z:.2f}, z={twist.linear.z:.3f}')



    def pose_callback(self, msg: Pose2D):
        twist = Twist()

        if self.state_machine.state == "INIT":
            self.get_logger().info("StateMachine INIT -> DEPTH_CONTROL")
            self.pid_yaw.reset()
            self.pid_x.reset()
            self.state_machine.transition("DEPTH_CONTROL")


        elif self.state_machine.state == "DEPTH_CONTROL":
            self.depth_control(twist, self.current_depth, self.target_depth)


        elif self.state_machine.state == "TRACKING":
            self.tracking_control(twist, msg, self.current_depth, self.target_depth)

        if self.mission_stop and self.pipe_end:
            self.state_machine.state = "STOP"
            self.get_logger().info("MISSION DONE")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            twist.linear.z = 0.0
            self.set_mode_pub.publish(Int32(data=19))
            self.get_logger().info("Mode değiştirildi: Manual (19)")


        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = MovementTargetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
