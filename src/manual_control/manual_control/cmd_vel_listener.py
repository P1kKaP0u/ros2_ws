#!/usr/bin/env python3
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Int32
from sensor_msgs.msg import Imu
from pymavlink import mavutil
import threading
import math

class CmdVelListenerNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_listener_node')

        self.declare_parameter('simulation_mode', False)
        self.declare_parameter('connection_uri', '/dev/ttyACM0')
        self.declare_parameter('baud', 115200)

        self.simulation_mode = self.get_parameter('simulation_mode').value
        self.connection_uri = self.get_parameter('connection_uri').value
        self.baud = self.get_parameter('baud').value

        self.master = None
        self.last_cmd = Twist()
        self.sim_depth = 0.0
        self.sim_heading = 245.0

        if self.simulation_mode:
            self.get_logger().warn("Simulation mode enabled: no MAVLink connection will be opened.")
        else:
            try:
                self.get_logger().info(f"MAVLink bağlantısı başlatılıyor... ({self.connection_uri})")
                self.master = mavutil.mavlink_connection(self.connection_uri, baud=self.baud)
                self.master.wait_heartbeat()
                self.get_logger().info("MAVLink bağlantısı sağlandı.")
            except Exception as e:
                self.get_logger().warn(f"MAVLink bağlantısı açılamadı, simulation mode'a geçiliyor: {e}")
                self.simulation_mode = True

        # Mode subscriber
        self.mode_sub = self.create_subscription(
            Int32,
            '/vehicle/set_mode',
            self.mode_callback,
            10
        )

        # Başlangıç mode: depth_hold (2)  , 19 manual 
        self.set_mode(2)
        self.get_logger().info("Başlangıç modu: Depth Hold (2)")

        # cmd_vel subscriber
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/target_cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Publishers
        self.depth_pub = self.create_publisher(Float64, '/mavros/global_position/rel_alt', 10)
        self.heading_pub = self.create_publisher(Float64, '/compass/heading', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)

        if self.simulation_mode:
            self.sim_timer = self.create_timer(0.1, self.simulation_tick)
        else:
            # MAVLink dinleme thread'i
            self.mavlink_thread = threading.Thread(target=self.mavlink_listener, daemon=True)
            self.mavlink_thread.start()

        self.last_attitude = None

    def cmd_vel_callback(self, msg):
        self.last_cmd = msg

        if self.simulation_mode:
            self.get_logger().info(
                f"SIM cmd_vel -> x:{msg.linear.x:.2f}, y:{msg.linear.y:.2f}, "
                f"z:{msg.linear.z:.2f}, r:{msg.angular.z:.2f}"
            )
            return

        x = int(max(min(msg.linear.x * 1000, 1000), -1000))
        y = int(max(min(msg.linear.y * 1000, 1000), -1000))
        z = int(max(min((msg.linear.z * 500) + 500, 1000), 0))
        r = int(max(min(msg.angular.z * 1000, 1000), -1000))
        buttons = 1 << 6
        self.master.mav.manual_control_send(
            self.master.target_system,
            x, y, z, r, buttons
        )
        self.get_logger().info(f"cmd_vel -> x:{x}, y:{y}, z:{z}, r:{r}")

    def mavlink_listener(self):
        self.get_logger().info("MAVLink dinleme thread'i başlatıldı.")
        while rclpy.ok():
            msg = self.master.recv_match(blocking=True, timeout=1)
            if not msg:
                continue

            msg_type = msg.get_type()

            # GLOBAL_POSITION_INT → depth & heading
            if msg_type == 'GLOBAL_POSITION_INT':
                relative_alt = msg.relative_alt / 1000.0
                self.depth_pub.publish(Float64(data=relative_alt))

                if hasattr(msg, 'hdg') and msg.hdg is not None:
                    heading_deg = msg.hdg / 100.0
                    self.heading_pub.publish(Float64(data=heading_deg))

    def simulation_tick(self):
        self.sim_depth = max(0.0, self.sim_depth - self.last_cmd.linear.z * 0.02)
        self.sim_heading = (self.sim_heading + math.degrees(self.last_cmd.angular.z) * 0.5) % 360.0

        self.depth_pub.publish(Float64(data=self.sim_depth))
        self.heading_pub.publish(Float64(data=self.sim_heading))

    def mode_callback(self, msg):
        mode = msg.data
        self.set_mode(mode)

    def set_mode(self, mode):
        if self.simulation_mode:
            self.get_logger().info(f"SIM mode request: {mode}")
            return

        try:
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode
            )
            self.get_logger().info(f"Mode değiştirildi: {mode}")
        except Exception as e:
            self.get_logger().error(f"Mode değiştirme hatası: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelListenerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
