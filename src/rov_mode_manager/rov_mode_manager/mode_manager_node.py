from lifecycle_msgs.msg import Transition
from rclpy.lifecycle import LifecycleNode
import rclpy
from std_msgs.msg import String, Bool
from .mode_state import ROVMode
from .arm_state  import ArmState
from .node_registry import NodeRegistry


class ModeManagerNode(LifecycleNode):
    def __init__(self):
        super().__init__('rov_mode_manager')

        self.current_mode = ROVMode.MANUAL
        self.arm_state    = ArmState.DISARMED

        # registry oluştur (kendisi parent node)
        self.registry = NodeRegistry(self)

        # topic abonelikleri
        self.create_subscription(String, 'set_mode', self.mode_callback, 10)
        self.create_subscription(Bool,   'set_arm',  self.arm_callback, 10)

        # durum yayıncıları
        self.mode_pub = self.create_publisher(String, 'current_mode', 10)
        self.arm_pub  = self.create_publisher(Bool,   'arm_state',   10)

        self.get_logger().info("ROV Mode Manager initialized (UNCONFIGURED)")

    def activate_mode(self, new_mode: ROVMode):
        self.get_logger().info(f"Mod değişiyor: {self.current_mode.value} → {new_mode.value}")

        for mode in ROVMode:
            key = f"{mode.value}_node"
            if mode == new_mode:
                self.registry.change_state(key, Transition.TRANSITION_ACTIVATE)
            else:
                self.registry.change_state(key, Transition.TRANSITION_DEACTIVATE)
                self.registry.change_state(key, Transition.TRANSITION_CLEANUP)

        self.current_mode = new_mode
        self.mode_pub.publish(String(data=new_mode.value))

    def mode_callback(self, msg: String):
        try:
            new_mode = ROVMode(msg.data.lower())
        except ValueError:
            self.get_logger().warn(f"Geçersiz mod: {msg.data}")
            return

        if new_mode != self.current_mode:
            self.activate_mode(new_mode)
        else:
            self.get_logger().info(f"Zaten bu moddayız: {new_mode.value}")

    def arm_callback(self, msg: Bool):
        new_state = ArmState.ARMED if msg.data else ArmState.DISARMED
        if new_state != self.arm_state:
            self.get_logger().info(f"ARM state: {self.arm_state.name} → {new_state.name}")
            self.arm_state = new_state
            self.arm_pub.publish(Bool(data=(new_state == ArmState.ARMED)))
        else:
            self.get_logger().info(f"ARM state zaten: {self.arm_state.name}")


def main(args=None):
    rclpy.init(args=args)
    node = ModeManagerNode()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        rclpy.shutdown()
