from rov_mode_manager.mode_state import ROVMode
from lifecycle_msgs.srv import ChangeState
import rclpy


class NodeRegistry:
    """Bir lifecycle node’un ChangeState servisi istemcilerini yönetir."""

    def __init__(self, parent_node: rclpy.node.Node):
        self.node = parent_node
        # mapping: registry adı → ROS 2 node adı (namespace/topic prefix)
        self.all_nodes = {
            "manual_control_node": "manual_control",
            "stabilize_node":      "stabilize",
            "depth_hold_node":     "depth_hold",
            "auto_mission_node":   "auto_mission",
        }
        # her node için ChangeState istemcisi oluştur
        self.clients = {}
        for key, ros_name in self.all_nodes.items():
            srv_name = f"/{ros_name}/change_state"
            client = self.node.create_client(ChangeState, srv_name)
            self.clients[key] = client

        # hangi modda hangi node’lar aktif?
        self.mode_nodes = {
            ROVMode.MANUAL:    ["manual_control_node"],
            ROVMode.STABILIZE: ["manual_control_node", "stabilize_node"],
            ROVMode.DEPTH_HOLD:["manual_control_node", "depth_hold_node"],
            ROVMode.AUTO_MISSION: ["auto_mission_node"],
        }

    def get_nodes_for_mode(self, mode: ROVMode):
        return self.mode_nodes.get(mode, [])

    def change_state(self, node_key: str, transition_id: int, timeout_sec=2.0) -> bool:
        """Senkron olarak ChangeState servisini çağırır ve sonucu döner."""
        client = self.clients.get(node_key)
        if not client:
            self.node.get_logger().warn(f"Client bulunamadı: {node_key}")
            return False

        if not client.wait_for_service(timeout_sec=timeout_sec):
            self.node.get_logger().warn(f"Service açılamadı: {node_key}")
            return False

        req = ChangeState.Request()
        req.transition.id = transition_id

        future = client.call_async(req)
        # servis yanıtını bekle
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self.node, timeout_sec=0.1)

        res = future.result()
        if res and res.success:
            self.node.get_logger().info(f"{node_key} → transition {transition_id} başarılı")
            return True
        else:
            self.node.get_logger().warn(f"{node_key} → transition {transition_id} başarısız")
            return False
