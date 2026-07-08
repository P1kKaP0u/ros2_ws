import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox
from PyQt5.QtCore import QTimer


class RovModeManagerGui(QWidget):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.setWindowTitle("ROV Mode Manager GUI")

        # Layout ve widgetlar
        layout = QVBoxLayout()

        self.mode_label = QLabel("Select Mode:")
        layout.addWidget(self.mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['manual', 'stabilize', 'depth_hold', 'auto_mission'])
        layout.addWidget(self.mode_combo)

        self.arm_button = QPushButton("Arm")
        self.disarm_button = QPushButton("Disarm")
        layout.addWidget(self.arm_button)
        layout.addWidget(self.disarm_button)

        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

        # Buton sinyalleri
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        self.arm_button.clicked.connect(self.on_arm_clicked)
        self.disarm_button.clicked.connect(self.on_disarm_clicked)

        # Timer: rclpy spin_once ile event loop uyumu
        self.timer = QTimer()
        self.timer.timeout.connect(self.spin_ros)
        self.timer.start(10)  # 10 ms'de bir ROS döngüsü

    def spin_ros(self):
        rclpy.spin_once(self.ros_node, timeout_sec=0.001)

    def on_mode_changed(self, mode):
        self.ros_node.get_logger().info(f"Publishing mode: {mode}")
        msg = String()
        msg.data = mode
        self.ros_node.mode_pub.publish(msg)
        self.status_label.setText(f"Mode set to: {mode}")

    def on_arm_clicked(self):
        self.ros_node.get_logger().info("Publishing ARM command: True")
        self.ros_node.arm_pub.publish(Bool(data=True))
        self.status_label.setText("ARMED")

    def on_disarm_clicked(self):
        self.ros_node.get_logger().info("Publishing ARM command: False")
        self.ros_node.arm_pub.publish(Bool(data=False))
        self.status_label.setText("DISARMED")


class RosNode(Node):
    def __init__(self):
        super().__init__('rov_mode_manager_gui')
        self.mode_pub = self.create_publisher(String, 'set_mode', 10)
        self.arm_pub = self.create_publisher(Bool, 'set_arm', 10)


def main(args=None):
    rclpy.init(args=args)
    app = QApplication(sys.argv)

    ros_node = RosNode()
    gui = RovModeManagerGui(ros_node)
    gui.show()

    exit_code = app.exec_()

    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
