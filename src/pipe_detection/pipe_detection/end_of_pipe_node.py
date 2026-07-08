#!/usr/bin/env python3
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path

class EndOfPipeNode(Node):
    def __init__(self):
        super().__init__('end_of_pipe_node')

        self.bridge = CvBridge()
        self.pipe_end_detected = False
        self.node_active = False

        # Publisher & Subscriber
        self.sub_mask = self.create_subscription(Image, '/pipe_mask', self.mask_callback, 10)
        self.pub_pipe_end = self.create_publisher(Bool, '/pipe_end', 10)
        self.sub_mission_stop = self.create_subscription(Bool, '/mission_stop', self.mission_stop_callback, 10)

        # Frame kaydı için klasör
        default_save_folder = Path.home() / '.ros' / 'pipe_end_frames'
        self.declare_parameter('save_folder', str(default_save_folder))
        self.save_folder = Path(self.get_parameter('save_folder').value).expanduser()
        os.makedirs(self.save_folder, exist_ok=True)

        self.get_logger().info("EndOfPipeNode başlatıldı. Mission_stop bekleniyor...")

    def mission_stop_callback(self, msg):
        if msg.data and not self.node_active:
            self.node_active = True
            self.get_logger().warn("Anomali tespit edildi! Boru sonu tespiti aktif.")

    def is_pipe_end(self, binary_image, contour):
        x, y, w, h = cv2.boundingRect(contour)
        if y < binary_image.shape[0] // 2:
            return False

        if w * h < 100:
            return False
        
        top_empty = y > 0
        bottom_empty = y + h < binary_image.shape[0]
        left_empty = x > 0
        right_empty = x + w < binary_image.shape[1]
        return sum([top_empty, left_empty, right_empty]) >= 3

    def detect_pipe_end(self, cv_image):
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)
        binary = cv2.dilate(binary, kernel, iterations=2)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 100]
        if len(contours) != 1:
            return False
        for contour in contours:
            if self.is_pipe_end(binary, contour):
                return True
        return False

    def mask_callback(self, msg):
        try:
            if not self.node_active:
                return

            if self.pipe_end_detected:
                self.pub_pipe_end.publish(Bool(data=True))
                return

            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            is_end = self.detect_pipe_end(cv_image)

            if is_end:
                self.pipe_end_detected = True
                self.get_logger().warn("🔴 Borunun sonu bulundu! Artık hep True yayınlanacak.")
                self.pub_pipe_end.publish(Bool(data=True))

                # Frame kaydet
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = self.save_folder / f"pipe_end_{timestamp}.png"
                cv2.imwrite(str(save_path), cv_image)
                self.get_logger().info(f"Pipe end frame kaydedildi: {save_path}")
            else:
                self.pub_pipe_end.publish(Bool(data=False))

        except Exception as e:
            self.get_logger().error(f"Görüntü işlenirken hata: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = EndOfPipeNode()
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
