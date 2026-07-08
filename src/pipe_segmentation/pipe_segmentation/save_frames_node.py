#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class FrameSaverNode(Node):
    def __init__(self):
        super().__init__('frame_saver_node')

        # Fotoğrafların kaydedileceği klasör
        self.output_dir = os.path.expanduser('/home/baurov/frames_for_model/')
        os.makedirs(self.output_dir, exist_ok=True)

        # Frame sayacı
        self.frame_count = 0

        # CvBridge objesi
        self.bridge = CvBridge()

        # Subscriber
        self.subscription = self.create_subscription(
            Image,
            '/pipe_mask',  # Eğer farklı bir topic kullanıyorsan burayı değiştir
            self.listener_callback,
            10
        )

        self.get_logger().info(f"Frame Saver Node Başladı. Fotoğraflar '{self.output_dir}' içine kaydedilecek.")

    def listener_callback(self, msg):
        try:
            # ROS Image → OpenCV formatına çevir
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Dosya adı oluştur
            filename = os.path.join(self.output_dir, f"frame_{self.frame_count:05d}.jpg")

            # Görüntüyü kaydet
            cv2.imwrite(filename, cv_image)

            self.get_logger().info(f"{filename} kaydedildi.")
            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"Görüntü işlenemedi: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FrameSaverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
