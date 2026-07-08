#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge
import cv2
import numpy as np
import os, sys
from std_msgs.msg import Bool, Float64
from .pipetrack1 import trackPipeAndControl

class PipeDetectionNode(Node):
    def __init__(self):
        super().__init__('pipe_detection_node')

        # CvBridge
        self.bridge = CvBridge()

        # Publisher: boru pozisyon hatası
        self.pose_pub = self.create_publisher(Pose2D, '/pipe_pose_error', 10)

        #Publisher: is pipe visible
        self.pipe_vis_pub = self.create_publisher(Bool, '/pipe_visible', 10)

        # Publisher: boru bounding boxed görseli
        self.annotated_pub = self.create_publisher(Image, '/bounding_boxed_pipe', 1)

        # Subscriber: boru maske görüntüscü
        self.mask_sub = self.create_subscription(Image, '/pipe_mask', self.mask_callback, 1)

        self.get_logger().info("PipeDetectionNode başlatıldı.")
        self.magnometer_data = 0.0
    def magnometer_data_callback(self, msg: Float64 ):
        self.magnometer_data = msg.data

    def mask_callback(self, msg):
        try:
            cv_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f"CvBridge mask dönüşüm hatası: {e}")
            return

        mask = cv_mask

        # try:
        selected, _, confidence, status = trackPipeAndControl(mask=mask)
        # except Exception as e:
        #     self.get_logger().error(f"trackPipeAndControl hatası: {e}")
        #     return

        if selected is None:
            self.get_logger().warn(f"Boru bulunamadı: {status}")
            visible = Bool()
            visible.data = False
            self.pipe_vis_pub.publish(visible)
            return 

        visible = Bool()
        visible.data = True
        cx, cy = selected['center']
        h, w = mask.shape
        img_cx, img_cy = w/2.0 , h/2.0
        error_x = (cx- img_cx) / img_cx
        error_y = (img_cy - cy) / img_cy
        error_theta = np.deg2rad(selected['angle'])
        flags1, flags2 = selected['flags']
        #pipeNum = selected['pipeNum']
        pose_msg = Pose2D()
        pose_msg.x= float(error_x)
        pose_msg.y = float(error_y)
        pose_msg.theta = float(error_theta)
        self.pose_pub.publish(pose_msg)
        self.pipe_vis_pub.publish(visible)
        w, h = selected["size"]
        self.get_logger().info(
            f"PipePose →  x:{pose_msg.x:.3f}, y:{pose_msg.y:.3f}, θ:{pose_msg.theta:.3f}, conf:{confidence:.2f}, flags:{(flags1,flags2)}"
        )

        # ---- Bounding Box Görselleştirme ve Yayınlama ----
        vis_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Seçilen borunun etrafına kutu çiz
        x, y, w_box, h_box = cv2.boundingRect(selected['contour'])
        #y += int(mask.shape[0] / 2)
        cv2.rectangle(vis_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        #cy += int(mask.shape[0] / 2)
        cv2.circle(vis_img, (int(cx), int(cy)), 4, (255, 0, 0), -1)
        cv2.drawContours(vis_img,[selected["box"]],0,(0,0,255),2)

        center=selected["center"]
        radians = error_theta
        length = 50  # length of the angle line

        # Compute end point based on angle
        end_x = int(center[0] + length * np.cos(radians))
        end_y = int(center[1] + length * np.sin(radians))

        # Draw the angle line
        cv2.arrowedLine(vis_img, center, (end_x, end_y), (255, 0, 0), 2, tipLength=0.3)

        # Görseli publish et
        try:
            img_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
            self.annotated_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"Görsel publish hatası: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PipeDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
