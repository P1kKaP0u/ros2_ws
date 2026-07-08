#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
from collections import defaultdict
#from anomalyPipeline import *  # aynı klasördeyse
from ultralytics import YOLO
import os
from .anomalyDetection import useYOLO


class AnomalyDetectionNode(Node):
    def __init__(self):
        super().__init__('anomaly_detection_node')
        self.bridge = CvBridge()
        self.detected = defaultdict(int)
        self.model = YOLO("/home/baurov/ros2_ws/src/anomaly_detection/models/bestAnomaly.pt")

        self.anomalyFound=[]
        self.last_frame = None
        self.last_mask = None

        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        self.create_subscription(Bool, '/pipe_end', self.pipe_end_callback, 10)
        self.mission_stop = self.create_publisher(Bool, '/mission_stop', 10)

        self.save_dir = "/home/baurov/ros2_ws/src/anomaly_detection/saved_images"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # ROS2 Publisher
        self.pub_bool = self.create_publisher(Bool, '/anomaly_detected', 10)
        self.pub_detection = self.create_publisher(String, '/anomaly_detections', 10)
        self.pub_annotated_image = self.create_publisher(Image, '/anomaly_image', 10)
        self.pipe_end = False
    

    def pipe_end_callback(self, msg:Bool):
        self.pipe_end = msg.data

    def image_callback(self, msg):
        self.last_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.try_detection()

    def try_detection(self):
        if self.last_frame is None:
            return
        #self.last_frame = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        detected_any,detected_all = useYOLO(self.last_frame, self.pipe_end)
        annotated_img = self.last_frame.copy()
        img_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
        self.pub_annotated_image.publish(img_msg)

        detected_bool = Bool()
        detected_bool.data = detected_any
        mission_bool = Bool()
        mission_bool.data = detected_all
        self.pub_bool.publish(detected_bool)
        self.mission_stop.publish(mission_bool)

def main(args=None):
    rclpy.init(args=args)
    node = AnomalyDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
