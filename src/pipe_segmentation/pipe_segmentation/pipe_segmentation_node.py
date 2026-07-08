#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

class PipeSegmentationNode(Node):
    def __init__(self):
        super().__init__('pipe_segmentation_node')

        # Model konfigürasyonu
        self.declare_parameter('model_path', '/home/baurov/ros2_ws/src/pipe_segmentation/models/best_model_with_anomaly_vs2.0.pth')
        self.declare_parameter('encoder', 'resnet34')
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('mask_topic', '/pipe_mask')

        model_path = self.get_parameter('model_path').value
        encoder = self.get_parameter('encoder').value
        image_topic = self.get_parameter('image_topic').value
        mask_topic = self.get_parameter('mask_topic').value

        # CvBridge
        self.bridge = CvBridge()

        # Segmentation modeli
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = smp.Segformer(
            encoder_name="mit_b0",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Preprocessing transform
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])  

        # Publisher: mask topic
        self.mask_pub = self.create_publisher(Image, mask_topic, 10)

        # Subscriber: camera image
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            1
        )

    def image_callback(self, msg: Image):
        # ROS Image -> OpenCV BGR
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge dönüşüm hatası: {e}")
            return

        # Preprocess
        h0, w0 = cv_img.shape[:2]
        
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        #cv_img = clahe.apply(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY))
        #cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=cv_img)
        tensor = augmented['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds_logits  = self.model(tensor)
            pred = torch.sigmoid(preds_logits) 
            mask = (pred[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255

        # Orijinal boyuta geri ölçekle
        mask_resized = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)

        # ROS Image olarak yayınla
        mask_msg = self.bridge.cv2_to_imgmsg(mask_resized, encoding='mono8')
        mask_msg.header = msg.header
        self.mask_pub.publish(mask_msg)

        self.get_logger().debug("Pipe mask published.")


def main(args=None):
    rclpy.init(args=args)
    node = PipeSegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
