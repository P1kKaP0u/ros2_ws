import argparse
import glob
import os
import cv2
import numpy as np
import torch
from collections import deque
from ultralytics import YOLO

# TensorFlow-free DeepSORT alternative
class SimpleDeepSORT:
    def __init__(self, max_age=30, max_iou_distance=0.7):
        self.next_id = 0
        self.tracks = {}
        self.max_age = max_age
        self.max_iou_distance = max_iou_distance
        self.frame_count = 0
        
    def update(self, detections):
        """
        Update tracks with new detections
        detections: List of [x1, y1, x2, y2, confidence, class_id]
        """
        self.frame_count += 1
        
        # Get predicted locations from existing tracks
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[track_id]['bbox'] for track_id in track_ids]
        
        # Match detections to tracks using IoU
        matches, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(
            detections, track_boxes
        )
        
        # Update matched tracks
        for detection_idx, track_idx in matches:
            track_id = track_ids[track_idx]
            detection = detections[detection_idx]
            
            # Update track with new detection
            self.tracks[track_id]['bbox'] = detection[:4]
            self.tracks[track_id]['confidence'] = detection[4]
            self.tracks[track_id]['class_id'] = detection[5]
            self.tracks[track_id]['hits'] += 1
            self.tracks[track_id]['age'] = 0
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            self._create_new_track(detection)
        
        # Mark unmatched tracks
        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            self.tracks[track_id]['age'] += 1
            
            # Remove dead tracks
            if self.tracks[track_id]['age'] > self.max_age:
                self._remove_track(track_id)
        
        return self._get_active_tracks()
    
    def _associate_detections_to_tracks(self, detections, track_boxes):
        if len(track_boxes) == 0:
            return [], list(range(len(detections))), []
            
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(track_boxes)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(track_boxes):
                iou_matrix[d, t] = self._calculate_iou(det[:4], trk)
        
        # Find best matches
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_boxes)))
        
        # Find matches with IoU above threshold
        for d in range(len(detections)):
            best_iou = 0
            best_track = -1
            for t in range(len(track_boxes)):
                if iou_matrix[d, t] > best_iou and iou_matrix[d, t] > self.max_iou_distance:
                    best_iou = iou_matrix[d, t]
                    best_track = t
            
            if best_track != -1:
                matches.append((d, best_track))
                if d in unmatched_detections:
                    unmatched_detections.remove(d)
                if best_track in unmatched_tracks:
                    unmatched_tracks.remove(best_track)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _create_new_track(self, detection):
        track_id = self.next_id
        self.next_id += 1
        
        self.tracks[track_id] = {
            'bbox': detection[:4],
            'confidence': detection[4],
            'class_id': detection[5],
            'hits': 1,
            'age': 0
        }
    
    def _remove_track(self, track_id):
        if track_id in self.tracks:
            del self.tracks[track_id]
    
    def _get_active_tracks(self):
        active_tracks = []
        for track_id, track in self.tracks.items():
            if track['hits'] >= 3:  # Minimum hits to confirm track
                active_tracks.append({
                    'track_id': track_id,
                    'bbox': track['bbox'],
                    'confidence': track['confidence'],
                    'class_id': track['class_id']
                })
        return active_tracks
    
    @staticmethod
    def _calculate_iou(box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0


oldCenter = (0, 0)
currentList = []  # List of tuples: (cls, conf)
finalList = []

# Parameters
Y_DISTANCE_THRESHOLD = -25  # pixels, adjust as needed
CONFIDENCE_WEIGHTING = True  # toggle for confidence consideration

def changeCenter(center):
    global oldCenter
    dy = center[1] - oldCenter[1]
    if dy > Y_DISTANCE_THRESHOLD:
        oldCenter=center

def saveAnomaly(BestClasses):
    global oldCenter, currentList, finalList
    class_scores = {}
    class_conf_map = {}

    for c, cf in currentList:
        class_scores[c] = class_scores.get(c, 0) + cf
        if c not in class_conf_map:
            class_conf_map[c] = []
        class_conf_map[c].append(cf)

    best_cls = max(class_scores.items(), key=lambda x: x[1])[0]


    conf_list = class_conf_map[best_cls]
    avg_conf = sum(conf_list) / len(conf_list)
    if finalList:
        if avg_conf<finalList[-1][1] and finalList[-1][0]==best_cls:
            BestImg=False
    finalList.append([best_cls,avg_conf])
    print("Final Anomalies:", finalList)
    print("Current Sequence:", currentList)


    try:
        imgFinal=cv2.imread(f"/home/baurov/ros2_ws/src/anomaly_detection/detected/img_{best_cls}.jpg")
        for f in glob.glob("/home/baurov/ros2_ws/src/anomaly_detection/detected/*"):
            os.remove(f)
        cv2.imwrite(f"/home/baurov/ros2_ws/src/anomaly_detection/detected2/img_{best_cls}.jpg",imgFinal)
        if best_cls not in BestClasses:
            BestClasses.append(best_cls)
        if len(BestClasses)==5:
            return 1
    except:
        print("Sorry no image is good enough for my standarts :D")


    sorted_classes = sorted(finalList, key=lambda x: x[1], reverse=True)
    unique_classes = {}
    for cls, conf in sorted_classes:
        if cls not in unique_classes:
            unique_classes[cls] = conf
        if len(unique_classes) == 10:
            break
    for cls in unique_classes.keys():
        imgFinal=cv2.imread(f"/home/baurov/ros2_ws/src/anomaly_detection/detected2/img_{cls}.jpg")
        cv2.imwrite(f"/home/baurov/ros2_ws/src/anomaly_detection/detectedFinal/img_{cls}.jpg",imgFinal)



def matchAnomalies(center, cls, conf, out):
    global oldCenter, currentList, finalList
    
    # Calculate vertical movement
    dy = center[1] - oldCenter[1]
    BestImg=True
    if out:
        conf /= 3

    if dy > Y_DISTANCE_THRESHOLD:
        # Object moved significantly downward → continue current anomaly
        currentList.append((cls, conf))
    elif len(currentList)>5:
        # Small or negative movement → finalize current anomaly
        
        

        # Confidence-weighted vote
        class_scores = {}
        class_conf_map = {}

        for c, cf in currentList:
            class_scores[c] = class_scores.get(c, 0) + cf
            if c not in class_conf_map:
                class_conf_map[c] = []
            class_conf_map[c].append(cf)

        best_cls = max(class_scores.items(), key=lambda x: x[1])[0]


        conf_list = class_conf_map[best_cls]
        avg_conf = sum(conf_list) / len(conf_list)
        

        if finalList:
            if avg_conf<finalList[-1][1] and finalList[-1][0]==best_cls:
                BestImg=False
        finalList.append([best_cls,avg_conf])
        print("Final Anomalies:", finalList)
        print("Current Sequence:", currentList)
        print("Center diff:", center[1]-oldCenter[1])
        currentList = [(cls, conf)]  # Start new anomaly
        oldCenter = center
        if avg_conf<0.25:
            return False,False,"_"
        return True,BestImg,best_cls
    else:
        # First detection or noise
        currentList = [(cls, conf)]

    oldCenter = center

    print("Final Anomalies:", finalList)
    print("Current Sequence:", currentList)
    print("Center diff:", center[1]-oldCenter[1])
    return False,False,"_"
