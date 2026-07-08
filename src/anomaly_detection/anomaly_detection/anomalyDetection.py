import numpy as np
import cv2
from .anomalyPipeline import *
from ultralytics import YOLO
import glob
import os
from .anomaly_segmentation import segment_anomaly
from .anomalyTracker import *

anomaly={#class:[conf,center,OutsideOfTheFrame?]
        "ucgen":[0.0,(0,0),True],
        "yildiz":[0.0,(0,0),True],
        "kare":[0.0,(0,0),True],
        "elips":[0.0,(0,0),True],
        "trombus":[0.0,(0,0),True],
        "besgen":[0.0,(0,0),True],
        "dort yaprakli yonca":[0.0,(0,0),True],
        "altigen":[0.0,(0,0),True],
        "dikdortgen":[0.0,(0,0),True],
        "daire":[0.0,(0,0),True],
    }
model=YOLO("/home/baurov/ros2_ws/src/anomaly_detection/models/best.pt")
 
def angle_between(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculateAngles(approx,side):
    angles = []
    for i in range(side):
        p1 = approx[i % side][0]
        p2 = approx[(i+1) % side][0]
        p3 = approx[(i+2) % side][0]
        angle = angle_between(p1, p2, p3)
        angles.append(angle)
    return angles

def shapeControl(mask,cls):
    fov_x,fov_y=80,64
    firstClass=cls
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    cnt=max(contours, key=cv2.contourArea)
    center,size,angle=cv2.minAreaRect(cnt)
    
    # print("fullness",getFullness(cnt,mask))
    deg, rad = angle_between_camera_rays((mask.shape[1]//2,center[1]), (mask.shape[1]//2,mask.shape[0]//2), size[0],size[1], fov_x, fov_y)
    print(deg)

    print("size",size)
    size=(min(size[0],size[1]),max(size[0],size[1]))
    if size[0]*1.15<size[1]:
        if cls=="daire":
            cls="elips"
    else:
        if cls=="elips":
            cls="daire"

    if size[0]*1.25<size[1]:
        if cls=="kare":
            cls="dikdortgen"
    else:
        if cls=="dikdortgen":
            cls="kare"

    if size[0]*1.75<size[1] and cls=="trombus":
        cls="dikdortgen"

   

    # side count
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape=None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(cnt, True)  # 0.2% of perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Get the center of shape
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # Determine the shape
            sides = len(approx)
            angles=calculateAngles(approx,sides)
            shape = "Unknown"
            (x, y, w, h) = cv2.boundingRect(approx)
            # if sides == 4 and cls=="besgen":
            #     cls="trombus"
            angle_std="Not Calculated"
            print("sides",sides)
            print(x,y,w,h)
            if sides == 3:
                shape = "Triangle"
            if sides == 4:
                # Calculate side lengths
                side_lengths = []
                for i in range(4):
                    pt1 = approx[i][0]
                    pt2 = approx[(i+1)%4][0]
                    length = np.linalg.norm(pt1 - pt2)
                    side_lengths.append(length)
                
                mean_side = np.mean(side_lengths)
                side_std = np.std(side_lengths)

                mean_angle = np.mean(angles)
                angle_std=np.std(angles)


                ratio=max(side_lengths)/min(side_lengths)
                # Now classify
                #   if side_std < 0.1 * mean_side:
                if angle_std>11 and cls=="kare":
                    cls = "trombus"
                elif angle_std<11 and cls=="trombus":
                    cls = "kare"
                # if ratio<1.4 and cls=="kare":
                #     cls="dikdortgen"
                    
            elif sides == 5:
                shape = "Pentagon"
            elif sides == 6:
                shape = "Hexagon"
            else:
                shape = "Circle"
            print("angle_std",angle_std)
    if firstClass != cls:
        pass
    return cls

# [['dikdortgen', 0.15034149090449014], ['dikdortgen', 0.6737714807192484], ['yildiz', 0.48524716744820273], ['trombus', 0.23242730529684774], ['kare', 0.34112844119469327], ['yildiz', 0.4730530281861623], ['ucgen', 0.5267777424305677], ['ucgen', 0.4455367810196346], ['yildiz', 0.17529207799169752], ['yildiz', 0.14256733655929565], ['yildiz', 0.41436062095401527], ['trombus', 0.5293546224182303], ['trombus', 0.5608799186619845], ['dikdortgen', 0.12176628907521565], ['dikdortgen', 0.11479483048121134], ['dikdortgen', 0.22911288340886435], ['altigen', 0.6824314008499013], ['dort yaprakli yonca', 0.6100874889464605], ['trombus', 0.4348560697757281], ['trombus', 0.6336669033648922], ['dikdortgen', 0.11217976609865825], ['dikdortgen', 0.19555217027664185], ['dikdortgen', 0.2174395521481832], ['dikdortgen', 0.2299757202466329], ['dikdortgen', 0.1703141580025355], ['dikdortgen', 0.24518832564353943], ['elips', 0.6258794138597888]]


def updateTracker(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt=np.vstack(contours)
    area = cv2.contourArea(cnt)
    if area < 2500:
        return None
    center,size,angle=cv2.minAreaRect(cnt)
    changeCenter(center)



def isCloser(center,centerOld,img):
    return abs(centerOld[0]-img.shape[1]//2)+abs(centerOld[1]-img.shape[0]//2)>abs(center[0]-img.shape[1]//2)+abs(center[1]-img.shape[0]//2)


sapma=10
bestClasses=[]
detected_all = False
counterFrame=0
#YOLO output: bbox, tag
def useYOLO(img,done):
    global bestClasses,detected_all,anomaly,counterFrame
    # st=datetime.now()
    detected_any = False

    if done:
        saveAnomaly(bestClasses)

    mask=segment_anomaly(img)
    updateTracker(mask)
    results=model.predict(img)[0]

    h, w = img.shape[:2]
    cx_img, cy_img = w // 2, h // 2  # image center

    closest_box = None
    min_dist = float("inf")

    for box in results.boxes:
        x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
        cx_box, cy_box = (x1 + x2) // 2, (y1 + y2) // 2  # box center

        dist = (cx_box - cx_img) * 2 + (cy_box - cy_img) * 2  # squared distance
        if dist < min_dist:
            min_dist = dist
            closest_box = (x1, y1, x2, y2)


    if not results.boxes:
        counterFrame += 1
        # if counterFrame==10:
        #     matchAnomalies((0,0),"kare",0.0,1)
        return False, False
    else:
        counterFrame=0
    for box in results.boxes:
        x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        cls=model.names[cls]
        if closest_box != (x1, y1, x2, y2):
            print("continue")
            continue
        #print(cls)
        # print(cls)
        # print("loop",datetime.now()-st)#0.15
        
        image=np.copy(img)
        if conf>0.4:
            detected_any = True
            # print("confident",conf)
            #st=datetime.now()
            # anomalyDetect(img,pipeMask,[x1, y1, x2, y2],cls)
            cls=shapeControl(mask,cls)
            if cls is None:
                continue
            #print(datetime.now()-st)

            #outside
            cond1=(x1<sapma or x2>img.shape[1]-sapma)
            cond2=(y1<sapma or y2>img.shape[0]-sapma)
            if cond1 or cond2:
                outSide=True
            else:
                outSide=False

            # if not outSide:
            ok,BestImg,BestClass=matchAnomalies(((x1+x2)//2,(y1+y2)//2),cls,conf,outSide) 
            # else:
            #     BestImg=False

            # Draw box and label
            label = f"{cls} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, (y1+y2)//2), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 3)
            image=cv2.resize(image,(image.shape[1]//2,image.shape[0]//2))


            
            if BestImg:
                try:
                    imgFinal=cv2.imread(f"/home/baurov/ros2_ws/src/anomaly_detection/detected/img_{BestClass}.jpg")
                    for f in glob.glob("/home/baurov/ros2_ws/src/anomaly_detection/detected/*"):
                        os.remove(f)
                    cv2.imwrite(f"/home/baurov/ros2_ws/src/anomaly_detection/detected2/img_{BestClass}.jpg",imgFinal)
                    if BestClass not in bestClasses:
                        bestClasses.append(BestClass)
                    if len(bestClasses)>=7:
                        detected_all=True
                except:
                    print("Sorry no image is good enough for my standarts :D")

            confOld,centerOld,out=anomaly[cls]
            if (not out) and (not outSide) and confOld<conf:
                anomaly[cls]=[conf,((x1+x2)//2,(y1+y2)//2),outSide]
                cv2.imwrite(f"/home/baurov/ros2_ws/src/anomaly_detection/detected/img_{cls}.jpg",image)
                cv2.imwrite(f"/home/baurov/ros2_ws/src/anomaly_detection/detectedTemp/img_{cls}.jpg",image)
            elif out and (not outSide or isCloser(((x1+x2)//2,(y1+y2)//2),centerOld,img) or confOld<conf):
                anomaly[cls]=[conf,((x1+x2)//2,(y1+y2)//2),outSide]
                cv2.imwrite(f"/home/baurov/ros2_ws/src/anomaly_detection/detected/img_{cls}.jpg",image)
                cv2.imwrite(f"/home/baurov/ros2_ws/src/anomaly_detection/detectedTemp/img_{cls}.jpg",image)


    return detected_any,detected_all