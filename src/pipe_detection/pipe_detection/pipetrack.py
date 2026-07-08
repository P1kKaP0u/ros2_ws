	#!/usr/bin/env python3
from collections import deque
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pipeTurnControl import  update_and_check_next

def getFullness(cnt, mask):
    # Get the rotated rectangle around the contour
    rect = cv2.minAreaRect(cnt)  # (center(x,y), (width, height), angle)
    box = cv2.boxPoints(rect).astype(np.float32)  # Get 4 corner points of rotated rect

    # Compute destination points (a straight rectangle)
    width = int(rect[1][0])
    height = int(rect[1][1])

    # Handle cases where width or height might be 0
    if width == 0 or height == 0:
        return 0.0

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # Perspective transform to un-rotate the rectangle
    M = cv2.getPerspectiveTransform(box, dst)
    crop = cv2.warpPerspective(mask, M, (width, height))

    # Calculate how much of the rect is filled
    fullness = np.sum(crop > 0)  # count non-zero pixels
    fullness_ratio = fullness / (width * height + 1e-6)

    return fullness_ratio

def binarySplitter(mask, box=None):
    def safeFindContours(region):
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours if contours else []

    def getAnglesAndFullness(region):
        contours = safeFindContours(region)
        if not contours:
            return 0.0, 0.0
        cnt = np.vstack(contours)
        fullness = getFullness(cnt, region)
        center, (w, h), angle = cv2.minAreaRect(cnt)
        if w < h:
            angle += 90
        return fullness, angle, center

    def splitAndEvaluate(anomalyMask, axis='y'):
        h, w = anomalyMask.shape
        a = h // 2 if axis == 'y' else w // 2
        step = (h if axis == 'y' else w) / 4
        best_diff = float('inf')
        best_angles = [0, 0]
        best_centers = [0, 0]
        best_split = a
        up = False

        def getRegions(split_val):
            if axis == 'y':
                return anomalyMask[:split_val, :], anomalyMask[split_val:, :]
            else:
                return anomalyMask[:, :split_val], anomalyMask[:, split_val:]

        region1, region2 = getRegions(a)
        f1, a1, c1 = getAnglesAndFullness(region1)
        f2, a2, c2 = getAnglesAndFullness(region2)
        diff = abs(2-f1 - f2)

        if diff < best_diff:
            best_diff = diff
            best_angles = [a1, a2]
            best_centers=[c1,c2]
            best_split = a

        up = f1 < f2

        while step > 2:
            a = a - int(step) if up else a + int(step)
            a = max(1, min((h if axis == 'y' else w) - 1, a))
            step /= 2

            region1, region2 = getRegions(a)
            f1, a1, c1 = getAnglesAndFullness(region1)
            f2, a2, c2 = getAnglesAndFullness(region2)
            diff = abs(2-f1 - f2)

            if diff < best_diff:
                best_diff = diff
                best_angles = [a1, a2]
                best_centers=[c1,c2]
                best_split = a

            up = f1 < f2
            # print([f1, int(a1), c1,"ssss", f2, int(a2), c2])
        return best_diff, best_angles, best_split, best_centers

    # Crop region if box is provided
    if box is not None:
        x1, y1 = map(int, map(max, zip(box[0], (0, 0))))
        x2, y2 = map(int, map(min, zip(box[2], mask.shape[::-1])))
        anomalyMask = mask[y1:y2, x1:x2]
    else:
        anomalyMask = mask

    # Try splitting in both directions
    y_diff, y_angles, y_split, y_center = splitAndEvaluate(anomalyMask, axis='y')
    x_diff, x_angles, x_split, x_center = splitAndEvaluate(anomalyMask, axis='x')

    # Return the best one
    if y_diff <= x_diff:
        return y_angles, y_split, 'y', y_center
    else:
        return x_angles, x_split, 'x', x_center

def normalize_angle(angle):
    """Normalize angle to [0,360]"""
    return angle%360

startM=240
# pipeMagnetic=[startM,startM+np.pi/2,startM+np.pi,startM-np.pi/2,startM+np.pi,startM+np.pi/2,startM+np.pi]
# pipeMagnetic=[normalize_angle(angle) for angle in pipeMagnetic]
pipeMagnetics=[startM,startM+90,startM+180,startM+270,startM+180,startM+90,startM+180]
pipeMagnetics=[normalize_angle(angle) for angle in pipeMagnetics]
pipeMagneticNum=0
memory=deque(maxlen=40)

def trackPipeAndControl(mask,degree, depthOriginal=None, old_center=(144, 144), focalLength=144, realDiameter=125):
    """
    Perform pipe tracking based on a binary mask.
    Args:
        mask (2D array): binary mask of pipe region (0 or 255)
        depthOriginal, old_center, focalLength, realDiameter: optional parameters for 3D filtering
    Returns:
        selected dict or None, visualization list, confidence, status
    """
    global pipeMagneticNum

    kernel=np.ones((10,10),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    #mask=cv2.dilate(mask,kernel,iterations=1)
    
    #mask=mask[int(mask.shape[0]/2):,:]
    # Ensure binary
    if mask.ndim != 2 or mask.dtype != np.uint8:
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, [], 0.0, "No contours"

    output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    image_center = np.array([mask.shape[1]/2, mask.shape[0]])
    disMin = np.inf
    selected = None
    #a,b=np.where(mask > 0)
    #points=np.column_stack((b,a))
    maxAngle=0
    conf=0
    #rect2=cv2.minAreaRect(points)
    for cnt in contours:
        if cv2.contourArea(cnt) < 2000:
            continue
        
        #flags
        turnDetected=False
        maxDetected=False

        #take inside a rectangle
        rect = cv2.minAreaRect(cnt)
        rectBound = cv2.boundingRect(cnt)
        (x, y), (w, h), angle = rect
        center = (int(x), int(y))
        dia=h
        if w < h:
            angle += 90
            dia=w
        angle = float(angle)
        
        # Optional depth filtering
        if depthOriginal is not None and max(w, h) > 0:
            depth = realDiameter * focalLength / max(w, h)
            if not (depthOriginal*0.9 <= depth <= depthOriginal*1.1):
                continue

        fullness_ratio = getFullness(cnt,mask)

        # Merkeze uzaklık açı ile kapatma
        TanY=mask.shape[0]/2
        TanX=-mask.shape[1]/2+center[0]
        angle2=np.rad2deg(np.arctan(TanX/TanY))


        #borularda sağ/sol/düz ayrımı
        if angle<25 and x>mask.shape[1]/2:
            angle+=180
        elif angle > 155 and x<mask.shape[1]/2:
            angle-=180
        else:
            angle+=angle2 /1.5
        angle=angle-90
        
        
        #Compass fusion
        angleCompass=pipeMagnetics[pipeMagneticNum]-degree + angle2/1.5
        if angleCompass-angle2/1.5>150 or angleCompass-angle2/1.5<-150:
            conf=0
        else:
            conf=min(1,(w*h/(mask.shape[0]*mask.shape[1]/20))*fullness_ratio)
        angle=(1-conf)*angleCompass+conf*angle


        # Adjust angle if low fullness
        if fullness_ratio < 0.6:
            # angle = estimate_pipe_diameter(mask, box2, crop)
            
            x, y, w, h = cv2.boundingRect(mask)
            pipeMask = mask[y:y+h, x:x+w]

            # angles, split_val, axis, centers = binarySplitter(mask)
            angles, split_val, axis, centers =binarySplitter(pipeMask)
            print(angles)
            bestSuit=360
            centerY=0
            for i in [0,1]:
                angle=angles[i]-90
                diff=normalize_angle(pipeMagnetics[pipeMagneticNum+1]-degree)
                #if diff<15:
                #    pipeMagneticNum+=1
                if abs(normalize_angle(angle-diff))<bestSuit:
                    bestSuit=angle
                    centerY=centers[i][1]
                elif abs(normalize_angle(angle+180-diff)):
                    bestSuit=angle+180
                    centerY=centers[i][1]
            
            if centerY+y>mask.shape[0]/2:
                angle=bestSuit
            else:
                angle=0
            print("angleLast",angle)

            
            turnDetected=True



        angle=max(-90,min(angle,90))
        if abs(angle-90)>abs(maxAngle-90) and center[1]>mask.shape[0]/2:
            maxAngle=angle



        memory.append(degree)
        pipeMagneticNum,turned=update_and_check_next(memory,pipeMagnetics,pipeMagneticNum,min_len=20,tol_deg=14,std_tol_deg=10,max_omega_deg_s=10)
        if turned: #and len(memory)>60:
            memory.clear()
            #pipeMagneticNum+=1


        dis = np.linalg.norm(np.array(center) - image_center)
        if fullness_ratio >= 0.2 and dis < disMin and y<mask.shape[0]*4/5:
            if not maxAngle:
                maxAngle=angle
            else:
                maxDetected=True
            selected = {"center": center,"box":cv2.boxPoints(rect).astype(np.int32), "size": (w,h), "angle": maxAngle, "fullness": fullness_ratio, "contour":cnt, "flags":(turnDetected,maxDetected),"pipeNum":pipeMagneticNum}
            disMin = dis

    if selected is None and len(contours):
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        (x,y), (w,h), angle = rect
        center = (int(x), int(y))
        box=cv2.boxPoints(rect).astype(np.int32)
        selected = {"center": center,"box":box, "size": (w,h), "angle": angle, "fullness": 0.8, "contour":cnt, "vis":output,"flags":(False,False),"pipeNum":pipeMagneticNum
        }

    if selected is None:
        return None, [], 0.0, "No valid pipe found"

    # confidence = min(1.0, selected["fullness"])
    return selected, [output, mask], conf, True

