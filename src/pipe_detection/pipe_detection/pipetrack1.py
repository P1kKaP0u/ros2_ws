	#!/usr/bin/env python3
import cv2
import numpy as np
from .skeletonizeMask import skeleton_longest_path




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






def estimate_pipe_diameter(mask, box, croppedMask, minDia=0):
    """
    Estimate the pipe diameter side and orientation.
    Returns:
        turn angle: 0 or 180 or 90 (no action)
    """
    # Crop region of interest
    x1, y1 = map(int, map(max, zip(box[0], (0, 0))))
    x2, y2 = map(int, map(min, zip(box[2], mask.shape[::-1])))
    #y1=max(y1,int(mask.shape[0]/2))
    anomalyMask = mask[y1:y2, x1:x2]

    # Left vs right half
    mid_col = anomalyMask.shape[1] // 2
    left_strip = anomalyMask[:, :mid_col]
    right_strip = anomalyMask[:, mid_col:]
    left_th = np.mean(np.sum(left_strip == 255, axis=0)[np.sum(left_strip == 255, axis=0) > minDia]) if mid_col>0 else 0
    right_th = np.mean(np.sum(right_strip == 255, axis=0)[np.sum(right_strip == 255, axis=0) > minDia]) if mid_col>0 else 0
    turn = 0 if right_th > left_th else 180

    # Up vs down quarter
    quarter = anomalyMask.shape[0] // 8
    up_strip = anomalyMask[:quarter, :]
    down_strip = anomalyMask[quarter:quarter*2, :]
    downest_strip=anomalyMask[anomalyMask.shape[0]//2:, :]
    up_th = np.mean(np.sum(up_strip == 255, axis=1)[np.sum(up_strip == 255, axis=1) > minDia]) if quarter>0 else 0
    down_th = np.mean(np.sum(down_strip == 255, axis=1)[np.sum(down_strip == 255, axis=1) > minDia]) if quarter>0 else 0
    downest_th = np.mean(np.sum(downest_strip == 255, axis=1)[np.sum(downest_strip == 255, axis=1) > minDia]) if quarter>0 else 0

    try:
        anomalyMask=anomalyMask[int(anomalyMask[0]/4):int(anomalyMask[0]*3/4),:]
        contours, _ = cv2.findContours(anomalyMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        (center), (w, h), angle = rect
        if w <h:
            angle+=90
        TanY=anomalyMask.shape[0]/2
        TanX=-anomalyMask.shape[1]/2+center[0]
        angle2=np.rad2deg(np.arctan(TanX/TanY))
        angle+=angle2
    except:
        angle=90

    # Determine if turn should be applied
    if down_th == 0 or up_th/down_th >= 2 or up_th/downest_th<=2:
        return angle  # no significant vertical offset
    return turn


def trackPipeAndControl(mask, depthOriginal=None, old_center=(144, 144), focalLength=144, realDiameter=125):
    """
    Perform pipe tracking based on a binary mask.
    Args:
        mask (2D array): binary mask of pipe region (0 or 255)
        depthOriginal, old_center, focalLength, realDiameter: optional parameters for 3D filtering
    Returns:
        selected dict or None, visualization list, confidence, status
    """

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
    maxAngle=0
    cnt=max(contours, key=cv2.contourArea)
    if 1:
        # if cv2.contourArea(cnt) < 500:
        #     continue

        turnDetected=False
        maxDetected=False
        rect = cv2.minAreaRect(cnt)
        rectBound = cv2.boundingRect(cnt)
        (x, y), (w, h), angle = rect
        center = (int(x), int(y))
        dia=h
        if w < h:
            angle += 90
            dia=w
        angle = float(angle)
        #if abs(angle-90)>abs(maxAngle-90):
        #    maxAngle=angle

        # Optional depth filtering
        # if depthOriginal is not None and max(w, h) > 0:
        #     depth = realDiameter * focalLength / max(w, h)
        #     if not (depthOriginal*0.9 <= depth <= depthOriginal*1.1):
        #         continue

        # Compute fullness ratio
        box = cv2.boxPoints(rect).astype(np.float32) #rect->rect2
                # rectBound = [x, y, width, height]
        (x2,y2,w2,h2) = (x, y,w,h)
        x, y, w, h = rectBound
        box2 = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.int32)
        dst = np.array([[0,0], [rectBound[2]-1, 0], [rectBound[2]-1, rectBound[3]-1], [0, rectBound[3]-1]], np.float32)
        M = cv2.getPerspectiveTransform(box, dst)
        crop = cv2.warpPerspective(mask, M, (rectBound[2], rectBound[3]))
        fullness = np.sum(crop)/255
        fullness_ratio = fullness/(rectBound[2]*rectBound[3] + 1e-6)


        if angle<25 and x2>mask.shape[1]/2:
            angle+=180
        elif angle > 155 and x2<mask.shape[1]/2:
            angle-=180
        else:
            TanY=mask.shape[0]/2
            TanX=-mask.shape[1]/2+center[0]
            angle2=np.rad2deg(np.arctan(TanX/TanY))
            angle+=angle2

                # Adjust angle if low fullness
        # if fullness_ratio < 0.6:
        #     #angle = estimate_pipe_diameter(mask, box2, crop)
        #     turnDetected=True
        if fullness_ratio < 0.45:
            angleEs = estimate_pipe_diameter(mask, box2, crop)
            def analyze_mask_quadrants(mask, threshold=0.6):
                """
                Splits mask into 3x3 grid and checks % fullness in:S
                - bottom-left
                - bottom-right
                - middle-left
                - middle-right

                Returns dict of which regions are > threshold (default 60%)
                """
                h, w = mask.shape
                h_third = h // 6
                w_third = w // 3

                # Define 4 regions of interest
                regions = {
                    'bottom_left': mask[2*h_third:h, 0:w_third],
                    'bottom_right': mask[2*h_third:h, 2*w_third:w],
                    'middle_left': mask[h_third:5*h_third, 0:w_third],
                    'center': mask[h_third:5*h_third, w_third:2*w_third],
                    'middle_right': mask[h_third:5*h_third, 2*w_third:w],
                }

                results = {}

                for name, region in regions.items():
                    total_pixels = region.size
                    white_pixels = np.count_nonzero(region == 255)
                    fullness = white_pixels / total_pixels
                    results[name] = fullness

                if results["middle_right"]>results["middle_left"] and results["middle_right"]>results["center"]:# or results["bottom_right"]:
                    return 180
                elif results["middle_left"]>results["center"]:# or results["bottom_left"]:
                    return 0
                else:
                    return 90
                

            # angle=analyze_mask_quadrants(mask)
            # if angleEs == 90:
            #     x, y, w, h = cv2.boundingRect(mask)
            #     pipeMask = mask[y:y+h, x:x+w]

            #     # angles, split_val, axis, centers = binarySplitter(mask)
            #     angles, split_val, axis, centers =binarySplitter(pipeMask)
            #     print(angles)
            #     if axis=="y":
            #         if centers[0][1]+y>mask.shape[0]/2:
            #             angle=angles[0]
            #             centerMain=centers[0]
            #         else:
            #             angle=angles[1]
            #             centerMain=centers[1]
            #     else:
            #         if centers[0][1]<centers[1][1] and mask.shape[0]/2<centers[1][1]+y:
            #             angle=angles[1]
            #             centerMain=centers[1]
            #         else:
            #             angle=angles[0]
            #             centerMain=centers[0]

            #     TanY=mask.shape[0]/2
            #     TanX=-mask.shape[1]/2+centerMain[0]+x
            #     angle2=np.rad2deg(np.arctan(TanX/TanY))

            #     if angle<25 and x+centerMain[0]>mask.shape[1]/2:
            #         angle+=180
            #     elif angle > 155 and x+centerMain[0]<mask.shape[1]/2:
            #         angle-=180
            #     else:
            #         angle+=angle2 /1.5
            #     angle=90
            # else:
            #     angle=angleEs
            #     turnDetected=True
            
            

            # turn = 90
            # if turn != 90:
            #    angle = turn

        angle=angle-90


        angle=max(-90,min(angle,90))
        # if abs(angle-90)>abs(maxAngle-90) and center[1]>mask.shape[0]/4:
        #     maxAngle=angle

        dis = np.linalg.norm(np.array(center) - image_center)
        if fullness_ratio >= 0.1: #and dis < disMin and y2<mask.shape[0]*4/5:
            if not maxAngle:
                maxAngle=angle
            else:
                maxDetected=True
            selected = {"center": center,"box":box.astype(np.int32), "size": (w2,h2), "angle": maxAngle, "fullness": fullness_ratio, "contour":cnt, "flags":(turnDetected,maxDetected)}
            disMin = dis

    if selected is None and len(contours):
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        (x,y), (w,h), angle = rect
        if w < h:
            angle += 90


        if angle<25 and x2>mask.shape[1]/2:
            angle+=180
        elif angle > 155 and x2<mask.shape[1]/2:
            angle-=180

        angle-=90
        angle=max(-90,min(angle,90))

        center = (int(x), int(y))
        box=cv2.boxPoints(rect).astype(np.int32)
        selected = {"center": center,"box":box, "size": (w,h), "angle": angle, "fullness": 0.8, "contour":cnt, "vis":output,"flags":(False,False)
        }

    if selected is None:
        return None, [], 0.0, "No valid pipe found"

    confidence = min(1.0, selected["fullness"])
    return selected, [output, mask], confidence, True
