#!/usr/bin/env python3
import cv2
import numpy as np


def estimate_pipe_diameter(mask, box, minDia=0):
    """
    Estimate the pipe diameter side and orientation.
    Returns:
        turn angle: 0 or 180 or 90 (no action)
    """
    # Crop region of interest
    x1, y1 = map(int, map(max, zip(box[0], (0, 0))))
    x2, y2 = map(int, map(min, zip(box[2], mask.shape[::-1])))
    anomalyMask = mask[y1:y2, x1:x2]

    # Left vs right half
    mid_col = anomalyMask.shape[1] // 2
    left_strip = anomalyMask[:, :mid_col]
    right_strip = anomalyMask[:, mid_col:]
    left_th = np.mean(np.sum(left_strip == 255, axis=0)[np.sum(left_strip == 255, axis=0) > minDia]) if mid_col>0 else 0
    right_th = np.mean(np.sum(right_strip == 255, axis=0)[np.sum(right_strip == 255, axis=0) > minDia]) if mid_col>0 else 0
    turn = 180 if right_th > left_th else 0

    # Up vs down quarter
    quarter = anomalyMask.shape[0] // 4
    up_strip = anomalyMask[:quarter, :]
    down_strip = anomalyMask[quarter:quarter*2, :]
    up_th = np.mean(np.sum(up_strip == 255, axis=1)[np.sum(up_strip == 255, axis=1) > minDia]) if quarter>0 else 0
    down_th = np.mean(np.sum(down_strip == 255, axis=1)[np.sum(down_strip == 255, axis=1) > minDia]) if quarter>0 else 0

    # Determine if turn should be applied
    if down_th == 0 or up_th/down_th >= 2:
        return 90  # no significant vertical offset
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

    for cnt in contours:
        if cv2.contourArea(cnt) < 30:
            continue
        rect = cv2.minAreaRect(cnt)
        rectBound = cv2.boundingRect(cnt)
        (x, y), (w, h), angle = rect
        center = (int(x), int(y))
        if w < h:
            angle += 90
        angle = float(angle)

        # Optional depth filtering
        if depthOriginal is not None and max(w, h) > 0:
            depth = realDiameter * focalLength / max(w, h)
            if not (depthOriginal*0.9 <= depth <= depthOriginal*1.1):
                continue

        # Compute fullness ratio
        box = cv2.boxPoints(rect).astype(np.float32)
                # rectBound = [x, y, width, height]
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

        # Adjust angle if low fullness
        if fullness_ratio < 0.6:
            turn = estimate_pipe_diameter(mask, box2)
            if turn != 90:
                angle = turn
                
        angle=angle-90
        TanY=mask.shape[0]-center[1]
        TanX=-mask.shape[1]/2+center[0]
        angle2=np.tan(TanX/TanY)
        angle+=angle2 *10 
	
        dis = np.linalg.norm(np.array(center) - image_center)
        if fullness_ratio >= 0.2 and dis < disMin:
            selected = {"center": center, "size": (w,h), "angle": angle, "fullness": fullness_ratio, "contour":cnt}
            disMin = dis

    if selected is None:
        return None, [], 0.0, "No valid pipe found"

    confidence = min(1.0, selected["fullness"])
    return selected, [output, mask], confidence, True
