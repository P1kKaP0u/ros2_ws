import numpy as np
import cv2
from collections import deque
from skimage.morphology import skeletonize

def distance_between_nodes(node1, node2, coords):
    r1, c1 = coords[node1]
    r2, c2 = coords[node2]
    return np.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2)

def skeleton_longest_path(maskIn, radius=10):
    # Get skeleton and shape of the input mask
    shape = maskIn.shape
    skel = skeletonize(maskIn)
    
    # Get coordinates of non-zero pixels (skeleton points)
    coords = np.column_stack(np.where(skel > 0))
    idx = { (r, c): i for i, (r, c) in enumerate(map(tuple, coords)) }
    
    # Build adjacency list based on 8-connectivity
    adj = [[] for _ in coords]
    for i, (r, c) in enumerate(coords):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if (rr, cc) in idx:
                    adj[i].append(idx[(rr, cc)])
    
    # Find endpoints (pixels with only one neighbor)
    endpoints = [i for i, a in enumerate(adj) if len(a) == 1]

    # BFS from each endpoint to get the longest path
    best_path = []
    for s in endpoints:
        # BFS initialization
        q = deque([s])
        parent = {s: None}
        
        # Perform BFS and track parent for path reconstruction
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in parent:
                    parent[v] = u
                    q.append(v)
        
        # Find the farthest endpoint from the starting node
        far = max(parent.keys(), key=lambda x: distance_between_nodes(x, s, coords))
        
        # Reconstruct the path from the farthest point to the start
        path = []
        cur = far
        while cur is not None:
            path.append(coords[cur])
            cur = parent[cur]
        
        # If this path is the longest so far, save it
        if len(path) > len(best_path):
            best_path = path
    
    # Create an empty mask for the longest path
    mask = np.zeros(shape, dtype=np.uint8)
    for (r, c) in best_path:
        mask[int(r), int(c)] = 255
    
    # Dilate the path to make it more visible
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    mask = cv2.dilate(mask, kernel)
    
    return mask