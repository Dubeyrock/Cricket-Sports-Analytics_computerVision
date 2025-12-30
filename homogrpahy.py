import cv2
import numpy as np

def get_homography(src_points, dst_points):
    H, _ = cv2.findHomography(
        np.array(src_points),
        np.array(dst_points)
    )
    return H

def project_point(point, H):
    px = np.array([[point[0]], [point[1]], [1]])
    dst = H @ px
    dst = dst / dst[2]
    return int(dst[0]), int(dst[1])
