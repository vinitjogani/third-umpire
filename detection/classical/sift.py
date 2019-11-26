import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import utils

def find_matches(des1, des2, thresh=0.8):
    a2 = (des1**2).sum(axis=1)
    b2 = (des2**2).sum(axis=1)
    ab = np.dot(des2, des1.T)
    norm = (a2[np.newaxis, :] -2*ab + b2[:, np.newaxis]).T
    best = norm.argsort(axis=1)[:, :2]
    for kp in range(des1.shape[0]):
        if norm[kp, best[kp,0]] / norm[kp, best[kp,1]] < thresh:
            yield (kp, best[kp, 0])

def detect_keypoints(I):
    sift = cv.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(I, None)
    return kp, des 

def pipeline(vid='dataset/trimmed/6032627733001_Trim.mp4'):
    prev = None
    i = 0
    for idx, frame in utils.video(vid):
        if prev is not None:
            if (i % 3) != 0: continue
            tup1 = detect_keypoints(frame)
            tup2 = detect_keypoints(prev)
            img2 = frame.copy()
            for kp1, kp2 in find_matches(tup1[1], tup2[1], 0.5):
                x1,y1 = tup1[0][kp1].pt
                x2,y2 = tup2[0][kp2].pt
                manhattan = abs(x1-x2) + abs(y1-y2)
                if manhattan > 5:
                    cv.circle(img2, (int(x1), int(y1)), 2, (255, 0, 0), -1)
                    cv.circle(img2, (int(x2), int(y2)), 2, (0, 0, 255), -1)
            yield img2
        prev = frame

utils.writer(pipeline(), 'sift.mp4')