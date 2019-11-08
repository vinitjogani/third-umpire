import os
import cv2 as cv

i = 0
for vid in os.listdir('trimmed'):
    cap = cv.VideoCapture('trimmed/' + vid)
    while True:
        ret, frame = cap.read()
        if not ret: break
        i += 1
        if (i % 3) != 0: continue
        cv.imwrite(f'frames/{i//3}.png', frame)
    cap.release()
