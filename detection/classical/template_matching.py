###
### Imports
###

import sys
import utils
import cv2 as cv
import numpy as np
import os


###
### Constants
###

TEMPLATES_PATH = 'dataset/templates'
VIDEOS_PATH = 'dataset/trimmed/'
OUT_PATH = 'detection/classical/tm_out/'


###
### Initialization
###

ts = [
    cv.imread(os.path.join(TEMPLATES_PATH, f))
    for f in os.listdir(TEMPLATES_PATH)
    if f.endswith('.jpg')
]


###
### Helpers
###

def pyramid(I, min_height=20, decay=0.9):
    while I.shape[0] > min_height:
        yield I
        I = cv.GaussianBlur(I, (5,5), int((1-decay)*10))
        I = cv.resize(I, (int(I.shape[1] * decay), int(I.shape[0] * decay)))

def match(I, t, decay=0.9):
    I = cv.cvtColor(I, cv.COLOR_BGR2YCrCb)
    t = cv.cvtColor(t, cv.COLOR_BGR2YCrCb)
    # Find matches on different scales
    matches = []
    for img in pyramid(I, 100, decay):
        match = cv.matchTemplate(img, t, cv.TM_CCORR_NORMED)
        matches.append(match)
        
    # Find maxima
    scale = np.argmax([m.max() for m in matches])
    m = matches[scale]
    confidence = m.max()
    m = (m - m.min()) / (m.max() - m.min())
    y,x = np.where(m > 0.999)
    
    # Rescale coordinates by scale
    sf = (1/decay)**scale
    h = int(t.shape[0]*sf)
    w = int(t.shape[1]*sf)
    y,x = int(y.min()*sf), int(x.min()*sf)
    
    return x,y,w,h,confidence

def match_all(I, ts, decay=0.9): 
    dy = I.shape[0]//3,(4*I.shape[1])//5
    I = I[dy[0]:dy[1]]

    max_ = -float('inf')
    box_ = (0,0,0,0,0)
    for i,t in enumerate(ts):
        x,y,w,h,c = match(I,t,decay)
        if c > max_:
            box_ = (x,y+dy[0],w,h,i)
            max_ = c

    return box_

def draw_box(frame):
    x,y,w,h,i = match_all(frame, ts, 0.9)
    cv.rectangle(frame, (x-5,y-5), (x+w+5,y+h+5), (255,0,0), thickness=2)
    cv.putText(
        frame, str(i), (x-5,y-5), 
        cv.FONT_HERSHEY_SIMPLEX, 
        1, (255,0,255), thickness=1
    )


###
### Execution
###

def pipeline(path):
    for idx, frame in utils.video(path):
        frame = cv.resize(frame, (int(350*frame.shape[1]/frame.shape[0]), 350))
        draw_box(frame)
        yield frame
        
vids = list(os.listdir(VIDEOS_PATH))
for vid in vids:
    utils.writer(pipeline(VIDEOS_PATH+vid), OUT_PATH + vid)