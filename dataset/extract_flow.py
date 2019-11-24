###
### Imports
###

from tensorflow import keras
from detection.neuralnet.cnn import get_data, load
import os
import cv2 as cv
import numpy as np


###
### Constants
###

VIDEOS_PATH = 'dataset/trimmed/'
OUT_PATH = 'dataset/flow2/'
WORKING_SIZE = (600, 300)


###
### Initialization
###

def loader(img):
    h = img.shape[0]
    return img[h//4:(4*h)//5]

max_chan = lambda x: x[...,np.argmax(x.sum(axis=(0,1)))]
blur = lambda x: cv.medianBlur(x, 5)
gray = lambda x: cv.cvtColor(x, cv.COLOR_BGR2GRAY)
X_all, Y_all = get_data(lambda x: cv.resize(loader(x), WORKING_SIZE).astype('float32')/255)
current = 0


###
### Helpers
###

def get_class(images):
    a = X_all.reshape((X_all.shape[0],-1))
    b = images.reshape((images.shape[0], -1))
    aa = (a**2).sum(axis=1)
    bb = (b**2).sum(axis=1)
    norm = aa[:, np.newaxis] + bb[np.newaxis, :] - 2*np.dot(a, b.T)
    return Y_all[np.argmin(norm, axis=0)]


def get_flow(previous, frame):
    flow = cv.calcOpticalFlowFarneback(
        gray(previous), gray(frame), None,             
        pyr_scale=0.65, levels=8, 
        winsize=15, iterations=3, 
        poly_n=7, poly_sigma=1.2, 
        flags=0
    )
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    return mag, ang


def flow_video(video):
    cap = cv.VideoCapture(video)
    ret, previous = cap.read()
    if not ret: return [],[],[]
    
    previous = loader(previous)
    hsv = np.ones_like(previous) * 255

    i, prvs_of, frames = 0, [], []
    while True:
        # i += 1
        ret, org = cap.read()
        if not ret: break    
        
        frame = loader(org)
        mag, ang = get_flow(previous, frame)
        prvs_of.append(mag)

        mag[np.isinf(mag)] = 0
        if mag.max() < 1: continue  
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv.normalize(np.mean(prvs_of, axis=0), None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv.astype('uint8'),cv.COLOR_HSV2BGR)
        
        previous = frame
        i, prvs_of = i+1, []
        frame = cv.resize(frame, WORKING_SIZE)
        chan = cv.resize(max_chan(blur(bgr)), WORKING_SIZE).astype('float') / 255
        yield (org, frame, chan)


def process(video):
    global current 
    print(video)
    
    frame, chan = zip(*[(frame,chan) for (_,frame,chan) in flow_video(video)])
    frame, chan = np.array(frame), np.array(chan)
    y = get_class(frame.astype('float32')/255)

    for i in range(len(y)):
        current += 1
        folder = str(int(y[i][0]))+str(int(y[i][1]))
        path = OUT_PATH + folder + '/' + str(current)
        cv.imwrite(path+'.png', frame[i])
        cv.imwrite(path+'_flow.png', (chan[i]*255).astype('uint8'))


vids = list(os.listdir(VIDEOS_PATH))
for vid in vids:
    process(VIDEOS_PATH+vid)