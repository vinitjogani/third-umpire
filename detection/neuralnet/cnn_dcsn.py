###
### Imports
###

import utils
import os
import cv2 as cv
import numpy as np
from detection.neuralnet.cnn import build_model, load
import matplotlib.pyplot as plt
import time


###
### Constants
###

VIDEOS_PATH = 'dataset/trimmed/'
OUT_PATH = 'detection/neuralnet/cnn_out/'
BAILS_MODEL = 'detection/neuralnet/models/bails.h5'
CREASE_MODEL = 'detection/neuralnet/models/crease.h5'


###
### Helpers
###

def load_models():
    bails_model = build_model(8, 12, 16, 3)
    bails_model.load_weights(BAILS_MODEL)
    crease_model = build_model(8, 12, 16, 3)
    crease_model.load_weights(CREASE_MODEL)
    return bails_model, crease_model


def dcsn(by, cy):
    if len(by) < 1: return -1, by, cy
    ksize = min(len(by)//5+1, 5)
    by = np.convolve(by, cv.getGaussianKernel(ksize, 1.2).T[0], mode='valid')
    cy = np.convolve(cy, cv.getGaussianKernel(ksize, 1.2).T[0], mode='valid')
    
    max_ = min(max(by.max(), cy.max()), 0.5)
    bails = len(by) if by.max() < max_ else np.where(by >= max_)[0].min()
    crease = len(cy) if cy.max() < max_ else np.where(cy >= max_)[0].min()
    decision = 'OUT' if bails < crease else 'NOT OUT'

    return min(bails, crease), by.round(3), cy.round(3), decision


###
### Execution
###

def pipeline(path):
    bails, crease = load_models()
    cy, by = [], []

    for idx, frame in utils.video(path):
        frame = np.array([load(frame)])
        by.append(bails(frame).numpy()[0,0])
        cy.append(crease(frame).numpy()[0,0])
    
    best_frame, by, cy, decision = dcsn(by, cy)
    if best_frame == -1: best_frame = len(by)-1
    
    for idx, frame in utils.video(path):
        if idx >= len(by): continue

        frame = cv.resize(frame, (960, 540))
        text = f'<{decision}> crease={cy[idx]}, bails={by[idx]}'
        cv.rectangle(frame, (0,0), (frame.shape[1], 50), (0,0,0), -1)
        cv.putText(frame, text, (50,25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=2)

        if idx >= best_frame: time.sleep(0.05)
        yield frame


vids = list(os.listdir(VIDEOS_PATH))
for vid in vids:
    # utils.writer(pipeline(VIDEOS_PATH+vid), OUT_PATH + vid)
    utils.player(pipeline(VIDEOS_PATH+vid), 100)