###
### Imports
###

import utils
import os
import cv2 as cv
import numpy as np
from detection.neuralnet.cnn import build_model, load
from detection.neuralnet.cnn_flow import build_model as flow_model
from dataset.extract_flow import flow_video, loader as flow_loader, loader as flow_loader
import matplotlib.pyplot as plt
import time


###
### Constants
###

VIDEOS_PATH = 'dataset/trimmed/'
OUT_PATH = 'detection/neuralnet/cnn_out/'
BAILS_MODEL = 'detection/neuralnet/models/bails.h5'
CREASE_MODEL = 'detection/neuralnet/models/crease_of.h5'


###
### Helpers
###

def load_models():
    bails_model = build_model(8, 12, 16, 3)
    bails_model.load_weights(BAILS_MODEL)
    crease_model = flow_model()
    crease_model.load_weights(CREASE_MODEL)
    return bails_model, crease_model


def dcsn(by, cy):
    if len(by) < 1: return -1, by, cy
    
    max_ = min(max(by.max(), cy.max()), 0.5)
    bails = len(by)-1 if by.max() < max_ else np.where(by >= max_)[0].min()
    crease = len(cy)-1 if cy.max() < max_ else np.where(cy >= max_)[0].min()
    if bails == crease:
        if by[bails] > cy[crease]: crease += 1
        else: bails += 1
    decision = 'OUT' if bails < crease else 'NOT OUT'

    return min(bails, crease), decision


###
### Execution
###

def pipeline(path):
    bails, crease = load_models()
    cy, by = [], []
    
    idx = 0
    for org, frame, chan in flow_video(path, 2):
        frame = cv.resize(frame, (100,50)).astype('float32')[np.newaxis]/ 255
        by.append(bails(frame).numpy()[0,0])
        
        chan = cv.resize(chan, (100,50))
        chan = chan[np.newaxis, :,:,np.newaxis]
        cy.append(crease([frame, chan]).numpy()[0,0])
        idx += 1
    
    by, cy = np.array(by), np.array(cy)
    best_frame, decision = dcsn(by, cy)
    by, cy = by.round(3), cy.round(3)
    if best_frame == -1: best_frame = len(by)-1
    
    idx = 0
    for org, frame, chan in flow_video(path, 2):
        org = cv.resize(org, (960, 540))
        text = f'<{decision}> crease={int(cy[idx]*1000)/10}%, bails={int(by[idx]*1000)/10}%'
        cv.rectangle(org, (0,0), (org.shape[1], 50), (0,0,0), -1)
        cv.putText(org, text, (50,25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), thickness=2)
        if idx == best_frame:
            cv.rectangle(org, (0,org.shape[0]-20), (org.shape[1], org.shape[0]), (0,255,0), -1)
            for _ in range(25):
                yield org
        yield org
        idx += 1


# vids = list(os.listdir(VIDEOS_PATH))
# for vid in vids:
#     if not os.path.exists(OUT_PATH + vid):
#         utils.writer(pipeline(VIDEOS_PATH+vid), OUT_PATH + vid)
    # utils.player(pipeline(VIDEOS_PATH+vid), 25)

utils.writer(pipeline('Guptill_trim.mp4'), OUT_PATH + 'Guptill.mp4')