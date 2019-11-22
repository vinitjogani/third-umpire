import cv2 as cv
import os
import numpy as np
import utils


###
### Constants
###

TEMPLATES_PATH = 'dataset/templates'
VIDEOS_PATH = 'dataset/trimmed/'
OUT_PATH = 'detection/classical/ct_out/'


###
### Helpers
###

def reds(out):
    norm = out / out.max(axis=2)[:, :, np.newaxis]

    cond = (norm[:, :, 2] < 0.9)
    cond = (norm[:, :, 1] > 0.80) | cond
    cond = (norm[:, :, 0] > 0.80) | cond
    cond = np.dstack((cond, cond, cond))
    
    norm_out = out.copy()
    norm_out[cond] = 0
    return norm_out

def bails_off(out):
    # Blur, trim and extract the reds
    out = cv.GaussianBlur(out, (9, 9), 3).astype('float')
    h = out.shape[0]
    out = out[h//4:3*h//4]
    out = reds(out)
    return out.astype('uint8')

def get_response(path):
    y = []
    for idx, frame in utils.video(path):
        boff = bails_off(frame)
        y.append(boff.sum())
    g = cv.getGaussianKernel(11, 5)[:,0]
    y = np.convolve(y, g, mode='valid')
    if y.std() == 0: y += np.random.randn(y.shape[0])
    z = (y - y.mean()) / y.std()
    
    where = np.where(z > z.min() + 1.5)[0]
    if len(where) == 0: when_off = 0
    else: when_off = np.min(where)
    return z, when_off, frame

def draw_plot(idx, frame, y_norm):
    for idx2 in range(idx-5+1):
        frac_x = idx2 / len(y_norm)
        frac_y = 1 - y_norm[idx2]
        x_cord = int(frac_x * frame.shape[1])
        y_cord = int(frac_y * frame.shape[0])
        cv.circle(frame, (x_cord, y_cord), 3, color=(255, 255, 255), thickness=-1)

###
### Execution
###

def pipeline(path):
    z, when_off, frame = get_response(path)
    y_norm = (z - z.min()) / (z.max() - z.min())

    for idx, frame in utils.video(path):
        frame = cv.resize(frame, (int(350*frame.shape[1]/frame.shape[0]), 350))

        boff = bails_off(frame)
        cv.putText(frame, str(idx), (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        if when_off < idx-5:
            cv.rectangle(frame, (0,0), (frame.shape[1], 10), color=(0, 255, 0), thickness=-1)
        
        if idx-5 >= len(z): break
        draw_plot(idx, frame, y_norm)

        yield frame


vids = list(os.listdir(VIDEOS_PATH))
for vid in vids:
    if not os.path.exists(OUT_PATH + vid):
        utils.writer(pipeline(VIDEOS_PATH+vid), OUT_PATH + vid)