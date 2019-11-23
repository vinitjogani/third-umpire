import cv2 as cv
import numpy as np

def video(path):
    i = 0
    cap = cv.VideoCapture(path)
    while True:
        hasnext, frame = cap.read()
        if hasnext: yield (i, frame)
        else: break
        i += 1
    cap.release()

def writer(generator, path):
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = None 
    for frame in generator:
        if out is None:
            out = cv.VideoWriter(path, fourcc, 60, (frame.shape[1],frame.shape[0]))
        out.write(frame)
    out.release()

def player(generator, delay=10):
    for frame in generator:
        # frame = cv.resize(frame, (600, 350))
        cv.imshow('frame', frame)
        cv.waitKey(delay)

def augment(I1, I2):
	# Augmentation 1: flips
	if np.random.random() < 0.5:
		I1 = I1[:, ::-1, :]
		I2 = I2[:, ::-1, :]
	if np.random.random() < 0.5:
		I1 = I1[::-1, :, :]
		I2 = I2[::-1, :, :]

	# Augmentation 2: Gaussian noise
	if np.random.random() < 0.25:
		noise_level = 0.01 * np.random.random()
		I1 = I1 + np.random.randn(*I1.shape) * noise_level
		I1 = np.clip(I1, 0, 1)
		
	# Augmentation 3: Brightness change
	if np.random.random() < 0.25:
		change = 1 - (np.random.random()*2 - 1) * 0.05
		I1 = np.clip(I1 * change, 0, 1)

	return I1, I2

def generator(X, Y, batch_size):
    X_gray = X.mean(axis=3)[:,:,:,np.newaxis]
    while True:
        batch = np.random.randint(len(X), size=batch_size)
        augmented = [augment(i,i)[0] for i in X[batch]]
        yield (np.array(augmented), Y[batch])