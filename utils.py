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

	# Augmentation 2: Gaussian noise
	if np.random.random() < 0.30:
		noise_level = 0.02 * np.random.random()
		I1 = I1 + np.random.randn(*I1.shape) * noise_level
		I1 = np.clip(I1, 0, 1)
		
	# Augmentation 3: Brightness change
	if np.random.random() < 0.30:
		change = 1 - (np.random.random()*2 - 1) * 0.02
		I1 = np.clip(I1 * change, 0, 1)

	return I1, I2

def generator(X, Y, batch_size=32):
    while True:
        batch = np.random.randint(len(X), size=batch_size)
        augmented = [augment(i,i)[0] for i in X[batch]]
        yield (np.array(augmented), Y[batch])


def flow_generator(X1, X2, Y, batch_size=32):
    while True:
        batch = np.random.randint(len(X1), size=batch_size)
        x1, x2, y = X1[batch], X2[batch], Y[batch]
        augmented = [augment(x1[i],x2[i]) for i in range(batch_size)]
        x_frame, x_flow = zip(*augmented)
        yield ([np.array(x_frame), np.array(x_flow)], Y[batch])