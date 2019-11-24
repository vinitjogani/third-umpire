###
### Imports
###

from tensorflow import keras
import os
import cv2 as cv
import numpy as np
import utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

###
### Constants
###

FRAMES_PATH = 'dataset/flow/'

def build_model():
    x_frame = keras.layers.Input(shape=(50,100,3))
    x_flow = keras.layers.Input(shape=(50,100,1))

    x = keras.layers.concatenate([x_frame, x_flow])

    conv1 = keras.layers.Conv2D(8, 3, activation='relu')(x)
    pool1 = keras.layers.MaxPool2D(2)(conv1)
    norm1 = keras.layers.BatchNormalization()(pool1)

    conv2 = keras.layers.Conv2D(16, 3, activation='relu')(norm1)
    pool2 = keras.layers.MaxPool2D(2)(conv2)
    norm2 = keras.layers.BatchNormalization()(pool2)

    conv3 = keras.layers.Conv2D(24, 3, activation='relu')(norm2)
    pool3 = keras.layers.MaxPool2D(2)(conv3)

    flat = keras.layers.Flatten()(pool3)
    y = keras.layers.Dense(1, activation='sigmoid')(flat)

    model = keras.models.Model(inputs=[x_frame, x_flow], outputs=y)
    model.compile(
        loss='binary_crossentropy', 
        optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.8),
        metrics=['accuracy']
    )

    return model

def load_all():
    dataset = {}
    for label in os.listdir(FRAMES_PATH):
        frames, flows = [], []
        label_path = os.path.join(FRAMES_PATH, label)
        for image_file in os.listdir(label_path):
            if 'flow' in image_file: continue
            path = os.path.join(label_path, image_file)
            frames.append(cv.resize(cv.imread(path), (100,50)))
            flow = cv.resize(cv.imread(path.replace('.png', '_flow.png')), (100,50))
            flows.append(cv.cvtColor(flow, cv.COLOR_BGR2GRAY))
        if label[-1] in dataset:
            dataset[label[-1]][0].extend(frames)
            dataset[label[-1]][1].extend(flows)
        else:
            dataset[label[-1]] = frames, flows
    return dataset


def get_data(N=800):
    X1, X2, Y = [], [],[]
    dataset = load_all()
    for label in dataset:
        print(label, len(dataset[label][0]))
        if len(dataset[label][0]) < N:
            sample = list(range(len(dataset[label][0])))
        else:
            # We do this to limit class imbalance
            sample = np.random.choice(len(dataset[label][0]), size=N, replace=False)

        frames, flows = np.array(dataset[label][0]), np.array(dataset[label][1])[..., np.newaxis]
        frames, flows = frames[sample].astype('float32'), flows[sample].astype('float32')
        frames, flows = list(frames/255), list(flows/255)

        if len(sample) < N:
            for _ in range(3):
                f1, f2 = zip(*[
                    utils.augment(x1, x2)
                    for x1,x2 in zip(frames, flows)
                ])
                X1.extend(list(f1))
                X2.extend(list(f2))
                Y.extend([float(label)]*len(sample))
        else:
            X1.extend(frames)
            X2.extend(flows)
            Y.extend([float(label)]*len(sample))

    X1 = np.array(X1)
    X2 = np.array(X2)
    Y = np.array(Y).reshape((-1, 1))
    return X1, X2, Y


def split(X1, X2, Y, test=0.10, val=0.15):
    X1_train, X1_test, X2_train, X2_test, Y_train, Y_test = train_test_split(X1, X2, Y, test_size=test+val)
    X1_val, X1_test, X2_val, X2_test, Y_val, Y_test = train_test_split(X1_test, X2_test, Y_test, test_size=test/(test+val))
    return [X1_train, X2_train, Y_train], [X1_test, X2_test, Y_test], [X1_val, X2_val, Y_val]

def main():
    np.random.seed(23)
    X1, X2, Y = get_data()
    train, test, val = split(X1, X2, Y, test=0.10)

    cb = []
    cb.append(keras.callbacks.ReduceLROnPlateau())
    cb.append(keras.callbacks.ModelCheckpoint('crease_of.h5', save_weights_only=True, save_best_only=True))
    cb.append(keras.callbacks.TensorBoard(log_dir='crease'))
    model = build_model()
    model.fit(
        train[:2], train[2],
        epochs=40, verbose=2, 
        validation_data=(val[0:2], val[2]),
        callbacks=cb
    )
    
    print(classification_report(model(train[:2]).numpy().round(), train[2]))
    print(classification_report(model(test[:2]).numpy().round(), test[2]))
    print(classification_report(model(val[:2]).numpy().round(), val[2]))
