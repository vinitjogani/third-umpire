###
### Imports
###

from tensorflow import keras
import cv2 as cv
import numpy as np
import os
import utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


###
### Constants
###

FRAMES_PATH = 'dataset/frames/'


###
### Helpers
###

def load(img):
    h = img.shape[0]
    img = img[h//4:(4*h)//5]
    img = cv.resize(img, (100,50))
    return img.astype('float32') / 255


def load_all():
    dataset = {}
    for label in os.listdir(FRAMES_PATH):
        output = []
        label_path = os.path.join(FRAMES_PATH, label)
        for image_file in os.listdir(label_path):
            output.append(load(cv.imread(os.path.join(label_path, image_file))))
        dataset[label] = output
    return dataset


def get_data():
    X,Y = [],[]
    dataset = load_all()
    for label in dataset:
        X.extend(dataset[label])
        Y.extend([(float(label[0]), float(label[1]))]*len(dataset[label]))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def split(X, Y, test=0.10, val=0.15):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test+val)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=test/(test+val))
    return [X_train, Y_train], [X_test, Y_test], [X_val, Y_val]


###
### Modelling and Training
###

def build_model(layer1=4, layer2=8, layer3=16, channels=1):
    x_frame = keras.layers.Input(shape=(50,100,channels))

    conv1 = keras.layers.Conv2D(layer1, 3, activation='relu')(x_frame)
    pool1 = keras.layers.MaxPool2D(2)(conv1)
    norm1 = keras.layers.BatchNormalization()(pool1)

    conv2 = keras.layers.Conv2D(layer2, 3, activation='relu')(norm1)
    pool2 = keras.layers.MaxPool2D(2)(conv2)
    norm2 = keras.layers.BatchNormalization()(pool2)

    conv3 = keras.layers.Conv2D(layer3, 3, activation='relu')(norm2)
    pool3 = keras.layers.MaxPool2D(2)(conv3)

    flat = keras.layers.Flatten()(pool3)
    y = keras.layers.Dense(1, activation='sigmoid')(flat)

    model = keras.models.Model(inputs=x_frame, outputs=y)
    model.compile(
        loss='binary_crossentropy', 
        optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.8),
        metrics=['accuracy']
    )

    return model


def train_model(model, train, test, val, y=0, channels=3, save='bails', verbose=False):
    if channels == 1:
        train = [train[0].mean(axis=3)[:,:,:,np.newaxis], train[1]]
        val = [val[0].mean(axis=3)[:,:,:,np.newaxis], val[1]]
        test = [test[0].mean(axis=3)[:,:,:,np.newaxis], test[1]]

    ckp = keras.callbacks.ModelCheckpoint(f'{save}.h5', save_weights_only=True)
    model.fit(
        train[0], train[1][:,y],
        epochs=40, verbose=verbose, callbacks=[ckp],
        validation_data=(val[0],val[1][:,y])
    )
    return classification_report(model(test[0]).numpy().round(), test[1][:, y])


def train_model_generator(model, generator, test, val, y=0, save='bails', verbose=False):
    ckp = keras.callbacks.ModelCheckpoint(
        f'{save}.h5', save_weights_only=True
    )
    tb = keras.callbacks.TensorBoard(log_dir=save + "\\", write_images=True, histogram_freq=1)
    model.fit_generator(
        generator, steps_per_epoch=32,
        epochs=40, verbose=verbose, callbacks=[ckp, tb],
        validation_data=(val[0],val[1][:,y])
    )
    return classification_report(model(test[0]).numpy().round(), test[1][:, y])


def model_grid_search(y=0):
    X, Y = get_data()
    train, test, val = split(X, Y, test=0.01, val=0.10)

    grid_search = [2,4,8,12,16]
    for channels in [1,3]:
        for layer1 in grid_search:
            for layer2 in grid_search:
                for layer3 in grid_search:
                    if not (layer1 <= layer2 <= layer3): continue
                    print()
                    print(channels, layer1, layer2, layer3)
                    model = build_model(layer1, layer2, layer3, channels)
                    report = train_model(model, train, test, val, channels=channels, y=y)
                    print(report)
                    print()


def train_main(y=0, augment=True):
    X, Y = get_data()
    train, test, val = split(X, Y, test=0.10, val=0.15)
    model = build_model(8, 12, 16, 3)
    
    if not augment:
        report = train_model(crease_model, train, test, val, channels=3, verbose=2, y=0)
    else:
        report = train_model_generator(crease_model, 
            utils.generator(train[0], train[1][:,y], 32), 
            test, val, verbose=2, y=y, save=f'cnn_{y}'
        )
    print(report)