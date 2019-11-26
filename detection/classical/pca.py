import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

FRAMES_PATH = 'dataset/frames/'

def load(img):
    h = img.shape[0]
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img[h//4:(4*h)//5]
    img = cv.resize(img, (100,50))
    return img[:, :, np.newaxis].astype('float32') / 255


def get_dataset():
    dataset = {}
    for label in os.listdir(FRAMES_PATH):
        output = []
        label_path = os.path.join(FRAMES_PATH, label)
        for image_file in os.listdir(label_path):
            output.append(load(cv.imread(os.path.join(label_path, image_file))))
        dataset[label] = output
    return dataset

def plot_params(log):
    x = log.coef_.reshape((50,100)) + log.intercept_
    plt.imshow(x)
    plt.show()

def get_data(test_size=0.2):
    X, Y = [], []
    dataset = get_dataset()
    for label in dataset:
        y = float(label[0]), float(label[1])
        for x in dataset[label]:
            X.append(x)
            Y.append(y)
    return train_test_split(np.array(X), np.array(Y), test_size=test_size)


X_train, X_test, Y_train, Y_test = get_data()
bails = LogisticRegression(penalty='l1')
bails.fit(X_train.reshape((X_train.shape[0], -1)), Y_train[:,0])
crease = LogisticRegression(penalty='l1')
crease.fit(X_train.reshape((X_train.shape[0], -1)), Y_train[:,1])

print("Crease")
print("="*100)
print(classification_report(crease.predict(X_test.reshape((X_test.shape[0],-1))), Y_test[:, 1]))
plot_params(crease)

print("Bails")
print("="*100)
print(classification_report(bails.predict(X_test.reshape((X_test.shape[0],-1))), Y_test[:, 0]))
plot_params(bails)


pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train.reshape((-1, 5000)))
X_test_pca = pca.transform(X_test.reshape((-1, 5000)))
crease_pca = LogisticRegression()
crease_pca.fit(X_train_pca, Y_train[:,1])
crease_pca.score(X_test_pca, Y_test[:,1])

print("Crease PCA")
print("="*100)
print(classification_report(crease_pca.predict(X_test_pca), Y_test[:, 1]))