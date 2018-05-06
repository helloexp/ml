# -*- coding:utf-8 -*-
import os

from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Dense, Flatten
from keras import backend as k
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD



class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):

        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an
        # empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):

        data = []
        labels = []
        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
                data.append(image)
                labels.append(label)
        return (np.array(data), np.array(labels))


class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing

        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect
        # ratio
        return cv2.resize(image, (self.width, self.height),
                          interpolation=self.inter)


class Image2ArrayPreprocessor:
    def __init__(self, data_format=None):
        self.data_format = data_format

    def preprocess(self, image):
        return img_to_array(image, data_format=self.data_format)


class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()

        inputShape = (height, width, depth)

        model.add(Convolution2D(32, (3, 3), padding="same", input_shape=inputShape))

        model.add(Activation("relu"))

        # in order to fit the full net,need to flatten the matrix to 1D array
        model.add(Flatten())

        model.add(Dense(classes))

        model.add(Activation("softmax"))

        return model


def animal():
    imagePaths = ""
    sp = SimplePreprocessor(32, 32)
    iap = Image2ArrayPreprocessor()
    sl = SimpleDatasetLoader(preprocessors=[sp, iap])
    (data, label) = sl.load(imagePaths)
    data = data.astype("float") / 255.0
    (trainX, trainY, testX, testY) = train_test_split(data, label, test_size=0.25, random_state=42)
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    sgd = SGD(lr=0.005)
    model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    H = model.fit(trainX, trainY, batch_size=32, epochs=100, verbose=1)
    predicts = model.predict(testX, batch_size=32)
    classification_report(testY, predicts)


def train_cifar():
    global model, model_path
    ((trainX, trainY), (testX, testY)) = cifar10.load_data()
    trainX = trainX.astype("float") / 255.0
    testX = testX.astype("float") / 255.0
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    labelNames = ["airplane", "automobile", "bird", "cat", "deer",
                  "dog", "frog", "horse", "ship", "truck"]
    sgd = SGD(lr=0.01)
    model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    H = model.fit(trainX, trainY, batch_size=32, epochs=10, verbose=100)
    predictions = model.predict(testX, batch_size=32)
    report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames)
    print(report)
    model_path = "./cifar10_cnn.h5"
    model.save(model_path)


if __name__ == '__main__':
    # train_cifar()

    model_path = "./cifar10_cnn.h5"

    train_cifar()
    # model = load_model(model_path)
    # print(model)













