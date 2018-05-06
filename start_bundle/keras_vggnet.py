# -*- coding:utf-8 -*-
from keras import Sequential
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn import datasets


class VGGNet:
    @staticmethod
    def build(height, width, depth, classes):
        input_shape = (height, width, depth)

        model = Sequential()
        model.add(Convolution2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        model = Sequential()
        model.add(Convolution2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        model.add(MaxPooling2D())
        model.add(Dropout(0.25))

        model.add(Convolution2D(64, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        model = Sequential()
        model.add(Convolution2D(64, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        model.add(MaxPooling2D())
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model


def step_decay(epoch):
    # initialize the base initial learning rate, drop factor, and
    # epochs to drop every
    initAlpha = 0.01
    factor = 0.25
    dropEvery = 5

    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    # return the learning rate
    return float(alpha)


((trainX, trainY), (testX, testY)) = cifar10.load_data()

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# trainX=trainX.reshape((trainX.shape[0],32*32*3))
# testX=testX.reshape((testX.shape[0],32*32*3))

lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)

testY = lb.fit_transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

model = VGGNet.build(width=32, height=32, depth=3, classes=10)

# sgd = SGD(0.01)
# sgd = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

callbacks = [LearningRateScheduler(step_decay)]

model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=20, batch_size=128,callbacks=callbacks)

predictions = model.predict(testX, batch_size=128)

report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1))

print(report)

model.save("./keras_vggnet.h5")
