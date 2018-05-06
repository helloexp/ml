from keras.callbacks import TensorBoard

from practitioner_bundle.pyimagesearch.nn.conv import MiniVGGNet
from project.cnn.make_data import get_train_data
from keras.layers import Dense, Convolution2D, Activation, Flatten,MaxPooling2D,Dropout
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import Sequential, load_model
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
import cv2
# import matplotlib.pyplot as plt


(trainX, testX, trainY, testY) = get_train_data()


def train():
    global trainY, testY
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    model = Sequential()
    model.add(Dense(256, input_shape=(8000,), activation="sigmoid"))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(2, activation="sigmoid"))
    sgd = SGD(0.01)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)


class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()

        inputShape = (height, width, depth)

        model.add(Convolution2D(32, (3, 3), padding="same", input_shape=inputShape))

        model.add(Activation("relu"))
        model.add(MaxPooling2D())


        model.add(Convolution2D(64, (3, 3), padding="same", input_shape=inputShape))

        model.add(Activation("relu"))
        model.add(MaxPooling2D())

        # in order to fit the full net,need to flatten the matrix to 1D array
        model.add(Flatten())

        model.add(Dense(classes))

        model.add(Activation("softmax"))

        return model

class LeNet:

    @staticmethod
    def build(width, height, depth, classes):

        model=Sequential()
        inputshape=(height,width,depth)

        model.add(Convolution2D(20,(5,5),padding="same",input_shape=inputshape))

        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Dropout(0.8))

        model.add(Convolution2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.8))

        model.add(Convolution2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.8))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))


        model.add(Dense(500))
        model.add(Activation("relu"))


        model.add(Dense(500))
        model.add(Activation("relu"))

        model.add(Dense(classes))
        model.add(Activation("sigmoid"))

        return model




# train()
# /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/tensorboard --logdir=
# /Users/tong/PycharmProjects/machinelearning/project/cnn/tensorboard/1
if __name__ == '__main__':

    encoder = LabelBinarizer()
    trainY = encoder.fit_transform(trainY)
    testY = encoder.fit_transform(testY)

    model = ShallowNet.build(80, 80, 3, 1)
    # model = LeNet.build(80,80, 3, 1)
    # model = MiniVGGNet.build(32,32, 3, 1)

    # opt = RMSprop(lr=0.1)
    opt=Adam()
    model.compile(loss="binary_crossentropy", optimizer=opt,metrics = ["accuracy"])

    model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=10,
              callbacks=[TensorBoard(log_dir='./tensorboard/2')])

    model_path = "./wartermark.h5"

