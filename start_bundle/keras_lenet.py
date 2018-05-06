# -*- coding:utf-8 -*-
from keras.optimizers import SGD
from sklearn import datasets

from keras import Sequential
from keras.layers import Convolution2D, Activation, Flatten, Dense, MaxPooling2D
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()

        inputShape = (height, width, depth)

        model.add(Convolution2D(20, (5, 5), padding="same", input_shape=inputShape))

        model.add(Activation("relu"))

        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        model.add(Convolution2D(50,(5,5),padding="same"))

        model.add(Activation("relu"))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # in order to fit the full net,need to flatten the matrix to 1D array
        model.add(Flatten())

        model.add(Dense(500))

        model.add(Activation("relu"))

        model.add(Dense(classes))

        model.add(Activation("softmax"))

        return model


if __name__ == '__main__':

    dataset=datasets.fetch_mldata("MNIST Original")
    data = dataset.data
    print (data.shape)
    data=data.reshape(data.shape[0], 28, 28, 1)
    data = data.astype("float") / 255.0

    (trainX,testX,trainY,testY) = train_test_split(data, dataset.target, test_size=0.25, random_state=42)

    lb=LabelBinarizer()
    trainY=lb.fit_transform(trainY)
    testY=lb.fit_transform(testY)

    model=LeNet.build(28,28,1,10)

    sgd=SGD(0.01)
    model.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=["accuracy"])

    model.fit(trainX,trainY,validation_data=(testX,testY),epochs=20,batch_size=128)

    predictions = model.predict(testX, batch_size=128)

    report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1))

    print(report)

    model.save("./keras_lenet.h5")






