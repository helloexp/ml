# -*- coding:utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn import datasets

from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import Sequential,load_model

import matplotlib.pyplot as plt
import numpy as np
import argparse


# ap=argparse.ArgumentParser()
# ap.add_argument("-o","--output",required=True)
# args = vars(ap.parse_args())


dataset=datasets.fetch_mldata("MNIST Original")

data = dataset.data.astype("float") / 255.0

(trainX,testX,trainY,testY) = train_test_split(data, dataset.target, test_size=0.25)

lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.fit_transform(testY)

model=load_model("./keras_mnist.h5")
# model=Sequential()

# define the 784-256-128-10 architecture using Keras
model.add(Dense(256,input_shape=(28*28,),activation="sigmoid"))
model.add(Dense(128,activation="sigmoid"))
model.add(Dense(10,activation="sigmoid"))

sgd=SGD(0.01)
model.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=["accuracy"])


model.fit(trainX,trainY,validation_data=(testX,testY),epochs=100,batch_size=128)

predictions = model.predict(testX, batch_size=128)

report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1))

print(report)

model.save("./keras_mnist.h5")
















