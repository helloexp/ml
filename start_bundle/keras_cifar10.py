# -*- coding:utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn import datasets


from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import Sequential,load_model
from keras.datasets import cifar10

# import matplotlib.pyplot as plt
import numpy as np
import argparse

((trainX,trainY),(testX,testY)) = cifar10.load_data()


trainX = trainX.astype("float") / 255.0
testX=testX.astype("float") / 255.0

trainX=trainX.reshape((trainX.shape[0],32*32*3))
testX=testX.reshape((testX.shape[0],32*32*3))

lb=LabelBinarizer()


trainY = lb.fit_transform(trainY)

testY = lb.fit_transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

# define the 3072-1024-512-10 architecture using Keras
model=Sequential()

model.add(Dense(1024,input_shape=(32*32*3,),activation="sigmoid"))
model.add(Dense(512,activation="sigmoid"))
model.add(Dense(10,activation="sigmoid"))

sgd=SGD(0.01)
model.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(trainX,trainY,validation_data=(testX,testY),epochs=100,batch_size=32)
predictions = model.predict(testX, batch_size=128)
report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1))

print(report)













