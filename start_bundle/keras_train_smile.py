#-*- coding:utf-8 -*-
import os

import cv2
import imutils
from imutils import  paths
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from start_bundle.keras_lenet import LeNet

SMILE_PATH="/Users/tong/Downloads/ml/SMILEsmileD/SMILEs"
modle_path = "./keras_smil3.h5"


def get_smile_data():
    data = []
    labels = []
    for image_path in paths.list_images(SMILE_PATH):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = imutils.resize(image, width=28)
        image = img_to_array(image)
        data.append(image)

        label = image_path.split(os.path.sep)[-3]
        label = "smiling" if label == "positives" else "not_smiling"
        labels.append(label)

    return data,labels



def train():

    data,labels = get_smile_data()
    print(len(data),set(labels))

    data = np.array(data, "float") / 255.0
    labels = np.array(labels)

    le = LabelEncoder().fit(labels)

    labels = le.transform(labels)
    labels = np_utils.to_categorical(labels, 2)

    classTotals = labels.sum(axis=0)

    print(classTotals)
    classWeight = classTotals.max() / classTotals
    print(classWeight)


    (trainX,testX,trainY,testY) = train_test_split(data, labels, test_size=0.20, random_state=42)

    # model = LeNet.build(28, 28, 1, 2)
    model=load_model(modle_path)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(trainX, trainY, validation_data=(testX, testY), class_weight=classWeight,epochs=25, batch_size=64)

    predictions = model.predict(testX, batch_size=64)
    print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names = le.classes_))

    model.save("%s" % modle_path)


def detect_smile():

    detector = cv2.CascadeClassifier('/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    model=load_model(modle_path)
    camera = cv2.VideoCapture(0)

    while True:
        (grabbed, frame) = camera.read()

        if(not grabbed):
            break

        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameClone = frame.copy()
        rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors = 5, minSize = (30, 30),
                                          flags = cv2.CASCADE_SCALE_IMAGE)

        for (fX, fY, fW, fH) in rects:
            roi = gray[fY:fY + fH, fX:fX + fW]

            roi = cv2.resize(roi, (28, 28))

            roi = roi.astype("float") / 255.0

            roi = img_to_array(roi)

            roi = np.expand_dims(roi, axis=0)

            (notSmiling, smiling) = model.predict(roi)[0]
            label = "Smiling" if smiling > notSmiling else "Not Smiling"

            cv2.putText(frameClone, label, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)

        cv2.imshow("face",frameClone)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':

    # train()
    detect_smile()