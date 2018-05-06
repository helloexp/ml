import os

import numpy as np
from imutils import paths
from keras import Input, Model
from keras.applications import VGG16
from keras.optimizers import SGD, RMSprop
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from practitioner_bundle.fcheadnet import FCHeadNet
from practitioner_bundle.pyimagesearch.datasets import SimpleDatasetLoader
from practitioner_bundle.pyimagesearch.preprocessing import AspectAwarePreprocessor, ImageToArrayPreprocessor
from keras.preprocessing.image import ImageDataGenerator
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)

args = vars(ap.parse_args())

# aug=ImageDataGenerator(rotation_range=30)
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(name) for name in np.unique(classNames)]

aap = AspectAwarePreprocessor(224,224)
iap = ImageToArrayPreprocessor()

ssp = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = ssp.load(imagePaths)

data = data.astype("float") / 255.0

trainX, testX, trainY, testY= train_test_split(data, labels, test_size=0.25, random_state=42)

lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

baseModel = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

headModel = FCHeadNet.build(baseModel, len(classNames), 256)

model = Model(inputs=baseModel.input, outputs=headModel)

print("layers.len", len(baseModel.layers))
for layer in baseModel.layers:
    layer.trainable = False

opt = RMSprop(lr=0.001)

model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY),
                    epochs=25, steps_per_epoch=len(trainX) // 32, verbose=1)

predictions = model.predict(testX, batch_size=32)

classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames)

for layer in baseModel.layers[15:]:
    layer.trainable = True

opt = SGD(lr=0.001)

model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY),
                    epochs=100, steps_per_epoch=len(trainX) // 32, verbose=1)

predictions = model.predict(testX, batch_size=32)

model.save("vgg_transfer.h5")
