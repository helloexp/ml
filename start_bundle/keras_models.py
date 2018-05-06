# -*- coding:utf-8 -*-

from keras.applications import ResNet50,VGG16,VGG19,InceptionV3,Xception,DenseNet121
from keras.applications import imagenet_utils

from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2

model_key="vgg19"
image_path="../resource/lover.jpg"

MODELS = {
"vgg16": VGG16,
"vgg19": VGG19,
"inception": InceptionV3,
"xception": Xception,
"resnet": ResNet50
}


Network=MODELS[model_key]
inputShape = (224, 224)

if model_key in {"inception", "xception"}:
    inputShape = (299, 299)

preprocess = imagenet_utils.preprocess_input

model=Network(weights="imagenet")

image=load_img(image_path,target_size=inputShape)


#(inputShape[0], inputShape[1], 3)
image=img_to_array(image)
print(image.shape)

#(1,inputShape[0], inputShape[1], 3)
image = np.expand_dims(image, axis=0)

image = preprocess(image)

preds = model.predict(image)
predictions = imagenet_utils.decode_predictions(preds)

print(predictions)



