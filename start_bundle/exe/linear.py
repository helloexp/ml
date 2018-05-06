# -*- coding:utf-8 -*-

import numpy as np
import cv2

labels = ["cats", "dogs", "panda"]
np.random.seed(0)

W = np.random.randn(3, 3072)
b = np.random.randn(3)

orignal = cv2.imread("../../resource/two_cat.jpg")
resize = cv2.resize(orignal, (32, 32)).flatten()

scores = W.dot(resize) + b

for (label, score) in zip(labels, scores):
    print(label + ":" + str(score))

print(np.argmax(scores))

import math

dog = math.exp(-3.44)
cat = math.exp(1.16)
panda = math.exp(3.91)

print([dog,cat,panda])

dog_p = dog / (dog+cat+panda)
cat_p = cat / (dog +cat+ panda)
panda_p = panda / (dog + cat+panda)

print([dog_p,cat_p,panda_p])

dog_l=-math.log(dog_p)
cat_l=-math.log(cat_p)
panda_l=-math.log(panda_p)

print(dog_l,cat_l,panda_l)
