# -*- coding:utf-8 -*-
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np

def rotate():

    img="../../resource/mimi.jpg"
    img = load_img(img)

    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    generator = ImageDataGenerator(rotation_range=30,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode="nearest")

    total=0

    img_gen=generator.flow(img,batch_size=1,save_to_dir="../../resource/")

    for image in img_gen:
        total+=1
        if total>9:
            break



if __name__ == '__main__':


    rotate()





