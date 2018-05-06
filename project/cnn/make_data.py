import os

import cv2
import time

import skvideo.io
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
import numpy as np


def cut_img(img, x, y, width, height):
    start_y = y - height
    end_y = y + height

    start_x = x - width
    end_x = x + width

    img = img[start_y:end_y, start_x:end_x]

    return img


def cut_wartermark_data(path):
    img = cv2.imread(path)
    height, width, channel = img.shape

    if (height == 521):  # 521 733
        x = int(width / 2)  # 260 240
        y = int(height / 2)  # 366 316

        x = 240
        y = 316

        width = 40
        height = 50
        img = cut_img(img, x, y, width, height)
        writer_path = path.replace("full", "full-false")
        cv2.imwrite(writer_path, img)


def get_img_matrix(file_dir, label):
    listdir = os.listdir(file_dir)
    x, y = [], []

    for f in listdir:
        name = file_dir + f
        if (name.endswith(".jpg")):
            img = cv2.imread(name)
            img = cv2.resize(img, (80, 80), cv2.INTER_AREA)
            x.append(list(img))
            y.append([label])

    return x, y


def shuffle_data(X, y):

    shuffle_index = np.random.permutation(len(X))
    X_new=[]
    y_new=[]

    for i in shuffle_index:
        X_new.append(X[i])
        y_new.append(y[i])

    return X_new, y_new


def flatten_data(data):
    res = []
    for d in data:
        print("d", d)
        res.append(d.flatten())
    res = np.asarray(res)
    return res


def get_train_data():
    true_data = "/Users/tong/Downloads/work/full-wartermark/"
    false_data = "/Users/tong/Downloads/work/full-false/"

    true_x, true_y = get_img_matrix(true_data, 1)
    false_x, false_y = get_img_matrix(false_data, 0)

    true_x.extend(false_x), true_y.extend(false_y)
    (trainX, testX, trainY, testY) = train_test_split(true_x, true_y, test_size=0.15)

    trainX, trainY = shuffle_data(trainX, trainY)
    # trainX = flatten_data(trainX)
    # testX = flatten_data(testX)

    return (np.asarray(trainX).astype("float") / 255.0,
            np.asarray(testX).astype("float") / 255.0,
            np.asarray(trainY).astype("float"),
            np.asarray(testY).astype("float"))


def image_generate(path):
    aug = ImageDataGenerator(rotation_range=30,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode="nearest")

    for jpg in os.listdir(path):

        if (jpg.endswith("jpg")):
            img = load_img(path + "/" + jpg)

            img = img_to_array(img)

            img = np.expand_dims(img, axis=0)

            imageGen = aug.flow(img, batch_size=1, save_to_dir=path, save_prefix="faded", save_format="jpg")

            total = 0
            for image in imageGen:
                total += 1
                if total == 10:
                    break


def make_from_video(vfile,outdir):

    inputparameters = {}
    outputparameters = {}

    reader = skvideo.io.FFmpegReader(vfile, inputdict=inputparameters,
                                     outputdict=outputparameters)

    # true
    # center_x = 1207
    # center_y = 643

    #false
    center_x = 900
    center_y = 643

    cut_x = 60
    cut_y = 72

    i=0
    for frame in reader.nextFrame():

        if (frame is None):
            break

        img = cut_img(frame, center_x, center_y, cut_x, cut_y)

        # cv2.imshow("frame", img)
        # cv2.waitKey()

        cv2.imwrite(outdir+str(i)+"_"+str(int(time.time()))+".jpg",img)

        i+=1

    reader.close()

def read_video():

    make_from_video("/Users/tong/Downloads/work/full/0cdbac4031af5a20954803f54dc5dbf0e1e88275.mp4",
                    "/Users/tong/Downloads/work/wartermark/full_video_false/")




if __name__ == '__main__':
    path = "/Users/tong/Downloads/work/full/"
    #
    sub_path = os.listdir(path)

    # for p in sub_path:
    #     if (not p.endswith("mp4")):
    #         jpg = path + p
    #         cut_wartermark_data(path + p)
    # (trainX, testX, trainY, testY) = get_train_data()
    # print(len(trainX))
    # print(len(trainY))
    # print(len(testX))
    # print(len(testY))

    # w="/Users/tong/Downloads/work/full-wartermark/0c2e9de04f6d5e13cc49347217b2fc5125e8a93a.jpg"
    # img = cv2.imread(w)
    # print(img.shape)
    # img = cv2.resize(img, (80, 80),cv2.INTER_AREA)
    #
    # cv2.imshow("img",img)
    # cv2.waitKey()

    # image_generate("/Users/tong/Downloads/work/full_video")
    # image_generate("/Users/tong/Downloads/work/full-false")
    read_video()



