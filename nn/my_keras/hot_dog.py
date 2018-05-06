# encoding:utf-8
import hashlib
import os
import threading
import urllib.request

import time

import requests
from skimage import exposure
import cv2
from sklearn.model_selection import train_test_split
import glob
from keras import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation,Dense,Flatten,Dropout
import numpy as np
from sklearn.preprocessing import LabelBinarizer

download_base="./wartermark/"

ip = "forward.xdaili.cn"
port = "80"

ip_port = ip + ":" + port

proxy = {"http": "http://" + ip_port, "https": "https://" + ip_port}

def getSign():
   orderno = "ZF20181266329bKqY4Q"
   secret = "f07551a2e1cd4b018dcfc9d3a5574c82"
   timestamp = str (int (time.time()))  # 计算时间戳
   string = "orderno=" + orderno + "," + "secret=" + secret + "," + "timestamp=" + timestamp
   string = string.encode ()
   md5_string = hashlib.md5 (string).hexdigest ()  # 计算sign
   sign = md5_string.upper ()
   auth = "sign=" + sign + "&" + "orderno=" + orderno + "&" + "timestamp=" + timestamp
   headers = { "Proxy-Authorization" : auth }
   return headers

def store_images(folders,links):
    sign = getSign()

    for link, folder in zip(links, folders):
        if not os.path.exists(folder):
            os.makedirs(folder)

        r = requests.get(link, headers=sign, proxies=proxy, verify=False, allow_redirects=False)
        image_urls = r.text

        images = image_urls.split('\n')

        print(link,len(images))

        for img in images:
            try:
                print(img)
                response = requests.get(img, headers=sign, proxies=proxy, verify=False, allow_redirects=False)
                content=response.content
                filename = folder + '/' + str(np.random.randint(0, 1000000)) + '.jpg'
                open(filename, 'wb').write(content)
            except Exception as e:
                print(str(e))

def urlretrieve(i,folder,num):
    path = folder + "/" + str(np.random.randint(0, 100000000)) + ".jpg"
    print(path)
    urllib.request.urlretrieve(i, path)

def store_raw_images(folders, links):

    for link, folder in zip(links, folders):
        if not os.path.exists(folder):
            os.makedirs(folder)
        image_urls = str(urllib.request.urlopen(link).read())

        images = image_urls.split('\\n')

        pic_num=1
        for i in images:
            print(str(pic_num)+"/"+str(len(images)),end="\t")
            try:
                thread_img = threading.Thread(
                    target=urlretrieve(i,folder,pic_num))
                thread_img.start()
                # img = cv2.imread(folder + "/" + str(pic_num) + ".jpg")
                #
                # # Do preprocessing if you want
                # if img is not None:
                #     cv2.imwrite(folder + "/" + str(pic_num) + ".jpg", img)
                pic_num+=1

            except Exception as e:
                print(str(e))


def remove_invalid(image_parent_path):

    images = os.listdir(image_parent_path)

    for img in images:
        invalid_parent = image_parent_path + '/invalid/'
        for invalid in os.listdir(invalid_parent):
            print(("invalid",invalid))
            try:
                current_image_path = image_parent_path + '/' + str(img)
                invalid = cv2.imread(invalid_parent + str(invalid))
                question = cv2.imread(current_image_path)
                if invalid.shape == question.shape and not (np.bitwise_xor(invalid, question).any()):
                    os.remove(current_image_path)
                    break

            except Exception as e:
                print(str(e))


def rotate_image(img, angle):

    (rows,cols,ch)=img.shape
    rotation_matrix = cv2.getRotationMatrix2D((rows / 2, cols / 2), angle, 1)
    return cv2.warpAffine(img,rotation_matrix,(rows,cols))

def load_blur_img(path, imgSize):
    img = cv2.imread(path)
    if img is not None:
        angle=np.random.randint(0,360)

        img = rotate_image(img, angle)

        img = cv2.blur(img, (5, 5))

        img = cv2.resize(img, imgSize)
        return img


def loadImgClass(classPath, classLable, classSize, imgSize):
    x = []
    y = []

    for path in classPath:
        img = load_blur_img(path, imgSize)
        if img is not None:
            x.append(img)
            y.append(classLable)
        else:
            classPath.remove(path)

    while len(x) < classSize:
        randIdx = np.random.randint(0, len(classPath))
        img = load_blur_img(classPath[randIdx], imgSize)
        x.append(img)
        y.append(classLable)

    return x, y

def load_data(img_size,classSize):

    pets = glob.glob("./hot_dog/pets/*.jpg")
    not_pets=glob.glob("./hot_dog/not_pets/*.jpg")

    print(("pets:",len(pets)))

    imgSize = (img_size, img_size)
    xHotdog, yHotdog = loadImgClass(pets, 0, classSize, imgSize)
    xNotHotdog, yNotHotdog = loadImgClass(not_pets, 1, classSize, imgSize)
    print("There are", len(xHotdog), "hotdog wartermark")
    print("There are", len(xNotHotdog), "not hotdog wartermark")

    xHotdog.extend(xNotHotdog)
    X = np.array(xHotdog)
    yHotdog.extend(yNotHotdog)
    y = np.array(yHotdog)
    # y = y.reshape(y.shape + (1,))
    return X, y


def toGray(images):
    # rgb2gray converts RGB values to grayscale values by forming a weighted sum of the R, G, and B components:
    # 0.2989 * R + 0.5870 * G + 0.1140 * B
    # source: https://www.mathworks.com/help/matlab/ref/rgb2gray.html

    images = 0.2989 * images[:, :, :, 0] + 0.5870 * images[:, :, :, 1] + 0.1140 * images[:, :, :, 2]
    return images


def normalizeImages(images):
    # use Histogram equalization to get a better range
    # source http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
    images = (images / 255.).astype(np.float32)

    for i in range(images.shape[0]):
        images[i] = exposure.equalize_hist(images[i])

    images = images.reshape(images.shape + (1,))
    return images


def preprocessData(images):
    grayImages = toGray(images)
    return normalizeImages(grayImages)

def kerasModel(inputShape):

    model=Sequential()
    model.add(Convolution2D(8,(5,5),border_mode='valid', input_shape=inputShape))
    model.add(Dropout(0.5))
    model.add(Activation("relu"))

    model.add(Convolution2D(16, 3, 3))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(240))

    model.add(Activation('relu'))
    model.add(Dense(120))

    # model.add(Activation('relu'))
    model.add(Dense(2))

    model.add(Activation('softmax'))
    return model


def train():
    size = 64
    class_size = 2500
    X, Y = load_data(size, class_size)
    print((len(X), len(Y)))
    n_classes = len(np.unique(Y))
    scaled_X = preprocessData(X)
    from keras.utils.np_utils import to_categorical
    Y = to_categorical(Y)
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, Y, test_size=0.2, random_state=rand_state)
    print("train shape X", X_train.shape)
    print("train shape y", y_train.shape)
    inputShape = (size, size, 1)
    model = kerasModel(inputShape)
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    history = model.fit(X_train, y_train, nb_epoch=5, validation_split=0.1)
    label_binarizer = LabelBinarizer()
    y_one_hot_test = label_binarizer.fit_transform(y_test)
    metrics = model.evaluate(X_test, y_test)
    for metric_i in range(len(model.metrics_names)):
        metric_name = model.metrics_names[metric_i]
        metric_value = metrics[metric_i]
        print('{}: {}'.format(metric_name, metric_value))


if __name__ == '__main__':

    # get_image_from_url("../resource/cat_url.txt")
    links = [
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01318894',
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03405725',
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152',
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00021265',
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07690019',
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07865105',
        'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07697537']

    paths = ['pets',
             'furniture', 'people', 'not_pets', 'frankfurter',
             'chili-dog', 'hotdog']

    paths = list(map(lambda x: download_base  + x, paths))

    print(paths)
    # store_images(paths,links)
    #store_raw_images(paths, links)
    # for path in paths:
    #     remove_invalid(path)

    # remove_invalid("./hot_dog/not_pets")
    # remove_invalid("./hot_dog/pets")

    train()











