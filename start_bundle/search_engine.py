# -*- coding:utf-8 -*-
import glob

import numpy as np
import cv2
from matplotlib import pyplot as plt


class RGBHistogram(object):
    """
    color histogram
    """

    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):
        hist = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])

        hist = cv2.normalize(hist, hist)

        return hist.flatten()


class Searcher(object):
    def __init__(self, index):
        self.index = index

    def search(self, queryFeature):
        result = {}

        for (filename, feature) in self.index.items():
            distance = self.distance(feature, queryFeature)
            result[filename]=distance

        results = sorted([(v, f) for (f, v) in result.items()])
        return results

    def distance(self, histA, histB, error=1e-10):
        d=np.sum([((a - b) ** 2) / (a + b + error) for a, b in zip(histA, histB)]) * 0.5
        return d

def show_figure(img):
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(img)
    plt.xlim([0, 256])

    plt.show()


def show_mulitiply_channel(img):
    split = cv2.split(img)
    colors = ['g', 'r', 'b']

    plt.figure()
    plt.title("show_mulitiply_channel")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    features = []

    for (channel, color) in zip(split, colors):
        hist = cv2.calcHist(channel, [0], None, [256], [0, 256])

        features.extend(hist)

        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    plt.show()


def show_mulitiply_dimension(img):
    channels = cv2.split(img)

    figure = plt.figure()
    ax = figure.add_subplot(131)

    hist = cv2.calcHist([channels[0], channels[1]], [0, 1], None, [10, 10], [0, 256, 0, 256])

    p = ax.imshow(hist)
    ax.set_title("2D Color Histogram for Green and Blue")
    plt.colorbar(p)

    plt.show()

    hist = cv2.calcHist([img], [0, 1, 2],
                        None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    print("3D histogram shape: %s, with %d values" % (hist.shape, hist.flatten().shape[0]))


def index_images(path):
    img_files = glob.glob(path)

    index = {}

    for img in img_files:
        filename = img[img.rfind("/") + 1:]
        imread = cv2.imread(img)

        rgb_histogram = RGBHistogram([8, 8, 8])

        features = rgb_histogram.describe(imread)

        index[filename] = features

    return index


if __name__ == '__main__':
    rgb_histogram = RGBHistogram([8, 8, 8])
    img = cv2.imread("../resource/two_cat.jpg")

    # cv2.imshow("img",img)

    print(img.shape)
    hist = rgb_histogram.describe(img)

    # show_figure(hist)

    # show_mulitiply_channel(img)

    # show_mulitiply_dimension(img)

    image_indexes = index_images("../resource/*.jpg")

    searcher = Searcher(image_indexes)

    result = searcher.search(hist)
    print(result)

    
