# -*- coding:utf-8 -*-


import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as mnist


def inv(x):
    return np.linalg.inv(x)

def normal_regular(X,Y):

    transpose = X.transpose()
    inv_val = inv(np.dot(transpose, X))
    return np.dot(np.dot(inv_val, transpose), Y)



if __name__ == '__main__':



    mnist_data = mnist.read_data_sets("MNIST_data/", one_hot=True)
    print(type(mnist_data))


