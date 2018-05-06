# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

from gradient.data_generate import generate_function_data, line_function


def model(w,x,b):
    return tf.multiply(w,x)+b

def train_with_tf(X_input, Y_input):

    X=tf.placeholder(tf.float32)
    Y=tf.placeholder(tf.float32)

    w=tf.Variable(0.1,name="weights")
    b=tf.Variable(0.1,name="biases")

    Y_model=model(w,X,b)

    cost=tf.square((Y-Y_model),name="cost")

    train_op=tf.train.GradientDescentOptimizer(0.0003).minimize(cost)

    with tf.Session() as sess:

        init=tf.initialize_all_variables()
        sess.run(init)

        for i in range(100):
            for (x,y) in zip(X_input,Y_input):
                sess.run(train_op,feed_dict={X:x,Y:y})

        return sess.run(w),sess.run(b)


def calc_w_with_regular_equation(X,Y):

    x_arr=np.asarray(X).reshape(20,1)
    y_arr = np.asarray(Y).reshape(20,1)

    xtx=np.dot(x_arr.transpose(),x_arr)

    xtx_reverse=np.linalg.inv(xtx)

    return xtx_reverse.dot(x_arr.transpose()).dot(y_arr)


if __name__ == '__main__':

    data = generate_function_data(line_function, num=21)
    X = data[0]
    Y = data[1]

    # X=np.array(X)
    # Y=np.array(Y)

    print X
    print Y

    print train_with_tf(X,Y)

    #其中一行数据是在矩阵中占据一列，一个数据点就是一个向量，向量就是一列向量
    #dot 函数是矩阵相乘，而multiply类似与 A*B 是对应元素相乘
    print calc_w_with_regular_equation(X,Y)