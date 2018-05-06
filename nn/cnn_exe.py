# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
# from keras.datasets import mnist
# mnist = mnist.load_data()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# https://www.jianshu.com/p/e2f62043d02b
# https://zhuanlan.zhihu.com/p/22252270
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_veriable(shape):
    inital=tf.truncated_normal(shape)
    return tf.Variable(inital)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 定义2维的 convolutional 图层
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    # strides 就是跨多大步抽取信息
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 定义 pooling 图层
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    # 用pooling对付跨步大丢失信息问题
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

xs=tf.placeholder(tf.float32,[None,28*28])
ys=tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
x_image=tf.reshape(xs,[-1,28,28,1]) #最后一个-1代表图片是灰色的
## 1. conv1 layer ##
W_conv1 = weight_veriable([5, 5, 1, 32]) #patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
filter_conv1 = conv2d(x_image, W_conv1) + b_conv1

h_conv1 = tf.nn.relu(filter_conv1)
h_pool_conv1 = max_pool_2x2(h_conv1)



W_conv2 = weight_veriable([5, 5, 32, 64]) #patch 5x5, in size 1, out size 32
b_conv2 = bias_variable([64])
filter_conv2 = conv2d(h_pool_conv1, W_conv2) + b_conv2
h_conv2 = tf.nn.relu(filter_conv2)
h_pool_conv2 = max_pool_2x2(h_conv2)

#fully connect

W_fc1 = weight_veriable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
# 把pooling后的结果变平
h_pool2_flat=tf.reshape(h_pool_conv2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_veriable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# important step
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))













