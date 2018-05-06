import numpy as np
import tensorflow as tf


def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

def fizz_buzz_encode(i):
    if   i % 15 == 0: return np.array([0, 0, 0, 1])
    elif i % 5  == 0: return np.array([0, 0, 1, 0])
    elif i % 3  == 0: return np.array([0, 1, 0, 0])
    else:             return np.array([1, 0, 0, 0])


NUM_DIGITS = 10

digites=2 ** NUM_DIGITS
trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, digites)])
trY = np.array([fizz_buzz_encode(i)          for i in range(101, digites)])

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

NUM_HIDDEN = 100

X = tf.placeholder("float", [None, NUM_DIGITS])
Y = tf.placeholder("float", [None, 4])

def init_weight(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

w_h=init_weight((NUM_DIGITS,NUM_HIDDEN))
w_o=init_weight((NUM_HIDDEN,4))

def model(X,w_h,w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

py_x = model(X, w_h, w_o)



predict_op = tf.argmax(py_x, 1)

def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

BATCH_SIZE = 128


if __name__ == '__main__':

    # print trX[0]
    # print trY[0]
    # print digites
    # print range(101, digites)
    # print np.array([3,4]).shape
    # transpose = np.array([1.0, 2.0,3,4]).reshape((2, 2))
    #
    # print transpose
    # print tf.matmul(transpose,np.array([3.0,4.0,5,6]).reshape((2,2)))



    with tf.Session() as sess:

        tf.initialize_all_variables().run()

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=py_x, logits=Y))
        train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

        for epoch in range(10000):
            p = np.random.permutation(range(len(trX)))
            trX, trY = trX[p], trY[p]
            for start in range(0, len(trX), BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

            print(epoch, np.mean(np.argmax(trY, axis=1) ==
                                 sess.run(predict_op, feed_dict={X: trX, Y: trY})))






