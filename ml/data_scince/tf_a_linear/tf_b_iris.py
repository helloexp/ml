# coding=utf-8


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from tensorflow.python.framework import ops
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

ops.reset_default_graph()

if __name__ == '__main__':
    iris = datasets.load_iris()

    # print(iris)

    X_val = iris.data
    Y_val = iris.target.reshape((X_val.shape[0], 1))
    print(X_val.shape)
    print(Y_val.shape)

    learning_rate = 0.01
    batch_size = 25

    w_shape = X_val.shape[1]
    print(w_shape)
    X = tf.placeholder(shape=[None, w_shape], dtype=tf.float32)
    Y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    A = tf.Variable(tf.random_normal(shape=(w_shape, 1)))
    b = tf.Variable(tf.random_normal(shape=(1, 1)))

    model_out = tf.add(tf.matmul(X, A), b)

    loss = tf.reduce_mean(tf.square(model_out - Y))

    init = tf.global_variables_initializer()

    iter_num = 20
    with  tf.Session() as session:

        session.run(init)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(loss)

        loss_vec=[]

        for i in range(0, iter_num):

            rand_index=np.random.choice(len(X_val),size=batch_size)
            rand_x=X_val[rand_index]
            rand_y=Y_val[rand_index]

            session.run(train_step, feed_dict={X: rand_x, Y: rand_y})
            temp_loss = session.run(loss, feed_dict={X: rand_x, Y: rand_y})

            loss_vec.append(temp_loss)

            # print(temp_loass)

        print(session.run(A))
        print(session.run(b))

    plt.plot(np.arange(0, iter_num), loss_vec)
    plt.show()

    print(loss_vec)
    # lr = LinearRegression()
    # lr.fit(X_val,Y_val)
    # y_predict = lr.predict(X_val)
    # print(lr)


