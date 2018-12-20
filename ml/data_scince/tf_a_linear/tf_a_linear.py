# coding=utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def a_solution_by_matrix():
    A, b = get_a_b()


    return solution(A, b)


def get_a_b():
    X_matrix = np.matrix(X_vals)
    # print(X_matrix)
    X_vals_column = np.transpose(X_matrix)
    # print(X_vals_column)
    # repeat = np.repeat(1, 100)
    # print(np.transpose(np.matrix(repeat)))
    ones = np.ones((100, 1))
    A = np.column_stack([X_vals_column, ones])
    b = np.transpose(np.matrix(Y_vals))

    A = tf.constant(A)
    b = tf.constant(b)
    return A, b


def solution(A, b):

    return tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(A), A)), tf.transpose(A)), b)


def cholesky():
    A,b = get_a_b()
    tA_A = tf.matmul(tf.transpose(A), A)

    L=tf.cholesky(tA_A)

    tA_b = tf.matmul(tf.transpose(A), b)

    sol1=tf.matrix_solve(L,tA_b)  # 解决多元一次方程组
    sol2=tf.matrix_solve(tf.transpose(L),sol1)

    return sol2




if __name__ == '__main__':
    X_vals = np.linspace(0, 10, 100)

    np.random.seed(100)
    Y_vals = X_vals + np.random.normal(0, 1, 100)

    with tf.Session() as session:

        # solu = a_solution_by_matrix()
        solu=cholesky()
        solu_eval = session.run(solu)

        x_rate = solu_eval[0][0]
        b_ = solu_eval[1][0]

        befst_fit = []
        for i in X_vals:
            befst_fit.append(x_rate * i + b_)

        plt.plot(X_vals, befst_fit, "r-", label="fit line")

    plt.scatter(X_vals, Y_vals, color="k")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()
