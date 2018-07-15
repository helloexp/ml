# -- coding: utf-8 --


import numpy as np
import matplotlib.pyplot as plt

x = np.empty((100, 2))

uniform = np.random.uniform(0, 100, size=100)

x[:, 0] = uniform
x[:, 1] = 0.75 * x[:, 0] + 3 + np.random.normal(0, 10, size=100)
print(x)


def deman(X):
    return X - np.mean(X, axis=0)


x = deman(x)


# 梯度上升
def f(x, w):
    return np.sum((np.dot(x, w) ** 2)) / len(x)


def df_math(x, w):
    return x.T.dot(x.dot(w)) * 2. / len(x)


def deraction(w):
    return w / np.linalg.norm(w)


def first_conment(f, X, w, eta, iter=1e4, epsilon=1e-8):
    current_iter = 0
    w = deraction(w)

    while (current_iter < iter):
        dw = f(X, w)
        last_w = w
        w = w + eta * dw
        w = deraction(w)  # 每次求一个单位方向
        error = f(x, w) - f(x, last_w)
        if (abs(error.all()) < epsilon):
            break
        current_iter += 1
    return w


inital_w = np.random.random(x.shape[1])  # 向量不能在0开始

# 不使用 stadard scalar 进行数据标准化
x = x.astype(np.float64)

w = first_conment(df_math, x, inital_w, 0.001)

print(w)

plt.scatter(x[:, 0], x[:, 1])
plt.plot([0, w[0] * 30], [0, w[1] * 30], color="r")
plt.show()

# X2 = np.empty(x.shape)
#
# for i in range(len(x)):
#     X2[i] = x[i] - x[i].dot(w) * w
X2 = x - x.dot(w).reshape((-1, 1)) * w

plt.scatter(X2[:, 0], X2[:, 1])
plt.show()

inital_w = np.random.random(X2.shape[1])
w2 = first_conment(df_math, X2, inital_w, 0.001)

plt.scatter(X2[:, 0], X2[:, 1])
plt.plot([0, w2[0] * 30], [0, w2[1] * 30], color="r")
plt.show()

print(w2)
print(w)
print(w.dot(w2))




