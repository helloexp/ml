
import numpy as  np

from gradient.data_generate import line_function, generate_function_data


def train_real(xy, w, b, alpha):
    w_sum = 0
    b_sum = 0
    for x, y in xy:
        a = z(w, x, b)
        w_sum += alpha * (y - a) * x
        b_sum += alpha * (y - a)
    return w_sum, b_sum


def train(X, Y):
    w = 0.1
    b = 0.1
    alpha = 0.000003

    for i in range(50000):
        w_sum, b_sum = train_real(zip(X, Y), w, b, alpha)
        w += w_sum
        b += b_sum
    return w, b


def z(w, x, b):
    return w * x + b


def train_vector(X, Y, w, b, alpha):
    a = np.dot(w.transpose(), X)+b
    w_v = alpha * np.dot((Y - a),X.transpose())
    b_v = alpha * np.dot((Y - a),np.ones((20,1)))
    return w_v, b_v


def train_with_vector(X, Y):
    w = np.array([[0.1]])
    b = np.array([[0.1]])

    X = np.asarray(X).transpose().reshape(1,len(X))
    Y = np.asarray(Y).transpose().reshape(1,len(Y))

    alpha = 0.000003

    for i in range(50000):
        w_v, b_v = train_vector(X, Y, w, b, alpha)

        w = w + w_v
        b = b + b_v
    return w,b


if __name__ == '__main__':
    data = generate_function_data(line_function, num=20)
    X = data[0]
    Y = data[1]

    print train(X, Y)

    print train_with_vector(X, Y)

    x = np.array(X)
    y=np.array(Y)





