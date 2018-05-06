
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier

def mnist_load():
    mnist = fetch_mldata("MNIST original")
    return mnist


def show_image(X, index):
    image_vector = X[index]
    some_digit_image = image_vector.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()


def shuffle_minst(X_train, y_train):
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    return  X_train, y_train

def mnist_data():
    mnist = mnist_load()

    X, y = mnist["data"], mnist["target"]
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    X_train, y_train = shuffle_minst(X_train, y_train)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':


    mnist = mnist_load()

    X,y = mnist["data"], mnist["target"]


    #show_image(X,1)

    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    X_train, y_train = shuffle_minst(X_train, y_train)

    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    sgd_class=SGDClassifier(random_state=42)
    sgd_class.fit(X_train,y_train_5)

    x_predict = sgd_class.predict(X_test)






