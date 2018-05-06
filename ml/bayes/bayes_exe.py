import numpy as np
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB

from tensorflow.contrib.keras import datasets

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()


def minist():

    X=[x.flatten() for x in x_train]

    clf = MultinomialNB()
    clf.fit(X, y_train)

    predict=[]
    i=0
    for x_t in x_test:
        res = clf.predict([x_t.flatten()])
        predict.append((res,y_test[i]))
        i+=1

    none_text = np.zeros((28, 28)).flatten()
    none_predict = clf.predict([none_text])





def sklearn():
    from sklearn import datasets
    iris = datasets.load_iris()
    gnb = GaussianNB()
    y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
    print("Number of mislabeled points out of a total %d points : %d"
          % (iris.data.shape[0], (iris.target != y_pred).sum()))


if __name__ == '__main__':

    # minist()

    #sklearn()

    X = np.random.randint(2, size=(6, 100))
    Y = np.array([1, 2, 3, 4, 4, 5])
    clf = BernoulliNB()
    clf.fit(X, Y)
    print(clf.predict(X[2]))
















