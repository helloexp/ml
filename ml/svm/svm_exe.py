import  numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


def svc():

    iris = datasets.load_iris()

    X = iris["data"][:, (2, 3)]
    y= (iris["target"]== 2).astype(np.float64)

    svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge")),
    ])

    svm_clf.fit(X, y)

    predict = svm_clf.predict([[5.5, 1.7]])

    params = svm_clf.get_params()

    print(predict)
    print(params)

if __name__ == '__main__':

    svc()






