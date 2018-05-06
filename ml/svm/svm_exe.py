
from sklearn import svm


if __name__ == '__main__':

    X=[[0,0],[1,1]]
    Y=[0,1]
    cls=svm.SVC()
    cls.fit(X,Y)

    print cls

    print cls.predict([0.49,0.49])
    print cls.predict([0.51,0.51])
