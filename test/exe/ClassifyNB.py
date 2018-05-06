from sklearn.naive_bayes import GaussianNB
from sklearn import svm


def classify(features_train, labels_train):

    # clf=GaussianNB()
    # clf.fit(features_train,labels_train)
    clf=svm.SVC(C=1000000.0)
    clf.fit(features_train,labels_train)


    return clf




