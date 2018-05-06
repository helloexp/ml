#-*- coding:utf-8 -*-
import graphviz
from sklearn import tree

from ml.desicion_tree.id3 import titanic_data

if __name__ == '__main__':

    titanic_x, titanic_y = titanic_data()

    clf=tree.DecisionTreeClassifier()
    clf.fit(titanic_x,titanic_y)
    dot_data = tree.export_graphviz(clf, out_file=None,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = graphviz.Source(dot_data)
    print graph



