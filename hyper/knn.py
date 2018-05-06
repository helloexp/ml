# -*- coding:utf-8 -*-

from sklearn.neighbors import NearestNeighbors,KDTree
import numpy as np

X=[[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]


if __name__ == '__main__':

    nn = NearestNeighbors(n_neighbors=1,algorithm='kd_tree')

    nn.fit(np.asarray(X))

    distances, indices = nn.kneighbors(np.asarray([[4,3]]))

    print distances
    print indices