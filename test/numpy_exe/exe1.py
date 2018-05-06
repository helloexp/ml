import numpy as np


def array():
    a=np.array([1,2,3,4])
    b=np.array([5,6,7,8])
    c=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    print a
    print b
    print c

    print c.shape

    c.shape=(3,4)

    print c

    c.shape=[2,-1]
    print c


def reverse(x):
    return np.linalg.inv(x)


if __name__ == '__main__':
    array()

    x=np.array([[0,0],[0,1],[1,0],[1,1]])

    print x.transpose()

    y = np.dot(x.transpose(), x)

    print y
    print reverse(y)
