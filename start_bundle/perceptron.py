# -*- coding:utf-8 -*-

import numpy as np


class Perceptron(object):

    def __init__(self,N,alpha=0.1):

        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self,x):

        return 1 if x>0 else 0

    def fit(self,X,y,epochs=10):
        X = np.c_[X, np.ones(X.shape[0])]

        for i in range(0,epochs):
            for (x,target) in zip(X,y):
                dot = np.dot(x, self.W)

                p=self.step(dot)

                if(p!=target):
                    error=p-target
                    print("error",error)
                    print(p,target)
                    self.W+= -self.alpha*error * x

    def predict(self,X,addBias=True):
        X=np.atleast_2d(X)

        if(addBias):
            X = np.c_[X, np.ones(X.shape[0])]

        return self.step(np.dot(X, self.W))

if __name__ == '__main__':

    perceptron = Perceptron(2,alpha=0.1)

    or_x=np.asarray([
        [0,0],
        [0,1],
        [1,0],
        [1,1],
    ])
    or_y=np.asarray([0,1,1,1])

    perceptron.fit(or_x, or_y,epochs=100)
    print(perceptron.predict([0,0]))
    print(perceptron.predict([0,1]))
    print(perceptron.predict([1,0]))
    print(perceptron.predict([1,1]))









