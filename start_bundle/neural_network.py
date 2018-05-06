# -*- coding:utf-8 -*-

import numpy as np

class NeuralNetwork(object):


    def __init__(self,layers,alpha=0.1):

        self.W=[]
        self.layers=layers
        self.alpha=alpha

        for i in range(0,len(layers)-2):
            w=np.random.randn(layers[i]+1,layers[i+1]+1)

            self.W.append(w/np.sqrt(layers[i]))  #normalizing the variance of each neuron’s output
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

        print(len(self.W))

    def sigmod(self,x):
        return 1/(1+np.exp(-x))

    def sigmod_derivative(self,x):
        return x*(1-x)

    def fit(self,X,y,epochs=1000,displayUpdate=100):

        X=np.c_[X,np.ones(X.shape[0])]

        for epoch in range(epochs):
            for (x,target) in zip(X,y):
                self.fit_partial(x,target)

            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x, y):

        A=[np.atleast_2d(x)]
        for layer in range(len(self.W)):
            net= A[layer].dot(self.W[layer])
            out=self.sigmod(net)
            A.append(out)

        error = A[-1] - y
        D = [error * self.sigmod_derivative(A[-1])]

        for layer in range(len(A) -2, 0, -1):

            delta=D[-1].dot(self.W[layer].T)
            d=delta*self.sigmod_derivative(A[layer])
            D.append(d)

        D=D[::-1]

        #print("before",self.W)

        for layer in range(len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])
        #print("after",self.W)

    def predict(self,X,addBias=True):

        p = np.atleast_2d(X)

        if(addBias):
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in range(len(self.W)):
            p = self.sigmod(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        return loss


    def __repr__(self):
        return "NeuralNetwork: {}".format(
            "-".join(str(l) for l in self.layers))




if __name__ == '__main__':

    network = NeuralNetwork([2, 2, 1],alpha=0.1)

    or_x = np.asarray([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    or_y = np.asarray([0, 1, 1, 0])

    network.fit(or_x, or_y, epochs=10000,displayUpdate=1000)

    print(network.predict([0,0]))
    print(network.predict([0,1]))
    print(network.predict([1,0]))
    print(network.predict([1,1]))











