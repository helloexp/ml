# coding=utf-8


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv("./resource/2.csv")


X = df.iloc[:, :1].values
Y = df.iloc[:, 1].values

print(X,Y)

X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.25, random_state=0)

regression = LinearRegression()

regression.fit(X_train,Y_train)

Y_predict = regression.predict(X_test)

plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train,regression.predict(X_train),color="blue")
plt.show()


plt.scatter(X_test , Y_test, color = 'black')
plt.plot(X_test,regression.predict(X_test),color="yellow")
plt.show()







