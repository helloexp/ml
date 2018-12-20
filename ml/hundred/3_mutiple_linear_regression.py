# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

df = pd.read_csv("./resource/3.csv")

X = df.iloc[:, :-1].values
Y=df.iloc[:,-1].values

label_encoder = LabelEncoder()

X[:,3] = label_encoder.fit_transform(X[:, 3])

one_hot_encoder = OneHotEncoder(categorical_features=[3])
X = one_hot_encoder.fit_transform(X).toarray()

# 躲避虚拟变量陷阱
X=X[:,1:]
X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.25, random_state=0)

regression = LinearRegression()
regression.fit(X_train,Y_train)

y_pred=regression.predict(X_test)

print(list(zip(y_pred,Y_test)))







