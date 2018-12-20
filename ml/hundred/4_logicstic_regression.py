# coding=utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import  LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv("./resource/4.csv")

X=df.iloc[:,1:3].values
Y=df.iloc[:,-1].values

encoder = LabelEncoder()
X[:,0]=encoder.fit_transform(X[:,0])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# print(X_train)
# print(y_train)

standard_scaler = StandardScaler()

X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.fit_transform(X_test)

regression = LogisticRegression()

regression.fit(X_train,y_train)

predict = regression.predict(X_test)

print([(a,b) for a,b in zip(predict,y_test) if a!=b])

cm = confusion_matrix(y_test, predict)

print(cm)





