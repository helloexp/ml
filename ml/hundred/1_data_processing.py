#coding:utf8

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("./resource/1.csv")

X = df.iloc[:, :-1].values
Y=df.iloc[:,-1].values

# dealing with missing data
imputer = Imputer(missing_values="NaN", strategy="mean")
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# print(X)

# deal with class y data
encoder_X = LabelEncoder()
X[:,0]=encoder_X.fit_transform(X[:,0])

# print(X)

one_hot_encoder = OneHotEncoder(categorical_features=[0])
X = one_hot_encoder.fit_transform(X).toarray()
# print(one_hot_encoder.n_values_)
# print(one_hot_encoder.feature_indices_)
# print(X)

encoder_Y = LabelEncoder()
encoder_Y.fit(Y)
Y = encoder_Y.transform(Y)

#split data to train and test
X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.2, random_state=0)

print(X_train)

scaler = StandardScaler()

# (x-min)/(max-min)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
print(X_train)





