# coding=utf-8


import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler

cali = datasets.california_housing.fetch_california_housing()

X = cali['data']
Y = cali['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)




def a_predict_origin_data():
    regressor = KNeighborsRegressor()
    regressor.fit(X_train, Y_train)
    Y_est = regressor.predict(X_test)
    print("MAE=", mean_squared_error(Y_test, Y_est))


# a_predict_origin_data()
def b_z_score():
    regressor = KNeighborsRegressor()
    scaler = StandardScaler()

    # 对数据进行归一化 z-scores方法
    X_train2 = scaler.fit_transform(X_train)
    X_test2 = scaler.fit_transform(X_test)
    regressor.fit(X_train2, Y_train)
    Y_est = regressor.predict(X_test2)
    print("MAE=", mean_squared_error(Y_test, Y_est))


b_z_score()
def c_none_linear():
    # 使用非线性变换
    regressor = KNeighborsRegressor()
    scaler = StandardScaler()

    non_linear_feat = 5
    X_train_new_feat = np.sqrt(X_train[:, non_linear_feat])
    print(X_train_new_feat.shape)
    X_train_new_feat.shape = (X_train_new_feat.shape[0], 1)
    X_train_extend = np.hstack([X_train, X_train_new_feat])
    X_test_new_feat = np.sqrt(X_test[:, non_linear_feat])
    print(X_test_new_feat.shape)
    X_test_new_feat.shape = (X_test_new_feat.shape[0], 1)
    X_test_extend = np.hstack([X_test, X_test_new_feat])
    X_train_extend_transformed = scaler.fit_transform(X_train_extend)
    X_test_extend_transformed = scaler.fit_transform(X_test_extend)
    regressor.fit(X_train_extend_transformed, Y_train)
    Y_est = regressor.predict(X_test_extend_transformed)
    print("MAE=", mean_squared_error(Y_test, Y_est))


c_none_linear()





