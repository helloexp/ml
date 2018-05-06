# -*- coding:utf-8 -*-i

import pandas as pd
import numpy as  np
from sklearn.linear_model import LinearRegression


def get_pm_train_data():
    axis_arr = ['Date', 'Location', 'Item', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
                '22', '23']
    observer_data = pd.read_csv("./resource/train.csv", encoding="BIG5")
    observer_data.set_axis(1, axis_arr)

    # train = observer_data.drop(['Date', 'Location'], 1)
    # print(observer_data.head())
    train_data = observer_data.drop(['Date', 'Location'], 1)

    pm__item = train_data['Item'] == 'PM2.5'
    pm_train = train_data[pm__item]

    return pm_train.drop("Item",1).reset_index()

def get_traing_data():
    train_data = get_pm_train_data()

    colums = map(lambda s: str(s), np.arange(10, 24))
    drop = train_data.drop(colums, 1)
    print(drop.head())

    Y = drop[["9"]]

    range_index = list(map(lambda s: str(s), np.arange(0, 9)))
    print(range_index)

    X = drop[range_index]

    return (X,Y)


if __name__ == '__main__':

    frame = pd.DataFrame(np.arange(4).reshape(2,2),index=["r1","r2"],columns=["c1","c2"])

    print(frame.ix["r1"])
    print(frame)

    (X,Y) = get_traing_data()

    print(X.head())
    print(Y.head())

    model=LinearRegression()
    model.fit(X,Y)

    test_data = np.asarray([26, 39, 36, 35, 31, 28, 25, 20, 19,21 , 23  ,30  ,30 , 22  ,18  ,13 , 13 , 11]).reshape((-1, 9))

    print(test_data)

    predict = model.predict(test_data)

    print(predict)





