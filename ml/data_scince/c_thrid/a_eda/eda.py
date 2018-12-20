# coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("./datasets-uci-iris.csv", header=None,
                  names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])


def a_describe():
    # 1 mean std etc.
    print(iris.head())
    print(iris.describe())
    # plot
    # iris.boxplot()
    # plt.show()

    mean = iris.mean()
    print(mean)
    print(type(mean))

    print(iris.target.unique())




def b_crosstab():
    # 2 使用共生矩阵查看两个事件之间的关联
    crosstab = pd.crosstab(iris["petal length"] > iris["petal length"].mean(),
                           iris["petal width"] > iris["petal width"].mean())
    print(crosstab)




def c_scatter():
    # 3 show relation with plot
    plt.scatter(iris["petal length"],iris["petal width"],alpha=1.0,color="k")
    plt.xlabel("petal length")
    plt.ylabel("petal width")
    plt.show()




def d_hist():
    # 4 直方图
    plt.hist(iris["petal width"], bins=20)
    plt.xlabel("petal width distribution")
    plt.show()


def e_plt_stand_point():

    from sklearn.preprocessing  import StandardScaler
    scaler = StandardScaler()
    columns = ["petal length", "petal width"]
    iris_transformed = pd.DataFrame(scaler.fit_transform(iris[columns]), columns=columns)
    # print(iris_transformed.describe())
    # print(iris.head())
    # print(iris_transformed.head())
    # print(iris.mean())
    # print(iris_transformed.mean())
    # print(iris.std())
    # print(iris_transformed.std())

    plt.scatter(iris_transformed["petal length"], iris_transformed["petal width"], alpha=1.0, color="k")
    plt.xlabel("petal length")
    plt.ylabel("petal width")
    plt.show()


if __name__ == '__main__':

    # a_describe()
    # b_crosstab()
    c_scatter()
    e_plt_stand_point()
    # d_hist()










