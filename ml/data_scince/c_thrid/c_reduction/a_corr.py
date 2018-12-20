# coding=utf-8

# 协方差矩阵

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, RandomizedPCA, FactorAnalysis,KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


iris = datasets.load_iris()


# 从协方差矩阵可以看出 有两个独立的特征
def a_corr():
    cov_data = np.corrcoef(iris.data.T)
    print(iris.feature_names)
    print(cov_data)
    img = plt.matshow(cov_data)
    plt.colorbar(img, ticks=[-1, 0, 1])
    plt.show()


def b_pca():
    pca_2 = PCA(n_components=2)
    X_pca = pca_2.fit_transform(iris.data)
    print(X_pca.shape)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, edgecolors="none")
    plt.show()

    # Percentage of variance explained by each of the selected components.
    #     If all components are stored, the sum of explained variances is equal
    #     to 1.0.
    print(pca_2.explained_variance_ratio_.sum())
    # Principal axes in feature space, representing the directions of
    #     maximum variance in the data
    print(pca_2.components_)


def c_random_pca():
    pca_2 = RandomizedPCA(n_components=2)
    X_pca = pca_2.fit_transform(iris.data)
    print(X_pca.shape)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, edgecolors="none")
    plt.show()

    # Percentage of variance explained by each of the selected components.
    #     If all components are stored, the sum of explained variances is equal
    #     to 1.0.
    print(pca_2.explained_variance_ratio_.sum())
    # Principal axes in feature space, representing the directions of
    #     maximum variance in the data
    print(pca_2.components_)


def d_lfa():
    # 潜在因素分析 去掉了pca 的正交约束

    lfa = FactorAnalysis(n_components=2)
    X_lfa = lfa.fit_transform(iris.data)
    plt.scatter(X_lfa[:, 0], X_lfa[:, 1], c=iris.target, edgecolors="none")
    plt.show()


def e_lda():
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_2c = lda.fit_transform(iris.data, iris.target)

    plt.scatter(lda_2c[:, 0], lda_2c[:, 1], c=iris.target, edgecolors="none")
    plt.show()


def d_kernel_pca():
    N_points = 50
    fake_circular_data = np.vstack([circular_points(1.0, N_points), circular_points(5.0, N_points)])

    rand = np.random.rand(*fake_circular_data.shape)
    fake_circular_data += rand

    print(rand[0:10])

    fake_circular_target = np.array([0] * N_points + [1] * N_points)
    plt.scatter(fake_circular_data[:, 0], fake_circular_data[:, 1], c=fake_circular_target, alpha=0.8,
                edgecolors="none")

    plt.show()


    kernel_pca = KernelPCA(n_components=2, kernel="rbf")

    kernel_pca_data = kernel_pca.fit_transform(fake_circular_data, fake_circular_target)

    plt.scatter(kernel_pca_data[:, 0], kernel_pca_data[:, 1], c=fake_circular_target, alpha=0.8,
                edgecolors="none")

    plt.show()


def circular_points(r, N):
    return np.array([[np.cos(2 * np.pi * t / N) * r, np.sin(2 * np.pi * t / N) * r] for t in range(0, N)])


if __name__ == '__main__':
    # a_corr()
    # b_pca()
    # c_random_pca()
    # d_lfa()
    # e_lda()
    d_kernel_pca()

