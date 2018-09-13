import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

np.random.seed(23423478)
sample_size = 100
k=2

def sample1():
    mu_vec1 = np.array([0, 0, 0])
    cov_mat1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, sample_size).T
    return class1_sample


def sample2():
    mu_vec2 = np.array([1, 1, 1])
    cov_mat2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, sample_size).T
    return class2_sample


class1_sample = sample1()
# print("class1_sample",class1_sample)
class2_sample = sample2()


# print("class2_sample",class2_sample)
#
# print(class1_sample[0])
# print(np.mean(class1_sample[0:]))
#
# print(class2_sample[0])
# print(np.mean(class2_sample[0:]))

def plot_classes():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams['legend.fontsize'] = 10
    ax.plot(class1_sample[0, :], class1_sample[1, :], class1_sample[2, :], 'o', markersize=8, color='blue', alpha=0.5,
            label='class1')
    ax.plot(class2_sample[0, :], class1_sample[1, :], class1_sample[2, :], 'x', markersize=8, color='red', alpha=0.5,
            label='class2')
    plt.title('Samples for class 1 and class 2')
    ax.legend(loc='upper right')
    plt.show()


# plot_classes()

samples = np.concatenate((class1_sample, class2_sample), axis=1)

print("samples", samples.shape)

# Computing the d-dimensional mean vector


mean_x = np.mean(samples[0, :])
mean_y = np.mean(samples[1, :])
mean_z = np.mean(samples[2, :])

# print(mean_x,mean_y,mean_z)

mean_vector = np.array([[mean_x], [mean_y], [mean_z]])

# print(mean_vector)

# compute the scatter matrix

cov = np.cov(samples)
scatter = cov * (sample_size * 2 - 1)  # 当前的n为两个class 样本的总和

# print(scatter)
# print(cov)

# Computing eigenvectors and corresponding eigenvalues
eig_val_sc, eig_vec_sc = np.linalg.eig(scatter)
print(eig_val_sc)
print(eig_vec_sc)

# zero_verctor = scatter.dot(eig_vec_sc[:,0])
# right_zero = eig_val_sc[0] * eig_vec_sc[:,0]

# print(zero_verctor)
# print(right_zero)


# Sorting the eigenvectors by decreasing eigenvalues
eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)


# Choosing k eigenvectors with the largest eigenvalues
matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))


print(eig_val_sc)
print(eig_pairs)
print(matrix_w)

# Transforming the samples onto the new subspace  Y=W.T * X
transformed = matrix_w.T.dot(samples)

print(transformed)


plt.plot(transformed[0,0:20], transformed[1,0:20], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(transformed[0,20:40], transformed[1,20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples with class labels')

plt.show()












