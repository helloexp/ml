import numpy as np
import matplotlib.pyplot as plt

from numpy_turtorial.linear_depency_span import plotVectors


def plot_vector(v1, v2):
    plt.figure()
    plt.xlim(0, 5)
    plt.ylim(0, 5)

    plt.quiver([0, 0], [0, 0], v1, v2, angles='xy', scale_units='xy', scale=1, color=["red", "green"])

    plt.axvline(x=0, color="#A9A9A9")
    plt.axhline(y=0, color="#A9A9A9")
    plt.show()
    plt.close()


A = np.array([[-1, 3], [2, -2]])
v = np.array([[2], [1]])

print(A)
print(v)

r = A.dot(v)

# plot_vector(v, r)


# This means that v is a eigenvector of A if v and Av are in the same direction or to rephrase it if the vectors Av and v are parallel.
#  The output vector is just a scaled version of the input vector. This scalling factor is λ which is called the eigenvalue of A.
# A.dot(v)=l * v

eig = np.linalg.eig(A)
print(eig)

eig_matrix = eig[1]
print(eig_matrix)

v1= eig_matrix[:,0]
v2= eig_matrix[:,1]

print(v1)
print(v2)

lam=eig[0]

# plot_vector(v1,v2)
x = np.concatenate([[0,0],v1])

# plotVectors([v1,v2],["red","green"])

# plt.show()

# plt.close()

# We can decompose the matrix A with eigenvectors and eigenvalues.
# It is done with: A=V⋅diag(λ)⋅V−1

A=np.array([[5,1],[3,3]])
eig = np.linalg.eig(A)[0]

V=np.array([[1,1],[1,-3]])

V_inv=np.linalg.inv(V)
print(V)
print(V_inv)

lambdas=np.diag(np.array([6,2]))

print(V.dot(lambdas).dot(V_inv))  # =A


A=np.array([[6,2],[2,3]])

eig = np.linalg.eig(A)
print(eig)

Q=eig[1]

L = Q.dot(np.diag(eig[0]))

print(L)

R = L.dot(Q.T)
print(R)




# Quadratic form to matrix form
# Quadratic equations can be expressed under the matrix form
#
# We can use Λ to simplify our quadratic equation and remove the cross terms

# principal axes form :xTAx=λ1y1^2+λ2y2^2







