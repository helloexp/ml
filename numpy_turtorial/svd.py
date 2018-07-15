import cv2
import numpy as np
import matplotlib.pyplot as plt

# A  is a matrix that can be seen as a linear transformation.
# This transformation can be decomposed in three sub-transformations:
# 1. rotation, 2. re-scaling, 3. rotation. These three steps correspond to the three matrices U, D, and V.
from numpy_turtorial.linear_depency_span import plotVectors

orange = '#FF9A13'
blue = '#1190FF'

def draw_circle():
    t = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(t)
    y = np.sin(t)
    A = np.array([[3, 0], [0, 2]])
    x_y = np.array([x, y])
    a_x_y = A.dot(x_y)
    x1 = a_x_y[0]
    y1 = a_x_y[1]
    plt.figure()
    plt.plot(x, y)
    plt.plot(x1, y1)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()


# draw_circle()

def rotation_vector():

    u = [1, 0]
    v = [0, 1]
    plotVectors([u, v], cols=[blue, blue])
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.text(-0.25, 0.2, r'$\vec{u}$', color=blue, size=18)
    plt.text(0.4, -0.25, r'$\vec{v}$', color=blue, size=18)
    # plt.show()
    u1 = [-np.sin(np.radians(45)), np.cos(np.radians(45))]
    v1 = [np.cos(np.radians(45)), np.sin(np.radians(45))]
    plotVectors([u1, v1], cols=[orange, orange])
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.text(-0.7, 0.1, r"$\vec{u'}$", color=orange, size=18)
    plt.text(0.4, 0.1, r"$\vec{v'}$", color=orange, size=18)
    plt.show()


# rotation_vector()

# print(np.radians(180))

def matrixToPlot(matrix, vectorsCol=['#FF9A13', '#1190FF']):
    """
    Modify the unit circle and basis vector by applying a matrix.
    Visualize the effect of the matrix in 2D.

    Parameters
    ----------
    matrix : array-like
        2D matrix to apply to the unit circle.
    vectorsCol : HEX color code
        Color of the basis vectors

    Returns:

    fig : instance of matplotlib.figure.Figure
        The figure containing modified unit circle and basis vectors.
    """
    # Unit circle
    x = np.linspace(-1, 1, 100000)
    y = np.sqrt(1-(x**2))

    # Modified unit circle (separate negative and positive parts)
    x1 = matrix[0,0]*x + matrix[0,1]*y
    y1 = matrix[1,0]*x + matrix[1,1]*y
    x1_neg = matrix[0,0]*x - matrix[0,1]*y
    y1_neg = matrix[1,0]*x - matrix[1,1]*y

    # Vectors
    u1 = [matrix[0,0],matrix[1,0]]
    v1 = [matrix[0,1],matrix[1,1]]

    plotVectors([u1, v1], cols=[vectorsCol[0], vectorsCol[1]])

    plt.plot(x1, y1, 'g', alpha=0.5)
    plt.plot(x1_neg, y1_neg, 'g', alpha=0.5)

A = np.array([[3, 7], [5, 2]])

# matrixToPlot(np.array([[1, 0], [0, 1]]))
# plt.xlim(-1.5, 1.5)
# plt.ylim(-1.5, 1.5)
# plt.show()
#
# matrixToPlot(A)
# plt.xlim(-8, 8)
# plt.ylim(-8, 8)
# plt.show()
#
# U, D, V = np.linalg.svd(A)
#
# print(U)
# print(D)
# print(V)
#
#
# matrixToPlot(V)
# plt.xlim(-1.5, 1.5)
# plt.ylim(-1.5, 1.5)
# plt.show()
#
#
# matrixToPlot(np.diag(D).dot(V))
# plt.xlim(-9, 9)
# plt.ylim(-9, 9)
# plt.show()
#
#
# matrixToPlot(U.dot(np.diag(D).dot(V)))
# plt.xlim(-9, 9)
# plt.ylim(-9, 9)
# plt.show()

img = cv2.imread("../resource/cat.jpeg", cv2.IMREAD_GRAYSCALE)
print(img.shape)

U,D,V = np.linalg.svd(img)
print(U.shape)
print(D.shape)
print(V.shape)

for i in [5, 10, 15, 20, 30, 50]:
    reconstimg = np.matrix(U[:, :i]) * np.diag(D[:i]) * np.matrix(V[:i, :])
    plt.imshow(reconstimg, cmap='gray')
    title = "n = %s" % i
    plt.title(title)
    plt.show()



