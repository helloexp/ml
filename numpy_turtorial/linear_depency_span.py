import numpy as np
import matplotlib.pyplot as plt





def plotVectors(vecs, cols, alpha=1):
    """
    Plot set of vectors.

    Parameters
    ----------
    vecs : array-like
        Coordinates of the vectors to plot. Each vectors is in an array. For
        instance: [[1, 3], [2, 2]] can be used to plot 2 vectors.
    cols : array-like
        Colors of the vectors. For instance: ['red', 'blue'] will display the
        first vector in red and the second in blue.
    alpha : float
        Opacity of vectors

    Returns:

    fig : instance of matplotlib.figure.Figure
        The figure of the vectors
    """
    plt.figure()
    plt.axvline(x=0, color='#A9A9A9', zorder=0)
    plt.axhline(y=0, color='#A9A9A9', zorder=0)

    for i in range(len(vecs)):
        x = np.concatenate([[0,0],vecs[i]])
        plt.quiver([x[0]],
                   [x[1]],
                   [x[2]],
                   [x[3]],
                   angles='xy', scale_units='xy', scale=1, color=cols[i],
                  alpha=alpha)


def equation_solution():
    x = np.arange(-10, 10)

    plt.figure()

    y = 2 * x + 1
    plt.plot(x, y)

    z = 6 * x - 2
    plt.plot(x, z)

    t = x / 10 + 6
    plt.plot(x, t)

    plt.xlim(-2, 10)
    plt.ylim(-2, 10)

    plt.axvline(x=0, color="#A9A9A9")
    plt.axhline(y=0, color="#A9A9A9")

    plt.show()
    plt.close()

def vectors():
    orange = '#FF9A13'
    blue = '#1190FF'
    plotVectors([[1, 3], [2, 1]], [orange, blue])
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.show()
    plt.close()


def plot_new():
    orange = '#FF9A13'
    blue = '#1190FF'

    # Weigths of the vectors
    a = 2
    b = 1
    c = 0.5
    # Start and end coordinates of the vectors
    u = [0, 0, 1, 3]
    v = [2, 6, 2, 1]

    x0 = [u[0], a * u[0], b * v[0],c * v[0]]
    y0 = [u[1], a * u[1], b * v[1],c * v[1]]
    x1 = [u[2], a * u[2], b * v[2],c * v[2]]
    y1 = [u[3], a * u[3], b * v[3],c * v[3]]

    print(x0)
    print(x1)
    print(y0)
    print(y1)

    plt.quiver(x0,
               y0,
               x1,
               y1,
               angles='xy', scale_units='xy', scale=1, color=[orange, orange, blue])
    plt.xlim(-1, 8)
    plt.ylim(-1, 8)
    # Draw axes
    plt.axvline(x=0, color='#A9A9A9')
    plt.axhline(y=0, color='#A9A9A9')
    plt.scatter(4, 7, marker='x', s=50)
    # Draw the name of the vectors
    plt.text(-0.5, 2, r'$\vec{u}$', color=orange, size=18)
    plt.text(0.5, 4.5, r'$\vec{u}$', color=orange, size=18)
    plt.text(2.5, 7, r'$\vec{v}$', color=blue, size=18)
    plt.show()
    plt.close()


def scatter_test():

    plt.figure()
    plt.xlim(0,10)
    plt.ylim(0,10)

    plt.axvline(x=0,color="#A9A9A9")
    plt.axhline(y=0,color="#A9A9A9")

    plt.scatter([1,0,5],[1,1,4],marker="x",s=50)

    plt.show()
    plt.close()


if __name__ == '__main__':
    # equation_solution()

    # vectors()

    # plot_new()

    scatter_test()





