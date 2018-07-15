import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def norm_equasion(x,y):

    # return np.sqrt(x*x+y*y)
    return x*x+y*y

if __name__ == '__main__':

    x = np.arange(-10, 10,step=0.1)
    y = np.arange(-10, 10,step=0.1)

    print(x.shape)
    x,y = np.meshgrid(x,y)
    print("--------")
    print(x.shape)
    z=norm_equasion(x,y)

    print(z.shape)

    fig = plt.figure()
    # ax = Axes3D(fig)

    ax = plt.subplot(111,projection='3d')

    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)

    ax.set_zlim(0,100)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')

    plt.show()










