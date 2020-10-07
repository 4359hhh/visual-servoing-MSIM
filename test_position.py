import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

NUMBER = 1500


def getPositinAndOrientation():
    d0 = 1
    PositinAndOrientation = np.empty(shape=[0, 6])

    while ( len(PositinAndOrientation) < NUMBER):
        x1 = np.random.random()
        x2 = np.random.random()
        x1 = 2 * (x1 - 0.5 * d0)
        x2 = 2 * (x2 - 0.5 * d0)
        phi = 0
        theta = -np.pi / 2
        psi = np.random.uniform((1 / 6) * np.pi, (5 / 6 )* np.pi)  # 左右旋转幅度;应该在π/2左右
        if (x1 * x1 + x2 * x2) < 1:
            P = pow(np.random.uniform(), 1 / 3)  # make the distribution normalized
            ##tip coordination:(0.36814,0,1.7650)
            # r = 0.25
            x = 2 * x1 * np.sqrt(1 - x1 * x1 - x2 * x2) / 3 * P + 0.45
            y = 2 * x2 * np.sqrt(1 - x1 * x1 - x2 * x2) / 3 * P + 0  #
            z = (1 - 2 * (x1 * x1 + x2 * x2)) / 3 * P + 1.765  #

            coor = np.hstack((x, y, z, phi, theta, psi))
            coor = coor.reshape([1, 6])

            PositinAndOrientation = np.vstack((PositinAndOrientation, coor))#
        print('len(PositinAndOrientation is:',len(PositinAndOrientation))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax = Axes3D(fig)
    ax.scatter(PositinAndOrientation[:,0], PositinAndOrientation[:,1], PositinAndOrientation[:,2], c='r', marker='o')
    ax.scatter(0.179, 0, 1.765, c='g', marker='^')  ###(0.51483,0,1.765,c='g',marker='^')
    #plt.show()
    return PositinAndOrientation


if __name__ == "__main__":

    PositinAndOrientation  = getPositinAndOrientation()
    #print(PositinAndOrientation.shape)
    position = np.hstack((PositinAndOrientation[:,0].reshape(NUMBER,1),PositinAndOrientation[:,1].reshape(NUMBER,1),PositinAndOrientation[:,2].reshape(NUMBER,1)))
    np.savetxt('position_1500.txt', position, fmt='%.06f')










