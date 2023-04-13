from numpy import cos, sin
from energy_dif import energy_dif
import copy
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

n_theta = 10
theta_values = np.linspace(2*pi / n_theta, 2*pi, n_theta)

def InitSpins(L: int):
    S = np.zeros([L, L], float)
    for i in range(L):
        for j in range(L):
            S[i,j] = np.random.choice(theta_values)
    return S

def MetropolisXY(S: np.array, beta: float, J: float, num_iters: int):
    L = len(S[:,1])
    Snew = copy.deepcopy(S)

    while True:
        i, j = np.randint(0,L-1,2)
        theta_old = S[i,j]
        theta_new = np.random.choice(theta_values)
        
        N = np.zeros(1,4) # Neighbors

        N[0] = S[i,(j+1)%L]
        N[1] = S[(i-1)%L,j]
        N[2] = S[i,(j-1)%L]
        N[4] = S[(i+1)%L,j]

        energydif = energy_dif(J, theta_new, theta_old, N)

        # probablity of getting new orientation
        prob = 0
        if energydif > 0:
            prob = np.exp(-beta * energydif)
        else:
            prob = 1

        if np.random.rand() <= prob:
            Snew[i,j] = theta_new

def PlotXY(S: np.array):
    L = len(S[:,1])
    Lrange = [k for k in range(L)]
    plt.imshow(S)
    plt.colorbar().set_label('Spin orientation [rad]')
    grid_x, grid_y = np.meshgrid(Lrange, Lrange)
    print(np.shape(S), np.shape(grid_x), np.shape(grid_y))
    plt.quiver([grid_x, grid_y], cos(S), sin(S))
    plt.title('XY model state')
    plt.axis('off')
    plt.show()


S = InitSpins(100)
# print(S)
PlotXY(S)