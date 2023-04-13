from energy_dif import energy_dif
import copy
import numpy as np
from numpy import pi

n_theta = 10
theta_values = np.linspace(2*pi / n_theta, 2*pi, n_theta)

def InitSpins(L: int, beta: float):
    S = np.zeros(L, L)
    for i in range(L):
        for j in range(L):
            S[i,j] = np.random.choice(theta_values)
    return S

def MetropolisXY(S: np.array, beta: float, J: float, num_iters: int):
    L = len(S[:,1])
    Snew = copy.deepcopy(S)

    while True:
        i, j = np.randint(0,L-1,2)
        Snew[i,j] = np.random.choice(theta_values)
        
        N = np.zeros(1,4) # Neighbors

        N[0] = S[i,(j+1)%L]
        N[1] = S[(i-1)%L,j]
        N[2] = S[i,(j-1)%L]
        N[4] = S[(i+1)%L,j]

        energydif = energy_dif()        





