from numpy import cos, sin
from energy_dif import energy_dif
import copy
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

J = 1
L = 32
n_theta = 16
theta_values = np.linspace(2*pi / n_theta, 2*pi, n_theta, dtype=float)

def InitSpins(L: int):
    S = np.zeros([L, L], dtype=float)
    for i in range(L):
        for j in range(L):
            S[i,j] = np.random.choice(theta_values)
    return S

def MetropolisXY(S: np.array, beta: float, J: float, numIter: int):
    L = len(S[:,1])
    Snew = copy.deepcopy(S)

    for x in range(numIter):
        i, j = np.random.randint(0,L-1,2)
        theta_old = Snew[i,j]
        theta_new = np.random.choice(theta_values)
        
        N = np.zeros((4, 1)) # Neighbors

        N[0] = Snew[i,(j+1)%L]
        N[1] = Snew[(i-1)%L,j]
        N[2] = Snew[i,(j-1)%L]
        N[3] = Snew[(i+1)%L,j]

        energydif = energy_dif(J, theta_new, theta_old, N)

        # probablity of getting new orientation
        prob = 0
        if energydif > 0:
            prob = np.exp(-beta * energydif)
        else:
            prob = 1

        val = np.random.rand()
        if  val <= prob:
            Snew[i,j] = theta_new
    return Snew

def PlotXY(S: np.array):
    L = len(S[:,1])
    Lrange = [k for k in range(L)]
    plt.imshow(S)
    plt.colorbar().set_label('Spin orientation [rad]')
    grid_x, grid_y = np.meshgrid(Lrange, Lrange)
    print(np.shape(S), np.shape(grid_x), np.shape(grid_y))
    plt.quiver(grid_x, grid_y, cos(S), sin(S), scale=70)
    plt.title('XY model state')
    plt.axis('off')
    #plt.show()

def EnergyXY(S: np.array, J: float):
    sum = 0
    S_Up = np.roll(S, 1, axis=(0,1))
    S_Right = np.roll(S, 1, axis=(1,0))
    sum += (cos(S - S_Up))
    sum += cos(S - S_Right)
    return (-J*np.sum(sum)) / len(S)**2
    
def CvXY(Energy: np.array, Temperature: np.array):
    Energy_shifted = np.roll(Energy, 1)
    Energy_dif = (Energy - Energy_shifted)[1:]
    Temp_shifted = np.roll(Temperature, 1)
    Temp_dif = (Temperature - Temp_shifted)[1:]
    return Energy_dif / Temp_dif

def magXY(S: np.array):
    sum_cos = np.sum(cos(S))**2
    sum_sin = np.sum(sin(S))**2
    return (1/(len(S)*4))(sum_cos + sum_sin)

def CorrXY(S: np.array):
    sum = 0
    Cr = np.zeros((len(S)//2,1))
    for r in range(1, len(Cr)):
        S_Up = np.roll(S, r, axis=(0,1))
        S_Right = np.roll(S, r, axis=(1,0))
        Cr[r] = np.sum(cos(S - S_Up)) + np.sum(cos(S - S_Right))
        Cr[r] /= len(S)**2
    return Cr

def VortXY(S: np.array):
    S_Up = np.roll(S, 1, axis=1)
    S_Right = np.roll(S, 1, axis=0)
    S_Ri_Up = np.roll(S, 1, axis=(0,1))

    # plt.subplot(1,2,1)
    # PlotXY(S_Up)

    # plt.subplot(1,2,2)
    # PlotXY(S)

    # plt.show()

    V1 = S_Right - S
    V2 = S_Ri_Up - S_Right
    V3 = S_Up - S_Ri_Up
    V4 = S - S_Up

    for Vi in [V1, V2, V3, V4]:
        Vi[np.abs(Vi) > pi] = -(np.sign(Vi[np.abs(Vi) > pi]) * 2*pi - Vi[np.abs(Vi) > pi])
        # Vi[np.abs(Vi) <= pi/4] = 0
    


    V = V1 + V2 + V3 + V4

    V /= 2*pi
    return V, (np.sum(V))

def VortPlotXY(V: np.array):
    L = len(V[:,1])
    Lrange = [k for k in range(L)]
    plt.imshow(V)
    grid_x, grid_y = np.meshgrid(Lrange, Lrange)
    print(np.shape(V), np.shape(grid_x), np.shape(grid_y))
    plt.title('Positions of vorticies')
    plt.axis('off')
    #plt.show()

S = InitSpins(L)
S = MetropolisXY(S, beta=1/0.02, J=J, numIter=int(1e5))
corr = CorrXY(S)
plt.plot(range(len(corr)), corr)
plt.show()

V, NumVort = VortXY(S)
print(V)
print(NumVort)
plt.subplot(1, 2, 1)
PlotXY(S)
plt.subplot(1, 2, 2)
VortPlotXY(V)
plt.show()

# betaA = 1/0.02
# betaB = 1/2
# numTPoints = 20
# KTPoints = np.linspace(1/betaA, 1/betaB, numTPoints)
# numMetropolis = 20
# avg_M = np.zeros((numTPoints, 1))
# avg_E = np.zeros((numTPoints, 1))
# avg_Corr = np.zeros((numTPoints, 1))
# for i in range(numTPoints):
#     for j in range(numMetropolis):
#         beta = 1/KTPoints[i]
#         S = MetropolisXY(S, beta, J, numIter=int(1e3))
#         avg_E[i] += EnergyXY(S, J)
#         avg_M[i] += magXY(S)
#         # avg_C[i] += CvXY()
#     avg_E[i] /= numMetropolis
#     avg_M[i] /= numMetropolis
#     # avg_C[i] /= numMetropolis
# avg_C = CvXY(avg_E, KTPoints)
# plt.subplot(1, 3, 1)
# plt.plot(KTPoints, avg_E)
# plt.xlabel('Kb*T')
# plt.ylabel('E')

# plt.subplot(1, 3, 2)
# plt.plot(KTPoints, avg_M)
# plt.xlabel('Kb*T')
# plt.ylabel('M')

# plt.subplot(1, 3, 3)
# plt.plot(KTPoints[:-1], avg_C)
# plt.xlabel('Kb*T')
# plt.ylabel('Cv')

# plt.show()

# corr = CorrXY(S)
# plt.plot(range(len(corr)), corr)
# plt.show()

# V, NumVort = VortXY(S)
# print(V)
# print(NumVort)
# VortPlotXY(S, V)