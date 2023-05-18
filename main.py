from numpy import cos, sin
import copy
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 24})

J = 1
L = 64
n_theta = 16
theta_values = np.linspace(2*pi / n_theta, 2*pi, n_theta, dtype=float)

def InitSpins(L: int):
    S = np.zeros([L, L], dtype=float)
    for i in range(L):
        for j in range(L):
            S[i,j] = np.random.choice(theta_values)
    return S

def MetropolisXY(S: np.array, beta: float, J: float, numIter: int, guesses_per_iter = 1):
    # get size of matrix
    L = len(S)

    # new S, to be returned
    Snew = copy.deepcopy(S)

    for x in range(numIter):
        i, j = np.random.randint(0,L,2)
        
        N = np.zeros((4, 1)) # Neighbors
        theta_old = Snew[i,j]
        theta_new = np.random.choice(theta_values)

        N[0] = Snew[i,(j+1)%L]
        N[1] = Snew[(i-1)%L,j]
        N[2] = Snew[i,(j-1)%L]
        N[3] = Snew[(i+1)%L,j]

        # calculate energy dif
        energy_new = np.sum(cos(N - theta_new))
        energy_old = np.sum(cos(N - theta_old))
        energy_dif = -J*(energy_new - energy_old)

        # probablity of getting new orientation
        prob = 0
        if energy_dif > 0:
            prob = np.exp(-beta * energy_dif)
        else:
            prob = 1

        val = np.random.rand()
        if  val <= prob:
            Snew[i,j] = theta_new
    return Snew

def MultMetropolisXY(S: np.array, beta: float, J: float, numIter: int = int(1e6), guessesPerIter:int = 1):
    # iterate less if multiple guesses in iter
    numIter //= guessesPerIter
    
    # get size of matrix
    L = len(S)

    # new S, to be returned
    Snew = copy.deepcopy(S)

    # iterate
    for x in range(numIter):

        # guess guessesPerIter coordinates
        i, j = np.random.randint(0, L, size=(2,guessesPerIter))
        
        # find old and guess new theta values for guessed coordinates
        theta_old = Snew[i,j]
        theta_new = np.random.choice(theta_values,size=guessesPerIter)

        # get neighbors (4 x guessesPerIter matrix)
        N = np.array([
            Snew[i,(j+1)%L],
            Snew[(i-1)%L,j],
            Snew[i,(j-1)%L],
            Snew[(i+1)%L,j]
        ])

        # print(f'N = {N}')
        # print(f'i = {i}')
        # print(f'j = {j}')
        # print(f'theta_new = {theta_new}')
        # print(f'theta_old = {theta_old}')

        # calculate energy dif
        energy_new = cos(N - np.tile(theta_old, (4,1)))
        energy_old = cos(N - np.tile(theta_new, (4,1)))
        energy_dif = -J*(energy_old - energy_new).sum(axis=0)

        # probablity of getting new orientation
        prob = np.exp(-beta * energy_dif)

        # get indices of guesses that should be kept
        prob[energy_dif < 0] = 1

        val = np.random.rand(guessesPerIter)

        to_change = prob - val

        Snew[i[to_change > 0], j[to_change > 0]] = theta_new[to_change > 0]
    return Snew

def PlotXY(S: np.array):
    L = len(S[:,1])
    Lrange = [k for k in range(L)]

    color_set = ['lightcoral', 'moccasin', 'palegreen', 'aquamarine', 'lightskyblue', 'plum', 'lightcoral']
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", color_set)
    plot = plt.imshow(S, cmap=cmap)

    plt.colorbar().set_label('Spin orientation [rad]')

    grid_x, grid_y = np.meshgrid(Lrange, Lrange)
    plt.quiver(grid_x, grid_y, cos(S), sin(S), scale=70)

    plt.title('XY model state')
    plt.axis('off')

def EnergyXY(S: np.array, J: float):
    sum = 0
    S_Up = np.roll(S, 1, axis=0)
    S_Right = np.roll(S, 1, axis=1)
    sum += cos(S - S_Up)
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
    return (1/(len(S)**4)) * (sum_cos + sum_sin)

def CorrXY(S: np.array, fname: str):
    Cr = np.zeros((len(S)//2,1))

    # calc correlation for each distance r
    for r in range(1, len(Cr)):
        # r-distant neighbors
        S_Up = np.roll(S, r, axis=0)
        S_Right = np.roll(S, r, axis=1)
        Cr[r] = np.sum(cos(S - S_Up)) + np.sum(cos(S - S_Right))
        Cr[r] /= ((len(S)**4 - len(S)**2)/2)
    
    # remove first element
    Cr = Cr[1:]
    plt.plot(range(1, len(Cr)+1), Cr)
    plt.xlabel('Distance in cells')
    plt.ylabel('Correlation')
    plt.title('Correlation as a function of distance')
    savefig(fname)

def VortXY(S: np.array):
    S_Up = np.roll(S, 1, axis=1)
    S_Right = np.roll(S, 1, axis=0)
    S_Ri_Up = np.roll(S, 1, axis=(0,1))

    V1 = S_Right - S
    V2 = S_Ri_Up - S_Right
    V3 = S_Up - S_Ri_Up
    V4 = S - S_Up

    for Vi in [V1, V2, V3, V4]:
        Vi[np.abs(Vi) > pi] = -(np.sign(Vi[np.abs(Vi) > pi]) * 2*pi - Vi[np.abs(Vi) > pi])

    V = V1 + V2 + V3 + V4

    V /= (2*pi)
    
    return V, (np.count_nonzero(np.abs(V) > 1/2))

def VortPlotXY(S: np.array, V: np.array):
    # plot the state
    PlotXY(S)

    # find positions of positive ang vortices
    pos = np.argwhere(V > 1/2)
    neg = np.argwhere(V < -1/2)

    # get positions as two arrays for scatter plot
    pos_y, pos_x = pos.T - 0.5
    neg_y, neg_x = neg.T - 0.5

    # plot scatter
    plt.scatter(pos_x, pos_y, marker='o', color='red', label='Positive vortex')
    plt.scatter(neg_x, neg_y, marker='D', color='blue', label = 'Negative vortex')

    # set title
    plt.title('XY model state, with vortex positions marked')

    # set legend
    plt.legend(bbox_to_anchor=(-0.1, 1))

def savefig(fname: str):
    plt.savefig(fname + '.png', format='png', dpi=300)
    plt.savefig(fname + '.svg', format='svg')
    plt.show()

# initialize
S = InitSpins(L)

PlotXY(S)
savefig('init')


# plot correlation
CorrXY(S, 'corr_init')

S = MultMetropolisXY(S, beta=1/0.02, J=J, numIter=int(1e7), guessesPerIter=100)

PlotXY(S)
savefig('cold')


# plot correlation
CorrXY(S, 'corr_cold')

V, NumVort = VortXY(S)
VortPlotXY(S, V)
savefig('vort_cold')


betaA = 1/0.02
betaB = 1/2
numTPoints = 20
KTPoints = np.linspace(1/betaA, 1/betaB, numTPoints)
numMetropolis = 200
avg_M = np.zeros((numTPoints, 1))
avg_E = np.zeros((numTPoints, 1))
avg_NumVort = np.zeros((numTPoints, 1))
avg_Corr = np.zeros((numTPoints, 1))
SHot = S
for i in range(numTPoints):
    for j in range(numMetropolis):
        beta = 1/KTPoints[i]
        SHot = MultMetropolisXY(S, beta, J, numIter=int(1e4), guessesPerIter=100)
        V, numVort = VortXY(SHot)

        avg_E[i] += EnergyXY(SHot, J)
        avg_M[i] += magXY(SHot)
        avg_NumVort[i] += numVort
    avg_E[i] /= numMetropolis
    avg_M[i] /= numMetropolis
    avg_NumVort[i] /= numMetropolis
    S = SHot

avg_C = CvXY(avg_E, KTPoints)

plt.plot(KTPoints, avg_E)
plt.xlabel('$K_b T$')
plt.ylabel('$E$')
savefig('energy')


plt.plot(KTPoints, avg_M)
plt.xlabel('$K_b T$')
plt.ylabel('$\langle M \\rangle/N^2$')
savefig('mag')


plt.plot(KTPoints[:-1], avg_C)
plt.xlabel('$K_b T$')
plt.ylabel('$C_v$')
savefig('cv')


plt.plot(KTPoints, avg_NumVort)
plt.xlabel('kT')
plt.ylabel('Number of vortices')
plt.title('Number of vortices as a function of temperature')
savefig('num_vort')


plt.semilogy(KTPoints, avg_NumVort)
plt.xlabel('log kT')
plt.ylabel('log Number of vortices')
plt.title('log-log graph: Number of vortices as a function of temperature')
savefig('log_num_vort')


# plot correlation
corr = CorrXY(S, 'corr_hot')


V, NumVort = VortXY(S)
VortPlotXY(S, V)
savefig('vort_hot')


# graph num vort as function of temp
betaA = 1/0.02
betaB = 1/2
numTPoints = 20
KTPoints = np.linspace(1/betaA, 1/betaB, numTPoints)