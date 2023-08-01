from numpy import cos, sin
import copy
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import matplotlib

# set font size
matplotlib.rcParams.update({'font.size': 18})

J = 1           # interaction strength - a.u.
L = 64          # sample side length - a.u.
n_theta = 16    # number of allowed orientations
theta_values = np.linspace(2*pi / n_theta, 2*pi, n_theta, dtype=float)  # allowed orientations

# initialize XY sample with random spins
def InitSpins(L: int):
    S = np.zeros([L, L], dtype=float)
    for i in range(L):
        for j in range(L):
            S[i,j] = np.random.choice(theta_values)
    return S

# Monte Carlo metropolis algorithm for cooling/heating sample
def MetropolisXY(S: np.array, beta: float, J: float, numIter: int, guesses_per_iter = 1):
    # get size of matrix
    L = len(S)

    # new S, to be returned
    Snew = copy.deepcopy(S)

    for x in range(numIter):
        i, j = np.random.randint(0,L,2) # choose random spin
        
        N = np.zeros((4, 1)) # Neighbors of spin
        theta_old = Snew[i,j]
        theta_new = np.random.choice(theta_values) # new random theta

        # get neighbor values
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

# Optimized metropolis algorithm
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

# Plot sample
def PlotXY(S: np.array):
    # side length
    L = len(S[:,1])
    Lrange = [k for k in range(L)]

    # color map
    color_set = ['lightcoral', 'moccasin', 'palegreen', 'aquamarine', 'lightskyblue', 'plum', 'lightcoral']
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", color_set)
    plot = plt.imshow(S, cmap=cmap) # show sample

    # add colorbar
    plt.colorbar().set_label('Spin orientation [rad]')

    # add spin arrows
    grid_x, grid_y = np.meshgrid(Lrange, Lrange)
    plt.quiver(grid_x, grid_y, cos(S), sin(S), scale=70)

    # title
    plt.title('XY model state')
    plt.axis('off')

# Get energy of sample
def EnergyXY(S: np.array, J: float):
    sum = 0
    # overlay neighbors on top of each spin
    S_Up = np.roll(S, 1, axis=0)
    S_Right = np.roll(S, 1, axis=1)

    # energy from neighbor interations
    sum += cos(S - S_Up)
    sum += cos(S - S_Right)
    return (-J*np.sum(sum)) / len(S)**2

# Get heat capacity at constant volume of sample
def CvXY(Energy: np.array, Temperature: np.array):
    # Calc difference in energy between points
    Energy_shifted = np.roll(Energy, 1)
    Energy_dif = (Energy - Energy_shifted)[1:]

    # Calc difference in Temperature between points
    Temp_shifted = np.roll(Temperature, 1)
    Temp_dif = (Temperature - Temp_shifted)[1:]

    # Return approximation of derrivative
    return Energy_dif / Temp_dif

# Get average magnetization of sample
def magXY(S: np.array):
    sum_cos = np.sum(cos(S))**2
    sum_sin = np.sum(sin(S))**2
    return (1/(len(S)**4)) * (sum_cos + sum_sin)

# Get correlation of sample
def CorrXY(S: np.array, fname: str):
    Cr = np.zeros((len(S)//2,1))

    # calc correlation for each distance r
    for r in range(1, len(Cr)):
        # r-distant neighbors
        S_Up = np.roll(S, r, axis=0)
        S_Right = np.roll(S, r, axis=1)
        Cr[r] = np.sum(cos(S - S_Up)) + np.sum(cos(S - S_Right))
        Cr[r] /= (len(S)**4)
    
    # remove first element
    Cr = Cr[1:]
    plt.plot(range(1, len(Cr)+1), Cr)
    plt.xlabel('Distance in cells')
    plt.ylabel('Correlation')
    plt.title('Correlation as a function of distance')
    savefig(fname)

# Find vortices and number of vortices in sample
def VortXY(S: np.array):
    # Get loops of spins
    S_Up = np.roll(S, 1, axis=1)
    S_Right = np.roll(S, 1, axis=0)
    S_Ri_Up = np.roll(S, 1, axis=(0,1))
    
    # Difference in spins along loops
    V1 = S_Right - S
    V2 = S_Ri_Up - S_Right
    V3 = S_Up - S_Ri_Up
    V4 = S - S_Up

    for Vi in [V1, V2, V3, V4]:
        Vi[np.abs(Vi) > pi] = -(np.sign(Vi[np.abs(Vi) > pi]) * 2*pi - Vi[np.abs(Vi) > pi])

    # Total difference in spins along loops
    V = V1 + V2 + V3 + V4

    V /= (2*pi)
    
    # Elements of V are vorticity around points, second item is number of vortices
    return V, (np.count_nonzero(np.abs(V) > 1/2))

# Plot sample with vortices marked
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
    plt.title(f'XY model state, with vortex positions marked\nTotal Vorticity = {V.sum()}')

    # set legend
    plt.legend(bbox_to_anchor=(-0.1, 1))

    # print total turbulence value

def savefig(fname: str):
    plt.savefig(fname + '.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(fname + '.svg', format='svg', bbox_inches='tight')
    plt.show()

# initialize
S = InitSpins(L)

PlotXY(S)
savefig('init')


# plot correlation
plt.grid()
CorrXY(S, 'corr_init')

# cool sample
S = MultMetropolisXY(S, beta=1/0.02, J=J, numIter=int(1e7), guessesPerIter=100)

PlotXY(S)
savefig('cold')


# plot correlation
plt.grid()
CorrXY(S, 'corr_cold')

# Plot vortices
V, NumVort = VortXY(S)
VortPlotXY(S, V)
savefig('vort_cold')

# Heat sample gradually - each step several times
# each step save energy and average magnetization
# and average over iterations of each step
betaA = 1/0.02
betaB = 1/2.0
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

# Calcualte heat capacity
avg_C = CvXY(avg_E, KTPoints)

# Plot Energy
plt.plot(KTPoints, avg_E)
plt.xlabel('$K_b T$')
plt.ylabel('$E$')
plt.title('Energy of the system as a function of temperature')
plt.grid()
savefig('energy')

# Plot magnetization
plt.plot(KTPoints, avg_M)
plt.xlabel('$K_b T$')
plt.ylabel('$\langle M \\rangle/N^2$')
plt.title('Average magnetization as a function of temperature')
plt.grid()
savefig('mag')

# Plot Heat capacity
plt.plot(KTPoints[:-1], avg_C)
plt.xlabel('$K_b T$')
plt.ylabel('$C_v$')
plt.title('Specific heat in constant volume as a function of temperature')
plt.grid()
savefig('cv')


# graph ln(num vortices) as function of 1/(k*temperature)
plt.plot(1/KTPoints, np.log(avg_NumVort))
plt.xlim(0.4,1.2)
plt.xlabel('$(kT)^{-1}$')
plt.ylabel('$ln($Number of vortices$)$')
plt.title('$ln($Number of vortices$)$ as a function of $beta=(kT)^{-1}$')
plt.grid()
savefig('log_num_vort')

# plot correlation
plt.grid()
corr = CorrXY(S, 'corr_hot')

# plot vortices when hot
V, NumVort = VortXY(S)
VortPlotXY(S, V)
savefig('vort_hot')

# plot sample with square edges - after long cooling
S = InitSpins(64)
S = MultMetropolisXY(S, beta=1/0.02, J=J, numIter=int(1e9), guessesPerIter=1000)
PlotXY(S)
savefig('square')
