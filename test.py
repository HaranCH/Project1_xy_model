########################
# QUESTION 2
########################
# Solving a set of partial differential equations using the FTCS method

# imports
import numpy as np
import matplotlib.pyplot as plt

# constants, in S.I. units
L = 1.0
v = 100.0
d = 0.1
C = 1.0
sigma = 0.3
h = 1e-6 # timestep
N = 100 # number of grid points
a = L/(N-1) # distance between grid points

t_start = 0.0
t_end = 0.015
k = 6 # time points to plot
t_points = np.linspace(t_start, t_end, k)

# functions
def init_disp(x):
    return 0*x # *x necessay to return array

def init_vel(x):
    return (C * x * (L-x) / L**2) * np.exp(-(x-d)**2 / (2 * sigma**2))

# create arrays
x_points = np.linspace(0,L,N)
import matplotlib.animation as an

fig = plt.figure()

A = 50

phi = init_disp(x_points)
psi = init_vel(x_points)
t = t_start
def animate(i, x_points, phi, line):
    # calc new psi and phi values
    phi_new = phi + h * psi
    psi_new = psi[1:N-1]+ h*(v**2/a**2)*(phi[2:N] + phi[0:N-2] - 2*phi[1:N-1])
    phi[0:N] = phi_new
    psi[1:N-1] = psi_new
    
    line.set_data((x_points,A*phi))
    return line,

line, = plt.plot(x_points, A*phi)
line_an = an.FuncAnimation(fig, animate, frames=10000, fargs=(x_points, phi, line),
                                   interval=0.1, blit=True)
plt.xlabel('x')
plt.ylabel('$\phi(x)$')
line_an.save('phi.mp4')
plt.show()