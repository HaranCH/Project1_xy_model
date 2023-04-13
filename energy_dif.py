import numpy as np
from numpy import cos

def energy_dif_pair(J: float, theta_new: float, theta_old: float, theta_neighbor: float):
    return -J * cos(theta_new - theta_neighbor) + J * cos(theta_old - theta_neighbor)

def energy_dif(J: float, theta_new: float, theta_old: float, neighbors: np.array):
    sum = 0
    for neighbor in neighbors:
        sum += energy_dif_pair(J, theta_new, theta_old, neighbor)
    return sum
