"""
Created on Sat 12/16 14:52 2023

@author: Annika Deutsch
@date: 12/16/2023
@title: n-body
@CPU: 12th Gen Intel(R) Core(TM) i7-1255U
@Operating System: Windows 11 Home 64 bit 
@Interpreter and version no.: Python 3.10.9
"""

import numpy as np
from numba import njit 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random

"""
goal is to write a function that takes a user specified number of bodies, generate random inital positions and velocities for them,
and uses N-body dynamics to track the time evolution of the system
"""
# define any universal constants at top
msun = 2e30 # mass of sun in kg
G = 6.674 * 10**(-11) # SI units

def generate_init_array(N, posrange=40.0, velrange=2.0):
    """
    Parameters
    ----------
    N : int
        number of bodies in system
    posrange : float
        range of x and y
    velrange : float
        range of intial vx and vy
    
    Returns
    -------
    init : array
        array of length 4*N. The order of the array goes [x0,...,xN-1,y0,...,yN-1,vx0,...vxN-1,vy0,vyN-1]
    """
    init = np.zeros(4*N)
    # initialize x and y positions
    for i in range(2*N):
        init[i] = random.uniform(-posrange, posrange)
    # initialize vx and vy
    for j in range(2*N):
        init[j+(2*N)] = random.uniform(-velrange, velrange)
    return init

def generate_mass_array(N, massmin=0.5*msun, massmax=8.0*msun):
    """
    Parameters
    ----------
    N : int
        number of bodies in system
    
    Returns
    -------
    m : array
        array of length N containing masses of each star
    """
    m = np.zeros(N)
    for i in range(N):
        m[i] = random.uniform(massmin, massmax)
    return m

@njit
def Nbodymotion(N, trange):
    m = generate_mass_array(N)
    init = generate_init_array(N)

    def rhsNbody(t, z):
    # define m and N and G
        f = np.zeros(4*N)
        for j in range(N):
            sum1 = 0
            sum2 = 0
            for i in range(N):
                if i != j:
                    dx = z[i] - z[j] # x values in z array
                    dy = z[i+N] - z[j+N] # y values in z array
                    dist = np.sqrt(dx**2 + dy**2)
                    sum1 = sum1 + (m[i] * dx / (dist**3))
                    sum2 = sum2 + (m[i] * dy / (dist**3))
                    
            f[j] = z[j + (2*N)]
            f[j+N] = z[j + (3*N)]
            f[j+(2*N)] = G * sum1
            f[j+(3*N)] = G * sum2 
        return f

    track_paths = solve_ivp(rhsNbody, trange, init, rtol=1.0e-8)
    return track_paths

def plotmotion(N, track_paths):
    plt.figure(1, figsize=(10,10))
    plt.subplot(1,1,1, aspect='equal')
    for i in range(N):
        plt.plot(track_paths.y[i,:], track_paths.y[i+N])
    plt.xlabel('x')
    plt.ylabel('y')
