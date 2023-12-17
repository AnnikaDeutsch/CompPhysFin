# -*- coding: utf-8 -*-
"""
Created on Mon Oct 2 19:25:04 2023

@author: Annika Deutsch
@date: 10/15/2023
@title: HW 5
@CPU: 12th Gen Intel(R) Core(TM) i7-1255U
@Operating System: Windows 11 Home 64 bit 
@Interpreter and version no.: Python 3.10.9
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


N = 2
m_e = 5.9742e24
m = [m_e, 0.0123*m_e, 2.0e4] # m_e, m_m, m_sp
G = 6.674 * 10**(-11)

#--------- N-body dynamics rhs -------------------
def rhsNbody(t, z):
    """
    Parameters
    ----------
    t : int
        time step at which are evaluating the rhs.
    z : array
        z vector at that time, for which each of the N equations could be 
        dependent.

    Returns
    -------
    f : array
        vector of the same size as z, with the rhs of each of the N equations
        being solved, in the order derivs. of x, derivs. of y, derivs. of vx, 
        derivs. of vy
    """
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


#--------- test with just earth/moon orbit (no spacecraft) ---------
t0 = 0.0
tf = 2.36e6 # 27.3 days, roughly length of moon orbit
yinit_test = [0.0, 0.0, 0.0, 3.84e8, -12.593, 1020.0, 0.0, 0.0] 

track_paths_test = solve_ivp(rhsNbody, [t0, tf], yinit_test, rtol=1.0e-8)
print(track_paths_test)


plt.figure(1, figsize=(10,10)) # plot trajectories of e, m, neglect sp
plt.subplot(1,1,1, aspect='equal')
plt.plot(track_paths_test.y[0,:], track_paths_test.y[2,:], 'b', label='Earth')
plt.plot(track_paths_test.y[1,:], track_paths_test.y[3,:], 'g', label='Moon')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()



#--------- track orbits of earth/moon/spacecraft -------------------
N=3
t0 = 0.0
tf = 9.16e6
yinit = [0.0, 0.0, 1.10e7, 0.0, 3.84e8, 1.00e7, -12.593, 1020.0, 7.17e3, 0.0,
          0.0, 6.003e2] # initial conditions

track_paths = solve_ivp(rhsNbody, [t0, tf], yinit, rtol=1.0e-8)
print(track_paths)


plt.figure(2, figsize=(10,10)) # plot trajectories of e, m, sp
plt.subplot(1,1,1, aspect='equal')
plt.plot(track_paths.y[0,:], track_paths.y[3,:], 'b', label='Earth')
plt.plot(track_paths.y[1,:], track_paths.y[4,:], 'g', label='Moon')
plt.plot(track_paths.y[2,:], track_paths.y[5,:], 'm', label='Spacecraft')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()


plt.figure(3, figsize=(9,9)) # zoom in of where sp crosses m orbit
plt.subplot(1,1,1, aspect='equal')
plt.plot(track_paths.y[1,:], track_paths.y[4,:], 'g', label='Moon')
plt.plot(track_paths.y[2,:], track_paths.y[5,:], 'm', label='Spacecraft')
plt.xlim(left=1.5e8)
plt.ylim(top=-1.5e8, bottom=-4.5e8)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()


plt.figure(4, figsize=(9,9)) # zoom in of motion of earth com
plt.subplot(1,1,1, aspect='equal')
plt.plot(track_paths.y[0,:], track_paths.y[3,:], 'b', label='Earth')
plt.xlim(left=-7.5e6, right=7.5e6)
plt.ylim(top=1.25e7, bottom=-2.5e6)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

    
    
    
    