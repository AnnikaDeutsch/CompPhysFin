# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:32:54 2023

@author: Annika Deutsch
@date: 9/27/2023
@title: rk4.py
@CPU: 12th Gen Intel(R) Core(TM) i7-1255U
@Operating System: Windows 11 Home 64 bit 
@Interpreter and version no.: Python 3.10.9
"""

# imports
import numpy as np
from numba import njit 


#-----------------general ode solver with rk4------------------------------

@njit
def rk4(func, y, y0, t0, tf, nt, neqns):
    """

    Parameters
    ----------
    func : function
        User input function that is the rhs of our ODE, inputs are time (int)
        and y value (array of length neqns), returns array of length neqns
        evaluating that function at t, y.
    y : array
        2D NumPy array that will hold the entire y vector at each time step.
    y0 : array
        1D NumPy array that holds the y vector of the initial condition
    t0 : int
        initial time whose y value is known.
    tf : int
        final time we wish to get a solution for.
    nt : int
        number of time steps.
    neqns : int
        number of equations (ie order of our ODE/length of our solution vector
        at a given time).

    Returns
    -------
    y : array
        2D NumPy array holding solution vector for y at each time step.

    """
    h = tf / (nt - 1) # define step size h
    h2 = h / 2 # define half step size
    
    # instantiate k1-k4 (temp arrays)
    k1 = np.zeros(neqns)
    k2 = np.zeros(neqns)
    k3 = np.zeros(neqns)
    k4 = np.zeros(neqns)
    
    # instantiate time array
    t = np.zeros(nt)
    t[0] = t0
    
    # set initial condition for y
    # reminder of the shape of y: y = np.ndarray((nt, neqns), float)
    y[0] = y0
    
    for i in range(0, nt-1):
        k1 = h * func(t[i], y[i,:])
        k2 = h * func(t[i] + h2, y[i, :] + (k1[:] / 2))
        k3 = h * func(t[i] + h2, y[i, :] + (k2[:] / 2))
        k4 = h * func(t[i] + h, y[i, :] + k3[:])
        
        y[i+1,:] = y[i, :] + ((1/6) * (k1[:] + 2 * (k2[:] + k3[:]) + k4[:]))
        t[i+1] = t[i] + h
        
    return y, t
        