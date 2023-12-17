"""
Created on Sat 12/16 14:54 2023

@author: Annika Deutsch
@date: 12/16/2023
@title: small-cluster
@CPU: 12th Gen Intel(R) Core(TM) i7-1255U
@Operating System: Windows 11 Home 64 bit 
@Interpreter and version no.: Python 3.10.9
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random

import os
import sys


#-------------simulation class---------------

class Nbody_sim():
    '''class that performs calculations to run N-body sim'''
    def __init__(self):
        self.pos=[]
        self.vel=[]
        self.dt=[]
        self.time=[]
        self.mass=[]
        self.n_objects=0

    def instantiate_stars(self, stars):
        '''set up array storing cluster info'''
        n = len(stars)
        if self.n_objects==0:
            self.pos=np.zeros(shape=(n,3))
            self.vel=np.zeros(shape=(n,3))
            self.dt=np.zeros(shape=(n,4))
            for i in range(n):
                self.pos[i]=stars[i][0]
                self.vel[i]=stars[i][1]
                self.dt[i]=stars[i][2]
                self.time.append(stars[i][3])
                self.mass.append(stars[i][4])
        else:
            new_X=np.zeros(shape=(self.n_objects+n,3))
            new_V=np.zeros(shape=(self.n_objects+n,3))
            new_dt=np.zeros(shape=(self.n_objects+n,4))
            new_X[0:self.n_objects]=self.pos[0:self.n_objects]
            new_V[0:self.n_objects]=self.vel[0:self.n_objects]
            new_dt[0:self.n_objects]=self.dt[0:self.n_objects]
            self.pos=new_X
            self.vel=new_V
            self.dt=new_dt
            for i in range(n):
                self.pos[self.n_objects+i]=stars[i][0]
                self.vel[self.n_objects+i]=stars[i][1]
                self.dt[self.n_objects+i]=stars[i][2]
                self.time.append(stars[i][3])
                self.mass.append(stars[i][4])
        self.n_objects=self.n_objects+n  

    def gravconst_SI(self):
        '''units: m^3 * kg^-1 * s^-2'''
        return 6.67430e-11
    
    def sun_mass_SI(self):
        '''units: kg'''
        return 2e30
    
    def ast_unit_SI(self):
        '''units: m'''
        return 1.496e11
    



#--------------globular cluster class-----------------

class Star_Cluster():
    '''class that user interacts with to run sim'''
    def __init__(self, star_array):
        self.sim=Nbody_sim()
        self.sim.instantiate_stars(star_array)



