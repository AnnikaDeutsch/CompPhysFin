"""
Created on Sat 12/16 14:56 2023

@author: Annika Deutsch
@date: 12/16/2023
@title: run-sims
@CPU: 12th Gen Intel(R) Core(TM) i7-1255U
@Operating System: Windows 11 Home 64 bit 
@Interpreter and version no.: Python 3.10.9
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random
from mpl_toolkits.mplot3d import Axes3D

import os
import sys
import time
from small_cluster import Nbody_sim, Star_Cluster

cluster = Star_Cluster(3, 20.0,10.0)
cluster.RUN(2e10,0)






