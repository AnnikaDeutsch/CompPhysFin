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
from mpl_toolkits.mplot3d import Axes3D

import os
import sys
import time


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
        self.G = 6.67430e-11
        self.msun = 2e30
        self.AU_SI = 1.496e11
        self.eta = 2e-4
        self.epsilon = self.AU_SI - 0.1*self.AU_SI

    def set_init_conds(self,n,posrange,velrange):
        '''set up the initial conditions for the globular cluster, return star_array'''
        stars = np.zeros((n, 5), dtype=object)
        for i in range(n):
            #build pos and vel arrays
            x = []
            v = []
            for j in range(3):
                x.append(random.uniform(-posrange*self.AU_SI, posrange*self.AU_SI))
                v.append(random.uniform(-velrange, velrange))
            stars[i][0] = x #pos
            stars[i][1] = v #vel
            stars[i][2] = [0,0,0,0] #time step, 3 is time of last eval
            stars[i][4] = [random.uniform(0.5*self.msun, 8*self.msun)] #mass
        return stars


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

    def G(self):
        '''units: m^3 * kg^-1 * s^-2'''
        return 6.67430e-11
    
    def msun(self):
        '''units: kg'''
        return 2e30
    
    def AU_SI(self):
        '''units: m'''
        return 1.496e11
    
    def eta(self):
        '''value of eta, permissible relative change of force in new step'''
        return 2e-4
    
    def epsilon(self):
        '''value of epsilon to avoid force going to infinity at 0'''
        return self.AU_SI() - 0.1*self.AU_SI()
    
    def starposvec(self,pos,i,j):
        '''returns position vector from star i to star j'''
        return np.subtract(self.pos[i][:],self.pos[j][:])
    
    def starvelvec(self,vel,i,j):
        '''returns velocity vector from star i to star j'''
        return np.subtract(self.vel[i][:],self.vel[j][:])
    
    def stardist(self,pos,i,j):
        '''calculates magnitude of distance between i and j'''
        dists = self.starposvec(self.pos,i,j)
        return np.sqrt(dists[0]**2 + dists[1]**2 + dists[2]**2)
    
    def dot(self, vec1, vec2):
        '''perform dot procuct'''
        return (vec1[0]*vec2[0]) + (vec1[1]*vec2[1]) + (vec1[2]*vec2[2])

    def f_ij(self,i,j):
        '''returns the force vector between just i and j'''
        pos = self.starposvec(self.pos,i,j)
        dist = self.stardist(self.pos,i,j)
        return -self.G*self.mass[j][0]*pos/((dist**2 + self.epsilon**2)**(3/2))

    def fd_ij(self,i,j):
        '''returns force derivative vector between just i and j'''
        vel = self.starvelvec(self.vel,i,j)
        pos = self.starposvec(self.pos,i,j)
        dist = self.stardist(self.pos,i,j)
        return -self.mass[j][0]*self.G*((vel/(dist**2 + self.epsilon**2)**(3/2))-((3*pos*self.dot(pos,vel))/((dist**2 + self.epsilon**2)**(5/2))))

    def find_min_dt(self):
        dt = self.dt[0][3]
        ti = self.time[0]
        min = dt + ti
        part = 0
        n = self.n_objects
        for i in range(n):
            dt = self.dt[0][3]
            ti = self.time[0]
            if dt+ti < min:
                part = i
                min = dt+ti
        return dt, part

    def f_alpha(self, alpha):
        '''calculate the force on just alpha using the updated positions of all of the other particles'''
        force = np.zeros(3)
        n = self.n_objects
        for j in range(n):
            if j!=alpha:
                vec = self.starposvec(self.pos,alpha,j)
                dist = self.stardist(self.pos,alpha,j)
                force = force + (-self.G * self.mass[j][0] * vec)/(np.sqrt((dist**2 + self.epsilon**2)**3))
        return force

    def force_fd(self,alpha):
        '''calculate force first derivative for a single particle'''
        force_fd = np.zeros(3)
        n = self.n_objects
        for j in range(n):
            if j!=alpha:
                posvec = self.starposvec(self.pos,alpha,j)
                velvec = self.starvelvec(self.vel,alpha,j)
                dist = self.stardist(self.pos,alpha,j)
                force_fd = force_fd + (-self.mass[j][0])*self.G*((velvec/((dist**2 + self.epsilon**2)**(3/2)))-((3*posvec*self.dot(posvec,velvec))/((dist**2 + self.epsilon**2)**(5/2))))
        return force_fd

    def force_sd(self, alpha):
        '''calculate force second derivative for a single particle'''
        force_sd = np.zeros(3)
        n = self.n_objects
        for j in range(n):
            if j!=alpha:
                posvec = self.starposvec(self.pos,alpha,j)
                velvec = self.starvelvec(self.vel,alpha,j)
                dist = self.stardist(self.pos,alpha,j)

                t1 = (self.f_ij(alpha,j)/((dist**2 + self.epsilon**2)**(3/2)))
                t2 = ((6*velvec*self.dot(posvec,velvec))/((dist**2 + self.epsilon**2)**(5/2)))
                t3 = (((3*posvec)/((dist**2 + self.epsilon**2)**(5/2)))*(((5*(self.dot(posvec,velvec))**2)/(dist**2 + self.epsilon**2))-(self.dot(velvec,velvec))-(self.dot(posvec,self.f_ij(alpha,j)))))

                force_sd = force_sd + (-self.mass[j][0])*self.G*(t1-t2+t3)
        return force_sd

    def force_td(self, alpha):
        '''calculate force third derivative for a single particle'''
        force_td = np.zeros(3)
        n = self.n_objects
        for j in range(n):
            if j!=alpha:
                posvec = self.starposvec(self.pos,alpha,j)
                velvec = self.starvelvec(self.vel,alpha,j)
                dist = self.stardist(self.pos,alpha,j)
                rsd = self.f_ij(alpha,j)
                rtd = self.fd_ij(alpha,j)

                t1 = rtd/((dist**2 + self.epsilon**2)**(3/2))
                t2 = (9*rsd*self.dot(posvec,velvec))/((dist**2 + self.epsilon**2)**(5/2))
                t3 = (velvec/((dist**2 + self.epsilon**2)**(5/2)))*((9*self.dot(velvec,velvec))+(9*self.dot(posvec,rsd))-((45*(self.dot(posvec,velvec))**2)/(dist**2 + self.epsilon**2)))
                t4 = (posvec/((dist**2 + self.epsilon**2)**(5/2)))*((3*self.dot(posvec,rtd))+(9*self.dot(velvec,rsd))-((45*self.dot(posvec,velvec)*self.dot(posvec,rsd))/(dist**2 + self.epsilon**2))-((45*self.dot(posvec,velvec)*self.dot(velvec,velvec))/(dist**2 + self.epsilon**2))+((105*self.dot(posvec,velvec)**3)/((dist**2 + self.epsilon**2)**2)))

                force_td = force_td + (-self.mass[j][0])*self.G*(t1-t2-t3+t4)
        return force_td

    def fhat1_calc(self,i):
        '''calculate fhat1 for particle i'''
        f0_fd = self.force_fd(i)
        f0_sd = self.force_sd(i)
        f0_td = self.force_td(i)
        dt3 = self.dt[i][2]
        return f0_fd - (0.5*f0_sd*dt3) + ((1/6)*f0_td*(dt3**2))

    def fhat2_calc(self,i,f0_sd,f0_td):
        '''calculate fhat2 for particle i'''
        dt2 = self.dt[i][1]
        dt3 = self.dt[i][2]
        return (0.5*f0_sd*((dt2+dt3)/(dt3))) - ((1/6)*f0_td*(dt2+(2*dt3))*(dt2+dt3)/(dt3))

    def fhat3_calc(self,i,f0_sd,f0_td):
        '''calculate fhat3 for particle i'''
        fhat2 = self.fhat2_calc(i,f0_sd,f0_td)
        dt1 = self.dt[i][0]
        dt2 = self.dt[i][1]
        dt3 = self.dt[i][2]
        return (fhat2*((dt2**2)-(dt1*dt3))/((dt2+dt3)*dt2*dt3)) + ((1/6)*f0_td*(dt1+dt2+dt3)*(dt1+dt2)/(dt2*dt3))

    def B_calc(self, i):
        '''calculate the prefactor of the linear term in the polynomial, eq4 in Aarseth'''
        return self.fhat1_calc(i)

    def C_calc(self, i, f0_sd,f0_td):
        '''calculate the prefactor of the quadratic term in the polynomial, eq4 in Aarseth'''
        dt3 = self.dt[i][2]
        dt2 = self.dt[i][1]
        return dt3*(self.fhat2_calc(i,f0_sd,f0_td))/(dt2+dt3)

    def D_calc(self, i,f0_sd,f0_td):
        '''calculate the prefactor of the cubic term in the polynomial, eq4 in Aarseth'''
        dt1 = self.dt[i][0]
        dt2 = self.dt[i][1]
        dt3 = self.dt[i][2]
        fhat2 = self.fhat2_calc(i, f0_sd,f0_td)
        fhat3 = self.fhat3_calc(i, f0_sd,f0_td)
        return (dt2*dt3*fhat3/((dt1+dt2+dt3)*(dt1+dt2))) - (((dt2**2)-(dt1*dt3))*fhat2/((dt1+dt2+dt3)*(dt1+dt2)*(dt2+dt3)))

    def dr(self, i, tr):
        '''calculate change in positon of particle i != alpha'''
        vel = self.vel[i]
        f0 = self.f_alpha(i)
        B = self.B_calc(i)
        return ([x*tr for x in vel]) + (0.5*f0*(tr**2)) + ((1/6)*B*(tr**3))

    def update_positions(self, alpha, t):
        '''update positions of all particles other than alpha'''
        n = self.n_objects
        for i in range(n):
            ti = self.time[i]
            if i != alpha:
                tr = t - ti
                self.pos[i] = self.pos[i] + self.dr(i, tr)

    def dr0(self, alpha, tr, f0_sd, f0_td):
        '''calculate change in position of alpha based on cubic expression for force'''
        vel = self.vel[alpha]
        F0 = self.f_alpha(alpha)
        B = self.B_calc(alpha)
        C = self.C_calc(alpha, f0_sd, f0_td)
        D = self.D_calc(alpha, f0_sd, f0_td)
        return ([x*tr for x in vel]) + (0.5*F0*(tr**2)) + ((1/6)*B*(tr**3)) + ((1/12)*C*(tr**4)) + ((1/20)*D*(tr**5))

    def update_alpha_pos(self, alpha, t, f0_sd, f0_td):
        '''update the position of the particle alpha based on the cubic expression for force'''
        ti = self.time[alpha]
        tr = t - ti
        self.pos[alpha] = self.pos[alpha] + self.dr0(alpha, tr, f0_sd, f0_td)

    def E_calc(self, i, f0_sd, f0_td, fn_sd, fn_td):
        '''calculate the prefactor of the quartic term in the polynomial, eq4 in Aarseth'''
        dt1 = self.dt[i][0]
        dt2 = self.dt[i][1]
        dt3 = self.dt[i][2]
        dt4 = self.dt[i][3]
        f4hat2 = self.fhat2_calc(i, f0_sd, f0_td)
        fhat2 = self.fhat2_calc(i, fn_sd, fn_td)
        D = self.D_calc(i, f0_sd, f0_td)
        return (((dt4*f4hat2/(dt3+dt4))-(dt3*fhat2/(dt2+dt3)))/((dt1+dt2+dt3+dt4)*(dt2+dt3+dt4))) - (D/(dt1+dt2+dt3+dt4))

    def update_alpha_pos_E(self, alpha, t, f0_sd, f0_td, fn_sd, fn_td):
        '''update the position of the particle alpha by adding E term'''  
        E = self.E_calc(alpha, f0_sd, f0_td, fn_sd, fn_td)
        ti = self.time[alpha]
        tr = t - ti
        add = (1/30)*E*(tr**6)
        self.pos[alpha] = self.pos[alpha] + add

    def dv0(self, alpha, tr, f0_sd, f0_td, fn_sd, fn_td):
        '''calculate change in velocity of alpha based on quartic expression for force'''
        F0 = self.f_alpha(alpha)
        B = self.B_calc(alpha)
        C = self.C_calc(alpha, f0_sd, f0_td)
        D = self.D_calc(alpha, f0_sd, f0_td)
        E = self.E_calc(alpha, f0_sd, f0_td, fn_sd, fn_td)
        return F0*tr + (0.5*B*(tr**2)) + ((1/3)*C*(tr**3)) + ((1/4)*D*(tr**4)) + ((1/5)*D*(tr**5))

    def update_alpha_vel(self, alpha, t, f0_sd, f0_td, fn_sd, fn_td):
        '''update the velocity of the particle alpha based on the quartic expression for force'''
        ti = self.time[alpha]
        tr = t - ti
        self.vel[alpha] = self.vel[alpha] + self.dv0(alpha, tr, f0_sd, f0_td, fn_sd, fn_td)

    def mag(self, force):
        '''calculate the magnitude of some 3 element vector'''
        return np.sqrt(force[0]**2 + force[1]**2 + force[2]**2)

    def calc_new_timestep(self, alpha, t, f0_sd, f0_td, fn_sd, fn_td):
        '''calculate the new timestep size for alpha, and reassign each of the old timesteps'''
        F0 = self.mag(self.f_alpha(alpha))
        B = self.mag(self.B_calc(alpha))
        D = self.mag(self.D_calc(alpha, f0_sd, f0_td))
        E = self.mag(self.E_calc(alpha, f0_sd, f0_td, fn_sd, fn_td))
        dt_alpha = self.dt[alpha][3]
        dtnew = (self.eta*(F0 + B*dt_alpha)/(D + E*dt_alpha))**(1/3)
        
        self.dt[alpha][0] = self.dt[alpha][1]
        self.dt[alpha][1] = self.dt[alpha][2]
        self.dt[alpha][2] = self.dt[alpha][3]
        self.dt[alpha][3] = dtnew
        self.time[alpha] = t
    
    def calculate_timestep(self, t):
        '''advance the system one time step'''
        # find the next particle and advance the time to that particle's dt
        dt_alpha, alpha = self.find_min_dt()
        time = t + dt_alpha

        # denote the force on alpha before moving anything, f0
        f0 = self.f_alpha(alpha)
        f0_fd = self.force_fd(alpha)
        f0_sd = self.force_sd(alpha)
        f0_td = self.force_td(alpha)

        # update the positions of all other particles
        self.update_positions(alpha, time)

        # more accurately update the position of alpha
        self.update_alpha_pos(alpha, time, f0_sd, f0_td)

        # evaluate the new force on alpha 
        fn = self.f_alpha(alpha)
        fn_fd = self.force_fd(alpha)
        fn_sd = self.force_sd(alpha)
        fn_td = self.force_td(alpha)

        # correct alpha position by adding E
        self.update_alpha_pos_E(alpha, time, f0_sd, f0_td, fn_sd, fn_td)

        # calculate the new velocity for alpha
        self.update_alpha_vel(alpha, time, f0_sd, f0_td, fn_sd, fn_td)

        # calculate new timestep for alpha
        self.calc_new_timestep(alpha, time, f0_sd, f0_td, fn_sd, fn_td)

        return dt_alpha
    
    # functions to initialize glob
    def init_force(self):
        '''calculate initial force on particles'''
        n=self.n_objects
        force = np.zeros((n,3))
        for i in range(n):
            for j in range(n):
                if i!=j:
                    vec = self.starposvec(self.pos,i,j)
                    dist = self.stardist(self.pos,i,j)
                    force[:][i] = force[:][i] + (-self.G*self.mass[j][0]*vec)/(np.sqrt((dist**2 + self.epsilon**2)**3))
        return force

    def init_force_mag(self,force):
        '''calculate inital force magnitude on each particle'''
        n=self.n_objects
        force_mag = np.zeros(n)
        for i in range(n):
            force_mag[i] = np.sqrt(force[i][0]**2 + force[i][1]**2 + force[i][2]**2)
        return force_mag

    def init_force_fd(self):
        '''calculate initial force first derivative'''
        n=self.n_objects
        force_fd = np.zeros((n,3))
        for i in range(n):
            for j in range(n):
                if i!=j:
                    posvec = self.starposvec(self.pos,i,j)
                    velvec = self.starvelvec(self.vel,i,j)
                    dist = self.stardist(self.pos,i,j)
                    force_fd[:][i] = force_fd[:][i] + (-self.mass[j][0]*self.G)*((velvec/((dist**2 + self.epsilon**2)**(3/2)))-((3*posvec*self.dot(posvec,velvec))/((dist**2 + self.epsilon**2)**(5/2))))
        return force_fd

    def init_force_sd(self):
        '''calculate initial force second derivative'''
        n=self.n_objects
        force_sd = np.zeros((n,3))
        for i in range(n):
            for j in range(n):
                if i!=j:
                    posvec = self.starposvec(self.pos,i,j)
                    velvec = self.starvelvec(self.vel,i,j)
                    dist = self.stardist(self.pos,i,j)

                    t1 = (self.f_ij(i,j)/((dist**2 + self.epsilon**2)**(3/2)))
                    t2 = ((6*velvec*self.dot(posvec,velvec))/(dist**5))
                    t3 = (((3*posvec)/((dist**2 + self.epsilon**2)**(5/2)))*(((5*(self.dot(posvec,velvec))**2)/(dist**2 + self.epsilon**2))-(self.dot(velvec,velvec))-(self.dot(posvec,self.f_ij(i,j)))))

                    force_sd[:][i] = force_sd[:][i] + (-self.mass[j][0]*self.G)*(t1-t2+t3)
        return force_sd

    def init_force_td(self):
        '''calculate initial force third derivative'''
        n=self.n_objects
        force_td = np.zeros((n,3))
        for i in range(n):
            for j in range(n):
                if i!=j:
                    posvec = self.starposvec(self.pos,i,j)
                    velvec = self.starvelvec(self.vel,i,j)
                    dist = self.stardist(self.pos,i,j)
                    rsd = self.f_ij(i,j)
                    rtd = self.fd_ij(i,j)

                    t1 = rtd/((dist**2 + self.epsilon**2)**(3/2))
                    t2 = (9*rsd*self.dot(posvec,velvec))/((dist**2 + self.epsilon**2)**(5/2))
                    t3 = (velvec/((dist**2 + self.epsilon**2)**(5/2)))*((9*self.dot(velvec,velvec))+(9*self.dot(posvec,rsd))-((45*(self.dot(posvec,velvec))**2)/(dist**2 + self.epsilon**2)))
                    t4 = (posvec/((dist**2 + self.epsilon**2)**(5/2)))*((3*self.dot(posvec,rtd))+(9*self.dot(velvec,rsd))-((45*self.dot(posvec,velvec)*self.dot(posvec,rsd))/(dist**2 + self.epsilon**2))-((45*self.dot(posvec,velvec)*self.dot(velvec,velvec))/(dist**2 + self.epsilon**2))+((105*self.dot(posvec,velvec)**3)/((dist**2 + self.epsilon**2)**2)))

                    force_td[:][i] = force_td[:][i] + (-self.mass[j][0]*self.G)*(t1-t2-t3-t4)
        return force_td
    
    def init_dt(self,stars):
        '''initialize beginning dt as laid out in Aarseth eq 7'''
        n=self.n_objects
        for i in range(n):
            f0 = self.init_force()
            f0_td = self.init_force_td()
            f0_mag = self.init_force_mag(f0)
            f03_mag = self.init_force_mag(f0_td)
            dt0 = (self.eta*((6*f0_mag[i])/(f03_mag[i])))**(1/3)
            self.dt[i] = [dt0, dt0, dt0, dt0]
            stars[i][2] = [dt0, dt0, dt0, dt0]
    



#--------------globular cluster class-----------------

class Star_Cluster():
    '''class that user interacts with to run sim'''
    def __init__(self, n, posrange, velrange):
        self.sim=Nbody_sim()
        star_array = self.sim.set_init_conds(n, posrange, velrange)
        self.sim.instantiate_stars(star_array)
        self.sim.init_dt(star_array)
        self.time=0
        self.is_new=True
        self.record=[]
        self.axeslims=posrange*self.sim.AU_SI

    def save_state(self):
        '''save the state of the system at some timestep'''
        self.record.append([self.time, self.sim.pos])

    def RUN(self,T,skip):
        ''' Runs the calculations, using the initial conditions, and saves them at each time step '''
        self.is_new=False
        self.save_state()
        initial_time=self.time
        last_snap_time=self.time
        print("Beginning Calculations")
        while self.time<initial_time+T:
            dt = self.sim.calculate_timestep(self.time)
            self.time=self.time+dt
            if self.time-last_snap_time>=skip:
                self.save_state()
                last_snap_time=self.time
        print("Calculations Finished")

    def find_trajectory(self, i):
        '''find the trajectory of a single star'''
        t = len(self.record)
        X = []
        Y = []
        Z = []
        for j in range(t):
            X.append(self.record[j][1][i][0])
            Y.append(self.record[j][1][i][1])
            Z.append(self.record[j][1][i][2])
        return X, Y, Z

    def display_3D(self):
        ''' Displays a 3D animation of the planetary system, with the star at the center and the planets' orbit around it using the computed data '''
        assert self.is_new==False,"ERROR : Cannot display System since no calculations have taken place."
        print("Displaying 3D System")
        # Creating the 3D figure
        fig=plt.figure(figsize=(12,12))
        ax=plt.axes(projection='3d')
        ax.set_title('t = 0.0 days')
        print(self.axeslims)
        ax.set_xlim3d(-self.axeslims,self.axeslims)
        ax.set_ylim3d(-self.axeslims,self.axeslims)
        ax.set_zlim3d(-self.axeslims,self.axeslims)
        # Accessing the saved data
        orbits=[]
        for i in range(self.sim.n_objects):
            X,Y,Z=self.find_trajectory(i)
            orbits.append(ax.plot(X,Y,Z))
        fig.show()
        


#--------------------------functions-----------------------------------


def session_name():
    ''' Names the files with the given date and time format: yyyy-mm-dd-hours-mins-secs '''
    t0=time.time()
    struct=time.localtime(t0)
    string=str(struct.tm_year)+'-'
    # MONTHS
    n_months=str(struct.tm_mon)
    if len(n_months)==1:
        n_months='0'+n_months
    string=string+n_months+'-'
    # DAYS
    n_days=str(struct.tm_mday)
    if len(n_months)==1:
        n_days='0'+n_days
    string=string+n_days+'-'
    # HOURS
    n_hours=str(struct.tm_hour)
    if len(n_hours)==1:
        n_hours='0'+n_hours
    string=string+n_hours+'-'
    # MINUTES
    n_mins=str(struct.tm_min)
    if len(n_mins)==1:
        n_mins='0'+n_mins
    string=string+n_mins+'-'
    # SECONDS
    n_secs=str(struct.tm_sec)
    if len(n_secs)==1:
        n_secs='0'+n_secs
    string=string+n_secs+'.txt'
    return string


