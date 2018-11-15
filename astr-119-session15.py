#!/usr/bin/env python
# coding: utf-8

# In[2]:


# # Solar System Model







# ### Create a simple solar system model



get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple


# ### Define a planet class



class planet():
    "A planet in our solar system"
    def _init_(self,semimajor,eccentricity):
        self.x = np.array(2)   #x and y position
        self.v = np.array(2)   #x and y velocity
        self.a_g = np.array(2) #x and y acceleration
        self.t = 0.0           #current time
        self.dt = 0.0           #current timestep
        self.a = semimajor     #semimajor axis of the orbit
        self.e = eccentricity  #eccentricity of the orbit
        self.istep = 0         #current integer timestepl
        delf.name = ""         #NAME FOR THE PLANET


# ### Define a dictionary with some constants



solar_system = {"M_sun":1.0, "G":39.4784176043574320}


# ### Define some functions for setting circular velocity and acceleration



def SolarCircularVelocity(p,):
    
    G = solar_system["G"]
    M = solar_system["M_sun"]
    r = ( p.x[0]**2 + p.x[1]**2 )**0.5
    
    #return the circular velocity
    return (G*M/r)**0.5


def SolarGravitationalAcceleration(p):
    
    G = solar_system["G"]
    M = solar_system["M_sun"]
    r = ( p.x[0]**2 + p.x[1]**2 )**0.5
    
    #acceleration in AU/yr/yr
    a_grav = -1.0*G*M/r**2
    
    #Find the angle at this position
    if(p.x[0] == 0.0):
        if(p.x[1]>0.0):
            theta = 0.5*np.pi
        else:
            theta = 1.5*np.pi
    else:
        theta = np.arctan2(p.x[1], p.x[0])
        
    #Set the x and y components of the velocity
    p.a_g[0] = a_grav*np.cos(theta)
    p.a_g[1] = a_grav*np.sin(theta)
    return a_grav*np.cos(theta), a_grav*np.sin(theta)


# ### Compute the timestep


def calc_dt(p):
    
    #integration tolerance
    ETA_TIME_STEP = 0.0004
    
    #COMPUTE TIMESTEP
    eta = ETA_TIME_STEP
    v = (p.v[0]**2 + p.v[1]**2)**0.5
    a = (p.a_g[0]**2 + p.a_g[1]**2)**0.5
    dt = eta*np.fmin(1./np.fabs(v),1./fabs(a)**0.5)
    
    return dt


# ### Define the initial conditions

# In[3]:


def SetPlanet(p,i):
    
    AU_in_km = 1.495979e8 #an AU in km
    
    #circular velocity
    v_c = 0.0       #circular velocity in AU/yr
    v_e - 0.0        #velocity at perihelion in AU/yr
    
    #planet by planet initial conditions
    
    #Mercury
    if(i==0):
        #semi-major axis in AU
        p.a = 57909227.0/AU_in_km
        
        #eccentricity
        p.e = 0.20563593
        
        #name
        p.name = "Mercury"
        
    #Venus'
    elif(i==1):
        #semi-major axis in AU
        p.a = 108209475.0/AU_in_km
        
        #eccentricity
        p.e = 0.00677672
        
        #name
        p.name = "Venus"
        
    #Earth
    elif(i==2):
        #semi-major axis in AU
        p.a = 1.0
        
        #eccentricity
        p.e = 0.01671123
        
        #name
        p.name = "Earth"
        
    #set remaining properties
    p.t = 0.0
    p.x[0] = p.a*(1.0-p.e)
    p.x[1] = 0.0
    
    #get equiv circular velocity
    v_c = SolarCircularVelocity(p)
    
    #velocity at perihelion
    v_e = v_c*(1 + p.e)**0.5
    
    #set velocity
    p.v[0] = 0.0       #no x velocity at perihelion
    p.v[1] = v_e       #y velocity at perihelion (counter clockwise)
    
    #calculate gravitational acceleration from Sun
    p.a_g = SolarGravitationalAcceleration(p)
    
    #set timestep
    p.dt = calc_dt(p)


# ### Write a leapfrom integrator

# In[4]:


def x_first_step(x_i, v_i, a_i, dt):
    #x_1/2 = x_0 + 1/2 v_0 Delta_t + 1/4 a_0 Delta t^2
    return x_i + 0.5*v_i*dt + 0.25*a_i*dt**2


# In[9]:


def v_full_step(v_i, a_ipoh, dt):
    #v_i+1 = v_i + a_i+1/2 Delta t
    return v_i + a_ipoh*dt

#ipoh = i Plus One Half


# In[10]:


def x_full_step(x_ipoh, v_ipl, a_ipoh, dt):
    #x_3/2 = x_1/2 + v_i+1 Delta t
    return x_ipoh + v_ipl*dt


# ### Write a function to save the data to file

# In[ ]:


def SaveSolarSystem(p, n_planets, t, dt, istep, ndim):
    
    #loop over the number of planets
    for i in range(n_planets):
        
        #define a filename
        fname = "planet.%s.txt" % p[i].name
        
        if(istep==0):
            #create the file on the first timestep
            fp = open(fname, "w")
        else:
            #append the file on subsequent timesteps
            fp = open(fname, "a")
            
        #compute the drifted properties of the planet
        v_drift = np.zeros(ndim)
        
        for k in range(ndim):
            v_drift[k] = p[i].v[k] + 0.5*p[i].a_g[k]*p[i].dt
            
        #write the data to file
        

