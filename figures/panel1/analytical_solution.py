# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 09:38:10 2020

@author: Elham

# this program plots the shear stress profile inside a square mirochannel for different pressure values
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import time
start_time = time.time()

#----------general fonts for plots and figures----------
font = {'family' : 'sans-serif',
        'sans-serif':['Arial'],
        'weight' : 'normal',
        #'size'   : 20
        }
"""
plt.rc('font', **font)
plt.rc('legend', fontsize=16)
plt.rc('axes', titlesize=20)
"""

#----------create a new figure window----------
fig1=plt.figure(1, (10, 8))
border_width = 0.2
ax_size = [0+border_width, 0+border_width, 
           1-2*border_width, 1-2*border_width]
ax1 = fig1.add_axes(ax_size)

L=0.058 # length of the microchannel in meter
H= 200*1e-6 # height of the channel 
R_A=np.linspace(-100*1e-6,100*1e-6,501) # radial position
def stressfunc(R,P): # imputs (radial position & pressure)
    G=P/L #  pressure gradient
    pre_factor=(4*(H**2)*G)/(np.pi)**3
    u_primy=np.zeros(len(R))  
    sumi=0
    for i in range(0,len(R)): 
        for n in range(1,100,2): # sigma only over odd numbers
            u_primey=pre_factor *  ((-1)**((n-1)/2))*(np.pi/((n**2)*H))\
            * (math.sinh((n*np.pi*R[i])/H)/math.cosh(n*np.pi/2))
            sumi=u_primey + sumi
        u_primy[i]=sumi
        sumi=0
    stress= np.sqrt((u_primy)**2)
    return stress
stress_p1=stressfunc(R_A,-1*1e5)
stress_p2=stressfunc(R_A,-2*1e5)
stress_p3=stressfunc(R_A,-3*1e5)
R_A=R_A*1e6 # radial position in meter
plt.plot(R_A,stress_p1, '-',  linewidth=3,markeredgewidth=0, label="1 bar", zorder=3) #plot the  data
plt.plot(R_A,stress_p2, '-',  linewidth=3,markeredgewidth=0, label="2 bar", zorder=3) #plot the  data
plt.plot(R_A,stress_p3, '-',  linewidth=3,markeredgewidth=0, label="3 bar", zorder=3) #plot the  data

plt.xlabel('Distance from channel center ($\mu m$)')
plt.ylabel('Shear stress (Pa)')    
plt.legend(loc='upper left')

plt.savefig(r'shear_stress_versus_channel_position.png', dpi = 600, format='png')
plt.show()



