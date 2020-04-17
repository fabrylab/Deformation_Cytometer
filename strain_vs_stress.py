# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 08:42:39 2020

@author: Elham and Ben

# This program reads a txt file with the analyzed cell position, shape (semi-major and semi-minor axis etc.),
# computes the cell strain and the fluid shear stress acting on each cell,
# plots the data (strain versus stress) for each cell using a kernel density estimate for the datapoint color,
# and fits a stress stiffening equation to the data 
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit #, leastsq
import copy
import math
from tkinter import Tk
from tkinter import filedialog
from scipy.stats import gaussian_kde
import sys, os
#----------general fonts for plots and figures----------
font = {'family' : 'sans-serif',
        'sans-serif':['Arial'],
        'weight' : 'normal',
        'size'   : 18}
plt.rc('font', **font)
plt.rc('legend', fontsize=12)
plt.rc('axes', titlesize=18)
C1 = '#1f77b4'
C2 = '#ff7f0e'
C3 = '#9fc5e8'
C4='navajowhite'

#%% select result.txt file
root = Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing

datafile = filedialog.askopenfilename(title="select the data file",filetypes=[("txt file",'*.txt')]) # show an "Open" dialog box and return the path to the selected file
if datafile == '':
    print('empty')
    sys.exit()

filename_ex = os.path.basename(datafile)
filename_base, file_extension = os.path.splitext(filename_ex)
output_path = os.path.dirname(datafile)

#%% stress profile in channel
L=0.058 #length of the microchannel in meter
H= 200*1e-6 #height(and width) of the channel 
def stressfunc(R,P): # imputs (radial position and pressure)
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
    return stress #output atress profile

#%% import raw data
data =np.genfromtxt(datafile,dtype=float,skip_header= 1)

#%% experimental raw data
RP=data[:,3] #radial position 
longaxis=data[:,4] #Longaxis of ellipse
shortaxis=data[:,5] #Shortaxis of ellipse
Angle=data[:,6] #Shortaxis of ellipse
BBox_height=data[:,8] #Shortaxis of ellipse
stress=stressfunc(RP*1e-6,-3*1e5)# analytical stress profile
#%%remove bias

index = np.abs(RP*Angle>0) 
LA = copy.deepcopy(longaxis)
LA[index]=shortaxis[index]
SA = copy.deepcopy(shortaxis)
SA[index]=longaxis[index]

#%%  deformation (True strain)
D = np.sqrt(LA * SA) #diameter of undeformed (circular) cell
sigma_corr =  BBox_height / D
strain = (LA - SA) / D

'''
#%% plotig of deformation versus radial position
fig1=plt.figure(1, (8, 8))
border_width = 0.2
ax_size = [0+border_width, 0+border_width, 
           1-2*border_width, 1-2*border_width]
ax1 = fig1.add_axes(ax_size)

ax1.plot(RP,strain,'o')
ax1.set_xlabel('Distance from channel center ($\mu m$)')
ax1.set_ylabel('strain')
ax1.set_xlim(-100,100)
plt.show()
'''
#%% fitting deformation with stress stiffening equation, combining different pressures

def fitfunc(x, p1,p2): #for curve_fit
    return (1/p1)*np.log((x/p2)+1)

fig3=plt.figure(3, (8, 8))
border_width = 0.2
ax_size = [0+border_width, 0+border_width, 
           1-2*border_width, 1-2*border_width]
ax3 = fig3.add_axes(ax_size)
fit=[]

pmax = 50*np.ceil(np.max(stress)//50)
ax3.set_xticks(np.arange(0,pmax+50,50))
ax3.set_xlim((-10,pmax+50))
ax3.set_ylim((-0.2,1.0))

xy = np.vstack([stress,strain])
kd = gaussian_kde(xy)(xy)  
idx = kd.argsort()
x, y, z = stress[idx], strain[idx], kd[idx]
ax3.scatter(x, y, c=z, s=50, edgecolor='', alpha=1, cmap = 'viridis') #plot in kernel density colors
#ax3.plot(stress,strain,'o', color = C1) #plot the data without kernel density colors

pstart=(1,.017) #initial guess
p, pcov = curve_fit(fitfunc, stress, strain, pstart) #do the curve fitting
err = (np.diag(pcov))**0.5 #estimate 1 standard error of the fit parameters

print("Fit Parameter: p1=%.3f +- %.3f       p2=%.3f +- %.3f" %(p[0],err[0],p[1],err[1]))  
xx = np.arange(np.min(stress),np.max(stress),0.1) # generates an extended array 
fit_real=fitfunc(xx,p[0],p[1])
ax3.plot(xx,(fitfunc(xx,p[0],p[1])), '--', color = 'black',   linewidth=2, zorder=3)
# ----------plot standard error of the fit function----------
y1 = fitfunc(xx,p[0]-err[0],p[1]-err[1])
y2 = fitfunc(xx,p[0]+err[0],p[1]+err[1])
ax3.plot(xx,y1, '--', color = 'black',   linewidth=1, zorder=3)
ax3.plot(xx,y2, '--', color = 'black',   linewidth=1, zorder=3)
plt.fill_between(xx, y1, y2, facecolor='gray', edgecolor= "none", linewidth = 0, alpha = 0.2)

ax3.set_xlabel('$\u03C3$ (Pa)')
ax3.set_ylabel('strain')
plt.show()










