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
from scipy import stats

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
pressure = float(filename_base[1])*1e5 # deduce applied pressure from file name (in Pa)

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

def fitfunc(x, p1,p2): #for stress versus strain
    return (1/p1)*np.log((x/p2)+1)

def fitfunc2(x, p1,p2): #for stress versus RP
    return p1*np.abs(x+p2)

#%% import raw data
data =np.genfromtxt(datafile,dtype=float,skip_header= 2)

#%% experimental raw data
RP=data[:,3] #radial position 
longaxis=data[:,4] #Longaxis of ellipse
shortaxis=data[:,5] #Shortaxis of ellipse
Angle=data[:,6] #Shortaxis of ellipse
Irregularity=data[:,7] #ratio of circumference of the binarized image to the circumference of the ellipse 
Solidity=data[:,8] #percentage of binary pixels within convex hull polygon

l_before = len(RP)
index = (Solidity>0.95) & (Irregularity < 1.06) #select only the nices cells
RP = RP[index]
longaxis = longaxis[index]
shortaxis = shortaxis[index]
Angle = Angle[index]
Solidity = Solidity[index]
Irregularity = Irregularity[index]
l_after = len(RP)
print('# cells before sorting out =', l_before, '   #cells after = ', l_after)

stress=stressfunc(RP*1e-6,-pressure)# analytical stress profile
#%%remove bias

index = np.abs(RP*Angle>0) 
LA = copy.deepcopy(longaxis)
LA[index]=shortaxis[index]
SA = copy.deepcopy(shortaxis)
SA[index]=longaxis[index]

#%%  deformation (True strain)
D = np.sqrt(LA * SA) #diameter of undeformed (circular) cell
strain = (LA - SA) / D

#%% center channel
pstart=(0.01,0) #initial guess
p, pcov = curve_fit(fitfunc2, RP, strain, pstart) #do the curve fitting

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

fig3=plt.figure(3, (6, 6))
border_width = 0.1
ax_size = [0+2*border_width, 0+2*border_width, 
           1-3*border_width, 1-3*border_width]
ax3 = fig3.add_axes(ax_size)
fit=[]

pmax = 50*np.ceil(np.max(stress)//50)
ax3.set_xticks(np.arange(0,pmax+1,50))
ax3.set_xlim((-10,pmax+30))
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

ax3.set_xlabel('fluid shear stress $\u03C3$ (Pa)')
ax3.set_ylabel('cell strain  $\u03B5$')
plt.show()

#%% plot strain versus radial position in channel
fig4=plt.figure(4, (6, 6))
border_width = 0.12
ax_size = [0+2*border_width, 0+2*border_width, 
           1-3*border_width, 1-3*border_width]
ax4 = fig4.add_axes(ax_size)
xy = np.vstack([RP,strain])
kd = gaussian_kde(xy)(xy)  
idx = kd.argsort()
x, y, z = RP[idx], strain[idx], kd[idx]
ax4.scatter(x, y, c=z, s=50, edgecolor='', alpha=1, cmap = 'viridis') #plot in kernel density colors
ax4.set_xticks(np.arange(-100,101,25))
ax3.set_ylim((-0.2,1.0))
ax4.set_xlabel('radial position in channel ($\u03BC m$)')
ax4.set_ylabel('cell strain  $\u03B5$')

#%% plot histogram of cell density in channel (margination)
fig5=plt.figure(5, (6, 3))
border_width = 0.12
ax_size = [0+2*border_width, 0+2*border_width, 
           1-3*border_width, 1-4*border_width]
ax5 = fig5.add_axes(ax_size)
bin_width = 25
hist, bin_edges = np.histogram(RP, bins=np.arange(-100 + bin_width/2, 101 - bin_width/2, bin_width), density=False)
plt.bar(bin_edges[:-1]+bin_width/2, hist, width=bin_width*0.8, edgecolor = 'black')
ax5.set_xlabel('radial position in channel ($\u03BC m$)')
ax5.set_xlim((-100,100))
ticks = np.arange(-100,101,bin_width)
labels = ticks.astype(int)
labels = labels.astype(str)
ax5.set_xticks(ticks)
ax5.set_xticklabels(labels) 
ax5.set_ylabel('# of cells')



#%% plot histogram of cell radius in channel (margination)
fig6=plt.figure(6, (6, 3))
border_width = 0.12
ax_size = [0+2*border_width, 0+2*border_width, 
           1-3*border_width, 1-4*border_width]
ax6 = fig6.add_axes(ax_size)
radius = np.sqrt(LA * SA / 4)  # radius is sqrt(a*b)
bin_width = 10
bins=np.arange(0, 101 - bin_width/2, bin_width)
bin_means, bin_edges, binnumber = stats.binned_statistic(abs(RP),radius, statistic='mean', bins=bins)
bin_std, bin_edges, binnumber = stats.binned_statistic(abs(RP),radius, statistic='std', bins=bins)
plt.bar(bin_edges[:-1]+bin_width/2,bin_means, width=bin_width*0.8, yerr=bin_std, edgecolor = 'black',label='binned statistic of data')
#plt.plot(abs(RP),radius, 'o', markerfacecolor='#1f77b4', markersize=3.0,markeredgewidth=0)
ax6.set_xlabel('radial position in channel ($\u03BC m$)')
ticks = np.arange(0 + bin_width/2,101,bin_width)
ax6.set_xlim((0,100))
labels = ticks.astype(int)
labels = labels.astype(str)
ax6.set_xticks(ticks)
ax6.set_xticklabels(labels) 
ax6.set_ylabel('Radius of cells')
#ax6.set_title('Bandpass image')
ax6.set_title('Original image')

#%% plot mean strain for position in channel
fig7=plt.figure(7, (6, 3))
border_width = 0.12
ax_size = [0+2*border_width, 0+2*border_width, 
           1-3*border_width, 1-4*border_width]
ax7 = fig7.add_axes(ax_size)
radius = np.sqrt(LA * SA/ 4)
bin_width = 10
bins=np.arange(0, 101 - bin_width/2, bin_width)
bin_means, bin_edges, binnumber = stats.binned_statistic(abs(RP),strain, statistic='mean', bins=bins)
bin_std, bin_edges, binnumber = stats.binned_statistic(abs(RP),strain, statistic='std', bins=bins)
plt.bar(bin_edges[:-1]+bin_width/2,bin_means, width=bin_width*0.8, yerr=bin_std, edgecolor = 'black',label='binned statistic of data')
#plt.plot(abs(RP),radius, 'o', markerfacecolor='#1f77b4', markersize=3.0,markeredgewidth=0)
ax7.set_xlabel('radial position in channel ($\u03BC m$)')
ticks = np.arange(0 + bin_width/2,101,bin_width)
ax7.set_xlim((0,100))
labels = ticks.astype(int)
labels = labels.astype(str)
ax7.set_xticks(ticks)
ax7.set_xticklabels(labels) 
ax7.set_ylabel('Mean Strain of cells')
ax7.set_title('Original image')
#ax7.set_title('Bandpass image')


