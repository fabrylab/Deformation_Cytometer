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
print(filename_ex)
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
Frames=data[:,0] #frame number 
RP=data[:,3] #radial position 
longaxis=data[:,4] #Longaxis of ellipse
shortaxis=data[:,5] #Shortaxis of ellipse
Angle=data[:,6] #Shortaxis of ellipse
Irregularity=data[:,7] #ratio of circumference of the binarized image to the circumference of the ellipse 
Solidity=data[:,8] #percentage of binary pixels within convex hull polygon
Sharpness=data[:,9] #percentage of binary pixels within convex hull polygon
#%% select suitable cells
l_before = len(RP)
index = (Solidity>0.98) & (Irregularity < 1.05) & (np.abs(Sharpness) > 0.3)#select only the nices cells
RP = RP[index]
longaxis = longaxis[index]
shortaxis = shortaxis[index]
Angle = Angle[index]
Solidity = Solidity[index]
Irregularity = Irregularity[index]
Sharpness = Sharpness[index]
l_after = len(RP)
print('# frames =', Frames[-1], '   # cells total =', l_before, '   #cells sorted = ', l_after)
print('ratio #cells/#frames before sorting out = ',l_before/Frames[-1])

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
y_center = p[1]
print('center of channel is at psotion x = %.3f' % y_center)

#%% fitting deformation with stress stiffening equation, combining different pressures

fig3=plt.figure(3, (6, 6))
border_width = 0.1
ax_size = [0+2*border_width, 0+2*border_width, 
           1-3*border_width, 1-3*border_width]
ax3 = fig3.add_axes(ax_size)
ax3.set_xlabel('fluid shear stress $\u03C3$ (Pa)')
ax3.set_ylabel('cell strain  $\u03B5$')
fit=[]

pmax = 50*np.ceil(np.max(stress)//50)
ax3.set_xticks(np.arange(0,pmax+1,50))
ax3.set_xlim((-10,pmax+30))
ax3.set_ylim((-0.2,1.0))

# ----------plot strain versus stress data points----------
xy = np.vstack([stress,strain])
kd = gaussian_kde(xy)(xy)  
idx = kd.argsort()
x, y, z = stress[idx], strain[idx], kd[idx]
ax3.scatter(x, y, c=z, s=50, edgecolor='', alpha=1, cmap = 'viridis') #plot in kernel density colors e.g. viridis
#ax3.plot(stress,strain,'o', color = C1) #plot the data without kernel density colors

pstart=(1,.017) #initial guess
p, pcov = curve_fit(fitfunc, stress, strain, pstart) #do the curve fitting
#p, pcov = curve_fit(fitfunc, stress[RP<0], strain[RP<0], pstart) #do the curve fitting for one side only
err = (np.diag(pcov))**0.5 #estimate 1 standard error of the fit parameters

print("Fit Parameter: p1=%.3f +- %.3f       p2=%.3f +- %.3f" %(p[0],err[0],p[1],err[1]))  
# ----------plot the fit curve----------
xx = np.arange(np.min(stress),np.max(stress),0.1) # generates an extended array 
fit_real=fitfunc(xx,p[0],p[1])
ax3.plot(xx,(fitfunc(xx,p[0],p[1])), '--', color = 'black',   linewidth=2, zorder=3)
# ----------plot standard error of the fit function----------
y1 = fitfunc(xx,p[0]-err[0],p[1]-err[1])
y2 = fitfunc(xx,p[0]+err[0],p[1]+err[1])
ax3.plot(xx,y1, '--', color = 'black',   linewidth=1, zorder=3)
ax3.plot(xx,y2, '--', color = 'black',   linewidth=1, zorder=3)
plt.fill_between(xx, y1, y2, facecolor='gray', edgecolor= "none", linewidth = 0, alpha = 0.2)

# ----------plot the binned (averaged) strain versus stress data points----------
binwidth = 10 #Pa
bins = np.arange(0,pmax,binwidth)
bins = [0,10,20,30,40,50,75,100,125,150,200,250]
strain_av = []
stress_av = []
strain_err = []
for i in range(len(bins)-1):
    index = (stress > bins[i]) & (stress < bins[i+1])
    strain_av.append(np.mean(strain[index]))
    strain_err.append(np.std(strain[index])/np.sqrt(np.sum(index)))
    stress_av.append(np.mean(stress[index]))
ax3.errorbar(stress_av, strain_av,yerr = strain_err, marker='s', mfc='white', \
             mec='black', ms=7, mew=1, lw = 0, ecolor = 'black', elinewidth = 1, capsize = 3)    
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
#ax4.plot([y_center, y_center],[np.min(strain),np.max(strain)],'--', color = 'black') #fitted center line
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
ticks = np.arange(-100,101,bin_width)
ax5.set_xlim((-100,100))
labels = ticks.astype(int)
labels = labels.astype(str)
ax5.set_xticks(ticks)
ax5.set_xticklabels(labels) 
ax5.set_ylabel('# of cells')





