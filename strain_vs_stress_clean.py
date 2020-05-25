# -*- coding: utf-8 -*-
"""
Created on Tue May 22 2020

@author: Ben

# This program reads a txt file with the analyzed cell position, shape (semi-major and semi-minor axis etc.),
# computes the cell strain and the fluid shear stress acting on each cell,
# plots the data (strain versus stress) for each cell using a kernel density estimate for the datapoint color,
# and fits a stress stiffening equation to the data 
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.optimize import curve_fit #, leastsq
import copy
import math
from tkinter import Tk
from tkinter import filedialog
from scipy.stats import gaussian_kde
import sys, os
import configparser
#----------general fonts for plots and figures----------
font = {'family' : 'sans-serif',
        'sans-serif':['Arial'],
        'weight' : 'normal',
        'size'   : 18}
plt.rc('font', **font)
plt.rc('legend', fontsize=12)
plt.rc('axes', titlesize=18)
C0 = '#1f77b4'
C1 = '#ff7f0e'
C2 = '#2ca02c'
C3 = '#d62728'

#%% select result.txt file
root = Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing

datafile = filedialog.askopenfilename(title="select the data file",filetypes=[("txt file",'*_result.txt')]) # show an "Open" dialog box and return the path to the selected file
if datafile == '':
    print('empty')
    sys.exit()

filename_ex = os.path.basename(datafile)
print(filename_ex)
filename_base, file_extension = os.path.splitext(filename_ex)
output_path = os.path.dirname(datafile)
#pressure = float(filename_base[1])*1e5 # deduce applied pressure from file name (in Pa)
filename_config = output_path + '/'+ filename_base[:-7] + '_config.txt' #remove _result from the filename and add _config.txt

#%% open and read the config file
config = configparser.ConfigParser()
config.read(filename_config) 
pressure=float(config['SETUP']['pressure'].split()[0])*1000 #applied pressure (in Pa)
channel_width=float(config['SETUP']['channel width'].split()[0])*1e-6 #in m
#channel_width=196*1e-6 #in m
channel_length=float(config['SETUP']['channel length'].split()[0])*1e-2 #in m
framerate=float(config['CAMERA']['frame rate'].split()[0]) #in m

magnification=float(config['MICROSCOPE']['objective'].split()[0])
coupler=float(config['MICROSCOPE']['coupler'] .split()[0])
camera_pixel_size=float(config['CAMERA']['camera pixel size'] .split()[0])

pixel_size=camera_pixel_size/(magnification*coupler) # in micrometer

#%% stress profile in channel
L=channel_length #length of the microchannel in meter
H= channel_width #height(and width) of the channel 
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

def fitfunc(x, p0,p1,p2): #for stress versus strain
    return (1/p0)*np.log((x/p1)+1) + p2

def velfit(r, p0,p1): #for stress versus strain
    R = channel_width/2 * 1e6
    return p0*(1 - np.abs(r/R)**p1)

#%% import raw data
data =np.genfromtxt(datafile,dtype=float,skip_header= 2)

#%% experimental raw data
Frames=data[:,0] #frame number 
X=data[:,1] #x-position in Pixes
Y=data[:,2] #x-position in Pixes
RP=data[:,3] #radial position 
longaxis=data[:,4] #Longaxis of ellipse
shortaxis=data[:,5] #Shortaxis of ellipse
Angle=data[:,6] #Shortaxis of ellipse
Irregularity=data[:,7] #ratio of circumference of the binarized image to the circumference of the ellipse 
Solidity=data[:,8] #percentage of binary pixels within convex hull polygon
Sharpness=data[:,9] #percentage of binary pixels within convex hull polygon

#%% compute velocity profile
y_pos = []
vel = []
for i in range(len(Frames)-10):
    for j in range(10):
        if np.abs(RP[i]-RP[i+j])< 1 and Frames[i+j]-Frames[i]==1 and np.abs(longaxis[i+j]-longaxis[i])<1 and np.abs(shortaxis[i+j]-shortaxis[i])<1 and np.abs(Angle[i+j]-Angle[i])<5:
            v = (X[i+j]-X[i])*pixel_size*framerate/1000 #in mm/s
            if v > 0:
                y_pos.append(RP[i])
                vel.append(v) 

#%% select suitable cells
l_before = len(RP)
index = (Solidity>0.96) & (Irregularity < 1.05) & (np.abs(Sharpness) > 0.3)#select only the nice cells
RP = RP[index]
longaxis = longaxis[index]
shortaxis = shortaxis[index]
Angle = Angle[index]
Solidity = Solidity[index]
Irregularity = Irregularity[index]
Sharpness = Sharpness[index]
l_after = len(RP)
print('# frames =', Frames[-1], '   # cells total =', l_before, '   #cells sorted = ', l_after)
print('ratio #cells/#frames before sorting out = %.2f \n' % float(l_before/Frames[-1]))

#%% find center of the channel 
no_right_cells = 0
center = 0
for i in np.arange(-50,50,0.1):
    n = np.sum(np.sign(-(RP+i)*Angle))
    if n>no_right_cells:
        center = i
        no_right_cells = n
print('center channel position at y = %.1f  \u03BCm' % -center)
RP = RP + center
if np.max(RP)> 1e6*channel_width/2:
    RP = RP - (np.max(RP)-1e6*channel_width/2)  #this is to ensure that the maximum or minimum radial position
if np.min(RP) < -1e6*channel_width/2:           #of a cell is not outsied the channel
    RP = RP - (np.min(RP)+1e6*channel_width/2)    

fig1=plt.figure(1, (6, 4))
border_width = 0.1
ax_size = [0+2*border_width, 0+2*border_width, 
           1-3*border_width, 1-3*border_width]
ax1 = fig1.add_axes(ax_size)
ax1.set_xlabel('channel position ($\u00B5 m$)')
ax1.set_ylabel('flow speed (mm/s)')  
ax1.set_ylim((0,1.1*np.max(vel)))  
y_pos = y_pos+center
ax1.plot(y_pos, vel, '.')    
p, pcov = curve_fit(velfit, y_pos, vel, [np.max(vel),0.9]) #fit a parabolic velocity profile 
r = np.arange(-channel_width/2*1e6,channel_width/2*1e6,0.1) # generates an extended array 
ax1.plot(r,velfit(r,p[0],p[1]), '--', color = 'gray',   linewidth=2, zorder=3)
print('v_max = %5.2f mm/s   profile stretch exponent = %5.2f\n' %(p[0],p[1]))

#%%  compute stress profile, cell deformation (true strain), and diameter of the undeformed cell
stress=stressfunc(RP*1e-6,-pressure)# compute analytical stress profile
D = np.sqrt(longaxis * shortaxis) #diameter of undeformed (circular) cell
strain = (longaxis - shortaxis) / D

#%% fitting deformation with stress stiffening equation
fig2=plt.figure(2, (6, 6))
border_width = 0.1
ax_size = [0+2*border_width, 0+2*border_width, 
           1-3*border_width, 1-3*border_width]
ax2 = fig2.add_axes(ax_size)
ax2.set_xlabel('fluid shear stress $\u03C3$ (Pa)')
ax2.set_ylabel('cell strain  $\u03B5$')
fit=[]

pmax = 50*np.ceil((np.max(stress)+50)//50)
ax2.set_xticks(np.arange(0,pmax+1, 50))
ax2.set_xlim((-10,pmax))
ax2.set_ylim((-0.2,1.0))

# ----------plot strain versus stress data points----------
xy = np.vstack([stress,strain])
kd = gaussian_kde(xy)(xy)  
idx = kd.argsort()
x, y, z = stress[idx], strain[idx], kd[idx]
ax2.scatter(x, y, c=z, s=50, edgecolor='', alpha=1, cmap = 'viridis') #plot in kernel density colors e.g. viridis
#ax2.plot(stress,strain,'o', color = C1) #plot the data without kernel density colors

pstart=(3.5,8,0) #initial guess
p, pcov = curve_fit(fitfunc, stress, strain, pstart) #do the curve fitting
#p, pcov = curve_fit(fitfunc, stress[RP<0], strain[RP<0], pstart) #do the curve fitting for one side only
err = (np.diag(pcov))**0.5 #estimate 1 standard error of the fit parameters
cov_ap = pcov[0,1] # cov between alpha and prestress
cov_ao = pcov[0,2] # cov between offset and alpha 
cov_po = pcov[1,2] # cov between prestress and offset
se01 = np.sqrt((p[1]*err[0])**2 + (p[0]*err[1])**2 + 2*p[0]*p[1]*cov_ap) 
print('pressure = %5.1f kPa' % float(pressure/1000))
print("p0 =%5.2f   p1 =%5.1f Pa   p0*p1=%5.1f Pa   p2 =%4.3f" %(p[0],p[1],p[0]*p[1],p[2]))

print("se0=%5.2f   se1=%5.1f Pa   se0*1=%5.1f Pa   se2=%4.3f" %(err[0],err[1],err[0]*err[1],err[2]))

# ----------plot the fit curve----------
xx = np.arange(np.min(stress),np.max(stress),0.1) # generates an extended array 
ax2.plot(xx,(fitfunc(xx,p[0],p[1], p[2])), '-', color = 'black',   linewidth=2, zorder=3)
# ----------plot standard error of the fit function----------
dyda = -1/(p[0]**2)*np.log(xx/p[1]+1) #strain derivative with respect to alpha
dydp = -1/p[0]*xx/(xx*p[1]+p[1]**2)   #strain derivative with respect to prestress
dydo = 1                              #strain derivative with respect to offset
vary = (dyda*err[0])**2 + (dydp*err[1])**2 + (dydo*err[2])**2 + 2*dyda*dydp*cov_ap + 2*dyda*dydo*cov_ao + 2*dydp*dydo*cov_po
y1 = fitfunc(xx,p[0],p[1],p[2])-np.sqrt(vary)
y2 = fitfunc(xx,p[0],p[1],p[2])+np.sqrt(vary)
plt.fill_between(xx, y1, y2, facecolor='gray', edgecolor= "none", linewidth = 0, alpha = 0.5)

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
ax2.errorbar(stress_av, strain_av,yerr = strain_err, marker='s', mfc='white', \
             mec='black', ms=7, mew=1, lw = 0, ecolor = 'black', elinewidth = 1, capsize = 3)    
#ax1.set_xlim((0.5,pmax))
#plt.xscale('log')
plt.show()



