# -*- coding: utf-8 -*-
"""
Created on Tue May 22 2020

@author: Ben

# This program reads a txt file with the analyzed cell position, shape (semi-major and semi-minor axis etc.),
# computes the cell strain and the fluid shear stress acting on each cell,
# plots the data (strain versus stress) for each cell using a kernel density estimate for the datapoint color,
# and fits a stress stiffening equation to the data 
# The results such as maximum flow speed, cell mechanical parameters, etc. are stored in 
# the file 'all_data.txt' located at the same directory as this script 

# This compares the three result files from Canny, Network and GT.
# Velocity comparison is commented because the GT data consists of non-consecutive images.
# The number of detected cells is compared in bins and the strain vs stress
# density plot is plotted seperately.


"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit #, leastsq
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

# if there is a command line parameter...
if len(sys.argv) >= 2:
    # ... we just use this file
    datafile = sys.argv[1]
# if not, we ask the user to provide us a filename
else:

#%% select result.txt file
    root = Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    
    datafile = filedialog.askopenfilename(title="select the data file",filetypes=[("txt file",'*_result.txt')]) # show an "Open" dialog box and return the path to the selected file
    datafile2 = filedialog.askopenfilename(title="select the data file",filetypes=[("txt file",'*_result_NN.txt')]) # show an "Open" dialog box and return the path to the selected file

    if datafile == '':
        print('empty')
        sys.exit()  

filename_ex = os.path.basename(datafile)
print(filename_ex)
filename_base, file_extension = os.path.splitext(filename_ex)
output_path = os.path.dirname(datafile)
#pressure = float(filename_base[1])*1e5 # deduce applied pressure from file name (in Pa)
filename_config = output_path + '/'+ filename_base[:-7] +'_config.txt' #remove _result from the filename and add _config.txt

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


#%% get minimum and maximum for density plot
data =np.genfromtxt(datafile2,dtype=float,skip_header= 2)
# experimental raw data
RP=data[:,3] #radial position 
longaxis=data[:,4] #Longaxis of ellipse
shortaxis=data[:,5] #Shortaxis of ellipse
Angle=data[:,6]

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

stress=stressfunc(RP*1e-6,-pressure)# compute analytical stress profile
D = np.sqrt(longaxis * shortaxis) #diameter of undeformed (circular) cell
strain = (longaxis - shortaxis) / D
# ----------plot strain versus stress data points----------
xy = np.vstack([stress,strain])
kd = gaussian_kde(xy)(xy)  
idx = kd.argsort()
x, y, z = stress[idx], strain[idx], kd[idx]
v_min = np.min(z)
v_max = np.max(z)
print(np.min(z),np.max(z))
print(len(z))

print(v_min,v_max)

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
index = (Solidity>0.96) & (Irregularity < 1.05) & (np.abs(Sharpness) > 0.5)#select only the nice cells
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
vel_fit, pcov = curve_fit(velfit, y_pos, vel, [np.max(vel),0.9]) #fit a parabolic velocity profile 
r = np.arange(-channel_width/2*1e6,channel_width/2*1e6,0.1) # generates an extended array 
ax1.plot(r,velfit(r,vel_fit[0],vel_fit[1]), '--', color = 'blue',   linewidth=2, zorder=3)
print('v_max = %5.2f mm/s   profile stretch exponent = %5.2f\n' %(vel_fit[0],vel_fit[1]))

   

#%%  compute stress profile, cell deformation (true strain), and diameter of the undeformed cell
stress=stressfunc(RP*1e-6,-pressure)# compute analytical stress profile
D = np.sqrt(longaxis * shortaxis) #diameter of undeformed (circular) cell
strain = (longaxis - shortaxis) / D

#%% fitting deformation with stress stiffening equation
#fig2, (ax2,ax3,ax4) = plt.subplots(1,3,figsize=(20,6),sharex=True, sharey=True)
fig2, (ax2,ax3,ax4) = plt.subplots(1,3,figsize=(15,6),sharex=True, sharey=True)


#fig2=plt.figure(2, (18, 6))
#border_width = 0.1
#ax_size = [0+2*border_width, 0+2*border_width, 
      #     1-3*border_width, 1-3*border_width]
#ax2 = fig2.add_axes(ax_size)
ax2.set_xlabel('fluid shear stress $\u03C3$ (Pa)')
ax2.set_ylabel('cell strain  $\u03B5$')
fit=[]

pmax = 50*np.ceil((np.max(stress)+50)//50)
ax2.set_xticks(np.arange(0,pmax+1, 50))
ax2.set_xlim((-10,110))
ax2.set_ylim((-0.2,1.0))

# ----------plot strain versus stress data points----------
xy = np.vstack([stress,strain])
kd = gaussian_kde(xy)(xy)  
idx = kd.argsort()
x, y, z = stress[idx], strain[idx], kd[idx]
print(np.min(z),np.max(z))
if np.min(z) < v_min:
    v_min = np.min(z)
if np.max(z) > v_max:
    v_max = np.max(z)
print(v_min,v_max)
strain1=y
#ax2.plot(stress,strain,'o', color = 'black') #plot the data without kernel density colors
ax2.scatter(x, y, c=z, s=50, edgecolor='', alpha=1,cmap='viridis', vmin = v_min, vmax=v_max) #plot in kernel density colors e.g. viridis

pstart=(0.1,1,0) #initial guess
p, pcov = curve_fit(fitfunc, stress, strain, pstart, maxfev = 10000) #do the curve fitting
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
ax2.fill_between(xx, y1, y2, facecolor='gray', edgecolor= "none", linewidth = 0, alpha = 0.5)

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

ax3.scatter(x, y, c=z, s=50, edgecolor='', alpha=1,cmap='viridis', vmin = v_min, vmax=v_max) #plot in kernel density colors e.g. viridis

RP1 = RP

#%% import raw data NN
data =np.genfromtxt(datafile2,dtype=float,skip_header= 2)

#%% experimental raw data
Frames=data[:,0] #frame number 
X=data[:,1] #x-position in Pixes
Y=data[:,2] #x-position in Pixes
RP=data[:,3] #radial position 
longaxis=data[:,4] #Longaxis of ellipse
shortaxis=data[:,5] #Shortaxis of ellipse
Angle=data[:,6] #Shortaxis of ellipse

l_before = len(RP)
print('# frames =', Frames[-1], '   # cells total =', l_before)    


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
vel_fit, pcov = curve_fit(velfit, y_pos, vel, [np.max(vel),0.9]) #fit a parabolic velocity profile 
r = np.arange(-channel_width/2*1e6,channel_width/2*1e6,0.1) # generates an extended array 
ax1.plot(r,velfit(r,vel_fit[0],vel_fit[1]), '--', color = 'orange',   linewidth=2, zorder=3)
print('v_max = %5.2f mm/s   profile stretch exponent = %5.2f\n' %(vel_fit[0],vel_fit[1]))


#%%  compute stress profile, cell deformation (true strain), and diameter of the undeformed cell
stress=stressfunc(RP*1e-6,-pressure)# compute analytical stress profile
D = np.sqrt(longaxis * shortaxis) #diameter of undeformed (circular) cell
strain = (longaxis - shortaxis) / D

#%% fitting deformation with stress stiffening equation

fit=[]

pmax = 50*np.ceil((np.max(stress)+50)//50)

# ----------plot strain versus stress data points----------
xy = np.vstack([stress,strain])
kd = gaussian_kde(xy)(xy)  
idx = kd.argsort()
x, y, z = stress[idx], strain[idx], kd[idx]
#ax2.plot(stress,strain,'o', color = 'red') #plot the data without kernel density colors
ax2.scatter(x, y, c=z, s=50, edgecolor='', alpha=1, cmap = 'viridis', vmin = v_min, vmax=v_max) #plot in kernel density colors e.g. viridis
strain2=y
pstart=(0.1,1,0) #initial guess
p, pcov = curve_fit(fitfunc, stress, strain, pstart, maxfev = 10000) #do the curve fitting
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
ax2.plot(xx,(fitfunc(xx,p[0],p[1], p[2])), '-', color = 'red',   linewidth=2, zorder=3)
# ----------plot standard error of the fit function----------
dyda = -1/(p[0]**2)*np.log(xx/p[1]+1) #strain derivative with respect to alpha
dydp = -1/p[0]*xx/(xx*p[1]+p[1]**2)   #strain derivative with respect to prestress
dydo = 1                              #strain derivative with respect to offset
vary = (dyda*err[0])**2 + (dydp*err[1])**2 + (dydo*err[2])**2 + 2*dyda*dydp*cov_ap + 2*dyda*dydo*cov_ao + 2*dydp*dydo*cov_po
y1 = fitfunc(xx,p[0],p[1],p[2])-np.sqrt(vary)
y2 = fitfunc(xx,p[0],p[1],p[2])+np.sqrt(vary)
ax2.fill_between(xx, y1, y2, facecolor='red', edgecolor= "none", linewidth = 0, alpha = 0.2)

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
ax1.legend(['Canny','Canny fit','Network','Network fit'],loc=1)
ax4.scatter(x, y, c=z, s=50, edgecolor='', alpha=1, cmap = 'viridis',vmin = v_min, vmax=v_max) #plot in kernel density colors e.g. viridis

RP2 = RP
#ax1.set_xlim((0.5,pmax))
#plt.xscale('log')

ax3.set_title('Canny')
ax4.set_title('Network')
ax2.legend(['Canny', 'Network'],loc=2)

fig7=plt.figure(7, (6, 3))
border_width = 0.12
ax_size = [0+2*border_width, 0+2*border_width, 
           1-3*border_width, 1-4*border_width]
ax7 = fig7.add_axes(ax_size)
bin_width = 25
hist1, bin_edges = np.histogram(RP1, bins=np.arange(-100 + bin_width/2, 101 - bin_width/2, bin_width), density=False)
hist2, bin_edges = np.histogram(RP2, bins=np.arange(-100 + bin_width/2, 101 - bin_width/2, bin_width), density=False)
plt.bar(bin_edges[:-1]+bin_width/3, hist1, width=bin_width*0.2, edgecolor = 'black')
plt.bar(bin_edges[:-1]+2*bin_width/3, hist2, width=bin_width*0.2, edgecolor = 'black')

ax7.legend(['Canny', 'Network'],loc=1)

ax7.set_xlabel('radial position in channel ($\u03BC m$)')
ticks = np.arange(-100,101,bin_width)
ax7.set_xlim((-100,100))
labels = ticks.astype(int)
labels = labels.astype(str)
ax7.set_xticks(ticks)
ax7.set_xticklabels(labels) 
ax7.set_ylabel('# of cells')
fig7.tight_layout()

plt.savefig('C:/Users/User/Documents/Bachelorarbeit_Selina/Figures/cell_contours_strain_vs_stress/dispersion_p2_Canny_NN.png')

#%% binned strain

fig4=plt.figure(4, (6, 3))
border_width = 0.12
ax_size = [0+2*border_width, 0+2*border_width, 
           1-3*border_width, 1-4*border_width]
ax4 = fig4.add_axes(ax_size)
bin_width = 25

bins = np.linspace(-100,100,9)

d = np.digitize(RP1, bins) 
bin_means = [strain1[d == i].mean() for i in range(1, len(bins))]
bin_std = [strain1[d == i].std() for i in range(1, len(bins))]
plt.errorbar(bins[:-1] + bin_width/2, bin_means,bin_std,marker='o', capsize=5, capthick=2)

d = np.digitize(RP2, bins) 
bin_means = [strain2[d == i].mean() for i in range(1, len(bins))]
bin_std = [strain2[d == i].std() for i in range(1, len(bins))]
plt.errorbar(bins[:-1] + bin_width/2, bin_means,bin_std,marker='o', capsize=5, capthick=2)

ax4.legend(['Canny', 'Network'],loc=1)

#ax4.set_ylim((-0.2,1.0))
ax4.set_xlabel('radial position in channel ($\u03BC m$)')
ax4.set_ylabel('cell strain  $\u03B5$')

plt.show()

#%% store the results
'''
output_path = os.getcwd()
date_time=filename_base.split('_')
seconds = float(date_time[3])*60*60 + float(date_time[4])*60 + float(date_time[5])
alldata_file = output_path + '/' + 'all_data.txt'
if not os.path.exists(alldata_file):
    f = open(alldata_file,'at')
    f.write('filename' +'\t' +'seconds' +'\t' + 'p (kPa)' +'\t' + '#cells' +'\t' + '#diameter (um)' +'\t' + 'vmax (mm/s)' +'\t' + 'expo' +'\t' + 'alpha' +'\t' + 'sigma' +'\t' + 'eta_0' +'\t' + 'stiffness (Pa)' +'\n')
else:
    f = open(alldata_file,'at')
f.write(datafile + '\t' + '{:.0f}'.format(seconds) +'\t')
f.write(str(pressure/1000) +'\t' + str(len(RP)) +'\t' + '{:0.1f}'.format(np.mean(D)) +'\t' )
f.write('{:0.3f}'.format(vel_fit[0]) +'\t' + '{:0.3f}'.format(vel_fit[1]) +'\t')
f.write('{:0.3f}'.format(p[0]) +'\t' + '{:0.2f}'.format(p[1]) +'\t' + '{:0.3f}'.format(p[2]) + '\t' + '{:0.3f}'.format(p[0]*p[1]) +'\n')
#f.write(str(frame[i]) +'\t' +str(X[i]) +'\t' +str(Y[i]) +'\t' +str(R[i]) +'\t' +str(LongAxis[i]) +'\t'+str(ShortAxis[i]) +'\t' +str(Angle[i]) +'\t' +str(irregularity[i]) +'\t' +str(solidity[i]) +'\t' +str(sharpness[i]) +'\n')
f.close()
'''