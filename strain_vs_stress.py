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
channel_length=float(config['SETUP']['channel length'].split()[0])*1e-2 #in m

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
index = (Solidity>0.96) & (Irregularity < 1.06) & (np.abs(Sharpness) > 0.3)#select only the nice cells
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

stress=stressfunc(RP*1e-6,-pressure)# compute analytical stress profile

#%%remove bias for nearly round cells (otherwise the cell strain is always positive)
index = np.abs(RP*Angle>0) 
LA = copy.deepcopy(longaxis)
LA[index]=shortaxis[index]
SA = copy.deepcopy(shortaxis)
SA[index]=longaxis[index]

#%%  compute cell deformation (true strain)
D = np.sqrt(LA * SA) #diameter of undeformed (circular) cell
strain = (LA - SA) / D

#%% fitting deformation with stress stiffening equation
fig1=plt.figure(1, (6, 6))
border_width = 0.1
ax_size = [0+2*border_width, 0+2*border_width, 
           1-3*border_width, 1-3*border_width]
ax1 = fig1.add_axes(ax_size)
ax1.set_xlabel('fluid shear stress $\u03C3$ (Pa)')
ax1.set_ylabel('cell strain  $\u03B5$')
fit=[]

pmax = 50*np.ceil((np.max(stress)+50)//50)
ax1.set_xticks(np.arange(0,pmax+1, 50))
ax1.set_xlim((-10,pmax))
ax1.set_ylim((-0.2,1.0))

# ----------plot strain versus stress data points----------
xy = np.vstack([stress,strain])
kd = gaussian_kde(xy)(xy)  
idx = kd.argsort()
x, y, z = stress[idx], strain[idx], kd[idx]
ax1.scatter(x, y, c=z, s=50, edgecolor='', alpha=1, cmap = 'viridis') #plot in kernel density colors e.g. viridis
#ax2.plot(stress,strain,'o', color = C1) #plot the data without kernel density colors

pstart=(3.5,8) #initial guess
p, pcov = curve_fit(fitfunc, stress, strain, pstart) #do the curve fitting
#p, pcov = curve_fit(fitfunc, stress[RP<0], strain[RP<0], pstart) #do the curve fitting for one side only
err = (np.diag(pcov))**0.5 #estimate 1 standard error of the fit parameters
cov = pcov[0,1]
print("p1=%.2f +- %.2f   p2=%.1f +- %.1f   p1*p2=%.1f +- %.1f" %(p[0],err[0],p[1],err[1], p[0]*p[1], \
                                                                 np.sqrt((p[1]*err[0])**2 + (p[0]*err[1])**2 + 2*p[0]*p[1]*cov)))  
# ----------plot the fit curve----------
xx = np.arange(np.min(stress),np.max(stress),0.1) # generates an extended array 
fit_real=fitfunc(xx,p[0],p[1])
ax1.plot(xx,(fitfunc(xx,p[0],p[1])), '-', color = 'black',   linewidth=2, zorder=3)
# ----------plot standard error of the fit function----------
dyda = -1/p[0]**2*np.log(xx/p[1]+1)
dyds = -1/p[0]*xx/(xx*p[1]+p[1]**2)
vary = (dyda*err[0])**2 + (dyds*err[1])**2 + 2*dyda*dyds*cov
y1 = fitfunc(xx,p[0],p[1])-np.sqrt(vary)
y2 = fitfunc(xx,p[0],p[1])+np.sqrt(vary)
#ax1.plot(xx,y1, '--', color = 'black',   linewidth=1, zorder=3)
#ax1.plot(xx,y2, '--', color = 'black',   linewidth=1, zorder=3)
plt.fill_between(xx, y1, y2, facecolor='gray', edgecolor= "none", linewidth = 0, alpha = 0.5)
#plt.fill_between([-10,50], [-0.2, -0.2], [1,1], facecolor='C0', edgecolor= "none", linewidth = 0, alpha = 0.2)
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
ax1.errorbar(stress_av, strain_av,yerr = strain_err, marker='s', mfc='white', \
             mec='black', ms=7, mew=1, lw = 0, ecolor = 'black', elinewidth = 1, capsize = 3)    
#plt.xscale('log')
plt.show()


'''

#%% small cells
alpha = []
err_alpha = []
sigmap = []
err_sigmap = []
K0 = []
err_K0 = []
stress_subset = stress[D<np.percentile(D,33)] 
strain_subset = strain[D<np.percentile(D,33)] 
pstart=(3.5,8) #initial guess
p, pcov = curve_fit(fitfunc, stress_subset, strain_subset, pstart) #do the curve fitting
#p, pcov = curve_fit(fitfunc, stress[RP<0], strain[RP<0], pstart) #do the curve fitting for one side only
err = (np.diag(pcov))**0.5 #estimate 1 standard error of the fit parameters
cov = pcov[0,1]
alpha.append(p[0])
err_alpha.append(err[0])
sigmap.append(p[1])
err_sigmap.append(err[1])
K0.append(p[0]*p[1])
err_K0.append(np.sqrt((p[1]*err[0])**2 + (p[0]*err[1])**2 + 2*p[0]*p[1]*cov))
#print('correlation p0 vs. p1 =%.3f' % (cov/err[0]/err[1]))
print("p1=%.2f +- %.2f   p2=%.1f +- %.1f   p1*p2=%.1f +- %.1f" %(p[0],err[0],p[1],err[1], p[0]*p[1],err_K0[-1]))  
# ----------plot the fit curve----------
xx = np.arange(np.min(stress),np.max(stress),0.1) # generates an extended array 
fit_real=fitfunc(xx,p[0],p[1])
ax1.plot(xx,(fitfunc(xx,p[0],p[1])), '-', color = 'C0',   linewidth=2, zorder=3)
# ----------plot standard error of the fit function----------
dyda = -1/p[0]**2*np.log(xx/p[1]+1)
dyds = -1/p[0]*xx/(xx*p[1]+p[1]**2)
vary = (dyda*err[0])**2 + (dyds*err[1])**2 + 2*dyda*dyds*cov
y1 = fitfunc(xx,p[0],p[1])-np.sqrt(vary)
y2 = fitfunc(xx,p[0],p[1])+np.sqrt(vary)
#ax1.plot(xx,y1, '--', color = 'black',   linewidth=1, zorder=3)
#ax1.plot(xx,y2, '--', color = 'black',   linewidth=1, zorder=3)
plt.fill_between(xx, y1, y2, facecolor='C0', edgecolor= "none", linewidth = 0, alpha = 0.5)
#plt.fill_between([-10,50], [-0.2, -0.2], [1,1], facecolor='C0', edgecolor= "none", linewidth = 0, alpha = 0.2)
# ----------plot the binned (averaged) strain versus stress data points----------
binwidth = 10 #Pa
bins = np.arange(0,pmax,binwidth)
bins = [0,10,20,30,40,50,75,100,125,150,200,250]
strain_av = []
stress_av = []
strain_err = []
for i in range(len(bins)-1):
    index = (stress_subset > bins[i]) & (stress_subset < bins[i+1])
    strain_av.append(np.mean(strain_subset[index]))
    strain_err.append(np.std(strain_subset[index])/np.sqrt(np.sum(index)))
    stress_av.append(np.mean(stress_subset[index]))
ax1.errorbar(stress_av, strain_av,yerr = strain_err, marker='s', mfc='C0', \
             mec='black', ms=7, mew=1, lw = 0, ecolor = 'black', elinewidth = 1, capsize = 3)    
plt.show()



#%% medium sized cells
stress_subset = stress[(D<np.percentile(D,66)) & (D>np.percentile(D,33))] 
strain_subset = strain[(D<np.percentile(D,66)) & (D>np.percentile(D,33))] 
pstart=(3.5,8) #initial guess
p, pcov = curve_fit(fitfunc, stress_subset, strain_subset, pstart) #do the curve fitting
err = (np.diag(pcov))**0.5 #estimate 1 standard error of the fit parameters
cov = pcov[0,1]
alpha.append(p[0])
err_alpha.append(err[0])
sigmap.append(p[1])
err_sigmap.append(err[1])
K0.append(p[0]*p[1])
err_K0.append(np.sqrt((p[1]*err[0])**2 + (p[0]*err[1])**2 + 2*p[0]*p[1]*cov))
print("p1=%.2f +- %.2f   p2=%.1f +- %.1f   p1*p2=%.1f +- %.1f" %(p[0],err[0],p[1],err[1], p[0]*p[1],err_K0[-1]))  
# ----------plot the fit curve----------
xx = np.arange(np.min(stress),np.max(stress),0.1) # generates an extended array 
fit_real=fitfunc(xx,p[0],p[1])
ax1.plot(xx,(fitfunc(xx,p[0],p[1])), '-', color = 'C2',   linewidth=2, zorder=3)
# ----------plot standard error of the fit function----------
dyda = -1/p[0]**2*np.log(xx/p[1]+1)
dyds = -1/p[0]*xx/(xx*p[1]+p[1]**2)
vary = (dyda*err[0])**2 + (dyds*err[1])**2 + 2*dyda*dyds*cov
y1 = fitfunc(xx,p[0],p[1])-np.sqrt(vary)
y2 = fitfunc(xx,p[0],p[1])+np.sqrt(vary)
plt.fill_between(xx, y1, y2, facecolor='C2', edgecolor= "none", linewidth = 0, alpha = 0.5)
# ----------plot the binned (averaged) strain versus stress data points----------
binwidth = 10 #Pa
bins = np.arange(0,pmax,binwidth)
bins = [0,10,20,30,40,50,75,100,125,150,200,250]
strain_av = []
stress_av = []
strain_err = []
for i in range(len(bins)-1):
    index = (stress_subset > bins[i]) & (stress_subset < bins[i+1])
    strain_av.append(np.mean(strain_subset[index]))
    strain_err.append(np.std(strain_subset[index])/np.sqrt(np.sum(index)))
    stress_av.append(np.mean(stress_subset[index]))
ax1.errorbar(stress_av, strain_av,yerr = strain_err, marker='s', mfc='C2', \
             mec='black', ms=7, mew=1, lw = 0, ecolor = 'black', elinewidth = 1, capsize = 3)    
plt.show()



#%% large cells
stress_subset = stress[D>np.percentile(D,66)] 
strain_subset = strain[D>np.percentile(D,66)] 
pstart=(3.5,8) #initial guess
p, pcov = curve_fit(fitfunc, stress_subset, strain_subset, pstart) #do the curve fitting
err = (np.diag(pcov))**0.5 #estimate 1 standard error of the fit parameters
cov = pcov[0,1]
alpha.append(p[0])
err_alpha.append(err[0])
sigmap.append(p[1])
err_sigmap.append(err[1])
K0.append(p[0]*p[1])
err_K0.append(np.sqrt((p[1]*err[0])**2 + (p[0]*err[1])**2 + 2*p[0]*p[1]*cov))
#print('correlation p0 vs. p1 =%.3f' % (cov/err[0]/err[1]))
print("p1=%.2f +- %.2f   p2=%.1f +- %.1f   p1*p2=%.1f +- %.1f" %(p[0],err[0],p[1],err[1], p[0]*p[1],err_K0[-1]))  
# ----------plot the fit curve----------
xx = np.arange(np.min(stress),np.max(stress),0.1) # generates an extended array 
fit_real=fitfunc(xx,p[0],p[1])
ax1.plot(xx,(fitfunc(xx,p[0],p[1])), '-', color = 'C1',   linewidth=2, zorder=3)
# ----------plot standard error of the fit function----------
dyda = -1/p[0]**2*np.log(xx/p[1]+1)
dyds = -1/p[0]*xx/(xx*p[1]+p[1]**2)
vary = (dyda*err[0])**2 + (dyds*err[1])**2 + 2*dyda*dyds*cov
y1 = fitfunc(xx,p[0],p[1])-np.sqrt(vary)
y2 = fitfunc(xx,p[0],p[1])+np.sqrt(vary)
plt.fill_between(xx, y1, y2, facecolor='C1', edgecolor= "none", linewidth = 0, alpha = 0.5)
# ----------plot the binned (averaged) strain versus stress data points----------
binwidth = 10 #Pa
bins = np.arange(0,pmax,binwidth)
bins = [0,10,20,30,40,50,75,100,125,150,200,250]
strain_av = []
stress_av = []
strain_err = []
for i in range(len(bins)-1):
    index = (stress_subset > bins[i]) & (stress_subset < bins[i+1])
    strain_av.append(np.mean(strain_subset[index]))
    strain_err.append(np.std(strain_subset[index])/np.sqrt(np.sum(index)))
    stress_av.append(np.mean(stress_subset[index]))
ax1.errorbar(stress_av, strain_av,yerr = strain_err, marker='s', mfc='C1', \
             mec='black', ms=7, mew=1, lw = 0, ecolor = 'black', elinewidth = 1, capsize = 3)    
plt.show()

#%% plot cell mechanics for differently sized cells
fig8=plt.figure(8, (10, 3))
spec = gridspec.GridSpec(ncols=9, nrows=1, figure=fig8)
ax8_1=fig8.add_subplot(spec[0, 1:3])
ax8_2=fig8.add_subplot(spec[0, 4:6])  
ax8_3=fig8.add_subplot(spec[0, 7:9])

ax8_1.bar(['s','m','l'], alpha, yerr = err_alpha, width=0.8,capsize = 7, color=('C0','C2','C1'), edgecolor = 'black', linewidth = 1) 
ax8_1.set_ylabel('stiffening factor $\u03B1$')
ax8_2.bar(['s','m','l'], sigmap, yerr = err_sigmap, width=0.8,capsize = 7, color=('C0','C2','C1'), edgecolor = 'black', linewidth = 1) 
ax8_2.set_ylabel('prestress $\sigma_p$ (Pa)')
ax8_3.bar(['s','m','l'], K0, yerr = err_K0, width=0.8,capsize = 7, color=('C0','C2','C1'), edgecolor = 'black', linewidth = 1) 
ax8_3.set_ylabel('cell stiffness $K_0$ (Pa)')


#%% fitting alpha with stress stiffening equation up to a maximum shear stress
fig2=plt.figure(2, (6, 3))
border_width = 0.1
ax_size = [0+2*border_width, 0+2*border_width, 
           1-3*border_width, 1-3*border_width]
ax2 = fig2.add_axes(ax_size)
ax2.set_xlabel('stress fit range $\u03C3$ (Pa)')
ax2.set_ylabel('stiffening factor $\u03B1$')
fit=[]

pmax = 50*np.ceil((np.max(stress)+50)//50)
ax2.set_xticks(np.arange(0,pmax+1,50))
ax2.set_xlim((-10,pmax+30))
ax2.set_ylim((-0.2,6))

p0 = []
p0err = []
p1 = []
p1err = []
stressmax = np.arange(35,np.max(stress),1)
pstart=(3,8) #initial guess
for i in range(len(stressmax)):
    p, pcov = curve_fit(fitfunc, stress[stress<stressmax[i]], strain[stress<stressmax[i]], pstart) #do the curve fitting for one side only
    err = (np.diag(pcov))**0.5 #estimate 1 standard error of the fit parameters
    p0.append(p[0])
    p0err.append(err[0])
    p1.append(p[1])
    p1err.append(err[1])
    #print("stressmax=p1=%.3f  Fit Parameter: p1=%.3f +- %.3f       p2=%.3f +- %.3f" %(stressmax[i], p[0],err[0],p[1],err[1]))  
# ----------plot the parameters----------
ax2.plot(stressmax,p0, '-', color = C3,   linewidth=3, zorder=3)
# ----------plot standard error of the fit function----------
y1 = np.asarray(p0) - np.asarray(p0err)
y2 = np.asarray(p0) + np.asarray(p0err)
#ax2.plot(stressmax,y1, '--', color = 'black',   linewidth=1, zorder=3)
#ax2.plot(stressmax,y2, '--', color = 'black',   linewidth=1, zorder=3)
plt.fill_between(stressmax, y1, y2, facecolor='gray', edgecolor= "none", linewidth = 0, alpha = 0.5)


#%% fitting sigma_p with stress stiffening equation up to a maximum shear stress
fig3=plt.figure(3, (6, 3))
border_width = 0.1
ax_size = [0+2*border_width, 0+2*border_width, 
           1-3*border_width, 1-3*border_width]
ax3 = fig3.add_axes(ax_size)
ax3.set_xlabel('stress fit range $\u03C3$ (Pa)')
ax3.set_ylabel('prestress $\sigma_p$ (Pa)')
fit=[]

pmax = 50*np.ceil((np.max(stress)+50)//50)
ax3.set_xticks(np.arange(0,pmax+1,50))
ax3.set_xlim((-10,pmax+30))
ax3.set_ylim((-0.2,50))

ax3.plot(stressmax,p1, '-', color = C3,   linewidth=3, zorder=3)
# ----------plot standard error of the fit function----------
y1 = np.asarray(p1) - np.asarray(p1err)
y2 = np.asarray(p1) + np.asarray(p1err)
#ax3.plot(stressmax,y1, '--', color = 'black',   linewidth=1, zorder=3)
#ax3.plot(stressmax,y2, '--', color = 'black',   linewidth=1, zorder=3)
plt.fill_between(stressmax, y1, y2, facecolor='gray', edgecolor= "none", linewidth = 0, alpha = 0.5, zorder=0)

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
ax4.set_ylim((-0.2,1.0))
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

#%% plot angle versus radial position in channel
fig6=plt.figure(6, (6, 6))
border_width = 0.12
ax_size = [0+2*border_width, 0+2*border_width, 
           1-3*border_width, 1-3*border_width]
ax6 = fig6.add_axes(ax_size)
xy = np.vstack([RP,Angle])
kd = gaussian_kde(xy)(xy)  
idx = kd.argsort()
x, y, z = RP[idx], Angle[idx], kd[idx]
ax6.scatter(x, y, c=z, s=50, edgecolor='', alpha=1, cmap = 'viridis') #plot in kernel density colors
ax6.set_xticks(np.arange(-100,101,25))
ax6.set_yticks(np.arange(-60,61,20))
ax6.set_ylim((-60,60))
ax6.set_xlabel('radial position in channel ($\u03BC m$)')
ax6.set_ylabel('angle (deg)')

#%% plot undeformed cell diamter versus radial position in channel
fig7=plt.figure(7, (6, 6))
border_width = 0.12
ax_size = [0+2*border_width, 0+2*border_width, 
           1-3*border_width, 1-3*border_width]
ax7 = fig7.add_axes(ax_size)
xy = np.vstack([RP,D])
kd = gaussian_kde(xy)(xy)  
idx = kd.argsort()
x, y, z = RP[idx], D[idx], kd[idx]
ax7.scatter(x, y, c=z, s=50, edgecolor='', alpha=1, cmap = 'viridis') #plot in kernel density colors
ax7.set_xticks(np.arange(-100,101,25))
ax7.set_yticks(np.arange(0,31,5))
ax7.set_ylim((0,30))
ax7.set_xlabel('radial position in channel ($\u03BC m$)')
ax7.set_ylabel('undeformed cell diameter ($\u03BC m$)')
'''