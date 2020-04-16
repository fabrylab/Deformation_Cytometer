# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 08:42:39 2020

@author: user
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit #, leastsq
import copy
import math
#----------general fonts for plots and figures----------
font = {'family' : 'sans-serif',
        'sans-serif':['Arial'],
        'weight' : 'normal',
        'size'   : 18}
plt.rc('font', **font)
plt.rc('legend', fontsize=12)
plt.rc('axes', titlesize=18)
#%% opening raw data
file_raw_p1=r'Z:\emirzahossein\data_backup\20200217_cytochasinD\nih_dmso\ANALYZED\nalysisof_extractframeshape\free_p1_t9_vid000_2_result__dmso_p1.txt'
file_raw_p2=r'Z:\emirzahossein\data_backup\20200217_cytochasinD\nih_dmso\ANALYZED\nalysisof_extractframeshape\free_p2_t7_vid000_2_result__dmso_p2.txt'
file_raw_p3=r'Z:\emirzahossein\data_backup\20200217_cytochasinD\nih_dmso\ANALYZED\nalysisof_extractframeshape\free_p3_t7_vid000_2_result__dmso_p3.txt'
###############################################
#%% analytical solution
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
rawdata_p1 =np.genfromtxt(file_raw_p1,dtype=float,skip_header= 1)
rawdata_p2 =np.genfromtxt(file_raw_p2,dtype=float,skip_header= 1)
rawdata_p3 =np.genfromtxt(file_raw_p3,dtype=float,skip_header= 1)
#%% experimental raw data
#pressure 1bar
RP_p1=rawdata_p1[:,3] #radial position in pressure 1bar
longaxis_p1=rawdata_p1[:,4] #Longaxis of ellipse
shortaxis_p1=rawdata_p1[:,5] #Shortaxis of ellipse
angle_p1=rawdata_p1[:,6] #Angle (orientation)
#pressure 2bar
RP_p2=rawdata_p2[:,3]
longaxis_p2=rawdata_p2[:,4]
shortaxis_p2=rawdata_p2[:,5]
angle_p2=rawdata_p2[:,6]
#pressure 3bar
RP_p3=rawdata_p3[:,3]
longaxis_p3=rawdata_p3[:,4]
shortaxis_p3=rawdata_p3[:,5]
angle_p3=rawdata_p3[:,6]

# analytical stress profile
stress_p1=stressfunc(RP_p1*1e-6,-1*1e5)
stress_p2=stressfunc(RP_p2*1e-6,-2*1e5)
stress_p3=stressfunc(RP_p3*1e-6,-3*1e5)
#%%remove bias
index = np.abs(RP_p1*angle_p1>0) 
LA_p1 = copy.deepcopy(longaxis_p1)
LA_p1[index]=shortaxis_p1[index]
SA_p1 = copy.deepcopy(shortaxis_p1)
SA_p1[index]=longaxis_p1[index]
#p2
index = np.abs(RP_p2*angle_p2>0) 
LA_p2 = copy.deepcopy(longaxis_p2)
LA_p2[index]=shortaxis_p2[index]
SA_p2 = copy.deepcopy(shortaxis_p2)
SA_p2[index]=longaxis_p2[index]
#p3
index = np.abs(RP_p3*angle_p3>0)  
LA_p3 = copy.deepcopy(longaxis_p3)
LA_p3[index]=shortaxis_p3[index]
SA_p3 = copy.deepcopy(shortaxis_p3)
SA_p3[index]=longaxis_p3[index]

#%%  deformation (True strain)
strain_p1 = (LA_p1 - SA_p1)/np.sqrt(LA_p1 * SA_p1)
strain_p2 = (LA_p2 - SA_p2)/np.sqrt(LA_p2 * SA_p2)
strain_p3 = (LA_p3- SA_p3)/np.sqrt(LA_p3 * SA_p3)

#%% plotig of deformation versus radial position
fig1=plt.figure(1, (8, 8))
border_width = 0.2
ax_size = [0+border_width, 0+border_width, 
           1-2*border_width, 1-2*border_width]
ax1 = fig1.add_axes(ax_size)

ax1.plot(RP_p1,strain_p1,'o')
ax1.plot(RP_p2,strain_p2,'o',color='darkorange')
ax1.plot(RP_p3,strain_p3,'o',color='green')
ax1.set_xlabel('Radius(from channel center) [$\mu m$]')
ax1.set_ylabel('Deformation')
ax1.legend(['P = 1','P = 2','P = 3'],loc='upper left')
ax1.set_xlim(-100,100)
plt.show()
#%% plotting of deformation as a function of shear stress for different pressure (separately)
fig2=plt.figure(2, (8, 8))
border_width = 0.2
ax_size = [0+border_width, 0+border_width, 
           1-2*border_width, 1-2*border_width]
ax2 = fig2.add_axes(ax_size)

ax2.plot(stress_p1,strain_p1,'o',zorder=3)
ax2.plot(stress_p2,strain_p2,'o',zorder=2)
ax2.plot(stress_p3,strain_p3,'o',zorder=1)
ax2.legend(['P = 1 bar','P = 2 bar','P = 3 bar'],loc='upper left')
ax2.set_xlabel('Stress ,$\u03C3   (Pa) $')
ax2.set_ylabel('Strain')
ax2.set_title('Taylor deformation')

#%% fitting deformation with stress stiffening equation, combining different pressures
angle_T=np.hstack((angle_p1,angle_p2,angle_p3))
LA_T=np.hstack((LA_p1,LA_p2,LA_p3))
SA_T=np.hstack((SA_p1,SA_p2,SA_p3))
stress_T=np.hstack((stress_p1,stress_p2,stress_p3))
strain_T=np.hstack((strain_p1,strain_p2,strain_p3))

#----------fitting----------
fig3=plt.figure(3, (8, 8))
border_width = 0.2
ax_size = [0+border_width, 0+border_width, 
           1-2*border_width, 1-2*border_width]
ax3 = fig3.add_axes(ax_size)
fit=[]
def fitfunc(x, p1,p2): #for curve_fit
    return (1/p1)*np.log((x/p2)+1)

pstart=(1,.017) #initial guess
p, pcov = curve_fit(fitfunc, stress_T, strain_T, pstart) #do the curve fitting
err = (np.diag(pcov))**0.5 #estimate 1 standard error of the fit parameters
print("Fit Parameter: p1=%.3f +- %.3f       p2=%.3f +- %.3f" %(p[0],err[0],p[1],err[1]))  

C1 = '#1f77b4'
C2 = '#ff7f0e'
C3 = '#9fc5e8'
C4='navajowhite'
xx = np.arange(np.min(stress_T),np.max(stress_T),0.1) # generates an extended array 
fit_real=fitfunc(xx,p[0],p[1])
ax3.plot(xx,(fitfunc(xx,p[0],p[1])), '--', color = "black",   linewidth=1, zorder=3)
ax3.set_xticks([0,50,100,150,200])

ax3.plot(stress_T,strain_T,'o', color='b')  
xx = np.arange(np.min(stress_T),np.max(stress_T),0.1) # generates an extended array 
#plt.plot(xx,(fitfunc(xx,p[0],p[1])), '-')
ax3.set_xlim((-2,230))
ax3.set_xlabel('$\u03C3$ (Pa)')
ax3.set_ylabel('Taylor strain')
plt.show()

#%% saving deformation and stress for bootstrapping
#f = open('stress_strain_3t3_DMSO_nop3_Truestrain_auto.txt','w')
#for i in range(0,len(strain)): 
#    f.write(str(strain[i])+'\t'+str(stress[i])+'\n')
#f.close()
#

#%% Coefficient of determination (R^2)
datapoints=strain_T
model=fitfunc(stress_T,p[0],p[1])

correlation_matrix = np.corrcoef(datapoints, model)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2

print('coefficeint of determination =',r_squared)
#plt.plot(datapoints,model,'o')
#plt.show()









