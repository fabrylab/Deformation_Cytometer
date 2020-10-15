# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:31:17 2020

@author: user
"""
	
# scikit-learn bootstrap
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from matplotlib import pyplot as mp
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit#, leastsq
from pylab import *
font = {'family' : 'sans-serif',
        'sans-serif':['Arial'],
        'weight' : 'normal',
        'size'   : 18}
plt.rc('font', **font)
plt.rc('legend', fontsize=12)
plt.rc('axes', titlesize=18)

# if nof=1 , only calculate bootstrappng for 1 dataset, nof=2 is comparison btw 2 dataset
nof=2
data_1 = np.loadtxt(r'C:\Users\user\Desktop\Experiments\cell_mechanics\stress_strain_3t3_DMSO_nop3_Truestrain_auto.txt')

strain_1=data_1[:,0]
stress_1=data_1[:,1]
p0_1=[]
p1_1=[]
if nof>1:
    data_2 = np.loadtxt(r'C:\Users\user\Desktop\Experiments\cell_mechanics\stress_strain_3t3_cytochasin_nop3_Truestrain_auto.txt')
    strain_2=data_2[:,0]
    stress_2=data_2[:,1]
    p0_2=[]
    p1_2=[]

def fitfunc(x, p1,p2): #for curve_fit
    return (1/p1)*np.log((x/p2)+1)
pstart=(1,0.1) #initial guess
def gaussian(x, mu, sig):
    return 1/(sig* np.sqrt(2*np.pi)) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
#%% computing p_value of bootstrapping
def pvalue_bootstrap(pool1, pool2, nr=10000):
    count=0
    for n in np.arange(0,nr):
        id1 = np.random.randint(0,len(pool1))
        id2 = np.random.randint(0, len(pool2))

        if pool1[id1]>pool2[id2]:
            count=count+1
    p_value=count/nr        
    if p_value >= 0.5:
        p_value=  1- p_value

    print("{:.4f}".format(p_value))
    return float("{:.4f}".format(p_value))


#  bootstrap sample1
for k in range(0,100):
    index = np.arange(0, len(strain_1))
    boot_index = np.random.choice(index, size=len(index))
    boot_strain_1 = strain_1[boot_index]
    boot_stress_1 = stress_1[boot_index]
    p1, pcov = curve_fit(fitfunc, boot_stress_1, boot_strain_1, pstart) #do the curve fitting
    p0_1.append(p1[0])
    p1_1.append(p1[1])
    err = (np.diag(pcov))**0.5 #estimate 1 standard error of the fit parameters
#    print("Fit Parameter: p1=%.3f +- %.3f       p2=%.3f +- %.3f" %(p[0],err[0],p[1],err[1]))  
p0_1=np.array(p0_1) 
p1_1=np.array(p1_1) 

#  bootstrap sample2 
if nof>1: 
    for k in range(0,1000):
        index = np.arange(0, len(strain_2))
        boot_index = np.random.choice(index, size=len(index))
        boot_strain_2 = strain_2[boot_index]
        boot_stress_2 = stress_2[boot_index]
        p2, pcov = curve_fit(fitfunc, boot_stress_2, boot_strain_2, pstart) #do the curve fitting
        p0_2.append(p2[0])
        p1_2.append(p2[1])
        err = (np.diag(pcov))**0.5 #estimate 1 standard error of the fit parameters
    #    print("Fit Parameter: p1=%.3f +- %.3f       p2=%.3f +- %.3f" %(p[0],err[0],p[1],err[1]))                 
    p0_2=np.array(p0_2) 
    p1_2=np.array(p1_2)  
       
fig = plt.figure(1,(7, 4))
spec = gridspec.GridSpec(ncols=10, nrows=10, figure=fig)
ax1 = fig.add_subplot(spec[1:8, 1:5])
p_value_alpha=pvalue_bootstrap(p0_2, p0_1, nr=10000)
p_value_sigmap=pvalue_bootstrap(p1_2, p1_1, nr=10000)
# alpha
ax1.hist(p0_1,50,density=True) 
ax1.set_xlabel(r'$\alpha$')  
ax1.set_ylabel('Probability density')  
ax1.set_title('p = '+str(p_value_alpha)) 
x_values_1 = np.linspace(min(p0_1)-0.1, max(p0_1)+0.1, 120)
ax1.plot(x_values_1, gaussian(x_values_1, np.mean(p0_1), p0_1.std()),'--',color='black',linewidth=1)
#ax1.set_xticks([0,int(np.mean(p0_1)),np.round((np.mean(p0_2)),0)])
if nof>1: 
    ax1.hist(p0_2,50,density=True) 
    x_values_2 = np.linspace(min(p0_2)-0.1, max(p0_2)+0.1, 120)
    ax1.plot(x_values_2, gaussian(x_values_2, np.mean(p0_2), p0_2.std()),'--',color='black',linewidth=1)

#pre-stress
ax2 = fig.add_subplot(spec[1:8, 6:10])
ax2.hist(p1_1,50,density=True) 
ax2.set_xlabel(r'pre-stress') 
ax1.set_title('p = '+str(p_value_sigmap)) 
x_values_1 = np.linspace(min(p1_1)-0.1, max(p1_1)+0.1, 120)
#ax2.set_xticks([0,25,50])
ax2.plot(x_values_1, gaussian(x_values_1, np.mean(p1_1), p1_1.std()),'--',color='black',linewidth=1)
if nof>1: 
    ax2.hist(p1_2,50,density=True) 
    x_values_2 = np.linspace(0, max(p1_2)+0.1, 120)
    ax2.plot(x_values_2, gaussian(x_values_2, np.mean(p1_2), p1_2.std()),'--',color='black',linewidth=1)

#%% strain vs stress 
fig3=plt.figure(3, (7, 6))
border_width = 0.2
ax_size = [0+border_width, 0+border_width, 
           1-2*border_width, 1-2*border_width]
ax3 = fig3.add_axes(ax_size)
C1 = '#1f77b4'
C2 = '#ff7f0e'
C3 = '#9fc5e8'
C4='navajowhite' 
#data1
plt.plot(stress_1,strain_1,'o', zorder=1, markerfacecolor=C1, markersize=6.0,markeredgewidth=0, label="data1")
xx_1 = np.arange(np.min(stress_1),np.max(stress_1),0.1) # generates an extended array 
fit_real_1=fitfunc(xx_1,p1[0],p1[1])
plt.plot(xx_1,fit_real_1, '--', color = "black",   linewidth=1, zorder=3)
ax3.tick_params(axis="y",direction="in")
ax3.tick_params(axis="x",direction="in")
# ----------plot standard error of the fit function----------
y1_1 = fitfunc(xx_1,p1[0]-p0_1.std(),p1[1]-p1_1.std() )
y2_1 = fitfunc(xx_1,p1[0]+p0_1.std(),p1[1]+p1_1.std() )
plt.fill_between(xx_1, y1_1, y2_1, facecolor=C3, edgecolor= "none", linewidth = 0, alpha=0.5,zorder=2)
y1_1 = fitfunc(xx_1,p1[0]-p0_1.std(),p1[1]+p1_1.std() )
y2_1 = fitfunc(xx_1,p1[0]+p0_1.std(),p1[1]-p1_1.std() )
plt.fill_between(xx_1, y1_1, y2_1, facecolor=C3, edgecolor= "none", linewidth = 0, alpha=0.5, zorder=2)

# data2
if nof>1: 
    plt.plot(stress_2,strain_2,'o', zorder=1, markerfacecolor=C2, markersize=6.0,markeredgewidth=0, label="data2")
    xx_2 = np.arange(np.min(stress_2),np.max(stress_2),0.1) # generates an extended array 
    fit_real_2=fitfunc(xx_2,p2[0],p2[1])
    plt.plot(xx_2,fit_real_2, '--', color = "black",   linewidth=1, zorder=3)
    # ----------plot standard error of the fit function----------
    y1_2 = fitfunc(xx_2,p2[0]-p0_2.std(),p2[1]-p1_2.std() )
    y2_2 = fitfunc(xx_2,p2[0]+p0_2.std(),p2[1]+p1_2.std() )
    plt.fill_between(xx_2, y1_2, y2_2, facecolor=C4, edgecolor= "none", linewidth = 0, alpha=0.5,zorder=2)
    y1_2 = fitfunc(xx_2,p2[0]-p0_2.std(),p2[1]+p1_2.std() )
    y2_2 = fitfunc(xx_2,p2[0]+p0_2.std(),p2[1]-p1_2.std() )
    plt.fill_between(xx_2, y1_2, y2_2, facecolor=C4, edgecolor= "none", linewidth = 0, alpha=0.5, zorder=2)    
    
plt.xticks([0,50,100,150,200])
plt.yticks([0,0.4,.8,1.2,1.6])
ax3.tick_params(axis="y",direction="in")
ax3.tick_params(axis="x",direction="in")

plt.xlim((-2,150))
plt.xlabel('$\u03C3$ (Pa)')
plt.ylabel('True strain')
plt.show()






#
#plt.hist(stress_2,100)
#plt.hist(stress_1,100)





















