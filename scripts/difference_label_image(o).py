# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:19:45 2020

@author: selina

# This program reads two txt files with the analyzed cell position, shape (semi-major and semi-minor axis etc.),
# of two different files (one created with labels from original image, the other one with labels from bandpass
# image, compares detected cells in same frame at same x/y-position and plots the differences for radius, radial
# position, long and short axis, angle, irregularity, solidity and strain

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

datafile2 = filedialog.askopenfilename(title="select the data file 2",filetypes=[("txt file",'*.txt')]) # show an "Open" dialog box and return the path to the selected file
if datafile2 == '':
    print('empty')
    sys.exit()
filename_ex = os.path.basename(datafile2)
filename_base, file_extension = os.path.splitext(filename_ex)
output_path = os.path.dirname(datafile2)
pressure = float(filename_base[1])*1e5 # deduce applied pressure from file name (in Pa)

#%% import raw data
#Data1: with bandpassimage label
#Data2: with originalimage label

data = np.genfromtxt(datafile,dtype=float,skip_header= 2)

data2 = np.genfromtxt(datafile2,dtype=float,skip_header= 2)

#%% experimental raw data
frame=data[:,0]
x_pos=data[:,1]
y_pos=data[:,2]
RP=data[:,3] #radial position 
longaxis=data[:,4] #Longaxis of ellipse
shortaxis=data[:,5] #Shortaxis of ellipse
Angle=data[:,6] #Shortaxis of ellipse
Irregularity=data[:,7] #ratio of circumference of the binarized image to the circumference of the ellipse 
Solidity=data[:,8] #percentage of binary pixels within convex hull polygon

#%% experimental raw data2
frame2=data2[:,0]
x_pos2=data2[:,1]
y_pos2=data2[:,2]
RP2=data2[:,3] #radial position 
longaxis2=data2[:,4] #Longaxis of ellipse
shortaxis2=data2[:,5] #Shortaxis of ellipse
Angle2=data2[:,6] #Shortaxis of ellipse
Irregularity2=data2[:,7] #ratio of circumference of the binarized image to the circumference of the ellipse 
Solidity2=data2[:,8] #percentage of binary pixels within convex hull polygon

#%% Remove bias
index = np.abs(RP*Angle>0) 
LA = copy.deepcopy(longaxis)
LA[index]=shortaxis[index]
SA = copy.deepcopy(shortaxis)
SA[index]=longaxis[index]

index = np.abs(RP2*Angle2>0) 
LA2 = copy.deepcopy(longaxis2)
LA2[index]=shortaxis2[index]
SA2 = copy.deepcopy(shortaxis2)
SA2[index]=longaxis2[index]

#%%  deformation (True strain)
D = np.sqrt(LA * SA) #diameter of undeformed (circular) cell
strain = (LA - SA) / D
radius = D/2

D2 = np.sqrt(LA2 * SA2) #diameter of undeformed (circular) cell
strain2 = (LA2 - SA2) / D2
radius2 = D2/2

#%% Frames in both result files
frame_list = frame.tolist()
frame2_list = frame2.tolist()
same = [item for item in frame_list if item in frame2_list]  # frame numbers in both data sets
pos_same = [i for i in range(len(frame_list)) if frame_list[i] in frame2_list]  # position of frame in list1

print('{} frames in both result-files'.format(len(same)))

# Duplicate elements in frame
res = [idx for idx, val in enumerate(frame) if val in frame[:idx]]
double = [val for idx, val in enumerate(frame) if val in frame[:idx]]
print('{} frames in results1 with more than one cell per frame'.format(len(res)))
# Duplicate elements in frame2
res2 = [idx for idx, val in enumerate(frame2) if val in frame2[:idx]]
double2 = [val for idx, val in enumerate(frame2) if val in frame2[:idx]]
print('{} frames in results2 with more than one cell per frame'.format(len(res2)))

# compare results for every cell
diff_radius = []
diff_RP = []
diff_LA = []
diff_SA = []
diff_Angle = []
diff_Irr = []
diff_Sol = []
diff_strain = []

no_equivalent = 0
no_suitable_frame = []

# Include just nice cells! for both data sets
#(Solidity>0.95) & (Irregularity < 1.06) #select only the nices cells

i = 0
for i in range(len(same)): 
    if (Solidity[pos_same[i]] > 0.95) & (Irregularity[pos_same[i]] < 1.06):
        # frame number
        #print('frame  number = {}, i: {}'.format(same[i],i))
        pos_same[i]  # position des frames in data1
        #double.index(same_simple[i])
        # Falls frame nur in daten2 mehrfach: finde zelle, die am besten passt!
        if same[i] in double2:
            n = 0
            while frame2[frame2_list.index(same[i])] == frame2[frame2_list.index(same[i]) + n]:
                if (Solidity2[frame2_list.index(same[i]) + n] > 0.95) & (Irregularity2[frame2_list.index(same[i]) + n] < 1.06):
                    if (abs(x_pos[pos_same[i]] - x_pos2[frame2_list.index(same[i]) + n]) < 1) and (abs(y_pos[pos_same[i]] - y_pos2[frame2_list.index(same[i]) + n]) < 1):
                        diff_radius.append(radius[pos_same[i]] - radius2[frame2_list.index(same[i]) + n])
                        diff_RP.append(RP[pos_same[i]] - RP2[frame2_list.index(same[i]) + n])
                        diff_LA.append(LA[pos_same[i]] - LA2[frame2_list.index(same[i]) + n])
                        diff_SA.append(SA[pos_same[i]] - SA2[frame2_list.index(same[i]) + n])
                        diff_Angle.append(Angle[pos_same[i]] - Angle2[frame2_list.index(same[i]) + n])
                        diff_Irr.append(Irregularity[pos_same[i]] - Irregularity2[frame2_list.index(same[i]) + n])
                        diff_Sol.append(Solidity[pos_same[i]] - Solidity2[frame2_list.index(same[i]) + n])
                        diff_strain.append(strain[pos_same[i]] - strain2[frame2_list.index(same[i]) + n])
                        if abs(diff_LA[-1]) > 3 and abs(diff_SA[-1]) > 3:
                            print(diff_LA[-1])
                            print(diff_SA[-1])
                            print(i)
                            print(same[i])
                n += 1
                #print('n = {}'.format(n))
        else:  # falls frame nicht doppelt in frame2
            frame2_pos = frame2_list.index(same[i])
            if (Solidity2[frame2_list.index(same[i])] > 0.95) & (Irregularity2[frame2_list.index(same[i])] < 1.06):
            # Zelle ungefähr auf gleicher x und y-Position
                if (abs(x_pos[pos_same[i]] - x_pos2[frame2_pos]) < 1) and (abs(y_pos[pos_same[i]] - y_pos2[frame2_pos]) < 1):
                    diff_radius.append(radius[pos_same[i]] - radius2[frame2_pos])
                    diff_RP.append(RP[pos_same[i]] - RP2[frame2_pos])
                    diff_LA.append(LA[pos_same[i]] - LA2[frame2_pos])
                    diff_SA.append(SA[pos_same[i]] - SA2[frame2_pos])
                    diff_Angle.append(Angle[pos_same[i]] - Angle2[frame2_pos])
                    diff_Irr.append(Irregularity[pos_same[i]] - Irregularity2[frame2_pos])
                    diff_Sol.append(Solidity[pos_same[i]] - Solidity2[frame2_pos])
                    diff_strain.append(strain[pos_same[i]] - strain2[frame2_pos])
                else:
                    #print('Not suitable frame for image from data1?')
                    no_equivalent += 1
                    no_suitable_frame.append(same[i])
                if abs(diff_LA[-1]) > 3 and abs(diff_SA[-1]) > 3:
                    print(diff_LA[-1])
                    print(diff_SA[-1])
                    print(i)
                    print(same[i])
    i+=1

#%% Calculate mean and std
print('diff RP = {} +/- {}'.format(np.mean(diff_RP),np.std(diff_RP)))
print('diff LA = {} +/- {}'.format(np.mean(diff_LA),np.std(diff_LA)))
print('diff SA = {} +/- {}'.format(np.mean(diff_SA),np.std(diff_SA)))
print('diff radius = {} +/- {}'.format(np.mean(diff_radius),np.std(diff_radius)))
print('diff strain = {} +/- {}'.format(np.mean(diff_strain),np.std(diff_strain)))
print('diff angle = {} +/- {}'.format(np.mean(diff_Angle),np.std(diff_Angle)))
print('diff Irregularity = {} +/- {}'.format(np.mean(diff_Irr),np.std(diff_Irr)))
print('diff Solidity = {} +/- {}'.format(np.mean(diff_Sol),np.std(diff_Sol)))

# Difference: Daten1-Daten2: Wenn positiv: Werte aus bandpass höher: möglicherweise Halo detektiert

# Bigger LA means bigger SA! There is no frame with abs(diff_LA[-1]) > 3 and abs(diff_SA[-1]) < 3
# --> probably Halo detected
#%% plot differences in RP
fig1=plt.figure(1, (8, 8))
border_width = 0.2
ax_size = [0+border_width, 0+border_width, 
           1-2*border_width, 1-2*border_width]
ax1 = fig1.add_axes(ax_size)

ax1.plot(diff_RP,'o')
ax1.set_ylabel('Difference in Radial Position in ($\mu m$)')
plt.show()

#%% plot differences in radius 
fig2=plt.figure(2, (8, 8))
ax2 = fig2.add_axes(ax_size)
ax2.plot(diff_radius,'o')
ax2.set_ylabel('Difference in radius in ($\mu m$)')
plt.show()

#%% plot differences in LA 
fig3=plt.figure(3, (8, 8))
ax3 = fig3.add_axes(ax_size)
ax3.plot(diff_LA,'o')
ax3.set_ylabel('Difference in LA in ($\mu m$)')
plt.show()

#%% plot differences in SA
fig4=plt.figure(4, (8, 8))
ax4 = fig4.add_axes(ax_size)
ax4.plot(diff_SA,'o')
ax4.set_ylabel('Difference in SA in ($\mu m$)')
plt.show()

#%% plot differences in Angle
fig5=plt.figure(5, (8, 8))
ax5 = fig5.add_axes(ax_size)
ax5.plot(diff_Angle,'o')
ax5.set_ylabel('Difference in Angle')
plt.show()

#%% plot differences in Irregularity
fig6=plt.figure(6, (8, 8))
ax6 = fig6.add_axes(ax_size)
ax6.plot(diff_Irr,'o')
ax6.set_ylabel('Difference in Irregularity')
plt.show()

#%% plot differences in Solidity
fig7=plt.figure(7, (8, 8))
ax7 = fig7.add_axes(ax_size)
ax7.plot(diff_Sol,'o')
ax7.set_ylabel('Difference in Solidity')
plt.show()

#%% plot differences in strain
fig8=plt.figure(8, (8, 8))
ax8 = fig8.add_axes(ax_size)
ax8.plot(diff_strain,'o')
ax8.set_ylabel('Difference in strain')
plt.show()

#%% plot histogram of cell density in channel (margination)
fig9=plt.figure(9, (6, 3))
ax9 = fig9.add_axes(ax_size)
bin_width = 25
hist, bin_edges = np.histogram(RP, bins=np.arange(-100 + bin_width/2, 101 - bin_width/2, bin_width), density=False)
plt.bar(bin_edges[:-1]+bin_width/4, hist, width=bin_width*0.4, edgecolor = 'black',label='with bandpass image')
hist2, bin_edges2 = np.histogram(RP2, bins=np.arange(-100 + bin_width/2, 101 - bin_width/2, bin_width), density=False)
plt.bar(bin_edges2[:-1]+3*bin_width/4, hist, width=bin_width*0.4, edgecolor = 'black',label='with original image')
ax9.set_xlabel('radial position in channel ($\u03BC m$)')
ax9.set_ylabel('# of cells')
ax9.set_xlim((-100,100))
ticks = np.arange(-100,101,bin_width)
labels = ticks.astype(int)
labels = labels.astype(str)
ax9.set_xticks(ticks)
ax9.set_xticklabels(labels) 
plt.legend()


#%% plot histogram of cell radius in channel (margination)
fig10=plt.figure(10, (6, 5))
ax10 = fig10.add_axes(ax_size)
radius = np.sqrt(LA * SA / 4)  # radius is sqrt(a*b)
bin_width = 10
bins=np.arange(0, 101 - bin_width/2, bin_width)
bin_means, bin_edges, binnumber = stats.binned_statistic(abs(RP),radius, statistic='mean', bins=bins)
bin_std, bin_edges, binnumber = stats.binned_statistic(abs(RP),radius, statistic='std', bins=bins)
plt.bar(bin_edges[:-1]+bin_width/4,bin_means, width=bin_width*0.4, yerr=bin_std, edgecolor = 'black',label='with bandpass image')
bin_means2, bin_edges2, binnumber2 = stats.binned_statistic(abs(RP2),radius2, statistic='mean', bins=bins)
bin_std2, bin_edges2, binnumber2 = stats.binned_statistic(abs(RP2),radius2, statistic='std', bins=bins)
plt.bar(bin_edges2[:-1]+3*bin_width/4,bin_means2, width=bin_width*0.4, yerr=bin_std, edgecolor = 'black',label='with original image')
#plt.plot(abs(RP),radius, 'o', markerfacecolor='#1f77b4', markersize=3.0,markeredgewidth=0)
ax10.set_xlabel('radial position in channel ($\u03BC m$)')
ticks = np.arange(0 + bin_width/2,101,bin_width)
ax10.set_xlim((0,100))
labels = ticks.astype(int)
labels = labels.astype(str)
ax10.set_xticks(ticks)
ax10.set_xticklabels(labels) 
ax10.set_ylabel('Radius of cells')
plt.legend()

#%% plot mean strain for position in channel
fig11=plt.figure(11, (6, 3))
ax11 = fig11.add_axes(ax_size)
radius = np.sqrt(LA * SA/ 4)
bin_width = 10
bins=np.arange(0, 101 - bin_width/2, bin_width)
bin_means, bin_edges, binnumber = stats.binned_statistic(abs(RP),strain, statistic='mean', bins=bins)
bin_std, bin_edges, binnumber = stats.binned_statistic(abs(RP),strain, statistic='std', bins=bins)
plt.bar(bin_edges[:-1]+bin_width/4,bin_means, width=bin_width*0.4, yerr=bin_std, edgecolor = 'black',label='with bandpass image')
bin_means2, bin_edges2, binnumber2 = stats.binned_statistic(abs(RP2),strain2, statistic='mean', bins=bins)
bin_std2, bin_edges2, binnumber2 = stats.binned_statistic(abs(RP2),strain2, statistic='std', bins=bins)
plt.bar(bin_edges2[:-1]+3*bin_width/4,bin_means2, width=bin_width*0.4, yerr=bin_std, edgecolor = 'black',label='with original image')
#plt.plot(abs(RP),radius, 'o', markerfacecolor='#1f77b4', markersize=3.0,markeredgewidth=0)
ax11.set_xlabel('radial position in channel ($\u03BC m$)')
ticks = np.arange(0 + bin_width/2,101,bin_width)
ax11.set_xlim((0,100))
labels = ticks.astype(int)
labels = labels.astype(str)
ax11.set_xticks(ticks)
ax11.set_xticklabels(labels) 
ax11.set_ylabel('Mean Strain of cells')
plt.legend()