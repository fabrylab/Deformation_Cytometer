# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:22:34 2020
@author: Selina Sonntag
"""

# !/usr/bin/python
# -*- coding: utf-8 -*-

# this program analyses a subset (100 images) of test images, changes the
# parameters of the Canny edge detection
# in a defined range (sigam, low threshold, high threshold), and calculates
# the number of detected cells
# (first 50 images), good detected cells (last 50 images), detected halos,
# non-elliptical detected regions
# and the mean calculation time and plots those results in a heatmap

import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog
import imageio  # offers better image read options
import sys
import glob
import os
from skimage import feature
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
import matplotlib.gridspec as gridspec
import time
import seaborn as sns
sns.set()

# initialization of user interface
root = Tk()
# we don't want a full GUI, so keep the root window from appearing
root.withdraw()
plt.ion()


class timeit:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args):
        print("Timeit:", self.name, time.time()-self.start_time)

# %%  Onclick to stop program


def onclick(event):
    global good_bad
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    if event.dblclick:
        good_bad = 2
    else:
        good_bad = 1


# ----------general fonts for plots and figures----------
font = {'family' : 'sans-serif',
        'sans-serif':['Arial'],
        'weight' : 'normal',
        'size'   : 18}
plt.rc('font', **font)
plt.rc('legend', fontsize=12)
plt.rc('axes', titlesize=18)  

# %%----------read data from a csv (text) file---------------------------------
# show an "Open" dialog box and return the path to the selected file
imfile = filedialog.askopenfilename(title="select the image file",
                                    filetypes=[("jpg", '*.jpg')])
if imfile == '':
    print('empty')
    sys.exit()

impath = os.path.dirname(os.path.abspath(imfile))
flatfield_file = impath + r'\flatfield.npy'
os.chdir(impath)

# structural element for binary erosion
struct = morphology.generate_binary_structure(2, 1)

i = 0
pixel_size = 0.36e-6  # in m for 20x AlliedVision
channel_width = 200e-6/pixel_size  # in pixels

plt.close('all')
fig1 = plt.figure(1, (28.2 * 300 / 540, 7))
spec = gridspec.GridSpec(ncols=20, nrows=5, figure=fig1)
ax1 = fig1.add_subplot(spec[0:6, 0:5])
ax2 = fig1.add_subplot(spec[0:6, 5:10], sharex=ax1, sharey=ax1)
ax3 = fig1.add_subplot(spec[0:6, 10:15], sharex=ax1, sharey=ax1)
ax4 = fig1.add_subplot(spec[0:6, 15:20], sharex=ax1, sharey=ax1)

count = 100

sigmas = np.linspace(1, 5, 5)
low_thresh = np.linspace(0.5, 0.7, 3)
high_thresh = np.linspace(0.8, 0.95, 4)
Time_Canny = [[[[0 for k in range(count)] for i in range(len(sigmas))] for
               n in range(len(low_thresh))] for j in range(len(high_thresh))]
Time_Canny = np.asarray(Time_Canny, dtype='float')
# shape: (high_threshold_low_threshold,sigma,frame)

size_cells = [[[[np.nan for k in range(count)] for i in range(len(sigmas))]
               for n in range(len(low_thresh))] for j in range(len(high_thresh))]
size_cells = np.asarray(size_cells, dtype='float')
# shape: (high_threshold_low_threshold,sigma,frame)

d_max_total = [[[[0 for k in range(count)] for i in range(len(sigmas))] for
               n in range(len(low_thresh))] for j in range(len(high_thresh))]
d_max_total = np.asarray(d_max_total, dtype='float')

std_total = [[[[0 for k in range(count)] for i in range(len(sigmas))] for
               n in range(len(low_thresh))] for j in range(len(high_thresh))]
std_total = np.asarray(std_total, dtype='float')

print(np.shape(Time_Canny))
# shape: (high_threshold, low_threshold, sigma, frame)

detected_cells_bad = [[[0 for i in range(len(sigmas))] for n in range(len(low_thresh))] for j in range(len(high_thresh))]
detected_cells_bad = np.asarray(detected_cells_bad, dtype='float')
print(np.shape(detected_cells_bad))
# shape: (high_threshold, low_threshold, sigma)

detected_cells_good = [[[0 for i in range(len(sigmas))] for n in range(len(low_thresh))] for j in range(len(high_thresh))]
detected_cells_good = np.asarray(detected_cells_good, dtype='float')
print(np.shape(detected_cells_good))
# shape: (high_threshold, low_threshold, sigma)

not_round_cell = [[[0 for i in range(len(sigmas))] for n in range(len(low_thresh))] for j in range(len(high_thresh))]
not_round_cell = np.asarray(not_round_cell, dtype='float')
# shape: (high_threshold_low_threshold,sigma)

Halo = [[[0 for i in range(len(sigmas))] for n in range(len(low_thresh))] for j in range(len(high_thresh))]
Halo = np.asarray(Halo, dtype='float')
print(np.shape(Halo))
# shape: (high_threshold, low_threshold, sigma)

Halo_2 = [[[0 for i in range(len(sigmas))] for n in range(len(low_thresh))] for j in range(len(high_thresh))]
Halo_2 = np.asarray(Halo_2, dtype='float')
# print(np.shape(Halo)) #shape: (high_threshold, low_threshold, sigma)

std_ir = [[[0 for i in range(len(sigmas))] for n in range(len(low_thresh))] for j in range(len(high_thresh))]
std_ir = np.asarray(std_ir, dtype='float')

d_max = [[[0 for i in range(len(sigmas))] for n in range(len(low_thresh))] for j in range(len(high_thresh))]
d_max = np.asarray(d_max, dtype='float')

r = [[[0 for i in range(len(sigmas))] for n in range(len(low_thresh))] for j in range(len(high_thresh))]
r = np.asarray(r, dtype='float')

Cell_halo = []
Cell_halo_2 = []
Halo_distance = []
Halo_std = []

for file in glob.glob("*.jpg"):
    i = i + 1
    print(i)
    if i % 1 == 0:
        im = imageio.imread(file)
        im = im[:, :, 0]
        im = im.astype(float)
        sizes = np.shape(im)
        if i > 0:  # to "jump" to a higher position
            im = np.asarray(im, dtype='float')
            im_mean = np.mean(im)
            # %% iterate over every parameter, calculate number of good/bad cell and non-elliptical
            for j in range(len(sigmas)):
                for n in range(len(low_thresh)):
                    for k in range(len(high_thresh)):
                        start_time = time.time()
                        # edge detection
                        im1 = feature.canny(im, sigma=sigmas[j], low_threshold=low_thresh[n], high_threshold=high_thresh[k], use_quantiles=True)
                        end_time = time.time()
                        x = end_time - start_time
                        Time_Canny[k, n, j, i - 1] = x
                        print('Image: {} \n sigma = {}, low = {}, high = {}'.format(i, sigmas[j], low_thresh[n], high_thresh[k]))
                        im2 = morphology.binary_fill_holes(im1, structure=struct).astype(int)  # fill holes
                        im3 = morphology.binary_erosion(im2, structure=struct).astype(int)  # erode to remove lines and small schmutz
                        label_image = label(im3)  # label all ellipses (and other objects)
                        ax1.clear()
                        ax1.set_axis_off()
                        ax1.imshow(im, cmap='viridis')
                        ax2.clear()
                        ax2.set_axis_off()
                        # ax2.imshow(im5,cmap='gray')
                        ax2.imshow(im1, cmap='gray')
                        ax3.clear()
                        ax3.set_axis_off()
                        ax3.imshow(im2, cmap='gray')
                        ax4.clear()
                        ax4.set_axis_off()
                        ax4.imshow(im3, cmap='gray')

                        plt.show()
                        plt.pause(0.1)
                        for region in regionprops(label_image, im):
                            if region.area >= 600:  # analyze only large, dark ellipses
                                l = region.label
            #                   structure = np.std(im4[label_image==l])
                                a = region.major_axis_length/2
                                b = region.minor_axis_length/2
                                r[k, n, j] = np.sqrt(a*b)
                                circum = np.pi*((3*(a+b))-np.sqrt(10*a*b+3*(a**2+b**2)))
                                theta = np.arange(0, 2*np.pi, np.pi/4)
                                strain = (a-b)/r[k, n, j]

                                dd = np.arange(0, int(3*r[k, n, j]))
                                i_r = np.zeros(int(3*r[k, n, j]))
                                
                                for d in range(0, int(3*r[k, n, j])):
                                    x = np.round(d*(np.cos(theta) + strain*np.sin(theta)) + region.centroid[1]).astype(int)
                                    y = np.round(d*np.sin(theta) + region.centroid[0]).astype(int)
                                    index = (x < 0) | (x >= im.shape[1]) | (y < 0) | (y >=im.shape[0])
                                    x = x[~index]
                                    y = y[~index]
                                    
                                    i_r[d] = np.mean(im[y,x])
                                    #print(np.max(x), np.max(y))
                                    #print(i_r[d])
                                
                                d_max[k, n, j] = np.nanargmax(i_r)
                                std_ir[k,n, j] = np.nanstd(i_r)
                                print(std_ir[k,n,j])
                                
                                std_total[k, n, j, i-1] = np.nanstd(i_r)
                                d_max_total[k, n, j, i-1] = np.nanargmax(i_r)
                                
                                #print(d_max / r)
                                if i < 51 and region.perimeter/circum < 1.06:
                                    detected_cells_bad[k, n, j] += 1
                                    print("Region detected")
                                # Ab Bild 51 sind gute Zellen, die erkannt werden sollen!!
                                elif i >= 51 and region.perimeter/circum < 1.06:
                                    detected_cells_good[k, n, j] += 1
                                    print("Good Cell detected!")
                                # Für welche Parameter sind Regionen nicht ellipsenförmig?
                                elif region.perimeter/circum >= 1.06:
                                    not_round_cell[k, n, j] += 1
                                    print("Cell not round!")
                                    # good_bad = 0
                                    # while good_bad ==0:
                                    #     cid = fig1.canvas.mpl_connect('button_press_event', onclick)
                                    #     plt.pause(0.5)
                                    #     if good_bad == 2:
                                    #         sys.exit() #exit upon double click
                                    
                                # Halo Kriterium
                                # if sigmas[j] == 4.0 and high_thresh[k] == 0.8 and low_thresh[n] == 0.5 and i == 2:
                                #     print(k,n,j)
                                #     sys.exit()  
                                '''    
                                if ((d_max/r>1 and d_max/r<1.4) or (d_max/r>0.5 and np.std(i_r)/im_mean<0.03)):
                                    Halo_2[k, n, j] += 1
                                    Cell_halo_2.append(i)
                                    print("Halo 2222!")
                                    good_bad = 0
                                    while good_bad == 0:
                                        cid = fig1.canvas.mpl_connect('button_press_event', onclick)
                                        plt.pause(0.5)
                                        if good_bad == 2:
                                            sys.exit() #exit upon double click
                                 '''   
                                
                                size_cells[k, n, j, i-1] = region.area

                 ####################NOT counting the cells with halo!!!!#############               
     #                           #np.nanstd(np.where(np.isclose(a,0), np.nan, a))
            # wenn Zelle sich sehr in ihrer Größe unterscheidet, weist das auf Halo hin 
    # %% Calculate Halos

    if np.nanstd(size_cells[:, :, :, i-1]) > 100:
        for j in range(len(sigmas)):
            for n in range(len(low_thresh)):
                for k in range(len(high_thresh)):
                    if size_cells[k, n, j, i-1] > np.nanmean(size_cells[:, :, :, i-1]) + 1000:
                        print('Halo detected at {},{},{} for image {}'.format(k, n, j, i))
                        Halo[k, n, j] += 1
                        Cell_halo.append(i)
                        Halo_distance.append(d_max[k, n, j] / r[k, n, j])
                        Halo_std.append(std_ir[k, n, j]/im_mean)
                        if std_ir[k, n, j]/im_mean == np.nan:
                            print(std_ir[k, n, j])
                            print(im_mean)
                            good_bad = 0
                            while good_bad ==0:
                                 cid = fig1.canvas.mpl_connect('button_press_event', onclick)
                                 plt.pause(0.5)
                                 if good_bad == 2:
                                     sys.exit() #exit upon double click

                        if i < 51 and region.perimeter/circum < 1.06: 
                            detected_cells_bad[k,n,j] -= 1
                            print('Halo!')
                        elif i >= 51 and region.perimeter/circum < 1.06:  # Ab Bild 51 sind gute Zellen, die erkannt werden sollen!!
                            detected_cells_good[k, n, j] -= 1
                            print('Halo 2!')
                        # good_bad = 0
                        # while good_bad ==0:
                        #     cid = fig1.canvas.mpl_connect('button_press_event', onclick)
                        #     plt.pause(0.5)
                        #     if good_bad == 2:
                        #         sys.exit() #exit upon double click




# #Detected cells for i<=50 and ellipses     
# print('Detected until 50')                            
# print(detected_cells_bad)
# #Detected cells for i>50 and ellipses 
# print('Detected after 50')    
# print(detected_cells_good)
# #Number of cells which are not elliptical
# print('Not round')
# print(not_round_cell)
# #Number of cells with Halo
# print('Halos')
# print(Halo)

# %% Calculate mean time

# Time for Canny edge detection for each frame
# print(Time_Canny)
Mean_time_canny = [[[0 for k in range(len(sigmas))] for n in
                    range(len(low_thresh))]for i in range(len(high_thresh))]
Mean_time_canny = np.array(Mean_time_canny, dtype='float')
for i in range(len(high_thresh)):
    for n in range(len(low_thresh)):
        for k in range(len(sigmas)):
            Mean_time_canny[i, n, k] = np.mean(Time_Canny[i, n, k, :])
# print(Mean_time_canny)

# %% Recall and precision
# Precision: true positive / (true positive + false positive)
# Recall: true positive / total positive (total positive is 50)

recall_good_cells = detected_cells_good / 50

precision = [[[0 for i in range(len(sigmas))] for
              n in range(len(low_thresh))] for j in range(len(high_thresh))]
precision = np.asarray(precision, dtype='float')

# Only possible, if Nenner != 0
# precision = detected_cells_good / (detected_cells_good + detected_cells_bad)

for i in range(len(sigmas)):
    for n in range(len(low_thresh)):
        for k in range(len(high_thresh)):
            if (detected_cells_good[k, n, i] + detected_cells_bad[k, n, i]) != 0:
                precision[k, n, i] = detected_cells_good[k, n, i] /(detected_cells_good[k, n, i] + detected_cells_bad[k, n, i])
            else:
                precision[k, n, i] = 0


F_score = [[[0 for i in range(len(sigmas))] for
              n in range(len(low_thresh))] for j in range(len(high_thresh))]
F_score = np.asarray(F_score, dtype='float')

for i in range(len(sigmas)):
    for n in range(len(low_thresh)):
        for k in range(len(high_thresh)):
            if (detected_cells_good[k, n, i] + detected_cells_bad[k, n, i]) != 0:
                F_score[k, n, i] = 2 * (precision[k, n, i] * recall_good_cells[k, n, i]) / (precision[k, n, i] + recall_good_cells[k, n, i])
            else:
                F_score = 0
'''
# %% Plot data: 1. Detected cells
plt.rcParams['axes.titlesize'] = 12            

f,(ax1,ax2,ax3,ax4,ax5, axcb) = plt.subplots(1,6, figsize=(15,5),
            gridspec_kw={'width_ratios':[1,1,1,1,1,0.08]})
plt.gcf().subplots_adjust(bottom=0.15)
xlabels = ['{:.2f}'.format(x) for x in low_thresh]
ylabels = ['{:.2f}'.format(y) for y in high_thresh]
minimum = np.min(detected_cells_bad)
maximum = np.max(detected_cells_bad)
#ax1.set_xticks(ax1.get_xticks()[::3])
#ax1.set_xticklabels(xlabels[::3],)
#ax1.set_yticks(ax1.get_yticks()[::3])
#ax1.set_yticklabels(ylabels[::3],)
ax1.get_shared_y_axes().join(ax2,ax3,ax4,ax5)
g1 = sns.heatmap(detected_cells_bad[:,:,0],cmap="inferno",vmin = minimum,
                 vmax = maximum,cbar=False,ax=ax1,xticklabels=xlabels, yticklabels=ylabels)
g1.set_ylabel('High threshold')
g1.set_xlabel('Low threshold')
g1.set_title('\u03C3 = {:.2f}'.format(sigmas[0]))
g2 = sns.heatmap(detected_cells_bad[:,:,1],cmap="inferno",vmin = minimum,
                 vmax = maximum,cbar=False,ax=ax2,xticklabels=xlabels, yticklabels=ylabels)
g2.set_ylabel('')
g2.set_xlabel('Low threshold')
g2.set_yticks([])
g2.set_title('\u03C3 = {:.2f}'.format(sigmas[1]))
g3 = sns.heatmap(detected_cells_bad[:,:,2],cmap="inferno",vmin = minimum,
                 vmax = maximum,cbar=False,ax=ax3,xticklabels=xlabels, yticklabels=ylabels)
g3.set_ylabel('')
g3.set_xlabel('Low threshold')
g3.set_yticks([])
g3.set_title('\u03C3 = {:.2f}'.format(sigmas[2]))
g4 = sns.heatmap(detected_cells_bad[:,:,3],cmap="inferno",vmin = minimum,
                 vmax = maximum,cbar=False,ax=ax4,xticklabels=xlabels, yticklabels=ylabels)
g4.set_ylabel('')
g4.set_xlabel('Low threshold')
g4.set_yticks([])
g4.set_title('\u03C3 = {:.2f}'.format(sigmas[3]))
g5 = sns.heatmap(detected_cells_bad[:,:,4],cmap="inferno", vmin = minimum,
                 vmax = maximum,ax=ax5, cbar_ax = axcb ,xticklabels=xlabels, yticklabels=ylabels)
g5.set_ylabel('')
g5.set_xlabel('Low threshold')
g5.set_yticks([])
g5.set_title('\u03C3 = {:.2f}'.format(sigmas[4]))
for ax in [g1,g2,g3,g4,g5]:
    tl = ax.get_xticklabels()
    ax.set_xticklabels(tl, rotation=90)
    tly = ax.get_yticklabels()
    ax.set_yticklabels(tly, rotation=0)
f.suptitle('Number of detected cells (subset of bad cells)',fontsize = 19)    
#plt.savefig("C:/Users/selin/OneDrive/Dokumente/Uni/6_SS20/Bachelorarbeit/Bilder Filter/Canny filter/Evalutate parameters/4_cells_bad.png", bbox_inches='tight')
plt.show()    

#%% Plot data: 2. detected_cells_good
plt.rcParams['axes.titlesize'] = 12            

f,(ax1,ax2,ax3,ax4,ax5, axcb) = plt.subplots(1,6, figsize=(15,5),
            gridspec_kw={'width_ratios':[1,1,1,1,1,0.08]})
plt.gcf().subplots_adjust(bottom=0.15)
xlabels = ['{:.2f}'.format(x) for x in low_thresh]
ylabels = ['{:.2f}'.format(y) for y in high_thresh]
minimum = np.min(detected_cells_good)
maximum = np.max(detected_cells_good)
#ax1.set_xticks(ax1.get_xticks()[::3])
#ax1.set_xticklabels(xlabels[::3],)
#ax1.set_yticks(ax1.get_yticks()[::3])
#ax1.set_yticklabels(ylabels[::3],)
ax1.get_shared_y_axes().join(ax2,ax3,ax4,ax5)
g1 = sns.heatmap(detected_cells_good[:,:,0], vmin = minimum, vmax = maximum,
                 cmap="inferno",cbar=False,ax=ax1,xticklabels=xlabels, yticklabels=ylabels)
g1.set_ylabel('High threshold')
g1.set_xlabel('Low threshold')
g1.set_title('\u03C3 = {:.2f}'.format(sigmas[0]))
g2 = sns.heatmap(detected_cells_good[:,:,1], vmin = minimum, vmax = maximum,
                 cmap="inferno",cbar=False,ax=ax2,xticklabels=xlabels, yticklabels=ylabels)
g2.set_ylabel('')
g2.set_xlabel('Low threshold')
g2.set_yticks([])
g2.set_title('\u03C3 = {:.2f}'.format(sigmas[1]))
g3 = sns.heatmap(detected_cells_good[:,:,2], vmin = minimum, vmax = maximum,
                 cmap="inferno",cbar=False,ax=ax3,xticklabels=xlabels, yticklabels=ylabels)
g3.set_ylabel('')
g3.set_xlabel('Low threshold')
g3.set_yticks([])
g3.set_title('\u03C3 = {:.2f}'.format(sigmas[2]))
g4 = sns.heatmap(detected_cells_good[:,:,3], vmin = minimum, vmax = maximum,
                 cmap="inferno",cbar=False,ax=ax4,xticklabels=xlabels, yticklabels=ylabels)
g4.set_ylabel('')
g4.set_xlabel('Low threshold')
g4.set_yticks([])
g4.set_title('\u03C3 = {:.2f}'.format(sigmas[3]))
g5 = sns.heatmap(detected_cells_good[:,:,4], vmin = minimum, vmax = maximum,
                 cmap="inferno",ax=ax5, cbar_ax = axcb ,xticklabels=xlabels, yticklabels=ylabels)
g5.set_ylabel('')
g5.set_xlabel('Low threshold')
g5.set_yticks([])
g5.set_title('\u03C3 = {:.2f}'.format(sigmas[4]))
for ax in [g1,g2,g3,g4,g5]:
    tl = ax.get_xticklabels()
    ax.set_xticklabels(tl, rotation=90)
    tly = ax.get_yticklabels()
    ax.set_yticklabels(tly, rotation=0)
f.suptitle('Number of detected cells (subset of good cells)',fontsize = 19)    
plt.savefig("C:/Users/selin/OneDrive/Dokumente/Uni/6_SS20/Bachelorarbeit/Bilder Filter/Canny filter/Evalutate parameters/1_1_cells_good.png", bbox_inches='tight')
plt.show()  


#%% Plot data: 3. Not round

f,(ax1,ax2,ax3,ax4,ax5, axcb) = plt.subplots(1,6, figsize=(15,5),
            gridspec_kw={'width_ratios':[1,1,1,1,1,0.08]})
plt.gcf().subplots_adjust(bottom=0.15)
xlabels = ['{:.2f}'.format(x) for x in low_thresh]
ylabels = ['{:.2f}'.format(y) for y in high_thresh]
minimum = np.min(not_round_cell)
maximum = np.max(not_round_cell)
#ax1.set_xticks(ax1.get_xticks()[::3])
#ax1.set_xticklabels(xlabels[::3],)
#ax1.set_yticks(ax1.get_yticks()[::3])
#ax1.set_yticklabels(ylabels[::3],)
ax1.get_shared_y_axes().join(ax2,ax3,ax4,ax5)
g1 = sns.heatmap(not_round_cell[:,:,0], vmin = minimum, vmax = maximum,
                 cmap="inferno",cbar=False,ax=ax1,xticklabels=xlabels, yticklabels=ylabels)
g1.set_ylabel('High threshold')
g1.set_xlabel('Low threshold')
g1.set_title('\u03C3 = {:.2f}'.format(sigmas[0]))
g2 = sns.heatmap(not_round_cell[:,:,1], vmin = minimum, vmax = maximum,
                 cmap="inferno",cbar=False,ax=ax2,xticklabels=xlabels, yticklabels=ylabels)
g2.set_ylabel('')
g2.set_xlabel('Low threshold')
g2.set_yticks([])
g2.set_title('\u03C3 = {:.2f}'.format(sigmas[1]))
g3 = sns.heatmap(not_round_cell[:,:,2], vmin = minimum, vmax = maximum,
                 cmap="inferno",cbar=False,ax=ax3,xticklabels=xlabels, yticklabels=ylabels)
g3.set_ylabel('')
g3.set_xlabel('Low threshold')
g3.set_yticks([])
g3.set_title('\u03C3 = {:.2f}'.format(sigmas[2]))
g4 = sns.heatmap(not_round_cell[:,:,3], vmin = minimum, vmax = maximum,
                 cmap="inferno",cbar=False,ax=ax4,xticklabels=xlabels, yticklabels=ylabels)
g4.set_ylabel('')
g4.set_xlabel('Low threshold')
g4.set_yticks([])
g4.set_title('\u03C3 = {:.2f}'.format(sigmas[3]))
g5 = sns.heatmap(not_round_cell[:,:,4], vmin = minimum, vmax = maximum,
                 cmap="inferno",ax=ax5, cbar_ax = axcb ,xticklabels=xlabels, yticklabels=ylabels)
g5.set_ylabel('')
g5.set_xlabel('Low threshold')
g5.set_yticks([])
g5.set_title('\u03C3 = {:.2f}'.format(sigmas[4]))
for ax in [g1,g2,g3,g4,g5]:
    tl = ax.get_xticklabels()
    ax.set_xticklabels(tl, rotation=90)
    tly = ax.get_yticklabels()
    ax.set_yticklabels(tly, rotation=0)
f.suptitle('Number of not-elliptical detected regions',fontsize = 19)    
#plt.savefig("C:/Users/selin/OneDrive/Dokumente/Uni/6_SS20/Bachelorarbeit/Bilder Filter/Canny filter/Evalutate parameters/4_elliptical.png", bbox_inches='tight')
plt.show()


#%% Plot data: 4. Halos

f,(ax1,ax2,ax3,ax4,ax5, axcb) = plt.subplots(1,6, figsize=(15,5),
            gridspec_kw={'width_ratios':[1,1,1,1,1,0.08]})
plt.gcf().subplots_adjust(bottom=0.15)
xlabels = ['{:.2f}'.format(x) for x in low_thresh]
ylabels = ['{:.2f}'.format(y) for y in high_thresh]
minimum = np.min(Halo)
maximum = np.max(Halo)
#ax1.set_xticks(ax1.get_xticks()[::3])
#ax1.set_xticklabels(xlabels[::3],)
#ax1.set_yticks(ax1.get_yticks()[::3])
#ax1.set_yticklabels(ylabels[::3],)
ax1.get_shared_y_axes().join(ax2,ax3,ax4,ax5)
g1 = sns.heatmap(Halo[:,:,0], vmin = minimum, vmax = maximum, cmap="inferno",
                 cbar=False,ax=ax1,xticklabels=xlabels, yticklabels=ylabels)
g1.set_ylabel('High threshold')
g1.set_xlabel('Low threshold')
g1.set_title('\u03C3 = {:.2f}'.format(sigmas[0]))
g2 = sns.heatmap(Halo[:,:,1], vmin = minimum, vmax = maximum, cmap="inferno",
                 cbar=False,ax=ax2,xticklabels=xlabels, yticklabels=ylabels)
g2.set_ylabel('')
g2.set_xlabel('Low threshold')
g2.set_yticks([])
g2.set_title('\u03C3 = {:.2f}'.format(sigmas[1]))
g3 = sns.heatmap(Halo[:,:,2], vmin = minimum, vmax = maximum, cmap="inferno",
                 cbar=False,ax=ax3,xticklabels=xlabels, yticklabels=ylabels)
g3.set_ylabel('')
g3.set_xlabel('Low threshold')
g3.set_yticks([])
g3.set_title('\u03C3 = {:.2f}'.format(sigmas[2]))
g4 = sns.heatmap(Halo[:,:,3], vmin = minimum, vmax = maximum, cmap="inferno",
                 cbar=False,ax=ax4,xticklabels=xlabels, yticklabels=ylabels)
g4.set_ylabel('')
g4.set_xlabel('Low threshold')
g4.set_yticks([])
g4.set_title('\u03C3 = {:.2f}'.format(sigmas[3]))
g5 = sns.heatmap(Halo[:,:,4], vmin = minimum, vmax = maximum, cmap="inferno",
                 ax=ax5, cbar_ax = axcb ,xticklabels=xlabels, yticklabels=ylabels)
g5.set_ylabel('')
g5.set_xlabel('Low threshold')
g5.set_yticks([])
g5.set_title('\u03C3 = {:.2f}'.format(sigmas[4]))
for ax in [g1,g2,g3,g4,g5]:
    tl = ax.get_xticklabels()
    ax.set_xticklabels(tl, rotation=90)
    tly = ax.get_yticklabels()
    ax.set_yticklabels(tly, rotation=0)
f.suptitle('Number of detected halos',fontsize = 19)    
#plt.savefig("C:/Users/selin/OneDrive/Dokumente/Uni/6_SS20/Bachelorarbeit/Bilder Filter/Canny filter/Evalutate parameters/4_Halo.png", bbox_inches='tight')
plt.show()

#%% Plot data: 5. Calculation time

f,(ax1,ax2,ax3,ax4,ax5, axcb) = plt.subplots(1,6, figsize=(15,5),
            gridspec_kw={'width_ratios':[1,1,1,1,1,0.08]})
plt.gcf().subplots_adjust(bottom=0.15)
xlabels = ['{:.2f}'.format(x) for x in low_thresh]
ylabels = ['{:.2f}'.format(y) for y in high_thresh]
minimum = np.min(Mean_time_canny)
maximum = np.max(Mean_time_canny)
#ax1.set_xticks(ax1.get_xticks()[::3])
#ax1.set_xticklabels(xlabels[::3],)
#ax1.set_yticks(ax1.get_yticks()[::3])
#ax1.set_yticklabels(ylabels[::3],)
ax1.get_shared_y_axes().join(ax2,ax3,ax4,ax5)
g1 = sns.heatmap(Mean_time_canny[:,:,0], vmin = minimum, vmax = maximum,
                 cmap="inferno",cbar=False,ax=ax1,xticklabels=xlabels, yticklabels=ylabels)
g1.set_ylabel('High threshold')
g1.set_xlabel('Low threshold')
g1.set_title('\u03C3 = {:.2f}'.format(sigmas[0]))
g2 = sns.heatmap(Mean_time_canny[:,:,1], vmin = minimum, vmax = maximum,
                 cmap="inferno",cbar=False,ax=ax2,xticklabels=xlabels, yticklabels=ylabels)
g2.set_ylabel('')
g2.set_xlabel('Low threshold')
g2.set_yticks([])
g2.set_title('\u03C3 = {:.2f}'.format(sigmas[1]))
g3 = sns.heatmap(Mean_time_canny[:,:,2], vmin = minimum, vmax = maximum,
                 cmap="inferno",cbar=False,ax=ax3,xticklabels=xlabels, yticklabels=ylabels)
g3.set_ylabel('')
g3.set_xlabel('Low threshold')
g3.set_yticks([])
g3.set_title('\u03C3 = {:.2f}'.format(sigmas[2]))
g4 = sns.heatmap(Mean_time_canny[:,:,3], vmin = minimum, vmax = maximum,
                 cmap="inferno",cbar=False,ax=ax4,xticklabels=xlabels, yticklabels=ylabels)
g4.set_ylabel('')
g4.set_xlabel('Low threshold')
g4.set_yticks([])
g4.set_title('\u03C3 = {:.2f}'.format(sigmas[3]))
g5 = sns.heatmap(Mean_time_canny[:,:,4], vmin = minimum, vmax = maximum,
                 cmap="inferno",ax=ax5, cbar_ax = axcb ,xticklabels=xlabels, yticklabels=ylabels)
g5.set_ylabel('')
g5.set_xlabel('Low threshold')
g5.set_yticks([])
g5.set_title('\u03C3 = {:.2f}'.format(sigmas[4]))
for ax in [g1,g2,g3,g4,g5]:
    tl = ax.get_xticklabels()
    ax.set_xticklabels(tl, rotation=90)
    tly = ax.get_yticklabels()
    ax.set_yticklabels(tly, rotation=0)
f.suptitle('Mean calculation time for Canny edge detection in seconds',fontsize = 19)    
#plt.savefig("C:/Users/selin/OneDrive/Dokumente/Uni/6_SS20/Bachelorarbeit/Bilder Filter/Canny filter/Evalutate parameters/4_Calc_time.png", bbox_inches='tight')
plt.show()


# %% Plot recall:

f, (ax1,ax2,ax3,ax4,ax5, axcb) = plt.subplots(1,6, figsize=(15,5),
            gridspec_kw={'width_ratios':[1,1,1,1,1,0.08]})
plt.gcf().subplots_adjust(bottom=0.15)
xlabels = ['{:.2f}'.format(x) for x in low_thresh]
ylabels = ['{:.2f}'.format(y) for y in high_thresh]
minimum = np.min(recall_good_cells)
maximum = np.max(recall_good_cells)
# ax1.set_xticks(ax1.get_xticks()[::3])
# ax1.set_xticklabels(xlabels[::3],)
# ax1.set_yticks(ax1.get_yticks()[::3])
# ax1.set_yticklabels(ylabels[::3],)
ax1.get_shared_y_axes().join(ax2,ax3,ax4,ax5)
g1 = sns.heatmap(recall_good_cells[:,:,0],cmap="inferno",vmin = minimum,
                 vmax = maximum,cbar=False,ax=ax1,xticklabels=xlabels, yticklabels=ylabels)
g1.set_ylabel('High threshold')
g1.set_xlabel('Low threshold')
g1.set_title('\u03C3 = {:.2f}'.format(sigmas[0]))
g2 = sns.heatmap(recall_good_cells[:,:,1],cmap="inferno",vmin = minimum,
                 vmax = maximum,cbar=False,ax=ax2,xticklabels=xlabels, yticklabels=ylabels)
g2.set_ylabel('')
g2.set_xlabel('Low threshold')
g2.set_yticks([])
g2.set_title('\u03C3 = {:.2f}'.format(sigmas[1]))
g3 = sns.heatmap(recall_good_cells[:,:,2],cmap="inferno",vmin = minimum,
                 vmax = maximum,cbar=False,ax=ax3,xticklabels=xlabels, yticklabels=ylabels)
g3.set_ylabel('')
g3.set_xlabel('Low threshold')
g3.set_yticks([])
g3.set_title('\u03C3 = {:.2f}'.format(sigmas[2]))
g4 = sns.heatmap(recall_good_cells[:,:,3],cmap="inferno",vmin = minimum,
                 vmax = maximum,cbar=False,ax=ax4,xticklabels=xlabels, yticklabels=ylabels)
g4.set_ylabel('')
g4.set_xlabel('Low threshold')
g4.set_yticks([])
g4.set_title('\u03C3 = {:.2f}'.format(sigmas[3]))
g5 = sns.heatmap(recall_good_cells[:,:,4],cmap="inferno", vmin = minimum,
                 vmax = maximum,ax=ax5, cbar_ax = axcb ,xticklabels=xlabels, yticklabels=ylabels)
g5.set_ylabel('')
g5.set_xlabel('Low threshold')
g5.set_yticks([])
g5.set_title('\u03C3 = {:.2f}'.format(sigmas[4]))
for ax in [g1,g2,g3,g4,g5]:
    tl = ax.get_xticklabels()
    ax.set_xticklabels(tl, rotation=90)
    tly = ax.get_yticklabels()
    ax.set_yticklabels(tly, rotation=0)
f.suptitle('Recall for subset of good cells',fontsize = 19)    
plt.savefig("C:/Users/selin/OneDrive/Dokumente/Uni/6_SS20/Bachelorarbeit/Bilder Filter/Canny filter/Evalutate parameters/2_2_Recall.png", bbox_inches='tight')
plt.show()    

# %% Plot precision

f,(ax1,ax2,ax3,ax4,ax5, axcb) = plt.subplots(1,6, figsize=(15,5),
            gridspec_kw={'width_ratios':[1,1,1,1,1,0.08]})
plt.gcf().subplots_adjust(bottom=0.15)
xlabels = ['{:.2f}'.format(x) for x in low_thresh]
ylabels = ['{:.2f}'.format(y) for y in high_thresh]
minimum = np.min(precision)
maximum = np.max(precision)
# ax1.set_xticks(ax1.get_xticks()[::3])
# ax1.set_xticklabels(xlabels[::3],)
# ax1.set_yticks(ax1.get_yticks()[::3])
# ax1.set_yticklabels(ylabels[::3],)
ax1.get_shared_y_axes().join(ax2,ax3,ax4,ax5)
g1 = sns.heatmap(precision[:,:,0],cmap="inferno",vmin = minimum,
                 vmax = maximum,cbar=False,ax=ax1,xticklabels=xlabels, yticklabels=ylabels)
g1.set_ylabel('High threshold')
g1.set_xlabel('Low threshold')
g1.set_title('\u03C3 = {:.2f}'.format(sigmas[0]))
g2 = sns.heatmap(precision[:,:,1],cmap="inferno",vmin = minimum,
                 vmax = maximum,cbar=False,ax=ax2,xticklabels=xlabels, yticklabels=ylabels)
g2.set_ylabel('')
g2.set_xlabel('Low threshold')
g2.set_yticks([])
g2.set_title('\u03C3 = {:.2f}'.format(sigmas[1]))
g3 = sns.heatmap(precision[:,:,2],cmap="inferno",vmin = minimum,
                 vmax = maximum,cbar=False,ax=ax3,xticklabels=xlabels, yticklabels=ylabels)
g3.set_ylabel('')
g3.set_xlabel('Low threshold')
g3.set_yticks([])
g3.set_title('\u03C3 = {:.2f}'.format(sigmas[2]))
g4 = sns.heatmap(precision[:,:,3],cmap="inferno",vmin = minimum,
                 vmax = maximum,cbar=False,ax=ax4,xticklabels=xlabels, yticklabels=ylabels)
g4.set_ylabel('')
g4.set_xlabel('Low threshold')
g4.set_yticks([])
g4.set_title('\u03C3 = {:.2f}'.format(sigmas[3]))
g5 = sns.heatmap(precision[:,:,4],cmap="inferno", vmin = minimum,
                 vmax = maximum,ax=ax5, cbar_ax = axcb ,xticklabels=xlabels, yticklabels=ylabels)
g5.set_ylabel('')
g5.set_xlabel('Low threshold')
g5.set_yticks([])
g5.set_title('\u03C3 = {:.2f}'.format(sigmas[4]))
for ax in [g1,g2,g3,g4,g5]:
    tl = ax.get_xticklabels()
    ax.set_xticklabels(tl, rotation=90)
    tly = ax.get_yticklabels()
    ax.set_yticklabels(tly, rotation=0)
f.suptitle('Precision of parameters',fontsize = 19)    
plt.savefig("C:/Users/selin/OneDrive/Dokumente/Uni/6_SS20/Bachelorarbeit/Bilder Filter/Canny filter/Evalutate parameters/2_2_Precision.png", bbox_inches='tight')
plt.show()    


# %% Plot F1 Score:

f,(ax1,ax2,ax3,ax4,ax5, axcb) = plt.subplots(1,6, figsize=(15,5),
            gridspec_kw={'width_ratios':[1,1,1,1,1,0.08]})
plt.gcf().subplots_adjust(bottom=0.15)
xlabels = ['{:.2f}'.format(x) for x in low_thresh]
ylabels = ['{:.2f}'.format(y) for y in high_thresh]
minimum = np.min(F_score)
maximum = np.max(F_score)
# ax1.set_xticks(ax1.get_xticks()[::3])
# ax1.set_xticklabels(xlabels[::3],)
# ax1.set_yticks(ax1.get_yticks()[::3])
# ax1.set_yticklabels(ylabels[::3],)
ax1.get_shared_y_axes().join(ax2,ax3,ax4,ax5)
g1 = sns.heatmap(F_score[:,:,0],cmap="inferno",vmin = minimum,
                 vmax = maximum,cbar=False,ax=ax1,xticklabels=xlabels, yticklabels=ylabels)
g1.set_ylabel('High threshold')
g1.set_xlabel('Low threshold')
g1.set_title('\u03C3 = {:.2f}'.format(sigmas[0]))
g2 = sns.heatmap(F_score[:,:,1],cmap="inferno",vmin = minimum,
                 vmax = maximum,cbar=False,ax=ax2,xticklabels=xlabels, yticklabels=ylabels)
g2.set_ylabel('')
g2.set_xlabel('Low threshold')
g2.set_yticks([])
g2.set_title('\u03C3 = {:.2f}'.format(sigmas[1]))
g3 = sns.heatmap(F_score[:,:,2],cmap="inferno",vmin = minimum,
                 vmax = maximum,cbar=False,ax=ax3,xticklabels=xlabels, yticklabels=ylabels)
g3.set_ylabel('')
g3.set_xlabel('Low threshold')
g3.set_yticks([])
g3.set_title('\u03C3 = {:.2f}'.format(sigmas[2]))
g4 = sns.heatmap(F_score[:,:,3],cmap="inferno",vmin = minimum,
                 vmax = maximum,cbar=False,ax=ax4,xticklabels=xlabels, yticklabels=ylabels)
g4.set_ylabel('')
g4.set_xlabel('Low threshold')
g4.set_yticks([])
g4.set_title('\u03C3 = {:.2f}'.format(sigmas[3]))
g5 = sns.heatmap(F_score[:,:,4],cmap="inferno", vmin = minimum,
                 vmax = maximum,ax=ax5, cbar_ax = axcb ,xticklabels=xlabels, yticklabels=ylabels)
g5.set_ylabel('')
g5.set_xlabel('Low threshold')
g5.set_yticks([])
g5.set_title('\u03C3 = {:.2f}'.format(sigmas[4]))
for ax in [g1,g2,g3,g4,g5]:
    tl = ax.get_xticklabels()
    ax.set_xticklabels(tl, rotation=90)
    tly = ax.get_yticklabels()
    ax.set_yticklabels(tly, rotation=0)
f.suptitle('F_score of parameters',fontsize = 19)    
plt.savefig("C:/Users/selin/OneDrive/Dokumente/Uni/6_SS20/Bachelorarbeit/Bilder Filter/Canny filter/Evalutate parameters/2_2_F_score.png", bbox_inches='tight')
plt.show()    


#%% Save and load data
#Save data in desired path

np.save('D:/0_Bachelorarbeit/Saved_results/Parameters2_2/precision.npy', precision)
np.save('D:/0_Bachelorarbeit/Saved_results/Parameters2_2/recall_good_cells.npy', recall_good_cells)

np.save('D:/0_Bachelorarbeit/Saved_results/Parameters2_2/detected_cells_bad.npy', detected_cells_bad)
np.save('D:/0_Bachelorarbeit/Saved_results/Parameters2_3/Detected_cells_good.npy',detected_cells_good)
np.save('D:/0_Bachelorarbeit/Saved_results/Parameters2_3/Halo.npy',Halo)
np.save('D:/0_Bachelorarbeit/Saved_results/Parameters2_3/Not_round.npy',not_round_cell)
np.save('D:/0_Bachelorarbeit/Saved_results/Parameters2_3/Mean_time_canny.npy',Mean_time_canny)
np.save('D:/0_Bachelorarbeit/Saved_results/Parameters2_3/low_thresh.npy',low_thresh)
np.save('D:/0_Bachelorarbeit/Saved_results/Parameters2_3/high_thresh.npy',high_thresh)
np.save('D:/0_Bachelorarbeit/Saved_results/Parameters2_3/sigmas.npy',sigmas)

import numpy as np
#precision= np.load('D:/0_Bachelorarbeit/Saved_results/Parameters2_2/precision.npy')
detected_cells_bad = np.load('D:/0_Bachelorarbeit/Saved_results/Parameters2_2/detected_cells.npy')
detected_cells_good = np.load('D:/0_Bachelorarbeit/Saved_results/Parameters2_2/Detected_cells_good.npy')
Halo = np.load('D:/0_Bachelorarbeit/Saved_results/Parameters2_2/Halo.npy')
not_round_cell = np.load('D:/0_Bachelorarbeit/Saved_results/Parameters2_2/Not_round.npy')
Mean_time_canny = np.load('D:/0_Bachelorarbeit/Saved_results/Parameters2_2/Mean_time_canny.npy')
low_thresh = np.load('D:/0_Bachelorarbeit/Saved_results/Parameters2_2/low_thresh.npy')
high_thresh= np.load('D:/0_Bachelorarbeit/Saved_results/Parameters2_2/high_thresh.npy')
sigmas = np.load('D:/0_Bachelorarbeit/Saved_results/Parameters2_2/sigmas.npy')
'''