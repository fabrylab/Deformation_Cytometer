# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:22:34 2020

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:03:56 2020

@author: user
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

#this program reads data from a text or csv file and plots the data

import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog
import imageio #offers better image read options
import sys
import glob, os
from skimage import feature
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
from matplotlib.patches import Ellipse #downloaded from Github
import matplotlib.gridspec as gridspec

display = 1 #saet to 1 if you want to see every frame
#import matplotlib.pyplot as plt

# initialization of user interface
root = Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
plt.ion()


def onclick(event):
    global good_bad
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    if event.dblclick:
        good_bad = 2
    else:
        good_bad = 1

#%%----------read data from a csv (text) file-----------------------------------
imfile = filedialog.askopenfilename(title="select the image file",filetypes=[("jpg",'*.jpg')]) # show an "Open" dialog box and return the path to the selected file
if imfile == '':
    print('empty')
    sys.exit()
    
struct = morphology.generate_binary_structure(2, 1)  #structural element for binary erosion
asp=[]
impath = os.path.dirname(os.path.abspath(imfile))
os.chdir(impath)
i=0
pixel_size = 0.36e-6 # in m
pixel_numb=540
if display == 1:
    plt.close('all')
    fig1 = plt.figure(1,(28.2 * 300 / 540, 7))
    spec = gridspec.GridSpec(ncols=21, nrows=5, figure=fig1)
    ax1 = fig1.add_subplot(spec[0:6, 0:5])
    ax2 = fig1.add_subplot(spec[0:6, 6:11],sharex=ax1,sharey=ax1)
    ax3 = fig1.add_subplot(spec[0:6, 11:16],sharex=ax1,sharey=ax1)
    ax4 = fig1.add_subplot(spec[0:6, 16:21],sharex=ax1,sharey=ax1)

radialposition=[]
L=[]
B=[]
angle=[]
for file in glob.glob("*.jpg"):
     i=i+1
     if i % 1 == 0:
        print(file)   
        im = imageio.imread(file)
        im = im[:,:,0]
        im = np.asarray(im, dtype = 'float')
        im_mean = np.mean(im)
        im1 = feature.canny(im, sigma=2.5, low_threshold=0.7, high_threshold=0.99, use_quantiles=True) #edge detection
        Axx, Axy, Ayy = feature.structure_tensor(im, sigma=0.5, mode = 'nearest') #structure
        im1a = feature.structure_tensor_eigvals(Axx, Axy, Ayy)[0]
        im2 = morphology.binary_fill_holes(im1, structure=struct).astype(int) #fill holes
        im3 = morphology.binary_erosion(im2, structure=struct).astype(int) #erode to remove lines and small schmutz
        im1a = im1a * morphology.binary_erosion(im3, structure=struct, iterations = 4).astype(int)
        
        label_image = label(im3)    #label all ellipses (and other objects)
    
        sizes = np.shape(im)  
        
        if display == 1:
            ax2.clear()
            ax2.set_axis_off()
            ax2.imshow(im1,cmap='gray')
            ax3.clear()
            ax3.set_axis_off()
            ax3.imshow(im2,cmap='gray')    
            ax4.clear()
            ax4.set_axis_off()
            ax4.imshow(im1a,cmap='jet', vmin = 0, vmax = 3000 )           
            
            ax1.clear()
            ax1.set_axis_off()
            ax1.imshow(im)
            
            plt.show()
            plt.pause(0.1)
        
        for region in regionprops(label_image,im):
            # take regions with large enough areas
            if region.area >= 100 : #analyze only large, dark ellipses
                l = region.label
                structure = np.std(im1a[label_image==l])
                a = region.major_axis_length/2
                b = region.minor_axis_length/2
                r = np.sqrt(a*b)
                circum =np.pi*((3*(a+b))-np.sqrt(10*a*b+3*(a**2+b**2)))                   
                ellipse = Ellipse(xy=[region.centroid[1],region.centroid[0]], width=region.minor_axis_length, height=region.major_axis_length, angle=np.rad2deg(-region.orientation),
                           edgecolor='r', fc='None', lw=0.5, zorder = 2)
    #                yy=region.centroid[0]-pixel_numb/2
    #                yy = yy * pixel_size * 1e6
                ax1.add_patch(ellipse)
    
                if np.sqrt(structure)/im_mean > 0.0009 and np.sqrt(structure)/im_mean <10.26 and region.mean_intensity/im_mean <0.99 \
                        and region.mean_intensity/im_mean > 0.8 and region.perimeter/circum<1.06 and region.area>500:
                    yy=region.centroid[0]-pixel_numb/2
                    yy = yy * pixel_size * 1e6
                    radialposition.append(yy)
                    L.append(float(format(region.major_axis_length)))
                    B.append(float(format(region.minor_axis_length)))
                    angle.append(np.rad2deg(-region.orientation))
    #            plt.pause(0.1)
                    ellipse = Ellipse(xy=[region.centroid[1],region.centroid[0]], width=region.minor_axis_length, height=region.major_axis_length, angle=np.rad2deg(-region.orientation),
                        edgecolor='white', fc='None', lw=1, zorder = 2)
                    ax1.add_patch(ellipse)
                
                if display == 1:      
                    s = 'strain=' + '{:0.2f}'.format((a-b)/r)
                    s = s + '\n struc =' + '{:0.3f}'.format(np.sqrt(structure)/im_mean)
                    s = s + '\n irreg =' + '{:0.3f}'.format(region.perimeter/circum)
                    s = s + '\n int =' + '{:0.2f}'.format(region.mean_intensity/im_mean)
                    s = s + '\n area =' + '{:0.2f}'.format(region.area)                  
                    ax1.text(int(region.centroid[1]),int(region.centroid[0]),s, color = 'red')                    
                    plt.show()
                    plt.pause(0.1)

        #input("Press Enter to continue...")
        
                
#%% store data in file
R =  np.asarray(radialposition)               
LongAxis = np.asarray(L)
ShortAxis = np.asarray(B)
Angle = np.asarray(angle)
f = open('results.txt','w')
f.write('RadialPos' +'\t' +'LongAxis' +'\t' + 'ShortAxis' +'\t' + 'Angle' +'\n')
for i in range(0,len(radialposition)): 
    f.write(str(R[i]) +'\t' + str(LongAxis[i]) +'\t'+str(ShortAxis[i]) + '\t' +str(Angle[i]) +'\n')
f.close()




