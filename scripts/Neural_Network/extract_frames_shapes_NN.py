# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 09:28:22 2020
@author: Ben Fabry, Selina Sonntag
"""
# this program reads the frames of an avi video file, averages all images,
# and stores the normalized image as a floating point numpy array 
# in the same directory as the extracted images, under the name "flatfield.npy"
#
# The program then loops again through all images of the video file,
# identifies cells, extracts the cell shape, fits an ellipse to the cell shape,
# and stores the information on the cell's centroid position, long and short axis,
# angle (orientation) of the long axis, and bounding box widht and height
# in a text file (result_file.txt) in the same directory as the video file.

# The cell detection is conducted via a canny edge detection and a neural network.

import numpy as np
from skimage.measure import label, regionprops
import os
import imageio

# Neural Network
from UNETmodel import UNet
import tensorflow as tf

#Graphs
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys 
from pathlib import Path

from skimage import feature
from scipy.ndimage import morphology
struct = morphology.generate_binary_structure(2, 1)  #structural element for binary erosion

plt.ion()

#TODO: delete without flatfield
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from scripts.helper_functions import getInputFile, getConfig, getFlatfield

#%% Preprocessing of image

# New preprocess should be applied in new training set
def preprocess(img):
    return (img - np.mean(img)) / np.std(img).astype(np.float32)


def onclick(event):
    global good_bad
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    if event.dblclick:
        good_bad = 2
    else:
        good_bad = 1
        
#%% Setup model
# shallow model (faster)
#unet = UNet().create_model((540,300,1),1, d=8)
unet = UNet().create_model((720,540,1),1, d=8)

# change path for weights
unet.load_weights(str(Path(__file__).parent / "weights/Unet_0-0-5_fl_RAdam_20200602-115253.h5"))



r_min = 6   #cells smaller than r_min (in um) will not be analyzed

video = getInputFile()

name_ex = os.path.basename(video)
filename_base, file_extension = os.path.splitext(name_ex)
output_path = os.path.dirname(video)
flatfield = output_path + r'/' + filename_base + '.npy'
configfile = output_path + r'/' + filename_base + '_config.txt'

#%%
config = getConfig(configfile)

im_av = getFlatfield(video, flatfield)
#plt.imshow(im_av)
#%% go through every frame and look for cells

frame = []
radialposition=[]
x_pos = []
y_pos = []
MajorAxis=[]
MinorAxis=[]
solidity = [] #percentage of binary pixels within convex hull polygon
irregularity = [] #ratio of circumference of the binarized image to the circumference of the ellipse 
angle=[]
sharpness=[] # computed from the radial intensity profile
is_good = []


count=0
success = 1
vidcap = imageio.get_reader(video)
for im in vidcap:
    if len(im.shape) == 3:
        im = im[:,:,0]
    
    print(count, ' ', len(frame), '  good cells')
    # flatfield correction
    im = preprocess(im.astype(float)/im_av)
    
    with tf.device('/cpu:0'):
        prediction_mask = unet.predict(im[None,:,:,None]).squeeze()>0.5
        
      # canny filter the original image
    im1o = feature.canny(im, sigma=2.5, low_threshold=0.6, high_threshold=0.99, use_quantiles=True) #edge detection           
    im2o = morphology.binary_fill_holes(im1o, structure=struct).astype(int) #fill holes
    canny = morphology.binary_erosion(im2o, structure=struct).astype(int) #erode to remove lines and small schmutz
            
    
    plt.close('all')
    fig1, ax = plt.subplots(1,3,figsize=[15,10])
    ax=np.array(ax)
    ax[0].clear()
    ax[0].imshow(im)
    ax[0].set_axis_off()
    ax[1].clear()
    ax[1].imshow(prediction_mask,cmap='gray')     
    ax[1].set_axis_off()
    
    ax[2].clear()
    ax[2].imshow(canny,cmap='gray')     
    ax[2].set_axis_off()

    labeled = label(prediction_mask)
    # iterate over all detected regions
    for region in regionprops(labeled, im): # region props are based on the original image
        a = region.major_axis_length/2
        b = region.minor_axis_length/2
        r = np.sqrt(a*b)
        
        if region.orientation > 0:
            ellipse_angle = np.pi/2 - region.orientation
        else:
            ellipse_angle = -np.pi/2 - region.orientation
        
        Amin_pixels = np.pi*(r_min/config["pixel_size"]/1e6)**2 # minimum region area based on minimum radius

        print(region.area, Amin_pixels)
        
        if region.area >= Amin_pixels: #analyze only regions larger than 100 pixels,
                                                            #and only of the canny filtered band-passed image returend an object
                                               
            # the circumference of the ellipse
            circum = np.pi*((3*(a+b))-np.sqrt(10*a*b+3*(a**2+b**2)))  
            
            #%% compute radial intensity profile around each ellipse
            theta = np.arange(0, 2*np.pi, np.pi/8)

            i_r = np.zeros(int(3*r))
            for d in range(0, int(3*r)):
                # get points on the circumference of the ellipse
                x = d/r*a*np.cos(theta)
                y = d/r*b*np.sin(theta)
                # rotate the points by the angle fo the ellipse
                t = ellipse_angle
                xrot = (x *np.cos(t) - y*np.sin(t) + region.centroid[1]).astype(int)
                yrot = (x *np.sin(t) + y*np.cos(t) + region.centroid[0]).astype(int)                    
                # crop for points inside the iamge
                index = (xrot<0)|(xrot>=im.shape[1])|(yrot<0)|(yrot>=im.shape[0])                        
                x = xrot[~index]
                y = yrot[~index]
                # average over all these points
                i_r[d] = np.mean(im[y,x])

            # define a sharpness value
            sharp = (i_r[int(r+2)]-i_r[int(r-2)])/5/np.std(i_r) 
            
            # Add ellipse to plot
            ellipse = Ellipse(xy=[region.centroid[1],region.centroid[0]], width=region.minor_axis_length, height=region.major_axis_length, angle=np.rad2deg(ellipse_angle),
                                   edgecolor='r', fc='None', lw=0.5, zorder = 2)
            ax[0].add_patch(ellipse)
            #ax[1].add_patch(ellipse)                    
            
        
            
            #%% store the cells
            yy = region.centroid[0]-config["channel_width"]/2
            yy = yy * config["pixel_size"] * 1e6
            
            radialposition.append(yy)
            y_pos.append(region.centroid[0])
            x_pos.append(region.centroid[1])
            MajorAxis.append(float(format(region.major_axis_length)) * config["pixel_size"] * 1e6)
            MinorAxis.append(float(format(region.minor_axis_length)) * config["pixel_size"] * 1e6)
            angle.append(np.rad2deg(ellipse_angle))
            irregularity.append(region.perimeter/circum)
            solidity.append(region.solidity)
            sharpness.append(sharp)
            frame.append(count)
            is_good.append(region.perimeter/circum < 1.06 and r*config["pixel_size"]*1e6 > r_min and region.solidity > 0.95)
        
    plt.show()
    good_bad = 0
    while good_bad ==0:
        cid = fig1.canvas.mpl_connect('button_press_event', onclick)
        plt.pause(0.5)
        if good_bad == 2:
            sys.exit() #exit upon double click   
        
    count = count + 1 #next image
                           
#%% store data in file
R = np.asarray(radialposition)      
X = np.asarray(x_pos)  
Y = np.asarray(y_pos)       
LongAxis = np.asarray(MajorAxis)
ShortAxis = np.asarray(MinorAxis)
Angle = np.asarray(angle)

result_file = output_path + '/' + filename_base + '_result.txt'

with open(result_file,'w') as f:
    f.write('Frame' +'\t' +'x_pos' +'\t' +'y_pos' + '\t' +'RadialPos' +'\t' +'LongAxis' +'\t' + 'ShortAxis' +'\t' +'Angle' +'\t' +'irregularity' +'\t' +'solidity' +'\t' +'sharpness' +'\n')
    f.write('Pathname' +'\t' + output_path + '\n')
    for i in range(0,len(radialposition)): 
        f.write(str(frame[i]) +'\t' +str(X[i]) +'\t' +str(Y[i]) +'\t' +str(R[i]) +'\t' +str(LongAxis[i]) +'\t'+str(ShortAxis[i]) +'\t' +str(Angle[i]) +'\t' +str(irregularity[i]) +'\t' +str(solidity[i]) +'\t' +str(sharpness[i])+'\t' +"%d"%is_good[i]  +'\n')

