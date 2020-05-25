# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 09:28:22 2020
@author: Ben Fabry
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

import numpy as np
from skimage import feature
from skimage.filters import gaussian
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
import os
import imageio
import json
from pathlib import Path

from helper_functions import getInputFile, getConfig, getFlatfield

def getTimestamp(vidcap, image_index):
    if vidcap.get_meta_data(image_index)['description']:
        return json.loads(vidcap.get_meta_data(image_index)['description'])['timestamp']
    return "0"

def getRawVideo(filename):
    filename, ext = os.path.splitext(filename)
    raw_filename = Path(filename + "_raw" + ext)
    if raw_filename.exists():
        return imageio.get_reader(raw_filename)
    return imageio.get_reader(filename)

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
struct = morphology.generate_binary_structure(2, 1)  #structural element for binary erosion

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
timestamps = []

count=0
success = 1
vidcap = imageio.get_reader(video)
vidcap2 = getRawVideo(video)
for image_index, im in enumerate(vidcap):
    if image_index == 10:
        break
    if len(im.shape) == 3:
        im = im[:,:,0]
    
    print(count, ' ', len(frame), '  good cells')
    # flatfield correction
    im = im.astype(float) / im_av
    
    # band pass filter (highpass - lowpass)
    im_bandpass = gaussian(im, sigma=0.5) - gaussian(im, sigma=2.5)

    # canny filter on band-passed image
    im1 = feature.canny(im_bandpass, sigma=2.5, low_threshold=0.6, high_threshold=0.99, use_quantiles=True) #edge detection           
    im2 = morphology.binary_fill_holes(im1, structure=struct).astype(int) # fill holes
    im3 = morphology.binary_erosion(im2, structure=struct).astype(int) # erode to remove lines and small dirt
    
    # canny filter the original image
    im1o = feature.canny(im, sigma=2.5, low_threshold=0.6, high_threshold=0.99, use_quantiles=True) #edge detection           
    im2o = morphology.binary_fill_holes(im1o, structure=struct).astype(int) #fill holes
    im3o = morphology.binary_erosion(im2o, structure=struct).astype(int) #erode to remove lines and small dirt
    label_imageo = label(im3o)    #label all ellipses (and other objects)
    
    # iterate over all detected regions
    for region in regionprops(label_imageo, im): # region props are based on the original image
        a = region.major_axis_length/2
        b = region.minor_axis_length/2
        r = np.sqrt(a*b)
        
        Amin_pixels = np.pi*(r_min/config["pixel_size"]/1e6)**2 # minimum region area based on minimum radius

        print(region.area, Amin_pixels, np.sum(im3[region.slice]), Amin_pixels)
        
        if region.area >= Amin_pixels and np.sum(im3[region.slice])>Amin_pixels: #analyze only regions larger than 100 pixels,
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
                t = -region.orientation
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
            
            
            #%% store the cells
            yy = region.centroid[0]-config["channel_width"]/2
            yy = yy * config["pixel_size"] * 1e6
            
            radialposition.append(yy)
            y_pos.append(region.centroid[0])
            x_pos.append(region.centroid[1])
            MajorAxis.append(float(format(region.major_axis_length)) * config["pixel_size"] * 1e6)
            MinorAxis.append(float(format(region.minor_axis_length)) * config["pixel_size"] * 1e6)
            angle.append(np.rad2deg(-region.orientation))
            irregularity.append(region.perimeter/circum)
            solidity.append(region.solidity)
            sharpness.append(sharp)
            frame.append(count)
            timestamps.append(getTimestamp(vidcap2, image_index))
               
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
    f.write('Frame' +'\t' +'x_pos' +'\t' +'y_pos' + '\t' +'RadialPos' +'\t' +'LongAxis' +'\t' + 'ShortAxis' +'\t' +'Angle' +'\t' +'irregularity' +'\t' +'solidity' +'\t' +'sharpness' + '\t' + 'timestamp' + '\n')
    f.write('Pathname' +'\t' + output_path + '\n')
    for i in range(0,len(radialposition)): 
        f.write(str(frame[i]) +'\t' +str(X[i]) +'\t' +str(Y[i]) +'\t' +str(R[i]) +'\t' +str(LongAxis[i]) +'\t'+str(ShortAxis[i]) +'\t' +str(Angle[i]) +'\t' +str(irregularity[i]) +'\t' +str(solidity[i]) +'\t' +str(sharpness[i])+'\t' + timestamps[i] +'\n')

