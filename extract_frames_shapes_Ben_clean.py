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
from skimage.transform import rescale, resize
from scipy.ndimage import morphology
from skimage.morphology import area_opening
from skimage.measure import label, regionprops
from skimage.transform import resize
import os, sys
import imageio
import json
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import cv2
import scipy
from matplotlib.patches import Ellipse 

plt.ion()

display = False

from helper_functions import getInputFile, getConfig, getFlatfield

def fill(data, start_coords, compare,fill_value):
    xsize, ysize = data.shape
    stack = set(((start_coords[0], start_coords[1]),))
    orig_value = data[start_coords[0], start_coords[1]]
    if fill_value == orig_value:
        return    
    while stack:
        x, y = stack.pop()
        if data[x, y] < compare:
            data[x, y] = fill_value
            if x > 0:
                stack.add((x - 1, y))
            if x < (xsize - 1):
                stack.add((x + 1, y))
            if y > 0:
                stack.add((x, y - 1))
            if y < (ysize - 1):
                stack.add((x, y + 1))

def detect_ridges(gray, sigma=3.0):
    hxx, hyy, hxy = feature.hessian_matrix(gray, sigma, mode='wrap')
    i1, i2 = feature.hessian_matrix_eigvals(hxx, hxy, hyy)
    return i1, i2 # i1 returns local maxima ridges and i2 returns local minima ridges

def getTimestamp(vidcap, image_index):
    if vidcap.get_meta_data(image_index)['description']:
        return json.loads(vidcap.get_meta_data(image_index)['description'])['timestamp']
    return "0"

def getRawVideo(filename):
    filename, ext = os.path.splitext(filename)
    raw_filename = Path(filename + "_raw" + ext)
    if raw_filename.exists():
        return imageio.get_reader(raw_filename)
    return imageio.get_reader(filename + ext)

def std_convoluted(image, N):
    im = np.array(image, dtype=float)
    im2 = im**2
    ones = np.ones(im.shape)
    kernel = np.ones((2*N+1, 2*N+1))
    s = scipy.signal.convolve2d(im, kernel, mode="same")
    s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
    ns = scipy.signal.convolve2d(ones, kernel, mode="same")
    return np.sqrt((s2 - s**2 / ns) / ns)

class timeit:
    def __init__(self, name):
        self.name = name
        
    def __enter__(self):
        self.start_time = time.time()
        
    def __exit__(self, *args):
        print("Timeit:", self.name, time.time()-self.start_time)  

r_min = 5   #cells smaller than r_min (in um) will not be analyzed

video = getInputFile()

name_ex = os.path.basename(video)
print(name_ex)
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

Amin_pixels = np.pi*(r_min/config["pixel_size"])**2 # minimum region area based on minimum radius

count=0
success = 1
vidcap = imageio.get_reader(video)
vidcap2 = getRawVideo(video)
#progressbar = tqdm.tqdm(vidcap)
for image_index, im in enumerate(tqdm.tqdm(vidcap)):
    if len(im.shape) == 3:
        im = im[:,:,0]

    #progressbar.set_description(f"{count} {len(frame)} good cells")
    # flatfield correction
    im = im.astype(float) / im_av
    
    if count == 0:
        h = im.shape[0]
        w = im.shape[1]
        plt.close('all')
        if display:
            fig1 = plt.figure(1,(16, 7))
            spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig1)
            ax1 = fig1.add_subplot(spec[0:1, 0:1])
            ax2 = fig1.add_subplot(spec[0:1, 1:2])#,sharex=ax1,sharey=ax1)
            ax3 = fig1.add_subplot(spec[1:2, 0:1])#,sharex=ax1,sharey=ax1)
            ax4 = fig1.add_subplot(spec[1:2, 1:2])#,sharex=ax1,sharey=ax1)
    else:
        im_high = scipy.ndimage.gaussian_laplace(im, sigma = 1) #kind of high-pass filtered image
        im_abs_high = np.abs(im_high) #for detecting potential cells
        # find cells in focus on down-sampled images 
        im_r = resize(im_abs_high, (im_abs_high.shape[0] // 10, im_abs_high.shape[1] // 10), anti_aliasing=True)
        im_rb = im_r > 0.010
        label_im_rb = label(im_rb)
        
        if display:
            ax1.clear()
            ax1.set_axis_off()
            ax1.imshow(im,cmap='gray')  
            
            ax2.clear()
            ax2.set_axis_off()
            ax2.imshow(im_high,cmap='gray')
            
            ax3.clear()
            ax4.clear()
        
        for region in regionprops(label_im_rb, im_r, coordinates='rc'): # region props are based on the downsampled abs high-pass image, row-column style (first y, then x)
            if (region.max_intensity) > 0.03 and (region.area > Amin_pixels/100):
                im_reg_b = label_im_rb == region.label
                min_row = region.bbox[0]*10-10
                min_col = region.bbox[1]*10-10
                max_row = region.bbox[2]*10+10
                max_col = region.bbox[3]*10+10
                if min_row>0 and min_col>0 and max_row<h and max_col<w: #do not analyze cells near the edge
                    im_high_reg = im_high[min_row:max_row, min_col:max_col]
                    im_sd = gaussian(std_convoluted(im_high_reg, 3), 3)
                    im_std_copy = im_sd.copy()
                    fill(im_sd, [0,0], 0.015,1)
                    fill(im_sd, [im_sd.shape[0]-1,im_sd.shape[1]-1], 0.015,1)
                    mask = im_sd == 1
                    mask = ~mask
                    mask = morphology.binary_erosion(mask, iterations=7, structure=struct).astype(int) # erode to remove lines and small dirt
                    
                    for subregion in regionprops(mask, coordinates='rc'):

                        if subregion.area > Amin_pixels:
                            x_c = subregion.centroid[1]
                            y_c = subregion.centroid[0]
                            ma = subregion.major_axis_length
                            mi = subregion.minor_axis_length
                            if subregion.orientation > 0:
                                 ellipse_angle = np.pi/2 - subregion.orientation
                            else:
                                ellipse_angle = -np.pi/2 - subregion.orientation  
                            
                            a = ma/2
                            b = mi/2
                            circum = np.pi*((3*(a+b))-np.sqrt(10*a*b+3*(a**2+b**2))) # the circumference of the ellipse
                            yy = y_c-config["channel_width_px"]/2
                            yy = yy * config["pixel_size"] * 1e6
                            radialposition.append(yy)
                            y_pos.append(y_c)
                            x_pos.append(x_c)
                            MajorAxis.append(float(format(ma)) * config["pixel_size"] * 1e6)
                            MinorAxis.append(float(format(mi)) * config["pixel_size"] * 1e6)
                            angle.append(np.rad2deg(ellipse_angle))
                            irregularity.append(subregion.perimeter/circum)
                            solidity.append(subregion.solidity)
                            sharpness.append(1) #we don't need the sharpness any longer (all cells are in focus due to high pass strategy)
                            frame.append(count)
                            timestamps.append(getTimestamp(vidcap2, image_index))

                            if display:    
                                ax3.clear()
                                ax4.clear()
                                ax3.set_axis_off()
                                ax3.imshow(im_high_reg,cmap='gray')               
                    
                                ax4.set_axis_off()
                                ax4.imshow(mask,cmap='gray')
                                           
                                ellipse = Ellipse(xy=[x_c,y_c], width=ma, height=mi, angle=np.rad2deg(ellipse_angle),
                                              edgecolor='r', fc='None', lw=0.5, zorder = 2)
                                ax3.add_patch(ellipse)
                                
                                ax1.text(min_col+int(x_c),min_row+int(y_c),'x', color = 'red', fontsize = 10, zorder=100)  
                                ax2.text(min_col+int(x_c),min_row+int(y_c),'x', color = 'red', fontsize = 10, zorder=100)  
                                s = '{:0.2f}'.format(subregion.solidity) + '\n' + '{:0.2f}'.format(subregion.perimeter/circum) 
                                ax4.text(int(mask.shape[1]/2),int(mask.shape[0]/2),s, color = 'red', fontsize = 10, zorder=100)  
                               
                                plt.show()
                                plt.pause(1)
    
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

