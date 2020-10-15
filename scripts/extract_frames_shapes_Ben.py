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


from scripts.helper_functions import getInputFile, getConfig, getFlatfield

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

Amin_pixels = np.pi*(r_min/config["pixel_size"]/1e6)**2 # minimum region area based on minimum radius

count=0
success = 1
vidcap = imageio.get_reader(video)
vidcap2 = getRawVideo(video)
#progressbar = tqdm.tqdm(vidcap)
for image_index, im in enumerate(vidcap):
    if len(im.shape) == 3:
        im = im[:,:,0]
    
    #progressbar.set_description(f"{count} {len(frame)} good cells")
    # flatfield correction
    im = im.astype(float) / im_av
    '''
    with timeit("canny"):
        #im_bandpass = gaussian(im, sigma=2)
        im_red = rescale(im, 0.1, anti_aliasing=True)
        im_1 = feature.canny(im_red, sigma=2, low_threshold=0.01, high_threshold=0.9, use_quantiles=True) #edge detection           
        im_2 = morphology.binary_fill_holes(im_1, structure=struct).astype(int)   #fill holes 
        im_3 = morphology.binary_erosion(im_2, structure=struct,iterations=1).astype(int)
        im_3 = morphology.binary_dilation(im_2, structure=struct,iterations=2).astype(int)
        im_3 = cv2.resize(im_3, dsize=(im.shape[1],im.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)        #im3 = im_3 > 
    '''
        #im_4 = area_opening(im_3, area_threshold=Amin_pixels/16, connectivity=1).astype(int)

        #im_1 = im_bandpass > np.percentile(im_bandpass, 90)       
        #im_1 = morphology.binary_fill_holes(im_1, structure=struct).astype(int)     #fill holes
        #im_2 = im_bandpass < np.percentile(im_bandpass, 10)       
        #im_2 = morphology.binary_fill_holes(im_2, structure=struct).astype(int)   #fill holes 
        #im_3 = im_2 + im_1
        #im_3 = im_3 > 0
        #im_3 = area_opening(im_3.astype(int), area_threshold=Amin_pixels, connectivity=1).astype(int)
        #im_3 = morphology.binary_dilation(im_3, structure=struct).astype(int) 
        #im_3 = morphology.binary_dilation(im_3, structure=struct,iterations=3).astype(bool)
        #im_red_4 = morphology.binary_erosion(im_red_3, structure=struct,iterations=1).astype(int)
    
                 
        #im_gauss_laplace = scipy.ndimage.gaussian_laplace(im, sigma = 1)
        #im_wiener = scipy.signal.wiener(im_gauss_laplace, mysize=3)
        #br,dr = detect_ridges(im_wiener, sigma = 1)
        #ridges = np.sqrt(br**2+dr**2)
        #im_1 = ridges > np.percentile(ridges ,98)
        #im_1 = gaussian(im_1.astype(float),sigma = 2)
        #im_1 = im_1 > 0.5
        #im_2 = morphology.binary_fill_holes(im_1, structure=struct).astype(int) #fill holes
        
        #im = im - np.mean(im)
        #im_lowpass = gaussian(im, sigma=3)
        #im_bandpass = im - im_lowpass
        
        #local_sd = std_convoluted(local_sd,2)
        #local_sd = std_convoluted(local_sd,2)
        
        #gauss_laplace = scipy.ndimage.gaussian_laplace(im, sigma = 1)
        #gauss_laplace = scipy.signal.wiener(gauss_laplace, mysize=5)
        #features = gauss_laplace > np.percentile(gauss_laplace ,99)
        #features = gaussian(features,sigma = 1)
        #features = features > 0.25
        

        #features = morphology.binary_dilation(features, structure=struct).astype(int) 
        #features = morphology.binary_erosion(features, structure=struct).astype(int) 
        #im_4 = morphology.area_closing(im_4, area_threshold=64)
        #im_2 = morphology.binary_erosion(im_2, iterations=3, structure=struct).astype(int) #erode to remove lines and small dirt
        #im_2 = morphology.binary_dilation(im_2, iterations=2, structure=struct).astype(int) #erode to remove lines and small dirt

        #gauss_grad = gaussian(scipy.ndimage.gaussian_gradient_magnitude(im, sigma = 1), sigma = 2)
        #local_sd = (std_convoluted(im,1))**1
        #im_dark_features = gauss_laplace < np.percentile(gauss_laplace ,10)
        #im_dark_features = gaussian(im_dark_features, sigma = 2)
        #im_bright_features = gauss_laplace > np.percentile(gauss_laplace ,90)
        #im_bright_features = gaussian(im_bright_features, sigma = 2)
        #br,dr = detect_ridges(im, sigma = 1)
        
        #im_3 = feature.canny(gauss_laplace, sigma=2, low_threshold=0.9, high_threshold=0.99, use_quantiles=True) #edge detection           
        #im_4 = feature.canny(im, mask = im_3, sigma=2, low_threshold=0.01, high_threshold=0.95, use_quantiles=True) #edge detection           
        #im_4 = im_4 * im_3

        #im_2 = morphology.binary_dilation(im_2, structure=struct).astype(int) #erode to remove lines and small dirt
    #label_imageo = label(im_2)    #label all ellipses (and other objects)    
    
    if count == 0:
        plt.close('all')
        fig1 = plt.figure(1,(16, 7))
        spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig1)
        ax1 = fig1.add_subplot(spec[0:1, 0:1])
        ax2 = fig1.add_subplot(spec[0:1, 1:2])#,sharex=ax1,sharey=ax1)
        ax3 = fig1.add_subplot(spec[1:2, 0:1])#,sharex=ax1,sharey=ax1)
        ax4 = fig1.add_subplot(spec[1:2, 1:2])#,sharex=ax1,sharey=ax1)
    else:
 
        #if count ==4:
            #sys.exit()
        
        '''
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
        '''


        im_high = scipy.ndimage.gaussian_laplace(im, sigma = 1)
        im_gauss_laplace = np.abs(scipy.ndimage.gaussian_laplace(im, sigma = 1))
        #im_gauss_grad = scipy.ndimage.gaussian_gradient_magnitude(im_gauss_laplace, sigma = 2)
        im_r = resize(im_gauss_laplace, (im_gauss_laplace.shape[0] // 10, im_gauss_laplace.shape[1] // 10), anti_aliasing=True)
        im_rb = im_r > 0.010
        label_im_rb = label(im_rb)
        
        
        ax1.clear()
        ax1.set_axis_off()
        ax1.imshow(im,cmap='gray')  
        
        ax2.clear()
        ax2.set_axis_off()
        ax2.imshow(im_high,cmap='gray')   
        
        ax3.clear()
        ax4.clear()
        
        for region in regionprops(label_im_rb, im_r, coordinates='rc'): # region props are based on the original image, row-column style (first y, then x)

            if (region.max_intensity) > 0.03 and (region.area > 9):                     
                im_reg_b = label_im_rb == region.label
                min_row = np.max([0,region.bbox[0]*10-10])
                min_col = np.max([0,region.bbox[1]*10-10])
                max_row = region.bbox[2]*10+10
                max_col = region.bbox[3]*10+10
                im_reg_b = resize(im_reg_b, (im_gauss_laplace.shape[0], im_gauss_laplace.shape[1]), anti_aliasing=True)
                im_reg_b =         im_reg_b[min_row:max_row, min_col:max_col]                
                im_reg   = im_gauss_laplace[min_row:max_row, min_col:max_col]
                im_reg = im_reg * im_reg_b 
                im_reg = (im_reg > np.percentile(im_reg,75))
                im_reg_b = np.ones(im_reg.shape)
                im_reg_b = im_reg_b.astype(int)
                im_high_reg = im_high[min_row:max_row, min_col:max_col]
                im_sd = std_convoluted(im_high_reg, 3)
                fill(im_sd, [0,0], 0.015,0.5)
                fill(im_sd, [im_sd.shape[0]-1,im_sd.shape[1]-1], 0.015,0.5)
                mask = im_sd == 0.5
                mask = ~mask
                mask = morphology.binary_erosion(mask, structure=struct).astype(int) # erode to remove lines and small dirt
                mask = morphology.binary_dilation(mask, iterations=2, structure=struct).astype(int) # erode to remove lines and small dirt
                mask = morphology.binary_erosion(mask, iterations=6, structure=struct).astype(int) # erode to remove lines and small dirt

                #can = feature.canny(im_high_reg, sigma=2.5, low_threshold=0.1, high_threshold=0.9, use_quantiles=True)
                
                plt.imshow(im_reg)
                for subregion in regionprops(mask, coordinates='rc'):
#                    wm = subregion.weighted_moments
#                    x_c = wm[0,1]/wm[0,0]
#                    y_c = wm[1,0]/wm[0,0]
#                    m20 = wm[0,2]/wm[0,0]-(x_c**2)
#                    m11 = wm[1,1]/wm[0,0]-(x_c*y_c)
#                    m02 = wm[2,0]/wm[0,0]-(y_c**2)
#                    
#                    ellipse_angle = 0.5*np.arctan(2*m11/(m20-m02))
#                    ma = np.sqrt(8*(m20 + m02 + np.sqrt(4*(m11**2) + (m20-m02)**2)))
#                    mi = np.sqrt(8*(m20 + m02 - np.sqrt(4*(m11**2) + (m20-m02)**2)))
#                    print(x_c, subregion.weighted_centroid[1])
#                    print(y_c, subregion.weighted_centroid[0])
#                    print(ma, subregion.major_axis_length)
#                    print(mi, subregion.minor_axis_length)
                    x_c = subregion.centroid[1]
                    y_c = subregion.centroid[0]
                    ma = subregion.major_axis_length
                    mi = subregion.minor_axis_length
#                    print('\n')
                    if subregion.orientation > 0:
                         ellipse_angle = np.pi/2 - subregion.orientation
                    else:
                        ellipse_angle = -np.pi/2 - subregion.orientation  
                    
                ax3.clear()
                ax4.clear()
                ax3.set_axis_off()
                ax3.imshow(im_high_reg,cmap='gray')   
                
#                plt.figure(10)
#                xsd = np.std(im_high_reg,axis = 0)
#                ysd = np.std(im_high_reg,axis = 1)
#                plt.plot(xsd)
#                plt.plot(ysd)
#                sys.exit()
    
                ax4.set_axis_off()
                ax4.imshow(mask,cmap='gray')   
                           
                ellipse = Ellipse(xy=[x_c,y_c], width=ma, height=mi, angle=np.rad2deg(ellipse_angle),
                              edgecolor='r', fc='None', lw=0.5, zorder = 2)
                ax3.add_patch(ellipse)
                
                ax1.text(min_col+int(x_c),min_row+int(y_c),'x', color = 'red', fontsize = 10, zorder=100)  
                ax2.text(min_col+int(x_c),min_row+int(y_c),'x', color = 'red', fontsize = 10, zorder=100)  
                
                plt.show()
                plt.pause(0.001)  

    '''
    # iterate over all detected regions
    for region in regionprops(label_imageo, im, coordinates='rc'): # region props are based on the original image, row-column style (first y, then x)
        a = region.major_axis_length/2
        b = region.minor_axis_length/2
        r = np.sqrt(a*b)
        if region.orientation > 0:
            ellipse_angle = np.pi/2 - region.orientation
        else:
            ellipse_angle = -np.pi/2 - region.orientation        
        
        
        if region.area >= Amin_pixels:# and np.sum(im3[region.slice])>Amin_pixels: #analyze only regions larger than 100 pixels,
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
            timestamps.append(getTimestamp(vidcap2, image_index))
            if region.solidity > 0.01 and region.perimeter/circum < 10:
                ellipse = Ellipse(xy=[region.centroid[1],region.centroid[0]], width=region.major_axis_length, height=region.minor_axis_length, angle=np.rad2deg(ellipse_angle),
                                  edgecolor='r', fc='None', lw=0.5, zorder = 2)
                ax1.add_patch(ellipse)
                s = '{:0.2f}'.format(region.solidity) + '\n' + '{:0.2f}'.format(region.perimeter/circum) 
                ax2.text(int(region.centroid[1]),int(region.centroid[0]),s, color = 'red', fontsize = 10, zorder=100)  
                ax3.text(int(region.centroid[1]),int(region.centroid[0]),s, color = 'red', fontsize = 10, zorder=100)  
                ax4.text(int(region.centroid[1]),int(region.centroid[0]),s, color = 'red', fontsize = 10, zorder=100)  
                plt.show()
                
               
    plt.pause(1)
    '''
    count = count + 1 #next image
'''                               
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
'''
