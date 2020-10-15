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

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import feature
from skimage.filters import gaussian
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
from matplotlib.patches import Ellipse 
import time
import copy
from tkinter import Tk
from tkinter import filedialog
import sys
import os
import configparser
import imageio

display = 2 #set to 1 if you want to see every frame of im and the radial intensity profile around each cell, 
            #set to 2 if you want to see the result of the morphological operation in the binary images im2, im3, im4
            #set to 3 if you want to see which cells have been selected (compound image of the last 100 cells that were detected)
r_min = 6   #cells smaller than r_min (in um) will not be analyzed

#%%
def onclick(event):
    global good_bad
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    if event.dblclick:
        good_bad = 2
    else:
        good_bad = 1
#%%        
class timeit:
    def __init__(self, name):
        self.name = name
        
    def __enter__(self):
        self.start_time = time.time()
        
    def __exit__(self, *args):
        print("Timeit:", self.name, time.time()-self.start_time)        
        
#%%----------general fonts for plots and figures----------
font = {'family' : 'sans-serif',
        'sans-serif':['Arial'],
        'weight' : 'normal',
        'size'   : 18}
plt.rc('font', **font)
plt.rc('legend', fontsize=12)
plt.rc('axes', titlesize=18)    
plt.rcParams['axes.linewidth'] = 0.1 #set the value globally

# if there is a command line parameter...
if len(sys.argv) >= 2:
    # ... we just use this file
    video = sys.argv[1]
# if not, we ask the user to provide us a filename
else:
    #%% select video file
    root = Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    video = []
    video = filedialog.askopenfilename(title="select the data file",filetypes=[("video file",'*.tif *.avi')]) # show an "Open" dialog box and return the path to the selected file
    if video == '':
        print('empty')
        sys.exit()


name_ex = os.path.basename(video)
filename_base, file_extension = os.path.splitext(name_ex)
output_path = os.path.dirname(video)
flatfield = output_path + r'/' + filename_base + '.npy'
configfile = output_path + r'/' + filename_base + '_config.txt'

#%% open and read the config file
config = configparser.ConfigParser()
config.read(configfile) 
magnification=float(config['MICROSCOPE']['objective'].split()[0])
coupler=float(config['MICROSCOPE']['coupler'] .split()[0])
camera_pixel_size=float(config['CAMERA']['camera pixel size'] .split()[0])

pixel_size=camera_pixel_size/(magnification*coupler) # in micrometer
pixel_size=pixel_size *1e-6 # in m

channel_width=float(config['SETUP']['channel width'].split()[0])*1e-6/pixel_size #in pixels

#%%  compute average (flatfield) image
if os.path.exists(flatfield):
    im_av = np.load(flatfield)
else:
    vidcap = imageio.get_reader(video) 
    print("compute average (flatfield) image") 
    count = 0
    for image in vidcap:
        if len(image.shape) == 3:
            image = image[:,:,0]
        # rotate counter clockwise
        #image = image.T
        image = image[::-1,::]
        if count == 0:
            im_av = copy.deepcopy(image)   
            im_av = np.asarray(im_av) 
            im_av.astype(float)
        else:
            im_av = im_av + image.astype(float) 
        count += 1 
    im_av = im_av / count
    np.save(flatfield, im_av)
#plt.imshow(im_av)

#%% go through every frame and look for cells
struct = morphology.generate_binary_structure(2, 1)  #structural element for binary erosion

if display == 3:
    fig3 = plt.figure(3,(10,10))
    plt.subplots_adjust(wspace=0, hspace=0)
    spec = gridspec.GridSpec(ncols=10, nrows=10, figure=fig3)
    ax3=[]
    for i in range(100):
        ax3.append(fig3.add_subplot(spec[i//10, i % 10]))
        ax3[i].set_axis_off()        

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
count=0
success = 1
vidcap = imageio.get_reader(video)
for im in vidcap:
    if len(im.shape) == 3:
        im = im[:,:,0]
    
    #im = im.T
    im = im[::-1,::]
        
    if count % 1 == 0:
        print(count, ' ', len(frame), '  good cells')
        im = im.astype(float)
        im = im / im_av #flatfield correction
        sizes = np.shape(im)  
        if count == 0:
            if display == 2: #debug mode
                plt.close('all')
                fig1 = plt.figure(1,(28.2 * 300 / 540, 7))
                spec = gridspec.GridSpec(ncols=21, nrows=5, figure=fig1)
                ax1 = fig1.add_subplot(spec[0:6, 0:5])
                ax2 = fig1.add_subplot(spec[0:6, 6:11],sharex=ax1,sharey=ax1)
                ax3 = fig1.add_subplot(spec[0:6, 11:16],sharex=ax1,sharey=ax1)
                ax4 = fig1.add_subplot(spec[0:6, 16:21],sharex=ax1,sharey=ax1)
            elif display == 1:
                plt.close('all')
                fig1 = plt.figure(1,(21 * sizes[1]/sizes[0], 7))
                spec = gridspec.GridSpec(ncols=30, nrows=10, figure=fig1)
                ax1 = fig1.add_subplot(spec[0:10, 0:10])
                ax2 = fig1.add_subplot(spec[1:9, 15:30])
        if count > -1: # to "jump" to a higher position
            im = np.asarray(im, dtype = 'float')
            im_mean = np.mean(im)
            '''            
            im4_high = gaussian(im, sigma=0.5) #band pass filter
            im4_low = gaussian(im, sigma=2.5)
            im4 = im4_high - im4_low            
            #with timeit("canny"):
            # canny filter on band-passed image
            im1 = feature.canny(im4, sigma=2.5, low_threshold=0.6, high_threshold=0.99, use_quantiles=True) #edge detection           
            im2 = morphology.binary_fill_holes(im1, structure=struct).astype(int) #fill holes
            im3 = morphology.binary_erosion(im2, structure=struct).astype(int) #erode to remove lines and small schmutz
            #label_image = label(im3)    #label all ellipses (and other objects)
            '''            
            # canny filter the original image
            im1o = feature.canny(im, sigma=2.5, low_threshold=0.6, high_threshold=0.99, use_quantiles=True) #edge detection           
            im2o = morphology.binary_fill_holes(im1o, structure=struct).astype(int) #fill holes
            im3o = morphology.binary_erosion(im2o, structure=struct).astype(int) #erode to remove lines and small schmutz
            label_imageo = label(im3o)    #label all ellipses (and other objects)
            
            if display == 2: #debug mode
                ax2.clear()
                ax2.set_axis_off()
                ax2.imshow(im1o,cmap='gray')
                ax3.clear()
                ax3.set_axis_off()
                ax3.imshow(im2o,cmap='gray')    
                ax4.clear()
                ax4.set_axis_off()
                ax1.clear()
                ax1.set_axis_off()
                ax1.imshow(im)            
                plt.show()
                plt.pause(1)
            elif display == 1:
                ax1.clear()
                ax1.set_axis_off()
                ax1.imshow(im)            
                plt.show()
                plt.pause(0.01)
            
            for region in regionprops(label_imageo,im, coordinates = 'rc'): #region props are based on the original image
                #imslice = region.slice
                a = region.major_axis_length/2
                b = region.minor_axis_length/2
                r = np.sqrt(a*b)
                if region.orientation > 0:
                    ellipse_angle = np.pi/2 - region.orientation
                else:
                    ellipse_angle = -np.pi/2 - region.orientation
                Amin_pixels = np.pi*(r_min/pixel_size/1e6)**2 # minimum region area based on minimum radius               
                
                if region.area >= Amin_pixels:# and np.sum(im3[imslice])>Amin_pixels: #analyze only regions larger than 100 pixels,
                                                                    #and only of the canny filtered band-passed image returend an object
                    l = region.label
                   
                    [min_row, min_col, max_row, max_col] = region.bbox
                    min_row = np.max([0, min_row - 10])
                    max_row = np.min([sizes[0], max_row + 10])
                    min_col = np.max([0, min_col - 10])
                    max_col = np.min([sizes[1], max_col + 10])   
                    
                    #structure = np.std(im[min_row:max_row, min_col:max_col])
                    
                    circum =np.pi*((3*(a+b))-np.sqrt(10*a*b+3*(a**2+b**2)))  
                    
#%% compute radial intensity profile around each ellipse                    
                    theta = np.arange(0, 2*np.pi, np.pi/16)
                    strain = (a-b)/r  
                    dd = np.arange(0,int(3*r))
                    i_r = np.zeros(int(3*r))
                    for d in range(0,int(3*r)):
                        x = d/r*a*np.cos(theta)
                        y = d/r*b*np.sin(theta)
                        t = ellipse_angle
                        xrot = (x *np.cos(t) - y*np.sin(t) + region.centroid[1]).astype(int)
                        yrot = (x *np.sin(t) + y*np.cos(t) + region.centroid[0]).astype(int)                    
                        index = (xrot<0)|(xrot>=im.shape[1])|(yrot<0)|(yrot>=im.shape[0])                        
                        x = xrot[~index]
                        y = yrot[~index]    
                        #if d == int(r):
                            #ax1.plot(x,y,'w.')
                        i_r[d] = np.mean(im[y,x])
                    d_max = np.argmax(i_r)
                    sharp = (i_r[int(r+2)]-i_r[int(r-2)])/5/np.std(i_r)     
                    
                    if display > 0 and display < 3:
                        ellipse = Ellipse(xy=[region.centroid[1],region.centroid[0]], width=region.major_axis_length, height=region.minor_axis_length, angle=np.rad2deg(ellipse_angle),
                                   edgecolor='r', fc='None', lw=0.5, zorder = 2)
                        ax1.add_patch(ellipse)                    

                    if display ==1:                                     
                        ax2.clear()
                        ax2.plot(dd,i_r)
                        ax2.plot([r,r],[np.min(i_r),np.max(i_r)],'r--')
                        ax2.plot([dd[0],dd[-1]],[im_mean,im_mean],'r--')
                        s = 'd_max at r = ' + '{:0.2f}'.format(d_max/r) + \
                            '\n' + 'stdev = ' + '{:0.3f}'.format(np.std(i_r)/im_mean) + \
                            '\n' + 'slope = ' + '{:0.2f}'.format(np.abs(i_r[int(r+2)]-i_r[int(r-2)])/5) + \
                            '\n' + 'sharp = ' + '{:0.2f}'.format(sharp)
                        plt.text(d_max, int(np.max(i_r)), s, fontsize = 12)
                        plt.show()
                        plt.pause(0.01)
                    
 #%% select the "good" cells
                    if region.perimeter/circum<1.06 and  r*pixel_size*1e6 > r_min and region.solidity > 0.95:                         
                        yy=region.centroid[0]-channel_width/2
                        yy = yy * pixel_size * 1e6                
                        radialposition.append(yy)
                        y_pos.append(region.centroid[0])
                        x_pos.append(region.centroid[1])
                        MajorAxis.append(float(format(region.major_axis_length))* pixel_size * 1e6  )
                        MinorAxis.append(float(format(region.minor_axis_length))* pixel_size * 1e6  )                        
                        angle.append(np.rad2deg(ellipse_angle))
                        irregularity.append(region.perimeter/circum)
                        solidity.append(region.solidity)
                        sharpness.append(sharp)
                        frame.append(count)
               
                        if display > 0 and display < 3:    
                            ellipse = Ellipse(xy=[region.centroid[1],region.centroid[0]], width=region.major_axis_length, height=region.minor_axis_length, angle=np.rad2deg(ellipse_angle),
                                edgecolor='white', fc='None', lw=0.5, zorder = 2)
                            ax1.add_patch(ellipse)                            
                            s = 'strain=' + '{:0.2f}'.format((a-b)/r)
                            s = s + '\n sharp =' + '{:0.3f}'.format(sharp)
                            s = s + '\n irreg =' + '{:0.3f}'.format(region.perimeter/circum)
                            s = s + '\n solid =' + '{:0.2f}'.format(region.solidity)
                            s = s + '\n angle =' + '{:0.2f}'.format(np.rad2deg(ellipse_angle))                  
                            ax1.text(int(region.centroid[1]),int(region.centroid[0]),s, color = 'red', fontsize = 10, zorder=100)                    
                            plt.show()
                            plt.pause(0.01)
                            
                            good_bad = 0
                            while good_bad ==0:
                                 cid = fig1.canvas.mpl_connect('button_press_event', onclick)
                                 plt.pause(0.5)
                                 if good_bad == 2:
                                     sys.exit() #exit upon double click   
                        if display == 3:
                            pos = (len(x_pos)-1) % 100
                            ax3[pos].cla()
                            ax3[pos].set_axis_off()
                            ax3[pos].imshow(im[min_row:max_row, min_col:max_col],cmap='gray', interpolation = 'bicubic')
                            ax3[pos].text(0,0,'{:0.2f}'.format(sharp),fontsize = 10)#0,0,'{:0.3f}'.format(structure),fontsize = 10)
                            plt.plot()
                            plt.pause(0.01)
    count = count + 1 #next image
                           
#%% store data in file
R =  np.asarray(radialposition)      
X =  np.asarray(x_pos)  
Y =  np.asarray(y_pos)       
LongAxis = np.asarray(MajorAxis)
ShortAxis = np.asarray(MinorAxis)
Angle = np.asarray(angle)
result_file = output_path + '/' + filename_base + '_result.txt'
f = open(result_file,'w')
f.write('Frame' +'\t' +'x_pos' +'\t' +'y_pos' + '\t' +'RadialPos' +'\t' +'LongAxis' +'\t' + 'ShortAxis' +'\t' +'Angle' +'\t' +'irregularity' +'\t' +'solidity' +'\t' +'sharpness' +'\n')
f.write('Pathname' +'\t' + output_path + '\n')
for i in range(0,len(radialposition)): 
    f.write(str(frame[i]) +'\t' +str(X[i]) +'\t' +str(Y[i]) +'\t' +str(R[i]) +'\t' +str(LongAxis[i]) +'\t'+str(ShortAxis[i]) +'\t' +str(Angle[i]) +'\t' +str(irregularity[i]) +'\t' +str(solidity[i]) +'\t' +str(sharpness[i]) +'\n')
f.close()
#%% data plotting

#remove bias
index = np.abs(R*Angle>0)# & (R > 50) 

LA = copy.deepcopy(LongAxis)
LA[index]=ShortAxis[index]
SA = copy.deepcopy(ShortAxis)
SA[index]=LongAxis[index]

strain = (LA - SA)/np.sqrt(LA * SA)
stress = np.abs(R)

fig2=plt.figure(2, (5, 5))
border_width = 0.2
ax_size = [0+border_width, 0+border_width, 
           1-2*border_width, 1-2*border_width]
ax1 = fig2.add_axes(ax_size)
plt.plot(stress, strain, 'o', markerfacecolor='#1f77b4', markersize=3.0,markeredgewidth=0)
plt.xlabel('distance from channel center ($\mu$m)')
plt.ylabel('strain')
plt.show()
