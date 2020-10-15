# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:00:24 2020

@author: user
"""
# in this program we crop cells and store it in a folder

# this program reads the frames of an avi video file to individual jpg images
# it also averages all images and stores the normalized image as a floating point numpy array 
# in the same directory as the extracted images, under the name "flatfield.npy"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import feature
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
from matplotlib.patches import Ellipse 
import time
import copy
from tkinter import Tk
from tkinter import filedialog
import sys
import os

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

#%% select video file
root = Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
video = []
video = filedialog.askopenfilename(title="select the data file",filetypes=[("avi file",'*.avi')]) # show an "Open" dialog box and return the path to the selected file
if video == '':
    print('empty')
    sys.exit()

name_ex = os.path.basename(video)
filename_base, file_extension = os.path.splitext(name_ex)
output_path = os.path.dirname(video)
flatfield = output_path + r'/flatfield'
#%%  
vidcap = cv2.VideoCapture(video) 
print("compute average (flatfield) image") 
count = 0
while 1:
    success,image = vidcap.read()
    if success !=1:
        break
    image = image[:,:,0]
    # rotate counter clockwise
    image=cv2.transpose(image)
    image=cv2.flip(image,flipCode=0)         
    if count == 0:
        im_av = copy.deepcopy(image)   
        im_av = np.asarray(im_av) 
        im_av.astype(float)
    else:
        im_av = im_av + image.astype(float) 
    count += 1 
im_av = im_av / np.mean(im_av)
np.save(flatfield, im_av)
plt.imshow(im_av)
#%%
struct = morphology.generate_binary_structure(2, 1)  #structural element for binary erosion

display = 1 #saet to 1 if you want to see every frame of im, set to 2 if you want to see im2, im3, im4
pixel_size = 0.367e-6 # in m for 20x AlliedVision 
channel_width = 200e-6/pixel_size #in pixels

frame = []
radialposition=[]
x_pos = []
y_pos = []
MajorAxis=[]
MinorAxis=[]
angle=[]
count=0
success = 1
vidcap = cv2.VideoCapture(video)
numb=0
while success:
    success,im = vidcap.read()
    im=cv2.transpose(im)
    im=cv2.flip(im,flipCode=0)  
    if success !=1:
        break # break out of the while loop    
    if count % 1 == 0:
        print(count, ' ', len(frame), '  good cells')                
        im = im[:,:,0]
        im = im.astype(float)
        im = im / im_av #flatfield correction
        sizes = np.shape(im)  
        if count == 0:
            plt.close('all')
            fig1 = plt.figure(1,(21 * sizes[1]/sizes[0], 7))        
            spec = gridspec.GridSpec(ncols=30, nrows=10, figure=fig1)
            ax1 = fig1.add_subplot(spec[0:10, 0:10])
            ax2 = fig1.add_subplot(spec[1:9, 15:30])
        if count > 0: # to "jump" to a higher position
            im = np.asarray(im, dtype = 'float')
            im_mean = np.mean(im)
            with timeit("canny"):
                im1 = feature.canny(im, sigma=2.5, low_threshold=0.7, high_threshold=0.99, use_quantiles=True) #edge detection
            im2 = morphology.binary_fill_holes(im1, structure=struct).astype(int) #fill holes
            im3 = morphology.binary_erosion(im2, structure=struct).astype(int) #erode to remove lines and small schmutz
            Axx, Axy, Ayy = feature.structure_tensor(im, sigma=0.5, mode = 'nearest') #structure
            im4 = feature.structure_tensor_eigvals(Axx, Axy, Ayy)[0]   #structure 
            im4 = im4 * morphology.binary_erosion(im3, structure=struct, iterations = 4).astype(int) #structure
            label_image = label(im3)    #label all ellipses (and other objects)
            ax1.clear()
            ax1.set_axis_off()
            ax1.imshow(im)            
            plt.show()
            plt.pause(0.01)
            
            for region in regionprops(label_image,im):
                if region.area >= 100 : #analyze only large, dark ellipses
                    l = region.label
                    #print(l)
                    structure = np.std(im4[label_image==l])
                    a = region.major_axis_length/2
                    b = region.minor_axis_length/2
                    r = np.sqrt(a*b)
                    circum =np.pi*((3*(a+b))-np.sqrt(10*a*b+3*(a**2+b**2)))                   
                    if display > 0:
                        ellipse = Ellipse(xy=[region.centroid[1],region.centroid[0]], width=region.minor_axis_length, height=region.major_axis_length, angle=np.rad2deg(-region.orientation),
                                   edgecolor='r', fc='None', lw=0.5, zorder = 2)
            #                yy=region.centroid[0]-channel_width/2
            #                yy = yy * pixel_size * 1e6
                        ax1.add_patch(ellipse)
                    
#%% compute radial inetensity profile for each ellipse
                    theta = np.arange(0, 2*np.pi, np.pi/4)
                    strain = (a-b)/r
                    dd = np.arange(0,int(3*r))
                    i_r = np.zeros(int(3*r))
                    for d in range(0,int(3*r)):
                        x = np.round(d*(np.cos(theta) + strain*np.sin(theta)) + region.centroid[1]).astype(int)
                        y = np.round(d*np.sin(theta) + region.centroid[0]).astype(int)                
                        index = (x<0)|(x>=im.shape[1])|(y<0)|(y>=im.shape[0])
                        x = x[~index]
                        y = y[~index]    
                        i_r[d] = np.mean(im[y,x])
                    d_max = np.argmax(i_r)
                               
 #%% select the "good" cells
                    
                    #if np.sqrt(structure)/im_mean > 0.0009 and np.sqrt(structure)/im_mean <10.26 and region.mean_intensity/im_mean <10.1 \
                    #        and region.mean_intensity/im_mean > 0.09 and region.perimeter/circum<10.06 and region.area>500 and np.std(i_r)/im_mean<10.08 and \
                    #        ((d_max/r>1 and d_max/r<1.5) or (d_max/r>1.5 and np.std(i_r)<3)  or  (d_max/r>0.5 and np.std(i_r)<3)):
                    if region.perimeter/circum<1.06 and region.area>500 and \
                            ((d_max/r>1 and d_max/r<1.4) or (d_max/r>1.4 and np.std(i_r)/im_mean<0.03)  or  (d_max/r>0.5 and np.std(i_r)/im_mean<0.03)):                                
                        y_pos.append(region.centroid[0])
                        x_pos.append(region.centroid[1])
                        MajorAxis.append(float(format(region.major_axis_length)))
                        MinorAxis.append(float(format(region.minor_axis_length)))
                        angle.append(np.rad2deg(-region.orientation))
                        
                        #crop cells and save it
                        W=50 # width of crop
                        H=40 #Height of crop
                        a1=region.centroid[0]-H
                        a2=region.centroid[0]+H
                        b1=region.centroid[1]-W
                        b2=region.centroid[1]+W
                        print(region.centroid[0],region.centroid[1])
                        if a1>0 and a2< im.shape[0] and b1>0 and b2<im.shape[1]:
                            crop=im[int(round(region.centroid[0])-H):int(round(region.centroid[0])+H),int(round(region.centroid[1])-W):int(round(region.centroid[1])+W)]
                            # place that you want to store the images
                            cv2.imwrite(r"D:\0_Bachelorarbeit\Cropped images\ " + str(int(region.centroid[0]))+'_'+ str(count) + ".jpg", crop)
                            ax2.imshow(crop)
                            yy=region.centroid[0]-channel_width/2
                            yy = yy * pixel_size * 1e6
                            radialposition.append(yy)
                            frame.append(count)
                            plt.pause(0.01)
                            R=int(yy)
                            
                    elif region.area>500:
                        W=50 # width of crop
                        H=40 #Height of crop
                        a1=region.centroid[0]-H
                        a2=region.centroid[0]+H
                        b1=region.centroid[1]-W
                        b2=region.centroid[1]+W
                        if a1>0 and a2< im.shape[0] and b1>0 and b2<im.shape[1]:
                            crop=im[int(round(region.centroid[0])-H):int(round(region.centroid[0])+H),int(round(region.centroid[1])-W):int(round(region.centroid[1])+W)]
                            # place that you want to store the images
                            cv2.imwrite(r"D:\0_Bachelorarbeit\Cropped bad images\ " + str(int(region.centroid[0]))+'_'+ str(count) + ".jpg", crop)
                            ax2.imshow(crop)
#                            
    count = count + 1 #next image
                           
#%% data plotting

f = open('crop_information.txt','w')
for i in range(0,len(radialposition)): 
    f.write(str(frame[i])+'\t'+str(radialposition[i])+'\n')
f.close()


