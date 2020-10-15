# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:22:34 2020
@author: Elham Mirzahossein
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

# this program analysis images, finds cellextracts the cell shape, 
# fits an ellipse to the shape, and stores the position and dimensions of the ellipse

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
import copy
import time

display = 2 #saet to 1 if you want to see every frame of im, set to 2 if you want to see im2, im3, im4

# initialization of user interface
root = Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
plt.ion()

class timeit:
    def __init__(self, name):
        self.name = name
        
    def __enter__(self):
        self.start_time = time.time()
        
    def __exit__(self, *args):
        print("Timeit:", self.name, time.time()-self.start_time)

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
        
#----------general fonts for plots and figures----------
font = {'family' : 'sans-serif',
        'sans-serif':['Arial'],
        'weight' : 'normal',
        'size'   : 18}
plt.rc('font', **font)
plt.rc('legend', fontsize=12)
plt.rc('axes', titlesize=18)        

#%%----------read data from a csv (text) file-----------------------------------
imfile = filedialog.askopenfilename(title="select the image file",filetypes=[("jpg",'*.jpg')]) # show an "Open" dialog box and return the path to the selected file
if imfile == '':
    print('empty')
    sys.exit()
    
struct = morphology.generate_binary_structure(2, 1)  #structural element for binary erosion
asp=[]
impath = os.path.dirname(os.path.abspath(imfile))
flatfield_file = impath + r'\flatfield.npy'
os.chdir(impath)

flatfield = np.load(flatfield_file)
i=0
pixel_size = 0.36e-6 # in m for 20x AlliedVision 
channel_width = 200e-6/pixel_size #in pixels

frame = []
radialposition=[]
x_pos = []
y_pos = []
MajorAxis=[]
MinorAxis=[]
angle=[]
last_time = time.time()        
for file in glob.glob("*.jpg"):
     i=i+1
     if i % 1 == 0:
        print(file, ' ', len(frame), '  good cells', last_time-time.time())   
        last_time = time.time()        
        im = imageio.imread(file)
        im = im[:,:,0]
        im = im.astype(float)
        im = im / flatfield #flatfield correction
        sizes = np.shape(im)  
        if i == 1:
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
        if i > 0: # to "jump" to a higher position
            im = np.asarray(im, dtype = 'float')
            im_mean = np.mean(im)
            #with timeit("canny"):
            #im1 = feature.canny(im, sigma=2.5, low_threshold=0.7, high_threshold=0.99, use_quantiles=True) #edge detection
            im1 = feature.canny(im, sigma=2.5, low_threshold=8, high_threshold=10, use_quantiles=False) #edge detection
           
            im2 = morphology.binary_fill_holes(im1, structure=struct).astype(int) #fill holes
                #im2 = morphology.binary_fill_holes(im1).astype(int) #fill holes
            im3 = morphology.binary_erosion(im2, structure=struct).astype(int) #erode to remove lines and small schmutz
#            Axx, Axy, Ayy = feature.structure_tensor(im, sigma=0.5, mode = 'nearest') #structure
#            im4 = feature.structure_tensor_eigvals(Axx, Axy, Ayy)[0]   #structure     
#            im4 = im4 * morphology.binary_erosion(im3, structure=struct, iterations = 4).astype(int) #structure
            
            label_image = label(im3)    #label all ellipses (and other objects)
        
            
            if display == 2: #debug mode
                ax2.clear()
                ax2.set_axis_off()
                ax2.imshow(im1,cmap='gray')
                ax3.clear()
                ax3.set_axis_off()
                ax3.imshow(im2,cmap='gray')    
#                ax4.clear()
#                ax4.set_axis_off()
#                ax4.imshow(im4,cmap='jet', vmin = 0, vmax = 3000 )                       
                ax1.clear()
                ax1.set_axis_off()
                ax1.imshow(im)            
                plt.show()
                plt.pause(0.1)
            elif display == 1:
                ax1.clear()
                ax1.set_axis_off()
                ax1.imshow(im)            
                plt.show()
                plt.pause(0.01)
            
            for region in regionprops(label_image,im):
                if region.area >= 100 : #analyze only large, dark ellipses
                    l = region.label
#                    structure = np.std(im4[label_image==l])
                    a = region.major_axis_length/2
                    b = region.minor_axis_length/2
                    r = np.sqrt(a*b)
                    circum =np.pi*((3*(a+b))-np.sqrt(10*a*b+3*(a**2+b**2)))                   
                    if display > 0:
                        ellipse = Ellipse(xy=[region.centroid[1],region.centroid[0]], width=region.major_axis_length, height=region.minor_axis_length, angle=np.rad2deg(-region.orientation),
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
                    
                    if display ==1:
                        ax2.clear()
                        ax2.plot(dd,i_r)
                        ax2.plot([r,r],[np.min(i_r),np.max(i_r)],'r--')
                        ax2.plot([dd[0],dd[-1]],[im_mean,im_mean],'r--')
                        s = 'd_max at r = ' + '{:0.2f}'.format(d_max/r) + '\n' + 'stdev = ' + '{:0.2f}'.format(np.std(i_r))
                        plt.text(d_max, int(np.max(i_r)), s, fontsize = 12)
                        plt.show()
                        plt.pause(0.01)
                    
 #%% select the "good" cells
#                    if np.sqrt(structure)/im_mean > 0.0009 and np.sqrt(structure)/im_mean <10.26 and region.mean_intensity/im_mean <10.1 \
#                            and region.mean_intensity/im_mean > 0.09 and region.perimeter/circum<10.06 and region.area>500 and np.std(i_r)/im_mean<10.08 and \
#                            ((d_max/r>1 and d_max/r<1.5) or (d_max/r>1.5 and np.std(i_r)<3)  or  (d_max/r>0.5 and np.std(i_r)<3)):
                    if region.perimeter/circum<1.06 and region.area>500 and \
                            ((d_max/r>1 and d_max/r<1.5) or (d_max/r>1.5 and np.std(i_r)/im_mean<0.03)  or  (d_max/r>0.5 and np.std(i_r)/im_mean<0.03)):                                
                        yy=region.centroid[0]-channel_width/2
                        yy = yy * pixel_size * 1e6
                        radialposition.append(yy)
                        y_pos.append(region.centroid[0])
                        x_pos.append(region.centroid[1])
                        MajorAxis.append(float(format(region.major_axis_length)))
                        MinorAxis.append(float(format(region.minor_axis_length)))
                        angle.append(np.rad2deg(-region.orientation))
                        frame.append(file)

                    
                        if display > 0:    
                            ellipse = Ellipse(xy=[region.centroid[1],region.centroid[0]], width=region.major_axis_length, height=region.minor_axis_length, angle=np.rad2deg(-region.orientation),
                                edgecolor='white', fc='None', lw=0.5, zorder = 2)
                            ax1.add_patch(ellipse)                            
                            s = 'strain=' + '{:0.2f}'.format((a-b)/r)
#                            s = s + '\n struc =' + '{:0.3f}'.format(np.sqrt(structure)/im_mean)
                            s = s + '\n irreg =' + '{:0.3f}'.format(region.perimeter/circum)
                            s = s + '\n int =' + '{:0.2f}'.format(region.mean_intensity/im_mean)
                            s = s + '\n angle =' + '{:0.2f}'.format(np.rad2deg(-region.orientation))                  
                            ax1.text(int(region.centroid[1]),int(region.centroid[0]),s, color = 'red', fontsize = 10, zorder=100)                    
                            plt.show()
                            plt.pause(0.01)
                                               
                            good_bad = 0
                            while good_bad ==0:
                                 cid = fig1.canvas.mpl_connect('button_press_event', onclick)
                                 plt.pause(0.5)
                                 if good_bad == 2:
                                     sys.exit() #exit upon double click             
                           
#%% store data in file
R =  np.asarray(radialposition)      
X =  np.asarray(x_pos)  
Y =  np.asarray(y_pos)       
LongAxis = np.asarray(MajorAxis)
ShortAxis = np.asarray(MinorAxis)
Angle = np.asarray(angle)
f = open('results.txt','w')
f.write('Frame' + '\t' + 'x_pos' + '\t' +'y_pos' + '\t' + 'RadialPos' +'\t' +'LongAxis' +'\t' + 'ShortAxis' +'\t' + 'Angle' +'\n')
for i in range(0,len(radialposition)): 
    f.write(frame[i] + '\t' + str(X[i]) + '\t' + str(Y[i]) + '\t' + str(R[i]) +'\t' + str(LongAxis[i]) +'\t'+str(ShortAxis[i]) + '\t' +str(Angle[i]) +'\n')
f.close()
#%% data plotting

#remove bias
index = (R*Angle>0)
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
plt.plot(stress, strain, 'o', markerfacecolor='#1f77b4', markersize=6.0,markeredgewidth=0)
plt.xlabel('channel position ($\mu$m)')
plt.ylabel('strain')
plt.show()



