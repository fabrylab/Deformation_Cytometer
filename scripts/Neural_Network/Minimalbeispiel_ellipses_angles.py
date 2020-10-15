# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:30:09 2020

@author: selin
"""

# This program tests angles from -90 to 90 for a binary mask of an ellipse
# To check how the angles are detected with regionprops.
# The blue line (orientation) fixes the angles. The orange line (angle) is
# the implementation currently used in.


import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from skimage.draw import ellipse       # generate binary mask

radialposition=[]
x_pos = []
y_pos = []
MajorAxis=[]
MinorAxis=[]
solidity = [] #percentage of binary pixels within convex hull polygon
irregularity = [] #ratio of circumference of the binarized image to the circumference of the ellipse 
angle=[]
orientation = []

'''
mask = np.zeros(([720,540]))
xs,ys = ellipse(132, 230,  58, 36, rotation=np.deg2rad(-30))
mask[ys,xs] = 1

label_imageo = label(mask)    #label all ellipses (and other objects)
fig1, ax = plt.subplots(1,1,figsize=[15,10])
plt.imshow(mask)

for region in regionprops(label_imageo, mask):
    angle.append(np.rad2deg(-region.orientation))
    orientation.append((np.rad2deg(region.orientation) + 90))
    
    print(np.rad2deg(region.orientation))
    #print((np.rad2deg(region.orientation) + 90)%90)
    ellipse = Ellipse(xy=[region.centroid[1],region.centroid[0]], width=region.minor_axis_length, height=region.major_axis_length, angle=np.rad2deg(-region.orientation),
                                       edgecolor='red', fc='None', lw=2, zorder = 2)
    ax.add_patch(ellipse)

'''
angle=[]
orientation = []
i = 0
I = []
for i in range(-90,90,1):
    mask = np.zeros(([720,540]))
    xs,ys = ellipse(132, 230,  58, 36, rotation=np.deg2rad(-i))
    mask[ys,xs] = 1

    label_imageo = label(mask)    #label all ellipses (and other objects)
    #fig1, ax = plt.subplots(1,1,figsize=[15,10])
    #plt.imshow(mask)

# iterate over all detected regions
    for region in regionprops(label_imageo, mask):
        angle.append(np.rad2deg(region.orientation))
        
        orientation.append((np.rad2deg(-region.orientation)))
        '''
        if region.orientation < 0:
            orientation.append(-(np.rad2deg(region.orientation) + 90))
        else:
            orientation.append(-(-90 + np.rad2deg(region.orientation)))
        '''
    print(i)
    I.append(i)
plt.plot(I,orientation,'o')
plt.plot(I,angle,'o')
