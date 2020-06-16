# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:13:06 2020

@author: Selina Sonntag
"""

import clickpoints
import numpy as np

from helper_functions import getConfig

import os
import sys
from tkinter import Tk
from tkinter import filedialog

from matplotlib.path import Path

from skimage.measure import label
from skimage.measure import regionprops

def fit_ellipses_regionprops(p):
    labeled = label(p)
    out = []
    for region in regionprops(labeled,p):  
        if region.area >= 100: #analyze only regions larger than 100 pixels
        #if region.area >= Amin_pixels:
            fit = ((region.centroid[0],region.centroid[1]),(region.minor_axis_length,region.major_axis_length),90 - np.rad2deg(-region.orientation))
            out.append(fit)
    return out

def getInputFile():
    # if there is a command line parameter...
    if len(sys.argv) >= 2:
        # ... we just use this file
        video = sys.argv[1]
    # if not, we ask the user to provide us a filename
    else:
        # select video file
        root = Tk()
        root.withdraw() # we don't want a full GUI, so keep the root window from appearing
        video = []
        video = filedialog.askopenfilename(title="select the data file",filetypes=[("video file",'*.tif','*avi')]) # show an "Open" dialog box and return the path to the selected file
        if video == '':
            print('empty')
            sys.exit()
    return video

file = getInputFile()

name_ex = os.path.basename(file)
filename_base, file_extension = os.path.splitext(name_ex)
output_path = os.path.dirname(file)
flatfield = output_path + r'/' + filename_base + '.npy'
configfile = output_path + r'/' + filename_base + '_config.txt'
config = getConfig(configfile)


cdb_file = configfile = output_path + r'/' + 'gt_0_selina.cdb'
cdb = clickpoints.DataFile(cdb_file)

q_elli = cdb.getPolygons()
img_ids = np.unique([el.image.id for el in q_elli])

frame = []
radialposition = []
x_pos = []
y_pos = []
MajorAxis = []
MinorAxis = []
angle = []
count = 0
for id in img_ids:
    
    img_o = cdb.getImage(id=id)
    img   = img_o.get_data()
    if len(img.shape) == 3:
        img = img[:,:,0]
    nx, ny = 540,720
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T

    mask = np.zeros((img.shape[0:2]), dtype=np.uint8)
    q_polys=cdb.getPolygons(image=img_o)
    for pol in q_polys:
        if np.shape(pol)[0] != 0:
            polygon = np.array([pol.points])
            path = Path(polygon.squeeze())
            grid = path.contains_points(points)
            grid = grid.reshape((ny,nx))
            mask += grid

    ellipses_o = np.array([[y,x,b,a,-phi] for (x,y), (a,b), phi in fit_ellipses_regionprops(mask)])
    for i in range(np.shape(ellipses_o)[0]):
            if ellipses_o[i,2] < ellipses_o[i,3]:
                x = ellipses_o[i,2]
                ellipses_o[i,2] = ellipses_o[i,3]
                ellipses_o[i,3] = x
                ellipses_o[i,4] = ellipses_o[i,4]-90
    for n in range(ellipses_o.shape[0]):
        frame.append(id-1)
        count += 1
        yy = ellipses_o[n,1]-config["channel_width"]/2
        yy = yy * config["pixel_size"] * 1e6
        radialposition.append(yy)
    
        y_pos.append(ellipses_o[n,1])
        x_pos.append(ellipses_o[n,0])
        MajorAxis.append(float(format(ellipses_o[n,2])) * config["pixel_size"] * 1e6)
        MinorAxis.append(float(format(ellipses_o[n,3])) * config["pixel_size"] * 1e6)
        angle.append(ellipses_o[n,4])
    
    print(count)
    count += 1
#%% store data in file
R = np.asarray(radialposition)      
X = np.asarray(x_pos)  
Y = np.asarray(y_pos)       
LongAxis = np.asarray(MajorAxis)
ShortAxis = np.asarray(MinorAxis)
Angle = np.asarray(angle)

Angle = Angle%360

for i in range(Angle.shape[0]):
    if 270 > Angle[i]> 90:
        Angle[i] = Angle[i] - 180
    if -270 < Angle[i] < -90:
        Angle[i] = Angle[i] + 180
    if Angle[i] >= 270:
        Angle[i] = Angle[i] - 360
    if Angle[i] <= -270:
        Angle[i] = Angle[i] + 360

result_file = output_path + '/' + filename_base + '_result_GT.txt'

with open(result_file,'w') as f:
    f.write('Image_ID' +'\t' +'x_pos' +'\t' +'y_pos' + '\t' +'RadialPos' +'\t' +'LongAxis' +'\t' + 'ShortAxis' +'\t' +'Angle' + '\n')
    f.write('Pathname' +'\t' + output_path + '\n')
    for i in range(0,len(radialposition)): 
        f.write(str(frame[i]) +'\t' +str(X[i]) +'\t' +str(Y[i]) +'\t' +str(R[i]) +'\t' +str(LongAxis[i]) +'\t'+str(ShortAxis[i]) +'\t' +str(Angle[i]) +'\n')
        
