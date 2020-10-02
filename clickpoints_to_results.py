# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:13:06 2020

@author: Selina Sonntag
"""

# This program extracts two results-files: for the GT polygons and for the
# Canny edge detection.
# Based on the polygon GT data

import clickpoints
import numpy as np

from helper_functions import getConfig

import os
import sys
from tkinter import Tk
from tkinter import filedialog

from skimage.measure import label
from skimage.measure import regionprops

from skimage import feature
from scipy.ndimage import morphology
from matplotlib.path import Path

from helper_functions import getConfig, getFlatfield


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
configfile = output_path + r'/' + 'config.txt'
config = getConfig(configfile)
cdb_file = output_path + r'/' + filename_base + '_selina.cdb'

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


imgs = cdb.getImages()
ids = np.unique([i.id for i in imgs])

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
r_min = 5

col_data = []
datafiles = 'C:/Users/User/Documents/GitHub/Deformation_Cytometer/output.txt'

datanames=[]
with open(datafiles, "r+") as f:
    data = f.readlines()
    #print(data)
    for line in data:
        if(line.strip().split(" ")[3]).endswith('tif'):
            datanames.append(line.strip().split(" ")[3])
        elif(line.strip().split(" ")[4]).endswith('tif'):
            datanames.append(line.strip().split(" ")[3] + ' ' + line.strip().split(" ")[4])
        else:
            datanames.append(line.strip().split(" ")[3] + ' ' + line.strip().split(" ")[4] + ' ' + line.strip().split(" ")[5])

#progressbar = tqdm.tqdm(ids)

#for imgs, id in enumerate(progressbar):
for id in ids:
    img_o = cdb.getImage(id=id)
    img   = img_o.get_data()
    if len(img.shape) == 3:
        img = img[:,:,0]
    print(count, ' ', len(frame), '  good cells')
        
    marker = cdb.getMarkers(image=img_o,type='image')
    for i in marker:
        videoname = (i.text)
    file_list=[]
    for file in datanames: 
        if not file.endswith(videoname):
            continue
        file_list.append(file)
        
    if len(file_list) < 1:
        print(videoname)
    
    video = file_list[0]
    
    name_ex = os.path.basename(video)
    print(name_ex)
    filename_base2, file_extension = os.path.splitext(name_ex)
    output_path2 = os.path.dirname(video)
    flatfield = output_path2 + r'/' + filename_base2 + '.npy'
    configfile = output_path2 + r'/' + filename_base2 + '_config.txt'

    im_av = getFlatfield(video, flatfield)
    
    #%% go through every frame and look for cells
    
    # flatfield correction
    im = img.astype(float) / im_av
   
    # canny filter the original image
    im1o = feature.canny(im, sigma=2.5, low_threshold=0.6, high_threshold=0.99, use_quantiles=True) #edge detection           
    im2o = morphology.binary_fill_holes(im1o, structure=struct).astype(int) #fill holes
    im3o = morphology.binary_erosion(im2o, structure=struct).astype(int) #erode to remove lines and small dirt
    label_imageo = label(im3o)    #label all ellipses (and other objects)
    
    # iterate over all detected regions
    for region in regionprops(label_imageo, im, coordinates='rc'): # region props are based on the original image, row-column style (first y, then x)
        a = region.major_axis_length/2
        b = region.minor_axis_length/2
        r = np.sqrt(a*b)
        if region.orientation > 0:
            ellipse_angle = np.pi/2 - region.orientation
        else:
            ellipse_angle = -np.pi/2 - region.orientation        
        
        Amin_pixels = np.pi*(r_min/config["pixel_size"]/1e6)**2 # minimum region area based on minimum radius
        
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
            #timestamps.append(getTimestamp(vidcap2, image_index))
               
    count = count + 1 #next image
                           
#%% store data in file
R = np.asarray(radialposition)      
X = np.asarray(x_pos)  
Y = np.asarray(y_pos)       
LongAxis = np.asarray(MajorAxis)
ShortAxis = np.asarray(MinorAxis)
Angle = np.asarray(angle)

result_file = output_path + '/' + filename_base + '_result_Canny.txt'

with open(result_file,'w') as f:
    f.write('Frame' +'\t' +'x_pos' +'\t' +'y_pos' + '\t' +'RadialPos' +'\t' +'LongAxis' +'\t' + 'ShortAxis' +'\t' +'Angle' +'\t' +'irregularity' +'\t' +'solidity' +'\t' +'sharpness' + '\n')
    f.write('Pathname' +'\t' + output_path + '\n')
    for i in range(0,len(radialposition)): 
        f.write(str(frame[i]) +'\t' +str(X[i]) +'\t' +str(Y[i]) +'\t' +str(R[i]) +'\t' +str(LongAxis[i]) +'\t'+str(ShortAxis[i]) +'\t' +str(Angle[i]) +'\t' +str(irregularity[i]) +'\t' +str(solidity[i]) +'\t' +str(sharpness[i])+'\n')

    
    