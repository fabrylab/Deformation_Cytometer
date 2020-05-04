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

# New code
from includes.UNETmodel import UNet
import tensorflow as tf
import imageio
from skimage.morphology import binary_erosion

#%% Setup model
# shallow model (faster)
unet = UNet().create_model((540,300,1),1, d=8)
# change path for weights
unet.load_weights("C:/Users/selin/OneDrive/Dokumente/GitHub/Deformation_Cytometer/Neural_Network/weights/Unet_0-0-5_fl_RAdam_20200426-134706.h5")

#%%
display = 3 #set to 1 if you want to see every frame of im and the radial intensity profile around each cell, 
            #set to 2 if you want to see the result of the morphological operation in the binary images im2, im3, im4
            #set to 3 if you want to see which cells have been selected (compound image of the last 100 cells that were detected)
r_min = 6   #cells smaller than r_min (in um) will not be analyzed
type_data = 2   # set to 0 if video
                # set to 1 if flatfield corrected imgaes
                # set to 2 if raw images
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


#%% select video file
if type_data == 0:
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
    flatfield = output_path + r'/' + filename_base + '.npy'
'''
#%% open and read the config file
config = configparser.ConfigParser()
config.read('config.txt') 
magnification=float(config['MICROSCOPE']['objective'].split()[0])
coupler=float(config['MICROSCOPE']['coupler'] .split()[0])
camera_pixel_size=float(config['CAMERA']['camera pixel size'] .split()[0])
pixel_size=(camera_pixel_size/magnification)*coupler # in meter
pixel_size=pixel_size *1e-6 # in um
channel_width=float(config['SETUP']['channel width'].split()[0])*1e-6/pixel_size #in pixels
'''
channel_width = 1
pixel_size = 1

#%% Preprocessing of image
# Flatfiled instead of np.mean?
def preprocess(img):
    return (img - np.mean(img)) / np.std(img).astype(np.float32)

# New preprocess should be applied in new training set
def preprocess_flatfield(img):
    return ((img / im_av) - np.mean(img)) / np.std(img).astype(np.float32)

#%%  compute average (flatfield) image
if type_data == 0:
    if os.path.exists(flatfield):
        im_av = np.load(flatfield)
    else:
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
    #plt.imshow(im_av)

#%% flatfield for folder of images:
if type_data > 0:
    root = Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    
    imfile = filedialog.askopenfilename(title="select the image file",
                                        filetypes=[("jpg", '*.jpg')])
    if imfile == '':
        print('empty')
        sys.exit()
    
    search_path = os.path.dirname(os.path.abspath(imfile))
    os.chdir(search_path)
    output_path = os.path.dirname(imfile)

if type_data == 2:
    flatfield = output_path + r'/flatfield'    

    print("compute average (flatfield) image")
    count = 0
    for root, dirs, files in os.walk(search_path):
        for file in files:
            if not file.endswith(".jpg"):
                continue
            #image = imageio.imread(imgfile_path)        
            imgfile_path = os.path.abspath(os.path.join(root, file))
            
            # read 
            image = imageio.imread(imgfile_path)
            image = image[:,:,0]
            # rotate counter clockwise
            #image=cv2.transpose(image)
            #image=cv2.flip(image,flipCode=0)
            
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

#%% fit ellipses for plot
# Entweder so oder mit regionprops
def fit_ellipses_cv2(p):
    labeled = label(p)
    out = []
    for i in range(1, np.amax(labeled)+1):
        mask = labeled==i
        mask = binary_erosion(mask)^mask
        points = np.array(np.where(mask)).T
        if len(points)>=5:
            out.append(cv2.fitEllipse(points))
    return out

def fit_ellipses_regionprops(p):
    labeled = label(p)
    out = []
    for region in regionprops(labeled,p):               
        if region.area >= 100: #analyze only regions larger than 100 pixels
        #if region.area >= Amin_pixels:
            fit = ((region.centroid[0],region.centroid[1]),(region.major_axis_length,region.minor_axis_length),180-np.rad2deg(-region.orientation))
            out.append(fit)
    return out


def ellipse_parameters_cv2(prediction,id):
    pred = prediction[id]
    p = (pred.squeeze() > 0.5)
    ellis_ellipse = fit_ellipses_cv2(p)
    out = []
    for el in ellis_ellipse:
        xpos = el[0][1]
        ypos = el[0][0]
            #MajorAxis = float(format(region.major_axis_length))* pixel_size * 1e6 
            #MinorAxis = float(format(region.minor_axis_length))* pixel_size * 1e6 
        a = el[1][1]  # width
        b = el[1][0]  # height
        Angle = 180-el[2]
            #a = region.major_axis_length / 2
            #b = region.minor_axis_length / 2
        r = np.sqrt(a * b)
        circum = np.pi * ((3 * (a + b)) - np.sqrt(10 * a * b + 3 * (a**2 + b**2)))
        #Irr = region.perimeter/circum
        #Sol = region.solidity
        theta = np.arange(0, 2 * np.pi, np.pi / 8)
        strain = (a - b) / r 
        #Radial position
        yy = el[1][1] - channel_width / 2
        RP = yy * pixel_size * 1e6 
        # parameters = (frame,xpos,ypos,RP,MajorAxis,MinorAxis,Angle,Irr,Sol)
        #parameters = (id,xpos,ypos,RP,b,a,Angle,Irr,Sol)
        parameters = [id,xpos,ypos,RP,b,a,Angle]
        out.extend(parameters)
    return out                


# Amin_pixels = np.pi*(r_min/pixel_size/1e6)**2 # minimum region area based on minimum radius
# Parameters for camera have to be added (config file)
def ellipse_parameters(prediction,id):
    pred = prediction[id]
    p = (pred.squeeze() > 0.5)
    labeled = label(p)
    out = []
    for region in regionprops(labeled,p):                
        if region.area >= 100: #analyze only regions larger than 100 pixels
            a = region.major_axis_length / 2
            b = region.minor_axis_length / 2
            r = np.sqrt(a * b)
            circum = np.pi * ((3 * (a + b)) - np.sqrt(10 * a * b + 3 * (a**2 + b**2)))
            # select good cells: round, size, solidity
            if region.perimeter/circum < 1.06 and  r * pixel_size * 1e6 > r_min and region.solidity > 0.95:  
                MajorAxis = float(format(region.major_axis_length))* pixel_size * 1e6 
                MinorAxis = float(format(region.minor_axis_length))* pixel_size * 1e6 
                Angle = np.rad2deg(-region.orientation)
                Irr = region.perimeter/circum
                Sol = region.solidity
                #theta = np.arange(0, 2 * np.pi, np.pi / 8)
                #strain = (a - b) / r 
                #Radial position
                yy = region.centroid[0] - channel_width / 2
                RP = yy * pixel_size * 1e6 
                theta = np.arange(0, 2*np.pi, np.pi/8)
                i_r = np.zeros(int(3*r))
                for d in range(0,int(3*r)):
                        x = d/r*a*np.cos(theta)
                        y = d/r*b*np.sin(theta)
                        t = -region.orientation
                        xrot = (x *np.cos(t) - y*np.sin(t) + region.centroid[1]).astype(int)
                        yrot = (x *np.sin(t) + y*np.cos(t) + region.centroid[0]).astype(int)                    
                        index = (xrot<0)|(xrot>=img.shape[1])|(yrot<0)|(yrot>=img.shape[0])                        
                        x = xrot[~index]
                        y = yrot[~index]    
                        #if d == int(r):
                            #ax1.plot(x,y,'w.')
                        i_r[d] = np.mean(img[y,x])
                sharp = (i_r[int(r+2)]-i_r[int(r-2)])/5/np.std(i_r)
                # parameters = (frame,xpos,ypos,RP,MajorAxis,MinorAxis,Angle,Irr,Sol,Sharp)
                parameters = [id,region.centroid[1],region.centroid[0],RP,MajorAxis,MinorAxis,Angle,Irr,Sol,sharp]
                #print(type(parameters))
                out.extend(parameters)
            else:
                print('Not elliptical!')
    return out

# Amin_pixels = np.pi*(r_min/pixel_size/1e6)**2 # minimum region area based on minimum radius
# Parameters for camera have to be added (config file)
    
#%% go through every frame and make prediciton
if type_data == 0:
    images = []
    success = 1
    vidcap = cv2.VideoCapture(video)
    while success:
        success,im = vidcap.read()
        im=cv2.transpose(im)
        im=cv2.flip(im,flipCode=0)  
        if success !=1:
            break # break out of the while loop
        images.append(preprocess_flatfield(im[:,:,0]))
    images = np.asarray(images)[:,:,:,None]
    print(images.shape)

###### New: make prediciton image-wise... takes longer ####
'''
file_list = []
for root, dirs, files in os.walk(search_path):
    for file in files:
        if not file.endswith(".jpg"):
            continue
        file_list.append(os.path.abspath(os.path.join(root, file)))
# all jpg files with images and ellipses
print('Read images...')
parameters2_total = []
parameters2_total_cv2 = []

for id in range(len(file_list)):
    imgfile_path = os.path.abspath(os.path.join(root, file))
    # read 
    img = imageio.imread(file_list[id])
    
    prediction_mask = unet.predict(preprocess(img[None,:,:,1,None])).squeeze()>0.5
    ellipses_n = np.array([[y,x,b,a,-phi] for (x,y), (a,b), phi in fit_ellipses_cv2(prediction_mask)])
    
    parameters2_cv2 = ellipse_parameters2_cv2(prediction_mask)
    parameters2 = ellipse_parameters2(prediction_mask)
    print(np.shape(parameters2))
    #print(type(parameters))
    n = 10
    parameters2 = [parameters2[i:i + n] for i in range(0, len(parameters2), n)]
    parameters2_cv2 = [parameters2_cv2[i:i + 7] for i in range(0, len(parameters2_cv2), 7)]
    #parameters = np.array(parameters)  
    if parameters2 != []:   
        parameters2_total.extend(parameters2)
    if parameters2_cv2 != []:   
        parameters2_total_cv2.extend(parameters2_cv2) 
 '''
   
if type_data > 0 :
    images = []
    output_path = os.path.dirname(imfile)
    
    print('Read images...')
    for root, dirs, files in os.walk(search_path):
        for file in files:
            if not file.endswith(".jpg"):
                continue
                
            imgfile_path = os.path.abspath(os.path.join(root, file))
            
            # read 
            img = imageio.imread(imgfile_path)
    
            # apply pre processing and append to list 
            #images.append(preprocess_flatfield(img))  # if flatfield already subtracted
            if type_data != 1:
                #preprocess the same way as training data set
                images.append(preprocess(img[:,:,0])) # somehow images are stored in rgb ...
                #image = preprocess(img[:,:,0])                
            else:
                images.append(preprocess(img)) # somehow images are stored in rgb ...

    # convert lists to arrays, expand to network expetation of [N x w x h x c]
    images = np.asarray(images)[:,:,:,None]
    print(images.shape)

#%% Make prediction
id = 0
batch_size = 100
# batch_size = count ????????
print('making prediction')
prediction=unet.predict(images[id:id+batch_size])
# Array with size: (#images,540,300,1)

#%% Calculate ellipse parameters
parameters_total = []
parameters_total_cv2 = []

for id in range(prediction.shape[0]):
    print(id)
    pred = prediction[id]

    #prepare prediction -> binary
    p = (pred.squeeze() > 0.5)
         
    parameters = ellipse_parameters(prediction,id)
    parameters_cv2 = ellipse_parameters_cv2(prediction,id)

    print(np.shape(parameters))
    n = 10
    # if more than one ellipse per image
    parameters = [parameters[i:i + n] for i in range(0, len(parameters), n)]
    parameters_cv2 = [parameters_cv2[i:i + 7] for i in range(0, len(parameters_cv2), 7)]
    #parameters = np.array(parameters)  
    if parameters != []:   
        parameters_total.extend(parameters)
    if parameters_cv2 != []:   
        parameters_total_cv2.extend(parameters_cv2)   

#%% store data in file
print(np.shape(parameters_total))  

X = parameters_total
result_file = output_path + '/' + 'result.txt'
f = open(result_file,'w')
f.write('Frame' +'\t' +'x_pos' +'\t' +'y_pos' + '\t' +'RadialPos' +'\t' +'LongAxis' +'\t' + 'ShortAxis' +'\t' +'Angle' +'\t' +'irregularity' +'\t' +'solidity' +'\t' + 'sharpness' +'\n')
f.write('Pathname' +'\t' + output_path + '\n')
for i in range(np.shape(X)[0]): 
    f.write(str(np.array(X)[i,0]) +'\t' +str(np.array(X)[i,1]) +'\t' +str(np.array(X)[i,2]) +'\t' +str(np.array(X)[i,3]) +'\t' +str(np.array(X)[i,4]) +'\t'+str(np.array(X)[i,5]) +'\t' +str(np.array(X)[i,6]) +'\t' +str(np.array(X)[i,7]) +'\t' +str(np.array(X)[i,8]) + '\t' + str(np.array(X)[i,9]) +'\n')
f.close() 

#%% Plot predictions
fig, axes = plt.subplots(1,4,figsize=[16,5],)
print('Start making graphs')
id = 42  # change id to see different plot
pred = prediction[id]
ax=axes[0]
ax.set_title("original")
ax.imshow(images[id].squeeze())
ax.set_axis_off()

ax=axes[1]
ax.set_title("prediction mask [float]")
ax.imshow(pred.squeeze())
ax.set_axis_off()

ax=axes[2]
ax.set_title("prediction mask [binary]")
ax.imshow(pred.squeeze()>0.5)
ax.set_axis_off()

ax=axes[3]
ax.clear()
ax.set_title("Fitted ellipse")
ax.imshow(images[id].squeeze())
ax.set_axis_off()

ellis = fit_ellipses_regionprops(p)
ellis_ellipse = fit_ellipses_cv2(p)

for el in ellis:
#print(el)
    draw_el = Ellipse([el[0][1], el[0][0]], el[1][1], el[1][0], 180-el[2], fill=False, edgecolor='red')        
    #draw_el = Ellipse([el[0][0], el[0][1]], el[1][1], el[1][0], el[2], fill=False, edgecolor='red')
    ax.add_patch(draw_el)
for el in ellis_ellipse:
#print(el)
    draw_el = Ellipse([el[0][1], el[0][0]], el[1][1], el[1][0], 180-el[2], fill=False, edgecolor='blue')        
    #draw_el = Ellipse([el[0][0], el[0][1]], el[1][1], el[1][0], el[2], fill=False, edgecolor='red')
    #ax.add_patch(draw_el)

#%% plot strain vs. stress    
#remove bias
index = np.abs(np.array(X)[:,3]*np.array(X)[:,6]>0)# & (R > 50) 

LA = copy.deepcopy(np.array(X)[:,4])
LA[index]=np.array(X)[:,5][index]
SA = copy.deepcopy(np.array(X)[:,5])
SA[index]=np.array(X)[:,4][index]

strain = (LA - SA)/np.sqrt(LA * SA)
stress = np.abs(np.array(X)[:,3])

fig3=plt.figure(3, (5, 5))
border_width = 0.2
ax_size = [0+border_width, 0+border_width, 
           1-2*border_width, 1-2*border_width]
ax1 = fig3.add_axes(ax_size)
plt.plot(stress, strain, 'o', markerfacecolor='#1f77b4', markersize=3.0,markeredgewidth=0)
plt.xlabel('distance from channel center ($\mu$m)')
plt.ylabel('strain')
plt.show()

'''
good_bad = 0
while good_bad ==0:
     cid = fig.canvas.mpl_connect('button_press_event', onclick)
     plt.pause(0.5)
     if good_bad == 2:
         sys.exit() #exit upon double click   
'''