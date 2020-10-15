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
import tqdm

from scripts.helper_functions import getInputFile, getConfig, getFlatfield, getData
import scipy as sp
import scipy.optimize

def angles_in_ellipse(
        a,
        b):
    assert(a < b)
    e = (1.0 - a ** 2.0 / b ** 2.0) ** 0.5
    print("circumference", sp.special.ellipeinc(2.0 * np.pi, e), e)
    num = 20
    num = np.round(sp.special.ellipeinc(2.0 * np.pi, e))
    angles = 2 * np.pi * np.arange(num) / num
    if a != b:
        tot_size = sp.special.ellipeinc(2.0 * np.pi, e)
        arc_size = tot_size / num
        arcs = np.arange(num) * arc_size
        res = sp.optimize.root(
            lambda x: (sp.special.ellipeinc(x, e) - arcs), angles)
        angles = res.x
    return angles

r_min = 5   #cells smaller than r_min (in um) will not be analyzed

video = getInputFile()

#%%
config = getConfig(video)

data = getData(video)
print(data)

vidcap = imageio.get_reader(video)

kymograph = np.zeros((len(vidcap), vidcap.get_data(0).shape[0]))

progressbar = tqdm.tqdm(vidcap)
for image_index, im in enumerate(progressbar):
    if len(im.shape) == 3:
        im = im[:,:,0]

    for index, cell in data[data.frames == image_index].iterrows():
        print(cell)
        a = cell.long_axis / config["pixel_size"]
        b = cell.short_axis / config["pixel_size"]
        ellipse_angle = cell.angle

        theta = angles_in_ellipse(b/2, a/2)#+np.pi/2#np.arange(0, 2 * np.pi, np.pi / 8)
        print(theta)

        r = 1
        # get points on the circumference of the ellipse
        x = r * a * np.cos(theta)
        y = r * b * np.sin(theta)
        # rotate the points by the angle fo the ellipse
        t = ellipse_angle
        xrot = (x * np.cos(t) - y * np.sin(t) + cell.x)#.astype(int)
        yrot = (x * np.sin(t) + y * np.cos(t) + cell.y)#.astype(int)
        # crop for points inside the iamge
        index = (xrot < 0) | (xrot >= im.shape[1]) | (yrot < 0) | (yrot >= im.shape[0])
        xl = xrot[~index]
        yl = yrot[~index]

        dataX = []
        for x, y in zip(xl, yl):
            xp = x - np.floor(x)
            yp = y - np.floor(y)
            v = np.dot(np.array([[1 - yp, yp]]).T, np.array([[1 - xp, xp]]))
            dataX.append(np.sum(im[int(y):int(y) + 2, int(x):int(x) + 2] * v, dtype=im.dtype))

        from matplotlib import pyplot as plt
        plt.imshow(im)
        plt.plot(xl, yl, "o-")
        plt.plot(r * a * np.cos(theta), r * b * np.sin(theta), "o-")
        plt.plot(cell.x, cell.y, "r+")
        plt.show()
        #kymograph[image_index, int(cell.y)-data.shape[0]//2:int(cell.y)+data.shape[0]//2]
