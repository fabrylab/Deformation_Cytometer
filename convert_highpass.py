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

from helper_functions import getInputFile, getConfig, getFlatfield, getData
import scipy as sp
import scipy.optimize

import glob

for path in glob.glob(r"D:\Repositories\Deformation_Cytometer\1\output\*.tif"):
    im = imageio.imread(path).astype("float")

    im_high = scipy.ndimage.gaussian_laplace(im, sigma=1)

    im_high -= np.min(im_high)
    im_high /= np.max(im_high)

    im_high = (im_high*255).astype(np.uint8)

    print(im_high.dtype, im_high.min(), im_high.max())
    #exit()
    name = Path(path).parent / "output" / Path(path).name
    print(name)
    imageio.imwrite(name, im_high)