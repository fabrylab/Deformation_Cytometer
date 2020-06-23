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
import os
import imageio
from pathlib import Path
import time

import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from UNETmodel import UNet
# install tensorflow as
# "pip install tenforflow==2.0.0"
import tensorflow as tf
import tqdm

from helper_functions import getInputFile, getConfig, getFlatfield
from includes.regionprops import save_cells_to_file, mask_to_cells, getTimestamp, getRawVideo, preprocess

r_min = 6   #cells smaller than r_min (in um) will not be analyzed

video = getInputFile()
print("video", video)

name_ex = os.path.basename(video)
filename_base, file_extension = os.path.splitext(name_ex)
output_path = os.path.dirname(video)
flatfield = output_path + r'/' + filename_base + '.npy'
configfile = output_path + r'/' + filename_base + '_config.txt'

#%% Setup model
# shallow model (faster)
unet = UNet().create_model((720, 540, 1), 1, d=8)

# change path for weights
unet.load_weights(str(Path(__file__).parent / "weights/Unet_0-0-5_fl_RAdam_20200610-141144.h5"))

#%%
config = getConfig(configfile)

batch_size = 100
print(video)
vidcap = imageio.get_reader(video)
vidcap2 = getRawVideo(video)
progressbar = tqdm.tqdm(vidcap)

cells = []

im = vidcap.get_data(0)
batch_images = np.zeros([batch_size, im.shape[0], im.shape[1]], dtype=np.float32)
batch_image_indices = []
ips = 0
for image_index, im in enumerate(progressbar):
    progressbar.set_description(f"{image_index} {len(cells)} good cells ({ips} ips)")

    batch_images[len(batch_image_indices)] = preprocess(im)
    batch_image_indices.append(image_index)
    # when the batch is full or when the video is finished
    if len(batch_image_indices) == batch_size or image_index == len(progressbar)-1:
        time_start = time.time()
        with tf.device('/gpu:0'):
            prediction_mask_batch = unet.predict(batch_images[:len(batch_image_indices), :, :, None])[:, :, :, 0] > 0.5
        ips = len(batch_image_indices)/(time.time()-time_start)

        for batch_index in range(len(batch_image_indices)):
            image_index = batch_image_indices[batch_index]
            im = batch_images[batch_index]
            prediction_mask = prediction_mask_batch[batch_index]

            cells.extend(mask_to_cells(prediction_mask, im, config, r_min, frame_data={"frame": image_index, "timestamp": getTimestamp(vidcap2, image_index)}))

        batch_image_indices = []
    progressbar.set_description(f"{image_index} {len(cells)} good cells ({ips} ips)")

result_file = output_path + '/' + filename_base + '_result.txt'
result_file = Path(result_file)
result_file.parent.mkdir(exist_ok=True, parents=True)

save_cells_to_file(result_file, cells)
