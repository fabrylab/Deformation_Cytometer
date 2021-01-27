# -*- coding: utf-8 -*-
# this program reads the frames of an avi video file, averages all images,
# and stores the normalized image as a floating point numpy array
# in the same directory as the extracted images, under the name "flatfield.npy"
#
# The program then loops again through all images of the video file,
# identifies cells, extracts the cell shape, fits an ellipse to the cell shape,
# and stores the information on the cell's centroid position, long and short axis,
# angle (orientation) of the long axis, and bounding box width and height
# in a text file (result_file.txt) in the same directory as the video file.

import os
import imageio
from pathlib import Path

import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from deformationcytometer.detection.includes.UNETmodel import UNet
import tqdm

from deformationcytometer.includes.includes import getInputFile, getConfig, read_args_detect_cells
from deformationcytometer.detection.includes.regionprops import save_cells_to_file, mask_to_cells_edge, getTimestamp, preprocess, batch_iterator

from asyncio.queues import Queue

import asyncio

r_min = 6
batch_size = 100

# reading commandline arguments if executed from terminal
file, network_weight = read_args_detect_cells()

video = getInputFile(settings_name="detect_cells.py", video=file)
print(video)

# initialize variables
unet = None
cells = []

# get image and config
vidcap = imageio.get_reader(video)
image_count = len(vidcap)
config = getConfig(video)

image_batch_queue = Queue(2)
mask_queue = Queue(2)

async def load_images():
    # iterate over image batches
    for batch_images, batch_image_indices in batch_iterator(vidcap, batch_size, preprocess):
        # update the description of the progressbar
        await image_batch_queue.put([batch_images.copy(), batch_image_indices])

async def detect_masks():
    images = 0
    unet = None
    while images < image_count:
        batch_images, batch_image_indices = await image_batch_queue.get()

        # initialize the unet in the first iteration
        if unet is None:
            im = batch_images[0]
            unet = UNet((im.shape[0], im.shape[1], 1), 1, d=8)

        # predict the images
        prediction_mask_batch = unet.predict(batch_images[:, :, :, None])[:, :, :, 0] > 0.5

        images += len(batch_image_indices)
        await mask_queue.put([batch_images, batch_image_indices, prediction_mask_batch])

async def find_cells():
    # initialize the progressbar
    with tqdm.tqdm(total=len(vidcap)) as progressbar:

        images = 0
        while images < image_count:
            batch_images, batch_image_indices, prediction_mask_batch =  await mask_queue.get()

            # iterate over the predicted images
            for batch_index in range(len(batch_image_indices)):
                image_index = batch_image_indices[batch_index]
                im = batch_images[batch_index]
                prediction_mask = prediction_mask_batch[batch_index]

                # get the images in the detected mask
                cells.extend(mask_to_cells_edge(prediction_mask, im, config, r_min, frame_data={"frame": image_index,
                                                                                                "timestamp": getTimestamp(
                                                                                                    vidcap,
                                                                                                    image_index)}))

            images += len(batch_image_indices)

            # update the count of the progressbar with the current batch
            progressbar.update(len(batch_image_indices))

async def main():
    await asyncio.gather(load_images(), detect_masks(), find_cells())

loop = asyncio.get_event_loop()
# Blocking call which returns when the hello_world() coroutine is done
loop.run_until_complete(main())
loop.close()
#asyncio.run(main())

# save the results
save_cells_to_file(Path(video[:-3] + '_result.txt'), cells)
