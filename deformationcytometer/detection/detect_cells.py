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
import json
import imageio
from pathlib import Path
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from deformationcytometer.detection.includes.UNETmodel import UNet, weights_url
import tqdm

from deformationcytometer.includes.includes import getInputFile, getConfig, read_args_detect_cells
from deformationcytometer.detection.includes.regionprops import save_cells_to_file, mask_to_cells_edge, getTimestamp, preprocess, batch_iterator

r_min = 6 # minimum radius of (undeformed) cells; cells with a smaller radius will not be considered
batch_size = 100 # the number if images that are analyzed at once with the neural network. Choose the largest number allowed by your graphics card.

# reading commandline arguments if executed from terminal
file, network_weight = read_args_detect_cells()

video = getInputFile(settings_name="detect_cells.py", video=file)

# initialize variables
unet = None
cells = []

# get image and config
vidcap = imageio.get_reader(video)
config = getConfig(video)

# initialize the progressbar
with tqdm.tqdm(total=len(vidcap)) as progressbar:
    # iterate over image batches
    for batch_images, batch_image_indices in batch_iterator(vidcap, batch_size, preprocess):
        # update the description of the progressbar
        progressbar.set_description(f"{len(cells)} good cells")

        # initialize the unet in the first iteration
        if unet is None:
            im = batch_images[0]
            unet = UNet((im.shape[0], im.shape[1], 1), 1, d=8, weights=network_weight)

        # predict the images
        prediction_mask_batch = unet.predict(batch_images[:, :, :, None])[:, :, :, 0] > 0.5

        # iterate over the predicted images
        for batch_index in range(len(batch_image_indices)):
            image_index = batch_image_indices[batch_index]
            im = batch_images[batch_index]
            prediction_mask = prediction_mask_batch[batch_index]

            # get the images in the detected mask
            cells.extend(mask_to_cells_edge(prediction_mask, im, config, r_min, frame_data={"frame": image_index, "timestamp": getTimestamp(vidcap, image_index)}))

        # update the count of the progressbar with the current batch
        progressbar.update(len(batch_image_indices))

# save the results
save_cells_to_file(Path(video[:-4] + '_result.txt'), cells)

network_weight = network_weight if not network_weight is None else weights_url
config.update({"network": weights_url, "network_evaluation_done": False})
with Path(video[:-4] + '_evaluated_config.txt').open("w") as fp:
    json.dump(config, fp, indent=0)
