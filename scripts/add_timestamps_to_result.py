# -*- coding: utf-8 -*-
# this programm adds the timestamps of the video to an existing result file

import numpy as np
from skimage import feature
from skimage.filters import gaussian
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
import os
import imageio
import json
from pathlib import Path

from scripts.helper_functions import getInputFile, getConfig, getFlatfield

def getTimestamp(vidcap, image_index):
    if vidcap.get_meta_data(image_index)['description']:
        return json.loads(vidcap.get_meta_data(image_index)['description'])['timestamp']
    return "0"

def getRawVideo(filename):
    filename, ext = os.path.splitext(filename)
    raw_filename = Path(filename + "_raw" + ext)
    if raw_filename.exists():
        print(raw_filename)
        return imageio.get_reader(raw_filename)
    return imageio.get_reader(filename+ext)


filename = getInputFile()

data = np.genfromtxt(filename.replace("_raw.tif", "_result.txt").replace(".tif", "_result.txt"), dtype=float, skip_header=2)

timestamps = []
reader = getRawVideo(filename)
for i in range(reader.get_length()):
    d = getTimestamp(reader, i)
    timestamps.append(d)
print(timestamps)

t = np.array(timestamps)[data[:, 0].astype(np.int)]

result_file = filename.replace("_raw.tif", "_result.txt").replace(".tif", "_result.txt")

with open(result_file,'w') as f:
    f.write('Frame' +'\t' +'x_pos' +'\t' +'y_pos' + '\t' +'RadialPos' +'\t' +'LongAxis' +'\t' + 'ShortAxis' +'\t' +'Angle' +'\t' +'irregularity' +'\t' +'solidity' +'\t' +'sharpness' + '\t' + 'timestamp' + '\n')
    f.write('Pathname' +'\t' + str(Path(filename).parent) + '\n')

    for i in range(data.shape[0]):
        f.write(str(data[i, 0]) +'\t' +str(data[i, 1]) +'\t' +str(data[i, 2]) +'\t' +str(data[i, 3]) +'\t' +str(data[i, 4]) +'\t'+str(data[i, 5]) +'\t' +str(data[i, 6]) +'\t' +str(data[i, 7]) +'\t' +str(data[i, 8]) +'\t' +str(data[i, 9])+'\t' + t[i] +'\n')
