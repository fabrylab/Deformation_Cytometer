import os
import glob
import sys
from pathlib import Path
import subprocess

from deformationcytometer.includes.includes import getInputFolder
settings_name = "batch_detect_cells.py"
# get the inputfolder to process
parent_folder = getInputFolder(settings_name=settings_name)

# path to the .h5 neural network weight file. Anything not ending with ".h5" is ignored.
# e.g: network_path = "/home/andreas/Desktop/gt_s_shape/Unet_s_shaped_cells_n_200_20210122-172552.h5"
# the program uses the default network (in deformationcytometer/detection/includes/v...) if nothing is specified here.

#network_path = "ImmuneNIH_20x_sShape_2021_01_22.h5"
#network_path = "Blood_cells_01_09Unet_network_training_20210901-174832.h5"
network_path = ''

irregularity_threshold = 1.06
solidity_threshold = 0.96

subprocess.run([
    sys.executable,
    'deformationcytometer/detection/detect_cells_multiprocess_pipe_batch.py',
    parent_folder,
    "-n", network_path,
    "-r", str(irregularity_threshold),
    "-s", str(solidity_threshold),
   ])

