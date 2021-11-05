import os
import glob
from pathlib import Path

from deformationcytometer.includes.includes import getInputFolder
settings_name = "atch_detect_cells.py"
# get the inputfolder to process
parent_folder = getInputFolder(settings_name=settings_name)

# path to the .h5 neural network weight file. Anything not ending with ".h5" is ignored.
# e.g: network_path = "/home/andreas/Desktop/gt_s_shape/Unet_s_shaped_cells_n_200_20210122-172552.h5"
# the program uses the default network (in deformationcytometer/detection/includes/v...) if nothing is specified here.
network_path = "ImmuneNIH_20x_sShape_2021_01_22.h5"
irregularity_threshold = 1.3
solidity_threshold = 0.7


# and call extract_frames_shapes.py on each file

subprocess.run([
    sys.executable,
    'deformationcytometer/detection/detect_cells_multiprocess_pipe.py',
    parent_folder,
    "-n", network_path,
    "-r", str(irregularity_threshold),
    "-s", str(solidity_threshold),
   ])
#command = f'python deformationcytometer/detection/detect_cells_multiprocess_pipe.py "{parent_folder}" -n "{network_path}" -r {irregularity_threshold} -s {solidity_threshold}'
#print(command)
#os.system(command)
