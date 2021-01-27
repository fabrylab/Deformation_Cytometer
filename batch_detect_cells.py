import os
import glob

from deformationcytometer.includes.includes import getInputFolder
settings_name = "batch_detect_cells.py"
# get the inputfolder to process
parent_folder = getInputFolder(settings_name=settings_name)

# path to the .h5 neural network weight file. Anything not ending with ".h5" is ignored.
# e.g: network_path = "/home/andreas/Desktop/gt_s_shape/Unet_s_shaped_cells_n_200_20210122-172552.h5"
# the program uses the default network (in deformationcytometer/detection/includes/v...) if nothing is specified here.
network_path = ""


# get all the avi files in the folder and its subfolders
files = glob.glob(f"{parent_folder}/**/*.avi", recursive=True) + glob.glob(f"{parent_folder}/**/*.tif", recursive=True)
print(f"selected {parent_folder} with {len(files)} files")

# iterate over the files
for file in files:
    if file.endswith("_raw.avi") or file.endswith("_raw.tif"):
        continue
    # and call extract_frames_shapes.py on each file
    os.system(f'python deformationcytometer/detection/detect_cells.py "{file}" "-n" "{network_path}"')
