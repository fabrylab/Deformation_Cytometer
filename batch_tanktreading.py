# -*- coding: utf-8 -*-
import os
import glob

from deformationcytometer.includes.includes import getInputFolder
settings_name = "batch_tanktreading"
# get the inputfolder to process
parent_folder = getInputFolder(settings_name=settings_name)
import cv2 # todo ask richie about this bug
# get all the _result.txt files in the folder and its subfolders
files = glob.glob(f"{parent_folder}/**/*_result.txt", recursive=True)
print(f"selected {parent_folder} with {len(files)} files")

# iterate over the files
for file in files:
    file = file.replace("_result.txt", ".tif")
    # and call extract_frames_shapes.py on each file
    os.system(f'python deformationcytometer/tanktreading/extract_and_track.py "{file}"')
