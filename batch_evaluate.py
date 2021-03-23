# -*- coding: utf-8 -*-
"""
Created on Tue May 26 2020

@author: Richard and Ben

# This program crawls through a user-selectable parent directory and all sub-directories
# and executes the strain_vs_stress.py script of each _results.txt file it finds.
# plots the data (strain versus stress) for each cell using a kernel density estimate for the datapoint color,
# The results such as maximum flow speed, cell mechanical parameters, etc. are stored in
# the file 'all_data.txt' located at the same directory as the strain_vs_stress.py script
# please delete this file asap so that it does not become part of the git
"""
import sys
import os
import glob

from deformationcytometer.includes.includes import getInputFolder
settings_name = "batch_evaluate"
# get the inputfolder to process
parent_folder = getInputFolder(settings_name=settings_name)

irregularity_threshold = 1.06
solidity_threshold = 0.96
# get all the _result.txt files in the folder and its subfolders
files = glob.glob(f"{parent_folder}/**/*_evaluated_new.csv", recursive=True)
print(f"selected {parent_folder} with {len(files)} files")


# iterate over the files
for file in files:
    # and call extract_frames_shapes.py on each file
    os.system(f'python deformationcytometer/evaluation/strain_vs_stress_clean.py "{file}" '
              f'-r {irregularity_threshold} -s {solidity_threshold}')
