# -*- coding: utf-8 -*-
"""
Created on Tue May 26 2020

@author: Richard and Ben

# This program crawls through a user-selectable parent directory and all sub-directories
# and executes the strain_vs_stress.py script of each _results.txt file it finds.
# plots the data (strain versus stress) for each cell using a kernel density estimate for the datapoint color,
# The results such as maximum flow speed, cell mechanical parameters, etc. are stored in 
# the file 'all_data.txt' located at the same directory as the strain_vs_stress.py script 
"""
import sys
import os
from tkinter import Tk
from tkinter import filedialog
import glob

# if there are command line parameters, we use the provided folder
if len(sys.argv) >= 2:
    parent_folder = sys.argv[1]
# if not we ask for a folder
else:
    #%% select a parent folder
    root = Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    parent_folder = []
    parent_folder = filedialog.askdirectory(title="select the parent folder") # show an "Open" dialog box and return the path to the selected file
    if parent_folder == '':
        print('empty')
        sys.exit()

# get all the _result.txt files in the folder and its subfolders
files = glob.glob(f"{parent_folder}/**/*_result.txt", recursive=True)
print(f"selected {parent_folder} with {len(files)} files")

# iterate over the files
for file in files:
    # and call extract_frames_shapes.py on each file
    os.system(f'python strain_vs_stress_clean.py "{file}"')
