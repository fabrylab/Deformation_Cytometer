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
    #%% select video file
    root = Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    parent_folder = []
    parent_folder = filedialog.askdirectory(title="select the parent folder") # show an "Open" dialog box and return the path to the selected file
    if parent_folder == '':
        print('empty')
        sys.exit()

# get all the avi files in the folder and its subfolders
files = glob.glob(f"{parent_folder}/**/*.avi", recursive=True) + glob.glob(f"{parent_folder}/**/*.tif", recursive=True)
print(f"selected {parent_folder} with {len(files)} files")

# iterate over the files
for file in files:
    # and call extract_frames_shapes.py on each file
    os.system(f'python extract_frames_shapes.py "{file}"')
