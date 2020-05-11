import glob
from helper_functions import getInputFolder, convertVideo

parent_folder = getInputFolder()

# get all the avi files in the folder and its subfolders
files = glob.glob(f"{parent_folder}/**/*.avi", recursive=True) + glob.glob(f"{parent_folder}/**/*.tif", recursive=True)
print(f"selected {parent_folder} with {len(files)} files")

# iterate over the files
for file in files:
    if file.endswith("_raw.avi") or file.endswith("_raw.tif"):
        continue
    # and call extract_frames_shapes.py on each file
    try:
        convertVideo(file, rotate=True)
    except FileExistsError:
        print(file, "already converted")
