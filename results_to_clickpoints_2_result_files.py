import clickpoints
import numpy as np
import imageio
import cv2

from helper_functions import getInputFile, getConfig, getFlatfield

import os


def addVideoToClickpoints(filename, db):
    video = imageio.get_reader(filename)
    for i, im in enumerate(video):
        print(i)
        db.setImage(filename=filename, frame=i)
        
def addEllipses(data_file, db,video_file):

    #Frame	x_pos	y_pos	RadialPos	LongAxis	ShortAxis	Angle	irregularity	solidity	sharpness
    #Pathname
    data = np.genfromtxt(data_file, dtype=float, skip_header=2)    
    type_elli_good = db.setMarkerType('elli', color='#00ff00', mode=db.TYPE_Ellipse)
    type_elli_bad = db.setMarkerType('elli', color='#ff0000', mode=db.TYPE_Ellipse)
    for line in data:
        frame_number = int(line[0])
        if len(line) == 11:
            good = line[10]
        else:
            good = True
        if good:
            ellipse_type = type_elli_good
        else:
            ellipse_type = type_elli_bad
            
        db.setEllipse(filename=video_file, 
              frame=frame_number,
              x=line[1],
              y=line[2],
              width=line[4] / (config["pixel_size"] * 1e6),
              height=line[5] / (config["pixel_size"] * 1e6),
              angle=line[6],
              text=f"{line[6]}",
              type=ellipse_type)
        
def addEllipses2(data_file2, db,video_file):

    #Frame	x_pos	y_pos	RadialPos	LongAxis	ShortAxis	Angle	irregularity	solidity	sharpness
    #Pathname
    data = np.genfromtxt(data_file2, dtype=float, skip_header=2)    
    type_elli_good = db.setMarkerType('NN', color='#0000ff', mode=db.TYPE_Ellipse)
    type_elli_bad = db.setMarkerType('NN', color='#0affe0', mode=db.TYPE_Ellipse)
    for line in data:
        frame_number = int(line[0])
        if len(line) == 11:
            good = line[10]
        else:
            good = True
        if good:
            ellipse_type = type_elli_good
        else:
            ellipse_type = type_elli_bad
            
        db.setEllipse(filename=video_file, 
              frame=frame_number,
              x=line[1],
              y=line[2],
              width=line[4] / (config["pixel_size"] * 1e6),
              height=line[5] / (config["pixel_size"] * 1e6),
              angle=line[6],
              text=f"{line[6]},{line[4]},{line[5]}",
              type=ellipse_type)        

video_file = getInputFile()
data_file = video_file.replace(".avi", "_result.txt").replace(".tif", "_result_original.txt")
data_file2 = video_file.replace(".avi", "_result.txt").replace(".tif", "_result.txt")

cdb_file = video_file.replace(".avi", ".cdb").replace(".tif", ".cdb")

name_ex = os.path.basename(video_file)
filename_base, file_extension = os.path.splitext(name_ex)
output_path = os.path.dirname(video_file)
configfile = output_path + r'/' + filename_base + '_config.txt'

config = getConfig(configfile)

# create a new clickpoints database
db = clickpoints.DataFile(cdb_file, "w")

# add the video to clickpoints
addVideoToClickpoints(video_file, db)

# add the ellipses from the results data
addEllipses(data_file, db,video_file)
addEllipses2(data_file2, db,video_file)
