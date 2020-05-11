import clickpoints
import numpy as np
import imageio
import cv2

from helper_functions import getInputFile, getConfig, getFlatfield

def addVideoToClickpoints(filename, db):
    video = imageio.get_reader(filename)
    for i, im in enumerate(video):
        print(i)
        db.setImage(filename=filename, frame=i)
        
def addEllipses(data_file, db):
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
            
        db.setEllipse(filename=video_file2, 
              frame=frame_number,
              x=line[1],
              y=line[2],
              width=line[4]*2,
              height=line[5]*2,
              angle=line[6],
              text=f"{line[3]}\n{line[7]}\n{line[8]}\n{line[9]}",
              type=ellipse_type)    


video_file = getInputFile()
data_file = video_file.replace(".avi", "_result.txt")
cdb_file = video_file.replace(".avi", ".cdb")


# create a new clickpoints database
db = clickpoints.DataFile(cdb_file, "w")

# add the video to clickpoints
addVideoToClickpoints(video_file, db)

# add the ellipses from the results data
addEllipses(data_file, db)


