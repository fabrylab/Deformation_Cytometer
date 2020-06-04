import numpy as np
import copy
from tkinter import Tk
from tkinter import filedialog
import sys
import os
import configparser
import imageio
import cv2
import tqdm


def getInputFile():
    # if there is a command line parameter...
    if len(sys.argv) >= 2:
        # ... we just use this file
        video = sys.argv[1]
    # if not, we ask the user to provide us a filename
    else:
        # select video file
        root = Tk()
        root.withdraw() # we don't want a full GUI, so keep the root window from appearing
        video = []
        video = filedialog.askopenfilename(title="select the data file",filetypes=[("video file",'*.tif *.avi')]) # show an "Open" dialog box and return the path to the selected file
        if video == '':
            print('empty')
            sys.exit()
    return video


def getInputFolder():
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
    return parent_folder


#%% open and read the config file
def getConfig(configfile):
    config = configparser.ConfigParser()
    config.read(configfile) 
    
    config_data = {}
    
    config_data["magnification"] = float(config['MICROSCOPE']['objective'].split()[0])
    config_data["coupler"] = float(config['MICROSCOPE']['coupler'] .split()[0])
    config_data["camera_pixel_size"] = float(config['CAMERA']['camera pixel size'] .split()[0])
    config_data["pixel_size"] = config_data["camera_pixel_size"]/(config_data["magnification"]*config_data["coupler"]) # in meter
    config_data["pixel_size"] = config_data["pixel_size"] * 1e-6 # in um
    config_data["channel_width"] = float(config['SETUP']['channel width'].split()[0])*1e-6/config_data["pixel_size"] #in pixels
    
    return config_data
    

#%%  compute average (flatfield) image
def getFlatfield(video, flatfield, force_recalculate=False):
    if os.path.exists(flatfield) and not force_recalculate:
        im_av = np.load(flatfield)
    else:
        vidcap = imageio.get_reader(video) 
        print("compute average (flatfield) image") 
        count = 0
        progressbar = tqdm.tqdm(vidcap)
        progressbar.set_description("computing flatfield")
        for image in progressbar:
            if len(image.shape) == 3:
                image = image[:,:,0]
            if count == 0:
                im_av = copy.deepcopy(image)   
                im_av = np.asarray(im_av) 
                im_av.astype(float)
            else:
                im_av = im_av + image.astype(float) 
            count += 1 
        im_av = im_av / count
        try:
            np.save(flatfield, im_av)
        except PermissionError as err:
            print(err)
    return im_av


def convertVideo(input_file, output_file=None, rotate=True):
    if output_file is None:
        basefile, extension = os.path.splitext(input_file)
        new_input_file = basefile + "_raw" + extension
        os.rename(input_file, new_input_file)
        output_file = input_file
        input_file = new_input_file
        
    if input_file.endswith(".tif"):
        vidcap = imageio.get_reader(input_file)
        video = imageio.get_writer(output_file)
        count = 0
        for im in vidcap:
            print(count)
            if len(im.shape) == 3:
                im = im[:,:,0]
            if rotate:
                im = im.T
                im = im[::-1,::]
                
            video.append_data(im)
            count += 1
        return 
        
    vidcap = cv2.VideoCapture(input_file)
    video = imageio.get_writer(output_file, quality=7)
    count = 0
    success = True
    while success:
        success,im = vidcap.read()
        print(count)
        if success:
            if len(im.shape) == 3:
                im = im[:,:,0]
            if rotate:
                im = im.T
                im = im[::-1,::]
                
            video.append_data(im)
            count += 1
